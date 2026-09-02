from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = EXPERIMENT_ROOT / "configs"
LAUNCHER = EXPERIMENT_ROOT / "submit_qwen235b_math_grpo.sh"
VERIFIER = EXPERIMENT_ROOT / "verify_composed_configs.py"

DEFAULT_SMALL_CAPTURE_SIZES = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
    216,
    224,
    232,
    240,
    248,
    256,
    272,
    288,
    304,
    320,
    336,
    352,
    368,
    384,
    400,
    416,
    432,
    448,
    464,
    480,
    496,
    512,
]


class Qwen235BMathGrpoContractTest(unittest.TestCase):
    PERFORMANCE_RECIPE = (
        "/home/sna/nemorl-q235-math-product-20260828/examples/configs/"
        "recipes/llm/performance/grpo-qwen3-235b-32n4g.yaml"
    )

    def load_config(self, arm: str) -> dict[str, object]:
        path = CONFIG_ROOT / f"{arm}.yaml"
        self.assertTrue(path.is_file(), f"missing config: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def prepare_submission_fixture(
        self, temporary_root: Path
    ) -> tuple[Path, Path, Path, dict[str, str]]:
        fixture_root = temporary_root / "fixture"
        durable_root = temporary_root / "durable"
        fake_bin = temporary_root / "bin"
        fixture_root.mkdir()
        (fixture_root / "configs").mkdir()
        fake_bin.mkdir()
        shutil.copy2(CONFIG_ROOT / "baseline_cg2048.yaml", fixture_root / "configs")
        shutil.copy2(VERIFIER, fixture_root)

        fixture_launcher = fixture_root / LAUNCHER.name
        launcher = LAUNCHER.read_text(encoding="utf-8")
        launcher = re.sub(
            r"^readonly SOURCE_ROOT=.*$",
            f"readonly SOURCE_ROOT='{temporary_root / 'source'}'",
            launcher,
            flags=re.MULTILINE,
        )
        launcher = launcher.replace(
            "readonly SBATCH_TIMEOUT_SECONDS=30",
            "readonly SBATCH_TIMEOUT_SECONDS=1",
        )
        launcher = re.sub(
            r"^readonly DURABLE_ROOT=.*$",
            f"readonly DURABLE_ROOT='{durable_root}'",
            launcher,
            flags=re.MULTILINE,
        )
        launcher = re.sub(
            r"^HARNESS_SHA=.*$",
            "HARNESS_SHA=test-harness-sha",
            launcher,
            flags=re.MULTILINE,
        )
        launcher = re.sub(
            r"preflight\(\) \{\n.*?\n\}\n\nrun_id\(\)",
            "preflight() {\n  :\n}\n\nrun_id()",
            launcher,
            count=1,
            flags=re.DOTALL,
        )
        copy_verifier = (
            '  cp "${SCRIPT_DIR}/verify_composed_configs.py" '
            '"${artifact}/verify_composed_configs.py"\n'
        )
        launcher = launcher.replace(
            copy_verifier,
            copy_verifier
            + """  if [[ -n "${Q235_TEST_MUTATE_SOURCE_AFTER_COPY:-}" ]]; then
    printf '\n# source mutation after copy\n' >>"${SCRIPT_DIR}/verify_composed_configs.py"
  fi
  if [[ -n "${Q235_TEST_MUTATE_ARTIFACT_AFTER_COPY:-}" ]]; then
    printf '\n# artifact mutation after copy\n' >>"${artifact}/verify_composed_configs.py"
  fi
""",
        )
        submit_identity = """  --submit)
    preflight "${arm}"
    identity="$(submission_identity "${arm}")"
"""
        launcher = launcher.replace(
            submit_identity,
            submit_identity
            + """    if [[ -n "${Q235_TEST_MUTATE_CONFIG_AFTER_IDENTITY:-}" ]]; then
      printf '\n ' >>"${SCRIPT_DIR}/configs/${arm}.yaml"
    fi
""",
        )
        fixture_launcher.write_text(launcher, encoding="utf-8")
        fixture_launcher.chmod(0o700)

        fake_sbatch = fake_bin / "sbatch"
        fake_sbatch.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >>"${FAKE_SBATCH_CALLS:?}"
if [[ "${1:-}" == --test-only ]]; then
  echo 'sbatch: Job 9001 to start at 2026-09-02T12:00:00 using 128 processors on nodes test'
  exit 0
fi
case "${FAKE_SBATCH_MODE:-accepted}" in
  accepted) echo "Submitted batch job ${FAKE_JOB_ID:-12345}" ;;
  ambiguous)
    echo "Submitted batch job ${FAKE_JOB_ID:-12345}"
    exit 1
    ;;
  large) python3 -c 'import sys; sys.stdout.write("x" * 100000)' ;;
  hang) sleep 2 ;;
  malformed) echo 'scheduler wrapper returned without a job id' ;;
esac
""",
            encoding="utf-8",
        )
        fake_sbatch.chmod(0o700)
        calls = temporary_root / "sbatch-calls.log"
        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_SBATCH_CALLS": str(calls),
            "Q235_MAX_STEPS": "20",
        }
        return fixture_launcher, durable_root, calls, env

    def run_fixture_launcher(
        self,
        launcher: Path,
        mode: str,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(launcher), mode, "baseline_cg2048"],
            cwd=launcher.parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_matrix_contains_matched_dspark_k3_k5_k7_arms(self) -> None:
        expected = {
            "baseline": (None, 0),
            "dspark_k3": ("dspark", 3),
            "dspark_k5": ("dspark", 5),
            "dspark_k7": ("dspark", 7),
        }

        for arm, (method, k) in expected.items():
            with self.subTest(arm=arm):
                config = self.load_config(arm)
                if method is None:
                    self.assertNotIn("policy", config)
                else:
                    policy = config["policy"]
                    self.assertIsInstance(policy, dict)
                    generation = policy["generation"]
                    self.assertIsInstance(generation, dict)
                    kwargs = generation["vllm_kwargs"]
                    self.assertIsInstance(kwargs, dict)
                    speculative = kwargs.get("speculative_config")
                    self.assertIsInstance(speculative, dict)
                    self.assertEqual(speculative["method"], method)
                    self.assertEqual(speculative["num_speculative_tokens"], k)

    def test_expanded_matrix_contains_exactly_four_matched_pairs(self) -> None:
        expected = {
            "baseline_cg2048": "baseline",
            "dspark_k3_cg2048": "dspark_k3",
            "dspark_k5_cg2048": "dspark_k5",
            "dspark_k7_cg2048": "dspark_k7",
        }

        for arm, base_arm in expected.items():
            with self.subTest(arm=arm):
                config = self.load_config(arm)
                base = self.load_config(base_arm)
                self.assertEqual(config["defaults"], base["defaults"])
                self.assertEqual(config["grpo"], base["grpo"])
                if base_arm == "baseline":
                    self.assertEqual(set(config), {"defaults", "grpo", "policy"})
                    self.assertEqual(set(config["policy"]), {"generation"})
                    self.assertEqual(
                        set(config["policy"]["generation"]), {"vllm_kwargs"}
                    )
                    self.assertEqual(
                        set(config["policy"]["generation"]["vllm_kwargs"]),
                        {"compilation_config"},
                    )
                else:
                    expanded_kwargs = config["policy"]["generation"]["vllm_kwargs"]
                    base_kwargs = base["policy"]["generation"]["vllm_kwargs"]
                    self.assertEqual(
                        {
                            key: value
                            for key, value in expanded_kwargs.items()
                            if key != "compilation_config"
                        },
                        {
                            key: value
                            for key, value in base_kwargs.items()
                            if key != "compilation_config"
                        },
                    )

    def test_expanded_capture_sizes_union_defaults_and_exact_verifier_anchors(
        self,
    ) -> None:
        expected_anchors = {
            "baseline_cg2048": {1024, 2048},
            "dspark_k3_cg2048": {256, 512, 1024, 2048},
            "dspark_k5_cg2048": {384, 768, 1536},
            "dspark_k7_cg2048": {512, 1024, 2048},
        }
        for arm, anchors in expected_anchors.items():
            with self.subTest(arm=arm):
                config = self.load_config(arm)
                sizes = config["policy"]["generation"]["vllm_kwargs"][
                    "compilation_config"
                ]["cudagraph_capture_sizes"]
                self.assertEqual(sizes, sorted(set(sizes)))
                self.assertLessEqual(max(sizes), 2048)
                if arm == "baseline_cg2048":
                    required_small = set(DEFAULT_SMALL_CAPTURE_SIZES)
                else:
                    base_arm = arm.removesuffix("_cg2048")
                    required_small = set(
                        self.load_config(base_arm)["policy"]["generation"][
                            "vllm_kwargs"
                        ]["compilation_config"]["cudagraph_capture_sizes"]
                    )
                self.assertTrue(required_small <= set(sizes))
                self.assertTrue(anchors <= set(sizes))

    def test_composition_validator_agrees_with_expanded_config_profiles(self) -> None:
        spec = importlib.util.spec_from_file_location("q235_verifier", VERIFIER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for arm in (
            "baseline_cg2048",
            "dspark_k3_cg2048",
            "dspark_k5_cg2048",
            "dspark_k7_cg2048",
        ):
            with self.subTest(arm=arm):
                config = self.load_config(arm)
                actual = config["policy"]["generation"]["vllm_kwargs"][
                    "compilation_config"
                ]["cudagraph_capture_sizes"]
                self.assertEqual(module.expected_capture_sizes(arm), actual)

    def test_all_arms_are_thin_overrides_of_official_32n4g_recipe(self) -> None:
        for arm in ("baseline", "dspark_k3", "dspark_k5", "dspark_k7"):
            with self.subTest(arm=arm):
                config = self.load_config(arm)
                self.assertEqual(config["defaults"], self.PERFORMANCE_RECIPE)
                self.assertEqual(config["grpo"], {"max_num_steps": 20})
                if arm == "baseline":
                    self.assertEqual(set(config), {"defaults", "grpo"})
                    continue

                self.assertEqual(set(config), {"defaults", "grpo", "policy"})
                policy = config["policy"]
                self.assertEqual(set(policy), {"generation"})
                generation = policy["generation"]
                self.assertEqual(set(generation), {"vllm_kwargs"})
                kwargs = generation["vllm_kwargs"]
                self.assertEqual(
                    set(kwargs),
                    {"kernel_config", "speculative_config", "compilation_config"},
                )
                self.assertNotIn("cudagraph_mode", kwargs["compilation_config"])

    def test_launcher_emits_immutable_arm_manifests(self) -> None:
        self.assertTrue(LAUNCHER.is_file(), f"missing launcher: {LAUNCHER}")
        expected_max_steps = int(os.environ.get("Q235_MAX_STEPS", "20"))
        expected = {
            "baseline": (None, 0),
            "dspark_k3": ("dspark", 3),
            "dspark_k5": ("dspark", 5),
            "dspark_k7": ("dspark", 7),
        }
        for arm, (method, k) in expected.items():
            with self.subTest(arm=arm):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "--emit-manifest", arm],
                    cwd=EXPERIMENT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                manifest = json.loads(result.stdout)
                self.assertEqual(manifest["arm"], arm)
                self.assertEqual(manifest["method"], method)
                self.assertEqual(manifest["num_speculative_tokens"], k)
                self.assertEqual(manifest["max_steps"], expected_max_steps)
                self.assertEqual(manifest["slurm"]["nodes"], 32)
                self.assertEqual(manifest["slurm"]["gpus_per_node"], 4)
                self.assertEqual(manifest["slurm"]["segment"], 16)

    def test_expanded_manifests_are_truthful_and_dflash_stays_rejected(self) -> None:
        expected = {
            "baseline_cg2048": ("baseline", None, 0),
            "dspark_k3_cg2048": ("dspark_k3", "dspark", 3),
            "dspark_k5_cg2048": ("dspark_k5", "dspark", 5),
            "dspark_k7_cg2048": ("dspark_k7", "dspark", 7),
        }
        for arm, (base_arm, method, k) in expected.items():
            with self.subTest(arm=arm):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "--emit-manifest", arm],
                    cwd=EXPERIMENT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                manifest = json.loads(result.stdout)
                self.assertEqual(manifest["base_arm"], base_arm)
                self.assertEqual(manifest["graph_profile"], "expanded_2048")
                self.assertEqual(manifest["method"], method)
                self.assertEqual(manifest["num_speculative_tokens"], k)
                self.assertEqual(
                    manifest["cudagraph_capture_sizes_source"],
                    "arm-config-expanded-through-2048",
                )
                base_result = subprocess.run(
                    ["bash", str(LAUNCHER), "--emit-manifest", base_arm],
                    cwd=EXPERIMENT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(base_result.returncode, 0, base_result.stderr)
                base_manifest = json.loads(base_result.stdout)
                for immutable_field in (
                    "method",
                    "num_speculative_tokens",
                    "checkpoint",
                    "source",
                    "container",
                    "max_steps",
                    "wandb_project",
                    "wandb_group",
                    "cudagraph_mode_source",
                    "slurm",
                ):
                    self.assertEqual(
                        manifest[immutable_field],
                        base_manifest[immutable_field],
                        immutable_field,
                    )

        for stale_arm in ("dflash_k3", "dflash_k5", "dflash_k3_cg2048"):
            with self.subTest(stale_arm=stale_arm):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "--emit-manifest", stale_arm],
                    cwd=EXPERIMENT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)

    def test_launcher_accepts_only_the_base_dspark_checkpoint(self) -> None:
        for arm in (
            "baseline",
            "dspark_k3",
            "dspark_k5",
            "dspark_k7",
            "baseline_cg2048",
            "dspark_k3_cg2048",
            "dspark_k5_cg2048",
            "dspark_k7_cg2048",
        ):
            with self.subTest(arm=arm):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "--validate-arm-contract", arm],
                    cwd=EXPERIMENT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_dspark_arms_use_the_base_block_size_eight_checkpoint(self) -> None:
        for arm in (
            "dspark_k3",
            "dspark_k5",
            "dspark_k7",
            "dspark_k3_cg2048",
            "dspark_k5_cg2048",
            "dspark_k7_cg2048",
        ):
            with self.subTest(arm=arm):
                config = self.load_config(arm)
                generation = config["policy"]["generation"]
                kwargs = generation["vllm_kwargs"]
                speculative = kwargs["speculative_config"]
                self.assertIn(
                    "qwen3-235ba22b-base-nemotron-b8-s25391/dspark",
                    speculative["model"],
                )
                self.assertEqual(speculative["draft_tensor_parallel_size"], 1)

    def test_dspark_capture_sizes_cover_k_and_bonus_widths(self) -> None:
        batches = (1, 2, 4, 8, 16, 32)
        for k in (3, 5, 7):
            with self.subTest(k=k):
                config = self.load_config(f"dspark_k{k}")
                sizes = set(
                    config["policy"]["generation"]["vllm_kwargs"]["compilation_config"][
                        "cudagraph_capture_sizes"
                    ]
                )
                self.assertTrue({k * batch for batch in batches} <= sizes)
                self.assertTrue({(k + 1) * batch for batch in batches} <= sizes)

    def test_submission_record_identity_includes_config_and_harness(self) -> None:
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--emit-submission-record", "baseline"],
            cwd=EXPERIMENT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record_name = Path(result.stdout.strip()).name
        config_sha = hashlib.sha256(
            (CONFIG_ROOT / "baseline.yaml").read_bytes()
        ).hexdigest()
        harness_sha = subprocess.run(
            ["git", "-C", str(EXPERIMENT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertTrue(
            record_name.endswith(f"-{config_sha}-{harness_sha}.json"), record_name
        )

    def test_accepted_submission_receipt_prevents_repeat_sbatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _, calls, env = self.prepare_submission_fixture(Path(temporary))
            test_only = self.run_fixture_launcher(
                launcher, "--test-only", {**env, "ACCOUNT": "account_a"}
            )
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            first = self.run_fixture_launcher(
                launcher, "--submit", {**env, "ACCOUNT": "account_a"}
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(first.stdout.strip(), "Submitted batch job 12345")

            record_result = self.run_fixture_launcher(
                launcher, "--emit-submission-record", env
            )
            self.assertEqual(record_result.returncode, 0, record_result.stderr)
            record = Path(record_result.stdout.strip())
            receipt = json.loads(record.read_text(encoding="utf-8"))
            self.assertEqual(receipt["state"], "accepted")
            self.assertEqual(receipt["job_id"], "12345")

            repeat = self.run_fixture_launcher(
                launcher, "--submit", {**env, "ACCOUNT": "account_a"}
            )
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("submission already exists", repeat.stderr)
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 2)

    def test_ambiguous_scheduler_acceptance_is_durable_and_blocks_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _, calls, env = self.prepare_submission_fixture(Path(temporary))
            bound_env = {
                **env,
                "ACCOUNT": "account_a",
                "FAKE_SBATCH_MODE": "ambiguous",
                "FAKE_JOB_ID": "67890",
            }
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            submit = self.run_fixture_launcher(launcher, "--submit", bound_env)
            self.assertNotEqual(submit.returncode, 0)

            record_result = self.run_fixture_launcher(
                launcher, "--emit-submission-record", bound_env
            )
            record = Path(record_result.stdout.strip())
            receipt = json.loads(record.read_text(encoding="utf-8"))
            self.assertEqual(receipt["state"], "ambiguous")
            self.assertEqual(receipt["scheduler_exit_status"], 1)
            self.assertEqual(receipt["candidate_job_ids"], ["67890"])

            retry = self.run_fixture_launcher(launcher, "--submit", bound_env)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("submission already exists", retry.stderr)
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 2)

    def test_submit_freezes_the_identity_validated_by_test_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, durable_root, _, env = self.prepare_submission_fixture(
                Path(temporary)
            )
            bound_env = {**env, "ACCOUNT": "account_a"}
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            preflight_receipt = json.loads(
                (durable_root / "preflight" / "baseline_cg2048-steps20.json").read_text(
                    encoding="utf-8"
                )
            )

            submit = self.run_fixture_launcher(
                launcher,
                "--submit",
                {**bound_env, "Q235_TEST_MUTATE_SOURCE_AFTER_COPY": "1"},
            )
            self.assertEqual(submit.returncode, 0, submit.stderr)
            record_result = self.run_fixture_launcher(
                launcher, "--emit-submission-record", bound_env
            )
            record = json.loads(
                Path(record_result.stdout.strip()).read_text(encoding="utf-8")
            )
            self.assertEqual(record["identity"], preflight_receipt["identity"])

    def test_submit_rejects_copied_artifact_hash_drift_before_sbatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _, calls, env = self.prepare_submission_fixture(Path(temporary))
            bound_env = {**env, "ACCOUNT": "account_a"}
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)

            submit = self.run_fixture_launcher(
                launcher,
                "--submit",
                {**bound_env, "Q235_TEST_MUTATE_ARTIFACT_AFTER_COPY": "1"},
            )
            self.assertNotEqual(submit.returncode, 0)
            self.assertIn("rendered artifact content drift", submit.stderr)
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 1)

            record_result = self.run_fixture_launcher(
                launcher, "--emit-submission-record", bound_env
            )
            record = json.loads(
                Path(record_result.stdout.strip()).read_text(encoding="utf-8")
            )
            self.assertEqual(record["state"], "submitting")

    def test_submit_rejects_a_legacy_lock_before_creating_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _, calls, env = self.prepare_submission_fixture(Path(temporary))
            bound_env = {**env, "ACCOUNT": "account_a"}
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            record_result = self.run_fixture_launcher(
                launcher, "--emit-submission-record", bound_env
            )
            record = Path(record_result.stdout.strip())
            record.parent.mkdir(parents=True, exist_ok=True)
            legacy_lock = Path(f"{record}.lock")
            legacy_lock.write_text("legacy in-progress submission\n", encoding="utf-8")

            submit = self.run_fixture_launcher(launcher, "--submit", bound_env)
            self.assertNotEqual(submit.returncode, 0)
            self.assertIn("legacy submission lock", submit.stderr)
            self.assertFalse(record.exists())
            self.assertTrue(legacy_lock.exists())
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 1)

    def test_submit_rejects_a_different_hash_legacy_lock_for_the_logical_arm(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, durable_root, calls, env = self.prepare_submission_fixture(
                Path(temporary)
            )
            bound_env = {**env, "ACCOUNT": "account_a"}
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            submissions = durable_root / "submissions"
            submissions.mkdir(parents=True)
            prior_lock = submissions / (
                "baseline_cg2048-steps20-oldsource-oldconfig-oldharness.json.lock"
            )
            prior_lock.write_text("legacy submission in progress\n", encoding="utf-8")

            submit = self.run_fixture_launcher(launcher, "--submit", bound_env)
            self.assertNotEqual(submit.returncode, 0)
            self.assertIn("prior logical submission lock", submit.stderr)
            self.assertIn("reconcile", submit.stderr)
            self.assertTrue(prior_lock.is_file())
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 1)

    def test_submit_record_path_uses_the_frozen_config_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _, calls, env = self.prepare_submission_fixture(Path(temporary))
            bound_env = {**env, "ACCOUNT": "account_a"}
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            expected_record_result = self.run_fixture_launcher(
                launcher, "--emit-submission-record", bound_env
            )
            self.assertEqual(expected_record_result.returncode, 0)
            expected_record = Path(expected_record_result.stdout.strip())

            submit = self.run_fixture_launcher(
                launcher,
                "--submit",
                {**bound_env, "Q235_TEST_MUTATE_CONFIG_AFTER_IDENTITY": "1"},
            )
            self.assertNotEqual(submit.returncode, 0)
            self.assertIn("rendered artifact content drift", submit.stderr)
            self.assertTrue(expected_record.is_file())
            receipt = json.loads(expected_record.read_text(encoding="utf-8"))
            self.assertEqual(receipt["state"], "submitting")
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 1)

    def test_submit_rejects_prior_logical_arm_record_across_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, durable_root, calls, env = self.prepare_submission_fixture(
                Path(temporary)
            )
            bound_env = {**env, "ACCOUNT": "account_a"}
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            submissions = durable_root / "submissions"
            submissions.mkdir(parents=True)
            prior = submissions / (
                "baseline_cg2048-steps20-oldsource-oldconfig-oldharness.json"
            )
            prior.write_text('{"state":"accepted","job_id":"77"}\n', encoding="utf-8")

            submit = self.run_fixture_launcher(launcher, "--submit", bound_env)
            self.assertNotEqual(submit.returncode, 0)
            self.assertIn("prior logical submission", submit.stderr)
            self.assertIn("reconcile", submit.stderr)
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 1)
            claim = submissions / "claims" / "baseline_cg2048-steps20.json"
            self.assertTrue(claim.is_file())

    def test_atomic_receipts_retain_owner_only_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, durable_root, _, env = self.prepare_submission_fixture(
                Path(temporary)
            )
            bound_env = {**env, "ACCOUNT": "account_a"}
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            preflight = durable_root / "preflight" / "baseline_cg2048-steps20.json"
            self.assertEqual(stat.S_IMODE(preflight.stat().st_mode), 0o600)

            submit = self.run_fixture_launcher(launcher, "--submit", bound_env)
            self.assertEqual(submit.returncode, 0, submit.stderr)
            record_result = self.run_fixture_launcher(
                launcher, "--emit-submission-record", bound_env
            )
            record = Path(record_result.stdout.strip())
            self.assertEqual(stat.S_IMODE(record.stat().st_mode), 0o600)

    def test_scheduler_output_capture_is_bounded_and_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _, calls, env = self.prepare_submission_fixture(Path(temporary))
            bound_env = {**env, "ACCOUNT": "account_a", "FAKE_SBATCH_MODE": "large"}
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            submit = self.run_fixture_launcher(launcher, "--submit", bound_env)
            self.assertNotEqual(submit.returncode, 0)

            record_result = self.run_fixture_launcher(
                launcher, "--emit-submission-record", bound_env
            )
            receipt = json.loads(
                Path(record_result.stdout.strip()).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["state"], "ambiguous")
            self.assertFalse(receipt["scheduler_timed_out"])
            self.assertTrue(receipt["scheduler_output_truncated"])
            self.assertGreaterEqual(receipt["scheduler_output_bytes"], 100_000)
            self.assertLessEqual(len(receipt["scheduler_output"].encode()), 65_536)
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 2)

    def test_scheduler_capture_timeout_is_durable_and_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _, calls, env = self.prepare_submission_fixture(Path(temporary))
            bound_env = {**env, "ACCOUNT": "account_a", "FAKE_SBATCH_MODE": "hang"}
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            submit = self.run_fixture_launcher(launcher, "--submit", bound_env)
            self.assertNotEqual(submit.returncode, 0)

            record_result = self.run_fixture_launcher(
                launcher, "--emit-submission-record", bound_env
            )
            receipt = json.loads(
                Path(record_result.stdout.strip()).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["state"], "ambiguous")
            self.assertTrue(receipt["scheduler_timed_out"])
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 2)

    def test_test_only_receipt_binds_account_resources_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, durable_root, calls, env = self.prepare_submission_fixture(
                Path(temporary)
            )
            test_only = self.run_fixture_launcher(
                launcher, "--test-only", {**env, "ACCOUNT": "account_a"}
            )
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            receipt_path = durable_root / "preflight" / "baseline_cg2048-steps20.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            identity = receipt["identity"]
            self.assertEqual(identity["slurm"]["account"], "account_a")
            self.assertEqual(
                identity["slurm"],
                {
                    "account": "account_a",
                    "cpus_per_worker": 64,
                    "exclusive": True,
                    "gpus_per_node": 4,
                    "memory": "0",
                    "nodes": 32,
                    "partition": "batch",
                    "qos": "normal",
                    "segment": 16,
                    "time": "04:00:00",
                },
            )
            self.assertRegex(identity["content"]["launcher_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(identity["content"]["config_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(identity["content"]["verifier_sha256"], r"^[0-9a-f]{64}$")

            changed_account = self.run_fixture_launcher(
                launcher, "--submit", {**env, "ACCOUNT": "account_b"}
            )
            self.assertNotEqual(changed_account.returncode, 0)
            self.assertIn("invalid test-only receipt", changed_account.stderr)
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 1)

    def test_test_only_receipt_rejects_changed_render_input_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _, calls, env = self.prepare_submission_fixture(Path(temporary))
            bound_env = {**env, "ACCOUNT": "account_a"}
            test_only = self.run_fixture_launcher(launcher, "--test-only", bound_env)
            self.assertEqual(test_only.returncode, 0, test_only.stderr)
            verifier = launcher.parent / VERIFIER.name
            verifier.write_text(
                verifier.read_text(encoding="utf-8") + "\n# changed after test-only\n",
                encoding="utf-8",
            )

            submit = self.run_fixture_launcher(launcher, "--submit", bound_env)
            self.assertNotEqual(submit.returncode, 0)
            self.assertIn("invalid test-only receipt", submit.stderr)
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 1)

    def test_launcher_pins_clean_product_and_node_local_overlays(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("f6f8605da02675af4361cfc9fd4d5f4d23279ff1", launcher)
        self.assertIn("source is dirty", launcher)
        self.assertIn("Q235_MCORE_OVERLAY", launcher)
        self.assertIn("NRL_VENV_POST_SYNC_SCRIPT", launcher)
        self.assertIn("vllm-0.25.1-pr48167-runtime.patch", launcher)
        self.assertIn("vllm-0.25.1-pr48167-group-causality-followup.patch", launcher)
        self.assertIn(
            "504730a52614fddeb8ea899ec37a0aa820dcbc3a57c704fc13f5834fcc07b317",
            launcher,
        )
        self.assertIn(
            "8e5ff0e385ee44cf71e1e07031e5cd19658b29eb7b90bc172a4754c599d1dd90",
            launcher,
        )
        self.assertIn("verify_composed_configs.py", launcher)

    def test_rendered_dspark_job_uses_fap_overlay_only_for_dspark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rendered: dict[str, str] = {}
            for arm in ("baseline", "dspark_k3", "dspark_k5", "dspark_k7"):
                env = {**os.environ, "Q235_RENDER_ROOT": temporary}
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "--render-sbatch", arm],
                    cwd=EXPERIMENT_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                rendered[arm] = Path(result.stdout.strip()).read_text()

            self.assertNotIn("NRL_VENV_POST_SYNC_SCRIPT", rendered["baseline"])
            self.assertIn(
                'export PATH="/cm/local/apps/slurm/current/bin:${PATH}"',
                rendered["baseline"],
            )
            self.assertIn("#SBATCH --segment=16", rendered["baseline"])
            for arm in ("dspark_k3", "dspark_k5", "dspark_k7"):
                self.assertIn("NRL_VENV_POST_SYNC_SCRIPT", rendered[arm])
                self.assertIn("Q235_VLLM_OVERLAY", rendered[arm])
                self.assertIn("/raid:/raid", rendered[arm])

    def test_rendered_expanded_jobs_keep_unique_arm_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rendered_paths: set[Path] = set()
            for arm in (
                "baseline_cg2048",
                "dspark_k3_cg2048",
                "dspark_k5_cg2048",
                "dspark_k7_cg2048",
            ):
                with self.subTest(arm=arm):
                    env = {**os.environ, "Q235_RENDER_ROOT": temporary}
                    result = subprocess.run(
                        ["bash", str(LAUNCHER), "--render-sbatch", arm],
                        cwd=EXPERIMENT_ROOT,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    rendered_path = Path(result.stdout.strip())
                    self.assertNotIn(rendered_path, rendered_paths)
                    rendered_paths.add(rendered_path)
                    sbatch = rendered_path.read_text(encoding="utf-8")
                    self.assertIn(f"#SBATCH --job-name=q235-math-{arm}", sbatch)
                    driver = (rendered_path.parent / "driver.sh").read_text(
                        encoding="utf-8"
                    )
                    self.assertIn(f"step25391-{arm}-", driver)
                    self.assertIn(f"resolved-input-{arm}.yaml", driver)

    def test_rendered_jobs_claim_full_nodes_like_official_performance_launcher(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = {**os.environ, "Q235_RENDER_ROOT": temporary}
            result = subprocess.run(
                ["bash", str(LAUNCHER), "--render-sbatch", "baseline"],
                cwd=EXPERIMENT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            sbatch = Path(result.stdout.strip()).read_text(encoding="utf-8")

            self.assertIn("#SBATCH --exclusive", sbatch)
            self.assertIn("#SBATCH --mem=0", sbatch)
            self.assertIn("export NCCL_NVLS_ENABLE=0", sbatch)


if __name__ == "__main__":
    unittest.main()
