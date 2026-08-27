"""Executable contracts for the stable Q30 fixed-versus-always experiment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


EXPERIMENT = "qwen3_30ba3b_fixed_vs_always_stable_200step_20260827"
SOURCE_ROOT = "/home/sna/nemorl-q30-fixed-always-product-20260827"
SOURCE_SHA = "4ee518b5dc2ed16f75e31876b477ea5ecf7d8c9b"
DURABLE_ROOT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    f"experiments/{EXPERIMENT}"
)
MODEL = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "hf-local/Qwen/Qwen3-30B-A3B"
)
DFLASH = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/"
    "base-dflash/exported-checkpoint-25391"
)
DSPARK = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/"
    "base-dspark/exported-checkpoint-25391"
)
CAPTURE_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]
VARIANTS = (
    "dflash-fixed",
    "dflash-always",
    "dspark-fixed",
    "dspark-always",
)


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment_root() -> Path:
    return root() / "experiments" / EXPERIMENT


def harness() -> Path:
    return experiment_root() / "submit_qwen3_30ba3b_fixed_vs_always_200step.sh"


def config_for(variant: str) -> dict[str, object]:
    return json.loads((experiment_root() / "configs" / f"{variant}.yaml").read_text())


class ContractTest(unittest.TestCase):
    maxDiff = None

    def manifest(self, variant: str) -> dict[str, object]:
        result = subprocess.run(
            ["bash", str(harness()), "--emit-manifest", variant],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def render(self, variant: str, temporary: str) -> tuple[Path, str, str]:
        result = subprocess.run(
            ["bash", str(harness()), "--render-sbatch", variant],
            cwd=root(),
            text=True,
            capture_output=True,
            env={**os.environ, "Q30_FIXED_ALWAYS_RENDER_ROOT": temporary},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        sbatch = Path(result.stdout.strip())
        driver = sbatch.parent / "driver.sh"
        self.assertEqual(subprocess.run(["bash", "-n", str(sbatch)]).returncode, 0)
        self.assertEqual(subprocess.run(["bash", "-n", str(driver)]).returncode, 0)
        return sbatch, sbatch.read_text(), driver.read_text()

    def prepare_submission_fixture(
        self, temporary_root: Path, variant: str
    ) -> tuple[Path, Path, Path, Path, str]:
        fixture_source_sha = "test-source-sha"
        fixture_harness_sha = "test-harness-sha"
        fixture_root = temporary_root / "fixture"
        durable_root = temporary_root / "durable"
        fixture_root.mkdir()
        shutil.copytree(experiment_root() / "configs", fixture_root / "configs")
        for filename in (
            "check_checkpoint_state_dict.py",
            "verify_composed_configs.py",
        ):
            shutil.copy2(experiment_root() / filename, fixture_root / filename)

        fixture_harness = fixture_root / harness().name
        fixture_contents = harness().read_text()
        substitutions = {
            r"^readonly SOURCE_ROOT=.*$": f"readonly SOURCE_ROOT={temporary_root / 'source'}",
            r"^readonly SOURCE_SHA=.*$": f"readonly SOURCE_SHA={fixture_source_sha}",
            r"^readonly CONTAINER=.*$": f"readonly CONTAINER={temporary_root / 'container.sqsh'}",
            r"^readonly DURABLE_ROOT=.*$": f"readonly DURABLE_ROOT={durable_root}",
            r"^readonly HARNESS_SHA=.*$": f"readonly HARNESS_SHA={fixture_harness_sha}",
        }
        for pattern, replacement in substitutions.items():
            fixture_contents = re.sub(
                pattern, replacement, fixture_contents, flags=re.MULTILINE
            )
        fixture_contents = re.sub(
            r"preflight\(\) \{\n.*?\n\}\n\nwrite_sbatch\(\)",
            "preflight() {\n  :\n}\n\nwrite_sbatch()",
            fixture_contents,
            count=1,
            flags=re.DOTALL,
        )
        fixture_contents = re.sub(
            r"harness_guard\(\) \{\n.*?\n\}\n\npreflight\(\)",
            "harness_guard() {\n  :\n}\n\npreflight()",
            fixture_contents,
            count=1,
            flags=re.DOTALL,
        )
        fixture_harness.write_text(fixture_contents)
        fixture_harness.chmod(0o700)

        config_sha = hashlib.sha256(
            (fixture_root / "configs" / f"{variant}.yaml").read_bytes()
        ).hexdigest()
        input_identity = {
            "checkpoint_checker_sha256": hashlib.sha256(
                (fixture_root / "check_checkpoint_state_dict.py").read_bytes()
            ).hexdigest(),
            "composition_verifier_sha256": hashlib.sha256(
                (fixture_root / "verify_composed_configs.py").read_bytes()
            ).hexdigest(),
            "config_sha256": config_sha,
            "launcher_sha256": hashlib.sha256(fixture_harness.read_bytes()).hexdigest(),
        }
        preflight_receipt = durable_root / "preflight" / f"{variant}.json"
        preflight_receipt.parent.mkdir(parents=True)
        preflight_receipt.write_text(
            json.dumps(
                {
                    "config_sha": config_sha,
                    "harness_sha": fixture_harness_sha,
                    "input_identity": input_identity,
                    "source_sha": fixture_source_sha,
                    "test_only_output": "validated",
                    "variant": variant,
                },
                sort_keys=True,
            )
            + "\n"
        )
        submission_record = (
            durable_root
            / "submissions"
            / f"{variant}-{fixture_source_sha}-{fixture_harness_sha}.json"
        )
        fake_bin = temporary_root / "bin"
        fake_bin.mkdir()
        return (
            fixture_harness,
            fixture_root,
            submission_record,
            fake_bin,
            fixture_harness_sha,
        )

    def assert_attempt_identity(
        self, receipt: dict[str, object], submission_record: Path
    ) -> None:
        artifact_dir = Path(str(receipt["artifact_dir"]))
        self.assertEqual(receipt["run_id"], artifact_dir.name)
        self.assertEqual(receipt["sbatch_path"], str(artifact_dir / "job.sbatch"))
        self.assertEqual(
            artifact_dir.parent, submission_record.parent.parent / "artifacts"
        )

    def test_matrix_has_exactly_four_requested_k5_arms(self) -> None:
        for variant in VARIANTS:
            manifest = self.manifest(variant)
            self.assertEqual(manifest["variant"], variant)
            self.assertEqual(manifest["k"], 5)
        invalid = subprocess.run(
            ["bash", str(harness()), "--emit-manifest", "dflash-fixed10"],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(invalid.returncode, 0)

    def test_configs_preserve_the_matched_q30_workload(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                config = config_for(variant)
                self.assertEqual(
                    config["defaults"],
                    f"{SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
                )
                self.assertNotIn("cadence_runtime", config)
                self.assertEqual(config["grpo"]["max_num_steps"], 200)
                self.assertEqual(config["grpo"]["num_prompts_per_step"], 16)
                self.assertEqual(config["grpo"]["num_generations_per_prompt"], 32)
                self.assertEqual(config["grpo"]["val_period"], 0)
                self.assertEqual(config["checkpointing"]["keep_top_k"], 1)
                self.assertFalse(config["data_plane"]["enabled"])
                self.assertFalse(config["data"]["shuffle"])
                self.assertEqual(
                    config["data"]["train"]["dataset_name"], "OpenMathInstruct-2"
                )
                policy = config["policy"]
                self.assertEqual(policy["model_name"], MODEL)
                self.assertEqual(policy["train_global_batch_size"], 512)
                self.assertEqual(policy["max_total_sequence_length"], 8192)
                self.assertEqual(policy["sequence_packing"], {"enabled": True})
                self.assertEqual(
                    policy["megatron_cfg"],
                    {
                        "tensor_model_parallel_size": 2,
                        "pipeline_model_parallel_size": 1,
                        "expert_model_parallel_size": 8,
                        "context_parallel_size": 1,
                        "sequence_parallel": True,
                    },
                )
                generation = policy["generation"]
                self.assertEqual(generation["max_new_tokens"], 1024)
                self.assertEqual(
                    generation["vllm_cfg"],
                    {
                        "tensor_parallel_size": 1,
                        "max_model_len": 8192,
                        "enforce_eager": False,
                    },
                )
                self.assertEqual(
                    config["cluster"],
                    {"gpus_per_node": 4, "num_nodes": 4, "segment_size": 4},
                )

    def test_fixed_keeps_generation_specdec_but_disables_draft_training(self) -> None:
        for drafter, checkpoint in (("dflash", DFLASH), ("dspark", DSPARK)):
            config = config_for(f"{drafter}-fixed")
            draft = config["policy"]["draft"]
            speculative = config["policy"]["generation"]["vllm_kwargs"][
                "speculative_config"
            ]
            self.assertFalse(draft["enabled"])
            self.assertIsNone(draft["optimizer"])
            self.assertNotIn("update_schedule", draft)
            self.assertEqual(draft["model_name"], checkpoint)
            self.assertEqual(speculative["model"], checkpoint)
            self.assertEqual(speculative["method"], drafter)
            self.assertEqual(speculative["num_speculative_tokens"], 5)

    def test_always_enables_stable_online_training_without_cadence_schema(self) -> None:
        for drafter, checkpoint in (("dflash", DFLASH), ("dspark", DSPARK)):
            config = config_for(f"{drafter}-always")
            draft = config["policy"]["draft"]
            speculative = config["policy"]["generation"]["vllm_kwargs"][
                "speculative_config"
            ]
            self.assertTrue(draft["enabled"])
            self.assertEqual(
                draft["optimizer"],
                {"lr": 5e-6, "min_lr": 5e-7, "weight_decay": 0.01},
            )
            self.assertNotIn("update_schedule", draft)
            self.assertEqual(draft["model_name"], checkpoint)
            self.assertEqual(speculative["model"], checkpoint)

    def test_drafter_specific_geometry_is_k5_matched(self) -> None:
        dflash = config_for("dflash-always")["policy"]["draft"]
        dspark = config_for("dspark-always")["policy"]["draft"]
        self.assertEqual(dflash["gamma"], 5)
        self.assertEqual(dspark["block_size"], 5)
        for draft in (dflash, dspark):
            self.assertEqual(draft["anchors_per_sample"], 2)
            self.assertEqual(draft["mask_token_id"], 151669)
            self.assertEqual(
                draft["target_hidden_state_layer_ids"], [1, 12, 23, 34, 45]
            )
            self.assertEqual(draft["num_layers"], 5)

    def test_manifest_pins_source_runtime_resources_and_wandb(self) -> None:
        harness_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root(),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        for variant in VARIANTS:
            first = self.manifest(variant)
            second = self.manifest(variant)
            self.assertEqual(first["source"], {"root": SOURCE_ROOT, "sha": SOURCE_SHA})
            self.assertEqual(first["max_steps"], 200)
            self.assertEqual(first["wandb_project"], "sna-specdec")
            self.assertEqual(
                first["wandb_group"],
                "q30ba3b-fixed-vs-always-stable-200step-20260827",
            )
            self.assertEqual(
                first["slurm"],
                {
                    "account": "nemotron_n3_post",
                    "partition": "batch",
                    "time": "04:00:00",
                    "nodes": 4,
                    "gpus_per_node": 4,
                },
            )
            self.assertEqual(
                first["gates"],
                [
                    "source-clean",
                    "state-dict",
                    "wandb-auth",
                    "cudagraph",
                    "step1",
                    "step2",
                ],
            )
            self.assertTrue(
                first["wandb_run_id"].startswith(
                    f"q30ba3b-stable-200step-{variant}-k5-"
                )
            )
            self.assertNotEqual(first["wandb_run_id"], second["wandb_run_id"])
            self.assertEqual(
                first["submission_record"],
                f"{DURABLE_ROOT}/submissions/{variant}-{SOURCE_SHA}-{harness_sha}.json",
            )
            self.assertEqual(
                set(first["input_identity"]),
                {
                    "checkpoint_checker_sha256",
                    "composition_verifier_sha256",
                    "config_sha256",
                    "launcher_sha256",
                },
            )
            for digest in first["input_identity"].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_harness_guard_requires_a_clean_exact_git_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_root = Path(temporary) / "fixture"
            fixture_root.mkdir()
            shutil.copytree(experiment_root() / "configs", fixture_root / "configs")
            for filename in (
                harness().name,
                "check_checkpoint_state_dict.py",
                "verify_composed_configs.py",
            ):
                shutil.copy2(experiment_root() / filename, fixture_root / filename)
            subprocess.run(["git", "init", "-q"], cwd=fixture_root, check=True)
            subprocess.run(["git", "add", "."], cwd=fixture_root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Contract Test",
                    "-c",
                    "user.email=contract@example.invalid",
                    "commit",
                    "-qm",
                    "test fixture",
                ],
                cwd=fixture_root,
                check=True,
            )
            fixture_harness = fixture_root / harness().name

            clean = subprocess.run(
                ["bash", str(fixture_harness), "--assert-harness-clean"],
                cwd=fixture_root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(clean.returncode, 0, clean.stderr)

            with (fixture_root / "verify_composed_configs.py").open("a") as stream:
                stream.write("\n")
            dirty = subprocess.run(
                ["bash", str(fixture_harness), "--assert-harness-clean"],
                cwd=fixture_root,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(dirty.returncode, 0)
            self.assertIn("dirty", dirty.stderr.lower())

    def test_rendered_jobs_pin_runtime_and_gate_before_training_progress(self) -> None:
        for variant in VARIANTS:
            with (
                self.subTest(variant=variant),
                tempfile.TemporaryDirectory() as temporary,
            ):
                sbatch_path, sbatch, driver = self.render(variant, temporary)
                artifact_dir = sbatch_path.parent
                self.assertTrue(
                    (artifact_dir / f"resolved-input-{variant}.yaml").is_file()
                )
                self.assertIn("#SBATCH --account=nemotron_n3_post", sbatch)
                self.assertIn("#SBATCH --nodes=4", sbatch)
                self.assertIn("#SBATCH --segment=4", sbatch)
                self.assertIn("#SBATCH --gpus-per-node=4", sbatch)
                self.assertIn("export CPUS_PER_WORKER=64", sbatch)
                self.assertIn("NRL_FORCE_REBUILD_VENVS=true", sbatch)
                self.assertIn("uv run --with hydra-core==1.3.2", driver)
                self.assertNotIn("cadence_runtime", driver)
                self.assertIn("WANDB_AUTH_GATE_PASS", driver)
                self.assertIn("CUDAGRAPH_GATE_PASS", driver)
                self.assertIn("STEP1_GATE_PASS", driver)
                self.assertIn("STEP2_GATE_PASS", driver)
                self.assertIn("compilation_config.cudagraph_mode=PIECEWISE", driver)
                self.assertIn(
                    "compilation_config.cudagraph_capture_sizes=[1,2,4,8,12,16,24,32,40,48]",
                    driver,
                )
                self.assertIn("logger.wandb.project=sna-specdec", driver)
                self.assertIn(
                    "logger.wandb.group=q30ba3b-fixed-vs-always-stable-200step-20260827",
                    driver,
                )

    def test_rendered_wandb_group_override_composes_with_strict_hydra(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, driver = self.render("dflash-fixed", temporary)
            match = re.search(r"(?:^|\s)(\+?logger\.wandb\.group=[^\s'\"]+)", driver)
            self.assertIsNotNone(match)
            override = match.group(1)
            config_path = Path(temporary) / "strict-config.json"
            config_path.write_text(
                json.dumps(
                    {"logger": {"wandb": {"project": "default", "name": "default"}}}
                )
            )
            program = """
import sys

from nemo_rl.utils.config import load_config, parse_hydra_overrides

config = parse_hydra_overrides(load_config(sys.argv[1]), [sys.argv[2]])
print(config.logger.wandb.group)
"""
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "--with",
                    "hydra-core==1.3.2",
                    "python3",
                    "-c",
                    program,
                    str(config_path),
                    override,
                ],
                cwd=root(),
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.strip(),
                "q30ba3b-fixed-vs-always-stable-200step-20260827",
            )

    def test_capture_buckets_cover_every_active_shape(self) -> None:
        result = subprocess.run(
            ["bash", str(harness()), "--assert-capture-coverage"],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["capture_sizes"], CAPTURE_SIZES)
        self.assertEqual(
            payload["shape_to_bucket"],
            {
                str(shape): next(size for size in CAPTURE_SIZES if size >= shape)
                for shape in range(1, 49)
            },
        )

    def test_state_dict_gate_accepts_exact_and_rejects_extra_key(self) -> None:
        checker = experiment_root() / "check_checkpoint_state_dict.py"
        expected_result = subprocess.run(
            ["uv", "run", "--no-project", str(checker), "--print-expected", "dflash"],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(expected_result.returncode, 0, expected_result.stderr)
        expected = json.loads(expected_result.stdout)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            header = {key: {} for key in expected}
            encoded = json.dumps(header).encode()
            (checkpoint / "model.safetensors").write_bytes(
                struct.pack("<Q", len(encoded)) + encoded
            )
            accepted = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    str(checker),
                    "--variant",
                    "dflash",
                    "--checkpoint",
                    str(checkpoint),
                ],
                cwd=root(),
                text=True,
                capture_output=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            header["target-owned.weight"] = {}
            encoded = json.dumps(header).encode()
            (checkpoint / "model.safetensors").write_bytes(
                struct.pack("<Q", len(encoded)) + encoded
            )
            rejected = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    str(checker),
                    "--variant",
                    "dflash",
                    "--checkpoint",
                    str(checkpoint),
                ],
                cwd=root(),
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("target-owned.weight", rejected.stderr)

    def test_completed_or_dangling_submission_record_blocks_sbatch(self) -> None:
        variant = "dflash-fixed"
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            fixture_harness, fixture_root, record, fake_bin, _ = (
                self.prepare_submission_fixture(temporary_root, variant)
            )
            record.parent.mkdir()
            original = '{"job_id":"1","state":"accepted"}\n'
            record.write_text(original)
            sbatch_log = temporary_root / "sbatch.log"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nprintf called >\"${FAKE_SBATCH_LOG}\"\nprintf 'Submitted batch job 2\\n'\n"
            )
            fake_sbatch.chmod(0o700)
            environment = {
                **os.environ,
                "FAKE_SBATCH_LOG": str(sbatch_log),
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
            }
            first = subprocess.run(
                ["bash", str(fixture_harness), "--submit", variant],
                cwd=fixture_root,
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertNotEqual(first.returncode, 0)
            self.assertIn("reconcile", first.stderr.lower())
            self.assertFalse(sbatch_log.exists())
            self.assertEqual(record.read_text(), original)

            record.unlink()
            record.symlink_to(temporary_root / "missing-receipt.json")
            second = subprocess.run(
                ["bash", str(fixture_harness), "--submit", variant],
                cwd=fixture_root,
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("reconcile", second.stderr.lower())
            self.assertFalse(sbatch_log.exists())
            self.assertTrue(record.is_symlink())

    def test_mutated_helper_or_config_invalidates_receipt_before_sbatch(self) -> None:
        variant = "dflash-fixed"
        for relative_path in (
            Path("verify_composed_configs.py"),
            Path("configs") / f"{variant}.yaml",
        ):
            with (
                self.subTest(path=str(relative_path)),
                tempfile.TemporaryDirectory() as temporary,
            ):
                temporary_root = Path(temporary)
                fixture_harness, fixture_root, record, fake_bin, _ = (
                    self.prepare_submission_fixture(temporary_root, variant)
                )
                with (fixture_root / relative_path).open("a") as stream:
                    stream.write("\n# mutated after scheduler validation\n")
                sbatch_log = temporary_root / "sbatch.log"
                fake_sbatch = fake_bin / "sbatch"
                fake_sbatch.write_text(
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    'printf called >"${FAKE_SBATCH_LOG}"\n'
                    "printf 'Submitted batch job 2\\n'\n"
                )
                fake_sbatch.chmod(0o700)

                result = subprocess.run(
                    ["bash", str(fixture_harness), "--submit", variant],
                    cwd=fixture_root,
                    text=True,
                    capture_output=True,
                    env={
                        **os.environ,
                        "FAKE_SBATCH_LOG": str(sbatch_log),
                        "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    },
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid test-only receipt", result.stderr.lower())
                self.assertFalse(sbatch_log.exists())
                self.assertFalse(record.exists())

    def test_scheduler_ambiguity_is_durable_and_secret_free(self) -> None:
        variant = "dflash-fixed"
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            fixture_harness, fixture_root, record, fake_bin, fixture_harness_sha = (
                self.prepare_submission_fixture(temporary_root, variant)
            )
            before = temporary_root / "before.json"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'cp "${FAKE_SUBMISSION_RECORD}" "${FAKE_BEFORE_RECORD}"\n'
                "printf 'Submitted batch job 4242\\n'\n"
                "printf 'secret=%s\\n' \"${WANDB_API_KEY}\"\n"
                "exit 70\n"
            )
            fake_sbatch.chmod(0o700)
            result = subprocess.run(
                ["bash", str(fixture_harness), "--submit", variant],
                cwd=fixture_root,
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "FAKE_BEFORE_RECORD": str(before),
                    "FAKE_SUBMISSION_RECORD": str(record),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "WANDB_API_KEY": "must-not-be-written",
                },
            )
            self.assertNotEqual(result.returncode, 0)
            pre_sbatch = json.loads(before.read_text())
            self.assertEqual(pre_sbatch["state"], "submitting")
            self.assertEqual(pre_sbatch["harness_sha"], fixture_harness_sha)
            self.assertEqual(
                set(pre_sbatch["input_identity"]),
                {
                    "checkpoint_checker_sha256",
                    "composition_verifier_sha256",
                    "config_sha256",
                    "launcher_sha256",
                },
            )
            self.assert_attempt_identity(pre_sbatch, record)
            ambiguous = json.loads(record.read_text())
            self.assertEqual(ambiguous["state"], "ambiguous")
            self.assertEqual(ambiguous["scheduler_exit_status"], 70)
            self.assertEqual(ambiguous["candidate_job_ids"], ["4242"])
            self.assertNotIn("must-not-be-written", record.read_text())

    def test_repository_artifacts_do_not_contain_wandb_secret_values(self) -> None:
        for path in experiment_root().rglob("*"):
            if path.is_file():
                self.assertNotRegex(
                    path.read_text(errors="ignore"),
                    r"WANDB_API_KEY=(['\"])(?!\\?\$)",
                )


if __name__ == "__main__":
    unittest.main()
