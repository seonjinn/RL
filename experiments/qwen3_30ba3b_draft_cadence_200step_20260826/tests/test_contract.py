"""Executable contracts for the Qwen3-30B-A3B draft-cadence matrix."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


EXPERIMENT = "qwen3_30ba3b_draft_cadence_200step_20260826"
SOURCE_ROOT = "/home/sna/nemorl-q30-cadence-product-20260826"
SOURCE_SHA = "1ce79c48334496fe4d86cf99fb3d27208b9f9b51"
DURABLE_ROOT = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_draft_cadence_200step_20260826"
MODEL = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B"
DFLASH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dflash/exported-checkpoint-25391"
DSPARK = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dspark/exported-checkpoint-25391"
INTERVALS = (5, 10, 20)
VARIANTS = tuple(
    f"{drafter}-fixed{interval}"
    for drafter in ("dflash", "dspark")
    for interval in INTERVALS
)


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment_root() -> Path:
    return root() / "experiments" / EXPERIMENT


def harness() -> Path:
    return experiment_root() / "submit_qwen3_30ba3b_cadence_200step.sh"


def config_for(variant: str) -> dict[str, object]:
    return json.loads((experiment_root() / "configs" / f"{variant}.yaml").read_text())


class ContractTest(unittest.TestCase):
    maxDiff = None

    def test_legacy_grpo_forwards_controller_decision_to_policy_train(self) -> None:
        source = (root() / "nemo_rl" / "algorithms" / "grpo.py").read_text()
        policy_train = re.search(
            r"train_results = policy\.train\((.*?)\n\s*\)",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(policy_train)
        assert policy_train is not None
        self.assertIn("draft_update_decision=cadence_decision", policy_train.group(1))

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
        fixture_contents = re.sub(
            r"^readonly SOURCE_ROOT=.*$",
            f"readonly SOURCE_ROOT={temporary_root / 'source'}",
            fixture_contents,
            flags=re.MULTILINE,
        )
        fixture_contents = re.sub(
            r"^readonly SOURCE_SHA=.*$",
            f"readonly SOURCE_SHA={fixture_source_sha}",
            fixture_contents,
            flags=re.MULTILINE,
        )
        fixture_contents = re.sub(
            r"^readonly CONTAINER=.*$",
            f"readonly CONTAINER={temporary_root / 'container.sqsh'}",
            fixture_contents,
            flags=re.MULTILINE,
        )
        fixture_contents = re.sub(
            r"^readonly DURABLE_ROOT=.*$",
            f"readonly DURABLE_ROOT={durable_root}",
            fixture_contents,
            flags=re.MULTILINE,
        )
        fixture_contents = re.sub(
            r"^readonly HARNESS_SHA=.*$",
            f"readonly HARNESS_SHA={fixture_harness_sha}",
            fixture_contents,
            flags=re.MULTILINE,
        )
        fixture_contents = re.sub(
            r"preflight\(\) \{\n.*?\n\}\n\nwrite_sbatch\(\)",
            "preflight() {\n  :\n}\n\nwrite_sbatch()",
            fixture_contents,
            count=1,
            flags=re.DOTALL,
        )
        fixture_harness.write_text(fixture_contents)
        fixture_harness.chmod(0o700)

        config_sha = hashlib.sha256(
            (fixture_root / "configs" / f"{variant}.yaml").read_bytes()
        ).hexdigest()
        preflight_receipt = durable_root / "preflight" / f"{variant}.json"
        preflight_receipt.parent.mkdir(parents=True)
        preflight_receipt.write_text(
            json.dumps(
                {
                    "config_sha": config_sha,
                    "harness_sha": fixture_harness_sha,
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

    def manifest(self, variant: str) -> dict[str, object]:
        result = subprocess.run(
            ["bash", str(harness()), "--emit-manifest", variant],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def render(self, variant: str, temporary: str) -> tuple[str, str]:
        result = subprocess.run(
            ["bash", str(harness()), "--render-sbatch", variant],
            cwd=root(),
            text=True,
            capture_output=True,
            env={**os.environ, "Q30_CADENCE_RENDER_ROOT": temporary},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        sbatch = Path(result.stdout.strip())
        driver = sbatch.parent / "driver.sh"
        self.assertEqual(subprocess.run(["bash", "-n", str(sbatch)]).returncode, 0)
        self.assertEqual(subprocess.run(["bash", "-n", str(driver)]).returncode, 0)
        return sbatch.read_text(), driver.read_text()

    def test_matrix_has_exactly_six_requested_interval_arms(self) -> None:
        for variant in VARIANTS:
            self.assertEqual(self.manifest(variant)["variant"], variant)
        invalid = subprocess.run(
            ["bash", str(harness()), "--emit-manifest", "dflash-static"],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(invalid.returncode, 0)

    def test_configs_only_overlay_interval_drafter_fields(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                config = config_for(variant)
                self.assertEqual(
                    config["defaults"],
                    f"{SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
                )
                self.assertEqual(config["grpo"], {"max_num_steps": 200})
                self.assertNotIn("data", config)
                self.assertNotIn("data_plane", config)
                self.assertNotIn("checkpointing", config)
                self.assertNotIn("cluster", config)
                policy = config["policy"]
                self.assertEqual(policy["model_name"], MODEL)
                for inherited_key in (
                    "train_global_batch_size",
                    "max_total_sequence_length",
                    "make_sequence_length_divisible_by",
                    "sequence_packing",
                    "megatron_cfg",
                ):
                    self.assertNotIn(inherited_key, policy)
                generation = policy["generation"]
                self.assertNotIn("max_new_tokens", generation)
                self.assertNotIn("vllm_cfg", generation)
                self.assertEqual(set(generation["vllm_kwargs"]), {"speculative_config"})

    def test_drafter_and_schedule_are_encoded_exactly(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                drafter, cadence = variant.split("-", 1)
                checkpoint = DFLASH if drafter == "dflash" else DSPARK
                draft = config_for(variant)["policy"]["draft"]
                self.assertEqual(draft["speculator_type"], drafter)
                self.assertEqual(draft["model_name"], checkpoint)
                interval = int(cadence.removeprefix("fixed"))
                self.assertEqual(
                    draft["update_schedule"],
                    {
                        "mode": "fixed",
                        "action": "sparse_update",
                        "fixed_interval": interval,
                    },
                )
                self.assertEqual(
                    draft["optimizer"],
                    {"lr": 5e-6, "min_lr": 5e-7, "weight_decay": 0.01},
                )

    def test_benchmark_uses_wandb_metrics_without_durable_cadence_runtime(
        self,
    ) -> None:
        for variant in VARIANTS:
            config = config_for(variant)
            self.assertEqual(config["cadence_runtime"], {"enabled": False})
            self.assertNotIn("checkpointing", config)

        verifier = (experiment_root() / "verify_composed_configs.py").read_text()
        self.assertNotIn("cadence_runtime.required_checkpoint_steps", verifier)

    def test_manifest_pins_product_identity_and_wandb(self) -> None:
        harness_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root(),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                first = self.manifest(variant)
                second = self.manifest(variant)
                self.assertEqual(
                    first["source"], {"root": SOURCE_ROOT, "sha": SOURCE_SHA}
                )
                self.assertEqual(first["max_steps"], 200)
                self.assertEqual(first["wandb_project"], "sna-specdec")
                self.assertEqual(
                    first["wandb_group"], "q30ba3b-draft-cadence-200step-20260826"
                )
                self.assertTrue(
                    first["wandb_run_id"].startswith(f"q30ba3b-200step-{variant}-k5-")
                )
                self.assertNotEqual(first["wandb_run_id"], second["wandb_run_id"])
                self.assertEqual(
                    first["submission_record"],
                    f"{DURABLE_ROOT}/submissions/{variant}-{SOURCE_SHA}-{harness_sha}.json",
                )

    def test_completed_submission_record_prevents_resubmit(self) -> None:
        variant = "dflash-fixed5"
        original_record = (
            '{"job_output": "Submitted batch job 1", "variant": "dflash-fixed5"}\n'
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            (
                fixture_harness,
                fixture_root,
                submission_record,
                fake_bin,
                _,
            ) = self.prepare_submission_fixture(temporary_root, variant)
            submission_record.parent.mkdir()
            submission_record.write_text(original_record)

            sbatch_log = temporary_root / "sbatch.log"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nprintf submitted >\"${FAKE_SBATCH_LOG}\"\nprintf 'Submitted batch job 2\\n'\n"
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

            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("reconcile", result.stderr.lower())
            self.assertFalse(sbatch_log.exists(), result.stdout + result.stderr)
            self.assertEqual(submission_record.read_text(), original_record)

            submission_record.unlink()
            submission_record.symlink_to(temporary_root / "missing-receipt.json")
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

            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("reconcile", result.stderr.lower())
            self.assertFalse(sbatch_log.exists(), result.stdout + result.stderr)
            self.assertTrue(submission_record.is_symlink())

    def test_nonzero_scheduler_acceptance_persists_ambiguous_receipt(self) -> None:
        variant = "dflash-fixed5"
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            (
                fixture_harness,
                fixture_root,
                submission_record,
                fake_bin,
                fixture_harness_sha,
            ) = self.prepare_submission_fixture(temporary_root, variant)
            sbatch_log = temporary_root / "sbatch.log"
            pre_sbatch_record = temporary_root / "pre-sbatch-record.json"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'cp "${FAKE_SUBMISSION_RECORD}" "${FAKE_PRE_SBATCH_RECORD}"\n'
                "printf 'scheduler accepted\\n' >>\"${FAKE_SBATCH_LOG}\"\n"
                "printf 'Submitted batch job 4242\\n'\n"
                "printf 'opaque secret: must-not-be-written-to-receipt\\n'\n"
                "printf '%*s' 9000 '' | tr ' ' x\n"
                "exit 70\n"
            )
            fake_sbatch.chmod(0o700)
            environment = {
                **os.environ,
                "FAKE_PRE_SBATCH_RECORD": str(pre_sbatch_record),
                "FAKE_SBATCH_LOG": str(sbatch_log),
                "FAKE_SUBMISSION_RECORD": str(submission_record),
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "WANDB_API_KEY": "must-not-be-written-to-receipt",
            }

            first = subprocess.run(
                ["bash", str(fixture_harness), "--submit", variant],
                cwd=fixture_root,
                text=True,
                capture_output=True,
                env=environment,
            )

            self.assertNotEqual(first.returncode, 0, first.stdout)
            self.assertEqual(sbatch_log.read_text(), "scheduler accepted\n")
            self.assertTrue(submission_record.is_file())
            pre_sbatch = json.loads(pre_sbatch_record.read_text())
            self.assertEqual(pre_sbatch["state"], "submitting")
            self.assert_attempt_identity(pre_sbatch, submission_record)
            ambiguous = json.loads(submission_record.read_text())
            self.assertEqual(ambiguous["state"], "ambiguous")
            self.assertEqual(ambiguous["scheduler_exit_status"], 70)
            self.assertEqual(ambiguous["scheduler_output_bytes"], 9071)
            self.assertEqual(
                ambiguous["scheduler_safe_output"], ["Submitted batch job 4242"]
            )
            self.assertTrue(ambiguous["scheduler_output_truncated"])
            self.assertFalse(ambiguous["scheduler_timed_out"])
            self.assertEqual(ambiguous["candidate_job_ids"], ["4242"])
            self.assertEqual(ambiguous["harness_sha"], fixture_harness_sha)
            self.assert_attempt_identity(ambiguous, submission_record)
            self.assertNotIn(
                "must-not-be-written-to-receipt", submission_record.read_text()
            )
            self.assertNotIn(
                "must-not-be-written-to-receipt", first.stdout + first.stderr
            )

            second = subprocess.run(
                ["bash", str(fixture_harness), "--submit", variant],
                cwd=fixture_root,
                text=True,
                capture_output=True,
                env=environment,
            )

            self.assertNotEqual(second.returncode, 0, second.stdout)
            self.assertIn("reconcile", second.stderr.lower())
            self.assertEqual(sbatch_log.read_text(), "scheduler accepted\n")

    def test_truncated_success_output_persists_ambiguous_receipt(self) -> None:
        variant = "dflash-fixed5"
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            (
                fixture_harness,
                fixture_root,
                submission_record,
                fake_bin,
                fixture_harness_sha,
            ) = self.prepare_submission_fixture(temporary_root, variant)
            sbatch_log = temporary_root / "sbatch.log"
            pre_sbatch_record = temporary_root / "pre-sbatch-record.json"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'cp "${FAKE_SUBMISSION_RECORD}" "${FAKE_PRE_SBATCH_RECORD}"\n'
                "printf 'scheduler accepted\\n' >>\"${FAKE_SBATCH_LOG}\"\n"
                "printf 'Submitted batch job 4242\\n'\n"
                "printf '%*s' 9000 '' | tr ' ' x\n"
                "printf '\\nSubmitted batch job 5252\\n'\n"
            )
            fake_sbatch.chmod(0o700)
            environment = {
                **os.environ,
                "FAKE_PRE_SBATCH_RECORD": str(pre_sbatch_record),
                "FAKE_SBATCH_LOG": str(sbatch_log),
                "FAKE_SUBMISSION_RECORD": str(submission_record),
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "WANDB_API_KEY": "must-not-be-written-to-receipt",
            }

            first = subprocess.run(
                ["bash", str(fixture_harness), "--submit", variant],
                cwd=fixture_root,
                text=True,
                capture_output=True,
                env=environment,
            )

            self.assertNotEqual(first.returncode, 0, first.stdout)
            self.assertIn("reconcile", first.stderr.lower())
            self.assertEqual(sbatch_log.read_text(), "scheduler accepted\n")
            pre_sbatch = json.loads(pre_sbatch_record.read_text())
            self.assertEqual(pre_sbatch["state"], "submitting")
            self.assert_attempt_identity(pre_sbatch, submission_record)
            ambiguous = json.loads(submission_record.read_text())
            self.assertEqual(ambiguous["state"], "ambiguous")
            self.assertEqual(ambiguous["scheduler_exit_status"], 0)
            self.assertEqual(ambiguous["scheduler_output_bytes"], 9051)
            self.assertTrue(ambiguous["scheduler_output_truncated"])
            self.assertEqual(
                ambiguous["scheduler_safe_output"], ["Submitted batch job 4242"]
            )
            self.assertEqual(ambiguous["candidate_job_ids"], ["4242"])
            self.assertEqual(ambiguous["harness_sha"], fixture_harness_sha)
            self.assert_attempt_identity(ambiguous, submission_record)

            second = subprocess.run(
                ["bash", str(fixture_harness), "--submit", variant],
                cwd=fixture_root,
                text=True,
                capture_output=True,
                env=environment,
            )

            self.assertNotEqual(second.returncode, 0, second.stdout)
            self.assertIn("reconcile", second.stderr.lower())
            self.assertEqual(sbatch_log.read_text(), "scheduler accepted\n")

    def test_scheduler_acceptance_atomically_finalizes_receipt(self) -> None:
        variant = "dflash-fixed5"
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            (
                fixture_harness,
                fixture_root,
                submission_record,
                fake_bin,
                fixture_harness_sha,
            ) = self.prepare_submission_fixture(temporary_root, variant)
            sbatch_log = temporary_root / "sbatch.log"
            pre_sbatch_record = temporary_root / "pre-sbatch-record.json"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'cp "${FAKE_SUBMISSION_RECORD}" "${FAKE_PRE_SBATCH_RECORD}"\n'
                "printf 'scheduler accepted\\n' >>\"${FAKE_SBATCH_LOG}\"\n"
                "printf 'Submitted batch job 4242\\n'\n"
                "printf 'opaque secret: must-not-be-written-to-receipt\\n'\n"
            )
            fake_sbatch.chmod(0o700)
            environment = {
                **os.environ,
                "FAKE_PRE_SBATCH_RECORD": str(pre_sbatch_record),
                "FAKE_SBATCH_LOG": str(sbatch_log),
                "FAKE_SUBMISSION_RECORD": str(submission_record),
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "WANDB_API_KEY": "must-not-be-written-to-receipt",
            }

            first = subprocess.run(
                ["bash", str(fixture_harness), "--submit", variant],
                cwd=fixture_root,
                text=True,
                capture_output=True,
                env=environment,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(first.stdout, "Submitted batch job 4242\n")
            self.assertEqual(sbatch_log.read_text(), "scheduler accepted\n")
            pre_sbatch = json.loads(pre_sbatch_record.read_text())
            self.assertEqual(pre_sbatch["state"], "submitting")
            self.assert_attempt_identity(pre_sbatch, submission_record)
            accepted = json.loads(submission_record.read_text())
            self.assertEqual(accepted["state"], "accepted")
            self.assertEqual(accepted["job_id"], "4242")
            self.assertEqual(accepted["scheduler_exit_status"], 0)
            self.assertEqual(accepted["scheduler_output_bytes"], 71)
            self.assertEqual(
                accepted["scheduler_safe_output"], ["Submitted batch job 4242"]
            )
            self.assertFalse(accepted["scheduler_output_truncated"])
            self.assertFalse(accepted["scheduler_timed_out"])
            self.assertEqual(accepted["harness_sha"], fixture_harness_sha)
            self.assert_attempt_identity(accepted, submission_record)
            self.assertNotIn(
                "must-not-be-written-to-receipt", submission_record.read_text()
            )
            self.assertNotIn(
                "must-not-be-written-to-receipt", first.stdout + first.stderr
            )
            self.assertEqual(list(submission_record.parent.glob("*.tmp")), [])

            second = subprocess.run(
                ["bash", str(fixture_harness), "--submit", variant],
                cwd=fixture_root,
                text=True,
                capture_output=True,
                env=environment,
            )

            self.assertNotEqual(second.returncode, 0, second.stdout)
            self.assertIn("reconcile", second.stderr.lower())
            self.assertEqual(sbatch_log.read_text(), "scheduler accepted\n")

    def test_rendered_jobs_preserve_performance_runtime_and_pin_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    sbatch, driver = self.render(variant, temporary)
                    self.assertEqual(
                        re.findall(r"^#SBATCH --nodes=(\d+)$", sbatch, re.MULTILINE),
                        ["4"],
                    )
                    self.assertEqual(
                        re.findall(r"^#SBATCH --segment=(\d+)$", sbatch, re.MULTILINE),
                        ["4"],
                    )
                    self.assertIn("#SBATCH --account=nemotron_n3_post", sbatch)
                    self.assertIn("#SBATCH --partition=batch", sbatch)
                    self.assertIn("#SBATCH --time=04:00:00", sbatch)
                    self.assertIn("#SBATCH --mem=0", sbatch)
                    self.assertIn(
                        'export PATH="/cm/local/apps/slurm/current/bin:${PATH}"',
                        sbatch,
                    )
                    for slurm_command in ("scontrol", "sinfo", "srun"):
                        self.assertIn(
                            f"command -v {slurm_command} >/dev/null",
                            sbatch,
                        )
                    self.assertIn("/raid:/raid", sbatch)
                    self.assertIn("export CPUS_PER_WORKER=64", sbatch.splitlines())
                    self.assertIn("Q30_MCORE_OVERLAY", sbatch)
                    self.assertIn("MCORE_OVERLAY_GATE_PASS", driver)
                    self.assertIn('test -n "${WANDB_API_KEY:-}"', driver)
                    self.assertIn("logger.wandb.project=sna-specdec", driver)
                    self.assertIn(
                        "+logger.wandb.group=q30ba3b-draft-cadence-200step-20260826",
                        driver,
                    )
                    self.assertNotIn("data_plane.enabled=", driver)
                    stable_runtime = (
                        "UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv "
                        "uv run --frozen --no-sync"
                    )
                    self.assertGreaterEqual(driver.count(stable_runtime), 2)
                    for forbidden_override in (
                        "max_num_seqs=",
                        "compilation_config.backend=",
                        "compilation_config.cudagraph_mode=",
                        "compilation_config.cudagraph_capture_sizes=",
                    ):
                        self.assertNotIn(forbidden_override, driver)
                    self.assertIn(
                        "Step[[:space:]]+2[[:space:]]*/[[:space:]]*200", driver
                    )
                    interval = variant.rsplit("fixed", 1)[1]
                    self.assertIn(
                        f"draft_post_update_refit=complete step={interval}",
                        driver,
                    )
                    self.assertIn("DRAFT_REFIT_GATE_PASS", driver)


if __name__ == "__main__":
    unittest.main()
