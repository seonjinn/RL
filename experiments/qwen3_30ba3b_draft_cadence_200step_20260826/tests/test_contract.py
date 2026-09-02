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
SOURCE_ROOT = "/home/sna/nemorl-q30-cadence-syncfix-product-20260902"
SOURCE_SHA = "9be09f0eb9120e37ab9e4e51ecca98f11d9814da"
DURABLE_ROOT = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_draft_cadence_200step_20260826"
MODEL = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B"
DFLASH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dflash/exported-checkpoint-25391"
DSPARK = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dspark/exported-checkpoint-25391"
INTERVALS = (5, 10, 20)
BASELINE_VARIANT = "baseline"
DFLASH_FIXED20_RETRY = "dflash-fixed20-retry"
DSPARK_ALWAYS_CG2048_RETRY = "dspark-always-cg2048-retry"
VARIANTS = tuple(
    f"{drafter}-fixed{interval}"
    for drafter in ("dflash", "dspark")
    for interval in INTERVALS
)
CG2048_CADENCES = ("static", "always", "fixed5", "fixed10", "fixed20", "adaptive-v2")
CG2048_VARIANTS = tuple(
    f"{drafter}-{cadence}-cg2048"
    for drafter in ("dflash", "dspark")
    for cadence in CG2048_CADENCES
)
PAIRABLE_VARIANTS = tuple(
    f"{drafter}-{cadence}"
    for drafter in ("dflash", "dspark")
    for cadence in CG2048_CADENCES
)
DEFAULT_CAPTURE_SIZES = (
    1,
    2,
    4,
    *range(8, 256, 8),
    *range(256, 513, 16),
)
EXTENDED_CAPTURE_SIZES = (
    576,
    640,
    704,
    768,
    832,
    896,
    960,
    1024,
    1280,
    1536,
    1792,
    2048,
)
DFLASH_CAPTURE_SIZES = (
    DEFAULT_CAPTURE_SIZES
    + EXTENDED_CAPTURE_SIZES[:-1]
    + (
        2046,
        2048,
    )
)
DSPARK_CAPTURE_SIZES = DEFAULT_CAPTURE_SIZES + EXTENDED_CAPTURE_SIZES


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
        shutil.copytree(experiment_root() / "patches", fixture_root / "patches")
        for filename in (
            "check_checkpoint_state_dict.py",
            "prepare_vllm_dspark_fap_overlay.py",
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

    def test_matrix_keeps_all_six_requested_interval_arms(self) -> None:
        for variant in VARIANTS:
            self.assertEqual(self.manifest(variant)["variant"], variant)
        invalid = subprocess.run(
            ["bash", str(harness()), "--emit-manifest", "dflash-unknown"],
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
                self.assertIs(policy["offload_optimizer_for_refit"], False)
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

    def test_static_and_always_avoid_optimizer_cpu_copy_during_refit(self) -> None:
        for drafter in ("dflash", "dspark"):
            for cadence in ("static", "always"):
                for suffix in ("", "-cg2048"):
                    variant = f"{drafter}-{cadence}{suffix}"
                    with self.subTest(variant=variant):
                        policy = config_for(variant)["policy"]
                        self.assertIs(policy["offload_optimizer_for_refit"], False)

    def test_cg2048_variants_are_exact_compilation_only_siblings(self) -> None:
        for variant in CG2048_VARIANTS:
            with self.subTest(variant=variant):
                expected_sizes = list(
                    DFLASH_CAPTURE_SIZES
                    if variant.startswith("dflash-")
                    else DSPARK_CAPTURE_SIZES
                )
                manifest = self.manifest(variant)
                self.assertEqual(manifest["variant"], variant)
                config = config_for(variant)
                generation = config["policy"]["generation"]
                vllm_kwargs = generation["vllm_kwargs"]
                base_variant = variant.removesuffix("-cg2048")
                base_config = config_for(base_variant)
                self.assertEqual(
                    set(vllm_kwargs),
                    set(base_config["policy"]["generation"]["vllm_kwargs"])
                    | {"compilation_config"},
                )
                compilation = vllm_kwargs["compilation_config"]
                self.assertEqual(compilation["cudagraph_mode"], "FULL_AND_PIECEWISE")
                self.assertEqual(compilation["cudagraph_capture_sizes"], expected_sizes)
                self.assertIn(768, expected_sizes)
                self.assertEqual(expected_sizes[-1], 2048)
                if variant.startswith("dflash-"):
                    self.assertIn(2046, expected_sizes)
                del vllm_kwargs["compilation_config"]
                self.assertEqual(config, base_config)

    def test_launcher_allowlists_each_paired_base_and_cg2048_sibling(self) -> None:
        for variant in PAIRABLE_VARIANTS + CG2048_VARIANTS:
            with self.subTest(variant=variant):
                manifest = self.manifest(variant)
                self.assertEqual(manifest["variant"], variant)
                self.assertTrue(
                    manifest["wandb_run_id"].startswith(
                        f"q30ba3b-200step-{variant}-k5-"
                    )
                )

    def test_readme_documents_every_pair_and_forbids_cross_cohort_comparison(
        self,
    ) -> None:
        readme = (experiment_root() / "README.md").read_text()
        for variant in PAIRABLE_VARIANTS:
            with self.subTest(variant=variant):
                self.assertIn(f"`{variant}`", readme)
                self.assertIn(f"`{variant}-cg2048`", readme)
        self.assertIn(
            "Do not compare the legacy fixed-vs-always cohort directly with the "
            "official performance-recipe cohort.",
            readme,
        )

    def test_cg2048_render_selects_its_config_and_truthful_refit_gate(self) -> None:
        expected_refit_step = {
            "static": None,
            "always": 1,
            "fixed5": 5,
            "fixed10": 10,
            "fixed20": 20,
            "adaptive-v2": None,
        }
        with tempfile.TemporaryDirectory() as temporary:
            for variant in CG2048_VARIANTS:
                with self.subTest(variant=variant):
                    _, driver = self.render(variant, temporary)
                    cadence = variant.split("-", 1)[1].removesuffix("-cg2048")
                    self.assertIn(f"resolved-input-{variant}.yaml", driver)
                    refit_step = expected_refit_step[cadence]
                    if refit_step is None:
                        self.assertNotIn("DRAFT_REFIT_GATE_PASS", driver)
                    else:
                        self.assertIn(
                            "wait_for_gate "
                            f"'draft_post_update_refit=complete step={refit_step}' "
                            "DRAFT_REFIT_GATE_PASS 0",
                            driver,
                        )

    def test_render_uses_checkpoint_declared_by_selected_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for variant in PAIRABLE_VARIANTS + CG2048_VARIANTS:
                with self.subTest(variant=variant):
                    _, driver = self.render(variant, temporary)
                    checkpoint = config_for(variant.removesuffix("-cg2048"))["policy"][
                        "draft"
                    ]["model_name"]
                    self.assertIn(f'readonly CHECKPOINT="{checkpoint}"', driver)

    def test_baseline_only_overlays_step_count_and_local_target(self) -> None:
        config = config_for(BASELINE_VARIANT)
        self.assertEqual(
            config["defaults"],
            f"{SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
        )
        self.assertEqual(config["grpo"], {"max_num_steps": 200})
        self.assertEqual(config["cadence_runtime"], {"enabled": False})
        self.assertEqual(
            config["policy"],
            {
                "model_name": MODEL,
                "offload_optimizer_for_refit": False,
                "tokenizer": {"name": MODEL},
            },
        )
        self.assertNotIn("generation", config["policy"])
        self.assertNotIn("draft", config["policy"])

        verifier = (experiment_root() / "verify_composed_configs.py").read_text()
        self.assertIn('if variant == "baseline":', verifier)
        self.assertIn('assert set(vllm_kwargs) == {"moe_backend"}', verifier)
        self.assertIn("assert config.policy.draft.enabled is False", verifier)
        self.assertIn(
            "assert config.policy.offload_optimizer_for_refit is False", verifier
        )

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

        verifier = (experiment_root() / "verify_composed_configs.py").read_text()
        self.assertIn(
            "expected_block_size = 8 if legacy_fixed_vs_always else 5", verifier
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
        self.assertNotIn("cadence_runtime.result_dir", harness().read_text())

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
                self.assertEqual(
                    first["slurm"],
                    {
                        "account": "nemotron_n3_post",
                        "gpus_per_node": 4,
                        "nodes": 4,
                        "partition": "batch_long",
                        "time": "18:00:00",
                    },
                )
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

    def test_baseline_manifest_and_render_disable_specdec(self) -> None:
        manifest = self.manifest(BASELINE_VARIANT)
        self.assertTrue(
            manifest["wandb_run_id"].startswith("q30ba3b-200step-baseline-k0-")
        )
        self.assertEqual(manifest["max_steps"], 200)
        self.assertNotIn("state-dict", manifest["gates"])
        self.assertNotIn("draft-refit", manifest["gates"])

        with tempfile.TemporaryDirectory() as temporary:
            sbatch, driver = self.render(BASELINE_VARIANT, temporary)
        self.assertIn('export Q30_DRAFTER="none"', sbatch)
        self.assertIn('readonly DRAFTER="none"', driver)
        self.assertIn("resolved-input-baseline.yaml", driver)
        self.assertNotIn("check_checkpoint_state_dict.py", driver)
        self.assertNotIn("DRAFT_REFIT_GATE_PASS", driver)
        self.assertNotIn("NRL_VENV_POST_SYNC_SCRIPT", sbatch)

    def test_dflash_fixed20_retry_reuses_config_and_excludes_bad_node(self) -> None:
        manifest = self.manifest(DFLASH_FIXED20_RETRY)
        self.assertTrue(
            manifest["wandb_run_id"].startswith(
                "q30ba3b-200step-dflash-fixed20-retry-k5-"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            sbatch, driver = self.render(DFLASH_FIXED20_RETRY, temporary)
        self.assertIn("#SBATCH --exclude=nvl72047-T16", sbatch)
        self.assertIn('export Q30_DRAFTER="dflash"', sbatch)
        self.assertIn("resolved-input-dflash-fixed20.yaml", driver)
        self.assertIn(
            "wait_for_gate 'draft_post_update_refit=complete step=20' "
            "DRAFT_REFIT_GATE_PASS 0",
            driver,
        )

    def test_dspark_always_retry_reuses_config_and_excludes_oom_node(self) -> None:
        manifest = self.manifest(DSPARK_ALWAYS_CG2048_RETRY)
        self.assertTrue(
            manifest["wandb_run_id"].startswith(
                "q30ba3b-200step-dspark-always-cg2048-retry-k5-"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            sbatch, driver = self.render(DSPARK_ALWAYS_CG2048_RETRY, temporary)
        self.assertIn("#SBATCH --exclude=nvl72118-T01", sbatch)
        self.assertIn('export Q30_DRAFTER="dspark"', sbatch)
        self.assertIn("resolved-input-dspark-always-cg2048.yaml", driver)
        self.assertIn(
            "wait_for_gate 'draft_post_update_refit=complete step=1' "
            "DRAFT_REFIT_GATE_PASS 0",
            driver,
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

    def test_rendered_jobs_preserve_performance_runtime_and_pin_submission(
        self,
    ) -> None:
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
                    self.assertIn("#SBATCH --partition=batch_long", sbatch)
                    self.assertIn("#SBATCH --time=18:00:00", sbatch)
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
                    drafter = variant.split("-", 1)[0]
                    self.assertIn(f'export Q30_DRAFTER="{drafter}"', sbatch)
                    self.assertIn("Q30_VLLM_OVERLAY", sbatch)
                    self.assertIn("NEMO_RL_VENV_DIR", sbatch)
                    self.assertIn("export UV_HTTP_TIMEOUT=300", sbatch)
                    self.assertIn("export UV_HTTP_RETRIES=10", sbatch)
                    self.assertIn(
                        'export PYTHONPATH="${Q30_VLLM_OVERLAY}:${Q30_MCORE_OVERLAY}:${SOURCE_ROOT}:${PYTHONPATH:-}"',
                        sbatch,
                    )
                    artifact_match = re.search(
                        r'^readonly ARTIFACT_DIR="([^"]+)"$',
                        driver,
                        re.MULTILINE,
                    )
                    self.assertIsNotNone(artifact_match)
                    assert artifact_match is not None
                    self.assertTrue(
                        (
                            Path(artifact_match.group(1))
                            / "patches"
                            / "vllm-0.25.1-pr48167-runtime.patch"
                        ).is_file()
                    )
                    self.assertTrue(
                        (
                            Path(artifact_match.group(1))
                            / "patches"
                            / "vllm-0.25.1-pr48167-group-causality-followup.patch"
                        ).is_file()
                    )
                    self.assertNotIn('/opt/nemo_rl_venv/bin/python "', sbatch)
                    self.assertIn("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH", sbatch)
                    if drafter == "dspark":
                        self.assertIn("prepare_vllm_dspark_fap_overlay.py", sbatch)
                        self.assertIn('export NRL_VENV_POST_SYNC_SCRIPT="', sbatch)
                        self.assertIn(
                            "export NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
                            sbatch,
                        )
                        self.assertIn("DSPARK_VLLM_OVERLAY_GATE_PASS", driver)
                        self.assertIn("dspark-fap-vllm-48167-runtime.json", driver)
                        self.assertIn("vllm-dspark-fap-overlay-receipt.json", driver)
                        self.assertIn(
                            "8e5ff0e385ee44cf71e1e07031e5cd19658b29eb7b90bc172a4754c599d1dd90",
                            driver,
                        )
                        self.assertNotIn("import vllm", driver)
                    else:
                        self.assertNotIn("NRL_VENV_POST_SYNC_SCRIPT", sbatch)
                        self.assertIn("STOCK_VLLM_GATE_PASS", driver)
                        self.assertNotIn("import vllm", driver)
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
                    self.assertIn(
                        "wait_for_gate 'Capturing CUDA graphs.*100%|Graph capturing finished' CUDAGRAPH_GATE_PASS 2700",
                        driver,
                    )
                    self.assertIn(
                        "wait_for_gate 'Step[[:space:]]+1[[:space:]]*/[[:space:]]*200' STEP1_GATE_PASS 2700",
                        driver,
                    )
                    self.assertIn(
                        "wait_for_gate 'Step[[:space:]]+2[[:space:]]*/[[:space:]]*200' STEP2_GATE_PASS 2700",
                        driver,
                    )
                    interval = variant.rsplit("fixed", 1)[1]
                    self.assertIn(
                        f"wait_for_gate 'draft_post_update_refit=complete step={interval}' DRAFT_REFIT_GATE_PASS 0",
                        driver,
                    )
                    self.assertIn("DRAFT_REFIT_GATE_PASS", driver)


if __name__ == "__main__":
    unittest.main()
