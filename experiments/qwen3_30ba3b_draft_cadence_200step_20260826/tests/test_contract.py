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
SOURCE_SHA = "55607a6e784b00058587414ab31aa6ea663a4cfd"
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
SEGMENTED_VARIANTS = (BASELINE_VARIANT, *CG2048_VARIANTS)
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
        self,
        temporary_root: Path,
        variant: str,
        *,
        bypass_harness_guard: bool = True,
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
            r'^HARNESS_SHA=.*$',
            f"HARNESS_SHA={fixture_harness_sha}",
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
        if bypass_harness_guard:
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
        self.assertEqual(artifact_dir.name, "artifacts")
        self.assertEqual(receipt["run_id"], artifact_dir.parent.name)
        self.assertEqual(receipt["sbatch_path"], str(artifact_dir / "job.sbatch"))
        self.assertEqual(
            artifact_dir.parent.parent, submission_record.parent.parent / "runs"
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

    def test_segmented_renderer_rejects_every_legacy_and_retry_variant_without_writes(
        self,
    ) -> None:
        legacy_variants = (
            "dflash-static",
            "dflash-always",
            "dflash-fixed5",
            "dflash-fixed10",
            "dflash-fixed20",
            "dflash-fixed20-retry",
            "dflash-adaptive-v2",
            "dspark-static",
            "dspark-always",
            "dspark-always-cg2048-retry",
            "dspark-fixed5",
            "dspark-fixed10",
            "dspark-fixed20",
            "dspark-adaptive-v2",
        )
        with tempfile.TemporaryDirectory() as temporary:
            render_root = Path(temporary) / "render"
            for variant in legacy_variants:
                with self.subTest(variant=variant):
                    result = subprocess.run(
                        ["bash", str(harness()), "--render-sbatch", variant],
                        cwd=root(),
                        text=True,
                        capture_output=True,
                        env={
                            **os.environ,
                            "Q30_CADENCE_RENDER_ROOT": str(render_root),
                        },
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("not approved for segmented", result.stderr)
                    self.assertFalse(render_root.exists())

    def test_testonly_and_submit_reject_untrusted_harness_before_durable_writes(
        self,
    ) -> None:
        variant = "dflash-fixed5-cg2048"
        for git_mode in ("dirty", "unpushed"):
            with self.subTest(git_mode=git_mode), tempfile.TemporaryDirectory() as temporary:
                temporary_root = Path(temporary)
                (
                    fixture_harness,
                    fixture_root,
                    submission_record,
                    fake_bin,
                    _,
                ) = self.prepare_submission_fixture(
                    temporary_root,
                    variant,
                    bypass_harness_guard=False,
                )
                fake_git = fake_bin / "git"
                fake_git.write_text(
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    'case "$*" in\n'
                    '  *"status --porcelain"*) [[ "${FAKE_GIT_MODE}" != dirty ]] || printf "?? dirty\\n" ;;\n'
                    '  *"symbolic-ref --quiet --short HEAD"*) printf "feature\\n" ;;\n'
                    '  *"config --get branch.feature.remote"*) printf "origin\\n" ;;\n'
                    '  *"config --get branch.feature.merge"*) printf "refs/heads/feature\\n" ;;\n'
                    '  *"rev-parse HEAD"*) printf "local-head\\n" ;;\n'
                    '  *"rev-parse refs/remotes/origin/feature"*) printf "remote-head\\n" ;;\n'
                    "esac\n"
                )
                fake_git.chmod(0o700)
                fake_sbatch = fake_bin / "sbatch"
                fake_sbatch.write_text(
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    'printf called >"${FAKE_SBATCH_LOG}"\n'
                    "printf 'Submitted batch job 42\\n'\n"
                )
                fake_sbatch.chmod(0o700)
                sbatch_log = temporary_root / "sbatch.log"
                environment = {
                    **os.environ,
                    "FAKE_GIT_MODE": git_mode,
                    "FAKE_SBATCH_LOG": str(sbatch_log),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                }
                for action in ("--test-only", "--submit"):
                    with self.subTest(action=action):
                        result = subprocess.run(
                            ["bash", str(fixture_harness), action, variant],
                            cwd=fixture_root,
                            text=True,
                            capture_output=True,
                            env=environment,
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("harness", result.stderr.lower())
                        self.assertFalse(sbatch_log.exists())
                        self.assertFalse(submission_record.exists())
                        self.assertFalse(
                            (submission_record.parent.parent / "runs").exists()
                        )

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

    def test_cg2048_matrix_is_one_official_performance_recipe_cohort(self) -> None:
        variants = (BASELINE_VARIANT, *CG2048_VARIANTS)
        for variant in variants:
            with self.subTest(variant=variant):
                config = config_for(variant)
                self.assertEqual(
                    config["defaults"],
                    f"{SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
                )
                self.assertEqual(config["grpo"], {"max_num_steps": 200})
                self.assertEqual(config["cadence_runtime"], {"enabled": False})
                for inherited_key in ("data", "data_plane", "checkpointing", "cluster"):
                    self.assertNotIn(inherited_key, config)

                policy = config["policy"]
                self.assertEqual(policy["model_name"], MODEL)
                self.assertEqual(policy["tokenizer"], {"name": MODEL})
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
                compilation = generation["vllm_kwargs"]["compilation_config"]
                self.assertEqual(compilation["cudagraph_mode"], "FULL_AND_PIECEWISE")
                self.assertEqual(compilation["cudagraph_capture_sizes"][-1], 2048)

                if variant == BASELINE_VARIANT:
                    self.assertNotIn("draft", policy)
                    self.assertEqual(
                        set(generation["vllm_kwargs"]), {"compilation_config"}
                    )
                    continue
                drafter = variant.split("-", maxsplit=1)[0]
                expected_checkpoint = DFLASH if drafter == "dflash" else DSPARK
                self.assertEqual(policy["draft"]["model_name"], expected_checkpoint)

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
                compilation = vllm_kwargs["compilation_config"]
                self.assertEqual(compilation["cudagraph_mode"], "FULL_AND_PIECEWISE")
                self.assertEqual(compilation["cudagraph_capture_sizes"], expected_sizes)
                self.assertIn(768, expected_sizes)
                self.assertEqual(expected_sizes[-1], 2048)
                if variant.startswith("dflash-"):
                    self.assertIn(2046, expected_sizes)
                cadence = variant.split("-", maxsplit=1)[1].removesuffix("-cg2048")
                if cadence in {"fixed5", "fixed10", "fixed20", "adaptive-v2"}:
                    base_variant = variant.removesuffix("-cg2048")
                    base_config = config_for(base_variant)
                    self.assertEqual(
                        set(vllm_kwargs),
                        set(base_config["policy"]["generation"]["vllm_kwargs"])
                        | {"compilation_config"},
                    )
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

    def test_readme_documents_matched_matrix_and_forbids_cross_cohort_comparison(
        self,
    ) -> None:
        readme = (experiment_root() / "README.md").read_text()
        for variant in (BASELINE_VARIANT, *CG2048_VARIANTS):
            with self.subTest(variant=variant):
                self.assertIn(f"`{variant}`", readme)
        for variant in (
            "dflash-static",
            "dflash-always",
            "dspark-static",
            "dspark-always",
        ):
            with self.subTest(historical_variant=variant):
                self.assertIn(f"`{variant}`", readme)
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
            for variant in CG2048_VARIANTS:
                with self.subTest(variant=variant):
                    _, driver = self.render(variant, temporary)
                    checkpoint = config_for(variant)["policy"]["draft"]["model_name"]
                    self.assertIn(f'readonly CHECKPOINT="{checkpoint}"', driver)

    def test_baseline_overlays_matched_runtime_fields(self) -> None:
        config = config_for(BASELINE_VARIANT)
        self.assertEqual(
            config["defaults"],
            f"{SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
        )
        self.assertEqual(config["grpo"], {"max_num_steps": 200})
        self.assertEqual(config["cadence_runtime"], {"enabled": False})
        policy = config["policy"]
        self.assertEqual(policy["model_name"], MODEL)
        self.assertIs(policy["offload_optimizer_for_refit"], False)
        self.assertEqual(policy["tokenizer"], {"name": MODEL})
        self.assertEqual(
            set(policy["generation"]["vllm_kwargs"]), {"compilation_config"}
        )
        compilation = policy["generation"]["vllm_kwargs"]["compilation_config"]
        self.assertEqual(compilation["cudagraph_mode"], "FULL_AND_PIECEWISE")
        self.assertEqual(compilation["cudagraph_capture_sizes"][-1], 2048)
        self.assertNotIn("draft", config["policy"])

        verifier = (experiment_root() / "verify_composed_configs.py").read_text()
        self.assertIn('if variant == "baseline":', verifier)
        self.assertIn(
            'assert set(vllm_kwargs) == {"moe_backend", "compilation_config"}',
            verifier,
        )
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
        self.assertIn("cadence_runtime.required_checkpoint_steps", verifier)
        self.assertIn("cadence_runtime.result_dir", harness().read_text())

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
                        "account": "nemotron_n4_post",
                        "gpus_per_node": 4,
                        "nodes": 4,
                        "partition": "batch",
                        "time": "04:00:00",
                    },
                )
                self.assertEqual(first["wandb_project"], "sna-specdec")
                self.assertEqual(
                    first["wandb_group"], "q30ba3b-draft-cadence-200step-20260826"
                )
                self.assertTrue(
                    first["wandb_run_id"].startswith(f"q30ba3b-200step-{variant}-k5-")
                )
                self.assertEqual(first["wandb_run_id"], second["wandb_run_id"])
                self.assertEqual(
                    first["submission_record"],
                    f"{DURABLE_ROOT}/submissions/{variant}-{SOURCE_SHA}-{harness_sha}.json",
                )

    def test_segmented_batch_manifest_and_render_share_one_logical_run(self) -> None:
        variant = "dflash-fixed5-cg2048"
        first = self.manifest(variant)
        second = self.manifest(variant)
        self.assertEqual(first["wandb_run_id"], second["wandb_run_id"])
        self.assertEqual(first["checkpoint_root"], second["checkpoint_root"])
        self.assertEqual(
            first["slurm"],
            {
                "account": "nemotron_n4_post",
                "gpus_per_node": 4,
                "nodes": 4,
                "partition": "batch",
                "time": "04:00:00",
            },
        )
        self.assertEqual(first["segments"], 5)
        self.assertEqual(first["afterok_segments"], 4)
        self.assertEqual(first["wandb_resume"], ["allow", "must", "must", "must", "must"])
        self.assertEqual(
            first["checkpointing"],
            {
                "checkpoint_must_save_by": "00:02:45:00",
                "enabled": True,
                "ft_keep_latest_k": 2,
                "ft_save_period": 20,
                "keep_top_k": 1,
                "metric_name": None,
                "save_optimizer": True,
                "save_period": 200,
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            sbatch, driver = self.render(variant, temporary)
            checkpoint_root = Path(str(first["checkpoint_root"]))
            rendered_checkpoint_root = checkpoint_root.relative_to(DURABLE_ROOT)
            expected_root = Path(temporary) / rendered_checkpoint_root
            self.assertIn("#SBATCH --account=nemotron_n4_post", sbatch)
            self.assertIn("#SBATCH --partition=batch", sbatch)
            self.assertIn("#SBATCH --time=04:00:00", sbatch)
            self.assertIn('readonly SEGMENT_INDEX="${Q30_SEGMENT_INDEX:-0}"', driver)
            self.assertIn(f'readonly CHECKPOINT_ROOT="{expected_root}"', driver)
            self.assertIn('export WANDB_RESUME="allow"', driver)
            self.assertIn('export WANDB_RESUME="must"', driver)
            self.assertIn("checkpointing.enabled=true", driver)
            self.assertIn("checkpointing.save_optimizer=true", driver)
            self.assertIn("checkpointing.save_period=200", driver)
            self.assertIn("checkpointing.keep_top_k=1", driver)
            self.assertIn("checkpointing.metric_name=null", driver)
            self.assertIn("++checkpointing.ft_save_period=20", driver)
            self.assertIn("++checkpointing.ft_keep_latest_k=2", driver)
            self.assertIn(
                "checkpointing.checkpoint_must_save_by=00:02:45:00", driver
            )
            self.assertIn("prepare_mcore_checkpoint_overlay.py", sbatch)
            self.assertIn(
                "patches/mcore-precision-aware-lazy-state-checkpoint.patch", sbatch
            )
            self.assertIn("MCORE_CHECKPOINT_OVERLAY_GATE_PASS", driver)
            self.assertIn("mcore-checkpoint-overlay-receipt.json", driver)

    def test_segmented_runtime_uses_one_result_root_for_cadence_and_checkpoints(
        self,
    ) -> None:
        variant = "dflash-fixed5-cg2048"
        manifest = self.manifest(variant)
        self.assertIn("result_root", manifest)
        result_root = Path(str(manifest["result_root"]))
        self.assertEqual(Path(str(manifest["checkpoint_root"])), result_root / "checkpoints")
        self.assertEqual(
            Path(str(manifest["completion_receipt"])),
            result_root / "completion-receipt.json",
        )
        with tempfile.TemporaryDirectory() as temporary:
            _, driver = self.render(variant, temporary)
        rendered_result_root = Path(temporary) / result_root.relative_to(DURABLE_ROOT)
        rendered_checkpoint_root = rendered_result_root / "checkpoints"
        self.assertGreaterEqual(
            driver.count(f"++cadence_runtime.result_dir={rendered_result_root}"), 2
        )
        self.assertGreaterEqual(
            driver.count("++cadence_runtime.required_checkpoint_steps=[200]"), 2
        )
        self.assertGreaterEqual(
            driver.count(f"checkpointing.checkpoint_dir={rendered_checkpoint_root}"),
            2,
        )
        self.assertIn("--override", driver)

    def test_completion_receipt_requires_bound_terminal_checkpoint_artifacts(
        self,
    ) -> None:
        variant = BASELINE_VARIANT
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            sbatch, driver = self.render(variant, temporary)
            helpers = list(
                temporary_root.glob("runs/*/artifacts/completion_receipt.py")
            )
            self.assertEqual(len(helpers), 1)
            helper = helpers[0]
            result_root = helper.parents[1]
            checkpoint = result_root / "checkpoints" / "step_200"
            weights = checkpoint / "policy" / "weights"
            optimizer = checkpoint / "policy" / "optimizer"
            weights.mkdir(parents=True)
            optimizer.mkdir()
            (weights / "weights.bin").write_bytes(b"weights")
            (optimizer / "state.bin").write_bytes(b"optimizer")
            (checkpoint / "train_dataloader.pt").write_bytes(b"dataloader")
            (checkpoint / "training_info.json").write_text(
                json.dumps({"total_steps": 200}) + "\n"
            )
            (checkpoint / "config.yaml").write_text("checkpoint: terminal\n")

            absent = subprocess.run(
                ["python3", str(helper), "validate"],
                text=True,
                capture_output=True,
            )
            self.assertEqual(absent.returncode, 3, absent.stderr)
            written = subprocess.run(
                ["python3", str(helper), "write"],
                text=True,
                capture_output=True,
            )
            self.assertEqual(written.returncode, 0, written.stderr)
            receipt_path = result_root / "completion-receipt.json"
            self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
            valid = subprocess.run(
                ["python3", str(helper), "validate"],
                text=True,
                capture_output=True,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)

            receipt = json.loads(receipt_path.read_text())
            for artifact in (
                "policy_weights",
                "policy_optimizer",
                "train_dataloaders",
            ):
                self.assertIn("sha256", json.dumps(receipt["artifacts"][artifact]))
            (weights / "weights.bin").write_bytes(b"WEIGHTS")
            content_tampered = subprocess.run(
                ["python3", str(helper), "validate"],
                text=True,
                capture_output=True,
            )
            self.assertNotIn(content_tampered.returncode, (0, 3))
            self.assertIn("binding differs", content_tampered.stderr)
            (weights / "weights.bin").write_bytes(b"weights")
            self.assertEqual(
                subprocess.run(
                    ["python3", str(helper), "validate"], capture_output=True
                ).returncode,
                0,
            )
            receipt["source_sha"] = "tampered"
            receipt_path.write_text(json.dumps(receipt) + "\n")
            malformed = subprocess.run(
                ["python3", str(helper), "validate"],
                text=True,
                capture_output=True,
            )
            self.assertNotIn(malformed.returncode, (0, 3))
            self.assertIn("invalid completion receipt", malformed.stderr)
            self.assertRegex(sbatch, r'completion_receipt\.py"? validate')
            self.assertIn("completion receipt is malformed", sbatch)
            write_match = re.search(r'completion_receipt\.py"? write', driver)
            self.assertIsNotNone(write_match)
            assert write_match is not None
            self.assertGreater(write_match.start(), driver.rindex('wait "${train_pid}"'))

    def test_cadence_enabled_completion_requires_terminal_evidence_and_receipt(
        self,
    ) -> None:
        variant = "dflash-always-cg2048"
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            self.render(variant, temporary)
            helpers = list(
                temporary_root.glob("runs/*/artifacts/completion_receipt.py")
            )
            self.assertEqual(len(helpers), 1)
            helper = helpers[0]
            checkpoint = helper.parents[1] / "checkpoints" / "step_200"
            weights = checkpoint / "policy" / "weights"
            optimizer = checkpoint / "policy" / "optimizer"
            weights.mkdir(parents=True)
            optimizer.mkdir()
            (weights / "weights.bin").write_bytes(b"weights")
            (optimizer / "state.bin").write_bytes(b"optimizer")
            (checkpoint / "train_dataloader.pt").write_bytes(b"dataloader")
            (checkpoint / "training_info.json").write_text(
                json.dumps({"total_steps": 200}) + "\n"
            )
            (checkpoint / "config.yaml").write_text("checkpoint: terminal\n")
            missing = subprocess.run(
                ["python3", str(helper), "write"],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("cadence terminal evidence", missing.stderr)

            evidence = {"decision_id": 200, "terminal": True}
            (checkpoint / "training_info.json").write_text(
                json.dumps(
                    {"total_steps": 200, "draft_terminal_evidence": evidence}
                )
                + "\n"
            )
            cadence_receipt = {
                "successful": True,
                "checkpoint_path": str(checkpoint),
                "current_step": 200,
                "cadence_terminal_evidence": evidence,
            }
            (checkpoint / "cadence-checkpoint-receipt.json").write_text(
                json.dumps(cadence_receipt) + "\n"
            )
            missing_terminal_close = subprocess.run(
                ["python3", str(helper), "write"],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(missing_terminal_close.returncode, 0)
            self.assertIn("cadence terminal closure", missing_terminal_close.stderr)

            result_root = helper.parents[1]
            (result_root / "checkpoint-runtime.json").write_text(
                json.dumps(cadence_receipt) + "\n"
            )
            (result_root / "schedule-runtime.json").write_text(
                json.dumps({"mode": "always", "current_step": 200}) + "\n"
            )
            complete = subprocess.run(
                ["python3", str(helper), "write"],
                text=True,
                capture_output=True,
            )
            self.assertEqual(complete.returncode, 0, complete.stderr)

    def test_optimizer_metadata_without_payload_cannot_seal_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            self.render(BASELINE_VARIANT, temporary)
            helper = next(
                temporary_root.glob("runs/*/artifacts/completion_receipt.py")
            )
            checkpoint = helper.parents[1] / "checkpoints" / "step_200"
            weights = checkpoint / "policy" / "weights"
            iteration = weights / "iter_0000007"
            iteration.mkdir(parents=True)
            (weights / "weights.bin").write_bytes(b"weights")
            (weights / "latest_checkpointed_iteration.txt").write_text("7\n")
            (iteration / "metadata.json").write_text('{"iteration": 7}\n')
            (checkpoint / "train_dataloader.pt").write_bytes(b"dataloader")
            (checkpoint / "training_info.json").write_text(
                json.dumps({"total_steps": 200}) + "\n"
            )
            (checkpoint / "config.yaml").write_text("checkpoint: terminal\n")

            metadata_only = subprocess.run(
                ["python3", str(helper), "write"], text=True, capture_output=True
            )
            self.assertNotEqual(metadata_only.returncode, 0)
            self.assertIn("optimizer payload", metadata_only.stderr)
            (iteration / "shard_0.distcp").write_bytes(b"optimizer")
            sealed = subprocess.run(
                ["python3", str(helper), "write"], text=True, capture_output=True
            )
            self.assertEqual(sealed.returncode, 0, sealed.stderr)

    def test_final_segment_fails_when_step_200_receipt_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, driver = self.render(BASELINE_VARIANT, temporary)
        self.assertIn("if (( SEGMENT_INDEX == 4 )); then", driver)
        self.assertIn("final segment ended without valid Step-200", driver)

    def test_submit_builds_one_arm_local_five_segment_afterok_chain(self) -> None:
        variant = "dflash-fixed5-cg2048"
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            (
                fixture_harness,
                fixture_root,
                submission_record,
                fake_bin,
                _,
            ) = self.prepare_submission_fixture(temporary_root, variant)
            sbatch_log = temporary_root / "sbatch.log"
            counter = temporary_root / "counter"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'printf \'%s\\n\' \"$*\" >>\"${FAKE_SBATCH_LOG}\"\n'
                'count=0; test ! -f "${FAKE_COUNTER}" || count="$(cat "${FAKE_COUNTER}")"\n'
                'printf \'%s\\n\' "$((count + 1))" >"${FAKE_COUNTER}"\n'
                'printf \'Submitted batch job %s\\n\' "$((4100 + count))"\n'
            )
            fake_sbatch.chmod(0o700)
            result = subprocess.run(
                ["bash", str(fixture_harness), "--submit", variant],
                cwd=fixture_root,
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "FAKE_COUNTER": str(counter),
                    "FAKE_SBATCH_LOG": str(sbatch_log),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = sbatch_log.read_text().splitlines()
            self.assertEqual(len(calls), 5)
            self.assertNotIn("--dependency", calls[0])
            self.assertIn("--export=ALL,Q30_SEGMENT_INDEX=0", calls[0])
            for segment in range(1, 5):
                self.assertIn(
                    f"--dependency=afterok:{4099 + segment}", calls[segment]
                )
                self.assertIn(
                    f"--export=ALL,Q30_SEGMENT_INDEX={segment}", calls[segment]
                )
            receipt = json.loads(submission_record.read_text())
            self.assertEqual(receipt["state"], "accepted")
            self.assertEqual(
                receipt["job_ids"], ["4100", "4101", "4102", "4103", "4104"]
            )

    def test_completed_checkpoint_short_circuits_before_cluster_setup(self) -> None:
        variant = BASELINE_VARIANT
        with tempfile.TemporaryDirectory() as temporary:
            sbatch_text, driver_text = self.render(variant, temporary)
            checkpoint_match = re.search(
                r'^readonly CHECKPOINT_ROOT="([^"]+)"$', driver_text, re.MULTILINE
            )
            self.assertIsNotNone(checkpoint_match)
            assert checkpoint_match is not None
            checkpoint = Path(checkpoint_match.group(1)) / "step_200"
            checkpoint.mkdir(parents=True)
            (checkpoint / "training_info.json").write_text(
                json.dumps({"total_steps": 200}) + "\n"
            )
            helper = next(Path(temporary).glob("runs/*/artifacts/completion_receipt.py"))
            absent = subprocess.run(
                ["python3", str(helper), "validate"], capture_output=True
            )
            self.assertEqual(absent.returncode, 3)
            weights = checkpoint / "policy" / "weights"
            optimizer = checkpoint / "policy" / "optimizer"
            weights.mkdir(parents=True)
            optimizer.mkdir()
            (weights / "weights.bin").write_bytes(b"weights")
            (optimizer / "state.bin").write_bytes(b"optimizer")
            (checkpoint / "train_dataloader.pt").write_bytes(b"dataloader")
            (checkpoint / "config.yaml").write_text("checkpoint: terminal\n")
            sealed = subprocess.run(
                ["python3", str(helper), "write"], text=True, capture_output=True
            )
            self.assertEqual(sealed.returncode, 0, sealed.stderr)
            sbatch = next(Path(temporary).glob("runs/*/artifacts/job.sbatch"))
            result = subprocess.run(
                ["bash", str(sbatch)],
                text=True,
                capture_output=True,
                env={**os.environ, "Q30_SEGMENT_INDEX": "3"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("validated atomic completion receipt", result.stdout)
            self.assertIn("Q30_SEGMENT_COMPLETE_BEFORE_RAY", sbatch_text)
            self.assertLess(
                sbatch_text.index("Q30_SEGMENT_COMPLETE_BEFORE_RAY"),
                sbatch_text.index('exec bash "'),
            )

    def test_resume_segment_gates_follow_checkpoint_instead_of_early_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, driver = self.render("dflash-fixed5-cg2048", temporary)
        self.assertIn('readonly GATES_LOG="${SEGMENT_DIR}/gates.log"', driver)
        self.assertIn("if (( RESUME_STEP == 0 )); then", driver)
        self.assertIn(
            "wait_for_gate 'Step[[:space:]]+1[[:space:]]*/[[:space:]]*200' "
            "STEP1_GATE_PASS 2700",
            driver,
        )
        self.assertIn("RESUME_CHECKPOINT_LOAD_GATE_PASS", driver)
        self.assertIn("Checkpoint loaded", driver)
        self.assertIn("NEXT_STEP=$((RESUME_STEP + 1))", driver)
        self.assertIn("RESUME_NEXT_STEP_GATE_PASS", driver)
        self.assertNotIn('tee -a "${ARTIFACT_DIR}/gates.log"', driver)

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

    def test_dflash_fixed20_retry_manifest_is_not_segmented_runnable(self) -> None:
        manifest = self.manifest(DFLASH_FIXED20_RETRY)
        self.assertTrue(
            manifest["wandb_run_id"].startswith(
                "q30ba3b-200step-dflash-fixed20-retry-k5-"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                ["bash", str(harness()), "--render-sbatch", DFLASH_FIXED20_RETRY],
                text=True,
                capture_output=True,
                env={**os.environ, "Q30_CADENCE_RENDER_ROOT": temporary},
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not approved for segmented", result.stderr)

    def test_dspark_always_retry_manifest_is_not_segmented_runnable(self) -> None:
        manifest = self.manifest(DSPARK_ALWAYS_CG2048_RETRY)
        self.assertTrue(
            manifest["wandb_run_id"].startswith(
                "q30ba3b-200step-dspark-always-cg2048-retry-k5-"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    "bash",
                    str(harness()),
                    "--render-sbatch",
                    DSPARK_ALWAYS_CG2048_RETRY,
                ],
                text=True,
                capture_output=True,
                env={**os.environ, "Q30_CADENCE_RENDER_ROOT": temporary},
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not approved for segmented", result.stderr)

    def test_dspark_cg2048_online_arms_exclude_oom_node(self) -> None:
        variants = (
            "dspark-always-cg2048",
            "dspark-fixed5-cg2048",
            "dspark-fixed10-cg2048",
            "dspark-fixed20-cg2048",
            "dspark-adaptive-v2-cg2048",
        )
        with tempfile.TemporaryDirectory() as temporary:
            for variant in variants:
                with self.subTest(variant=variant):
                    sbatch, _ = self.render(variant, temporary)
                    self.assertIn("#SBATCH --exclude=nvl72118-T01", sbatch)

    def test_completed_submission_record_prevents_resubmit(self) -> None:
        variant = "dflash-fixed5-cg2048"
        original_record = (
            '{"job_output": "Submitted batch job 1", "variant": "dflash-fixed5-cg2048"}\n'
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
        variant = "dflash-fixed5-cg2048"
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
        variant = "dflash-fixed5-cg2048"
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
        variant = "dflash-fixed5-cg2048"
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
            self.assertEqual(first.stdout, "Submitted batch job 4242\n" * 5)
            self.assertEqual(sbatch_log.read_text(), "scheduler accepted\n" * 5)
            pre_sbatch = json.loads(pre_sbatch_record.read_text())
            self.assertEqual(pre_sbatch["state"], "submitting")
            self.assert_attempt_identity(pre_sbatch, submission_record)
            accepted = json.loads(submission_record.read_text())
            self.assertEqual(accepted["state"], "accepted")
            self.assertEqual(accepted["job_id"], "4242")
            self.assertEqual(accepted["job_ids"], ["4242"] * 5)
            self.assertEqual(accepted["scheduler_exit_status"], 0)
            self.assertEqual(accepted["scheduler_output_bytes"], 355)
            self.assertEqual(
                accepted["scheduler_safe_output"], ["Submitted batch job 4242"] * 5
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
            self.assertEqual(sbatch_log.read_text(), "scheduler accepted\n" * 5)

    def test_rendered_jobs_preserve_performance_runtime_and_pin_submission(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for variant in tuple(f"{item}-cg2048" for item in VARIANTS):
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
                    self.assertIn("#SBATCH --account=nemotron_n4_post", sbatch)
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
                    interval = variant.removesuffix("-cg2048").rsplit("fixed", 1)[1]
                    self.assertIn(
                        f"wait_for_gate 'draft_post_update_refit=complete step={interval}' DRAFT_REFIT_GATE_PASS 0",
                        driver,
                    )
                    self.assertIn("DRAFT_REFIT_GATE_PASS", driver)


if __name__ == "__main__":
    unittest.main()
