"""Executable contracts for the Qwen3-30B-A3B draft-cadence matrix."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


EXPERIMENT = "qwen3_30ba3b_draft_cadence_200step_20260826"
SOURCE_ROOT = "/home/sna/nemorl-q30-cadence-product-20260826"
SOURCE_SHA = "1be8237816bfd78dad752dd5c1e0149ae2420301"
MODEL = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B"
DFLASH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dflash-b8-16n/exported-checkpoint-25391"
DSPARK = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dspark-b8-16n/exported-checkpoint-25391"
CAPTURE_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]
VARIANTS = (
    "dflash-static",
    "dflash-always",
    "dflash-fixed10",
    "dspark-static",
    "dspark-always",
    "dspark-fixed10",
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

    def test_matrix_has_exactly_six_requested_arms(self) -> None:
        for variant in VARIANTS:
            self.assertEqual(self.manifest(variant)["variant"], variant)
        invalid = subprocess.run(
            ["bash", str(harness()), "--emit-manifest", "dflash-fixed5"],
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
                self.assertEqual(config["grpo"]["max_num_steps"], 200)
                self.assertEqual(config["grpo"]["num_prompts_per_step"], 16)
                self.assertEqual(config["grpo"]["num_generations_per_prompt"], 32)
                self.assertTrue(config["data_plane"]["enabled"])
                self.assertEqual(config["data"]["train"]["dataset_name"], "OpenMathInstruct-2")
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
                self.assertEqual(policy["generation"]["max_new_tokens"], 1024)
                self.assertEqual(config["cluster"], {"gpus_per_node": 4, "num_nodes": 4, "segment_size": 4})

    def test_drafter_and_schedule_are_encoded_exactly(self) -> None:
        schedules = {
            "static": {"mode": "fixed", "action": "sparse_update", "fixed_interval": 201},
            "always": {"mode": "always"},
            "fixed10": {"mode": "fixed", "action": "sparse_update", "fixed_interval": 10},
        }
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                drafter, cadence = variant.split("-", 1)
                checkpoint = DFLASH if drafter == "dflash" else DSPARK
                draft = config_for(variant)["policy"]["draft"]
                self.assertEqual(draft["speculator_type"], drafter)
                self.assertEqual(draft["model_name"], checkpoint)
                self.assertEqual(draft["update_schedule"], schedules[cadence])
                self.assertEqual(
                    draft["optimizer"],
                    {"lr": 5e-6, "min_lr": 5e-7, "weight_decay": 0.01},
                )

    def test_cadence_runtime_has_only_a_terminal_checkpoint(self) -> None:
        for variant in VARIANTS:
            config = config_for(variant)
            self.assertEqual(
                config["cadence_runtime"],
                {"enabled": True, "result_dir": f"/tmp/{variant}", "required_checkpoint_steps": [200]},
            )
            self.assertEqual(
                config["checkpointing"],
                {
                    "enabled": True,
                    "save_period": 200,
                    "keep_top_k": None,
                    "metric_name": None,
                    "save_optimizer": True,
                    "checkpoint_dir": f"/tmp/{variant}/checkpoints",
                },
            )

    def test_manifest_pins_product_identity_and_wandb(self) -> None:
        for variant in VARIANTS:
            first = self.manifest(variant)
            second = self.manifest(variant)
            self.assertEqual(first["source"], {"root": SOURCE_ROOT, "sha": SOURCE_SHA})
            self.assertEqual(first["max_steps"], 200)
            self.assertEqual(first["wandb_project"], "sna-specdec")
            self.assertEqual(first["wandb_group"], "q30ba3b-draft-cadence-200step-20260826")
            self.assertTrue(first["wandb_run_id"].startswith(f"q30ba3b-200step-{variant}-k5-"))
            self.assertNotEqual(first["wandb_run_id"], second["wandb_run_id"])

    def test_rendered_jobs_pin_slurm_wandb_and_cuda_graph_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    sbatch, driver = self.render(variant, temporary)
                    self.assertEqual(re.findall(r"^#SBATCH --nodes=(\d+)$", sbatch, re.MULTILINE), ["4"])
                    self.assertEqual(re.findall(r"^#SBATCH --segment=(\d+)$", sbatch, re.MULTILINE), ["4"])
                    self.assertIn("#SBATCH --account=nemotron_sw_post", sbatch)
                    self.assertIn("#SBATCH --partition=batch", sbatch)
                    self.assertIn("#SBATCH --time=08:00:00", sbatch)
                    self.assertIn('export MOUNTS="/lustre:/lustre,/home:/home"', sbatch)
                    self.assertIn('test -n "${WANDB_API_KEY:-}"', driver)
                    self.assertIn("logger.wandb.project=sna-specdec", driver)
                    self.assertIn("logger.wandb.group=q30ba3b-draft-cadence-200step-20260826", driver)
                    self.assertIn("data_plane.enabled=true", driver)
                    self.assertIn("cudagraph_mode=PIECEWISE", driver)
                    self.assertIn("cudagraph_capture_sizes=[1,2,4,8,12,16,24,32,40,48]", driver)
                    self.assertIn("Step[[:space:]]+2[[:space:]]*/[[:space:]]*200", driver)

    def test_capture_buckets_cover_all_runtime_batch_shapes(self) -> None:
        result = subprocess.run(
            ["bash", str(harness()), "--assert-capture-coverage"],
            cwd=root(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        coverage = json.loads(result.stdout)
        self.assertEqual(coverage["capture_sizes"], CAPTURE_SIZES)
        self.assertEqual(set(map(int, coverage["shape_to_bucket"])), set(range(1, 49)))


if __name__ == "__main__":
    unittest.main()
