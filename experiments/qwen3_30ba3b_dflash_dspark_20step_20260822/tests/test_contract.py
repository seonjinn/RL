"""Executable contracts for the replacement Qwen3-30B-A3B 20-step pair."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


EXPERIMENT = "qwen3_30ba3b_dflash_dspark_20step_20260822"
SOURCE_ROOT = "/home/sna/nemorl-pr11-final-df9"
SOURCE_SHA = "df9daf62fe4625609b3a71abd7179007cd6970f9"
MODEL = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B"
DFLASH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dflash-b8-16n/exported-checkpoint-25391"
DSPARK = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dspark-b8-16n/exported-checkpoint-25391"
CAPTURE_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def harness() -> Path:
    return root() / "experiments" / EXPERIMENT / "submit_qwen3_30ba3b_20step.sh"


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

    def test_configs_pin_the_matched_20_step_recipe(self) -> None:
        for variant, checkpoint, method in (
            ("dflash", DFLASH, "dflash"),
            ("dspark", DSPARK, "dspark"),
        ):
            with self.subTest(variant=variant):
                path = root() / "experiments" / EXPERIMENT / "configs" / f"{variant}.yaml"
                self.assertTrue(path.is_file(), f"missing committed {variant} config: {path}")
                config = json.loads(path.read_text())
                self.assertEqual(config["defaults"], f"{SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml")
                self.assertEqual(config["grpo"], {"max_num_steps": 20, "num_prompts_per_step": 16, "num_generations_per_prompt": 32, "seed": 42, "async_grpo": {"enabled": False}})
                self.assertFalse(config["data"]["shuffle"])
                self.assertEqual(config["data"]["train"], {"dataset_name": "OpenMathInstruct-2", "split_validation_size": 0, "seed": 42})
                self.assertFalse(config["checkpointing"]["enabled"])
                policy = config["policy"]
                self.assertEqual(policy["model_name"], MODEL)
                self.assertEqual(policy["tokenizer"]["name"], MODEL)
                self.assertEqual(policy["train_global_batch_size"], 512)
                self.assertEqual(policy["max_total_sequence_length"], 8192)
                self.assertEqual(policy["megatron_cfg"], {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "expert_model_parallel_size": 16})
                generation = policy["generation"]
                self.assertEqual(generation["max_new_tokens"], 1024)
                self.assertEqual(generation["vllm_cfg"], {"tensor_parallel_size": 1, "max_model_len": 8192, "enforce_eager": False})
                self.assertEqual(generation["vllm_kwargs"]["speculative_config"], {"method": method, "model": checkpoint, "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1})
                self.assertEqual(policy["draft"]["model_name"], checkpoint)
                self.assertEqual(policy["draft"]["speculator_type"], method)
                self.assertEqual(policy["draft"]["anchors_per_sample"], 2)
                self.assertEqual(policy["draft"]["mask_token_id"], 151669)
                self.assertEqual(policy["draft"]["target_hidden_state_layer_ids"], [1, 12, 23, 34, 45])
                self.assertEqual(policy["draft"]["num_layers"], 5)
                if variant == "dflash":
                    self.assertEqual(policy["draft"]["gamma"], 5)
                else:
                    self.assertEqual(policy["draft"]["block_size"], 8)
                    self.assertEqual(policy["draft"]["markov_rank"], 256)
                    self.assertEqual(policy["draft"]["markov_head_type"], "vanilla")
                    self.assertTrue(policy["draft"]["confidence_enabled"])
                    self.assertTrue(policy["draft"]["confidence_with_markov"])

    def test_harness_pins_clean_df9_and_never_reuses_wandb_ids(self) -> None:
        first = self.manifest("dflash")
        second = self.manifest("dflash")
        self.assertEqual(first["source"], {"root": SOURCE_ROOT, "sha": SOURCE_SHA})
        self.assertEqual(first["slurm"], {"account": "nemotron_n4_post", "partition": "batch", "qos": "normal", "time": "04:00:00", "nodes": 4, "gpus_per_node": 4})
        self.assertEqual(first["gates"], ["source-clean", "state-dict", "cudagraph", "step1", "step2"])
        self.assertEqual(first["wandb_reuse"], "never")
        self.assertNotEqual(first["wandb_run_id"], second["wandb_run_id"])
        self.assertTrue(first["wandb_run_id"].startswith("q30-20step-dflash-"))
        script = harness().read_text()
        self.assertIn("--untracked-files=all", script)
        self.assertIn("submodule status --recursive", script)
        self.assertIn(SOURCE_ROOT, script)
        self.assertNotIn("443e7243", script)

    def test_capture_buckets_cover_every_runtime_shape(self) -> None:
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
        self.assertTrue(all(bucket in CAPTURE_SIZES for bucket in coverage["shape_to_bucket"].values()))

    def test_rendered_jobs_are_receipt_gated_and_account_correct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = {**os.environ, "Q30_20STEP_RENDER_ROOT": temporary}
            rendered: list[tuple[str, str]] = []
            for variant in ("dflash", "dspark"):
                result = subprocess.run(
                    ["bash", str(harness()), "--render-sbatch", variant],
                    cwd=root(),
                    text=True,
                    capture_output=True,
                    env=environment,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                path = Path(result.stdout.strip())
                self.assertTrue(path.is_file())
                self.assertEqual(subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True).returncode, 0)
                driver = path.parent / "driver.sh"
                self.assertTrue(driver.is_file())
                self.assertEqual(subprocess.run(["bash", "-n", str(driver)], capture_output=True, text=True).returncode, 0)
                rendered.append((path.read_text(), driver.read_text()))
            for sbatch, driver in rendered:
                self.assertIn("#SBATCH --account=nemotron_n4_post", sbatch)
                self.assertIn("#SBATCH --partition=batch", sbatch)
                self.assertIn("#SBATCH --qos=normal", sbatch)
                self.assertIn("#SBATCH --time=04:00:00", sbatch)
                self.assertIn("UV_CACHE_DIR_OVERRIDE", sbatch)
                self.assertIn("UV_PROJECT_ENVIRONMENT", sbatch)
                self.assertIn("check_checkpoint_state_dict.py", driver)
                self.assertIn("verify_df9_configs.py", driver)
                self.assertIn("CUDAGRAPH_GATE_PASS", driver)
                self.assertIn("STEP1_GATE_PASS", driver)
                self.assertIn("STEP2_GATE_PASS", driver)
                self.assertIn("++policy.generation.vllm_kwargs.max_num_seqs=8", driver)
                self.assertIn("++policy.generation.vllm_kwargs.compilation_config.backend=eager", driver)
                self.assertIn("++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE", driver)
                self.assertIn("++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=[1,2,4,8,12,16,24,32,40,48]", driver)
            self.assertIn("--test-only", harness().read_text())
            preflight = harness().read_text().split("write_sbatch()", maxsplit=1)[0]
            self.assertNotIn("verify_df9_configs.py", preflight)


if __name__ == "__main__":
    unittest.main()
