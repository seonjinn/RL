"""Fail-closed contracts for the Qwen3-8B DAPO OSL32K training pilot."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


EXPERIMENT = "qwen3_8b_dapo_osl32k_pilot_20260823"
TARGET = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "hf_home/hub/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
DFLASH = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "hf_home/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/"
    "9b41424b7109f9c5413454f481b09a82b85333f4"
)
DSPARK = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "hf_home/hub/models--deepseek-ai--dspark_qwen3_8b_block7/snapshots/"
    "03326e5043815da1f81b109078b2889737c26017"
)
VARIANTS = {
    "baseline-k0": (None, None),
    "dflash-k5": ("dflash", DFLASH),
    "dspark-k5": ("dspark", DSPARK),
}
CAPTURE_SIZES = [1, 2, 4, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48, 56, 64]


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment() -> Path:
    return root() / "research" / EXPERIMENT


class PilotContractTest(unittest.TestCase):
    maxDiff = None

    def config(self, variant: str) -> dict[str, object]:
        return json.loads((experiment() / "configs" / f"{variant}.yaml").read_text())

    def test_three_arms_are_matched_cp1_two_step_training_runs(self) -> None:
        for variant, (method, checkpoint) in VARIANTS.items():
            with self.subTest(variant=variant):
                config = self.config(variant)
                self.assertEqual(config["grpo"]["max_num_steps"], 2)
                self.assertEqual(config["grpo"]["num_prompts_per_step"], 2)
                self.assertEqual(config["grpo"]["num_generations_per_prompt"], 4)
                self.assertEqual(config["grpo"]["seed"], 42)
                policy = config["policy"]
                self.assertEqual(policy["model_name"], TARGET)
                self.assertEqual(policy["tokenizer"], {"name": TARGET})
                self.assertEqual(policy["train_global_batch_size"], 8)
                self.assertEqual(policy["max_total_sequence_length"], 40960)
                self.assertEqual(policy["sequence_packing"], {"enabled": False})
                megatron = policy["megatron_cfg"]
                self.assertEqual(megatron["tensor_model_parallel_size"], 2)
                self.assertEqual(megatron["pipeline_model_parallel_size"], 1)
                self.assertEqual(megatron["context_parallel_size"], 1)
                self.assertFalse(megatron["sequence_parallel"])
                self.assertTrue(megatron["activation_checkpointing"])
                generation = policy["generation"]
                self.assertEqual(generation["max_new_tokens"], 32768)
                self.assertEqual(generation["vllm_cfg"]["max_model_len"], 40960)
                self.assertEqual(
                    generation["vllm_kwargs"]["compilation_config"]["cudagraph_capture_sizes"],
                    CAPTURE_SIZES,
                )
                if method is None:
                    self.assertNotIn("draft", policy)
                    self.assertNotIn("speculative_config", generation["vllm_kwargs"])
                    continue
                draft = policy["draft"]
                spec = generation["vllm_kwargs"]["speculative_config"]
                self.assertEqual(draft["model_name"], checkpoint)
                self.assertEqual(spec["method"], method)
                self.assertEqual(spec["model"], checkpoint)
                self.assertEqual(spec["num_speculative_tokens"], 5)
                if method == "dflash":
                    self.assertEqual(draft["gamma"], 5)
                else:
                    self.assertEqual(draft["block_size"], 7)
                    self.assertNotIn("gamma", draft)

    def test_model_length_has_k5_lookahead_headroom(self) -> None:
        self.assertGreaterEqual(40960, 2048 + 32768 + 5 + 1)


if __name__ == "__main__":
    unittest.main()
