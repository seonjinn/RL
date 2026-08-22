"""Executable contracts for the OCI-HSG Qwen3-30B-A3B 20-step pair."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


EXPERIMENT = "qwen3_30ba3b_dflash_dspark_20step_20260822"
SOURCE_SHA = "443e7243ae2a235b6dcd8f4918fea86e693630a9"
MODEL_PATH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B"
CONTAINER = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh"
CAPTURE_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]
TARGET_IDS = [1, 12, 23, 34, 45]


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _config(variant: str) -> dict[str, object]:
    path = _root() / "experiments" / EXPERIMENT / "configs" / f"{variant}.yaml"
    try:
        with path.open() as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise AssertionError(f"missing runnable {variant} recipe: {path}") from exc


class ContractTest(unittest.TestCase):
    def test_recipe_contract_is_exact(self) -> None:
        """Changing an experiment-shaping knob must fail before a costly launch."""
        cases = (
            (
                "dflash",
                "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dflash-b8-16n/exported-checkpoint-25391",
                "dflash",
            ),
            (
                "dspark",
                "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dspark-b8-16n/exported-checkpoint-25391",
                "dspark",
            ),
        )
        for variant, checkpoint, method in cases:
            with self.subTest(variant=variant):
                config = _config(variant)
                policy = config["policy"]
                generation = policy["generation"]
                vllm_cfg = generation["vllm_cfg"]
                vllm_kwargs = generation["vllm_kwargs"]
                draft = policy["draft"]
                self.assertEqual(
                    config["defaults"],
                    "../../../examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
                )
                self.assertEqual(
                    config["grpo"],
                    {
                        "max_num_steps": 20,
                        "num_prompts_per_step": 16,
                        "num_generations_per_prompt": 32,
                        "seed": 42,
                        "async_grpo": {"enabled": False},
                    },
                )
                self.assertFalse(config["checkpointing"]["enabled"])
                self.assertEqual(policy["model_name"], MODEL_PATH)
                self.assertEqual(policy["tokenizer"]["name"], MODEL_PATH)
                self.assertEqual(policy["train_global_batch_size"], 512)
                self.assertEqual(policy["max_total_sequence_length"], 8192)
                self.assertEqual(generation["max_new_tokens"], 1024)
                self.assertEqual(vllm_cfg["max_model_len"], 8192)
                self.assertEqual(vllm_cfg["max_num_seqs"], 8)
                self.assertFalse(vllm_cfg["enforce_eager"])
                self.assertEqual(
                    vllm_kwargs["compilation_config"],
                    {"backend": "eager", "cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": CAPTURE_SIZES},
                )
                self.assertEqual(
                    vllm_kwargs["speculative_config"],
                    {"method": method, "model": checkpoint, "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1},
                )
                self.assertTrue(draft["enabled"])
                self.assertEqual(draft["speculator_type"], method)
                self.assertEqual(draft["model_name"], checkpoint)
                self.assertEqual(draft["anchors_per_sample"], 2)
                self.assertEqual(draft["mask_token_id"], 151669)
                self.assertEqual(draft["target_hidden_state_layer_ids"], TARGET_IDS)
                self.assertEqual(draft["num_layers"], 5)
                if variant == "dflash":
                    self.assertEqual(draft["gamma"], 5)
                else:
                    self.assertEqual(draft["block_size"], 8)
                    self.assertEqual(draft["markov_rank"], 256)
                    self.assertEqual(draft["markov_head_type"], "vanilla")
                    self.assertTrue(draft["confidence_enabled"])
                    self.assertTrue(draft["confidence_with_markov"])

    def test_harness_emits_a_fail_closed_oci_submission_manifest(self) -> None:
        """Removing a preflight gate or changing OCI shape makes the launch contract fail."""
        harness = _root() / "experiments" / EXPERIMENT / "submit_qwen3_30ba3b_20step.sh"
        for variant in ("dflash", "dspark"):
            with self.subTest(variant=variant):
                try:
                    result = subprocess.run(
                        ["bash", str(harness), "--emit-manifest", variant],
                        check=True,
                        cwd=_root(),
                        capture_output=True,
                        text=True,
                    )
                except FileNotFoundError as exc:
                    self.fail(f"missing runnable submission harness: {harness} ({exc})")
                manifest = json.loads(result.stdout)
                self.assertEqual(manifest["variant"], variant)
                self.assertEqual(manifest["source_sha"], SOURCE_SHA)
                self.assertEqual(manifest["container"], CONTAINER)
                self.assertEqual(
                    manifest["slurm"],
                    {"partition": "batch", "qos": "normal", "time": "04:00:00", "nodes": 4, "gpus_per_node": 4},
                )
                self.assertEqual(manifest["gates"], ["setup", "state_dict", "cudagraph", "step1", "step2"])
                self.assertEqual(manifest["max_steps"], 20)
                self.assertTrue(manifest["wandb_run_id"].startswith(f"q30-20step-{variant}-"))

    def test_checkpoint_gate_reports_zero_missing_and_unexpected_keys(self) -> None:
        """A checkpoint schema drift must block launch before GPU setup."""
        gate = _root() / "experiments" / EXPERIMENT / "check_checkpoint_state_dict.py"
        for variant, checkpoint in (
            ("dflash", "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dflash-b8-16n/exported-checkpoint-25391"),
            ("dspark", "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dspark-b8-16n/exported-checkpoint-25391"),
        ):
            with self.subTest(variant=variant):
                try:
                    result = subprocess.run(
                        ["python3", str(gate), "--variant", variant, "--checkpoint", checkpoint],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except FileNotFoundError as exc:
                    self.fail(f"missing executable state-dict gate: {gate} ({exc})")
                except subprocess.CalledProcessError as exc:
                    self.fail(f"state-dict gate did not run: {exc.stderr.strip()}")
                self.assertIn("missing=0 unexpected=0", result.stdout)


if __name__ == "__main__":
    unittest.main()
