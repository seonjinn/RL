from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "experiments/q35_math_specdec_20260916/launch.py"


def run(arm: str, concurrency: str, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LAUNCHER), "--config-json", arm, concurrency, *extra],
        capture_output=True,
        text=True,
        check=False,
    )


class LaunchTest(unittest.TestCase):
    def config(self, arm: str, concurrency: str) -> dict[str, str]:
        result = run(arm, concurrency)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_all_arms_require_graphs_and_requested_moe_backend(self) -> None:
        for arm in ("baseline", "dflash", "dspark"):
            for concurrency in ("default", "64"):
                config = self.config(arm, concurrency)
                self.assertEqual(
                    config["policy.generation.vllm_cfg.enforce_eager"], "false"
                )
                self.assertEqual(
                    config["policy.generation.vllm_kwargs.moe_backend"],
                    "flashinfer_trtllm",
                )
                self.assertEqual(
                    config[
                        "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode"
                    ],
                    "FULL_AND_PIECEWISE",
                )
                self.assertEqual(config["policy.draft.enabled"], "false")

    def test_concurrency_does_not_change_workload_or_graph_envelope(self) -> None:
        for arm in ("baseline", "dflash", "dspark"):
            default = self.config(arm, "default")
            limited = self.config(arm, "64")
            self.assertEqual(
                limited.pop("policy.generation.vllm_kwargs.max_num_seqs"), "64"
            )
            self.assertEqual(default, limited)
            self.assertNotIn("policy.train_global_batch_size", default)
            self.assertNotIn("grpo.num_prompts_per_step", default)

    def test_baseline_has_no_drafter_and_specdec_uses_q35_replay_exports(self) -> None:
        baseline = self.config("baseline", "64")
        self.assertEqual(
            baseline["policy.generation.vllm_kwargs.speculative_config"], "null"
        )
        for method in ("dflash", "dspark"):
            cfg = self.config(method, "64")
            self.assertEqual(
                cfg["policy.generation.vllm_kwargs.speculative_config.method"], method
            )
            self.assertEqual(
                cfg[
                    "policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens"
                ],
                "5",
            )
            self.assertIn(
                f"sd2p3rp-q35-a3b-ptv3rp25-{method}-b8-16n/exported-checkpoint-44000",
                cfg["policy.generation.vllm_kwargs.speculative_config.model"],
            )

    def test_graph_envelope_covers_all_128_requests_per_engine(self) -> None:
        for arm, widths in (
            ("baseline", (1,)),
            ("dflash", (1, 6)),
            ("dspark", (1, 5, 6)),
        ):
            cfg = self.config(arm, "default")
            sizes = json.loads(
                cfg[
                    "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes"
                ]
            )
            for width in widths:
                for requests in range(1, 129):
                    eligible = [
                        s for s in sizes if s % width == 0 and s >= requests * width
                    ]
                    self.assertTrue(eligible)
                    self.assertLessEqual(min(eligible), 2 * requests * width)

    def test_invalid_inputs_rejected(self) -> None:
        for arm, limit, extra in (
            ("dflash2", "64", ()),
            ("baseline", "0", ()),
            ("baseline", "64", ("--steps", "0")),
        ):
            self.assertNotEqual(run(arm, limit, *extra).returncode, 0)

    def test_specdec_submission_requires_target_lineage_confirmation(self) -> None:
        result = subprocess.run(
            [sys.executable, str(LAUNCHER), "--submit", "dspark", "64"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("target lineage", result.stderr)

    def test_resolved_recipe_preserves_training_and_requires_requested_runtime(
        self,
    ) -> None:
        from omegaconf import OmegaConf
        from nemo_rl.utils.config import (
            load_config,
            parse_hydra_overrides,
            register_omegaconf_resolvers,
        )

        register_omegaconf_resolvers()
        recipe = (
            ROOT
            / "examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml"
        )
        for arm in ("baseline", "dflash", "dspark"):
            for limit in ("default", "64"):
                values = self.config(arm, limit)
                cfg = parse_hydra_overrides(
                    load_config(recipe), [f"++{k}={v}" for k, v in values.items()]
                )
                OmegaConf.resolve(cfg)
                self.assertEqual(cfg.policy.train_global_batch_size, 512)
                self.assertEqual(
                    cfg.grpo.num_prompts_per_step * cfg.grpo.num_generations_per_prompt,
                    512,
                )
                self.assertEqual(cfg.policy.max_total_sequence_length, 4096)
                self.assertEqual(cfg.policy.megatron_cfg.tensor_model_parallel_size, 2)
                self.assertEqual(cfg.policy.megatron_cfg.context_parallel_size, 2)
                self.assertEqual(cfg.policy.megatron_cfg.expert_model_parallel_size, 16)
                self.assertEqual(cfg.policy.generation.vllm_cfg.tensor_parallel_size, 4)
                self.assertFalse(cfg.policy.generation.vllm_cfg.enforce_eager)
                self.assertEqual(cfg.cluster.num_nodes * cfg.cluster.gpus_per_node, 16)
                self.assertFalse(
                    cfg.grpo.val_at_start or cfg.grpo.val_at_end or cfg.grpo.val_period
                )


if __name__ == "__main__":
    unittest.main()
