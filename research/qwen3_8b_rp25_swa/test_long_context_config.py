"""Resolve the long-context launch config and run its actual chunking validator."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from research.qwen3_8b_rp25_swa.study import build_new_arms, overrides


ROOT = Path(__file__).resolve().parents[2]


class LongContextConfigTests(unittest.TestCase):
    def test_resolved_configs_pass_chunking_and_length_constraints(self) -> None:
        register_omegaconf_resolvers()
        tree = ast.parse((ROOT / "nemo_rl/models/megatron/setup.py").read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_validate_chunking_config"
        )
        namespace = {"PolicyConfig": dict}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "setup.py", "exec"),
            namespace,
        )
        for arm in build_new_arms():
            if arm.cadence not in ("baseline", "static"):
                continue
            with self.subTest(arm=arm.name):
                config = parse_hydra_overrides(
                    load_config(ROOT / arm.config_path),
                    list(overrides(arm, "/lustre/test", long_context=True)),
                )
                OmegaConf.resolve(config)
                namespace["_validate_chunking_config"](config.policy)
                self.assertEqual(config.policy.train_global_batch_size, 512)
                self.assertEqual(
                    config.grpo.num_prompts_per_step
                    * config.grpo.num_generations_per_prompt,
                    512,
                )
                self.assertEqual(
                    config.data.max_input_seq_length
                    + config.policy.generation.max_new_tokens,
                    32768,
                )
                self.assertEqual(config.policy.max_total_sequence_length, 32768)
                self.assertFalse(config.policy.draft.enabled)
                self.assertEqual(config.grpo.max_num_steps, 3)
                if arm.drafter == "none":
                    self.assertNotIn(
                        "max_num_seqs", config.policy.generation.vllm_kwargs
                    )
                    self.assertNotIn(
                        "cudagraph_capture_sizes",
                        config.policy.generation.vllm_kwargs.compilation_config,
                    )
                else:
                    self.assertEqual(
                        config.policy.generation.vllm_kwargs.max_num_seqs, 8
                    )


if __name__ == "__main__":
    unittest.main()
