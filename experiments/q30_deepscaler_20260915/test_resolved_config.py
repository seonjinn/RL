from __future__ import annotations

import shlex
import unittest

from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from test_launcher import LAUNCHER, ROOT, render


class ResolvedConfigTest(unittest.TestCase):
    def test_workload_constraints_hold_after_inheritance(self) -> None:
        register_omegaconf_resolvers()
        for arm, concurrency, steps in (
            ("baseline", "default", 3),
            ("baseline", "default", 20),
            ("dflash_k5", "64", 20),
            ("dspark_k5", "64", 20),
        ):
            with self.subTest(arm=arm, steps=steps):
                result = render(LAUNCHER, arm, concurrency, steps)
                self.assertEqual(result.returncode, 0, result.stderr)
                line = next(
                    x
                    for x in result.stdout.splitlines()
                    if x.startswith("export COMMAND=")
                )
                tokens = shlex.split(
                    shlex.split(line.removeprefix("export COMMAND="))[0]
                )
                cli = tokens[tokens.index("--config") + 2 :]
                config = parse_hydra_overrides(
                    load_config(
                        ROOT
                        / "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
                    ),
                    cli,
                )
                OmegaConf.resolve(config)
                self.assertEqual(config.data.train.dataset_name, "DeepScaler")
                self.assertEqual(config.env.math.math_verify_impl, "hf_math_verify")
                self.assertEqual(
                    config.policy.train_global_batch_size,
                    config.grpo.num_prompts_per_step
                    * config.grpo.num_generations_per_prompt,
                )
                self.assertEqual(config.policy.train_global_batch_size, 2048)
                self.assertTrue(config.policy.megatron_cfg.defer_fp32_logits)
                self.assertLessEqual(
                    config.policy.logprob_chunk_size,
                    config.policy.max_total_sequence_length,
                )
                self.assertEqual(
                    config.data.max_input_seq_length
                    + config.policy.generation.max_new_tokens,
                    config.policy.max_total_sequence_length,
                )
                self.assertEqual(config.policy.max_total_sequence_length, 40960)
                self.assertIsNone(config.data.validation)
                self.assertFalse(
                    config.grpo.val_period > 0
                    or config.grpo.val_at_start
                    or config.grpo.val_at_end
                )


if __name__ == "__main__":
    unittest.main()
