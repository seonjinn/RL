from __future__ import annotations

import shlex
import unittest

from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from test_launcher import ROOT, render


class ResolvedConfigTest(unittest.TestCase):
    def test_no_validation_dataset_does_not_request_validation(self) -> None:
        register_omegaconf_resolvers()
        recipe = (
            ROOT
            / "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
        )
        for arm in ("baseline", "dflash_k5", "dspark_k5"):
            for concurrency in (16, 32, 64, 128):
                with self.subTest(arm=arm, concurrency=concurrency):
                    result = render(arm, str(concurrency), steps=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    line = next(
                        line
                        for line in result.stdout.splitlines()
                        if line.startswith("export COMMAND=")
                    )
                    command = shlex.split(line.removeprefix("export COMMAND="))[0]
                    tokens = shlex.split(command)
                    cli = tokens[tokens.index("--config") + 2 :]
                    config = parse_hydra_overrides(load_config(recipe), cli)
                    OmegaConf.resolve(config)
                    self.assertIsNone(config.data.validation)
                    self.assertEqual(config.data.train.split_validation_size, 0)
                    validation_requested = (
                        config.grpo.val_period > 0
                        or config.grpo.val_at_start
                        or config.grpo.val_at_end
                    )
                    self.assertFalse(
                        validation_requested,
                        "GRPO setup requires a validation dataset when any validation trigger is enabled",
                    )
                    self.assertEqual(config.policy.train_global_batch_size, 2048)
                    self.assertEqual(
                        config.grpo.num_prompts_per_step
                        * config.grpo.num_generations_per_prompt,
                        2048,
                    )
                    self.assertEqual(config.policy.generation.max_new_tokens, 47104)
                    self.assertEqual(
                        config.policy.generation.vllm_kwargs.max_num_seqs, concurrency
                    )


if __name__ == "__main__":
    unittest.main()
