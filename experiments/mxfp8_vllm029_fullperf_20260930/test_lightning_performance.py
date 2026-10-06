"""Check the eight matched Lightning configurations before GPU training."""

import fnmatch
import os
from pathlib import Path
import unittest

from omegaconf import OmegaConf

from nemo_rl.utils.config import register_omegaconf_resolvers
from preflight import ARMS, MODES, PROTECTED_KEYS, render_config, validate_config


class LightningPerformanceTest(unittest.TestCase):
    def test_matched_precision_matrix(self) -> None:
        register_omegaconf_resolvers()
        root = Path.cwd()
        for mode in MODES:
            baseline = None
            for arm in ARMS:
                with self.subTest(mode=mode, arm=arm):
                    config, original, fields = render_config(
                        root, "lightning", mode, arm
                    )
                    validate_config(config, original, fields, "lightning", mode, arm)
                    self.assertEqual(config.grpo.num_prompts_per_step, 64)
                    self.assertEqual(config.grpo.num_generations_per_prompt, 8)
                    self.assertEqual(config.policy.train_global_batch_size, 512)
                    self.assertEqual(config.policy.max_total_sequence_length, 4096)
                    self.assertEqual(config.cluster.num_nodes, 8)
                    self.assertEqual(
                        config.policy.generation.vllm_cfg.tensor_parallel_size, 4
                    )
                    self.assertEqual(
                        config.policy.generation.vllm_cfg.expert_parallel_size, 4
                    )
                    protected = {
                        key: OmegaConf.to_container(
                            OmegaConf.create({"value": OmegaConf.select(config, key)}),
                            resolve=True,
                        )["value"]
                        for key in PROTECTED_KEYS
                    }
                    if baseline is None:
                        baseline = protected
                    self.assertEqual(protected, baseline)
                    if arm.endswith("-mxfp8"):
                        vllm = config.policy.generation.vllm_cfg
                        self.assertEqual(vllm.num_first_layers_in_bf16, 2)
                        self.assertEqual(vllm.num_last_layers_in_bf16, 6)
                        patterns = list(vllm.quantization_ignore_patterns)
                        self.assertEqual(
                            patterns,
                            list(
                                original.policy.generation.vllm_cfg.quantization_ignore_patterns
                            ),
                        )
                        for prefix in ("model", "backbone"):
                            for projection in (
                                "qkv_proj",
                                "o_proj",
                                "in_proj",
                                "out_proj",
                                "gate",
                                "shared_experts.up_proj",
                                "shared_experts.down_proj",
                            ):
                                name = f"{prefix}.layers.4.mixer.{projection}"
                                self.assertTrue(
                                    any(
                                        fnmatch.fnmatchcase(name, pattern)
                                        for pattern in patterns
                                    ),
                                    name,
                                )
                            for projection in ("experts.up_proj", "experts.down_proj"):
                                name = f"{prefix}.layers.4.mixer.{projection}"
                                self.assertFalse(
                                    any(
                                        fnmatch.fnmatchcase(name, pattern)
                                        for pattern in patterns
                                    ),
                                    name,
                                )
                    if arm.startswith("mxfp8-"):
                        megatron = config.policy.megatron_cfg
                        self.assertTrue(megatron.first_last_layers_bf16)
                        self.assertEqual(megatron.num_layers_at_start_in_bf16, 2)
                        self.assertEqual(megatron.num_layers_at_end_in_bf16, 6)
                    print(f"LIGHTNING_CONFIG_PASS {mode}/{arm}", flush=True)

    @unittest.skipUnless(
        os.environ.get("LIGHTNING_SNAPSHOT"),
        "Set the existing local snapshot to test offline tokenizer loading",
    )
    def test_offline_tokenizer(self) -> None:
        from transformers import AutoConfig, AutoTokenizer

        snapshot = os.environ["LIGHTNING_SNAPSHOT"]
        config = AutoConfig.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=True
        )
        self.assertTrue(tokenizer.encode("A short offline tokenizer check."))
        self.assertGreater(config.num_hidden_layers, 8)
        print(
            f"LIGHTNING_TOKENIZER_PASS model_type={config.model_type} layers={config.num_hidden_layers}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
