import copy
import unittest

from experiments.supervl35_option_b_20261010.build_rollout_ab_configs import (
    build_configs,
)


class RolloutAbConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bf16 = {
            "policy": {
                "generation": {
                    "vllm_cfg": {
                        "precision": "bfloat16",
                        "is_mx": False,
                        "quantization_ignore_patterns": [],
                    },
                    "vllm_kwargs": {
                        "kernel_config": {"moe_backend": "flashinfer_trtllm"},
                        "moe_backend": "flashinfer_trtllm",
                    },
                }
            }
        }
        self.moe = copy.deepcopy(self.bf16)
        cfg = self.moe["policy"]["generation"]["vllm_cfg"]
        cfg["precision"] = "fp8"
        cfg["is_mx"] = True
        cfg["quantization_ignore_patterns"] = [
            "*model.layers.*.mixer.in_proj",
            "*model.layers.*.mixer.out_proj",
            "*model.layers.*.mixer.qkv_proj",
            "*model.layers.*.mixer.o_proj",
            "*model.layers.*.mixer.up_proj",
            "*model.layers.*.mixer.down_proj",
            "*model.layers.*.mixer.gate",
            "*model.layers.*.mixer.shared_experts.*",
            "*model.layers.*.mixer.fc1_latent_proj",
            "*model.layers.*.mixer.fc2_latent_proj",
            "*model.layers.0.*",
            "*model.layers.80.*",
            "*mtp*",
            "*lm_head",
            "*vision_model.*",
            "*sound_encoder.*",
        ]

    def test_scopes_change_only_rollout_precision_and_backend(self) -> None:
        configs = build_configs(self.bf16, self.moe)
        self.assertEqual(
            set(configs),
            {"bf16", "moe", "qkvo_mamba", "all_decoder"},
        )
        for config in configs.values():
            generation = config["policy"]["generation"]
            self.assertEqual(
                generation["vllm_kwargs"]["kernel_config"]["moe_backend"],
                "flashinfer_trtllm",
            )
            self.assertEqual(
                generation["vllm_kwargs"]["kernel_config"]["linear_backend"],
                "flashinfer_cutedsl",
            )
            self.assertEqual(
                generation["vllm_kwargs"]["moe_backend"],
                "flashinfer_trtllm",
            )
        self.assertEqual(
            self.bf16["policy"]["generation"]["vllm_kwargs"]["kernel_config"],
            {"moe_backend": "flashinfer_trtllm"},
        )

    def test_qkvo_mamba_keeps_boundary_and_non_text_modules_bf16(self) -> None:
        configs = build_configs(self.bf16, self.moe)
        ignore = configs["qkvo_mamba"]["policy"]["generation"]["vllm_cfg"][
            "quantization_ignore_patterns"
        ]
        for target in ("in_proj", "out_proj", "qkv_proj", "o_proj"):
            self.assertNotIn(f"*model.layers.*.mixer.{target}", ignore)
        for pattern in (
            "*model.layers.0.*",
            "*model.layers.80.*",
            "*model.layers.*.mixer.gate",
            "*vision_model.*",
            "*mtp*",
        ):
            self.assertIn(pattern, ignore)

    def test_all_decoder_still_excludes_unsupported_modules(self) -> None:
        configs = build_configs(self.bf16, self.moe)
        ignore = configs["all_decoder"]["policy"]["generation"]["vllm_cfg"][
            "quantization_ignore_patterns"
        ]
        self.assertNotIn("*model.layers.0.*", ignore)
        self.assertNotIn("*model.layers.80.*", ignore)
        self.assertNotIn("*model.layers.*.mixer.shared_experts.*", ignore)
        for pattern in (
            "*model.layers.*.mixer.gate",
            "*vision_model.*",
            "*sound_encoder.*",
            "*mtp*",
            "*lm_head",
        ):
            self.assertIn(pattern, ignore)

    def test_rejects_unexpected_baseline_scope(self) -> None:
        self.moe["policy"]["generation"]["vllm_cfg"]["quantization_ignore_patterns"] = [
            "*vision_model.*"
        ]
        with self.assertRaisesRegex(ValueError, "missing required exclusions"):
            build_configs(self.bf16, self.moe)


if __name__ == "__main__":
    unittest.main()
