import copy
import fnmatch
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
            {
                "bf16",
                "moe",
                "moe_all",
                "moe_attention",
                "moe_mamba",
                "qkvo_mamba",
                "all_decoder",
            },
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

    def test_attention_and_mamba_scopes_are_independent(self) -> None:
        configs = build_configs(self.bf16, self.moe)
        attention_ignore = configs["moe_attention"]["policy"]["generation"]["vllm_cfg"][
            "quantization_ignore_patterns"
        ]
        mamba_ignore = configs["moe_mamba"]["policy"]["generation"]["vllm_cfg"][
            "quantization_ignore_patterns"
        ]
        for target in ("qkv_proj", "o_proj"):
            self.assertNotIn(f"*model.layers.*.mixer.{target}", attention_ignore)
            self.assertIn(f"*model.layers.*.mixer.{target}", mamba_ignore)
        for target in ("in_proj", "out_proj"):
            self.assertIn(f"*model.layers.*.mixer.{target}", attention_ignore)
            self.assertNotIn(f"*model.layers.*.mixer.{target}", mamba_ignore)

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

    def test_all_layer_moe_quantizes_boundary_experts_only(self) -> None:
        configs = build_configs(self.bf16, self.moe)
        ignore = configs["moe_all"]["policy"]["generation"]["vllm_cfg"][
            "quantization_ignore_patterns"
        ]
        self.assertNotIn("*model.layers.0.*", ignore)
        self.assertNotIn("*model.layers.80.*", ignore)
        for pattern in (
            "*model.layers.*.mixer.qkv_proj",
            "*model.layers.*.mixer.in_proj",
            "*model.layers.*.mixer.shared_experts.*",
            "*model.layers.*.mixer.gate",
            "*vision_model.*",
            "*mtp*",
        ):
            self.assertIn(pattern, ignore)

    def test_vllm_realized_module_scope(self) -> None:
        configs = build_configs(self.bf16, self.moe)

        def excluded(scope: str, module: str) -> bool:
            patterns = configs[scope]["policy"]["generation"]["vllm_cfg"][
                "quantization_ignore_patterns"
            ]
            return any(fnmatch.fnmatchcase(module, pattern) for pattern in patterns)

        expert = "model.layers.80.mixer.experts"
        qkv = "model.layers.80.mixer.qkv_proj"
        shared = "model.layers.80.mixer.shared_experts.up_proj"
        gate = "model.layers.80.mixer.gate"
        self.assertTrue(excluded("moe", expert))
        for scope in ("moe_all", "all_decoder"):
            self.assertFalse(excluded(scope, expert))
            self.assertTrue(excluded(scope, gate))
            self.assertTrue(excluded(scope, "vision_model.radio_model.model"))
            self.assertTrue(excluded(scope, "mtp.layers.0.mixer.experts"))
        self.assertTrue(excluded("moe_all", qkv))
        self.assertTrue(excluded("moe_all", shared))
        self.assertFalse(excluded("all_decoder", qkv))
        self.assertFalse(excluded("all_decoder", shared))

    def test_rejects_unexpected_baseline_scope(self) -> None:
        self.moe["policy"]["generation"]["vllm_cfg"]["quantization_ignore_patterns"] = [
            "*vision_model.*"
        ]
        with self.assertRaisesRegex(ValueError, "missing required exclusions"):
            build_configs(self.bf16, self.moe)


if __name__ == "__main__":
    unittest.main()
