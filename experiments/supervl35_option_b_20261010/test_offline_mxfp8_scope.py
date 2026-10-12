import unittest

from offline_mxfp8_scope import build_scope


class OfflineMxFp8ScopeTest(unittest.TestCase):
    def test_only_middle_routed_experts_are_quantized(self) -> None:
        prefix = "language_model.backbone.layers"
        names = {
            f"{prefix}.{layer}.mixer.experts.{expert}.{projection}.weight"
            for layer in (1, 3, 5, 7)
            for expert in (0, 1)
            for projection in ("up_proj", "down_proj")
        }
        names.update(
            {
                f"{prefix}.3.mixer.qkv_proj.weight",
                "language_model.mtp.layers.0.mixer.experts.0.up_proj.weight",
                "lm_head.weight",
            }
        )

        targets, ignored = build_scope(
            names,
            ["mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe"],
            first_bf16=2,
            last_bf16=3,
        )

        self.assertEqual(len(targets), 4)
        self.assertTrue(all("layers.3.mixer.experts." in name for name in targets))
        self.assertIn("*layers.1.mixer.experts", ignored)
        self.assertIn("*layers.7.mixer.experts", ignored)
        self.assertIn(f"{prefix}.3.mixer.qkv_proj", ignored)
        self.assertIn("lm_head", ignored)
        self.assertNotIn("*layers.3.mixer.experts", ignored)

    def test_missing_expert_projection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing down_proj"):
            build_scope(
                {"language_model.backbone.layers.1.mixer.experts.0.up_proj.weight"},
                ["mamba", "moe", "mamba"],
                first_bf16=0,
                last_bf16=0,
            )


if __name__ == "__main__":
    unittest.main()
