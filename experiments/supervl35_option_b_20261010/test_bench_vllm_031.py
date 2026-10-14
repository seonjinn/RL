from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from bench_vllm_031 import build_benchmark_kwargs


HERE = Path(__file__).parent


class Vllm031BenchmarkConfigTest(unittest.TestCase):
    def test_bf16_tp4_preserves_recipe_backends(self) -> None:
        recipe = yaml.safe_load((HERE / "configs/bf16.yaml").read_text())
        kwargs = build_benchmark_kwargs(recipe, "/model", "bf16", 1, 16)
        self.assertEqual(kwargs["tensor_parallel_size"], 4)
        self.assertEqual(kwargs["max_num_seqs"], 16)
        self.assertEqual(kwargs["max_num_batched_tokens"], 65536)
        self.assertEqual(
            kwargs["kernel_config"]["linear_backend"], "flashinfer_cutedsl"
        )
        self.assertEqual(kwargs["moe_backend"], "flashinfer_trtllm")
        self.assertEqual(kwargs["compilation_config"]["backend"], "eager")
        self.assertNotIn("quantization", kwargs)

    def test_mxfp8_tp2dp2_keeps_legacy_ignore_scope(self) -> None:
        recipe = yaml.safe_load((HERE / "configs/moe.yaml").read_text())
        kwargs = build_benchmark_kwargs(recipe, "/model", "mxfp8_modelopt", 2, 16)
        self.assertEqual(kwargs["tensor_parallel_size"], 2)
        self.assertEqual(kwargs["max_num_seqs"], 8)
        self.assertEqual(kwargs["max_num_batched_tokens"], 32768)
        self.assertEqual(kwargs["quantization"], "fp8")
        ignore = kwargs["hf_overrides"]["quantization_config"]["ignore"]
        self.assertIn("*model.layers.0.*", ignore)
        self.assertIn("*model.layers.87.*", ignore)
        self.assertIn("*model.layers.*.mixer.qkv_proj", ignore)


if __name__ == "__main__":
    unittest.main()
