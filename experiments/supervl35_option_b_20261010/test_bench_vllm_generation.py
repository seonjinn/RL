import unittest

from experiments.supervl35_option_b_20261010.bench_vllm_generation import (
    _prompt_ids,
    build_engine_kwargs,
    summarize_outputs,
)


class VllmGenerationBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe = {
            "policy": {
                "model_name": "/model",
                "precision": "bfloat16",
                "generation": {
                    "model_name": "/model",
                    "vllm_cfg": {
                        "precision": "fp8",
                        "is_mx": True,
                        "tensor_parallel_size": 4,
                        "pipeline_parallel_size": 1,
                        "expert_parallel_size": 4,
                        "gpu_memory_utilization": 0.8,
                        "max_model_len": 65536,
                        "enable_prefix_caching": False,
                        "enforce_eager": False,
                    },
                    "vllm_kwargs": {
                        "hf_overrides": {"video_target_num_patches": 1024},
                        "kernel_config": {
                            "moe_backend": "flashinfer_trtllm",
                            "linear_backend": "flashinfer_cutedsl",
                        },
                    },
                },
            }
        }

    def test_engine_kwargs_use_hf_load_and_keep_model_overrides(self) -> None:
        kwargs = build_engine_kwargs(
            self.recipe,
            fp8_kwargs={
                "quantization": "fp8",
                "hf_overrides": {
                    "quantization_config": {
                        "quant_method": "modelopt",
                        "quant_algo": "MXFP8",
                        "ignore": ["*lm_head"],
                    }
                },
            },
        )
        self.assertEqual(kwargs["load_format"], "auto")
        self.assertEqual(kwargs["tensor_parallel_size"], 4)
        self.assertTrue(kwargs["enable_expert_parallel"])
        self.assertEqual(kwargs["dtype"], "bfloat16")
        self.assertEqual(
            kwargs["hf_overrides"]["quantization_config"]["quant_algo"],
            "MXFP8",
        )
        self.assertEqual(kwargs["hf_overrides"]["video_target_num_patches"], 1024)
        self.assertEqual(
            kwargs["kernel_config"]["linear_backend"], "flashinfer_cutedsl"
        )

    def test_bf16_engine_rejects_fp8_options(self) -> None:
        self.recipe["policy"]["generation"]["vllm_cfg"]["precision"] = "bfloat16"
        self.recipe["policy"]["generation"]["vllm_cfg"]["is_mx"] = False
        with self.assertRaisesRegex(ValueError, "BF16 arm must not have FP8"):
            build_engine_kwargs(self.recipe, fp8_kwargs={"quantization": "fp8"})

    def test_summary_requires_exact_output_length(self) -> None:
        outputs = [
            {"prompt_tokens": 10, "output_tokens": 4},
            {"prompt_tokens": 10, "output_tokens": 4},
        ]
        result = summarize_outputs(
            outputs, elapsed_s=2.0, expected_output_tokens=4, gpus=4
        )
        self.assertEqual(result["output_tokens_per_second"], 4.0)
        self.assertEqual(result["output_tokens_per_second_per_gpu"], 1.0)
        with self.assertRaisesRegex(ValueError, "output length"):
            summarize_outputs(outputs, elapsed_s=2.0, expected_output_tokens=5, gpus=4)

    def test_prompt_expands_until_target_token_count(self) -> None:
        class CoarseTokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                self.assertions = (tokenize, add_generation_prompt)
                return [7] * (len(messages[0]["content"]) // 100)

        tokenizer = CoarseTokenizer()
        self.assertEqual(len(_prompt_ids(tokenizer, 10500)), 10500)
        self.assertEqual(tokenizer.assertions, (True, True))

    def test_prompt_unwraps_batched_tokenizer_output(self) -> None:
        class BatchedTokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                return {"input_ids": [[3] * (len(messages[0]["content"]) // 10)]}

        self.assertEqual(_prompt_ids(BatchedTokenizer(), 10500), [3] * 10500)

    def test_prompt_rejects_non_token_output(self) -> None:
        class BrokenTokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                return {"input_ids": [["bad"]]}

        with self.assertRaisesRegex(TypeError, "token IDs"):
            _prompt_ids(BrokenTokenizer(), 10)


if __name__ == "__main__":
    unittest.main()
