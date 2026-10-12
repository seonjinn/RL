import unittest
from pathlib import Path

from experiments.supervl35_option_b_20261010.bench_vllm_generation import (
    _prompt_ids,
    benchmark_cases,
    build_profile_config,
    build_engine_kwargs,
    generate_case,
    summarize_outputs,
)
from experiments.supervl35_option_b_20261010.bench_vllm_dp import (
    aggregate_rank_runs,
    requests_for_rank,
    serialized_checkpoint_recipe,
    trials_for_batch,
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

    def test_attention_override_keeps_moe_and_linear_backends(self) -> None:
        kwargs = build_engine_kwargs(
            self.recipe,
            fp8_kwargs={"quantization": "fp8"},
            attention_backend="TRITON_ATTN",
        )
        self.assertEqual(kwargs["attention_backend"], "TRITON_ATTN")
        self.assertEqual(kwargs["kernel_config"]["moe_backend"], "flashinfer_trtllm")
        self.assertEqual(kwargs["kernel_config"]["linear_backend"], "flashinfer_cutedsl")
        self.assertNotIn("attention_backend", self.recipe["policy"]["generation"]["vllm_kwargs"])

    def test_serialized_checkpoint_uses_its_own_quantization_config(self) -> None:
        recipe = serialized_checkpoint_recipe(self.recipe, "/serialized-mxfp8")
        kwargs = build_engine_kwargs(recipe)
        self.assertEqual(kwargs["model"], "/serialized-mxfp8")
        self.assertEqual(kwargs["load_format"], "auto")
        self.assertEqual(kwargs["dtype"], "bfloat16")
        self.assertNotIn("quantization", kwargs)
        self.assertNotIn("quantization_config", kwargs["hf_overrides"])
        self.assertEqual(
            self.recipe["policy"]["generation"]["vllm_cfg"]["precision"], "fp8"
        )

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

    def test_concurrency_sweep_preserves_requested_sizes(self) -> None:
        self.assertEqual(
            benchmark_cases("1,2,4,8,16,32,128,256,512,1024", 128),
            [
                (1, 128),
                (2, 128),
                (4, 128),
                (8, 128),
                (16, 128),
                (32, 128),
                (128, 128),
                (256, 128),
                (512, 128),
                (1024, 128),
            ],
        )

    def test_concurrency_sweep_rejects_duplicate_or_invalid_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            benchmark_cases("1,2,2", 128)
        with self.assertRaisesRegex(ValueError, "positive"):
            benchmark_cases("1,2", 0)

    def test_profile_config_bounds_worker_trace_and_requires_absolute_dir(self) -> None:
        config = build_profile_config(Path("/raid/scratch/profile"), 64, 32)
        self.assertEqual(config["profiler"], "torch")
        self.assertEqual(config["torch_profiler_dir"], "/raid/scratch/profile")
        self.assertEqual(config["delay_iterations"], 64)
        self.assertEqual(config["max_iterations"], 32)
        with self.assertRaisesRegex(ValueError, "absolute"):
            build_profile_config(Path("relative/profile"), 0, 32)
        with self.assertRaisesRegex(ValueError, "positive"):
            build_profile_config(Path("/raid/scratch/profile"), 0, 0)

    def test_profiled_case_stops_profiler_after_generation_error(self) -> None:
        class FailingLLM:
            def __init__(self) -> None:
                self.calls = []

            def start_profile(self, name):
                self.calls.append(("start", name))

            def generate(self, prompts, params, *, use_tqdm):
                self.calls.append(("generate", len(prompts)))
                raise RuntimeError("generation failed")

            def stop_profile(self):
                self.calls.append(("stop",))

        llm = FailingLLM()
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            generate_case(llm, [{}], object(), profile_prefix="isl1024-c16")
        self.assertEqual(
            llm.calls,
            [("start", "isl1024-c16"), ("generate", 1), ("stop",)],
        )

    def test_profiled_case_returns_generated_output_and_elapsed(self) -> None:
        class LLM:
            def __init__(self) -> None:
                self.calls = []

            def start_profile(self, name):
                self.calls.append("start")

            def generate(self, prompts, params, *, use_tqdm):
                self.calls.append("generate")
                return ["sample"]

            def stop_profile(self):
                self.calls.append("stop")

        llm = LLM()
        outputs, elapsed = generate_case(llm, [{}], object(), "case")
        self.assertEqual(outputs, ["sample"])
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(llm.calls, ["start", "generate", "stop"])

    def test_dp_split_preserves_total_requests(self) -> None:
        self.assertEqual(requests_for_rank(5, 0, 2), 3)
        self.assertEqual(requests_for_rank(5, 1, 2), 2)
        self.assertEqual(requests_for_rank(2, 0, 2), 1)
        with self.assertRaisesRegex(ValueError, "at least one request"):
            requests_for_rank(1, 0, 2)

    def test_dp_length_sweep_trial_counts_match_tp4(self) -> None:
        self.assertEqual(trials_for_batch(128, 2, 1), 2)
        self.assertEqual(trials_for_batch(256, 2, 1), 1)

    def test_dp_aggregate_uses_shared_wall_clock_and_four_gpus(self) -> None:
        result = aggregate_rank_runs(
            [
                {
                    "rank": 0,
                    "started_s": 10.0,
                    "finished_s": 14.0,
                    "requests": 3,
                    "prompt_tokens": 3072,
                    "output_tokens": 384,
                },
                {
                    "rank": 1,
                    "started_s": 10.1,
                    "finished_s": 15.0,
                    "requests": 2,
                    "prompt_tokens": 2048,
                    "output_tokens": 256,
                },
            ],
            gpus=4,
        )
        self.assertEqual(result["requests"], 5)
        self.assertEqual(result["elapsed_s"], 5.0)
        self.assertEqual(result["output_tokens_per_second_per_gpu"], 32.0)

    def test_dp_aggregate_rejects_incomplete_ranks(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate rank"):
            aggregate_rank_runs(
                [
                    {
                        "rank": 0,
                        "started_s": 1,
                        "finished_s": 2,
                        "requests": 1,
                        "prompt_tokens": 1,
                        "output_tokens": 1,
                    }
                ]
                * 2,
                gpus=4,
            )


if __name__ == "__main__":
    unittest.main()
