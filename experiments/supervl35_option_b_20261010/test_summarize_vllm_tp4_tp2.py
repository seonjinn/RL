import unittest

from experiments.supervl35_option_b_20261010 import summarize_vllm_tp4_tp2


def result(elapsed_s: float, requests: int = 20) -> dict:
    return {
        "vllm_version": "0.29.0",
        "prompt_sha256": "same-prompt",
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "request_multiplier": requests // 2,
        "runs": [
            {
                "batch_size": 2,
                "output_length": 1024,
                "requests": requests,
                "prompt_tokens": requests * 1024,
                "output_tokens": requests * 1024,
                "elapsed_s": elapsed_s,
            }
        ],
    }


class SummarizeVllmTp4Tp2Tests(unittest.TestCase):
    def test_matched_pair_computes_throughput_ratio(self) -> None:
        summary = summarize_vllm_tp4_tp2.summarize_matched_runs(
            [(result(120.0), result(100.0))], concurrency=2
        )
        self.assertEqual(summary["requests"], 20)
        self.assertEqual(summary["output_tokens"], 20480)
        self.assertAlmostEqual(summary["target_vs_baseline"], 1.2)

    def test_split_runs_sum_tokens_and_elapsed_before_comparison(self) -> None:
        summary = summarize_vllm_tp4_tp2.summarize_matched_runs(
            [(result(120.0), result(100.0)), (result(180.0), result(200.0))],
            concurrency=2,
        )
        self.assertEqual(summary["requests"], 40)
        self.assertEqual(summary["baseline_elapsed_s"], 300.0)
        self.assertEqual(summary["target_elapsed_s"], 300.0)
        self.assertEqual(summary["target_vs_baseline"], 1.0)

    def test_rejects_mismatched_requests_and_disabled_cache(self) -> None:
        with self.assertRaisesRegex(ValueError, "request counts"):
            summarize_vllm_tp4_tp2.summarize_matched_runs(
                [(result(120.0), result(100.0, requests=18))], concurrency=2
            )
        disabled = result(100.0)
        disabled["enable_prefix_caching"] = False
        with self.assertRaisesRegex(ValueError, "prefix caching"):
            summarize_vllm_tp4_tp2.summarize_matched_runs(
                [(result(120.0), disabled)], concurrency=2
            )


if __name__ == "__main__":
    unittest.main()
