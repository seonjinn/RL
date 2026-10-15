"""Focused validation tests for the active-concurrency report."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from collect_vllm029_active_trend import collect_job


class ActiveTrendValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "vllm_length_sweep" / "isl1024_osl1024" / "123"
        self.run_dir.mkdir(parents=True)
        self.job = {"arm": "bf16", "concurrency": 256, "job_id": "123", "request_multiplier": 8}
        self.result = {
            "vllm_version": "0.29.0",
            "concurrencies": [256],
            "max_num_seqs": 256,
            "request_multiplier": 8,
            "prompt_tokens": 1024,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "moe_backend": "flashinfer_trtllm",
            "linear_backend": "flashinfer_cutedsl",
            "attention_backend": "FLASHINFER",
            "scoped_mxfp8_ignore": [],
            "model": "checkpoint",
            "prompt_sha256": "prompt",
            "config_sha256": "config",
            "runs": [
                {
                    "trial": trial,
                    "requests": 2048,
                    "batch_size": 256,
                    "output_length": 1024,
                    "prompt_tokens": 2048 * 1024,
                    "output_tokens": 2048 * 1024,
                    "elapsed_s": 256.0,
                    "output_tokens_per_second_per_gpu": 2048.0,
                }
                for trial in (0, 1)
            ],
        }
        (self.run_dir / "exit-code").write_text("0\n", encoding="utf-8")
        (self.run_dir / "driver.log").write_text(
            "enforce_eager=False cudagraph_mode=<CUDAGraphMode.PIECEWISE: 1> "
            "cudagraph_capture_sizes': [1, 256, 512], max_cudagraph_capture_size': 512\n"
            "Graph capturing finished in 16 secs\n"
            "Running: 256 reqs, Waiting: 100 reqs, GPU KV cache usage: 30.0%\n",
            encoding="utf-8",
        )
        self.save_result()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def save_result(self) -> None:
        (self.run_dir / "bf16.json").write_text(json.dumps(self.result), encoding="utf-8")

    def test_valid_pair_is_measured(self) -> None:
        record = collect_job(self.root, self.job)
        self.assertTrue(record["reached_target"])
        self.assertEqual(record["max_running"], 256)
        self.assertEqual(record["mean_output_tok_s_gpu"], 2048.0)

    def test_wrong_output_tokens_are_rejected(self) -> None:
        self.result["runs"][1]["output_tokens"] -= 1
        self.save_result()
        with self.assertRaisesRegex(ValueError, "Wrong output tokens"):
            collect_job(self.root, self.job)

    def test_incomplete_active_concurrency_is_marked(self) -> None:
        (self.run_dir / "driver.log").write_text(
            "enforce_eager=False cudagraph_mode=<CUDAGraphMode.PIECEWISE: 1> "
            "cudagraph_capture_sizes': [1, 256, 512], max_cudagraph_capture_size': 512\n"
            "Graph capturing finished in 16 secs\n"
            "Running: 128 reqs, Waiting: 100 reqs, GPU KV cache usage: 30.0%\n",
            encoding="utf-8",
        )
        self.assertFalse(collect_job(self.root, self.job)["reached_target"])


if __name__ == "__main__":
    unittest.main()
