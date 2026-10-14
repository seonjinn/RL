from __future__ import annotations

import unittest

from verify_active_concurrency import assess_active_concurrency


class ActiveConcurrencyTest(unittest.TestCase):
    def test_reached_from_runtime_log(self) -> None:
        result = {
            "concurrencies": [512],
            "max_num_seqs": 512,
            "runs": [{"requests": 1024}],
        }
        log = (
            "Running: 128 reqs, Waiting: 896 reqs, GPU KV cache usage: 7.3%\n"
            "Running: 512 reqs, Waiting: 512 reqs, GPU KV cache usage: 31.4%\n"
        )
        assessment = assess_active_concurrency(result, log, 512)
        self.assertTrue(assessment["reached"])
        self.assertEqual(assessment["max_running"], 512)
        self.assertEqual(assessment["max_kv_cache_usage_percent"], 31.4)

    def test_configured_cap_alone_does_not_prove_concurrency(self) -> None:
        result = {
            "concurrencies": [512],
            "max_num_seqs": 512,
            "runs": [{"requests": 1024}],
        }
        log = "Running: 128 reqs, Waiting: 896 reqs, GPU KV cache usage: 7.3%"
        self.assertFalse(assess_active_concurrency(result, log, 512)["reached"])

    def test_mismatched_cap_is_rejected(self) -> None:
        result = {
            "concurrencies": [512],
            "max_num_seqs": 128,
            "runs": [{"requests": 1024}],
        }
        with self.assertRaisesRegex(ValueError, "max_num_seqs"):
            assess_active_concurrency(result, "Running: 128 reqs", 512)


if __name__ == "__main__":
    unittest.main()
