import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_nemorl_specdec_parity import compare_samples, load_samples


class SpecDecParityTest(unittest.TestCase):
    def _load(self, rows):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            return load_samples(path)

    def test_greedy_parity_passes_exact_tokens_and_close_logprobs(self):
        baseline = self._load(
            [
                {
                    "prompt_id": "p0",
                    "sample_id": "s0",
                    "token_ids": [1, 2],
                    "token_logprobs": [-0.2, -0.3],
                    "reward": 1.0,
                }
            ]
        )
        specdec = self._load(
            [
                {
                    "prompt_id": "p0",
                    "sample_id": "s0",
                    "token_ids": [1, 2],
                    "token_logprobs": [-0.20001, -0.29999],
                    "reward": 1.0,
                }
            ]
        )

        result = compare_samples(
            baseline,
            specdec,
            mode="greedy",
            max_token_logprob_delta=1e-3,
            max_mean_logprob_delta=1e-3,
            max_first_token_tv=0.1,
            max_reward_delta=0.0,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["exact_tokens"])

    def test_sampled_parity_fails_shifted_first_token_distribution(self):
        baseline_rows = []
        specdec_rows = []
        for index in range(10):
            common = {
                "prompt_id": "p0",
                "sample_id": f"s{index}",
                "token_logprobs": [-0.5],
            }
            baseline_rows.append({**common, "token_ids": [index % 2]})
            specdec_rows.append({**common, "token_ids": [2]})

        result = compare_samples(
            self._load(baseline_rows),
            self._load(specdec_rows),
            mode="sampled",
            max_token_logprob_delta=1e-3,
            max_mean_logprob_delta=1e-3,
            max_first_token_tv=0.1,
            max_reward_delta=0.0,
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["first_token_distribution"])

    def test_greedy_token_mismatch_cannot_pass_logprob_parity(self):
        baseline = self._load(
            [
                {
                    "prompt_id": "p0",
                    "sample_id": "s0",
                    "token_ids": [1, 2],
                    "token_logprobs": [-0.2, -0.3],
                }
            ]
        )
        specdec = self._load(
            [
                {
                    "prompt_id": "p0",
                    "sample_id": "s0",
                    "token_ids": [1, 3],
                    "token_logprobs": [-0.2, -0.3],
                }
            ]
        )

        result = compare_samples(
            baseline,
            specdec,
            mode="greedy",
            max_token_logprob_delta=1e-3,
            max_mean_logprob_delta=1e-3,
            max_first_token_tv=0.1,
            max_reward_delta=0.0,
        )

        self.assertFalse(result["checks"]["exact_tokens"])
        self.assertFalse(result["checks"]["token_logprobs"])

    def test_missing_chosen_token_logprob_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt_id": "p0",
                        "sample_id": "s0",
                        "token_ids": [1],
                        "token_logprobs": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "length mismatch"):
                load_samples(path)


if __name__ == "__main__":
    unittest.main()
