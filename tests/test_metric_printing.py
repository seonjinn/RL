import unittest

from nemo_rl.utils.metric_printing import format_spec_decode_metrics


class TestFormatSpecDecodeMetrics(unittest.TestCase):
    def test_formats_parseable_scalar_values(self) -> None:
        line = format_spec_decode_metrics(
            {
                "vllm/spec_num_drafts": 10.0,
                "vllm/spec_acceptance_rate": 0.625,
                "vllm/spec_acceptance_length": 2.75,
                "reward": 0.5,
            },
            step=3,
        )

        self.assertEqual(
            line,
            'VLLM_SPEC_DECODE_METRICS {"step": 3, '
            '"vllm/spec_acceptance_length": 2.75, '
            '"vllm/spec_acceptance_rate": 0.625, '
            '"vllm/spec_num_drafts": 10.0}',
        )

    def test_returns_none_without_numeric_spec_metrics(self) -> None:
        line = format_spec_decode_metrics(
            {
                "vllm/spec_debug": {"unsupported": True},
                "reward": 0.5,
            },
            step=1,
        )

        self.assertIsNone(line)


if __name__ == "__main__":
    unittest.main()
