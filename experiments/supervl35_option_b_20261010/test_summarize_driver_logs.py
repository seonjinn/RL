"""Tests for step-aligned SuperVL driver-log timing summaries."""

import tempfile
import unittest
from pathlib import Path

from summarize_driver_logs import parse_steps, summarize


class DriverLogSummaryTest(unittest.TestCase):
    def test_extracts_completed_steps_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "driver.log"
            path.write_text(
                "(NemoGym pid=1) Collecting rollouts: 4/16\r"
                "(SingleControllerActor pid=2)   • Total step time: 200.00s\n"
                "(SingleControllerActor pid=2)   • policy_training: 150.00s (75.0%)\n"
                "(SingleControllerActor pid=2)   • policy_and_reference_logprobs: 30.00s\n"
                "(SingleControllerActor pid=2)   • weight_sync: 5.00s\n"
                "(SingleControllerActor pid=2)   • exposed_generation: 12.00s\n"
                "(SingleControllerActor pid=2)   • valid_tokens_per_sec_per_gpu: 20.00s\n"
                "(SingleControllerActor pid=2) train step 1/20 trainer_v=1\n"
                "(SingleControllerActor pid=2)   • Total step time: 180.00s\n"
                "(SingleControllerActor pid=2)   • policy_training: 130.00s\n"
                "(SingleControllerActor pid=2) train step 2/20 trainer_v=2\n"
                "(SingleControllerActor pid=2)   • Total step time: 170.00s\n",
                encoding="utf-8",
            )
            steps = parse_steps(path)
            self.assertEqual(sorted(steps), [1, 2])
            self.assertEqual(steps[1]["policy_training"], 150.0)
            self.assertEqual(steps[1]["valid_tokens_per_sec_per_gpu"], 20.0)
            self.assertEqual(steps[2]["total_step_time"], 180.0)
            self.assertNotIn("weight_sync", steps[2])

    def test_only_reports_metric_when_present_in_every_selected_step(self) -> None:
        steps = {
            1: {"total_step_time": 200.0, "weight_sync": 5.0},
            2: {"total_step_time": 180.0},
        }
        metrics = summarize(steps, [1, 2])
        self.assertEqual(metrics["total_step_time"], 190.0)
        self.assertIsNone(metrics["weight_sync"])


if __name__ == "__main__":
    unittest.main()
