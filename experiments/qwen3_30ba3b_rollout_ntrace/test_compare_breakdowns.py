from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from compare_breakdowns import compare, load_arm


def _summary(scale: float) -> dict:
    ranks = [
        {
            "raw_kernel_categories_s": {
                "expert_fc1_bmm": 2.0 * scale,
                "expert_fc2_bmm": 1.0 * scale,
            }
        }
        for _ in range(2)
    ]
    iterations = [
        {
            "index": index,
            "step_s": {"mean": 10.0 * scale, "max": 11.0 * scale},
            "active_s": {"mean": 8.0 * scale},
            "idle_s": {"mean": 2.0 * scale},
            "stack_categories_s": {"moe": 4.0 * scale},
        }
        for index in range(3)
    ]
    return {"ranks": ranks, "iterations": iterations}


LOG = """
 [ntrace.nemo_rl.rollout] armed rank=0 backend=cpp capture_iter=1 last_iter=3
========================= Step 1/4 =========================
[ntrace.nemo_rl.rollout] rollout started iteration=0 step_id=step1/attempt1
Generating responses for batch of size 8...
  • Mean Generation Length: 50.0
========================= Step 2/4 =========================
[ntrace.nemo_rl.rollout] rollout started iteration=1 step_id=step2/attempt1
[ntrace.nemo_rl.rollout] rollout started iteration=1 step_id=step2/attempt1
Generating responses for batch of size 8...
  • Mean Generation Length: 100.0
========================= Step 3/4 =========================
[ntrace.nemo_rl.rollout] rollout started iteration=2 step_id=step3/attempt1
Generating responses for batch of size 8...
  • Mean Generation Length: 200.0
========================= Step 4/4 =========================
[ntrace.nemo_rl.rollout] rollout started iteration=3 step_id=step4/attempt1
Generating responses for batch of size 8...
  • Mean Generation Length: 300.0
"""


class CompareBreakdownsTest(unittest.TestCase):
    def test_normalizes_matched_windows_per_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "run.log"
            log.write_text(LOG)
            arms = []
            for name, scale in (("bf16", 1.0), ("mxfp8", 0.5)):
                summary = root / f"{name}.json"
                summary.write_text(json.dumps(_summary(scale)))
                arms.append(load_arm(summary, log))

            result = compare(*arms)

        self.assertEqual(result["steady_iteration_indices"], [1, 2])
        self.assertAlmostEqual(result["ratios"]["throughput_speedup"], 2.0)
        self.assertAlmostEqual(result["ratios"]["active_time_reduction"], 0.5)
        self.assertAlmostEqual(result["ratios"]["moe_time_reduction"], 0.5)
        self.assertAlmostEqual(
            result["bf16"]["raw_s_per_mtoken"]["expert_fc1_bmm"], 2500.0
        )


if __name__ == "__main__":
    unittest.main()
