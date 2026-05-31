#!/usr/bin/env python3
"""Extract per-step timing from NeMo-RL async GRPO ray-driver.log.

Usage (run on cluster login node):
    python extract_step_times.py <job_id> [<job_id> ...]

Reads /lustre/fsw/portfolios/coreai/users/sna/repos/nemo-rl-qwen-swe/{job}-logs/ray-driver.log.
Emits per-step timings + summary (median over steady-state, cold step 0 dropped,
gen-bound outliers >800s dropped) for direct comparison vs throughput_tracker.html.

Output format mirrors the throughput tracker's reading guide:
  exposed_generation, policy_training, policy_and_reference_logprobs, E2E (sum).
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

REPO = Path("/lustre/fsw/portfolios/coreai/users/sna/repos/nemo-rl-qwen-swe")
STEP_RE = re.compile(r"Step coordination: training_step=(\d+)")
TIMING_KEYS = ("exposed_generation", "policy_training", "policy_and_reference_logprobs")
TIMING_RE = {k: re.compile(rf"• {k}: ([\d.]+)s") for k in TIMING_KEYS}
GEN_OUTLIER_THRESHOLD_S = 800.0


def extract(job_id: str) -> dict[int, dict[str, float]]:
    log = REPO / f"{job_id}-logs" / "ray-driver.log"
    if not log.exists():
        return {}
    by_step: dict[int, dict[str, float]] = {}
    current: int | None = None
    for line in log.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            current = int(m.group(1))
            by_step.setdefault(current, {})
            continue
        if current is None:
            continue
        for k, rx in TIMING_RE.items():
            mm = rx.search(line)
            if mm and k not in by_step[current]:
                by_step[current][k] = float(mm.group(1))
    return by_step


def summarize(job_id: str, by_step: dict[int, dict[str, float]]) -> None:
    print(f"\n=== Job {job_id} ===")
    if not by_step:
        print("  (no step timings recorded yet)")
        return
    print(f"  steps captured: {sorted(by_step.keys())}")
    rows = []
    for s, t in sorted(by_step.items()):
        if not all(k in t for k in TIMING_KEYS):
            continue
        e2e = sum(t[k] for k in TIMING_KEYS)
        rows.append(
            (
                s,
                t["exposed_generation"],
                t["policy_training"],
                t["policy_and_reference_logprobs"],
                e2e,
            )
        )
    if not rows:
        print("  (no fully-timed steps)")
        return
    print(f"  {'step':>4}  {'gen':>8}  {'train':>8}  {'logprob':>9}  {'E2E':>8}")
    for r in rows:
        print(f"  {r[0]:>4}  {r[1]:>8.2f}  {r[2]:>8.2f}  {r[3]:>9.2f}  {r[4]:>8.2f}")
    # steady-state median: drop step 0 + drop steps where gen > 800s
    steady = [r for r in rows if r[0] > 0 and r[1] <= GEN_OUTLIER_THRESHOLD_S]
    if steady:
        gen = [r[1] for r in steady]
        train = [r[2] for r in steady]
        logp = [r[3] for r in steady]
        e2e = [r[4] for r in steady]
        print(
            f"  --- steady-state medians (n={len(steady)}, gen<{GEN_OUTLIER_THRESHOLD_S:.0f}s) ---"
        )
        print(
            f"  gen={statistics.median(gen):.2f}s  train={statistics.median(train):.2f}s  "
            f"logprob={statistics.median(logp):.2f}s  E2E={statistics.median(e2e):.2f}s"
        )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for job_id in sys.argv[1:]:
        summarize(job_id, extract(job_id))


if __name__ == "__main__":
    main()
