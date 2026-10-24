"""Summarize agent startup bench runs: per-step seconds by variant and concurrency.

python summarize.py <results/run_dir> [summary.csv]
"""

import csv
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r"(?P<variant>\w+)\.n(?P<n>\d+)\.r(?P<rep>\d+)\.\d+\.csv")
STEPS = [
    "git_check",
    "poetry_version",
    "poetry_noop",
    "venv_noop",
    "infer_imports",
    "server_import",
    "total",
]
FIELDS = ["variant", "n", "rep", "step", "count", "failed", "mean", "p50", "max"]


def main() -> None:
    run = Path(sys.argv[1])
    times: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    failed: dict[tuple[str, int, int, str], int] = defaultdict(int)
    for f in run.glob("*.csv"):
        m = NAME.fullmatch(f.name)
        if not m:
            continue
        key = (m["variant"], int(m["n"]), int(m["rep"]))
        for step, t0, t1, rc in csv.reader(f.open()):
            if rc != "0":
                failed[(*key, step)] += 1
            times[(*key, step)].append(float(t1) - float(t0))
    rows = []
    for key in sorted(times, key=lambda k: (k[0], k[1], k[2], STEPS.index(k[3]))):
        xs = sorted(times[key])
        rows.append(
            dict(
                zip(FIELDS[:4], key),
                count=len(xs),
                failed=failed[key],
                mean=f"{st.mean(xs):.2f}",
                p50=f"{xs[len(xs) // 2]:.2f}",
                max=f"{xs[-1]:.2f}",
            )
        )
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    for r in rows:
        print(" ".join(f"{r[k]!s:>14}" for k in FIELDS))
    copy = run / "copy.csv"
    if copy.exists():
        _, t0, t1, rc = next(csv.reader(copy.open()))
        print(f"node-local copy {float(t1) - float(t0):.1f} s rc={rc}")


if __name__ == "__main__":
    main()
