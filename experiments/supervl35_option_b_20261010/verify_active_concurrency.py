"""Verify achieved TP4 concurrency from a completed vLLM benchmark run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEDULER_STATS = re.compile(
    r"Running:\s*(\d+)\s+reqs,\s*Waiting:\s*(\d+)\s+reqs,\s*"
    r"GPU KV cache usage:\s*([\d.]+)%"
)


def assess_active_concurrency(
    result: dict[str, Any], log_text: str, expected: int
) -> dict[str, Any]:
    if expected <= 0:
        raise ValueError("expected concurrency must be positive")
    if result.get("concurrencies") != [expected]:
        raise ValueError("result concurrency does not match expected concurrency")
    if result.get("max_num_seqs") != expected:
        raise ValueError("max_num_seqs does not match expected concurrency")
    runs = result.get("runs", [])
    if not runs or any(run.get("requests", 0) < expected for run in runs):
        raise ValueError("result has fewer requests than expected concurrency")
    samples = [
        (int(running), int(waiting), float(kv_usage))
        for running, waiting, kv_usage in SCHEDULER_STATS.findall(log_text)
    ]
    if not samples:
        raise ValueError("no vLLM scheduler statistics found in driver log")
    max_running = max(sample[0] for sample in samples)
    return {
        "configured_max_num_seqs": expected,
        "max_running": max_running,
        "max_waiting": max(sample[1] for sample in samples),
        "max_kv_cache_usage_percent": max(sample[2] for sample in samples),
        "scheduler_samples": len(samples),
        "reached": max_running >= expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("arm", choices=("bf16", "moe"))
    parser.add_argument("expected_concurrency", type=int)
    args = parser.parse_args()
    exit_code = (args.run_dir / "exit-code").read_text(encoding="utf-8").strip()
    if exit_code != "0":
        raise SystemExit(f"benchmark did not complete successfully: exit-code={exit_code}")
    result = json.loads((args.run_dir / f"{args.arm}.json").read_text(encoding="utf-8"))
    log = (args.run_dir / "driver.log").read_text(encoding="utf-8", errors="replace")
    assessment = assess_active_concurrency(result, log, args.expected_concurrency)
    print(json.dumps(assessment, indent=2, sort_keys=True))
    if not assessment["reached"]:
        raise SystemExit("configured concurrency was not observed at runtime")


if __name__ == "__main__":
    main()
