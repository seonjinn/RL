"""Summarize exact generated-token lengths from NeMo-RL train JSONL logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def percentile(values: list[int], percentage: int) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentage / 100) - 1)]


parser = argparse.ArgumentParser()
parser.add_argument("--log-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-output-length", type=int, required=True)
parser.add_argument("--expected-steps", type=int, nargs="+", required=True)
parser.add_argument("--expected-samples-per-step", type=int)
args = parser.parse_args()

lengths: list[int] = []
per_step: dict[str, dict[str, Any]] = {}
for step in args.expected_steps:
    matches = list(args.log_root.rglob(f"train_data_step{step}.jsonl"))
    if len(matches) != 1:
        raise SystemExit(f"expected one train_data_step{step}.jsonl, found {matches}")
    step_lengths: list[int] = []
    with matches[0].open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            mask = row.get("token_loss_mask")
            if (
                not isinstance(mask, list)
                or len(mask) != 1
                or not isinstance(mask[0], list)
            ):
                raise SystemExit(
                    f"invalid token_loss_mask at {matches[0]}:{line_number}"
                )
            step_lengths.append(sum(bool(value) for value in mask[0]))
    if not step_lengths:
        raise SystemExit(f"no samples in {matches[0]}")
    if (
        args.expected_samples_per_step is not None
        and len(step_lengths) != args.expected_samples_per_step
    ):
        raise SystemExit(
            f"step {step} sample count {len(step_lengths)} != "
            f"{args.expected_samples_per_step}"
        )
    lengths.extend(step_lengths)
    per_step[str(step)] = {
        "sample_count": len(step_lengths),
        "max": max(step_lengths),
        "cap_hit_count": sum(value >= args.max_output_length for value in step_lengths),
    }
metrics = {
    "steps": args.expected_steps,
    "sample_count": len(lengths),
    "max_output_length": args.max_output_length,
    "mean": sum(lengths) / len(lengths),
    "max": max(lengths),
    "quantiles": {f"p{p}": percentile(lengths, p) for p in (50, 90, 95, 99)},
    "cap_hit_count": sum(value >= args.max_output_length for value in lengths),
    "cap_hit_rate": sum(value >= args.max_output_length for value in lengths)
    / len(lengths),
    "per_step": per_step,
}
args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(
    "OUTPUT_LENGTH_GATE_PASS "
    f"samples={metrics['sample_count']} p50={metrics['quantiles']['p50']} "
    f"p95={metrics['quantiles']['p95']} p99={metrics['quantiles']['p99']} "
    f"max={metrics['max']} cap_hits={metrics['cap_hit_count']}"
)
