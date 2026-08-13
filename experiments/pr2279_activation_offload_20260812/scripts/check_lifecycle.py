#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RANK_ROW = re.compile(r"\bRank\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)")
SUMMARY_MARKER = "Activation Offload Summary (MB)"
REQUIRED_METRICS = ("train/loss", "train/grad_norm")


def _find_complete_nonzero_summary(
    log_text: str, expected_world_size: int
) -> dict[str, Any] | None:
    cleaned_log = ANSI_ESCAPE.sub("", log_text)
    expected_ranks = set(range(expected_world_size))
    for block in cleaned_log.split(SUMMARY_MARKER)[1:]:
        if "moe_act" not in block:
            continue
        ranks = {
            int(rank): {"moe_act_mib": float(moe_act), "total_mib": float(total)}
            for rank, moe_act, total in RANK_ROW.findall(block)
            if int(rank) in expected_ranks
        }
        if set(ranks) != expected_ranks:
            continue
        if all(
            values["moe_act_mib"] > 0.0 and values["total_mib"] > 0.0
            for values in ranks.values()
        ):
            return {
                "rank_count": len(ranks),
                "minimum_moe_act_mib": min(
                    values["moe_act_mib"] for values in ranks.values()
                ),
                "minimum_total_mib": min(
                    values["total_mib"] for values in ranks.values()
                ),
                "ranks": ranks,
            }
    return None


def _validate_metrics(
    metrics: dict[str, Any], expected_steps: int
) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {}
    errors: list[str] = []
    required_steps = set(range(1, expected_steps + 1))
    for metric_name in REQUIRED_METRICS:
        values = metrics.get(metric_name)
        if not isinstance(values, dict):
            errors.append(f"missing metric {metric_name}")
            continue
        try:
            steps = {int(step) for step in values}
        except (TypeError, ValueError):
            errors.append(f"{metric_name} has a non-integer step")
            continue
        if (
            not required_steps.issubset(steps)
            or max(steps, default=-1) != expected_steps
        ):
            errors.append(
                f"{metric_name} must contain steps 1..{expected_steps} and end at "
                f"step {expected_steps}; found {sorted(steps)}"
            )
            continue
        try:
            numeric_values = [float(values[str(step)]) for step in required_steps]
        except (KeyError, TypeError, ValueError):
            errors.append(f"{metric_name} contains a non-numeric value")
            continue
        if not all(math.isfinite(value) for value in numeric_values):
            errors.append(f"{metric_name} contains a non-finite value")
            continue
        summary[metric_name] = {
            "steps": sorted(required_steps),
            "values": numeric_values,
        }
    return summary, errors


def check_lifecycle(
    log_path: Path,
    metrics_path: Path,
    expected_steps: int,
    expected_world_size: int,
) -> dict[str, Any]:
    log_text = log_path.read_text(errors="replace")
    metrics = json.loads(metrics_path.read_text())
    if not isinstance(metrics, dict):
        raise ValueError("metrics JSON must contain an object")

    offload_summary = _find_complete_nonzero_summary(log_text, expected_world_size)
    metric_summary, errors = _validate_metrics(metrics, expected_steps)
    if offload_summary is None:
        errors.append(
            "no complete activation-offload summary has positive moe_act and Total "
            f"values for all {expected_world_size} ranks"
        )

    return {
        "accepted": not errors,
        "expected_steps": expected_steps,
        "expected_world_size": expected_world_size,
        "offload_summary": offload_summary,
        "metrics": metric_summary,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a NeMo-RL activation-offload lifecycle gate"
    )
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--expected-world-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        result = check_lifecycle(
            log_path=args.log,
            metrics_path=args.metrics,
            expected_steps=args.expected_steps,
            expected_world_size=args.expected_world_size,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result = {"accepted": False, "errors": [str(error)]}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["accepted"]:
        print(f"Activation-offload lifecycle accepted: {args.output}")
        return 0
    print(
        "Activation-offload lifecycle rejected: " + "; ".join(result["errors"]),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
