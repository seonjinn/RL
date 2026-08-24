#!/usr/bin/env python3
"""Summarize steady-state metrics from one NeMo-RL driver log."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_PATTERN = re.compile(r"Step\s+(\d+)/(\d+)")
METRIC_PATTERNS = {
    "loss": re.compile(r"Loss:\s+(-?[0-9.eE+]+)"),
    "gen_kl_error": re.compile(r"Generation KL Error:\s+(-?[0-9.eE+]+)"),
    "reward": re.compile(r"Avg Reward:\s+(-?[0-9.eE+]+)"),
    "generation_length": re.compile(r"Mean Generation Length:\s+(-?[0-9.eE+]+)"),
    "step_time_s": re.compile(r"Total step time:\s+([0-9.eE+]+)s"),
    "policy_training_s": re.compile(r"policy_training:\s+([0-9.eE+]+)s"),
    "logprob_s": re.compile(r"policy_and_reference_logprobs:\s+([0-9.eE+]+)s"),
    "generation_s": re.compile(r"generation:\s+([0-9.eE+]+)s"),
    "refit_total_s": re.compile(r"prepare_for_generation/total:\s+([0-9.eE+]+)s"),
    "transfer_update_s": re.compile(
        r"prepare_for_generation/transfer_and_update_weights:\s+([0-9.eE+]+)s"
    ),
    "e2e_tokens_per_s_per_gpu": re.compile(
        r"E2E \(Tokens/sec/gpu\):\s+([0-9.eE+]+)"
    ),
    "generation_tokens_per_s_per_gpu": re.compile(
        r"Generation Worker Group \(Tokens/sec/gpu\):\s+([0-9.eE+]+)"
    ),
}


def parse_steps(path: Path) -> list[dict[str, float | int]]:
    steps: dict[int, dict[str, float | int]] = {}
    current_step: int | None = None
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        step_match = STEP_PATTERN.search(line)
        if step_match is not None:
            current_step = int(step_match.group(1))
            steps.setdefault(current_step, {"step": current_step})
            continue
        if current_step is None:
            continue
        for name, pattern in METRIC_PATTERNS.items():
            match = pattern.search(line)
            if match is not None:
                steps[current_step][name] = float(match.group(1))
    return [steps[step] for step in sorted(steps)]


def summarize(
    steps: list[dict[str, float | int]], start_step: int, end_step: int | None
) -> dict[str, object]:
    selected = [
        step
        for step in steps
        if int(step["step"]) >= start_step
        and (end_step is None or int(step["step"]) <= end_step)
        and "step_time_s" in step
    ]
    if not selected:
        raise ValueError("no completed steps matched the requested window")

    metrics: dict[str, dict[str, float | int]] = {}
    for name in METRIC_PATTERNS:
        values = [float(step[name]) for step in selected if name in step]
        if not values:
            continue
        metrics[name] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    for metric in metrics.values():
        for key, value in tuple(metric.items()):
            if isinstance(value, float) and not math.isfinite(value):
                metric[key] = None
    return {
        "window": {
            "start_step": start_step,
            "end_step": end_step,
            "completed_steps": [int(step["step"]) for step in selected],
        },
        "metrics": metrics,
        "steps": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("driver_log", type=Path)
    parser.add_argument("--start-step", type=int, default=3)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = summarize(
        parse_steps(args.driver_log), args.start_step, args.end_step
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
