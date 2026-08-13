#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


COMPONENT_METRICS = (
    "timing/train/total_step_time",
    "timing/train/generation",
    "timing/train/policy_and_reference_logprobs",
    "timing/train/policy_training",
    "timing/train/prepare_for_generation/transfer_and_update_weights",
    "performance/tokens_per_sec_per_gpu",
    "performance/generation_tokens_per_sec_per_gpu",
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
    "performance/policy_training_tokens_per_sec_per_gpu",
    "timing/train/valid_tokens_per_sec_per_gpu",
)
TOKEN_METRIC = "train/total_num_tokens"
GPU_MEMORY_METRIC = re.compile(r"^ray/node\.\d+\.gpu\.\d+\.mem_gb$")
HOST_MEMORY_METRIC = re.compile(r"^ray/node\.\d+\.mem_gb$")


def _load_metrics(path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text())
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return metrics


def _post_warmup_values(
    metrics: dict[str, Any], metric_name: str, warmup_steps: int
) -> tuple[list[int], list[float]]:
    raw_values = metrics.get(metric_name)
    if not isinstance(raw_values, dict):
        raise ValueError(f"missing metric {metric_name}")
    values_by_step: dict[int, float] = {}
    for raw_step, raw_value in raw_values.items():
        step = int(raw_step)
        value = float(raw_value)
        if step > warmup_steps:
            if not math.isfinite(value):
                raise ValueError(f"{metric_name} step {step} is non-finite")
            values_by_step[step] = value
    if not values_by_step:
        raise ValueError(
            f"{metric_name} has no observations after {warmup_steps} warmup steps"
        )
    steps = sorted(values_by_step)
    return steps, [values_by_step[step] for step in steps]


def _compare_metric(
    off: dict[str, Any], on: dict[str, Any], metric_name: str, warmup_steps: int
) -> dict[str, Any]:
    off_steps, off_values = _post_warmup_values(off, metric_name, warmup_steps)
    on_steps, on_values = _post_warmup_values(on, metric_name, warmup_steps)
    if off_steps != on_steps:
        raise ValueError(
            f"{metric_name} step mismatch: OFF {off_steps}, ON {on_steps}"
        )
    off_mean = statistics.fmean(off_values)
    on_mean = statistics.fmean(on_values)
    if off_mean == 0.0:
        raise ValueError(f"{metric_name} OFF mean is zero")
    return {
        "steps": off_steps,
        "sample_count": len(off_steps),
        "off_mean": off_mean,
        "off_population_stddev": statistics.pstdev(off_values),
        "on_mean": on_mean,
        "on_population_stddev": statistics.pstdev(on_values),
        "on_vs_off_percent": (on_mean / off_mean - 1.0) * 100.0,
        "favorable_direction": (
            "lower" if metric_name.startswith("timing/") else "higher"
        ),
    }


def _peak_memory(
    metrics: dict[str, Any], pattern: re.Pattern[str], warmup_steps: int
) -> float | None:
    values: list[float] = []
    for metric_name in metrics:
        if pattern.fullmatch(metric_name):
            _, metric_values = _post_warmup_values(
                metrics, metric_name, warmup_steps
            )
            values.extend(metric_values)
    return max(values) if values else None


def analyze_pair(
    off: dict[str, Any], on: dict[str, Any], warmup_steps: int
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for metric_name in COMPONENT_METRICS:
        if metric_name in off and metric_name in on:
            comparisons[metric_name] = _compare_metric(
                off, on, metric_name, warmup_steps
            )

    token_comparison = _compare_metric(off, on, TOKEN_METRIC, warmup_steps)
    off_token_total = token_comparison["off_mean"] * token_comparison["sample_count"]
    on_token_total = token_comparison["on_mean"] * token_comparison["sample_count"]
    return {
        "warmup_steps": warmup_steps,
        "metrics": comparisons,
        "workload": {
            "steps": token_comparison["steps"],
            "off_total_tokens": off_token_total,
            "on_total_tokens": on_token_total,
            "token_drift_percent": (on_token_total / off_token_total - 1.0)
            * 100.0,
        },
        "memory": {
            "off_peak_gpu_mem_gb": _peak_memory(
                off, GPU_MEMORY_METRIC, warmup_steps
            ),
            "on_peak_gpu_mem_gb": _peak_memory(on, GPU_MEMORY_METRIC, warmup_steps),
            "off_peak_host_mem_gb": _peak_memory(
                off, HOST_MEMORY_METRIC, warmup_steps
            ),
            "on_peak_host_mem_gb": _peak_memory(
                on, HOST_MEMORY_METRIC, warmup_steps
            ),
            "note": "Maximum sampled telemetry value after warmup; not allocator peak.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare dependency-matched activation-offload metrics"
    )
    parser.add_argument("--off", required=True, type=Path)
    parser.add_argument("--on", required=True, type=Path)
    parser.add_argument("--warmup-steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        result = analyze_pair(
            _load_metrics(args.off), _load_metrics(args.on), args.warmup_steps
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Pair comparison written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
