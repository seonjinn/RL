"""Compare matched BF16 and MXFP8 ntrace summaries per generated token."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


STEP_PATTERN = re.compile(r"=+ Step (?P<step>\d+)/\d+ =+")
LENGTH_PATTERN = re.compile(r"Mean Generation Length:\s*(?P<length>[0-9.]+)")
BATCH_PATTERN = re.compile(r"Generating responses for batch of size (?P<size>\d+)")
ITERATION_PATTERN = re.compile(
    r"rollout started iteration=(?P<iteration>\d+) "
    r"step_id=step(?P<step>\d+)/"
)
CAPTURE_ITER_PATTERN = re.compile(r"armed rank=\d+ .*capture_iter=(?P<iteration>\d+)")


def parse_run_log(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    current_step: int | None = None
    lengths: dict[int, float] = {}
    for line in text.splitlines():
        if match := STEP_PATTERN.search(line):
            current_step = int(match.group("step"))
        elif current_step is not None and (match := LENGTH_PATTERN.search(line)):
            lengths[current_step] = float(match.group("length"))

    batch_sizes = {int(match.group("size")) for match in BATCH_PATTERN.finditer(text)}
    if len(batch_sizes) != 1:
        raise ValueError(f"expected one generation batch size, found {batch_sizes}")

    iteration_steps: dict[int, int] = {}
    for match in ITERATION_PATTERN.finditer(text):
        iteration = int(match.group("iteration"))
        step = int(match.group("step"))
        previous = iteration_steps.setdefault(iteration, step)
        if previous != step:
            raise ValueError(
                f"iteration {iteration} maps to both step {previous} and {step}"
            )

    if not iteration_steps:
        raise ValueError("no ntrace iteration markers found")
    capture_iters = {
        int(match.group("iteration")) for match in CAPTURE_ITER_PATTERN.finditer(text)
    }
    if len(capture_iters) > 1:
        raise ValueError(f"workers use different capture iterations: {capture_iters}")
    capture_iter = capture_iters.pop() if capture_iters else 0
    missing = sorted(set(iteration_steps.values()) - set(lengths))
    if missing:
        raise ValueError(f"missing generation lengths for steps {missing}")
    return {
        "batch_size": batch_sizes.pop(),
        "generation_lengths": lengths,
        "iteration_steps": iteration_steps,
        "capture_iter": capture_iter,
    }


def load_arm(summary_path: Path, log_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    run = parse_run_log(log_path)
    ranks = summary["ranks"]
    iterations = summary["iterations"]
    rank_count = len(ranks)
    if rank_count == 0:
        raise ValueError("summary has no ranks")
    summary_indices = {int(iteration["index"]) for iteration in iterations}
    if summary_indices != set(range(len(iterations))):
        raise ValueError(f"summary iterations are not contiguous: {summary_indices}")

    normalized_iterations = []
    for iteration in iterations:
        index = int(iteration["index"])
        rollout_iteration = index + run["capture_iter"]
        if rollout_iteration not in run["iteration_steps"]:
            raise ValueError(
                f"trace iteration {index} maps to missing rollout iteration "
                f"{rollout_iteration}"
            )
        step = run["iteration_steps"][rollout_iteration]
        mean_length = run["generation_lengths"][step]
        tokens_per_rank = mean_length * run["batch_size"] / rank_count
        categories = iteration["stack_categories_s"]
        normalized_iterations.append(
            {
                "index": index,
                "step": step,
                "mean_generation_length": mean_length,
                "tokens_per_rank": tokens_per_rank,
                "wall_s_mean": float(iteration["step_s"]["mean"]),
                "wall_s_max": float(iteration["step_s"]["max"]),
                "active_s_mean": float(iteration["active_s"]["mean"]),
                "idle_s_mean": float(iteration["idle_s"]["mean"]),
                "moe_s_mean": float(categories.get("moe", 0.0)),
            }
        )

    raw_keys = sorted(
        set().union(*(rank["raw_kernel_categories_s"] for rank in ranks))
    )
    mean_tokens_per_rank = statistics.fmean(
        iteration["tokens_per_rank"] for iteration in normalized_iterations
    )
    raw_s_per_mtoken = {
        key: statistics.fmean(
            rank["raw_kernel_categories_s"].get(key, 0.0) for rank in ranks
        )
        / mean_tokens_per_rank
        * 1e6
        for key in raw_keys
    }
    return {
        "rank_count": rank_count,
        "batch_size": run["batch_size"],
        "iterations": normalized_iterations,
        "raw_s_per_mtoken": raw_s_per_mtoken,
    }


def steady_summary(arm: dict[str, Any], indices: tuple[int, ...]) -> dict[str, float]:
    selected = [arm["iterations"][index] for index in indices]
    tokens = sum(iteration["tokens_per_rank"] for iteration in selected)

    def per_mtoken(key: str) -> float:
        return sum(iteration[key] for iteration in selected) / tokens * 1e6

    wall_s = sum(iteration["wall_s_max"] for iteration in selected)
    return {
        "tokens_per_rank": tokens,
        "wall_s_critical": wall_s,
        "throughput_tokens_per_s_per_gpu": tokens / wall_s,
        "active_s_per_mtoken": per_mtoken("active_s_mean"),
        "idle_s_per_mtoken": per_mtoken("idle_s_mean"),
        "moe_s_per_mtoken": per_mtoken("moe_s_mean"),
    }


def compare(
    bf16: dict[str, Any], mxfp8: dict[str, Any], indices: tuple[int, ...] = (1, 2)
) -> dict[str, Any]:
    if bf16["rank_count"] != mxfp8["rank_count"]:
        raise ValueError("rank counts differ")
    if bf16["batch_size"] != mxfp8["batch_size"]:
        raise ValueError("generation batch sizes differ")

    bf16_steady = steady_summary(bf16, indices)
    mxfp8_steady = steady_summary(mxfp8, indices)
    return {
        "steady_iteration_indices": list(indices),
        "bf16": {**bf16_steady, "raw_s_per_mtoken": bf16["raw_s_per_mtoken"]},
        "mxfp8": {
            **mxfp8_steady,
            "raw_s_per_mtoken": mxfp8["raw_s_per_mtoken"],
        },
        "ratios": {
            "throughput_speedup": mxfp8_steady["throughput_tokens_per_s_per_gpu"]
            / bf16_steady["throughput_tokens_per_s_per_gpu"],
            "active_time_reduction": 1.0
            - mxfp8_steady["active_s_per_mtoken"]
            / bf16_steady["active_s_per_mtoken"],
            "moe_time_reduction": 1.0
            - mxfp8_steady["moe_s_per_mtoken"]
            / bf16_steady["moe_s_per_mtoken"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bf16_summary", type=Path)
    parser.add_argument("bf16_log", type=Path)
    parser.add_argument("mxfp8_summary", type=Path)
    parser.add_argument("mxfp8_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = compare(
        load_arm(args.bf16_summary, args.bf16_log),
        load_arm(args.mxfp8_summary, args.mxfp8_log),
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["ratios"], indent=2))


if __name__ == "__main__":
    main()
