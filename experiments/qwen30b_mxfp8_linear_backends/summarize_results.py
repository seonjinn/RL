#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import fmean
from typing import NamedTuple, Sequence


BACKENDS = (
    "flashinfer_cutedsl",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
    "flashinfer_trtllm_adaptive",
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TOTAL_STEP_PATTERN = re.compile(r"Total step time:\s*([0-9.]+)s")
GENERATION_PATTERN = re.compile(r"generation:\s*([0-9.]+)s")
E2E_THROUGHPUT_PATTERN = re.compile(r"E2E \(Tokens/sec/gpu\):\s*([0-9.]+)")
GENERATION_THROUGHPUT_PATTERN = re.compile(
    r"Generation Worker Group \(Tokens/sec/gpu\):\s*([0-9.]+)"
)
MEAN_GENERATION_LENGTH_PATTERN = re.compile(
    r"Mean Generation Length:\s*([0-9.]+)"
)


class StepMetrics(NamedTuple):
    step: int
    mean_generation_length: float
    total_step_seconds: float
    generation_seconds: float
    e2e_tokens_per_sec_per_gpu: float
    generation_tokens_per_sec_per_gpu: float


class StepSummary(NamedTuple):
    first_step: int
    last_step: int
    num_steps: int
    mean_generation_length_mean: float
    total_step_seconds_mean: float
    generation_seconds_mean: float
    e2e_tokens_per_sec_per_gpu_mean: float
    generation_tokens_per_sec_per_gpu_mean: float


def _metric(pattern: re.Pattern[str], block: str) -> float | None:
    match = pattern.search(block)
    return float(match.group(1)) if match else None


def parse_training_results(log_text: str) -> list[StepMetrics]:
    clean_text = ANSI_ESCAPE.sub("", log_text)
    steps: list[StepMetrics] = []
    for block in clean_text.split("Training Results:")[1:]:
        mean_generation_length = _metric(MEAN_GENERATION_LENGTH_PATTERN, block)
        total_step_seconds = _metric(TOTAL_STEP_PATTERN, block)
        generation_seconds = _metric(GENERATION_PATTERN, block)
        e2e_throughput = _metric(E2E_THROUGHPUT_PATTERN, block)
        generation_throughput = _metric(GENERATION_THROUGHPUT_PATTERN, block)
        metrics = (
            mean_generation_length,
            total_step_seconds,
            generation_seconds,
            e2e_throughput,
            generation_throughput,
        )
        if any(metric is None for metric in metrics):
            continue
        steps.append(
            StepMetrics(
                step=len(steps) + 1,
                mean_generation_length=float(mean_generation_length),
                total_step_seconds=float(total_step_seconds),
                generation_seconds=float(generation_seconds),
                e2e_tokens_per_sec_per_gpu=float(e2e_throughput),
                generation_tokens_per_sec_per_gpu=float(generation_throughput),
            )
        )
    return steps


def summarize_steps(steps: Sequence[StepMetrics], first_step: int) -> StepSummary:
    selected = [step for step in steps if step.step >= first_step]
    if not selected:
        raise ValueError(f"No completed steps at or after step {first_step}")
    return StepSummary(
        first_step=selected[0].step,
        last_step=selected[-1].step,
        num_steps=len(selected),
        mean_generation_length_mean=fmean(
            step.mean_generation_length for step in selected
        ),
        total_step_seconds_mean=fmean(step.total_step_seconds for step in selected),
        generation_seconds_mean=fmean(step.generation_seconds for step in selected),
        e2e_tokens_per_sec_per_gpu_mean=fmean(
            step.e2e_tokens_per_sec_per_gpu for step in selected
        ),
        generation_tokens_per_sec_per_gpu_mean=fmean(
            step.generation_tokens_per_sec_per_gpu for step in selected
        ),
    )


def _find_driver_log(run_root: Path, backend: str) -> Path:
    matches = sorted((run_root / backend).glob("*-logs/ray-driver.log"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one ray-driver.log for {backend}, found {len(matches)}"
        )
    return matches[0]


def write_results(
    run_root: Path,
    output_dir: Path,
    first_step: int,
    backends: Sequence[str] = BACKENDS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    summaries: dict[str, dict[str, int | float]] = {}
    for backend in backends:
        log_path = _find_driver_log(run_root, backend)
        steps = parse_training_results(log_path.read_text(errors="replace"))
        for step in steps:
            rows.append({"backend": backend, **step._asdict()})
        summaries[backend] = summarize_steps(steps, first_step)._asdict()

    csv_path = output_dir / "step_metrics.csv"
    with csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-step", type=int, default=3)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_results(args.run_root, args.output_dir, args.first_step, args.backends)


if __name__ == "__main__":
    main()
