#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Mapping, NamedTuple, Sequence, TypedDict


BACKENDS = ("flashinfer_cutlass", "flashinfer_cutedsl")
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


class CsvRow(TypedDict):
    model: str
    backend: str
    step: int
    mean_generation_length: float
    total_step_seconds: float
    generation_seconds: float
    e2e_tokens_per_sec_per_gpu: float
    generation_tokens_per_sec_per_gpu: float


CSV_FIELDNAMES = tuple(CsvRow.__annotations__)


def _metric(pattern: re.Pattern[str], block: str) -> float | None:
    match = pattern.search(block)
    return float(match.group(1)) if match else None


def parse_training_results(log_text: str, source: str = "log") -> list[StepMetrics]:
    clean_text = ANSI_ESCAPE.sub("", log_text)
    steps: list[StepMetrics] = []
    for step, block in enumerate(clean_text.split("Training Results:")[1:], start=1):
        mean_generation_length = _metric(MEAN_GENERATION_LENGTH_PATTERN, block)
        total_step_seconds = _metric(TOTAL_STEP_PATTERN, block)
        generation_seconds = _metric(GENERATION_PATTERN, block)
        e2e_throughput = _metric(E2E_THROUGHPUT_PATTERN, block)
        generation_throughput = _metric(GENERATION_THROUGHPUT_PATTERN, block)
        if (
            mean_generation_length is None
            or total_step_seconds is None
            or generation_seconds is None
            or e2e_throughput is None
            or generation_throughput is None
        ):
            missing_metrics = [
                metric_name
                for metric_name, metric_value in (
                    ("Mean Generation Length", mean_generation_length),
                    ("Total step time", total_step_seconds),
                    ("generation", generation_seconds),
                    ("E2E (Tokens/sec/gpu)", e2e_throughput),
                    (
                        "Generation Worker Group (Tokens/sec/gpu)",
                        generation_throughput,
                    ),
                )
                if metric_value is None
            ]
            raise ValueError(
                f"Incomplete Training Results block {step} for {source}: missing "
                f"{', '.join(missing_metrics)}"
            )
        steps.append(
            StepMetrics(
                step=step,
                mean_generation_length=mean_generation_length,
                total_step_seconds=total_step_seconds,
                generation_seconds=generation_seconds,
                e2e_tokens_per_sec_per_gpu=e2e_throughput,
                generation_tokens_per_sec_per_gpu=generation_throughput,
            )
        )
    return steps


def _find_driver_log(model: str, run_root: Path, backend: str) -> Path:
    matches = sorted((run_root / backend).glob("*-logs/ray-driver.log"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one driver log for {model}/{backend}, found {len(matches)}"
        )
    return matches[0]


def _measured_steps(
    model: str,
    backend: str,
    steps: Sequence[StepMetrics],
    first_step: int,
    last_step: int,
) -> list[StepMetrics]:
    selected = [step for step in steps if first_step <= step.step <= last_step]
    expected_steps = list(range(first_step, last_step + 1))
    actual_steps = [step.step for step in selected]
    if actual_steps != expected_steps:
        raise ValueError(
            f"Expected complete measured steps for {model}/{backend}: "
            f"expected {expected_steps}, found {actual_steps}"
        )
    return selected


def summarize_steps(steps: Sequence[StepMetrics]) -> StepSummary:
    return StepSummary(
        first_step=steps[0].step,
        last_step=steps[-1].step,
        num_steps=len(steps),
        mean_generation_length_mean=fmean(
            step.mean_generation_length for step in steps
        ),
        total_step_seconds_mean=fmean(step.total_step_seconds for step in steps),
        generation_seconds_mean=fmean(step.generation_seconds for step in steps),
        e2e_tokens_per_sec_per_gpu_mean=fmean(
            step.e2e_tokens_per_sec_per_gpu for step in steps
        ),
        generation_tokens_per_sec_per_gpu_mean=fmean(
            step.generation_tokens_per_sec_per_gpu for step in steps
        ),
    )


def validate_paired_steps(
    model: str, cutlass_steps: Sequence[StepMetrics], cutedsl_steps: Sequence[StepMetrics]
) -> None:
    if len(cutlass_steps) != len(cutedsl_steps):
        raise ValueError(
            f"Paired measured-step count mismatch for {model}: "
            f"flashinfer_cutlass={len(cutlass_steps)}, "
            f"flashinfer_cutedsl={len(cutedsl_steps)}"
        )
    for cutlass_step, cutedsl_step in zip(cutlass_steps, cutedsl_steps, strict=True):
        if cutlass_step.step != cutedsl_step.step:
            raise ValueError(
                f"Paired measured-step mismatch for {model}: "
                f"flashinfer_cutlass={cutlass_step.step}, "
                f"flashinfer_cutedsl={cutedsl_step.step}"
            )
        if cutlass_step.mean_generation_length != cutedsl_step.mean_generation_length:
            raise ValueError(
                f"Paired mean generation length mismatch for {model} at step "
                f"{cutlass_step.step}: "
                f"flashinfer_cutlass={cutlass_step.mean_generation_length}, "
                f"flashinfer_cutedsl={cutedsl_step.mean_generation_length}"
            )


def _with_cutlass_normalization(
    summary: StepSummary, cutlass_summary: StepSummary
) -> dict[str, int | float]:
    metrics = summary._asdict()
    metrics.update(
        {
            "generation_tokens_per_sec_per_gpu_cutlass_normalized": (
                summary.generation_tokens_per_sec_per_gpu_mean
                / cutlass_summary.generation_tokens_per_sec_per_gpu_mean
            ),
            "e2e_tokens_per_sec_per_gpu_cutlass_normalized": (
                summary.e2e_tokens_per_sec_per_gpu_mean
                / cutlass_summary.e2e_tokens_per_sec_per_gpu_mean
            ),
            "generation_latency_speedup_vs_cutlass": (
                cutlass_summary.generation_seconds_mean
                / summary.generation_seconds_mean
            ),
            "e2e_latency_speedup_vs_cutlass": (
                cutlass_summary.total_step_seconds_mean
                / summary.total_step_seconds_mean
            ),
        }
    )
    return metrics


def validate_normalization_denominators(
    model: str,
    backend: str,
    summary: StepSummary,
    first_step: int,
    last_step: int,
) -> None:
    for metric_name, metric_value in (
        (
            "generation_tokens_per_sec_per_gpu_mean",
            summary.generation_tokens_per_sec_per_gpu_mean,
        ),
        ("e2e_tokens_per_sec_per_gpu_mean", summary.e2e_tokens_per_sec_per_gpu_mean),
        ("generation_seconds_mean", summary.generation_seconds_mean),
        ("total_step_seconds_mean", summary.total_step_seconds_mean),
    ):
        if metric_value <= 0:
            raise ValueError(
                f"Invalid normalization denominator for {model}/{backend}, "
                f"steps {first_step}-{last_step}: {metric_name} must be positive"
            )


def write_results(
    model_run_roots: Mapping[str, Path],
    output_dir: Path,
    first_step: int = 3,
    last_step: int = 8,
) -> None:
    if first_step > last_step:
        raise ValueError("first_step must be less than or equal to last_step")

    rows: list[CsvRow] = []
    summaries: dict[str, dict[str, StepSummary]] = {}
    for model, raw_run_root in model_run_roots.items():
        run_root = Path(raw_run_root)
        measured_steps: dict[str, list[StepMetrics]] = {}
        for backend in BACKENDS:
            log_path = _find_driver_log(model, run_root, backend)
            steps = parse_training_results(
                log_path.read_text(errors="replace"), source=f"{model}/{backend}"
            )
            rows.extend(
                CsvRow(model=model, backend=backend, **step._asdict())
                for step in steps
            )
            measured_steps[backend] = _measured_steps(
                model, backend, steps, first_step, last_step
            )

        validate_paired_steps(
            model,
            measured_steps["flashinfer_cutlass"],
            measured_steps["flashinfer_cutedsl"],
        )
        summaries[model] = {
            backend: summarize_steps(measured_steps[backend]) for backend in BACKENDS
        }
        for backend, summary in summaries[model].items():
            validate_normalization_denominators(
                model, backend, summary, first_step, last_step
            )

    normalized_summaries = {
        model: {
            backend: _with_cutlass_normalization(
                summary, model_summaries["flashinfer_cutlass"]
            )
            for backend, summary in model_summaries.items()
        }
        for model, model_summaries in summaries.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "step_metrics.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(normalized_summaries, indent=2, sort_keys=True) + "\n"
    )


def _parse_model_run(value: str) -> tuple[str, Path]:
    model, separator, run_root = value.partition("=")
    if not separator or not model or not run_root:
        raise argparse.ArgumentTypeError("Expected MODEL=RUN_ROOT")
    return model, Path(run_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-run",
        action="append",
        type=_parse_model_run,
        required=True,
        metavar="MODEL=RUN_ROOT",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-step", type=int, default=3)
    parser.add_argument("--last-step", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_run_roots = dict(args.model_run)
    if len(model_run_roots) != len(args.model_run):
        raise ValueError("Each MODEL may be supplied only once")
    write_results(
        model_run_roots,
        args.output_dir,
        first_step=args.first_step,
        last_step=args.last_step,
    )


if __name__ == "__main__":
    main()
