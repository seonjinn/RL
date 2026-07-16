# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class StepRow:
    step: int
    e2e_time_s: float
    generation_time_s: float
    policy_time_s: float
    logprob_time_s: float
    throughput_tps: float
    generation_ratio: float
    acceptance_rate: float
    mean_accepted_length: float
    model: str
    recipe: str
    variant: str
    vllm_version: str
    container: str
    cluster: str
    temperature: float
    top_p: float
    max_osl: int
    cuda_graph_mode: str
    job_id: str
    log_path: str
    wandb_url: str
    runner: str
    graph_mode: str


@dataclass(frozen=True)
class RunSummary:
    model: str
    recipe: str
    variant: str
    vllm_version: str
    container: str
    cluster: str
    temperature: float
    top_p: float
    max_osl: int
    cuda_graph_mode: str
    job_id: str
    log_path: str
    wandb_url: str
    runner: str
    graph_mode: str
    step_start: int
    step_end: int
    step_count: int
    is_partial: bool
    e2e_time_s: float
    generation_time_s: float
    policy_time_s: float
    logprob_time_s: float
    throughput_tps: float
    generation_ratio: float
    acceptance_rate: float
    mean_accepted_length: float
    e2e_time_speedup: float | None = None
    generation_time_speedup: float | None = None
    throughput_speedup: float | None = None


class IncompleteWindowError(ValueError):
    """Raised when a requested metric window does not have every step."""


class NoMatchingBaselineError(ValueError):
    """Raised when no baseline has the candidate's exact controlled identity."""


_IDENTITY_FIELDS = (
    "model",
    "recipe",
    "vllm_version",
    "container",
    "cluster",
    "temperature",
    "top_p",
    "max_osl",
    "cuda_graph_mode",
)

_RUN_METADATA_FIELDS = (
    "model",
    "recipe",
    "variant",
    "vllm_version",
    "container",
    "cluster",
    "temperature",
    "top_p",
    "max_osl",
    "cuda_graph_mode",
    "job_id",
    "log_path",
    "wandb_url",
    "runner",
    "graph_mode",
)

_CSV_FIELDS = (
    "model",
    "recipe",
    "variant",
    "vllm_version",
    "container",
    "cluster",
    "temperature",
    "top_p",
    "max_osl",
    "cuda_graph_mode",
    "job_id",
    "log_path",
    "wandb_url",
    "runner",
    "graph_mode",
    "step_start",
    "step_end",
    "step_count",
    "is_partial",
    "e2e_time_s",
    "generation_time_s",
    "policy_time_s",
    "logprob_time_s",
    "throughput_tps",
    "generation_ratio",
    "acceptance_rate",
    "mean_accepted_length",
    "e2e_time_speedup",
    "generation_time_speedup",
    "throughput_speedup",
)


def load_steps(path: Path) -> tuple[StepRow, ...]:
    rows: list[StepRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(decoded, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(
                parse_step(cast(Mapping[str, object], decoded), path, line_number)
            )
    return tuple(rows)


def parse_step(
    data: Mapping[str, object], path: Path | None = None, line_number: int | None = None
) -> StepRow:
    location = _location(path, line_number)
    return StepRow(
        step=_required_int(data, "step", location),
        e2e_time_s=_required_float(data, "e2e_time_s", location),
        generation_time_s=_required_float(data, "generation_time_s", location),
        policy_time_s=_required_float(data, "policy_time_s", location),
        logprob_time_s=_required_float(data, "logprob_time_s", location),
        throughput_tps=_required_float(data, "throughput_tps", location),
        generation_ratio=_required_float(data, "generation_ratio", location),
        acceptance_rate=_required_float(data, "acceptance_rate", location),
        mean_accepted_length=_required_float(data, "mean_accepted_length", location),
        model=_required_string(data, "model", location),
        recipe=_required_string(data, "recipe", location),
        variant=_required_string(data, "variant", location),
        vllm_version=_required_string(data, "vllm_version", location),
        container=_required_string(data, "container", location),
        cluster=_required_string(data, "cluster", location),
        temperature=_required_float(data, "temperature", location),
        top_p=_required_float(data, "top_p", location),
        max_osl=_required_int(data, "max_osl", location),
        cuda_graph_mode=_required_string(data, "cuda_graph_mode", location),
        job_id=_required_string(data, "job_id", location),
        log_path=_required_string(data, "log_path", location),
        wandb_url=_required_string(data, "wandb_url", location),
        runner=_required_string(data, "runner", location),
        graph_mode=_required_string(data, "graph_mode", location),
    )


def summarize_steps(
    rows: Iterable[StepRow], start: int = 2, end: int = 20, allow_partial: bool = False
) -> RunSummary:
    if start > end:
        raise ValueError(f"Invalid step window: {start} > {end}")

    selected: dict[int, StepRow] = {}
    for row in rows:
        if start <= row.step <= end:
            if row.step in selected:
                raise ValueError(f"Duplicate step in requested window: {row.step}")
            selected[row.step] = row

    expected_steps = set(range(start, end + 1))
    missing_steps = sorted(expected_steps - selected.keys())
    if missing_steps and not allow_partial:
        missing_text = ", ".join(str(step) for step in missing_steps)
        raise IncompleteWindowError(
            f"Incomplete step window; missing steps: {missing_text}"
        )
    if not selected:
        raise IncompleteWindowError(f"No steps found in requested window {start}-{end}")

    ordered_rows = tuple(selected[step] for step in sorted(selected))
    _validate_constant_run_fields(ordered_rows)
    reference = ordered_rows[0]
    count = len(ordered_rows)
    return RunSummary(
        model=reference.model,
        recipe=reference.recipe,
        variant=reference.variant,
        vllm_version=reference.vllm_version,
        container=reference.container,
        cluster=reference.cluster,
        temperature=reference.temperature,
        top_p=reference.top_p,
        max_osl=reference.max_osl,
        cuda_graph_mode=reference.cuda_graph_mode,
        job_id=reference.job_id,
        log_path=reference.log_path,
        wandb_url=reference.wandb_url,
        runner=reference.runner,
        graph_mode=reference.graph_mode,
        step_start=start,
        step_end=end,
        step_count=count,
        is_partial=bool(missing_steps),
        e2e_time_s=_mean(row.e2e_time_s for row in ordered_rows),
        generation_time_s=_mean(row.generation_time_s for row in ordered_rows),
        policy_time_s=_mean(row.policy_time_s for row in ordered_rows),
        logprob_time_s=_mean(row.logprob_time_s for row in ordered_rows),
        throughput_tps=_mean(row.throughput_tps for row in ordered_rows),
        generation_ratio=_mean(row.generation_ratio for row in ordered_rows),
        acceptance_rate=_mean(row.acceptance_rate for row in ordered_rows),
        mean_accepted_length=_mean(row.mean_accepted_length for row in ordered_rows),
    )


def match_baseline(
    candidate: RunSummary, baselines: Sequence[RunSummary]
) -> RunSummary:
    exact_matches = [
        baseline for baseline in baselines if _same_identity(candidate, baseline)
    ]
    if not exact_matches:
        mismatches = _baseline_mismatch_fields(candidate, baselines)
        fields = (
            ", ".join(mismatches) if mismatches else "no baseline summaries supplied"
        )
        raise NoMatchingBaselineError(
            f"No exact baseline match for candidate; mismatched fields: {fields}"
        )
    if len(exact_matches) > 1:
        raise NoMatchingBaselineError("Multiple exact baseline matches for candidate")

    baseline = exact_matches[0]
    return replace(
        candidate,
        e2e_time_speedup=_speedup(
            baseline.e2e_time_s, candidate.e2e_time_s, "e2e_time_s"
        ),
        generation_time_speedup=_speedup(
            baseline.generation_time_s, candidate.generation_time_s, "generation_time_s"
        ),
        throughput_speedup=_speedup(
            candidate.throughput_tps, baseline.throughput_tps, "throughput_tps"
        ),
    )


def render_reports(
    summaries: Sequence[RunSummary], csv_path: Path, markdown_path: Path
) -> None:
    ordered = tuple(sorted(summaries, key=_sort_key))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for summary in ordered:
            writer.writerow(
                {field: _format_value(getattr(summary, field)) for field in _CSV_FIELDS}
            )

    markdown_path.write_text(_render_markdown(ordered), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect vLLM drafter-matrix step metrics."
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="JSONL step files, one per run"
    )
    parser.add_argument(
        "--csv", required=True, type=Path, help="Output CSV report path"
    )
    parser.add_argument(
        "--markdown", required=True, type=Path, help="Output Markdown report path"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit incomplete step windows and label their summaries partial.",
    )
    args = parser.parse_args(argv)

    summaries = [
        summarize_steps(load_steps(path), allow_partial=args.allow_partial)
        for path in args.inputs
    ]
    baselines = [summary for summary in summaries if summary.variant == "baseline"]
    matched_summaries = [
        summary if summary.variant == "baseline" else match_baseline(summary, baselines)
        for summary in summaries
    ]
    render_reports(matched_summaries, args.csv, args.markdown)
    return 0


def _location(path: Path | None, line_number: int | None) -> str:
    if path is None:
        return "step row"
    if line_number is None:
        return str(path)
    return f"{path}:{line_number}"


def _required_string(data: Mapping[str, object], field: str, location: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}: {field} must be a non-empty string")
    return value


def _required_float(data: Mapping[str, object], field: str, location: str) -> float:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{location}: {field} must be a number")
    return float(value)


def _required_int(data: Mapping[str, object], field: str, location: str) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}: {field} must be an integer")
    return value


def _validate_constant_run_fields(rows: Sequence[StepRow]) -> None:
    reference = rows[0]
    for row in rows[1:]:
        changed_fields = [
            field
            for field in _RUN_METADATA_FIELDS
            if getattr(row, field) != getattr(reference, field)
        ]
        if changed_fields:
            raise ValueError(
                f"Run metadata changed across steps: {', '.join(changed_fields)}"
            )


def _same_identity(candidate: RunSummary, baseline: RunSummary) -> bool:
    return all(
        getattr(candidate, field) == getattr(baseline, field)
        for field in _IDENTITY_FIELDS
    )


def _baseline_mismatch_fields(
    candidate: RunSummary, baselines: Sequence[RunSummary]
) -> list[str]:
    if not baselines:
        return []
    mismatches: list[str] = []
    for field in _IDENTITY_FIELDS:
        if all(
            getattr(candidate, field) != getattr(baseline, field)
            for baseline in baselines
        ):
            mismatches.append(field)
    return mismatches


def _speedup(numerator: float, denominator: float, metric: str) -> float:
    if denominator == 0.0:
        raise ValueError(f"Cannot compute speedup for {metric}: denominator is zero")
    return numerator / denominator


def _mean(values: Iterable[float]) -> float:
    collected = tuple(values)
    if not collected:
        raise ValueError("Cannot average an empty sequence")
    return math.fsum(collected) / len(collected)


def _sort_key(summary: RunSummary) -> tuple[str, ...]:
    return (
        summary.model,
        summary.recipe,
        summary.cluster,
        summary.variant,
        summary.job_id,
    )


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _render_markdown(summaries: Sequence[RunSummary]) -> str:
    header = (
        "| model | recipe | variant | cluster | job_id | runner | graph_mode | "
        "e2e_time_s | generation_time_s | throughput_tps | e2e_time_speedup | "
        "generation_time_speedup | throughput_speedup | partial | log_path | wandb_url |\n"
    )
    separator = (
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | "
        "---: | ---: | --- | --- | --- |\n"
    )
    rows = [header, separator]
    for summary in summaries:
        values = (
            summary.model,
            summary.recipe,
            summary.variant,
            summary.cluster,
            summary.job_id,
            summary.runner,
            summary.graph_mode,
            _format_value(summary.e2e_time_s),
            _format_value(summary.generation_time_s),
            _format_value(summary.throughput_tps),
            _format_value(summary.e2e_time_speedup),
            _format_value(summary.generation_time_speedup),
            _format_value(summary.throughput_speedup),
            _format_value(summary.is_partial),
            summary.log_path,
            summary.wandb_url,
        )
        rows.append(f"| {' | '.join(values)} |\n")
    return "".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
