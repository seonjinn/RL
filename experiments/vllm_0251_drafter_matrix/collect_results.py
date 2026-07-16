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
from typing import Literal, cast


RunStatus = Literal["completed", "failed", "unsupported"]
_BASELINE_VARIANTS = frozenset(("baseline", "baseline_mrv1"))


@dataclass(frozen=True)
class RunMetadata:
    """Provenance and controlled settings shared by every row of a run."""

    model: str
    recipe: str
    variant: str
    vllm_version: str
    container: str
    cluster: str
    temperature: float
    top_p: float
    max_osl: int
    requested_cuda_graph_mode: str
    resolved_cuda_graph_mode: str
    cuda_graph_coverage: float | None
    job_id: str
    log_path: str
    wandb_url: str
    runner: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, object], location: str) -> RunMetadata:
        """Parse run provenance from one JSON record."""
        return cls(
            model=_required_string(data, "model", location),
            recipe=_required_string(data, "recipe", location),
            variant=_required_string(data, "variant", location),
            vllm_version=_required_string(data, "vllm_version", location),
            container=_required_string(data, "container", location),
            cluster=_required_string(data, "cluster", location),
            temperature=_required_nonnegative_float(data, "temperature", location),
            top_p=_required_ratio(data, "top_p", location),
            max_osl=_required_positive_int(data, "max_osl", location),
            requested_cuda_graph_mode=_required_string(
                data, "requested_cuda_graph_mode", location
            ),
            resolved_cuda_graph_mode=_required_string(
                data, "resolved_cuda_graph_mode", location
            ),
            cuda_graph_coverage=_optional_ratio(data, "cuda_graph_coverage", location),
            job_id=_required_string(data, "job_id", location),
            log_path=_required_string(data, "log_path", location),
            wandb_url=_required_string(data, "wandb_url", location),
            runner=_required_string(data, "runner", location),
        )


@dataclass(frozen=True)
class StepRow:
    """Validated metrics for one completed training step."""

    step: int
    e2e_time_s: float
    generation_time_s: float
    policy_time_s: float
    logprob_time_s: float
    e2e_throughput_tps_per_gpu: float
    generation_throughput_tps_per_gpu: float
    generation_ratio: float
    acceptance_rate: float | None
    mean_accepted_length: float | None
    metadata: RunMetadata


@dataclass(frozen=True)
class RunSummary:
    """Validated averages for a completed run over a requested step window."""

    metadata: RunMetadata
    step_start: int
    step_end: int
    step_count: int
    is_partial: bool
    e2e_time_s: float
    generation_time_s: float
    policy_time_s: float
    logprob_time_s: float
    e2e_throughput_tps_per_gpu: float
    generation_throughput_tps_per_gpu: float
    generation_ratio: float
    acceptance_rate: float | None
    mean_accepted_length: float | None
    e2e_time_speedup: float | None = None
    generation_time_speedup: float | None = None
    e2e_throughput_speedup: float | None = None
    generation_throughput_speedup: float | None = None


@dataclass(frozen=True)
class ReportRow:
    """A completed summary or an explicit failed or unsupported run outcome."""

    metadata: RunMetadata
    status: RunStatus
    reason: str | None = None
    summary: RunSummary | None = None

    def __post_init__(self) -> None:
        """Require metrics only for completed rows and reasons otherwise."""
        if self.status == "completed":
            if self.summary is None:
                raise ValueError("Completed report rows require a run summary")
            if self.reason is not None:
                raise ValueError("Completed report rows cannot have a reason")
            if self.summary.metadata != self.metadata:
                raise ValueError("Completed report row metadata must match its summary")
            return
        if self.summary is not None:
            raise ValueError(
                "Failed and unsupported report rows cannot have a run summary"
            )
        if not self.reason:
            raise ValueError("Failed and unsupported report rows require a reason")

    @classmethod
    def completed(cls, summary: RunSummary) -> ReportRow:
        """Create a completed report row from a validated summary."""
        return cls(metadata=summary.metadata, status="completed", summary=summary)


class IncompleteWindowError(ValueError):
    """Raised when a requested metric window does not have every step."""


class NoMatchingBaselineError(ValueError):
    """Raised when no baseline has the candidate's exact controlled identity."""


_IDENTITY_FIELD_NAMES = (
    "model",
    "recipe",
    "vllm_version",
    "container",
    "cluster",
    "temperature",
    "top_p",
    "max_osl",
    "requested_cuda_graph_mode",
    "resolved_cuda_graph_mode",
    "runner",
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
    "requested_cuda_graph_mode",
    "resolved_cuda_graph_mode",
    "cuda_graph_coverage",
    "job_id",
    "log_path",
    "wandb_url",
    "runner",
    "status",
    "reason",
    "step_start",
    "step_end",
    "step_count",
    "is_partial",
    "e2e_time_s",
    "generation_time_s",
    "policy_time_s",
    "logprob_time_s",
    "e2e_throughput_tps_per_gpu",
    "generation_throughput_tps_per_gpu",
    "generation_ratio",
    "acceptance_rate",
    "mean_accepted_length",
    "e2e_time_speedup",
    "generation_time_speedup",
    "e2e_throughput_speedup",
    "generation_throughput_speedup",
)


def load_steps(path: Path) -> tuple[StepRow, ...]:
    """Load completed step records from a JSONL file."""
    return tuple(
        parse_step(data, path, line_number)
        for line_number, data in _load_json_records(path)
    )


def load_report_row(path: Path, allow_partial: bool = False) -> ReportRow:
    """Load a completed, failed, or unsupported run record from JSONL."""
    records = _load_json_records(path)
    if not records:
        raise ValueError(f"{path}: no JSON records found")
    first_line_number, first_data = records[0]
    location = _location(path, first_line_number)
    status = _status(first_data, location)
    if status == "completed":
        if any(
            _status(data, _location(path, line_number)) != "completed"
            for line_number, data in records
        ):
            raise ValueError(f"{path}: completed step files cannot mix run statuses")
        return ReportRow.completed(
            summarize_steps(
                (parse_step(data, path, line_number) for line_number, data in records),
                allow_partial=allow_partial,
            )
        )
    if len(records) != 1:
        raise ValueError(
            f"{path}: {status} records must contain exactly one JSON object"
        )
    return ReportRow(
        metadata=RunMetadata.from_mapping(first_data, location),
        status=status,
        reason=_required_string(first_data, "reason", location),
    )


def parse_step(
    data: Mapping[str, object], path: Path | None = None, line_number: int | None = None
) -> StepRow:
    """Parse and validate one completed JSONL step record."""
    location = _location(path, line_number)
    if _status(data, location) != "completed":
        raise ValueError(f"{location}: only completed records contain step metrics")
    metadata = RunMetadata.from_mapping(data, location)
    if metadata.variant in _BASELINE_VARIANTS:
        acceptance_rate = _optional_ratio(data, "acceptance_rate", location)
        mean_accepted_length = _optional_nonnegative_float(
            data, "mean_accepted_length", location
        )
    else:
        acceptance_rate = _required_ratio(data, "acceptance_rate", location)
        mean_accepted_length = _required_positive_float(
            data, "mean_accepted_length", location
        )
    return StepRow(
        step=_required_positive_int(data, "step", location),
        e2e_time_s=_required_positive_float(data, "e2e_time_s", location),
        generation_time_s=_required_positive_float(data, "generation_time_s", location),
        policy_time_s=_required_nonnegative_float(data, "policy_time_s", location),
        logprob_time_s=_required_nonnegative_float(data, "logprob_time_s", location),
        e2e_throughput_tps_per_gpu=_required_positive_float(
            data, "e2e_throughput_tps_per_gpu", location
        ),
        generation_throughput_tps_per_gpu=_required_positive_float(
            data, "generation_throughput_tps_per_gpu", location
        ),
        generation_ratio=_required_ratio(data, "generation_ratio", location),
        acceptance_rate=acceptance_rate,
        mean_accepted_length=mean_accepted_length,
        metadata=metadata,
    )


def summarize_steps(
    rows: Iterable[StepRow], start: int = 2, end: int = 20, allow_partial: bool = False
) -> RunSummary:
    """Average validated step metrics over the requested inclusive step window."""
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
    _validate_constant_run_metadata(ordered_rows)
    reference = ordered_rows[0]
    return RunSummary(
        metadata=reference.metadata,
        step_start=start,
        step_end=end,
        step_count=len(ordered_rows),
        is_partial=bool(missing_steps),
        e2e_time_s=_mean(row.e2e_time_s for row in ordered_rows),
        generation_time_s=_mean(row.generation_time_s for row in ordered_rows),
        policy_time_s=_mean(row.policy_time_s for row in ordered_rows),
        logprob_time_s=_mean(row.logprob_time_s for row in ordered_rows),
        e2e_throughput_tps_per_gpu=_mean(
            row.e2e_throughput_tps_per_gpu for row in ordered_rows
        ),
        generation_throughput_tps_per_gpu=_mean(
            row.generation_throughput_tps_per_gpu for row in ordered_rows
        ),
        generation_ratio=_mean(row.generation_ratio for row in ordered_rows),
        acceptance_rate=_optional_mean(row.acceptance_rate for row in ordered_rows),
        mean_accepted_length=_optional_mean(
            row.mean_accepted_length for row in ordered_rows
        ),
    )


def match_baseline(
    candidate: RunSummary, baselines: Sequence[RunSummary]
) -> RunSummary:
    """Attach speedups after finding one exact controlled baseline."""
    _require_final_window(candidate)
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
    _require_final_window(baseline)
    return replace(
        candidate,
        e2e_time_speedup=_speedup(
            baseline.e2e_time_s, candidate.e2e_time_s, "e2e_time_s"
        ),
        generation_time_speedup=_speedup(
            baseline.generation_time_s, candidate.generation_time_s, "generation_time_s"
        ),
        e2e_throughput_speedup=_speedup(
            candidate.e2e_throughput_tps_per_gpu,
            baseline.e2e_throughput_tps_per_gpu,
            "e2e_throughput_tps_per_gpu",
        ),
        generation_throughput_speedup=_speedup(
            candidate.generation_throughput_tps_per_gpu,
            baseline.generation_throughput_tps_per_gpu,
            "generation_throughput_tps_per_gpu",
        ),
    )


def render_reports(
    rows: Sequence[ReportRow | RunSummary], csv_path: Path, markdown_path: Path
) -> None:
    """Write deterministic CSV and Markdown reports for every run outcome."""
    report_rows = tuple(
        row if isinstance(row, ReportRow) else ReportRow.completed(row) for row in rows
    )
    ordered = tuple(sorted(report_rows, key=_sort_key))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                dict(
                    zip(
                        _CSV_FIELDS,
                        (_format_value(value) for value in _report_values(row)),
                    )
                )
            )

    markdown_path.write_text(_render_markdown(ordered), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Collect drafter-matrix JSONL step files into CSV and Markdown reports."""
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

    report_rows = [
        load_report_row(path, allow_partial=args.allow_partial) for path in args.inputs
    ]
    baselines = [
        row.summary
        for row in report_rows
        if row.status == "completed"
        and row.summary is not None
        and not row.summary.is_partial
        and row.metadata.variant in _BASELINE_VARIANTS
    ]
    matched_rows = [
        ReportRow.completed(match_baseline(row.summary, baselines))
        if row.status == "completed"
        and row.summary is not None
        and not row.summary.is_partial
        and row.metadata.variant not in _BASELINE_VARIANTS
        else row
        for row in report_rows
    ]
    render_reports(matched_rows, args.csv, args.markdown)
    return 0


def _load_json_records(path: Path) -> tuple[tuple[int, Mapping[str, object]], ...]:
    records: list[tuple[int, Mapping[str, object]]] = []
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
            records.append((line_number, cast(Mapping[str, object], decoded)))
    return tuple(records)


def _location(path: Path | None, line_number: int | None) -> str:
    if path is None:
        return "step row"
    if line_number is None:
        return str(path)
    return f"{path}:{line_number}"


def _status(data: Mapping[str, object], location: str) -> RunStatus:
    value = data.get("status", "completed")
    if value not in ("completed", "failed", "unsupported"):
        raise ValueError(
            f"{location}: status must be completed, failed, or unsupported"
        )
    return cast(RunStatus, value)


def _required_string(data: Mapping[str, object], field: str, location: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}: {field} must be a non-empty string")
    return value


def _required_nonnegative_float(
    data: Mapping[str, object], field: str, location: str
) -> float:
    value = _number(data, field, location)
    if value < 0.0:
        raise ValueError(f"{location}: {field} must be nonnegative")
    return value


def _required_positive_float(
    data: Mapping[str, object], field: str, location: str
) -> float:
    value = _number(data, field, location)
    if value <= 0.0:
        raise ValueError(f"{location}: {field} must be positive")
    return value


def _required_ratio(data: Mapping[str, object], field: str, location: str) -> float:
    value = _number(data, field, location)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{location}: {field} must be in [0, 1]")
    return value


def _optional_ratio(
    data: Mapping[str, object], field: str, location: str
) -> float | None:
    if data.get(field) is None:
        return None
    return _required_ratio(data, field, location)


def _optional_nonnegative_float(
    data: Mapping[str, object], field: str, location: str
) -> float | None:
    if data.get(field) is None:
        return None
    value = _number(data, field, location)
    if value < 0.0:
        raise ValueError(f"{location}: {field} must be nonnegative")
    return value


def _number(data: Mapping[str, object], field: str, location: str) -> float:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{location}: {field} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{location}: {field} must be a finite number")
    return converted


def _required_positive_int(
    data: Mapping[str, object], field: str, location: str
) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location}: {field} must be a positive integer")
    return value


def _validate_constant_run_metadata(rows: Sequence[StepRow]) -> None:
    reference = rows[0].metadata
    for row in rows[1:]:
        if row.metadata != reference:
            raise ValueError("Run metadata changed across steps")


def _identity_values(metadata: RunMetadata) -> tuple[object, ...]:
    return (
        metadata.model,
        metadata.recipe,
        metadata.vllm_version,
        metadata.container,
        metadata.cluster,
        metadata.temperature,
        metadata.top_p,
        metadata.max_osl,
        metadata.requested_cuda_graph_mode,
        metadata.resolved_cuda_graph_mode,
        metadata.runner,
    )


def _same_identity(candidate: RunSummary, baseline: RunSummary) -> bool:
    return _identity_values(candidate.metadata) == _identity_values(baseline.metadata)


def _baseline_mismatch_fields(
    candidate: RunSummary, baselines: Sequence[RunSummary]
) -> list[str]:
    if not baselines:
        return []
    candidate_values = _identity_values(candidate.metadata)
    baseline_values = tuple(
        _identity_values(baseline.metadata) for baseline in baselines
    )
    return [
        field
        for index, field in enumerate(_IDENTITY_FIELD_NAMES)
        if all(candidate_values[index] != values[index] for values in baseline_values)
    ]


def _speedup(numerator: float, denominator: float, metric: str) -> float:
    if not math.isfinite(numerator) or numerator <= 0.0:
        raise ValueError(
            f"Cannot compute speedup for {metric}: numerator must be positive and finite"
        )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(
            f"Cannot compute speedup for {metric}: denominator must be positive and finite"
        )
    return numerator / denominator


def _require_final_window(summary: RunSummary) -> None:
    if (
        summary.is_partial
        or summary.step_start != 2
        or summary.step_end != 20
        or summary.step_count != 19
    ):
        raise IncompleteWindowError(
            "Baseline speedups require the complete steps 2-20 window"
        )


def _mean(values: Iterable[float]) -> float:
    collected = tuple(values)
    if not collected:
        raise ValueError("Cannot average an empty sequence")
    return math.fsum(collected) / len(collected)


def _optional_mean(values: Iterable[float | None]) -> float | None:
    collected = tuple(value for value in values if value is not None)
    return _mean(collected) if collected else None


def _report_values(row: ReportRow) -> tuple[object, ...]:
    metadata = row.metadata
    summary = row.summary
    return (
        metadata.model,
        metadata.recipe,
        metadata.variant,
        metadata.vllm_version,
        metadata.container,
        metadata.cluster,
        metadata.temperature,
        metadata.top_p,
        metadata.max_osl,
        metadata.requested_cuda_graph_mode,
        metadata.resolved_cuda_graph_mode,
        metadata.cuda_graph_coverage,
        metadata.job_id,
        metadata.log_path,
        metadata.wandb_url,
        metadata.runner,
        row.status,
        row.reason,
        summary.step_start if summary else None,
        summary.step_end if summary else None,
        summary.step_count if summary else None,
        summary.is_partial if summary else None,
        summary.e2e_time_s if summary else None,
        summary.generation_time_s if summary else None,
        summary.policy_time_s if summary else None,
        summary.logprob_time_s if summary else None,
        summary.e2e_throughput_tps_per_gpu if summary else None,
        summary.generation_throughput_tps_per_gpu if summary else None,
        summary.generation_ratio if summary else None,
        summary.acceptance_rate if summary else None,
        summary.mean_accepted_length if summary else None,
        summary.e2e_time_speedup if summary else None,
        summary.generation_time_speedup if summary else None,
        summary.e2e_throughput_speedup if summary else None,
        summary.generation_throughput_speedup if summary else None,
    )


def _sort_key(row: ReportRow) -> tuple[str, ...]:
    return tuple(_format_value(value) for value in _report_values(row))


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _render_markdown(rows: Sequence[ReportRow]) -> str:
    header = f"| {' | '.join(_CSV_FIELDS)} |\n"
    separator = f"| {' | '.join('---' for _ in _CSV_FIELDS)} |\n"
    rendered_rows = [header, separator]
    for row in rows:
        values = (
            _escape_markdown(_format_value(value)) for value in _report_values(row)
        )
        rendered_rows.append(f"| {' | '.join(values)} |\n")
    return "".join(rendered_rows)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


if __name__ == "__main__":
    raise SystemExit(main())
