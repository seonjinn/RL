#!/usr/bin/env python3
"""Normalize local W&B JSON exports into the experiment result schema."""

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

CSV_FIELDS = (
    "scope",
    "job_id",
    "status",
    "failure",
    "exit_code",
    "elapsed",
    "completed_steps",
    "step",
    "geometry_key",
    "capture_count",
    "replay_count",
    "cache_hit_count",
    "eviction_count",
    "fallback_count",
    "e2e_step_time",
    "e2e_tokens_per_sec_per_gpu",
    "generation_time",
    "generation_tokens_per_sec_per_gpu",
    "policy_training_time",
    "policy_training_tokens_per_sec_per_gpu",
    "logprob_time",
    "logprob_tokens_per_sec_per_gpu",
    "reward_mean",
    "generation_kl_error",
    "token_mult_prob_error",
    "policy_kl_error",
    "js_divergence_error",
    "sampling_importance_ratio",
    "num_masked_seqs_by_logprob_error",
    "policy_loss",
    "grad_norm",
    "peak_allocated_gib",
    "peak_reserved_gib",
)

WANDB_METRIC_MAP = {
    "e2e_tokens_per_sec_per_gpu": "performance/tokens_per_sec_per_gpu",
    "generation_tokens_per_sec_per_gpu": (
        "performance/generation_tokens_per_sec_per_gpu"
    ),
    "policy_training_tokens_per_sec_per_gpu": (
        "performance/policy_training_tokens_per_sec_per_gpu"
    ),
    "logprob_tokens_per_sec_per_gpu": (
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu"
    ),
    "e2e_step_time": "timing/train/total_step_time",
    "generation_time": "timing/train/generation",
    "policy_training_time": "timing/train/policy_training",
    "logprob_time": "timing/train/policy_and_reference_logprobs",
}

CORRECTNESS_METRIC_MAP = {
    "reward_mean": "train/reward",
    "generation_kl_error": "train/gen_kl_error",
    "token_mult_prob_error": "train/token_mult_prob_error",
    "policy_kl_error": "train/policy_kl_error",
    "js_divergence_error": "train/js_divergence_error",
    "sampling_importance_ratio": "train/sampling_importance_ratio",
    "num_masked_seqs_by_logprob_error": "train/num_masked_seqs_by_logprob_error",
    "policy_loss": "train/loss",
    "grad_norm": "train/grad_norm",
}

QUALITY_METRICS = (
    "train/reward",
    "train/accuracy",
    "train/gen_kl_error",
    "train/token_mult_prob_error",
    "train/policy_kl_error",
    "train/js_divergence_error",
    "train/sampling_importance_ratio",
    "train/num_masked_seqs_by_logprob_error",
    "train/loss",
    "train/grad_norm",
)

TELEMETRY_FIELDS = (
    "geometry_key",
    "capture_count",
    "replay_count",
    "cache_hit_count",
    "eviction_count",
    "fallback_count",
)

OPTIONAL_METRIC_MAP = {
    "peak_allocated_gib": "memory/peak_allocated_gib",
    "peak_reserved_gib": "memory/peak_reserved_gib",
}

PERFORMANCE_FIELDS = (
    "e2e_step_time",
    "e2e_tokens_per_sec_per_gpu",
    "generation_time",
    "generation_tokens_per_sec_per_gpu",
    "policy_training_time",
    "policy_training_tokens_per_sec_per_gpu",
    "logprob_time",
    "logprob_tokens_per_sec_per_gpu",
)

THROUGHPUT_FIELDS = (
    "e2e_tokens_per_sec_per_gpu",
    "generation_tokens_per_sec_per_gpu",
    "policy_training_tokens_per_sec_per_gpu",
    "logprob_tokens_per_sec_per_gpu",
)

CORRECTNESS_FIELDS = (
    "reward_mean",
    "generation_kl_error",
    "token_mult_prob_error",
    "policy_kl_error",
    "js_divergence_error",
    "sampling_importance_ratio",
    "num_masked_seqs_by_logprob_error",
    "policy_loss",
    "grad_norm",
)

BASELINE_SCOPES = frozenset({"baseline-no-cg", "no_cg", "[no_cg]"})


def _value(
    record: Mapping[str, Any],
    metrics: Mapping[str, Any],
    field: str,
) -> Any:
    if field in record:
        return record[field]
    for metric_name in (
        field,
        f"cuda_graph/{field}",
        f"policy/cuda_graph/{field}",
    ):
        if metric_name in metrics:
            return metrics[metric_name]
    return ""


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one local export row without contacting W&B."""
    nested_metrics = record.get("metrics", {})
    if not isinstance(nested_metrics, Mapping):
        raise TypeError("record.metrics must be a mapping")
    metrics = dict(record)
    metrics.update(nested_metrics)

    row = {field: "" for field in CSV_FIELDS}
    for field in (
        "scope",
        "job_id",
        "status",
        "failure",
        "exit_code",
        "elapsed",
        "completed_steps",
    ):
        row[field] = record.get(field, "")
    row["step"] = record.get("step", metrics.get("_step", ""))
    for field in TELEMETRY_FIELDS:
        row[field] = _value(record, metrics, field)
    for metric_map in (WANDB_METRIC_MAP, CORRECTNESS_METRIC_MAP):
        for output_field, metric_name in metric_map.items():
            row[output_field] = metrics.get(metric_name, "")
    if row["reward_mean"] == "":
        row["reward_mean"] = metrics.get("train/accuracy", "")
    for output_field, metric_name in OPTIONAL_METRIC_MAP.items():
        row[output_field] = metrics.get(metric_name, record.get(output_field, ""))
    return row


def steady_state_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    first_step: int = 6,
    last_step: int = 20,
) -> list[Mapping[str, str]]:
    """Return rows whose optimizer step is within the inclusive measurement window."""
    if first_step > last_step:
        raise ValueError("first_step must not exceed last_step")

    selected_rows = []
    for row in rows:
        step_text = row.get("step", "")
        if step_text == "":
            continue
        try:
            step = int(step_text)
        except ValueError as error:
            raise ValueError(
                f"invalid step for {row.get('scope', '')}/{row.get('job_id', '')}: "
                f"{step_text!r}"
            ) from error
        if first_step <= step <= last_step:
            selected_rows.append(row)
    return selected_rows


def _format_number(value: float) -> str:
    """Format finite aggregate values deterministically for CSV and HTML."""
    if not math.isfinite(value):
        raise ValueError(f"aggregate value must be finite, got {value!r}")
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _numeric_values(rows: Sequence[Mapping[str, str]], field: str) -> list[float]:
    """Read present numeric values for one normalized metric field."""
    values = []
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        try:
            values.append(float(value))
        except ValueError as error:
            raise ValueError(
                f"invalid {field} for {row.get('scope', '')}/{row.get('job_id', '')}: "
                f"{value!r}"
            ) from error
    return values


def _median_and_p95(rows: Sequence[Mapping[str, str]], field: str) -> tuple[str, str]:
    """Calculate the median and nearest-rank 95th percentile for one field."""
    values = sorted(_numeric_values(rows, field))
    if not values:
        return "", ""
    nearest_rank_index = math.ceil(0.95 * len(values)) - 1
    return _format_number(statistics.median(values)), _format_number(
        values[nearest_rank_index]
    )


def _median_value(rows: Sequence[Mapping[str, str]], field: str) -> float | None:
    """Return the unrounded median used by comparisons, if the field is present."""
    values = _numeric_values(rows, field)
    return statistics.median(values) if values else None


def _telemetry_maximum(rows: Sequence[Mapping[str, str]], field: str) -> int:
    """Read the final cumulative telemetry count from per-step snapshots."""
    maximum = 0
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        try:
            maximum = max(maximum, int(value))
        except ValueError as error:
            raise ValueError(
                f"invalid {field} for {row.get('scope', '')}/{row.get('job_id', '')}: "
                f"{value!r}"
            ) from error
    return maximum


def _field_issue(field: str, value: str | None, *, integer: bool) -> str | None:
    """Return a deterministic validation reason for one required field."""
    if value in {None, ""}:
        return f"{field}_missing"
    try:
        numeric_value = int(value) if integer else float(value)
    except ValueError:
        return f"{field}_invalid"
    if integer:
        if numeric_value < 0:
            return f"{field}_negative"
    elif not math.isfinite(numeric_value):
        return f"{field}_nonfinite"
    return None


def _group_invalid_reasons(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Validate every required steady-state field without averaging bad samples."""
    reasons = []
    for field in (*PERFORMANCE_FIELDS, *CORRECTNESS_FIELDS):
        for row in rows:
            reason = _field_issue(field, row.get(field), integer=False)
            if reason is not None:
                reasons.append(reason)
                break
    for field in ("eviction_count", "fallback_count"):
        for row in rows:
            reason = _field_issue(field, row.get(field), integer=True)
            if reason is not None:
                reasons.append(reason)
                break
        else:
            maximum = _telemetry_maximum(rows, field)
            if maximum:
                reasons.append(f"{field}={maximum}")
    return reasons


def _baseline_key(
    grouped_rows: Mapping[tuple[str, str], Sequence[Mapping[str, str]]],
) -> tuple[str, str] | None:
    """Select the deterministic no-CG baseline aggregate, if present."""
    baseline_keys = sorted(key for key in grouped_rows if key[0] in BASELINE_SCOPES)
    return baseline_keys[0] if baseline_keys else None


def _blank_comparisons(aggregate: dict[str, str]) -> None:
    """Clear comparison cells when no valid baseline comparison is possible."""
    for field in THROUGHPUT_FIELDS:
        aggregate[f"{field}_ratio_to_baseline"] = ""
    for field in CORRECTNESS_FIELDS:
        aggregate[f"{field}_delta"] = ""


def aggregate_performance(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Aggregate one steady-state row per scope and Slurm job.

    Performance ratios and correctness deltas compare each group with the
    deterministic no-CG baseline group. The caller selects the measurement
    window with :func:`steady_state_rows` before invoking this function.
    """
    grouped_rows: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        key = (row.get("scope", ""), row.get("job_id", ""))
        grouped_rows.setdefault(key, []).append(row)

    aggregates = []
    aggregate_by_key = {}
    for (scope, job_id), group_rows in sorted(grouped_rows.items()):
        invalid_reasons = _group_invalid_reasons(group_rows)
        aggregate = {
            "scope": scope,
            "job_id": job_id,
            "sample_count": str(len(group_rows)),
            "valid": "false" if invalid_reasons else "true",
            "invalid_reason": "; ".join(invalid_reasons),
        }
        _blank_comparisons(aggregate)
        for field in PERFORMANCE_FIELDS:
            median, p95 = (
                _median_and_p95(group_rows, field) if not invalid_reasons else ("", "")
            )
            aggregate[f"{field}_median"] = median
            aggregate[f"{field}_p95"] = p95

        aggregate_by_key[(scope, job_id)] = aggregate
        aggregates.append(aggregate)

    baseline_key = _baseline_key(grouped_rows)
    if baseline_key is None:
        baseline_rows = []
        baseline_medians: dict[str, float | None] = {}
    else:
        baseline_rows = grouped_rows[baseline_key]
        baseline_aggregate = aggregate_by_key[baseline_key]
        baseline_medians = (
            {
                field: _median_value(baseline_rows, field)
                for field in (*THROUGHPUT_FIELDS, *CORRECTNESS_FIELDS)
            }
            if baseline_aggregate["valid"] == "true"
            else {}
        )

    for key, group_rows in sorted(grouped_rows.items()):
        aggregate = aggregate_by_key[key]
        if key[0] in BASELINE_SCOPES:
            continue
        if baseline_key is None:
            aggregate["valid"] = "false"
            aggregate["invalid_reason"] = "; ".join(
                filter(None, (aggregate["invalid_reason"], "baseline_missing"))
            )
            continue
        baseline_aggregate = aggregate_by_key[baseline_key]
        if aggregate["valid"] == "false":
            continue
        if baseline_aggregate["valid"] == "false":
            aggregate["valid"] = "false"
            aggregate["invalid_reason"] = (
                f"baseline_invalid={baseline_key[0]}/{baseline_key[1]}"
            )
            continue

        for field in THROUGHPUT_FIELDS:
            median = _median_value(group_rows, field)
            baseline_median = baseline_medians[field]
            if median is None or baseline_median in {None, 0.0}:
                continue
            aggregate[f"{field}_ratio_to_baseline"] = _format_number(
                median / baseline_median
            )
        for field in CORRECTNESS_FIELDS:
            median = _median_value(group_rows, field)
            baseline_median = baseline_medians[field]
            if median is None or baseline_median is None:
                continue
            aggregate[f"{field}_delta"] = _format_number(median - baseline_median)
    return aggregates


def load_records(path: Path) -> list[Mapping[str, Any]]:
    """Load a JSON array/object or one JSON object per line."""
    if path.suffix == ".jsonl":
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
            records = payload["rows"]
        else:
            records = [payload]
    if not all(isinstance(record, Mapping) for record in records):
        raise TypeError("every input record must be a JSON object")
    return records


def write_csv(records: list[Mapping[str, Any]], output: Path) -> None:
    """Write normalized rows with a stable header."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(normalize_record(record) for record in records)


def parse_args() -> argparse.Namespace:
    """Parse local-only collector paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Collect one local result export."""
    args = parse_args()
    write_csv(load_records(args.input), args.output)


if __name__ == "__main__":
    main()
