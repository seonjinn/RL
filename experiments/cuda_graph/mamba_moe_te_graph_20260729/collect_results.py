#!/usr/bin/env python3
"""Normalize local W&B JSON exports into the experiment result schema."""

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CSV_FIELDS = (
    "scope",
    "job_id",
    "status",
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
    "reward_mean": "train/reward",
    "generation_kl_error": "train/token_mult_prob_error",
    "policy_loss": "train/loss",
}

QUALITY_METRICS = (
    "train/reward",
    "train/accuracy",
    "train/token_mult_prob_error",
    "train/loss",
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
    "grad_norm": "train/grad_norm",
    "peak_allocated_gib": "memory/peak_allocated_gib",
    "peak_reserved_gib": "memory/peak_reserved_gib",
}


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
    for field in ("scope", "job_id", "status"):
        row[field] = record.get(field, "")
    row["step"] = record.get("step", metrics.get("_step", ""))
    for field in TELEMETRY_FIELDS:
        row[field] = _value(record, metrics, field)
    for output_field, metric_name in WANDB_METRIC_MAP.items():
        row[output_field] = metrics.get(metric_name, "")
    if row["reward_mean"] == "":
        row["reward_mean"] = metrics.get("train/accuracy", "")
    for output_field, metric_name in OPTIONAL_METRIC_MAP.items():
        row[output_field] = metrics.get(metric_name, record.get(output_field, ""))
    return row


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
