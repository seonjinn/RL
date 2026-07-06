#!/usr/bin/env python3
"""Summarize SPEED-Bench official and Sync-RL overlay cohorts separately."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


MATCHED_BASELINE_FIELDS = (
    "cohort",
    "runtime_image_sha256",
    "model_config_hash",
    "prepared_manifest_hash",
    "request_plan_hash",
    "model",
    "dataset_config",
    "active_concurrency",
    "temperature",
    "top_p",
)


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def reduction_pct(value: float, baseline: float) -> float | None:
    return round((1.0 - value / baseline) * 100.0, 6) if baseline else None


def compare_rows(
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> dict[str, Any]:
    if baseline_row.get("cohort") != candidate_row.get("cohort"):
        raise ValueError(
            "cohort mismatch: "
            f"{baseline_row.get('cohort')} vs {candidate_row.get('cohort')}"
        )
    mismatches = {
        field: (baseline_row.get(field), candidate_row.get(field))
        for field in MATCHED_BASELINE_FIELDS
        if baseline_row.get(field) != candidate_row.get(field)
        and baseline_row.get(field) is not None
        and candidate_row.get(field) is not None
    }
    if mismatches:
        first = next(iter(mismatches))
        raise ValueError(f"matched runtime baseline mismatch: {first} {mismatches}")
    baseline_throughput = float(baseline_row.get("output_tok_s_per_gpu", 0.0))
    candidate_throughput = float(candidate_row.get("output_tok_s_per_gpu", 0.0))
    baseline_time = float(baseline_row.get("total_rollout_time_s", 0.0))
    candidate_time = float(candidate_row.get("total_rollout_time_s", 0.0))
    result = dict(candidate_row)
    result["throughput_speedup_vs_baseline"] = ratio(
        candidate_throughput,
        baseline_throughput,
    )
    result["rollout_time_reduction_vs_baseline_pct"] = reduction_pct(
        candidate_time,
        baseline_time,
    )
    return result


def row_from_result(path: Path) -> dict[str, Any] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    config = payload.get("config", {})
    summary = payload.get("summary", {})
    if not summary:
        return None
    metrics = summary.get("spec_decode_metrics", {})
    return {
        "cohort": config.get("cohort"),
        "variant": config.get("mode"),
        "runtime_image_sha256": config.get("runtime_image_sha256"),
        "model_config_hash": config.get("model_config_hash"),
        "prepared_manifest_hash": config.get("prepared_manifest_hash"),
        "request_plan_hash": config.get("request_plan_hash"),
        "model": config.get("model"),
        "dataset_config": config.get("dataset_config"),
        "active_concurrency": config.get("active_concurrency"),
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "total_rollout_time_s": summary.get("total_rollout_time_s"),
        "output_tok_s_per_gpu": summary.get("output_tok_s_per_gpu"),
        "total_output_tokens": summary.get("total_output_tokens"),
        "acceptance_rate": metrics.get("acceptance_rate"),
        "mean_acceptance_length": metrics.get("mean_acceptance_length"),
        "result_json": str(path),
    }


def load_rows(matrix_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(matrix_root.glob("**/result.json")):
        row = row_from_result(path)
        if row is not None:
            rows.append(row)
    return rows


def _group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in MATCHED_BASELINE_FIELDS)


def build_summary(matrix_root: Path) -> list[dict[str, Any]]:
    rows = load_rows(matrix_root)
    baselines = {
        _group_key(row): row
        for row in rows
        if row.get("variant") == "baseline"
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        baseline = baselines.get(_group_key(row))
        if baseline is None:
            output.append(row)
        else:
            output.append(compare_rows(baseline, row))
    return output


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    rows = build_summary(args.matrix_root)
    output_csv = args.output_csv or args.matrix_root / "speedbench_summary.csv"
    output_json = args.output_json or args.matrix_root / "speedbench_summary.json"
    write_csv(output_csv, rows)
    output_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"wrote {output_csv}")
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
