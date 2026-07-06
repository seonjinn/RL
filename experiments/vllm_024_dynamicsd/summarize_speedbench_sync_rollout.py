#!/usr/bin/env python3
"""Summarize SPEED-Bench official and Sync-RL overlay cohorts separately."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


OFFICIAL_INSTRUMENTATION_FIELDS = (
    "official_instrumentation_schema_version",
    "official_instrumentation_modelopt_commit",
    "official_instrumentation_source_sha256",
    "official_instrumentation_patch_sha256",
    "official_instrumentation_patched_source_sha256",
)

MATCHED_BASELINE_FIELDS = (
    "cohort",
    "runtime_image_sha256",
    "model_config_hash",
    "prepared_manifest_hash",
    "request_plan_hash",
    "prompt_set_hash",
    "model",
    "draft_model",
    "dataset_config",
    "active_concurrency",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "dtype",
    "kv_cache_dtype",
    "cudagraph_mode",
    "compilation_config",
    "temperature",
    "top_p",
    "sampling_protocol",
    "sampling",
    "max_model_len",
    "max_new_tokens",
    "samples_per_prompt",
    "rollout_batches",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "distributed_executor_backend",
    "distributed_timeout_seconds",
    "enable_expert_parallel",
    "model_loader_extra_config",
    "mamba_ssm_cache_dtype",
    "mamba_backend",
    "enable_mamba_cache_stochastic_rounding",
    "mamba_cache_philox_rounds",
    "moe_backend",
    *OFFICIAL_INSTRUMENTATION_FIELDS,
)

REQUIRED_PROVENANCE_FIELDS = (
    "cohort",
    "variant",
    "runtime_image_sha256",
    "model_config_hash",
    "prepared_manifest_hash",
    "request_plan_hash",
    "prompt_set_hash",
    "model",
    "draft_model",
    "method",
    "dataset_config",
    "active_concurrency",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "dtype",
    "kv_cache_dtype",
    "cudagraph_mode",
    "compilation_config",
    "temperature",
    "top_p",
    "sampling_protocol",
    "sampling",
    "max_model_len",
    "max_new_tokens",
    "samples_per_prompt",
    "rollout_batches",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "distributed_executor_backend",
    "distributed_timeout_seconds",
    "enable_expert_parallel",
    "model_loader_extra_config",
    "mamba_ssm_cache_dtype",
    "mamba_backend",
    "enable_mamba_cache_stochastic_rounding",
    "mamba_cache_philox_rounds",
    "moe_backend",
)


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def reduction_pct(value: float, baseline: float) -> float | None:
    return round((1.0 - value / baseline) * 100.0, 6) if baseline else None


def _canonical_match_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def validate_required_provenance(row: dict[str, Any]) -> None:
    for field in REQUIRED_PROVENANCE_FIELDS:
        value = row.get(field)
        if value is None or value == "unknown" or value == "":
            raise ValueError(f"{field} is required and must not be unknown")
    if row.get("cohort") == "official":
        for field in OFFICIAL_INSTRUMENTATION_FIELDS:
            value = row.get(field)
            if value is None or value == "unknown" or value == "":
                raise ValueError(f"{field} is required and must not be unknown")


def compare_rows(
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> dict[str, Any]:
    validate_required_provenance(baseline_row)
    validate_required_provenance(candidate_row)
    if baseline_row.get("cohort") != candidate_row.get("cohort"):
        raise ValueError(
            "cohort mismatch: "
            f"{baseline_row.get('cohort')} vs {candidate_row.get('cohort')}"
        )
    mismatches = {
        field: (baseline_row.get(field), candidate_row.get(field))
        for field in MATCHED_BASELINE_FIELDS
        if baseline_row.get(field) != candidate_row.get(field)
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
    instrumentation = config.get("official_instrumentation")
    if not isinstance(instrumentation, dict):
        instrumentation = {}
    return {
        "cohort": config.get("cohort"),
        "variant": config.get("mode"),
        "runtime_image_sha256": config.get("runtime_image_sha256"),
        "model_config_hash": config.get("model_config_hash"),
        "prepared_manifest_hash": config.get("prepared_manifest_hash"),
        "request_plan_hash": config.get("request_plan_hash"),
        "prompt_set_hash": config.get("prompt_set_hash"),
        "model": config.get("model"),
        "draft_model": config.get("draft_model"),
        "method": config.get("method"),
        "dataset_config": config.get("dataset_config"),
        "active_concurrency": config.get("active_concurrency"),
        "tensor_parallel_size": config.get("tensor_parallel_size"),
        "pipeline_parallel_size": config.get("pipeline_parallel_size"),
        "dtype": config.get("dtype"),
        "kv_cache_dtype": config.get("kv_cache_dtype"),
        "cudagraph_mode": config.get("cudagraph_mode"),
        "compilation_config": config.get("compilation_config"),
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "sampling_protocol": config.get("sampling_protocol"),
        "sampling": config.get("sampling"),
        "max_model_len": config.get("max_model_len"),
        "max_new_tokens": config.get("max_new_tokens"),
        "samples_per_prompt": config.get("samples_per_prompt"),
        "rollout_batches": config.get("rollout_batches"),
        "max_num_batched_tokens": config.get("max_num_batched_tokens"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization"),
        "distributed_executor_backend": config.get("distributed_executor_backend"),
        "distributed_timeout_seconds": config.get("distributed_timeout_seconds"),
        "enable_expert_parallel": config.get("enable_expert_parallel"),
        "model_loader_extra_config": config.get("model_loader_extra_config"),
        "mamba_ssm_cache_dtype": config.get("mamba_ssm_cache_dtype"),
        "mamba_backend": config.get("mamba_backend"),
        "enable_mamba_cache_stochastic_rounding": config.get(
            "enable_mamba_cache_stochastic_rounding"
        ),
        "mamba_cache_philox_rounds": config.get("mamba_cache_philox_rounds"),
        "moe_backend": config.get("moe_backend"),
        "official_instrumentation_schema_version": config.get(
            "official_instrumentation_schema_version",
            instrumentation.get("schema_version"),
        ),
        "official_instrumentation_modelopt_commit": config.get(
            "official_instrumentation_modelopt_commit",
            instrumentation.get("modelopt_commit"),
        ),
        "official_instrumentation_source_sha256": config.get(
            "official_instrumentation_source_sha256",
            instrumentation.get("source_sha256"),
        ),
        "official_instrumentation_patch_sha256": config.get(
            "official_instrumentation_patch_sha256",
            instrumentation.get("patch_sha256"),
        ),
        "official_instrumentation_patched_source_sha256": config.get(
            "official_instrumentation_patched_source_sha256",
            instrumentation.get("patched_source_sha256"),
        ),
        "total_rollout_time_s": summary.get("total_rollout_time_s"),
        "output_tok_s_per_gpu": summary.get("output_tok_s_per_gpu"),
        "total_output_tokens": summary.get("total_output_tokens"),
        "ttft_p50_s": summary.get("ttft_p50_s"),
        "ttft_p90_s": summary.get("ttft_p90_s"),
        "ttft_p99_s": summary.get("ttft_p99_s"),
        "completion_p50_s": summary.get("completion_p50_s"),
        "completion_p90_s": summary.get("completion_p90_s"),
        "completion_p99_s": summary.get("completion_p99_s"),
        "barrier_tail_gap_s": summary.get("barrier_tail_gap_s"),
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
    return tuple(
        _canonical_match_value(row.get(field))
        for field in MATCHED_BASELINE_FIELDS
    )


def build_summary(matrix_root: Path) -> list[dict[str, Any]]:
    rows = load_rows(matrix_root)
    for row in rows:
        validate_required_provenance(row)
    baselines = {
        _group_key(row): row
        for row in rows
        if row.get("variant") == "baseline"
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        baseline = baselines.get(_group_key(row))
        if baseline is None:
            raise ValueError(
                "missing complete baseline for "
                f"cohort={row.get('cohort')} variant={row.get('variant')}"
            )
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
