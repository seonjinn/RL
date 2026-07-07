#!/usr/bin/env python3
"""Summarize baseline-relative vLLM synchronous-rollout performance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MATCHED_CONFIG_FIELDS = (
    "runtime_image_sha256",
    "model_config_hash",
    "model_checkpoint_hash",
    "model_view_marker_hash",
    "drafter_config_hash",
    "drafter_checkpoint_hash",
    "drafter_view_marker_hash",
    "context_profile",
    "rope_config_hash",
    "prompt_set_hash",
    "request_plan_hash",
    "model",
    "draft_model",
    "node_count",
    "topology",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "dtype",
    "kv_cache_dtype",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "attention_backend",
    "distributed_executor_backend",
    "cudagraph_mode",
    "compilation_config",
    "num_prompts",
    "samples_per_prompt",
    "requests_per_rollout_batch",
    "rollout_batches",
    "max_prompt_tokens",
    "max_new_tokens",
    "temperature",
    "top_p",
    "seed",
    "prompt_jsonl",
    "prompt_offset",
    "prompt_batch_hashes",
)

REQUIRED_CONFIG_FIELDS = (
    "runtime_image_sha256",
    "model_config_hash",
    "model_checkpoint_hash",
    "model_view_marker_hash",
    "drafter_config_hash",
    "drafter_checkpoint_hash",
    "drafter_view_marker_hash",
    "context_profile",
    "rope_config_hash",
    "prompt_set_hash",
    "request_plan_hash",
    "model",
    "node_count",
    "topology",
    "distributed_executor_backend",
    "compilation_config",
)


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def reduction_pct(value: float, baseline: float) -> float | None:
    return round((1.0 - value / baseline) * 100.0, 6) if baseline else None


def validate_required_config(variant: str, config: dict[str, Any]) -> None:
    for field in REQUIRED_CONFIG_FIELDS:
        value = config.get(field)
        invalid_string = isinstance(value, str) and value.lower() in {
            "",
            "unknown",
            "none",
        }
        if value is None or invalid_string or value == {} or value == []:
            raise ValueError(
                f"{field} is required and must not be missing, unknown, or empty "
                f"for {variant}"
            )
    if not isinstance(config["node_count"], int) or config["node_count"] <= 0:
        raise ValueError(f"node_count must be a positive integer for {variant}")


def load_results(matrix_root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(matrix_root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        config = payload.get("config", {})
        summary = payload.get("summary", {})
        variant = str(config.get("mode", path.parent.name))
        if not summary or not summary.get("total_rollout_time_s"):
            continue
        payload["_path"] = str(path)
        results[variant] = payload
    return results


def validate_matched_configs(results: dict[str, dict[str, Any]]) -> None:
    if not results:
        return
    reference_variant = "baseline" if "baseline" in results else sorted(results)[0]
    reference = results[reference_variant]["config"]
    validate_required_config(reference_variant, reference)
    for variant, payload in results.items():
        config = payload["config"]
        validate_required_config(variant, config)
        mismatches = {
            field: (reference.get(field), config.get(field))
            for field in MATCHED_CONFIG_FIELDS
            if reference.get(field) != config.get(field)
        }
        if mismatches:
            raise ValueError(
                f"configuration mismatch for {variant} vs {reference_variant}: "
                f"{mismatches}"
            )


def output_hashes(payload: dict[str, Any]) -> list[str]:
    return [
        str(output_hash)
        for batch in payload.get("rollout_batches", [])
        for output_hash in batch.get("output_token_hashes", [])
    ]


OutputWork = tuple[list[int], list[int], list[bool]]


def output_work_counts(payload: dict[str, Any]) -> OutputWork | None:
    planned: list[int] = []
    actual: list[int] = []
    forced: list[bool] = []
    saw_counts = False
    for batch in payload.get("rollout_batches", []):
        batch_planned = batch.get("planned_output_tokens")
        batch_actual = batch.get("actual_output_tokens")
        batch_forced = batch.get("forced_output_mask")
        if not isinstance(batch_planned, list) or not isinstance(batch_actual, list):
            raise ValueError("exact output work counts must be arrays")
        if not isinstance(batch_forced, list):
            raise ValueError("exact output forced mask must be an array")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in [*batch_planned, *batch_actual]
        ):
            raise ValueError("exact output work count values must be nonnegative integers")
        if not all(isinstance(value, bool) for value in batch_forced):
            raise ValueError("exact output forced mask values must be booleans")
        if len(batch_planned) != len(batch_actual):
            raise ValueError("exact output work count arrays differ in length")
        if len(batch_planned) != len(batch_forced):
            raise ValueError("exact output forced mask differs in length")
        planned.extend(int(value) for value in batch_planned)
        actual.extend(int(value) for value in batch_actual)
        forced.extend(batch_forced)
        saw_counts = True
    return (planned, actual, forced) if saw_counts else None


def forced_planned_counts(work: OutputWork) -> list[int]:
    planned, _actual, forced = work
    return [
        planned_tokens
        for planned_tokens, is_forced in zip(planned, forced, strict=True)
        if is_forced
    ]


def validate_forced_actual_counts(variant: str, work: OutputWork) -> None:
    planned, actual, forced = work
    for index, (planned_tokens, actual_tokens, is_forced) in enumerate(
        zip(planned, actual, forced, strict=True)
    ):
        if is_forced and actual_tokens != planned_tokens:
            raise ValueError(
                f"forced actual output length does not match planned work for "
                f"{variant} at request {index}: planned={planned_tokens} "
                f"actual={actual_tokens}"
            )


def validate_exact_output_work(results: dict[str, dict[str, Any]]) -> None:
    baseline = results.get("baseline")
    if baseline is None:
        return
    baseline_work = output_work_counts(baseline)
    if baseline_work is None:
        raise ValueError("missing exact output work counts for baseline")
    validate_forced_actual_counts("baseline", baseline_work)
    for variant, payload in results.items():
        work = output_work_counts(payload)
        if work is None:
            raise ValueError(f"missing exact output work counts for {variant}")
        validate_forced_actual_counts(variant, work)
        if work != baseline_work:
            raise ValueError(
                f"exact output work arrays mismatch for {variant} vs baseline"
            )


def build_summary(matrix_root: Path) -> list[dict[str, Any]]:
    results = load_results(matrix_root)
    if "baseline" not in results:
        raise ValueError(
            "incomplete synchronous-rollout matrix; missing complete baseline"
        )
    variant_families = (
        ("static", "dynamic"),
        ("mtp_static", "mtp_dynamic"),
    )
    selected_family = next(
        (
            family
            for family in variant_families
            if all(variant in results for variant in family)
        ),
        None,
    )
    if selected_family is None:
        raise ValueError(
            "incomplete synchronous-rollout matrix; expected complete "
            "static/dynamic or mtp_static/mtp_dynamic variants"
        )
    static_variant, dynamic_variant = selected_family
    validate_matched_configs(results)
    validate_exact_output_work(results)
    baseline = results.get("baseline")
    static = results.get(static_variant)
    baseline_summary = baseline.get("summary", {}) if baseline else {}
    static_summary = static.get("summary", {}) if static else {}
    baseline_hashes = output_hashes(baseline) if baseline else []
    rows: list[dict[str, Any]] = []
    order = {"baseline": 0, static_variant: 1, dynamic_variant: 2}
    for variant, payload in sorted(
        results.items(), key=lambda item: (order.get(item[0], 99), item[0])
    ):
        config = payload["config"]
        summary = payload["summary"]
        metrics = summary.get("spec_decode_metrics", {})
        throughput = float(summary["output_tok_s_per_gpu"])
        rollout_time = float(summary["total_rollout_time_s"])
        baseline_throughput = float(
            baseline_summary.get("output_tok_s_per_gpu", 0.0)
        )
        baseline_time = float(baseline_summary.get("total_rollout_time_s", 0.0))
        static_throughput = float(static_summary.get("output_tok_s_per_gpu", 0.0))
        static_time = float(static_summary.get("total_rollout_time_s", 0.0))
        baseline_tokens = float(baseline_summary.get("total_output_tokens", 0.0))
        output_tokens = float(summary.get("total_output_tokens", 0.0))
        hashes = output_hashes(payload)
        work = output_work_counts(payload)
        baseline_work = output_work_counts(baseline) if baseline else None
        exact_work_match = work == baseline_work if work and baseline_work else None
        rows.append(
            {
                "variant": variant,
                "temperature": config.get("temperature"),
                "top_p": config.get("top_p"),
                "total_rollout_time_s": rollout_time,
                "output_tok_s_per_gpu": throughput,
                "total_output_tokens": output_tokens,
                "output_token_ratio_vs_baseline": ratio(
                    output_tokens, baseline_tokens
                ),
                "direct_time_comparison_valid": (
                    baseline_tokens > 0
                    and abs(output_tokens / baseline_tokens - 1.0) <= 0.01
                ),
                "throughput_speedup_vs_baseline": ratio(
                    throughput, baseline_throughput
                ),
                "rollout_time_reduction_vs_baseline_pct": reduction_pct(
                    rollout_time, baseline_time
                ),
                "throughput_speedup_vs_static": ratio(throughput, static_throughput),
                "rollout_time_reduction_vs_static_pct": reduction_pct(
                    rollout_time, static_time
                ),
                "acceptance_rate": metrics.get("acceptance_rate"),
                "mean_acceptance_length": metrics.get("mean_acceptance_length"),
                "exact_output_work_match_vs_baseline": (
                    exact_work_match
                ),
                "exact_output_hash_match_vs_baseline": (
                    hashes == baseline_hashes
                    if hashes and baseline_hashes
                    else None
                ),
                "result_json": payload["_path"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    rows = build_summary(args.matrix_root)
    output_csv = args.output_csv or args.matrix_root / "summary.csv"
    output_json = args.output_json or args.matrix_root / "summary.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        output_csv.write_text("", encoding="utf-8")
    output_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"wrote {output_csv}")
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
