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
    "prompt_set_hash",
    "request_plan_hash",
    "model",
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
    "cudagraph_mode",
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


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def reduction_pct(value: float, baseline: float) -> float | None:
    return round((1.0 - value / baseline) * 100.0, 6) if baseline else None


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
    for variant, payload in results.items():
        config = payload["config"]
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


def validate_exact_output_work(results: dict[str, dict[str, Any]]) -> None:
    baseline = results.get("baseline")
    if baseline is None:
        return
    baseline_hashes = output_hashes(baseline)
    if not baseline_hashes:
        return
    for variant, payload in results.items():
        hashes = output_hashes(payload)
        if hashes != baseline_hashes:
            raise ValueError(
                f"exact output work mismatch for {variant} vs baseline"
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
