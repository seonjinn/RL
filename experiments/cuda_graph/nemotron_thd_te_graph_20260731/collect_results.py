#!/usr/bin/env python3
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

"""Normalize local experiment JSON/JSONL into stable JSON and CSV ledgers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = EXPERIMENT_DIR / "results" / "raw"
DEFAULT_OUTPUT_JSON = EXPERIMENT_DIR / "results" / "results.json"
DEFAULT_OUTPUT_CSV = EXPERIMENT_DIR / "results" / "results.csv"

IDENTITY_FIELDS = (
    "model",
    "dispatcher",
    "scope",
    "router_replay",
    "status",
    "failure",
    "exit_code",
    "mode",
    "cluster",
    "profile",
    "phase",
    "steps",
    "step",
    "repeat",
    "run_group",
    "job_id",
)
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
GRAPH_FIELDS = (
    "graph_telemetry_status",
    "capture_count",
    "replay_count",
    "cache_hits",
    "cache_misses",
    "cache_evictions",
    "fallback_count",
    "graph_calls",
    "eligible_calls",
    "graph_coverage",
    "logical_tokens",
    "padded_tokens",
    "capacity_tokens",
    "capacity_utilization",
    "padding_utilization",
)
CORRECTNESS_FIELDS = (
    "reward",
    "policy_loss",
    "gen_kl_error",
    "token_mult_prob_error",
    "policy_kl_error",
    "js_divergence_error",
    "sampling_importance_ratio",
    "num_masked_seqs_by_logprob_error",
    "router_topk_parity",
    "expert_count_parity",
    "parameter_delta_parity",
    "parameter_delta_max_abs_error",
    "parameter_delta_max_rel_error",
    "grad_norm",
    "nan_inf_status",
)
PROVENANCE_FIELDS = (
    "nemo_rl_commit",
    "bridge_commit",
    "mcore_commit",
    "te_commit",
    "te_version",
    "container_sha256",
)
CSV_FIELDS = (
    *IDENTITY_FIELDS,
    *PERFORMANCE_FIELDS,
    *GRAPH_FIELDS,
    *CORRECTNESS_FIELDS,
    *PROVENANCE_FIELDS,
)
REQUIRED_REPORT_FIELDS = (
    "model",
    "dispatcher",
    "scope",
    "router_replay",
    "status",
    "mode",
    "cluster",
    "profile",
    "phase",
    "steps",
    "step",
    "repeat",
    "run_group",
    "job_id",
    *PERFORMANCE_FIELDS,
    *GRAPH_FIELDS,
    *CORRECTNESS_FIELDS,
    *PROVENANCE_FIELDS,
)

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "e2e_step_time": ("timing/train/total_step_time",),
    "e2e_tokens_per_sec_per_gpu": ("performance/tokens_per_sec_per_gpu",),
    "generation_time": ("timing/train/generation",),
    "generation_tokens_per_sec_per_gpu": (
        "performance/generation_tokens_per_sec_per_gpu",
    ),
    "policy_training_time": ("timing/train/policy_training",),
    "policy_training_tokens_per_sec_per_gpu": (
        "performance/policy_training_tokens_per_sec_per_gpu",
    ),
    "logprob_time": ("timing/train/policy_and_reference_logprobs",),
    "logprob_tokens_per_sec_per_gpu": (
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
    ),
    "graph_telemetry_status": ("graph_telemetry_status",),
    "capture_count": (
        "cuda_graph/capture_count",
        "train/cuda_graph/capture_count",
        "policy/cuda_graph/capture_count",
    ),
    "replay_count": (
        "cuda_graph/replay_count",
        "train/cuda_graph/replay_count",
        "policy/cuda_graph/replay_count",
    ),
    "cache_hits": (
        "cuda_graph/cache_hits",
        "cuda_graph/cache_hit_count",
        "train/cuda_graph/cache_hits",
        "train/cuda_graph/cache_hit_count",
        "policy/cuda_graph/cache_hits",
        "policy/cuda_graph/cache_hit_count",
    ),
    "cache_misses": (
        "cuda_graph/cache_misses",
        "cuda_graph/cache_miss_count",
        "train/cuda_graph/cache_misses",
        "train/cuda_graph/cache_miss_count",
        "policy/cuda_graph/cache_misses",
        "policy/cuda_graph/cache_miss_count",
    ),
    "cache_evictions": (
        "cuda_graph/cache_evictions",
        "cuda_graph/eviction_count",
        "train/cuda_graph/cache_evictions",
        "train/cuda_graph/eviction_count",
        "policy/cuda_graph/cache_evictions",
        "policy/cuda_graph/eviction_count",
    ),
    "fallback_count": (
        "cuda_graph/fallback_count",
        "train/cuda_graph/fallback_count",
        "policy/cuda_graph/fallback_count",
    ),
    "graph_calls": (
        "cuda_graph/graph_calls",
        "train/cuda_graph/graph_calls",
        "policy/cuda_graph/graph_calls",
    ),
    "eligible_calls": (
        "cuda_graph/eligible_calls",
        "train/cuda_graph/eligible_calls",
        "policy/cuda_graph/eligible_calls",
    ),
    "graph_coverage": (
        "cuda_graph/coverage",
        "train/cuda_graph/coverage",
        "policy/cuda_graph/coverage",
    ),
    "logical_tokens": (
        "cuda_graph/logical_tokens",
        "train/cuda_graph/logical_tokens",
        "policy/cuda_graph/logical_tokens",
    ),
    "padded_tokens": (
        "cuda_graph/padded_tokens",
        "train/cuda_graph/padded_tokens",
        "policy/cuda_graph/padded_tokens",
    ),
    "capacity_tokens": (
        "cuda_graph/capacity_tokens",
        "train/cuda_graph/capacity_tokens",
        "policy/cuda_graph/capacity_tokens",
    ),
    "capacity_utilization": (
        "cuda_graph/capacity_utilization",
        "train/cuda_graph/capacity_utilization",
        "policy/cuda_graph/capacity_utilization",
    ),
    "padding_utilization": (
        "cuda_graph/padding_utilization",
        "train/cuda_graph/padding_utilization",
        "policy/cuda_graph/padding_utilization",
    ),
    "reward": ("train/reward", "train/accuracy"),
    "policy_loss": ("train/loss",),
    "gen_kl_error": ("train/gen_kl_error",),
    "token_mult_prob_error": ("train/token_mult_prob_error",),
    "policy_kl_error": ("train/policy_kl_error",),
    "js_divergence_error": ("train/js_divergence_error",),
    "sampling_importance_ratio": ("train/sampling_importance_ratio",),
    "num_masked_seqs_by_logprob_error": (
        "train/num_masked_seqs_by_logprob_error",
        "train/num_mask_sample_filtered",
    ),
    "router_topk_parity": (
        "correctness/router_topk_parity",
        "cuda_graph/router_topk_parity",
    ),
    "expert_count_parity": (
        "correctness/expert_count_parity",
        "cuda_graph/expert_count_parity",
    ),
    "parameter_delta_parity": (
        "correctness/parameter_delta_parity",
        "parameter_delta_parity",
    ),
    "parameter_delta_max_abs_error": (
        "correctness/parameter_delta_max_abs_error",
        "parameter_delta_max_abs_error",
    ),
    "parameter_delta_max_rel_error": (
        "correctness/parameter_delta_max_rel_error",
        "parameter_delta_max_rel_error",
    ),
    "grad_norm": ("train/grad_norm",),
    "nan_inf_status": (
        "correctness/nan_inf_status",
        "train/nan_inf_status",
    ),
}


def _first_value(
    sources: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
) -> Any:
    for alias in aliases:
        for source in sources:
            if alias in source and source[alias] is not None:
                return source[alias]
    return ""


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value).lower()
    return value


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one local row while retaining failure-only records."""
    nested_metrics = record.get("metrics", {})
    nested_provenance = record.get("provenance", {})
    nested_parity = record.get("parity", {})
    if not isinstance(nested_metrics, Mapping):
        raise TypeError("record.metrics must be a mapping")
    if not isinstance(nested_provenance, Mapping):
        raise TypeError("record.provenance must be a mapping")
    if not isinstance(nested_parity, Mapping):
        raise TypeError("record.parity must be a mapping")

    row: dict[str, Any] = {field: "" for field in CSV_FIELDS}
    for field in IDENTITY_FIELDS:
        aliases = (field, "error") if field == "failure" else (field,)
        row[field] = _first_value((record,), aliases)
    if row["router_replay"] == "":
        row["router_replay"] = "off"
    if row["step"] == "":
        row["step"] = _first_value((nested_metrics, record), ("_step",))

    for field, aliases in METRIC_ALIASES.items():
        row[field] = _first_value(
            (record, nested_metrics, nested_parity), (field, *aliases)
        )
    for field in PROVENANCE_FIELDS:
        aliases = {
            "te_commit": ("te_commit", "transformer_engine_commit"),
            "te_version": ("te_version", "transformer_engine_version"),
        }.get(field, (field,))
        row[field] = _first_value((record, nested_provenance), aliases)

    graph_calls = row["graph_calls"]
    eligible_calls = row["eligible_calls"]
    if row["graph_coverage"] == "" and graph_calls != "" and eligible_calls != "":
        try:
            eligible = float(eligible_calls)
            row["graph_coverage"] = float(graph_calls) / eligible if eligible else 0.0
        except (TypeError, ValueError):
            pass

    numeric_values = [
        row[field]
        for field in (
            *PERFORMANCE_FIELDS,
            *GRAPH_FIELDS,
            "reward",
            "policy_loss",
            "gen_kl_error",
            "token_mult_prob_error",
            "policy_kl_error",
            "js_divergence_error",
            "sampling_importance_ratio",
            "num_masked_seqs_by_logprob_error",
            "parameter_delta_max_abs_error",
            "parameter_delta_max_rel_error",
            "grad_norm",
        )
        if field != "graph_telemetry_status" and row[field] != ""
    ]
    nonfinite = any(
        isinstance(value, float) and not math.isfinite(value)
        for value in numeric_values
    )
    if row["nan_inf_status"] == "" and numeric_values:
        row["nan_inf_status"] = "detected" if nonfinite else "clear"

    return {field: _json_safe(row[field]) for field in CSV_FIELDS}


def steady_state_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    first_step: int = 6,
    last_step: int | None = None,
) -> list[Mapping[str, Any]]:
    """Exclude warmup/capture steps and return the measurement window."""
    if first_step < 1:
        raise ValueError("first_step must be positive")
    if last_step is not None and first_step > last_step:
        raise ValueError("first_step must not exceed last_step")
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        raw_step = row.get("step", "")
        if raw_step == "":
            continue
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid optimizer step for job {row.get('job_id', '')}: {raw_step!r}"
            ) from error
        if step >= first_step and (last_step is None or step <= last_step):
            selected.append(row)
    return selected


def read_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Read JSON objects, JSON arrays, schema ledgers, or JSONL files."""
    records: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text().strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            payload = payload["rows"]
        elif isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list) or not all(
            isinstance(record, dict) for record in payload
        ):
            raise ValueError(f"result input must contain JSON objects: {path}")
        records.extend(payload)
    return records


def _atomic_text(output: Path, writer: Callable[[Any], None]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            writer(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
        output.chmod(0o644)
    except BaseException:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise


def write_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_json: Path,
    output_csv: Path,
) -> None:
    """Atomically publish the stable JSON and CSV result schemas."""
    if not rows:
        raise ValueError("refusing to publish results with no result rows")
    ordered_rows = [
        {field: _json_safe(row.get(field, "")) for field in CSV_FIELDS} for row in rows
    ]

    def write_json(stream: TextIO) -> None:
        json.dump(
            {
                "schema_version": 1,
                "fields": list(CSV_FIELDS),
                "rows": ordered_rows,
            },
            stream,
            allow_nan=False,
            indent=2,
        )
        stream.write("\n")

    def write_csv(stream: TextIO) -> None:
        fieldnames = [str(field) for field in CSV_FIELDS]
        csv_writer = csv.DictWriter(stream, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(ordered_rows)

    _atomic_text(output_json, write_json)
    _atomic_text(output_csv, write_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="JSON or JSONL input; repeat for multiple local artifacts",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.input or sorted(DEFAULT_RAW_DIR.glob("*.json*"))
    rows = [normalize_record(record) for record in read_records(inputs)]
    write_results(rows, output_json=args.output_json, output_csv=args.output_csv)
    print(
        json.dumps(
            {
                "input_count": len(inputs),
                "row_count": len(rows),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
