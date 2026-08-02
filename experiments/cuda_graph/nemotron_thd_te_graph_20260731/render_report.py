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

"""Render the normalized multi-model CUDA Graph ledger as static HTML."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXPERIMENT_DIR / "results" / "results.json"
DEFAULT_NSYS = EXPERIMENT_DIR / "results" / "nsys_cuda_graph_calls.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "results" / "report.html"

IDENTITY_FIELDS = (
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
NUMERIC_FIELDS = (
    *PERFORMANCE_FIELDS,
    *(field for field in GRAPH_FIELDS if field != "graph_telemetry_status"),
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
RUN_KEY_FIELDS = (
    "model",
    "dispatcher",
    "scope",
    "router_replay",
    "mode",
    "cluster",
    "profile",
    "phase",
    "steps",
    "repeat",
    "run_group",
    "job_id",
)
BASELINE_SCOPES = frozenset({"baseline", "baseline_no_cg"})
MATCH_FIELDS = (
    "model",
    "dispatcher",
    "router_replay",
    "mode",
    "cluster",
    "profile",
    "phase",
    "steps",
    "run_group",
    "repeat",
)
COMPARISON_GROUP_FIELDS = MATCH_FIELDS[:-1]
COMMIT_FIELDS = frozenset(
    {"nemo_rl_commit", "bridge_commit", "mcore_commit", "te_commit"}
)
PARITY_FIELDS = (
    "router_topk_parity",
    "expert_count_parity",
    "parameter_delta_parity",
)
CORRECTNESS_NUMERIC_FIELDS = (
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
REQUIRED_GRAPH_NUMERIC_FIELDS = (
    "capture_count",
    "replay_count",
    "cache_hits",
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
GRAPH_COUNTER_FIELDS = frozenset(
    {
        "capture_count",
        "replay_count",
        "cache_hits",
        "cache_misses",
        "cache_evictions",
        "fallback_count",
        "graph_calls",
        "eligible_calls",
        "logical_tokens",
        "padded_tokens",
        "capacity_tokens",
    }
)


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _is_failure(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status", "")).lower()
    failure = str(row.get("failure", ""))
    exit_code = row.get("exit_code", "")
    return (
        bool(failure)
        or str(exit_code) not in {"", "0"}
        or any(
            marker in status
            for marker in ("fail", "error", "cancel", "timeout", "oom", "invalid")
        )
    )


def _is_completed(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status", "")).lower()
    return not _is_failure(row) and status in {
        "pass",
        "passed",
        "complete",
        "completed",
        "success",
        "succeeded",
    }


def validate_completed_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Strictly validate rows when a caller explicitly promotes comparisons."""
    errors = [
        f"{row.get('model', '')}/{row.get('scope', '')}/{row.get('job_id', '')}: "
        + ", ".join(issues)
        for row in summarize_runs(rows)
        if (issues := comparison_issues(row))
    ]
    if errors:
        raise ValueError("completed report rows are incomplete: " + "; ".join(errors))


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _truth(value: Any) -> bool | None:
    if type(value) is bool:
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "pass", "passed"}:
        return True
    if normalized in {"false", "0", "fail", "failed"}:
        return False
    return None


def comparison_issues(row: Mapping[str, Any]) -> list[str]:
    """Return every reason a summarized run cannot support a comparison."""
    issues: list[str] = []
    if not _is_completed(row):
        issues.append("status is not completed")
    for field in (*MATCH_FIELDS, "job_id"):
        if row.get(field, "") == "":
            issues.append(f"missing {field}")
    repeat = _number(row.get("repeat", ""))
    if repeat is None or not repeat.is_integer() or repeat < 1:
        issues.append("repeat must be a positive integer")
    sample_count = _number(row.get("sample_count", ""))
    if sample_count in {None, 0.0}:
        issues.append("no steady-state samples")
    steps = _number(row.get("steps", ""))
    if (
        steps is not None
        and steps.is_integer()
        and steps >= 5
        and sample_count is not None
    ):
        expected_samples = int(steps) - 5
        if sample_count != expected_samples:
            issues.append(f"expected {expected_samples} steady-state samples")

    for field in (*PERFORMANCE_FIELDS, *CORRECTNESS_NUMERIC_FIELDS):
        value = _number(row.get(field, ""))
        if value is None:
            issues.append(f"{field} must be finite numeric data")
        elif field in PERFORMANCE_FIELDS and value <= 0:
            issues.append(f"{field} must be positive")
        elif (
            sample_count is not None
            and _number(row.get(f"{field}_sample_count", "")) != sample_count
        ):
            issues.append(f"{field} is incomplete across steady-state samples")
    if str(row.get("scope", "")) not in BASELINE_SCOPES:
        for field in REQUIRED_GRAPH_NUMERIC_FIELDS:
            value = _number(row.get(field, ""))
            if value is None:
                issues.append(f"{field} must be finite numeric data")
            elif value < 0:
                issues.append(f"{field} must be nonnegative")
            elif field in GRAPH_COUNTER_FIELDS and not value.is_integer():
                issues.append(f"{field} must be an integer")
            if (
                sample_count is not None
                and _number(row.get(f"{field}_sample_count", "")) != sample_count
            ):
                issues.append(f"{field} is incomplete across steady-state samples")
        for field in ("graph_coverage", "capacity_utilization", "padding_utilization"):
            value = _number(row.get(field, ""))
            if value is not None and not 0.0 <= value <= 1.0:
                issues.append(f"{field} must be between zero and one")
        if _number(row.get("fallback_count", "")) not in {0, 0.0}:
            issues.append("fallback_count must be zero")
    elif str(row.get("graph_telemetry_status", "")) != "not_applicable":
        issues.append("baseline graph telemetry must be explicitly not_applicable")
    cache_misses = _number(row.get("cache_misses", ""))
    if cache_misses is not None and (cache_misses < 0 or not cache_misses.is_integer()):
        issues.append("cache_misses must be a nonnegative integer")

    for field in PARITY_FIELDS:
        if _truth(row.get(field, "")) is not True:
            issues.append(f"{field} must be explicitly true")
        if (
            sample_count is not None
            and _number(row.get(f"{field}_sample_count", "")) != sample_count
        ):
            issues.append(f"{field} is incomplete across steady-state samples")
    if str(row.get("nan_inf_status", "")).strip().lower() != "clear":
        issues.append("nan_inf_status must be clear")
    if row.get("provenance_consistent") is not True:
        issues.append("provenance differs across steady-state samples")

    for field in PROVENANCE_FIELDS:
        value = str(row.get(field, ""))
        if field in COMMIT_FIELDS:
            valid = re.fullmatch(r"[0-9a-f]{40}", value) is not None
        elif field == "container_sha256":
            valid = re.fullmatch(r"[0-9a-f]{64}", value) is not None
        else:
            valid = re.fullmatch(r"\d+\.\d+(?:[A-Za-z0-9.+-]*)", value) is not None
        if not valid:
            issues.append(f"invalid {field}")
    return issues


def _step(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("step", 0))
    except (TypeError, ValueError):
        return 0


def summarize_runs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one steady-state median row per model/scope/job run."""
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if _is_failure(row):
            continue
        key = tuple(
            str(row.get(field, "off" if field == "router_replay" else ""))
            for field in RUN_KEY_FIELDS
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=_step)
        measurement_rows = [row for row in ordered if _step(row) >= 6]
        summary = {
            field: ordered[-1].get(
                field, "off" if field == "router_replay" else ""
            )
            for field in IDENTITY_FIELDS
        }
        summary["sample_count"] = len(measurement_rows)
        for field in NUMERIC_FIELDS:
            values = [
                numeric
                for row in measurement_rows
                if (numeric := _number(row.get(field, ""))) is not None
            ]
            summary[field] = statistics.median(values) if values else ""
            summary[f"{field}_sample_count"] = len(values)
        for field in PARITY_FIELDS:
            values = [row.get(field, "") for row in measurement_rows]
            present = [value for value in values if value != ""]
            summary[field] = (
                all(
                    str(value).lower() in {"true", "1", "pass", "passed"}
                    for value in present
                )
                if present
                else ""
            )
            summary[f"{field}_sample_count"] = len(present)
        nan_statuses = {
            str(row.get("nan_inf_status", ""))
            for row in measurement_rows
            if row.get("nan_inf_status", "") != ""
        }
        summary["nan_inf_status"] = ", ".join(sorted(nan_statuses))
        graph_statuses = {
            str(row.get("graph_telemetry_status", ""))
            for row in measurement_rows
            if row.get("graph_telemetry_status", "") != ""
        }
        summary["graph_telemetry_status"] = ", ".join(sorted(graph_statuses))
        for field in PROVENANCE_FIELDS:
            summary[field] = ordered[-1].get(field, "")
        summary["provenance_consistent"] = all(
            len({str(row.get(field, "")) for row in measurement_rows}) == 1
            for field in PROVENANCE_FIELDS
        )
        issues = comparison_issues(summary)
        summary["comparison_status"] = "eligible" if not issues else "provisional"
        summary["comparison_issues"] = ", ".join(issues)
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            str(row.get("model", "")),
            str(row.get("dispatcher", "")),
            str(row.get("scope", "")),
            str(row.get("job_id", "")),
        ),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of no values")
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _provenance_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in PROVENANCE_FIELDS)


def build_matched_comparisons(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair each eligible repeat with one exact-provenance no-CG baseline."""
    eligible = [row for row in summaries if not comparison_issues(row)]
    baselines: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in eligible:
        if str(row.get("scope", "")) in BASELINE_SCOPES:
            key = tuple(
                str(row.get(field, "off" if field == "router_replay" else ""))
                for field in MATCH_FIELDS
            )
            baselines.setdefault(key, []).append(row)

    paired: dict[tuple[str, ...], list[dict[str, float]]] = {}
    identities: dict[tuple[str, ...], dict[str, Any]] = {}
    for variant in eligible:
        scope = str(variant.get("scope", ""))
        if scope in BASELINE_SCOPES:
            continue
        match_key = tuple(
            str(variant.get(field, "off" if field == "router_replay" else ""))
            for field in MATCH_FIELDS
        )
        candidates = baselines.get(match_key, [])
        if len(candidates) != 1:
            continue
        baseline = candidates[0]
        if _provenance_key(variant) != _provenance_key(baseline):
            continue
        values: dict[str, float] = {}
        for field in PERFORMANCE_FIELDS:
            baseline_value = _number(baseline.get(field, ""))
            variant_value = _number(variant.get(field, ""))
            if baseline_value is None or baseline_value == 0.0 or variant_value is None:
                break
            values[field] = 100.0 * (variant_value - baseline_value) / baseline_value
        else:
            group_key = (
                *(str(variant.get(field, "")) for field in COMPARISON_GROUP_FIELDS),
                scope,
                *_provenance_key(variant),
            )
            paired.setdefault(group_key, []).append(values)
            identities[group_key] = {
                field: variant.get(field, "") for field in COMPARISON_GROUP_FIELDS
            } | {"scope": scope}

    comparisons: list[dict[str, Any]] = []
    for key, repeats in paired.items():
        comparison = {**identities[key], "repeat_count": len(repeats)}
        for field in PERFORMANCE_FIELDS:
            deltas = [repeat[field] for repeat in repeats]
            comparison[f"{field}_delta_pct_median"] = statistics.median(deltas)
            comparison[f"{field}_delta_pct_variance"] = statistics.pvariance(deltas)
            comparison[f"{field}_delta_pct_p95"] = _percentile(deltas, 0.95)
        comparisons.append(comparison)
    return sorted(
        comparisons,
        key=lambda row: (
            str(row.get("model", "")),
            str(row.get("dispatcher", "")),
            str(row.get("run_group", "")),
            str(row.get("scope", "")),
        ),
    )


def _format_value(field: str, value: Any) -> str:
    if value == "":
        return ""
    if field in {"graph_coverage", "capacity_utilization", "padding_utilization"}:
        numeric = _number(value)
        return f"{100 * numeric:.2f}%" if numeric is not None else str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
    *,
    empty_message: str,
) -> str:
    if not rows:
        return f'<p class="pending">{escape(empty_message)}</p>'
    headers = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{escape(_format_value(field, row.get(field, '')))}</td>"
            for field, _ in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + headers
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _nsys_rows(
    nsys_coverage: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for label, summary in sorted(nsys_coverage.items()):
        share = _number(
            summary.get("nsys_cuda_graph_launch_share_of_cuda_api_calls_pct", "")
        )
        rows.append(
            {
                "label": label,
                "profile_count": summary.get("nsys_profile_count", ""),
                "profiles_with_launches": summary.get(
                    "nsys_profiles_with_cuda_graph_launches", ""
                ),
                "api_share": f"{share:.2f}%" if share is not None else "",
            }
        )
    return rows


def render_html(
    rows: Sequence[Mapping[str, Any]],
    *,
    nsys_coverage: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Build a self-contained, escaped report for every model and dispatcher."""
    summaries = summarize_runs(rows)
    comparisons = build_matched_comparisons(summaries)
    provisional = [
        summary
        for summary in summaries
        if summary.get("comparison_status") != "eligible"
    ]
    failures = [
        {**row, "router_replay": row.get("router_replay") or "off"}
        for row in rows
        if _is_failure(row)
    ]
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    identity_columns = (
        ("model", "Model"),
        ("dispatcher", "Dispatcher"),
        ("scope", "Scope"),
        ("router_replay", "Router replay"),
        ("mode", "Mode"),
        ("cluster", "Cluster"),
        ("profile", "Profile"),
        ("phase", "Phase"),
        ("steps", "Steps"),
        ("repeat", "Repeat"),
        ("run_group", "Run group"),
        ("sample_count", "Steady samples"),
        ("comparison_status", "Comparison status"),
        ("status", "Status"),
        ("job_id", "Job ID"),
    )
    compact_identity_columns = (
        ("model", "Model"),
        ("dispatcher", "Dispatcher"),
        ("scope", "Scope"),
        ("router_replay", "Router replay"),
        ("phase", "Phase"),
        ("steps", "Steps"),
        ("repeat", "Repeat"),
        ("run_group", "Run group"),
        ("job_id", "Job ID"),
        ("comparison_status", "Comparison status"),
    )
    performance_columns = compact_identity_columns + tuple(
        (field, label)
        for field, label in (
            ("e2e_step_time", "E2E step time"),
            ("e2e_tokens_per_sec_per_gpu", "E2E tokens/s/GPU"),
            ("generation_time", "Generation time"),
            ("generation_tokens_per_sec_per_gpu", "Generation tokens/s/GPU"),
            ("policy_training_time", "Policy-training time"),
            (
                "policy_training_tokens_per_sec_per_gpu",
                "Policy-training tokens/s/GPU",
            ),
            ("logprob_time", "Logprob time"),
            ("logprob_tokens_per_sec_per_gpu", "Logprob tokens/s/GPU"),
        )
    )
    graph_columns = compact_identity_columns + tuple(
        (field, label)
        for field, label in (
            ("graph_telemetry_status", "Graph telemetry"),
            ("capture_count", "Captures"),
            ("replay_count", "Replays"),
            ("cache_hits", "Cache hits"),
            ("cache_misses", "Cache misses"),
            ("cache_evictions", "Cache evictions"),
            ("fallback_count", "Fallbacks"),
            ("graph_calls", "Graph calls"),
            ("eligible_calls", "Eligible calls"),
            ("graph_coverage", "Runtime graph coverage"),
            ("logical_tokens", "Logical tokens"),
            ("padded_tokens", "Padded tokens"),
            ("capacity_tokens", "Capacity tokens"),
            ("capacity_utilization", "Capacity utilization"),
            ("padding_utilization", "Padding utilization"),
        )
    )
    correctness_columns = compact_identity_columns + tuple(
        (field, field.replace("_", " ").title()) for field in CORRECTNESS_FIELDS
    )
    provenance_columns = compact_identity_columns + tuple(
        (field, field.replace("_", " ").title()) for field in PROVENANCE_FIELDS
    )
    failure_columns = (
        ("model", "Model"),
        ("scope", "Scope"),
        ("router_replay", "Router replay"),
        ("status", "Status"),
        ("failure", "Failure"),
        ("exit_code", "Exit code"),
        ("job_id", "Job ID"),
    )
    nsys_rows = _nsys_rows(nsys_coverage or {})
    nsys_columns = (
        ("label", "Profile label"),
        ("profile_count", "Profiles"),
        ("profiles_with_launches", "Profiles with graph launches"),
        ("api_share", "Graph launches / all CUDA API calls"),
    )
    comparison_columns: tuple[tuple[str, str], ...] = (
        ("model", "Model"),
        ("dispatcher", "Dispatcher"),
        ("scope", "Scope"),
        ("router_replay", "Router replay"),
        ("profile", "Profile"),
        ("phase", "Phase"),
        ("steps", "Steps"),
        ("run_group", "Run group"),
        ("repeat_count", "Matched repeats"),
    ) + tuple(
        column
        for field, label in (
            ("e2e_step_time", "E2E step time"),
            ("e2e_tokens_per_sec_per_gpu", "E2E tokens/s/GPU"),
            ("generation_time", "Generation time"),
            ("generation_tokens_per_sec_per_gpu", "Generation tokens/s/GPU"),
            ("policy_training_time", "Policy-training time"),
            (
                "policy_training_tokens_per_sec_per_gpu",
                "Policy-training tokens/s/GPU",
            ),
            ("logprob_time", "Logprob time"),
            ("logprob_tokens_per_sec_per_gpu", "Logprob tokens/s/GPU"),
        )
        for column in (
            (f"{field}_delta_pct_median", f"{label} delta median (%)"),
            (f"{field}_delta_pct_variance", f"{label} repeat variance"),
            (f"{field}_delta_pct_p95", f"{label} delta p95 (%)"),
        )
    )
    provisional_columns = (
        ("model", "Model"),
        ("dispatcher", "Dispatcher"),
        ("scope", "Scope"),
        ("router_replay", "Router replay"),
        ("profile", "Profile"),
        ("phase", "Phase"),
        ("steps", "Steps"),
        ("repeat", "Repeat"),
        ("run_group", "Run group"),
        ("job_id", "Job ID"),
        ("comparison_issues", "Why comparison is blocked"),
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nemotron THD Transformer Engine CUDA Graph Report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #202124; }}
h1, h2 {{ color: #17365d; }}
.table-wrap {{ overflow-x: auto; margin-bottom: 1.5rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
th, td {{ border: 1px solid #c8ccd0; padding: 0.4rem 0.55rem; text-align: left; white-space: nowrap; }}
th {{ background: #edf2f7; }}
.pending {{ color: #6b7280; font-style: italic; }}
.definition {{ background: #f6f8fa; border-left: 4px solid #4f78a8; padding: 0.75rem; }}
</style>
</head>
<body>
<h1>Nemotron packed-THD Transformer Engine CUDA Graph study</h1>
<p>Generated {escape(generated_at)} from local normalized artifacts.</p>
<h2>Run inventory</h2>
{_table(summaries, identity_columns, empty_message="No collected runs yet.")}
<h2>Matched baseline comparisons</h2>
{_table(comparisons, comparison_columns, empty_message="No comparison-eligible matched baseline pairs.")}
<h2>Provisional / incomplete runs</h2>
{_table(provisional, provisional_columns, empty_message="No provisional runs.")}
<h2>Performance</h2>
{_table(summaries, performance_columns, empty_message="No performance rows yet.")}
<h2>Runtime graph coverage (graph_calls / eligible_calls)</h2>
<p class="definition">This percentage measures eligible model-module calls replayed by the NeMo-RL runtime.</p>
{_table(summaries, graph_columns, empty_message="No runtime graph telemetry yet.")}
<h2>Nsight CUDA API launch share</h2>
<p class="definition">This percentage uses all CUDA runtime and driver API calls as the denominator. It is not runtime eligible-call coverage.</p>
{_table(nsys_rows, nsys_columns, empty_message="No Nsight profiles collected yet.")}
<h2>Correctness</h2>
{_table(summaries, correctness_columns, empty_message="No correctness rows yet.")}
<h2>Provenance</h2>
{_table(summaries, provenance_columns, empty_message="No provenance rows yet.")}
<h2>Raw failures</h2>
{_table(failures, failure_columns, empty_message="No failures recorded.")}
</body>
</html>
"""


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"normalized report input is missing: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("rows", [])
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError("normalized report input must contain a rows array")
    return payload


def read_nsys(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not all(
        isinstance(summary, dict) for summary in payload.values()
    ):
        raise ValueError("Nsight coverage input must be an object of summaries")
    return payload


def write_report(report: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            temporary.write(report)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
        output.chmod(0o644)
    except BaseException:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit("refusing to overwrite the report with no result rows")
    report = render_html(rows, nsys_coverage=read_nsys(args.nsys))
    write_report(report, args.output)
    print(
        json.dumps({"row_count": len(rows), "output": str(args.output)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
