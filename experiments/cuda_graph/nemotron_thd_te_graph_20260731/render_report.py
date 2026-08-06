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
from typing import Any, NotRequired, TypedDict
from urllib.parse import urlsplit, urlunsplit

try:
    from experiments.cuda_graph.nemotron_thd_te_graph_20260731.scope_matrix import (
        MODEL_NAMES,
        classify_scope,
        find_scope_row,
    )
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    from scope_matrix import MODEL_NAMES, classify_scope, find_scope_row


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXPERIMENT_DIR / "results" / "results.json"
DEFAULT_NSYS = EXPERIMENT_DIR / "results" / "nsys_cuda_graph_calls.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "results" / "report.html"
DEFAULT_CONTEXT = EXPERIMENT_DIR / "report_context.json"

SUPPORT_SCOPES = (
    ("baseline", "Baseline"),
    ("attn", "Attn"),
    ("mlp", "MLP"),
    ("mamba", "Mamba"),
    ("moe_router", "Router"),
    ("moe_router,moe_preprocess", "Router + preprocess"),
    (
        "attn,mamba,moe_router,moe_preprocess",
        "Attn + Mamba + router + preprocess",
    ),
)


class ReportItem(TypedDict):
    """One concise editorial report item."""

    text: str
    href: NotRequired[str]


class ReportContext(TypedDict):
    """Versioned editorial context; measured status stays derived from results."""

    schema_version: int
    current_status: list[ReportItem]
    changes: list[ReportItem]
    next_steps: list[ReportItem]


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
            field: ordered[-1].get(field, "off" if field == "router_replay" else "")
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
    evidence_pairs: dict[tuple[str, ...], list[str]] = {}
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
            evidence_pairs.setdefault(group_key, []).append(
                f"{baseline.get('job_id', '')} -> {variant.get('job_id', '')}"
            )
            identities[group_key] = {
                field: variant.get(field, "") for field in COMPARISON_GROUP_FIELDS
            } | {"scope": scope}

    comparisons: list[dict[str, Any]] = []
    for key, repeats in paired.items():
        comparison = {
            **identities[key],
            "repeat_count": len(repeats),
            "evidence_pairs": "; ".join(sorted(evidence_pairs[key])),
        }
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


def normalize_report_context(payload: object) -> ReportContext:
    """Validate editorial context at both file and rendering boundaries."""
    if not isinstance(payload, dict):
        raise ValueError("report context must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("report context schema_version must be 1")
    allowed_fields = {"schema_version", "current_status", "changes", "next_steps"}
    unknown_fields = sorted(set(payload) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            f"report context has unknown fields: {', '.join(unknown_fields)}"
        )

    context: ReportContext = {
        "schema_version": 1,
        "current_status": [],
        "changes": [],
        "next_steps": [],
    }
    for field in ("current_status", "changes", "next_steps"):
        items = payload.get(field, [])
        if not isinstance(items, list):
            raise ValueError(f"report context {field} must be an array")
        normalized_items: list[ReportItem] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict) or set(item) - {"text", "href"}:
                raise ValueError(
                    f"report context {field}[{index}] must contain text and optional href"
                )
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"report context {field}[{index}].text must be non-empty"
                )
            normalized_item: ReportItem = {"text": text.strip()}
            href = item.get("href")
            if href is not None:
                if not isinstance(href, str) or not href.strip():
                    raise ValueError(
                        f"report context {field}[{index}].href must be non-empty"
                    )
                href = href.strip()
                parsed = urlsplit(href)
                if (
                    parsed.scheme not in {"", "https"}
                    or (not parsed.scheme and parsed.netloc)
                    or (not parsed.scheme and parsed.path.startswith("/"))
                ):
                    raise ValueError(
                        f"report context {field}[{index}] has unsupported href"
                    )
                normalized_item["href"] = href
            normalized_items.append(normalized_item)
        context[field] = normalized_items
    return context


def rebase_report_context_links(
    context: Mapping[str, Any],
    *,
    context_path: Path,
    output_path: Path,
) -> ReportContext:
    """Rebase context-relative artifact links for the selected output path."""
    normalized = normalize_report_context(context)
    rebased: ReportContext = {
        "schema_version": 1,
        "current_status": [],
        "changes": [],
        "next_steps": [],
    }
    for field in ("current_status", "changes", "next_steps"):
        rebased_items: list[ReportItem] = []
        for item in normalized[field]:
            rebased_item: ReportItem = {"text": item["text"]}
            href = item.get("href")
            if href:
                parsed = urlsplit(href)
                if parsed.scheme == "https" or (not parsed.path and parsed.fragment):
                    rebased_item["href"] = href
                else:
                    target = (context_path.parent / parsed.path).resolve()
                    relative = os.path.relpath(
                        target,
                        start=output_path.parent.resolve(),
                    )
                    rebased_item["href"] = urlunsplit(
                        ("", "", relative, parsed.query, parsed.fragment)
                    )
            rebased_items.append(rebased_item)
        rebased[field] = rebased_items
    return rebased


def _items(items: Sequence[ReportItem], *, empty_message: str) -> str:
    if not items:
        return f'<p class="pending">{escape(empty_message)}</p>'
    rendered = []
    for item in items:
        text = escape(item["text"])
        href = item.get("href", "")
        if href:
            text = f'<a href="{escape(href)}">{text}</a>'
        rendered.append(f"<li>{text}</li>")
    return "<ul>" + "".join(rendered) + "</ul>"


def _support_matrix() -> str:
    status_labels = {
        "runnable": "runnable",
        "model-incompatible": "model-incompatible",
        "capacity-blocked": "capacity-blocked",
        "dependency-blocked": "dependency-blocked",
        "submitted": "submitted",
    }
    headers = "<th>Model</th>" + "".join(
        f"<th>{escape(label)}</th>" for _, label in SUPPORT_SCOPES
    )
    body = []
    for model in MODEL_NAMES:
        cells = [f"<td>{escape(model)}</td>"]
        for scope, _ in SUPPORT_SCOPES:
            classification = classify_scope(find_scope_row(scope), model=model)
            label = status_labels[classification.status]
            cells.append(
                f'<td class="status {escape(classification.status)}" '
                f'title="{escape(classification.reason)}">{escape(label)}</td>'
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + headers
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_html(
    rows: Sequence[Mapping[str, Any]],
    *,
    nsys_coverage: Mapping[str, Mapping[str, Any]] | None = None,
    report_context: Mapping[str, Any] | None = None,
) -> str:
    """Build a concise, self-contained report from canonical experiment data."""
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
    context = normalize_report_context(
        report_context if report_context is not None else {"schema_version": 1}
    )
    current_status = context["current_status"]
    changes = context["changes"]
    next_steps = context["next_steps"]
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    source_description = (
        "normalized result artifacts and versioned report context"
        if rows
        else "versioned report context; no normalized result rows are present"
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
        ("scope", "Scope"),
        ("router_replay", "Router replay"),
        ("steps", "Steps"),
        ("cluster", "Cluster"),
        ("profile", "Profile"),
        ("phase", "Phase"),
        ("run_group", "Run group"),
        ("repeat_count", "Matched repeats"),
        ("evidence_pairs", "Baseline -> graph jobs"),
        ("e2e_step_time_delta_pct_median", "E2E step time delta (%)"),
        (
            "e2e_tokens_per_sec_per_gpu_delta_pct_median",
            "E2E tokens/s/GPU delta (%)",
        ),
        ("generation_time_delta_pct_median", "Generation time delta (%)"),
        (
            "generation_tokens_per_sec_per_gpu_delta_pct_median",
            "Generation tokens/s/GPU delta (%)",
        ),
        ("policy_training_time_delta_pct_median", "Policy time delta (%)"),
        (
            "policy_training_tokens_per_sec_per_gpu_delta_pct_median",
            "Policy tokens/s/GPU delta (%)",
        ),
        ("logprob_time_delta_pct_median", "Logprob time delta (%)"),
        (
            "logprob_tokens_per_sec_per_gpu_delta_pct_median",
            "Logprob tokens/s/GPU delta (%)",
        ),
    )
    validation_columns = (
        ("model", "Model"),
        ("scope", "Scope"),
        ("router_replay", "Router replay"),
        ("steps", "Steps"),
        ("graph_coverage", "Runtime graph coverage"),
        ("fallback_count", "Fallbacks"),
        ("router_topk_parity", "Router parity"),
        ("expert_count_parity", "Expert parity"),
        ("parameter_delta_parity", "Update parity"),
        ("nan_inf_status", "NaN/Inf"),
        ("comparison_status", "Validation"),
        ("job_id", "Job ID"),
    )
    provisional_columns = (
        ("model", "Model"),
        ("scope", "Scope"),
        ("router_replay", "Router replay"),
        ("steps", "Steps"),
        ("job_id", "Job ID"),
        ("comparison_issues", "Blocked by"),
    )
    provenance_columns = (
        ("model", "Model"),
        ("scope", "Scope"),
        ("run_group", "Run group"),
        ("job_id", "Job ID"),
        ("cluster", "Cluster"),
        ("profile", "Profile"),
        ("phase", "Phase"),
        ("nemo_rl_commit", "NeMo-RL commit"),
        ("bridge_commit", "Bridge commit"),
        ("mcore_commit", "MCore commit"),
        ("te_commit", "TE commit"),
        ("container_sha256", "Container SHA256"),
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nemotron THD Transformer Engine CUDA Graph Report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1500px; padding: 0 1rem; color: #202124; }}
h1, h2, h3 {{ color: #17365d; }}
h1 {{ margin-bottom: 0.25rem; }}
h2 {{ border-bottom: 1px solid #d8dee4; padding-bottom: 0.35rem; }}
.table-wrap {{ overflow-x: auto; margin-bottom: 1rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
th, td {{ border: 1px solid #c8ccd0; padding: 0.4rem 0.55rem; text-align: left; white-space: nowrap; }}
th {{ background: #edf2f7; }}
a {{ color: #075985; }}
.counts {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.8rem 0 1rem; }}
.count {{ background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 0.35rem; padding: 0.55rem 0.75rem; }}
.pending {{ color: #6b7280; font-style: italic; }}
.definition {{ background: #f6f8fa; border-left: 4px solid #4f78a8; padding: 0.75rem; }}
.status {{ font-size: 0.78rem; }}
.runnable {{ background: #dcfce7; }}
.model-incompatible {{ background: #f3f4f6; color: #6b7280; }}
.capacity-blocked, .dependency-blocked {{ background: #fef3c7; }}
.submitted {{ background: #dbeafe; }}
</style>
</head>
<body>
<h1>Nemotron packed-THD Transformer Engine CUDA Graph study</h1>
<p>Generated {escape(generated_at)} from {escape(source_description)}.</p>
<p><a href="cudagraph_implementation_explainer.html">Open the implementation explainer</a></p>
<h2>Current status</h2>
{_items(current_status, empty_message="No editorial status update.")}
<div class="counts">
  <span class="count"><strong>{len(summaries)}</strong> normalized runs</span>
  <span class="count"><strong>{len(comparisons)}</strong> matched comparisons</span>
  <span class="count"><strong>{len(provisional)}</strong> provisional runs</span>
  <span class="count"><strong>{len(failures)}</strong> failures</span>
</div>
<h3>Static preflight support</h3>
<p class="definition">This table is derived from the canonical model selectors and scope classifier. Runnable means submission preflight passed; it is not runtime or correctness proof. Hover for the reason.</p>
{_support_matrix()}
<h2>Changes</h2>
{_items(changes, empty_message="No changes recorded.")}
<h2>Validation</h2>
<h3>Matched performance</h3>
{_table(comparisons, comparison_columns, empty_message="No comparison-eligible matched baseline pairs.")}
<h3>Runtime coverage and correctness</h3>
{_table(summaries, validation_columns, empty_message="No normalized validation rows yet.")}
{("<h3>Provisional / incomplete runs</h3>" + _table(provisional, provisional_columns, empty_message="No provisional runs.")) if provisional else ""}
{("<h3>Failures</h3>" + _table(failures, failure_columns, empty_message="No failures recorded.")) if failures else ""}
{("<h3>Nsight CUDA API launch share</h3>" + _table(nsys_rows, nsys_columns, empty_message="No Nsight profiles collected yet.")) if nsys_rows else ""}
{("<details><summary>Evidence and provenance</summary>" + _table(summaries, provenance_columns, empty_message="No provenance rows yet.") + "</details>") if summaries else ""}
<h2>Next steps</h2>
{_items(next_steps, empty_message="No next steps recorded.")}
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


def read_report_rows(path: Path) -> list[dict[str, Any]]:
    """Read normalized rows, allowing the ignored default ledger to be absent."""
    if path == DEFAULT_INPUT and not path.exists():
        return []
    return read_rows(path)


def read_report_context(path: Path) -> ReportContext:
    """Read concise editorial context and reject stale or unsafe schemas."""
    if not path.is_file():
        raise FileNotFoundError(f"report context is missing: {path}")
    return normalize_report_context(json.loads(path.read_text()))


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
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_report_rows(args.input)
    context = rebase_report_context_links(
        read_report_context(args.context),
        context_path=args.context,
        output_path=args.output,
    )
    if not rows and not any(
        context[field] for field in ("current_status", "changes", "next_steps")
    ):
        raise SystemExit("refusing to overwrite the report with no results or context")
    report = render_html(
        rows,
        nsys_coverage=read_nsys(args.nsys),
        report_context=context,
    )
    write_report(report, args.output)
    print(
        json.dumps({"row_count": len(rows), "output": str(args.output)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
