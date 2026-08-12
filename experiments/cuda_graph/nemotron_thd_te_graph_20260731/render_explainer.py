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

"""Render an interactive explanation of the packed-THD CUDA Graph changes."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_CONTEXT = EXPERIMENT_DIR / "explainer_context.json"
DEFAULT_PERFORMANCE = RESULTS_DIR / "persistent_bank_scope_sweep_steps11_19.csv"
DEFAULT_TELEMETRY = RESULTS_DIR / "persistent_bank_scope_sweep_telemetry_steps11_19.csv"
DEFAULT_CORRECTNESS = (
    RESULTS_DIR / "persistent_bank_scope_sweep_correctness_steps11_19.csv"
)
DEFAULT_OUTPUT = RESULTS_DIR / "cudagraph_implementation_explainer.html"


@dataclass(frozen=True)
class ScopeEvidence:
    """Measured performance, graph telemetry, and correctness for one scope."""

    scope: str
    job_id: str
    mean_tokens_per_sample: float
    e2e_tps: float
    generation_tps: float
    training_tps: float
    logprob_tps: float
    total_step_time: float
    generation_time: float
    training_time: float
    logprob_time: float
    e2e_speedup_pct: float | None
    training_speedup_pct: float | None
    graph_calls: int
    eligible_calls: int
    capture_count: int
    replay_count: int
    cache_hit_count: int
    cache_miss_count: int
    cache_hit_pct: float | None
    eviction_count: int
    fallback_count: int
    reward: float
    gen_kl_error: float
    policy_kl_error: float
    masked_sequences: int
    nonfinite_count: int
    validation_accuracy_step_20: float


def _read_indexed_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"evidence input is missing: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "Exp" not in reader.fieldnames:
            raise ValueError(f"evidence input must contain an Exp column: {path}")
        indexed: dict[str, dict[str, str]] = {}
        for row in reader:
            scope = row.get("Exp", "").strip()
            if not scope:
                raise ValueError(f"evidence row has an empty Exp value: {path}")
            if scope in indexed:
                raise ValueError(f"duplicate evidence scope {scope!r}: {path}")
            indexed[scope] = row
    if not indexed:
        raise ValueError(f"evidence input has no rows: {path}")
    return indexed


def _float(row: dict[str, str], field: str, *, scope: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{scope}: {field} must be numeric") from error


def _integer(row: dict[str, str], field: str, *, scope: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{scope}: {field} must be an integer") from error
    if value < 0:
        raise ValueError(f"{scope}: {field} must not be negative")
    return value


def _speedup(value: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline throughput must be positive")
    return (value / baseline - 1.0) * 100.0


def load_evidence(
    performance_path: Path,
    telemetry_path: Path,
    correctness_path: Path,
) -> list[ScopeEvidence]:
    """Load and join canonical evidence by experiment scope."""
    performance = _read_indexed_csv(performance_path)
    telemetry = _read_indexed_csv(telemetry_path)
    correctness = _read_indexed_csv(correctness_path)
    scope_sets = (set(performance), set(telemetry), set(correctness))
    if not (scope_sets[0] == scope_sets[1] == scope_sets[2]):
        raise ValueError("performance, telemetry, and correctness scopes must match")
    if "baseline" not in performance:
        raise ValueError("evidence must include a baseline row")

    baseline_e2e = _float(performance["baseline"], "E2E TPS/gpu", scope="baseline")
    baseline_training = _float(
        performance["baseline"],
        "Performance Breakdown - Train TPS/gpu",
        scope="baseline",
    )
    evidence: list[ScopeEvidence] = []
    for scope, performance_row in performance.items():
        telemetry_row = telemetry[scope]
        correctness_row = correctness[scope]
        graph_calls = _integer(telemetry_row, "Graph Calls", scope=scope)
        eligible_calls = _integer(telemetry_row, "Eligible Calls", scope=scope)
        if graph_calls > eligible_calls:
            raise ValueError(
                f"{scope}: graph calls ({graph_calls}) exceed eligible calls "
                f"({eligible_calls})"
            )
        cache_hits = _integer(telemetry_row, "Cache Hits", scope=scope)
        cache_misses = _integer(telemetry_row, "Cache Misses", scope=scope)
        cache_lookups = cache_hits + cache_misses
        e2e_tps = _float(performance_row, "E2E TPS/gpu", scope=scope)
        training_tps = _float(
            performance_row,
            "Performance Breakdown - Train TPS/gpu",
            scope=scope,
        )
        evidence.append(
            ScopeEvidence(
                scope=scope,
                job_id=performance_row.get("Job ID", ""),
                mean_tokens_per_sample=_float(
                    performance_row,
                    "Mean tokens per sample",
                    scope=scope,
                ),
                e2e_tps=e2e_tps,
                generation_tps=_float(
                    performance_row,
                    "Gen TPS/gpu",
                    scope=scope,
                ),
                training_tps=training_tps,
                logprob_tps=_float(
                    performance_row,
                    "Performance Breakdown - Logprob TPS/gpu",
                    scope=scope,
                ),
                total_step_time=_float(
                    performance_row,
                    "Total Step Time",
                    scope=scope,
                ),
                generation_time=_float(
                    performance_row,
                    "Time Breakdown - (Exposed) Generation",
                    scope=scope,
                ),
                training_time=_float(
                    performance_row,
                    "Time Breakdown - Policy Training",
                    scope=scope,
                ),
                logprob_time=_float(
                    performance_row,
                    "Time Breakdown - Policy and Reference Logprobs",
                    scope=scope,
                ),
                e2e_speedup_pct=(
                    None if scope == "baseline" else _speedup(e2e_tps, baseline_e2e)
                ),
                training_speedup_pct=(
                    None
                    if scope == "baseline"
                    else _speedup(training_tps, baseline_training)
                ),
                graph_calls=graph_calls,
                eligible_calls=eligible_calls,
                capture_count=_integer(telemetry_row, "Captures", scope=scope),
                replay_count=_integer(telemetry_row, "Replays", scope=scope),
                cache_hit_count=cache_hits,
                cache_miss_count=cache_misses,
                cache_hit_pct=(
                    None if cache_lookups == 0 else cache_hits / cache_lookups * 100.0
                ),
                eviction_count=_integer(telemetry_row, "Evictions", scope=scope),
                fallback_count=_integer(telemetry_row, "Fallbacks", scope=scope),
                reward=_float(correctness_row, "Reward Mean", scope=scope),
                gen_kl_error=_float(
                    correctness_row,
                    "Gen KL Error Mean",
                    scope=scope,
                ),
                policy_kl_error=_float(
                    correctness_row,
                    "Policy KL Error Mean",
                    scope=scope,
                ),
                masked_sequences=_integer(
                    correctness_row,
                    "Masked Sequences Max",
                    scope=scope,
                ),
                nonfinite_count=_integer(
                    correctness_row,
                    "Nonfinite Count",
                    scope=scope,
                ),
                validation_accuracy_step_20=_float(
                    correctness_row,
                    "Validation Accuracy Step 20",
                    scope=scope,
                ),
            )
        )
    return evidence


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _string_list(context: dict[str, Any], key: str) -> list[str]:
    value = context.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"explainer context field {key!r} must be a list of strings")
    return value


def _format_number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "baseline"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _scope_label(scope: str) -> str:
    labels = {
        "baseline": "CG 없는 baseline",
        "attn": "Attention",
        "attn,mamba": "Attention + Mamba",
        "attn,mamba,moe_router": "Attention + Mamba + router",
    }
    return labels.get(scope, scope.replace(",", " + "))


def _render_status(items: list[str]) -> str:
    if not items:
        return '<p class="muted">기록된 현재 상태가 없습니다.</p>'
    return "".join(
        f'<li><span class="status-dot" aria-hidden="true"></span>{_escape(item)}</li>'
        for item in items
    )


def _render_code_groups(context: dict[str, Any]) -> str:
    groups = context.get("code_groups", [])
    if not isinstance(groups, list):
        raise ValueError("explainer context field 'code_groups' must be a list")
    rendered: list[str] = []
    for index, raw_group in enumerate(groups, start=1):
        if not isinstance(raw_group, dict):
            raise ValueError("every code group must be an object")
        title = _escape(raw_group.get("title", f"Change group {index}"))
        purpose = _escape(raw_group.get("purpose", ""))
        excerpt = _escape(raw_group.get("excerpt", ""))
        files = raw_group.get("files", [])
        if not isinstance(files, list) or not all(
            isinstance(path, str) for path in files
        ):
            raise ValueError("code group files must be a list of paths")
        file_links = "".join(
            f'<a class="file-chip" href="../../../../{_escape(path)}">{_escape(path)}</a>'
            for path in files
        )
        rendered.append(
            f"""
            <article class="code-group">
              <div class="step-number">{index:02d}</div>
              <div class="code-copy">
                <h3>{title}</h3>
                <p>{purpose}</p>
                <div class="file-list">{file_links}</div>
                {f"<pre><code>{excerpt}</code></pre>" if excerpt else ""}
              </div>
            </article>
            """
        )
    return "".join(rendered)


def _render_problems(context: dict[str, Any]) -> str:
    problems = context.get("problems", [])
    if not isinstance(problems, list):
        raise ValueError("explainer context field 'problems' must be a list")
    rendered: list[str] = []
    for raw_problem in problems:
        if not isinstance(raw_problem, dict):
            raise ValueError("every problem must be an object")
        severity = str(raw_problem.get("severity", "measure"))
        if severity not in {"confirmed", "measure", "risk"}:
            raise ValueError(f"unsupported problem severity: {severity}")
        labels = {
            "confirmed": "확인된 문제",
            "measure": "추가 측정 필요",
            "risk": "Correctness 위험",
        }
        rendered.append(
            f"""
            <article class="problem-card {severity}">
              <span class="eyebrow">{labels[severity]}</span>
              <h3>{_escape(raw_problem.get("title", "Unresolved issue"))}</h3>
              <p>{_escape(raw_problem.get("detail", ""))}</p>
              <div class="next-action"><strong>다음 검증</strong>{_escape(raw_problem.get("next", ""))}</div>
            </article>
            """
        )
    return "".join(rendered)


def _render_evidence_rows(evidence: list[ScopeEvidence]) -> str:
    rows: list[str] = []
    for item in evidence:
        coverage = (
            "—"
            if item.eligible_calls == 0
            else f"{item.graph_calls / item.eligible_calls * 100.0:.1f}%"
        )
        hit_rate = "—" if item.cache_hit_pct is None else f"{item.cache_hit_pct:.1f}%"
        rows.append(
            f"""
            <tr>
              <th scope="row">{_escape(_scope_label(item.scope))}<small>job {_escape(item.job_id)}</small></th>
              <td>{_format_number(item.mean_tokens_per_sample, 0)}</td>
              <td>{_format_number(item.e2e_tps)} <span class="delta">{_format_delta(item.e2e_speedup_pct)}</span></td>
              <td>{_format_number(item.training_tps)} <span class="delta">{_format_delta(item.training_speedup_pct)}</span></td>
              <td>{_format_number(item.generation_tps)}</td>
              <td>{_format_number(item.logprob_tps)}</td>
              <td>{coverage}<small>{item.graph_calls:,}/{item.eligible_calls:,} calls</small></td>
              <td>{hit_rate}<small>{item.cache_hit_count} hit / {item.cache_miss_count} miss</small></td>
              <td>{item.eviction_count}</td>
              <td>{item.fallback_count}</td>
            </tr>
            """
        )
    return "".join(rows)


def _render_correctness_rows(evidence: list[ScopeEvidence]) -> str:
    return "".join(
        f"""
        <tr>
          <th scope="row">{_escape(_scope_label(item.scope))}</th>
          <td>{item.reward:.6f}</td>
          <td>{item.gen_kl_error:.6f}</td>
          <td>{item.policy_kl_error:.6f}</td>
          <td>{item.masked_sequences}</td>
          <td>{item.nonfinite_count}</td>
          <td>{item.validation_accuracy_step_20:.6f}</td>
        </tr>
        """
        for item in evidence
    )


def _render_perf_bars(evidence: list[ScopeEvidence]) -> str:
    max_tps = max(item.training_tps for item in evidence)
    return "".join(
        f"""
        <div class="bar-row">
          <div class="bar-label"><strong>{_escape(_scope_label(item.scope))}</strong><span>{item.training_tps:,.0f} train tok/s/GPU</span></div>
          <div class="bar-track"><span style="width:{item.training_tps / max_tps * 100.0:.2f}%"></span></div>
          <div class="bar-value">{_format_delta(item.training_speedup_pct)}</div>
        </div>
        """
        for item in evidence
    )


def _render_quiz(context: dict[str, Any]) -> str:
    questions = context.get("quiz", [])
    if not isinstance(questions, list) or len(questions) != 5:
        raise ValueError("explainer context must contain exactly five quiz questions")
    rendered: list[str] = []
    for question_index, raw_question in enumerate(questions, start=1):
        if not isinstance(raw_question, dict):
            raise ValueError("every quiz question must be an object")
        options = raw_question.get("options", [])
        answer = raw_question.get("answer")
        if (
            not isinstance(options, list)
            or len(options) < 2
            or not all(isinstance(option, str) for option in options)
            or isinstance(answer, bool)
            or not isinstance(answer, int)
            or not 0 <= answer < len(options)
        ):
            raise ValueError(
                f"quiz question {question_index} has invalid options or answer"
            )
        option_markup = "".join(
            f"""
            <label class="quiz-option">
              <input type="radio" name="quiz-{question_index}" value="{option_index}">
              <span>{_escape(option)}</span>
            </label>
            """
            for option_index, option in enumerate(options)
        )
        rendered.append(
            f"""
            <fieldset class="quiz-question" data-answer="{answer}" data-feedback="{_escape(raw_question.get("feedback", ""))}">
              <legend><span>{question_index:02d}</span>{_escape(raw_question.get("question", ""))}</legend>
              <div class="quiz-options">{option_markup}</div>
              <button type="button" class="check-answer">Check answer</button>
              <p class="quiz-feedback" aria-live="polite"></p>
            </fieldset>
            """
        )
    return "".join(rendered)


def render_html(context: dict[str, Any], evidence: list[ScopeEvidence]) -> str:
    """Render one self-contained, responsive implementation explanation."""
    if context.get("schema_version") != 1:
        raise ValueError("explainer context schema_version must be 1")
    if not evidence:
        raise ValueError("explainer evidence must not be empty")
    title = _escape(context.get("title", "CUDA Graph 구현 설명"))
    subtitle = _escape(context.get("subtitle", ""))
    updated = _escape(context.get("updated", datetime.now(UTC).date().isoformat()))
    status = _render_status(_string_list(context, "status"))
    beginner_background = _string_list(context, "beginner_background")
    relevant_background = _string_list(context, "relevant_background")
    background_markup = "".join(
        f"<p>{_escape(item)}</p>" for item in beginner_background
    )
    relevant_markup = "".join(f"<p>{_escape(item)}</p>" for item in relevant_background)
    code_groups = _render_code_groups(context)
    problems = _render_problems(context)
    evidence_rows = _render_evidence_rows(evidence)
    correctness_rows = _render_correctness_rows(evidence)
    bars = _render_perf_bars(evidence)
    quiz = _render_quiz(context)

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="NeMo-RL packed-THD Transformer Engine partial CUDA Graph 지원을 코드와 측정 결과로 설명합니다.">
<title>{title}</title>
<style>
:root {{
  --ink: #17233b;
  --muted: #62708a;
  --paper: #f7f6f1;
  --card: #ffffff;
  --line: #d9dfeb;
  --blue: #2457d6;
  --blue-soft: #eaf0ff;
  --green: #167d5a;
  --green-soft: #e6f5ee;
  --amber: #a76000;
  --amber-soft: #fff2d8;
  --red: #ae3d45;
  --red-soft: #fdebed;
  --shadow: 0 18px 50px rgba(23, 35, 59, 0.08);
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.65; }}
a {{ color: var(--blue); text-underline-offset: 0.16em; }}
a:hover {{ text-decoration-thickness: 2px; }}
button, input {{ font: inherit; }}
.hero {{ position: relative; overflow: hidden; padding: 5.5rem max(1.25rem, calc((100vw - 1180px) / 2)) 4.5rem; background: #13213b; color: white; }}
.hero::after {{ content: ""; position: absolute; width: 34rem; height: 34rem; right: -10rem; top: -18rem; border-radius: 50%; background: radial-gradient(circle, rgba(87, 145, 255, .46), rgba(87, 145, 255, 0)); }}
.kicker, .eyebrow {{ display: inline-block; color: #9dbbff; font-size: .73rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }}
.hero h1 {{ position: relative; max-width: 900px; margin: .7rem 0 1rem; font-family: Georgia, "Times New Roman", serif; font-size: clamp(2.6rem, 7vw, 5.7rem); line-height: .98; letter-spacing: -.05em; }}
.hero p {{ position: relative; max-width: 760px; margin: 0; color: #d7e2f8; font-size: clamp(1.05rem, 2vw, 1.3rem); }}
.hero-meta {{ position: relative; display: flex; flex-wrap: wrap; gap: .7rem; margin-top: 2rem; }}
.hero-meta span, .hero-meta a {{ padding: .42rem .72rem; border: 1px solid rgba(255,255,255,.24); border-radius: 999px; color: #e9f0ff; text-decoration: none; font-size: .83rem; }}
.layout {{ width: min(1180px, calc(100% - 2rem)); margin: 0 auto; display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 3rem; align-items: start; }}
.toc {{ position: sticky; top: 1rem; margin-top: 2rem; padding: 1.1rem; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.82); backdrop-filter: blur(14px); }}
.toc strong {{ display: block; margin-bottom: .55rem; font-size: .77rem; text-transform: uppercase; letter-spacing: .08em; }}
.toc a {{ display: block; padding: .3rem 0; color: var(--muted); text-decoration: none; font-size: .88rem; }}
.toc a:hover {{ color: var(--blue); }}
main {{ min-width: 0; padding: 2rem 0 6rem; }}
section {{ scroll-margin-top: 1rem; padding: 3.4rem 0; border-bottom: 1px solid var(--line); }}
section:last-child {{ border-bottom: 0; }}
.section-label {{ margin: 0 0 .6rem; color: var(--blue); font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }}
h2 {{ margin: 0 0 1rem; font-family: Georgia, "Times New Roman", serif; font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1.08; letter-spacing: -.035em; }}
h3 {{ margin: 0 0 .45rem; font-size: 1.08rem; line-height: 1.35; }}
.lede {{ max-width: 760px; color: var(--muted); font-size: 1.08rem; }}
.status-panel {{ display: grid; grid-template-columns: 1fr auto; gap: 2rem; margin-top: 2rem; padding: 1.35rem; border: 1px solid #b9d9cc; border-radius: 18px; background: var(--green-soft); }}
.status-panel ul {{ display: grid; gap: .65rem; margin: 0; padding: 0; list-style: none; }}
.status-panel li {{ display: flex; gap: .6rem; align-items: baseline; }}
.status-dot {{ flex: 0 0 .48rem; width: .48rem; height: .48rem; border-radius: 50%; background: var(--green); }}
.status-badge {{ align-self: start; padding: .45rem .75rem; border-radius: 999px; background: var(--green); color: white; font-size: .75rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; white-space: nowrap; }}
details.background-deep {{ margin-top: 1.6rem; border: 1px solid var(--line); border-radius: 14px; background: var(--card); }}
details.background-deep summary {{ cursor: pointer; padding: 1rem 1.15rem; font-weight: 750; }}
details.background-deep .details-body {{ padding: 0 1.15rem 1rem; color: #3f4c63; }}
.callout {{ margin: 1.5rem 0; padding: 1rem 1.15rem; border-left: 4px solid var(--blue); border-radius: 0 12px 12px 0; background: var(--blue-soft); }}
.callout strong {{ display: block; margin-bottom: .25rem; color: #183f9f; }}
.flow {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; margin: 2rem 0; }}
.flow-node {{ position: relative; padding: 1rem; min-height: 132px; border: 1px solid var(--line); border-radius: 15px; background: var(--card); box-shadow: var(--shadow); }}
.flow-node:not(:last-child)::after {{ content: "→"; position: absolute; right: -.64rem; top: 48%; z-index: 2; display: grid; place-items: center; width: 1.28rem; height: 1.28rem; border-radius: 50%; background: var(--blue); color: white; font-weight: 800; }}
.flow-node small {{ display: block; margin-bottom: .45rem; color: var(--blue); font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }}
.flow-node code {{ display: block; margin-top: .55rem; color: var(--muted); font-size: .76rem; overflow-wrap: anywhere; }}
.packing-demo {{ display: grid; grid-template-columns: 1fr 1.2fr; gap: 1.2rem; margin: 2rem 0; }}
.diagram-card {{ padding: 1.2rem; border: 1px solid var(--line); border-radius: 16px; background: var(--card); }}
.sequence-list {{ display: grid; gap: .55rem; margin-top: 1rem; }}
.sequence {{ height: 34px; border-radius: 8px; display: flex; align-items: center; padding: 0 .7rem; color: white; font-size: .78rem; font-weight: 700; }}
.sequence.one {{ width: 68%; background: #2457d6; }}
.sequence.two {{ width: 43%; background: #16806b; }}
.sequence.three {{ width: 25%; background: #8b5db7; }}
.packed-row {{ display: flex; height: 48px; margin-top: 1rem; overflow: hidden; border: 2px solid #243a63; border-radius: 9px; }}
.packed-row span {{ display: grid; place-items: center; color: white; font-size: .72rem; font-weight: 800; }}
.packed-row .p1 {{ flex: 7; background: #2457d6; }}
.packed-row .p2 {{ flex: 4; background: #16806b; }}
.packed-row .p3 {{ flex: 3; background: #8b5db7; }}
.packed-row .pad {{ flex: 2; background: repeating-linear-gradient(135deg, #c9cfdb, #c9cfdb 7px, #e7eaf0 7px, #e7eaf0 14px); color: #374151; }}
.timeline {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: .5rem; margin: 2rem 0; }}
.timeline-step {{ padding: .85rem .45rem; border-radius: 12px; background: var(--card); border: 1px solid var(--line); text-align: center; font-size: .78rem; }}
.timeline-step strong {{ display: block; font-size: .9rem; }}
.timeline-step.warm {{ background: var(--amber-soft); border-color: #e4bd75; }}
.timeline-step.capture {{ background: var(--blue-soft); border-color: #9eb8f3; }}
.timeline-step.replay {{ background: var(--green-soft); border-color: #9bcab8; }}
.code-groups {{ display: grid; gap: 1rem; margin-top: 2rem; }}
.code-group {{ display: grid; grid-template-columns: 56px minmax(0, 1fr); gap: 1rem; padding: 1.25rem; border: 1px solid var(--line); border-radius: 16px; background: var(--card); }}
.step-number {{ display: grid; place-items: center; width: 46px; height: 46px; border-radius: 12px; background: var(--ink); color: white; font-weight: 850; }}
.file-list {{ display: flex; flex-wrap: wrap; gap: .4rem; margin: .8rem 0; }}
.file-chip {{ padding: .25rem .52rem; border-radius: 7px; background: var(--blue-soft); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .7rem; text-decoration: none; overflow-wrap: anywhere; }}
pre {{ margin: .85rem 0 0; padding: 1rem; overflow-x: auto; border-radius: 12px; background: #111a2c; color: #e5edff; font: .78rem/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }}
.problem-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin-top: 2rem; }}
.problem-card {{ padding: 1.25rem; border: 1px solid var(--line); border-top: 5px solid var(--amber); border-radius: 14px; background: var(--card); }}
.problem-card .eyebrow {{ color: var(--amber); }}
.problem-card.confirmed {{ border-top-color: var(--blue); }}
.problem-card.confirmed .eyebrow {{ color: var(--blue); }}
.problem-card.risk {{ border-top-color: var(--red); }}
.problem-card.risk .eyebrow {{ color: var(--red); }}
.next-action {{ display: grid; gap: .15rem; margin-top: 1rem; padding-top: .8rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .88rem; }}
.evidence-note {{ display: flex; gap: .8rem; align-items: flex-start; margin: 1.5rem 0; padding: 1rem; border-radius: 12px; background: var(--amber-soft); color: #6e4309; }}
.table-wrap {{ overflow-x: auto; margin: 1rem 0 2rem; border: 1px solid var(--line); border-radius: 14px; background: var(--card); }}
table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
th, td {{ padding: .75rem .7rem; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
thead th {{ background: #e9edf4; color: #415069; font-size: .69rem; letter-spacing: .04em; text-transform: uppercase; }}
tbody th, thead th:first-child {{ text-align: left; }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
td small, th small {{ display: block; color: var(--muted); font-size: .68rem; font-weight: 500; }}
.delta {{ display: inline-block; margin-left: .2rem; color: var(--green); font-size: .68rem; font-weight: 800; }}
.bars {{ display: grid; gap: .75rem; margin: 1.2rem 0 2.3rem; }}
.bar-row {{ display: grid; grid-template-columns: 220px minmax(120px, 1fr) 74px; gap: .8rem; align-items: center; }}
.bar-label {{ display: grid; }}
.bar-label span {{ color: var(--muted); font-size: .72rem; }}
.bar-track {{ height: 14px; overflow: hidden; border-radius: 999px; background: #dfe4ec; }}
.bar-track span {{ display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #2457d6, #16806b); }}
.bar-value {{ color: var(--green); font-size: .78rem; font-weight: 800; text-align: right; }}
.quiz-list {{ display: grid; gap: 1rem; margin-top: 2rem; }}
.quiz-question {{ margin: 0; padding: 1.25rem; border: 1px solid var(--line); border-radius: 16px; background: var(--card); }}
.quiz-question legend {{ padding: 0 .25rem; font-weight: 760; }}
.quiz-question legend span {{ display: inline-grid; place-items: center; width: 32px; height: 32px; margin-right: .6rem; border-radius: 9px; background: var(--ink); color: white; font-size: .72rem; }}
.quiz-options {{ display: grid; gap: .55rem; margin: 1rem 0; }}
.quiz-option {{ display: flex; gap: .65rem; align-items: flex-start; padding: .7rem .8rem; border: 1px solid var(--line); border-radius: 10px; cursor: pointer; }}
.quiz-option:hover {{ border-color: #91a8dc; background: #f5f8ff; }}
.check-answer {{ padding: .55rem .85rem; border: 0; border-radius: 9px; background: var(--blue); color: white; cursor: pointer; font-weight: 750; }}
.check-answer:focus-visible, a:focus-visible, summary:focus-visible, input:focus-visible {{ outline: 3px solid #fdc35b; outline-offset: 3px; }}
.quiz-feedback {{ min-height: 1.5em; margin: .75rem 0 0; font-size: .9rem; }}
.quiz-feedback.correct {{ color: var(--green); }}
.quiz-feedback.incorrect {{ color: var(--red); }}
.muted {{ color: var(--muted); }}
.footer {{ padding: 2rem 1rem 4rem; color: var(--muted); text-align: center; font-size: .82rem; }}
@media (max-width: 900px) {{
  .layout {{ grid-template-columns: 1fr; }}
  .toc {{ position: static; display: flex; gap: .7rem 1rem; flex-wrap: wrap; margin-bottom: 0; }}
  .toc strong {{ width: 100%; margin: 0; }}
  .flow {{ grid-template-columns: 1fr 1fr; }}
  .flow-node:nth-child(2)::after {{ display: none; }}
  .packing-demo, .problem-grid {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 620px) {{
  .hero {{ padding-top: 4rem; }}
  .layout {{ width: min(100% - 1.25rem, 1180px); }}
  section {{ padding: 2.6rem 0; }}
  .status-panel {{ grid-template-columns: 1fr; }}
  .status-badge {{ justify-self: start; }}
  .flow {{ grid-template-columns: 1fr; }}
  .flow-node:not(:last-child)::after {{ content: "↓"; right: auto; left: 50%; top: auto; bottom: -.66rem; }}
  .timeline {{ grid-template-columns: repeat(3, 1fr); }}
  .code-group {{ grid-template-columns: 1fr; }}
  .bar-row {{ grid-template-columns: 1fr 58px; }}
  .bar-track {{ grid-column: 1 / -1; grid-row: 2; }}
}}
</style>
</head>
<body>
<header class="hero">
  <span class="kicker">Implementation explainer · 코드와 측정으로 확인</span>
  <h1>{title}</h1>
  <p>{subtitle}</p>
  <div class="hero-meta">
    <span>업데이트 {updated}</span>
    <span>Warmup: 성공한 optimizer step 3회</span>
    <a href="report.html">전체 실험 ledger 열기</a>
  </div>
</header>
<div class="layout">
  <nav class="toc" aria-label="목차">
    <strong>이 페이지의 내용</strong>
    <a href="#background">배경</a>
    <a href="#intuition">핵심 직관</a>
    <a href="#code">코드 변경</a>
    <a href="#problems">현재 문제</a>
    <a href="#evidence">측정 결과</a>
    <a href="#quiz">퀴즈</a>
  </nav>
  <main>
    <section id="background">
      <p class="section-label">01 · 배경</p>
      <h2>무엇을 다시 재생할 수 있게 만드는가?</h2>
      <p class="lede">CUDA Graph는 안정적인 GPU 실행 경로를 한 번 기록하고 재생하여 반복되는 CPU launch 작업을 줄입니다. 대신 tensor 주소와 shape, 제어 흐름, 분산 schedule geometry가 capture 당시와 호환되어야 합니다.</p>
      <div class="status-panel">
        <ul>{status}</ul>
        <span class="status-badge">20-step smoke 동작 확인</span>
      </div>
      <details class="background-deep">
        <summary>기초 배경: CUDA Graph, partial graph, THD packing</summary>
        <div class="details-body">{background_markup or "<p>CUDA Graph는 반복되는 GPU launch를 기록합니다. Partial graph는 안정적인 model submodule만 capture하고 동적인 작업은 eager로 남겨 둡니다.</p>"}</div>
      </details>
      <div class="flow" role="img" aria-label="NeMo-RL policy training이 Megatron-Core를 거쳐 Transformer Engine partial CUDA Graph로 재생되는 흐름">
        <div class="flow-node"><small>Orchestration</small><strong>NeMo-RL</strong><p>Policy phase를 고르고 sample을 pack하며 metric을 기록합니다.</p><code>policy.train()</code></div>
        <div class="flow-node"><small>Schedule</small><strong>Megatron worker</strong><p>THD geometry를 사전 검증하고 graph bank를 선택합니다.</p><code>ensure_active(schedule_key)</code></div>
        <div class="flow-node"><small>Model</small><strong>Megatron-Core</strong><p>고정된 schedule에서 attention, Mamba, MoE module을 실행합니다.</p><code>forward_backward_func</code></div>
        <div class="flow-node"><small>Replay</small><strong>Transformer Engine</strong><p>선택한 partial module scope를 capture하거나 replay합니다.</p><code>make_graphed_callables</code></div>
      </div>
      <div class="callout"><strong>구분해야 할 경계</strong>Generation에는 vLLM의 별도 CUDA Graph가 있습니다. 이 페이지에서 다루는 것은 Megatron policy-training 경로의 Transformer Engine partial graph입니다. Logprob 시간이 빨라졌다는 사실만으로 logprob 자체가 TE graph를 replay했다고 결론 내릴 수 없습니다.</div>
      <div class="narrow-background">{relevant_markup}</div>
    </section>

    <section id="intuition">
      <p class="section-label">02 · 핵심 직관</p>
      <h2>동적인 packed batch를 소수의 안정적인 schedule로 바꿉니다.</h2>
      <p class="lede">Sequence packing은 논리적으로 동적이지만, replay에는 물리 tensor가 합의된 capacity에 맞으면 됩니다. 구현은 token storage와 THD metadata를 고정된 한도까지 padding하고 schedule key별 graph bank를 cache합니다.</p>
      <div class="packing-demo">
        <div class="diagram-card">
          <h3>논리 sample</h3>
          <p class="muted">세 sequence의 길이가 서로 다릅니다.</p>
          <div class="sequence-list">
            <div class="sequence one">S1 · 7 tokens</div>
            <div class="sequence two">S2 · 4 tokens</div>
            <div class="sequence three">S3 · 3 tokens</div>
          </div>
        </div>
        <div class="diagram-card">
          <h3>고정된 replay storage</h3>
          <p class="muted">논리 token 14개가 capacity 16칸을 사용하며, cu_seqlens와 sequence slot도 같은 규칙으로 padding됩니다.</p>
          <div class="packed-row"><span class="p1">S1</span><span class="p2">S2</span><span class="p3">S3</span><span class="pad">pad</span></div>
          <p><code>cu_seqlens = [0, 7, 11, 14, 16]</code></p>
        </div>
      </div>
      <div class="timeline" aria-label="세 번의 warmup 이후 capture와 replay가 이어지는 과정">
        <div class="timeline-step warm"><strong>Step 1</strong>eager warmup</div>
        <div class="timeline-step warm"><strong>Step 2</strong>eager warmup</div>
        <div class="timeline-step warm"><strong>Step 3</strong>eager warmup</div>
        <div class="timeline-step capture"><strong>Step 4</strong>capture miss</div>
        <div class="timeline-step replay"><strong>Logprob</strong>bank 유지</div>
        <div class="timeline-step replay"><strong>Step 5</strong>replay hit</div>
      </div>
      <div class="callout"><strong>Coverage와 hit rate는 다릅니다.</strong>Graph-call coverage 100%는 graph 대상이었던 module call이 모두 graph를 사용했다는 뜻입니다. 필요한 schedule을 매번 새로 capture한 뒤 그 step 안에서 graph를 사용하면, coverage는 100%이면서 bank hit rate는 낮을 수 있습니다.</div>
    </section>

    <section id="code">
      <p class="section-label">03 · 코드</p>
      <h2>이번 변화는 단일 flag가 아니라 lifecycle입니다.</h2>
      <p class="lede">구현 범위는 configuration, packed data geometry, 분산 worker state, Transformer Engine storage, nested Megatron scope 지원, observability까지 이어집니다. 이 순서로 보면 각 계층이 다음 계층에 어떤 invariant를 전달하는지 알 수 있습니다.</p>
      <div class="code-groups">{code_groups}</div>
    </section>

    <section id="problems">
      <p class="section-label">04 · 현재 문제</p>
      <h2>기능적으로 동작하는 것과 완성된 것은 다릅니다.</h2>
      <p class="lede">최신 smoke run에서는 선택한 scope들이 fatal error 없이 capture와 replay를 수행했습니다. 이제 남은 질문은 cache 효율, memory 비용, phase별 성능 귀속, convergence 수준의 correctness입니다.</p>
      <div class="problem-grid">{problems}</div>
    </section>

    <section id="evidence">
      <p class="section-label">05 · 측정 결과</p>
      <h2>Nano의 step 11–19 측정 결과입니다.</h2>
      <p class="lede">모든 row는 같은 24-GPU performance recipe, nightly image, all-to-all dispatcher, sequence packing, warmup 3회, checkpoint 비활성화를 사용합니다. 아래 값은 HTML 생성 시 canonical CSV에서 계산됩니다.</p>
      <div class="evidence-note"><strong>비교 시 주의점</strong><span>Scope마다 sample당 평균 token 수가 다릅니다. 따라서 throughput을 우선 비교해야 하며, total step time만 보면 구현 비용과 실제 처리량 차이가 섞입니다.</span></div>
      <h3>Policy-training throughput</h3>
      <div class="bars">{bars}</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Scope</th><th>평균 token / sample</th><th>E2E tok/s/GPU</th><th>Train tok/s/GPU</th><th>Generation tok/s/GPU</th><th>Logprob tok/s/GPU</th><th>CG coverage</th><th>Bank hit rate</th><th>Eviction</th><th>Fallback</th></tr></thead>
          <tbody>{evidence_rows}</tbody>
        </table>
      </div>
      <h3>20-step correctness smoke</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Scope</th><th>Reward 평균</th><th>Gen KL 평균</th><th>Policy KL 평균</th><th>Masked seq 최대</th><th>Nonfinite</th><th>Validation accuracy · step 20</th></tr></thead>
          <tbody>{correctness_rows}</tbody>
        </table>
      </div>
      <div class="callout"><strong>현재까지 확인된 범위</strong>20-step run에서 NaN/Inf, masked sequence, graph fallback은 관찰되지 않았고 matched 100-step baseline·attention job도 정상 종료했습니다. 다만 job 성공과 독립 trajectory의 aggregate metric만으로 numerical parity를 증명할 수 없습니다. 100-step metric 분석과 고정 input의 output·gradient·parameter-delta parity가 다음 correctness gate입니다.</div>
    </section>

    <section id="quiz">
      <p class="section-label">06 · 퀴즈</p>
      <h2>이해한 내용을 확인합니다.</h2>
      <p class="lede">Graph coverage, cache 동작, packed geometry, correctness evidence의 차이를 묻습니다. 답을 고른 뒤 바로 설명을 확인할 수 있습니다.</p>
      <div class="quiz-list">{quiz}</div>
    </section>
  </main>
</div>
<footer class="footer">Versioned explanation context와 측정 CSV에서 생성했습니다. 코드나 결과가 바뀌면 입력을 갱신하고 renderer를 다시 실행합니다.</footer>
<script>
document.querySelectorAll('.quiz-question').forEach((question) => {{
  const button = question.querySelector('.check-answer');
  const feedback = question.querySelector('.quiz-feedback');
  button.addEventListener('click', () => {{
    const selected = question.querySelector('input:checked');
    feedback.classList.remove('correct', 'incorrect');
    if (!selected) {{
      feedback.textContent = '먼저 답을 선택하세요.';
      feedback.classList.add('incorrect');
      return;
    }}
    const correct = Number(selected.value) === Number(question.dataset.answer);
    feedback.textContent = (correct ? '정답입니다. ' : '다시 생각해 보세요. ') + question.dataset.feedback;
    feedback.classList.add(correct ? 'correct' : 'incorrect');
  }});
}});
</script>
</body>
</html>
"""


def read_context(path: Path) -> dict[str, Any]:
    """Read the versioned explainer context from JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"explainer context is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("explainer context must be a JSON object")
    return payload


def write_html(document: str, output: Path) -> None:
    """Atomically write one generated HTML document."""
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized_document = "\n".join(line.rstrip() for line in document.splitlines())
    if document.endswith(("\n", "\r")):
        normalized_document += "\n"
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
            temporary.write(normalized_document)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
        output.chmod(0o644)
    except BaseException:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise


def render_from_paths(
    *,
    context_path: Path,
    performance_path: Path,
    telemetry_path: Path,
    correctness_path: Path,
    output_path: Path,
) -> dict[str, int | str]:
    """Read canonical inputs, render the explainer, and return a summary."""
    context = read_context(context_path)
    evidence = load_evidence(performance_path, telemetry_path, correctness_path)
    document = render_html(context, evidence)
    write_html(document, output_path)
    quiz = context.get("quiz", [])
    return {
        "evidence_rows": len(evidence),
        "quiz_questions": len(quiz) if isinstance(quiz, list) else 0,
        "output": str(output_path),
    }


def main() -> None:
    """Render the canonical explainer or explicitly supplied inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--performance", type=Path, default=DEFAULT_PERFORMANCE)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--correctness", type=Path, default=DEFAULT_CORRECTNESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = render_from_paths(
        context_path=args.context,
        performance_path=args.performance,
        telemetry_path=args.telemetry,
        correctness_path=args.correctness,
        output_path=args.output,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
