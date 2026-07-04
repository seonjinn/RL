#!/usr/bin/env python3
"""Normalize and render vLLM 0.24 native profile benchmark results."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


PROFILE_ORDER = ["Native 32K", "YaRN 64K", "YaRN total-128K"]
METHOD_ORDER = {
    "baseline": 0,
    "dflash": 1,
    "pard": 2,
    "pard2": 3,
    "suffix": 4,
}
MATCH_KEYS = [
    "runtime",
    "model",
    "domain",
    "temperature",
    "top_p",
    "batch_size",
    "isl",
    "osl",
    "context_profile",
    "position_encoding",
    "cuda_graph",
    "setup",
]
CANONICAL_COLUMNS = [
    "source_status",
    "runtime",
    "domain",
    "model",
    "temperature",
    "top_p",
    "batch_size",
    "isl",
    "osl",
    "context_profile",
    "position_encoding",
    "cuda_graph",
    "setup",
    "attention_backend",
    "method",
    "k",
    "tok_s_gpu",
    "latency_s",
    "acceptance_rate",
    "mean_accept_len",
    "job_id",
    "source",
]


def _source_status(status: object) -> str:
    return "complete" if str(status).lower() == "complete" else "partial"


def _runtime_label(runtime: object) -> str:
    if isinstance(runtime, dict):
        version = runtime.get("vllm_version")
        if version:
            return f"vLLM {version}"
    return str(runtime)


def _domain(config: dict[str, Any], source: Path) -> str:
    text = " ".join(
        [
            str(config.get("prompt_jsonl", "")),
            str(config.get("tag", "")),
            str(source),
        ]
    ).lower()
    if "swe" in text:
        return "SWE"
    if "math" in text:
        return "Math"
    return "Unknown"


def _model_name(model_path: object) -> str:
    value = str(model_path).lower()
    if "qwen3-8b" in value:
        return "Qwen3-8B"
    return Path(str(model_path)).name or str(model_path)


def _position_encoding(model_path: object) -> str:
    return "yarn4" if "yarn4" in str(model_path).lower() else "native"


def _context_profile(isl: int, osl: int, position_encoding: str) -> str:
    if isl == 4096 and osl == 32768 and position_encoding == "native":
        return "Native 32K"
    if isl == 4096 and osl == 65536 and position_encoding == "yarn4":
        return "YaRN 64K"
    if isl == 4096 and osl == 126976 and position_encoding == "yarn4":
        return "YaRN total-128K"
    return f"ISL {isl} / OSL {osl}"


def _job_id(payload: dict[str, Any], source: Path) -> str:
    runtime = payload.get("runtime")
    if isinstance(runtime, dict):
        environment = runtime.get("environment")
        if isinstance(environment, dict):
            job_id = environment.get("SLURM_JOB_ID")
            if job_id:
                return str(job_id)
    matches = re.findall(r"(?<!\d)(\d{7})(?!\d)", str(source))
    return matches[-1] if matches else ""


def _method(config: dict[str, Any]) -> str:
    return str(config.get("mode", "baseline")).lower()


def _k(config: dict[str, Any]) -> float:
    speculative = config.get("speculative_config")
    if not isinstance(speculative, dict):
        return math.nan
    value = speculative.get("num_speculative_tokens")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _setup_signature(config: dict[str, Any]) -> str:
    keys = [
        "attention_backend",
        "dtype",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "enforce_eager",
        "engine_gpus",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "moe_backend",
        "pipeline_parallel_size",
        "pp",
        "tensor_parallel_size",
        "total_gpus",
        "tp",
    ]
    setup = {key: config.get(key) for key in keys}
    return json.dumps(setup, sort_keys=True, separators=(",", ":"))


def _acceptance_metric(batch_result: dict[str, Any], key: str) -> float:
    metrics = batch_result.get("spec_decode_metrics")
    if not isinstance(metrics, dict):
        return math.nan
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _latency_s(batch_result: dict[str, Any]) -> float:
    for key in ("mean_latency_s", "latency_s_mean", "latency_s"):
        value = batch_result.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return math.nan


def _tok_s_gpu(batch_result: dict[str, Any]) -> float:
    value = batch_result.get("output_tok_s_per_gpu")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _normalize_batch_result(
    payload: dict[str, Any], config: dict[str, Any], batch_result: dict[str, Any], source: Path
) -> dict[str, object]:
    model_path = config.get("model", "")
    isl = int(config.get("isl", 0))
    osl = int(config.get("osl", 0))
    position_encoding = _position_encoding(model_path)
    return {
        "source_status": _source_status(payload.get("status")),
        "runtime": _runtime_label(payload.get("runtime")),
        "domain": _domain(config, source),
        "model": _model_name(model_path),
        "temperature": float(config.get("temperature", math.nan)),
        "top_p": float(config.get("top_p", math.nan)),
        "batch_size": int(batch_result.get("bs", 0)),
        "isl": isl,
        "osl": osl,
        "context_profile": _context_profile(isl, osl, position_encoding),
        "position_encoding": position_encoding,
        "cuda_graph": str(config.get("cudagraph_mode", "")),
        "setup": _setup_signature(config),
        "attention_backend": str(config.get("attention_backend", "")),
        "method": _method(config),
        "k": _k(config),
        "tok_s_gpu": _tok_s_gpu(batch_result),
        "latency_s": _latency_s(batch_result),
        "acceptance_rate": _acceptance_metric(batch_result, "acceptance_rate"),
        "mean_accept_len": _acceptance_metric(batch_result, "mean_acceptance_length"),
        "job_id": _job_id(payload, source),
        "source": str(source),
    }


def load_profile_results(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(path) for path in paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config")
        results = payload.get("results")
        if not isinstance(config, dict) or not isinstance(results, list):
            continue
        for batch_result in results:
            if isinstance(batch_result, dict):
                rows.append(_normalize_batch_result(payload, config, batch_result, path))
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _speedup_label(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "waiting matched baseline"
    return f"{number:.2f}x" if math.isfinite(number) else "waiting matched baseline"


def match_profile_baselines(rows: pd.DataFrame) -> pd.DataFrame:
    matched = rows.copy()
    if matched.empty:
        matched["baseline_tok_s_gpu"] = pd.Series(dtype=float)
        matched["baseline_latency_s"] = pd.Series(dtype=float)
        matched["throughput_speedup"] = pd.Series(dtype=float)
        matched["latency_speedup"] = pd.Series(dtype=float)
        matched["throughput_speedup_label"] = pd.Series(dtype=str)
        matched["latency_speedup_label"] = pd.Series(dtype=str)
        return matched

    baseline_rows = matched.loc[
        matched["method"].eq("baseline")
        & matched["runtime"].astype(str).str.startswith("vLLM")
    ].copy()
    baseline_rows = baseline_rows[MATCH_KEYS + ["tok_s_gpu", "latency_s"]].rename(
        columns={
            "tok_s_gpu": "baseline_tok_s_gpu",
            "latency_s": "baseline_latency_s",
        }
    )

    matched = matched.merge(baseline_rows, on=MATCH_KEYS, how="left")
    matched["throughput_speedup"] = matched["tok_s_gpu"] / matched["baseline_tok_s_gpu"]
    matched["latency_speedup"] = matched["baseline_latency_s"] / matched["latency_s"]
    matched["throughput_speedup_label"] = matched["throughput_speedup"].map(_speedup_label)
    matched["latency_speedup_label"] = matched["latency_speedup"].map(_speedup_label)
    return matched


def _fmt_number(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _source_badge(source_status: object) -> str:
    status = str(source_status)
    style = (
        "display:inline-block;padding:0.15rem 0.45rem;border-radius:999px;"
        "font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0;"
    )
    if status == "complete":
        style += "background:#153a2a;color:#9fe3b5;border:1px solid #29543f;"
    else:
        style += "background:#4b3411;color:#ffd98a;border:1px solid #7a5620;"
    return f'<span class="source-status {escape(status)}" style="{style}">{escape(status)}</span>'


def _display_source(source: object) -> str:
    parts = Path(str(source)).parts
    if len(parts) <= 6:
        return str(source)
    return "/".join(parts[-6:])


def _profile_table(rows: pd.DataFrame, profile: str) -> str:
    profile_rows = rows.loc[rows["context_profile"].eq(profile)].copy()
    if profile_rows.empty:
        return ""
    profile_rows["method_rank"] = profile_rows["method"].map(METHOD_ORDER).fillna(999)
    profile_rows = profile_rows.sort_values(
        ["domain", "temperature", "method_rank", "batch_size", "source_status", "job_id"],
        kind="stable",
    )
    body: list[str] = []
    for row in profile_rows.itertuples(index=False):
        method_label = "Baseline" if row.method == "baseline" else f"{str(row.method).upper()} K={int(row.k)}"
        if row.method == "pard2":
            method_label = f"PARD-2 K={int(row.k)}"
        elif row.method == "pard":
            method_label = f"PARD K={int(row.k)}"
        elif row.method == "dflash":
            method_label = f"DFlash K={int(row.k)}"
        elif row.method == "suffix":
            method_label = f"Suffix K={int(row.k)}"

        source_cell = (
            f"{_source_badge(row.source_status)}<br>"
            f'<span class="source-path">{escape(_display_source(row.source))}</span>'
        )
        cells = [
            escape(str(row.domain)),
            f'<span class="num">{_fmt_number(row.temperature, 1)}</span>',
            f'<span class="num">{int(row.isl):,}</span>',
            f'<span class="num">{int(row.osl):,}</span>',
            f'<span class="num">{int(row.batch_size)}</span>',
            escape(method_label),
            f'<span class="num">{_fmt_number(row.tok_s_gpu)}</span>',
            escape(str(row.throughput_speedup_label)),
            escape(str(row.latency_speedup_label)),
            f'<span class="num">{_fmt_number(float(row.acceptance_rate) * 100)}%</span>',
            f'<span class="num">{_fmt_number(row.mean_accept_len)}</span>',
            source_cell,
        ]
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")

    headings = [
        "Domain",
        "Temperature",
        "ISL",
        "OSL",
        "Batch",
        "Method / K",
        "tok/s/GPU",
        "Throughput speedup",
        "Latency speedup",
        "Acceptance",
        "Mean accept length",
        "Source",
    ]
    header = "".join(f"<th>{escape(heading)}</th>" for heading in headings)
    return (
        f"<h3>{escape(profile)}</h3>"
        '<div class="table-wrap"><table>'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def render_profile_section(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""
    target_rows = rows.loc[rows["context_profile"].isin(PROFILE_ORDER)].copy()
    if target_rows.empty:
        return ""
    tables = "".join(_profile_table(target_rows, profile) for profile in PROFILE_ORDER)
    if not tables:
        return ""
    return (
        '<section class="section" id="vllm024-profile">'
        "<h2>vLLM 0.24 / Native Profile Results</h2>"
        '<p class="note">Qwen3-8B on Lyris GB200, matched only against exact '
        "vLLM-native baselines on runtime, model, domain, temperature, top-p, batch, "
        "ISL, OSL, context profile, position encoding, CUDA graph mode, and normalized "
        "setup. Persisted batch results from interrupted sources are shown as partial.</p>"
        f"{tables}</section>"
    )
