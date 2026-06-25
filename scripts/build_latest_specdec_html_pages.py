#!/usr/bin/env python3
"""Build latest SpecDec benchmark HTML pages from refreshed CSV artifacts."""

from __future__ import annotations

import datetime as dt
import html
import math
import re
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

MAIN_VLLM = DOCS / "vllm_standalone_all_batches_combined_20260619.csv"
VLLM_LIVE_SOURCES = [
    (
        DOCS / "oci_qmath_extra_k_live_log_metrics_20260620.csv",
        "OCI Math extra-K sweep, refreshed 2026-06-21",
        20,
    ),
    (
        DOCS / "oci_qmath_pard2_k_sweep_live_log_metrics_20260620.csv",
        "OCI Math PARD-2 K sweep, refreshed 2026-06-21",
        10,
    ),
    (
        DOCS / "oci_qmath_pard_pard2_k16_focus_live_log_metrics_20260620.csv",
        "OCI Math Qwen32 PARD/PARD-2 K16 retry, refreshed 2026-06-21",
        30,
    ),
    (
        DOCS / "lyris_qwen235b_swe_pard2_k_sweep_live_log_metrics_20260620.csv",
        "Lyris SWE Qwen235B PARD-2 K sweep",
        15,
    ),
]
DFLASH = DOCS / "qwen3_235b_dflash_retry28_openmath_metrics.csv"
VLLM_LEGACY_NORMALIZED = DOCS / "vllm_standalone_qwen30_qwen8_legacy_breakdowns_20260625.csv"
VLLM_TEMP_TRENDS = DOCS / "vllm_standalone_temp0_temp1_trends_20260616.csv"
VLLM_ADDED_OUT = DOCS / "vllm_standalone_added_results_latest.csv"
VLLM_HTML_LATEST = DOCS / "vllm_standalone_results_latest.html"
VLLM_HTML_DATED = DOCS / "vllm_standalone_results_20260621.html"

NEMORL_MANIFESTS = (
    sorted(ROOT.glob("latest_lyris_nemorl_qwen235b_*20260621_jobs.csv"))
    + sorted(DOCS.glob("latest_lyris_nemorl_perfcfg_*wandb_20260622_jobs.csv"))
    + sorted(DOCS.glob("latest_lyris_nemorl_*20260623_jobs.csv"))
    + sorted(DOCS.glob("latest_lyris_nemorl_*20260624_jobs.csv"))
    + sorted(DOCS.glob("latest_lyris_nemorl_*20260625_jobs.csv"))
)
NEMORL_SUMMARY = DOCS / "lyris_qwen235b_pr2879_live_summary_skip_step1_20260621.csv"
NEMORL_ADDITIONAL_SUMMARIES = [
    DOCS / "lyris_20260623_current_plus_eagerfalse_summary_skip_step1.csv",
    DOCS / "qwen32_pardk1_20260624_summary_skip1_latest.csv",
]
NEMORL_COMPARISON_SUMMARIES = [
    DOCS / "qwen32_pard_eagerfalse_compare_20260624.csv",
    DOCS / "nemorl_specdec_slowdown_watchlist_20260624.csv",
]
NEMORL_SACCT = DOCS / "lyris_qwen235b_pr2879_sacct_20260621.psv"
NEMORL_OUT = DOCS / "lyris_qwen235b_pr2879_live_enriched_20260621.csv"
NEMORL_LYRIS_HISTORICAL_SOURCES = [
    (
        DOCS / "lyris_nemorl_qwen30_qwen32_pr2879_step20_speedups_20260622.csv",
        "Lyris Qwen30/Qwen32 PerfCfg OSL4096 latest-main+PR2879 2026-06-22",
        "performance recipe default plus latest-main+PR2879 topology-aware fix, temp=1.0/top_p=1.0, step>=2 summary",
        1,
    ),
    (
        DOCS / "lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv",
        "Lyris Qwen30/Qwen32 PerfCfg OSL4096 2026-06-18",
        "performance recipe default, temp=1.0/top_p=1.0, step>=2 live summary",
        2,
    ),
]
NEMORL_OCI_HISTORICAL = DOCS / "nemorl_integrated_specdec_results_clean_20260617.csv"
NEMORL_LIVE_K_SWEEP_SUMMARY = DOCS / "lyris_nemorl_qwen30_qwen32_eagle3_k_sweep_live_summary_20260622.csv"
NEMORL_LIVE_K_SWEEP_SOURCE_GROUP = "Lyris Qwen30/Qwen32 PerfCfg OSL4096 enforce_eager=true K sweep 2026-06-22"
NEMORL_LIVE_K_SWEEP_CHECKED_AT = "2026-06-22 21:31 PDT"
NEMORL_COMBINED_OUT = DOCS / "lyris_nemorl_perfcfg_specdec_combined_latest.csv"
NEMORL_HTML = DOCS / "lyris_nemorl_perfcfg_specdec_live_status_latest.html"
NEMORL_HTML_DATED = DOCS / "lyris_nemorl_perfcfg_specdec_live_status_20260622.html"
WANDB_ENTITY = "nvidia"

NEMORL_LIVE_K_SWEEP_META = [
    {
        "job_id": "2177867",
        "model": "qwen30ba3b",
        "mode": "sync",
        "k": 5,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "FAILED",
        "elapsed": "00:08:00",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "failed_before_completed_step",
        "notes": "Old pre-fix K5 sync attempt; CUBLAS GEMM failure at step 1.",
        "error": "RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k5_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_sync_eagle3/2177867-logs/ray-driver.log",
    },
    {
        "job_id": "2177868",
        "model": "qwen30ba3b",
        "mode": "async-1off",
        "k": 5,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "TIMEOUT",
        "elapsed": "05:00:19",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "engine_error_timeout",
        "notes": "Timed out after vLLM EngineCore error; not clean performance data.",
        "error": "RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasSetStream",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k5_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_async1off_eagle3/2177868-logs/ray-driver.log",
    },
    {
        "job_id": "2177869",
        "model": "qwen32",
        "mode": "sync",
        "k": 5,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "COMPLETED",
        "elapsed": "02:21:17",
        "metric_state": "parsed_completed",
        "notes": "Completed 20/20; log confirms enforce_eager=true.",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k5_contextclamp_step20_recipe_osl_temp1/logs/qwen32_sync_eagle3/2177869-logs/ray-driver.log",
    },
    {
        "job_id": "2177870",
        "model": "qwen32",
        "mode": "async-1off",
        "k": 5,
        "nodes_x_gpus": "8x4",
        "segment": 8,
        "slurm_state": "TIMEOUT",
        "elapsed": "05:00:01",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "engine_error_timeout",
        "notes": "Timed out after vLLM EngineCore error; not clean performance data.",
        "error": "RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasGemmEx",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k5_contextclamp_step20_recipe_osl_temp1/logs/qwen32_async1off_eagle3/2177870-logs/ray-driver.log",
    },
    {
        "job_id": "2177871",
        "model": "qwen30ba3b",
        "mode": "sync",
        "k": 7,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "FAILED",
        "elapsed": "00:07:46",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "failed_before_completed_step",
        "notes": "Old pre-fix K7 sync attempt; Triton device-side assert at step 1.",
        "error": "RuntimeError: Triton Error [CUDA]: device-side assert triggered",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k7_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_sync_eagle3/2177871-logs/ray-driver.log",
    },
    {
        "job_id": "2177872",
        "model": "qwen30ba3b",
        "mode": "async-1off",
        "k": 7,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "TIMEOUT",
        "elapsed": "05:00:12",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "engine_error_timeout",
        "notes": "Timed out after vLLM EngineCore error; not clean performance data.",
        "error": "RuntimeError: Triton Error [CUDA]: device-side assert triggered",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k7_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_async1off_eagle3/2177872-logs/ray-driver.log",
    },
    {
        "job_id": "2177873",
        "model": "qwen32",
        "mode": "sync",
        "k": 7,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "COMPLETED",
        "elapsed": "02:26:16",
        "metric_state": "parsed_completed",
        "notes": "Completed 20/20; log confirms enforce_eager=true.",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k7_contextclamp_step20_recipe_osl_temp1/logs/qwen32_sync_eagle3/2177873-logs/ray-driver.log",
    },
    {
        "job_id": "2177874",
        "model": "qwen32",
        "mode": "async-1off",
        "k": 7,
        "nodes_x_gpus": "8x4",
        "segment": 8,
        "slurm_state": "TIMEOUT",
        "elapsed": "05:00:01",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "engine_error_timeout",
        "notes": "Timed out after vLLM EngineCore traceback; not clean performance data.",
        "error": "EngineCore traceback at step 1",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k7_contextclamp_step20_recipe_osl_temp1/logs/qwen32_async1off_eagle3/2177874-logs/ray-driver.log",
    },
    {
        "job_id": "2177875",
        "model": "qwen30ba3b",
        "mode": "sync",
        "k": 9,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "COMPLETED",
        "elapsed": "01:45:29",
        "metric_state": "parsed_completed",
        "notes": "Completed 20/20; log confirms enforce_eager=true.",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k9_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_sync_eagle3/2177875-logs/ray-driver.log",
    },
    {
        "job_id": "2177876",
        "model": "qwen30ba3b",
        "mode": "async-1off",
        "k": 9,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "COMPLETED",
        "elapsed": "01:59:05",
        "metric_state": "parsed_completed_with_shutdown_warning",
        "notes": "Completed 20/20 with enforce_eager=true; async log lacks generation-time breakdown, so throughput rows are cleaner than time-speedup rows.",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k9_contextclamp_step20_recipe_osl_temp1/logs/qwen30ba3b_async1off_eagle3/2177876-logs/ray-driver.log",
    },
    {
        "job_id": "2177877",
        "model": "qwen32",
        "mode": "sync",
        "k": 9,
        "nodes_x_gpus": "4x4",
        "segment": 4,
        "slurm_state": "COMPLETED",
        "elapsed": "02:39:13",
        "metric_state": "parsed_completed",
        "notes": "Completed 20/20; log confirms enforce_eager=true.",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k9_contextclamp_step20_recipe_osl_temp1/logs/qwen32_sync_eagle3/2177877-logs/ray-driver.log",
    },
    {
        "job_id": "2177878",
        "model": "qwen32",
        "mode": "async-1off",
        "k": 9,
        "nodes_x_gpus": "8x4",
        "segment": 8,
        "slurm_state": "TIMEOUT",
        "elapsed": "05:00:12",
        "completed_steps": 0,
        "last_step": 1,
        "metric_state": "engine_error_timeout",
        "notes": "Timed out after vLLM EngineCore error; not clean performance data.",
        "error": "RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasSetStream",
        "log_path": "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260622_lyris_nemorl_qwen30_qwen32_eagle3k9_contextclamp_step20_recipe_osl_temp1/logs/qwen32_async1off_eagle3/2177878-logs/ray-driver.log",
    },
]


MODEL_MAP = {
    "qwen235b": "Qwen3-235B-A22B",
    "qwen30ba3b": "Qwen3-30B-A3B",
    "qwen30": "Qwen3-30B-A3B",
    "qwen32": "Qwen3-32B",
    "qwen8": "Qwen3-8B",
}

PALETTE = {
    "baseline": "#6b7280",
    "baseline_fuselossfalse": "#b8b8b8",
    "eagle3_k3": "#1f78b4",
    "eagle3_k5": "#a6cee3",
    "eagle3_k7": "#6a3d9a",
    "eagle3_k9": "#cab2d6",
    "eagle3_k8": "#2563eb",
    "pard_k1_tp1": "#ff7f00",
    "pard_k1_tp2": "#fdbf6f",
    "pard_k5": "#e31a1c",
    "pard_k8": "#fb9a99",
    "pard_k12": "#b15928",
    "pard_k16": "#8b1a1a",
    "pard2": "#33a02c",
    "pard2_8b": "#b2df8a",
    "pard2_14b": "#ffff99",
    "pard2_k16": "#1b9e77",
    "pard2_k11": "#66a61e",
    "pard2_k9": "#d95f02",
    "pard2_k5": "#7570b3",
    "pard2_k3": "#e7298a",
    "pard2_k1": "#a6761d",
    "suffix_k32": "#17a398",
    "temp0": "#2563eb",
    "temp1": "#dc2626",
}

METRIC_PALETTE = {
    "Generation throughput": "#1f78b4",
    "E2E throughput": "#33a02c",
    "Generation time": "#fb9a99",
    "E2E step time": "#e31a1c",
}


WANDB_URL_RE = re.compile(r"https?://wandb\.ai/[^\s\x1b\"'<>]+")


def short_model(value: object) -> str:
    text = str(value)
    replacements = {
        "Qwen3-235B-A22B": "235B",
        "Qwen3-30B-A3B": "30B-A3B",
        "Qwen3-32B": "32B",
        "Qwen3-8B": "8B",
    }
    return replacements.get(text, text.replace("Qwen3-", ""))


def method_label(value: object) -> str:
    text = str(value)
    match = re.fullmatch(r"([a-z0-9]+)_k(\d+)", text)
    if not match:
        return text.replace("_", " ")
    base, k = match.groups()
    names = {
        "eagle3": "Eagle-3",
        "pard": "PARD",
        "pard2": "PARD-2",
        "suffix": "Suffix",
        "dflash": "DFlash",
    }
    return f"{names.get(base, base)} K{k}"


def nemorl_method_label(value: object) -> str:
    text = str(value)
    if text == "baseline":
        return "Baseline"
    if text == "baseline_fuselossfalse":
        return "Baseline fuse_loss=false"
    if text == "pard2":
        return "PARD-2"
    if text == "pard2_8b":
        return "PARD-2 8B"
    if text == "pard2_14b":
        return "PARD-2 14B"
    if text == "pard_k1_tp1":
        return "PARD K1 TP1"
    if text == "pard_k1_tp2":
        return "PARD K1 TP2"
    return method_label(text)


def chart_value(value: object, metric: str) -> str:
    if metric == "speedup":
        return fmt(value, 2, "x")
    if metric == "acceptance_pct":
        return fmt(value, 0, "%")
    return fmt(value, 2)


def chart_tick(value: float, metric: str) -> str:
    if metric == "speedup":
        return f"{value:.1f}x"
    if metric == "acceptance_pct":
        return f"{value:.0f}%"
    if metric == "mean_accept_len":
        return f"{value:.1f}"
    return f"{value:.1f}"


def chart_y_max(max_value: float, metric: str) -> float:
    if metric == "speedup":
        return max(1.1, max_value * 1.22)
    if metric == "acceptance_pct":
        return max(10.0, max_value * 1.22)
    if metric == "mean_accept_len":
        return max(1.0, max_value * 1.22)
    return max(1.0, max_value * 1.22)


def legend_svg(methods: list[str], x: float, y: float, gap: float = 116) -> str:
    width = max(0, (len(methods) - 1) * gap)
    start = x - width / 2
    chunks = []
    for idx, method in enumerate(methods):
        lx = start + idx * gap
        color = PALETTE.get(method, "#4b5563")
        chunks.append(
            f'<rect x="{lx:.1f}" y="{y - 8:.1f}" width="14" height="14" rx="2" fill="{color}"/>'
            f'<text x="{lx + 20:.1f}" y="{y + 3:.1f}" font-size="13" fill="#374151">{esc(method_label(method))}</text>'
        )
    return "".join(chunks)


def grouped_bar_svg(rows: pd.DataFrame, title: str, metric: str, methods: list[str]) -> str:
    if rows.empty:
        return ""
    models = [m for m in ["Qwen3-235B-A22B", "Qwen3-30B-A3B", "Qwen3-32B", "Qwen3-8B"] if m in set(rows["model"])]
    if not models:
        models = sorted(rows["model"].dropna().astype(str).unique())
    rows = rows[rows["method"].isin(methods)].copy()
    rows = rows.groupby(["model", "method"], as_index=False)[metric].mean()
    max_value = clean_float(rows[metric].max())
    if math.isnan(max_value) or max_value <= 0:
        return ""
    y_max = chart_y_max(max_value, metric)
    width, height = 760, 330
    left, right, top, bottom = 58, 22, 66, 48
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_for(group_idx: int, method_idx: int) -> float:
        group_w = plot_w / max(1, len(models))
        bar_gap = 4
        inner = min(104, group_w * 0.72)
        bar_w = (inner - bar_gap * (len(methods) - 1)) / len(methods)
        return left + group_idx * group_w + (group_w - inner) / 2 + method_idx * (bar_w + bar_gap)

    def y_for(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    group_w = plot_w / max(1, len(models))
    inner = min(104, group_w * 0.72)
    bar_gap = 4
    bar_w = (inner - bar_gap * (len(methods) - 1)) / len(methods)
    lookup = {(str(row["model"]), str(row["method"])): clean_float(row[metric]) for _, row in rows.iterrows()}
    baseline_line = ""
    if metric == "speedup" and y_max > 1:
        y = y_for(1)
        baseline_line = (
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" '
            'stroke="#94a3b8" stroke-dasharray="5 5"/>'
            f'<text x="{width - right - 72}" y="{y - 6:.1f}" font-size="12" fill="#64748b">1.0x baseline</text>'
        )
    grid = []
    for frac in [0, 0.5, 1.0]:
        value = y_max * frac
        y = y_for(value)
        label = chart_tick(value, metric)
        grid.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="13" fill="#64748b">{label}</text>'
        )
    bars = []
    for gi, model in enumerate(models):
        gx = left + gi * group_w + group_w / 2
        bars.append(f'<text x="{gx:.1f}" y="{height - 17}" text-anchor="middle" font-size="14" fill="#111827">{esc(short_model(model))}</text>')
        for mi, method in enumerate(methods):
            value = lookup.get((model, method), math.nan)
            if math.isnan(value):
                continue
            x = x_for(gi, mi)
            y = y_for(value)
            color = PALETTE.get(method, "#4b5563")
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" rx="3" fill="{color}"/>'
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="12" fill="#111827">{chart_value(value, metric)}</text>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{esc(title)}</text>'
        f'{legend_svg(methods, width / 2, 48)}'
        f'{"".join(grid)}{baseline_line}'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#cbd5e1"/>'
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#cbd5e1"/>'
        f'{"".join(bars)}</svg>'
    )


def line_svg(rows: pd.DataFrame, title: str, metric: str, x_key: str, series_key: str) -> str:
    if rows.empty:
        return ""
    rows = rows.dropna(subset=[metric, x_key, series_key]).copy()
    if rows.empty:
        return ""
    rows[x_key] = pd.to_numeric(rows[x_key], errors="coerce")
    rows = rows.dropna(subset=[x_key])
    series = sorted(rows[series_key].dropna().astype(str).unique())
    x_values = sorted(rows[x_key].dropna().unique())
    if not series or not x_values:
        return ""
    max_value = clean_float(rows[metric].max())
    if math.isnan(max_value) or max_value <= 0:
        return ""
    y_max = chart_y_max(max_value, metric)
    width, height = 760, 330
    left, right, top, bottom = 58, 24, 66, 48
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_for(value: float) -> float:
        if len(x_values) == 1:
            return left + plot_w / 2
        return left + (list(x_values).index(value) / (len(x_values) - 1)) * plot_w

    def y_for(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    grid = []
    for frac in [0, 0.5, 1.0]:
        value = y_max * frac
        y = y_for(value)
        label = chart_tick(value, metric)
        grid.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="13" fill="#64748b">{label}</text>'
        )
    axis_labels = [
        f'<text x="{x_for(v):.1f}" y="{height - 17}" text-anchor="middle" font-size="14" fill="#111827">{int(v)}</text>'
        for v in x_values
    ]
    lines = []
    for idx, item in enumerate(series):
        color = PALETTE.get(item, ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"][idx % 5])
        sub = rows[rows[series_key].astype(str) == item].sort_values(x_key)
        points = []
        for _, row in sub.iterrows():
            value = clean_float(row[metric])
            if math.isnan(value):
                continue
            points.append((x_for(row[x_key]), y_for(value), value))
        if not points:
            continue
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        lines.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y, value in points:
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
                f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12" fill="#111827">{chart_value(value, metric)}</text>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{esc(title)}</text>'
        f'{legend_svg(series, width / 2, 48, gap=122)}'
        f'{"".join(grid)}'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#cbd5e1"/>'
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#cbd5e1"/>'
        f'{"".join(axis_labels)}{"".join(lines)}</svg>'
    )


def nemorl_grouped_metric_svg(
    rows: pd.DataFrame,
    title: str,
    y_label: str,
    series: list[tuple[str, str, str]],
    *,
    reference_line: bool = False,
    lower_is_better: bool = False,
) -> str:
    if rows.empty:
        return ""
    rows = rows.copy()
    rows["method_display"] = rows["method_k"].map(nemorl_method_label)
    method_order = [
        "Baseline",
        "Eagle-3 K3",
        "Eagle-3 K5",
        "Suffix K32",
        "PARD K5",
        "PARD K16",
        "PARD-2 K5",
        "PARD-2 K16",
    ]
    methods = [method for method in method_order if method in set(rows["method_display"])]
    if not methods:
        methods = rows["method_display"].dropna().astype(str).tolist()
    plotted_series = []
    max_value = 0.0
    for column, label, color in series:
        values = []
        for method in methods:
            sub = rows[rows["method_display"] == method]
            value = clean_float(sub[column].iloc[0]) if not sub.empty else math.nan
            values.append(value)
            if not math.isnan(value):
                max_value = max(max_value, value)
        if any(not math.isnan(value) for value in values):
            plotted_series.append((column, label, color, values))
    if not plotted_series or max_value <= 0:
        return ""

    width, height = 920, 390
    left, right, top, bottom = 76, 28, 78, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = max(1.15 if reference_line else 0.1, max_value * 1.22)
    group_w = plot_w / max(1, len(methods))
    inner = min(132, group_w * 0.78)
    bar_gap = 4
    bar_w = (inner - bar_gap * (len(plotted_series) - 1)) / len(plotted_series)

    def y_for(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    def fmt_metric(value: float, column: str) -> str:
        if "speedup" in column:
            return f"{value:.2f}x"
        if "time" in column:
            return f"{value:.0f}s"
        return f"{value:.1f}"

    grid = []
    for frac in [0, 0.5, 1.0]:
        value = y_max * frac
        y = y_for(value)
        label = f"{value:.1f}x" if reference_line else f"{value:.0f}" if max_value > 20 else f"{value:.1f}"
        grid.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d1d5db" stroke-dasharray="6 6"/>'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="14" fill="#4b5563">{label}</text>'
        )

    baseline = ""
    if reference_line and y_max > 1:
        y = y_for(1.0)
        baseline = (
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#111827" stroke-dasharray="5 5" stroke-width="1.4"/>'
            f'<text x="{width - right - 76}" y="{y - 8:.1f}" font-size="13" fill="#111827">1.0x baseline</text>'
        )

    legend_parts = []
    legend_gap = 176
    legend_start = width / 2 - ((len(plotted_series) - 1) * legend_gap) / 2
    for idx, (_, label, color, _) in enumerate(plotted_series):
        x = legend_start + idx * legend_gap
        legend_parts.append(
            f'<rect x="{x:.1f}" y="37" width="15" height="15" rx="2" fill="{color}" stroke="#192133" stroke-width="1.8"/>'
            f'<text x="{x + 22:.1f}" y="50" font-size="14" fill="#111827">{esc(label)}</text>'
        )

    bars = []
    for group_idx, method in enumerate(methods):
        gx = left + group_idx * group_w + group_w / 2
        bars.append(f'<text x="{gx:.1f}" y="{height - 24}" text-anchor="middle" font-size="14" fill="#111827">{esc(method)}</text>')
        for series_idx, (column, _, color, values) in enumerate(plotted_series):
            value = values[group_idx]
            if math.isnan(value):
                continue
            x = left + group_idx * group_w + (group_w - inner) / 2 + series_idx * (bar_w + bar_gap)
            y = y_for(value)
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" rx="3" fill="{color}" stroke="#192133" stroke-width="1.8"/>'
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="11" fill="#111827">{fmt_metric(value, column)}</text>'
            )

    direction = "lower is better" if lower_is_better else "higher is better"
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">{esc(title)}</text>'
        f'{"".join(legend_parts)}'
        f'<text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="15" fill="#111827">{esc(y_label)}</text>'
        f'{"".join(grid)}{baseline}'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>'
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>'
        f'{"".join(bars)}'
        f'<text x="{width - right}" y="{height - 6}" text-anchor="end" font-size="12" fill="#64748b">{direction}</text>'
        '</svg>'
    )


def nemorl_chart_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    current = rows[pd.to_numeric(rows.get("max_steps"), errors="coerce") == 20].copy()
    current = current[pd.to_numeric(current.get("completed_steps"), errors="coerce").fillna(0) > 0]
    metric_cols = [
        "gen_tps_speedup",
        "e2e_tps_speedup",
        "generation_time_speedup",
        "e2e_step_time_speedup",
        "generation_worker_tokens_per_sec_per_gpu_mean",
        "e2e_tokens_per_sec_per_gpu_mean",
        "generation_time_s_mean",
        "total_step_time_s_mean",
    ]
    for col in metric_cols:
        current[col] = pd.to_numeric(current.get(col), errors="coerce")
    current = current.dropna(subset=["generation_worker_tokens_per_sec_per_gpu_mean"])
    if current.empty:
        return current
    current["has_gen_speedup"] = current["gen_tps_speedup"].notna()
    current = current.sort_values(
        ["has_gen_speedup", "completed_steps", "gen_tps_speedup"],
        ascending=[False, False, False],
    )
    return current.drop_duplicates(
        subset=["source_group", "model_name", "mode", "max_new_tokens", "method_k"],
        keep="first",
    ).drop(columns=["has_gen_speedup"], errors="ignore")


def nemorl_chart_model_order(models: list[str]) -> list[str]:
    preferred = ["Qwen3-235B-A22B", "Qwen3-30B-A3B", "Qwen3-32B", "Qwen3-8B"]
    present = list(dict.fromkeys(str(model) for model in models if str(model) and str(model) != "nan"))
    ordered = [model for model in preferred if model in present]
    ordered.extend(model for model in present if model not in ordered)
    return ordered


def nemorl_charts_section(rows: pd.DataFrame) -> str:
    chart_rows = nemorl_chart_rows(rows)
    if chart_rows.empty:
        return '<section><h2>Baseline-Relative Charts</h2><p class="note">No parsed step20 timing rows are available yet for charting.</p></section>'
    metric_specs = [
        ("Generation Throughput Speedup", "gen_tps_speedup", "Speedup vs baseline"),
        ("E2E Throughput Speedup", "e2e_tps_speedup", "Speedup vs baseline"),
        ("Generation Step-Time Speedup", "generation_time_speedup", "Baseline time / run time"),
        ("E2E Step-Time Speedup", "e2e_step_time_speedup", "Baseline time / run time"),
    ]
    model_sections = []
    for model in nemorl_chart_model_order(chart_rows["model_name"].astype(str).tolist()):
        sub = chart_rows[chart_rows["model_name"].astype(str) == model].copy()
        if sub.empty:
            continue
        cards = [
            nemorl_multigroup_metric_svg(
                sub,
                title,
                metric,
                y_label,
                include_model_in_group=False,
                max_groups=6,
            )
            for title, metric, y_label in metric_specs
        ]
        rendered = "".join(f'<div class="chart-card">{card}</div>' for card in cards if card)
        if not rendered:
            continue
        model_sections.append(
            f'<h3>{esc(model)}</h3>'
            '<p class="note">Within this model, x-axis groups are matched setup slices: mode, max OSL, and cluster/source. Method colors compare against the matched baseline inside each slice.</p>'
            f'<div class="model-charts">{rendered}</div>'
        )
    return (
        '<section><h2>Baseline-Relative Charts</h2>'
        '<p class="note">Charts use parsed step20 rows and are grouped by model. Baselines are matched by model, mode, max OSL, temperature/top_p, and source setup; each model section keeps those setup slices separate on the x-axis.</p>'
        + "".join(model_sections)
        + '</section>'
    )


def nemorl_group_label(row: pd.Series, *, include_model: bool = True) -> str:
    model = short_model(row.get("model_name", row.get("model", ""))) if include_model else ""
    mode = str(row.get("mode", "") or "sync")
    osl = clean_float(row.get("max_new_tokens"))
    cluster = str(row.get("cluster", "") or "").upper()
    osl_label = f"OSL{int(osl)}" if not math.isnan(osl) else "OSL?"
    return "\n".join(part for part in [model, mode, osl_label, cluster] if part)


def svg_multiline_text(x: float, y: float, lines: list[str], *, size: int = 12, anchor: str = "middle") -> str:
    tspans = []
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else size + 2
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="#111827">' + "".join(tspans) + "</text>"


def nemorl_method_order(methods: list[str]) -> list[str]:
    preferred = [
        "baseline",
        "baseline_fuselossfalse",
        "eagle3_k3",
        "eagle3_k5",
        "eagle3_k7",
        "eagle3_k9",
        "suffix_k32",
        "pard_k1_tp1",
        "pard_k1_tp2",
        "pard_k5",
        "pard_k8",
        "pard_k12",
        "pard_k16",
        "pard2",
        "pard2_k5",
        "pard2_k16",
        "pard2_8b",
        "pard2_14b",
    ]
    present = list(dict.fromkeys(str(method) for method in methods if str(method) and str(method) != "nan"))
    ordered = [method for method in preferred if method in present]
    ordered.extend(method for method in present if method not in ordered)
    return ordered


def nemorl_multigroup_metric_svg(
    rows: pd.DataFrame,
    title: str,
    metric: str,
    y_label: str,
    *,
    reference_line: bool = True,
    max_groups: int = 10,
    include_model_in_group: bool = True,
) -> str:
    if rows.empty:
        return ""
    rows = rows.copy()
    rows[metric] = pd.to_numeric(rows.get(metric), errors="coerce")
    rows = rows.dropna(subset=[metric])
    rows = rows[pd.to_numeric(rows.get("max_steps"), errors="coerce") == 20]
    rows = rows[pd.to_numeric(rows.get("completed_steps"), errors="coerce").fillna(0) > 0]
    if rows.empty:
        return ""
    rows["group_label"] = rows.apply(lambda row: nemorl_group_label(row, include_model=include_model_in_group), axis=1)
    rows["source_rank"] = rows["source_group"].astype(str).map(
        lambda value: 0 if "Qwen235B" in value else 1 if "Lyris" in value else 2
    )
    rows = rows.sort_values(["source_rank", "model_name", "mode", "max_new_tokens", "completed_steps"])
    group_labels = list(dict.fromkeys(rows["group_label"].astype(str).tolist()))[:max_groups]
    rows = rows[rows["group_label"].isin(group_labels)]
    methods = nemorl_method_order(rows["method_k"].astype(str).tolist())
    if not group_labels or not methods:
        return ""

    max_value = clean_float(rows[metric].max())
    if math.isnan(max_value) or max_value <= 0:
        return ""
    y_max = max(1.15 if reference_line else 0.1, max_value * 1.18)
    legend_cols = min(5, len(methods))
    legend_rows = math.ceil(len(methods) / legend_cols)
    width = max(820, 110 + 108 * len(group_labels))
    height = 338 + max(0, legend_rows - 1) * 22
    left, right, top, bottom = 62, 22, 68 + max(0, legend_rows - 1) * 22, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / max(1, len(group_labels))
    inner = min(98, group_w * 0.84)
    bar_gap = 2.5
    bar_w = max(6, (inner - bar_gap * (len(methods) - 1)) / len(methods))

    def y_for(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    lookup: dict[tuple[str, str], float] = {}
    for _, row in rows.iterrows():
        lookup[(str(row["group_label"]), str(row["method_k"]))] = clean_float(row.get(metric))

    grid = []
    for frac in [0, 0.5, 1.0]:
        value = y_max * frac
        y = y_for(value)
        label = f"{value:.1f}x" if reference_line else f"{value:.1f}"
        grid.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d1d5db" stroke-dasharray="6 6"/>'
            f'<text x="{left - 8}" y="{y + 5:.1f}" text-anchor="end" font-size="13" fill="#4b5563">{label}</text>'
        )

    baseline = ""
    if reference_line and y_max > 1:
        y = y_for(1.0)
        baseline = (
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#111827" stroke-dasharray="5 5" stroke-width="1.3"/>'
            f'<text x="{width - right - 82}" y="{y - 8:.1f}" font-size="12" fill="#111827">1.0x baseline</text>'
        )

    legend_cell_w = 150
    legend_total_w = legend_cols * legend_cell_w
    legend_start = max(left, (width - legend_total_w) / 2)
    legend_parts = []
    for idx, method in enumerate(methods):
        x = legend_start + (idx % legend_cols) * legend_cell_w
        y = 35 + (idx // legend_cols) * 22
        color = PALETTE.get(method, "#4b5563")
        legend_parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="13" height="13" rx="2" fill="{color}" stroke="#192133" stroke-width="1.3"/>'
            f'<text x="{x + 19:.1f}" y="{y + 11.5:.1f}" font-size="12.5" fill="#111827">{esc(nemorl_method_label(method))}</text>'
        )

    bars = []
    show_bar_labels = bar_w >= 11 and len(group_labels) * len(methods) <= 42
    for group_idx, label in enumerate(group_labels):
        gx = left + group_idx * group_w + group_w / 2
        bars.append(svg_multiline_text(gx, height - 58, label.split("\n"), size=12))
        for method_idx, method in enumerate(methods):
            value = lookup.get((label, method), math.nan)
            if math.isnan(value):
                continue
            x = left + group_idx * group_w + (group_w - inner) / 2 + method_idx * (bar_w + bar_gap)
            y = y_for(value)
            color = PALETTE.get(method, "#4b5563")
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" rx="2.5" fill="{color}" stroke="#192133" stroke-width="1.2">'
                f'<title>{esc(nemorl_method_label(method))}: {value:.2f}x</title></rect>'
                + (
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-size="10.5" fill="#111827">{value:.2f}x</text>'
                    if show_bar_labels
                    else ""
                )
            )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
        f'<text x="{width / 2}" y="23" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{esc(title)}</text>'
        f'{"".join(legend_parts)}'
        f'<text x="17" y="{top + plot_h / 2:.1f}" transform="rotate(-90 17 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="13" fill="#111827">{esc(y_label)}</text>'
        f'{"".join(grid)}{baseline}'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>'
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>'
        f'{"".join(bars)}</svg>'
    )


def charts_section(added: pd.DataFrame) -> str:
    if added.empty:
        return ""
    valid = added[added["valid_result"]].copy()
    if valid.empty:
        return ""
    focus_methods = ["eagle3_k8", "pard_k16", "pard2_k16"]
    cards = []
    for temp in [0.0, 1.0]:
        sub = valid[(valid["domain"] == "Math") & (valid["temperature"] == temp) & valid["method"].isin(focus_methods)]
        cards.append(grouped_bar_svg(sub, f"Math Temp {temp:.1f} Speedup", "speedup", focus_methods))
        cards.append(grouped_bar_svg(sub, f"Math Temp {temp:.1f} Acceptance", "acceptance_pct", focus_methods))
        cards.append(grouped_bar_svg(sub, f"Math Temp {temp:.1f} Mean Accepted Length", "mean_accept_len", focus_methods))
    for temp in [0.0, 1.0]:
        sub = valid[
            (valid["domain"] == "Math")
            & (valid["temperature"] == temp)
            & (valid["model"] == "Qwen3-235B-A22B")
            & valid["method"].isin(focus_methods)
        ].copy()
        cards.append(line_svg(sub, f"Qwen3-235B Math Temp {temp:.1f}: Speedup vs Batch", "speedup", "batch_size", "method"))
        cards.append(line_svg(sub, f"Qwen3-235B Math Temp {temp:.1f}: Mean Accepted Length vs Batch", "mean_accept_len", "batch_size", "method"))
    pard2 = valid[
        (valid["model"] == "Qwen3-235B-A22B")
        & (valid["method"].astype(str).str.startswith("pard2_k"))
    ].copy()
    if not pard2.empty:
        pard2["k"] = pard2["method"].astype(str).str.extract(r"k(\d+)").astype(float)
        for domain in ["Math", "SWE"]:
            sub = pard2[pard2["domain"] == domain].copy()
            if sub.empty:
                continue
            sub["series"] = sub["temperature"].map(lambda v: f"temp{float(v):.0f}")
            summary = sub.groupby(["k", "series"], as_index=False).agg(
                speedup=("speedup", "mean"),
                acceptance_pct=("acceptance_pct", "mean"),
                mean_accept_len=("mean_accept_len", "mean"),
            )
            cards.append(line_svg(summary, f"Qwen3-235B {domain} PARD-2 K Sweep Speedup", "speedup", "k", "series"))
            cards.append(line_svg(summary, f"Qwen3-235B {domain} PARD-2 K Sweep Acceptance", "acceptance_pct", "k", "series"))
            cards.append(line_svg(summary, f"Qwen3-235B {domain} PARD-2 K Sweep Mean Accepted Length", "mean_accept_len", "k", "series"))
    cards = [card for card in cards if card]
    if not cards:
        return ""
    return (
        '<section class="section"><h2>Visual Summary</h2>'
        '<p class="note">Charts use matched-baseline speedups and average repeated batch rows where needed. Legends are centered; tables below keep exact row provenance.</p>'
        '<div class="charts">'
        + "".join(f'<div class="chart-card">{card}</div>' for card in cards)
        + "</div></section>"
    )


def esc(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return html.escape(str(value), quote=True)


def text_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def first_text(row: pd.Series, *keys: str) -> str:
    for key in keys:
        value = text_value(row.get(key, ""))
        if value:
            return value
    return ""


def normalize_wandb_url(value: object) -> str:
    text = text_value(value)
    if not text:
        return ""
    match = WANDB_URL_RE.search(text)
    if match:
        return match.group(0).rstrip(".,)")
    return text if text.startswith(("http://", "https://")) else ""


def link_html(value: object, label: str = "W&B") -> str:
    url = normalize_wandb_url(value)
    if not url:
        return ""
    return f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(label)}</a>'


def wandb_link_html(row: pd.Series) -> str:
    direct_url = normalize_wandb_url(row.get("wandb_url", ""))
    if direct_url:
        return link_html(direct_url, "run")
    project = first_text(row, "wandb_project")
    if not project:
        return ""
    return link_html(f"https://wandb.ai/{WANDB_ENTITY}/{project}", "project")


def clean_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return math.nan
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    value = clean_float(value)
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def fmt_x(value: object) -> str:
    return fmt(value, 2, "x")


def fmt_pct(value: object) -> str:
    return fmt(value, 1, "%")


def model_name(value: object) -> str:
    text = str(value)
    lower = text.lower()
    for key, name in MODEL_MAP.items():
        if key in lower:
            return name
    if "235b" in lower:
        return "Qwen3-235B-A22B"
    if "30b" in lower:
        return "Qwen3-30B-A3B"
    if "32b" in lower:
        return "Qwen3-32B"
    if "8b" in lower:
        return "Qwen3-8B"
    return text


def method_with_k(method: object, k: object) -> str:
    method_text = str(method).strip()
    if not method_text or method_text == "baseline":
        return method_text or "baseline"
    k_value = clean_float(k)
    if math.isnan(k_value):
        return method_text
    return f"{method_text}_k{int(k_value)}"


def refine_nemorl_method_from_run(method_k: object, run_id: object) -> str:
    method = str(method_k)
    run = str(run_id).lower()
    if method == "pard_k1":
        if "drafttp1_targettp1" in run:
            return "pard_k1_tp1"
        if "pardk1" in run:
            return "pard_k1_tp2"
    return method


def parse_completed_last(value: object) -> tuple[float, float]:
    match = re.search(r"(\d+)\s*/\s*(\d+)", str(value))
    if not match:
        return math.nan, math.nan
    return float(match.group(1)), float(match.group(2))


def normalize_nemorl_method(method: object, label: object = "", k: object = math.nan) -> str:
    method_text = str(method).strip()
    label_text = str(label).strip()
    lower = f"{method_text} {label_text}".lower()
    k_value = clean_float(k)
    if "baseline" in lower:
        if "fuse_loss=false" in lower or "fuselossfalse" in lower:
            return "baseline_fuselossfalse"
        return "baseline"
    if "eagle" in lower:
        return f"eagle3_k{int(k_value)}" if not math.isnan(k_value) and k_value > 0 else "eagle3_k3"
    if "suffix" in lower:
        return f"suffix_k{int(k_value)}" if not math.isnan(k_value) and k_value > 0 else "suffix_k32"
    if "pard-2" in lower or "pard2" in lower:
        if not math.isnan(k_value) and k_value > 0:
            return f"pard2_k{int(k_value)}"
        if "14b" in lower:
            return "pard2_14b"
        if "8b" in lower:
            return "pard2_8b"
        return "pard2"
    if "pard" in lower:
        if not math.isnan(k_value) and k_value > 0:
            return f"pard_k{int(k_value)}"
        match = re.search(r"k[=_-]?(\d+)", lower)
        return f"pard_k{match.group(1)}" if match else "pard"
    return method_text.lower().replace(" ", "_")


def normalize_nemorl_diagnostic_method(method: object) -> str:
    text = str(method).strip()
    lower = text.lower()
    base = normalize_nemorl_method(text)
    if base == "pard_k1":
        if "tp1" in lower:
            return "pard_k1_tp1"
        if "tp2" in lower:
            return "pard_k1_tp2"
    return base


def effective_metric(row: pd.Series, final_col: str, live_col: str) -> float:
    final = clean_float(row.get(final_col))
    if not math.isnan(final):
        return final
    return clean_float(row.get(live_col))


def baseline_lookup(main: pd.DataFrame) -> dict[tuple[object, ...], float]:
    lookup: dict[tuple[object, ...], float] = {}
    baselines = main[main["method"].astype(str) == "baseline"]
    for _, row in baselines.iterrows():
        key = (
            str(row["domain"]),
            str(row["model"]),
            float(row["temperature"]),
            int(row["batch_size"]),
            int(row["isl"]),
            int(row["osl"]),
        )
        lookup[key] = float(row["tok_s_gpu"])
    return lookup


def vllm_baseline_key(row: pd.Series) -> tuple[object, ...] | None:
    try:
        return (
            str(row["domain"]),
            str(row["model"]),
            float(row["temperature"]),
            int(clean_float(row["batch_size"])),
            int(clean_float(row["isl"])),
            int(clean_float(row["osl"])),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def fill_vllm_added_speedups(main: pd.DataFrame, added: pd.DataFrame) -> pd.DataFrame:
    if added.empty:
        return added
    added = added.copy()
    baselines = baseline_lookup(main)
    valid_baselines = added[(added["method"].astype(str) == "baseline") & (added["valid_result"])]
    for _, row in valid_baselines.iterrows():
        key = vllm_baseline_key(row)
        tok = clean_float(row.get("tok_s_gpu"))
        if key is not None and not math.isnan(tok):
            baselines.setdefault(key, tok)
    for idx, row in added.iterrows():
        key = vllm_baseline_key(row)
        if key is None:
            continue
        baseline = baselines.get(key, math.nan)
        tok = clean_float(row.get("tok_s_gpu"))
        if str(row.get("method")) == "baseline" and not math.isnan(tok):
            added.at[idx, "baseline_tok_s_gpu"] = tok
            added.at[idx, "speedup"] = 1.0
            continue
        if math.isnan(clean_float(row.get("baseline_tok_s_gpu"))) and not math.isnan(baseline):
            added.at[idx, "baseline_tok_s_gpu"] = baseline
        if math.isnan(clean_float(row.get("speedup"))) and not math.isnan(tok) and not math.isnan(baseline) and baseline:
            added.at[idx, "speedup"] = tok / baseline
    return added


def load_vllm_added(main: pd.DataFrame) -> pd.DataFrame:
    baselines = baseline_lookup(main)
    parts: list[pd.DataFrame] = []
    for path, source_label, priority in VLLM_LIVE_SOURCES:
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        rows = []
        for _, row in raw.iterrows():
            domain = "Math" if str(row.get("domain", "")).lower() == "math" else "SWE"
            model = model_name(row.get("model_group", ""))
            temperature = clean_float(row.get("temperature"))
            batch = int(clean_float(row.get("batch_size")))
            isl = int(clean_float(row.get("isl")))
            osl = int(clean_float(row.get("osl")))
            method = method_with_k(row.get("method"), row.get("k"))
            tok = effective_metric(row, "final_tok_s_gpu", "live_tok_s_gpu_approx")
            acceptance = effective_metric(row, "final_acceptance_pct", "live_acceptance_pct")
            mean_len = effective_metric(row, "final_mean_accept_len", "live_mean_accept_len")
            baseline = baselines.get((domain, model, temperature, batch, isl, osl), math.nan)
            speedup = tok / baseline if not math.isnan(tok) and not math.isnan(baseline) and baseline else math.nan
            rows.append(
                {
                    "domain": domain,
                    "model": model,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "batch_size": batch,
                    "isl": isl,
                    "osl": osl,
                    "method": method,
                    "job_id": str(row.get("job_id", "")),
                    "state": str(row.get("state", "")),
                    "tok_s_gpu": tok,
                    "baseline_tok_s_gpu": baseline,
                    "speedup": speedup,
                    "acceptance_pct": acceptance,
                    "mean_accept_len": mean_len,
                    "basis": "final breakdown" if str(row.get("breakdown_valid", "")) == "1" else "live log",
                    "source": str(path.relative_to(ROOT)),
                    "source_label": source_label,
                    "source_priority": priority,
                    "logs_dir": str(row.get("logs_dir", "")),
                    "valid_result": str(row.get("state", "")) == "COMPLETED" and not math.isnan(tok),
                }
            )
        parts.append(pd.DataFrame(rows))
    if VLLM_LEGACY_NORMALIZED.exists():
        legacy = pd.read_csv(VLLM_LEGACY_NORMALIZED)
        parts.append(legacy)
    if DFLASH.exists():
        dflash = pd.read_csv(DFLASH)
        rows = []
        for _, row in dflash.iterrows():
            rows.append(
                {
                    "domain": "Math",
                    "model": "Qwen3-235B-A22B",
                    "temperature": math.nan,
                    "top_p": math.nan,
                    "batch_size": int(clean_float(row.get("batch_size"))),
                    "isl": math.nan,
                    "osl": math.nan,
                    "method": str(row.get("method", "")).lower().replace(" ", "_"),
                    "job_id": "",
                    "state": "COMPLETED",
                    "tok_s_gpu": clean_float(row.get("output_tok_s_per_gpu")),
                    "baseline_tok_s_gpu": math.nan,
                    "speedup": math.nan,
                    "acceptance_pct": clean_float(row.get("acceptance_pct")),
                    "mean_accept_len": clean_float(row.get("mean_acceptance_length")),
                    "basis": "legacy DFlash OpenMath; no matched OSL32K baseline",
                    "source": str(DFLASH.relative_to(ROOT)),
                    "source_label": "DFlash OpenMath retry28",
                    "source_priority": 5,
                    "logs_dir": "",
                    "valid_result": True,
                }
            )
        parts.append(pd.DataFrame(rows))
    if not parts:
        return pd.DataFrame()
    added = pd.concat(parts, ignore_index=True)
    added["valid_result"] = added["valid_result"].astype(str).str.lower().isin({"1", "true", "yes"})
    added = fill_vllm_added_speedups(main, added)
    added = added.sort_values(
        [
            "domain",
            "model",
            "temperature",
            "batch_size",
            "method",
            "source_priority",
            "valid_result",
        ],
        na_position="last",
    )
    key = ["domain", "model", "temperature", "batch_size", "isl", "osl", "method"]
    added = added.groupby(key, dropna=False, as_index=False).tail(1).copy()
    added = fill_vllm_added_speedups(main, added)
    added = added.sort_values(["domain", "temperature", "model", "method", "batch_size"], na_position="last")
    return added


def aggregate_added(added: pd.DataFrame) -> pd.DataFrame:
    if added.empty:
        return added
    valid = added[added["valid_result"]].copy()
    if valid.empty:
        return valid
    grouped = (
        valid.groupby(["domain", "temperature", "model", "method", "source_label"], dropna=False)
        .agg(
            rows=("job_id", "count"),
            batches=("batch_size", lambda s: "/".join(str(int(v)) for v in sorted(pd.to_numeric(s, errors="coerce").dropna()))),
            isl=("isl", "first"),
            osl=("osl", "first"),
            tok_s_gpu=("tok_s_gpu", "mean"),
            speedup=("speedup", "mean"),
            acceptance_pct=("acceptance_pct", "mean"),
            mean_accept_len=("mean_accept_len", "mean"),
            basis=("basis", "first"),
            source=("source", "first"),
        )
        .reset_index()
    )
    return grouped.sort_values(["domain", "temperature", "model", "method"], na_position="last")


def matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, temp, model, method), group in df.groupby(["domain", "temperature", "model", "method"], dropna=False):
        row = {
            "domain": domain,
            "temperature": temp,
            "model": model,
            "method": method,
        }
        for batch in [1, 2, 4, 8, 16, 32]:
            values = group[group["batch_size"] == batch]["speedup"].dropna()
            row[f"b{batch}_speedup"] = float(values.iloc[-1]) if len(values) else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["domain", "temperature", "model", "method"], na_position="last")


def temp_trends_section() -> str:
    if not VLLM_TEMP_TRENDS.exists():
        return ""
    rows = pd.read_csv(VLLM_TEMP_TRENDS)
    if rows.empty:
        return ""
    rows = rows.copy()
    rows = rows.rename(
        columns={
            "mean_speedup_vs_baseline": "mean_speedup",
            "mean_tok_s_per_gpu": "mean_tok_s_gpu",
            "mean_acceptance_pct": "mean_acceptance",
            "mean_acceptance_length": "mean_accept_len",
        }
    )
    rows = rows.sort_values(["domain", "model", "temperature", "method"], na_position="last")
    return (
        '<section class="section"><h2>Historical Temp0/Temp1 Trend Summary</h2>'
        '<p class="note">This preserves the older extensive Math/SWE temperature analysis page. It is an aggregate view; exact batch-level rows are reflected in the detailed sections below when the underlying CSV or breakdown JSON exists.</p>'
        '<div class="table-wrap">'
        + table(
            rows,
            [
                ("domain", "Domain", "text"),
                ("dataset", "Dataset", "text"),
                ("model", "Model", "text"),
                ("temperature", "Temp", "text"),
                ("method", "Method", "text"),
                ("rows", "Rows", "int"),
                ("mean_speedup", "Mean speedup", "x"),
                ("min_speedup", "Min", "x"),
                ("max_speedup", "Max", "x"),
                ("mean_tok_s_gpu", "tok/s/GPU", "num"),
                ("mean_acceptance", "Acceptance", "pct"),
                ("mean_accept_len", "Mean len", "num"),
                ("basis", "Basis", "text"),
                ("source", "Source", "text"),
            ],
        )
        + "</div></section>"
    )


def related_vllm_reports_section() -> str:
    reports = [
        (
            "Temp0/Temp1 Trend Page",
            "Math/SWE temperature 0 vs 1 aggregate trends and key interpretation.",
            "vllm_standalone_temp0_temp1_trends_20260616.html",
        ),
        (
            "Broad SpecDec Dashboard",
            "Older wide dashboard with vLLM standalone, SWE/Math snapshots, and status fragments.",
            "specdec_benchmark_metrics_dashboard_20260616.html",
        ),
        (
            "Clean Primary Results",
            "Curated 2026-06-17 vLLM standalone primary/supplemental split.",
            "vllm_standalone_clean_results_20260617.html",
        ),
        (
            "6/19 Batch Matrix",
            "Earlier all-batch report before the latest legacy-source refresh.",
            "vllm_standalone_results_20260619.html",
        ),
        (
            "Qwen235B SWE Batch Sweep",
            "Dedicated Qwen3-235B SWE OSL32K batch-sweep speedup page.",
            "lyris_qwen235b_swebench_osl32k_batch_sweep_speedups_20260612.html",
        ),
        (
            "Qwen235B Diagnostics",
            "Live diagnostic page from the older Qwen3-235B standalone runs.",
            "lyris_qwen235b_standalone_live_diagnostics_20260613.html",
        ),
    ]
    items = []
    for title, desc, href in reports:
        if not (DOCS / href).exists():
            continue
        items.append(
            '<div class="card">'
            f'<b><a href="{esc(href)}">{esc(title)}</a></b>'
            f'<span>{esc(desc)}</span>'
            f'<code>{esc(href)}</code>'
            '</div>'
        )
    if not items:
        return ""
    return (
        '<section class="section"><h2>Related Broader Reports</h2>'
        '<p class="note">This latest page is intentionally focused on matched ISL4096/OSL32768 comparisons. Use these archive pages for broader historical, long-OSL, partial, or aggregate views that are not all directly comparable in one speedup matrix.</p>'
        '<div class="cards">'
        + "".join(items)
        + "</div></section>"
    )


def table(rows: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    if rows.empty:
        return '<p class="note">No rows.</p>'
    head = "".join(f"<th>{esc(label)}</th>" for _, label, _ in columns)
    body = []
    for _, row in rows.iterrows():
        cells = []
        for key, _, kind in columns:
            value = row.get(key, "")
            cls = "num" if kind in {"num", "x", "pct"} else ""
            if key == "slurm_state":
                cls = str(value).strip()
            if kind == "num":
                text = fmt(value, 2)
            elif kind == "int":
                text = "n/a" if pd.isna(value) else str(int(float(value)))
            elif kind == "x":
                text = fmt_x(value)
            elif kind == "pct":
                text = fmt_pct(value)
            elif kind == "temp":
                text = "n/a" if pd.isna(value) else f"{float(value):.1f}"
            elif kind == "link":
                text = wandb_link_html(row) if key == "wandb_url" else link_html(value)
            else:
                text = esc(value)
            cells.append(f'<td class="{cls}">{text}</td>' if cls else f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table>"


def build_vllm_html(main: pd.DataFrame, added: pd.DataFrame) -> str:
    updated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added_summary = aggregate_added(added)
    main_matrix = pd.read_csv(DOCS / "vllm_standalone_all_batches_combined_matrix_20260619.csv")
    added_matrix = matrix(added[added["valid_result"]]) if not added.empty else pd.DataFrame()
    focus = added[
        added["method"].isin(["pard_k16", "pard2_k16"])
        & added["model"].isin(["Qwen3-32B", "Qwen3-235B-A22B", "Qwen3-30B-A3B", "Qwen3-8B"])
    ].copy()
    failed = added[~added["valid_result"]].copy() if not added.empty else pd.DataFrame()
    eagle8 = added[
        (added["model"] == "Qwen3-235B-A22B")
        & (added["method"] == "eagle3_k8")
        & (added["domain"] == "Math")
        & added["valid_result"]
    ]
    eagle_lines = []
    for temp, label in [(0.0, "temp0"), (1.0, "temp1")]:
        sub = eagle8[eagle8["temperature"] == temp]
        if not sub.empty:
            eagle_lines.append(
                f"Qwen3-235B Eagle-3 K8 Math {label}: mean speedup {sub['speedup'].mean():.2f}x, "
                f"acceptance {sub['acceptance_pct'].mean():.1f}%."
            )
    q32_pard16 = added[
        (added["model"] == "Qwen3-32B")
        & (added["method"] == "pard_k16")
        & (added["temperature"] == 1.0)
        & added["valid_result"]
    ]
    if not q32_pard16.empty:
        eagle_lines.append(
            f"Qwen3-32B PARD K16 temp1 retry completed {len(q32_pard16)} rows, "
            f"mean speedup {q32_pard16['speedup'].mean():.2f}x."
        )
    key_finding = " ".join(eagle_lines) if eagle_lines else "Latest CSV refresh completed; no new valid rows found."
    css = """
:root{--text:#111827;--muted:#6b7280;--line:#d8dee8;--bg:#f7f8fb;--panel:#fff;--blue:#1f5fbf;--good:#e8f3ff;--bad:#fff0f0;--warn:#fff7df}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;font-size:15px;line-height:1.42}main{max-width:1500px;margin:0 auto;padding:24px}h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:28px 0 10px}h3{font-size:16px;margin:18px 0 8px}.sub,.note{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:18px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}.card b{display:block;font-size:22px}.pill{display:inline-block;border:1px solid var(--line);background:#fff;border-radius:999px;padding:4px 9px;margin:2px 4px 2px 0;color:#374151}.section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;margin:14px 0}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:12px}.chart-card{border:1px solid var(--line);border-radius:8px;background:#fff;padding:10px;min-width:0}.chart-card svg{width:100%;height:auto;display:block}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:#fff;margin:8px 0 14px}th,td{border:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}th{background:#eef2f7;font-size:13px}.num{text-align:right;font-variant-numeric:tabular-nums}.good{background:var(--good)}.bad{background:var(--bad)}.warn{background:var(--warn)}code{background:#f3f4f6;padding:1px 4px;border-radius:4px}@media(max-width:1000px){.charts{grid-template-columns:1fr}}@media(max-width:900px){main{padding:16px}.cards{grid-template-columns:1fr 1fr}table{font-size:13px}}"""
    parts = [
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        f"<title>vLLM Standalone SpecDec Results</title><style>{css}</style></head><body><main>",
        "<h1>vLLM Standalone SpecDec Results</h1>",
        f"<p class=\"sub\">Updated {esc(updated)}. Data refresh from the 6/19 batch matrix, 6/20 extra-K/PARD sweeps, 6/16 temp0/temp1 trend analysis, and refreshed Lyris legacy breakdown JSONs for Qwen3-30B-A3B and Qwen3-8B.</p>",
        "<div><span class=\"pill\">ISL 4096</span><span class=\"pill\">OSL 32768</span><span class=\"pill\">batch 1/2/4/8/16/32</span><span class=\"pill\">temperature 0.0 and 1.0</span><span class=\"pill\">top_p 1.0 where available</span></div>",
        "<div class=\"cards\">",
        f"<div class=\"card\"><b>{len(main)}</b><span>existing 6/19 rows</span></div>",
        f"<div class=\"card\"><b>{int(added['valid_result'].sum()) if not added.empty else 0}</b><span>valid added rows</span></div>",
        f"<div class=\"card\"><b>{len(failed)}</b><span>failed or invalid added rows</span></div>",
        f"<div class=\"card\"><b>{len(added_summary)}</b><span>added summary groups</span></div>",
        "</div>",
        "<section class=\"section\"><h2>Scope</h2><p>This page is the matched-comparison view for <b>ISL 4096 / OSL 32768</b>. It keeps speedup cells blank when the exact baseline is missing for the same domain, model, temperature, batch size, ISL, and OSL.</p></section>",
        related_vllm_reports_section(),
        "<section class=\"section\"><h2>Key Findings</h2><p>" + esc(key_finding) + "</p><p class=\"note\">Speedups are computed only when a matched baseline exists with the same domain, model, temperature, batch size, ISL and OSL.</p></section>",
        charts_section(added),
        temp_trends_section(),
        "<section class=\"section\"><h2>PARD / PARD-2 K=16 Focus</h2><div class=\"table-wrap\">",
        table(
            focus,
            [
                ("domain", "Domain", "text"),
                ("model", "Model", "text"),
                ("temperature", "Temp", "temp"),
                ("batch_size", "Batch", "int"),
                ("method", "Method", "text"),
                ("state", "State", "text"),
                ("tok_s_gpu", "tok/s/GPU", "num"),
                ("speedup", "Speedup", "x"),
                ("acceptance_pct", "Acceptance", "pct"),
                ("mean_accept_len", "Mean len", "num"),
                ("basis", "Basis", "text"),
            ],
        ),
        "</div></section>",
        "<section class=\"section\"><h2>Main 6/19 Batch-Speedup Matrix</h2><p class=\"note\">This is the existing baseline/reference matrix kept for continuity.</p>",
        "<div class=\"table-wrap\">",
        table(
            main_matrix,
            [
                ("domain", "Domain", "text"),
                ("temperature", "Temp", "temp"),
                ("model", "Model", "text"),
                ("method", "Method", "text"),
                ("batch_1_speedup", "B1", "x"),
                ("batch_2_speedup", "B2", "x"),
                ("batch_4_speedup", "B4", "x"),
                ("batch_8_speedup", "B8", "x"),
                ("batch_16_speedup", "B16", "x"),
                ("batch_32_speedup", "B32", "x"),
            ],
        ),
        "</div></section>",
        "<section class=\"section\"><h2>Added And Legacy Results Summary</h2><div class=\"table-wrap\">",
        table(
            added_summary,
            [
                ("domain", "Domain", "text"),
                ("temperature", "Temp", "temp"),
                ("model", "Model", "text"),
                ("method", "Method", "text"),
                ("rows", "Rows", "int"),
                ("batches", "Batches", "text"),
                ("tok_s_gpu", "tok/s/GPU", "num"),
                ("speedup", "Speedup", "x"),
                ("acceptance_pct", "Acceptance", "pct"),
                ("mean_accept_len", "Mean len", "num"),
                ("source_label", "Source", "text"),
            ],
        ),
        "</div></section>",
        "<section class=\"section\"><h2>Added And Legacy Speedup Matrix</h2><div class=\"table-wrap\">",
        table(
            added_matrix,
            [
                ("domain", "Domain", "text"),
                ("temperature", "Temp", "temp"),
                ("model", "Model", "text"),
                ("method", "Method", "text"),
                ("b1_speedup", "B1", "x"),
                ("b2_speedup", "B2", "x"),
                ("b4_speedup", "B4", "x"),
                ("b8_speedup", "B8", "x"),
                ("b16_speedup", "B16", "x"),
                ("b32_speedup", "B32", "x"),
            ],
        ),
        "</div></section>",
        "<section class=\"section\"><h2>Failed Or Invalid Added Rows</h2><div class=\"table-wrap\">",
        table(
            failed,
            [
                ("domain", "Domain", "text"),
                ("model", "Model", "text"),
                ("temperature", "Temp", "temp"),
                ("batch_size", "Batch", "int"),
                ("method", "Method", "text"),
                ("job_id", "Job", "text"),
                ("state", "State", "text"),
                ("basis", "Basis", "text"),
                ("source_label", "Source", "text"),
            ],
        ),
        "</div></section>",
        "<section class=\"section\"><h2>Sources</h2><p class=\"note\"><code>docs/vllm_standalone_all_batches_combined_20260619.csv</code>, <code>docs/vllm_standalone_all_batches_combined_matrix_20260619.csv</code>, <code>docs/vllm_standalone_temp0_temp1_trends_20260616.csv</code>, <code>docs/vllm_standalone_qwen30_qwen8_legacy_breakdowns_20260625.csv</code>, <code>docs/oci_qmath_extra_k_live_log_metrics_20260620.csv</code>, <code>docs/oci_qmath_pard2_k_sweep_live_log_metrics_20260620.csv</code>, <code>docs/oci_qmath_pard_pard2_k16_focus_live_log_metrics_20260620.csv</code>, <code>docs/lyris_qwen235b_swe_pard2_k_sweep_live_log_metrics_20260620.csv</code>, and <code>docs/qwen3_235b_dflash_retry28_openmath_metrics.csv</code>.</p></section>",
        "</main></body></html>",
    ]
    return "\n".join(parts)


def load_sacct() -> pd.DataFrame:
    if not NEMORL_SACCT.exists():
        return pd.DataFrame(columns=["job_id", "job_name", "slurm_state", "exit_code", "elapsed", "start", "end"])
    rows = []
    for line in NEMORL_SACCT.read_text().splitlines():
        parts = line.split("|")
        if len(parts) < 7 or "." in parts[0]:
            continue
        rows.append(
            {
                "job_id": parts[0],
                "job_name": parts[1],
                "slurm_state": parts[2],
                "exit_code": parts[3],
                "elapsed": parts[4],
                "start": parts[5],
                "end": parts[6],
            }
        )
    return pd.DataFrame(rows)


def load_nemorl_manifest() -> pd.DataFrame:
    parts = []
    for path in NEMORL_MANIFESTS:
        raw = pd.read_csv(path)
        raw["manifest"] = str(path.relative_to(ROOT))
        parts.append(raw)
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True)
    rows["job_id"] = rows["job_id"].astype(str)
    rows = rows.drop_duplicates(subset=["job_id"], keep="last")
    return rows


def load_nemorl_summary() -> pd.DataFrame:
    parts = []
    for path in [NEMORL_SUMMARY, *NEMORL_ADDITIONAL_SUMMARIES]:
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        raw["summary_source"] = str(path.relative_to(ROOT))
        parts.append(raw)
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True, sort=False)
    rows["job_id"] = rows["job_id"].astype(str)
    return rows.drop_duplicates(subset=["job_id"], keep="last")


def nemorl_source_group_from_run_id(run_id: object) -> str:
    text = str(run_id)
    if "20260624" in text:
        return "Lyris PerfCfg enforce_eager=false PARD diagnostics 2026-06-24"
    if "20260623" in text or "cudagraphoff" in text:
        return "Lyris PerfCfg enforce_eager=false triton W&B matrix 2026-06-23"
    if "eagerfalse_triton" in text and "wandb" in text:
        return "Lyris PerfCfg enforce_eager=false triton W&B matrix 2026-06-22"
    return "Lyris Qwen235B PR2879 OSL8192 2026-06-21"


def nemorl_config_basis_from_run_id(run_id: object) -> str:
    text = str(run_id)
    if "20260624" in text:
        return (
            "performance recipe default plus latest-main+PR2879 topology-aware fix; "
            "enforce_eager=false, MoE backend=triton, max_num_seqs=64, max_num_batched_tokens=32760/32768; PARD diagnostic sweep"
        )
    if "20260623" in text or "cudagraphoff" in text:
        return (
            "performance recipe default plus latest-main+PR2879 topology-aware fix; "
            "enforce_eager=false, MoE backend=triton, max_num_seqs=64, max_num_batched_tokens=32768, W&B enabled"
        )
    if "eagerfalse_triton" in text and "wandb" in text:
        return (
            "performance recipe default plus latest-main+PR2879 topology-aware fix; "
            "enforce_eager=false, MoE backend=triton, max_num_seqs=64, max_num_batched_tokens=32768, W&B enabled"
        )
    return "performance recipe default, latest main plus PR2879 topology-aware fix"


def enrich_nemorl() -> pd.DataFrame:
    manifest = load_nemorl_manifest()
    summary = load_nemorl_summary()
    sacct = load_sacct()
    if manifest.empty:
        return pd.DataFrame()
    if not summary.empty:
        summary["job_id"] = summary["job_id"].astype(str)
    rows = manifest.merge(summary, on="job_id", how="left", suffixes=("", "_metric"))
    if not sacct.empty:
        rows = rows.merge(sacct, on="job_id", how="left")
    for col in ["wandb_enabled", "wandb_project", "wandb_name", "wandb_url"]:
        metric_col = f"{col}_metric"
        if col not in rows:
            rows[col] = ""
        if metric_col in rows:
            rows[col] = rows[col].where(rows[col].map(text_value).ne(""), rows[metric_col])
    rows["wandb_url"] = rows["wandb_url"].map(normalize_wandb_url)
    rows["method_k"] = rows.apply(lambda r: method_with_k(r.get("method"), r.get("num_speculative_tokens")), axis=1)
    rows["method_k"] = rows.apply(lambda r: refine_nemorl_method_from_run(r.get("method_k"), r.get("run_id")), axis=1)
    rows["model_name"] = rows["model"].map(model_name)
    rows["cluster"] = "lyris"
    rows["source_group"] = rows["run_id"].map(nemorl_source_group_from_run_id)
    rows["config_basis"] = rows["run_id"].map(nemorl_config_basis_from_run_id)
    rows["enforce_eager"] = rows["run_id"].map(lambda value: False if "eagerfalse" in str(value) else "")
    rows["source_priority"] = 0
    if "slurm_state" not in rows:
        rows["slurm_state"] = ""
    rows["slurm_state"] = rows["slurm_state"].where(rows["slurm_state"].map(text_value).ne(""), "SUBMITTED")
    rows["completed_last_step"] = rows.apply(
        lambda r: (
            f"{int(clean_float(r.get('completed_steps')))}/{int(clean_float(r.get('last_step')))}"
            if not math.isnan(clean_float(r.get("completed_steps"))) and not math.isnan(clean_float(r.get("last_step")))
            else "0/0"
        ),
        axis=1,
    )
    for col in [
        "generation_worker_tokens_per_sec_per_gpu_mean",
        "e2e_tokens_per_sec_per_gpu_mean",
        "generation_time_s_mean",
        "total_step_time_s_mean",
    ]:
        rows[col] = pd.to_numeric(rows.get(col), errors="coerce")
    for col in [
        "gen_tps_speedup",
        "e2e_tps_speedup",
        "generation_time_speedup",
        "e2e_step_time_speedup",
    ]:
        rows[col] = math.nan
    group_cols = ["model", "mode", "max_steps", "max_new_tokens", "temperature", "top_p"]
    for _, idx in rows.groupby(group_cols, dropna=False).groups.items():
        sub = rows.loc[list(idx)]
        base = sub[sub["method"].astype(str) == "baseline"]
        if base.empty:
            continue
        base = base.iloc[0]
        base_gen = clean_float(base.get("generation_worker_tokens_per_sec_per_gpu_mean"))
        base_e2e = clean_float(base.get("e2e_tokens_per_sec_per_gpu_mean"))
        base_gen_time = clean_float(base.get("generation_time_s_mean"))
        base_step_time = clean_float(base.get("total_step_time_s_mean"))
        for row_idx in idx:
            gen = clean_float(rows.at[row_idx, "generation_worker_tokens_per_sec_per_gpu_mean"])
            e2e = clean_float(rows.at[row_idx, "e2e_tokens_per_sec_per_gpu_mean"])
            gen_time = clean_float(rows.at[row_idx, "generation_time_s_mean"])
            step_time = clean_float(rows.at[row_idx, "total_step_time_s_mean"])
            if not math.isnan(base_gen) and not math.isnan(gen) and base_gen:
                rows.at[row_idx, "gen_tps_speedup"] = gen / base_gen
            if not math.isnan(base_e2e) and not math.isnan(e2e) and base_e2e:
                rows.at[row_idx, "e2e_tps_speedup"] = e2e / base_e2e
            if not math.isnan(base_gen_time) and not math.isnan(gen_time) and gen_time:
                rows.at[row_idx, "generation_time_speedup"] = base_gen_time / gen_time
            if not math.isnan(base_step_time) and not math.isnan(step_time) and step_time:
                rows.at[row_idx, "e2e_step_time_speedup"] = base_step_time / step_time
    rows = rows.sort_values(["max_steps", "method", "job_id"], ascending=[False, True, True])
    return rows


def load_lyris_historical_nemorl() -> pd.DataFrame:
    rows = []
    for path, source_group, config_basis, source_priority in NEMORL_LYRIS_HISTORICAL_SOURCES:
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        raw = raw[raw["model"].astype(str).isin(["Qwen3-30B-A3B", "Qwen3-32B"])].copy()
        for _, row in raw.iterrows():
            completed, last = parse_completed_last(row.get("completed_last_step"))
            rows.append(
                {
                    "job_id": str(row.get("job_id", "")),
                    "model": str(row.get("model", "")),
                    "model_name": str(row.get("model", "")),
                    "mode": str(row.get("mode", "")),
                    "method": str(row.get("method", "")),
                    "method_k": normalize_nemorl_method(row.get("method"), row.get("label")),
                    "max_steps": 20,
                    "max_new_tokens": clean_float(row.get("max_osl")),
                    "temperature": clean_float(row.get("temperature")),
                    "top_p": clean_float(row.get("top_p")),
                    "isl": row.get("isl", ""),
                    "cluster": "lyris",
                    "source_group": source_group,
                    "config_basis": config_basis,
                    "source_priority": source_priority,
                    "slurm_state": str(row.get("slurm_state", "")),
                    "exit_code": "",
                    "completed_steps": completed,
                    "last_step": last,
                    "completed_last_step": str(row.get("completed_last_step", "")),
                    "metric_state": str(row.get("metric_state", "")),
                    "total_step_time_s_mean": clean_float(row.get("e2e_step_time_s")),
                    "generation_time_s_mean": clean_float(row.get("generation_time_s")),
                    "e2e_tokens_per_sec_per_gpu_mean": clean_float(row.get("e2e_throughput_tok_s_gpu")),
                    "generation_worker_tokens_per_sec_per_gpu_mean": clean_float(row.get("generation_throughput_tok_s_gpu")),
                    "e2e_step_time_speedup": clean_float(row.get("e2e_step_time_speedup")),
                    "e2e_tps_speedup": clean_float(row.get("e2e_throughput_speedup")),
                    "generation_time_speedup": clean_float(row.get("generation_time_speedup")),
                    "gen_tps_speedup": clean_float(row.get("generation_throughput_speedup")),
                    "vllm_token_acceptance_pct": clean_float(row.get("acceptance_pct")),
                    "vllm_acceptance_length_mean_weighted_mean": clean_float(row.get("mean_accept_len")),
                    "manifest": str(path.relative_to(ROOT)),
                    "wandb_enabled": str(row.get("wandb_enabled", "")),
                    "wandb_project": str(row.get("wandb_project", "")),
                    "wandb_name": str(row.get("wandb_name", "")),
                    "wandb_url": normalize_wandb_url(row.get("wandb_url", "")),
                    "notes": str(row.get("notes", "")).strip(),
                    "log_path": str(row.get("source_log", "")),
                }
            )
    return pd.DataFrame(rows)


def load_oci_historical_nemorl() -> pd.DataFrame:
    if not NEMORL_OCI_HISTORICAL.exists():
        return pd.DataFrame()
    raw = pd.read_csv(NEMORL_OCI_HISTORICAL)
    raw = raw[
        raw["domain"].astype(str).eq("Math-RL")
        & raw["run_group"].astype(str).str.contains("step20 temp1 OSL1024", na=False)
        & raw["model"].astype(str).isin(["Qwen3-30B-A3B", "Qwen3-32B"])
    ].copy()
    rows = []
    for _, row in raw.iterrows():
        completed = clean_float(row.get("completed_steps"))
        max_steps = clean_float(row.get("max_steps"))
        rows.append(
            {
                "job_id": str(row.get("job_id", "")),
                "model": str(row.get("model", "")),
                "model_name": str(row.get("model", "")),
                "mode": "sync",
                "method": str(row.get("method", "")),
                "method_k": normalize_nemorl_method(row.get("method"), k=row.get("k")),
                "max_steps": max_steps,
                "max_new_tokens": clean_float(row.get("max_new_tokens")),
                "temperature": 1.0,
                "top_p": 1.0,
                "isl": "",
                "cluster": "oci-hsg",
                "source_group": "OCI-HSG Qwen30/Qwen32 Math-RL OSL1024 2026-06-16",
                "config_basis": str(row.get("config_basis", "")),
                "source_priority": 2,
                "slurm_state": str(row.get("state", "")),
                "exit_code": str(row.get("exit_code", "")),
                "completed_steps": completed,
                "last_step": clean_float(row.get("parsed_steps")),
                "completed_last_step": (
                    f"{int(completed)}/{int(max_steps)}"
                    if not math.isnan(completed) and not math.isnan(max_steps)
                    else ""
                ),
                "metric_state": str(row.get("metric_status", "")),
                "total_step_time_s_mean": clean_float(row.get("e2e_step_time_s")),
                "generation_time_s_mean": clean_float(row.get("generation_time_s")),
                "e2e_tokens_per_sec_per_gpu_mean": clean_float(row.get("e2e_tokens_per_sec_per_gpu")),
                "generation_worker_tokens_per_sec_per_gpu_mean": clean_float(row.get("generation_worker_tokens_per_sec_per_gpu")),
                "e2e_step_time_speedup": math.nan,
                "e2e_tps_speedup": clean_float(row.get("e2e_throughput_speedup")),
                "generation_time_speedup": math.nan,
                "gen_tps_speedup": clean_float(row.get("generation_throughput_speedup")),
                "vllm_token_acceptance_pct": clean_float(row.get("acceptance_rate_pct")),
                "vllm_acceptance_length_mean_weighted_mean": clean_float(row.get("mean_accepted_length")),
                "manifest": str(NEMORL_OCI_HISTORICAL.relative_to(ROOT)),
                "wandb_enabled": str(row.get("wandb_enabled", "")),
                "wandb_project": str(row.get("wandb_project", "")),
                "wandb_name": str(row.get("wandb_name", "")),
                "wandb_url": normalize_wandb_url(row.get("wandb_url", "")),
                "notes": str(row.get("notes", "")).strip(),
                "log_path": str(row.get("sources", "")),
            }
        )
    return pd.DataFrame(rows)


def load_lyris_live_k_sweep_nemorl() -> pd.DataFrame:
    summary = pd.read_csv(NEMORL_LIVE_K_SWEEP_SUMMARY) if NEMORL_LIVE_K_SWEEP_SUMMARY.exists() else pd.DataFrame()
    if not summary.empty:
        summary["job_id"] = summary["job_id"].astype(str)
        summary = summary.set_index("job_id")
    rows = []
    for meta in NEMORL_LIVE_K_SWEEP_META:
        job_id = meta["job_id"]
        metric = summary.loc[job_id].to_dict() if not summary.empty and job_id in summary.index else {}
        completed = clean_float(metric.get("completed_steps", meta.get("completed_steps", math.nan)))
        last = clean_float(metric.get("last_step", meta.get("last_step", math.nan)))
        if math.isnan(completed):
            completed = clean_float(meta.get("completed_steps", math.nan))
        if math.isnan(last):
            last = clean_float(meta.get("last_step", math.nan))
        completed_last = (
            f"{int(completed)}/20 last {int(last)}"
            if not math.isnan(completed) and not math.isnan(last) and last > 0
            else "0/20"
        )
        metric_state = str(meta.get("metric_state", ""))
        if metric and completed > 0 and metric_state in {"partial_live", ""}:
            metric_state = str(metric.get("partial_result_state", "partial_live"))
        latest_error = str(meta.get("error", "")).strip()
        if not latest_error:
            raw_error = metric.get("latest_error", "")
            if raw_error is not None and not pd.isna(raw_error):
                latest_error = str(raw_error).strip()
        rows.append(
            {
                "job_id": job_id,
                "model": meta["model"],
                "model_name": model_name(meta["model"]),
                "mode": meta["mode"],
                "method": f"Eagle-3 K={meta['k']}",
                "method_k": f"eagle3_k{meta['k']}",
                "max_steps": 20,
                "max_new_tokens": 4096,
                "temperature": 1.0,
                "top_p": 1.0,
                "enforce_eager": True,
                "isl": "performance recipe default",
                "cluster": "lyris",
                "source_group": NEMORL_LIVE_K_SWEEP_SOURCE_GROUP,
                "comparison_group": "Lyris Qwen30/Qwen32 PerfCfg OSL4096 latest-main+PR2879 2026-06-22",
                "config_basis": (
                    "performance recipe default plus latest-main+PR2879 topology-aware fix; "
                    "enforce_eager=true, prefix caching disabled, MoE backend=triton; "
                    f"context-clamp K sweep checked {NEMORL_LIVE_K_SWEEP_CHECKED_AT}"
                ),
                "source_priority": 0.5,
                "slurm_state": meta["slurm_state"],
                "exit_code": "",
                "elapsed": meta.get("elapsed", ""),
                "completed_steps": completed,
                "last_step": last,
                "completed_last_step": completed_last,
                "metric_state": metric_state,
                "total_step_time_s_mean": clean_float(metric.get("total_step_time_s_mean")),
                "generation_time_s_mean": clean_float(metric.get("generation_time_s_mean")),
                "e2e_tokens_per_sec_per_gpu_mean": clean_float(metric.get("e2e_tokens_per_sec_per_gpu_mean")),
                "generation_worker_tokens_per_sec_per_gpu_mean": clean_float(metric.get("generation_worker_tokens_per_sec_per_gpu_mean")),
                "e2e_step_time_speedup": math.nan,
                "e2e_tps_speedup": math.nan,
                "generation_time_speedup": math.nan,
                "gen_tps_speedup": math.nan,
                "vllm_token_acceptance_pct": clean_float(metric.get("vllm_token_acceptance_pct")),
                "vllm_acceptance_length_mean_weighted_mean": clean_float(metric.get("vllm_acceptance_length_mean_weighted_mean")),
                "manifest": str(NEMORL_LIVE_K_SWEEP_SUMMARY.relative_to(ROOT)) if NEMORL_LIVE_K_SWEEP_SUMMARY.exists() else "",
                "wandb_enabled": str(meta.get("wandb_enabled", metric.get("wandb_enabled", ""))),
                "wandb_project": str(meta.get("wandb_project", metric.get("wandb_project", ""))),
                "wandb_name": str(meta.get("wandb_name", metric.get("wandb_name", ""))),
                "wandb_url": normalize_wandb_url(meta.get("wandb_url", metric.get("wandb_url", ""))),
                "notes": str(meta.get("notes", "")).strip(),
                "latest_error": latest_error,
                "log_path": str(meta.get("log_path", metric.get("log_path", ""))),
                "nodes_x_gpus": meta.get("nodes_x_gpus", ""),
                "segment": meta.get("segment", ""),
            }
        )
    return pd.DataFrame(rows)


def load_nemorl_comparison_summaries() -> pd.DataFrame:
    rows = []
    for path in NEMORL_COMPARISON_SUMMARIES:
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        for _, row in raw.iterrows():
            method = row.get("method", "")
            job_id = str(row.get("job_id", ""))
            if not job_id or job_id.lower() == "nan":
                continue
            completed = row.get("completed", row.get("steps", ""))
            completed_steps, last_step = parse_completed_last(completed)
            max_steps = clean_float(row.get("max_steps"))
            if math.isnan(max_steps):
                max_steps = 20
            max_osl = clean_float(row.get("max_osl"))
            if math.isnan(max_osl):
                max_osl = 4096
            model_text = text_value(row.get("model", ""))
            if not model_text and "qwen32" in path.name.lower():
                model = "Qwen3-32B"
            else:
                model = model_name(model_text)
            source_group = "Lyris PerfCfg enforce_eager=false PARD diagnostics 2026-06-24"
            status = first_text(row, "status")
            rows.append(
                {
                    "job_id": job_id,
                    "model": model,
                    "model_name": model,
                    "mode": str(row.get("mode", "sync") or "sync"),
                    "method": str(method),
                    "method_k": normalize_nemorl_diagnostic_method(method),
                    "max_steps": max_steps,
                    "max_new_tokens": max_osl,
                    "temperature": clean_float(row.get("temp", 1.0)),
                    "top_p": clean_float(row.get("top_p", 1.0)),
                    "enforce_eager": row.get("enforce_eager", False),
                    "isl": "",
                    "cluster": "lyris",
                    "source_group": source_group,
                    "comparison_group": source_group,
                    "config_basis": (
                        "performance recipe default plus latest-main+PR2879 topology-aware fix; "
                        "enforce_eager=false, MoE backend=triton; diagnostic CSV with precomputed baseline-relative speedups"
                    ),
                    "source_priority": 0.25,
                    "slurm_state": status,
                    "exit_code": "",
                    "completed_steps": completed_steps,
                    "last_step": last_step,
                    "completed_last_step": str(completed),
                    "metric_state": status,
                    "total_step_time_s_mean": clean_float(row.get("e2e_step_time_s")),
                    "generation_time_s_mean": clean_float(row.get("generation_time_s")),
                    "e2e_tokens_per_sec_per_gpu_mean": clean_float(row.get("e2e_tps_gpu")),
                    "generation_worker_tokens_per_sec_per_gpu_mean": clean_float(row.get("generation_tps_gpu")),
                    "e2e_step_time_speedup": clean_float(row.get("e2e_step_time_vs_baseline_speedup", row.get("e2e_step_time_speedup"))),
                    "e2e_tps_speedup": clean_float(row.get("e2e_tps_vs_baseline_speedup", row.get("e2e_throughput_speedup"))),
                    "generation_time_speedup": clean_float(row.get("generation_time_vs_baseline_speedup", row.get("generation_time_speedup"))),
                    "gen_tps_speedup": clean_float(row.get("generation_tps_vs_baseline_speedup", row.get("generation_throughput_speedup"))),
                    "vllm_token_acceptance_pct": clean_float(row.get("acceptance_pct")),
                    "vllm_acceptance_length_mean_weighted_mean": clean_float(row.get("mean_accept_len")),
                    "manifest": str(path.relative_to(ROOT)),
                    "wandb_enabled": "true" if normalize_wandb_url(row.get("wandb_url", row.get("wandb_or_run", ""))) else "",
                    "wandb_project": "",
                    "wandb_name": "",
                    "wandb_url": normalize_wandb_url(row.get("wandb_url", row.get("wandb_or_run", ""))),
                    "notes": str(row.get("action_note", "")),
                    "latest_error": "",
                    "log_path": str(row.get("source", "")),
                }
            )
    return pd.DataFrame(rows)


def fill_nemorl_speedups(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    if "comparison_group" not in rows:
        rows["comparison_group"] = ""
    comparison_group = rows["comparison_group"].map(text_value)
    rows["_comparison_group"] = comparison_group.where(comparison_group.ne(""), rows["source_group"])
    for col in [
        "generation_worker_tokens_per_sec_per_gpu_mean",
        "e2e_tokens_per_sec_per_gpu_mean",
        "generation_time_s_mean",
        "total_step_time_s_mean",
        "gen_tps_speedup",
        "e2e_tps_speedup",
        "generation_time_speedup",
        "e2e_step_time_speedup",
    ]:
        rows[col] = pd.to_numeric(rows.get(col), errors="coerce")
    group_cols = ["_comparison_group", "model_name", "mode", "max_steps", "max_new_tokens", "temperature", "top_p"]
    for _, idx in rows.groupby(group_cols, dropna=False).groups.items():
        sub = rows.loc[list(idx)]
        base = sub[sub["method_k"].astype(str) == "baseline"]
        if base.empty:
            continue
        base = base.iloc[0]
        base_gen = clean_float(base.get("generation_worker_tokens_per_sec_per_gpu_mean"))
        base_e2e = clean_float(base.get("e2e_tokens_per_sec_per_gpu_mean"))
        base_gen_time = clean_float(base.get("generation_time_s_mean"))
        base_step_time = clean_float(base.get("total_step_time_s_mean"))
        for row_idx in idx:
            gen = clean_float(rows.at[row_idx, "generation_worker_tokens_per_sec_per_gpu_mean"])
            e2e = clean_float(rows.at[row_idx, "e2e_tokens_per_sec_per_gpu_mean"])
            gen_time = clean_float(rows.at[row_idx, "generation_time_s_mean"])
            step_time = clean_float(rows.at[row_idx, "total_step_time_s_mean"])
            if not math.isnan(base_gen) and not math.isnan(gen) and base_gen and math.isnan(clean_float(rows.at[row_idx, "gen_tps_speedup"])):
                rows.at[row_idx, "gen_tps_speedup"] = gen / base_gen
            if not math.isnan(base_e2e) and not math.isnan(e2e) and base_e2e and math.isnan(clean_float(rows.at[row_idx, "e2e_tps_speedup"])):
                rows.at[row_idx, "e2e_tps_speedup"] = e2e / base_e2e
            if not math.isnan(base_gen_time) and not math.isnan(gen_time) and gen_time and math.isnan(clean_float(rows.at[row_idx, "generation_time_speedup"])):
                rows.at[row_idx, "generation_time_speedup"] = base_gen_time / gen_time
            if not math.isnan(base_step_time) and not math.isnan(step_time) and step_time and math.isnan(clean_float(rows.at[row_idx, "e2e_step_time_speedup"])):
                rows.at[row_idx, "e2e_step_time_speedup"] = base_step_time / step_time
    return rows.drop(columns=["_comparison_group"], errors="ignore")


def combine_nemorl_rows(live_rows: pd.DataFrame) -> pd.DataFrame:
    parts = [
        part
        for part in [
            live_rows,
            load_nemorl_comparison_summaries(),
            load_lyris_live_k_sweep_nemorl(),
            load_lyris_historical_nemorl(),
            load_oci_historical_nemorl(),
        ]
        if not part.empty
    ]
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True, sort=False)
    rows = fill_nemorl_speedups(rows)
    rows["has_speedup_metric"] = pd.to_numeric(rows.get("gen_tps_speedup"), errors="coerce").notna()
    rows["completed_steps_numeric"] = pd.to_numeric(rows.get("completed_steps"), errors="coerce").fillna(0)
    rows = rows.sort_values(
        ["source_group", "job_id", "method_k", "has_speedup_metric", "completed_steps_numeric"],
        ascending=[True, True, True, False, False],
        na_position="last",
    )
    rows = rows.drop_duplicates(subset=["source_group", "job_id", "method_k"], keep="first")
    rows = rows.sort_values(
        [
            "source_priority",
            "model_name",
            "mode",
            "max_new_tokens",
            "method_k",
            "job_id",
        ],
        ascending=[True, True, True, True, True, True],
        na_position="last",
    )
    return rows.drop(columns=["has_speedup_metric", "completed_steps_numeric"], errors="ignore")


def nemorl_live_k_sweep_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "job_id" not in rows:
        return pd.DataFrame()
    ids = {item["job_id"] for item in NEMORL_LIVE_K_SWEEP_META}
    live = rows[rows["job_id"].astype(str).isin(ids)].copy()
    if live.empty:
        return live
    live["k_sort"] = live["method_k"].astype(str).str.extract(r"k(\d+)").astype(float)
    mode_rank = {"sync": 0, "async-1off": 1}
    live["mode_rank"] = live["mode"].astype(str).map(mode_rank).fillna(9)
    live["model_rank"] = live["model_name"].astype(str).map({"Qwen3-30B-A3B": 0, "Qwen3-32B": 1}).fillna(9)
    return live.sort_values(["model_rank", "mode_rank", "k_sort", "job_id"], na_position="last")


def nemorl_fresh_finding(live_rows: pd.DataFrame) -> str:
    if live_rows.empty:
        return "No fresh K-sweep rows were available in the local artifacts."
    clean = live_rows[
        (live_rows["mode"].astype(str) == "sync")
        & pd.to_numeric(live_rows.get("completed_steps"), errors="coerce").fillna(0).gt(0)
    ].copy()
    if clean.empty:
        return "Fresh K-sweep jobs are submitted, but no sync row has completed enough steps for timing metrics yet."
    clean["gen_tps_speedup"] = pd.to_numeric(clean.get("gen_tps_speedup"), errors="coerce")
    clean["e2e_tps_speedup"] = pd.to_numeric(clean.get("e2e_tps_speedup"), errors="coerce")
    clean = clean.sort_values("gen_tps_speedup", ascending=False)
    best = clean.iloc[0]
    q32 = clean[clean["model_name"].astype(str) == "Qwen3-32B"]
    q32_text = ""
    if not q32.empty:
        q32_bits = [
            f"{nemorl_method_label(row.method_k)} {fmt_x(row.gen_tps_speedup)} gen"
            for row in q32.itertuples()
            if not math.isnan(clean_float(row.gen_tps_speedup))
        ]
        if q32_bits:
            q32_text = " Qwen3-32B partial sync rows: " + ", ".join(q32_bits) + "."
    return (
        f"Fresh K-sweep signal: {best['model_name']} {best['mode']} {nemorl_method_label(best['method_k'])} "
        f"reached {best['completed_last_step']} with {fmt_x(best['gen_tps_speedup'])} generation throughput "
        f"and {fmt_x(best['e2e_tps_speedup'])} E2E throughput vs the matched OSL4096 baseline."
        + q32_text
    )


def chapter_card(title: str, body: str, href: str) -> str:
    return (
        f'<a class="chapter-card" href="{esc(href)}">'
        f"<strong>{esc(title)}</strong><span>{esc(body)}</span></a>"
    )


def build_nemorl_html(rows: pd.DataFrame) -> str:
    updated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = rows.copy() if not rows.empty else rows
    if not rows.empty:
        rows["method_display"] = rows["method_k"].map(nemorl_method_label)
    running = int((rows.get("slurm_state", pd.Series(dtype=str)).astype(str) == "RUNNING").sum()) if not rows.empty else 0
    pending = int((rows.get("slurm_state", pd.Series(dtype=str)).astype(str) == "PENDING").sum()) if not rows.empty else 0
    completed_metric = int(pd.to_numeric(rows.get("completed_steps"), errors="coerce").fillna(0).gt(0).sum()) if not rows.empty else 0
    current = rows[pd.to_numeric(rows.get("max_steps"), errors="coerce") == 20].copy() if not rows.empty else pd.DataFrame()
    smoke = rows[pd.to_numeric(rows.get("max_steps"), errors="coerce") == 3].copy() if not rows.empty else pd.DataFrame()
    live_k = nemorl_live_k_sweep_rows(rows)
    fresh_key = nemorl_fresh_finding(live_k)
    async_engine_errors = int(
        (
            live_k.get("metric_state", pd.Series(dtype=str)).astype(str).str.contains("engine_error", na=False)
        ).sum()
    ) if not live_k.empty else 0
    best = current[~current["method_k"].astype(str).str.startswith("baseline")].copy()
    best = best[pd.to_numeric(best["completed_steps"], errors="coerce").fillna(0) > 0]
    best = best[pd.to_numeric(best["gen_tps_speedup"], errors="coerce").notna()]
    if not best.empty:
        top = best.sort_values("gen_tps_speedup", ascending=False).iloc[0]
        key = (
            f"Best parsed NeMo-RL step20 row is {top['model_name']} {top['mode']} {nemorl_method_label(top['method_k'])} "
            f"({top['source_group']}) with "
            f"{fmt_x(top['gen_tps_speedup'])} generation throughput speedup and "
            f"{fmt_x(top['e2e_tps_speedup'])} E2E throughput speedup vs the matched baseline snapshot."
        )
    else:
        key = "Step20 rows are running or pending; matched speedup will update as baseline and spec rows complete more steps."
    css = """
:root{--ink:#111827;--muted:#5f6b7a;--line:#d6dee9;--bg:#f4f6f9;--panel:#fff;--soft:#eef3f8;--blue:#2457a6;--green:#157f47;--amber:#946200;--red:#b42318}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;font:15px/1.48 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;color:var(--ink);background:var(--bg)}header{background:linear-gradient(180deg,#ffffff 0,#f8fafc 100%);border-bottom:1px solid var(--line)}.hero{max-width:1480px;margin:0 auto;padding:26px 28px 18px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--blue);margin-bottom:8px}main{max-width:1480px;margin:0 auto;padding:20px 28px 42px}h1{margin:0 0 8px;font-size:34px;line-height:1.12;letter-spacing:0}h2{margin:0 0 12px;font-size:21px}h3{margin:18px 0 6px;font-size:16px}.subtitle,.note{color:var(--muted)}.toc{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}.toc a{border:1px solid var(--line);background:#fff;color:#263448;text-decoration:none;border-radius:6px;padding:7px 10px;font-size:13px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:4px 9px;margin:2px 4px 2px 0;background:#fff}.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:12px 0 18px}.kpi{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px}.kpi b{display:block;font-size:24px;line-height:1.05}.kpi span{color:var(--muted)}section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px;margin:0 0 18px}.chapter-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.chapter-card{display:block;text-decoration:none;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:8px;padding:13px}.chapter-card strong{display:block;margin-bottom:5px}.chapter-card span{display:block;color:var(--muted);font-size:13px}.callout{border-left:4px solid var(--blue);background:#f8fbff;padding:12px 14px;border-radius:6px;margin:10px 0}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:12px}.model-charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:10px 0 18px}.chart-card{border:1px solid var(--line);border-radius:8px;background:#fff;padding:8px;min-width:0}.chart-card svg{width:100%;height:auto;display:block}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:#fff}th,td{border:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}th{background:#eef2f7;font-size:13px}.num{text-align:right;font-variant-numeric:tabular-nums}.RUNNING,.COMPLETED{color:var(--green);font-weight:700}.PENDING,.SUBMITTED{color:var(--amber);font-weight:700}.FAILED,.TIMEOUT,.CANCELLED{color:var(--red);font-weight:700}code{background:#f3f4f6;padding:1px 4px;border-radius:4px}@media(max-width:1100px){.charts,.model-charts,.chapter-grid{grid-template-columns:1fr 1fr}.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:900px){.hero,main{padding-left:16px;padding-right:16px}.model-charts,.kpis,.chapter-grid{grid-template-columns:1fr}h1{font-size:28px}table{font-size:13px}}@media(max-width:620px){.charts,.kpis,.chapter-grid{grid-template-columns:1fr}}"""
    cols = [
        ("source_group", "Source group", "text"),
        ("cluster", "Cluster", "text"),
        ("job_id", "Job", "text"),
        ("wandb_url", "W&B", "link"),
        ("wandb_name", "W&B name", "text"),
        ("model_name", "Model", "text"),
        ("mode", "Mode", "text"),
        ("method_display", "Method", "text"),
        ("enforce_eager", "enforce_eager", "text"),
        ("max_steps", "Max steps", "int"),
        ("max_new_tokens", "Max OSL", "int"),
        ("slurm_state", "SLURM", "text"),
        ("completed_last_step", "completed/last", "text"),
        ("total_step_time_s_mean", "E2E step", "num"),
        ("e2e_step_time_speedup", "Step-time speedup", "x"),
        ("e2e_tokens_per_sec_per_gpu_mean", "E2E tok/s/GPU", "num"),
        ("e2e_tps_speedup", "E2E tput speedup", "x"),
        ("generation_time_s_mean", "Generation time", "num"),
        ("generation_time_speedup", "Gen-time speedup", "x"),
        ("generation_worker_tokens_per_sec_per_gpu_mean", "Gen tok/s/GPU", "num"),
        ("gen_tps_speedup", "Gen tput speedup", "x"),
        ("vllm_token_acceptance_pct", "Acceptance", "pct"),
        ("vllm_acceptance_length_mean_weighted_mean", "Mean len", "num"),
        ("manifest", "Manifest", "text"),
    ]
    live_cols = [
        ("job_id", "Job", "text"),
        ("wandb_url", "W&B", "link"),
        ("wandb_name", "W&B name", "text"),
        ("model_name", "Model", "text"),
        ("mode", "Mode", "text"),
        ("method_display", "Method", "text"),
        ("enforce_eager", "enforce_eager", "text"),
        ("nodes_x_gpus", "Nodes x GPUs", "text"),
        ("segment", "segment", "int"),
        ("slurm_state", "SLURM", "text"),
        ("completed_last_step", "completed/last", "text"),
        ("generation_worker_tokens_per_sec_per_gpu_mean", "Gen tok/s/GPU", "num"),
        ("gen_tps_speedup", "Gen tput speedup", "x"),
        ("generation_time_s_mean", "Gen time", "num"),
        ("generation_time_speedup", "Gen-time speedup", "x"),
        ("e2e_tokens_per_sec_per_gpu_mean", "E2E tok/s/GPU", "num"),
        ("e2e_tps_speedup", "E2E tput speedup", "x"),
        ("total_step_time_s_mean", "E2E step", "num"),
        ("e2e_step_time_speedup", "E2E step speedup", "x"),
        ("vllm_token_acceptance_pct", "Acceptance", "pct"),
        ("vllm_acceptance_length_mean_weighted_mean", "Mean len", "num"),
        ("metric_state", "Metric state", "text"),
        ("notes", "Notes", "text"),
        ("latest_error", "First severe error", "text"),
    ]
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
            f"<title>Lyris NeMo-RL SpecDec Status Latest</title><style>{css}</style></head><body>",
            "<header><div class=\"hero\"><div class=\"eyebrow\">LIVE REPORT · SPECULATIVE DECODING · 2026</div><h1>Lyris NeMo-RL SpecDec Status</h1>",
            f"<div class=\"subtitle\">Updated {esc(updated)}. Fresh K-sweep check: {esc(NEMORL_LIVE_K_SWEEP_CHECKED_AT)}. Data covers Qwen3-235B PR2879/latest-main rows, 2026-06-23 enforce_eager=false W&B rows, 2026-06-24 PARD diagnostics, and historical Qwen3-30B-A3B/Qwen3-32B Lyris/OCI-HSG artifacts.</div>",
            "<nav class=\"toc\"><a href=\"#overview\">Overview</a><a href=\"#fresh\">Fresh enforce_eager=true K Sweep</a><a href=\"#methodology\">Methodology</a><a href=\"#charts\">Charts</a><a href=\"#step20\">Step20 Tables</a><a href=\"#smoke\">Step3 Smoke</a><a href=\"#sources\">Sources</a></nav></div></header><main>",
            "<div><span class=\"pill\">performance recipe configs</span><span class=\"pill\">temperature=1.0</span><span class=\"pill\">top_p=1.0</span><span class=\"pill\">enforce_eager shown per row</span><span class=\"pill\">Max OSL separated by section</span><span class=\"pill\">step>=2 metrics where noted</span><span class=\"pill\">GB200 segment captured</span></div>",
            "<div class=\"kpis\">",
            f"<div class=\"kpi\"><b>{running}</b><span>running jobs</span></div>",
            f"<div class=\"kpi\"><b>{pending}</b><span>pending jobs</span></div>",
            f"<div class=\"kpi\"><b>{async_engine_errors}</b><span>async rows with engine errors</span></div>",
            f"<div class=\"kpi\"><b>{completed_metric}</b><span>rows with completed steps</span></div>",
            f"<div class=\"kpi\"><b>{len(rows)}</b><span>tracked rows</span></div>",
            "</div>",
            "<section id=\"overview\"><h2>Overview</h2>",
            f"<div class=\"callout\"><strong>Key finding.</strong> {esc(key)}<br><strong>Fresh update.</strong> {esc(fresh_key)}</div>",
            "<div class=\"chapter-grid\">",
            chapter_card("Fresh enforce_eager=true K Sweep", "Newest Lyris Eagle-3 K5/K7/K9 state and parsed speedups.", "#fresh"),
            chapter_card("Matched Charts", "Generation/E2E throughput and step-time speedups by model.", "#charts"),
            chapter_card("Step20 Snapshot", "All current and historical step20 rows with acceptance metrics.", "#step20"),
            chapter_card("Raw Evidence", "CSV, log path, and source provenance links for reproducibility.", "#sources"),
            "</div></section>",
            "<section id=\"fresh\"><h2>Fresh enforce_eager=true K Sweep</h2>",
            "<p class=\"note\">Newest Lyris run set for Eagle-3 K5/K7/K9 on performance recipes with <code>policy.generation.vllm_cfg.enforce_eager=true</code>, prefix caching disabled, and MoE backend=triton. Sync rows with completed steps are baseline-relative against the matched 2026-06-22 OSL4096 baseline rows. Async timeout rows are listed for status but should not be treated as clean performance data while EngineCore errors are present.</p><div class=\"table-wrap\">",
            table(live_k, live_cols),
            "</div></section>",
            "<section id=\"methodology\"><h2>Evaluation Methodology</h2><ul>",
            "<li>Recipes: NeMo-RL <code>examples/configs/recipes/llm/performance</code>.</li>",
            "<li>Matched comparisons keep model, mode, max OSL, temperature=1.0, top_p=1.0, and cluster/source setup fixed.</li>",
            "<li>SpecDec rows add only the generation speculative decoding method, drafter/checkpoint, and <code>num_speculative_tokens</code>; baseline rows use the same recipe with SpecDec disabled.</li>",
            "<li>Fresh 2026-06-22 Qwen3-30B-A3B/Qwen3-32B Lyris rows use latest-main+PR2879, recipe OSL4096, and step2-20 averages where available.</li>",
            "<li>The fresh K-sweep section is explicitly <code>enforce_eager=true</code>; the pending W&B matrix rows are separately labeled <code>enforce_eager=false</code> when present.</li>",
            "</ul></section>",
            f"<section><h2>Metric Notes</h2><p>{esc(fresh_key)}</p><p class=\"note\">Acceptance metrics are shown only when the NeMo-RL driver log includes vLLM SpecDec metrics; Qwen3-235B current driver snapshots mostly expose timing/throughput, while historical Qwen30/Qwen32 rows include acceptance when available.</p></section>",
            '<div id="charts">',
            nemorl_charts_section(rows),
            "</div>",
            "<section id=\"step20\"><h2>Step20 Current And Historical Snapshot</h2><div class=\"table-wrap\">",
            table(current, cols),
            "</div></section>",
            "<section id=\"smoke\"><h2>Step3 Smoke / K Sweep</h2><div class=\"table-wrap\">",
            table(smoke, cols),
            "</div></section>",
            "<section id=\"sources\"><h2>Sources</h2><p class=\"note\"><code>docs/lyris_nemorl_qwen30_qwen32_eagle3_k_sweep_live_summary_20260622.csv</code>, <code>docs/lyris_qwen235b_pr2879_live_summary_skip_step1_20260621.csv</code>, <code>docs/lyris_20260623_current_plus_eagerfalse_summary_skip_step1.csv</code>, <code>docs/qwen32_pardk1_20260624_summary_skip1_latest.csv</code>, <code>docs/qwen32_pard_eagerfalse_compare_20260624.csv</code>, <code>docs/nemorl_specdec_slowdown_watchlist_20260624.csv</code>, <code>docs/lyris_qwen235b_pr2879_sacct_20260621.psv</code>, <code>latest_lyris_nemorl_*20260621-20260625_jobs.csv</code>, <code>docs/lyris_nemorl_qwen30_qwen32_pr2879_step20_speedups_20260622.csv</code>, <code>docs/lyris_nemorl_qwen30_qwen32_pr2879_status_20260622.csv</code>, <code>docs/lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv</code>, and <code>docs/nemorl_integrated_specdec_results_clean_20260617.csv</code>.</p></section>",
            "</main></body></html>",
        ]
    )


def main() -> None:
    main_vllm = pd.read_csv(MAIN_VLLM)
    added = load_vllm_added(main_vllm)
    added.to_csv(VLLM_ADDED_OUT, index=False)
    vllm_html = build_vllm_html(main_vllm, added)
    VLLM_HTML_DATED.write_text(vllm_html, encoding="utf-8")
    shutil.copyfile(VLLM_HTML_DATED, VLLM_HTML_LATEST)

    nemorl_rows = enrich_nemorl()
    nemorl_rows.to_csv(NEMORL_OUT, index=False)
    nemorl_combined = combine_nemorl_rows(nemorl_rows)
    nemorl_combined.to_csv(NEMORL_COMBINED_OUT, index=False)
    nemorl_html = build_nemorl_html(nemorl_combined)
    NEMORL_HTML_DATED.write_text(nemorl_html, encoding="utf-8")
    shutil.copyfile(NEMORL_HTML_DATED, NEMORL_HTML)

    print(VLLM_ADDED_OUT)
    print(VLLM_HTML_LATEST)
    print(NEMORL_OUT)
    print(NEMORL_COMBINED_OUT)
    print(NEMORL_HTML)


if __name__ == "__main__":
    main()
