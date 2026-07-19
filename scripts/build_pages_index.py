#!/usr/bin/env python3
# pyright: reportCallIssue=false, reportArgumentType=false, reportAssignmentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportGeneralTypeIssues=false
"""Build the GitLab Pages landing page for SpecDec RL benchmark results."""

from __future__ import annotations

import html
import math
import shutil
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PUBLIC = ROOT / "public"
REPORTS = PUBLIC / "reports"
DATA = PUBLIC / "data"
ARCHIVE = PUBLIC / "archive"
FIGURES = PUBLIC / "figures"
DFLARE_COMPLETED = (
    ROOT / "experiments/vllm_024_dynamicsd/report/dflare_completed_latest.csv"
)
DFLARE_STATUS = (
    ROOT / "experiments/vllm_024_dynamicsd/report/dflare_job_status_latest.csv"
)
VLLM024_PROFILES = (
    ROOT / "experiments/vllm_024_dynamicsd/report/vllm024_profiles_latest.csv"
)
DFLARE_PROFILES = {"Native 32K", "YaRN 64K", "YaRN total-128K"}

LOCAL_REF_SKIP_SCHEMES = {"http", "https", "mailto", "tel", "ftp", "javascript", "data"}

MODELS = ["Qwen3-30B-A3B", "Qwen3-32B", "Qwen3-235B-A22B"]
MODEL_SHORT = {
    "Qwen3-30B-A3B": "30B-A3B",
    "Qwen3-32B": "32B",
    "Qwen3-235B-A22B": "235B",
}
PAIRED = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]

REPORT_GROUPS = [
    {
        "title": "DynamicSD (vLLM 0.24)",
        "summary": "Dynamic speculative decoding under synchronous RL rollout: BSxK profiling grids, derived K schedules, and baseline/fixed-K/DynamicSD rollout comparisons.",
        "items": [
            (
                "DynamicSD sync-rollout results",
                "dynamic_sd_sync_rollout_results_latest.html",
                "Qwen3-30B-A3B/32B/235B with EAGLE3 Thinking drafters on GB200; NeMo-RL SyncRL recipe shapes.",
            ),
            (
                "NemoGym SWE rollout inefficiency report",
                "nemogym_swe_efficiency_report.html",
                "Measured timeline decomposition (PR #3243 + PR #1825 profiler), 8 inefficiencies with code segments, fix plan with trade-off and correctness analysis.",
            ),
        ],
    },
    {
        "title": "vLLM Standalone",
        "summary": "Standalone vLLM benchmark views for Math/SWE, temperature 0/1, batch sweeps, and Qwen235B focused diagnostics.",
        "items": [
            (
                "Canonical latest matched matrix",
                "vllm_standalone_results_latest.html",
                "Current ISL4096/OSL32768 matched-baseline view; use this as the canonical standalone page.",
            ),
            (
                "Latest dated mirror",
                "vllm_standalone_results_20260621.html",
                "Dated mirror of the canonical latest page.",
            ),
            (
                "6/20 historical page",
                "vllm_standalone_results_20260620.html",
                "Superseded historical added-result matrix; retained for traceability.",
            ),
            (
                "6/19 batch matrix",
                "vllm_standalone_results_20260619.html",
                "All-batch standalone report before later refreshes.",
            ),
            (
                "Clean result split",
                "vllm_standalone_clean_results_20260617.html",
                "Curated primary and supplemental standalone results.",
            ),
            (
                "Temp0 vs Temp1 trends",
                "vllm_standalone_temp0_temp1_trends_20260616.html",
                "Historical aggregate temperature analysis; later standalone additions are in the canonical latest page.",
            ),
            (
                "Qwen235B SWE batch sweep",
                "lyris_qwen235b_swebench_osl32k_batch_sweep_speedups_20260612.html",
                "Dedicated SWE OSL32K batch-sweep speedups.",
            ),
            (
                "Qwen235B diagnostics",
                "lyris_qwen235b_standalone_live_diagnostics_20260613.html",
                "Older live diagnostics for Qwen235B standalone jobs.",
            ),
            (
                "Qwen235B PARD snapshot",
                "lyris_qwen235b_swebench_osl32k_pard_live_snapshot_20260613.html",
                "PARD-focused live snapshot.",
            ),
            (
                "Expected performance",
                "lyris_specdec_expected_performance_20260612.html",
                "Early expected-performance summary.",
            ),
            (
                "Qwen235B SWE/Math status",
                "qwen235b_specdec_swe_math_status_20260613.html",
                "Combined historical SWE and Math status page.",
            ),
        ],
    },
    {
        "title": "NeMo-RL",
        "summary": "NeMo-RL performance recipe and SpecDec integration pages, including live Lyris and OCI-HSG status snapshots.",
        "items": [
            (
                "Cross-framework lessons and NeMo-RL gaps",
                "specdec_rl_framework_lessons_and_nemorl_gaps_20260709.html",
                "Primary upstream lessons matrix covering veRL, slime, Miles, SGLang/vLLM, and current NeMo-RL gaps.",
            ),
            (
                "Latest NeMo-RL status",
                "lyris_nemorl_perfcfg_specdec_live_status_latest.html",
                "Current performance-config SpecDec status.",
            ),
            (
                "6/22 NeMo-RL status",
                "lyris_nemorl_perfcfg_specdec_live_status_20260622.html",
                "Dated latest NeMo-RL page.",
            ),
            (
                "6/21 NeMo-RL status",
                "lyris_nemorl_perfcfg_specdec_live_status_20260621.html",
                "Previous Lyris status snapshot.",
            ),
            (
                "6/19 NeMo-RL status",
                "lyris_nemorl_perfcfg_specdec_live_status_20260619.html",
                "Older performance-config status.",
            ),
            (
                "6/18 OSL step20 matrix",
                "lyris_nemorl_perfcfg_specdec_live_status_20260618.html",
                "Current recipe OSL step20 matrix snapshot.",
            ),
            (
                "PARD/PARD-2 status",
                "nemorl_pard_pard2_status_20260615.html",
                "Focused PARD and PARD-2 integration status.",
            ),
            (
                "OCI-HSG Math RL",
                "oci_hsg_mathrl_multimodel_specdec_step20_status_20260616.html",
                "OCI-HSG Math RL multimodel step20 page.",
            ),
        ],
    },
    {
        "title": "Broad Dashboards And Background",
        "summary": "Cross-cutting dashboards, clean summaries, and older background reports that preserve context outside the latest matched matrix.",
        "items": [
            (
                "Broad metrics dashboard",
                "specdec_benchmark_metrics_dashboard_20260616.html",
                "Wide dashboard with vLLM, Math, SWE, and status fragments.",
            ),
            (
                "Clean benchmark results",
                "specdec_clean_benchmark_results_20260617.html",
                "Clean combined benchmark report.",
            ),
            (
                "Background observations",
                "specdec_background_and_observations_charts.html",
                "Early charts and observations.",
            ),
            (
                "Completed eval bars",
                "specdec_completed_eval_bar_graphs.html",
                "Older completed evaluation bar charts.",
            ),
            (
                "Qwen235B team report",
                "qwen3_235b_team_report_20260606.html",
                "Historical team-facing Qwen235B report.",
            ),
        ],
    },
]

PINNED_REPORT_FILES = {
    "specdec_rl_framework_lessons_and_nemorl_gaps_20260709.html",
    "vllm_standalone_results_20260619.html",
}

PRIMARY_LINKS = [
    (
        "Cross-framework lessons",
        "reports/specdec_rl_framework_lessons_and_nemorl_gaps_20260709.html",
    ),
    ("vLLM standalone latest", "reports/vllm_standalone_results_latest.html"),
    (
        "DFlare result-table design",
        "specs/2026-07-03-dflare-html-results-design.html",
    ),
    (
        "vLLM standalone 6/19 all-batch matrix",
        "reports/vllm_standalone_results_20260619.html",
    ),
    (
        "NeMo-RL latest",
        "reports/lyris_nemorl_perfcfg_specdec_live_status_latest.html",
    ),
    (
        "Combined clean report",
        "reports/specdec_clean_benchmark_results_20260617.html",
    ),
    (
        "Qwen235B SWE/Math historical",
        "reports/qwen235b_specdec_swe_math_status_20260613.html",
    ),
    (
        "Temp0/Temp1 trends",
        "reports/vllm_standalone_temp0_temp1_trends_20260616.html",
    ),
    ("Archive: old Eagle3 report", "archive/specdec_math_progress_report.html"),
]

REPORT_FILE_NAMES = sorted(
    {filename for group in REPORT_GROUPS for _, filename, _ in group["items"]}
)

REPORT_COMPANION_FILES = [
    "qwen235b_specdec_swe_math_status_20260613.csv",
    "qwen235b_specdec_swe_math_status_20260613.png",
]


class LocalRefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.refs.append(value)


def esc(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return html.escape(text)


def as_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    num = as_float(value)
    if math.isnan(num):
        return "n/a"
    return f"{num:.{digits}f}{suffix}"


def fmt_int(value: object) -> str:
    num = as_float(value)
    if math.isnan(num):
        return "n/a"
    return str(int(round(num)))


def copy_if_exists(src: Path, dst_dir: Path) -> Path | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    if dst.suffix in {".csv", ".html", ".json", ".txt"}:
        raw = dst.read_bytes()
        dst.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return dst


def normalize_text_file(path: Path) -> None:
    if path.suffix in {
        ".csv",
        ".html",
        ".json",
        ".txt",
        ".md",
        ".py",
        ".sh",
        ".yaml",
        ".yml",
    }:
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def resolve_local_ref(base: Path, ref: str) -> Path | None:
    if not ref or ref.startswith("#") or ref.startswith("//"):
        return None
    parsed = urlsplit(ref)
    if parsed.scheme.lower() in LOCAL_REF_SKIP_SCHEMES:
        return None
    if parsed.path.startswith("/") or not parsed.path:
        return None
    return (base.parent / unquote(parsed.path)).resolve()


def public_destination_for_source(src: Path) -> Path | None:
    src = src.resolve()
    if src.suffix.lower() == ".html":
        return None
    try:
        return REPORTS / src.relative_to(DOCS)
    except ValueError:
        pass
    try:
        return PUBLIC / src.relative_to(ROOT)
    except ValueError:
        return None


def copy_report_local_refs(report_src: Path) -> list[Path]:
    if not report_src.exists():
        return []
    parser = LocalRefParser()
    parser.feed(report_src.read_text(encoding="utf-8", errors="ignore"))
    copied: list[Path] = []
    seen: set[Path] = set()
    for ref in parser.refs:
        src = resolve_local_ref(report_src, ref)
        if src is None or src in seen or not src.exists() or src.is_dir():
            continue
        seen.add(src)
        dst = public_destination_for_source(src)
        if dst is None:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        normalize_text_file(dst)
        copied.append(dst)
    return copied


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def latest_dflare_performance_row(rows: pd.DataFrame) -> pd.Series:
    numeric_job_ids = cast(pd.Series, pd.to_numeric(rows["job_id"], errors="coerce"))
    if bool(numeric_job_ids.notna().any()):
        return rows.loc[int(numeric_job_ids.idxmax())]
    return rows.iloc[-1]


def vllm_best_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rel in [
        "vllm_standalone_all_batches_combined_20260619.csv",
        "vllm_standalone_added_results_latest.csv",
    ]:
        path = DOCS / rel
        df = load_csv(path)
        if df.empty:
            continue
        if "valid_result" in df.columns:
            df = df[df["valid_result"].astype(str).str.lower().eq("true")]
        df["source_file"] = f"docs/{rel}"
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    rows = pd.concat(frames, ignore_index=True, sort=False)
    rows = rows[rows["model"].isin(MODELS)]
    rows = rows[rows["method"].astype(str) != "baseline"]
    for column in [
        "temperature",
        "top_p",
        "batch_size",
        "isl",
        "osl",
        "tok_s_gpu",
        "speedup",
        "acceptance_pct",
        "mean_accept_len",
    ]:
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=["speedup"])
    if rows.empty:
        return rows
    idx = rows.groupby(["model", "domain", "temperature"], dropna=False)[
        "speedup"
    ].idxmax()
    keep = [
        "domain",
        "model",
        "temperature",
        "method",
        "batch_size",
        "isl",
        "osl",
        "tok_s_gpu",
        "speedup",
        "acceptance_pct",
        "mean_accept_len",
        "source_file",
    ]
    return rows.loc[idx, keep].sort_values(["model", "domain", "temperature"])


def load_nemorl_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    perf_sources = [
        DOCS / "lyris_nemorl_qwen30_qwen32_pr2879_step20_speedups_20260622.csv",
        DOCS / "lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv",
    ]
    for perf_path in perf_sources:
        perf = load_csv(perf_path)
        if perf.empty:
            continue
        perf = perf[perf["model"].isin(["Qwen3-30B-A3B", "Qwen3-32B"])]
        perf = perf[
            ~perf["method"].astype(str).str.contains("baseline", case=False, na=False)
        ]
        for column in [
            "generation_throughput_speedup",
            "generation_time_speedup",
            "e2e_throughput_speedup",
            "e2e_step_time_speedup",
            "acceptance_pct",
            "mean_accept_len",
            "generation_throughput_tok_s_gpu",
            "generation_time_s",
            "e2e_step_time_s",
        ]:
            perf[column] = pd.to_numeric(perf[column], errors="coerce")
        for _, row in perf.dropna(subset=["generation_throughput_speedup"]).iterrows():
            rows.append(
                {
                    "source_file": f"docs/{perf_path.name}",
                    "job_id": row["job_id"],
                    "model": row["model"],
                    "mode": row["mode"],
                    "method": row["method"],
                    "completed": row["completed_last_step"],
                    "max_osl": row["max_osl"],
                    "gen_tps": row["generation_throughput_tok_s_gpu"],
                    "gen_tps_speedup": row["generation_throughput_speedup"],
                    "gen_time_speedup": row["generation_time_speedup"],
                    "e2e_tps_speedup": row["e2e_throughput_speedup"],
                    "e2e_step_speedup": row["e2e_step_time_speedup"],
                    "acceptance_pct": row["acceptance_pct"],
                    "mean_accept_len": row["mean_accept_len"],
                }
            )

    q235 = load_csv(DOCS / "lyris_qwen235b_pr2879_live_enriched_20260621.csv")
    if not q235.empty:
        q235 = q235[q235["model_name"].eq("Qwen3-235B-A22B")]
        q235 = q235[
            ~q235["method_k"].astype(str).str.contains("baseline", case=False, na=False)
        ]
        for column in [
            "gen_tps_speedup",
            "generation_time_speedup",
            "e2e_tps_speedup",
            "e2e_step_time_speedup",
            "generation_worker_tokens_per_sec_per_gpu_mean",
            "vllm_token_acceptance_pct",
            "vllm_acceptance_length_mean_weighted_mean",
        ]:
            q235[column] = pd.to_numeric(q235[column], errors="coerce")
        for _, row in q235.dropna(subset=["gen_tps_speedup"]).iterrows():
            rows.append(
                {
                    "source_file": "docs/lyris_qwen235b_pr2879_live_enriched_20260621.csv",
                    "job_id": row["job_id"],
                    "model": row["model_name"],
                    "mode": row["mode"],
                    "method": row["method_k"],
                    "completed": row["completed_last_step"],
                    "max_osl": row["max_new_tokens"],
                    "gen_tps": row["generation_worker_tokens_per_sec_per_gpu_mean"],
                    "gen_tps_speedup": row["gen_tps_speedup"],
                    "gen_time_speedup": row["generation_time_speedup"],
                    "e2e_tps_speedup": row["e2e_tps_speedup"],
                    "e2e_step_speedup": row["e2e_step_time_speedup"],
                    "acceptance_pct": row.get("vllm_token_acceptance_pct", math.nan),
                    "mean_accept_len": row.get(
                        "vllm_acceptance_length_mean_weighted_mean", math.nan
                    ),
                }
            )

    combined = load_csv(DOCS / "lyris_nemorl_perfcfg_specdec_combined_latest.csv")
    if not combined.empty:
        fresh_ids = {str(job_id) for job_id in range(2177867, 2177879)}
        fresh = combined[combined["job_id"].astype(str).isin(fresh_ids)].copy()
        fresh = fresh[fresh["mode"].astype(str).eq("sync")]
        for column in [
            "gen_tps_speedup",
            "generation_time_speedup",
            "e2e_tps_speedup",
            "e2e_step_time_speedup",
            "generation_worker_tokens_per_sec_per_gpu_mean",
            "vllm_token_acceptance_pct",
            "vllm_acceptance_length_mean_weighted_mean",
        ]:
            fresh[column] = pd.to_numeric(fresh[column], errors="coerce")
        for _, row in fresh.dropna(subset=["gen_tps_speedup"]).iterrows():
            rows.append(
                {
                    "source_file": "docs/lyris_nemorl_perfcfg_specdec_combined_latest.csv",
                    "job_id": row["job_id"],
                    "model": row["model_name"],
                    "mode": row["mode"],
                    "method": row["method_k"],
                    "completed": row["completed_last_step"],
                    "max_osl": row["max_new_tokens"],
                    "gen_tps": row["generation_worker_tokens_per_sec_per_gpu_mean"],
                    "gen_tps_speedup": row["gen_tps_speedup"],
                    "gen_time_speedup": row["generation_time_speedup"],
                    "e2e_tps_speedup": row["e2e_tps_speedup"],
                    "e2e_step_speedup": row["e2e_step_time_speedup"],
                    "acceptance_pct": row.get("vllm_token_acceptance_pct", math.nan),
                    "mean_accept_len": row.get(
                        "vllm_acceptance_length_mean_weighted_mean", math.nan
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def nemorl_best_rows() -> pd.DataFrame:
    df = load_nemorl_rows()
    if df.empty:
        return df
    idx = df.groupby("model")["gen_tps_speedup"].idxmax()
    return df.loc[idx].sort_values("model")


def short_model(value: object) -> str:
    return MODEL_SHORT.get(str(value), str(value).replace("Qwen3-", ""))


def display_method(value: object) -> str:
    text = str(value).strip()
    lower = text.lower().replace("_", " ")
    replacements = {
        "eagle-3": "Eagle-3 K3",
        "suffix": "Suffix K32",
        "pard k=5": "PARD K5",
        "pard-2": "PARD-2",
        "eagle3 k3": "Eagle-3 K3",
        "eagle3 k5": "Eagle-3 K5",
        "eagle3 k7": "Eagle-3 K7",
        "eagle3 k9": "Eagle-3 K9",
        "pard k5": "PARD K5",
        "pard k16": "PARD K16",
        "pard2 k16": "PARD-2 K16",
        "suffix k32": "Suffix K32",
    }
    return replacements.get(lower, text.replace("_", " "))


def vllm_chart_rows(vllm: pd.DataFrame) -> pd.DataFrame:
    if vllm.empty:
        return pd.DataFrame()
    rows = vllm.copy()
    rows["model_short"] = rows["model"].map(short_model)
    rows["series"] = rows.apply(
        lambda row: f"{row['domain']} T{int(float(row['temperature']))}",
        axis=1,
    )
    return rows


def nemorl_chart_rows(nemorl: pd.DataFrame) -> pd.DataFrame:
    if nemorl.empty:
        return pd.DataFrame()
    rows = nemorl[nemorl["mode"].astype(str) == "sync"].copy()
    rows = rows.dropna(subset=["gen_tps_speedup"])
    if rows.empty:
        return rows
    rows["model_short"] = rows["model"].map(short_model)
    rows["method_display"] = rows["method"].map(display_method)
    idx = rows.groupby(["model", "method_display"], dropna=False)[
        "gen_tps_speedup"
    ].idxmax()
    return rows.loc[idx].sort_values(["model", "method_display"])


def plot_grouped_bar(
    rows: pd.DataFrame,
    *,
    x_col: str,
    hue_col: str,
    y_col: str,
    x_order: list[str],
    hue_order: list[str],
    title: str,
    ylabel: str,
    out_base: Path,
    baseline_line: bool = True,
) -> str:
    rows = rows.dropna(subset=[x_col, hue_col, y_col]).copy()
    if rows.empty:
        return ""

    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    x_order = [item for item in x_order if item in set(rows[x_col])]
    hue_order = [item for item in hue_order if item in set(rows[hue_col])]
    if not x_order or not hue_order:
        return ""

    fig_width = max(7.8, 1.15 * len(x_order) + 1.0 * len(hue_order))
    fig, ax = plt.subplots(figsize=(fig_width, 4.25))
    group_positions = list(range(len(x_order)))
    group_width = 0.76
    bar_width = group_width / max(1, len(hue_order))
    color_map = {hue: PAIRED[idx % len(PAIRED)] for idx, hue in enumerate(hue_order)}

    max_y = 0.0
    for hue_idx, hue in enumerate(hue_order):
        offset = -group_width / 2 + bar_width / 2 + hue_idx * bar_width
        values = []
        for x in x_order:
            sub = rows[(rows[x_col] == x) & (rows[hue_col] == hue)]
            values.append(as_float(sub[y_col].iloc[0]) if not sub.empty else math.nan)
        xs = [
            pos + offset
            for pos, value in zip(group_positions, values)
            if not math.isnan(value)
        ]
        ys = [value for value in values if not math.isnan(value)]
        if ys:
            max_y = max(max_y, max(ys))
            ax.bar(
                xs,
                ys,
                width=bar_width * 0.86,
                label=hue,
                color=color_map[hue],
                edgecolor="#192133",
                linewidth=2.0,
                zorder=10,
            )

    if baseline_line:
        ax.axhline(y=1.0, linestyle="--", linewidth=1.1, color="black", zorder=3)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=34)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(group_positions)
    ax.set_xticklabels(x_order, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, linestyle="--", dashes=(6, 6), linewidth=1.1, axis="y", zorder=0)
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_linewidth(2.0)
        ax.spines[side].set_color("black")
    if max_y > 0:
        ax.set_ylim(0, max(1.15, max_y * 1.22))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend().remove()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
            fontsize=12,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for suffix in (".png", ".pdf"):
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return f"figures/{out_base.with_suffix('.png').name}"


def figure_html(src: str, caption: str) -> str:
    if not src:
        return ""
    return (
        '<figure class="chart-figure">'
        f'<img src="{esc(src)}" alt="{esc(caption)}">'
        f"<figcaption>{esc(caption)}</figcaption>"
        "</figure>"
    )


def source_artifact_link(value: object) -> str:
    text = str(value or "")
    if not text or text.lower() == "nan":
        return "n/a"
    name = Path(text).name
    if not name:
        return f"<code>{esc(text)}</code>"
    if (DATA / name).exists() or (DOCS / name).exists():
        return f'<a class="source-link" href="data/{esc(name)}"><code>{esc(name)}</code></a>'
    return f"<code>{esc(text)}</code>"


def report_link_card(label: str, filename: str, description: str) -> str:
    local_src = DOCS / filename
    published_src = REPORTS / filename
    if local_src.exists() or published_src.exists():
        return (
            '<a class="report-link" '
            f'href="reports/{esc(filename)}">'
            f"<b>{esc(label)}</b>"
            f"<span>{esc(description)}</span>"
            f'<code class="file-code">{esc(filename)}</code>'
            "</a>"
        )
    return (
        '<div class="report-link missing">'
        f"<b>{esc(label)}</b>"
        f"<span>Missing local file: {esc(filename)}</span>"
        f'<code class="file-code">{esc(filename)}</code>'
        "</div>"
    )


def primary_links_html() -> str:
    return "\n      ".join(
        f'<a href="{esc(href)}">{esc(label)}</a>' for label, href in PRIMARY_LINKS
    )


def report_hub_html() -> str:
    groups = []
    for group in REPORT_GROUPS:
        primary = []
        archive = []
        for index, (label, filename, description) in enumerate(group["items"]):
            card = report_link_card(label, filename, description)
            if index == 0 or "latest" in filename or filename in PINNED_REPORT_FILES:
                primary.append(card)
            else:
                archive.append(card)
        archive_html = ""
        if archive:
            archive_cards = "".join(archive)
            archive_html = (
                '<details class="archive-links"><summary>Historical, diagnostic, and dated pages</summary>'
                f'<div class="report-buttons archive">{archive_cards}</div>'
                "</details>"
            )
        groups.append(
            '<section class="report-panel">'
            f"<h3>{esc(group['title'])}</h3>"
            f"<p>{esc(group['summary'])}</p>"
            f'<div class="report-buttons primary">{"".join(primary)}</div>'
            f"{archive_html}"
            "</section>"
        )
    return "".join(groups)


def build_chart_gallery(
    vllm: pd.DataFrame, nemorl_all: pd.DataFrame
) -> tuple[str, str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for old in FIGURES.glob("*.png"):
        old.unlink()
    for old in FIGURES.glob("*.pdf"):
        old.unlink()

    vcharts = vllm_chart_rows(vllm)
    ncharts = nemorl_chart_rows(nemorl_all)

    vllm_html = ""
    if not vcharts.empty:
        series_order = ["Math T0", "Math T1", "SWE T0", "SWE T1"]
        x_order = [short_model(model) for model in MODELS]
        speed = plot_grouped_bar(
            vcharts,
            x_col="model_short",
            hue_col="series",
            y_col="speedup",
            x_order=x_order,
            hue_order=series_order,
            title="vLLM Standalone: Best Speedup by Domain and Temperature",
            ylabel="Throughput speedup vs baseline",
            out_base=FIGURES / "vllm_best_speedup",
        )
        acc = plot_grouped_bar(
            vcharts,
            x_col="model_short",
            hue_col="series",
            y_col="acceptance_pct",
            x_order=x_order,
            hue_order=series_order,
            title="vLLM Standalone: Acceptance Rate",
            ylabel="Acceptance rate (%)",
            out_base=FIGURES / "vllm_best_acceptance",
            baseline_line=False,
        )
        vllm_html = "".join(
            [
                figure_html(
                    speed,
                    "Best matched-baseline throughput speedup. Method, batch size, ISL, and OSL are listed in the table below.",
                ),
                figure_html(
                    acc,
                    "Acceptance rate for the same best rows; temperature 1 generally lowers acceptance and accepted length.",
                ),
            ]
        )

    nemorl_html = ""
    if not ncharts.empty:
        method_order = [
            "Eagle-3 K3",
            "Eagle-3 K5",
            "Suffix K32",
            "PARD K5",
            "PARD K16",
            "PARD-2",
            "PARD-2 K16",
        ]
        x_order = [short_model(model) for model in MODELS]
        chart_specs = [
            (
                "gen_tps_speedup",
                "nemorl_generation_throughput_speedup",
                "NeMo-RL: Generation Throughput Speedup",
                "Generation throughput speedup",
                "Generation worker tokens/sec/GPU speedup vs matched baseline.",
            ),
            (
                "e2e_tps_speedup",
                "nemorl_e2e_throughput_speedup",
                "NeMo-RL: E2E Throughput Speedup",
                "E2E throughput speedup",
                "End-to-end tokens/sec/GPU speedup vs matched baseline.",
            ),
            (
                "gen_time_speedup",
                "nemorl_generation_time_speedup",
                "NeMo-RL: Generation Step-Time Speedup",
                "Generation time speedup",
                "Baseline generation time divided by SpecDec generation time; higher is faster.",
            ),
            (
                "e2e_step_speedup",
                "nemorl_e2e_step_time_speedup",
                "NeMo-RL: E2E Step-Time Speedup",
                "E2E step-time speedup",
                "Baseline total step time divided by SpecDec total step time; higher is faster.",
            ),
        ]
        chunks = []
        for metric, name, title, ylabel, caption in chart_specs:
            src = plot_grouped_bar(
                ncharts,
                x_col="model_short",
                hue_col="method_display",
                y_col=metric,
                x_order=x_order,
                hue_order=method_order,
                title=title,
                ylabel=ylabel,
                out_base=FIGURES / name,
            )
            chunks.append(figure_html(src, caption))
        nemorl_html = "".join(chunks)
    return vllm_html, nemorl_html


def rows_to_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def vllm_table(df: pd.DataFrame) -> str:
    if df.empty:
        return '<p class="muted">No vLLM rows are available in the local CSV artifacts.</p>'
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                esc(row["domain"]),
                esc(row["model"]),
                fmt(row["temperature"], 1),
                esc(row["method"]),
                fmt_int(row["batch_size"]),
                f"{fmt_int(row['isl'])}/{fmt_int(row['osl'])}",
                fmt(row["tok_s_gpu"], 2),
                fmt(row["speedup"], 2, "x"),
                fmt(row["acceptance_pct"], 1, "%"),
                fmt(row["mean_accept_len"], 2),
                source_artifact_link(row["source_file"]),
            ]
        )
    return rows_to_table(
        [
            "Domain",
            "Model",
            "Temp",
            "Method",
            "BS",
            "ISL/OSL",
            "tok/s/GPU",
            "Speedup",
            "Acceptance",
            "Mean accept len",
            "Source",
        ],
        rows,
    )


def nemorl_table(df: pd.DataFrame) -> str:
    if df.empty:
        return '<p class="muted">No NeMo-RL speedup rows are available in the local CSV artifacts.</p>'
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                f"<code>{esc(row['job_id'])}</code>",
                esc(row["model"]),
                esc(row["mode"]),
                esc(row["method"]),
                esc(row["completed"]),
                fmt_int(row["max_osl"]),
                fmt(row["gen_tps"], 2),
                fmt(row["gen_tps_speedup"], 2, "x"),
                fmt(row["gen_time_speedup"], 2, "x"),
                fmt(row["e2e_tps_speedup"], 2, "x"),
                fmt(row["e2e_step_speedup"], 2, "x"),
                fmt(row["acceptance_pct"], 1, "%"),
                fmt(row["mean_accept_len"], 2),
                source_artifact_link(row["source_file"]),
            ]
        )
    return rows_to_table(
        [
            "Job",
            "Model",
            "Mode",
            "Method",
            "Completed",
            "Max OSL",
            "Gen tok/s/GPU",
            "Gen TPS speedup",
            "Gen time speedup",
            "E2E TPS speedup",
            "E2E step speedup",
            "Acceptance",
            "Mean accept len",
            "Source",
        ],
        rows,
    )


def read_job_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def build() -> None:
    PUBLIC.mkdir(exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    report_files = [DOCS / filename for filename in REPORT_FILE_NAMES]
    data_files = [
        DOCS / "vllm_standalone_added_results_latest.csv",
        DOCS / "vllm_standalone_all_batches_combined_20260619.csv",
        DOCS / "lyris_qwen235b_pr2879_live_enriched_20260621.csv",
        DOCS / "lyris_nemorl_perfcfg_specdec_combined_latest.csv",
        DOCS / "lyris_nemorl_qwen30_qwen32_eagle3_k_sweep_live_summary_20260622.csv",
        DOCS / "lyris_nemorl_qwen30_qwen32_pr2879_step20_speedups_20260622.csv",
        DOCS / "lyris_nemorl_qwen30_qwen32_pr2879_status_20260622.csv",
        DOCS / "lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv",
        DOCS
        / "latest_lyris_nemorl_cudagraphoff_wandb_best_qwen32_async_20260623_jobs.csv",
        DOCS / "nemorl_clean_results_20260617.csv",
        DOCS / "nemorl_integrated_specdec_results_clean_20260617.csv",
        VLLM024_PROFILES,
        DFLARE_COMPLETED,
        DFLARE_STATUS,
        DOCS / "lyris_angelslim_checkpoint_prewarm_summary_20260622.json",
        ROOT / "latest_lyris_angelslim_checkpoint_prewarm_20260622_jobs.txt",
    ]
    archive_files = [
        ROOT / "experiments/eagle3_qwen3_235b/specdec_math_progress_report.html",
    ]
    for src in report_files:
        copy_if_exists(src, REPORTS)
        copy_report_local_refs(src)
    for filename in REPORT_COMPANION_FILES:
        copy_if_exists(DOCS / filename, REPORTS)
    for src in data_files:
        copy_if_exists(src, DATA)
    for src in archive_files:
        copy_if_exists(src, ARCHIVE)

    vllm = vllm_best_rows()
    dflare = load_csv(DFLARE_COMPLETED)
    dflare_status_rows = load_csv(DFLARE_STATUS)
    nemorl_all = load_nemorl_rows()
    nemorl = nemorl_best_rows()
    vllm_charts, nemorl_charts = build_chart_gallery(vllm, nemorl_all)
    report_hub = report_hub_html()
    job = read_job_file(
        ROOT / "latest_lyris_angelslim_checkpoint_prewarm_20260622_jobs.txt"
    )
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    job_id = job.get("job_id", "pending")
    job_status = job.get("status", "submitted")
    sacct_state = job.get("sacct_state", "")
    logs_dir = job.get("logs_dir", "")
    summary_json = job.get("summary_json", "")
    model_ids = job.get("model_ids", "")
    status_class = "ok" if sacct_state == "COMPLETED" else "warn"
    status_label = sacct_state or job_status
    if dflare.empty:
        dflare_completed = pd.DataFrame()
    else:
        complete_mask = dflare["status"].astype(str).eq("complete")
        profile_mask = dflare["context_profile"].astype(str).isin(DFLARE_PROFILES)
        dflare_completed = dflare[complete_mask & profile_mask]
    if dflare_completed.empty:
        dflare_summary = "No completed target-profile DFlare rows are available yet."
    else:
        latest_dflare = latest_dflare_performance_row(dflare_completed)
        completed_jobs = dflare_completed["job_id"].astype(str).nunique()
        performance_rows = len(dflare_completed)
        failed_rows = len(dflare_status_rows)
        timeout_rows = (
            dflare_status_rows["state"].astype(str).eq("TIMEOUT").sum()
            if not dflare_status_rows.empty
            else 0
        )
        acceptance = as_float(latest_dflare.get("acceptance_rate")) * 100
        dflare_summary = (
            f"{completed_jobs} completed target-profile DFlare job(s) produced "
            f"{performance_rows} performance row(s). "
            f"{failed_rows} failure/status row(s) remain separate, including "
            f"{int(timeout_rows)} TIMEOUT row(s). Latest performance row: "
            f"{latest_dflare.get('context_profile', 'n/a')} "
            f"{latest_dflare.get('domain', 'n/a')} "
            f"temp={fmt(latest_dflare.get('temperature'), 1)}, "
            f"{fmt(latest_dflare.get('tok_s_gpu'))} tok/s/GPU, "
            f"{fmt(acceptance)}% acceptance, mean accepted length "
            f"{fmt(latest_dflare.get('mean_accept_len'))}, job "
            f"{fmt_int(latest_dflare.get('job_id'))}."
        )

    html_text = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>SpecDec RL Benchmark Dashboard</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #151922;
      --muted: #5d6675;
      --line: #d7dce5;
      --blue: #1d5fbf;
      --green: #0c7a4b;
      --amber: #986100;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
      line-height: 1.45;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 26px 18px 44px; }}
    h1 {{ margin: 0; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 10px; font-size: 20px; letter-spacing: 0; }}
    p {{ color: var(--muted); margin: 7px 0 0; }}
    code {{ background: #eef1f6; border: 1px solid var(--line); border-radius: 4px; padding: 1px 4px; font-size: 12px; }}
    a {{ color: var(--blue); }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(16,24,40,.06); }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 700; letter-spacing: .04em; }}
    .metric {{ font-size: 24px; font-weight: 780; margin-top: 6px; }}
    .muted {{ color: var(--muted); }}
    .pill {{ display: inline-flex; align-items: center; min-height: 26px; padding: 2px 9px; border-radius: 999px; background: #eef1f6; border: 1px solid var(--line); font-weight: 700; font-size: 12px; }}
    .pill.ok {{ color: var(--green); }}
    .pill.warn {{ color: var(--amber); }}
    .links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    .links a {{ display: inline-flex; align-items: center; min-height: 34px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 7px 10px; text-decoration: none; font-weight: 700; }}
    .report-grid {{ display: grid; grid-template-columns: repeat(1, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
    .report-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 15px; box-shadow: 0 1px 2px rgba(16,24,40,.06); }}
    .report-panel h3 {{ margin: 0; font-size: 17px; }}
    .report-buttons {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 9px; margin-top: 12px; }}
    .report-buttons.primary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .report-buttons.archive {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .archive-links {{ margin-top: 10px; border-top: 1px solid var(--line); padding-top: 9px; }}
    .archive-links summary {{ cursor: pointer; color: var(--muted); font-weight: 750; }}
    .report-link {{ display: flex; min-height: 82px; flex-direction: column; gap: 5px; justify-content: flex-start; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; padding: 10px; text-decoration: none; }}
    .report-link b {{ color: var(--ink); font-size: 14px; line-height: 1.25; }}
    .report-link span {{ color: var(--muted); font-size: 12px; line-height: 1.3; }}
    .report-link code {{ margin-top: auto; overflow-wrap: anywhere; }}
    .report-link .file-code {{ color: var(--muted); font-size: 11px; }}
    .report-link.missing {{ opacity: .56; }}
    .source-link code {{ color: var(--blue); }}
    .table-scroll {{ width: 100%; overflow-x: auto; margin-top: 10px; }}
    table {{ min-width: 1060px; width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #eef1f6; }}
    tr:last-child td {{ border-bottom: 0; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 12px; }}
    .chart-figure {{ margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; box-shadow: 0 1px 2px rgba(16,24,40,.06); }}
    .chart-figure img {{ display: block; width: 100%; height: auto; }}
    .chart-figure figcaption {{ color: var(--muted); font-size: 13px; margin-top: 7px; }}
    .note {{ border-left: 4px solid var(--blue); background: var(--panel); border-radius: 8px; padding: 13px 14px; border-top: 1px solid var(--line); border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); margin-top: 14px; }}
    @media (max-width: 980px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 980px) {{ .report-buttons,.report-buttons.primary,.report-buttons.archive {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 840px) {{ .grid {{ grid-template-columns: 1fr; }} .report-buttons {{ grid-template-columns: 1fr; }} main {{ padding: 18px 12px 32px; }} }}
  </style>
</head>
<body>
<main id=\"overview\">
  <h1>SpecDec RL Benchmark Dashboard</h1>
  <p>Updated {esc(generated_at)}. This Pages entry point mirrors the latest local vLLM standalone and NeMo-RL benchmark artifacts for Qwen3 speculative decoding.</p>

  <div class=\"grid\">
    <div class=\"card\"><div class=\"label\">vLLM scope</div><div class=\"metric\">Math + SWE</div><p>Batch sweeps and temp 0/1 comparisons with ISL/OSL shown in the result tables.</p></div>
    <div class=\"card\"><div class=\"label\">NeMo-RL scope</div><div class=\"metric\">Perf recipe</div><p>Qwen30/32 use recipe OSL4096; Qwen235B PR2879 rows use recipe OSL8192.</p></div>
    <div class=\"card\"><div class=\"label\">AngelSlim staging</div><div class=\"metric\"><code>{esc(job_id)}</code></div><p>HF download job state: <span class=\"pill {status_class}\">{esc(status_label)}</span></p></div>
  </div>

  <section>
    <h2>Primary Links</h2>
    <div class=\"links\">
      {primary_links_html()}
    </div>
  </section>

  <section>
    <h2>Report Hub</h2>
    <p>These buttons preserve the broader local HTML archive. The latest vLLM page is intentionally scoped to matched ISL4096/OSL32768 comparisons; older pages keep historical, partial, long-OSL, and diagnostic context separate.</p>
    <div class=\"report-grid\">{report_hub}</div>
  </section>

  <section>
    <h2>vLLM Standalone Charts</h2>
    <p>Each bar uses the best valid matched-baseline row available for that model/domain/temperature cell. The table below keeps the method, batch size, ISL, and OSL visible.</p>
    <div class=\"chart-grid\">{vllm_charts}</div>
  </section>

  <section>
    <h2>Best vLLM Standalone Rows</h2>
    <p>Best rows are selected by matched baseline speedup for each model, domain, and temperature.</p>
    {vllm_table(vllm)}
  </section>

  <section>
    <h2>NeMo-RL Baseline-Relative Charts</h2>
    <p>Sync-mode rows with parsed matched-baseline metrics are plotted. Qwen30/32 are performance recipe OSL4096; Qwen235B is latest nightly/PR2879 with vLLM 0.20 path and recipe OSL8192.</p>
    <div class=\"chart-grid\">{nemorl_charts}</div>
  </section>

  <section>
    <h2>Best NeMo-RL Rows</h2>
    <p>Best rows are selected by generation throughput speedup against the matched baseline. E2E throughput and step-time speedup are shown separately.</p>
    {nemorl_table(nemorl)}
  </section>

  <section>
    <h2>DFlare and AngelSlim Status</h2>
    <div class=\"note\">
      <span class=\"pill {status_class}\">{esc(status_label)}</span>
      <p>{esc(dflare_summary)}</p>
      <p><a href=\"reports/vllm_standalone_results_latest.html#vllm024-profile\">Open the vLLM-native section</a>, <a href=\"reports/vllm_standalone_results_latest.html#vllm024-dflare\">open the completed DFlare table</a>, or <a href=\"reports/vllm_standalone_results_latest.html#vllm024-dflare-status\">open the failure/status table</a>. DFlare uses AngelSlim's standalone runtime and is kept separate from vLLM-native speedups.</p>
      <p>DFlare public checkpoints staged here include <code>AngelSlim/Qwen3-4b-dflare</code>, <code>AngelSlim/Qwen3-8b-dflare</code>, and <code>AngelSlim/Gpt-oss-20b-dflare</code>.</p>
      <p>Models requested in staging job: <code>{esc(model_ids)}</code></p>
      <p>Logs: <code>{esc(logs_dir)}</code></p>
      <p>Summary JSON: <code>{esc(summary_json)}</code></p>
    </div>
  </section>

  <section>
    <h2>Data Artifacts</h2>
    <div class=\"links\">
      <a href=\"data/vllm_standalone_added_results_latest.csv\">vLLM added CSV</a>
      <a href=\"data/vllm_standalone_all_batches_combined_20260619.csv\">vLLM all-batch CSV</a>
      <a href=\"data/lyris_qwen235b_pr2879_live_enriched_20260621.csv\">Qwen235B NeMo-RL CSV</a>
      <a href=\"data/lyris_nemorl_perfcfg_specdec_combined_latest.csv\">Combined NeMo-RL latest CSV</a>
      <a href=\"data/lyris_nemorl_qwen30_qwen32_eagle3_k_sweep_live_summary_20260622.csv\">Fresh Eagle-3 K sweep CSV</a>
      <a href=\"data/lyris_nemorl_qwen30_qwen32_pr2879_step20_speedups_20260622.csv\">Qwen30/32 NeMo-RL 2026-06-22 CSV</a>
      <a href=\"data/lyris_nemorl_qwen30_qwen32_pr2879_status_20260622.csv\">Qwen30/32 NeMo-RL 2026-06-22 status</a>
      <a href=\"data/lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv\">Qwen30/32 NeMo-RL CSV</a>
      <a href=\"data/lyris_angelslim_checkpoint_prewarm_summary_20260622.json\">AngelSlim prewarm summary</a>
      <a href=\"data/latest_lyris_angelslim_checkpoint_prewarm_20260622_jobs.txt\">AngelSlim job record</a>
      <a href=\"data/vllm024_profiles_latest.csv\">vLLM 0.24 native profiles</a>
      <a href=\"data/dflare_completed_latest.csv\">DFlare completed rows</a>
      <a href=\"data/dflare_job_status_latest.csv\">DFlare failure and status rows</a>
    </div>
  </section>
</main>
</body>
</html>
"""
    (PUBLIC / "index.html").write_text(html_text)

    local_html_text = html_text
    local_html_text = local_html_text.replace('href="reports/', 'href="')
    local_html_text = local_html_text.replace(
        'href="archive/', 'href="../public/archive/'
    )
    local_html_text = local_html_text.replace(
        'src="figures/', 'src="../public/figures/'
    )
    local_html_text = local_html_text.replace(
        'href="data/latest_lyris_angelslim_checkpoint_prewarm_20260622_jobs.txt"',
        'href="../latest_lyris_angelslim_checkpoint_prewarm_20260622_jobs.txt"',
    )
    local_html_text = local_html_text.replace('href="data/', 'href="')
    (DOCS / "specdec_reports_index_latest.html").write_text(local_html_text)
    print(PUBLIC / "index.html")
    print(DOCS / "specdec_reports_index_latest.html")


if __name__ == "__main__":
    build()
