#!/usr/bin/env python3
"""Normalize and render completed vLLM 0.24 DFlare benchmark results."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_COLUMNS = [
    "status",
    "runtime",
    "backend",
    "attention_backend",
    "domain",
    "model",
    "temperature",
    "top_p",
    "batch_size",
    "isl",
    "osl",
    "context_profile",
    "position_encoding",
    "method",
    "k",
    "tok_s_gpu",
    "speedup",
    "speedup_label",
    "acceptance_rate",
    "mean_accept_len",
    "job_id",
    "source",
]

PROFILE_ORDER = ["Native 32K", "YaRN 64K", "YaRN total-128K"]


def _domain(dataset: object) -> str:
    value = str(dataset).lower()
    if "math" in value or "dapo" in value:
        return "Math"
    if "swe" in value:
        return "SWE"
    return str(dataset)


def _model_name(target_model: object) -> str:
    value = str(target_model).lower()
    if "qwen3-8b" in value:
        return "Qwen3-8B"
    return Path(str(target_model)).name


def _job_id(path: Path) -> str:
    matches = re.findall(r"(?<!\d)(\d{7})(?!\d)", str(path))
    return matches[-1] if matches else ""


def _position_encoding(target_model: object) -> str:
    return "yarn4" if "/yarn4/" in str(target_model).lower() else "native"


def _context_profile(isl: int, osl: int, position_encoding: str) -> str:
    if isl == 4096 and osl == 32_768 and position_encoding == "native":
        return "Native 32K"
    if isl == 4096 and osl == 65_536 and position_encoding == "yarn4":
        return "YaRN 64K"
    if isl == 4096 and osl == 126_976 and position_encoding == "yarn4":
        return "YaRN total-128K"
    return f"ISL {isl} / OSL {osl}"


def _per_gpu_batch(config: dict[str, Any]) -> int:
    samples = int(config.get("max_samples", 1))
    world_size = max(int(config.get("world_size", 1)), 1)
    return max(math.ceil(samples / world_size), 1)


def _paired_speedup(config: dict[str, Any], results: dict[str, Any]) -> float:
    run_mode = str(config.get("run_mode", ""))
    has_pair = {
        "baseline_decode_tok_s",
        "spec_decode_tok_s",
        "decode_throughput_speedup",
    }.issubset(results)
    if run_mode == "spec" or not has_pair:
        return math.nan
    return float(results["decode_throughput_speedup"])


def normalize_dflare_result(payload: dict[str, Any], source: Path) -> dict[str, object]:
    config = dict(payload.get("config", {}))
    results = dict(payload.get("results", {}))
    target_model = config.get("target_model", "")
    isl = int(config.get("input_length", 0))
    osl = int(config.get("max_new_tokens", 0))
    position_encoding = _position_encoding(target_model)
    k = int(config.get("block_size", 0))
    speedup = _paired_speedup(config, results)
    return {
        "status": str(payload.get("status", "")),
        "runtime": "AngelSlim",
        "backend": str(payload.get("backend", "")),
        "attention_backend": str(config.get("attention_backend", "torch.sdpa")),
        "domain": _domain(config.get("dataset", "")),
        "model": _model_name(target_model),
        "temperature": float(config.get("temperature", math.nan)),
        "top_p": float(config.get("top_p", math.nan)),
        "batch_size": _per_gpu_batch(config),
        "isl": isl,
        "osl": osl,
        "context_profile": _context_profile(isl, osl, position_encoding),
        "position_encoding": position_encoding,
        "method": f"dflare_k{k}",
        "k": k,
        "tok_s_gpu": float(results.get("spec_decode_tok_s", math.nan)),
        "speedup": speedup,
        "speedup_label": f"{speedup:.2f}x" if math.isfinite(speedup) else "waiting matched baseline",
        "acceptance_rate": float(results.get("acceptance_rate", math.nan)),
        "mean_accept_len": float(results.get("mean_acceptance_length", math.nan)),
        "job_id": _job_id(source),
        "source": str(source),
    }


def load_completed_dflare_results(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        if payload.get("config", {}).get("draft_arch") != "dflare":
            continue
        rows.append(normalize_dflare_result(payload, path))
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def match_dflare_baselines(rows: pd.DataFrame) -> pd.DataFrame:
    matched = rows.copy()
    if matched.empty:
        return matched
    matched["speedup"] = pd.to_numeric(matched["speedup"], errors="coerce")
    matched["speedup_label"] = matched["speedup"].map(
        lambda value: f"{value:.2f}x" if pd.notna(value) else "waiting matched baseline"
    )
    return matched


def target_profile_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return rows.loc[rows["context_profile"].isin(PROFILE_ORDER)].copy()


def relativize_sources(rows: pd.DataFrame, root: Path) -> pd.DataFrame:
    relative = rows.copy()
    if relative.empty:
        return relative
    resolved_root = root.resolve()

    def convert(value: object) -> str:
        path = Path(str(value)).resolve()
        try:
            return str(path.relative_to(resolved_root))
        except ValueError:
            return str(value)

    relative["source"] = relative["source"].map(convert)
    return relative


def _fmt_number(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _profile_table(rows: pd.DataFrame, profile: str) -> str:
    profile_rows = rows.loc[rows["context_profile"].eq(profile)].copy()
    if profile_rows.empty:
        return ""
    profile_rows = profile_rows.sort_values(
        ["domain", "temperature", "method", "batch_size", "job_id"],
        kind="stable",
    )
    body: list[str] = []
    for row in profile_rows.itertuples(index=False):
        cells = [
            escape(str(row.domain)),
            f'<span class="num">{_fmt_number(row.temperature, 1)}</span>',
            f'<span class="num">{int(row.isl):,}</span>',
            f'<span class="num">{int(row.osl):,}</span>',
            f'<span class="num">{int(row.batch_size)}</span>',
            escape(str(row.method)),
            f'<span class="num">{_fmt_number(row.tok_s_gpu)}</span>',
            escape(str(row.speedup_label)),
            f'<span class="num">{_fmt_number(float(row.acceptance_rate) * 100)}%</span>',
            f'<span class="num">{_fmt_number(row.mean_accept_len)}</span>',
            escape(str(row.job_id or "n/a")),
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
        "Speedup",
        "Acceptance",
        "Mean accept length",
        "Job ID",
    ]
    header = "".join(f"<th>{escape(heading)}</th>" for heading in headings)
    return (
        f"<h3>{escape(profile)}</h3>"
        '<div class="table-wrap"><table>'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def render_dflare_section(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""
    completed = target_profile_rows(
        rows.loc[rows["status"].eq("complete")].copy()
    )
    if completed.empty:
        return ""
    tables = "".join(_profile_table(completed, profile) for profile in PROFILE_ORDER)
    if not tables:
        return ""
    return (
        '<section class="section" id="vllm024-dflare">'
        "<h2>vLLM 0.24 / DFlare Completed Results</h2>"
        '<p class="note">Qwen3-8B on Lyris GB200; AngelSlim Transformers-native runtime, '
        "temperature 0/1, top-p 1.0, and DFlare K16. FlashAttention is unavailable, "
        "so these runs use PyTorch SDPA. Speedup is reported only for an exact "
        "AngelSlim baseline; spec-only rows remain waiting matched baseline.</p>"
        f"{tables}</section>"
    )
