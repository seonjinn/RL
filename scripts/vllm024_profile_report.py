#!/usr/bin/env python3
"""Normalize and render vLLM 0.24 native profile benchmark results."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
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
SPECULATIVE_EXCLUDED_CONFIG_KEYS = frozenset(
    {
        "batch_sizes",
        "draft_model",
        "mode",
        "prompt_count_loaded",
        "speculative_config",
        "tag",
    }
)
RUNTIME_ENV_EXCLUDED_KEYS = frozenset({"CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID"})
MATCH_KEYS = [
    "runtime_family",
    "runtime_provenance",
    "model_checkpoint",
    "domain",
    "temperature",
    "top_p",
    "batch_size",
    "isl",
    "osl",
    "context_profile",
    "position_encoding",
    "cuda_graph",
    "setup_signature",
]
CANONICAL_COLUMNS = [
    "runtime_family",
    "runtime",
    "runtime_provenance",
    "domain",
    "model",
    "model_checkpoint",
    "temperature",
    "top_p",
    "batch_size",
    "isl",
    "osl",
    "context_profile",
    "position_encoding",
    "cuda_graph",
    "setup_signature",
    "attention_backend",
    "method",
    "k",
    "tok_s_gpu",
    "latency_s",
    "acceptance_rate",
    "mean_accept_len",
    "job_id",
    "source_status",
    "source",
]


def _to_float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return math.nan
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return math.nan
    return math.nan


def _to_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _stable_value(value: object) -> object:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key in sorted(str(item) for item in value.keys()):
            normalized[key] = _stable_value(value.get(key))
        return normalized
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_json(value: object) -> str:
    return json.dumps(
        _stable_value(value),
        sort_keys=True,
        separators=(",", ":"),
    )


def _runtime_family(payload: Mapping[str, object]) -> str | None:
    runtime = payload.get("runtime")
    if isinstance(runtime, Mapping) and "vllm_version" in runtime:
        return "vllm_native"
    return None


def _runtime_label(runtime: object) -> str:
    if isinstance(runtime, Mapping):
        version = runtime.get("vllm_version")
        if isinstance(version, str) and version:
            return f"vLLM {version}"
    return str(runtime)


def _runtime_provenance(runtime: object) -> str:
    if not isinstance(runtime, Mapping):
        return str(runtime)
    runtime_dict = dict(runtime)
    environment = runtime_dict.get("environment")
    if isinstance(environment, Mapping):
        runtime_dict["environment"] = {
            str(key): environment.get(key)
            for key in sorted(str(item) for item in environment.keys())
            if str(key) not in RUNTIME_ENV_EXCLUDED_KEYS
        }
    return _stable_json(runtime_dict)


def _domain(config: Mapping[str, object], source: Path) -> str:
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


def _job_id(payload: Mapping[str, object], source: Path) -> str:
    runtime = payload.get("runtime")
    if isinstance(runtime, Mapping):
        environment = runtime.get("environment")
        if isinstance(environment, Mapping):
            job_id = environment.get("SLURM_JOB_ID")
            if job_id is not None:
                return str(job_id)
    matches = re.findall(r"(?<!\d)(\d{5,})(?!\d)", str(source))
    return matches[-1] if matches else ""


def _method(config: Mapping[str, object]) -> str:
    return str(config.get("mode", "baseline")).lower()


def _k(config: Mapping[str, object]) -> float:
    speculative = config.get("speculative_config")
    if not isinstance(speculative, Mapping):
        return math.nan
    return _to_float(speculative.get("num_speculative_tokens"))


def _setup_signature(config: Mapping[str, object]) -> str:
    filtered_config = {
        str(key): config.get(key)
        for key in sorted(str(item) for item in config.keys())
        if str(key) not in SPECULATIVE_EXCLUDED_CONFIG_KEYS
    }
    return _stable_json(filtered_config)


def _acceptance_metric(batch_result: Mapping[str, object], key: str) -> float:
    metrics = batch_result.get("spec_decode_metrics")
    if not isinstance(metrics, Mapping):
        return math.nan
    return _to_float(metrics.get(key))


def _latency_s(batch_result: Mapping[str, object]) -> float:
    for key in ("mean_latency_s", "latency_s_mean", "latency_s"):
        value = batch_result.get(key)
        number = _to_float(value)
        if math.isfinite(number):
            return number
    return math.nan


def _tok_s_gpu(batch_result: Mapping[str, object]) -> float:
    return _to_float(batch_result.get("output_tok_s_per_gpu"))


def _source_status(status: object) -> str:
    return "complete" if str(status).lower() == "complete" else "partial"


def _normalize_batch_result(
    payload: Mapping[str, object],
    config: Mapping[str, object],
    batch_result: Mapping[str, object],
    source: Path,
) -> dict[str, object]:
    runtime = payload.get("runtime")
    model_path = str(config.get("model", ""))
    isl = _to_int(config.get("isl"))
    osl = _to_int(config.get("osl"))
    position_encoding = _position_encoding(model_path)
    return {
        "runtime_family": "vllm_native",
        "runtime": _runtime_label(runtime),
        "runtime_provenance": _runtime_provenance(runtime),
        "domain": _domain(config, source),
        "model": _model_name(model_path),
        "model_checkpoint": model_path,
        "temperature": _to_float(config.get("temperature")),
        "top_p": _to_float(config.get("top_p")),
        "batch_size": _to_int(batch_result.get("bs")),
        "isl": isl,
        "osl": osl,
        "context_profile": _context_profile(isl, osl, position_encoding),
        "position_encoding": position_encoding,
        "cuda_graph": str(config.get("cudagraph_mode", "")),
        "setup_signature": _setup_signature(config),
        "attention_backend": str(config.get("attention_backend", "")),
        "method": _method(config),
        "k": _k(config),
        "tok_s_gpu": _tok_s_gpu(batch_result),
        "latency_s": _latency_s(batch_result),
        "acceptance_rate": _acceptance_metric(batch_result, "acceptance_rate"),
        "mean_accept_len": _acceptance_metric(batch_result, "mean_acceptance_length"),
        "job_id": _job_id(payload, source),
        "source_status": _source_status(payload.get("status")),
        "source": str(source),
    }


def load_profile_results(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(path) for path in paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            continue
        if _runtime_family(payload) != "vllm_native":
            continue
        config = payload.get("config")
        results = payload.get("results")
        if not isinstance(config, Mapping) or not isinstance(results, list):
            continue
        for batch_result in results:
            if isinstance(batch_result, Mapping):
                rows.append(_normalize_batch_result(payload, config, batch_result, path))
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _speedup_label(value: object) -> str:
    number = _to_float(value)
    return f"{number:.2f}x" if math.isfinite(number) else "waiting matched baseline"


def _empty_matched_frame(rows: pd.DataFrame) -> pd.DataFrame:
    matched = rows.copy()
    matched["baseline_tok_s_gpu"] = pd.Series(dtype=float)
    matched["baseline_latency_s"] = pd.Series(dtype=float)
    matched["throughput_speedup"] = pd.Series(dtype=float)
    matched["latency_speedup"] = pd.Series(dtype=float)
    matched["throughput_speedup_label"] = pd.Series(dtype=str)
    matched["latency_speedup_label"] = pd.Series(dtype=str)
    return matched


def _prepare_baseline_lookup(rows: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = rows.loc[rows["method"].eq("baseline")].copy()
    if baseline_rows.empty:
        return pd.DataFrame(columns=MATCH_KEYS + ["baseline_tok_s_gpu", "baseline_latency_s"])

    baseline_rows["status_rank"] = baseline_rows["source_status"].map({"complete": 0, "partial": 1}).fillna(2)
    baseline_rows = baseline_rows.sort_values(
        MATCH_KEYS + ["status_rank", "source", "job_id"],
        kind="stable",
    )
    baseline_rows["preferred_status_rank"] = baseline_rows.groupby(
        MATCH_KEYS,
        dropna=False,
    )["status_rank"].transform("min")
    preferred_rows = baseline_rows.loc[
        baseline_rows["status_rank"].eq(baseline_rows["preferred_status_rank"])
    ].copy()

    duplicate_mask = preferred_rows.duplicated(subset=MATCH_KEYS, keep=False)
    if duplicate_mask.any():
        raise ValueError("ambiguous duplicate baseline exact keys")

    return preferred_rows[MATCH_KEYS + ["tok_s_gpu", "latency_s"]].rename(
        columns={
            "tok_s_gpu": "baseline_tok_s_gpu",
            "latency_s": "baseline_latency_s",
        }
    )


def match_profile_baselines(rows: pd.DataFrame) -> pd.DataFrame:
    native_rows = rows.loc[rows["runtime_family"].eq("vllm_native")].copy()
    if native_rows.empty:
        return _empty_matched_frame(native_rows)

    native_rows["_row_order"] = range(len(native_rows))
    baseline_lookup = _prepare_baseline_lookup(native_rows)
    matched = native_rows.merge(
        baseline_lookup,
        on=MATCH_KEYS,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    matched = matched.sort_values("_row_order", kind="stable").drop(columns="_row_order")
    matched["throughput_speedup"] = matched["tok_s_gpu"] / matched["baseline_tok_s_gpu"]
    matched["latency_speedup"] = matched["baseline_latency_s"] / matched["latency_s"]
    matched["throughput_speedup_label"] = matched["throughput_speedup"].map(_speedup_label)
    matched["latency_speedup_label"] = matched["latency_speedup"].map(_speedup_label)
    return matched


def _fmt_number(value: object, digits: int = 2) -> str:
    number = _to_float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _fmt_percent(value: object) -> str:
    number = _to_float(value)
    return f"{number * 100:.2f}%" if math.isfinite(number) else "n/a"


def _fmt_k(value: object) -> str:
    number = _to_float(value)
    if not math.isfinite(number):
        return "n/a"
    return str(int(number))


def _method_label(method: object, k: object) -> str:
    text = str(method)
    if text == "baseline":
        return "Baseline"
    labels = {
        "dflash": "DFlash",
        "pard": "PARD",
        "pard2": "PARD-2",
        "suffix": "Suffix",
    }
    return f"{labels.get(text, text)} K={_fmt_k(k)}"


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
    text = str(source)
    parts = Path(text).parts
    if len(parts) <= 6:
        return text
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
            escape(_method_label(row.method, row.k)),
            f'<span class="num">{_fmt_number(row.tok_s_gpu)}</span>',
            escape(str(row.throughput_speedup_label)),
            escape(str(row.latency_speedup_label)),
            f'<span class="num">{_fmt_percent(row.acceptance_rate)}</span>',
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
    native_rows = rows.loc[rows["runtime_family"].eq("vllm_native")].copy()
    if native_rows.empty:
        return ""
    target_rows = native_rows.loc[native_rows["context_profile"].isin(PROFILE_ORDER)].copy()
    if target_rows.empty:
        return ""
    if "throughput_speedup_label" not in target_rows.columns:
        target_rows["throughput_speedup_label"] = target_rows["throughput_speedup"].map(_speedup_label)
    if "latency_speedup_label" not in target_rows.columns:
        target_rows["latency_speedup_label"] = target_rows["latency_speedup"].map(_speedup_label)
    tables = "".join(_profile_table(target_rows, profile) for profile in PROFILE_ORDER)
    if not tables:
        return ""
    return (
        '<section class="section" id="vllm024-profile">'
        "<h2>vLLM 0.24 / Native Profile Results</h2>"
        '<p class="note">Qwen3-8B on Lyris GB200, matched only against exact '
        "vLLM-native baselines on runtime provenance, target checkpoint identity, domain, "
        "temperature, top-p, batch, ISL, OSL, profile, position encoding, CUDA graph mode, "
        "and normalized non-speculative setup. Persisted batch results from interrupted "
        "sources are shown as partial.</p>"
        f"{tables}</section>"
    )
