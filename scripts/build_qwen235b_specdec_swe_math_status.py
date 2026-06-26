#!/usr/bin/env python3
"""Build a focused SWE/Math SpecDec status report from local benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_SHORT = {
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "Qwen/Qwen3-14B": "Qwen3-14B",
    "Qwen/Qwen3-30B-A3B": "Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "Qwen3-30B-Think",
    "Qwen/Qwen3-235B-A22B": "Qwen3-235B-A22B",
}

METHOD_COLORS = {
    "baseline": "#6b7280",
    "suffix": "#2a9d8f",
    "pard": "#c2410c",
    "pard2": "#7c3aed",
    "eagle3": "#2563eb",
}

QWEN235B_MODEL = "Qwen/Qwen3-235B-A22B"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def as_int(value: str | None) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def fmt(value: Any, precision: int = 2, suffix: str = "") -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{float(value):.{precision}f}{suffix}"


def model_short(model: str) -> str:
    return MODEL_SHORT.get(model, model.replace("Qwen/", ""))


def dataset_display(dataset: str) -> str:
    mapping = {
        "full": "SWE-Bench full",
        "verified": "SWE-Bench-Verified",
        "math500": "MATH500",
    }
    return mapping.get(dataset, dataset)


def method_family(method: str) -> str:
    method = method.lower()
    if method.startswith("suffix"):
        return "suffix"
    if method.startswith("pard2") or "pard-2" in method:
        return "pard2"
    if method.startswith("pard"):
        return "pard"
    if method.startswith("eagle3") or "eagle-3" in method:
        return "eagle3"
    if method.startswith("baseline"):
        return "baseline"
    return method


def method_display(method: str, k: str | None = None) -> str:
    raw = method.lower()
    if k is None or k == "":
        match = re.search(r"(?:^|_)k(\d+)(?:_|$)", raw)
        if match:
            k = match.group(1)
    if raw in ("baseline", "none"):
        return "baseline"
    if raw.startswith("suffix"):
        return f"Suffix K{k}" if k else "Suffix"
    if raw.startswith("pard2"):
        return f"PARD-2 K{k}" if k else "PARD-2"
    if raw.startswith("pard"):
        return f"PARD K{k}" if k else "PARD"
    if raw.startswith("eagle3"):
        return f"Eagle-3 K{k}" if k else "Eagle-3"
    if raw == "draft_model":
        return f"PARD K{k}" if k else "PARD"
    return method


def method_sort_key(method: str) -> tuple[int, int, str]:
    family = method_family(method)
    order = {"baseline": 0, "suffix": 1, "pard": 2, "pard2": 3, "eagle3": 4}
    match = re.search(r"\bK(\d+)\b", method)
    k = int(match.group(1)) if match else -1
    return (order.get(family, 9), k, method)


def source_from_row(row: dict[str, str], path: Path | None = None) -> str:
    text = " ".join(
        part
        for part in [
            row.get("tag", ""),
            row.get("breakdown_json", ""),
            str(path or ""),
        ]
        if part
    ).lower()
    if "oci" in text or "/lustre/fs1/" in text:
        return "OCI-HSG"
    if "lyris" in text or "/lustre/fsw/" in text:
        return "Lyris"
    return ""


def normalize_swe_final_speedups(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        method = row.get("method", "")
        rows.append(
            {
                "domain": "SWE OSL32K",
                "dataset": row.get("dataset", ""),
                "dataset_display": dataset_display(row.get("dataset", "")),
                "model": "Qwen/Qwen3-235B-A22B",
                "model_short": "Qwen3-235B-A22B",
                "source": "Lyris",
                "batch_size": as_int(row.get("batch_size")),
                "method": method_display(method),
                "method_family": method_family(method),
                "measurement": "final_spec_vs_live_baseline",
                "tok_s_per_gpu": as_float(row.get("tok_s_per_gpu")),
                "baseline_tok_s_per_gpu": as_float(row.get("baseline_live_tok_s_per_gpu")),
                "speedup_vs_baseline": as_float(row.get("provisional_speedup_vs_live_baseline")),
                "acceptance_pct": as_float(row.get("acceptance_rate_pct")),
                "mean_acceptance_length": as_float(row.get("mean_acceptance_length")),
                "job_id": "",
                "state": "",
                "final_rows": "",
                "note": row.get("basis", ""),
            }
        )
    return rows


def normalize_swe_pard_live(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        method = row.get("method", "")
        rows.append(
            {
                "domain": "SWE OSL32K",
                "dataset": row.get("dataset", ""),
                "dataset_display": dataset_display(row.get("dataset", "")),
                "model": "Qwen/Qwen3-235B-A22B",
                "model_short": "Qwen3-235B-A22B",
                "source": "Lyris",
                "batch_size": as_int(row.get("batch_size")),
                "method": method_display(method),
                "method_family": method_family(method),
                "measurement": "live_only",
                "tok_s_per_gpu": as_float(row.get("live_tok_s_per_gpu")),
                "baseline_tok_s_per_gpu": as_float(row.get("baseline_live_tok_s_per_gpu")),
                "speedup_vs_baseline": as_float(row.get("live_speedup_vs_live_baseline")),
                "acceptance_pct": as_float(row.get("live_acceptance_pct")),
                "mean_acceptance_length": as_float(row.get("live_mean_acceptance_length")),
                "job_id": row.get("job_id", ""),
                "state": row.get("state", ""),
                "final_rows": row.get("completed_batch_rows", ""),
                "note": row.get("basis", ""),
            }
        )
    return rows


def normalize_swe_breakdown_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        model = row.get("model", QWEN235B_MODEL)
        if model and model != QWEN235B_MODEL:
            continue
        label = row.get("label", "")
        if label.startswith("verified_"):
            dataset = "verified"
        elif label.startswith("full_"):
            dataset = "full"
        else:
            continue
        method = row.get("spec_method", "")
        if row.get("spec_active") == "False":
            method = "baseline"
        method_name = method_display(method, row.get("num_speculative_tokens"))
        rows.append(
            {
                "domain": "SWE OSL32K",
                "dataset": dataset,
                "dataset_display": dataset_display(dataset),
                "model": model,
                "model_short": model_short(model),
                "source": source_from_row(row, path),
                "batch_size": as_int(row.get("batch_size")),
                "method": method_name,
                "method_family": method_family(method_name),
                "measurement": "final_breakdown_no_baseline",
                "tok_s_per_gpu": as_float(row.get("output_tok_s_per_gpu")),
                "baseline_tok_s_per_gpu": "",
                "speedup_vs_baseline": as_float(row.get("speedup_vs_baseline")),
                "acceptance_pct": as_float(row.get("acceptance_rate_pct")),
                "mean_acceptance_length": as_float(row.get("mean_acceptance_length")),
                "job_id": "",
                "state": "",
                "final_rows": "1",
                "note": label,
            }
        )
    return rows


def normalize_math500(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        label = row.get("label", "")
        if not label.startswith("math500_osl32k"):
            continue
        model = row.get("model", "")
        if model and model != QWEN235B_MODEL:
            continue
        spec_method = row.get("spec_method", "")
        method = "baseline" if row.get("spec_active") == "False" else spec_method
        if method == "pard2":
            k = row.get("num_speculative_tokens")
            method_name = f"PARD-2 K{k}" if k else "PARD-2"
        else:
            method_name = method_display(method, row.get("num_speculative_tokens"))
        rows.append(
            {
                "domain": "MATH500 OSL32K",
                "dataset": "math500",
                "dataset_display": "MATH500",
                "model": model,
                "model_short": model_short(model),
                "source": source_from_row(row, path),
                "batch_size": as_int(row.get("batch_size")),
                "method": method_name,
                "method_family": method_family(method_name),
                "measurement": "final_breakdown",
                "tok_s_per_gpu": as_float(row.get("output_tok_s_per_gpu")),
                "baseline_tok_s_per_gpu": "",
                "speedup_vs_baseline": as_float(row.get("speedup_vs_baseline")),
                "acceptance_pct": as_float(row.get("acceptance_rate_pct")),
                "mean_acceptance_length": as_float(row.get("mean_acceptance_length")),
                "job_id": "",
                "state": "",
                "final_rows": "1",
                "note": row.get("label", ""),
            }
        )
    return rows


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            r["domain"],
            r["model_short"],
            r["dataset"],
            r.get("source", ""),
            r["batch_size"] if r["batch_size"] is not None else -1,
            method_sort_key(r["method"]),
            r["measurement"],
        ),
    )


def fill_final_baseline_speedups(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, str, int | None], float] = {}
    for row in rows:
        if row.get("measurement") != "final_breakdown" or row.get("method") != "baseline":
            continue
        tok_s = row.get("tok_s_per_gpu")
        if tok_s is None or tok_s == 0:
            continue
        key = (row["domain"], row["dataset"], row["model"], row["batch_size"])
        baselines[key] = float(tok_s)
        row["baseline_tok_s_per_gpu"] = float(tok_s)
        if row.get("speedup_vs_baseline") is None:
            row["speedup_vs_baseline"] = 1.0

    for row in rows:
        if row.get("measurement") != "final_breakdown":
            continue
        if row.get("method") == "baseline":
            continue
        tok_s = row.get("tok_s_per_gpu")
        if tok_s is None:
            continue
        key = (row["domain"], row["dataset"], row["model"], row["batch_size"])
        baseline = baselines.get(key)
        if not baseline:
            continue
        row["baseline_tok_s_per_gpu"] = baseline
        if row.get("speedup_vs_baseline") is None:
            row["speedup_vs_baseline"] = float(tok_s) / baseline


def write_normalized_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "dataset",
        "model",
        "source",
        "batch_size",
        "method",
        "measurement",
        "tok_s_per_gpu",
        "baseline_tok_s_per_gpu",
        "speedup_vs_baseline",
        "acceptance_pct",
        "mean_acceptance_length",
        "job_id",
        "state",
        "final_rows",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def range_summary(rows: list[dict[str, Any]], method: str, dataset: str | None = None) -> str:
    selected = [
        row
        for row in rows
        if row["method"] == method and (dataset is None or row["dataset"] == dataset)
    ]
    if not selected:
        return "not available"
    speedups = [row["speedup_vs_baseline"] for row in selected if row["speedup_vs_baseline"] is not None]
    accs = [row["acceptance_pct"] for row in selected if row["acceptance_pct"] is not None]
    toks = [row["tok_s_per_gpu"] for row in selected if row["tok_s_per_gpu"] is not None]
    parts = []
    if toks:
        parts.append(f"{min(toks):.2f}-{max(toks):.2f} tok/s/GPU")
    if speedups:
        parts.append(f"{min(speedups):.2f}x-{max(speedups):.2f}x")
    if accs:
        parts.append(f"{min(accs):.2f}%-{max(accs):.2f}% acceptance")
    return ", ".join(parts) if parts else "not available"


def range_summary_any(rows: list[dict[str, Any]], methods: set[str], dataset: str | None = None) -> str:
    selected = [
        row
        for row in rows
        if row["method"] in methods and (dataset is None or row["dataset"] == dataset)
    ]
    if not selected:
        return "not available"
    speedups = [row["speedup_vs_baseline"] for row in selected if row["speedup_vs_baseline"] is not None]
    accs = [row["acceptance_pct"] for row in selected if row["acceptance_pct"] is not None]
    toks = [row["tok_s_per_gpu"] for row in selected if row["tok_s_per_gpu"] is not None]
    parts = []
    if toks:
        parts.append(f"{min(toks):.2f}-{max(toks):.2f} tok/s/GPU")
    if speedups:
        parts.append(f"{min(speedups):.2f}x-{max(speedups):.2f}x")
    if accs:
        parts.append(f"{min(accs):.2f}%-{max(accs):.2f}% acceptance")
    return ", ".join(parts) if parts else "not available"


def final_count_text(rows: list[dict[str, Any]], methods: list[str]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        method = row["method"]
        if method not in methods:
            continue
        counts[method] = counts.get(method, 0) + int(row.get("final_rows") or 1)
    if not counts:
        return "0"
    total = sum(counts.values())
    detail = ", ".join(f"{method}={counts[method]}" for method in methods if method in counts)
    return f"{total} ({detail})"


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Domain | Dataset | Model | Source | Batch | Method | Measurement | tok/s/GPU | Baseline tok/s/GPU | Speedup | Acceptance | Mean accept len | Job | State | Final rows |",
        "| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {domain} | {dataset} | `{model}` | {source} | {batch} | {method} | {measurement} | {tok} | {base} | {speedup} | {acc} | {mean_len} | {job} | {state} | {final_rows} |".format(
                domain=row["domain"],
                dataset=row["dataset_display"],
                model=row["model"],
                source=row.get("source", ""),
                batch=row["batch_size"] if row["batch_size"] is not None else "",
                method=row["method"],
                measurement=row["measurement"],
                tok=fmt(row["tok_s_per_gpu"], 2),
                base=fmt(row["baseline_tok_s_per_gpu"], 2),
                speedup=fmt(row["speedup_vs_baseline"], 3, "x"),
                acc=fmt(row["acceptance_pct"], 2, "%"),
                mean_len=fmt(row["mean_acceptance_length"], 2),
                job=f"`{row['job_id']}`" if row.get("job_id") else "",
                state=f"`{row['state']}`" if row.get("state") else "",
                final_rows=row.get("final_rows", ""),
            )
        )
    return lines


def oci_math_status_note(paths: list[Path], completed_rows_present: bool = False) -> str:
    prefix = (
        "Qwen3-235B MATH500 OSL32K final coverage is partial; "
        if completed_rows_present
        else "Qwen3-235B MATH500 OSL32K rows are not present in local metrics yet; "
    )
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return (
            prefix
            + "no OCI-HSG fallback status file was found."
        )
    status_rows: list[dict[str, str]] = []
    try:
        for path in existing_paths:
            with path.open(newline="", encoding="utf-8") as f:
                status_rows.extend(csv.DictReader(f))
    except Exception as exc:
        return f"{prefix}OCI-HSG status could not be read: {exc}."
    if not status_rows:
        return prefix + "OCI-HSG status has no rows."
    states: dict[str, int] = {}
    job_bits: list[str] = []
    for row in status_rows:
        state = row.get("queue_state") or row.get("acct_state") or "UNKNOWN"
        states[state] = states.get(state, 0) + 1
        if row.get("job_id") and row.get("method_detail"):
            job_bits.append(f"{row['method_detail']} `{row['job_id']}`")
    state_text = ", ".join(f"{state}={count}" for state, count in sorted(states.items()))
    jobs_text = ", ".join(job_bits[:16])
    return (
        prefix
        + f"OCI-HSG fallback jobs are submitted ({state_text})."
        + (f" Jobs: {jobs_text}." if jobs_text else "")
    )


def oci_math_live_note(path: Path) -> str:
    rows = read_csv(path)
    if not rows:
        return ""
    bits: list[str] = []
    for row in rows:
        method = row.get("method_detail", "")
        speed = row.get("live_speedup_vs_baseline", "")
        gen = row.get("live_generation_tok_s", "")
        acc = row.get("live_draft_acceptance_pct", "")
        gen_samples = row.get("live_generation_samples", "")
        spec_samples = row.get("live_spec_samples", "")
        sample_note = ""
        if gen_samples or spec_samples:
            sample_parts = []
            if gen_samples:
                sample_parts.append(f"gen n={gen_samples}")
            if spec_samples:
                sample_parts.append(f"spec n={spec_samples}")
            sample_note = f"; {', '.join(sample_parts)}"
        if method == "baseline" and gen:
            if sample_note:
                bits.append(f"baseline {gen} gen tok/s ({sample_note.lstrip('; ')})")
            else:
                bits.append(f"baseline {gen} gen tok/s")
        elif speed and gen:
            suffix = f", {acc}% draft acceptance" if acc else ""
            bits.append(f"{method} {gen} gen tok/s ({speed}x{suffix}{sample_note})")
    if not bits:
        return ""
    return (
        "OCI-HSG MATH500 live logger telemetry, not final breakdown "
        "and volatile by prompt: "
        + "; ".join(bits)
        + "."
    )


def qwen235b_math_completed_note(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    parts: list[str] = []
    has_baseline = any(row["method"] == "baseline" for row in rows)
    for row in sorted_rows(rows):
        batch = row["batch_size"] if row["batch_size"] is not None else "?"
        detail = (
            f"{row['method']} batch {batch}"
            + (f" on {row['source']}" if row.get("source") else "")
            + ": "
            f"{fmt(row['tok_s_per_gpu'], 2)} tok/s/GPU"
        )
        acc = fmt(row["acceptance_pct"], 2, "%")
        mean_len = fmt(row["mean_acceptance_length"], 2)
        if acc:
            detail += f", {acc} acceptance"
        if mean_len:
            detail += f", mean accept len {mean_len}"
        parts.append(detail)
    suffix = ""
    if not has_baseline:
        suffix = " Matched Qwen3-235B baseline final row is not available yet, so final speedup remains blank."
    return "Qwen3-235B MATH500 completed final rows: " + "; ".join(parts) + "." + suffix


def write_markdown(
    path: Path,
    png_path: Path,
    csv_path: Path,
    rows: list[dict[str, Any]],
    oci_math_status_csv: Path,
    oci_math_extra_status_csvs: list[Path],
    oci_math_live_csv: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    swe_final = [
        r for r in rows if r["domain"] == "SWE OSL32K" and r["measurement"].startswith("final")
    ]
    swe_live = [r for r in rows if r["domain"] == "SWE OSL32K" and r["measurement"] == "live_only"]
    math_rows = [r for r in rows if r["domain"] == "MATH500 OSL32K"]
    qwen235b_math = [r for r in math_rows if r["model"] == "Qwen/Qwen3-235B-A22B"]
    lyris_full_sweep_final = [
        r for r in swe_final if r.get("source") == "Lyris" and r["dataset"] in {"full", "verified"}
    ]
    pard_sweep_methods = ["PARD K9", "PARD K11"]
    pard2_sweep_methods = ["PARD-2 K9", "PARD-2 K11"]

    lines = [
        "# Qwen3-235B SpecDec SWE/Math Status - 2026-06-13",
        "",
        f"![SpecDec SWE/Math status]({png_path.name})",
        "",
        "## Current Read",
        "",
        f"- SWE-Bench full Suffix K32: {range_summary(swe_final, 'Suffix K32', 'full')}.",
        f"- SWE-Bench-Verified Suffix K32: {range_summary(swe_final, 'Suffix K32', 'verified')}.",
        f"- SWE-Bench full Eagle-3 K3: {range_summary(swe_final, 'Eagle-3 K3', 'full')}.",
        f"- SWE-Bench-Verified Eagle-3 K3: {range_summary(swe_final, 'Eagle-3 K3', 'verified')}.",
        f"- SWE-Bench full PARD K9/K11 final breakdown rows without matched final baseline: {range_summary_any(swe_final, set(pard_sweep_methods), 'full')}.",
        f"- SWE-Bench full PARD-2 K9/K11 final breakdown rows: {range_summary_any(swe_final, set(pard2_sweep_methods), 'full')}.",
        f"- SWE PARD K5 live-only: {range_summary(swe_live, 'PARD K5')}.",
        f"- SWE PARD-2 K1 live-only: {range_summary(swe_live, 'PARD-2 K1')}.",
        f"- Current Lyris K9/K11 sweep has {final_count_text(lyris_full_sweep_final, pard_sweep_methods)} PARD final rows and {final_count_text(lyris_full_sweep_final, pard2_sweep_methods)} PARD-2 final rows; older PARD K5/PARD-2 K1 rows remain live telemetry only because `completed_batch_rows=0`.",
        "- Suffix K32 remains the strongest completed Qwen3-235B SWE OSL32K setting with baseline-relative speedup, while newer Lyris K8/K16 Suffix, PARD K9, and Eagle-3 K9/K11 rows are arriving as final breakdowns but still need matched final baseline rows for speedup.",
    ]
    if not qwen235b_math:
        lines.append(f"- {oci_math_status_note([oci_math_status_csv, *oci_math_extra_status_csvs])}")
        live_note = oci_math_live_note(oci_math_live_csv)
        if live_note:
            lines.append(f"- {live_note}")
    else:
        lines.append(f"- {qwen235b_math_completed_note(qwen235b_math)}")
        lines.append(
            f"- {oci_math_status_note([oci_math_status_csv, *oci_math_extra_status_csvs], completed_rows_present=True)}"
        )
        live_note = oci_math_live_note(oci_math_live_csv)
        if live_note:
            lines.append(f"- {live_note}")
    lines.extend(
        [
            f"- Raw normalized CSV: `{csv_path.name}`",
            "",
            "## Qwen3-235B SWE Final/Provisional Rows",
            "",
            *markdown_table(swe_final),
            "",
            "## Qwen3-235B SWE PARD/PARD-2 Live Rows",
            "",
            *markdown_table(swe_live),
            "",
            "## MATH500 OSL32K Completed Rows",
            "",
            *markdown_table(math_rows),
            "",
            "Notes:",
            "",
            "- `final_spec_vs_live_baseline` means the SpecDec row has a final breakdown, but the speedup uses live baseline telemetry until matching baseline final breakdown rows are collected.",
            "- `live_only` means the job had telemetry but no final breakdown row in the latest local refresh.",
            "- `final_breakdown` is a completed benchmark JSON row with speedup from the benchmark parser.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def html_table(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    current: tuple[str, str, int | None] | None = None
    for row in rows:
        group = (row["dataset_display"], row["model_short"], row["batch_size"])
        if group != current:
            current = group
            body.append(
                "<tr class=\"group\"><td colspan=\"15\">"
                f"{html.escape(row['dataset_display'])} / {html.escape(row['model_short'])} / batch {html.escape(str(row['batch_size']))}"
                "</td></tr>"
            )
        speed = row.get("speedup_vs_baseline")
        speed_class = ""
        if isinstance(speed, float):
            speed_class = " good" if speed >= 1.0 else " bad"
        body.append(
            "<tr>"
            f"<td>{html.escape(row['domain'])}</td>"
            f"<td>{html.escape(row['dataset_display'])}</td>"
            f"<td><code>{html.escape(row['model'])}</code></td>"
            f"<td>{html.escape(row.get('source', ''))}</td>"
            f"<td class=\"num\">{html.escape(str(row['batch_size'] or ''))}</td>"
            f"<td>{html.escape(row['method'])}</td>"
            f"<td>{html.escape(row['measurement'])}</td>"
            f"<td class=\"num\">{html.escape(fmt(row['tok_s_per_gpu'], 2))}</td>"
            f"<td class=\"num\">{html.escape(fmt(row['baseline_tok_s_per_gpu'], 2))}</td>"
            f"<td class=\"num{speed_class}\">{html.escape(fmt(speed, 3, 'x'))}</td>"
            f"<td class=\"num\">{html.escape(fmt(row['acceptance_pct'], 2, '%'))}</td>"
            f"<td class=\"num\">{html.escape(fmt(row['mean_acceptance_length'], 2))}</td>"
            f"<td><code>{html.escape(row.get('job_id', ''))}</code></td>"
            f"<td><code>{html.escape(row.get('state', ''))}</code></td>"
            f"<td class=\"num\">{html.escape(str(row.get('final_rows', '')))}</td>"
            "</tr>"
        )
    return "\n".join(body)


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def fmt_na(value: Any, precision: int = 2, suffix: str = "") -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{precision}f}{suffix}"


def best_label(row: dict[str, Any] | None) -> str:
    if not row:
        return "n/a"
    batch = row["batch_size"] if row.get("batch_size") is not None else "?"
    return f"{row['dataset_display']} {row['method']} bs{batch}"


def speedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("method") != "baseline" and finite_float(row.get("speedup_vs_baseline")) is not None
    ]


def best_speedup_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = speedup_rows(rows)
    if not candidates:
        return None
    return max(candidates, key=lambda row: finite_float(row.get("speedup_vs_baseline")) or -1.0)


def best_rows_by_scope(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in speedup_rows(rows):
        key = (row["domain"], row["dataset_display"])
        current = grouped.get(key)
        if current is None or (finite_float(row["speedup_vs_baseline"]) or -1.0) > (
            finite_float(current["speedup_vs_baseline"]) or -1.0
        ):
            grouped[key] = row
    return [grouped[key] for key in sorted(grouped)]


def waiting_baseline_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("method") != "baseline"
        and row.get("measurement") == "final_breakdown_no_baseline"
        and finite_float(row.get("tok_s_per_gpu")) is not None
    ]
    candidates.sort(key=lambda row: finite_float(row["tok_s_per_gpu"]) or -1.0, reverse=True)
    return candidates[:limit]


def summary_cards_html(rows: list[dict[str, Any]]) -> str:
    swe_rows = [row for row in rows if row["domain"] == "SWE OSL32K"]
    math_rows = [row for row in rows if row["domain"] == "MATH500 OSL32K"]
    swe_best = best_speedup_row(swe_rows)
    math_best = best_speedup_row(math_rows)
    final_rows = sum(1 for row in rows if str(row.get("measurement", "")).startswith("final"))
    live_rows = sum(1 for row in rows if row.get("measurement") == "live_only")
    cards = [
        ("Best SWE speedup", fmt_na(swe_best.get("speedup_vs_baseline") if swe_best else None, 2, "x"), best_label(swe_best)),
        ("Best Math speedup", fmt_na(math_best.get("speedup_vs_baseline") if math_best else None, 2, "x"), best_label(math_best)),
        ("Completed/final rows", str(final_rows), "Rows parsed from final benchmark breakdowns"),
        ("Live-only rows", str(live_rows), "Telemetry rows kept separate from final rows"),
    ]
    return "".join(
        "<div class=\"metric-card\">"
        f"<div class=\"label\">{html.escape(label)}</div>"
        f"<div class=\"metric\">{html.escape(value)}</div>"
        f"<p>{html.escape(detail)}</p>"
        "</div>"
        for label, value, detail in cards
    )


def methodology_html() -> str:
    pills = [
        "Qwen3-235B-A22B",
        "Historical cut: 2026-06-13",
        "vLLM standalone",
        "SWE-Bench OSL32K",
        "MATH500 OSL32K",
        "tok/s/GPU",
        "speedup vs matched baseline where available",
        "acceptance rate",
        "mean accepted length",
    ]
    return "".join(f"<span class=\"pill\">{html.escape(pill)}</span>" for pill in pills)


def freshness_panel_html() -> str:
    links = [
        (
            "Latest vLLM standalone matrix",
            "vLLM standalone report with the newer Qwen3-235B, Qwen3-30B-A3B, Qwen3-32B, and Qwen3-8B rollups.",
            "vllm_standalone_results_latest.html",
        ),
        (
            "2026-06-20 standalone page",
            "Dated standalone report with the newer chart layout and expanded batch/method coverage.",
            "vllm_standalone_results_20260620.html",
        ),
        (
            "SpecDec report hub",
            "Top-level index for vLLM standalone, NeMo-RL, and historical reports.",
            "../index.html",
        ),
    ]
    cards = []
    for title, detail, href in links:
        cards.append(
            '<a class="link-card" href="'
            + html.escape(href)
            + '"><strong>'
            + html.escape(title)
            + "</strong><span>"
            + html.escape(detail)
            + "</span></a>"
        )
    return (
        '<div class="freshness">'
        "<p><strong>Data cutoff.</strong> This page is the cleaned 2026-06-13 historical Qwen3-235B snapshot. "
        "Newer Qwen3-235B Math/SWE rows, DFlash rows, broader batch sweeps, and Qwen3-30B/Qwen3-8B/Qwen3-32B comparisons live in the current standalone pages below.</p>"
        f'<div class="link-grid">{"".join(cards)}</div>'
        "</div>"
    )


def key_findings_html(
    rows: list[dict[str, Any]],
    qwen235b_math_count: int,
    qwen235b_math_note: str,
    qwen235b_math_live_note: str,
) -> str:
    swe_best = best_speedup_row([row for row in rows if row["domain"] == "SWE OSL32K"])
    math_best = best_speedup_row([row for row in rows if row["domain"] == "MATH500 OSL32K"])
    waiting_count = len(waiting_baseline_rows(rows, limit=10_000))
    bullets = []
    if swe_best:
        bullets.append(
            f"Best SWE row with a baseline is {best_label(swe_best)} at "
            f"{fmt_na(swe_best['speedup_vs_baseline'], 2, 'x')} and "
            f"{fmt_na(swe_best['acceptance_pct'], 1, '%')} acceptance."
        )
    if math_best:
        bullets.append(
            f"Best MATH500 row with a baseline is {best_label(math_best)} at "
            f"{fmt_na(math_best['speedup_vs_baseline'], 2, 'x')} and "
            f"{fmt_na(math_best['acceptance_pct'], 1, '%')} acceptance."
        )
    if waiting_count:
        bullets.append(
            f"{waiting_count} final breakdown rows are shown without speedup because the matched final baseline is missing."
        )
    bullets.append(f"Qwen3-235B MATH500 completed rows found locally: {qwen235b_math_count}.")
    if qwen235b_math_note:
        bullets.append(qwen235b_math_note.split(" Jobs:", 1)[0] + ".")
    if qwen235b_math_live_note:
        bullets.append("OCI-HSG MATH500 live logger telemetry is available; detailed live values are kept in Notes and Sources.")
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in bullets) + "</ul>"


def compact_rows_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class=\"muted\">No rows available.</p>"
    body = []
    for row in rows:
        speed = finite_float(row.get("speedup_vs_baseline"))
        speed_class = " good" if speed is not None and speed >= 1.0 else " bad" if speed is not None else ""
        body.append(
            "<tr>"
            f"<td>{html.escape(row['domain'])}</td>"
            f"<td>{html.escape(row['dataset_display'])}</td>"
            f"<td class=\"num\">{html.escape(str(row['batch_size'] or ''))}</td>"
            f"<td>{html.escape(row['method'])}</td>"
            f"<td>{html.escape(row.get('source', ''))}</td>"
            f"<td class=\"num\">{html.escape(fmt_na(row.get('tok_s_per_gpu'), 2))}</td>"
            f"<td class=\"num{speed_class}\">{html.escape(fmt_na(row.get('speedup_vs_baseline'), 2, 'x'))}</td>"
            f"<td class=\"num\">{html.escape(fmt_na(row.get('acceptance_pct'), 1, '%'))}</td>"
            f"<td class=\"num\">{html.escape(fmt_na(row.get('mean_acceptance_length'), 2))}</td>"
            f"<td>{html.escape(row['measurement'])}</td>"
            "</tr>"
        )
    return (
        "<div class=\"table-wrap compact\"><table>"
        "<thead><tr><th>Domain</th><th>Dataset</th><th>Batch</th><th>Method</th><th>Source</th>"
        "<th>tok/s/GPU</th><th>Speedup</th><th>Acceptance</th><th>Mean len</th><th>Basis</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def cell_class(value: float | None) -> str:
    if value is None:
        return "empty"
    if value >= 1.0:
        return "good-cell"
    return "bad-cell"


def speed_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return '<td class="empty">n/a</td>'
    speed = finite_float(row.get("speedup_vs_baseline"))
    if speed is None:
        tok = fmt_na(row.get("tok_s_per_gpu"), 1)
        acc = fmt_na(row.get("acceptance_pct"), 1, "%")
        return (
            '<td class="empty">'
            f'<span class="cell-main">{html.escape(tok)}</span>'
            f'<span class="cell-sub">acc {html.escape(acc)}</span>'
            '<span class="cell-sub">waiting baseline</span>'
            "</td>"
        )
    return (
        f'<td class="{cell_class(speed)}">'
        f'<span class="cell-main">{html.escape(fmt_na(speed, 2, "x"))}</span>'
        f'<span class="cell-sub">{html.escape(fmt_na(row.get("tok_s_per_gpu"), 1))} tok/s/GPU</span>'
        f'<span class="cell-sub">acc {html.escape(fmt_na(row.get("acceptance_pct"), 1, "%"))}</span>'
        "</td>"
    )


def method_priority(method: str) -> tuple[int, int, str]:
    family = method_family(method)
    family_order = {"suffix": 0, "eagle3": 1, "pard": 2, "pard2": 3, "baseline": 9}
    match = re.search(r"\bK(\d+)\b", method)
    k = int(match.group(1)) if match else -1
    return (family_order.get(family, 8), k, method)


def speedup_matrix_table(
    rows: list[dict[str, Any]],
    *,
    domain: str,
    dataset: str,
    title: str,
) -> str:
    selected = [
        row
        for row in rows
        if row["domain"] == domain
        and row["dataset"] == dataset
        and row.get("method") != "baseline"
        and str(row.get("measurement", "")).startswith("final")
    ]
    if not selected:
        return ""
    batches = sorted({row["batch_size"] for row in selected if row.get("batch_size") is not None})
    methods = sorted({row["method"] for row in selected}, key=method_priority)
    best_by_key: dict[tuple[str, int | None], dict[str, Any]] = {}
    for row in selected:
        key = (row["method"], row["batch_size"])
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = row
            continue
        current_speed = finite_float(current.get("speedup_vs_baseline"))
        row_speed = finite_float(row.get("speedup_vs_baseline"))
        if (row_speed if row_speed is not None else -1.0) > (current_speed if current_speed is not None else -1.0):
            best_by_key[key] = row
    header = "".join(f"<th>B{html.escape(str(batch))}</th>" for batch in batches)
    body = []
    for method in methods:
        cells = "".join(speed_cell(best_by_key.get((method, batch))) for batch in batches)
        body.append(f"<tr><td class=\"method-cell\">{html.escape(method)}</td>{cells}</tr>")
    return (
        "<div>"
        f"<h3>{html.escape(title)}</h3>"
        '<div class="table-wrap matrix-wrap"><table class="matrix-table">'
        f"<thead><tr><th>Method</th>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
        "</div>"
    )


def matrix_section_html(rows: list[dict[str, Any]]) -> str:
    parts = [
        speedup_matrix_table(rows, domain="SWE OSL32K", dataset="full", title="SWE-Bench Full"),
        speedup_matrix_table(rows, domain="SWE OSL32K", dataset="verified", title="SWE-Bench Verified"),
        speedup_matrix_table(rows, domain="MATH500 OSL32K", dataset="math500", title="MATH500 OSL32K"),
    ]
    return "".join(part for part in parts if part)


def write_html(
    path: Path,
    png_path: Path,
    csv_path: Path,
    rows: list[dict[str, Any]],
    oci_math_status_csv: Path,
    oci_math_extra_status_csvs: list[Path],
    oci_math_live_csv: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    swe_final = [
        r for r in rows if r["domain"] == "SWE OSL32K" and r["measurement"].startswith("final")
    ]
    swe_live = [r for r in rows if r["domain"] == "SWE OSL32K" and r["measurement"] == "live_only"]
    math_rows = [r for r in rows if r["domain"] == "MATH500 OSL32K"]
    qwen235b_math_count = sum(1 for r in math_rows if r["model"] == "Qwen/Qwen3-235B-A22B")
    qwen235b_math_note = oci_math_status_note(
        [oci_math_status_csv, *oci_math_extra_status_csvs],
        completed_rows_present=qwen235b_math_count > 0,
    )
    qwen235b_math_live_note = oci_math_live_note(oci_math_live_csv)
    qwen235b_math_completed = qwen235b_math_completed_note(
        [r for r in math_rows if r["model"] == "Qwen/Qwen3-235B-A22B"]
    )
    best_scope_rows = best_rows_by_scope(rows)
    waiting_rows = waiting_baseline_rows(rows)
    matrix_html = matrix_section_html(rows)
    latest_update = "2026-06-13 historical snapshot, regenerated from local artifacts"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3-235B SpecDec SWE/Math Status</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #667085;
      --line: #d7dce2;
      --good: #047857;
      --bad: #b42318;
      --accent: #22577a;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 26px 24px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.55;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .hero p {{
      max-width: 920px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
    .metric {{
      font-size: 24px;
      font-weight: 800;
      margin-top: 5px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 3px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fbfcfe;
      color: #344054;
      font-size: 12px;
      font-weight: 700;
      margin: 2px 6px 2px 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .chart {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      overflow-x: auto;
      margin: 0;
    }}
    .chart img {{
      display: block;
      width: min(100%, 1180px);
      max-width: 100%;
      height: auto;
      margin: 0 auto;
    }}
    .note {{
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      color: var(--muted);
    }}
    .freshness {{
      margin-top: 14px;
      border: 1px solid #bfd2e6;
      border-radius: 8px;
      background: #f6f9fc;
      padding: 12px 14px;
    }}
    .freshness p {{
      margin: 0 0 10px;
      max-width: 1080px;
    }}
    .link-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .link-card {{
      display: block;
      min-height: 90px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      text-decoration: none;
    }}
    .link-card strong {{
      display: block;
      margin-bottom: 6px;
      color: var(--accent);
    }}
    .link-card span {{
      display: block;
      color: var(--muted);
      line-height: 1.4;
    }}
    .table-wrap {{
      overflow: auto;
      border-radius: 8px;
      border: 1px solid var(--line);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      font-size: 13px;
    }}
    .compact table {{
      min-width: 920px;
    }}
    .matrix-wrap table {{
      min-width: 760px;
    }}
    .matrix-table th,
    .matrix-table td {{
      text-align: center;
      vertical-align: middle;
    }}
    .matrix-table th:first-child,
    .matrix-table td:first-child {{
      text-align: left;
    }}
    .method-cell {{
      font-weight: 750;
      white-space: nowrap;
    }}
    .cell-main {{
      display: block;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}
    .cell-sub {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
      margin-top: 2px;
    }}
    .good-cell {{
      background: #eefaf4;
      color: var(--good);
    }}
    .bad-cell {{
      background: #fff1f0;
      color: var(--bad);
    }}
    .empty {{
      background: #f8fafc;
      color: #64748b;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid #e7ebef;
      vertical-align: top;
    }}
    th {{
      background: #eef3f8;
      color: #344054;
      text-align: left;
      white-space: nowrap;
    }}
    td {{
      overflow-wrap: anywhere;
    }}
    .group td {{
      background: #f4f7fb;
      color: #1d3557;
      font-weight: 750;
    }}
    .num {{
      text-align: right;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}
    .good {{
      color: var(--good);
      font-weight: 700;
    }}
    .bad {{
      color: var(--bad);
      font-weight: 700;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      word-break: break-word;
    }}
    a {{
      color: var(--accent);
    }}
    details {{
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      color: var(--text);
      font-weight: 750;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 1080px) {{
      .cards, .grid, .link-grid {{
        grid-template-columns: 1fr;
      }}
      main {{
        padding: 18px 12px 36px;
      }}
    }}
  </style>
</head>
<body>
<main>
  <div class="hero">
    <h1>Qwen3-235B SpecDec SWE/Math Status</h1>
    <p>Focused snapshot from local benchmark artifacts. Final benchmark rows, live-only telemetry, and rows waiting for matched baselines are deliberately separated so speedup interpretation stays clear.</p>
    <p class="muted">{html.escape(latest_update)}</p>
    <div>{methodology_html()}</div>
    {freshness_panel_html()}
    <div class="cards">{summary_cards_html(rows)}</div>
  </div>

  <section>
    <h2>Key Findings</h2>
    {key_findings_html(rows, qwen235b_math_count, qwen235b_math_note, qwen235b_math_live_note)}
  </section>

  <section>
    <h2>Best Rows With Matched Baselines</h2>
    <p>Rows below have a usable baseline-relative speedup. Speedups are not computed for final breakdown rows that lack a matched final baseline.</p>
    {compact_rows_table(best_scope_rows)}
  </section>

  <section>
    <h2>Speedup Matrices</h2>
    <p>Each cell shows speedup, throughput, and acceptance for the best available row for that method and batch size. Cells marked waiting baseline have final throughput but no matched baseline speedup.</p>
    <div class="grid">{matrix_html}</div>
  </section>

  <section>
    <h2>Final Rows Waiting For Matched Baselines</h2>
    <p>These rows have final benchmark throughput and acceptance metrics, but speedup is intentionally blank until a matched final baseline exists for the same dataset, model, batch, and setup.</p>
    {compact_rows_table(waiting_rows)}
  </section>

  <section>
    <h2>Charts</h2>
    <div class="chart"><img src="{html.escape(png_path.name)}" alt="SpecDec SWE Math status chart"></div>
  </section>

  <section>
    <h2>Qwen3-235B SWE Final/Provisional Rows</h2>
    <p>Full row-level detail is kept here for traceability. The matrix above is the faster way to compare methods.</p>
    <details>
    <summary>Show detailed SWE final/provisional rows</summary>
    <div class="table-wrap detail-wrap">
    <table>
      <thead><tr><th>Domain</th><th>Dataset</th><th>Model</th><th>Source</th><th>Batch</th><th>Method</th><th>Measurement</th><th>tok/s/GPU</th><th>Baseline tok/s/GPU</th><th>Speedup</th><th>Acceptance</th><th>Mean accept len</th><th>Job</th><th>State</th><th>Final rows</th></tr></thead>
      <tbody>{html_table(swe_final)}</tbody>
    </table>
    </div>
    </details>
  </section>

  <section>
    <h2>Qwen3-235B SWE PARD/PARD-2 Live Rows</h2>
    <details>
    <summary>Show live-only telemetry rows</summary>
    <div class="table-wrap detail-wrap">
    <table>
      <thead><tr><th>Domain</th><th>Dataset</th><th>Model</th><th>Source</th><th>Batch</th><th>Method</th><th>Measurement</th><th>tok/s/GPU</th><th>Baseline tok/s/GPU</th><th>Speedup</th><th>Acceptance</th><th>Mean accept len</th><th>Job</th><th>State</th><th>Final rows</th></tr></thead>
      <tbody>{html_table(swe_live)}</tbody>
    </table>
    </div>
    </details>
  </section>

  <section>
    <h2>MATH500 OSL32K Completed Rows</h2>
    <details open>
    <summary>Show MATH500 completed rows</summary>
    <div class="table-wrap detail-wrap">
    <table>
      <thead><tr><th>Domain</th><th>Dataset</th><th>Model</th><th>Source</th><th>Batch</th><th>Method</th><th>Measurement</th><th>tok/s/GPU</th><th>Baseline tok/s/GPU</th><th>Speedup</th><th>Acceptance</th><th>Mean accept len</th><th>Job</th><th>State</th><th>Final rows</th></tr></thead>
      <tbody>{html_table(math_rows)}</tbody>
    </table>
    </div>
    </details>
  </section>

  <section>
    <h2>Notes And Sources</h2>
    <div class="note">
      <p>SWE Suffix/Eagle-3 speedups use final SpecDec breakdown rows divided by live baseline telemetry. PARD/PARD-2 live rows remain separate from final benchmark rows.</p>
      <details>
        <summary>Qwen3-235B MATH500 completed final rows</summary>
        <p>{html.escape(qwen235b_math_completed)}</p>
      </details>
      <details>
        <summary>OCI-HSG fallback status and live telemetry</summary>
        <p>{html.escape(qwen235b_math_note)}</p>
        <p>{html.escape(qwen235b_math_live_note)}</p>
      </details>
      <p>Raw normalized CSV: <a href="{html.escape(csv_path.name)}">{html.escape(csv_path.name)}</a></p>
    </div>
  </section>
</main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def plot_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    swe_final = [
        r for r in rows if r["domain"] == "SWE OSL32K" and r["measurement"].startswith("final")
    ]
    swe_live = [r for r in rows if r["domain"] == "SWE OSL32K" and r["measurement"] == "live_only"]
    math_rows = [
        r
        for r in rows
        if r["domain"] == "MATH500 OSL32K"
        and r["method"] != "baseline"
        and r["speedup_vs_baseline"] is not None
    ]

    fig, axes = plt.subplots(3, 1, figsize=(16, 18), constrained_layout=True)
    panels = [
        ("Qwen3-235B SWE final SpecDec rows vs live baseline", swe_final),
        ("Qwen3-235B SWE PARD/PARD-2 live-only rows", swe_live),
        ("MATH500 OSL32K completed final rows", math_rows),
    ]
    for ax, (title, panel_rows) in zip(axes, panels):
        panel_rows = [
            row for row in panel_rows if row.get("speedup_vs_baseline") is not None
        ]
        if not panel_rows:
            ax.text(0.5, 0.5, "No rows available", ha="center", va="center")
            ax.set_axis_off()
            continue
        labels = [
            f"{row['dataset_display']} | {row['model_short']} | bs{row['batch_size']} | {row['method']}"
            for row in panel_rows
        ]
        y = list(range(len(panel_rows)))
        values = [float(row["speedup_vs_baseline"]) for row in panel_rows]
        colors = [METHOD_COLORS.get(row["method_family"], "#4b5563") for row in panel_rows]
        bars = ax.barh(y, values, color=colors, edgecolor="#222222", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(1.0, color="#111111", linewidth=1, linestyle="--")
        ax.set_xlabel("Speedup vs baseline")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        xmax = max(values) * 1.2
        ax.set_xlim(0, max(1.5, xmax))
        for bar, row, value in zip(bars, panel_rows, values):
            acc = row.get("acceptance_pct")
            text = f"{value:.2f}x"
            if acc is not None:
                text += f" / acc {acc:.1f}%"
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                " " + text,
                va="center",
                ha="left",
                fontsize=8,
            )
    fig.suptitle("SpecDec performance impact snapshot from local Lyris artifacts", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    math500_metrics = [args.math500_metrics_csv, *args.extra_math500_metrics_csv]
    swe_metrics = [*args.extra_swe_metrics_csv]
    rows = [
        *normalize_swe_final_speedups(args.swe_speedups_csv),
        *normalize_swe_pard_live(args.swe_pard_live_csv),
        *(row for path in swe_metrics for row in normalize_swe_breakdown_metrics(path)),
        *(row for path in math500_metrics for row in normalize_math500(path)),
    ]
    fill_final_baseline_speedups(rows)
    return sorted_rows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--swe-speedups-csv",
        type=Path,
        default=Path("docs/lyris_qwen235b_swebench_osl32k_batch_sweep_speedups_20260612.csv"),
    )
    parser.add_argument(
        "--swe-pard-live-csv",
        type=Path,
        default=Path("docs/lyris_qwen235b_swebench_osl32k_pard_live_snapshot_20260613.csv"),
    )
    parser.add_argument(
        "--extra-swe-metrics-csv",
        action="append",
        type=Path,
        default=[
            Path("docs/lyris_qwen235b_standalone_fast_20260613_metrics.csv"),
        ],
    )
    parser.add_argument(
        "--math500-metrics-csv",
        type=Path,
        default=Path("docs/lyris_math500_osl32k_metrics_20260612.csv"),
    )
    parser.add_argument(
        "--extra-math500-metrics-csv",
        action="append",
        type=Path,
        default=[
            Path("docs/oci_qwen235b_math500_osl32k_metrics_20260613.csv"),
            Path("docs/oci_qwen235b_math500_suffix_py312_retry1_metrics_20260613.csv"),
            Path("docs/oci_qwen235b_math500_drafter_k9_metrics_20260613.csv"),
            Path("docs/oci_qwen235b_math500_drafter_k11_metrics_20260613.csv"),
            Path("docs/lyris_qwen235b_standalone_fast_20260613_metrics.csv"),
        ],
    )
    parser.add_argument(
        "--oci-math500-status-csv",
        type=Path,
        default=Path("docs/oci_qwen235b_math500_osl32k_status_20260613.csv"),
    )
    parser.add_argument(
        "--oci-math500-extra-status-csv",
        action="append",
        type=Path,
        default=[
            Path("docs/oci_qwen235b_math500_suffix_py312_retry1_status_20260613.csv"),
            Path("docs/oci_qwen235b_math500_drafter_k9_status_20260613.csv"),
            Path("docs/oci_qwen235b_math500_drafter_k11_status_20260613.csv"),
        ],
    )
    parser.add_argument(
        "--oci-math500-live-csv",
        type=Path,
        default=Path("docs/oci_qwen235b_math500_live_progress_20260613.csv"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("docs/qwen235b_specdec_swe_math_status_20260613"),
    )
    args = parser.parse_args()

    rows = build_rows(args)
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    html_path = args.output_prefix.with_suffix(".html")
    png_path = args.output_prefix.with_suffix(".png")

    plot_rows(png_path, rows)
    write_normalized_csv(csv_path, rows)
    write_markdown(
        md_path,
        png_path,
        csv_path,
        rows,
        args.oci_math500_status_csv,
        args.oci_math500_extra_status_csv,
        args.oci_math500_live_csv,
    )
    write_html(
        html_path,
        png_path,
        csv_path,
        rows,
        args.oci_math500_status_csv,
        args.oci_math500_extra_status_csv,
        args.oci_math500_live_csv,
    )
    print(csv_path)
    print(md_path)
    print(html_path)
    print(png_path)


if __name__ == "__main__":
    main()
