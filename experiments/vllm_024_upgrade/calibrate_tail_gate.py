#!/usr/bin/env python3
"""Convert measured tail-gate component latencies into a roofline JSON file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = (
    "model",
    "target_tp",
    "draft_tp",
    "cluster",
    "container",
    "container_sha256",
    "vllm_commit",
    "gpu",
)
CONSTANT_FIELDS = (
    "W_t",
    "W_d",
    "C_dense",
    "C_attn",
    "kappa_theoretical",
    "F_eff",
    "BW_peak",
    "F_peak",
    "c_comm",
)
MEASUREMENT_FIELDS = ("B", "S", "K", "T_T", "T_D", "T_V")
REQUIRED_FIELDS = IDENTITY_FIELDS + CONSTANT_FIELDS + MEASUREMENT_FIELDS


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Measured latency CSV"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    return parser.parse_args(argv)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = [
            field for field in REQUIRED_FIELDS if field not in fieldnames
        ]
        if missing_columns:
            raise ValueError(f"missing required column: {missing_columns[0]}")
        rows = list(reader)
    if not rows:
        raise ValueError("measurement CSV is empty")
    return rows


def _require_single_value(rows: list[dict[str, str]], field: str) -> str:
    values = {row[field].strip() for row in rows}
    if "" in values:
        raise ValueError(f"missing required value: {field}")
    if len(values) != 1:
        raise ValueError(f"mixed {field} rows are not supported in one calibration")
    return values.pop()


def _parse_positive(
    row: dict[str, str], field: str, *, integer: bool = False
) -> float | int:
    value = row[field].strip()
    if not value:
        raise ValueError(f"missing required value: {field}")
    try:
        parsed: float | int = int(value) if integer else float(value)
    except ValueError as error:
        raise ValueError(f"invalid {field}: {value}") from error
    if not math.isfinite(float(parsed)) or parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _validated_values(rows: list[dict[str, str]]) -> dict[str, Any]:
    identity = {field: _require_single_value(rows, field) for field in IDENTITY_FIELDS}
    constants: dict[str, float | int] = {}
    for field in CONSTANT_FIELDS:
        values = {
            _parse_positive(row, field, integer=field == "kappa_theoretical")
            for row in rows
        }
        if len(values) != 1:
            raise ValueError(f"mixed {field} rows are not supported in one calibration")
        constants[field] = values.pop()
    for row in rows:
        for field in MEASUREMENT_FIELDS:
            _parse_positive(row, field, integer=field in {"B", "S", "K"})
    return {**identity, **constants}


def _estimate_kappa_eff(
    rows: list[dict[str, str]], bandwidth: float, maximum: int
) -> float:
    points = sorted(
        (int(row["B"]) * int(row["S"]), float(row["T_T"]) * 1e-3) for row in rows
    )
    slopes = [
        (right_time - left_time) / (right_tokens - left_tokens)
        for (left_tokens, left_time), (right_tokens, right_time) in zip(
            points, points[1:]
        )
        if right_tokens > left_tokens and right_time > left_time
    ]
    if not slopes:
        return max(1.0, maximum / 2.0)
    return min(float(maximum), max(1.0, statistics.median(slopes) * bandwidth))


def _fit_payload(
    rows: list[dict[str, str]], values: dict[str, Any], input_sha256: str
) -> dict[str, Any]:
    target_timings = [float(row["T_T"]) for row in rows]
    draft_timings = [float(row["T_D"]) for row in rows]
    verify_timings = [float(row["T_V"]) for row in rows]
    batches = [int(row["B"]) for row in rows]
    bandwidth = min(
        float(values["BW_peak"]),
        max(1.0, float(values["W_t"]) / (statistics.median(target_timings) * 1e-3)),
    )
    kappa_eff = _estimate_kappa_eff(rows, bandwidth, int(values["kappa_theoretical"]))
    eta_d = max(
        1.0,
        statistics.median(draft_timings)
        / statistics.median(target_timings)
        * float(values["W_t"])
        / float(values["W_d"]),
    )
    median_batch = statistics.median(batches)
    k_values = sorted({int(row["K"]) for row in rows})
    metadata = {
        "calibration_schema": "efficientrollout-sd-toggle-v1",
        "cluster": values["cluster"],
        "container": values["container"],
        "container_sha256": values["container_sha256"],
        "draft_tp": int(values["draft_tp"]),
        "input_sha256": input_sha256,
        "k_values": k_values,
        "measurement_rows": len(rows),
        "model": values["model"],
        "target_tp": int(values["target_tp"]),
        "vllm_commit": values["vllm_commit"],
    }
    return {
        "hardware": {
            "BW_eff": bandwidth,
            "BW_peak": float(values["BW_peak"]),
            "F_peak": float(values["F_peak"]),
            "c_comm": float(values["c_comm"]),
            "gpu": values["gpu"],
            "tp": int(values["target_tp"]),
        },
        "model": {
            "C_attn": float(values["C_attn"]),
            "C_dense": float(values["C_dense"]),
            "W_d": float(values["W_d"]),
            "W_t": float(values["W_t"]),
            "gqa": 1,
            "kappa_theoretical": int(values["kappa_theoretical"]),
            "name": values["model"],
            "rho": float(values["W_d"]) / float(values["W_t"]),
        },
        "calibration": {
            "F_eff": float(values["F_eff"]),
            "beta": 0.0,
            "c_D": statistics.median(draft_timings) * 1000.0 / median_batch,
            "c_T": statistics.median(target_timings) * 1000.0 / median_batch,
            "c_V": statistics.median(verify_timings) * 1000.0 / median_batch,
            "eta_d": eta_d,
            "kappa_eff": kappa_eff,
            "per_gamma": {},
        },
        "metadata": metadata,
    }


def _output_name(values: dict[str, Any], k_values: list[int]) -> str:
    model = re.sub(r"[^a-z0-9]+", "-", values["model"].lower()).strip("-")
    cluster = re.sub(r"[^a-z0-9]+", "-", values["cluster"].lower()).strip("-")
    k_label = "-".join(str(value) for value in k_values)
    container_hash = values["container_sha256"][:12]
    return (
        f"{model}-tp{values['target_tp']}-dtp{values['draft_tp']}-"
        f"{cluster}-c{container_hash}-k{k_label}.json"
    )


def _write_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    output_path.with_suffix(f"{output_path.suffix}.sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        rows = _load_rows(args.input)
        values = _validated_values(rows)
        payload = _fit_payload(
            rows, values, hashlib.sha256(args.input.read_bytes()).hexdigest()
        )
        output_path = args.output_dir / _output_name(
            payload["metadata"], payload["metadata"]["k_values"]
        )
        _write_payload(payload, output_path)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
