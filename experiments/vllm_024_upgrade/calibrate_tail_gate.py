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
from datetime import datetime
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
    "target_checkpoint_revision",
    "draft_checkpoint_revision",
    "calibration_timestamp",
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
    identity: dict[str, Any] = {
        field: _require_single_value(rows, field) for field in IDENTITY_FIELDS
    }
    for field in ("target_tp", "draft_tp"):
        value = identity[field]
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"{field} must be a positive integer")
        identity[field] = int(value)
    for field in ("target_checkpoint_revision", "draft_checkpoint_revision"):
        if re.fullmatch(r"[0-9a-fA-F]{40}", identity[field]) is None:
            raise ValueError(
                f"{field} must be an exact 40-character hexadecimal revision"
            )
    timestamp = identity["calibration_timestamp"]
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            "calibration_timestamp must be an ISO-8601 timestamp"
        ) from error
    if parsed_timestamp.utcoffset() is None:
        raise ValueError("calibration_timestamp must include a timezone")
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


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    size = len(values)
    augmented = [row[:] + [value] for row, value in zip(matrix, values)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-18:
            raise ValueError("calibration measurements do not span a fit-able grid")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * reference
                for current, reference in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _least_squares(
    features: list[list[float]], observations: list[float]
) -> list[float]:
    width = len(features[0])
    normal_matrix = [
        [sum(row[left] * row[right] for row in features) for right in range(width)]
        for left in range(width)
    ]
    normal_values = [
        sum(row[column] * value for row, value in zip(features, observations))
        for column in range(width)
    ]
    return _solve_linear_system(normal_matrix, normal_values)


def _fit_target_parameters(
    rows: list[dict[str, str]], values: dict[str, Any]
) -> tuple[float, float]:
    unique_measurements = sorted(
        {(int(row["B"]), int(row["S"]), float(row["T_T"])) for row in rows}
    )
    c_comm = float(values["c_comm"])
    features = [
        [1.0, float(batch * sequence), float(batch)]
        for batch, sequence, _timing in unique_measurements
    ]
    observations = [
        timing_ms * 1e-3 - c_comm
        for _batch, _sequence, timing_ms in unique_measurements
    ]
    intercept, kv_slope, _batch_overhead = _least_squares(features, observations)
    if intercept <= 0.0 or kv_slope <= 0.0:
        raise ValueError("target timing fit produced non-positive roofline parameters")
    bandwidth = min(float(values["BW_peak"]), float(values["W_t"]) / intercept)
    kappa_eff = kv_slope * bandwidth
    if bandwidth <= 0.0 or kappa_eff <= 0.0:
        raise ValueError("target timing fit produced non-positive roofline parameters")
    return bandwidth, kappa_eff


def _target_base_seconds(
    row: dict[str, str], values: dict[str, Any], bandwidth: float, kappa_eff: float
) -> float:
    batch = int(row["B"])
    sequence = int(row["S"])
    memory = (float(values["W_t"]) + kappa_eff * batch * sequence) / bandwidth
    compute = (
        batch * float(values["C_dense"]) + batch * sequence * float(values["C_attn"])
    ) / float(values["F_eff"])
    return max(memory, compute) + float(values["c_comm"])


def _fit_eta_d(
    rows: list[dict[str, str]],
    values: dict[str, Any],
    bandwidth: float,
    kappa_eff: float,
) -> float:
    k_values = sorted({int(row["K"]) for row in rows})
    features: list[list[float]] = []
    observations: list[float] = []
    for row in rows:
        batch = int(row["B"])
        sequence = int(row["S"])
        gamma = int(row["K"])
        gamma_features = [float(batch) if gamma == value else 0.0 for value in k_values]
        features.append([1.0, *gamma_features])
        observations.append(
            float(row["T_D"]) * 1e-3
            - float(values["c_comm"])
            - kappa_eff * batch * sequence / bandwidth
        )
    intercept, *_per_gamma_slopes = _least_squares(features, observations)
    eta_d = intercept * bandwidth / float(values["W_d"])
    return max(1.0, eta_d)


def _draft_base_seconds(
    row: dict[str, str],
    values: dict[str, Any],
    bandwidth: float,
    kappa_eff: float,
    eta_d: float,
) -> float:
    batch = int(row["B"])
    sequence = int(row["S"])
    memory = (eta_d * float(values["W_d"]) + kappa_eff * batch * sequence) / bandwidth
    compute = (
        batch * float(values["C_dense"]) + batch * sequence * float(values["C_attn"])
    ) / float(values["F_eff"])
    return max(memory, compute) + float(values["c_comm"])


def _verify_base_seconds(
    row: dict[str, str], values: dict[str, Any], bandwidth: float, kappa_eff: float
) -> float:
    batch = int(row["B"])
    sequence = int(row["S"])
    gamma = int(row["K"])
    memory = (float(values["W_t"]) + kappa_eff * batch * sequence) / bandwidth
    compute = (
        batch * (gamma + 1) * float(values["C_dense"])
        + batch * sequence * (gamma + 1) * float(values["C_attn"])
    ) / float(values["F_eff"])
    return max(memory, compute) + float(values["c_comm"])


def _residual_overhead_us(
    rows: list[dict[str, str]], measured_field: str, base_seconds: Sequence[float]
) -> float:
    residuals = [
        max(0.0, (float(row[measured_field]) * 1e-3 - base) / (int(row["B"]) * 1e-6))
        for row, base in zip(rows, base_seconds)
    ]
    return statistics.median(residuals)


def _fit_per_gamma(
    rows: list[dict[str, str]],
    values: dict[str, Any],
    bandwidth: float,
    kappa_eff: float,
    eta_d: float,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for gamma in sorted({int(row["K"]) for row in rows}):
        gamma_rows = [row for row in rows if int(row["K"]) == gamma]
        c_t = _residual_overhead_us(
            gamma_rows,
            "T_T",
            [
                _target_base_seconds(row, values, bandwidth, kappa_eff)
                for row in gamma_rows
            ],
        )
        c_d = _residual_overhead_us(
            gamma_rows,
            "T_D",
            [
                _draft_base_seconds(row, values, bandwidth, kappa_eff, eta_d)
                for row in gamma_rows
            ],
        )
        c_v = _residual_overhead_us(
            gamma_rows,
            "T_V",
            [
                _verify_base_seconds(row, values, bandwidth, kappa_eff)
                for row in gamma_rows
            ],
        )
        result[str(gamma)] = {"R2": 0.0, "c_D": c_d, "c_T": c_t, "c_V": c_v}
    return result


def _fit_payload(
    rows: list[dict[str, str]], values: dict[str, Any], input_sha256: str
) -> dict[str, Any]:
    bandwidth, kappa_eff = _fit_target_parameters(rows, values)
    eta_d = _fit_eta_d(rows, values, bandwidth, kappa_eff)
    per_gamma = _fit_per_gamma(rows, values, bandwidth, kappa_eff, eta_d)
    k_values = sorted({int(row["K"]) for row in rows})
    metadata = {
        "calibration_schema": "efficientrollout-sd-toggle-v1",
        "cluster": values["cluster"],
        "container": values["container"],
        "container_sha256": values["container_sha256"],
        "draft_tp": int(values["draft_tp"]),
        "draft_checkpoint_revision": values["draft_checkpoint_revision"],
        "calibration_timestamp": values["calibration_timestamp"],
        "input_sha256": input_sha256,
        "k_values": k_values,
        "measurement_rows": len(rows),
        "model": values["model"],
        "target_tp": int(values["target_tp"]),
        "target_checkpoint_revision": values["target_checkpoint_revision"],
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
            "c_D": 0.0,
            "c_T": 0.0,
            "c_V": 0.0,
            "eta_d": eta_d,
            "kappa_eff": kappa_eff,
            "per_gamma": per_gamma,
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
