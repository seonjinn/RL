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

"""Fail-closed validation for immutable Qwen campaign gate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import sys
from pathlib import Path
from typing import NoReturn, cast


FULL_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
FULL_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
ARMS = frozenset({"A", "B", "C", "E"})
PROVENANCE_FIELDS = frozenset(
    {
        "nemo_rl_commit",
        "bridge_commit",
        "mcore_commit",
        "container_sha256",
        "runtime_attestation_sha256",
    }
)
R3_DIAGNOSTIC = {
    "model": "Qwen/Qwen3-235B-A22B",
    "num_prompts": 128,
    "max_tokens": 256,
    "max_model_len": 8192,
    "prompt_repeat": 128,
    "tensor_parallel_size": 8,
    "pipeline_parallel_size": 1,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.4,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
    "enforce_eager": False,
    "moe_backend": "triton",
    "num_outputs": 128,
    "num_failures": 0,
}
ARM_FIELDS = frozenset(
    {
        "job_id",
        "status",
        "completed_steps",
        "metrics_finite",
        "correctness_passed",
        "undeclared_fallbacks",
        "router_replay",
        "graph_coverage_status",
        "r3_trace_status",
    }
)


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _regular_absolute_file(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        _fail(f"{label} must be an absolute path")
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_mapping(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    actual_fields = set(value)
    if actual_fields != fields:
        missing = sorted(fields - actual_fields)
        unknown = sorted(actual_fields - fields)
        _fail(f"{label} fields mismatch: missing={missing}, unknown={unknown}")
    return cast(dict[str, object], value)


def _positive_number(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a positive numeric value")
    if not math.isfinite(value) or value <= 0:
        _fail(f"{label} must be a positive finite numeric value")


def _require(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        _fail(f"{label} must equal {expected!r}")


def _validate_provenance(
    value: object,
    *,
    nemo_rl_commit: str,
    bridge_commit: str,
    mcore_commit: str,
    container_sha256: str,
    runtime_attestation_sha256: str,
) -> None:
    provenance = _exact_mapping(value, PROVENANCE_FIELDS, "provenance")
    expected = {
        "nemo_rl_commit": nemo_rl_commit,
        "bridge_commit": bridge_commit,
        "mcore_commit": mcore_commit,
        "container_sha256": container_sha256,
        "runtime_attestation_sha256": runtime_attestation_sha256,
    }
    for field, expected_value in expected.items():
        candidate = provenance[field]
        if not isinstance(candidate, str):
            _fail(f"provenance.{field} must be a string")
        pattern = FULL_SHA256 if field.endswith("sha256") else FULL_COMMIT
        if pattern.fullmatch(candidate) is None:
            _fail(f"provenance.{field} must be a full lowercase digest")
        _require(candidate, expected_value, f"provenance.{field}")


def _validate_r3(payload: dict[str, object], args: argparse.Namespace) -> None:
    gate = _exact_mapping(
        payload,
        frozenset({"gate_type", "status", "model", "slurm_job_id", "provenance", "diagnostic"}),
        "R3 gate",
    )
    _require(gate["gate_type"], "qwen235_r3_routes", "gate_type")
    _require(gate["status"], "passed", "status")
    _require(gate["model"], "qwen3_235b", "model")
    _positive_number(gate["slurm_job_id"], "slurm_job_id")
    _validate_provenance(
        gate["provenance"],
        nemo_rl_commit=args.nemo_rl_commit,
        bridge_commit=args.bridge_commit,
        mcore_commit=args.mcore_commit,
        container_sha256=args.container_sha256,
        runtime_attestation_sha256=args.runtime_attestation_sha256,
    )
    diagnostic = _exact_mapping(gate["diagnostic"], frozenset(R3_DIAGNOSTIC), "diagnostic")
    for field, expected in R3_DIAGNOSTIC.items():
        _require(diagnostic[field], expected, f"diagnostic.{field}")


def _validate_arm(arm: str, value: object) -> None:
    evidence = _exact_mapping(value, ARM_FIELDS, f"arms.{arm}")
    _positive_number(evidence["job_id"], f"arms.{arm}.job_id")
    _require(evidence["status"], "passed", f"arms.{arm}.status")
    _require(evidence["completed_steps"], 5, f"arms.{arm}.completed_steps")
    _require(evidence["metrics_finite"], True, f"arms.{arm}.metrics_finite")
    _require(evidence["correctness_passed"], True, f"arms.{arm}.correctness_passed")
    _require(evidence["undeclared_fallbacks"], 0, f"arms.{arm}.undeclared_fallbacks")
    r3_on = arm in {"C", "E"}
    _require(evidence["router_replay"], "on" if r3_on else "off", f"arms.{arm}.router_replay")
    graph_status = "passed" if arm in {"B", "E"} else "not_applicable"
    _require(evidence["graph_coverage_status"], graph_status, f"arms.{arm}.graph_coverage_status")
    _require(evidence["r3_trace_status"], "passed" if r3_on else "not_applicable", f"arms.{arm}.r3_trace_status")


def _validate_promotion(payload: dict[str, object], args: argparse.Namespace) -> None:
    gate = _exact_mapping(
        payload,
        frozenset({"gate_type", "status", "model", "phase", "steps", "provenance", "arms"}),
        "promotion gate",
    )
    _require(gate["gate_type"], "smoke_promotion", "gate_type")
    _require(gate["status"], "passed", "status")
    _require(gate["model"], args.model, "model")
    _require(gate["phase"], "smoke", "phase")
    _require(gate["steps"], 5, "steps")
    _validate_provenance(
        gate["provenance"],
        nemo_rl_commit=args.nemo_rl_commit,
        bridge_commit=args.bridge_commit,
        mcore_commit=args.mcore_commit,
        container_sha256=args.container_sha256,
        runtime_attestation_sha256=args.runtime_attestation_sha256,
    )
    arms = gate["arms"]
    if not isinstance(arms, dict) or not arms:
        _fail("arms must be a non-empty JSON object")
    unknown = set(arms) - ARMS
    if unknown:
        _fail(f"arms contains unsupported arm(s): {sorted(unknown)}")
    for arm in args.arm:
        if arm not in arms:
            _fail(f"promotion gate does not cover requested arm {arm}")
        _validate_arm(arm, arms[arm])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("r3", "promotion"))
    parser.add_argument("--gate-file", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--model", required=True, choices=("qwen3_30ba3b", "qwen3_235b"))
    parser.add_argument("--nemo-rl-commit", required=True)
    parser.add_argument("--bridge-commit", required=True)
    parser.add_argument("--mcore-commit", required=True)
    parser.add_argument("--container-sha256", required=True)
    parser.add_argument("--runtime-attestation", required=True)
    parser.add_argument("--arm", action="append", default=[])
    args = parser.parse_args()
    if FULL_SHA256.fullmatch(args.gate_sha256) is None:
        parser.error("--gate-sha256 must be a full lowercase SHA256")
    for option in ("nemo_rl_commit", "bridge_commit", "mcore_commit"):
        if FULL_COMMIT.fullmatch(getattr(args, option)) is None:
            parser.error(f"--{option.replace('_', '-')} must be a full lowercase commit")
    if FULL_SHA256.fullmatch(args.container_sha256) is None:
        parser.error("--container-sha256 must be a full lowercase SHA256")
    if args.kind == "r3" and args.arm:
        parser.error("R3 validation does not accept --arm")
    if args.kind == "promotion":
        if not args.arm or len(set(args.arm)) != len(args.arm) or set(args.arm) - ARMS:
            parser.error("promotion validation requires unique A, B, C, or E --arm values")
    return args


def main() -> int:
    args = _parse_args()
    try:
        gate_path = _regular_absolute_file(args.gate_file, "gate file")
        runtime_path = _regular_absolute_file(args.runtime_attestation, "runtime attestation")
        if not hmac.compare_digest(_sha256(gate_path), args.gate_sha256):
            _fail("gate file SHA256 does not match --gate-sha256")
        try:
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            _fail(f"gate file is not valid JSON: {error}")
        if not isinstance(payload, dict):
            _fail("gate file must contain a JSON object")
        args.runtime_attestation_sha256 = _sha256(runtime_path)
        if args.kind == "r3":
            _validate_r3(payload, args)
        else:
            _validate_promotion(payload, args)
    except ValueError as error:
        print(f"Campaign gate rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
