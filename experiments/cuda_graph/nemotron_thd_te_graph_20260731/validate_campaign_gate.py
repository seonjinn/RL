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
import os
import re
import stat
import sys
from pathlib import Path
from typing import NoReturn, cast


FULL_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
FULL_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
PROFILE_ASSIGNMENT = re.compile(r"([A-Z][A-Z0-9_]*)=([A-Za-z0-9_./,:=-]*)\Z")
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
PROFILE_FIELDS = frozenset(
    {
        "PROFILE_ID",
        "ACCOUNT",
        "PARTITION",
        "CONTAINER",
        "CONTAINER_SHA256",
        "MOUNTS",
        "SBATCH_GPUS_PER_NODE",
        "SBATCH_GRES",
        "SBATCH_SEGMENT_SIZE",
        "TIME_LIMIT",
        "RUNTIME_ATTESTATION",
        "RUNTIME_PREFLIGHT_JOB_ID",
        "EXPECTED_TE_SHA",
        "EXPECTED_NEMORL_SHA",
        "EXPECTED_BRIDGE_SHA",
        "EXPECTED_MCORE_SHA",
    }
)
PROFILE_REQUIRED = frozenset(
    {
        "CONTAINER_SHA256",
        "RUNTIME_ATTESTATION",
        "EXPECTED_NEMORL_SHA",
        "EXPECTED_BRIDGE_SHA",
        "EXPECTED_MCORE_SHA",
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


def _read_regular_file(path: Path, label: str) -> bytes:
    """Read one absolute regular non-symlink file through one opened descriptor."""
    if not path.is_absolute():
        _fail(f"{label} must be an absolute path")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        _fail(f"{label} cannot be opened without following symlinks")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as error:
        _fail(f"{label} cannot be opened as a non-symlink file: {error}")
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            _fail(f"{label} must be a regular file")
        content = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return bytes(content)
            content.extend(chunk)
    except OSError as error:
        _fail(f"{label} cannot be read: {error}")
    finally:
        os.close(descriptor)


def _validate_profile_directory(path: Path) -> None:
    if not path.is_absolute():
        _fail("profile directory must be an absolute path")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        _fail("profile directory cannot be opened without following symlinks")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow | directory)
    except OSError as error:
        _fail(f"profile directory is not trusted: {error}")
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            _fail("profile directory must be a directory")
    finally:
        os.close(descriptor)


def _parse_json(content: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(content.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"gate file is not valid JSON: {error}")
    if not isinstance(payload, dict):
        _fail("gate file must contain a JSON object")
    return cast(dict[str, object], payload)


def _parse_profile(content: bytes) -> dict[str, str]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"profile is not valid UTF-8: {error}")
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        match = PROFILE_ASSIGNMENT.fullmatch(line)
        if match is None:
            _fail(f"profile line {line_number} must be a literal NAME=value assignment")
        name, value = match.groups()
        if name not in PROFILE_FIELDS:
            _fail(f"profile line {line_number} uses unknown field {name}")
        if name in values:
            _fail(f"profile line {line_number} duplicates field {name}")
        values[name] = value
    missing = sorted(PROFILE_REQUIRED - set(values))
    if missing:
        _fail(f"profile is missing required fields: {missing}")
    for field in PROFILE_REQUIRED:
        value = values[field]
        if not value or value.startswith("__REQUIRED_"):
            _fail(f"profile has unresolved {field}")
    return values


def _profile_values(args: argparse.Namespace) -> dict[str, str]:
    profile_dir = Path(args.profile_dir)
    _validate_profile_directory(profile_dir)
    if args.profile_file is None:
        candidate = profile_dir / f"{args.cluster}.env"
        if not os.path.lexists(candidate):
            candidate = profile_dir / f"{args.cluster}.env.example"
    else:
        candidate = Path(args.profile_file)
    if not candidate.is_absolute() or candidate.parent != profile_dir:
        _fail("profile file must be a direct child of the trusted profile directory")
    values = _parse_profile(_read_regular_file(candidate, "profile file"))
    if FULL_COMMIT.fullmatch(values["EXPECTED_NEMORL_SHA"]) is None:
        _fail("profile EXPECTED_NEMORL_SHA must be a full lowercase commit")
    if FULL_COMMIT.fullmatch(values["EXPECTED_BRIDGE_SHA"]) is None:
        _fail("profile EXPECTED_BRIDGE_SHA must be a full lowercase commit")
    if FULL_COMMIT.fullmatch(values["EXPECTED_MCORE_SHA"]) is None:
        _fail("profile EXPECTED_MCORE_SHA must be a full lowercase commit")
    if FULL_SHA256.fullmatch(values["CONTAINER_SHA256"]) is None:
        _fail("profile CONTAINER_SHA256 must be a full lowercase SHA256")
    if not Path(values["RUNTIME_ATTESTATION"]).is_absolute():
        _fail("profile RUNTIME_ATTESTATION must be an absolute path")
    return values


def _exact_mapping(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    actual_fields = set(value)
    if actual_fields != fields:
        missing = sorted(fields - actual_fields)
        unknown = sorted(actual_fields - fields)
        _fail(f"{label} fields mismatch: missing={missing}, unknown={unknown}")
    return cast(dict[str, object], value)


def _positive_job_id(value: object, label: str) -> None:
    if type(value) is not int or value <= 0:
        _fail(f"{label} must be a positive integer job ID")


def _require(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        _fail(f"{label} must equal {expected!r}")


def _validate_provenance(value: object, expected: dict[str, str]) -> None:
    provenance = _exact_mapping(value, PROVENANCE_FIELDS, "provenance")
    for field, expected_value in expected.items():
        candidate = provenance[field]
        if not isinstance(candidate, str):
            _fail(f"provenance.{field} must be a string")
        pattern = FULL_SHA256 if field.endswith("sha256") else FULL_COMMIT
        if pattern.fullmatch(candidate) is None:
            _fail(f"provenance.{field} must be a full lowercase digest")
        _require(candidate, expected_value, f"provenance.{field}")


def _validate_r3(payload: dict[str, object], expected: dict[str, str]) -> None:
    gate = _exact_mapping(
        payload,
        frozenset({"gate_type", "status", "model", "slurm_job_id", "provenance", "diagnostic"}),
        "R3 gate",
    )
    _require(gate["gate_type"], "qwen235_r3_routes", "gate_type")
    _require(gate["status"], "passed", "status")
    _require(gate["model"], "qwen3_235b", "model")
    _positive_job_id(gate["slurm_job_id"], "slurm_job_id")
    _validate_provenance(gate["provenance"], expected)
    diagnostic = _exact_mapping(gate["diagnostic"], frozenset(R3_DIAGNOSTIC), "diagnostic")
    for field, expected_value in R3_DIAGNOSTIC.items():
        _require(diagnostic[field], expected_value, f"diagnostic.{field}")


def _validate_arm(arm: str, value: object) -> None:
    evidence = _exact_mapping(value, ARM_FIELDS, f"arms.{arm}")
    _positive_job_id(evidence["job_id"], f"arms.{arm}.job_id")
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


def _validate_promotion(payload: dict[str, object], args: argparse.Namespace, expected: dict[str, str]) -> None:
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
    _validate_provenance(gate["provenance"], expected)
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
    parser.add_argument("--profile-file")
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--cluster", required=True, choices=("ptyche", "oci-hsg", "lyris"))
    parser.add_argument("--arm", action="append", default=[])
    args = parser.parse_args()
    if FULL_SHA256.fullmatch(args.gate_sha256) is None:
        parser.error("--gate-sha256 must be a full lowercase SHA256")
    if args.kind == "r3":
        if args.model != "qwen3_235b":
            parser.error("R3 validation requires --model qwen3_235b")
        if args.arm:
            parser.error("R3 validation does not accept --arm")
    if args.kind == "promotion":
        if not args.arm or len(set(args.arm)) != len(args.arm) or set(args.arm) - ARMS:
            parser.error("promotion validation requires unique A, B, C, or E --arm values")
    return args


def main() -> int:
    args = _parse_args()
    try:
        profile = _profile_values(args)
        gate_content = _read_regular_file(Path(args.gate_file), "gate file")
        if not hmac.compare_digest(hashlib.sha256(gate_content).hexdigest(), args.gate_sha256):
            _fail("gate file SHA256 does not match --gate-sha256")
        runtime_content = _read_regular_file(
            Path(profile["RUNTIME_ATTESTATION"]), "runtime attestation"
        )
        expected = {
            "nemo_rl_commit": profile["EXPECTED_NEMORL_SHA"],
            "bridge_commit": profile["EXPECTED_BRIDGE_SHA"],
            "mcore_commit": profile["EXPECTED_MCORE_SHA"],
            "container_sha256": profile["CONTAINER_SHA256"],
            "runtime_attestation_sha256": hashlib.sha256(runtime_content).hexdigest(),
        }
        payload = _parse_json(gate_content)
        if args.kind == "r3":
            _validate_r3(payload, expected)
        else:
            _validate_promotion(payload, args, expected)
    except ValueError as error:
        print(f"Campaign gate rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
