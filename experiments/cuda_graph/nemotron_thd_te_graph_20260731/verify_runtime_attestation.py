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

"""Verify one immutable OCI runtime preflight artifact for a leaf job."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MINIMUM_TE_VERSION = (2, 16)
REQUIRED_PACKAGES = frozenset(
    (
        "torch",
        "transformer_engine.pytorch",
        "megatron.core",
        "megatron.bridge",
        "mamba_ssm",
        "causal_conv1d",
        "cupy",
        "grouped_gemm",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_pair(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\D|$)", version)
    if match is None:
        raise ValueError(f"unparseable Transformer Engine version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def _require_full_commit(label: str, commit: str) -> None:
    if FULL_COMMIT.fullmatch(commit) is None:
        raise ValueError(f"{label} must be a full lowercase 40-character SHA")


def _container_identity(container: Path) -> dict[str, int]:
    if container.is_symlink():
        raise ValueError(f"immutable container must not be a symlink: {container}")
    if not container.is_file():
        raise ValueError(f"immutable container is missing: {container}")
    status = container.stat()
    return {
        "container_device": status.st_dev,
        "container_inode": status.st_ino,
        "container_size": status.st_size,
        "container_mtime_seconds": int(status.st_mtime),
        "container_ctime_seconds": int(status.st_ctime),
    }


def _require_path_within(*, label: str, path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} is outside {root}: {path}") from error


def _read_attestation(attestation: Path) -> dict[str, Any]:
    if attestation.is_symlink():
        raise ValueError(f"runtime attestation must not be a symlink: {attestation}")
    if not attestation.is_file():
        raise ValueError(f"runtime attestation is missing: {attestation}")
    try:
        payload = json.loads(attestation.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"runtime attestation is not valid JSON: {attestation}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("runtime attestation must contain a JSON object")
    return payload


def validate_attestation(
    *,
    attestation: Path,
    container: Path,
    expected_container_sha256: str,
    nemo_rl_commit: str,
    bridge_commit: str,
    mcore_commit: str,
    uv_lock: Path,
    expected_te_commit: str,
    expected_device_count: int,
    expected_python_version: str,
    expected_python_install_dir: Path,
) -> dict[str, Any]:
    """Require exact source, image, TE, GPU, and worker-stack provenance."""
    if FULL_SHA256.fullmatch(expected_container_sha256) is None:
        raise ValueError(
            "expected container SHA256 must be 64 lowercase hexadecimal characters"
        )
    for label, commit in (
        ("NeMo-RL commit", nemo_rl_commit),
        ("Megatron-Bridge commit", bridge_commit),
        ("Megatron-LM commit", mcore_commit),
        ("Transformer Engine commit", expected_te_commit),
    ):
        _require_full_commit(label, commit)
    if expected_device_count <= 0:
        raise ValueError("expected device count must be positive")
    if re.fullmatch(r"\d+\.\d+\.\d+", expected_python_version) is None:
        raise ValueError("expected Python version must be an exact X.Y.Z version")
    if not expected_python_install_dir.is_absolute():
        raise ValueError("expected Python install directory must be absolute")
    if expected_python_install_dir.is_symlink():
        raise ValueError(
            "expected Python install directory must not be a symlink: "
            f"{expected_python_install_dir}"
        )
    expected_python_install_dir = expected_python_install_dir.resolve(strict=False)
    if not expected_python_install_dir.is_dir():
        raise ValueError(
            "expected Python install directory is missing: "
            f"{expected_python_install_dir}"
        )
    if not uv_lock.is_file():
        raise ValueError(f"uv.lock is missing: {uv_lock}")

    payload = _read_attestation(attestation)
    if payload.get("status") != "passed":
        raise ValueError("runtime attestation status is not passed")

    expected_provenance: dict[str, object] = {
        "container_image": str(container),
        "container_sha256": expected_container_sha256,
        "nemo_rl_commit": nemo_rl_commit,
        "bridge_commit": bridge_commit,
        "mcore_commit": mcore_commit,
        "expected_te_commit": expected_te_commit,
        "transformer_engine_vcs_commit": expected_te_commit,
        "device_count": expected_device_count,
        "expected_device_count": expected_device_count,
        "expected_python_version": expected_python_version,
        "python_version": expected_python_version,
        "uv_python_install_dir": str(expected_python_install_dir),
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in expected_provenance.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "runtime attestation provenance mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    actual_identity = _container_identity(container)
    identity_mismatches = {
        key: {"expected": payload.get(key), "actual": actual}
        for key, actual in actual_identity.items()
        if payload.get(key) != actual
    }
    if identity_mismatches:
        raise ValueError(
            "container identity mismatch after runtime preflight: "
            + json.dumps(identity_mismatches, sort_keys=True)
        )

    lock_sha256 = _sha256(uv_lock)
    if payload.get("uv_lock_sha256") != lock_sha256:
        raise ValueError(
            "uv.lock SHA256 mismatch after runtime preflight: "
            f"expected {payload.get('uv_lock_sha256')}, got {lock_sha256}"
        )

    python_base_executable_value = payload.get("python_base_executable")
    if not isinstance(python_base_executable_value, str):
        raise ValueError("runtime attestation lacks managed Python executable")
    python_base_executable = Path(python_base_executable_value)
    if not python_base_executable.is_absolute():
        raise ValueError("managed Python executable must be absolute")
    if python_base_executable.is_symlink():
        raise ValueError(
            f"managed Python executable must not be a symlink: {python_base_executable}"
        )
    python_base_executable = python_base_executable.resolve(strict=False)
    _require_path_within(
        label="managed Python executable",
        path=python_base_executable,
        root=expected_python_install_dir,
    )
    if not python_base_executable.is_file():
        raise ValueError(
            f"managed Python executable is missing: {python_base_executable}"
        )
    expected_python_sha256 = payload.get("python_base_executable_sha256")
    if (
        not isinstance(expected_python_sha256, str)
        or FULL_SHA256.fullmatch(expected_python_sha256) is None
    ):
        raise ValueError("runtime attestation lacks managed Python executable SHA256")
    actual_python_sha256 = _sha256(python_base_executable)
    if actual_python_sha256 != expected_python_sha256:
        raise ValueError(
            "managed Python executable SHA256 mismatch: "
            f"expected {expected_python_sha256}, got {actual_python_sha256}"
        )

    packages = payload.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError("runtime attestation packages must be a JSON object")
    missing_packages = sorted(REQUIRED_PACKAGES.difference(packages))
    if missing_packages:
        raise ValueError(
            "runtime attestation is missing required packages: "
            + ", ".join(missing_packages)
        )
    te_package = packages["transformer_engine.pytorch"]
    if not isinstance(te_package, Mapping) or not isinstance(
        te_package.get("version"), str
    ):
        raise ValueError("runtime attestation lacks Transformer Engine version")
    te_version = te_package["version"]
    if _version_pair(te_version) < MINIMUM_TE_VERSION:
        raise ValueError(
            f"runtime requires Transformer Engine >= 2.16, got {te_version}"
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--container", required=True, type=Path)
    parser.add_argument("--expected-container-sha256", required=True)
    parser.add_argument("--nemo-rl-commit", required=True)
    parser.add_argument("--bridge-commit", required=True)
    parser.add_argument("--mcore-commit", required=True)
    parser.add_argument("--uv-lock", required=True, type=Path)
    parser.add_argument("--expected-te-commit", required=True)
    parser.add_argument("--expected-device-count", required=True, type=int)
    parser.add_argument("--expected-python-version", required=True)
    parser.add_argument("--expected-python-install-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_attestation(
        attestation=args.attestation,
        container=args.container,
        expected_container_sha256=args.expected_container_sha256,
        nemo_rl_commit=args.nemo_rl_commit,
        bridge_commit=args.bridge_commit,
        mcore_commit=args.mcore_commit,
        uv_lock=args.uv_lock,
        expected_te_commit=args.expected_te_commit,
        expected_device_count=args.expected_device_count,
        expected_python_version=args.expected_python_version,
        expected_python_install_dir=args.expected_python_install_dir,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
