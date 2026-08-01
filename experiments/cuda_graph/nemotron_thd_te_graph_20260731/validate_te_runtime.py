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

"""Verify an immutable, container-owned Transformer Engine runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MINIMUM_TE_VERSION = (2, 16)
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


def validate_resolved_paths(
    resolved: Mapping[str, Path],
    *,
    site_packages: Path,
) -> dict[str, str]:
    """Require imported Python and native TE artifacts under one prefix."""
    outside = {
        name: str(path.resolve(strict=False))
        for name, path in resolved.items()
        if not _within(path, site_packages)
    }
    if outside:
        raise ValueError(
            "Transformer Engine resolved outside the native runtime: "
            + json.dumps(outside, sort_keys=True)
        )
    return {name: str(path.resolve(strict=False)) for name, path in resolved.items()}


def _version_pair(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\D|$)", version)
    if match is None:
        raise ValueError(f"unparseable Transformer Engine version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def _verify_ancestry(
    source_repository: Path,
    *,
    minimum_commit: str,
    native_commit: str,
) -> None:
    if not source_repository.is_dir():
        raise ValueError(
            f"trusted Transformer Engine source repository is missing: {source_repository}"
        )
    for label, revision in (
        ("minimum commit", minimum_commit),
        ("native commit", native_commit),
    ):
        if FULL_COMMIT.fullmatch(revision) is None:
            raise ValueError(f"{label} must be a full lowercase 40-character SHA")
        exists = subprocess.run(
            [
                "git",
                "-C",
                str(source_repository),
                "cat-file",
                "-e",
                f"{revision}^{{commit}}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if exists.returncode != 0:
            raise ValueError(
                f"{label} {revision} is absent from trusted repository "
                f"{source_repository}"
            )

    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(source_repository),
            "merge-base",
            "--is-ancestor",
            minimum_commit,
            native_commit,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode == 1:
        raise ValueError(
            f"native commit {native_commit} is not at or after minimum commit "
            f"{minimum_commit}"
        )
    if ancestry.returncode != 0:
        raise RuntimeError(
            "git could not verify Transformer Engine commit ancestry: "
            + ancestry.stderr.strip()
        )


def _validate_imports(site_packages: Path, expected_version: str) -> dict[str, str]:
    sys.path.insert(0, str(site_packages))
    try:
        importlib.invalidate_caches()
        transformer_engine = importlib.import_module("transformer_engine")
        transformer_engine_pytorch = importlib.import_module(
            "transformer_engine.pytorch"
        )
        transformer_engine_torch = importlib.import_module("transformer_engine_torch")
        actual_version = str(transformer_engine.__version__)
        if actual_version != expected_version:
            raise ValueError(
                "Transformer Engine imported version mismatch: "
                f"expected {expected_version}, got {actual_version}"
            )

        common = importlib.import_module("transformer_engine.common")
        core_library = Path(common._get_shared_object_file("core"))
        module_files = {
            "transformer_engine": transformer_engine.__file__,
            "transformer_engine.pytorch": transformer_engine_pytorch.__file__,
            "transformer_engine_torch": transformer_engine_torch.__file__,
        }
        missing_locations = [
            name for name, path in module_files.items() if path is None
        ]
        if missing_locations:
            raise ValueError(
                "Transformer Engine imports have no filesystem location: "
                + ", ".join(missing_locations)
            )
        return validate_resolved_paths(
            {
                name: Path(path)
                for name, path in module_files.items()
                if path is not None
            }
            | {"libtransformer_engine": core_library},
            site_packages=site_packages,
        )
    finally:
        sys.path.pop(0)


def _write_json_atomic(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(payload, temporary, allow_nan=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
        output.chmod(0o644)
    except BaseException:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise


def validate_runtime(
    *,
    provenance_path: Path,
    site_packages: Path,
    container: Path,
    expected_container_sha256: str,
    source_repository: Path,
    minimum_commit: str,
    output: Path,
    validate_imports: bool,
) -> dict[str, Any]:
    """Validate immutable TE provenance and publish success evidence."""
    if FULL_SHA256.fullmatch(expected_container_sha256) is None:
        raise ValueError(
            "expected container SHA256 must be 64 lowercase hex characters"
        )
    if not container.is_file():
        raise ValueError(f"immutable container is missing: {container}")
    actual_container_sha256 = _sha256(container)
    if actual_container_sha256 != expected_container_sha256:
        raise ValueError(
            "container SHA256 mismatch: "
            f"expected {expected_container_sha256}, got {actual_container_sha256}"
        )
    if not provenance_path.is_file():
        raise ValueError(f"Transformer Engine provenance is missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    if not isinstance(provenance, dict):
        raise ValueError("Transformer Engine provenance must be a JSON object")

    expected_provenance = {
        "container": str(container.resolve()),
        "container_sha256": expected_container_sha256,
        "install_prefix": str(site_packages.parent.resolve()),
        "site_packages": str(site_packages.resolve()),
        "source_repository": str(source_repository.resolve()),
    }
    mismatches = {
        key: {"expected": expected, "actual": provenance.get(key)}
        for key, expected in expected_provenance.items()
        if provenance.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Transformer Engine runtime provenance mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    version = provenance.get("transformer_engine_version")
    native_commit = provenance.get("transformer_engine_commit")
    if not isinstance(version, str):
        raise ValueError("provenance lacks transformer_engine_version")
    if not isinstance(native_commit, str):
        raise ValueError("provenance lacks transformer_engine_commit")
    if _version_pair(version) < MINIMUM_TE_VERSION:
        raise ValueError(f"runtime requires Transformer Engine >= 2.16, got {version}")
    _verify_ancestry(
        source_repository,
        minimum_commit=minimum_commit,
        native_commit=native_commit,
    )

    required_artifacts = (
        site_packages / "transformer_engine" / "__init__.py",
        site_packages / "transformer_engine" / "libtransformer_engine.so",
    )
    missing_artifacts = [str(path) for path in required_artifacts if not path.is_file()]
    if missing_artifacts:
        raise ValueError(
            "Transformer Engine native runtime is incomplete: "
            + json.dumps(missing_artifacts)
        )

    resolved_imports = (
        _validate_imports(site_packages, version) if validate_imports else {}
    )
    result: dict[str, Any] = {
        "status": "passed",
        "container": str(container.resolve()),
        "container_sha256": actual_container_sha256,
        "install_prefix": str(site_packages.parent.resolve()),
        "site_packages": str(site_packages.resolve()),
        "source_repository": str(source_repository.resolve()),
        "transformer_engine_version": version,
        "transformer_engine_commit": native_commit,
        "minimum_commit": minimum_commit,
        "ancestry_verified": True,
        "imports_validated": validate_imports,
        "resolved_imports": resolved_imports,
    }
    _write_json_atomic(result, output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--site-packages", required=True, type=Path)
    parser.add_argument("--container", required=True, type=Path)
    parser.add_argument("--expected-container-sha256", required=True)
    parser.add_argument("--source-repository", required=True, type=Path)
    parser.add_argument("--minimum-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validate-imports", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = validate_runtime(
            provenance_path=args.provenance,
            site_packages=args.site_packages,
            container=args.container,
            expected_container_sha256=args.expected_container_sha256,
            source_repository=args.source_repository,
            minimum_commit=args.minimum_commit,
            output=args.output,
            validate_imports=args.validate_imports,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
