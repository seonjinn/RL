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

"""Probe the four-GPU OCI training runtime and record package versions."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


REQUIRED_MODULE_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "torch": ("torch",),
    "transformer_engine.pytorch": ("transformer-engine",),
    "megatron.core": ("megatron-core",),
    "megatron.bridge": ("megatron-bridge",),
    "mamba_ssm": ("mamba-ssm",),
    "causal_conv1d": ("causal-conv1d",),
    "cupy": ("cupy-cuda13x", "cupy-cuda12x", "cupy"),
    "grouped_gemm": ("nv-grouped-gemm", "grouped-gemm"),
}
EDITABLE_PROJECT_MODULES = frozenset(("megatron.core", "megatron.bridge"))
FULL_COMMIT_LENGTH = 40


def _distribution_version(
    candidates: tuple[str, ...],
    version_getter: Callable[[str], str],
) -> tuple[str, str]:
    errors: list[str] = []
    for distribution in candidates:
        try:
            return distribution, version_getter(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(distribution)
    raise RuntimeError(
        "no installed distribution metadata for any of: " + ", ".join(errors)
    )


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_path_within(*, label: str, path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"{label} resolved outside expected uv environment: {path} (root: {root})"
        ) from error


def _distribution_vcs_commit(
    distribution: str,
    distribution_getter: Callable[[str], Any] = importlib.metadata.distribution,
) -> str:
    metadata = distribution_getter(distribution).read_text("direct_url.json")
    if metadata is None:
        raise RuntimeError(f"{distribution} has no direct_url.json provenance")
    try:
        direct_url = json.loads(metadata)
        vcs_info = direct_url["vcs_info"]
        vcs = vcs_info["vcs"]
        commit = vcs_info["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{distribution} direct_url.json lacks Git commit provenance"
        ) from error
    if vcs != "git" or not isinstance(commit, str) or len(commit) != FULL_COMMIT_LENGTH:
        raise RuntimeError(
            f"{distribution} direct_url.json has invalid Git commit provenance"
        )
    try:
        int(commit, 16)
    except ValueError as error:
        raise RuntimeError(
            f"{distribution} direct_url.json has invalid Git commit provenance"
        ) from error
    return commit.lower()


def _version_pair(version: str) -> tuple[int, int]:
    components = version.split(".", 2)
    if len(components) < 2:
        raise RuntimeError(f"unparseable Transformer Engine version: {version!r}")
    try:
        return int(components[0]), int(components[1])
    except ValueError as error:
        raise RuntimeError(
            f"unparseable Transformer Engine version: {version!r}"
        ) from error


def probe_runtime(
    *,
    expected_device_count: int,
    expected_environment_root: Path | None = None,
    expected_project_root: Path | None = None,
    importer: Callable[[str], Any] = importlib.import_module,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    interpreter_path: str | os.PathLike[str] = sys.executable,
    runtime_prefix: str | os.PathLike[str] = sys.prefix,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    """Import the training stack and require exactly the allocated GPUs."""
    environment_root: Path | None = None
    project_root = (
        expected_project_root.resolve(strict=False)
        if expected_project_root is not None
        else None
    )
    python_executable = _absolute_path(interpreter_path)
    python_prefix = _absolute_path(runtime_prefix)
    if expected_environment_root is not None:
        environment_root = expected_environment_root.resolve(strict=False)
        for variable in ("PYTHONHOME", "PYTHONPATH"):
            if environment.get(variable):
                raise RuntimeError(
                    f"ambient {variable} must be unset for the immutable container probe"
                )
        configured_environment = environment.get("UV_PROJECT_ENVIRONMENT")
        if configured_environment is None:
            raise RuntimeError("UV_PROJECT_ENVIRONMENT is not set")
        if _absolute_path(configured_environment) != environment_root:
            raise RuntimeError(
                "UV_PROJECT_ENVIRONMENT does not match the expected uv environment: "
                f"{configured_environment} != {environment_root}"
            )
        _require_path_within(
            label="python executable", path=python_executable, root=environment_root
        )
        _require_path_within(
            label="runtime prefix", path=python_prefix, root=environment_root
        )

    torch_module = importer("torch")
    cuda = torch_module.cuda
    cuda_available = bool(cuda.is_available())
    device_count = int(cuda.device_count())
    if not cuda_available:
        raise RuntimeError("torch.cuda.is_available() is false")
    if device_count != expected_device_count:
        raise RuntimeError(
            f"expected exactly {expected_device_count} visible CUDA devices, "
            f"got {device_count}"
        )

    modules = {"torch": torch_module}
    for module_name in REQUIRED_MODULE_DISTRIBUTIONS:
        if module_name != "torch":
            modules[module_name] = importer(module_name)

    packages: dict[str, dict[str, str]] = {}
    for module_name, distributions in REQUIRED_MODULE_DISTRIBUTIONS.items():
        distribution, version = _distribution_version(distributions, version_getter)
        module_file = getattr(modules[module_name], "__file__", None)
        if module_file is None:
            raise RuntimeError(f"import {module_name} has no filesystem location")
        module_path = Path(module_file).resolve(strict=False)
        if environment_root is not None:
            try:
                _require_path_within(
                    label=f"import {module_name}",
                    path=module_path,
                    root=environment_root,
                )
            except RuntimeError:
                if module_name not in EDITABLE_PROJECT_MODULES or project_root is None:
                    raise
                _require_path_within(
                    label=f"editable import {module_name}",
                    path=module_path,
                    root=project_root,
                )
        packages[module_name] = {
            "distribution": distribution,
            "version": str(version),
            "path": str(module_path),
        }

    devices = [
        {
            "index": index,
            "name": str(cuda.get_device_name(index)),
            "capability": list(cuda.get_device_capability(index)),
        }
        for index in range(device_count)
    ]
    torch_version = getattr(torch_module, "version", None)
    return {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "expected_device_count": expected_device_count,
        "expected_environment_root": (
            str(environment_root) if environment_root is not None else None
        ),
        "expected_project_root": str(project_root)
        if project_root is not None
        else None,
        "python_executable": str(python_executable),
        "runtime_prefix": str(python_prefix),
        "torch_cuda_version": getattr(torch_version, "cuda", None),
        "devices": devices,
        "packages": packages,
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--container-sha256", required=True)
    parser.add_argument("--expected-device-count", required=True, type=int)
    parser.add_argument("--expected-environment-root", required=True, type=Path)
    parser.add_argument("--expected-project-root", required=True, type=Path)
    parser.add_argument("--nemo-rl-commit", required=True)
    parser.add_argument("--bridge-commit", required=True)
    parser.add_argument("--mcore-commit", required=True)
    parser.add_argument("--uv-lock-sha256", required=True)
    parser.add_argument("--expected-te-commit", required=True)
    parser.add_argument("--container-device", required=True, type=int)
    parser.add_argument("--container-inode", required=True, type=int)
    parser.add_argument("--container-size", required=True, type=int)
    parser.add_argument("--container-mtime-seconds", required=True, type=int)
    parser.add_argument("--container-ctime-seconds", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = {
        "container_image": args.container_image,
        "container_sha256": args.container_sha256,
        "nemo_rl_commit": args.nemo_rl_commit,
        "bridge_commit": args.bridge_commit,
        "mcore_commit": args.mcore_commit,
        "uv_lock_sha256": args.uv_lock_sha256,
        "expected_te_commit": args.expected_te_commit,
        "container_device": args.container_device,
        "container_inode": args.container_inode,
        "container_size": args.container_size,
        "container_mtime_seconds": args.container_mtime_seconds,
        "container_ctime_seconds": args.container_ctime_seconds,
    }
    try:
        runtime = probe_runtime(
            expected_device_count=args.expected_device_count,
            expected_environment_root=args.expected_environment_root,
            expected_project_root=args.expected_project_root,
        )
        te_version = runtime["packages"]["transformer_engine.pytorch"]["version"]
        if _version_pair(te_version) < (2, 16):
            raise RuntimeError(
                f"runtime requires Transformer Engine >= 2.16, got {te_version}"
            )
        te_commit = _distribution_vcs_commit("transformer-engine")
        if te_commit != args.expected_te_commit:
            raise RuntimeError(
                "Transformer Engine VCS commit mismatch: "
                f"expected {args.expected_te_commit}, got {te_commit}"
            )
        payload = {
            "status": "passed",
            **context,
            "transformer_engine_vcs_commit": te_commit,
            **runtime,
        }
    except Exception as error:
        payload = {
            "status": "failed",
            **context,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json_atomic(payload, args.output)
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(1) from error
    _write_json_atomic(payload, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
