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
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


REQUIRED_MODULE_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "torch": ("torch",),
    "transformer_engine.pytorch": ("transformer-engine",),
    "megatron.core": ("megatron-core",),
    "megatron.core.extensions.transformer_engine": ("megatron-core",),
    "megatron.bridge": ("megatron-bridge",),
    "mamba_ssm": ("mamba-ssm",),
    "causal_conv1d": ("causal-conv1d",),
    "cupy": ("cupy-cuda13x", "cupy-cuda12x", "cupy"),
}
TE_EVAL_FEATURE_SET = "te_eval_capability_8"
BRIDGE_EVAL_FEATURE_SET = "bridge_forward_only_eval_8"
NARROW_EVAL_FEATURE_SETS = frozenset((TE_EVAL_FEATURE_SET, BRIDGE_EVAL_FEATURE_SET))
NANO_HYBRIDEP_FEATURE_SET = "dropless_hybridep_nano16"
QWEN30_ALLTOALL_FEATURE_SET = "dropless_alltoall_qwen30_16"
SUPER_ALLTOALL_FEATURE_SET = "dropless_alltoall_super32"
QWEN235_HYBRIDEP_FEATURE_SET = "dropless_hybridep_qwen235_64"
DROPLESS_MOE_FEATURE_SETS = frozenset(
    (
        NANO_HYBRIDEP_FEATURE_SET,
        QWEN30_ALLTOALL_FEATURE_SET,
        SUPER_ALLTOALL_FEATURE_SET,
        QWEN235_HYBRIDEP_FEATURE_SET,
    )
)
HYBRIDEP_FEATURE_SETS = frozenset(
    (NANO_HYBRIDEP_FEATURE_SET, QWEN235_HYBRIDEP_FEATURE_SET)
)
ALLTOALL_FEATURE_SETS = frozenset(
    (QWEN30_ALLTOALL_FEATURE_SET, SUPER_ALLTOALL_FEATURE_SET)
)
TE_EVAL_EXCLUDED_PACKAGES = (
    "causal-conv1d",
    "deep-ep",
    "fast-hadamard-transform",
    "mamba-ssm",
)
HYBRIDEP_EXCLUDED_PACKAGES = ("fast-hadamard-transform",)
ALLTOALL_EXCLUDED_PACKAGES = ("deep-ep", "fast-hadamard-transform")
RUNTIME_FEATURE_EXCLUSIONS = {
    TE_EVAL_FEATURE_SET: TE_EVAL_EXCLUDED_PACKAGES,
    BRIDGE_EVAL_FEATURE_SET: TE_EVAL_EXCLUDED_PACKAGES,
    **{
        feature_set: HYBRIDEP_EXCLUDED_PACKAGES for feature_set in HYBRIDEP_FEATURE_SETS
    },
    **{
        feature_set: ALLTOALL_EXCLUDED_PACKAGES for feature_set in ALLTOALL_FEATURE_SETS
    },
}
TE_EVAL_OPTIONAL_MODULES = frozenset(("mamba_ssm", "causal_conv1d"))
EDITABLE_PROJECT_MODULES = frozenset(
    (
        "megatron.core",
        "megatron.core.extensions.transformer_engine",
        "megatron.bridge",
    )
)
TE_GROUPED_LINEAR_MODULE = "megatron.core.extensions.transformer_engine"
REQUIRED_TE_GROUPED_LINEAR_SYMBOLS = (
    "TEColumnParallelGroupedLinear",
    "TERowParallelGroupedLinear",
)
FULL_COMMIT_LENGTH = 40
DEFAULT_BASE_EXECUTABLE = getattr(sys, "_base_executable", sys.executable)
NCCL_EP_EXTENSION_SYMBOLS = (
    "ep_initialize",
    "ep_finalize",
    "ep_get_zero_copy",
    "ep_handle_mem_size",
    "ep_prepare",
    "ep_dispatch",
    "ep_combine",
    "ep_dispatch_bwd",
    "ep_combine_bwd",
)
HYBRIDEP_BUFFER_SYMBOL = "HybridEPBuffer"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_transformer_engine_identities(
    *,
    version: str,
    source_commit: str,
    expected_source_commit: str,
    expected_version_base_commit: str,
) -> dict[str, str]:
    """Validate distinct install-source and version-base TE commit identities."""
    for label, commit in (
        ("source commit", source_commit),
        ("expected source commit", expected_source_commit),
        ("expected version-base commit", expected_version_base_commit),
    ):
        if (
            len(commit) != FULL_COMMIT_LENGTH
            or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        ):
            raise RuntimeError(
                f"Transformer Engine {label} must be one lowercase full SHA"
            )
    if source_commit != expected_source_commit:
        raise RuntimeError(
            "Transformer Engine source commit mismatch: "
            f"expected {expected_source_commit}, got {source_commit}"
        )
    version_match = re.search(r"\+([0-9a-f]{8})(?:\D|$)", version)
    if (
        version_match is None
        or version_match.group(1) != expected_version_base_commit[:8]
    ):
        raise RuntimeError(
            "Transformer Engine version-base commit mismatch: "
            f"expected {expected_version_base_commit[:8]} in {version}"
        )
    return {
        "transformer_engine_source_commit": source_commit,
        "transformer_engine_version_base_commit": expected_version_base_commit,
    }


def probe_runtime(
    *,
    expected_device_count: int,
    expected_environment_root: Path | None = None,
    expected_project_root: Path | None = None,
    expected_python_version: str | None = None,
    expected_python_install_dir: Path | None = None,
    expected_uv_version: str | None = None,
    expected_uv_executable: Path | None = None,
    expected_nvte_with_nccl_ep: str | None = None,
    expected_runtime_feature_set: str | None = None,
    expected_excluded_packages: tuple[str, ...] | None = None,
    expected_torch_cuda_arch_list: str | None = None,
    expected_nvte_cuda_archs: str | None = None,
    importer: Callable[[str], Any] = importlib.import_module,
    optional_importer: Callable[[str], Any] = importlib.import_module,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    interpreter_path: str | os.PathLike[str] = sys.executable,
    base_interpreter_path: str | os.PathLike[str] = DEFAULT_BASE_EXECUTABLE,
    runtime_prefix: str | os.PathLike[str] = sys.prefix,
    python_version: str = platform.python_version(),
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    """Import the training stack and require exactly the allocated GPUs."""
    if expected_runtime_feature_set is not None:
        feature_exclusions = RUNTIME_FEATURE_EXCLUSIONS.get(
            expected_runtime_feature_set
        )
        if feature_exclusions is None:
            raise RuntimeError("unsupported runtime feature set")
        if expected_excluded_packages != feature_exclusions:
            raise RuntimeError("runtime exclusions do not match the typed feature set")
        assert expected_excluded_packages is not None
        if environment.get("RUNTIME_FEATURE_SET") != expected_runtime_feature_set:
            raise RuntimeError("RUNTIME_FEATURE_SET mismatch")
        if environment.get("RUNTIME_EXCLUDED_PACKAGES") != ",".join(
            expected_excluded_packages
        ):
            raise RuntimeError("RUNTIME_EXCLUDED_PACKAGES mismatch")
        if environment.get("TORCH_CUDA_ARCH_LIST") != expected_torch_cuda_arch_list:
            raise RuntimeError("TORCH_CUDA_ARCH_LIST mismatch")
        if environment.get("NVTE_CUDA_ARCHS") != expected_nvte_cuda_archs:
            raise RuntimeError("NVTE_CUDA_ARCHS mismatch")
    nvte_with_nccl_ep = environment.get("NVTE_WITH_NCCL_EP")
    if expected_nvte_with_nccl_ep is not None:
        if expected_nvte_with_nccl_ep not in {"0", "1"}:
            raise RuntimeError("expected NVTE_WITH_NCCL_EP must be 0 or 1")
        if nvte_with_nccl_ep != expected_nvte_with_nccl_ep:
            raise RuntimeError(
                "NVTE_WITH_NCCL_EP mismatch: "
                f"expected {expected_nvte_with_nccl_ep}, got {nvte_with_nccl_ep!r}"
            )
    transformer_engine_nccl_ep_available: bool | None = None
    transformer_engine_nccl_ep_symbols: list[str] | None = None
    if expected_nvte_with_nccl_ep is not None:
        try:
            transformer_engine_torch = optional_importer("transformer_engine_torch")
        except ImportError:
            transformer_engine_nccl_ep_symbols = []
        else:
            transformer_engine_nccl_ep_symbols = [
                symbol
                for symbol in NCCL_EP_EXTENSION_SYMBOLS
                if hasattr(transformer_engine_torch, symbol)
            ]
        if transformer_engine_nccl_ep_symbols and len(
            transformer_engine_nccl_ep_symbols
        ) != len(NCCL_EP_EXTENSION_SYMBOLS):
            raise RuntimeError(
                "Transformer Engine has incomplete NCCL-EP extension bindings: "
                + ", ".join(transformer_engine_nccl_ep_symbols)
            )
        transformer_engine_nccl_ep_available = bool(transformer_engine_nccl_ep_symbols)
        if expected_nvte_with_nccl_ep == "0" and transformer_engine_nccl_ep_available:
            raise RuntimeError(
                "Transformer Engine NCCL-EP module is available despite "
                "NVTE_WITH_NCCL_EP=0"
            )
        if (
            expected_nvte_with_nccl_ep == "1"
            and not transformer_engine_nccl_ep_available
        ):
            raise RuntimeError(
                "Transformer Engine NCCL-EP module is unavailable despite "
                "NVTE_WITH_NCCL_EP=1"
            )
    environment_root: Path | None = None
    project_root = (
        expected_project_root.resolve(strict=False)
        if expected_project_root is not None
        else None
    )
    python_executable = _absolute_path(interpreter_path)
    python_base_executable = _absolute_path(base_interpreter_path).resolve(strict=False)
    python_prefix = _absolute_path(runtime_prefix)
    if (expected_python_version is None) != (expected_python_install_dir is None):
        raise RuntimeError(
            "expected Python version and install directory must be provided together"
        )
    python_install_dir: Path | None = None
    python_base_executable_sha256: str | None = None
    if expected_python_version is not None and expected_python_install_dir is not None:
        python_install_dir = expected_python_install_dir.resolve(strict=False)
        configured_install_dir = environment.get("UV_PYTHON_INSTALL_DIR")
        if configured_install_dir is None:
            raise RuntimeError("UV_PYTHON_INSTALL_DIR is not set")
        if (
            _absolute_path(configured_install_dir).resolve(strict=False)
            != python_install_dir
        ):
            raise RuntimeError(
                "UV_PYTHON_INSTALL_DIR does not match the expected managed Python "
                f"directory: {configured_install_dir} != {python_install_dir}"
            )
        if environment.get("UV_MANAGED_PYTHON") != "1":
            raise RuntimeError("UV_MANAGED_PYTHON must be 1")
        if environment.get("UV_PYTHON_DOWNLOADS") != "never":
            raise RuntimeError("UV_PYTHON_DOWNLOADS must be never during validation")
        if python_version != expected_python_version:
            raise RuntimeError(
                "Python version mismatch: "
                f"expected {expected_python_version}, got {python_version}"
            )
        _require_path_within(
            label="base Python executable",
            path=python_base_executable,
            root=python_install_dir,
        )
        if not python_base_executable.is_file():
            raise RuntimeError(
                f"base Python executable is missing: {python_base_executable}"
            )
        python_base_executable_sha256 = _sha256(python_base_executable)
    if (expected_uv_version is None) != (expected_uv_executable is None):
        raise RuntimeError(
            "expected uv version and executable must be provided together"
        )
    uv_version: str | None = None
    uv_executable: Path | None = None
    uv_executable_sha256: str | None = None
    if expected_uv_version is not None and expected_uv_executable is not None:
        configured_uv_executable = environment.get("UV_EXECUTABLE")
        if configured_uv_executable is None:
            raise RuntimeError("UV_EXECUTABLE is not set")
        if environment.get("PINNED_UV_VERSION") != expected_uv_version:
            raise RuntimeError(
                "PINNED_UV_VERSION does not match the expected uv version"
            )
        if expected_uv_executable.is_symlink():
            raise RuntimeError(
                f"uv executable must not be a symlink: {expected_uv_executable}"
            )
        uv_executable = expected_uv_executable.resolve(strict=False)
        if (
            _absolute_path(configured_uv_executable).resolve(strict=False)
            != uv_executable
        ):
            raise RuntimeError(
                "UV_EXECUTABLE does not match the expected executable: "
                f"{configured_uv_executable} != {uv_executable}"
            )
        if not uv_executable.is_file() or not os.access(uv_executable, os.X_OK):
            raise RuntimeError(f"uv executable is missing: {uv_executable}")
        uv_process = subprocess.run(
            [str(uv_executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        version_fields = uv_process.stdout.strip().split()
        if (
            uv_process.returncode != 0
            or len(version_fields) < 2
            or version_fields[0] != "uv"
        ):
            raise RuntimeError(
                f"failed to read uv version from {uv_executable}: "
                f"{uv_process.stderr.strip()}"
            )
        uv_version = version_fields[1]
        if uv_version != expected_uv_version:
            raise RuntimeError(
                f"uv version mismatch: expected {expected_uv_version}, got {uv_version}"
            )
        uv_executable_sha256 = _sha256(uv_executable)
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
    hybridep_buffer_available: bool | None = None
    required_module_distributions = dict(REQUIRED_MODULE_DISTRIBUTIONS)
    if expected_runtime_feature_set in NARROW_EVAL_FEATURE_SETS:
        required_module_distributions = {
            name: distributions
            for name, distributions in required_module_distributions.items()
            if name not in TE_EVAL_OPTIONAL_MODULES
        }
    elif expected_runtime_feature_set in HYBRIDEP_FEATURE_SETS:
        required_module_distributions["deep_ep"] = ("deep-ep",)
    for module_name in required_module_distributions:
        if module_name != "torch":
            modules[module_name] = importer(module_name)

    if (
        expected_runtime_feature_set in HYBRIDEP_FEATURE_SETS
        and getattr(modules["deep_ep"], HYBRIDEP_BUFFER_SYMBOL, None) is None
    ):
        raise RuntimeError(f"DeepEP runtime is missing {HYBRIDEP_BUFFER_SYMBOL}")
    if expected_runtime_feature_set in HYBRIDEP_FEATURE_SETS:
        hybridep_buffer_available = True

    te_extension = modules[TE_GROUPED_LINEAR_MODULE]
    transformer_engine_grouped_linear_symbols = [
        symbol
        for symbol in REQUIRED_TE_GROUPED_LINEAR_SYMBOLS
        if getattr(te_extension, symbol, None) is not None
    ]
    unavailable_grouped_linear_symbols = sorted(
        set(REQUIRED_TE_GROUPED_LINEAR_SYMBOLS).difference(
            transformer_engine_grouped_linear_symbols
        )
    )
    if unavailable_grouped_linear_symbols:
        raise RuntimeError(
            "TE grouped-linear backend is unavailable: "
            + ", ".join(unavailable_grouped_linear_symbols)
        )

    packages: dict[str, dict[str, str]] = {}
    for module_name, distributions in required_module_distributions.items():
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
        "all_eval_callables_supported": "not_tested",
        "mcore_eval_reuse_graph_io": "not_implemented",
        "raw_te_eval_reuse_graph_io": "not_tested",
        "candidate_sha": None,
        "integration_sha": None,
        "test_row_id": "runtime_preflight",
        "runtime_feature_set": expected_runtime_feature_set,
        "hybridep_buffer_available": hybridep_buffer_available,
        "excluded_packages": (
            list(expected_excluded_packages)
            if expected_excluded_packages is not None
            else None
        ),
        "torch_cuda_arch_list": expected_torch_cuda_arch_list,
        "nvte_cuda_archs": expected_nvte_cuda_archs,
        "topology": {
            "num_nodes": 1,
            "gpus_per_node": device_count,
            "world_size": device_count,
        },
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
        "python_base_executable": str(python_base_executable),
        "python_base_executable_sha256": python_base_executable_sha256,
        "python_version": python_version,
        "uv_python_install_dir": (
            str(python_install_dir) if python_install_dir is not None else None
        ),
        "uv_version": uv_version,
        "uv_executable": str(uv_executable) if uv_executable is not None else None,
        "uv_executable_sha256": uv_executable_sha256,
        "nvte_with_nccl_ep": nvte_with_nccl_ep,
        "transformer_engine_nccl_ep_available": (transformer_engine_nccl_ep_available),
        "transformer_engine_nccl_ep_symbols": transformer_engine_nccl_ep_symbols,
        "transformer_engine_grouped_linear_symbols": (
            transformer_engine_grouped_linear_symbols
        ),
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
    parser.add_argument("--expected-python-version", required=True)
    parser.add_argument("--expected-python-install-dir", required=True, type=Path)
    parser.add_argument("--expected-uv-version", required=True)
    parser.add_argument("--expected-uv-executable", required=True, type=Path)
    parser.add_argument("--expected-nvte-with-nccl-ep", required=True)
    parser.add_argument("--runtime-feature-set", required=True)
    parser.add_argument("--excluded-packages", required=True)
    parser.add_argument("--torch-cuda-arch-list", required=True)
    parser.add_argument("--nvte-cuda-archs", required=True)
    parser.add_argument("--nemo-rl-commit", required=True)
    parser.add_argument("--bridge-commit", required=True)
    parser.add_argument("--mcore-commit", required=True)
    parser.add_argument("--uv-lock-sha256", required=True)
    parser.add_argument("--expected-te-commit", required=True)
    parser.add_argument("--expected-te-version-base-commit", required=True)
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
        "expected_te_version_base_commit": args.expected_te_version_base_commit,
        "expected_python_version": args.expected_python_version,
        "expected_uv_version": args.expected_uv_version,
        "expected_nvte_with_nccl_ep": args.expected_nvte_with_nccl_ep,
        "runtime_feature_set": args.runtime_feature_set,
        "excluded_packages": args.excluded_packages.split(","),
        "torch_cuda_arch_list": args.torch_cuda_arch_list,
        "nvte_cuda_archs": args.nvte_cuda_archs,
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
            expected_python_version=args.expected_python_version,
            expected_python_install_dir=args.expected_python_install_dir,
            expected_uv_version=args.expected_uv_version,
            expected_uv_executable=args.expected_uv_executable,
            expected_nvte_with_nccl_ep=args.expected_nvte_with_nccl_ep,
            expected_runtime_feature_set=args.runtime_feature_set,
            expected_excluded_packages=tuple(args.excluded_packages.split(",")),
            expected_torch_cuda_arch_list=args.torch_cuda_arch_list,
            expected_nvte_cuda_archs=args.nvte_cuda_archs,
        )
        te_version = runtime["packages"]["transformer_engine.pytorch"]["version"]
        if _version_pair(te_version) < (2, 16):
            raise RuntimeError(
                f"runtime requires Transformer Engine >= 2.16, got {te_version}"
            )
        te_commit = _distribution_vcs_commit("transformer-engine")
        te_identities = validate_transformer_engine_identities(
            version=te_version,
            source_commit=te_commit,
            expected_source_commit=args.expected_te_commit,
            expected_version_base_commit=args.expected_te_version_base_commit,
        )
        payload = {
            "status": "passed",
            **context,
            "transformer_engine_vcs_commit": te_commit,
            **te_identities,
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
