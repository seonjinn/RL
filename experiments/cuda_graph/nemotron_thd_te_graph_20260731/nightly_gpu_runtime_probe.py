#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-device-count", type=int, required=True)
    parser.add_argument("--expected-python-version", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-uv-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _capture_cuda_graph(torch: Any, device_index: int) -> float:
    with torch.cuda.device(device_index):
        tensor = torch.ones((64, 64), device="cuda", dtype=torch.float32)
        for _ in range(3):
            tensor.add_(1.0)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tensor.mul_(2.0)
        graph.replay()
        torch.cuda.synchronize()
        return float(tensor.sum().item())


def main() -> None:
    args = _parse_args()
    uv_executable = "/opt/nemo_rl_venv/bin/uv"
    uv_version = subprocess.run(
        [uv_executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if uv_version != f"uv {args.expected_uv_version}":
        raise RuntimeError(
            f"uv version mismatch: expected uv {args.expected_uv_version}, got {uv_version}"
        )

    managed_python = subprocess.run(
        [
            uv_executable,
            "python",
            "find",
            args.expected_python_version,
            "--no-python-downloads",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    managed_python_version = subprocess.run(
        [
            managed_python,
            "-c",
            "import platform; print(platform.python_version())",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if managed_python_version != args.expected_python_version:
        raise RuntimeError(
            f"Managed Python version mismatch: expected {args.expected_python_version}, "
            f"got {managed_python_version}"
        )

    image_source_commit = os.environ.get("NEMO_RL_COMMIT", "")
    if image_source_commit != args.expected_source_commit:
        raise RuntimeError(
            f"Image source commit mismatch: expected {args.expected_source_commit}, "
            f"got {image_source_commit or 'unset'}"
        )

    imported_modules = {
        module_name: importlib.import_module(module_name)
        for module_name in (
            "causal_conv1d",
            "cupy",
            "mamba_ssm",
            "megatron.core",
            "torch",
            "transformer_engine.pytorch",
        )
    }
    torch = imported_modules["torch"]
    cupy = imported_modules["cupy"]

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    device_count = torch.cuda.device_count()
    if device_count != args.expected_device_count:
        raise RuntimeError(
            f"CUDA device count mismatch: expected {args.expected_device_count}, got {device_count}"
        )

    devices: list[dict[str, Any]] = []
    for device_index in range(device_count):
        graph_checksum = _capture_cuda_graph(torch, device_index)
        with cupy.cuda.Device(device_index):
            cupy_array = cupy.arange(32, dtype=cupy.float32)
            cupy_checksum = float(cupy_array.sum().get())
        devices.append(
            {
                "index": device_index,
                "name": torch.cuda.get_device_name(device_index),
                "capability": list(torch.cuda.get_device_capability(device_index)),
                "torch_cuda_graph_checksum": graph_checksum,
                "cupy_checksum": cupy_checksum,
            }
        )

    result = {
        "schema": "nemo-rl-nightly-gpu-runtime-smoke-v1",
        "status": "passed",
        "bootstrap_python": sys.version,
        "managed_python": {
            "executable": managed_python,
            "version": managed_python_version,
        },
        "uv": uv_version,
        "nemo_rl_commit": image_source_commit,
        "versions": {
            "causal_conv1d": _package_version("causal-conv1d"),
            "cupy": _package_version("cupy-cuda12x"),
            "mamba_ssm": _package_version("mamba-ssm"),
            "megatron_core": _package_version("megatron-core"),
            "torch": torch.__version__,
            "transformer_engine": _package_version("transformer-engine"),
        },
        "devices": devices,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
