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


def _parse_uv_version(output: str) -> str:
    fields = output.split()
    if len(fields) < 2 or fields[0] != "uv":
        raise RuntimeError(f"Unexpected uv version output: {output}")
    return fields[1]


def _validate_cuda_graph_checksums(
    captured: float,
    first_replay: float,
    second_replay: float,
    expected_captured: float,
    expected_first_replay: float,
    expected_second_replay: float,
) -> None:
    if captured != expected_captured:
        raise RuntimeError(
            "CUDA graph capture checksum mismatch: "
            f"expected {expected_captured}, got {captured}"
        )
    if first_replay != expected_first_replay:
        raise RuntimeError(
            "CUDA graph first replay checksum mismatch: "
            f"expected {expected_first_replay}, got {first_replay}"
        )
    if second_replay != expected_second_replay:
        raise RuntimeError(
            "CUDA graph second replay checksum mismatch: "
            f"expected {expected_second_replay}, got {second_replay}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-device-count", type=int, required=True)
    parser.add_argument("--expected-python-version", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-uv-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _capture_cuda_graph(torch: Any, device_index: int) -> tuple[float, float, float]:
    with torch.cuda.device(device_index):
        tensor = torch.ones((64, 64), device="cuda", dtype=torch.float32)
        current_stream = torch.cuda.current_stream(device_index)
        warmup_stream = torch.cuda.Stream(device=device_index)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                tensor.mul_(2.0)
                tensor.fill_(1.0)
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(device_index)

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=device_index)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.graph(graph, stream=capture_stream):
            tensor.mul_(2.0)
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(device_index)
        captured_checksum = float(tensor.sum().item())

        graph.replay()
        torch.cuda.synchronize(device_index)
        first_replay_checksum = float(tensor.sum().item())

        graph.replay()
        torch.cuda.synchronize(device_index)
        second_replay_checksum = float(tensor.sum().item())
        elements = float(tensor.numel())
        _validate_cuda_graph_checksums(
            captured_checksum,
            first_replay_checksum,
            second_replay_checksum,
            expected_captured=elements,
            expected_first_replay=elements * 2.0,
            expected_second_replay=elements * 4.0,
        )
        return captured_checksum, first_replay_checksum, second_replay_checksum


def main() -> None:
    args = _parse_args()
    uv_executable = "/root/.local/bin/uv"
    uv_version_output = subprocess.run(
        [uv_executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    uv_version = _parse_uv_version(uv_version_output)
    if uv_version != args.expected_uv_version:
        raise RuntimeError(
            f"uv version mismatch: expected {args.expected_uv_version}, got {uv_version}"
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
        (
            captured_checksum,
            first_replay_checksum,
            second_replay_checksum,
        ) = _capture_cuda_graph(torch, device_index)
        with cupy.cuda.Device(device_index):
            cupy_array = cupy.arange(32, dtype=cupy.float32)
            cupy_checksum = float(cupy_array.sum().get())
        devices.append(
            {
                "index": device_index,
                "name": torch.cuda.get_device_name(device_index),
                "capability": list(torch.cuda.get_device_capability(device_index)),
                "torch_cuda_graph_capture_checksum": captured_checksum,
                "torch_cuda_graph_first_replay_checksum": first_replay_checksum,
                "torch_cuda_graph_checksum": second_replay_checksum,
                "cupy_checksum": cupy_checksum,
            }
        )

    result = {
        "schema": "nemo-rl-nightly-gpu-runtime-smoke-v2",
        "status": "passed",
        "bootstrap_python": sys.version,
        "managed_python": {
            "executable": managed_python,
            "version": managed_python_version,
        },
        "uv": uv_version_output,
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
