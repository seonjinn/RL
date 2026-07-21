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

"""Fail-closed GPU and dependency smoke test for a staged NeMo-RL image."""

import importlib
import importlib.metadata
import json
import platform
from pathlib import Path

import cutlass
import torch
from cutlass import cute

import nemo_rl
import transformer_engine.pytorch as te


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no visible CUDA devices")

    modules = {
        "cutlass": cutlass,
        "cutlass.cute": cute,
        "nemo_rl": nemo_rl,
        "transformer_engine.pytorch": te,
        "megatron.core": importlib.import_module("megatron.core"),
        "megatron.bridge": importlib.import_module("megatron.bridge"),
        "modelopt.torch": importlib.import_module("modelopt.torch"),
    }
    actual_devices = torch.cuda.device_count()
    device_names = [
        torch.cuda.get_device_name(index) for index in range(actual_devices)
    ]
    if not all("GB200" in device_name for device_name in device_names):
        raise RuntimeError(f"expected only GB200 devices, found {device_names}")
    for device_index in range(actual_devices):
        device = torch.device(f"cuda:{device_index}")
        layer = te.Linear(64, 64).to(device)
        inputs = torch.randn(8, 64, device=device, requires_grad=True)
        outputs = layer(inputs)
        outputs.sum().backward()
        torch.cuda.synchronize(device)
        if not torch.isfinite(outputs).all():
            raise RuntimeError(
                f"Transformer Engine produced non-finite output on {device}"
            )
        if inputs.grad is None or not torch.isfinite(inputs.grad).all():
            raise RuntimeError(
                f"Transformer Engine produced an invalid input gradient on {device}"
            )
    evidence = {
        "architecture": platform.machine(),
        "cuda_available": True,
        "cuda_device_count": actual_devices,
        "cuda_device_name": device_names[0],
        "cuda_device_names": device_names,
        "cuda_version": torch.version.cuda,
        "transformer_engine_linear_backward": "pass",
        "module_paths": {
            name: str(Path(module.__file__).resolve())
            if getattr(module, "__file__", None)
            else None
            for name, module in modules.items()
        },
        "package_versions": {
            "megatron-bridge": _distribution_version("megatron-bridge"),
            "modelopt": _distribution_version("nvidia-modelopt"),
            "nemo-rl": _distribution_version("nemo-rl"),
            "torch": str(torch.__version__),
            "transformer-engine": _distribution_version("transformer-engine"),
        },
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
