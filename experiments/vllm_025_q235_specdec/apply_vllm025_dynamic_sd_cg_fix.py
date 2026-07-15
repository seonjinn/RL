#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import site
from pathlib import Path


def _replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected vLLM 0.25 source for {label}")
    return text.replace(old, new), True


def _find_site_packages() -> Path:
    for candidate in map(Path, site.getsitepackages()):
        if (candidate / "vllm").is_dir():
            return candidate
    raise RuntimeError("Could not find the installed vLLM package")


def apply_patch(site_packages: Path) -> bool:
    cudagraph_utils = site_packages / "vllm/v1/worker/gpu/cudagraph_utils.py"
    speculator = (
        site_packages / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    )
    if not cudagraph_utils.is_file() or not speculator.is_file():
        raise FileNotFoundError(f"vLLM 0.25 sources not found under {site_packages}")

    cudagraph_text = cudagraph_utils.read_text()
    cudagraph_text, signature_changed = _replace_once(
        cudagraph_text,
        "class CudaGraphManager:\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: VllmConfig,\n"
        "        device: torch.device,\n"
        "        cudagraph_mode: CUDAGraphMode,\n"
        "        decode_query_len: int,\n"
        "        lora_capture_cases: list[int] | None = None,\n"
        "    ):\n",
        "class CudaGraphManager:\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: VllmConfig,\n"
        "        device: torch.device,\n"
        "        cudagraph_mode: CUDAGraphMode,\n"
        "        decode_query_len: int,\n"
        "        lora_capture_cases: list[int] | None = None,\n"
        "        use_dynamic_decode_shapes: bool = True,\n"
        "    ):\n",
        "CudaGraphManager signature",
    )
    cudagraph_text, attribute_changed = _replace_once(
        cudagraph_text,
        "        self.decode_query_len = decode_query_len\n\n"
        "        self.dp_size = vllm_config.parallel_config.data_parallel_size\n",
        "        self.decode_query_len = decode_query_len\n"
        "        self.use_dynamic_decode_shapes = use_dynamic_decode_shapes\n\n"
        "        self.dp_size = vllm_config.parallel_config.data_parallel_size\n",
        "CudaGraphManager attribute",
    )
    cudagraph_text, condition_changed = _replace_once(
        cudagraph_text,
        "            and speculative_config.uses_dynamic_speculative_decoding()\n"
        "        ):\n",
        "            and speculative_config.uses_dynamic_speculative_decoding()\n"
        "            and self.use_dynamic_decode_shapes\n"
        "        ):\n",
        "DynamicSD graph condition",
    )

    speculator_text = speculator.read_text()
    speculator_text, speculator_changed = _replace_once(
        speculator_text,
        "            cudagraph_mode,\n            decode_query_len=1,\n        )\n",
        "            cudagraph_mode,\n"
        "            decode_query_len=1,\n"
        "            use_dynamic_decode_shapes=False,\n"
        "        )\n",
        "autoregressive draft decode manager",
    )

    changed = any(
        (signature_changed, attribute_changed, condition_changed, speculator_changed)
    )
    if changed:
        cudagraph_utils.write_text(cudagraph_text)
        speculator.write_text(speculator_text)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", type=Path)
    args = parser.parse_args()
    site_packages = args.site_packages or _find_site_packages()
    changed = apply_patch(site_packages)
    state = "applied" if changed else "already applied"
    print(f"vLLM 0.25 DynamicSD CUDA graph fix: {state}")


if __name__ == "__main__":
    main()
