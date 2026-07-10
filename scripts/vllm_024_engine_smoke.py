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

"""Run one real CUDA Graph-enabled generation with vLLM 0.24."""

from __future__ import annotations

import json
import os
import time

from vllm import LLM, SamplingParams


def main() -> None:
    model = os.environ.get("ENGINE_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
    started_at = time.perf_counter()
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.5,
        enforce_eager=False,
    )
    initialized_at = time.perf_counter()
    outputs = llm.generate(
        ["Return only the number that is the sum of 20 and 22."],
        SamplingParams(temperature=0.0, max_tokens=16),
    )
    finished_at = time.perf_counter()
    output = outputs[0].outputs[0]
    assert output.token_ids, "vLLM returned no generated tokens"

    print(
        json.dumps(
            {
                "cuda_graph_enabled": True,
                "generated_text": output.text,
                "generated_tokens": len(output.token_ids),
                "generation_seconds": finished_at - initialized_at,
                "initialization_seconds": initialized_at - started_at,
                "model": model,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
