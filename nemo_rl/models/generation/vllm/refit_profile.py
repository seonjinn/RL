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

import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

import torch


class RefitPhaseProfiler:
    """Collect opt-in wall-clock, CUDA-event, and counter refit metrics."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._wall_seconds: dict[str, float] = {}
        self._cuda_events: dict[str, list[tuple[Any, Any]]] = {}
        self._counters: dict[str, int] = {}

    @contextmanager
    def wall_phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        torch.accelerator.synchronize()
        started_at = time.perf_counter()
        try:
            yield
        finally:
            torch.accelerator.synchronize()
            elapsed = time.perf_counter() - started_at
            self._wall_seconds[name] = self._wall_seconds.get(name, 0.0) + elapsed

    @contextmanager
    def cuda_phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._cuda_events.setdefault(name, []).append((start, end))

    def increment(self, name: str, value: int = 1) -> None:
        if self.enabled:
            self._counters[name] = self._counters.get(name, 0) + value

    def finish(self) -> dict[str, float | int]:
        if not self.enabled:
            return {}

        torch.accelerator.synchronize()
        metrics: dict[str, float | int] = {
            f"{name}_s": seconds for name, seconds in self._wall_seconds.items()
        }
        for name, event_pairs in self._cuda_events.items():
            elapsed_seconds = (
                sum(start.elapsed_time(end) for start, end in event_pairs) / 1000.0
            )
            metrics[f"{name}_gpu_s"] = elapsed_seconds
        metrics.update(self._counters)
        return metrics


@contextmanager
def profile_vllm_layerwise_kernels(
    profiler: RefitPhaseProfiler,
) -> Iterator[None]:
    """Measure the expensive vLLM layerwise transformations without changing them."""
    if not profiler.enabled:
        yield
        return

    from vllm.model_executor.layers.fused_moe.oracle import unquantized
    from vllm.model_executor.model_loader.reload import layerwise

    original_layout_conversion = (
        unquantized.convert_moe_weights_to_flashinfer_trtllm_block_layout
    )
    original_kernel_copy = layerwise._copy_and_restore_kernel_tensors

    @wraps(original_layout_conversion)
    def profiled_layout_conversion(*args: Any, **kwargs: Any) -> Any:
        with profiler.cuda_phase("moe_layout_conversion"):
            return original_layout_conversion(*args, **kwargs)

    @wraps(original_kernel_copy)
    def profiled_kernel_copy(*args: Any, **kwargs: Any) -> Any:
        with profiler.cuda_phase("kernel_storage_copy"):
            return original_kernel_copy(*args, **kwargs)

    unquantized.convert_moe_weights_to_flashinfer_trtllm_block_layout = (
        profiled_layout_conversion
    )
    layerwise._copy_and_restore_kernel_tensors = profiled_kernel_copy
    try:
        yield
    finally:
        unquantized.convert_moe_weights_to_flashinfer_trtllm_block_layout = (
            original_layout_conversion
        )
        layerwise._copy_and_restore_kernel_tensors = original_kernel_copy
