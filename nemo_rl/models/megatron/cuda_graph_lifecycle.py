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

from typing import Protocol


class TECudaGraphHelperProtocol(Protocol):
    def create_cudagraphs(self) -> None: ...

    def graphs_created(self) -> bool: ...

    def cuda_graph_set_manual_hooks(self) -> None: ...

    def delete_cuda_graphs(self) -> None: ...


class TECudaGraphLifecycle:
    """Own the warmup, capture, and cleanup state for one TE graph helper."""

    def __init__(
        self,
        helper: TECudaGraphHelperProtocol,
        warmup_steps: int,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("CUDA Graph warmup steps must be non-negative")
        self.helper = helper
        self.warmup_steps = warmup_steps
        self.successful_steps = 0
        self._capture_attempted = False
        self._closed = False

    def record_successful_step(self) -> None:
        self.successful_steps += 1

    def ready_to_capture(self) -> bool:
        return (
            not self._capture_attempted and self.successful_steps >= self.warmup_steps
        )

    def capture_if_ready(self) -> bool:
        if not self.ready_to_capture():
            return False

        self._capture_attempted = True
        self.helper.create_cudagraphs()
        if not self.helper.graphs_created():
            raise RuntimeError(
                "Transformer Engine CUDA Graph capture found no graphable layers "
                "for the requested scope"
            )
        return True

    def graphs_created(self) -> bool:
        return self.helper.graphs_created()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.helper.graphs_created():
            self.helper.delete_cuda_graphs()
