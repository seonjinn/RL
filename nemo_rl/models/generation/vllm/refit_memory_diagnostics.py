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

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RefitMemorySnapshot:
    """Bounded host and device memory values for one refit boundary."""

    phase: str
    pid: int
    rss_bytes: int
    mem_available_bytes: int
    cuda_allocated_bytes: int
    cuda_reserved_bytes: int


def _read_process_rss_bytes() -> int:
    with open("/proc/self/statm", encoding="utf-8") as statm:
        fields = statm.readline().split()
    if len(fields) < 2:
        raise RuntimeError("/proc/self/statm does not contain an RSS field")
    return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))


def _read_available_memory_bytes() -> int:
    with open("/proc/meminfo", encoding="utf-8") as meminfo:
        for record in meminfo:
            if record.startswith("MemAvailable:"):
                fields = record.split()
                if len(fields) != 3 or fields[2] != "kB":
                    raise RuntimeError("MemAvailable has an unexpected format")
                return int(fields[1]) * 1024
    raise RuntimeError("/proc/meminfo does not contain MemAvailable")


def capture_refit_memory_snapshot(phase: str) -> RefitMemorySnapshot:
    """Capture current-process RSS, host availability, and CUDA allocator bytes."""
    if torch.cuda.is_available():
        cuda_allocated_bytes = int(torch.cuda.memory_allocated())
        cuda_reserved_bytes = int(torch.cuda.memory_reserved())
    else:
        cuda_allocated_bytes = 0
        cuda_reserved_bytes = 0

    return RefitMemorySnapshot(
        phase=phase,
        pid=os.getpid(),
        rss_bytes=_read_process_rss_bytes(),
        mem_available_bytes=_read_available_memory_bytes(),
        cuda_allocated_bytes=cuda_allocated_bytes,
        cuda_reserved_bytes=cuda_reserved_bytes,
    )
