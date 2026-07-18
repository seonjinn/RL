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

from __future__ import annotations

import json
import resource
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, TypeVar


T = TypeVar("T")
G_LOG_PREFIX = "[NRL_REFIT_OFFLOAD]"


@dataclass(frozen=True, slots=True)
class HostMemorySnapshot:
    """Process and node memory counters sampled around one refit phase."""

    rss_bytes: int | None
    major_faults: int
    mem_available_bytes: int | None


def _read_kib_value(path: Path, key: str) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    prefix = f"{key}:"
    for line in lines:
        if not line.startswith(prefix):
            continue
        fields = line[len(prefix) :].split()
        if not fields:
            return None
        try:
            return int(fields[0]) * 1024
        except ValueError:
            return None
    return None


def capture_host_memory_snapshot() -> HostMemorySnapshot:
    """Capture current process RSS, major faults, and node available memory."""
    return HostMemorySnapshot(
        rss_bytes=_read_kib_value(Path("/proc/self/status"), "VmRSS"),
        major_faults=resource.getrusage(resource.RUSAGE_SELF).ru_majflt,
        mem_available_bytes=_read_kib_value(Path("/proc/meminfo"), "MemAvailable"),
    )


def _delta(after: int | None, before: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def measure_refit_phase(
    operation: Callable[[], T],
    *,
    phase: str,
    rank: int,
    optimizer_cuda_bytes: int | None,
    capture_snapshot: Callable[[], HostMemorySnapshot] = capture_host_memory_snapshot,
    monotonic: Callable[[], float] = time.perf_counter,
    hostname: str | None = None,
    stream: TextIO | None = None,
) -> T:
    """Run one refit phase and emit a single structured rank-local record."""
    before = capture_snapshot()
    started = monotonic()
    status = "ok"
    error_type: str | None = None
    try:
        result = operation()
    except Exception as error:
        status = "error"
        error_type = type(error).__name__
        raise
    finally:
        elapsed_s = monotonic() - started
        after = capture_snapshot()
        payload: dict[str, object] = {
            "schema_version": 1,
            "event": "refit_offload_phase",
            "phase": phase,
            "rank": rank,
            "hostname": socket.gethostname() if hostname is None else hostname,
            "status": status,
            "elapsed_s": elapsed_s,
            "optimizer_cuda_bytes": optimizer_cuda_bytes,
            "rss_bytes_before": before.rss_bytes,
            "rss_bytes_after": after.rss_bytes,
            "rss_bytes_delta": _delta(after.rss_bytes, before.rss_bytes),
            "major_faults_before": before.major_faults,
            "major_faults_after": after.major_faults,
            "major_faults_delta": after.major_faults - before.major_faults,
            "mem_available_bytes_before": before.mem_available_bytes,
            "mem_available_bytes_after": after.mem_available_bytes,
            "mem_available_bytes_delta": _delta(
                after.mem_available_bytes, before.mem_available_bytes
            ),
        }
        if error_type is not None:
            payload["error_type"] = error_type
        destination = sys.stdout if stream is None else stream
        print(
            f"{G_LOG_PREFIX} {json.dumps(payload, sort_keys=True, separators=(',', ':'))}",
            file=destination,
            flush=True,
        )
    return result
