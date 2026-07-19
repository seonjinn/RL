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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, TypeVar


T = TypeVar("T")
G_LOG_PREFIX = "[NRL_REFIT_OFFLOAD]"


@dataclass(frozen=True, slots=True)
class HostMemorySnapshot:
    """Process and node memory counters sampled around one refit phase."""

    rss_bytes: int | None
    major_faults: int | None
    mem_available_bytes: int | None
    minor_faults: int | None = None


@dataclass(frozen=True, slots=True)
class PinnedSlabEntry:
    """Location of one contiguous optimizer tensor inside a pinned slab."""

    slab_index: int
    offset_bytes: int
    num_bytes: int


@dataclass(frozen=True, slots=True)
class PinnedSlabPlan:
    """Bounded pinned allocations and entry placements for optimizer offload."""

    slab_sizes: tuple[int, ...]
    entries: tuple[PinnedSlabEntry, ...]


def plan_pinned_slabs(
    entry_sizes: Sequence[int],
    *,
    slab_bytes: int,
    alignment: int,
) -> PinnedSlabPlan:
    """Pack entries into fixed-size slabs without splitting an entry."""
    if slab_bytes <= 0:
        raise ValueError("slab_bytes must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")

    slab_sizes: list[int] = []
    entries: list[PinnedSlabEntry] = []
    active_slab: int | None = None
    active_offset = 0

    for num_bytes in entry_sizes:
        if num_bytes <= 0:
            raise ValueError("entry sizes must be positive")
        if num_bytes > slab_bytes:
            raise ValueError(
                f"optimizer entry size {num_bytes} exceeds pinned slab size "
                f"{slab_bytes}"
            )

        offset = (active_offset + alignment - 1) // alignment * alignment
        if active_slab is None or offset + num_bytes > slab_bytes:
            active_slab = len(slab_sizes)
            slab_sizes.append(slab_bytes)
            offset = 0
        entries.append(PinnedSlabEntry(active_slab, offset, num_bytes))
        active_offset = offset + num_bytes

    return PinnedSlabPlan(tuple(slab_sizes), tuple(entries))


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
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return HostMemorySnapshot(
        rss_bytes=_read_kib_value(Path("/proc/self/status"), "VmRSS"),
        major_faults=usage.ru_majflt,
        mem_available_bytes=_read_kib_value(Path("/proc/meminfo"), "MemAvailable"),
        minor_faults=usage.ru_minflt,
    )


def _delta(after: int | None, before: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _capture_snapshot_safely(
    capture_snapshot: Callable[[], HostMemorySnapshot],
) -> HostMemorySnapshot:
    try:
        return capture_snapshot()
    except Exception:
        return HostMemorySnapshot(
            rss_bytes=None,
            major_faults=None,
            mem_available_bytes=None,
            minor_faults=None,
        )


def _monotonic_safely(monotonic: Callable[[], float]) -> float | None:
    try:
        return monotonic()
    except Exception:
        return None


def _emit_record_safely(
    *,
    phase: str,
    rank: int,
    optimizer_cuda_bytes: int | None,
    before: HostMemorySnapshot,
    after: HostMemorySnapshot,
    elapsed_s: float | None,
    status: str,
    error_type: str | None,
    hostname: str | None,
    stream: TextIO | None,
) -> None:
    try:
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
            "major_faults_delta": _delta(after.major_faults, before.major_faults),
            "minor_faults_before": before.minor_faults,
            "minor_faults_after": after.minor_faults,
            "minor_faults_delta": _delta(after.minor_faults, before.minor_faults),
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
    except Exception:
        pass


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
    before = _capture_snapshot_safely(capture_snapshot)
    started = _monotonic_safely(monotonic)
    try:
        result = operation()
    except Exception as error:
        finished = _monotonic_safely(monotonic)
        after = _capture_snapshot_safely(capture_snapshot)
        _emit_record_safely(
            phase=phase,
            rank=rank,
            optimizer_cuda_bytes=optimizer_cuda_bytes,
            before=before,
            after=after,
            elapsed_s=(
                finished - started
                if finished is not None and started is not None
                else None
            ),
            status="error",
            error_type=type(error).__name__,
            hostname=hostname,
            stream=stream,
        )
        raise
    finished = _monotonic_safely(monotonic)
    after = _capture_snapshot_safely(capture_snapshot)
    _emit_record_safely(
        phase=phase,
        rank=rank,
        optimizer_cuda_bytes=optimizer_cuda_bytes,
        before=before,
        after=after,
        elapsed_s=(
            finished - started if finished is not None and started is not None else None
        ),
        status="ok",
        error_type=None,
        hostname=hostname,
        stream=stream,
    )
    return result
