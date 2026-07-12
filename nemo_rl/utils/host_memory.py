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
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, Optional

import psutil

_GIB = 1024**3
_PROC_SELF_CGROUP_PATH = Path("/proc/self/cgroup")
_CGROUP_ROOT = Path("/sys/fs/cgroup")


class HostMemorySnapshot(NamedTuple):
    process_rss_gib: Optional[float]
    system_available_gib: Optional[float]
    cgroup_memory_current_gib: Optional[float]
    cgroup_memory_max_gib: Optional[float]
    cgroup_memory_peak_gib: Optional[float]


def emit_structured_stdout(message: str) -> None:
    """Emit one fail-visible diagnostic line without changing runtime semantics."""
    try:
        print(message, flush=True)
    except Exception:
        pass


def _read_cgroup_gib(path: Path, *, allow_max: bool = False) -> Optional[float]:
    try:
        raw = path.read_text().strip()
        if allow_max and raw == "max":
            return None
        value = int(raw)
        return value / _GIB if value >= 0 else None
    except (OSError, ValueError):
        return None


def _get_cgroup_memory_values() -> tuple[
    Optional[float], Optional[float], Optional[float]
]:
    try:
        cgroup_lines = _PROC_SELF_CGROUP_PATH.read_text().splitlines()
    except OSError:
        return None, None, None
    relative_path = None
    for line in cgroup_lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            relative_path = parts[2]
            break
    if relative_path is None:
        return None, None, None

    relative_dir = _CGROUP_ROOT / relative_path.lstrip("/")
    metric_names = ("memory.current", "memory.max", "memory.peak")
    try:
        has_relative_metrics = any(
            (relative_dir / metric_name).is_file() for metric_name in metric_names
        )
    except OSError:
        has_relative_metrics = False
    cgroup_dir = relative_dir if has_relative_metrics else _CGROUP_ROOT

    return (
        _read_cgroup_gib(cgroup_dir / "memory.current"),
        _read_cgroup_gib(cgroup_dir / "memory.max", allow_max=True),
        _read_cgroup_gib(cgroup_dir / "memory.peak"),
    )


def _get_host_memory_snapshot() -> HostMemorySnapshot:
    try:
        process_rss_gib = psutil.Process(os.getpid()).memory_info().rss / _GIB
    except Exception:
        process_rss_gib = None

    try:
        system_available_gib = psutil.virtual_memory().available / _GIB
    except Exception:
        system_available_gib = None

    try:
        cgroup_current_gib, cgroup_max_gib, cgroup_peak_gib = (
            _get_cgroup_memory_values()
        )
    except Exception:
        cgroup_current_gib, cgroup_max_gib, cgroup_peak_gib = None, None, None

    return HostMemorySnapshot(
        process_rss_gib=process_rss_gib,
        system_available_gib=system_available_gib,
        cgroup_memory_current_gib=cgroup_current_gib,
        cgroup_memory_max_gib=cgroup_max_gib,
        cgroup_memory_peak_gib=cgroup_peak_gib,
    )


def _format_gib(value: Optional[float]) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _format_delta(current: Optional[float], before: Optional[float]) -> str:
    if current is None or before is None:
        return "unavailable"
    return f"{current - before:.3f}"


def emit_host_memory_event(
    *,
    event: str,
    phase: str,
    fields: Optional[Mapping[str, object]] = None,
    before_snapshot: Optional[HostMemorySnapshot] = None,
    include_deltas: bool = False,
) -> Optional[HostMemorySnapshot]:
    """Capture host memory and emit a single structured, best-effort event."""
    try:
        prefix = f"event={event} phase={phase}"
        if fields:
            prefix += "".join(f" {key}={value}" for key, value in fields.items())

        snapshot = _get_host_memory_snapshot()
        message = f"{prefix} process_rss_gib={_format_gib(snapshot.process_rss_gib)}"
        if include_deltas:
            before_rss = None if before_snapshot is None else before_snapshot.process_rss_gib
            message += (
                " process_rss_delta_gib="
                f"{_format_delta(snapshot.process_rss_gib, before_rss)}"
            )
        message += (
            " system_available_gib="
            f"{_format_gib(snapshot.system_available_gib)}"
        )
        if include_deltas:
            before_available = (
                None if before_snapshot is None else before_snapshot.system_available_gib
            )
            message += (
                " system_available_delta_gib="
                f"{_format_delta(snapshot.system_available_gib, before_available)}"
            )
        message += (
            " cgroup_memory_current_gib="
            f"{_format_gib(snapshot.cgroup_memory_current_gib)}"
            " cgroup_memory_max_gib="
            f"{_format_gib(snapshot.cgroup_memory_max_gib)}"
            " cgroup_memory_peak_gib="
            f"{_format_gib(snapshot.cgroup_memory_peak_gib)}"
        )
        emit_structured_stdout(message)
        return snapshot
    except Exception:
        return None
