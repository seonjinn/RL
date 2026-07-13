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

import faulthandler
import json
import os
import re
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

_DEBUG_ENV = "NRL_DEBUG_REFERENCE_MODEL_SETUP"
_DISTRIBUTED_TIMEOUT_ENV = "NRL_MEGATRON_NCCL_TIMEOUT_SECONDS"
_MARKER_DIR_ENV = "NRL_REFERENCE_SETUP_MARKER_DIR"
_STACK_DUMP_INTERVAL_ENV = "NRL_REFERENCE_SETUP_STACK_DUMP_SECONDS"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_CHECKPOINT_MARKERS = (
    "run_config.yaml",
    "train_state.pt",
    "metadata.json",
    ".metadata",
    "latest_train_state.pt",
    "latest_checkpointed_iteration.txt",
)


def _diagnostics_enabled() -> bool:
    return os.getenv(_DEBUG_ENV, "").strip().lower() in _TRUE_VALUES


def buffer_memory_metadata(buffer: Any) -> dict[str, int]:
    """Return tensor sizes without touching buffer storage or devices."""
    metadata: dict[str, int] = {}
    for attribute, field_prefix in (
        ("param_data", "param"),
        ("grad_data", "grad"),
        ("param_data_cpu", "param_cpu"),
    ):
        tensor = getattr(buffer, attribute, None)
        if tensor is None:
            continue
        numel = int(tensor.numel())
        metadata[f"{field_prefix}_numel"] = numel
        metadata[f"{field_prefix}_bytes"] = numel * int(tensor.element_size())
    return metadata


def _numa_memory_metadata(
    node_root: str | os.PathLike[str] = "/sys/devices/system/node",
) -> dict[str, int]:
    """Return per-NUMA-node memory totals from Linux sysfs."""
    metadata: dict[str, int] = {}
    try:
        entries = os.listdir(node_root)
    except OSError:
        return metadata

    field_names = {
        "MemFree": "free",
        "MemTotal": "total",
        "MemUsed": "used",
    }
    for entry in sorted(entries):
        match = re.fullmatch(r"node(\d+)", entry)
        if match is None:
            continue
        node_id = match.group(1)
        meminfo_path = os.path.join(os.fspath(node_root), entry, "meminfo")
        try:
            with open(meminfo_path) as meminfo_file:
                for line in meminfo_file:
                    key, separator, value = line.partition(":")
                    field_name = key.split()[-1] if key.split() else ""
                    if not separator or field_name not in field_names:
                        continue
                    metadata[
                        f"numa_{node_id}_mem_{field_names[field_name]}_kb"
                    ] = int(value.split()[0])
        except (OSError, ValueError):
            continue
    return metadata


def _cgroup_memory_metadata(
    proc_self_cgroup: str | os.PathLike[str] = "/proc/self/cgroup",
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
) -> dict[str, int]:
    """Return cgroup-v2 memory usage, limits, events, and NUMA accounting."""
    metadata: dict[str, int] = {}
    try:
        with open(proc_self_cgroup) as cgroup_file:
            cgroup_path = next(
                (
                    line.rstrip("\n").split("::", 1)[1]
                    for line in cgroup_file
                    if line.startswith("0::")
                ),
                None,
            )
    except OSError:
        return metadata
    if cgroup_path is None:
        return metadata

    memory_dir = os.path.join(os.fspath(cgroup_root), cgroup_path.lstrip("/"))
    for filename, field_name in (
        ("memory.current", "cgroup_memory_current_bytes"),
        ("memory.peak", "cgroup_memory_peak_bytes"),
        ("memory.max", "cgroup_memory_max_bytes"),
    ):
        try:
            with open(os.path.join(memory_dir, filename)) as memory_file:
                metadata[field_name] = int(memory_file.read().strip())
        except (OSError, ValueError):
            continue

    try:
        with open(os.path.join(memory_dir, "memory.events")) as events_file:
            for line in events_file:
                field_name, value = line.split()
                if field_name in {"high", "max", "oom", "oom_kill"}:
                    metadata[f"cgroup_memory_event_{field_name}"] = int(value)
    except (OSError, ValueError):
        pass

    try:
        with open(os.path.join(memory_dir, "memory.numa_stat")) as numa_file:
            for line in numa_file:
                fields = line.split()
                if not fields or fields[0] not in {"anon", "file"}:
                    continue
                memory_kind = fields[0]
                for field in fields[1:]:
                    node, value = field.split("=", 1)
                    if re.fullmatch(r"N\d+", node) is None:
                        continue
                    metadata[
                        f"cgroup_memory_numa_{memory_kind}_{node.lower()}_bytes"
                    ] = int(value)
    except (OSError, ValueError):
        pass
    return metadata


def _linux_memory_metadata() -> dict[str, int]:
    metadata: dict[str, int] = {}
    proc_files = (
        (
            "/proc/self/status",
            {
                "VmHWM": "process_vmhwm_kb",
                "VmLck": "process_vmlck_kb",
                "VmPin": "process_vmpin_kb",
                "VmRSS": "process_vmrss_kb",
            },
        ),
        (
            "/proc/meminfo",
            {
                "MemAvailable": "node_mem_available_kb",
                "MemFree": "node_mem_free_kb",
            },
        ),
    )
    for path, field_names in proc_files:
        try:
            with open(path) as proc_file:
                for line in proc_file:
                    key, separator, value = line.partition(":")
                    if not separator or key not in field_names:
                        continue
                    metadata[field_names[key]] = int(value.split()[0])
        except (OSError, ValueError):
            continue
    metadata.update(_numa_memory_metadata())
    metadata.update(_cgroup_memory_metadata())
    return metadata


def _write_rank_marker(metadata: dict[str, Any]) -> None:
    marker_dir = os.getenv(_MARKER_DIR_ENV)
    if not marker_dir:
        return

    rank = re.sub(r"[^A-Za-z0-9_.-]", "_", str(metadata["rank"]))
    marker_path = os.path.join(marker_dir, f"rank-{rank}.jsonl")
    try:
        os.makedirs(marker_dir, exist_ok=True)
        with open(marker_path, "a") as marker_file:
            marker_file.write(json.dumps(metadata, sort_keys=True, default=str))
            marker_file.write("\n")
    except OSError as error:
        print(
            f"NRL_REFERENCE_SETUP_MARKER_ERROR path={marker_path} error={error}",
            file=sys.stderr,
            flush=True,
        )


def log_reference_setup_stage(stage: str, **fields: Any) -> None:
    """Emit one process-local reference setup marker when diagnostics are enabled."""
    if not _diagnostics_enabled():
        return

    metadata = {
        "epoch_s": f"{time.time():.6f}",
        "host": socket.gethostname(),
        "local_rank": os.getenv("LOCAL_RANK", "unknown"),
        "pid": os.getpid(),
        "rank": os.getenv("RANK", "unknown"),
        "stage": stage,
        "world_size": os.getenv("WORLD_SIZE", "unknown"),
        **_linux_memory_metadata(),
        **fields,
    }
    payload = " ".join(f"{key}={value}" for key, value in metadata.items())
    print(f"NRL_REFERENCE_SETUP {payload}", file=sys.stderr, flush=True)
    _write_rank_marker(metadata)


def distributed_timeout_override() -> timedelta | None:
    """Return an optional default process-group timeout configured in seconds."""
    raw_value = os.getenv(_DISTRIBUTED_TIMEOUT_ENV)
    if raw_value is None:
        return None

    timeout_seconds = int(raw_value)
    if timeout_seconds <= 0:
        raise ValueError(f"{_DISTRIBUTED_TIMEOUT_ENV} must be greater than zero")
    return timedelta(seconds=timeout_seconds)


def checkpoint_marker_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Collect lightweight rank-local checkpoint visibility metadata."""
    if not _diagnostics_enabled():
        return {}

    checkpoint_path = os.fspath(path)
    metadata: dict[str, Any] = {"realpath": os.path.realpath(checkpoint_path)}
    for marker in _CHECKPOINT_MARKERS:
        marker_path = os.path.join(checkpoint_path, marker)
        try:
            stat_result = os.stat(marker_path)
        except OSError as error:
            metadata[marker] = {
                "errno": error.errno,
                "exists": False,
            }
        else:
            metadata[marker] = {
                "device": stat_result.st_dev,
                "exists": True,
                "inode": stat_result.st_ino,
                "mtime_ns": stat_result.st_mtime_ns,
                "size": stat_result.st_size,
            }
    return metadata


def _stack_dump_interval_seconds() -> int:
    raw_value = os.getenv(_STACK_DUMP_INTERVAL_ENV, "0")
    try:
        return int(raw_value)
    except ValueError:
        log_reference_setup_stage(
            "diagnostics.invalid_stack_dump_interval", value=raw_value
        )
        return 0


def _dump_stacks_periodically(
    stop_event: threading.Event, interval_seconds: int
) -> None:
    while not stop_event.wait(interval_seconds):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


@contextmanager
def reference_setup_stack_dumps() -> Iterator[None]:
    """Periodically dump Python stacks while reference-model setup is in progress."""
    interval_seconds = _stack_dump_interval_seconds()
    armed = _diagnostics_enabled() and interval_seconds > 0
    stop_event = None
    dump_thread = None
    if armed:
        stop_event = threading.Event()
        dump_thread = threading.Thread(
            target=_dump_stacks_periodically,
            args=(stop_event, interval_seconds),
            name="nrl-reference-setup-stack-dump",
            daemon=True,
        )
        dump_thread.start()
    try:
        yield
    finally:
        if stop_event is not None and dump_thread is not None:
            stop_event.set()
            dump_thread.join(timeout=1.0)
