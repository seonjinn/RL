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
import fcntl
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
_OFFLOAD_LOCK_DIR_ENV = "NRL_REFERENCE_CPU_OFFLOAD_LOCK_DIR"
_SERIALIZE_OFFLOAD_ENV = "NRL_SERIALIZE_REFERENCE_CPU_OFFLOAD"
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


def reference_cpu_offload_serialization_enabled() -> bool:
    """Return whether initial reference setup should serialize CPU offload per host."""
    return os.getenv(_SERIALIZE_OFFLOAD_ENV, "").strip().lower() in _TRUE_VALUES


@contextmanager
def reference_cpu_offload_lock(*, enabled: bool) -> Iterator[None]:
    """Serialize large pinned-memory allocations among ranks on the same host."""
    if not enabled:
        yield
        return

    lock_dir = os.getenv(_OFFLOAD_LOCK_DIR_ENV, "/tmp")
    hostname = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
    lock_path = os.path.join(lock_dir, f"nrl-reference-offload-{hostname}.lock")
    os.makedirs(lock_dir, exist_ok=True)
    log_reference_setup_stage("worker.before_reference_offload_lock")
    with open(lock_path, "a") as lock_file:
        wait_started = time.monotonic()
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        log_reference_setup_stage(
            "worker.after_reference_offload_lock",
            lock_wait_s=f"{time.monotonic() - wait_started:.6f}",
        )
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


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
