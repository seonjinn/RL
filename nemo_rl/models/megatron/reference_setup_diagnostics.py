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
import os
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
        **fields,
    }
    payload = " ".join(f"{key}={value}" for key, value in metadata.items())
    print(f"NRL_REFERENCE_SETUP {payload}", file=sys.stderr, flush=True)


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
