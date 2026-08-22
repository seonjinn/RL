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

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch


@dataclass(frozen=True, slots=True)
class DraftPerfSnapshot:
    """Performance counters captured for one draft training step."""

    global_rank: int
    step: int
    microbatches: int
    region_seconds: dict[str, float]
    calls: dict[str, int]
    bytes: dict[str, int]
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def to_json(self) -> str:
        """Serialize the snapshot as a compact, deterministically ordered JSON row."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class DraftPerfSink:
    """Rank-qualified destination for draft performance artifacts."""

    output_dir: Path
    global_rank: int

    @classmethod
    def from_env(cls, global_rank: int) -> DraftPerfSink | None:
        """Configure the process-local sink when profiling is enabled by the environment."""
        global _SINK

        output_dir = os.environ.get("NRL_DRAFT_PERF_OUTPUT_DIR")
        if os.environ.get("NRL_DRAFT_PERF_PROFILE") != "1" or not output_dir:
            _SINK = None
            return None
        _SINK = cls(output_dir=Path(output_dir), global_rank=global_rank)
        return _SINK

    @property
    def rank_dir(self) -> Path:
        """Return the directory that holds this rank's artifacts."""
        return self.output_dir / f"rank-{self.global_rank}"

    def append(self, snapshot: DraftPerfSnapshot) -> None:
        """Append one durable JSONL row for a completed step."""
        counters_path = self.rank_dir / "counters.jsonl"
        with counters_path.open("a", encoding="utf-8") as output:
            output.write(snapshot.to_json() + "\n")
            output.flush()
            os.fsync(output.fileno())


@dataclass(slots=True)
class _DraftPerfStep:
    sink: DraftPerfSink
    step: int
    microbatches: int
    profiler: Any
    trace_path: Path
    counters: dict[str, tuple[int, int]] = field(default_factory=dict)
    regions: set[str] = field(default_factory=set)


_COUNTERS: ContextVar[_DraftPerfStep | None] = ContextVar(
    "draft_perf_counters", default=None
)
_SINK: DraftPerfSink | None = None


def begin_draft_perf_step(step: int, *, microbatches: int) -> None:
    """Start optional performance collection for a draft training step."""
    if _COUNTERS.get() is not None:
        raise RuntimeError("a draft performance step is already active")
    sink = _SINK
    if (
        sink is None
        or os.environ.get("NRL_DRAFT_PERF_PROFILE") != "1"
        or not os.environ.get("NRL_DRAFT_PERF_OUTPUT_DIR")
    ):
        return

    sink.rank_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=True,
    )
    profiler.__enter__()
    _COUNTERS.set(
        _DraftPerfStep(
            sink=sink,
            step=step,
            microbatches=microbatches,
            profiler=profiler,
            trace_path=sink.rank_dir / f"step-{step}.trace.json",
        )
    )


def finish_draft_perf_step(step: int) -> DraftPerfSnapshot:
    """Finish collection, export the trace, and durably append a counter row."""
    state = _COUNTERS.get()
    if state is None:
        return DraftPerfSnapshot(
            global_rank=0,
            step=step,
            microbatches=0,
            region_seconds={},
            calls={},
            bytes={},
            peak_allocated_bytes=0,
            peak_reserved_bytes=0,
        )
    if step != state.step:
        _discard_draft_perf_step(state)
        raise ValueError(f"draft performance step changed from {state.step} to {step}")

    try:
        state.profiler.__exit__(None, None, None)
        snapshot = DraftPerfSnapshot(
            global_rank=state.sink.global_rank,
            step=state.step,
            microbatches=state.microbatches,
            region_seconds=_region_seconds(state),
            calls={name: values[0] for name, values in state.counters.items()},
            bytes={name: values[1] for name, values in state.counters.items()},
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
        state.profiler.export_chrome_trace(str(state.trace_path))
        state.sink.append(snapshot)
        return snapshot
    finally:
        _COUNTERS.set(None)


def abort_draft_perf_step() -> None:
    """Discard an in-progress profiling step without emitting a completed row."""
    state = _COUNTERS.get()
    if state is None:
        return
    _discard_draft_perf_step(state)


def draft_perf_region(name: str) -> AbstractContextManager[None]:
    """Record a named profiler and NVTX region only while collection is active."""
    if _COUNTERS.get() is None:
        return nullcontext()
    return _enabled_draft_perf_region(name)


@contextmanager
def _enabled_draft_perf_region(name: str) -> Iterator[None]:
    state = _COUNTERS.get()
    if state is None:
        yield
        return
    state.regions.add(name)
    torch.cuda.nvtx.range_push(name)
    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        torch.cuda.nvtx.range_pop()


def count_draft_perf(
    name: str,
    *,
    calls: int = 1,
    num_bytes: int = 0,
) -> None:
    """Increment pure-Python counters for an active draft profiling step."""
    state = _COUNTERS.get()
    if state is None:
        return
    old_calls, old_bytes = state.counters.get(name, (0, 0))
    state.counters[name] = (old_calls + calls, old_bytes + num_bytes)


def _region_seconds(state: _DraftPerfStep) -> dict[str, float]:
    return {
        event.key: _event_seconds(event)
        for event in state.profiler.key_averages()
        if event.key in state.regions
    }


def _event_seconds(event: Any) -> float:
    for attribute in (
        "device_time_total",
        "cuda_time_total",
        "self_cuda_time_total",
        "cpu_time_total",
    ):
        microseconds = getattr(event, attribute, 0.0)
        if microseconds:
            return float(microseconds) / 1_000_000.0
    return 0.0


def _discard_draft_perf_step(state: _DraftPerfStep) -> None:
    try:
        state.profiler.__exit__(None, None, None)
    finally:
        state.trace_path.unlink(missing_ok=True)
        _COUNTERS.set(None)
