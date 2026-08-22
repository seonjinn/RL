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
        existed = counters_path.exists()
        original_size = counters_path.stat().st_size if existed else 0
        try:
            with counters_path.open("a+", encoding="utf-8") as output:
                output.seek(0, os.SEEK_END)
                original_size = output.tell()
                output.write(snapshot.to_json() + "\n")
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            try:
                with counters_path.open("r+b") as output:
                    output.truncate(original_size)
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException:
                pass
            if not existed:
                try:
                    counters_path.unlink(missing_ok=True)
                except BaseException:
                    pass
            raise


@dataclass(slots=True)
class _DraftPerfStep:
    sink: DraftPerfSink
    step: int
    microbatches: int
    profiler: Any | None
    trace_path: Path
    counters: dict[str, tuple[int, int]] = field(default_factory=dict)
    regions: set[str] = field(default_factory=set)
    region_seconds: dict[str, float] = field(default_factory=dict)
    trace_parts: list[Path] = field(default_factory=list)
    phase: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0


_COUNTERS: ContextVar[_DraftPerfStep | None] = ContextVar(
    "draft_perf_counters", default=None
)
_SINK: DraftPerfSink | None = None
_DEFERRED: _DraftPerfStep | None = None


def begin_draft_perf_step(step: int, *, microbatches: int) -> None:
    """Start optional performance collection for a draft training step."""
    global _DEFERRED

    if _COUNTERS.get() is not None:
        raise RuntimeError("a draft performance step is already active")
    if _DEFERRED is not None:
        stale = _DEFERRED
        _DEFERRED = None
        _discard_draft_perf_step(stale)
    sink = _SINK
    if (
        sink is None
        or os.environ.get("NRL_DRAFT_PERF_PROFILE") != "1"
        or not os.environ.get("NRL_DRAFT_PERF_OUTPUT_DIR")
    ):
        return

    sink.rank_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    profiler = _new_profiler()
    try:
        profiler.__enter__()
    except BaseException:
        try:
            profiler.__exit__(None, None, None)
        except BaseException:
            pass
        raise
    _COUNTERS.set(
        _DraftPerfStep(
            sink=sink,
            step=step,
            microbatches=microbatches,
            profiler=profiler,
            trace_path=sink.rank_dir / f"step-{step}.trace.json",
        )
    )


def finish_draft_perf_step(
    step: int, *, defer_refit: bool = False
) -> DraftPerfSnapshot:
    """Close one phase and optionally defer artifact commit through refit."""
    global _DEFERRED

    state = _COUNTERS.get()
    if state is None:
        return _empty_snapshot(step)
    if step != state.step:
        _COUNTERS.set(None)
        _discard_draft_perf_step(state)
        raise ValueError(f"draft performance step changed from {state.step} to {step}")

    try:
        _finish_draft_perf_phase(state)
        snapshot = _snapshot(state)
        _COUNTERS.set(None)
        if defer_refit:
            if _DEFERRED is not None:
                stale = _DEFERRED
                _DEFERRED = None
                _discard_draft_perf_step(stale)
            _DEFERRED = state
        else:
            _commit_draft_perf_artifacts(state, snapshot)
    except BaseException:
        _COUNTERS.set(None)
        if _DEFERRED is state:
            _DEFERRED = None
        _discard_draft_perf_step(state)
        raise
    return snapshot


def begin_draft_perf_refit(step: int) -> None:
    """Resume a deferred training step only for its associated refit RPC."""
    global _DEFERRED

    if _COUNTERS.get() is not None:
        raise RuntimeError("a draft performance step is already active")
    state = _DEFERRED
    if state is None:
        return
    _DEFERRED = None
    if step != state.step:
        _discard_draft_perf_step(state)
        raise ValueError(f"draft performance step changed from {state.step} to {step}")
    try:
        torch.cuda.reset_peak_memory_stats()
        profiler = _new_profiler()
        state.profiler = profiler
        profiler.__enter__()
        state.regions.clear()
        _COUNTERS.set(state)
    except BaseException:
        _COUNTERS.set(None)
        _discard_draft_perf_step(state)
        raise


def finish_draft_perf_refit(step: int) -> DraftPerfSnapshot:
    """Commit one deferred training/refit snapshot and combined trace."""
    return finish_draft_perf_step(step)


def abort_draft_perf_step() -> None:
    """Discard an in-progress profiling step without emitting a completed row."""
    global _DEFERRED

    state = _COUNTERS.get()
    deferred = _DEFERRED
    _COUNTERS.set(None)
    _DEFERRED = None
    if state is not None:
        _discard_draft_perf_step(state)
    if deferred is not None and deferred is not state:
        _discard_draft_perf_step(deferred)


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


def increment_draft_perf_microbatches(microbatches: int = 1) -> None:
    """Record completed microbatches for the active logical training step."""
    state = _COUNTERS.get()
    if state is None:
        return
    state.microbatches += microbatches


def _region_seconds(profiler: Any, regions: set[str]) -> dict[str, float]:
    return {
        event.key: _event_seconds(event)
        for event in profiler.key_averages()
        if event.key in regions
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
    profiler = state.profiler
    state.profiler = None
    try:
        if profiler is not None:
            profiler.__exit__(None, None, None)
    except BaseException:
        pass
    for path in (
        *state.trace_parts,
        state.trace_path,
        _trace_staging_path(state),
    ):
        try:
            path.unlink(missing_ok=True)
        except BaseException:
            pass


def _new_profiler() -> Any:
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=True,
    )


def _finish_draft_perf_phase(state: _DraftPerfStep) -> None:
    profiler = state.profiler
    if profiler is None:
        raise RuntimeError("draft performance phase has no active profiler")
    profiler.__exit__(None, None, None)
    state.profiler = None
    phase_regions = _region_seconds(profiler, state.regions)
    for name, seconds in phase_regions.items():
        state.region_seconds[name] = state.region_seconds.get(name, 0.0) + seconds
    state.peak_allocated_bytes = max(
        state.peak_allocated_bytes,
        int(torch.cuda.max_memory_allocated()),
    )
    state.peak_reserved_bytes = max(
        state.peak_reserved_bytes,
        int(torch.cuda.max_memory_reserved()),
    )
    trace_part = state.trace_path.with_name(
        f"step-{state.step}.phase-{state.phase}.trace.json.tmp"
    )
    try:
        trace_part.unlink(missing_ok=True)
    except BaseException:
        pass
    state.trace_parts.append(trace_part)
    profiler.export_chrome_trace(str(trace_part))
    state.phase += 1


def _snapshot(state: _DraftPerfStep) -> DraftPerfSnapshot:
    return DraftPerfSnapshot(
        global_rank=state.sink.global_rank,
        step=state.step,
        microbatches=state.microbatches,
        region_seconds=dict(sorted(state.region_seconds.items())),
        calls={name: values[0] for name, values in state.counters.items()},
        bytes={name: values[1] for name, values in state.counters.items()},
        peak_allocated_bytes=state.peak_allocated_bytes,
        peak_reserved_bytes=state.peak_reserved_bytes,
    )


def _empty_snapshot(step: int) -> DraftPerfSnapshot:
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


def _trace_staging_path(state: _DraftPerfStep) -> Path:
    return state.trace_path.with_name(f"{state.trace_path.name}.tmp")


def _commit_draft_perf_artifacts(
    state: _DraftPerfStep, snapshot: DraftPerfSnapshot
) -> None:
    staging_path = _trace_staging_path(state)
    try:
        staging_path.unlink(missing_ok=True)
    except BaseException:
        pass
    try:
        _merge_trace_parts(state.trace_parts, staging_path)
        os.replace(staging_path, state.trace_path)
        state.sink.append(snapshot)
    except BaseException:
        for path in (staging_path, state.trace_path, *state.trace_parts):
            try:
                path.unlink(missing_ok=True)
            except BaseException:
                pass
        raise
    finally:
        for path in state.trace_parts:
            try:
                path.unlink(missing_ok=True)
            except BaseException:
                pass
        state.trace_parts.clear()


def _merge_trace_parts(parts: list[Path], output_path: Path) -> None:
    if not parts:
        raise RuntimeError("draft performance step produced no profiler trace")
    if len(parts) == 1:
        os.replace(parts[0], output_path)
        return

    combined: dict[str, Any] | None = None
    trace_events: list[Any] = []
    for part in parts:
        payload = json.loads(part.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(
            payload.get("traceEvents"), list
        ):
            raise ValueError(f"invalid profiler trace payload in {part}")
        if combined is None:
            combined = dict(payload)
        trace_events.extend(payload["traceEvents"])
    assert combined is not None
    combined["traceEvents"] = trace_events
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(combined, output, separators=(",", ":"))
        output.flush()
        os.fsync(output.fileno())
