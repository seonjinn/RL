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
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from nemo_rl.models.megatron.draft.perf_counters import (
    DraftPerfSink,
    abort_draft_perf_step,
    begin_draft_perf_step,
    count_draft_perf,
    draft_perf_region,
    finish_draft_perf_step,
)


class _ForbiddenOperationRecorder(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def __torch_dispatch__(
        self,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        self.calls.append(str(func))
        return func(*args, **({} if kwargs is None else kwargs))  # type: ignore[operator]


class _ProfilerEvent:
    key = "draft/metadata"
    cuda_time_total = 2_500.0


class _FakeProfiler:
    def __init__(self) -> None:
        self.exited = False

    def __enter__(self) -> _FakeProfiler:
        return self

    def __exit__(self, *args: object) -> None:
        self.exited = True

    def key_averages(self) -> list[_ProfilerEvent]:
        return [_ProfilerEvent()]

    def export_chrome_trace(self, path: str) -> None:
        Path(path).write_text("{}", encoding="utf-8")


def test_disabled_counters_issue_no_tensor_or_collective_operations(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("NRL_DRAFT_PERF_PROFILE", raising=False)
    monkeypatch.delenv("NRL_DRAFT_PERF_OUTPUT_DIR", raising=False)

    with _ForbiddenOperationRecorder() as recorder:
        begin_draft_perf_step(1, microbatches=1)
        with draft_perf_region("draft/metadata"):
            count_draft_perf("metadata_collective", calls=2, num_bytes=64)
        finish_draft_perf_step(1)

    forbidden = (
        "_local_scalar_dense",
        "clone",
        "cat",
        "all_gather",
        "broadcast",
        "all_reduce",
    )
    operation_counts = {
        name: sum(name in call for call in recorder.calls) for name in forbidden
    }
    assert operation_counts == dict.fromkeys(forbidden, 0)


def test_enabled_sink_writes_trace_and_fsynced_rank_jsonl(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setenv("NRL_DRAFT_PERF_PROFILE", "1")
    monkeypatch.setenv("NRL_DRAFT_PERF_OUTPUT_DIR", str(tmp_path))
    profiler = _FakeProfiler()
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: profiler)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 101)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 202)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    sink = DraftPerfSink.from_env(global_rank=3)

    assert sink is not None
    begin_draft_perf_step(7, microbatches=2)
    with draft_perf_region("draft/metadata"):
        count_draft_perf("metadata_collective", calls=2, num_bytes=64)
    snapshot = finish_draft_perf_step(7)

    rank_dir = tmp_path / "rank-3"
    trace_path = rank_dir / "step-7.trace.json"
    rows = (rank_dir / "counters.jsonl").read_text(encoding="utf-8").splitlines()
    assert list(rank_dir.glob("step-*.trace.json")) == [trace_path]
    assert len(rows) == 1
    assert json.loads(rows[0]) == json.loads(snapshot.to_json())
    assert snapshot.region_seconds == {"draft/metadata": 0.0025}
    assert snapshot.peak_allocated_bytes == 101
    assert snapshot.peak_reserved_bytes == 202
    assert snapshot.calls == {"metadata_collective": 2}
    assert snapshot.bytes == {"metadata_collective": 64}
    assert profiler.exited

    begin_draft_perf_step(8, microbatches=2)
    abort_draft_perf_step()

    assert (
        len((rank_dir / "counters.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    )
    assert not (rank_dir / "step-8.trace.json").exists()
