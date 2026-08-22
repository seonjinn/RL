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

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from nemo_rl.models.megatron.draft import perf_counters
from nemo_rl.models.megatron.draft.perf_counters import (
    DraftPerfSink,
    DraftPerfSnapshot,
    abort_draft_perf_step,
    begin_draft_perf_refit,
    begin_draft_perf_step,
    count_draft_perf,
    draft_perf_region,
    finish_draft_perf_refit,
    finish_draft_perf_step,
    finish_deferred_draft_perf_step,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_PRODUCER_PATHS = (
    "nemo_rl/algorithms/loss/draft.py",
    "nemo_rl/algorithms/loss/wrapper.py",
    "nemo_rl/models/megatron/draft/hidden_capture.py",
    "nemo_rl/models/megatron/draft/diagnostics.py",
    "nemo_rl/models/megatron/draft/training.py",
    "nemo_rl/models/megatron/draft/step_state.py",
    "nemo_rl/models/megatron/draft/utils.py",
    "nemo_rl/models/policy/workers/megatron_policy_worker.py",
)


def _source(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _class_method_source(class_name: str, method_name: str) -> str:
    source = _source("nemo_rl/models/policy/workers/megatron_policy_worker.py")
    tree = ast.parse(source)
    worker_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in worker_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def _lifecycle_worker_class() -> type:
    source = _source("nemo_rl/models/policy/workers/megatron_policy_worker.py")
    tree = ast.parse(source)
    worker_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    method_names = {
        "train",
        "begin_train_step",
        "finish_train_step",
        "abort_train_step",
    }
    methods = [
        node
        for node in worker_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {method.name for method in methods} == method_names
    for method in methods:
        method.decorator_list = []
        method.returns = None
        for argument in (
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
        ):
            argument.annotation = None
    lifecycle_class = ast.ClassDef(
        name="LifecycleWorker",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[lifecycle_class], type_ignores=[])
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "Iterable": object,
        "LossFunction": object,
        "Optional": object,
        "_best_effort_abort_draft_perf": lambda: None,
        "nullcontext": __import__("contextlib").nullcontext,
    }
    exec(compile(module, "<draft-perf-lifecycle>", "exec"), namespace)
    namespace["_best_effort_abort_draft_perf"] = lambda: namespace[
        "abort_draft_perf_step"
    ]()
    return namespace["LifecycleWorker"]


def test_integration_producers_cover_final_head_hot_path() -> None:
    sources = {path: _source(path) for path in _PRODUCER_PATHS}

    assert set(sources) == set(_PRODUCER_PATHS)
    assert '"metadata_collective"' in sources[_PRODUCER_PATHS[0]]
    assert 'draft_perf_region("draft.loss_backward")' in sources[_PRODUCER_PATHS[0]]
    assert '"tensor_materialization"' in sources[_PRODUCER_PATHS[1]]
    assert 'count_draft_perf("scalar_materialization"' in sources[_PRODUCER_PATHS[1]]
    assert 'draft_perf_region("draft.hidden_capture")' in sources[_PRODUCER_PATHS[2]]
    assert 'count_draft_perf("scalar_materialization"' in sources[_PRODUCER_PATHS[3]]
    assert 'draft_perf_region("draft.provider_forward")' in sources[_PRODUCER_PATHS[4]]
    assert sources[_PRODUCER_PATHS[4]].count("@_profile_provider_forward") == 2
    assert 'count_draft_perf("scalar_materialization"' in sources[_PRODUCER_PATHS[5]]
    assert (
        'draft_perf_region("draft.export_reconstruct")' in sources[_PRODUCER_PATHS[6]]
    )
    assert 'draft_perf_region("draft.refit_transfer")' in sources[_PRODUCER_PATHS[6]]
    assert '"refit_payload_collective"' in sources[_PRODUCER_PATHS[6]]

    worker_source = sources[_PRODUCER_PATHS[7]]
    sync_train = _class_method_source("MegatronPolicyWorkerImpl", "_train_body")
    split_finish = _class_method_source(
        "MegatronPolicyWorkerImpl", "_finish_train_step_body"
    )
    assert sync_train.count('draft_perf_region("draft.optimizer_finalize")') == 1
    assert split_finish.count('draft_perf_region("draft.optimizer_finalize")') == 1
    assert 'draft_perf_region("draft.finish_normalization")' in worker_source
    assert worker_source.count("begin_draft_perf_step(") == 2
    assert worker_source.count("finish_draft_perf_step(") == 2
    assert worker_source.count("_best_effort_abort_draft_perf(") >= 4
    assert worker_source.count("increment_draft_perf_microbatches(") == 2

    init_source = _class_method_source("MegatronPolicyWorkerImpl", "__init__")
    assert init_source.count("DraftPerfSink.from_env(global_rank=self.rank)") == 1
    assert "if self.draft_provider is not None:" in init_source
    assert "draft_perf" not in _source("nemo_rl/models/megatron/train.py")


def test_step_snapshot_lifecycle_matches_monolithic_split_and_abort(
    monkeypatch: Any,
) -> None:
    begins: list[tuple[int, int]] = []
    finishes: list[int] = []
    aborts: list[None] = []
    worker_type = _lifecycle_worker_class()
    monkeypatch.setitem(
        worker_type.train.__globals__,
        "begin_draft_perf_step",
        lambda step, *, microbatches: begins.append((step, microbatches)),
    )
    monkeypatch.setitem(
        worker_type.train.__globals__,
        "finish_draft_perf_step",
        lambda step, **_kwargs: finishes.append(step),
    )
    monkeypatch.setitem(
        worker_type.train.__globals__,
        "abort_draft_perf_step",
        lambda: aborts.append(None),
    )

    monolithic = worker_type()
    monolithic.draft_provider = object()
    monolithic.scheduler = SimpleNamespace(num_steps=12)
    monolithic._train_body = MagicMock(return_value={"mode": "monolithic"})
    assert monolithic.train(object(), object()) == {"mode": "monolithic"}
    assert begins == [(12, 0)]
    assert finishes == [12]
    assert aborts == []

    split = worker_type()
    split.draft_provider = object()
    split.scheduler = SimpleNamespace(num_steps=13)
    split.model = MagicMock()
    split.model.modules.return_value = ()
    split.optimizer = MagicMock()
    split._split_step_state_init = MagicMock(return_value={})

    def finish_split(_state: dict[str, Any]) -> dict[str, str]:
        split._train_step_state = None
        return {"mode": "split"}

    split._finish_train_step_body = MagicMock(side_effect=finish_split)
    split._restore_saved_mcore_hooks = MagicMock()
    split._assert_step_open = lambda: split._train_step_state
    split.begin_train_step(object())
    assert split.finish_train_step() == {"mode": "split"}
    assert begins[-1] == (13, 0)
    assert finishes[-1] == 13

    split.scheduler.num_steps = 14
    split.begin_train_step(object())
    split.abort_train_step()
    assert begins[-1] == (14, 0)
    assert finishes == [12, 13]
    assert len(aborts) == 1

    monolithic._train_body.side_effect = RuntimeError("monolithic failure")
    with pytest.raises(RuntimeError, match="monolithic failure"):
        monolithic.train(object(), object())
    assert finishes == [12, 13]
    assert len(aborts) == 2

    split.scheduler.num_steps = 15
    split._finish_train_step_body.side_effect = RuntimeError("split failure")
    split.begin_train_step(object())
    with pytest.raises(RuntimeError, match="split failure"):
        split.finish_train_step()
    assert finishes == [12, 13]
    assert len(aborts) == 3

    fixed = worker_type()
    fixed.draft_provider = None
    fixed._train_body = MagicMock(return_value={"mode": "fixed"})
    assert fixed.train(object(), object()) == {"mode": "fixed"}

    fixed.model = MagicMock()
    fixed.model.modules.return_value = ()
    fixed.optimizer = MagicMock()
    fixed.scheduler = SimpleNamespace(num_steps=16)
    fixed._split_step_state_init = MagicMock(return_value={})

    def finish_fixed_split(_state: dict[str, Any]) -> dict[str, str]:
        fixed._train_step_state = None
        return {"mode": "fixed-split"}

    fixed._finish_train_step_body = MagicMock(side_effect=finish_fixed_split)
    fixed._restore_saved_mcore_hooks = MagicMock()
    fixed._assert_step_open = lambda: fixed._train_step_state
    fixed.begin_train_step(object())
    assert fixed.finish_train_step() == {"mode": "fixed-split"}
    fixed.begin_train_step(object())
    fixed.abort_train_step()
    assert begins == [(12, 0), (13, 0), (14, 0), (12, 0), (15, 0)]
    assert finishes == [12, 13]
    assert len(aborts) == 3


def test_step_snapshot_counts_only_completed_microbatches(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setenv("NRL_DRAFT_PERF_PROFILE", "1")
    monkeypatch.setenv("NRL_DRAFT_PERF_OUTPUT_DIR", str(tmp_path))
    profiler = _FakeProfiler()
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: profiler)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    assert DraftPerfSink.from_env(global_rank=5) is not None
    begin_draft_perf_step(21, microbatches=0)
    perf_counters.increment_draft_perf_microbatches(2)
    perf_counters.increment_draft_perf_microbatches()
    snapshot = finish_draft_perf_step(21)

    assert snapshot.microbatches == 3


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
    def __init__(
        self,
        *,
        events: list[Any] | None = None,
        trace_events: list[dict[str, object]] | None = None,
    ) -> None:
        self.exited = False
        self.exit_calls = 0
        self.events = events if events is not None else [_ProfilerEvent()]
        self.trace_events = trace_events if trace_events is not None else []

    def __enter__(self) -> _FakeProfiler:
        return self

    def __exit__(self, *args: object) -> None:
        self.exited = True
        self.exit_calls += 1

    def key_averages(self) -> list[_ProfilerEvent]:
        return self.events

    def export_chrome_trace(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"traceEvents": self.trace_events}), encoding="utf-8"
        )


def test_deferred_refit_emits_one_snapshot_with_both_phases(
    monkeypatch: Any, tmp_path: Path
) -> None:
    class _TrainingEvent:
        key = "draft.provider_forward"
        cuda_time_total = 1_000.0

    class _RefitEvent:
        key = "draft.export_reconstruct"
        cuda_time_total = 2_000.0

    profilers = iter(
        (
            _FakeProfiler(
                events=[_TrainingEvent()],
                trace_events=[{"name": "train"}],
            ),
            _FakeProfiler(
                events=[_RefitEvent()],
                trace_events=[{"name": "refit"}],
            ),
        )
    )
    monkeypatch.setenv("NRL_DRAFT_PERF_PROFILE", "1")
    monkeypatch.setenv("NRL_DRAFT_PERF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: next(profilers))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 101)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 202)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    assert DraftPerfSink.from_env(global_rank=7) is not None
    begin_draft_perf_step(31, microbatches=0)
    perf_counters.increment_draft_perf_microbatches(2)
    perf_counters.increment_draft_perf_microbatches(3)
    count_draft_perf("metadata_collective", calls=4, num_bytes=64)
    with draft_perf_region("draft.provider_forward"):
        pass
    finish_draft_perf_step(31, defer_refit=True)

    rank_dir = tmp_path / "rank-7"
    assert not (rank_dir / "counters.jsonl").exists()
    assert not (rank_dir / "step-31.trace.json").exists()

    begin_draft_perf_refit(31)
    count_draft_perf("refit_payload_collective", calls=2, num_bytes=4096)
    with draft_perf_region("draft.export_reconstruct"):
        pass
    snapshot = finish_draft_perf_refit(31)

    rows = (rank_dir / "counters.jsonl").read_text(encoding="utf-8").splitlines()
    trace = json.loads((rank_dir / "step-31.trace.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert json.loads(rows[0]) == json.loads(snapshot.to_json())
    assert snapshot.microbatches == 5
    assert snapshot.region_seconds == {
        "draft.export_reconstruct": 0.002,
        "draft.provider_forward": 0.001,
    }
    assert snapshot.calls == {
        "metadata_collective": 4,
        "refit_payload_collective": 2,
    }
    assert snapshot.bytes["refit_payload_collective"] == 4096
    assert [event["name"] for event in trace["traceEvents"]] == ["train", "refit"]
    assert not list(rank_dir.glob("*.tmp"))


def test_deferred_refit_failure_discards_the_whole_step(
    monkeypatch: Any, tmp_path: Path
) -> None:
    profilers = iter((_FakeProfiler(), _FakeProfiler()))
    monkeypatch.setenv("NRL_DRAFT_PERF_PROFILE", "1")
    monkeypatch.setenv("NRL_DRAFT_PERF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: next(profilers))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    assert DraftPerfSink.from_env(global_rank=8) is not None
    begin_draft_perf_step(32, microbatches=0)
    finish_draft_perf_step(32, defer_refit=True)
    begin_draft_perf_refit(32)
    abort_draft_perf_step()

    rank_dir = tmp_path / "rank-8"
    assert not (rank_dir / "counters.jsonl").exists()
    assert not (rank_dir / "step-32.trace.json").exists()
    assert not list(rank_dir.glob("step-32*"))


def test_deferred_step_can_commit_without_a_refit_phase(
    monkeypatch: Any, tmp_path: Path
) -> None:
    profiler = _FakeProfiler(trace_events=[{"name": "train"}])
    monkeypatch.setenv("NRL_DRAFT_PERF_PROFILE", "1")
    monkeypatch.setenv("NRL_DRAFT_PERF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: profiler)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 11)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 22)

    assert DraftPerfSink.from_env(global_rank=9) is not None
    begin_draft_perf_step(34, microbatches=0)
    perf_counters.increment_draft_perf_microbatches(3)
    count_draft_perf("metadata_collective", calls=2, num_bytes=128)
    finish_draft_perf_step(34, defer_refit=True)

    rank_dir = tmp_path / "rank-9"
    assert not (rank_dir / "counters.jsonl").exists()
    snapshot = finish_deferred_draft_perf_step(34)

    rows = (rank_dir / "counters.jsonl").read_text(encoding="utf-8").splitlines()
    trace = json.loads((rank_dir / "step-34.trace.json").read_text(encoding="utf-8"))
    assert json.loads(rows[0]) == json.loads(snapshot.to_json())
    assert snapshot.microbatches == 3
    assert snapshot.calls == {"metadata_collective": 2}
    assert snapshot.bytes == {"metadata_collective": 128}
    assert [event["name"] for event in trace["traceEvents"]] == ["train"]
    assert profiler.exit_calls == 1
    assert not list(rank_dir.glob("*.tmp"))


def test_deferred_no_refit_commit_failure_discards_artifacts_and_state(
    monkeypatch: Any, tmp_path: Path
) -> None:
    profiler = _FakeProfiler(trace_events=[{"name": "train"}])
    monkeypatch.setenv("NRL_DRAFT_PERF_PROFILE", "1")
    monkeypatch.setenv("NRL_DRAFT_PERF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: profiler)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    sink = DraftPerfSink.from_env(global_rank=10)
    assert sink is not None
    begin_draft_perf_step(35, microbatches=1)
    finish_draft_perf_step(35, defer_refit=True)

    def fail_append(_sink: DraftPerfSink, _snapshot: DraftPerfSnapshot) -> None:
        raise OSError("append failed")

    monkeypatch.setattr(DraftPerfSink, "append", fail_append)

    with pytest.raises(OSError, match="append failed"):
        finish_deferred_draft_perf_step(35)

    rank_dir = tmp_path / "rank-10"
    assert not list(rank_dir.glob("step-35*"))
    assert not (rank_dir / "counters.jsonl").exists()
    assert finish_deferred_draft_perf_step(35).microbatches == 0


def test_finish_failure_rolls_back_partial_artifacts_and_allows_next_step(
    monkeypatch: Any, tmp_path: Path
) -> None:
    profiler = _FakeProfiler(trace_events=[{"name": "partial"}])
    monkeypatch.setenv("NRL_DRAFT_PERF_PROFILE", "1")
    monkeypatch.setenv("NRL_DRAFT_PERF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: profiler)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    real_fsync = os.fsync
    fsync_calls = 0

    def fail_first_fsync(fd: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            raise OSError("fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_first_fsync)
    assert DraftPerfSink.from_env(global_rank=9) is not None
    begin_draft_perf_step(33, microbatches=1)
    with pytest.raises(OSError, match="fsync failed"):
        finish_draft_perf_step(33)

    rank_dir = tmp_path / "rank-9"
    assert not (rank_dir / "step-33.trace.json").exists()
    counters_path = rank_dir / "counters.jsonl"
    assert not counters_path.exists() or counters_path.read_bytes() == b""
    assert not list(rank_dir.glob("*.tmp"))

    begin_draft_perf_step(34, microbatches=1)
    abort_draft_perf_step()


def test_trace_export_failure_removes_partial_trace(
    monkeypatch: Any, tmp_path: Path
) -> None:
    class _ExportFailureProfiler(_FakeProfiler):
        def export_chrome_trace(self, path: str) -> None:
            Path(path).write_text("partial", encoding="utf-8")
            raise OSError("trace export failed")

    profiler = _ExportFailureProfiler()
    monkeypatch.setenv("NRL_DRAFT_PERF_PROFILE", "1")
    monkeypatch.setenv("NRL_DRAFT_PERF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: profiler)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    assert DraftPerfSink.from_env(global_rank=11) is not None
    begin_draft_perf_step(37, microbatches=1)
    with pytest.raises(OSError, match="trace export failed"):
        finish_draft_perf_step(37)

    rank_dir = tmp_path / "rank-11"
    assert not list(rank_dir.glob("step-37*"))
    assert not (rank_dir / "counters.jsonl").exists()


def test_abort_suppresses_profiler_and_unlink_cleanup_failures(
    monkeypatch: Any, tmp_path: Path
) -> None:
    class _ExitFailureProfiler(_FakeProfiler):
        def __exit__(self, *args: object) -> None:
            super().__exit__(*args)
            raise RuntimeError("profiler exit failed")

    profilers = iter((_ExitFailureProfiler(), _FakeProfiler()))
    monkeypatch.setenv("NRL_DRAFT_PERF_PROFILE", "1")
    monkeypatch.setenv("NRL_DRAFT_PERF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: next(profilers))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    assert DraftPerfSink.from_env(global_rank=10) is not None
    begin_draft_perf_step(35, microbatches=1)
    real_unlink = Path.unlink

    def fail_unlink(self: Path, *, missing_ok: bool = False) -> None:
        if "step-35" in self.name:
            raise OSError("unlink failed")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    abort_draft_perf_step()

    begin_draft_perf_step(36, microbatches=1)
    abort_draft_perf_step()


def test_disabled_counters_issue_no_tensor_or_collective_operations(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("NRL_DRAFT_PERF_PROFILE", raising=False)
    monkeypatch.delenv("NRL_DRAFT_PERF_OUTPUT_DIR", raising=False)

    with _ForbiddenOperationRecorder() as recorder:
        begin_draft_perf_step(1, microbatches=1)
        perf_counters.increment_draft_perf_microbatches()
        with draft_perf_region("draft/metadata"):
            count_draft_perf("metadata_collective", calls=2, num_bytes=64)
        finish_draft_perf_step(1)
        finish_deferred_draft_perf_step(1)

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
    reset_calls: list[None] = []
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: profiler)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: reset_calls.append(None)
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 101)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 202)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    sink = DraftPerfSink.from_env(global_rank=3)

    assert sink is not None
    begin_draft_perf_step(7, microbatches=2)
    assert reset_calls == [None]
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


def test_mismatched_finish_cleans_up_before_raising(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setenv("NRL_DRAFT_PERF_PROFILE", "1")
    monkeypatch.setenv("NRL_DRAFT_PERF_OUTPUT_DIR", str(tmp_path))
    profiler = _FakeProfiler()
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: profiler)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    sink = DraftPerfSink.from_env(global_rank=3)

    assert sink is not None
    begin_draft_perf_step(9, microbatches=2)
    partial_trace = tmp_path / "rank-3" / "step-9.trace.json"
    partial_trace.write_text("partial", encoding="utf-8")
    with pytest.raises(ValueError, match="changed from 9 to 10"):
        finish_draft_perf_step(10)

    assert profiler.exit_calls == 1
    assert not partial_trace.exists()
    assert not (tmp_path / "rank-3" / "counters.jsonl").exists()

    begin_draft_perf_step(11, microbatches=2)
    abort_draft_perf_step()

    assert profiler.exit_calls == 2
