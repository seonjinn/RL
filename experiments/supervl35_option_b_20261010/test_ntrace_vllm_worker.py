"""Exercise bounded ntrace capture without importing CUDA or vLLM."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


MODULE = Path(__file__).with_name("ntrace_vllm_worker.py")


class FakeController:
    def __init__(self, *, rank: int) -> None:
        self.rank = rank
        self.events: list[object] = []

    def begin_engine_initialization(self, *, name: str) -> str:
        self.events.append(("init_start", name))
        return "token"

    def end_engine_initialization(self, token: str) -> None:
        self.events.append(("init_end", token))

    def begin_rollout(self, *, step_id: int) -> int:
        self.events.append(("begin", step_id))
        return step_id

    def finish_rollout(self) -> None:
        self.events.append("finish")

    def abort_rollout(self, *, reason: str) -> None:
        self.events.append(("abort", reason))

    def assert_trace_completed(self) -> None:
        self.events.append("assert_complete")

    def close(self) -> None:
        self.events.append("close")


class FakeWorker:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.rank = 2
        self.calls = 0
        self.fail_at: int | None = None
        self.model_runner = SimpleNamespace(capture_model=lambda: "capture")

    def init_device(self) -> None:
        pass

    def compile_or_warm_up_model(self) -> str:
        return self.model_runner.capture_model()

    def execute_model(self, scheduler_output: object) -> int:
        self.calls += 1
        if self.calls == self.fail_at:
            raise ValueError("execute failed")
        return self.calls

    def shutdown(self) -> None:
        self.did_shutdown = True


@pytest.fixture
def worker_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    ntrace = ModuleType("ntrace")
    ntrace.NemoRLRolloutTraceController = FakeController
    gpu_worker = ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker.Worker = FakeWorker
    monkeypatch.setitem(sys.modules, "ntrace", ntrace)
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", ModuleType("vllm.v1.worker"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker)
    spec = importlib.util.spec_from_file_location("test_ntrace_worker_under_test", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_model_is_scoped_and_restored(worker_module: ModuleType) -> None:
    worker = worker_module.NtraceWorker()
    worker.init_device()
    original = worker.model_runner.capture_model
    assert worker.compile_or_warm_up_model() == "capture"
    assert worker.model_runner.capture_model is original
    assert worker.trace_controller.events == [
        ("init_start", "vllm_cudagraph_capture"),
        ("init_end", "token"),
    ]


def test_only_requested_decode_window_is_traced(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NTRACE_DECODE_DELAY_ITERATIONS", "3")
    monkeypatch.setenv("NTRACE_DECODE_NUM_ITERS", "2")
    worker = worker_module.NtraceWorker()
    worker.init_device()
    worker.profile(is_start=True)
    for _ in range(7):
        worker.execute_model(object())
    worker.profile(is_start=False)
    assert worker.trace_controller.events == [
        ("begin", 3), "finish", ("begin", 4), "finish", "assert_complete"
    ]


def test_worker_aborts_trace_on_model_failure(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NTRACE_DECODE_DELAY_ITERATIONS", "0")
    monkeypatch.setenv("NTRACE_DECODE_NUM_ITERS", "2")
    worker = worker_module.NtraceWorker()
    worker.init_device()
    worker.fail_at = 1
    worker.profile(is_start=True)
    with pytest.raises(ValueError, match="execute failed"):
        worker.execute_model(object())
    assert worker.trace_controller.events == [
        ("begin", 0), ("abort", "execute_model_failed")
    ]


def test_base_shutdown_runs_when_trace_validation_fails(worker_module: ModuleType) -> None:
    worker = worker_module.NtraceWorker()
    worker.init_device()

    def fail_close() -> None:
        raise RuntimeError("incomplete trace")

    worker.trace_controller.close = fail_close
    with pytest.raises(RuntimeError, match="incomplete trace"):
        worker.shutdown()
    assert worker.did_shutdown
