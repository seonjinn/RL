# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import asyncio
import sys
import threading
import time
from types import ModuleType
from typing import Any

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


class _BlockingLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.sampling_params: Any = None

    def generate(self, *, sampling_params: Any, **_kwargs: Any):
        self.sampling_params = sampling_params

        async def stream():
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
            yield None

        return stream()


def _make_async_worker(llm: _BlockingLLM) -> Any:
    worker: Any = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker._refit_failure_reason = None
    worker.cfg = {
        "_pad_token_id": 0,
        "ignore_eos": False,
        "max_new_tokens": 4,
        "stop_strings": None,
        "stop_token_ids": [9],
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
        "vllm_cfg": {"async_engine": True, "max_model_len": 32},
    }
    worker.SamplingParams = lambda **kwargs: kwargs
    worker.llm = llm
    return worker


def test_generate_async_cancellation_stops_child_request_task() -> None:
    async def exercise() -> None:
        llm = _BlockingLLM()
        worker = _make_async_worker(llm)
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "input_lengths": torch.tensor([2], dtype=torch.long),
            }
        )
        generation = worker.generate_async(data, validation=True)
        next_result = asyncio.create_task(anext(generation))

        await asyncio.wait_for(llm.started.wait(), timeout=0.5)
        assert llm.sampling_params["extra_args"] == {"nemo_rl": {"validation": True}}
        next_result.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_result

        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.1)

    asyncio.run(exercise())


def test_generate_text_async_cancellation_stops_child_request_task() -> None:
    async def exercise() -> None:
        llm = _BlockingLLM()
        worker = _make_async_worker(llm)
        worker.cfg["ignore_eos"] = True
        data = BatchedDataDict({"prompts": ["hello"]})
        generation = worker.generate_text_async(data)
        next_result = asyncio.create_task(anext(generation))

        await asyncio.wait_for(llm.started.wait(), timeout=0.5)
        assert llm.sampling_params["ignore_eos"] is True
        assert llm.sampling_params["logprobs"] is None
        next_result.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_result

        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.1)

    asyncio.run(exercise())


def test_generate_async_marks_output_context_exhaustion_as_truncated() -> None:
    async def exercise() -> None:
        llm = _BlockingLLM()
        worker = _make_async_worker(llm)
        worker.cfg["_output_max_model_len"] = 2
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "input_lengths": torch.tensor([2], dtype=torch.long),
            }
        )

        results = [result async for result in worker.generate_async(data)]

        assert len(results) == 1
        sample_idx, result = results[0]
        assert sample_idx == 0
        assert result["generation_lengths"].tolist() == [0]
        assert result["truncated"].tolist() == [True]
        assert llm.sampling_params is None

    asyncio.run(exercise())


def test_shutdown_stops_vllm_metrics_logger(monkeypatch) -> None:
    calls = 0

    def get_metrics_snapshot():
        nonlocal calls
        calls += 1
        return []

    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    v1_module = ModuleType("vllm.v1")
    v1_module.__path__ = []
    metrics_module = ModuleType("vllm.v1.metrics")
    metrics_module.__path__ = []
    reader_module = ModuleType("vllm.v1.metrics.reader")
    reader_module.Gauge = type("Gauge", (), {})
    reader_module.Counter = type("Counter", (), {})
    reader_module.get_metrics_snapshot = get_metrics_snapshot
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.metrics", metrics_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.metrics.reader", reader_module)

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "vllm_cfg": {
            "async_engine": True,
            "enable_vllm_metrics_logger": True,
            "vllm_metrics_logger_interval": 0.01,
        }
    }
    worker.is_model_owner = True
    worker._vllm_metrics_lock = threading.Lock()
    worker.llm = None
    worker.tokenizer = None
    worker.server_thread = None
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    worker._start_vllm_metrics_logger()
    try:
        deadline = time.monotonic() + 0.5
        while calls == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert calls > 0

        assert asyncio.run(worker.shutdown()) is True
        assert worker._vllm_metrics_logger_stop_event.is_set()
        assert not worker._vllm_metrics_logger_thread.is_alive()
    finally:
        worker._vllm_metrics_logger_stop_event.set()
        worker._vllm_metrics_logger_thread.join(timeout=0.2)
