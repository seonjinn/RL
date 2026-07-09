# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import asyncio
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
        generation = worker.generate_async(data)
        next_result = asyncio.create_task(anext(generation))

        await asyncio.wait_for(llm.started.wait(), timeout=0.5)
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
