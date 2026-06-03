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

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_rl.models.generation.vllm.vllm_worker_async import VllmAsyncGenerationWorker


def _async_worker_class():
    metadata = getattr(VllmAsyncGenerationWorker, "__ray_metadata__", None)
    return getattr(metadata, "modified_class", VllmAsyncGenerationWorker)


def _make_worker():
    worker = _async_worker_class().__new__(_async_worker_class())
    worker.cfg = {"vllm_cfg": {"async_engine": True}}
    worker.llm = MagicMock()
    worker.llm.collective_rpc = AsyncMock(return_value=[(True, None)])
    worker.llm.reset_mm_cache = AsyncMock()
    worker.llm.wake_up = AsyncMock()
    return worker


@pytest.mark.parametrize(
    "method_name",
    [
        "update_weights_from_collective_async",
        "update_weights_via_ipc_zmq_async",
        "wake_up_async",
    ],
)
@pytest.mark.asyncio
async def test_reset_mm_cache_called_at_all_refit_boundaries(method_name):
    worker = _make_worker()

    result = await getattr(worker, method_name)()

    if method_name != "wake_up_async":
        assert result is True
    worker.llm.reset_mm_cache.assert_awaited_once_with()


def test_async_vllm_kwargs_forwarded_to_asyncllm(monkeypatch):
    captured = {}

    class FakeCompilationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["engine_kwargs"] = kwargs

    class FakeAsyncLLM:
        @staticmethod
        def from_engine_args(engine_args, stat_loggers):
            captured["engine_args"] = engine_args
            captured["stat_loggers"] = stat_loggers
            return "fake-llm"

    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.engine", types.ModuleType("vllm.engine"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(
        sys.modules, "vllm.v1.engine", types.ModuleType("vllm.v1.engine")
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.metrics", types.ModuleType("vllm.v1.metrics")
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        types.SimpleNamespace(CompilationConfig=FakeCompilationConfig),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.engine.arg_utils",
        types.SimpleNamespace(AsyncEngineArgs=FakeAsyncEngineArgs),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.async_llm",
        types.SimpleNamespace(AsyncLLM=FakeAsyncLLM),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.metrics.loggers",
        types.SimpleNamespace(PrometheusStatLogger=object),
    )

    worker = _async_worker_class().__new__(_async_worker_class())
    worker.cfg = {"vllm_cfg": {"enable_vllm_metrics_logger": False}}

    worker._create_engine(
        {
            "model": "test-model",
            "limit_mm_per_prompt": {"image": 3},
            "skip_mm_profiling": True,
            "mm_processor_cache_gb": 0,
            "logprobs_mode": "raw_logprobs",
        }
    )

    assert worker.llm == "fake-llm"
    assert captured["engine_kwargs"]["limit_mm_per_prompt"] == {"image": 3}
    assert captured["engine_kwargs"]["skip_mm_profiling"] is True
    assert captured["engine_kwargs"]["mm_processor_cache_gb"] == 0
    assert captured["engine_kwargs"]["logprobs_mode"] == "raw_logprobs"
