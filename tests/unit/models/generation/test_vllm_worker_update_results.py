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

import asyncio
from typing import Any
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


@pytest.fixture(autouse=True)
def _disable_nvtx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


class _SyncCollectiveRpc:
    def __init__(self, worker_results: list[bool | None]) -> None:
        self.worker_results = worker_results
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def collective_rpc(self, method: str, args: tuple[Any, ...]) -> list[Any]:
        self.calls.append((method, args))
        if method == "report_device_id":
            return ["device-0", "device-1"]
        return self.worker_results


class _AsyncCollectiveRpc:
    def __init__(
        self,
        worker_results: list[bool | None],
        *,
        return_nested_awaitable: bool = False,
    ) -> None:
        self.worker_results = worker_results
        self.return_nested_awaitable = return_nested_awaitable
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def collective_rpc(self, method: str, args: tuple[Any, ...]) -> Any:
        self.calls.append((method, args))
        if method == "report_device_id":
            return ["device-0", "device-1"]
        if not self.return_nested_awaitable:
            return self.worker_results

        async def resolve_worker_results() -> list[bool | None]:
            return self.worker_results

        return resolve_worker_results()


class _SyncLifecycleLLM:
    def __init__(self) -> None:
        self.draft_weight = object()
        self.sleep_levels: list[int] = []
        self.llm_engine = SimpleNamespace(reset_prefix_cache=lambda: None)

    def sleep(self, *, level: int) -> None:
        self.sleep_levels.append(level)
        if level != 1:
            self.draft_weight = None

    def wake_up(self, **_kwargs: Any) -> None:
        return None


class _AsyncLifecycleLLM:
    def __init__(self) -> None:
        self.draft_weight = object()
        self.sleep_levels: list[int] = []

    async def reset_prefix_cache(self) -> None:
        return None

    async def sleep(self, *, level: int) -> None:
        self.sleep_levels.append(level)
        if level != 1:
            self.draft_weight = None

    async def wake_up(self, **_kwargs: Any) -> None:
        return None


def test_sync_prefix_reset_forwards_running_request_preemption() -> None:
    calls: list[bool] = []

    class Engine:
        def reset_prefix_cache(self, *, reset_running_requests: bool) -> bool:
            calls.append(reset_running_requests)
            return False

    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.llm = SimpleNamespace(llm_engine=Engine())
    worker.cfg = {"vllm_cfg": {"async_engine": False}}

    result = worker.reset_prefix_cache(reset_running_requests=True)

    assert result is False
    assert calls == [True]


def test_async_prefix_reset_forwards_running_request_preemption() -> None:
    async def exercise() -> None:
        calls: list[bool] = []

        class Engine:
            async def reset_prefix_cache(self, *, reset_running_requests: bool) -> bool:
                calls.append(reset_running_requests)
                return False

        worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
        worker.llm = Engine()
        worker.cfg = {"vllm_cfg": {"async_engine": True}}

        result = await worker.reset_prefix_cache_async(reset_running_requests=True)

        assert result is False
        assert calls == [True]

    asyncio.run(exercise())


def test_generation_cache_invalidation_preempts_running_requests(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class WorkerGroup:
        def run_all_workers_single_data(self, method_name: str, **kwargs: Any):
            calls.append((method_name, kwargs))
            return [object()]

    generation = VllmGeneration.__new__(VllmGeneration)
    generation.cfg = {"vllm_cfg": {"async_engine": True}}
    generation.worker_group = WorkerGroup()
    monkeypatch.setattr("ray.get", lambda _futures: [True])

    assert generation.invalidate_kv_cache() is True
    assert calls == [
        (
            "reset_prefix_cache_async",
            {
                "run_rank_0_only_axes": ["tensor_parallel", "pipeline_parallel"],
                "reset_running_requests": True,
            },
        )
    ]


@pytest.mark.parametrize(
    ("worker_results", "expected"),
    [
        ([], False),
        ([None], False),
        ([True, None], True),
        ([False, None], False),
    ],
)
def test_generation_cache_invalidation_requires_a_successful_owner_result(
    monkeypatch: pytest.MonkeyPatch,
    worker_results: list[bool | None],
    expected: bool,
) -> None:
    generation = VllmGeneration.__new__(VllmGeneration)
    generation.cfg = {"vllm_cfg": {"async_engine": True}}
    generation.worker_group = SimpleNamespace(
        run_all_workers_single_data=lambda *_args, **_kwargs: [object()]
    )
    monkeypatch.setattr("ray.get", lambda _futures: worker_results)

    assert generation.invalidate_kv_cache() is expected


def test_finish_generation_does_not_preempt_running_requests(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class WorkerGroup:
        def run_all_workers_single_data(self, method_name: str, **kwargs: Any):
            calls.append((method_name, kwargs))
            return [object()]

    generation = VllmGeneration.__new__(VllmGeneration)
    generation.cfg = {
        "colocated": {"enabled": False},
        "vllm_cfg": {"async_engine": True},
    }
    generation.worker_group = WorkerGroup()
    monkeypatch.setattr("ray.get", lambda _futures: [True])

    assert generation.finish_generation() is True
    assert calls == [
        (
            "reset_prefix_cache_async",
            {"run_rank_0_only_axes": ["tensor_parallel", "pipeline_parallel"]},
        )
    ]


def _make_sync_worker(
    worker_results: list[bool | None], *, load_mtp_from_disk: bool = False
) -> VllmGenerationWorkerImpl:
    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.llm = _SyncCollectiveRpc(worker_results)
    worker.cfg = {"vllm_cfg": {"async_engine": False}}
    worker._mtp_load_from_disk = load_mtp_from_disk
    worker._refit_failure_reason = None
    worker.model_name = "test-model"
    return worker


def _make_async_worker(
    worker_results: list[bool | None],
    *,
    load_mtp_from_disk: bool = False,
    return_nested_awaitable: bool = False,
) -> VllmAsyncGenerationWorkerImpl:
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.llm = _AsyncCollectiveRpc(
        worker_results, return_nested_awaitable=return_nested_awaitable
    )
    worker.cfg = {"vllm_cfg": {"async_engine": True}}
    worker._mtp_load_from_disk = load_mtp_from_disk
    worker._refit_failure_reason = None
    worker.model_name = "test-model"
    return worker


def _empty_generation_batch() -> Any:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    return BatchedDataDict(
        {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "input_lengths": torch.empty(0, dtype=torch.long),
        }
    )


@pytest.mark.parametrize(
    "method_name",
    ["update_weights_via_ipc_zmq", "update_weights_from_collective"],
)
@pytest.mark.parametrize(
    ("worker_results", "expected"),
    [
        ([True, True], True),
        ([True, False], False),
        ([False, True], False),
        ([False], False),
        ([], False),
    ],
)
def test_sync_weight_update_requires_every_worker_to_succeed(
    method_name: str, worker_results: list[bool], expected: bool
) -> None:
    worker = _make_sync_worker(worker_results)

    assert getattr(worker, method_name)() is expected


@pytest.mark.parametrize(
    "method_name",
    [
        "update_weights_via_ipc_zmq_async",
        "update_weights_from_collective_async",
    ],
)
@pytest.mark.parametrize("return_nested_awaitable", [False, True])
@pytest.mark.parametrize(
    ("worker_results", "expected"),
    [
        ([True, True], True),
        ([True, False], False),
        ([False, True], False),
        ([False], False),
        ([], False),
    ],
)
def test_async_weight_update_requires_every_worker_to_succeed(
    method_name: str,
    return_nested_awaitable: bool,
    worker_results: list[bool],
    expected: bool,
) -> None:
    worker = _make_async_worker(
        worker_results, return_nested_awaitable=return_nested_awaitable
    )

    assert asyncio.run(getattr(worker, method_name)()) is expected


@pytest.mark.parametrize(
    "method_name",
    ["update_weights_via_ipc_zmq", "update_weights_from_collective"],
)
def test_sync_failed_weight_update_prevents_worker_reuse(method_name: str) -> None:
    worker = _make_sync_worker([True, False])

    assert getattr(worker, method_name)() is False
    with pytest.raises(RuntimeError, match="weight refit failed.*restart"):
        worker.generate(_empty_generation_batch())


@pytest.mark.parametrize(
    "method_name",
    [
        "update_weights_via_ipc_zmq_async",
        "update_weights_from_collective_async",
    ],
)
def test_async_failed_weight_update_prevents_worker_reuse(method_name: str) -> None:
    worker = _make_async_worker([True, False])

    assert asyncio.run(getattr(worker, method_name)()) is False

    async def consume_generation() -> None:
        async for _ in worker.generate_async(_empty_generation_batch()):
            pass

    with pytest.raises(RuntimeError, match="weight refit failed.*restart"):
        asyncio.run(consume_generation())


@pytest.mark.parametrize("worker_results", [[None, True, False], [None, None], []])
def test_sync_mtp_startup_rejects_incomplete_loads(
    worker_results: list[bool | None],
) -> None:
    worker = _make_sync_worker(worker_results, load_mtp_from_disk=True)

    with pytest.raises(RuntimeError, match="MTP draft weight loading failed"):
        worker.post_init()


def test_sync_mtp_startup_accepts_complete_load() -> None:
    worker = _make_sync_worker([None, None, True, True], load_mtp_from_disk=True)

    worker.post_init()

    assert worker.vllm_device_ids == ["device-0", "device-1"]


@pytest.mark.parametrize("worker_results", [[None, True, False], [None, None], []])
def test_async_mtp_startup_rejects_incomplete_loads(
    worker_results: list[bool | None],
) -> None:
    worker = _make_async_worker(worker_results, load_mtp_from_disk=True)

    with pytest.raises(RuntimeError, match="MTP draft weight loading failed"):
        asyncio.run(worker.post_init_async())


def test_async_mtp_startup_accepts_complete_load() -> None:
    worker = _make_async_worker([None, None, True, True], load_mtp_from_disk=True)

    asyncio.run(worker.post_init_async())

    assert worker.vllm_device_ids == ["device-0", "device-1"]


def test_sync_prepare_refit_forwards_mtp_draft_requirement() -> None:
    worker = _make_sync_worker([True])
    worker.cfg["_mtp_weights_from_refit"] = True
    state_dict_info = {"model.weight": (torch.Size([1]), torch.float32)}

    worker.prepare_refit_info(state_dict_info)

    assert worker.llm.calls[-1] == (
        "prepare_refit_info",
        (state_dict_info, True),
    )


def test_async_prepare_refit_forwards_mtp_draft_requirement() -> None:
    worker = _make_async_worker([True])
    worker.cfg["_mtp_weights_from_refit"] = True
    state_dict_info = {"model.weight": (torch.Size([1]), torch.float32)}

    asyncio.run(worker.prepare_refit_info_async(state_dict_info))

    assert worker.llm.calls[-1] == (
        "prepare_refit_info",
        (state_dict_info, True),
    )


def test_sync_sleep_wake_uses_level_one_and_preserves_drafter() -> None:
    worker = _make_sync_worker([True])
    llm = _SyncLifecycleLLM()
    worker.llm = llm
    original_draft_weight = llm.draft_weight

    worker.sleep()
    worker.wake_up()

    assert llm.sleep_levels == [1]
    assert llm.draft_weight is original_draft_weight


def test_async_sleep_wake_uses_level_one_and_preserves_drafter() -> None:
    worker = _make_async_worker([True])
    llm = _AsyncLifecycleLLM()
    worker.llm = llm
    original_draft_weight = llm.draft_weight

    asyncio.run(worker.sleep_async())
    asyncio.run(worker.wake_up_async())

    assert llm.sleep_levels == [1]
    assert llm.draft_weight is original_draft_weight
