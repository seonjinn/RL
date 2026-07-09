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


def _make_sync_worker(
    worker_results: list[bool | None], *, load_mtp_from_disk: bool = False
) -> VllmGenerationWorkerImpl:
    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.llm = _SyncCollectiveRpc(worker_results)
    worker.cfg = {"vllm_cfg": {"async_engine": False}}
    worker._mtp_load_from_disk = load_mtp_from_disk
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
    worker.model_name = "test-model"
    return worker


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
