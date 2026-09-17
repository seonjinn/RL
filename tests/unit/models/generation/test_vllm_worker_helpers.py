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

"""Tests for vLLM worker helper functions."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.vllm_worker import (
    VllmGenerationWorkerImpl,
    _refit_sleep_level,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.models.generation.vllm.worker_utils import (
    find_tokenizer_required_architectures,
    resolve_data_parallel_local_rank,
    resolve_distributed_executor_backend,
)


def _refit_test_config(
    mode: str | None = None,
    *,
    speculative_config: dict | None = None,
) -> VllmConfig:
    config: dict = {
        "vllm_cfg": {"async_engine": False},
        "vllm_kwargs": {},
    }
    if mode is not None:
        config["refit_cfg"] = {"memory_lifecycle": {"mode": mode}}
    if speculative_config is not None:
        config["vllm_kwargs"]["speculative_config"] = speculative_config
    return cast(VllmConfig, config)


def _sleep_test_worker(
    *, uses_specdec_deep_refit: bool
) -> tuple[VllmGenerationWorkerImpl, MagicMock]:
    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.cfg = _refit_test_config(
        "specdec_deep_refit" if uses_specdec_deep_refit else None
    )
    worker.uses_specdec_deep_refit = uses_specdec_deep_refit  # type: ignore[attr-defined]
    fake_llm = MagicMock()
    worker.llm = fake_llm
    return worker, fake_llm


def _post_init_test_worker(
    *,
    mode: str | None,
    speculative_config: dict | None,
    draft_weights_from_refit: bool = False,
    mtp_weights_from_refit: bool = False,
    drafter_rpc_results: list[bool] | None = None,
) -> tuple[VllmGenerationWorkerImpl, MagicMock]:
    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.cfg = _refit_test_config(
        mode,
        speculative_config=speculative_config,
    )
    worker.uses_specdec_deep_refit = mode == "specdec_deep_refit"
    worker._draft_weights_from_refit = draft_weights_from_refit
    worker._mtp_weights_from_refit = mtp_weights_from_refit
    worker._mtp_speculative_enabled = False
    worker._mtp_load_from_disk = False
    worker._sparse_refit_receiver = None
    worker.model_name = "target"
    worker.report_device_id = MagicMock(return_value=[0])
    worker._record_deep_refit_memory = MagicMock()
    fake_llm = MagicMock()

    def collective_rpc(method: str, *, args: tuple):
        del args
        if method in ("snapshot_static_drafter", "restore_static_drafter"):
            return drafter_rpc_results if drafter_rpc_results is not None else [True]
        return [None]

    fake_llm.collective_rpc.side_effect = collective_rpc
    worker.llm = fake_llm
    return worker, fake_llm


def _async_test_worker(
    *,
    mode: str | None,
    speculative_config: dict | None,
    draft_weights_from_refit: bool = False,
    mtp_weights_from_refit: bool = False,
    drafter_rpc_results: list[bool] | None = None,
) -> tuple[VllmAsyncGenerationWorkerImpl, MagicMock]:
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = _refit_test_config(
        mode,
        speculative_config=speculative_config,
    )
    worker.cfg["vllm_cfg"]["async_engine"] = True
    worker.uses_specdec_deep_refit = mode == "specdec_deep_refit"
    worker._draft_weights_from_refit = draft_weights_from_refit
    worker._mtp_weights_from_refit = mtp_weights_from_refit
    worker._mtp_speculative_enabled = False
    worker._mtp_load_from_disk = False
    worker._sparse_refit_receiver = None
    worker._http_engine_client = None
    worker.model_name = "target"
    worker.report_device_id_async = AsyncMock(return_value=[0])
    worker._record_deep_refit_memory = MagicMock()
    fake_llm = MagicMock(
        spec=["collective_rpc", "reset_prefix_cache", "sleep", "wake_up"]
    )

    async def collective_rpc(method: str, *, args: tuple):
        del args
        if method in ("snapshot_static_drafter", "restore_static_drafter"):
            return drafter_rpc_results if drafter_rpc_results is not None else [True]
        return [None]

    fake_llm.collective_rpc = AsyncMock(side_effect=collective_rpc)
    fake_llm.reset_prefix_cache = AsyncMock()
    fake_llm.sleep = AsyncMock()
    fake_llm.wake_up = AsyncMock()
    worker.llm = fake_llm
    return worker, fake_llm


def test_refit_sleep_level_defaults_to_level_one() -> None:
    assert _refit_sleep_level(_refit_test_config()) == 1


def test_refit_sleep_level_selects_level_two_only_when_explicit() -> None:
    assert _refit_sleep_level(_refit_test_config("specdec_deep_refit")) == 2


def test_refit_sleep_legacy_worker_remains_level_one_without_rpc() -> None:
    worker, fake_llm = _sleep_test_worker(uses_specdec_deep_refit=False)
    worker._record_deep_refit_memory = MagicMock()

    worker.sleep()

    fake_llm.sleep.assert_called_once_with(level=1)
    fake_llm.collective_rpc.assert_not_called()
    worker._record_deep_refit_memory.assert_not_called()


def test_refit_sleep_deep_worker_selects_level_two() -> None:
    worker, fake_llm = _sleep_test_worker(uses_specdec_deep_refit=True)
    worker._record_deep_refit_memory = MagicMock()

    worker.sleep()

    fake_llm.sleep.assert_called_once_with(level=2)
    assert worker._record_deep_refit_memory.call_args_list == [
        call("before_sleep_level2"),
        call("after_sleep_level2"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_level"),
    [(None, 1), ("specdec_deep_refit", 2)],
)
async def test_async_refit_sleep_matches_selected_lifecycle(
    mode: str | None,
    expected_level: int,
) -> None:
    worker, fake_llm = _async_test_worker(
        mode=mode,
        speculative_config={"method": "dflash"} if mode else None,
    )

    await worker.sleep_async()

    fake_llm.sleep.assert_awaited_once_with(level=expected_level)
    if mode is None:
        worker._record_deep_refit_memory.assert_not_called()
    else:
        assert worker._record_deep_refit_memory.call_args_list == [
            call("before_sleep_level2"),
            call("after_sleep_level2"),
        ]


@pytest.mark.asyncio
async def test_async_deep_refit_static_drafter_snapshots_and_restores() -> None:
    worker, fake_llm = _async_test_worker(
        mode="specdec_deep_refit",
        speculative_config={"method": "dspark"},
        drafter_rpc_results=[False, True],
    )

    await worker.post_init_async()

    assert await worker.restore_drafter_after_refit_async() is True
    assert call("snapshot_static_drafter", args=tuple()) in (
        fake_llm.collective_rpc.await_args_list
    )
    assert call("restore_static_drafter", args=tuple()) in (
        fake_llm.collective_rpc.await_args_list
    )


@pytest.mark.asyncio
async def test_async_legacy_refit_avoids_drafter_rpcs() -> None:
    worker, fake_llm = _async_test_worker(
        mode=None,
        speculative_config={"method": "dflash"},
    )

    await worker.post_init_async()
    await worker.sleep_async()

    assert await worker.restore_drafter_after_refit_async() is True
    draft_methods = {
        args.args[0] for args in fake_llm.collective_rpc.await_args_list if args.args
    }
    assert "snapshot_static_drafter" not in draft_methods
    assert "restore_static_drafter" not in draft_methods


def test_deep_refit_requires_speculative_config() -> None:
    worker, _ = _post_init_test_worker(
        mode="specdec_deep_refit",
        speculative_config=None,
    )

    with pytest.raises(ValueError, match="speculative_config"):
        worker.post_init()


def test_deep_refit_static_drafter_requires_owning_snapshot() -> None:
    worker, fake_llm = _post_init_test_worker(
        mode="specdec_deep_refit",
        speculative_config={"method": "dflash"},
        drafter_rpc_results=[False, False],
    )

    with pytest.raises(RuntimeError, match="owning.*snapshot"):
        worker.post_init()

    assert call("snapshot_static_drafter", args=tuple()) in (
        fake_llm.collective_rpc.call_args_list
    )


def test_deep_refit_static_drafter_snapshots_and_restores() -> None:
    worker, fake_llm = _post_init_test_worker(
        mode="specdec_deep_refit",
        speculative_config={"method": "dspark"},
        drafter_rpc_results=[False, True],
    )

    worker.post_init()

    assert worker.restore_drafter_after_refit() is True
    assert call("snapshot_static_drafter", args=tuple()) in (
        fake_llm.collective_rpc.call_args_list
    )
    assert call("restore_static_drafter", args=tuple()) in (
        fake_llm.collective_rpc.call_args_list
    )


@pytest.mark.parametrize(
    ("draft_weights_from_refit", "mtp_weights_from_refit"),
    [(True, False), (False, True)],
)
def test_deep_refit_streamed_drafter_does_not_snapshot_or_restore(
    draft_weights_from_refit: bool,
    mtp_weights_from_refit: bool,
) -> None:
    worker, fake_llm = _post_init_test_worker(
        mode="specdec_deep_refit",
        speculative_config={"method": "dflash"},
        draft_weights_from_refit=draft_weights_from_refit,
        mtp_weights_from_refit=mtp_weights_from_refit,
    )

    worker.post_init()

    assert worker.restore_drafter_after_refit() is True
    draft_methods = {
        args.args[0] for args in fake_llm.collective_rpc.call_args_list if args.args
    }
    assert "snapshot_static_drafter" not in draft_methods
    assert "restore_static_drafter" not in draft_methods


@pytest.mark.parametrize(
    "speculative_config",
    [None, {"method": "dflash"}],
)
def test_legacy_refit_keeps_post_init_sleep_and_restore_free_of_draft_rpcs(
    speculative_config: dict | None,
) -> None:
    worker, fake_llm = _post_init_test_worker(
        mode=None,
        speculative_config=speculative_config,
    )

    worker.post_init()
    worker.sleep()

    assert worker.restore_drafter_after_refit() is True
    draft_methods = {
        args.args[0] for args in fake_llm.collective_rpc.call_args_list if args.args
    }
    assert "snapshot_static_drafter" not in draft_methods
    assert "restore_static_drafter" not in draft_methods


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (None, []),
        ([], []),
        (["Gemma4ForCausalLM"], []),
        (
            ["Gemma4ForConditionalGeneration"],
            ["Gemma4ForConditionalGeneration"],
        ),
        (
            [
                "Gemma4ForCausalLM",
                "Gemma4UnifiedForConditionalGeneration",
                "Mistral3ForConditionalGeneration",
            ],
            [
                "Gemma4UnifiedForConditionalGeneration",
                "Mistral3ForConditionalGeneration",
            ],
        ),
    ],
)
def test_find_tokenizer_required_architectures(architectures, expected):
    assert find_tokenizer_required_architectures(architectures) == expected


@pytest.mark.parametrize(
    ("tp", "pp", "ep", "expected"),
    [
        (2, 1, 2, "ray"),
        (1, 2, 2, "ray"),
        (1, 1, 8, "uni"),
        (1, 1, 1, None),
    ],
)
def test_resolve_distributed_executor_backend(tp, pp, ep, expected):
    assert resolve_distributed_executor_backend(tp, pp, ep) == expected


@pytest.mark.parametrize(
    ("rank", "model_parallel_size", "executor_backend", "expected"),
    [
        (7, 1, "uni", 0),
        (6, 2, "ray", 3),
    ],
)
def test_resolve_data_parallel_local_rank(
    rank, model_parallel_size, executor_backend, expected
):
    assert (
        resolve_data_parallel_local_rank(rank, model_parallel_size, executor_backend)
        == expected
    )
