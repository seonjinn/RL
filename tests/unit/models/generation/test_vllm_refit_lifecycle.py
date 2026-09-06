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

"""Gym-independent vLLM refit lifecycle and acknowledgement contracts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.interfaces import (
    GenerationLifecycleNotDispatchedError,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


@pytest.mark.parametrize("version", [True, -1, 1.0, "1", None])
def test_worker_rejects_non_exact_or_negative_weight_versions(version):
    worker = SimpleNamespace(_rollout_weight_version=9)

    with pytest.raises(ValueError, match="exact nonnegative int"):
        asyncio.run(
            VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, version)
        )

    assert worker._rollout_weight_version == 9


class _RemoteMethod:
    def __init__(self, result=True, *, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class _Leader:
    def __init__(self, index: int) -> None:
        self.index = index
        self.setup_token_capture = _RemoteMethod()
        self.set_rollout_weight_version = _RemoteMethod()
        self.reset_prefix_cache_async = _RemoteMethod()
        self.pause_generation_async = _RemoteMethod()
        self.resume_generation_async = _RemoteMethod()
        self.update_weights_from_collective_async = _RemoteMethod()
        self.nccl_reshard_refit_async = _RemoteMethod()


def _generation_with_mock_group(*, async_engine: bool = True) -> VllmGeneration:
    generation = object.__new__(VllmGeneration)
    generation.cfg = {"vllm_cfg": {"async_engine": async_engine}}
    leaders = [_Leader(0), _Leader(1)]
    generation.worker_group = SimpleNamespace(
        workers=leaders,
        dp_size=2,
        shutdown=lambda **_: True,
    )
    generation.dp_size = 2
    generation._refit_membership = None
    generation._refit_commit_poison = None
    generation.weight_synchronizer = None
    return generation


def test_generation_setup_token_capture_fans_out(monkeypatch):
    generation = _generation_with_mock_group()
    ray_timeouts = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: ray_timeouts.append(timeout) or futures,
    )
    assert (
        generation.setup_token_capture({"backend": "simple"}, "rollout_staging") is True
    )
    assert ray_timeouts == [300.0]
    for leader in generation.worker_group.workers:
        assert leader.setup_token_capture.calls == [
            {
                "dp_cfg": {"backend": "simple"},
                "staging_partition": "rollout_staging",
            }
        ]


def test_generation_setup_token_capture_requires_async_engine():
    generation = _generation_with_mock_group(async_engine=False)
    with pytest.raises(AssertionError, match="async vLLM engine"):
        generation.setup_token_capture({}, "rollout_staging")


def test_generation_set_rollout_weight_version_fans_out(monkeypatch):
    generation = _generation_with_mock_group()
    ray_timeouts = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: ray_timeouts.append(timeout) or futures,
    )
    assert generation.set_rollout_weight_version(7, refit_timeout_s=17.5) is True
    assert ray_timeouts == [17.5]
    for leader in generation.worker_group.workers:
        assert leader.set_rollout_weight_version.calls == [{"version": 7}]


def test_generation_stamps_only_surviving_refit_leaders(monkeypatch):
    generation = _generation_with_mock_group()
    dead = generation.worker_group.workers[1]
    survivor = _Leader(2)
    generation.worker_group.workers.append(survivor)
    generation.dp_size = 3
    generation.worker_group.dp_size = 3
    generation._refit_membership = SimpleNamespace(
        shard_prefixes={0: 0, 2: 1}, workers_per_shard=1
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: futures,
    )

    assert generation.set_rollout_weight_version(8) is True

    assert generation.worker_group.workers[0].set_rollout_weight_version.calls == [
        {"version": 8}
    ]
    assert dead.set_rollout_weight_version.calls == []
    assert survivor.set_rollout_weight_version.calls == [{"version": 8}]


@pytest.mark.parametrize("result", [[True], [True, False], [True, None], []])
def test_generation_stamp_requires_exact_true_ack_from_every_survivor(
    monkeypatch, result
):
    generation = _generation_with_mock_group()
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: result,
    )

    with pytest.raises(RuntimeError, match="acknowledgement|worker"):
        generation.set_rollout_weight_version(7)


def test_unknown_stamp_outcome_poison_prevents_retry_and_preserves_cause(monkeypatch):
    generation = _generation_with_mock_group()
    first_failure = TimeoutError("stamp RPC timed out")
    ray_calls = 0

    def _fail(_futures, *, timeout):
        nonlocal ray_calls
        del timeout
        ray_calls += 1
        raise first_failure

    monkeypatch.setattr("nemo_rl.models.generation.vllm.vllm_generation.ray.get", _fail)

    with pytest.raises(TimeoutError) as first:
        generation.set_rollout_weight_version(7)
    assert first.value is first_failure

    with pytest.raises(RuntimeError, match="poisoned") as retry:
        generation.set_rollout_weight_version(7)

    assert retry.value.__cause__ is first_failure
    assert ray_calls == 1
    for leader in generation.worker_group.workers:
        assert leader.set_rollout_weight_version.calls == [{"version": 7}]


def test_first_remote_failure_is_typed_not_dispatched_and_retry_safe(monkeypatch):
    generation = _generation_with_mock_group()
    first_failure = RuntimeError("first leader submission failed")
    first_method = _RemoteMethod(error=first_failure)
    generation.worker_group.workers[0].set_rollout_weight_version = first_method
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: futures,
    )

    with pytest.raises(GenerationLifecycleNotDispatchedError) as first:
        generation.set_rollout_weight_version(7)

    assert first.value.__cause__ is first_failure
    assert generation._refit_commit_poison is None
    assert first_method.calls == [{"version": 7}]
    assert generation.worker_group.workers[1].set_rollout_weight_version.calls == []

    first_method.error = None
    assert generation.set_rollout_weight_version(7) is True
    assert first_method.calls == [{"version": 7}, {"version": 7}]
    assert generation.worker_group.workers[1].set_rollout_weight_version.calls == [
        {"version": 7}
    ]


def test_later_remote_failure_poison_preserves_partial_dispatch_cause():
    generation = _generation_with_mock_group()
    partial_failure = RuntimeError("second leader submission failed")
    second_method = _RemoteMethod(error=partial_failure)
    generation.worker_group.workers[1].set_rollout_weight_version = second_method

    with pytest.raises(RuntimeError) as first:
        generation.set_rollout_weight_version(7)

    assert first.value is partial_failure
    assert generation._refit_commit_poison is partial_failure
    assert generation.worker_group.workers[0].set_rollout_weight_version.calls == [
        {"version": 7}
    ]
    assert second_method.calls == [{"version": 7}]

    with pytest.raises(RuntimeError, match="poisoned") as retry:
        generation.set_rollout_weight_version(7)
    assert retry.value.__cause__ is partial_failure
    assert generation.worker_group.workers[0].set_rollout_weight_version.calls == [
        {"version": 7}
    ]
    assert second_method.calls == [{"version": 7}]


def test_poisoned_stamp_rejects_a_later_refit_before_dispatch(monkeypatch):
    generation = _generation_with_mock_group()
    first_failure = TimeoutError("stamp RPC timed out")
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: (_ for _ in ()).throw(first_failure),
    )
    with pytest.raises(TimeoutError):
        generation.set_rollout_weight_version(7)

    with pytest.raises(RuntimeError, match="poisoned") as caught:
        generation.update_weights_from_collective(refit_timeout_s=2.0)

    assert caught.value.__cause__ is first_failure
    for leader in generation.worker_group.workers:
        assert leader.update_weights_from_collective_async.calls == []


def test_cache_invalidation_is_bounded_exact_and_poisoning(monkeypatch):
    generation = _generation_with_mock_group()
    malformed = [True, None]
    ray_timeouts = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: ray_timeouts.append(timeout) or malformed,
    )

    with pytest.raises(RuntimeError, match="expected exactly True") as first:
        generation.invalidate_kv_cache(refit_timeout_s=6.25)

    assert ray_timeouts == [6.25]
    with pytest.raises(RuntimeError, match="poisoned") as retry:
        generation.set_rollout_weight_version(9)
    assert retry.value.__cause__ is first.value


def test_refit_pause_resume_are_bounded_and_use_only_survivor_leaders(monkeypatch):
    generation = _generation_with_mock_group()
    assert generation.supports_refit_pause_resume is True
    assert generation.supports_refit_pause_resume_timeout is True
    dead = generation.worker_group.workers[1]
    survivor = _Leader(2)
    generation.worker_group.workers.append(survivor)
    generation.dp_size = 3
    generation.worker_group.dp_size = 3
    generation._refit_membership = SimpleNamespace(
        shard_prefixes={0: 0, 2: 1}, workers_per_shard=1
    )
    ray_timeouts = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: ray_timeouts.append(timeout) or futures,
    )

    assert (
        generation.pause_generation_for_refit(clear_cache=True, refit_timeout_s=4.5)
        is True
    )
    assert generation.resume_generation_after_refit(refit_timeout_s=3.25) is True

    assert ray_timeouts == [4.5, 3.25]
    assert generation.worker_group.workers[0].pause_generation_async.calls == [
        {"clear_cache": True}
    ]
    assert survivor.pause_generation_async.calls == [{"clear_cache": True}]
    assert dead.pause_generation_async.calls == []
    assert generation.worker_group.workers[0].resume_generation_async.calls == [{}]
    assert survivor.resume_generation_async.calls == [{}]
    assert dead.resume_generation_async.calls == []


def test_refit_pause_requires_exact_ack_and_poison_blocks_resume(monkeypatch):
    generation = _generation_with_mock_group()
    malformed = [True, 1]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: malformed,
    )

    with pytest.raises(RuntimeError, match="expected exactly True") as first:
        generation.pause_generation_for_refit(clear_cache=True, refit_timeout_s=2.0)

    with pytest.raises(RuntimeError, match="poisoned") as retry:
        generation.resume_generation_after_refit(refit_timeout_s=2.0)
    assert retry.value.__cause__ is first.value
    for leader in generation.worker_group.workers:
        assert leader.resume_generation_async.calls == []
