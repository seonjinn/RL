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

import threading
import time
from typing import Any
from unittest import mock

import pytest

from nemo_rl.algorithms.async_utils import trajectory_collector as collector_module
from nemo_rl.algorithms.async_utils.trajectory_collector import (
    AsyncTrajectoryCollector,
)
from nemo_rl.algorithms.grpo import MasterConfig


def _local_collector() -> Any:
    collector_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    config = MasterConfig.model_construct(
        grpo={
            "num_prompts_per_step": 1,
            "num_generations_per_prompt": 1,
            "max_rollout_turns": 1,
            "async_grpo": {
                "max_trajectory_age_steps": 1,
                "in_flight_weight_updates": False,
                "recompute_kv_cache_after_weight_updates": False,
                "pending_generation_timeout_s": 600.0,
            },
        },
        policy={
            "make_sequence_length_divisible_by": 1,
            "max_total_sequence_length": 16,
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {"async_engine": False},
            },
        },
    )
    return collector_cls(
        policy_generation=mock.MagicMock(),
        tokenizer=mock.MagicMock(),
        task_to_env={},
        master_config=config,
        replay_buffer=mock.MagicMock(),
    )


@pytest.mark.parametrize("pause_kind", ["refit", "manual"])
def test_pause_is_a_barrier_for_worker_launch(
    monkeypatch: pytest.MonkeyPatch, pause_kind: str
) -> None:
    collector = _local_collector()
    collector.running = True
    target_weight = 1
    collector._get_next_target_for_generation = mock.MagicMock(
        side_effect=lambda _version: (
            collector._generating_targets.add(target_weight) or target_weight
        )
    )
    collector.replay_buffer.get_trajectories_needed.remote.return_value = 1
    monkeypatch.setattr(collector_module.ray, "get", lambda value: value)

    semaphore_entered = threading.Event()
    allow_semaphore = threading.Event()

    class _BlockingSemaphore:
        def acquire(self) -> None:
            semaphore_entered.set()
            assert allow_semaphore.wait(timeout=2)

        def release(self) -> None:
            pass

    worker_started = threading.Event()

    class _RecordingThread:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def start(self) -> None:
            worker_started.set()

        def is_alive(self) -> bool:
            return False

    class _Batch:
        size = 1

        def slice(self, _start: int, _end: int) -> _Batch:
            return self

        def repeat_interleave(self, _repeats: int) -> _Batch:
            return self

    collector._inflight_sema = _BlockingSemaphore()
    process = threading.Thread(target=collector._process_batch, args=(_Batch(),))
    monkeypatch.setattr(collector_module._threading, "Thread", _RecordingThread)

    process.start()
    assert semaphore_entered.wait(timeout=2)

    if pause_kind == "refit":
        collector.prepare_for_refit()
    else:
        collector.pause(wait_for_pending_generations=True)
    allow_semaphore.set()
    time.sleep(0.05)

    assert not worker_started.is_set()
    if pause_kind == "refit":
        collector.resume_after_refit()
    else:
        collector.resume()
    process.join(timeout=2)
    assert not process.is_alive()
    assert worker_started.is_set()


def test_collection_loop_failure_is_reported_by_health_check() -> None:
    collector = _local_collector()

    class _FailingDataloader:
        def __iter__(self):
            raise RuntimeError("dataloader exploded")

    collector.dataloader = _FailingDataloader()
    collector.running = True
    collector._collection_loop()

    with pytest.raises(RuntimeError, match="dataloader exploded"):
        collector.check_health()


def test_collection_loop_exhaustion_is_reported_by_health_check() -> None:
    collector = _local_collector()
    collector.dataloader = []
    collector.running = True

    collector._collection_loop()

    with pytest.raises(RuntimeError, match="stopped before training completed"):
        collector.check_health()


def test_process_batch_failure_is_reported_by_health_check() -> None:
    collector = _local_collector()
    collector.running = True
    collector._get_next_target_for_generation = mock.MagicMock(
        side_effect=RuntimeError("batch exploded")
    )

    collector._process_batch(mock.MagicMock())

    with pytest.raises(RuntimeError, match="batch exploded"):
        collector.check_health()


def test_prompt_worker_failure_is_reported_by_health_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _local_collector()
    collector.running = True
    monkeypatch.setattr(
        "nemo_rl.algorithms.grpo._should_use_nemo_gym", lambda _config: False
    )
    monkeypatch.setattr(
        collector_module,
        "run_async_multi_turn_rollout",
        mock.MagicMock(side_effect=RuntimeError("rollout exploded")),
    )

    collector._run_prompt_group_worker(
        repeated_batch=mock.MagicMock(),
        generation_weight_version=0,
        target_weight_version=1,
        prompt_idx=0,
    )

    with pytest.raises(RuntimeError, match="rollout exploded"):
        collector.check_health()


def test_pending_generation_drain_timeout_is_reported_by_health_check() -> None:
    collector = _local_collector()
    collector.running = True
    pending_thread = mock.MagicMock()
    pending_thread.is_alive.return_value = True
    collector._inflight_threads.add(pending_thread)

    with pytest.raises(TimeoutError, match="timed out"):
        collector.wait_for_pending_generations(timeout_s=0.01)

    with pytest.raises(RuntimeError, match="timed out"):
        collector.check_health()


def test_paused_collector_abandons_full_buffer_enqueue_for_gap_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _local_collector()
    collector.running = True
    collector._manual_pause_cleared.clear()
    monkeypatch.setattr(
        "nemo_rl.algorithms.grpo._should_use_nemo_gym", lambda _config: False
    )
    final_batch = mock.MagicMock()
    final_batch.to.return_value = final_batch
    monkeypatch.setattr(
        collector_module,
        "run_async_multi_turn_rollout",
        mock.MagicMock(return_value=(final_batch, {})),
    )
    collector.replay_buffer.add.remote.return_value = "full"
    monkeypatch.setattr(collector_module.ray, "get", lambda value: value)

    worker = threading.Thread(
        target=collector._run_prompt_group_worker,
        kwargs={
            "repeated_batch": mock.MagicMock(),
            "generation_weight_version": 0,
            "target_weight_version": 1,
            "prompt_idx": 0,
        },
    )
    worker.start()
    worker.join(timeout=0.2)
    was_still_running = worker.is_alive()
    if was_still_running:
        collector.running = False
        worker.join(timeout=1)

    assert not was_still_running
    collector.running = True
    collector.check_health()
    assert collector._buffered_per_target.get(1, 0) == 0


def test_drained_refit_invalidates_prefix_cache_before_resume() -> None:
    collector = _local_collector()
    collector._refit_pause_cleared.clear()
    collector.policy_generation.invalidate_kv_cache.return_value = True

    collector.resume_after_refit()

    collector.policy_generation.invalidate_kv_cache.assert_called_once_with()
    assert collector._refit_pause_cleared.is_set()


def test_drained_refit_stays_paused_when_prefix_cache_reset_fails() -> None:
    collector = _local_collector()
    collector._refit_pause_cleared.clear()
    collector.policy_generation.invalidate_kv_cache.return_value = False

    with pytest.raises(RuntimeError, match="cache invalidation failed"):
        collector.resume_after_refit()

    assert not collector._refit_pause_cleared.is_set()


def test_drained_refit_skips_reset_when_prefix_cache_is_disabled() -> None:
    collector = _local_collector()
    collector.master_config.policy["generation"]["vllm_cfg"][
        "enable_prefix_caching"
    ] = False
    collector._refit_pause_cleared.clear()

    collector.resume_after_refit()

    collector.policy_generation.invalidate_kv_cache.assert_not_called()
    assert collector._refit_pause_cleared.is_set()


def test_complete_refit_updates_version_before_resuming() -> None:
    collector = _local_collector()
    calls: list[tuple[str, int | None]] = []
    collector.set_weight_version = mock.MagicMock(
        side_effect=lambda version: calls.append(("set", version))
    )
    collector.resume_after_refit = mock.MagicMock(
        side_effect=lambda: calls.append(("resume", None))
    )

    collector.complete_refit(7)

    assert calls == [("set", 7), ("resume", None)]
