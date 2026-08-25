# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Minimal behavioral invariants for the data-plane wiring.

* ``examples/run_grpo._select_trainer`` dispatches the legacy trainer
  when ``data_plane`` is absent and the sync trainer when enabled.
* The ``DataPlaneClient`` ABC carries every method adapters depend on.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]


def test_run_grpo_dispatches_both_trainers():
    """``examples/run_grpo._select_trainer`` returns the TQ-mediated
    ``grpo_train_sync`` iff ``data_plane.enabled`` is true, and the
    legacy ``grpo_train`` otherwise."""
    import sys

    sys.path.insert(0, str(REPO / "examples"))
    try:
        from run_grpo import _select_trainer
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync

    cfg_legacy = MasterConfig.model_construct(data_plane=None)
    assert _select_trainer(cfg_legacy) is grpo_train

    cfg_sync = MasterConfig.model_construct(data_plane={"enabled": True})
    assert _select_trainer(cfg_sync) is grpo_train_sync


def test_sync_trainer_rejects_message_level_advantage_penalties():
    from nemo_rl.algorithms.grpo import GRPOConfig, MasterConfig
    from nemo_rl.algorithms.grpo_sync import (
        _raise_if_message_level_advantage_penalties_enabled,
    )

    cfg_disabled = MasterConfig.model_construct(grpo=GRPOConfig())
    _raise_if_message_level_advantage_penalties_enabled(cfg_disabled)

    cfg_enabled = MasterConfig.model_construct(
        grpo=GRPOConfig(
            invalid_tool_call_advantage=-5.0,
            malformed_thinking_advantage=None,
        )
    )
    with pytest.raises(
        NotImplementedError,
        match="grpo.invalid_tool_call_advantage",
    ):
        _raise_if_message_level_advantage_penalties_enabled(cfg_enabled)


@pytest.mark.parametrize(
    "method",
    [
        "register_partition",
        "claim_meta",
        "get_data",
        "put_samples",
        "get_samples",
        "clear_samples",
        "check_consumption_status",
        "close",
    ],
)
def test_data_plane_client_abc_method_present(method: str) -> None:
    """The ``DataPlaneClient`` ABC is the swap surface; a silent rename
    is a breaking change for every adapter."""
    from nemo_rl.data_plane.interfaces import DataPlaneClient

    assert hasattr(DataPlaneClient, method), (
        f"DataPlaneClient ABC is missing required method {method!r}. "
        "This is a breaking change for every adapter."
    )


def test_sync_rollout_actor_shutdown_is_explicit_and_idempotent(monkeypatch) -> None:
    from nemo_rl.algorithms import grpo_sync

    actor = MagicMock()
    shutdown_ref = object()
    actor.shutdown.remote.return_value = shutdown_ref
    ray_get = MagicMock()
    ray_kill = MagicMock()
    monkeypatch.setattr(grpo_sync.ray, "get", ray_get)
    monkeypatch.setattr(grpo_sync.ray, "kill", ray_kill)
    monkeypatch.setattr(grpo_sync, "_active_sync_rollout_actor", actor)

    grpo_sync.shutdown_active_sync_rollout_actor()
    grpo_sync.shutdown_active_sync_rollout_actor()

    actor.shutdown.remote.assert_called_once_with()
    ray_get.assert_called_once_with(shutdown_ref, timeout=10)
    ray_kill.assert_called_once_with(actor)
    assert grpo_sync._active_sync_rollout_actor is None


def test_sync_trainer_releases_actor_after_each_direct_call(monkeypatch) -> None:
    from nemo_rl.algorithms import grpo_sync

    calls: list[str] = []
    monkeypatch.setattr(
        grpo_sync,
        "_grpo_train_sync_impl",
        lambda *_args, **_kwargs: calls.append("train"),
    )
    monkeypatch.setattr(
        grpo_sync,
        "shutdown_active_sync_rollout_actor",
        lambda: calls.append("shutdown"),
    )

    args = [MagicMock() for _ in range(12)]
    grpo_sync.grpo_train_sync(*args)
    grpo_sync.grpo_train_sync(*args)

    assert calls == ["train", "shutdown", "train", "shutdown"]


def test_sync_trainer_reentry_does_not_kill_existing_actor(monkeypatch) -> None:
    from nemo_rl.algorithms import grpo_sync

    actor = MagicMock()
    shutdown = MagicMock()
    monkeypatch.setattr(grpo_sync, "_active_sync_rollout_actor", actor)
    monkeypatch.setattr(grpo_sync, "shutdown_active_sync_rollout_actor", shutdown)

    with pytest.raises(RuntimeError, match="already active"):
        grpo_sync.grpo_train_sync(*[MagicMock() for _ in range(12)])

    shutdown.assert_not_called()
    assert grpo_sync._active_sync_rollout_actor is actor


def test_sync_resource_shutdown_orders_actor_before_environment_and_policy(
    monkeypatch,
) -> None:
    import sys

    sys.path.insert(0, str(REPO / "examples"))
    try:
        import run_grpo
    finally:
        sys.path.pop(0)

    calls: list[str] = []
    policy = MagicMock()
    generation = MagicMock()
    monkeypatch.setattr(
        run_grpo,
        "shutdown_active_sync_rollout_actor",
        lambda: calls.append("rollout_actor"),
    )
    monkeypatch.setattr(
        run_grpo,
        "shutdown_environments",
        lambda *_: calls.append("environments"),
    )
    generation.shutdown.side_effect = lambda: calls.append("generation")
    policy.shutdown.side_effect = lambda: calls.append("policy")

    run_grpo._shutdown_training_resources(
        policy=policy,
        policy_generation=generation,
        task_to_env={},
        val_task_to_env={},
        data_plane_enabled=True,
    )

    assert calls == ["rollout_actor", "environments", "generation", "policy"]
