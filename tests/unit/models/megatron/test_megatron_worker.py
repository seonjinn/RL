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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.algorithms.draft_update_schedule import DraftUpdateDecision
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorker,
    draft_execution_inputs,
)


pytestmark = pytest.mark.mcore


def _decision(*, update_requested: bool) -> DraftUpdateDecision:
    return DraftUpdateDecision(
        global_step=2,
        decision_id=3,
        update_requested=update_requested,
        draft_refit_requested=update_requested,
        reason="always" if update_requested else "none",
        observed_acceptance=None,
    )


def test_skip_omits_every_draft_execution_input() -> None:
    draft_model = object()
    draft_provider = object()

    inputs = draft_execution_inputs(
        _decision(update_requested=False), draft_model, draft_provider
    )

    assert inputs == {
        "run_draft": False,
        "enable_hidden_capture": False,
        "draft_model": None,
        "draft_provider": None,
    }


def test_requested_update_preserves_every_draft_execution_input() -> None:
    draft_model = object()
    draft_provider = object()

    inputs = draft_execution_inputs(
        _decision(update_requested=True), draft_model, draft_provider
    )

    assert inputs == {
        "run_draft": True,
        "enable_hidden_capture": True,
        "draft_model": draft_model,
        "draft_provider": draft_provider,
    }


def _worker_with_param_gather_overlap(
    *,
    hook_enabled: bool,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
) -> MegatronPolicyWorker:
    worker = object.__new__(MegatronPolicyWorker)
    worker.model = SimpleNamespace(
        ddp_config=SimpleNamespace(overlap_grad_reduce=overlap_grad_reduce)
    )
    worker.megatron_cfg = SimpleNamespace(
        ddp=SimpleNamespace(
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
        )
    )
    worker._forward_pre_hook_enabled = MagicMock(return_value=hook_enabled)
    worker.disable_forward_pre_hook = MagicMock()
    worker.enable_forward_pre_hook = MagicMock()
    return worker


def test_conditional_draft_skip_temporarily_uses_synchronous_ddp_lifecycle() -> None:
    worker = _worker_with_param_gather_overlap(hook_enabled=True)
    param_sync_func = object()
    grad_sync_func = object()
    model_config = SimpleNamespace(
        param_sync_func=param_sync_func,
        grad_sync_func=grad_sync_func,
    )

    with patch(
        "nemo_rl.models.policy.workers.megatron_policy_worker.get_model_config",
        return_value=model_config,
    ):
        with worker._conditional_draft_skip_ddp_sync(
            draft_enabled=True, run_draft=False
        ):
            worker.disable_forward_pre_hook.assert_called_once_with(param_sync=True)
            assert model_config.param_sync_func is None
            assert model_config.grad_sync_func is None
            assert worker.model.ddp_config.overlap_grad_reduce is False
            worker.enable_forward_pre_hook.assert_not_called()

    worker.enable_forward_pre_hook.assert_called_once_with()
    assert model_config.param_sync_func is param_sync_func
    assert model_config.grad_sync_func is grad_sync_func
    assert worker.model.ddp_config.overlap_grad_reduce is True


def test_conditional_draft_skip_restores_hooks_after_body_failure() -> None:
    worker = _worker_with_param_gather_overlap(hook_enabled=True)
    param_sync_func = object()
    grad_sync_func = object()
    model_config = SimpleNamespace(
        param_sync_func=param_sync_func,
        grad_sync_func=grad_sync_func,
    )

    with (
        patch(
            "nemo_rl.models.policy.workers.megatron_policy_worker.get_model_config",
            return_value=model_config,
        ),
        pytest.raises(RuntimeError, match="forward failed"),
    ):
        with worker._conditional_draft_skip_ddp_sync(
            draft_enabled=True, run_draft=False
        ):
            raise RuntimeError("forward failed")

    worker.enable_forward_pre_hook.assert_called_once_with()
    assert model_config.param_sync_func is param_sync_func
    assert model_config.grad_sync_func is grad_sync_func
    assert worker.model.ddp_config.overlap_grad_reduce is True


@pytest.mark.parametrize(
    ("draft_enabled", "run_draft"),
    [
        (False, False),
        (True, True),
    ],
)
def test_conditional_draft_skip_preserves_inactive_entry_state(
    draft_enabled: bool,
    run_draft: bool,
) -> None:
    worker = _worker_with_param_gather_overlap(hook_enabled=True)
    param_sync_func = object()
    grad_sync_func = object()
    model_config = SimpleNamespace(
        param_sync_func=param_sync_func,
        grad_sync_func=grad_sync_func,
    )

    with patch(
        "nemo_rl.models.policy.workers.megatron_policy_worker.get_model_config",
        return_value=model_config,
    ):
        with worker._conditional_draft_skip_ddp_sync(
            draft_enabled=draft_enabled, run_draft=run_draft
        ):
            assert model_config.param_sync_func is param_sync_func
            assert model_config.grad_sync_func is grad_sync_func
            assert worker.model.ddp_config.overlap_grad_reduce is True

    worker.disable_forward_pre_hook.assert_not_called()
    worker.enable_forward_pre_hook.assert_not_called()


@pytest.mark.parametrize(
    ("overlap_param_gather", "hook_enabled"),
    [
        (False, False),
        (True, False),
    ],
)
def test_conditional_draft_skip_disables_grad_overlap_without_param_hooks(
    overlap_param_gather: bool,
    hook_enabled: bool,
) -> None:
    worker = _worker_with_param_gather_overlap(
        hook_enabled=hook_enabled,
        overlap_param_gather=overlap_param_gather,
    )
    grad_sync_func = object()
    model_config = SimpleNamespace(
        param_sync_func=object(),
        grad_sync_func=grad_sync_func,
    )

    with patch(
        "nemo_rl.models.policy.workers.megatron_policy_worker.get_model_config",
        return_value=model_config,
    ):
        with worker._conditional_draft_skip_ddp_sync(
            draft_enabled=True, run_draft=False
        ):
            assert worker.model.ddp_config.overlap_grad_reduce is False
            assert model_config.grad_sync_func is None

    assert worker.model.ddp_config.overlap_grad_reduce is True
    assert model_config.grad_sync_func is grad_sync_func
    worker.disable_forward_pre_hook.assert_not_called()
    worker.enable_forward_pre_hook.assert_not_called()


def test_conditional_draft_skip_is_noop_without_ddp_overlap() -> None:
    worker = _worker_with_param_gather_overlap(
        hook_enabled=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )

    with worker._conditional_draft_skip_ddp_sync(draft_enabled=True, run_draft=False):
        assert worker.model.ddp_config.overlap_grad_reduce is False

    worker.disable_forward_pre_hook.assert_not_called()
    worker.enable_forward_pre_hook.assert_not_called()
