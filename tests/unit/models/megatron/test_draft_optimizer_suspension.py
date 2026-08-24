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

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

from nemo_rl.models.megatron.draft.optimizer import suspend_draft_optimizer_groups
from nemo_rl.models.megatron.draft.optimizer import (
    initialize_sparse_draft_optimizer_state,
    install_sparse_draft_optimizer_checkpointing,
)


pytestmark = pytest.mark.mcore


class _MutationTrackingGroups(list[dict[str, object]]):
    def __init__(self, groups: list[dict[str, object]]) -> None:
        super().__init__(groups)
        self.slice_mutations = 0

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, slice):
            self.slice_mutations += 1
        super().__setitem__(key, value)


def _draft_parameter(value: float = 1.0) -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.tensor([value]))
    parameter.grad_norm_group = "draft"
    return parameter


def _clone_state(state: dict[str, object]) -> dict[str, object]:
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }


def _run_mcore_master_parameter_skip(rank: int, world_size: int) -> None:
    from megatron.core.optimizer import optimizer as mcore_optimizer

    from nemo_rl.models.megatron.draft.utils import register_draft_grad_norm_group

    assert world_size == 1
    original_grad_norm_groups = mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS
    register_draft_grad_norm_group()
    device = torch.device("cuda", rank)
    policy = torch.nn.Parameter(
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
    )
    draft = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16, device=device))
    draft.grad_norm_group = "draft"
    base_optimizer = torch.optim.AdamW(
        [{"params": [policy]}, {"params": [draft]}],
        lr=0.1,
        weight_decay=0.1,
    )
    wrapped_optimizer = Float16OptimizerWithFloat16Params(
        base_optimizer,
        OptimizerConfig(
            optimizer="adam",
            lr=0.1,
            bf16=True,
            clip_grad=0.0,
        ),
        None,
        lambda optimizer, config: None,
    )
    wrapped_optimizer.grad_stats_parallel_group = torch.distributed.group.WORLD
    wrapped_optimizer.tp_group = torch.distributed.group.WORLD
    optimizer = ChainedOptimizer([wrapped_optimizer])

    try:
        policy.grad = torch.ones_like(policy)
        draft.grad = torch.ones_like(draft)
        update_successful, _, _ = optimizer.step()
        assert update_successful is True

        policy_main = wrapped_optimizer.fp32_from_float16_groups[0][0]
        draft_main = wrapped_optimizer.fp32_from_float16_groups[1][0]
        policy_model_before = policy.detach().clone()
        policy_before = policy_main.detach().clone()
        draft_before = draft.detach().clone()
        draft_main_before = draft_main.detach().clone()
        draft_state_before = _clone_state(base_optimizer.state[draft_main])
        groups_before = list(base_optimizer.param_groups)

        policy.grad = torch.ones_like(policy)
        draft.grad = torch.ones_like(draft)
        with suspend_draft_optimizer_groups(optimizer):
            assert len(base_optimizer.param_groups) == 1
            assert base_optimizer.param_groups[0] is groups_before[0]
            update_successful, _, _ = optimizer.step()

        assert update_successful is True
        assert not torch.equal(policy, policy_model_before)
        assert not torch.equal(policy_main, policy_before)
        assert torch.equal(draft, draft_before)
        assert torch.equal(draft_main, draft_main_before)
        assert len(base_optimizer.param_groups) == len(groups_before)
        assert all(
            restored is original
            for restored, original in zip(
                base_optimizer.param_groups, groups_before, strict=True
            )
        )
        for key in ("exp_avg", "exp_avg_sq", "step"):
            actual = base_optimizer.state[draft_main][key]
            expected = draft_state_before[key]
            assert isinstance(actual, torch.Tensor)
            assert isinstance(expected, torch.Tensor)
            assert torch.equal(actual, expected)
    finally:
        mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS = original_grad_norm_groups


def test_skip_preserves_draft_bytes_moments_and_step_while_scheduler_advances() -> None:
    policy = torch.nn.Parameter(torch.tensor([1.0]))
    draft = _draft_parameter()
    optimizer = torch.optim.AdamW(
        [{"params": [policy]}, {"params": [draft]}],
        lr=0.1,
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    policy.grad = torch.ones_like(policy)
    draft.grad = torch.ones_like(draft)
    optimizer.step()
    draft_before = draft.detach().clone()
    draft_state_before = _clone_state(optimizer.state[draft])
    groups_before = list(optimizer.param_groups)

    policy.grad = torch.ones_like(policy)
    draft.grad = torch.ones_like(draft)
    policy_before = policy.detach().clone()
    with suspend_draft_optimizer_groups(optimizer):
        optimizer.step()
    scheduler.step()

    assert not torch.equal(policy, policy_before)
    assert torch.equal(draft, draft_before)
    assert optimizer.param_groups == groups_before
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.05)
    for key, expected in draft_state_before.items():
        actual = optimizer.state[draft][key]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected


def test_mcore_chained_skip_preserves_draft_master_state(
    distributed_test_runner: Callable[..., None],
) -> None:
    distributed_test_runner(_run_mcore_master_parameter_skip, world_size=1)


def test_restores_groups_when_optimizer_step_raises() -> None:
    policy = torch.nn.Parameter(torch.tensor([1.0]))
    draft = _draft_parameter()
    optimizer = torch.optim.SGD([{"params": [policy]}, {"params": [draft]}])
    groups_before = list(optimizer.param_groups)

    with pytest.raises(RuntimeError, match="step failed"):
        with suspend_draft_optimizer_groups(optimizer):
            assert optimizer.param_groups == [groups_before[0]]
            raise RuntimeError("step failed")

    assert optimizer.param_groups == groups_before


def test_mcore_wrapper_and_chained_optimizers_suspend_each_draft_group() -> None:
    first_policy = torch.nn.Parameter(torch.tensor([1.0]))
    first_draft = _draft_parameter()
    second_policy = torch.nn.Parameter(torch.tensor([2.0]))
    second_draft = _draft_parameter(2.0)
    first = torch.optim.SGD(
        [{"params": [first_policy]}, {"params": [first_draft]}], lr=0.1
    )
    second = torch.optim.SGD(
        [{"params": [second_policy]}, {"params": [second_draft]}], lr=0.1
    )
    first_groups = list(first.param_groups)
    second_groups = list(second.param_groups)
    chained = SimpleNamespace(
        chained_optimizers=(
            SimpleNamespace(optimizer=first),
            SimpleNamespace(optimizer=second),
        )
    )

    with suspend_draft_optimizer_groups(chained):
        assert first.param_groups == [first_groups[0]]
        assert second.param_groups == [second_groups[0]]

    assert first.param_groups == first_groups
    assert second.param_groups == second_groups


def test_mixed_group_in_later_chained_optimizer_fails_before_any_mutation() -> None:
    first_policy = torch.nn.Parameter(torch.tensor([1.0]))
    first_draft = _draft_parameter()
    mixed_policy = torch.nn.Parameter(torch.tensor([2.0]))
    mixed_draft = _draft_parameter(2.0)
    first = torch.optim.SGD(
        [{"params": [first_policy]}, {"params": [first_draft]}], lr=0.1
    )
    second = torch.optim.SGD([{"params": [mixed_policy, mixed_draft]}], lr=0.1)
    first.param_groups = _MutationTrackingGroups(first.param_groups)
    second.param_groups = _MutationTrackingGroups(second.param_groups)
    first_groups = list(first.param_groups)
    second_groups = list(second.param_groups)
    chained = SimpleNamespace(
        chained_optimizers=(
            SimpleNamespace(optimizer=first),
            SimpleNamespace(optimizer=second),
        )
    )

    with pytest.raises(RuntimeError, match="mixes policy and draft"):
        with suspend_draft_optimizer_groups(chained):
            raise AssertionError("mixed groups must fail on context entry")

    assert first.param_groups == first_groups
    assert second.param_groups == second_groups
    assert first.param_groups.slice_mutations == 0
    assert second.param_groups.slice_mutations == 0


def test_sparse_checkpoint_initialization_preserves_live_optimizer_state() -> None:
    policy = torch.nn.Parameter(torch.tensor([1.0]))
    draft = _draft_parameter()
    optimizer = torch.optim.AdamW(
        [{"params": [policy]}, {"params": [draft]}],
        lr=0.1,
        weight_decay=0.1,
    )

    policy.grad = torch.ones_like(policy)
    with suspend_draft_optimizer_groups(optimizer):
        optimizer.step()
    policy_state_before = _clone_state(optimizer.state[policy])
    policy_before = policy.detach().clone()
    draft_before = draft.detach().clone()
    policy_grad = torch.full_like(policy, 2.0)
    draft_grad = torch.full_like(draft, 3.0)
    policy.grad = policy_grad
    draft.grad = draft_grad
    groups_before = list(optimizer.param_groups)
    lrs_before = [group["lr"] for group in groups_before]

    initialized = initialize_sparse_draft_optimizer_state(optimizer)

    assert initialized == 1
    assert torch.equal(policy, policy_before)
    assert torch.equal(draft, draft_before)
    assert policy.grad is policy_grad
    assert draft.grad is draft_grad
    assert list(optimizer.param_groups) == groups_before
    assert [group["lr"] for group in groups_before] == lrs_before
    for key, expected in policy_state_before.items():
        actual = optimizer.state[policy][key]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected
    assert optimizer.state[draft]["step"].item() == 0
    assert torch.count_nonzero(optimizer.state[draft]["exp_avg"]) == 0
    assert torch.count_nonzero(optimizer.state[draft]["exp_avg_sq"]) == 0


class _RejectsHeterogeneousSteps:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, int]:
        steps = {int(state["step"].item()) for state in self.optimizer.state.values()}
        assert len(steps) <= 1, f"steps: {sorted(steps)}"
        return {"step": max(steps, default=0)}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        for state in self.optimizer.state.values():
            state["step"].fill_(state_dict["step"])

    def sharded_state_dict(self) -> dict[str, object]:
        header = self.state_dict()
        parameter_steps = [
            int(state["step"].item()) for state in self.optimizer.state.values()
        ]
        return {"header": header, "parameter_steps": parameter_steps}


def test_sparse_checkpoint_uses_uniform_header_but_preserves_parameter_steps() -> None:
    policy = torch.nn.Parameter(torch.tensor([1.0]))
    draft = _draft_parameter()
    base_optimizer = torch.optim.AdamW(
        [{"params": [policy]}, {"params": [draft]}],
        lr=0.1,
    )
    wrapped_optimizer = _RejectsHeterogeneousSteps(base_optimizer)
    optimizer = SimpleNamespace(chained_optimizers=(wrapped_optimizer,))

    for step in range(3):
        policy.grad = torch.ones_like(policy)
        draft.grad = torch.ones_like(draft)
        context = (
            suspend_draft_optimizer_groups(base_optimizer)
            if step > 0
            else nullcontext()
        )
        with context:
            base_optimizer.step()

    with pytest.raises(AssertionError, match=r"steps: \[1, 3\]"):
        wrapped_optimizer.sharded_state_dict()

    assert install_sparse_draft_optimizer_checkpointing(optimizer) == 1
    assert install_sparse_draft_optimizer_checkpointing(optimizer) == 0
    checkpoint = wrapped_optimizer.sharded_state_dict()

    assert checkpoint == {
        "header": {
            "step": 3,
            "nemo_rl_sparse_draft_optimizer_step": 1,
        },
        "parameter_steps": [3, 1],
    }
    assert [int(state["step"].item()) for state in base_optimizer.state.values()] == [
        3,
        1,
    ]

    for state in base_optimizer.state.values():
        state["step"].fill_(9)
    wrapped_optimizer.load_state_dict(checkpoint["header"])

    assert [int(state["step"].item()) for state in base_optimizer.state.values()] == [
        3,
        1,
    ]


class _GroupStepOptimizer:
    def __init__(self, policy: torch.nn.Parameter, draft: torch.nn.Parameter) -> None:
        self.param_groups = [
            {"params": [policy], "lr": 0.1, "step": 4},
            {"params": [draft], "lr": 0.1},
        ]
        self.state: dict[torch.nn.Parameter, dict[str, torch.Tensor]] = {
            policy: {"master_param": policy.detach().clone()}
        }

    def step(self) -> None:
        for group in self.param_groups:
            active = [
                parameter for parameter in group["params"] if parameter.grad is not None
            ]
            if not active:
                continue
            group["step"] = int(group.get("step", 0)) + 1
            for parameter in active:
                self.state.setdefault(
                    parameter,
                    {"master_param": parameter.detach().clone()},
                )

    def zero_grad(self, set_to_none: bool = True) -> None:
        assert set_to_none
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter.grad = None

    def state_dict(self) -> dict[str, int]:
        steps = {int(group["step"]) for group in self.param_groups}
        assert len(steps) == 1, sorted(steps)
        return {"step": steps.pop()}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        for group in self.param_groups:
            group["step"] = state_dict["step"]


def test_sparse_checkpoint_preserves_fused_optimizer_group_steps() -> None:
    policy = torch.nn.Parameter(torch.tensor([1.0]))
    draft = _draft_parameter()
    optimizer = _GroupStepOptimizer(policy, draft)
    policy_grad = torch.full_like(policy, 2.0)
    draft_grad = torch.full_like(draft, 3.0)
    policy.grad = policy_grad
    draft.grad = draft_grad

    assert initialize_sparse_draft_optimizer_state(optimizer) == 1

    assert optimizer.param_groups[0]["step"] == 4
    assert optimizer.param_groups[1]["step"] == 0
    assert "master_param" in optimizer.state[draft]
    assert policy.grad is policy_grad
    assert draft.grad is draft_grad
    assert install_sparse_draft_optimizer_checkpointing(optimizer) == 1

    checkpoint = optimizer.state_dict()

    assert checkpoint == {
        "step": 4,
        "nemo_rl_sparse_draft_optimizer_step": 0,
    }
    assert [group["step"] for group in optimizer.param_groups] == [4, 0]

    for group in optimizer.param_groups:
        group["step"] = 9
    optimizer.load_state_dict(checkpoint)

    assert [group["step"] for group in optimizer.param_groups] == [4, 0]
