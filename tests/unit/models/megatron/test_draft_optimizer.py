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
from unittest.mock import MagicMock

import pytest
import torch

# Module-level megatron imports would break COLLECTION on non-mcore CI shards
# (marks only deselect at run time); skip collection gracefully instead.
pytest.importorskip("megatron.bridge.training.config")
pytest.importorskip("megatron.core.optimizer")

from megatron.bridge.training.config import OptimizerConfigOverrideProviderContext
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,
    ParamGroupOverride,
    combine_param_group_overrides,
)
from pydantic import ValidationError

from nemo_rl.models.megatron.draft.optimizer import (
    DraftOptimizerConfigOverrideProvider,
    build_draft_optimizer_override_provider,
    draft_param_groups,
    frozen_param_groups,
)
from nemo_rl.models.megatron.draft.utils import DRAFT_GRAD_NORM_GROUP
from nemo_rl.models.megatron.param_group_scheduler import (
    ParamGroupWarmupScheduler,
    enable_param_group_warmup,
)
from nemo_rl.models.policy.draft_config import (
    DraftOptimizerConfig,
    Eagle3DraftConfig,
)

pytestmark = pytest.mark.mcore


def _combined_overrides_for_parameter(
    provider: DraftOptimizerConfigOverrideProvider,
    parameter: torch.nn.Parameter,
    name: str,
    *,
    policy_lr: float = 2.0e-3,
    policy_min_lr: float = 2.0e-4,
    scheduler_config: object | None = None,
) -> ParamGroupOverride:
    overrides = provider.build_config_overrides(
        OptimizerConfigOverrideProviderContext(
            scheduler_config=scheduler_config or MagicMock(),
            optimizer_config=OptimizerConfig(
                lr=policy_lr,
                min_lr=policy_min_lr,
                weight_decay=0.1,
            ),
            model=MagicMock(),
        )
    )
    assert overrides is not None
    return combine_param_group_overrides(
        [
            override
            for key, override in overrides.items()
            if key.matches(parameter, name)
        ]
    )


def _linear_scheduler(optimizer: torch.optim.Optimizer) -> OptimizerParamScheduler:
    return OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=2.0e-3,
        min_lr=2.0e-4,
        lr_warmup_steps=2,
        lr_decay_steps=6,
        lr_decay_style="linear",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=6,
        wd_incr_style="constant",
    )


@pytest.mark.parametrize(
    "config",
    [
        {"lr": 0.0},
        {"lr": -1.0},
        {"lr": 1.0e-5, "min_lr": -1.0e-6},
        {"lr": 1.0e-5, "min_lr": 2.0e-5},
        {"lr": 1.0e-5, "weight_decay": -0.1},
        {"lr": 1.0e-5, "learning_rate": 2.0e-5},
        {"lr": 1.0e-5, "warmup_iters": -1},
    ],
)
def test_draft_optimizer_config_rejects_invalid_values(
    config: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        DraftOptimizerConfig(**config)


def test_draft_optimizer_override_is_opt_in() -> None:
    configured_but_disabled = Eagle3DraftConfig(
        enabled=False,
        optimizer={"lr": 1.0e-5},
    )

    assert build_draft_optimizer_override_provider(None, samples_per_iter=1) is None
    assert (
        build_draft_optimizer_override_provider(
            Eagle3DraftConfig(enabled=True), samples_per_iter=1
        )
        is None
    )
    assert (
        build_draft_optimizer_override_provider(
            configured_but_disabled, samples_per_iter=1
        )
        is None
    )


def test_draft_optimizer_provider_only_overrides_tagged_parameters() -> None:
    draft_parameter = torch.nn.Parameter(torch.ones(2, 2))
    draft_parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
    policy_parameter = torch.nn.Parameter(torch.ones(2, 2))
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig(lr=1.0e-3, min_lr=1.0e-4, weight_decay=0.02),
        samples_per_iter=1,
    )

    assert _combined_overrides_for_parameter(
        provider, draft_parameter, "draft.weight"
    ) == {
        "max_lr": 1.0e-3,
        "min_lr": 1.0e-4,
        "start_wd": 0.02,
        "end_wd": 0.02,
    }
    assert (
        _combined_overrides_for_parameter(provider, policy_parameter, "policy.weight")
        == {}
    )


def test_draft_weight_decay_override_keeps_norm_and_bias_at_zero_decay() -> None:
    """Megatron's standard wd_mult=0.0 for 1-D params survives the draft override:
    the scheduler multiplies get_wd(group) * wd_mult, so draft norms/biases keep
    zero decay even when draft.optimizer.weight_decay is set."""
    from megatron.core.optimizer import get_standard_config_overrides

    bias_like = torch.nn.Parameter(torch.ones(4))
    bias_like.grad_norm_group = DRAFT_GRAD_NORM_GROUP
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig(lr=1.0e-3, weight_decay=0.02), samples_per_iter=1
    )
    context = OptimizerConfigOverrideProviderContext(
        scheduler_config=MagicMock(),
        optimizer_config=OptimizerConfig(lr=2.0e-3, min_lr=2.0e-4, weight_decay=0.1),
        model=MagicMock(),
    )
    draft_overrides = provider.build_config_overrides(context)
    assert draft_overrides is not None
    standard_overrides = get_standard_config_overrides(context.optimizer_config)

    merged = {**standard_overrides, **draft_overrides}
    combined = combine_param_group_overrides(
        [
            override
            for key, override in merged.items()
            if key.matches(bias_like, "draft.norm.bias")
        ]
    )

    assert combined["start_wd"] == 0.02
    assert combined["wd_mult"] == 0.0
    assert combined["start_wd"] * combined["wd_mult"] == 0.0


def test_draft_optimizer_provider_rejects_incompatible_inherited_min_lr() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig(lr=1.0e-5), samples_per_iter=1
    )

    # The floor was never configured on the draft block, so the message has to
    # name the inheritance and the knob that decouples it -- otherwise the user
    # sees a bound they never set and has nothing to act on.
    with pytest.raises(
        ValueError,
        match=r"inherited from the policy optimizer.*draft\.optimizer\.min_lr",
    ):
        _combined_overrides_for_parameter(
            provider,
            parameter,
            "draft.weight",
            policy_lr=2.0e-5,
            policy_min_lr=2.0e-5,
        )


def test_draft_optimizer_provider_rejects_explicit_min_lr_above_lr() -> None:
    """The non-inherited branch of the same guard.

    `DraftOptimizerConfig` already rejects `min_lr > lr` at validation, so this
    state is only reachable if that validator is bypassed; `model_construct`
    reproduces it so the defensive branch stays covered and does not claim the
    floor was inherited when the user set it themselves.
    """
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig.model_construct(lr=1.0e-5, min_lr=2.0e-5),
        samples_per_iter=1,
    )

    with pytest.raises(ValueError, match="draft optimizer lr") as excinfo:
        _combined_overrides_for_parameter(provider, parameter, "draft.weight")
    assert "inherited" not in str(excinfo.value)


def test_real_scheduler_applies_distinct_draft_lr_trajectory() -> None:
    policy_parameter = torch.nn.Parameter(torch.ones(1))
    draft_parameter = torch.nn.Parameter(torch.ones(1))
    draft_parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig(lr=1.0e-3, min_lr=1.0e-4), samples_per_iter=1
    )
    draft_overrides = _combined_overrides_for_parameter(
        provider, draft_parameter, "draft.weight"
    )
    optimizer = torch.optim.SGD(
        [
            {"params": [policy_parameter]},
            {"params": [draft_parameter], **draft_overrides},
        ],
        lr=2.0e-3,
    )
    scheduler = _linear_scheduler(optimizer)

    trajectories: list[tuple[float, float]] = []
    for step in (0, 2, 4, 6):
        scheduler.step(step - scheduler.num_steps)
        trajectories.append(
            (optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])
        )

    assert trajectories == [
        (0.0, 0.0),
        (2.0e-3, 1.0e-3),
        (1.1e-3, 5.5e-4),
        (2.0e-4, 1.0e-4),
    ]


def test_draft_warmup_iters_converts_to_scheduler_steps() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig(lr=1.0e-3, min_lr=1.0e-4, warmup_iters=3),
        samples_per_iter=8,
    )

    combined = _combined_overrides_for_parameter(
        provider,
        parameter,
        "draft.weight",
        scheduler_config=SimpleNamespace(lr_decay_steps=100),
    )

    assert combined["lr_warmup_steps"] == 24


def test_draft_warmup_is_capped_before_lr_decay_ends() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig(lr=1.0e-3, min_lr=1.0e-4, warmup_iters=3),
        samples_per_iter=2,
    )

    with pytest.warns(UserWarning, match="capping it to 5"):
        combined = _combined_overrides_for_parameter(
            provider,
            parameter,
            "draft.weight",
            scheduler_config=SimpleNamespace(lr_decay_steps=6),
        )

    assert combined["lr_warmup_steps"] == 5


@pytest.mark.parametrize(
    ("warmup_iters", "expected_draft_lrs"),
    [
        # No warmup: the draft starts at lr and decays over all 6 steps.
        (0, [1.0e-3, 7.0e-4, 4.0e-4, 1.0e-4]),
        # 2 iterations x 2 samples = 4 warmup steps, longer than the policy's 2.
        (2, [0.0, 5.0e-4, 1.0e-3, 1.0e-4]),
    ],
)
def test_draft_warmup_is_independent_of_policy_warmup(
    warmup_iters: int, expected_draft_lrs: list[float]
) -> None:
    policy_parameter = torch.nn.Parameter(torch.ones(1))
    draft_parameter = torch.nn.Parameter(torch.ones(1))
    draft_parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig(lr=1.0e-3, min_lr=1.0e-4, warmup_iters=warmup_iters),
        samples_per_iter=2,
    )
    draft_overrides = _combined_overrides_for_parameter(
        provider,
        draft_parameter,
        "draft.weight",
        scheduler_config=SimpleNamespace(lr_decay_steps=6),
    )
    optimizer = torch.optim.SGD(
        [
            {"params": [policy_parameter]},
            {"params": [draft_parameter], **draft_overrides},
        ],
        lr=2.0e-3,
    )
    scheduler = _linear_scheduler(optimizer)
    enable_param_group_warmup(scheduler)

    policy_lrs: list[float] = []
    draft_lrs: list[float] = []
    for step in (0, 2, 4, 6):
        scheduler.step(step - scheduler.num_steps)
        policy_lrs.append(optimizer.param_groups[0]["lr"])
        draft_lrs.append(optimizer.param_groups[1]["lr"])

    assert isinstance(scheduler, ParamGroupWarmupScheduler)
    assert policy_lrs == pytest.approx([0.0, 2.0e-3, 1.1e-3, 2.0e-4])
    assert draft_lrs == pytest.approx(expected_draft_lrs)


def test_enable_param_group_warmup_refreshes_lr_set_by_constructor() -> None:
    draft_parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD(
        [
            {
                "params": [draft_parameter],
                "max_lr": 1.0e-3,
                "min_lr": 1.0e-4,
                "lr_warmup_steps": 0,
            }
        ],
        lr=2.0e-3,
    )
    scheduler = _linear_scheduler(optimizer)
    # The constructor applied the global warmup, so the first optimizer step
    # would still use init_lr without the refresh.
    assert optimizer.param_groups[0]["lr"] == 0.0

    enable_param_group_warmup(scheduler)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert scheduler.num_steps == 0


@pytest.mark.parametrize(
    "saved_warmup_steps",
    [
        # A checkpoint from before warmup_iters, or one where Megatron matched
        # the draft group to a policy group with the same lr and min_lr.
        None,
        # A checkpoint saved with another warmup_iters value.
        99,
    ],
)
def test_draft_warmup_survives_optimizer_checkpoint_load(
    saved_warmup_steps: int | None,
) -> None:
    def build() -> tuple[torch.optim.Optimizer, OptimizerParamScheduler]:
        optimizer = torch.optim.SGD(
            [
                {"params": [torch.nn.Parameter(torch.ones(1))]},
                {
                    "params": [torch.nn.Parameter(torch.ones(1))],
                    "max_lr": 1.0e-3,
                    "min_lr": 1.0e-4,
                    "lr_warmup_steps": 0,
                },
            ],
            lr=2.0e-3,
        )
        scheduler = _linear_scheduler(optimizer)
        enable_param_group_warmup(scheduler)
        return optimizer, scheduler

    optimizer, scheduler = build()
    scheduler.step(2)
    optimizer_state = optimizer.state_dict()
    saved_draft_group = optimizer_state["param_groups"][1]
    if saved_warmup_steps is None:
        del saved_draft_group["lr_warmup_steps"]
    else:
        saved_draft_group["lr_warmup_steps"] = saved_warmup_steps

    resumed_optimizer, resumed_scheduler = build()
    # Megatron-Bridge loads the optimizer state first and the scheduler second.
    # The optimizer load replaces the group dicts with the saved ones.
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler.state_dict())
    draft_lrs = [resumed_optimizer.param_groups[1]["lr"]]
    resumed_scheduler.step(2)
    draft_lrs.append(resumed_optimizer.param_groups[1]["lr"])

    assert resumed_optimizer.param_groups[1]["lr_warmup_steps"] == 0
    assert draft_lrs == pytest.approx([7.0e-4, 4.0e-4])


def test_enable_param_group_warmup_is_noop_without_group_warmup() -> None:
    optimizer = torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.ones(1))], "max_lr": 1.0e-3}],
        lr=2.0e-3,
    )
    scheduler = _linear_scheduler(optimizer)

    enable_param_group_warmup(scheduler)

    assert type(scheduler) is OptimizerParamScheduler


def test_draft_lr_override_produces_finite_stable_toy_updates() -> None:
    def run(*, draft_lr: float | None) -> tuple[list[float], float, float]:
        policy_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        draft_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        draft_group: dict[str, object] = {"params": [draft_parameter]}
        if draft_lr is not None:
            draft_group.update({"max_lr": draft_lr, "min_lr": draft_lr / 10})
        optimizer = torch.optim.SGD(
            [{"params": [policy_parameter]}, draft_group],
            lr=2.0e-3,
        )
        scheduler = _linear_scheduler(optimizer)
        losses: list[float] = []
        for _ in range(6):
            optimizer.zero_grad()
            loss = policy_parameter.square().sum() + draft_parameter.square().sum()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step(1)
        return losses, policy_parameter.item(), draft_parameter.item()

    baseline_losses, baseline_policy, baseline_draft = run(draft_lr=None)
    override_losses, override_policy, override_draft = run(draft_lr=1.0e-3)

    assert torch.isfinite(torch.tensor(baseline_losses + override_losses)).all()
    assert baseline_losses[-1] < baseline_losses[0]
    assert override_losses[-1] < override_losses[0]
    assert override_policy == baseline_policy
    assert override_draft > baseline_draft


def _param_group(*parameters: torch.nn.Parameter, lr: float) -> dict:
    return {"params": list(parameters), "lr": lr}


def _draft_tagged_parameter() -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
    return parameter


def test_draft_param_groups_selects_only_draft_owned_groups() -> None:
    draft_group = _param_group(_draft_tagged_parameter(), lr=1.0e-4)
    policy_group = _param_group(torch.nn.Parameter(torch.ones(2)), lr=1.0e-6)
    empty_group = _param_group(lr=1.0e-4)
    optimizer = SimpleNamespace(param_groups=[policy_group, draft_group, empty_group])

    assert draft_param_groups(optimizer) == [draft_group]


def test_draft_param_groups_rejects_mixed_groups() -> None:
    """Catches a group whose lr=0 freeze would also freeze policy parameters."""
    mixed_group = _param_group(
        _draft_tagged_parameter(), torch.nn.Parameter(torch.ones(2)), lr=1.0e-4
    )

    with pytest.raises(ValueError, match="mixes draft and policy parameters"):
        draft_param_groups(SimpleNamespace(param_groups=[mixed_group]))


def test_frozen_param_groups_keeps_adamw_parameters_unchanged() -> None:
    """Zero gradients alone still move AdamW parameters via momentum and decay."""
    draft_parameter = _draft_tagged_parameter()
    policy_parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW(
        [
            {"params": [policy_parameter], "lr": 1.0e-2},
            {"params": [draft_parameter], "lr": 1.0e-2},
        ],
        weight_decay=0.1,
    )
    (policy_parameter.sum() + draft_parameter.sum()).backward()
    optimizer.step()
    draft_before = draft_parameter.detach().clone()
    policy_before = policy_parameter.detach().clone()

    optimizer.zero_grad(set_to_none=False)
    (policy_parameter.sum() + 0.0 * draft_parameter.sum()).backward()
    with frozen_param_groups(draft_param_groups(optimizer)):
        optimizer.step()

    assert torch.equal(draft_parameter, draft_before)
    assert not torch.equal(policy_parameter, policy_before)
    assert [group["lr"] for group in optimizer.param_groups] == [1.0e-2, 1.0e-2]


def test_frozen_param_groups_restores_lr_after_an_error() -> None:
    group = _param_group(_draft_tagged_parameter(), lr=3.0e-4)

    with pytest.raises(RuntimeError, match="step failed"):
        with frozen_param_groups([group]):
            assert group["lr"] == 0.0
            raise RuntimeError("step failed")

    assert group["lr"] == 3.0e-4
