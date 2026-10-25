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

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from megatron.bridge.training.config import (
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
)
from megatron.core.optimizer import ParamKey, ParamPredicate
from megatron.core.optimizer_param_scheduler import ParamGroupOverride

from nemo_rl.models.megatron.draft.utils import is_draft_parameter
from nemo_rl.models.megatron.param_group_scheduler import ParamGroupScheduleOverride
from nemo_rl.models.policy.draft_config import DraftOptimizerConfig


class DraftOptimizerConfigOwner(Protocol):
    """Draft config fields required to opt in to optimizer overrides."""

    enabled: bool
    optimizer: DraftOptimizerConfig | None


@dataclass
class DraftOptimizerConfigOverrideProvider(OptimizerConfigOverrideProvider):
    """Add scheduler overrides only for parameters owned by a draft model."""

    draft_optimizer: DraftOptimizerConfig
    # NeMo RL steps the scheduler by the global batch size, so one training
    # iteration is this many scheduler steps.
    samples_per_iter: int

    def build_config_overrides(
        self, context: OptimizerConfigOverrideProviderContext
    ) -> dict[ParamKey, ParamGroupOverride] | None:
        """Combine standard Megatron groups with the opt-in draft schedule."""
        overrides = super().build_config_overrides(context) or {}
        minimum_lr = self.draft_optimizer.min_lr
        if minimum_lr is None:
            minimum_lr = context.optimizer_config.min_lr
        if minimum_lr is not None and minimum_lr > self.draft_optimizer.lr:
            inherited = self.draft_optimizer.min_lr is None
            raise ValueError(
                f"draft optimizer lr ({self.draft_optimizer.lr}) is below "
                f"min_lr ({minimum_lr})"
                + (
                    ", which was inherited from the policy optimizer; set "
                    "policy.draft.optimizer.min_lr to decouple the draft schedule"
                    if inherited
                    else ""
                )
            )

        draft_override = ParamGroupScheduleOverride(max_lr=self.draft_optimizer.lr)
        if minimum_lr is not None:
            draft_override["min_lr"] = minimum_lr
        if self.draft_optimizer.warmup_iters is not None:
            warmup_steps = self.draft_optimizer.warmup_iters * self.samples_per_iter
            decay_steps = context.scheduler_config.lr_decay_steps
            # Same invariant and cap as Megatron-Bridge for the policy warmup.
            if warmup_steps >= decay_steps:
                capped = decay_steps - 1
                warnings.warn(
                    f"draft optimizer warmup ({warmup_steps} scheduler steps) >= "
                    f"lr_decay_steps ({decay_steps}); capping it to {capped}. "
                    "Reduce policy.draft.optimizer.warmup_iters for short runs.",
                    UserWarning,
                    stacklevel=2,
                )
                warmup_steps = capped
            draft_override["lr_warmup_steps"] = warmup_steps
        if self.draft_optimizer.weight_decay is not None:
            # Megatron's standard overrides still set wd_mult=0.0 for norm/bias
            # and 1-D draft params, and the scheduler multiplies:
            # weight_decay = get_wd(group) * wd_mult. So this override changes
            # decay only for weight matrices; draft norms/biases stay at 0,
            # matching the main model's convention.
            draft_override["start_wd"] = self.draft_optimizer.weight_decay
            draft_override["end_wd"] = self.draft_optimizer.weight_decay

        overrides[
            ParamKey(
                predicate=ParamPredicate(
                    name="draft_parameter",
                    fn=is_draft_parameter,
                )
            )
        ] = draft_override
        return overrides


def build_draft_optimizer_override_provider(
    draft_config: DraftOptimizerConfigOwner | None,
    samples_per_iter: int,
) -> DraftOptimizerConfigOverrideProvider | None:
    """Build an override provider only when a draft schedule is configured."""
    if (
        draft_config is None
        or not draft_config.enabled
        or draft_config.optimizer is None
    ):
        return None
    return DraftOptimizerConfigOverrideProvider(
        draft_config.optimizer, samples_per_iter=samples_per_iter
    )


def draft_param_groups(optimizer: Any) -> list[dict[str, Any]]:
    """Return the optimizer param groups that hold only draft parameters.

    Groups with no parameters on this rank are skipped.

    Raises:
        ValueError: If a group mixes draft and policy parameters. Such a group
            cannot freeze the draft without also freezing the policy.
    """
    groups = []
    for group in optimizer.param_groups:
        draft_flags = {is_draft_parameter(parameter) for parameter in group["params"]}
        if draft_flags == {True}:
            groups.append(group)
        elif len(draft_flags) > 1:
            raise ValueError(
                "an optimizer param group mixes draft and policy parameters; "
                "set policy.draft.optimizer so the draft owns its param groups"
            )
    return groups


def _torch_optimizers(optimizer: Any) -> Iterator[torch.optim.Optimizer]:
    """Yield the torch optimizers inside a Megatron optimizer wrapper."""
    if isinstance(optimizer, torch.optim.Optimizer):
        yield optimizer
    elif hasattr(optimizer, "chained_optimizers"):
        for chained_optimizer in optimizer.chained_optimizers:
            yield from _torch_optimizers(chained_optimizer)
    elif getattr(optimizer, "optimizer", None) is not None:
        yield from _torch_optimizers(optimizer.optimizer)


def _snapshot(values: dict[Any, Any], skip_key: str | None = None) -> dict[Any, Any]:
    # Non-tensor values are kept by reference. Optimizers replace them, they
    # do not change them in place.
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
        if key != skip_key
    }


def _restore(
    values: dict[Any, Any], saved: dict[Any, Any], skip_key: str | None = None
) -> None:
    for key in [key for key in values if key != skip_key and key not in saved]:
        del values[key]
    for key, saved_value in saved.items():
        current = values.get(key)
        if (
            isinstance(current, torch.Tensor)
            and isinstance(saved_value, torch.Tensor)
            and current.shape == saved_value.shape
            and current.dtype == saved_value.dtype
        ):
            # In place, because fused optimizers can hold references to state tensors.
            current.copy_(saved_value)
        else:
            values[key] = saved_value


@contextmanager
def frozen_param_groups(
    optimizer: Any, param_groups: Sequence[dict[str, Any]]
) -> Iterator[None]:
    """Keep the parameters and optimizer state of these groups unchanged for one step.

    Zero gradients alone do not freeze a parameter: Adam momentum and
    decoupled weight decay still move it. Both updates scale with the group
    learning rate, so the step runs with lr=0. The step still decays the Adam
    moments and advances the step count, so the context restores the group
    settings and the per-parameter state after the step. The scaling factors
    of a precision-aware optimizer are not restored.

    Args:
        optimizer: The optimizer that owns ``param_groups``. It can be a
            torch optimizer or a Megatron optimizer that wraps one or more.
        param_groups: Param groups of ``optimizer`` to freeze.

    Raises:
        ValueError: If ``optimizer`` does not own one of ``param_groups``.
    """
    owners = {
        id(group): torch_optimizer
        for torch_optimizer in _torch_optimizers(optimizer)
        for group in torch_optimizer.param_groups
    }
    if any(id(group) not in owners for group in param_groups):
        raise ValueError("the optimizer does not own a param group to freeze")

    with torch.no_grad():
        saved_groups = [_snapshot(group, skip_key="params") for group in param_groups]
        saved_states = [
            (
                owners[id(group)].state,
                parameter,
                _snapshot(owners[id(group)].state[parameter])
                if parameter in owners[id(group)].state
                else None,
            )
            for group in param_groups
            for parameter in group["params"]
        ]
    for group in param_groups:
        group["lr"] = 0.0
    try:
        yield
    finally:
        with torch.no_grad():
            for group, saved_group in zip(param_groups, saved_groups, strict=True):
                _restore(group, saved_group, skip_key="params")
            for state, parameter, saved_state in saved_states:
                if saved_state is None:
                    state.pop(parameter, None)
                else:
                    _restore(state.setdefault(parameter, {}), saved_state)
