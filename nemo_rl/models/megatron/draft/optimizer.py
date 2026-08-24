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

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterator, Mapping, Protocol

import torch
from megatron.bridge.training.config import (
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
)
from megatron.core.optimizer import ParamKey, ParamPredicate
from megatron.core.optimizer_param_scheduler import ParamGroupOverride

from nemo_rl.models.policy.draft_config import DraftOptimizerConfig


class DraftOptimizerConfigOwner(Protocol):
    """Draft config fields required to opt in to optimizer overrides."""

    enabled: bool
    optimizer: DraftOptimizerConfig | None


_DRAFT_OPTIMIZER_STEP_KEY = "nemo_rl_sparse_draft_optimizer_step"
_SPARSE_CHECKPOINTING_MARKER = "_nemo_rl_sparse_draft_checkpointing"


def _is_draft_parameter(parameter: torch.nn.Parameter) -> bool:
    return getattr(parameter, "grad_norm_group", None) == "draft"


@dataclass
class DraftOptimizerConfigOverrideProvider(OptimizerConfigOverrideProvider):
    """Add scheduler overrides only for parameters owned by a draft model."""

    draft_optimizer: DraftOptimizerConfig | None

    def build_config_overrides(
        self, context: OptimizerConfigOverrideProviderContext
    ) -> dict[ParamKey, ParamGroupOverride] | None:
        """Combine standard Megatron groups with the opt-in draft schedule."""
        overrides = super().build_config_overrides(context) or {}
        draft_override = ParamGroupOverride()
        if self.draft_optimizer is not None:
            minimum_lr = self.draft_optimizer.min_lr
            if minimum_lr is None:
                minimum_lr = context.optimizer_config.min_lr
            if minimum_lr is not None and minimum_lr > self.draft_optimizer.lr:
                raise ValueError(
                    "draft optimizer lr must be at least the inherited optimizer min_lr"
                )

            draft_override["max_lr"] = self.draft_optimizer.lr
            if minimum_lr is not None:
                draft_override["min_lr"] = minimum_lr
            if self.draft_optimizer.weight_decay is not None:
                draft_override["start_wd"] = self.draft_optimizer.weight_decay
                draft_override["end_wd"] = self.draft_optimizer.weight_decay

        overrides[
            ParamKey(
                predicate=ParamPredicate(
                    name="draft_parameter",
                    fn=_is_draft_parameter,
                )
            )
        ] = draft_override
        return overrides


def build_draft_optimizer_override_provider(
    draft_config: DraftOptimizerConfigOwner | None,
) -> DraftOptimizerConfigOverrideProvider | None:
    """Build a draft group selector whenever online draft training is enabled."""
    if draft_config is None or not draft_config.enabled:
        return None
    return DraftOptimizerConfigOverrideProvider(draft_config.optimizer)


def _optimizer_param_group_owners(optimizer: Any) -> tuple[Any, ...]:
    chained_optimizers = getattr(optimizer, "chained_optimizers", None)
    if chained_optimizers is not None:
        return tuple(
            owner
            for chained_optimizer in chained_optimizers
            for owner in _optimizer_param_group_owners(chained_optimizer)
        )
    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    return () if base_optimizer is None else (base_optimizer,)


def _optimizer_leaves(optimizer: Any) -> tuple[Any, ...]:
    chained_optimizers = getattr(optimizer, "chained_optimizers", None)
    if chained_optimizers is None:
        return (optimizer,)
    return tuple(
        leaf
        for chained_optimizer in chained_optimizers
        for leaf in _optimizer_leaves(chained_optimizer)
    )


def _parameters(optimizer: Any) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )


def _group_kind(group: Mapping[str, Any]) -> str:
    parameters = tuple(group.get("params", ()))
    has_draft = any(_is_draft_parameter(parameter) for parameter in parameters)
    has_policy = any(not _is_draft_parameter(parameter) for parameter in parameters)
    if has_draft and has_policy:
        raise RuntimeError(
            "optimizer parameter group mixes policy and draft parameters"
        )
    return "draft" if has_draft else "policy"


def _copy_step(container: Mapping[str, Any]) -> Any:
    step = container["step"]
    return step.detach().clone() if isinstance(step, torch.Tensor) else step


def _set_step(container: dict[str, Any], step: int) -> None:
    current = container["step"]
    if isinstance(current, torch.Tensor):
        current.fill_(step)
    else:
        container["step"] = step


def initialize_sparse_draft_optimizer_state(optimizer: Any) -> int:
    """Allocate lazy draft optimizer state without advancing or changing weights.

    Megatron's distributed checkpoint path expects every owned parameter to have
    optimizer tensors. Sparse cadence deliberately omits draft parameters from
    most optimizer steps, so a static drafter can otherwise reach its first
    checkpoint without ``master_param``/Adam state. A zero-learning-rate step on
    only the missing draft parameters allocates that state, after which its step
    counter is restored to zero. Existing policy state and gradients are untouched.
    """
    initialized = 0
    for current in _optimizer_param_group_owners(optimizer):
        parameters = _parameters(current)
        missing = tuple(
            parameter
            for parameter in parameters
            if _is_draft_parameter(parameter) and not current.state.get(parameter)
        )
        if not missing:
            continue

        saved_lrs = tuple(group.get("lr") for group in current.param_groups)
        saved_group_steps = tuple(
            ("step" in group, _copy_step(group) if "step" in group else None)
            for group in current.param_groups
        )
        saved_gradients = tuple(
            (
                parameter,
                parameter.grad,
                hasattr(parameter, "decoupled_grad"),
                getattr(parameter, "decoupled_grad", None),
            )
            for parameter in parameters
        )
        sub_optimizer_requires_grad = tuple(
            (parameter, parameter.requires_grad)
            for sub_optimizer in getattr(current, "sub_optimizers", ())
            for parameter in _parameters(sub_optimizer)
        )
        step_succeeded = False
        try:
            for group in current.param_groups:
                group["lr"] = 0.0
            for parameter, _, has_decoupled_grad, _ in saved_gradients:
                parameter.grad = None
                if has_decoupled_grad:
                    parameter.decoupled_grad = None
            for parameter in missing:
                gradient = torch.zeros_like(parameter)
                if hasattr(parameter, "decoupled_grad"):
                    parameter.decoupled_grad = gradient
                else:
                    parameter.grad = gradient

            current.step()
            step_succeeded = True
            current.zero_grad(set_to_none=True)
        finally:
            for group, lr in zip(current.param_groups, saved_lrs, strict=True):
                if lr is None:
                    group.pop("lr", None)
                else:
                    group["lr"] = lr
            for group, (had_step, saved_step) in zip(
                current.param_groups, saved_group_steps, strict=True
            ):
                if step_succeeded and _group_kind(group) == "draft":
                    if "step" in group:
                        _set_step(group, 0)
                elif had_step:
                    if "step" not in group:
                        group["step"] = saved_step
                    elif isinstance(group["step"], torch.Tensor):
                        group["step"].copy_(saved_step)
                    else:
                        group["step"] = saved_step
                else:
                    group.pop("step", None)
            for (
                parameter,
                gradient,
                has_decoupled_grad,
                decoupled_gradient,
            ) in saved_gradients:
                parameter.grad = gradient
                if has_decoupled_grad:
                    parameter.decoupled_grad = decoupled_gradient
            for parameter, requires_grad in sub_optimizer_requires_grad:
                parameter.requires_grad = requires_grad

        for parameter in missing:
            state = current.state.get(parameter)
            if not state:
                raise RuntimeError(
                    "draft optimizer state remained uninitialized after a zero-LR "
                    "initialization step"
                )
            step = state.get("step")
            if isinstance(step, torch.Tensor):
                step.zero_()
            elif step is not None:
                state["step"] = 0
        initialized += len(missing)
    return initialized


def _step_states(optimizer: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        state
        for state in optimizer.state.values()
        if isinstance(state, dict) and "step" in state
    )


def _step_value(container: Mapping[str, Any]) -> int:
    step = container["step"]
    return int(step.item()) if isinstance(step, torch.Tensor) else int(step)


def _step_containers(optimizer: Any) -> tuple[dict[str, Any], ...]:
    group_steps = tuple(group for group in optimizer.param_groups if "step" in group)
    return (*_step_states(optimizer), *group_steps)


@contextmanager
def _uniform_optimizer_step_header(optimizer: Any) -> Iterator[None]:
    containers = _step_containers(optimizer)
    if not containers:
        yield
        return
    canonical_step = max(_step_value(container) for container in containers)
    saved_steps = tuple((container, _copy_step(container)) for container in containers)
    try:
        for container, _ in saved_steps:
            _set_step(container, canonical_step)
        yield
    finally:
        for container, saved_step in saved_steps:
            current = container["step"]
            if isinstance(current, torch.Tensor):
                current.copy_(saved_step)
            else:
                container["step"] = saved_step


def _draft_optimizer_step(optimizer: Any) -> int | None:
    draft_parameters = tuple(
        parameter
        for parameter in _parameters(optimizer)
        if _is_draft_parameter(parameter)
    )
    if not draft_parameters:
        return None
    steps: set[int] = set()
    missing_steps: list[torch.nn.Parameter] = []
    for group in optimizer.param_groups:
        if _group_kind(group) != "draft":
            continue
        if "step" in group:
            steps.add(_step_value(group))
            continue
        for parameter in group.get("params", ()):
            state = optimizer.state.get(parameter, {})
            if "step" in state:
                steps.add(_step_value(state))
            else:
                missing_steps.append(parameter)
    if missing_steps:
        raise RuntimeError(
            "draft optimizer checkpoint state is incomplete; call "
            "initialize_sparse_draft_optimizer_state after optimizer setup"
        )
    if len(steps) != 1:
        raise RuntimeError("draft optimizer parameters have inconsistent step counters")
    return steps.pop()


def _restore_draft_optimizer_step(optimizer: Any, step: int) -> None:
    for group in optimizer.param_groups:
        if _group_kind(group) != "draft":
            continue
        restored = False
        if "step" in group:
            _set_step(group, step)
            restored = True
        for parameter in group.get("params", ()):
            state = optimizer.state.get(parameter, {})
            if "step" in state:
                _set_step(state, step)
                restored = True
        if not restored:
            raise RuntimeError("loaded draft optimizer state has no step counter")
    sync_to_sub_optimizers = getattr(
        optimizer, "_sync_hdo_state_to_sub_optimizers", None
    )
    if sync_to_sub_optimizers is not None:
        sync_to_sub_optimizers()


def install_sparse_draft_optimizer_checkpointing(optimizer: Any) -> int:
    """Preserve independent policy and draft Adam clocks in MCore checkpoints.

    MCore serializes one optimizer step in its non-parameter header and omits
    per-parameter ``step`` tensors from every distributed checkpoint format.
    Sparse draft updates therefore need a second header value. The installed
    wrappers present a uniform step only while MCore builds its legacy header,
    store the real draft step alongside it, and restore that value after load.
    """
    installed = 0
    for leaf in _optimizer_leaves(optimizer):
        if getattr(leaf, _SPARSE_CHECKPOINTING_MARKER, False):
            continue
        current = getattr(leaf, "optimizer", leaf)
        if _draft_optimizer_step(current) is None:
            continue

        original_state_dict = leaf.state_dict
        original_load_state_dict = leaf.load_state_dict

        def compatible_state_dict(
            _leaf: Any,
            *args: Any,
            _current: Any = current,
            _original_state_dict: Any = original_state_dict,
            **kwargs: Any,
        ) -> dict[str, Any]:
            del _leaf
            draft_step = _draft_optimizer_step(_current)
            with _uniform_optimizer_step_header(_current):
                state_dict = _original_state_dict(*args, **kwargs)
            state_dict[_DRAFT_OPTIMIZER_STEP_KEY] = draft_step
            return state_dict

        def compatible_load_state_dict(
            _leaf: Any,
            state_dict: Mapping[str, Any],
            *args: Any,
            _current: Any = current,
            _original_load_state_dict: Any = original_load_state_dict,
            **kwargs: Any,
        ) -> Any:
            del _leaf
            result = _original_load_state_dict(state_dict, *args, **kwargs)
            draft_step = state_dict.get(_DRAFT_OPTIMIZER_STEP_KEY)
            if draft_step is not None:
                _restore_draft_optimizer_step(_current, int(draft_step))
            return result

        leaf.state_dict = MethodType(compatible_state_dict, leaf)
        leaf.load_state_dict = MethodType(compatible_load_state_dict, leaf)
        setattr(leaf, _SPARSE_CHECKPOINTING_MARKER, True)
        installed += 1
    return installed


@contextmanager
def suspend_draft_optimizer_groups(optimizer: Any) -> Iterator[None]:
    """Temporarily remove homogeneous draft-only optimizer parameter groups."""
    planned: list[tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for current in _optimizer_param_group_owners(optimizer):
        original = list(current.param_groups)
        kept: list[dict[str, Any]] = []
        for group in original:
            parameters = group.get("params", ())
            has_draft = any(_is_draft_parameter(parameter) for parameter in parameters)
            has_policy = any(
                not _is_draft_parameter(parameter) for parameter in parameters
            )
            if has_draft and has_policy:
                raise RuntimeError(
                    "optimizer parameter group mixes policy and draft parameters"
                )
            if not has_draft:
                kept.append(group)
        planned.append((current, original, kept))

    try:
        for current, _, kept in planned:
            current.param_groups[:] = kept
        yield
    finally:
        for current, original, _ in reversed(planned):
            current.param_groups[:] = original
