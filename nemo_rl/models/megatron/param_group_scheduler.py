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

import copy
from typing import Any, cast

from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,
    ParamGroupOverride,
)

LR_WARMUP_STEPS_KEY = "lr_warmup_steps"


class ParamGroupScheduleOverride(ParamGroupOverride, total=False):
    """Megatron param-group override that can also set the group's own warmup."""

    # Same units as OptimizerParamScheduler.lr_warmup_steps (samples in NeMo RL).
    lr_warmup_steps: int


class ParamGroupWarmupScheduler(OptimizerParamScheduler):
    """Megatron LR scheduler whose param groups can set their own linear warmup.

    Megatron reads ``max_lr`` and ``min_lr`` per group but applies one global
    warmup to every group. A group that sets ``lr_warmup_steps`` runs the same
    warmup and decay shape with its own warmup length. Other groups keep the
    global schedule.
    """

    # Group position -> configured warmup. An optimizer checkpoint load replaces
    # the group dicts, and Megatron does not match groups by this key, so the
    # loaded groups can lose it or carry an old value. step() writes it back.
    group_warmup_steps: dict[int, int]

    def step(self, increment: int) -> None:
        param_groups = self.optimizer.param_groups
        for index, warmup_steps in self.group_warmup_steps.items():
            param_groups[index][LR_WARMUP_STEPS_KEY] = warmup_steps
        super().step(increment)

    def get_lr(self, param_group: dict[str, Any]) -> float:
        if LR_WARMUP_STEPS_KEY not in param_group:
            return super().get_lr(param_group)
        group_schedule = copy.copy(self)
        group_schedule.lr_warmup_steps = param_group[LR_WARMUP_STEPS_KEY]
        return OptimizerParamScheduler.get_lr(group_schedule, param_group)


def enable_param_group_warmup(scheduler: OptimizerParamScheduler) -> None:
    """Make a built scheduler honor per-group warmup when a group sets one.

    Megatron-Bridge builds the scheduler, so the class is switched in place.
    Call this before a checkpoint load. Megatron keeps the group order of the
    current optimizer on load, so the warmups are kept by position. The
    constructor already set every group's lr from the global warmup, so the
    current lrs are recomputed.
    """
    group_warmup_steps = {
        index: group[LR_WARMUP_STEPS_KEY]
        for index, group in enumerate(scheduler.optimizer.param_groups)
        if LR_WARMUP_STEPS_KEY in group
    }
    if not group_warmup_steps:
        return
    scheduler.__class__ = ParamGroupWarmupScheduler
    warmup_scheduler = cast(ParamGroupWarmupScheduler, scheduler)
    warmup_scheduler.group_warmup_steps = group_warmup_steps
    warmup_scheduler.step(increment=0)
