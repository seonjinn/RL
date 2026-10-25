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
"""Decide on which policy steps the online draft model trains.

The trainer calls ``observe`` with the rollout metrics of a step and then
``begin_step`` before the policy update. The scheduler pauses or resumes the
draft update on the policy workers only when its decision changes.

The scheduler state is not saved in checkpoints. A resumed run starts with
the draft update enabled.
"""

import logging
import math
from collections.abc import Mapping
from typing import Any, Protocol

from nemo_rl.models.policy.draft_config import (
    AcceptanceDraftUpdateSchedule,
    IntervalDraftUpdateSchedule,
    coerce_draft_config,
)

ACCEPTANCE_LENGTH_METRIC = "vllm/spec_acceptance_length"
NUM_DRAFTS_METRIC = "vllm/spec_num_drafts"

log = logging.getLogger(__name__)


class DraftUpdateTarget(Protocol):
    def set_draft_update(self, enabled: bool) -> None: ...


class DraftUpdatePolicy(Protocol):
    def observe(self, acceptance_length: float) -> None: ...

    def should_update(self, step: int) -> bool: ...

    def metrics(self) -> dict[str, float]: ...


class IntervalDraftUpdatePolicy:
    """Update the draft on steps that are a multiple of the interval."""

    def __init__(self, schedule: IntervalDraftUpdateSchedule):
        self.interval = schedule.interval

    def observe(self, acceptance_length: float) -> None:
        del acceptance_length

    def should_update(self, step: int) -> bool:
        return step % self.interval == 0

    def metrics(self) -> dict[str, float]:
        return {}


class AcceptanceDraftUpdatePolicy:
    """Pause the draft update while the rollout acceptance length holds steady.

    While the draft trains, the policy tracks the best smoothed acceptance
    length. After ``patience`` observations without a relative gain of
    ``min_gain``, it pauses the draft. A paused draft resumes when the
    acceptance length falls by ``drift_tolerance`` below its value at the
    pause, or after ``max_skip_steps`` paused steps.

    A resume clears the best value. The acceptance length at the resume then
    sets a new reference, so the next pause needs ``patience`` more
    observations without gain. A best value kept from before the pause would
    count every recovering observation as stale and pause the draft again
    before it recovers.
    """

    def __init__(self, schedule: AcceptanceDraftUpdateSchedule):
        self.schedule = schedule
        self.ema: float | None = None
        self.best: float | None = None
        self.stale_observations = 0
        self.paused = False
        self.pause_reference = 0.0
        self.paused_steps = 0

    def observe(self, acceptance_length: float) -> None:
        if self.ema is None:
            self.ema = acceptance_length
        else:
            weight = self.schedule.ema
            self.ema = weight * acceptance_length + (1.0 - weight) * self.ema
        if self.paused:
            drift_floor = self.pause_reference * (1.0 - self.schedule.drift_tolerance)
            if self.ema < drift_floor:
                self._resume()
            return
        if self.best is None or self.ema > self.best * (1.0 + self.schedule.min_gain):
            self.best = self.ema
            self.stale_observations = 0
            return
        self.stale_observations += 1
        if self.stale_observations >= self.schedule.patience:
            self.paused = True
            self.pause_reference = self.ema
            self.paused_steps = 0

    def should_update(self, step: int) -> bool:
        del step
        max_skip_steps = self.schedule.max_skip_steps
        if (
            self.paused
            and max_skip_steps is not None
            and self.paused_steps >= max_skip_steps
        ):
            self._resume()
        if self.paused:
            self.paused_steps += 1
        return not self.paused

    def metrics(self) -> dict[str, float]:
        if self.ema is None:
            return {}
        return {"draft_update/acceptance_ema": self.ema}

    def _resume(self) -> None:
        self.paused = False
        self.best = None
        self.stale_observations = 0


class DraftUpdateScheduler:
    """Apply a draft update policy to the policy workers."""

    def __init__(self, target: DraftUpdateTarget, policy: DraftUpdatePolicy):
        self.target = target
        self.policy = policy
        # Workers start with draft updates enabled.
        self.enabled = True
        self._decided_step: int | None = None
        self._warned_missing_metrics = False

    def observe(self, metrics: Mapping[str, Any]) -> None:
        """Record the rollout acceptance length of one step, if the step drafted.

        With async rollouts, these metrics come from generations of an older
        policy, so the decision lags the draft weights by the rollout lag.
        """
        if ACCEPTANCE_LENGTH_METRIC not in metrics or NUM_DRAFTS_METRIC not in metrics:
            if (
                isinstance(self.policy, AcceptanceDraftUpdatePolicy)
                and not self._warned_missing_metrics
            ):
                log.warning(
                    "The acceptance draft update schedule found no %s or %s in "
                    "the rollout metrics. The draft keeps its current update state.",
                    ACCEPTANCE_LENGTH_METRIC,
                    NUM_DRAFTS_METRIC,
                )
                self._warned_missing_metrics = True
            return
        if float(metrics[NUM_DRAFTS_METRIC]) <= 0:
            return
        acceptance_length = float(metrics[ACCEPTANCE_LENGTH_METRIC])
        if not math.isfinite(acceptance_length):
            return
        self.policy.observe(acceptance_length)

    def begin_step(self, step: int) -> None:
        """Pause or resume the draft update before the policy update of ``step``.

        A repeated call for the same step keeps the first decision, so a
        retried step does not count as an extra paused step.
        """
        if step == self._decided_step:
            return
        self._decided_step = step
        enabled = self.policy.should_update(step)
        if enabled != self.enabled:
            self.target.set_draft_update(enabled)
            self.enabled = enabled

    def metrics(self) -> dict[str, float]:
        return {"draft_update/enabled": float(self.enabled), **self.policy.metrics()}


def build_draft_update_scheduler(
    draft_config: Any, target: DraftUpdateTarget
) -> DraftUpdateScheduler | None:
    """Return a scheduler, or None when the draft trains on every step."""
    draft = coerce_draft_config(draft_config)
    if draft is None or not draft.enabled:
        return None
    schedule = getattr(draft, "update_schedule", None)
    if isinstance(schedule, IntervalDraftUpdateSchedule):
        return DraftUpdateScheduler(target, IntervalDraftUpdatePolicy(schedule))
    if isinstance(schedule, AcceptanceDraftUpdateSchedule):
        return DraftUpdateScheduler(target, AcceptanceDraftUpdatePolicy(schedule))
    return None
