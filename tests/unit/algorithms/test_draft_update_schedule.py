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

import logging
from typing import Any

import pytest

from nemo_rl.algorithms.draft_update_schedule import (
    ACCEPTANCE_LENGTH_METRIC,
    NUM_DRAFTS_METRIC,
    AcceptanceDraftUpdatePolicy,
    DraftUpdateScheduler,
    IntervalDraftUpdatePolicy,
    build_draft_update_scheduler,
)
from nemo_rl.models.policy.draft_config import (
    AcceptanceDraftUpdateSchedule,
    IntervalDraftUpdateSchedule,
    MegatronDFlashDraftConfig,
)


class RecordingTarget:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def set_draft_update(self, enabled: bool) -> None:
        self.calls.append(enabled)


def _rollout_metrics(acceptance_length: float, num_drafts: float = 10.0) -> dict:
    return {
        ACCEPTANCE_LENGTH_METRIC: acceptance_length,
        NUM_DRAFTS_METRIC: num_drafts,
    }


def _acceptance_scheduler(
    target: RecordingTarget, **schedule: Any
) -> DraftUpdateScheduler:
    # ema=1.0 makes the smoothed value equal the latest observation.
    values = {"min_gain": 0.02, "patience": 2, "drift_tolerance": 0.05, "ema": 1.0}
    values.update(schedule)
    policy = AcceptanceDraftUpdatePolicy(AcceptanceDraftUpdateSchedule(**values))
    return DraftUpdateScheduler(target, policy)


def _dflash_config(**values: Any) -> MegatronDFlashDraftConfig:
    return MegatronDFlashDraftConfig(
        enabled=True,
        gamma=5,
        anchors_per_sample=4,
        mask_token_id=151665,
        target_hidden_state_layer_ids=[1, 17, 33],
        **values,
    )


def test_interval_updates_on_multiples_and_calls_target_only_on_change() -> None:
    target = RecordingTarget()
    scheduler = DraftUpdateScheduler(
        target, IntervalDraftUpdatePolicy(IntervalDraftUpdateSchedule(interval=5))
    )

    enabled_steps = []
    for step in range(11):
        scheduler.begin_step(step)
        if scheduler.enabled:
            enabled_steps.append(step)

    assert enabled_steps == [0, 5, 10]
    assert target.calls == [False, True, False, True]


def test_acceptance_pauses_after_patience_and_resumes_on_drift() -> None:
    target = RecordingTarget()
    scheduler = _acceptance_scheduler(target)

    scheduler.observe(_rollout_metrics(2.0))
    scheduler.begin_step(0)
    scheduler.observe(_rollout_metrics(2.01))
    scheduler.begin_step(1)
    assert target.calls == []

    scheduler.observe(_rollout_metrics(2.02))
    scheduler.begin_step(2)
    assert target.calls == [False]

    # 1.95 stays above the drift floor 2.02 * 0.95 = 1.919.
    scheduler.observe(_rollout_metrics(1.95))
    scheduler.begin_step(3)
    assert target.calls == [False]

    scheduler.observe(_rollout_metrics(1.90))
    scheduler.begin_step(4)
    assert target.calls == [False, True]
    assert scheduler.metrics() == {
        "draft_update/enabled": 1.0,
        "draft_update/acceptance_ema": 1.90,
    }


def test_acceptance_keeps_training_while_acceptance_improves() -> None:
    target = RecordingTarget()
    scheduler = _acceptance_scheduler(target)

    for step, acceptance_length in enumerate([2.0, 2.1, 2.2, 2.3, 2.4]):
        scheduler.observe(_rollout_metrics(acceptance_length))
        scheduler.begin_step(step)

    assert target.calls == []
    assert scheduler.enabled


def test_acceptance_resume_requires_fresh_patience_before_next_pause() -> None:
    target = RecordingTarget()
    scheduler = _acceptance_scheduler(target)

    for acceptance_length in [2.0, 2.0, 2.0]:
        scheduler.observe(_rollout_metrics(acceptance_length))
    scheduler.begin_step(0)
    scheduler.observe(_rollout_metrics(1.5))
    scheduler.begin_step(1)
    assert target.calls == [False, True]

    # The resume clears the best value, so 1.5 becomes the new reference and
    # the next pause needs patience=2 more steps without gain.
    for step, acceptance_length in enumerate([1.5, 1.5], start=2):
        scheduler.observe(_rollout_metrics(acceptance_length))
        scheduler.begin_step(step)
    assert target.calls == [False, True]
    scheduler.observe(_rollout_metrics(1.5))
    scheduler.begin_step(4)
    assert target.calls == [False, True, False]


def test_acceptance_resumes_after_max_skip_steps() -> None:
    target = RecordingTarget()
    scheduler = _acceptance_scheduler(target, max_skip_steps=2)

    for acceptance_length in [2.0, 2.0, 2.0]:
        scheduler.observe(_rollout_metrics(acceptance_length))

    decisions = []
    for step in range(3):
        scheduler.begin_step(step)
        decisions.append(scheduler.enabled)

    assert decisions == [False, False, True]
    assert target.calls == [False, True]


@pytest.mark.parametrize(
    "metrics",
    [
        {ACCEPTANCE_LENGTH_METRIC: 1.0, NUM_DRAFTS_METRIC: 0.0},
        {ACCEPTANCE_LENGTH_METRIC: 1.0},
        {},
    ],
)
def test_steps_without_drafts_are_not_observed(metrics: dict) -> None:
    target = RecordingTarget()
    scheduler = _acceptance_scheduler(target)

    for _ in range(5):
        scheduler.observe(metrics)
        scheduler.begin_step(0)

    assert target.calls == []
    assert scheduler.metrics() == {"draft_update/enabled": 1.0}


@pytest.mark.parametrize("acceptance_length", [float("nan"), float("inf")])
def test_non_finite_acceptance_lengths_are_not_observed(
    acceptance_length: float,
) -> None:
    target = RecordingTarget()
    scheduler = _acceptance_scheduler(target)

    for step in range(5):
        scheduler.observe(_rollout_metrics(acceptance_length))
        scheduler.begin_step(step)

    assert target.calls == []
    assert scheduler.metrics() == {"draft_update/enabled": 1.0}


def test_repeated_begin_step_keeps_the_first_decision() -> None:
    """A retried step must not count as an extra paused step."""
    target = RecordingTarget()
    scheduler = _acceptance_scheduler(target, max_skip_steps=2)
    for acceptance_length in [2.0, 2.0, 2.0]:
        scheduler.observe(_rollout_metrics(acceptance_length))

    decisions = []
    for step in [0, 0, 0, 1, 1, 2]:
        scheduler.begin_step(step)
        decisions.append(scheduler.enabled)

    assert decisions == [False, False, False, False, False, True]
    assert target.calls == [False, True]


def test_missing_acceptance_metrics_warn_once(caplog: pytest.LogCaptureFixture) -> None:
    scheduler = _acceptance_scheduler(RecordingTarget())

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            scheduler.observe({})

    assert len(caplog.records) == 1
    assert ACCEPTANCE_LENGTH_METRIC in caplog.records[0].getMessage()


def test_interval_schedule_does_not_warn_without_acceptance_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scheduler = DraftUpdateScheduler(
        RecordingTarget(),
        IntervalDraftUpdatePolicy(IntervalDraftUpdateSchedule(interval=5)),
    )

    with caplog.at_level(logging.WARNING):
        scheduler.observe({})

    assert caplog.records == []


@pytest.mark.parametrize(
    "draft_config",
    [
        None,
        {"enabled": False},
        _dflash_config(),
    ],
)
def test_builder_returns_none_when_draft_trains_every_step(draft_config: Any) -> None:
    assert build_draft_update_scheduler(draft_config, RecordingTarget()) is None


@pytest.mark.parametrize(
    ("update_schedule", "policy_type"),
    [
        ({"type": "interval", "interval": 5}, IntervalDraftUpdatePolicy),
        ({"type": "acceptance"}, AcceptanceDraftUpdatePolicy),
    ],
)
def test_builder_selects_policy_from_config(
    update_schedule: dict, policy_type: type
) -> None:
    draft_config = _dflash_config(
        optimizer={"lr": 1.0e-4}, update_schedule=update_schedule
    )

    scheduler = build_draft_update_scheduler(draft_config, RecordingTarget())

    assert scheduler is not None
    assert isinstance(scheduler.policy, policy_type)
