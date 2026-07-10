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

"""CPU lifecycle tests for the mini tail-gated vLLM scheduler contract."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from nemo_rl.models.generation.vllm.tail_gate import (
    TailGateConfig,
    TailGateController,
    TailGateObservation,
)


SCHEDULER_MODULE = "nemo_rl.models.generation.vllm.tail_gate_scheduler"


class _LocalScheduler:
    """Minimal CPU stand-in for only the external vLLM scheduler mechanics."""

    def __init__(
        self, vllm_config: SimpleNamespace, *_args: object, **_kwargs: object
    ) -> None:
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_sampled_tokens_per_step = 1
        self.requests: dict[str, _ActiveRequest] = {}
        self.running: list[SimpleNamespace] = []
        self.waiting: list[object] = []
        self.skipped_waiting: list[object] = []
        self.schedule_outputs: list[SimpleNamespace] = []

    def schedule(self, throttle_prefills: bool = False) -> SimpleNamespace:
        del throttle_prefills
        return self.schedule_outputs.pop(0)

    def update_from_output(
        self,
        scheduler_output: SimpleNamespace,
        model_runner_output: SimpleNamespace,
    ) -> None:
        del scheduler_output, model_runner_output

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        del invalid_block_ids, num_scheduled_tokens
        return set()

    def get_num_unfinished_requests(self) -> int:
        return len(self.running) + len(self.waiting) + len(self.skipped_waiting)


class _ActiveRequest:
    def is_finished(self) -> bool:
        return False


@pytest.fixture
def scheduler_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the production wrapper with its unavailable vLLM base replaced."""
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    v1 = ModuleType("vllm.v1")
    v1.__path__ = []
    core = ModuleType("vllm.v1.core")
    core.__path__ = []
    sched = ModuleType("vllm.v1.core.sched")
    sched.__path__ = []
    scheduler = ModuleType("vllm.v1.core.sched.scheduler")
    scheduler.Scheduler = _LocalScheduler

    vllm.v1 = v1
    v1.core = core
    core.sched = sched
    sched.scheduler = scheduler
    for name, module in {
        "vllm": vllm,
        "vllm.v1": v1,
        "vllm.v1.core": core,
        "vllm.v1.core.sched": sched,
        "vllm.v1.core.sched.scheduler": scheduler,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, SCHEDULER_MODULE, raising=False)
    return importlib.import_module(SCHEDULER_MODULE)


def _vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_speculative_tokens=5,
        speculative_config=SimpleNamespace(
            sd_tail_gate_mode="threshold",
            sd_tail_gate_threshold=4,
            sd_tail_gate_consecutive_checks=10,
            sd_tail_gate_margin=0.05,
        ),
    )


def _running_requests(count: int, *, prefill: bool = False) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(is_prefill_chunk=prefill, num_tokens=1024 + index)
        for index in range(count)
    ]


def _scheduler_output(*, drafts: dict[str, list[int]] | None = None) -> SimpleNamespace:
    scheduled_drafts = drafts or {}
    return SimpleNamespace(
        scheduled_spec_decode_tokens=scheduled_drafts,
        num_scheduled_tokens={request_id: 1 for request_id in scheduled_drafts},
        num_spec_tokens_to_schedule=5,
    )


def _model_runner_output(
    request_id: str | None = None,
    sampled_token_ids: list[int] | None = None,
) -> SimpleNamespace:
    if request_id is None:
        return SimpleNamespace(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
    return SimpleNamespace(
        req_ids=[request_id],
        req_id_to_index={request_id: 0},
        sampled_token_ids=[sampled_token_ids or []],
    )


def test_mini_scheduler_tail_gate_full_rollout_lifecycle(
    scheduler_module: ModuleType,
) -> None:
    scheduler: Any = scheduler_module.TailGatedScheduler(_vllm_config())
    scheduler.running = _running_requests(8)
    scheduler.schedule_outputs = [_scheduler_output() for _ in range(11)]
    scheduler.schedule_outputs.append(
        _scheduler_output(drafts={"decode-0": [101, 102, 103, 104, 105]})
    )

    outputs = [scheduler.schedule()]
    scheduler.running = _running_requests(4)
    outputs.extend(scheduler.schedule() for _ in range(10))
    scheduler.running = _running_requests(8)
    outputs.append(scheduler.schedule())

    assert (
        outputs[0].tail_gate_state,
        outputs[0].tail_gate_tick,
        outputs[0].tail_gate_active_requests,
        outputs[0].num_spec_tokens_to_schedule,
    ) == ("ARMED_OFF", 1, 8, 0)
    assert [output.num_spec_tokens_to_schedule for output in outputs] == [
        *([0] * 10),
        5,
        5,
    ]
    assert [
        (
            output.tail_gate_tick,
            output.tail_gate_active_requests,
            output.tail_gate_decode_active_requests,
        )
        for output in outputs
        if output.tail_gate_just_activated
    ] == [(11, 4, 4)]
    assert [
        (output.tail_gate_state, output.tail_gate_just_activated)
        for output in outputs[-2:]
    ] == [("ON_LATCHED", True), ("ON_LATCHED", False)]

    scheduler.requests["decode-0"] = _ActiveRequest()
    scheduler.update_from_output(
        outputs[-1],
        _model_runner_output("decode-0", [201, 202, 203, 204]),
    )

    assert (scheduler._accepted_tokens, scheduler._draft_cycles) == (3, 1)

    scheduler.running.clear()
    scheduler.update_from_output(_scheduler_output(), _model_runner_output())

    assert (
        scheduler._tail_gate.enabled,
        scheduler._tail_gate.telemetry.state,
        scheduler._tail_gate.telemetry.tick,
        scheduler._tail_gate._qualifying_checks,
        scheduler._tail_gate.expected_accept_length,
        scheduler._accepted_tokens,
        scheduler._draft_cycles,
    ) == (False, "RAMPING_OFF", 0, 0, 4.0, 0, 0)

    scheduler.running = _running_requests(8)
    scheduler.schedule_outputs = [_scheduler_output()]
    next_rollout = scheduler.schedule()

    assert (
        next_rollout.tail_gate_state,
        next_rollout.tail_gate_tick,
        next_rollout.num_spec_tokens_to_schedule,
        next_rollout.tail_gate_expected_accept_length,
        next_rollout.tail_gate_just_activated,
    ) == ("ARMED_OFF", 1, 0, 4.0, False)

    next_rollout_outputs = [next_rollout]
    for _ in range(10):
        scheduler.running = _running_requests(4)
        scheduler.schedule_outputs = [_scheduler_output()]
        next_rollout_outputs.append(scheduler.schedule())

    assert [output.num_spec_tokens_to_schedule for output in next_rollout_outputs] == [
        *([0] * 10),
        5,
    ]
    assert next_rollout_outputs[-1].tail_gate_just_activated is True


def test_all_prefill_batch_does_not_arm_tail_gate(
    scheduler_module: ModuleType,
) -> None:
    scheduler: Any = scheduler_module.TailGatedScheduler(_vllm_config())
    scheduler.running = _running_requests(8, prefill=True)
    scheduler.schedule_outputs = [_scheduler_output()]

    output = scheduler.schedule()

    assert (
        output.tail_gate_state,
        output.tail_gate_active_requests,
        output.tail_gate_decode_active_requests,
        output.num_spec_tokens_to_schedule,
        output.tail_gate_just_activated,
    ) == ("RAMPING_OFF", 8, 0, 0, False)


@pytest.mark.parametrize(
    "threshold",
    [pytest.param(8, id="at-capacity"), pytest.param(9, id="above-capacity")],
)
def test_threshold_at_or_above_capacity_is_left_to_launcher_validation(
    threshold: int,
) -> None:
    controller = TailGateController(
        TailGateConfig(
            mode="threshold",
            threshold=threshold,
            consecutive_checks=10,
            gamma=5,
        )
    )

    decision = controller.observe(
        TailGateObservation(
            active_requests=8,
            mean_sequence_length=1024,
            is_decode=True,
        )
    )

    assert (
        controller.config.threshold,
        decision.enabled,
        decision.reason,
        decision.telemetry.state,
    ) == (threshold, False, "ramp_guard", "RAMPING_OFF")
