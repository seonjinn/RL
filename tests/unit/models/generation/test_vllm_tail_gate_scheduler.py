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

"""Unit tests for the vLLM tail-gated scheduler wrapper."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCHEDULER_MODULE = "nemo_rl.models.generation.vllm.tail_gate_scheduler"


class _StubScheduler:
    """Small vLLM Scheduler stand-in used without importing vLLM."""

    def __init__(self, vllm_config: SimpleNamespace, *_args: object, **_kwargs: object) -> None:
        self.vllm_config = vllm_config
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_sampled_tokens_per_step = 1
        self.running: list[SimpleNamespace] = []
        self.waiting: list[SimpleNamespace] = []
        self.skipped_waiting: list[SimpleNamespace] = []
        self.schedule_outputs: list[SimpleNamespace] = []
        self.schedule_throttle_prefills: list[bool] = []
        self.update_result: object = object()

    def schedule(self, throttle_prefills: bool = False) -> SimpleNamespace:
        self.schedule_throttle_prefills.append(throttle_prefills)
        return self.schedule_outputs.pop(0)

    def update_from_output(
        self,
        _scheduler_output: SimpleNamespace,
        _model_runner_output: SimpleNamespace,
    ) -> object:
        return self.update_result

    def get_num_unfinished_requests(self) -> int:
        return len(self.running) + len(self.waiting) + len(self.skipped_waiting)


@pytest.fixture
def scheduler_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the scheduler with only the vLLM types it directly needs."""
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    v1 = ModuleType("vllm.v1")
    v1.__path__ = []
    core = ModuleType("vllm.v1.core")
    core.__path__ = []
    sched = ModuleType("vllm.v1.core.sched")
    sched.__path__ = []
    scheduler = ModuleType("vllm.v1.core.sched.scheduler")
    scheduler.Scheduler = _StubScheduler

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
            sd_tail_gate_threshold=2,
            sd_tail_gate_consecutive_checks=1,
            sd_tail_gate_margin=0.05,
            num_speculative_tokens=5,
        ),
    )


def _scheduler_output(*, drafts: dict[str, list[int]] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_spec_decode_tokens=drafts or {},
        num_spec_tokens_to_schedule=5,
    )


def test_schedule_gates_the_next_proposal_and_preserves_pending_drafts(
    scheduler_module: ModuleType,
) -> None:
    scheduler = scheduler_module.TailGatedScheduler(_vllm_config())
    scheduler.running = [
        SimpleNamespace(is_prefill_chunk=False, num_tokens=10),
        SimpleNamespace(is_prefill_chunk=False, num_tokens=12),
        SimpleNamespace(is_prefill_chunk=False, num_tokens=14),
    ]
    pending_drafts = {"request-1": [1, 2, 3]}
    scheduler.schedule_outputs = [_scheduler_output(drafts=pending_drafts)]

    output = scheduler.schedule(throttle_prefills=True)

    assert scheduler.schedule_throttle_prefills == [True]
    assert output.num_spec_tokens_to_schedule == 0
    assert output.scheduled_spec_decode_tokens == pending_drafts
    assert output.tail_gate_state == "ARMED_OFF"
    assert output.tail_gate_tick == 1
    assert output.tail_gate_active_requests == 3
    assert output.tail_gate_decode_active_requests == 3
    assert output.tail_gate_mean_sequence_length == 12
    assert output.tail_gate_predicted_speedup_sum == 0.0
    assert output.tail_gate_predicted_speedup_count == 0
    assert output.tail_gate_expected_accept_length == 3.0
    assert output.tail_gate_just_activated is False

    scheduler.running.pop()
    scheduler.schedule_outputs = [_scheduler_output()]

    activated_output = scheduler.schedule()

    assert activated_output.num_spec_tokens_to_schedule == 5
    assert activated_output.tail_gate_state == "ON_LATCHED"
    assert activated_output.tail_gate_tick == 2
    assert activated_output.tail_gate_active_requests == 2
    assert activated_output.tail_gate_mean_sequence_length == 11
    assert activated_output.tail_gate_just_activated is True


def test_update_records_acceptance_before_resetting_after_skipped_waiting_drains(
    scheduler_module: ModuleType,
) -> None:
    scheduler = scheduler_module.TailGatedScheduler(_vllm_config())
    scheduler.running = [
        SimpleNamespace(is_prefill_chunk=False, num_tokens=10),
        SimpleNamespace(is_prefill_chunk=False, num_tokens=10),
        SimpleNamespace(is_prefill_chunk=False, num_tokens=10),
    ]
    scheduler.schedule_outputs = [_scheduler_output()]
    scheduler.schedule()
    scheduler.running.pop()
    scheduler.schedule_outputs = [_scheduler_output()]
    scheduler.schedule()
    assert scheduler._tail_gate.enabled is True

    scheduler.running.clear()
    scheduler.skipped_waiting.append(SimpleNamespace())
    scheduler_output = _scheduler_output(drafts={"request-1": [1, 2, 3]})
    model_runner_output = SimpleNamespace(
        req_ids=["request-1"],
        req_id_to_index={"request-1": 0},
        sampled_token_ids=[[10, 11, 12]],
    )

    result = scheduler.update_from_output(scheduler_output, model_runner_output)

    assert result is scheduler.update_result
    assert scheduler._tail_gate.enabled is True

    scheduler.skipped_waiting.clear()
    result = scheduler.update_from_output(
        _scheduler_output(),
        SimpleNamespace(req_ids=[], req_id_to_index={}, sampled_token_ids=[]),
    )

    assert result is scheduler.update_result
    assert scheduler._tail_gate.enabled is False
    assert scheduler._tail_gate.expected_accept_length == 3.0
