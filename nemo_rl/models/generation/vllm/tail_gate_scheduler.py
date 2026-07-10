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

"""vLLM scheduler wrapper for monotone tail-gated speculative decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler

from nemo_rl.models.generation.vllm.sd_toggle import load_config
from nemo_rl.models.generation.vllm.tail_gate import (
    TailGateConfig,
    TailGateController,
    TailGateObservation,
)


class TailGatedScheduler(Scheduler):
    """Applies a rollout-local tail gate to vLLM's next proposal step."""

    def __init__(self, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            vllm_config,  # pyrefly: ignore[bad-argument-count]  vLLM is an external runtime dependency
            *args,
            **kwargs,
        )
        speculative_config = vllm_config.speculative_config
        if speculative_config is None:
            raise ValueError("TailGatedScheduler requires SpeculativeConfig")
        self._tail_gate = TailGateController(
            _tail_gate_config(speculative_config, self.num_spec_tokens)
        )
        self._accepted_tokens = 0
        self._draft_cycles = 0
        self._failed_output_request_ids: set[str] = set()

    def schedule(self, throttle_prefills: bool = False) -> Any:
        """Schedule normally, then gate only the following proposal cycle."""
        output = super().schedule(throttle_prefills=throttle_prefills)
        observation, decode_active_batch_size = self._build_observation()
        decision = self._tail_gate.observe(observation)
        output.num_spec_tokens_to_schedule = (
            self.num_spec_tokens if decision.enabled else 0
        )
        self._write_tail_gate_output(
            output,
            observation=observation,
            decode_active_batch_size=decode_active_batch_size,
            decision=decision,
        )
        return output

    def update_from_output(
        self, scheduler_output: Any, model_runner_output: Any
    ) -> Any:
        """Accumulate acceptance feedback while preserving Scheduler semantics."""
        acceptance_snapshot = self._snapshot_acceptance(
            scheduler_output, model_runner_output
        )
        self._failed_output_request_ids.clear()
        result = super().update_from_output(scheduler_output, model_runner_output)
        self._record_acceptance(
            acceptance_snapshot,
            self._failed_output_request_ids,
        )
        if self.get_num_unfinished_requests() == 0:
            self._tail_gate.finish_rollout(
                accepted_tokens=self._accepted_tokens,
                num_drafts=self._draft_cycles,
                validation=False,
            )
            self._accepted_tokens = 0
            self._draft_cycles = 0
        return result

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        failed_request_ids = super()._handle_invalid_blocks(
            invalid_block_ids, num_scheduled_tokens
        )
        self._failed_output_request_ids.update(failed_request_ids)
        return failed_request_ids

    def _build_observation(self) -> tuple[TailGateObservation, int]:
        active_requests = len(self.running)
        decode_active_batch_size = sum(
            not request.is_prefill_chunk for request in self.running
        )
        mean_sequence_length = (
            sum(request.num_tokens for request in self.running) // active_requests
            if active_requests
            else 0
        )
        return (
            TailGateObservation(
                active_requests=active_requests,
                mean_sequence_length=mean_sequence_length,
                is_decode=decode_active_batch_size > 0,
            ),
            decode_active_batch_size,
        )

    def _snapshot_acceptance(
        self, scheduler_output: Any, model_runner_output: Any
    ) -> dict[str, int]:
        acceptance_snapshot: dict[str, int] = {}
        for request_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(request_id)
            if request is None or request.is_finished():
                continue
            if request_id not in model_runner_output.req_id_to_index:
                continue
            draft_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                request_id
            )
            if not draft_token_ids:
                continue
            request_index = model_runner_output.req_id_to_index[request_id]
            sampled_token_ids = (
                model_runner_output.sampled_token_ids[request_index]
                if model_runner_output.sampled_token_ids
                else []
            )
            if not sampled_token_ids:
                continue
            acceptance_snapshot[request_id] = max(
                len(sampled_token_ids) - self.num_sampled_tokens_per_step,
                0,
            )
        return acceptance_snapshot

    def _record_acceptance(
        self,
        acceptance_snapshot: dict[str, int],
        failed_request_ids: set[str],
    ) -> None:
        for request_id, accepted_tokens in acceptance_snapshot.items():
            if request_id in failed_request_ids:
                continue
            self._accepted_tokens += accepted_tokens
            self._draft_cycles += 1

    def _write_tail_gate_output(
        self,
        output: Any,
        *,
        observation: TailGateObservation,
        decode_active_batch_size: int,
        decision: Any,
    ) -> None:
        telemetry = decision.telemetry
        predicted_speedup = telemetry.predicted_speedup
        output.tail_gate_state = telemetry.state
        output.tail_gate_tick = telemetry.tick
        output.tail_gate_active_requests = observation.active_requests
        output.tail_gate_decode_active_requests = decode_active_batch_size
        output.tail_gate_mean_sequence_length = observation.mean_sequence_length
        output.tail_gate_predicted_speedup_sum = predicted_speedup or 0.0
        output.tail_gate_predicted_speedup_count = int(predicted_speedup is not None)
        output.tail_gate_expected_accept_length = telemetry.expected_accept_length
        output.tail_gate_just_activated = decision.just_activated


def _tail_gate_config(speculative_config: Any, gamma: int) -> TailGateConfig:
    """Build a controller config from the patched vLLM SpeculativeConfig."""
    mode = _required_speculative_field(speculative_config, "sd_tail_gate_mode")
    threshold = _required_speculative_field(
        speculative_config, "sd_tail_gate_threshold"
    )
    consecutive_checks = _required_speculative_field(
        speculative_config, "sd_tail_gate_consecutive_checks"
    )
    margin = _required_speculative_field(speculative_config, "sd_tail_gate_margin")
    roofline_config = None
    if mode == "roofline":
        config_path = _required_speculative_field(
            speculative_config, "sd_tail_gate_config_path"
        )
        if not isinstance(config_path, str) or not config_path:
            raise ValueError("sd_tail_gate_config_path must be a non-empty string")
        roofline_config = load_config(Path(config_path))
    return TailGateConfig(
        mode=mode,
        threshold=threshold,
        consecutive_checks=consecutive_checks,
        gamma=gamma,
        margin=margin,
        roofline_config=roofline_config,
    )


def _required_speculative_field(speculative_config: Any, name: str) -> Any:
    value = getattr(speculative_config, name, None)
    if value is None:
        raise ValueError(f"TailGatedScheduler requires SpeculativeConfig.{name}")
    return value
