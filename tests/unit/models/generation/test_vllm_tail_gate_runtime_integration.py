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

from __future__ import annotations

import ast
import copy
import math
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from nemo_rl.models.generation.vllm import patches
from nemo_rl.models.generation.vllm.utils import compute_spec_decode_metrics


_FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "vllm_v0_24_0"
_RUNTIME_PATHS = (
    "config/speculative.py",
    "v1/core/sched/output.py",
    "v1/worker/gpu/model_runner.py",
    "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
    "v1/worker/gpu_model_runner.py",
)
_SPEC_COUNTERS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)
_TELEMETRY_KEYS = (
    "vllm:spec_decode_tail_gate_decisions",
    "vllm:spec_decode_tail_gate_enabled_steps",
    "vllm:spec_decode_tail_gate_disabled_steps",
    "vllm:spec_decode_tail_gate_activations",
    "vllm:spec_decode_tail_gate_activation_batch_sum",
    "vllm:spec_decode_tail_gate_activation_tick_sum",
    "vllm:spec_decode_tail_gate_activation_tick_count",
    "vllm:spec_decode_tail_gate_k_0_steps",
    "vllm:spec_decode_tail_gate_k_5_steps",
)

type _CounterKey = str | tuple[str, int]


class _StopAfterTelemetry(RuntimeError):
    pass


class _ModelRunnerOutput:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _AsyncOutput:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _DraftTokenHandler:
    def __init__(self) -> None:
        self.published: list[torch.Tensor] = []

    def set_draft_tokens(
        self, _input_batch: SimpleNamespace, draft_tokens: torch.Tensor
    ) -> None:
        self.published.append(draft_tokens.clone())


class _V1Runner:
    execute_model_state = None
    speculative_config = SimpleNamespace(sd_tail_gate_mode="threshold")
    num_spec_tokens = 5

    @property
    def routed_experts_initialized(self) -> bool:
        raise _StopAfterTelemetry


def _load_method(
    path: Path,
    class_name: str,
    method_name: str,
    namespace: dict[str, object],
) -> Callable[..., Any]:
    source = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = copy.deepcopy(
        next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
    )
    method_node.decorator_list = []
    method_node.returns = None
    for node in ast.walk(method_node):
        if isinstance(node, ast.arg):
            node.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return cast(Callable[..., Any], namespace[method_name])


@pytest.fixture
def patched_runtime_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for relative_path in _RUNTIME_PATHS:
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_FIXTURE_ROOT / relative_path, destination)
        paths[relative_path] = destination

    monkeypatch.setattr(
        patches,
        "_get_vllm_file",
        lambda relative_path: str(paths[relative_path]),
    )
    patches._patch_vllm_runtime_tail_gating(SimpleNamespace(info=lambda *_args: None))
    return paths


def _scheduler_outputs() -> tuple[SimpleNamespace, ...]:
    return (
        _scheduler_output(
            runtime_k=0,
            tick=1,
            active_requests=4,
            mean_sequence_length=10.0,
            state="ARMED_OFF",
        ),
        _scheduler_output(
            runtime_k=0,
            tick=2,
            active_requests=3,
            mean_sequence_length=20.0,
            state="ARMED_OFF",
        ),
        _scheduler_output(
            runtime_k=5,
            tick=3,
            active_requests=2,
            mean_sequence_length=30.0,
            state="ON_LATCHED",
            just_activated=True,
        ),
        _scheduler_output(
            runtime_k=5,
            tick=4,
            active_requests=2,
            mean_sequence_length=40.0,
            state="ON_LATCHED",
        ),
    )


def _scheduler_output(
    *,
    runtime_k: int,
    tick: int,
    active_requests: int,
    mean_sequence_length: float,
    state: str,
    just_activated: bool = False,
) -> SimpleNamespace:
    enabled = runtime_k > 0
    return SimpleNamespace(
        num_spec_tokens_to_schedule=runtime_k,
        tail_gate_just_activated=just_activated,
        tail_gate_tick=tick,
        tail_gate_active_requests=active_requests,
        tail_gate_decode_active_requests=active_requests,
        tail_gate_mean_sequence_length=mean_sequence_length,
        tail_gate_predicted_speedup_sum=2.0 if enabled else 0.0,
        tail_gate_predicted_speedup_count=1 if enabled else 0,
        tail_gate_expected_accept_length=3.0,
        tail_gate_state=state,
    )


def _telemetry_runners(
    paths: dict[str, Path],
) -> tuple[
    dict[str, object],
    tuple[tuple[object, Callable[..., Any]], ...],
]:
    def stop_after_telemetry() -> None:
        raise _StopAfterTelemetry

    v2_runner = SimpleNamespace(
        speculative_config=SimpleNamespace(
            sd_tail_gate_mode="threshold",
            sd_tail_gate_off_mode="advance_only",
            method="eagle3",
        ),
        num_speculative_steps=5,
        update_pp_decode_requests=stop_after_telemetry,
    )
    v2_execute = _load_method(
        paths["v1/worker/gpu/model_runner.py"],
        "GPUModelRunner",
        "execute_model",
        {},
    )
    v1_runner = _V1Runner()
    v1_execute = _load_method(
        paths["v1/worker/gpu_model_runner.py"],
        "GPUModelRunner",
        "execute_model",
        {},
    )
    return (
        {"v1": v1_runner, "v2": v2_runner},
        ((v1_runner, v1_execute), (v2_runner, v2_execute)),
    )


def _observe_runtime_k(
    runners_and_methods: tuple[tuple[object, Callable[..., Any]], ...],
    scheduler_output: SimpleNamespace,
) -> None:
    for runner, execute_model in runners_and_methods:
        try:
            execute_model(runner, scheduler_output)
        except _StopAfterTelemetry:
            continue
        raise AssertionError("pinned runner continued past the telemetry fixture")


def _sampling_harness(
    paths: dict[str, Path],
) -> tuple[
    SimpleNamespace,
    Callable[..., Any],
    SimpleNamespace,
    _DraftTokenHandler,
    list[int],
    list[int],
]:
    class CUDAGraphMode:
        NONE = 0
        FULL = 1

    eagle_advances: list[int] = []
    decode_steps: list[int] = []
    input_batch = SimpleNamespace(
        num_tokens_after_padding=12,
        num_tokens=12,
        num_reqs=2,
        num_scheduled_tokens=torch.tensor([6, 6]),
        seq_lens_cpu_upper_bound=torch.tensor([8, 10]),
        seq_lens=torch.tensor([8, 10]),
        idx_mapping=torch.tensor([0, 2]),
        req_ids=["request-0", "request-2"],
        query_start_loc=torch.tensor([0, 6, 12]),
    )
    speculator = SimpleNamespace(
        num_speculative_steps=5,
        max_model_len=64,
        max_num_reqs=3,
        method="eagle3",
        hidden_states=torch.zeros((12, 4)),
        draft_tokens=torch.full((3, 5), -9, dtype=torch.int64),
        last_token_indices=torch.zeros(3, dtype=torch.int64),
        current_draft_step=torch.tensor(0, dtype=torch.int64),
        input_buffers=SimpleNamespace(),
        prefill_cudagraph_manager=None,
        decode_cudagraph_manager=None,
        dp_size=1,
        dp_rank=0,
        advance_draft_positions=True,
        supports_mm_inputs=False,
        _copy_request_inputs=lambda *_args, **_kwargs: None,
    )

    def advance_eagle_state(*_args: object, **_kwargs: object) -> None:
        eagle_advances.append(len(eagle_advances) + 1)
        speculator.draft_tokens[:2].fill_(eagle_advances[-1])

    speculator._prefill = advance_eagle_state
    speculator._multi_step_decode = lambda *_args, **_kwargs: decode_steps.append(
        len(decode_steps) + 1
    )

    def dispatch(
        _manager: object,
        _num_reqs: int,
        num_tokens: int,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[SimpleNamespace, None]:
        return SimpleNamespace(cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens), None

    propose = _load_method(
        paths["v1/worker/gpu/spec_decode/autoregressive/speculator.py"],
        "AutoRegressiveSpeculator",
        "propose",
        {
            "torch": torch,
            "CUDAGraphMode": CUDAGraphMode,
            "dispatch_cg_and_sync_dp": dispatch,
            "get_uniform_token_count": lambda *_args: 6,
            "prepare_prefill_inputs": lambda *_args, **_kwargs: None,
            "prepare_decode_inputs": lambda *_args, **_kwargs: None,
        },
    )
    speculator.propose = lambda *args, **kwargs: propose(speculator, *args, **kwargs)

    sampled_counts: Iterator[torch.Tensor] = iter(
        (
            torch.ones(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.int64),
            torch.tensor([3, 2], dtype=torch.int64),
        )
    )

    def sample(*_args: object) -> tuple[SimpleNamespace, torch.Tensor, torch.Tensor]:
        counts = next(sampled_counts)
        sampled_token_ids = torch.zeros(
            (2, int(counts.max().item())), dtype=torch.int64
        )
        return (
            SimpleNamespace(sampled_token_ids=sampled_token_ids),
            counts,
            torch.zeros(2),
        )

    draft_tokens_handler = _DraftTokenHandler()
    runner = SimpleNamespace(
        execute_model_state=None,
        is_last_pp_rank=True,
        pp_handler=None,
        sample=sample,
        prompt_logprobs_worker=SimpleNamespace(
            compute_prompt_logprobs=lambda *_args: {}
        ),
        model=SimpleNamespace(compute_logits=lambda *_args: None),
        req_states=SimpleNamespace(
            all_token_ids=SimpleNamespace(gpu=torch.zeros((3, 1))),
            num_computed_tokens=SimpleNamespace(gpu=torch.zeros(3)),
            prompt_len=SimpleNamespace(np=[1, 1, 1]),
            last_sampled_tokens=torch.zeros(3, dtype=torch.int64),
            next_prefill_tokens=torch.zeros(3, dtype=torch.int64),
            draft_tokens=torch.full((3, 5), -7, dtype=torch.int64),
        ),
        main_stream=object(),
        output_copy_stream=object(),
        speculator=speculator,
        postprocess_sampled=lambda *_args: None,
        sampler=SimpleNamespace(
            sampling_states=SimpleNamespace(
                temperature=SimpleNamespace(gpu=torch.ones(3)),
                seeds=SimpleNamespace(gpu=torch.zeros(3, dtype=torch.int64)),
            )
        ),
        num_speculative_steps=5,
        draft_tokens_handler=draft_tokens_handler,
        kv_connector=SimpleNamespace(post_forward=lambda _finished: None),
    )
    sample_tokens = _load_method(
        paths["v1/worker/gpu/model_runner.py"],
        "GPUModelRunner",
        "sample_tokens",
        {"AsyncOutput": _AsyncOutput, "ModelRunnerOutput": _ModelRunnerOutput},
    )
    return (
        runner,
        sample_tokens,
        input_batch,
        draft_tokens_handler,
        eagle_advances,
        decode_steps,
    )


def _snapshot(
    baseline: dict[_CounterKey, float], delta: dict[_CounterKey, float]
) -> dict[_CounterKey, float]:
    return {key: value + delta.get(key, 0.0) for key, value in baseline.items()}


def test_patched_v1_v2_tail_gate_lifecycle_reaches_wandb_metrics(
    patched_runtime_sources: dict[str, Path],
) -> None:
    telemetry_runners, runners_and_methods = _telemetry_runners(patched_runtime_sources)
    (
        sampling_runner,
        sample_tokens,
        input_batch,
        draft_tokens_handler,
        eagle_advances,
        decode_steps,
    ) = _sampling_harness(patched_runtime_sources)
    runtime_spec_counters: dict[_CounterKey, float] = {
        cast(_CounterKey, key): 0.0 for key in _SPEC_COUNTERS
    }
    cache_snapshots: list[list[list[int]]] = []
    pre_activation_telemetry: dict[_CounterKey, float] = {}

    for scheduler_output in _scheduler_outputs():
        _observe_runtime_k(runners_and_methods, scheduler_output)
        sampling_runner.execute_model_state = SimpleNamespace(
            input_batch=input_batch,
            attn_metadata={},
            slot_mappings_by_layer={},
            hidden_states=torch.zeros((12, 4)),
            aux_hidden_states=None,
            num_spec_tokens_to_schedule=(scheduler_output.num_spec_tokens_to_schedule),
            finished_req_ids=set(),
        )
        sample_output = cast(_AsyncOutput, sample_tokens(sampling_runner, None))
        published = draft_tokens_handler.published[-1]
        if published.shape[1] > 0:
            runtime_spec_counters["vllm:spec_decode_num_drafts"] += float(
                published.shape[0]
            )
            runtime_spec_counters["vllm:spec_decode_num_draft_tokens"] += float(
                published.numel()
            )
        runtime_spec_counters["vllm:spec_decode_num_accepted_tokens"] += float(
            (sample_output.num_sampled_tokens - 1).clamp_min(0).sum().item()
        )
        cache_snapshots.append(
            sampling_runner.req_states.draft_tokens.detach().cpu().tolist()
        )
        if scheduler_output.tail_gate_tick == 2:
            pre_activation_telemetry = dict(
                telemetry_runners["v2"]._nrl_tail_gate_metrics
            )

    final_telemetry = {
        name: {
            key: runner._nrl_tail_gate_metrics.get(key, 0.0) for key in _TELEMETRY_KEYS
        }
        for name, runner in telemetry_runners.items()
    }
    final_delta: dict[_CounterKey, float] = dict(
        telemetry_runners["v2"]._nrl_tail_gate_metrics
    )
    final_delta.update(runtime_spec_counters)
    all_keys: set[_CounterKey] = (
        set(final_delta) | set(pre_activation_telemetry) | set(_SPEC_COUNTERS)
    )
    baseline: dict[_CounterKey, float] = {key: 100.0 for key in all_keys}
    start_snapshot: dict[_CounterKey, float] = {
        key: value for key, value in baseline.items()
    }
    pre_activation_metrics = compute_spec_decode_metrics(
        start_snapshot,
        _snapshot(baseline, pre_activation_telemetry),
    )
    final_metrics = compute_spec_decode_metrics(
        start_snapshot,
        _snapshot(baseline, final_delta),
    )
    zero_safe_metric_names = (
        "vllm/spec_acceptance_rate",
        "vllm/tail_gate_enabled_step_ratio",
        "vllm/tail_gate_activation_batch",
        "vllm/tail_gate_activation_tick",
        "vllm/tail_gate_predicted_speedup",
        "vllm/tail_gate_activation_predicted_speedup",
    )

    observed = {
        "telemetry": final_telemetry,
        "proposal_lifecycle": {
            "published_widths": [
                proposal.shape[1] for proposal in draft_tokens_handler.published
            ],
            "active_cache_first_values": [
                [snapshot[index][0] for index in (0, 2)] for snapshot in cache_snapshots
            ],
            "inactive_cache_first_values": [
                snapshot[1][0] for snapshot in cache_snapshots
            ],
            "eagle_advance_ticks": eagle_advances,
            "decode_steps": decode_steps,
        },
        "pre_activation_metrics": {
            "drafts": pre_activation_metrics["vllm/spec_num_drafts"],
            "accepted": pre_activation_metrics["vllm/spec_num_accepted_tokens"],
            "acceptance_length": pre_activation_metrics["vllm/spec_acceptance_length"],
            "advance_only_ratio": pre_activation_metrics[
                "vllm/tail_gate_advance_only_step_ratio"
            ],
            "k0_steps": pre_activation_metrics["vllm/tail_gate_k_0_steps"],
            "zero_safe": {
                name: pre_activation_metrics[name] for name in zero_safe_metric_names
            },
            "all_finite": all(
                math.isfinite(pre_activation_metrics[name])
                for name in zero_safe_metric_names
            ),
        },
        "final_metrics": {
            "drafts": final_metrics["vllm/spec_num_drafts"],
            "draft_tokens": final_metrics["vllm/spec_num_draft_tokens"],
            "accepted": final_metrics["vllm/spec_num_accepted_tokens"],
            "acceptance_length": final_metrics["vllm/spec_acceptance_length"],
            "acceptance_rate": final_metrics["vllm/spec_acceptance_rate"],
            "enabled_ratio": final_metrics["vllm/tail_gate_enabled_step_ratio"],
            "advance_only_ratio": final_metrics[
                "vllm/tail_gate_advance_only_step_ratio"
            ],
            "activation_batch": final_metrics["vllm/tail_gate_activation_batch"],
            "activation_tick": final_metrics["vllm/tail_gate_activation_tick"],
            "active_requests": final_metrics["vllm/tail_gate_active_requests"],
            "mean_sequence_length": final_metrics[
                "vllm/tail_gate_mean_sequence_length"
            ],
            "predicted_speedup": final_metrics["vllm/tail_gate_predicted_speedup"],
            "activation_predicted_speedup": final_metrics[
                "vllm/tail_gate_activation_predicted_speedup"
            ],
            "k0_ratio": final_metrics["vllm/tail_gate_k_0_step_ratio"],
            "k5_ratio": final_metrics["vllm/tail_gate_k_5_step_ratio"],
        },
    }

    expected_telemetry = {
        "vllm:spec_decode_tail_gate_decisions": 4.0,
        "vllm:spec_decode_tail_gate_enabled_steps": 2.0,
        "vllm:spec_decode_tail_gate_disabled_steps": 2.0,
        "vllm:spec_decode_tail_gate_activations": 1.0,
        "vllm:spec_decode_tail_gate_activation_batch_sum": 2.0,
        "vllm:spec_decode_tail_gate_activation_tick_sum": 3.0,
        "vllm:spec_decode_tail_gate_activation_tick_count": 1.0,
        "vllm:spec_decode_tail_gate_k_0_steps": 2.0,
        "vllm:spec_decode_tail_gate_k_5_steps": 2.0,
    }
    assert observed == {
        "telemetry": {"v1": expected_telemetry, "v2": expected_telemetry},
        "proposal_lifecycle": {
            "published_widths": [0, 0, 5, 5],
            "active_cache_first_values": [[0, 0], [0, 0], [3, 3], [4, 4]],
            "inactive_cache_first_values": [-7, -7, -7, -7],
            "eagle_advance_ticks": [1, 2, 3, 4],
            "decode_steps": [1, 2],
        },
        "pre_activation_metrics": {
            "drafts": 0.0,
            "accepted": 0.0,
            "acceptance_length": 1.0,
            "advance_only_ratio": 1.0,
            "k0_steps": 2.0,
            "zero_safe": {
                "vllm/spec_acceptance_rate": 0.0,
                "vllm/tail_gate_enabled_step_ratio": 0.0,
                "vllm/tail_gate_activation_batch": 0.0,
                "vllm/tail_gate_activation_tick": 0.0,
                "vllm/tail_gate_predicted_speedup": 0.0,
                "vllm/tail_gate_activation_predicted_speedup": 0.0,
            },
            "all_finite": True,
        },
        "final_metrics": {
            "drafts": 4.0,
            "draft_tokens": 20.0,
            "accepted": 3.0,
            "acceptance_length": 1.75,
            "acceptance_rate": 0.15,
            "enabled_ratio": 0.5,
            "advance_only_ratio": 0.5,
            "activation_batch": 2.0,
            "activation_tick": 3.0,
            "active_requests": 2.75,
            "mean_sequence_length": 25.0,
            "predicted_speedup": 2.0,
            "activation_predicted_speedup": 2.0,
            "k0_ratio": 0.5,
            "k5_ratio": 0.5,
        },
    }
