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

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec

R3_MISSING_ROUTE_SENTINEL = -1


def validate_openai_sampling_request(
    request: Any, generation_config: dict[str, Any]
) -> None:
    """Reject HTTP sampling modifiers that training logprob replay cannot model.

    NeMo-RL recomputes the rollout distribution with temperature, top-p, and
    top-k only. Letting the OpenAI endpoint apply another logits processor makes
    the captured behavior policy inconsistent with training-side logprobs.
    """
    configured_top_k = generation_config["top_k"]
    effective_top_k = -1 if configured_top_k is None else configured_top_k
    if request.top_k not in (None, -1, effective_top_k):
        raise ValueError(
            "top_k must be unset or match the NeMo-RL generation config; "
            f"got {request.top_k!r}, expected {effective_top_k!r}."
        )
    if request.temperature != generation_config["temperature"]:
        raise ValueError(
            "temperature must match the NeMo-RL generation config; "
            f"got {request.temperature!r}, expected "
            f"{generation_config['temperature']!r}."
        )
    if request.top_p != generation_config["top_p"]:
        raise ValueError(
            "top_p must match the NeMo-RL generation config; "
            f"got {request.top_p!r}, expected {generation_config['top_p']!r}."
        )

    neutral_values = {
        "presence_penalty": (None, 0.0),
        "frequency_penalty": (None, 0.0),
        "repetition_penalty": (None, 1.0),
        "min_p": (None, 0.0),
        "logit_bias": (None, {}),
        "allowed_token_ids": (None,),
        "bad_words": (None, []),
        "use_beam_search": (None, False),
        "min_tokens": (None, 0),
        "thinking_token_budget": (None,),
        "structured_outputs": (None,),
        "response_format": (None,),
        "stop": (None, []),
        "stop_token_ids": (None, []),
        "include_stop_str_in_output": (None, False),
        "ignore_eos": (None, False),
        "repetition_detection": (None,),
        "vllm_xargs": (None, {}),
        "truncate_prompt_tokens": (None,),
        "truncation_side": (None,),
        "add_generation_prompt": (None, True),
        "continue_final_message": (None, False),
        "add_special_tokens": (None, False),
        "chat_template": (None,),
        "documents": (None, []),
        "media_io_kwargs": (None, {}),
        "mm_processor_kwargs": (None, {}),
    }
    for field, allowed in neutral_values.items():
        value = getattr(request, field, None)
        if value not in allowed:
            raise ValueError(
                f"{field}={value!r} changes the rollout distribution, but "
                "NeMo-RL training logprob replay does not model this modifier."
            )

    # Avoid model-level generation_config defaults silently enabling processors.
    request.top_k = effective_top_k
    request.repetition_penalty = 1.0
    request.min_p = 0.0


def extract_generated_token_logprobs(
    generated_token_ids: Sequence[int],
    token_logprobs: Sequence[Mapping[int, Any] | None] | None,
) -> list[float]:
    """Return one finite processed logprob for every generated token."""
    if not generated_token_ids:
        return []
    if not token_logprobs:
        raise RuntimeError(
            "vLLM did not return processed logprobs for generated tokens."
        )
    if len(token_logprobs) != len(generated_token_ids):
        raise RuntimeError(
            "vLLM generated-token logprob count does not match the generated "
            f"token count: logprobs={len(token_logprobs)}, "
            f"tokens={len(generated_token_ids)}."
        )

    values: list[float] = []
    for position, (token_id, candidates) in enumerate(
        zip(generated_token_ids, token_logprobs, strict=True)
    ):
        record = candidates.get(token_id) if candidates is not None else None
        if record is None:
            raise RuntimeError(
                "vLLM processed logprobs do not contain the chosen token "
                f"{token_id} at generated position {position}."
            )
        value = float(record.logprob)
        if not math.isfinite(value):
            raise RuntimeError(
                "vLLM returned a non-finite processed logprob for chosen token "
                f"{token_id} at generated position {position}: {value}."
            )
        values.append(value)
    return values


def format_prompt_for_vllm_generation(
    data: BatchedDataDict[GenerationDatumSpec], sample_idx: Optional[int] = None
) -> list[dict[str, Any]]:
    """Format a list of prompts for vllm generation (which requires a specific format for its own `generate` method).

    See https://docs.vllm.ai/en/v0.9.1/features/multimodal_inputs.html for prompt format for multimodal inputs.
    """
    # Prepare prompts for vLLM (removing padding)
    prompts = []

    input_ids = data["input_ids"]
    batch_size = input_ids.shape[0]
    input_lengths = data["input_lengths"]

    # if sample_idx is None, return list of all prompts for the entire batch
    # else, return the prompt for the single sample specified by sample_idx
    return_all = sample_idx is None
    if sample_idx is None:
        start_idx = 0
        end_idx = batch_size
    else:
        start_idx = sample_idx
        end_idx = sample_idx + 1

    def _get_regular_prompt(index: int):
        valid_length = input_lengths[index].item()
        valid_ids = (
            input_ids[index, :valid_length]
            if valid_length > 0
            else input_ids[index, :0]
        )
        token_ids = valid_ids.tolist()
        return {"prompt_token_ids": token_ids}

    # Check if this is VLM generation by looking for message_log with images
    # Support for videos/audio/etc. can be added here
    # if 'message_log' in data and any('images' in msg for msg in data['message_log']):
    if "vllm_content" in data:
        # VLM generation using content and multi_modal_data
        for i in range(start_idx, end_idx):
            msg = data["vllm_content"][i]
            # if msg is None, this conversation had no multimodal content, fallback to regular prompt
            if msg is None:
                prompts.append(_get_regular_prompt(i))
                continue
            # init prompt dict
            prompt_dict = {"prompt": msg}
            # collect multi_modal_data from images, audios, and videos
            multi_modal_data = {}
            images = data.get("vllm_images", None)
            if images is not None and len(images[i]) > 0:
                multi_modal_data["image"] = (
                    images[i][0] if len(images[i]) == 1 else images[i]
                )
            audios = data.get("vllm_audios", None)
            if audios is not None and len(audios[i]) > 0:
                multi_modal_data["audio"] = (
                    audios[i][0] if len(audios[i]) == 1 else audios[i]
                )
            videos = data.get("vllm_videos", None)
            if videos is not None and len(videos[i]) > 0:
                multi_modal_data["video"] = (
                    videos[i][0] if len(videos[i]) == 1 else videos[i]
                )
            if not multi_modal_data:
                prompts.append(_get_regular_prompt(i))
                continue
            prompt_dict["multi_modal_data"] = multi_modal_data
            prompts.append(prompt_dict)
    else:
        # Regular LLM generation using token_ids (pre-tokenized).
        # Note: eval.py uses raw prompt strings instead of token IDs because its
        # collate function produces message_log dicts, not tokenized tensors.
        # Both are valid vLLM input formats but may tokenize slightly differently.
        for i in range(start_idx, end_idx):
            # Use input_lengths to get only valid tokens (not padding)
            prompts.append(_get_regular_prompt(i))

    return prompts if return_all else prompts[0]


def pad_and_align_routed_expert_indices(
    request_output: Any,
    completion_output: Any,
    *,
    valid_length: int,
    padded_length: int,
    device: torch.device,
    require_complete_routed_experts: bool = False,
    allow_missing_routed_experts_fallback: bool = True,
    return_stats: bool = False,
) -> Optional[torch.Tensor] | tuple[Optional[torch.Tensor], dict[str, int]]:
    """Return full-sequence-aligned routed experts as ``[S, L, topk]`` int32."""
    routed = getattr(completion_output, "routed_experts", None)
    prompt_routed = getattr(request_output, "prompt_routed_experts", None)

    if prompt_routed is not None:
        prompt_routed = torch.as_tensor(prompt_routed, dtype=torch.int32, device=device)
    if routed is not None:
        routed = torch.as_tensor(routed, dtype=torch.int32, device=device)

    if prompt_routed is not None and routed is not None:
        routed = torch.cat((prompt_routed, routed), dim=0)
    elif prompt_routed is not None:
        routed = prompt_routed

    expected_routes = min(max(valid_length - 1, 0), padded_length)
    stats = {
        "actual_routes": 0,
        "expected_routes": expected_routes,
        "missing_routes": 0,
        "surplus_routes": 0,
    }

    if routed is None:
        return (None, stats) if return_stats else None
    if routed.dim() != 3:
        raise ValueError(
            "vLLM routed_experts must have shape [tokens, num_moe_layers, topk], "
            f"got {tuple(routed.shape)}"
        )

    stats["actual_routes"] = int(routed.shape[0])
    stats["missing_routes"] = max(expected_routes - int(routed.shape[0]), 0)
    stats["surplus_routes"] = max(int(routed.shape[0]) - (expected_routes + 1), 0)
    if (
        require_complete_routed_experts
        and stats["missing_routes"] > 0
        and not allow_missing_routed_experts_fallback
    ):
        # This has only been observed rarely with vLLM prefix caching plus
        # chunked prefill: a small number of samples can omit routed-expert
        # rows even though most requests are complete. Keep
        # tools/model_diagnostics/6.vllm_routed_experts_completeness.py as a
        # standalone reproducer for upstream vLLM bug reports.
        num_cached_tokens = getattr(request_output, "num_cached_tokens", None)
        raise ValueError(
            "vLLM returned incomplete routed_experts for router replay: "
            f"routes={routed.shape[0]}, expected_at_least={expected_routes}, "
            f"valid_length={valid_length}, padded_length={padded_length}, "
            f"num_cached_tokens={num_cached_tokens}. This usually means the "
            "generation backend did not return routed experts for every "
            "non-final token in the prompt+response sequence."
        )
    max_allowed_routes = expected_routes + 1
    if require_complete_routed_experts and routed.shape[0] > max_allowed_routes:
        num_cached_tokens = getattr(request_output, "num_cached_tokens", None)
        raise ValueError(
            "vLLM returned too many routed_experts routes for router replay: "
            f"routes={routed.shape[0]}, expected={expected_routes}, "
            f"max_allowed={max_allowed_routes}, valid_length={valid_length}, "
            f"padded_length={padded_length}, num_cached_tokens={num_cached_tokens}. "
            "Router replay allows at most one surplus final-token route."
        )

    default_route = torch.arange(
        routed.shape[2],
        dtype=torch.int32,
        device=device,
    )
    full = (
        default_route.view(1, 1, -1)
        .expand(padded_length, routed.shape[1], routed.shape[2])
        .clone()
    )
    full = full.to(dtype=torch.int32)
    routes_to_copy = min(expected_routes, routed.shape[0])
    if routes_to_copy > 0:
        full[:routes_to_copy] = routed[:routes_to_copy].to(device=device)
    if stats["missing_routes"] > 0:
        full[routes_to_copy:expected_routes] = R3_MISSING_ROUTE_SENTINEL
    return (full, stats) if return_stats else full


def attach_routed_experts_to_chat_response_choices(
    response: Any,
    final_request_output: Any,
    *,
    device: torch.device,
    logger: Any = None,
) -> Any:
    """Attach aligned routed experts to OpenAI chat response choices."""
    outputs_by_index = {
        output.index: output for output in getattr(final_request_output, "outputs", [])
    }
    prompt_token_count = len(
        getattr(final_request_output, "prompt_token_ids", []) or []
    )

    choices = list(getattr(response, "choices", []))
    attached_choice_indices = set()
    for choice in choices:
        generation_details = outputs_by_index.get(choice.index)
        if generation_details is None:
            continue
        attached_choice_indices.add(choice.index)

        generation_token_count = len(getattr(generation_details, "token_ids", []) or [])
        routed_result = pad_and_align_routed_expert_indices(
            final_request_output,
            generation_details,
            valid_length=prompt_token_count + generation_token_count,
            padded_length=prompt_token_count + generation_token_count,
            device=device,
            require_complete_routed_experts=True,
            return_stats=True,
        )
        if not isinstance(routed_result, tuple):
            raise RuntimeError(
                "Expected routed_experts alignment to return stats for the "
                "OpenAI-compatible chat endpoint."
            )
        routed_experts, r3_stats = routed_result
        if routed_experts is None:
            raise RuntimeError(
                "vLLM was asked to return routed experts for the "
                "OpenAI-compatible chat endpoint but the generation "
                "output did not include routed_experts."
            )
        if r3_stats["missing_routes"] > 0 and logger is not None:
            logger.warning(
                "R3 router replay fallback: vLLM returned incomplete "
                "routed_experts for chat choice_idx=%d, "
                "missing_token_routes=%d, actual_routes=%d, "
                "expected_routes=%d. Megatron will use its own router "
                "for those missing token routes.",
                choice.index,
                r3_stats["missing_routes"],
                r3_stats["actual_routes"],
                r3_stats["expected_routes"],
            )
        choice.message.routed_experts = routed_experts.to(dtype=torch.int32).tolist()

    if len(attached_choice_indices) != len(choices):
        missing_choice_indices = sorted(
            choice.index
            for choice in choices
            if choice.index not in attached_choice_indices
        )
        raise RuntimeError(
            "vLLM was asked to return routed experts for the "
            "OpenAI-compatible chat endpoint but response choices could not be "
            "matched to generation outputs: "
            f"missing_choice_indices={missing_choice_indices}."
        )

    return response


def model_dump_chat_response_with_routed_experts(response: Any) -> dict[str, Any]:
    """Dump a vLLM OpenAI chat response while preserving dynamic R3 fields."""
    response_dict = response.model_dump()
    for choice, choice_dict in zip(
        getattr(response, "choices", []), response_dict.get("choices", [])
    ):
        routed_experts = getattr(
            getattr(choice, "message", None), "routed_experts", None
        )
        if routed_experts is not None:
            choice_dict.setdefault("message", {})["routed_experts"] = routed_experts
    return response_dict


def aggregate_spec_decode_counters(
    worker_metrics: list[dict[str, float | list[float]]],
) -> dict[str | tuple[str, int], float]:
    """Aggregate speculative decoding counters from multiple workers.

    Combines spec decode metrics collected from DP leader workers into
    a single aggregated counter dictionary.

    Args:
        worker_metrics: List of metric dictionaries from each worker.
            Each dict maps metric names to float values or lists of floats
            (for per-position metrics).

    Returns:
        Dictionary mapping metric names to their aggregated float values.
        Per-position metrics use (name, position) tuples as keys.

    Example:
        >>> metrics_from_workers = policy_generation.get_metrics()
        >>> counters = aggregate_spec_decode_counters(metrics_from_workers)
        >>> print(counters.get("vllm:spec_decode_num_drafts", 0))
        1234.0
    """
    counters: dict[str | tuple[str, int], float] = defaultdict(float)

    for report in worker_metrics:
        for metric_name, value in report.items():
            if "spec_decode" in metric_name:
                if isinstance(value, list):
                    # Per-position metrics (e.g., acceptance counts at each draft position)
                    for position, pos_value in enumerate(value, 1):
                        counters[metric_name, position] += pos_value
                else:
                    counters[metric_name] += value

    return dict(counters)


def compute_spec_decode_metrics(
    start_counters: dict[str | tuple[str, int], float],
    end_counters: dict[str | tuple[str, int], float],
) -> dict[str, float]:
    """Compute delta and derived metrics for speculative decoding.

    Calculates the difference between two counter snapshots and derives
    acceptance rate and acceptance length metrics for logging.

    Args:
        start_counters: Counter snapshot taken before generation.
        end_counters: Counter snapshot taken after generation.

    Returns:
        Dictionary of metrics suitable for logging to wandb/tensorboard.
        Keys are prefixed with "vllm/" for namespace consistency.
        Includes:
            - vllm/spec_num_drafts: Total number of draft batches
            - vllm/spec_num_draft_tokens: Total draft tokens generated
            - vllm/spec_num_accepted_tokens: Total tokens accepted
            - vllm/spec_acceptance_length: Average accepted tokens per draft + 1
            - vllm/spec_acceptance_rate: Ratio of accepted to draft tokens
            - vllm/{metric}-{position}: Per-position acceptance counts
            - vllm/spec_acceptance_rate-pos-{position}: Per-position acceptance rates
    """
    keys = set(start_counters) | set(end_counters)
    delta = {k: end_counters.get(k, 0.0) - start_counters.get(k, 0.0) for k in keys}

    num_drafts = delta.get("vllm:spec_decode_num_drafts", 0.0)
    num_draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    num_accepted_tokens = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)

    # acceptance_length = 1 + (accepted / drafts) represents average tokens
    # generated per draft batch (1 target model token + accepted draft tokens)
    acceptance_length = (
        1.0 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0
    )
    acceptance_rate = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0.0
    )

    spec_metrics: dict[str, float] = {
        "vllm/spec_num_drafts": num_drafts,
        "vllm/spec_num_draft_tokens": num_draft_tokens,
        "vllm/spec_num_accepted_tokens": num_accepted_tokens,
        "vllm/spec_acceptance_length": acceptance_length,
        "vllm/spec_acceptance_rate": acceptance_rate,
    }

    tail_gate_prefix = "vllm:spec_decode_tail_gate_"
    if any(isinstance(key, str) and key.startswith(tail_gate_prefix) for key in keys):

        def tail_gate_ratio(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator > 0 else 0.0

        decisions = delta.get(f"{tail_gate_prefix}decisions", 0.0)
        enabled_steps = delta.get(f"{tail_gate_prefix}enabled_steps", 0.0)
        disabled_steps = delta.get(f"{tail_gate_prefix}disabled_steps", 0.0)
        activations = delta.get(f"{tail_gate_prefix}activations", 0.0)
        active_requests_sum = delta.get(f"{tail_gate_prefix}active_requests_sum", 0.0)
        active_requests_count = delta.get(
            f"{tail_gate_prefix}active_requests_count", 0.0
        )
        decode_active_requests_sum = delta.get(
            f"{tail_gate_prefix}decode_active_requests_sum", 0.0
        )
        decode_active_requests_count = delta.get(
            f"{tail_gate_prefix}decode_active_requests_count", 0.0
        )
        mean_sequence_length_sum = delta.get(
            f"{tail_gate_prefix}mean_sequence_length_sum", 0.0
        )
        mean_sequence_length_count = delta.get(
            f"{tail_gate_prefix}mean_sequence_length_count", 0.0
        )
        expected_accept_length_sum = delta.get(
            f"{tail_gate_prefix}expected_accept_length_sum", 0.0
        )
        expected_accept_length_count = delta.get(
            f"{tail_gate_prefix}expected_accept_length_count", 0.0
        )
        activation_batch_sum = delta.get(f"{tail_gate_prefix}activation_batch_sum", 0.0)
        activation_seq_len_sum = delta.get(
            f"{tail_gate_prefix}activation_sequence_length_sum", 0.0
        )
        activation_tick_sum = delta.get(f"{tail_gate_prefix}activation_tick_sum", 0.0)
        activation_tick_count = delta.get(
            f"{tail_gate_prefix}activation_tick_count", 0.0
        )
        predicted_speedup_sum = delta.get(
            f"{tail_gate_prefix}predicted_speedup_sum", 0.0
        )
        predicted_speedup_count = delta.get(
            f"{tail_gate_prefix}predicted_speedup_count", 0.0
        )
        activation_predicted_speedup_sum = delta.get(
            f"{tail_gate_prefix}activation_predicted_speedup_sum", 0.0
        )
        activation_predicted_speedup_count = delta.get(
            f"{tail_gate_prefix}activation_predicted_speedup_count", 0.0
        )

        spec_metrics.update(
            {
                "vllm/tail_gate_decisions": decisions,
                "vllm/tail_gate_enabled_steps": enabled_steps,
                "vllm/tail_gate_disabled_steps": disabled_steps,
                "vllm/tail_gate_activations": activations,
                "vllm/tail_gate_enabled_step_ratio": tail_gate_ratio(
                    enabled_steps, decisions
                ),
                "vllm/tail_gate_advance_only_step_ratio": tail_gate_ratio(
                    disabled_steps, decisions
                ),
                "vllm/tail_gate_activation_batch": tail_gate_ratio(
                    activation_batch_sum, activations
                ),
                "vllm/tail_gate_activation_seq_len": tail_gate_ratio(
                    activation_seq_len_sum, activations
                ),
                "vllm/tail_gate_activation_sequence_length": tail_gate_ratio(
                    activation_seq_len_sum, activations
                ),
                "vllm/tail_gate_activation_tick": tail_gate_ratio(
                    activation_tick_sum, activation_tick_count
                ),
                "vllm/tail_gate_active_requests": tail_gate_ratio(
                    active_requests_sum, active_requests_count
                ),
                "vllm/tail_gate_decode_active_requests": tail_gate_ratio(
                    decode_active_requests_sum, decode_active_requests_count
                ),
                "vllm/tail_gate_mean_sequence_length": tail_gate_ratio(
                    mean_sequence_length_sum, mean_sequence_length_count
                ),
                "vllm/tail_gate_expected_accept_length": tail_gate_ratio(
                    expected_accept_length_sum, expected_accept_length_count
                ),
                "vllm/tail_gate_predicted_speedup": tail_gate_ratio(
                    predicted_speedup_sum, predicted_speedup_count
                ),
                "vllm/tail_gate_activation_predicted_speedup": tail_gate_ratio(
                    activation_predicted_speedup_sum,
                    activation_predicted_speedup_count,
                ),
            }
        )

        for key, value in delta.items():
            if not isinstance(key, str):
                continue
            histogram_prefix = f"{tail_gate_prefix}k_"
            if key.startswith(histogram_prefix) and key.endswith("_steps"):
                metric_name = key.removeprefix("vllm:spec_decode_").removesuffix(
                    "_steps"
                )
                spec_metrics[f"vllm/{metric_name}_steps"] = value
                spec_metrics[f"vllm/{metric_name}_step_ratio"] = tail_gate_ratio(
                    value, decisions
                )

    cudagraph_roles = (
        "target",
        "draft",
        "draft_prefill",
        "draft_decode",
        "draft_query",
    )
    all_eager_calls = 0.0
    all_graph_calls = 0.0
    all_eager_tokens = 0.0
    all_graph_tokens = 0.0
    all_padded_graph_tokens = 0.0
    all_fallbacks: dict[str, float] = defaultdict(float)

    for role in cudagraph_roles:
        counter_prefix = f"vllm:spec_decode_cudagraph_{role}_"
        eager_calls = delta.get(f"{counter_prefix}calls_none", 0.0)
        piecewise_calls = delta.get(f"{counter_prefix}calls_piecewise", 0.0)
        full_calls = delta.get(f"{counter_prefix}calls_full", 0.0)
        graph_calls = piecewise_calls + full_calls
        total_calls = eager_calls + graph_calls
        if total_calls <= 0:
            continue

        eager_tokens = delta.get(f"{counter_prefix}unpadded_tokens_none", 0.0)
        graph_tokens = delta.get(
            f"{counter_prefix}unpadded_tokens_piecewise", 0.0
        ) + delta.get(f"{counter_prefix}unpadded_tokens_full", 0.0)
        padded_graph_tokens = delta.get(
            f"{counter_prefix}padded_tokens_piecewise", 0.0
        ) + delta.get(f"{counter_prefix}padded_tokens_full", 0.0)
        total_tokens = eager_tokens + graph_tokens
        all_eager_calls += eager_calls
        all_graph_calls += graph_calls
        all_eager_tokens += eager_tokens
        all_graph_tokens += graph_tokens
        all_padded_graph_tokens += padded_graph_tokens

        spec_metrics.update(
            {
                f"vllm/cudagraph_{role}_total_calls": total_calls,
                f"vllm/cudagraph_{role}_graph_calls": graph_calls,
                f"vllm/cudagraph_{role}_eager_calls": eager_calls,
                f"vllm/cudagraph_{role}_graph_call_ratio": graph_calls / total_calls,
                f"vllm/cudagraph_{role}_eager_call_ratio": eager_calls / total_calls,
                f"vllm/cudagraph_{role}_graph_token_ratio": (
                    graph_tokens / total_tokens if total_tokens > 0 else 0.0
                ),
                f"vllm/cudagraph_{role}_eager_token_ratio": (
                    eager_tokens / total_tokens if total_tokens > 0 else 0.0
                ),
                f"vllm/cudagraph_{role}_padding_overhead_ratio": (
                    (padded_graph_tokens - graph_tokens) / graph_tokens
                    if graph_tokens > 0
                    else 0.0
                ),
            }
        )
        for reason in (
            "uninitialized",
            "disabled",
            "missing_capture_limit",
            "oversize",
            "mode_restricted",
            "missing_key",
            "incompatible",
            "empty",
        ):
            key = f"{counter_prefix}fallback_{reason}"
            if key in delta:
                spec_metrics[f"vllm/cudagraph_{role}_fallback_{reason}"] = delta[key]
                all_fallbacks[reason] += delta[key]

    all_total_calls = all_eager_calls + all_graph_calls
    all_total_tokens = all_eager_tokens + all_graph_tokens
    if all_total_calls > 0:
        spec_metrics.update(
            {
                "vllm/cudagraph_all_total_calls": all_total_calls,
                "vllm/cudagraph_all_graph_calls": all_graph_calls,
                "vllm/cudagraph_all_eager_calls": all_eager_calls,
                "vllm/cudagraph_all_graph_call_ratio": all_graph_calls
                / all_total_calls,
                "vllm/cudagraph_all_eager_call_ratio": all_eager_calls
                / all_total_calls,
                "vllm/cudagraph_all_graph_token_ratio": (
                    all_graph_tokens / all_total_tokens if all_total_tokens > 0 else 0.0
                ),
                "vllm/cudagraph_all_eager_token_ratio": (
                    all_eager_tokens / all_total_tokens if all_total_tokens > 0 else 0.0
                ),
                "vllm/cudagraph_all_padding_overhead_ratio": (
                    (all_padded_graph_tokens - all_graph_tokens) / all_graph_tokens
                    if all_graph_tokens > 0
                    else 0.0
                ),
            }
        )
        for reason, value in all_fallbacks.items():
            spec_metrics[f"vllm/cudagraph_all_fallback_{reason}"] = value

    # Add per-position metrics for detailed analysis
    for key, value in delta.items():
        if isinstance(key, tuple):
            metric_name, position = key
            spec_metrics[f"vllm/{metric_name}-{position}"] = value
            if num_drafts > 0:
                spec_metrics[f"vllm/spec_acceptance_rate-pos-{position}"] = (
                    value / num_drafts
                )

    return spec_metrics


# TODO: Replace this hard-coded map with a generic plugin-registration
# hook on ``VllmGeneration`` (e.g. a ``worker_cls_overrides`` registry populated
# by ``nemo_rl.modelopt`` on import) so core has no knowledge of ModelOpt-specific
# worker classes.
GENERATION_WORKER_OVERRIDES = {
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker",
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker",
}


def resolve_generation_worker_cls(default_cls: str, config: dict) -> str:
    """Return the quantized vLLM generation worker FQN if ``quant_cfg`` is set, else ``default_cls``.

    Safe to call even when ModelOpt is not installed — returns ``default_cls``
    unchanged whenever ``quant_cfg`` is ``None``, so the core generation path
    stays import-free of ModelOpt.
    """
    if config.get("quant_cfg") is None:
        return default_cls
    return GENERATION_WORKER_OVERRIDES.get(default_cls, default_cls)
