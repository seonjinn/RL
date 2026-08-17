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

import hashlib
import inspect
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional

import torch

from nemo_rl.models.generation.interfaces import (
    ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.r3_trace import (
    trace_router_replay_action,
    trace_router_replay_assignment,
    trace_router_replay_graph_consumer,
)

_ROUTER_REPLAY_VALIDATE_ENV = "NRL_ROUTER_REPLAY_VALIDATE"
_ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY = "r3_router_cuda_graph_input_v1"
_ROUTER_REPLAY_CUDA_GRAPH_SCOPES = (
    frozenset(("moe_router",)),
    frozenset(("attn", "mamba", "moe_router")),
)
_MISSING_ROUTE_SENTINEL = ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL
_MISSING_ROUTE_FALLBACK_PATCH_ATTR = "_nrl_missing_route_fallback_patch"
ROUTER_REPLAY_GRAPH_COUNTER_FIELDS = (
    "route_payloads_produced",
    "route_payloads_copied",
    "route_graph_launches",
    "missing_route_count",
    "stale_generation_count",
    "malformed_route_count",
    "out_of_range_count",
    "duplicate_route_count",
    "cp_mismatch_count",
)
_ROUTER_REPLAY_GRAPH_CONSUMER_ATTRIBUTES = (
    "graph_input_launch_record",
    "graph_input_bank_id",
    "graph_input_graph_index",
    "graph_input_copy_generation",
    "_nrl_graph_input_schedule_key",
    "_nrl_route_digest",
    "_nrl_graph_input_signature",
)


@dataclass(frozen=True)
class RouterReplayGraphInputSignature:
    """Content-independent physical identity for one model-local route slice."""

    layer_number: int
    payload_idx: int
    shape: tuple[int, int]
    dtype: str
    device_type: str
    topk: int
    num_experts: int

    def trace_record(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device_type": self.device_type,
            "topk": self.topk,
            "num_experts": self.num_experts,
        }


def router_replay_enabled(config: PolicyConfig) -> bool:
    return bool((config.get("router_replay") or {}).get("enabled", False))


def configure_vllm_for_router_replay(config: PolicyConfig) -> None:
    """Apply vLLM settings required for Megatron router replay correctness."""
    if not router_replay_enabled(config):
        return

    generation = config.setdefault("generation", {})
    vllm_kwargs = generation.setdefault("vllm_kwargs", {})
    vllm_kwargs["enable_return_routed_experts"] = True


def resolve_router_replay_cuda_graph_input_capability() -> str | None:
    """Read the versioned route-input capability from the active MCore runtime."""
    try:
        from megatron.core.transformer.moe.router_replay import (
            ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY,
        )
    except ImportError:
        return None
    if not isinstance(ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY, str):
        return None
    return ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY


def validate_router_replay_cuda_graph_scope(
    *,
    enabled: bool,
    cuda_graph_impl: object,
    cuda_graph_modules: object,
    runtime_capability: str | None,
    validation_enabled: bool,
    router_fusion: bool,
    fixed_thd_capacity: bool,
    bf16: bool,
    hybridep: bool,
) -> None:
    """Allow graph-owned replay routing only for the exact proven v1 contract."""
    if not enabled or cuda_graph_impl == "none":
        return

    if isinstance(cuda_graph_modules, str):
        configured_modules: Iterable[object] = (
            module.strip() for module in cuda_graph_modules.split(",") if module.strip()
        )
    elif isinstance(cuda_graph_modules, (list, tuple)):
        configured_modules = cuda_graph_modules
    else:
        configured_modules = ()
    modules = {getattr(module, "name", module) for module in configured_modules}
    captures_router = not modules or bool(
        modules.intersection({"moe", "moe_router", "moe_preprocess"})
    )
    if not captures_router:
        return
    if cuda_graph_impl != "transformer_engine" or frozenset(modules) not in (
        _ROUTER_REPLAY_CUDA_GRAPH_SCOPES
    ):
        raise ValueError(
            "R3 v1 supports only tested partial graph-owned router scopes: "
            "{moe_router} or {attn,mamba,moe_router} with "
            "cuda_graph_impl='transformer_engine'."
        )
    if runtime_capability != _ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY:
        raise ValueError(
            "The active Megatron-Core runtime lacks "
            f"{_ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY}."
        )
    if not validation_enabled:
        raise ValueError("R3 router CUDA graph input requires route validation.")
    if router_fusion:
        raise ValueError("R3 router CUDA graph input does not support router fusion.")
    if not fixed_thd_capacity:
        raise ValueError("R3 router CUDA graph input requires fixed THD capacity.")
    if not bf16:
        raise ValueError(
            "R3 router CUDA graph input v1 requires BF16 without FP8 or NVFP4."
        )
    if not hybridep:
        raise ValueError("R3 router CUDA graph input v1 requires HybridEP.")


def _requested_fixed_thd_capacity(config: PolicyConfig) -> bool:
    megatron_cfg = config.get("megatron_cfg") or {}
    sequence_packing = config.get("sequence_packing") or {}
    dynamic_batching = config.get("dynamic_batching") or {}
    sequence_capacity = megatron_cfg.get("thd_max_packed_sequences")
    token_capacity = sequence_packing.get("train_mb_tokens")
    return (
        sequence_packing.get("enabled") is True
        and dynamic_batching.get("enabled") is False
        and isinstance(sequence_capacity, int)
        and not isinstance(sequence_capacity, bool)
        and sequence_capacity >= 2
        and isinstance(token_capacity, int)
        and not isinstance(token_capacity, bool)
        and token_capacity > 0
    )


def _requested_unquantized_bf16(config: PolicyConfig) -> bool:
    megatron_cfg = config.get("megatron_cfg") or {}
    fp8_cfg = megatron_cfg.get("fp8_cfg") or {}
    model_overrides = megatron_cfg.get("model_overrides") or {}
    return (
        config.get("precision") == "bfloat16"
        and fp8_cfg.get("enabled") is not True
        and not bool(model_overrides.get("fp8"))
        and not bool(model_overrides.get("fp4"))
    )


def validate_router_replay_config(config: PolicyConfig) -> None:
    if not router_replay_enabled(config):
        return

    generation = config.get("generation") or {}
    megatron_cfg = config.get("megatron_cfg") or {}

    if generation.get("backend") != "vllm":
        raise ValueError("router_replay.enabled requires vLLM generation.")
    if not megatron_cfg.get("enabled", False):
        raise ValueError("router_replay.enabled requires the Megatron policy backend.")

    validate_router_replay_cuda_graph_scope(
        enabled=True,
        cuda_graph_impl=megatron_cfg.get("cuda_graph_impl", "none"),
        cuda_graph_modules=megatron_cfg.get("cuda_graph_modules"),
        runtime_capability=resolve_router_replay_cuda_graph_input_capability(),
        validation_enabled=router_replay_validation_enabled(),
        router_fusion=(
            megatron_cfg.get("moe_router_fusion") is True
            or (megatron_cfg.get("model_overrides") or {}).get("moe_router_fusion")
            is True
        ),
        fixed_thd_capacity=_requested_fixed_thd_capacity(config),
        bf16=_requested_unquantized_bf16(config),
        hybridep=(
            megatron_cfg.get("moe_token_dispatcher_type") == "flex"
            and megatron_cfg.get("moe_flex_dispatcher_backend") == "hybridep"
        ),
    )

    vpp_size = megatron_cfg.get("virtual_pipeline_model_parallel_size")
    if vpp_size not in (None, 1):
        raise ValueError(
            "router_replay.enabled does not support virtual pipeline parallelism yet."
        )
    _install_missing_route_fallback_patch()


def _iter_model_modules(model: Any) -> Iterable[Any]:
    if isinstance(model, (list, tuple)):
        for item in model:
            yield from _iter_model_modules(item)
        return

    modules = getattr(model, "modules", None)
    if callable(modules):
        yield from modules()
    else:
        yield model


def _unwrap_model_config(model: Any) -> Optional[Any]:
    if isinstance(model, (list, tuple)):
        for item in model:
            cfg = _unwrap_model_config(item)
            if cfg is not None:
                return cfg
        return None

    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        cfg = getattr(current, "config", None)
        if cfg is not None:
            return cfg
        current = getattr(current, "module", None)
    return None


def _global_moe_layer_numbers(model_config: Any) -> list[int]:
    num_layers = int(getattr(model_config, "num_layers"))
    moe_layer_freq = getattr(model_config, "moe_layer_freq", 1)

    if isinstance(moe_layer_freq, int):
        if moe_layer_freq <= 0:
            raise ValueError(f"moe_layer_freq must be positive, got {moe_layer_freq}")
        pattern = [1 if i % moe_layer_freq == 0 else 0 for i in range(num_layers)]
    elif isinstance(moe_layer_freq, list):
        if len(moe_layer_freq) != num_layers:
            raise ValueError(
                f"moe_layer_freq has {len(moe_layer_freq)} entries but num_layers={num_layers}"
            )
        pattern = moe_layer_freq
    else:
        raise ValueError(f"Unsupported moe_layer_freq: {moe_layer_freq!r}")

    return [layer_idx + 1 for layer_idx, is_moe in enumerate(pattern) if is_moe]


def _router_replay_instances_for_model(model: Any) -> list[tuple[Any, int]]:
    instances: list[tuple[Any, int]] = []
    seen: set[int] = set()
    for module in _iter_model_modules(model):
        replay = getattr(module, "router_replay", None)
        layer_number = getattr(module, "layer_number", None)
        if replay is None or layer_number is None:
            continue
        if id(replay) in seen:
            continue
        seen.add(id(replay))
        instances.append((replay, int(layer_number)))
    return instances


def _local_layer_numbers_for_model(model: Any) -> set[int]:
    layer_numbers: set[int] = set()
    for module in _iter_model_modules(model):
        layer_number = getattr(module, "layer_number", None)
        if layer_number is None:
            continue
        layer_numbers.add(int(layer_number))
    return layer_numbers


def _normalize_routed_experts_for_mcore(routed_experts: torch.Tensor) -> torch.Tensor:
    if routed_experts.dim() == 4:
        if routed_experts.shape[0] == 1:
            return routed_experts.squeeze(0)
        return routed_experts.transpose(0, 1).reshape(
            routed_experts.shape[0] * routed_experts.shape[1],
            routed_experts.shape[2],
            routed_experts.shape[3],
        )
    if routed_experts.dim() == 3:
        return routed_experts
    raise ValueError(
        "routed_experts must have shape [1, T, L, K], [B, S, L, K], or [T, L, K]; "
        f"got {tuple(routed_experts.shape)}"
    )


def _payload_indices_for_moe_layers(
    *,
    global_moe_layers: list[int],
    num_payload_layers: int,
    total_num_layers: int,
) -> dict[int, int]:
    if num_payload_layers == len(global_moe_layers):
        return {
            layer_number: payload_idx
            for payload_idx, layer_number in enumerate(global_moe_layers)
        }

    if num_payload_layers == total_num_layers:
        return {layer_number: layer_number - 1 for layer_number in global_moe_layers}

    raise ValueError(
        "routed_experts layer axis does not match a supported payload layout: "
        f"payload={num_payload_layers}, moe_layers={len(global_moe_layers)}, "
        f"total_layers={total_num_layers}. Expected exactly "
        f"{len(global_moe_layers)} layers for compressed MoE-layer layout or "
        f"{total_num_layers} layers for vLLM full-transformer-layer layout."
    )


def router_replay_validation_enabled() -> bool:
    return os.getenv(_ROUTER_REPLAY_VALIDATE_ENV, "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _validate_microbatch_generation(microbatch_generation: int) -> None:
    if type(microbatch_generation) is not int:
        raise TypeError("microbatch_generation must be an int.")
    if microbatch_generation < 0:
        raise ValueError("microbatch_generation must be nonnegative.")


def _route_digest(replay_tensor: torch.Tensor) -> str:
    tensor = replay_tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _router_replay_counters(replay_instance: Any) -> dict[str, int]:
    counters = getattr(replay_instance, "_nrl_graph_route_counters", None)
    if not isinstance(counters, dict) or set(counters) != set(
        ROUTER_REPLAY_GRAPH_COUNTER_FIELDS
    ):
        counters = {name: 0 for name in ROUTER_REPLAY_GRAPH_COUNTER_FIELDS}
        replay_instance._nrl_graph_route_counters = counters
    return counters


def _increment_router_replay_counter(
    replay_instance: Any,
    name: str,
    amount: int = 1,
) -> None:
    if name not in ROUTER_REPLAY_GRAPH_COUNTER_FIELDS:
        raise ValueError(f"Unknown RouterReplay graph counter: {name}")
    _router_replay_counters(replay_instance)[name] += amount


def _route_error_counter_name(error: BaseException) -> str:
    message = str(error).lower()
    if "missing-route" in message or "missing route" in message:
        return "missing_route_count"
    if "duplicate" in message:
        return "duplicate_route_count"
    if "outside" in message or "expert range" in message:
        return "out_of_range_count"
    if (
        "context" in message
        or "cp-" in message
        or "token count" in message
        or "indices shape must equal" in message
        or "structural padding mask shape" in message
    ):
        return "cp_mismatch_count"
    if "generation" in message:
        return "stale_generation_count"
    return "malformed_route_count"


def _record_router_replay_error(model: Any, error: BaseException) -> None:
    counter_name = _route_error_counter_name(error)
    for replay_instance, _ in _router_replay_instances_for_model(model):
        _increment_router_replay_counter(replay_instance, counter_name)


def record_router_replay_graph_error(model: Any, error: BaseException) -> None:
    """Count a controller-side graph route validation failure by category."""
    _record_router_replay_error(model, error)


def snapshot_router_replay_graph_counters(model: Any) -> dict[str, int]:
    """Return process-local cumulative route lifecycle counters for this model."""
    totals = {name: 0 for name in ROUTER_REPLAY_GRAPH_COUNTER_FIELDS}
    for replay_instance, _ in _router_replay_instances_for_model(model):
        counters = _router_replay_counters(replay_instance)
        for name in ROUTER_REPLAY_GRAPH_COUNTER_FIELDS:
            totals[name] += int(counters[name])
    return totals


def _validate_replay_tensor(
    replay_tensor: torch.Tensor,
    model_config: Any,
    *,
    layer_number: int,
    payload_idx: int,
) -> None:
    if replay_tensor.numel() == 0 or not router_replay_validation_enabled():
        return

    missing_route_mask = replay_tensor.eq(_MISSING_ROUTE_SENTINEL).all(dim=-1)
    partial_missing_mask = replay_tensor.lt(0).any(dim=-1) & ~missing_route_mask
    if bool(partial_missing_mask.any().item()):
        bad_row = int(partial_missing_mask.nonzero()[0].item())
        bad_sample = replay_tensor[bad_row].detach().cpu().tolist()
        raise ValueError(
            "routed_experts fallback rows must use the all--1 sentinel. "
            f"layer_number={layer_number}, payload_idx={payload_idx}, "
            f"row={bad_row}, sample={bad_sample}, shape={tuple(replay_tensor.shape)}"
        )

    replay_tensor = replay_tensor[~missing_route_mask]
    if replay_tensor.numel() == 0:
        return

    sorted_indices = replay_tensor.sort(dim=-1).values
    duplicate_mask = sorted_indices[..., 1:] == sorted_indices[..., :-1]
    has_duplicate_topk = duplicate_mask.any()
    if bool(has_duplicate_topk.item()):
        duplicate_row = int(duplicate_mask.any(dim=-1).nonzero()[0].item())
        duplicate_sample = replay_tensor[duplicate_row].detach().cpu().tolist()
        raise ValueError(
            "routed_experts contains duplicate expert ids within a token's top-k "
            "selection. Missing or padded routed_experts rows must use a valid "
            "dummy top-k route, not repeated zeros. "
            f"layer_number={layer_number}, payload_idx={payload_idx}, "
            f"row={duplicate_row}, sample={duplicate_sample}, "
            f"shape={tuple(replay_tensor.shape)}"
        )

    num_moe_experts = getattr(model_config, "num_moe_experts", None)
    if num_moe_experts is None:
        return

    min_expert = int(replay_tensor.min().item())
    max_expert = int(replay_tensor.max().item())
    if min_expert < 0 or max_expert >= int(num_moe_experts):
        raise ValueError(
            "routed_experts contains expert ids outside Megatron's expert range: "
            f"min={min_expert}, max={max_expert}, num_moe_experts={num_moe_experts}, "
            f"layer_number={layer_number}, payload_idx={payload_idx}, "
            f"shape={tuple(replay_tensor.shape)}"
        )


def _install_missing_route_fallback_patch() -> None:
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    if getattr(RouterReplay.get_replay_topk, _MISSING_ROUTE_FALLBACK_PATCH_ATTR, False):
        return

    original_get_replay_topk = RouterReplay.get_replay_topk
    expected_non_receiver_params = [
        "scores",
        "topk",
        "num_groups",
        "group_topk",
        "default_compute_topk",
    ]
    actual_params = list(inspect.signature(original_get_replay_topk).parameters)
    # Wrapper receiver names are arbitrary; guard only Megatron's callable API.
    actual_non_receiver_params = actual_params[1:]
    if actual_non_receiver_params != expected_non_receiver_params:
        raise RuntimeError(
            "Unsupported Megatron RouterReplay.get_replay_topk signature for "
            "NeMo RL missing-route fallback patch: "
            f"expected_non_receiver_params={expected_non_receiver_params}, "
            f"actual={actual_params}. "
            "Update nemo_rl.models.megatron.router_replay before enabling "
            "policy.router_replay.enabled."
        )

    @wraps(original_get_replay_topk)
    def wrapped_get_replay_topk(
        replay_instance: Any,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Optional[Any] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = getattr(replay_instance, "router_replay_action", None)
        if action not in {
            RouterReplayAction.REPLAY_FORWARD,
            RouterReplayAction.REPLAY_BACKWARD,
        }:
            return original_get_replay_topk(
                replay_instance,
                scores,
                topk,
                num_groups,
                group_topk,
                default_compute_topk,
            )

        if action == RouterReplayAction.REPLAY_FORWARD:
            target_topk_idx = getattr(replay_instance, "target_topk_idx", None)
        else:
            replay_backward_list = getattr(replay_instance, "replay_backward_list", [])
            target_topk_idx = replay_backward_list[0] if replay_backward_list else None

        if target_topk_idx is None:
            return original_get_replay_topk(
                replay_instance,
                scores,
                topk,
                num_groups,
                group_topk,
                default_compute_topk,
            )

        target_topk_idx = target_topk_idx.to(scores.device)
        fallback_mask = target_topk_idx.eq(_MISSING_ROUTE_SENTINEL).all(dim=-1)
        if not bool(fallback_mask.any().item()):
            return original_get_replay_topk(
                replay_instance,
                scores,
                topk,
                num_groups,
                group_topk,
                default_compute_topk,
            )

        if default_compute_topk is None:
            raise RuntimeError(
                "RouterReplay missing-route fallback requires default_compute_topk."
            )

        _, default_indices = default_compute_topk(
            scores,
            topk,
            num_groups=num_groups,
            group_topk=group_topk,
        )
        effective_topk_idx = target_topk_idx.clone()
        effective_topk_idx[fallback_mask] = default_indices[fallback_mask]
        probs = scores.gather(1, effective_topk_idx)

        if action == RouterReplayAction.REPLAY_FORWARD:
            replay_backward_list = getattr(replay_instance, "replay_backward_list", [])
            if replay_backward_list:
                replay_backward_list[-1] = effective_topk_idx.detach()
        else:
            getattr(replay_instance, "replay_backward_list").pop(0)

        return probs, effective_topk_idx

    setattr(wrapped_get_replay_topk, _MISSING_ROUTE_FALLBACK_PATCH_ATTR, True)
    RouterReplay.get_replay_topk = wrapped_get_replay_topk


def _get_tensor_model_parallel_world_size() -> int:
    from megatron.core import parallel_state

    return int(parallel_state.get_tensor_model_parallel_world_size())


def _get_tensor_model_parallel_rank() -> int:
    from megatron.core import parallel_state

    return int(parallel_state.get_tensor_model_parallel_rank())


def _split_for_sequence_parallel(
    model_config: Any, routed_experts: torch.Tensor
) -> torch.Tensor:
    if not getattr(model_config, "sequence_parallel", False):
        return routed_experts

    tp_size = _get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return routed_experts
    if routed_experts.shape[0] % tp_size != 0:
        raise ValueError(
            "routed_experts token axis must be divisible by tensor parallel size "
            "when sequence_parallel is enabled: "
            f"tokens={routed_experts.shape[0]}, tp_size={tp_size}"
        )

    tp_rank = _get_tensor_model_parallel_rank()
    token_chunk = routed_experts.shape[0] // tp_size
    token_start = tp_rank * token_chunk
    token_end = token_start + token_chunk
    return routed_experts[token_start:token_end].contiguous()


def build_router_replay_tensors(
    model: Any,
    routed_experts: torch.Tensor,
) -> list[torch.Tensor]:
    """Build MCore RouterReplay tensors in model-local router order."""
    return [
        replay_tensor
        for _, replay_tensor in build_router_replay_assignments(model, routed_experts)
    ]


def build_router_replay_assignments(
    model: Any,
    routed_experts: torch.Tensor,
) -> list[tuple[Any, torch.Tensor]]:
    """Pair model-owned MCore RouterReplay instances with their replay tensors."""
    local_routed_experts = _normalize_routed_experts_for_mcore(routed_experts)
    if local_routed_experts.dim() != 3:
        raise ValueError(
            "normalized routed_experts must have shape [T, num_moe_layers, topk]"
        )

    model_config = _unwrap_model_config(model)
    if model_config is None:
        raise ValueError("Could not locate Megatron model config for router replay.")

    local_routed_experts = _split_for_sequence_parallel(
        model_config, local_routed_experts
    )
    global_moe_layers = _global_moe_layer_numbers(model_config)
    total_num_layers = int(getattr(model_config, "num_layers"))
    num_payload_layers = local_routed_experts.shape[1]
    moe_layer_to_payload_idx = _payload_indices_for_moe_layers(
        global_moe_layers=global_moe_layers,
        num_payload_layers=num_payload_layers,
        total_num_layers=total_num_layers,
    )
    model_instances = _router_replay_instances_for_model(model)
    if len(model_instances) == 0:
        local_moe_layers = _local_layer_numbers_for_model(model).intersection(
            global_moe_layers
        )
        if not local_moe_layers:
            return []
        raise ValueError(
            "Could not find any model-owned RouterReplay instances for local MoE "
            f"layers {sorted(local_moe_layers)}. Ensure Megatron was initialized "
            "with moe_enable_routing_replay=True."
        )

    replay_assignments = []
    for replay_instance, layer_number in model_instances:
        if layer_number not in moe_layer_to_payload_idx:
            raise ValueError(
                f"Router layer {layer_number} is not present in MoE layer pattern "
                f"{global_moe_layers}."
            )
        payload_idx = moe_layer_to_payload_idx[layer_number]
        replay_tensor = (
            local_routed_experts[:, payload_idx, :].to(dtype=torch.long).contiguous()
        )
        setattr(replay_instance, "_nrl_layer_number", layer_number)
        setattr(replay_instance, "_nrl_payload_idx", payload_idx)
        _validate_replay_tensor(
            replay_tensor,
            model_config,
            layer_number=layer_number,
            payload_idx=payload_idx,
        )
        trace_router_replay_assignment(
            layer_number=layer_number,
            payload_idx=payload_idx,
            replay_tensor=replay_tensor,
        )
        replay_assignments.append(
            (
                replay_instance,
                replay_tensor,
            )
        )

    return replay_assignments


def validate_router_replay_graph_microbatch(
    model: Any,
    routed_experts: torch.Tensor,
    structural_padding_mask: torch.Tensor,
    *,
    microbatch_generation: int,
) -> tuple[RouterReplayGraphInputSignature, ...]:
    """Validate current route identity without arming replay or graph state."""
    _validate_microbatch_generation(microbatch_generation)
    for replay_instance, _ in _router_replay_instances_for_model(model):
        previous_generation = getattr(
            replay_instance,
            "_nrl_last_graph_input_generation",
            None,
        )
        if previous_generation is not None and microbatch_generation <= int(
            previous_generation
        ):
            raise ValueError(
                "Router replay microbatch generation must strictly advance before "
                "graph activation: "
                f"current={microbatch_generation}, previous={previous_generation}."
            )
    if not isinstance(structural_padding_mask, torch.Tensor):
        raise TypeError(
            "Router replay graph preflight requires a structural padding Tensor."
        )
    if structural_padding_mask.dtype != torch.bool:
        raise TypeError(
            "Router replay graph structural padding mask must use torch.bool."
        )

    from megatron.core.transformer.moe import router_replay as mcore_router_replay

    capability = getattr(
        mcore_router_replay,
        "ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY",
        None,
    )
    validator = getattr(
        mcore_router_replay,
        "validate_router_replay_cuda_graph_input",
        None,
    )
    if capability != _ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY or not callable(
        validator
    ):
        raise RuntimeError(
            "The active Megatron-Core runtime lacks the exact "
            f"{_ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY} validation contract."
        )

    assignments = build_router_replay_assignments(model, routed_experts)
    model_config = _unwrap_model_config(model)
    if model_config is None:
        raise ValueError("Could not locate Megatron model config for route preflight.")
    num_experts = getattr(model_config, "num_moe_experts", None)
    if isinstance(num_experts, bool) or not isinstance(num_experts, int):
        raise ValueError(
            "Router replay graph preflight requires integer num_moe_experts."
        )

    flattened_mask = structural_padding_mask.reshape(-1).contiguous()
    flattened_mask = _split_for_sequence_parallel(model_config, flattened_mask)
    signatures: list[RouterReplayGraphInputSignature] = []
    for replay_instance, replay_tensor in assignments:
        layer_number = int(getattr(replay_instance, "_nrl_layer_number"))
        payload_idx = int(getattr(replay_instance, "_nrl_payload_idx"))
        signature = validator(
            replay_tensor,
            structural_padding_mask=flattened_mask,
            expected_tokens=int(replay_tensor.shape[0]),
            topk=int(replay_tensor.shape[1]),
            num_experts=num_experts,
        )
        signatures.append(
            RouterReplayGraphInputSignature(
                layer_number=layer_number,
                payload_idx=payload_idx,
                shape=(int(signature.shape[0]), int(signature.shape[1])),
                dtype=str(signature.dtype),
                device_type=str(signature.device_type),
                topk=int(signature.topk),
                num_experts=int(signature.num_experts),
            )
        )
    return tuple(signatures)


def set_router_replay_forward(
    model: Any,
    routed_experts: torch.Tensor,
    *,
    microbatch_generation: int,
) -> None:
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    try:
        _validate_microbatch_generation(microbatch_generation)
        assignments = build_router_replay_assignments(model, routed_experts)
        for replay_instance, _ in assignments:
            previous_generation = getattr(
                replay_instance,
                "_nrl_last_graph_input_generation",
                None,
            )
            if previous_generation is not None and microbatch_generation <= int(
                previous_generation
            ):
                raise ValueError(
                    "Router replay microbatch generation must strictly advance: "
                    f"current={microbatch_generation}, previous={previous_generation}."
                )
    except (TypeError, ValueError, RuntimeError) as error:
        _record_router_replay_error(model, error)
        raise

    _install_missing_route_fallback_patch()
    for replay_instance, replay_tensor in assignments:
        for attribute in _ROUTER_REPLAY_GRAPH_CONSUMER_ATTRIBUTES:
            setattr(replay_instance, attribute, None)
        replay_instance.graph_input_generation = microbatch_generation
        replay_instance._nrl_last_graph_input_generation = microbatch_generation
        replay_instance._nrl_route_digest = _route_digest(replay_tensor)
        model_config = _unwrap_model_config(model)
        num_experts = getattr(model_config, "num_moe_experts", -1)
        replay_instance._nrl_graph_input_signature = RouterReplayGraphInputSignature(
            layer_number=int(getattr(replay_instance, "_nrl_layer_number")),
            payload_idx=int(getattr(replay_instance, "_nrl_payload_idx")),
            shape=(int(replay_tensor.shape[0]), int(replay_tensor.shape[1])),
            dtype=str(replay_tensor.dtype),
            device_type=replay_tensor.device.type,
            topk=int(replay_tensor.shape[1]),
            num_experts=(
                int(num_experts)
                if isinstance(num_experts, int) and not isinstance(num_experts, bool)
                else -1
            ),
        )
        replay_instance.set_target_indices(replay_tensor)
        _increment_router_replay_counter(replay_instance, "route_payloads_produced")
        trace_router_replay_action(
            action="replay_forward",
            layer_number=getattr(replay_instance, "_nrl_layer_number", None),
            replay_tensor=replay_tensor,
            replay_backward_list_len=len(
                getattr(replay_instance, "replay_backward_list", [])
            ),
            microbatch_generation=microbatch_generation,
            route_digest=replay_instance._nrl_route_digest,
        )
        replay_instance.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)


def record_router_replay_graph_consumers(
    model: Any,
    *,
    microbatch_generation: int,
    schedule_key: int,
) -> None:
    """Record post-success MCore launch evidence for current route values."""
    _validate_microbatch_generation(microbatch_generation)
    if type(schedule_key) is not int or schedule_key < 1:
        raise ValueError("Router replay graph schedule_key must be a positive int.")
    for replay_instance, layer_number in _router_replay_instances_for_model(model):
        active_generation = getattr(replay_instance, "graph_input_generation", None)
        if active_generation != microbatch_generation:
            _increment_router_replay_counter(
                replay_instance, "stale_generation_count"
            )
            raise ValueError(
                "Router replay graph consumer generation is stale: "
                f"expected={microbatch_generation}, active={active_generation}."
            )

        record = getattr(replay_instance, "graph_input_launch_record", None)
        successful_graph_launch = record is not None
        bank_id: Optional[int] = None
        graph_index: Optional[int] = None
        copy_generation: Optional[int] = None
        if successful_graph_launch:
            mcore_bank_id = getattr(record, "bank_id", None)
            graph_index = getattr(record, "graph_index", None)
            copy_generation = getattr(record, "copy_generation", None)
            record_values = (mcore_bank_id, graph_index, copy_generation)
            if any(type(value) is not int for value in record_values):
                _increment_router_replay_counter(
                    replay_instance, "malformed_route_count"
                )
                raise RuntimeError(
                    "Router replay graph launch record has malformed integer fields."
                )
            if mcore_bank_id < 0 or graph_index < 0 or copy_generation < 1:
                _increment_router_replay_counter(
                    replay_instance, "malformed_route_count"
                )
                raise RuntimeError(
                    "Router replay graph launch record has invalid integer ranges."
                )
            previous_copy_generation = getattr(
                replay_instance,
                "_nrl_last_graph_input_copy_generation",
                None,
            )
            if previous_copy_generation is not None and copy_generation <= int(
                previous_copy_generation
            ):
                _increment_router_replay_counter(
                    replay_instance, "stale_generation_count"
                )
                raise RuntimeError(
                    "Router replay graph copy generation must strictly advance."
                )
            replay_instance._nrl_last_graph_input_copy_generation = copy_generation
            _increment_router_replay_counter(replay_instance, "route_payloads_copied")
            _increment_router_replay_counter(replay_instance, "route_graph_launches")
            # MCore's bank token is process-private. The normalized schedule key
            # identifies the selected bank without serializing that private value.
            bank_id = schedule_key

        replay_instance._nrl_graph_input_schedule_key = schedule_key
        signature = getattr(replay_instance, "_nrl_graph_input_signature", None)
        if not isinstance(signature, RouterReplayGraphInputSignature):
            _increment_router_replay_counter(replay_instance, "malformed_route_count")
            raise RuntimeError("Router replay graph consumer lacks a physical signature.")
        action = getattr(replay_instance, "router_replay_action", None)
        action_name = str(getattr(action, "value", action))
        trace_router_replay_graph_consumer(
            action=action_name,
            layer_number=layer_number,
            payload_idx=signature.payload_idx,
            microbatch_generation=microbatch_generation,
            route_digest=str(getattr(replay_instance, "_nrl_route_digest")),
            physical_signature=signature.trace_record(),
            bank_id=bank_id,
            graph_index=graph_index,
            schedule_key=schedule_key,
            copy_generation=copy_generation,
            successful_graph_launch=successful_graph_launch,
            capability_version=_ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY,
        )


def set_router_replay_backward(model: Any) -> None:
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    for replay_instance, _ in _router_replay_instances_for_model(model):
        replay_backward_list = getattr(replay_instance, "replay_backward_list", [])
        next_replay_tensor = replay_backward_list[0] if replay_backward_list else None
        trace_router_replay_action(
            action="replay_backward",
            layer_number=getattr(replay_instance, "_nrl_layer_number", None),
            replay_tensor=next_replay_tensor,
            replay_backward_list_len=len(replay_backward_list),
        )
        replay_instance.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)


def clear_router_replay(model: Optional[Any] = None) -> None:
    from megatron.core.transformer.moe.router_replay import RouterReplay

    if model is None:
        RouterReplay.clear_global_router_replay_action()
        RouterReplay.clear_global_indices()
        instances = RouterReplay.global_router_replay_instances
        for replay_instance in instances:
            replay_instance.graph_input_generation = None
            for attribute in _ROUTER_REPLAY_GRAPH_CONSUMER_ATTRIBUTES:
                setattr(replay_instance, attribute, None)
        return

    for replay_instance, _ in _router_replay_instances_for_model(model):
        replay_instance.clear_router_replay_action()
        replay_instance.clear_indices()
        replay_instance.graph_input_generation = None
        for attribute in _ROUTER_REPLAY_GRAPH_CONSUMER_ATTRIBUTES:
            setattr(replay_instance, attribute, None)


def clear_global_router_replay_instances() -> None:
    from megatron.core.transformer.moe.router_replay import RouterReplay

    RouterReplay.clear_global_router_replay_instances()
