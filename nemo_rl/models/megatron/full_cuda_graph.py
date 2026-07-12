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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Literal, Optional

import torch

FULL_CUDA_GRAPH_GLOBAL_VALID_SEQS = "__nemo_rl_full_cuda_graph_global_valid_seqs"
FULL_CUDA_GRAPH_GLOBAL_VALID_TOKS = "__nemo_rl_full_cuda_graph_global_valid_toks"

_SUPPORTED_OPERATIONS = Literal[
    "policy_training",
    "logprob",
    "eval",
    "split_policy_training",
    "colocated_refit",
]


@dataclass(frozen=True)
class TensorSignature:
    """Static tensor properties that a captured graph may safely reuse."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device_type: str
    device_index: Optional[int]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorSignature":
        return cls(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device_type=tensor.device.type,
            device_index=tensor.device.index,
        )


@dataclass(frozen=True)
class StaticMicrobatchSignature:
    """Tensor-only signature for one fixed-shape processed microbatch."""

    tensors: tuple[tuple[str, TensorSignature], ...]

    @classmethod
    def from_microbatch(cls, microbatch: Any) -> "StaticMicrobatchSignature":
        if getattr(microbatch, "packed_seq_params", None) is not None:
            raise ValueError(
                "full-iteration CUDA graph PolicyTraining does not support packed sequences"
            )

        data_dict = getattr(microbatch, "data_dict", None)
        if not isinstance(data_dict, Mapping):
            raise TypeError("ProcessedMicrobatch.data_dict must be a mapping")

        tensors: list[tuple[str, TensorSignature]] = []
        for key, value in sorted(data_dict.items()):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    "full-iteration CUDA graph static inputs require a tensor-only "
                    f"data_dict; key {key!r} has {type(value).__name__}"
                )
            tensors.append((f"data_dict.{key}", TensorSignature.from_tensor(value)))

        for field_name in (
            "input_ids",
            "input_ids_cp_sharded",
            "attention_mask",
            "position_ids",
            "cu_seqlens_padded",
            "mtp_loss_mask",
            "routed_experts",
            "routed_experts_cp_sharded",
        ):
            value = getattr(microbatch, field_name, None)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"ProcessedMicrobatch.{field_name} must be a tensor or None"
                    )
                tensors.append((field_name, TensorSignature.from_tensor(value)))

        return cls(tensors=tuple(tensors))

    def require_match(
        self,
        actual: "StaticMicrobatchSignature",
        *,
        stage: str,
        microbatch: int,
    ) -> None:
        if self != actual:
            raise ValueError(
                "full-iteration CUDA graph static input signature changed for "
                f"{stage} microbatch {microbatch}: expected {self}, got {actual}"
            )


@dataclass(frozen=True)
class FullCudaGraphCallSignature:
    """Python-side schedule properties baked into one captured graph."""

    num_microbatches: int
    seq_length: int
    micro_batch_size: int
    loss_signature: str

    def require_match(self, actual: "FullCudaGraphCallSignature") -> None:
        if self != actual:
            raise ValueError(
                "full-iteration CUDA graph call signature changed: "
                f"expected {self}, got {actual}"
            )


class ProcessedMicrobatchStaticBufferLoader:
    """Copy processed NeMo-RL microbatches into stable tensor allocations."""

    def __init__(self) -> None:
        self._buffers: dict[tuple[str, int], dict[str, Any]] = {}
        self._signatures: dict[tuple[str, int], StaticMicrobatchSignature] = {}
        self._stream: Optional[torch.cuda.Stream] = None

    @staticmethod
    def _serialize(microbatch: Any) -> dict[str, Any]:
        return {
            "data_dict": dict(microbatch.data_dict),
            "input_ids": microbatch.input_ids,
            "input_ids_cp_sharded": microbatch.input_ids_cp_sharded,
            "attention_mask": microbatch.attention_mask,
            "position_ids": microbatch.position_ids,
            "packed_seq_params": microbatch.packed_seq_params,
            "cu_seqlens_padded": microbatch.cu_seqlens_padded,
            "mtp_loss_mask": microbatch.mtp_loss_mask,
            "routed_experts": microbatch.routed_experts,
            "routed_experts_cp_sharded": microbatch.routed_experts_cp_sharded,
        }

    @staticmethod
    def _clone_structure(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {
                key: ProcessedMicrobatchStaticBufferLoader._clone_structure(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _copy_structure(target: Any, source: Any) -> None:
        if isinstance(source, torch.Tensor):
            target.copy_(source, non_blocking=True)
            return
        if isinstance(source, dict):
            for key, item in source.items():
                ProcessedMicrobatchStaticBufferLoader._copy_structure(target[key], item)

    @staticmethod
    def _deserialize(template: Any, static: dict[str, Any]) -> Any:
        data_dict_type = type(template.data_dict)
        return type(template)(
            data_dict=data_dict_type(static["data_dict"]),
            input_ids=static["input_ids"],
            input_ids_cp_sharded=static["input_ids_cp_sharded"],
            attention_mask=static["attention_mask"],
            position_ids=static["position_ids"],
            packed_seq_params=static["packed_seq_params"],
            cu_seqlens_padded=static["cu_seqlens_padded"],
            mtp_loss_mask=static["mtp_loss_mask"],
            routed_experts=static["routed_experts"],
            routed_experts_cp_sharded=static["routed_experts_cp_sharded"],
        )

    def __call__(self, inputs: Any, stage: str, microbatch: int) -> Any:
        if stage != "training":
            raise RuntimeError(
                "NeMo-RL full-iteration CUDA graph supports PolicyTraining only"
            )
        signature = StaticMicrobatchSignature.from_microbatch(inputs)
        key = (stage, microbatch)
        expected = self._signatures.get(key)
        if expected is None:
            self._signatures[key] = signature
        else:
            expected.require_match(signature, stage=stage, microbatch=microbatch)

        serialized = self._serialize(inputs)
        if key not in self._buffers:
            self._buffers[key] = self._clone_structure(serialized)
        else:
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            current_stream = torch.cuda.current_stream()
            self._stream.wait_stream(current_stream)
            with torch.cuda.stream(self._stream):
                self._copy_structure(self._buffers[key], serialized)
            current_stream.wait_stream(self._stream)
        return self._deserialize(inputs, self._buffers[key])

    def reset(self, stage: Optional[str] = None) -> None:
        if stage is None:
            self._buffers.clear()
            self._signatures.clear()
            return
        for key in tuple(self._buffers):
            if key[0] == stage:
                del self._buffers[key]
        for key in tuple(self._signatures):
            if key[0] == stage:
                del self._signatures[key]


class _NemoRLFullCudaGraphWrapperMixin:
    """Validation and lifecycle layer mixed into MCore's wrapper at runtime."""

    static_loader: ProcessedMicrobatchStaticBufferLoader
    _expected_call_signature: Optional[FullCudaGraphCallSignature]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("forward_only"):
            raise RuntimeError(
                "NeMo-RL full-iteration CUDA graph supports PolicyTraining only; "
                "Logprob/eval forward-only calls are not supported"
            )
        signature = kwargs.pop("nemo_rl_signature", None)
        if not isinstance(signature, FullCudaGraphCallSignature):
            raise TypeError(
                "nemo_rl_signature is required for full-iteration CUDA graph"
            )
        if self._expected_call_signature is None:
            self._expected_call_signature = signature
        else:
            self._expected_call_signature.require_match(signature)
        return super().__call__(*args, **kwargs)  # type: ignore[misc]

    def reset_cuda_graph(self, stage: Optional[str] = None) -> None:
        super().reset_cuda_graph(stage=stage)  # type: ignore[misc]
        self.static_loader.reset(stage=stage)
        if stage is None or stage == "training":
            self._expected_call_signature = None


@lru_cache(maxsize=None)
def _wrapper_type(upstream_wrapper_cls: type) -> type:
    return type(
        "_NemoRLFullCudaGraphWrapper",
        (_NemoRLFullCudaGraphWrapperMixin, upstream_wrapper_cls),
        {},
    )


class NemoRLFullCudaGraphWrapper:
    """Factory returning an MCore FullCudaGraphWrapper with NeMo-RL adapters."""

    def __new__(
        cls,
        forward_backward_func: Callable[..., Any],
        *,
        cuda_graph_warmup_steps: int,
        use_single_mempool: bool,
        upstream_wrapper_cls: Optional[type] = None,
    ) -> Any:
        del cls
        if upstream_wrapper_cls is None:
            from megatron.core.full_cuda_graph import FullCudaGraphWrapper

            upstream_wrapper_cls = FullCudaGraphWrapper
        concrete_cls = _wrapper_type(upstream_wrapper_cls)
        instance = concrete_cls(
            forward_backward_func,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
            use_single_mempool=use_single_mempool,
        )
        instance.static_loader = ProcessedMicrobatchStaticBufferLoader()
        instance._expected_call_signature = None
        instance.reset_cuda_graph()
        return instance


def validate_full_cuda_graph_policy_config(
    config: Mapping[str, Any], *, init_optimizer: bool
) -> None:
    """Fail at setup for modes outside the first graph-supported slice."""
    megatron_cfg = config["megatron_cfg"]
    if megatron_cfg.get("cuda_graph_impl") != "full_iteration":
        return

    errors: list[str] = []
    if not init_optimizer:
        errors.append("an optimizer-backed PolicyTraining worker is required")
    if config["dynamic_batching"]["enabled"]:
        errors.append("dynamic batching is not supported")
    sequence_packing = config.get("sequence_packing")
    if sequence_packing is not None and sequence_packing["enabled"]:
        errors.append("sequence packing is not supported")
    if megatron_cfg["context_parallel_size"] != 1:
        errors.append("context parallelism must be 1")
    generation = config.get("generation")
    if generation is not None and generation["colocated"]["enabled"]:
        errors.append("colocated generation/refit is not supported")
    if generation is not None and generation.get("backend") == "megatron":
        errors.append("Megatron generation refit/offload is not supported")

    if errors:
        raise ValueError(
            "megatron_cfg.cuda_graph_impl=full_iteration currently requires:\n  - "
            + "\n  - ".join(errors)
        )


def require_supported_full_cuda_graph_operation(
    *, enabled: bool, operation: _SUPPORTED_OPERATIONS
) -> None:
    """Reject operations that would replay a graph against invalid state."""
    if not enabled or operation == "policy_training":
        return
    labels = {
        "logprob": "Logprob",
        "eval": "evaluation",
        "split_policy_training": "split/async PolicyTraining",
        "colocated_refit": "colocated refit/offload",
    }
    raise RuntimeError(
        f"NeMo-RL full-iteration CUDA graph does not support {labels[operation]} yet"
    )


def _stable_signature(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if isinstance(value, Mapping):
        items = ",".join(
            f"{key!r}:{_stable_signature(item)}"
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return "{" + items + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_signature(item) for item in value) + "]"
    raise TypeError(
        "full-iteration CUDA graph loss configuration must contain only static "
        f"Python values; got {type(value).__name__}"
    )


def full_cuda_graph_loss_signature(loss_fn: Any) -> str:
    """Return a stable signature for supported graph-safe policy losses."""
    if type(loss_fn).__name__ not in {"ClippedPGLossFn", "NLLLossFn"}:
        raise TypeError(
            "full-iteration CUDA graph PolicyTraining currently supports only "
            "ClippedPGLossFn and NLLLossFn"
        )
    attributes = {
        key: value
        for key, value in vars(loss_fn).items()
        if key != "metric_normalizations"
    }
    return (
        f"{type(loss_fn).__module__}.{type(loss_fn).__qualname__}:"
        + _stable_signature(attributes)
    )


def attach_full_cuda_graph_normalizers(
    data_iterator: Any,
    *,
    global_valid_seqs: torch.Tensor,
    global_valid_toks: torch.Tensor,
) -> Any:
    """Move changing normalization tensors into the graph's static inputs."""
    for microbatch in data_iterator:
        data_dict = type(microbatch.data_dict)(dict(microbatch.data_dict))
        data_dict[FULL_CUDA_GRAPH_GLOBAL_VALID_SEQS] = global_valid_seqs
        data_dict[FULL_CUDA_GRAPH_GLOBAL_VALID_TOKS] = global_valid_toks
        yield type(microbatch)(
            data_dict=data_dict,
            input_ids=microbatch.input_ids,
            input_ids_cp_sharded=microbatch.input_ids_cp_sharded,
            attention_mask=microbatch.attention_mask,
            position_ids=microbatch.position_ids,
            packed_seq_params=microbatch.packed_seq_params,
            cu_seqlens_padded=microbatch.cu_seqlens_padded,
            mtp_loss_mask=microbatch.mtp_loss_mask,
            routed_experts=microbatch.routed_experts,
            routed_experts_cp_sharded=microbatch.routed_experts_cp_sharded,
        )


def materialize_full_cuda_graph_metrics(
    metrics: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy captured scalar metrics once per dtype/device and return Python values."""
    materialized = [dict(metric_dict) for metric_dict in metrics]
    groups: dict[
        tuple[torch.device, torch.dtype], list[tuple[int, str, torch.Tensor]]
    ] = {}
    for metric_index, metric_dict in enumerate(metrics):
        for name, value in metric_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim != 0:
                raise ValueError(
                    "full-iteration CUDA graph metrics must be scalar tensors; "
                    f"{name!r} has shape {tuple(value.shape)}"
                )
            groups.setdefault((value.device, value.dtype), []).append(
                (metric_index, name, value)
            )

    for entries in groups.values():
        host_values = torch.stack([value.detach() for _, _, value in entries]).cpu()
        for (metric_index, name, _), host_value in zip(
            entries, host_values.tolist(), strict=True
        ):
            materialized[metric_index][name] = host_value
    return materialized


def build_full_cuda_graph_schedule(
    *,
    raw_schedule: Callable[..., Any],
    model_config: Any,
    model: Any,
    optimizer: Any,
    copy_main_params: bool,
    paged_stash_cls: Optional[type] = None,
    upstream_wrapper_cls: Optional[type] = None,
) -> tuple[Callable[..., Any], Any]:
    """Compose MCore FullCudaGraphWrapper and optional PagedStashRunner."""
    graph = NemoRLFullCudaGraphWrapper(
        raw_schedule,
        cuda_graph_warmup_steps=model_config.cuda_graph_warmup_steps,
        use_single_mempool=model_config.cuda_graph_use_single_mempool,
        upstream_wrapper_cls=upstream_wrapper_cls,
    )
    schedule: Callable[..., Any] = graph
    if model_config.moe_expert_rank_capacity_factor is not None:
        if paged_stash_cls is None:
            from megatron.core.transformer.moe.paged_stash import PagedStashRunner

            paged_stash_cls = PagedStashRunner
        model_chunks = model if isinstance(model, list) else [model]
        schedule = paged_stash_cls(
            model_config,
            copy_main_params,
            model_chunks,
            optimizer,
            graph,
        )
    return schedule, graph
