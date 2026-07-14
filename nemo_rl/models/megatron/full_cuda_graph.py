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

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as dataclass_field
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
class FullCudaGraphExecutionStats:
    """Cumulative successful calls made by one NeMo-RL graph wrapper."""

    warmup_calls: int
    capture_calls: int
    replay_calls: int
    reset_calls: int


@dataclass(frozen=True, repr=False)
class _FullCudaGraphStorageEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    storage_data_ptr: int
    effective_data_ptr: int
    storage_offset: int
    stride: tuple[int, ...]

    def canonical_value(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "effective_data_ptr": self.effective_data_ptr,
            "name": self.name,
            "shape": self.shape,
            "storage_data_ptr": self.storage_data_ptr,
            "storage_offset": self.storage_offset,
            "stride": self.stride,
        }


_SAFE_STORAGE_NAME = re.compile(r"^[A-Za-z0-9_.\[\]=:-]+$")
_STABLE_SCALAR_TYPES = (
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
    torch.dtype,
    torch.device,
)


def _safe_storage_name(value: Any, *, kind: str) -> str:
    if type(value) is not str or _SAFE_STORAGE_NAME.fullmatch(value) is None:
        raise TypeError(
            "full-iteration CUDA graph storage signature encountered an "
            f"unsupported {kind}"
        ) from None
    return value


def _is_custom_fsdp_model(value: Any) -> bool:
    value_type = type(value)
    return value_type.__name__ == "FullyShardedDataParallel" or (
        value_type.__module__.startswith("megatron.core.distributed.fsdp")
    )


def _storage_identifier(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _raise_traversal_unavailable(*, component: str, field: str) -> None:
    raise TypeError(
        "full-iteration CUDA graph traversal unavailable "
        f"component_id_sha256={_storage_identifier(component)} field={field} "
        "reason=unsupported_traversal"
    ) from None


_MISSING_TRAVERSAL_ATTRIBUTE = object()


def _read_traversal_attribute(
    value: Any,
    *,
    component: str,
    field: str,
    attribute: Optional[str] = None,
    default: Any = _MISSING_TRAVERSAL_ATTRIBUTE,
) -> Any:
    attribute = attribute or field
    try:
        if default is _MISSING_TRAVERSAL_ATTRIBUTE:
            return getattr(value, attribute)
        return getattr(value, attribute, default)
    except Exception:
        _raise_traversal_unavailable(component=component, field=field)


def _iterate_traversal(value: Any, *, component: str, field: str) -> Any:
    try:
        iterator = iter(value)
    except Exception:
        _raise_traversal_unavailable(component=component, field=field)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except Exception:
            _raise_traversal_unavailable(component=component, field=field)


def _call_traversal(
    callable_value: Any,
    *args: Any,
    component: str,
    field: str,
) -> Any:
    try:
        return callable_value(*args)
    except Exception:
        _raise_traversal_unavailable(component=component, field=field)


def _mapping_items(value: Mapping[Any, Any], *, component: str) -> Any:
    field = "mapping_items"
    items = _read_traversal_attribute(
        value,
        component=component,
        field=field,
        attribute="items",
    )
    iterable = _call_traversal(
        items,
        component=component,
        field=field,
    )
    for item in _iterate_traversal(
        iterable,
        component=component,
        field=field,
    ):
        if type(item) not in (list, tuple) or len(item) != 2:
            _raise_traversal_unavailable(component=component, field=field)
        yield item[0], item[1]


def _read_tensor_attribute(tensor: torch.Tensor, *, name: str, attribute: str) -> Any:
    try:
        return getattr(tensor, attribute, None)
    except Exception:
        raise TypeError(
            "full-iteration CUDA graph tensor attribute unavailable "
            f"tensor_id_sha256={_storage_identifier(name)} field={attribute} "
            "reason=unsupported_tensor_attribute"
        ) from None


def _is_unsupported_distributed_tensor(value: torch.Tensor, *, name: str) -> bool:
    value_type = type(value)
    if value_type.__name__ == "DTensor" or value_type.__module__.startswith(
        "torch.distributed.tensor"
    ):
        return True
    fsdp_parameter = _read_tensor_attribute(
        value, name=name, attribute="__fsdp_param__"
    )
    try:
        return bool(fsdp_parameter)
    except Exception:
        raise TypeError(
            "full-iteration CUDA graph tensor attribute unavailable "
            f"tensor_id_sha256={_storage_identifier(name)} field=__fsdp_param__ "
            "reason=unsupported_tensor_attribute"
        ) from None


def _read_storage_metadata(
    tensor: torch.Tensor, *, name: str
) -> _FullCudaGraphStorageEntry:
    if _is_unsupported_distributed_tensor(tensor, name=name):
        raise TypeError(
            "full-iteration CUDA graph storage signature rejects DTensor/custom-FSDP "
            f"tensor_id_sha256={_storage_identifier(name)}"
        ) from None

    fields: dict[str, Any] = {}
    readers: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("shape", lambda: tuple(int(value) for value in tensor.shape)),
        ("dtype", lambda: str(tensor.dtype)),
        ("device", lambda: str(tensor.device)),
        ("storage_data_ptr", lambda: int(tensor.untyped_storage().data_ptr())),
        ("effective_data_ptr", lambda: int(tensor.data_ptr())),
        ("storage_offset", lambda: int(tensor.storage_offset())),
        ("stride", lambda: tuple(int(value) for value in tensor.stride())),
    )
    for field, reader in readers:
        try:
            fields[field] = reader()
        except Exception:
            raise TypeError(
                "full-iteration CUDA graph storage metadata unavailable "
                f"tensor_id_sha256={_storage_identifier(name)} field={field} "
                "reason=unsupported_tensor_metadata"
            ) from None
    return _FullCudaGraphStorageEntry(name=name, **fields)


class _FullCudaGraphStorageCapture:
    def __init__(self) -> None:
        self.entries: list[_FullCudaGraphStorageEntry] = []
        self.logical_names: set[str] = set()
        self.seen_tensor_ids: set[int] = set()

    def add_tensor(self, tensor: Any, *, name: str) -> None:
        if name in self.logical_names:
            raise TypeError(
                "full-iteration CUDA graph storage signature encountered a "
                "duplicate logical tensor name "
                f"tensor_id_sha256={_storage_identifier(name)}"
            ) from None
        self.logical_names.add(name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "full-iteration CUDA graph storage signature expected a live tensor "
                f"tensor_id_sha256={_storage_identifier(name)}"
            ) from None
        tensor_id = id(tensor)
        if tensor_id in self.seen_tensor_ids:
            return
        self.entries.append(_read_storage_metadata(tensor, name=name))
        self.seen_tensor_ids.add(tensor_id)


def _optimizer_leaves(optimizer: Any, *, component: str = "optimizer") -> list[Any]:
    children = _read_traversal_attribute(
        optimizer,
        component=component,
        field="chained_optimizers",
        default=None,
    )
    if children is None:
        return [optimizer]
    if not isinstance(children, (list, tuple)):
        raise TypeError(
            "full-iteration CUDA graph storage signature requires deterministic "
            "ChainedOptimizer children"
        ) from None
    leaves: list[Any] = []
    for child_index, child in enumerate(
        _iterate_traversal(
            children,
            component=component,
            field="chained_optimizers",
        )
    ):
        leaves.extend(
            _optimizer_leaves(
                child,
                component=f"{component}.child[{child_index}]",
            )
        )
    return leaves


def _mapping_key_canonical_value(
    key: Any, parameter_names: Mapping[int, str]
) -> tuple[Any, ...]:
    parameter_name = parameter_names.get(id(key))
    if parameter_name is not None:
        return ("parameter", parameter_name)
    if type(key) is str:
        return ("str", _safe_storage_name(key, kind="optimizer mapping key"))
    if key is None or type(key) in (bool, int, float):
        return (type(key).__name__, repr(key))
    if type(key) is tuple:
        return (
            "tuple",
            tuple(_mapping_key_canonical_value(part, parameter_names) for part in key),
        )
    raise TypeError(
        "full-iteration CUDA graph storage signature encountered a "
        "nondeterministic optimizer mapping key"
    ) from None


def _mapping_key_name(key: Any, parameter_names: Mapping[int, str]) -> str:
    canonical = json.dumps(
        _mapping_key_canonical_value(key, parameter_names),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"key_sha256={hashlib.sha256(canonical).hexdigest()}"


def _capture_optimizer_value(
    capture: _FullCudaGraphStorageCapture,
    value: Any,
    *,
    path: str,
    parameter_names: Mapping[int, str],
) -> None:
    if isinstance(value, torch.Tensor):
        capture.add_tensor(value, name=path)
        return
    if isinstance(value, Mapping):
        named_items = [
            (_mapping_key_name(key, parameter_names), item)
            for key, item in _mapping_items(value, component=path)
        ]
        key_names = [key_name for key_name, _ in named_items]
        if len(set(key_names)) != len(key_names):
            raise TypeError(
                "full-iteration CUDA graph storage signature encountered "
                "nondeterministic optimizer mapping keys"
            ) from None
        named_items.sort(key=lambda item: item[0])
        for key_name, item in named_items:
            _capture_optimizer_value(
                capture,
                item,
                path=f"{path}.{key_name}",
                parameter_names=parameter_names,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(
            _iterate_traversal(
                value,
                component=path,
                field="optimizer_sequence",
            )
        ):
            _capture_optimizer_value(
                capture,
                item,
                path=f"{path}[{index}]",
                parameter_names=parameter_names,
            )
        return
    if isinstance(value, (set, frozenset)):
        raise TypeError(
            "full-iteration CUDA graph storage signature encountered a "
            f"nondeterministic optimizer container path={path}"
        ) from None
    if not isinstance(value, _STABLE_SCALAR_TYPES):
        raise TypeError(
            "full-iteration CUDA graph storage signature encountered an "
            f"unsupported optimizer value path={path}"
        ) from None


@dataclass(frozen=True, repr=False, slots=True)
class FullCudaGraphStorageSignature:
    """Digestible live-storage snapshot for graph-resident policy state."""

    _entries: tuple[_FullCudaGraphStorageEntry, ...]
    _digest: str = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = [entry.name for entry in self._entries]
        if len(names) != len(set(names)):
            raise TypeError(
                "full-iteration CUDA graph storage signature encountered a "
                "duplicate logical tensor name"
            ) from None
        canonical = json.dumps(
            [entry.canonical_value() for entry in self._entries],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object.__setattr__(self, "_digest", hashlib.sha256(canonical).hexdigest())

    @classmethod
    def capture(cls, model: Any, optimizer: Any) -> "FullCudaGraphStorageSignature":
        """Capture live model, gradient, and optimizer tensors without state_dict."""
        model_chunks = model if isinstance(model, (list, tuple)) else (model,)

        capture = _FullCudaGraphStorageCapture()
        saw_model_chunk = False
        for chunk_index, model_chunk in enumerate(
            _iterate_traversal(
                model_chunks,
                component="model",
                field="model_chunks",
            )
        ):
            saw_model_chunk = True
            chunk_component = f"model_chunk[{chunk_index}]"
            if _is_custom_fsdp_model(model_chunk):
                raise TypeError(
                    "full-iteration CUDA graph storage signature rejects custom FSDP"
                ) from None
            named_parameters = _read_traversal_attribute(
                model_chunk,
                component=chunk_component,
                field="named_parameters",
                default=None,
            )
            if not callable(named_parameters):
                raise TypeError(
                    "full-iteration CUDA graph storage signature requires "
                    f"model_chunk[{chunk_index}].named_parameters"
                ) from None
            named_parameter_items = _call_traversal(
                named_parameters,
                component=chunk_component,
                field="named_parameters",
            )
            for named_parameter_item in _iterate_traversal(
                named_parameter_items,
                component=chunk_component,
                field="named_parameters",
            ):
                if (
                    type(named_parameter_item) not in (list, tuple)
                    or len(named_parameter_item) != 2
                ):
                    _raise_traversal_unavailable(
                        component=chunk_component,
                        field="named_parameters",
                    )
                parameter_name, parameter = named_parameter_item
                if (
                    type(parameter_name) is not str
                    or _SAFE_STORAGE_NAME.fullmatch(parameter_name) is None
                ):
                    _raise_traversal_unavailable(
                        component=chunk_component,
                        field="parameter_name",
                    )
                logical_name = f"model_chunk[{chunk_index}].parameter.{parameter_name}"
                capture.add_tensor(parameter, name=logical_name)
                for gradient_name in ("main_grad", "grad"):
                    gradient = _read_tensor_attribute(
                        parameter, name=logical_name, attribute=gradient_name
                    )
                    if gradient is not None:
                        capture.add_tensor(
                            gradient, name=f"{logical_name}.{gradient_name}"
                        )
        if not saw_model_chunk:
            raise TypeError(
                "full-iteration CUDA graph storage signature requires model chunks"
            ) from None

        leaves = _optimizer_leaves(optimizer)
        for leaf_index, leaf in enumerate(leaves):
            leaf_path = f"optimizer_leaf[{leaf_index}]"
            param_groups = _read_traversal_attribute(
                leaf,
                component=leaf_path,
                field="param_groups",
            )
            state = _read_traversal_attribute(
                leaf,
                component=leaf_path,
                field="state",
            )
            if not isinstance(param_groups, (list, tuple)):
                raise TypeError(
                    "full-iteration CUDA graph optimizer param_groups must be a "
                    f"deterministic sequence path={leaf_path}"
                ) from None

            parameter_names: dict[int, str] = {}
            for group_index, param_group in enumerate(
                _iterate_traversal(
                    param_groups,
                    component=leaf_path,
                    field="param_groups",
                )
            ):
                group_component = f"{leaf_path}.param_group[{group_index}]"
                if not isinstance(param_group, Mapping):
                    raise TypeError(
                        "full-iteration CUDA graph optimizer parameter group must be "
                        f"a mapping path={leaf_path}.param_group[{group_index}]"
                    ) from None
                parameter_getter = _read_traversal_attribute(
                    param_group,
                    component=group_component,
                    field="params",
                    attribute="get",
                )
                parameters = _call_traversal(
                    parameter_getter,
                    "params",
                    component=group_component,
                    field="params",
                )
                if not isinstance(parameters, (list, tuple)):
                    raise TypeError(
                        "full-iteration CUDA graph optimizer params must be a "
                        "deterministic sequence "
                        f"path={leaf_path}.param_group[{group_index}]"
                    ) from None
                for parameter_index, parameter in enumerate(
                    _iterate_traversal(
                        parameters,
                        component=group_component,
                        field="params",
                    )
                ):
                    parameter_name = (
                        f"param_group[{group_index}].param[{parameter_index}]"
                    )
                    parameter_names[id(parameter)] = parameter_name

            _capture_optimizer_value(
                capture,
                param_groups,
                path=f"{leaf_path}.param_groups",
                parameter_names=parameter_names,
            )
            _capture_optimizer_value(
                capture,
                state,
                path=f"{leaf_path}.state",
                parameter_names=parameter_names,
            )

        return cls(tuple(sorted(capture.entries, key=lambda entry: entry.name)))

    def digest(self) -> str:
        return self._digest

    def require_match(self, model: Any, optimizer: Any) -> None:
        actual = type(self).capture(model, optimizer)
        expected_by_name = {entry.name: entry for entry in self._entries}
        actual_by_name = {entry.name: entry for entry in actual._entries}
        names = sorted(expected_by_name.keys() | actual_by_name.keys())
        fields = (
            "shape",
            "dtype",
            "device",
            "storage_data_ptr",
            "effective_data_ptr",
            "storage_offset",
            "stride",
        )
        for name in names:
            expected = expected_by_name.get(name)
            observed = actual_by_name.get(name)
            if expected is None or observed is None:
                self._raise_mismatch(actual, name=name, field="presence")
            assert expected is not None and observed is not None
            for field in fields:
                if getattr(expected, field) != getattr(observed, field):
                    self._raise_mismatch(actual, name=name, field=field)

    def _raise_mismatch(
        self,
        actual: "FullCudaGraphStorageSignature",
        *,
        name: str,
        field: str,
    ) -> None:
        raise RuntimeError(
            "full-iteration CUDA graph storage signature mismatch "
            f"expected_sha256={self.digest()} actual_sha256={actual.digest()} "
            f"tensor_id_sha256={_storage_identifier(name)} field={field} "
            "reason=graph_storage_changed"
        ) from None

    def __repr__(self) -> str:
        return f"FullCudaGraphStorageSignature(sha256={self.digest()})"


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


class FullCudaGraphAuxLossScaleBuffer:
    """Keep the MoE/MTP gradient scale at one graph-stable address."""

    def __init__(self) -> None:
        self._value: Optional[torch.Tensor] = None
        self._signature: Optional[TensorSignature] = None

    def update(self, global_valid_toks: torch.Tensor) -> torch.Tensor:
        """Update the reciprocal token count without replacing captured storage."""
        if global_valid_toks.ndim != 0:
            raise ValueError(
                "full-iteration CUDA graph requires scalar global_valid_toks for "
                "MoE/MTP auxiliary loss scaling"
            )

        updated = global_valid_toks.detach().clamp(min=1).to(torch.float32).reciprocal()
        signature = TensorSignature.from_tensor(updated)
        if self._value is None:
            self._value = updated.clone()
            self._signature = signature
        else:
            if signature != self._signature:
                raise ValueError(
                    "full-iteration CUDA graph auxiliary loss scale signature changed: "
                    f"expected {self._signature}, got {signature}"
                )
            self._value.copy_(updated)
        return self._value


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
    """Validation and lifecycle layer mixed into MCore's wrapper at runtime.

    MCore's full-iteration graph state is class-global. NeMo-RL therefore keeps
    the existing invariant of one wrapper per Ray worker process.
    """

    static_loader: ProcessedMicrobatchStaticBufferLoader
    _expected_call_signature: Optional[FullCudaGraphCallSignature]
    _nemo_rl_bootstrap_reset: bool
    _nemo_rl_capture_calls: int
    _nemo_rl_phase_calls: int
    _nemo_rl_replay_calls: int
    _nemo_rl_reset_calls: int
    _nemo_rl_warmup_calls: int

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
        phase_call = self._nemo_rl_phase_calls
        result = super().__call__(*args, **kwargs)  # type: ignore[misc]
        if phase_call < self.cuda_graph_warmup_steps:
            self._nemo_rl_warmup_calls += 1
        elif phase_call == self.cuda_graph_warmup_steps:
            self._nemo_rl_capture_calls += 1
            self._nemo_rl_replay_calls += 1
        else:
            self._nemo_rl_replay_calls += 1
        self._nemo_rl_phase_calls += 1
        return result

    def execution_stats(self) -> FullCudaGraphExecutionStats:
        """Return an immutable snapshot of cumulative successful calls."""
        return FullCudaGraphExecutionStats(
            warmup_calls=self._nemo_rl_warmup_calls,
            capture_calls=self._nemo_rl_capture_calls,
            replay_calls=self._nemo_rl_replay_calls,
            reset_calls=self._nemo_rl_reset_calls,
        )

    def will_capture_next_call(self) -> bool:
        """Whether the next successful training call is the capture call."""
        return self._nemo_rl_phase_calls == self.cuda_graph_warmup_steps

    def reset_cuda_graph(self, stage: Optional[str] = None) -> None:
        super().reset_cuda_graph(stage=stage)  # type: ignore[misc]
        self.static_loader.reset(stage=stage)
        if stage is None or stage == "training":
            self._expected_call_signature = None
            self._nemo_rl_phase_calls = 0
        if not self._nemo_rl_bootstrap_reset:
            self._nemo_rl_reset_calls += 1


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
        if cuda_graph_warmup_steps < 1:
            raise ValueError(
                "full-iteration CUDA graph warmup steps must be at least 1"
            )
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
        instance._nemo_rl_bootstrap_reset = True
        instance._nemo_rl_capture_calls = 0
        instance._nemo_rl_phase_calls = 0
        instance._nemo_rl_replay_calls = 0
        instance._nemo_rl_reset_calls = 0
        instance._nemo_rl_warmup_calls = 0
        instance.reset_cuda_graph()
        instance._nemo_rl_bootstrap_reset = False
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
    ddp_config = megatron_cfg.get("distributed_data_parallel_config")
    if isinstance(ddp_config, Mapping) and ddp_config.get("use_custom_fsdp"):
        errors.append("custom FSDP/DTensor is not supported")
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
