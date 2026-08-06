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

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntFlag
from typing import Protocol


class _StorageLike(Protocol):
    def data_ptr(self) -> int: ...


class TensorStorageLike(Protocol):
    dtype: object
    device: object
    layout: object

    def data_ptr(self) -> int: ...

    def untyped_storage(self) -> _StorageLike: ...

    def storage_offset(self) -> int: ...

    def size(self) -> Iterable[int]: ...

    def stride(self) -> Iterable[int]: ...


class StorageChange(IntFlag):
    """Kinds of captured training storage whose identity changed."""

    NONE = 0
    MODEL = 1
    GRAD = 2


@dataclass(frozen=True)
class TensorStorageFingerprint:
    """Address and Tensor signature captured by a CUDA Graph."""

    name: str
    storage_ptr: int
    data_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: object
    device: object
    layout: object


@dataclass(frozen=True)
class GraphStorageFingerprint:
    """Model and gradient storage owned by training graph banks."""

    model: tuple[TensorStorageFingerprint, ...]
    grads: tuple[TensorStorageFingerprint, ...]


def fingerprint_named_tensors(
    tensors: Iterable[tuple[str, TensorStorageLike]],
) -> tuple[TensorStorageFingerprint, ...]:
    """Build a deterministic fingerprint without reading Tensor values."""
    fingerprints = []
    for name, tensor in tensors:
        fingerprints.append(
            TensorStorageFingerprint(
                name=name,
                storage_ptr=int(tensor.untyped_storage().data_ptr()),
                data_ptr=int(tensor.data_ptr()),
                storage_offset=int(tensor.storage_offset()),
                shape=tuple(int(dimension) for dimension in tensor.size()),
                stride=tuple(int(dimension) for dimension in tensor.stride()),
                dtype=tensor.dtype,
                device=tensor.device,
                layout=tensor.layout,
            )
        )
    return tuple(sorted(fingerprints, key=lambda fingerprint: fingerprint.name))


def classify_storage_change(
    before: GraphStorageFingerprint,
    after: GraphStorageFingerprint,
) -> StorageChange:
    """Classify address/signature changes without considering Tensor values."""
    change = StorageChange.NONE
    if before.model != after.model:
        change |= StorageChange.MODEL
    if before.grads != after.grads:
        change |= StorageChange.GRAD
    return change


def validate_training_graph_storage_lifecycle(
    *,
    cuda_graph_impl: str,
    generation_colocated: bool | None,
    generation_backend: str | None,
    fp8_enabled: bool,
    use_custom_fsdp: bool,
    offload_optimizer_for_logprob: bool,
) -> None:
    """Reject runtime policies that relocate TE training-graph storage."""
    if cuda_graph_impl != "transformer_engine":
        return
    if generation_colocated:
        raise ValueError(
            "Transformer Engine training CUDA Graph reuse requires "
            "non-colocated generation because colocated refit offloads and "
            "reallocates graph-owned model and gradient storage every step."
        )
    if generation_backend == "megatron":
        raise ValueError(
            "Transformer Engine training CUDA Graph reuse is incompatible with "
            "the Megatron generation backend because its refit lifecycle "
            "offloads graph-owned model and gradient storage."
        )
    if fp8_enabled:
        raise ValueError(
            "Transformer Engine training CUDA Graph reuse currently supports "
            "BF16 only; FP8 hidden metadata and workspace storage are not yet "
            "part of the replay fingerprint."
        )
    if use_custom_fsdp:
        raise ValueError(
            "Transformer Engine training CUDA Graph reuse currently supports "
            "MCore DDP only; custom FSDP storage owners are not yet part of "
            "the replay fingerprint."
        )
    if offload_optimizer_for_logprob:
        raise ValueError(
            "Transformer Engine training CUDA Graph reuse currently requires "
            "offload_optimizer_for_logprob=false."
        )
