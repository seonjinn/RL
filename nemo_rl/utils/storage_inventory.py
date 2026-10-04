# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Opt-in, read-only storage ownership probe; not an offload implementation."""

import json
import os
import socket
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass
class StorageRecord:
    device: str
    pointer: int
    nbytes: int
    aliases: list[str] = field(default_factory=list)


@dataclass
class StorageInventory:
    storages: list[StorageRecord] = field(default_factory=list)
    zero_storage_aliases: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def storage_inventory_enabled() -> bool:
    """Return whether this isolated diagnostic is explicitly enabled."""
    return os.environ.get("NRL_STORAGE_INVENTORY") == "1"


def _existing_tensors(
    name: str, value: Any, *, depth: int = 0
) -> Iterator[tuple[str, torch.Tensor]]:
    if depth > 8:
        raise ValueError(f"Storage probe container nesting exceeds eight: {name}")
    attributes = getattr(value, "__dict__", {})
    # TE wrappers have empty nominal storage. Read existing backing fields;
    # never use repr, dequantize, prepare_for_saving or create grouped members.
    payload_fields = (
        "_rowwise_data",
        "_columnwise_data",
        "_rowwise_scale_inv",
        "_columnwise_scale_inv",
        "rowwise_data",
        "columnwise_data",
        "scale_inv",
        "columnwise_scale_inv",
        "amax",
        "columnwise_amax",
        "scale",
        "first_dims",
        "last_dims",
        "tensor_offsets",
        "quantized_tensors",
    )
    fields = [key for key in payload_fields if key in attributes]
    if fields:
        for key in fields:
            yield from _existing_tensors(
                f"{name}.{key}", attributes[key], depth=depth + 1
            )
    elif isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, Mapping):
        for index, (key, item) in enumerate(value.items()):
            label = key if isinstance(key, str) else str(index)
            yield from _existing_tensors(f"{name}[{label}]", item, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _existing_tensors(f"{name}[{index}]", item, depth=depth + 1)


def collect_tensor_storages(
    named_values: Iterable[tuple[str, Any]],
) -> StorageInventory:
    """Count allocated storage bytes once, retaining each observed owner path."""
    result = StorageInventory()
    indexed: dict[tuple[str, int, int], StorageRecord] = {}
    iterator = iter(named_values)
    while True:
        try:
            name, value = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            result.errors.append(f"owner traversal: {type(exc).__name__}: {exc}")
            break
        try:
            for alias, tensor in _existing_tensors(name, value):
                try:
                    storage = tensor.untyped_storage()
                    nbytes = storage.nbytes()
                    if not nbytes:
                        result.zero_storage_aliases.append(alias)
                        continue
                    device = str(tensor.device)
                    pointer = storage.data_ptr()
                    key = (device, pointer, nbytes)
                    if key not in indexed:
                        record = StorageRecord(device, pointer, nbytes)
                        indexed[key] = record
                        result.storages.append(record)
                    if alias not in indexed[key].aliases:
                        indexed[key].aliases.append(alias)
                except Exception as exc:
                    result.errors.append(f"{alias}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            result.errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return result


def _tensor_owners(
    name: str, value: Any, *, depth: int = 0
) -> Iterator[tuple[str, Any]]:
    if depth > 8:
        return
    yield name, value
    if isinstance(value, torch.Tensor):
        attributes = vars(value)
        for key in (
            "main_param",
            "main_grad",
            "main_grad_copy_in_grad_buffer",
            "decoupled_grad",
        ):
            attached = attributes.get(key)
            if attached is not None:
                yield f"{name}.{key}", attached
                if isinstance(attached, torch.Tensor) and attached.is_leaf:
                    yield f"{name}.{key}.grad", attached.grad
        if value.is_leaf:
            yield f"{name}.grad", value.grad
    elif isinstance(value, Mapping):
        for index, (key, item) in enumerate(value.items()):
            label = key if isinstance(key, str) else str(index)
            yield from _tensor_owners(f"{name}[{label}]", item, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensor_owners(f"{name}[{index}]", item, depth=depth + 1)


def _policy_owners(model: Any, optimizer: Any) -> Iterator[tuple[str, Any]]:
    seen_optimizers: set[int] = set()

    def visit_model(name: str, chunk: Any, depth: int = 0) -> Iterator[tuple[str, Any]]:
        if depth > 8:
            return
        if isinstance(chunk, (list, tuple)):
            for index, item in enumerate(chunk):
                yield from visit_model(f"{name}[{index}]", item, depth + 1)
            return
        if not isinstance(chunk, torch.nn.Module):
            return
        for key, parameter in chunk.named_parameters(remove_duplicate=False):
            yield from _tensor_owners(f"{name}.parameters.{key}", parameter)
        for key, buffer in chunk.named_buffers(remove_duplicate=False):
            yield f"{name}.module_buffers.{key}", buffer
        for module_name, module in chunk.named_modules(remove_duplicate=False):
            attributes = vars(module)
            prefix = f"{name}.modules.{module_name}"
            yield f"{prefix}._fp8_workspaces", attributes.get("_fp8_workspaces")
            dispatcher = attributes.get("token_dispatcher")
            if dispatcher is not None:
                for key, value in getattr(dispatcher, "__dict__", {}).items():
                    yield f"{prefix}.dispatcher.{key}", value
            for group_name in ("buffers", "expert_parallel_buffers"):
                buffers = attributes.get(group_name, [])
                if not isinstance(buffers, (list, tuple)):
                    continue
                for index, buffer in enumerate(buffers):
                    buffer_name = f"{prefix}.{group_name}[{index}]"
                    for key in (
                        "param_data",
                        "grad_data",
                        "shared_buffer",
                        "extra_main_grads",
                        "param_data_cpu",
                    ):
                        yield (
                            f"{buffer_name}.{key}",
                            getattr(buffer, "__dict__", {}).get(key),
                        )
                    for bucket_index, bucket in enumerate(
                        getattr(buffer, "__dict__", {}).get("buckets", [])
                    ):
                        for key in ("param_data", "grad_data", "layerwise_gather_list"):
                            yield (
                                f"{buffer_name}.buckets[{bucket_index}].{key}",
                                getattr(bucket, "__dict__", {}).get(key),
                            )
            for group_name in ("bucket_groups", "expert_parallel_bucket_groups"):
                for index, group in enumerate(attributes.get(group_name, [])):
                    for key in (
                        "cached_param_buffer_shard_list",
                        "cached_grad_buffer_shard_list",
                    ):
                        yield (
                            f"{prefix}.{group_name}[{index}].{key}",
                            getattr(group, "__dict__", {}).get(key),
                        )

    def visit_optimizer(name: str, opt: Any) -> Iterator[tuple[str, Any]]:
        if opt is None or id(opt) in seen_optimizers:
            return
        seen_optimizers.add(id(opt))
        attributes = vars(opt)
        for key in (
            "shard_fp32_from_float16_groups",
            "model_float16_groups",
            "model_fp32_groups",
            "shard_float16_groups",
            "shard_fp32_groups",
            "float16_groups",
            "fp32_from_float16_groups",
            "fp32_from_fp32_groups",
            "param_groups",
            "cpu_param_groups",
            "gpu_param_groups",
            "param_to_fp32_param",
            "gpu_params_map_cpu_copy",
            "param_to_inner_param",
            "cpu_copy_map_grad",
        ):
            yield from _tensor_owners(f"{name}.{key}", attributes.get(key))
        for key in ("state", "_scales", "_dummy_overflow_buf"):
            yield f"{name}.{key}", attributes.get(key)
        yield from visit_model(f"{name}.model_chunks", attributes.get("model_chunks"))
        for key in ("chained_optimizers", "cpu_optimizers"):
            for index, child in enumerate(attributes.get(key, [])):
                yield from visit_optimizer(f"{name}.{key}[{index}]", child)
        for key in ("optimizer", "gpu_optimizer"):
            yield from visit_optimizer(f"{name}.{key}", attributes.get(key))

    yield from visit_model("model", model)
    yield from visit_optimizer("optimizer", optimizer)


def collect_policy_storages(model: Any, optimizer: Any) -> StorageInventory:
    """Inspect source-confirmed owner fields, without state/export APIs."""
    return collect_tensor_storages(_policy_owners(model, optimizer))


def _cuda_boundary(rank: int, phase: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rank": rank,
        "phase": phase,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "time_ns": time.time_ns(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch.cuda.is_initialized():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        result.update(
            device=device,
            device_uuid=str(getattr(properties, "uuid", "unavailable")),
            free_bytes=free,
            total_bytes=total,
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),
        )
    return result


def _emit_metadata(label: str, result: dict[str, Any]) -> None:
    try:
        print(label + " " + json.dumps(result, separators=(",", ":")), flush=True)
    except Exception:
        pass


def log_policy_storage_inventory(
    model: Any, optimizer: Any, *, rank: int, phase: str
) -> None:
    """Emit metadata only, with no allocator reset or GPU synchronization."""
    if not storage_inventory_enabled():
        return
    try:
        inventory = collect_policy_storages(model, optimizer)
        result = _cuda_boundary(rank, phase)
        totals: dict[str, int] = {}
        for record in inventory.storages:
            totals[record.device] = totals.get(record.device, 0) + record.nbytes
        result.update(inventory=asdict(inventory), storage_bytes_by_device=totals)
        _emit_metadata("[NRL_STORAGE_INVENTORY]", result)
    except Exception as exc:
        _emit_metadata(
            "[NRL_STORAGE_INVENTORY_ERROR]",
            {"rank": rank, "phase": phase, "error_type": type(exc).__name__},
        )


def log_rollout_storage_boundary(*, rank: int, phase: str) -> None:
    """Read counters on the actual vLLM CUDA worker, including its GPU UUID."""
    if storage_inventory_enabled():
        try:
            _emit_metadata(
                "[NRL_ROLLOUT_STORAGE_BOUNDARY]", _cuda_boundary(rank, phase)
            )
        except Exception as exc:
            _emit_metadata(
                "[NRL_STORAGE_INVENTORY_ERROR]",
                {"rank": rank, "phase": phase, "error_type": type(exc).__name__},
            )


def log_wake_event(
    *, phase: str, tags: Any, error: BaseException | None = None
) -> None:
    """Label the native wake call without touching CUDA in its control actor."""
    if storage_inventory_enabled():
        try:
            result = {
                "phase": phase,
                "tags": tags,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "time_ns": time.time_ns(),
            }
            if error is not None:
                result["error_type"] = type(error).__name__
                result["error"] = str(error)
            print("[NRL_ROLLOUT_WAKE] " + json.dumps(result), flush=True)
        except Exception:
            # Diagnostics must not mask a native wake exception, even if stdout
            # has closed or exception text is not serializable.
            pass
