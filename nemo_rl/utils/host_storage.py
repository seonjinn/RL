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

from collections.abc import Mapping
from typing import TypedDict

import torch


class CategoryStorage(TypedDict):
    bytes: int
    pinned_bytes: int
    unsupported_tensors: int


class HostStorageInventory(TypedDict):
    categories: dict[str, CategoryStorage]
    union_bytes: int
    cross_category_duplicate_bytes: int


class MegatronHostInventory(HostStorageInventory):
    uncovered: list[str]


def cpu_storage_inventory(groups: Mapping[str, object]) -> HostStorageInventory:
    """Count backing CPU storage in explicit tensor/container roots.

    Views count their entire backing storage, once per category and once in
    the union. This is not RSS: Python objects, allocators, and unreferenced
    caches are not measured. Tensor subclasses require callers to pass their
    physical payload/scale tensors explicitly; otherwise they are reported as
    unsupported. No copies, dequantization, or device synchronization occur.
    """
    union: dict[int, int] = {}
    categories: dict[str, CategoryStorage] = {}
    for name, root in groups.items():
        seen: set[int] = set()
        storages: dict[int, int] = {}
        pinned: dict[int, int] = {}
        unsupported = 0
        pending = [root]
        while pending:
            value = pending.pop()
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(value, torch.Tensor):
                if type(value) not in (torch.Tensor, torch.nn.Parameter):
                    unsupported += 1
                    continue
                if value.device.type != "cpu":
                    continue
                if value.layout != torch.strided:
                    unsupported += 1
                    continue
                storage = value.untyped_storage()
                size = storage.nbytes()
                if size:
                    key = storage.data_ptr()
                    storages[key] = size
                    union[key] = size
                    if storage.is_pinned():
                        pinned[key] = size
            elif isinstance(value, Mapping):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
        categories[name] = {
            "bytes": sum(storages.values()),
            "pinned_bytes": sum(pinned.values()),
            "unsupported_tensors": unsupported,
        }
    union_bytes = sum(union.values())
    return {
        "categories": categories,
        "union_bytes": union_bytes,
        "cross_category_duplicate_bytes": (
            sum(category["bytes"] for category in categories.values()) - union_bytes
        ),
    }


def megatron_cpu_storage_inventory(
    model: object,
    reference: object,
    optimizer: object,
    *,
    reference_swap: object = None,
) -> MegatronHostInventory:
    """Inspect existing ordinary DDP/optimizer CPU roots without exporting.

    Tensor subclasses and other workers are not covered by this adapter.
    Callers must keep roots alive and avoid concurrent offload/reload while
    taking this diagnostic snapshot. Returned data retains no tensor roots.
    """
    backups: list[object] = []
    states: list[object] = []
    masters: list[object] = []
    uncovered: list[str] = []
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for index, chunk in enumerate(chunks):
        attrs = vars(chunk) if hasattr(chunk, "__dict__") else {}
        if "buffers" not in attrs or "expert_parallel_buffers" not in attrs:
            uncovered.append(f"model.{index}: no ordinary DDP buffer lists")
        for kind in ("buffers", "expert_parallel_buffers"):
            for buffer in attrs.get(kind, ()):
                backups.append(vars(buffer).get("param_data_cpu"))

    seen: set[int] = set()
    pending = [optimizer]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        attrs = vars(current) if hasattr(current, "__dict__") else {}
        children = attrs.get("chained_optimizers")
        if children is not None:
            pending.extend(children)
            continue
        for key in (
            "shard_fp32_from_float16_groups",
            "shard_fp32_groups",
            "fp32_from_float16_groups",
        ):
            masters.append(attrs.get(key))
        inner = attrs.get("optimizer")
        concrete = vars(inner) if hasattr(inner, "__dict__") else {}
        if not isinstance(concrete.get("state"), Mapping):
            uncovered.append(f"optimizer.{len(seen)}: no concrete state mapping")
        else:
            states.append(concrete["state"])
        masters.append(concrete.get("param_groups"))

    inventory = cpu_storage_inventory(
        {
            "ddp_backups": backups,
            "reference": reference,
            "reference_swap": reference_swap,
            "optimizer_state": states,
            "optimizer_master": masters,
        }
    )
    return {**inventory, "uncovered": uncovered}
