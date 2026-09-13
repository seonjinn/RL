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
