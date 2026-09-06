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

"""Deterministic paired model-topology adapter registry."""

from __future__ import annotations

from dataclasses import dataclass

from nemo_rl.precision_policy.topology import ModelTopologyAdapter
from nemo_rl.precision_policy.topology_resolver import SelectionTopologyAdapter


def _validate_adapter_id(adapter_id: object, label: str) -> str:
    if (
        type(adapter_id) is not str
        or not adapter_id
        or adapter_id != adapter_id.strip()
        or any(character.isspace() for character in adapter_id)
    ):
        raise ValueError(f"{label} must be canonical non-empty text")
    return adapter_id


@dataclass(frozen=True, slots=True)
class PrecisionTopologyAdapterBundle:
    """One family adapter paired across source-neutral and runtime phases."""

    adapter_id: str
    selection: SelectionTopologyAdapter
    runtime: ModelTopologyAdapter

    def __post_init__(self) -> None:
        bundle_id = _validate_adapter_id(self.adapter_id, "adapter bundle ID")
        try:
            selection_id = self.selection.adapter_id
            selection_supports = self.selection.supports
            resolve_graph = self.selection.resolve_graph
        except AttributeError as error:
            raise TypeError(
                "selection does not implement SelectionTopologyAdapter"
            ) from error
        try:
            runtime_id = self.runtime.adapter_id
            runtime_supports = self.runtime.supports
            classify_graph = self.runtime.classify_graph
        except AttributeError as error:
            raise TypeError(
                "runtime does not implement ModelTopologyAdapter"
            ) from error
        selection_id = _validate_adapter_id(selection_id, "selection adapter ID")
        runtime_id = _validate_adapter_id(runtime_id, "runtime adapter ID")
        if not callable(selection_supports) or not callable(resolve_graph):
            raise TypeError("selection does not implement SelectionTopologyAdapter")
        if not callable(runtime_supports) or not callable(classify_graph):
            raise TypeError("runtime does not implement ModelTopologyAdapter")
        if selection_id != bundle_id:
            raise ValueError("selection adapter ID differs from bundle ID")
        if runtime_id != bundle_id:
            raise ValueError("runtime adapter ID differs from bundle ID")


def validate_precision_adapter_bundles(
    bundles: tuple[PrecisionTopologyAdapterBundle, ...],
) -> tuple[PrecisionTopologyAdapterBundle, ...]:
    """Validate and canonically order one duplicate-free paired registry."""
    if type(bundles) is not tuple:
        raise TypeError("precision topology adapter bundles must be an exact tuple")
    if any(type(bundle) is not PrecisionTopologyAdapterBundle for bundle in bundles):
        raise TypeError(
            "precision topology adapter registry requires exact bundle records"
        )
    for bundle in bundles:
        bundle.__post_init__()
    canonical = tuple(sorted(bundles, key=lambda bundle: bundle.adapter_id))
    adapter_ids = tuple(bundle.adapter_id for bundle in canonical)
    if len(adapter_ids) != len(set(adapter_ids)):
        raise ValueError("duplicate precision topology adapter ID")
    return canonical


BUILTIN_PRECISION_TOPOLOGY_ADAPTER_BUNDLES: tuple[
    PrecisionTopologyAdapterBundle, ...
] = validate_precision_adapter_bundles(())


# Legacy runtime-only discovery remains available until controller bootstrap owns
# the paired registry in every call path.
BUILTIN_TOPOLOGY_ADAPTERS: tuple[ModelTopologyAdapter, ...] = tuple(
    bundle.runtime for bundle in BUILTIN_PRECISION_TOPOLOGY_ADAPTER_BUNDLES
)
