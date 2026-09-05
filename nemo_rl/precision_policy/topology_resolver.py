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

"""Pure Phase 1 model-topology resolution for semantic precision policy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from nemo_rl.precision_policy.semantic import (
    DecoderLayerUniverse,
    ExpectedGraphDeclaration,
    GraphKind,
    ResolvedGraphTopology,
    ResolvedSelectionTopology,
    _compute_semantic_structure_digest,
    _graph_sort_key,
    _merge_selection_role_definitions,
)


@dataclass(frozen=True, slots=True, eq=False)
class _FrozenConfigMapping(Mapping[str, object]):
    _entries: tuple[tuple[str, object], ...]

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        return all(key in other and value == other[key] for key, value in self._entries)


def _freeze_plain_value(value: object, active_ids: set[int]) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("plain configuration floats must be finite")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("plain configuration must not contain cycles")
        active_ids.add(identity)
        try:
            if any(not isinstance(key, str) for key in value):
                raise TypeError("plain configuration mapping keys must be strings")
            frozen = {
                key: _freeze_plain_value(value[key], active_ids)
                for key in sorted(value)
            }
        finally:
            active_ids.remove(identity)
        return _FrozenConfigMapping(tuple(frozen.items()))
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("plain configuration must not contain cycles")
        active_ids.add(identity)
        try:
            return tuple(_freeze_plain_value(item, active_ids) for item in value)
        finally:
            active_ids.remove(identity)
    raise TypeError(
        "plain configuration values must be mappings, lists, tuples, or JSON scalars"
    )


def _freeze_plain_config(config: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_plain_value(config, set())
    if not isinstance(frozen, Mapping):
        raise TypeError("effective_model_config must be a mapping")
    return frozen


@dataclass(frozen=True, slots=True)
class GraphTopologyResolutionRequest:
    """Source-neutral, recursively frozen input for one graph resolution."""

    declaration: ExpectedGraphDeclaration
    effective_model_config: Mapping[str, object]
    resolved_model_revision: str
    decoder_layer_universe: DecoderLayerUniverse

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, ExpectedGraphDeclaration):
            raise TypeError("declaration must be ExpectedGraphDeclaration")
        if not isinstance(self.effective_model_config, Mapping):
            raise TypeError("effective_model_config must be a plain mapping")
        if (
            not isinstance(self.resolved_model_revision, str)
            or not self.resolved_model_revision
            or self.resolved_model_revision != self.resolved_model_revision.strip()
            or any(character.isspace() for character in self.resolved_model_revision)
        ):
            raise ValueError(
                "resolved_model_revision must be non-empty without whitespace"
            )
        if not isinstance(self.decoder_layer_universe, DecoderLayerUniverse):
            raise TypeError("decoder_layer_universe must be DecoderLayerUniverse")
        object.__setattr__(
            self,
            "effective_model_config",
            _freeze_plain_config(self.effective_model_config),
        )
        evidence = self.declaration.lifecycle.immutable_evidence
        if (
            evidence is not None
            and evidence.pinned_checkpoint_revision != self.resolved_model_revision
        ):
            raise ValueError(
                "resolved_model_revision must equal pinned checkpoint revision"
            )


class SelectionTopologyAdapter(Protocol):
    """Pure family-specific adapter for source-neutral topology discovery."""

    adapter_id: str

    def supports(self, model_config: Mapping[str, object]) -> bool:
        """Return whether this adapter owns the effective model configuration."""
        ...

    def resolve_graph(
        self,
        request: GraphTopologyResolutionRequest,
    ) -> ResolvedGraphTopology:
        """Derive one complete graph topology without runtime source state."""
        ...


def _validate_request_set(
    requests: tuple[GraphTopologyResolutionRequest, ...],
) -> tuple[GraphTopologyResolutionRequest, ...]:
    if not isinstance(requests, tuple):
        raise TypeError("topology resolution requests must be a tuple")
    if not requests:
        raise ValueError("topology resolution requires a complete non-empty graph set")
    if any(
        not isinstance(request, GraphTopologyResolutionRequest) for request in requests
    ):
        raise TypeError(
            "topology resolution requests must be GraphTopologyResolutionRequest"
        )
    graph_ids = tuple(request.declaration.graph_instance_id for request in requests)
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("topology resolution contains a duplicate graph declaration")
    main_ids = tuple(
        request.declaration.graph_instance_id
        for request in requests
        if request.declaration.lifecycle.graph_kind == GraphKind.MAIN
    )
    if main_ids != ("main",):
        raise ValueError(
            "topology resolution requires exactly one MAIN instance named main"
        )
    return tuple(
        sorted(
            requests,
            key=lambda request: _graph_sort_key(request.declaration.graph_instance_id),
        )
    )


def _select_adapter(
    request: GraphTopologyResolutionRequest,
    adapters: tuple[SelectionTopologyAdapter, ...],
) -> SelectionTopologyAdapter:
    matches: list[SelectionTopologyAdapter] = []
    for adapter in adapters:
        supported = adapter.supports(request.effective_model_config)
        if not isinstance(supported, bool):
            raise TypeError("selection topology adapter supports() must return bool")
        if supported:
            matches.append(adapter)
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one selection topology adapter for "
            f"{request.declaration.graph_instance_id}, got {len(matches)}"
        )
    return matches[0]


def _resolve_graph(
    request: GraphTopologyResolutionRequest,
    adapters: tuple[SelectionTopologyAdapter, ...],
) -> ResolvedGraphTopology:
    adapter = _select_adapter(request, adapters)
    graph = adapter.resolve_graph(request)
    if not isinstance(graph, ResolvedGraphTopology):
        raise TypeError("selection topology adapter must return ResolvedGraphTopology")
    if graph.declaration != request.declaration:
        raise ValueError("resolved graph declaration differs from its request")
    if graph.resolved_model_revision != request.resolved_model_revision:
        raise ValueError("resolved graph model revision differs from its request")
    if graph.adapter_id != adapter.adapter_id:
        raise ValueError("resolved graph adapter_id differs from selected adapter")
    if graph.decoder_layer_universe != request.decoder_layer_universe:
        raise ValueError(
            "adapter-derived decoder layer universe mismatch with declared universe"
        )
    return graph


def _validate_adapter_registry(
    adapters: tuple[SelectionTopologyAdapter, ...],
) -> tuple[SelectionTopologyAdapter, ...]:
    if not isinstance(adapters, tuple):
        raise TypeError("selection topology adapters must be a tuple")
    adapter_ids: list[str] = []
    for adapter in adapters:
        adapter_id = adapter.adapter_id
        if (
            not isinstance(adapter_id, str)
            or not adapter_id
            or adapter_id != adapter_id.strip()
            or any(character.isspace() for character in adapter_id)
        ):
            raise ValueError("selection topology adapter_id must be canonical text")
        adapter_ids.append(adapter_id)
    if len(adapter_ids) != len(set(adapter_ids)):
        raise ValueError("duplicate selection topology adapter_id")
    return adapters


def resolve_selection_topology(
    requests: tuple[GraphTopologyResolutionRequest, ...],
    schema_version: int,
    *,
    adapters: tuple[SelectionTopologyAdapter, ...] = (),
) -> ResolvedSelectionTopology:
    """Resolve a complete graph set into one deterministic Phase 1 topology."""
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("semantic schema_version must be an integer")
    ordered_requests = _validate_request_set(requests)
    adapter_registry = _validate_adapter_registry(adapters)
    graphs = tuple(
        _resolve_graph(request, adapter_registry) for request in ordered_requests
    )
    role_definitions = _merge_selection_role_definitions(graphs, schema_version)
    digest = _compute_semantic_structure_digest(
        schema_version=schema_version,
        graphs=graphs,
        role_definitions=role_definitions,
    )
    return ResolvedSelectionTopology(
        schema_version=schema_version,
        graphs=graphs,
        role_definitions=role_definitions,
        semantic_structure_digest=digest,
    )
