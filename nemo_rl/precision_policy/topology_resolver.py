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

from bisect import bisect_left
from collections.abc import ItemsView, Iterator, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol, cast

from nemo_rl.precision_policy.semantic import (
    DecoderLayerUniverse,
    ExpectedGraphDeclaration,
    GraphKind,
    ResolvedGraphTopology,
    ResolvedSelectionTopology,
    _compute_semantic_structure_digest,
    _graph_sort_key,
    _merge_selection_role_definitions,
    _validate_exact_source_neutral_topology_values,
    canonical_model_config_digest,
)


@dataclass(frozen=True, slots=True, eq=False)
class _FrozenConfigMapping(Mapping[str, object]):
    _entries: tuple[tuple[str, object], ...]

    def __getitem__(self, key: str) -> object:
        index = bisect_left(self._entries, key, key=lambda entry: entry[0])
        if index < len(self._entries):
            candidate, value = self._entries[index]
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self) -> ItemsView[str, object]:
        return _FrozenConfigItemsView(self)

    def __eq__(self, other: object) -> bool:
        if type(other) is _FrozenConfigMapping:
            return self._entries == other._entries
        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        return all(key in other and value == other[key] for key, value in self._entries)


class _FrozenConfigItemsView(ItemsView[str, object]):
    def __iter__(self) -> Iterator[tuple[str, object]]:
        mapping = cast(_FrozenConfigMapping, self._mapping)
        return iter(mapping._entries)


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


def _validate_exact_frozen_config(
    value: object,
    path: str,
    active_ids: set[int],
    completed_ids: set[int],
) -> None:
    value_type = type(value)
    if value_type in {bool, int, str, type(None)}:
        return
    if value_type is float:
        if not isfinite(cast(float, value)):
            raise ValueError(f"{path} floats must be finite")
        return
    if value_type not in {_FrozenConfigMapping, tuple}:
        raise TypeError(f"{path} contains a non-exact frozen config value")
    identity = id(value)
    if identity in active_ids:
        raise ValueError("request effective_model_config must not contain cycles")
    if identity in completed_ids:
        return
    active_ids.add(identity)
    try:
        if value_type is tuple:
            for index, item in enumerate(
                tuple.__iter__(cast(tuple[object, ...], value))
            ):
                _validate_exact_frozen_config(
                    item,
                    f"{path}[{index}]",
                    active_ids,
                    completed_ids,
                )
            return
        mapping = cast(_FrozenConfigMapping, value)
        entries = mapping._entries
        if type(entries) is not tuple:
            raise TypeError(f"{path} entries must be an exact tuple")
        keys: list[str] = []
        for index, entry in enumerate(tuple.__iter__(entries)):
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError(f"{path} entries must contain exact key/value tuples")
            key, item = entry
            if type(key) is not str:
                raise TypeError(f"{path} keys must be exact strings")
            keys.append(key)
            _validate_exact_frozen_config(
                item,
                f"{path}.{key}",
                active_ids,
                completed_ids,
            )
        if keys != sorted(set(keys)):
            raise ValueError(f"{path} keys must use unique canonical order")
    finally:
        active_ids.remove(identity)
        completed_ids.add(identity)


@dataclass(frozen=True, slots=True)
class GraphTopologyResolutionRequest:
    """Source-neutral, recursively frozen input for one graph resolution."""

    declaration: ExpectedGraphDeclaration
    effective_model_config: Mapping[str, object]
    resolved_model_revision: str
    decoder_layer_universe: DecoderLayerUniverse
    effective_model_config_digest: str = field(init=False)

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
        frozen_model_config = _freeze_plain_config(self.effective_model_config)
        object.__setattr__(self, "effective_model_config", frozen_model_config)
        object.__setattr__(
            self,
            "effective_model_config_digest",
            canonical_model_config_digest(frozen_model_config),
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
    if type(requests) is not tuple:
        raise TypeError("topology resolution requests must be an exact tuple")
    if not requests:
        raise ValueError("topology resolution requires a complete non-empty graph set")
    for request in requests:
        if type(request) is not GraphTopologyResolutionRequest:
            raise TypeError(
                "topology resolution requests must be exact "
                "GraphTopologyResolutionRequest records"
            )
        if type(request.effective_model_config) is not _FrozenConfigMapping:
            raise TypeError(
                "request effective_model_config must be the frozen constructor snapshot"
            )
        _validate_exact_frozen_config(
            request.effective_model_config,
            "request effective_model_config",
            set(),
            set(),
        )
        _validate_exact_source_neutral_topology_values(
            [request.declaration, request.decoder_layer_universe],
            replay_invariants=True,
        )
        if type(request.resolved_model_revision) is not str:
            raise TypeError("request resolved_model_revision must be an exact string")
        if type(request.effective_model_config_digest) is not str:
            raise TypeError(
                "request effective_model_config_digest must be an exact string"
            )
        expected_config_digest = canonical_model_config_digest(
            request.effective_model_config
        )
        if request.effective_model_config_digest != expected_config_digest:
            raise ValueError("request effective model config digest mismatch")
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
    if graph.effective_model_config_digest != request.effective_model_config_digest:
        raise ValueError(
            "resolved graph effective model config digest differs from its request"
        )
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
