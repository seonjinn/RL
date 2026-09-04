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

"""Pure model-topology discovery and semantic-classification contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import gcd, isfinite, prod
from typing import Literal, Protocol

from nemo_rl.precision_policy.semantic import (
    AxisDomain,
    AxisProjection,
    ComponentRole,
    EvidenceSource,
    ExpectedGraphDeclaration,
    FamilyIndexDomain,
    LayerDomain,
    LayerMember,
    OwnerFamilyReference,
    ParameterInventory,
    ParameterInventoryEntry,
    RoleDefinition,
    RoleExpectedDomain,
    RolloutParticipation,
    SemanticGraphManifest,
    SemanticManifestBundle,
    SemanticPredicate,
    SourceMutability,
    SourceOwnerInventoryEntry,
    ValueProvenance,
    builtin_role_definitions,
    resolve_component_axes,
)


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class _FrozenConfigMapping(Mapping[str, object]):
    entries: tuple[tuple[str, object], ...]

    def __getitem__(self, key: str) -> object:
        for item_key, value in self.entries:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        return all(key in other and value == other[key] for key, value in self.entries)


def _freeze_config_value(value: object, path: str) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_config_value(item, f"{path}.{key}")
        return _FrozenConfigMapping(tuple(sorted(frozen.items())))
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_config_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{path} floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{path} must contain only plain configuration values")


def _freeze_model_config(config: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_config_value(config, "model_config")
    if not isinstance(frozen, Mapping):  # pragma: no cover - fixed by input type
        raise AssertionError("model_config snapshot must be a mapping")
    return frozen


@dataclass(frozen=True, slots=True)
class GraphTopologyInput:
    """One declared graph paired with its own immutable topology inputs."""

    declaration: ExpectedGraphDeclaration
    model_config: Mapping[str, object]
    resolved_model_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, ExpectedGraphDeclaration):
            raise TypeError("declaration must be ExpectedGraphDeclaration")
        if not isinstance(self.model_config, Mapping):
            raise TypeError("model_config must be a mapping")
        if not isinstance(self.resolved_model_revision, str):
            raise TypeError("resolved_model_revision must be a string")
        if not self.resolved_model_revision.strip():
            raise ValueError("resolved_model_revision must be non-empty")
        object.__setattr__(
            self,
            "model_config",
            _freeze_model_config(self.model_config),
        )


def resolve_text_config(
    model_config: Mapping[str, object],
) -> Mapping[str, object]:
    """Resolve a wrapper's nested text config without loading backend objects."""
    nested = model_config.get("text_config")
    if nested is None:
        return model_config
    if not isinstance(nested, Mapping):
        raise TypeError("text_config must be a mapping")
    if any(not isinstance(key, str) for key in nested):
        raise TypeError("text_config keys must be strings")
    return nested


class SourceRecordProvenance(StrEnum):
    """Raw source authority recorded before semantic classification."""

    TRAINING_RUNTIME = "training_runtime"
    CHECKPOINT_STORAGE = "checkpoint_storage"
    BACKEND_DERIVED = "backend_derived"
    TIED_STORAGE = "tied_storage"


@dataclass(frozen=True, slots=True)
class SourceDiscoveryRecord:
    """Frozen native tensor metadata with no semantic or runtime binding."""

    record_id: str
    graph_instance_id: str
    source_native_name: str | None
    source_native_owner_id: str | None
    dtype: str
    shape: tuple[int, ...]
    provenance: SourceRecordProvenance
    provenance_evidence: EvidenceSource
    source_mutability: SourceMutability
    mutability_evidence: EvidenceSource

    def __post_init__(self) -> None:
        _require_text(self.record_id, "source discovery record_id")
        _require_text(self.graph_instance_id, "source discovery graph_instance_id")
        object.__setattr__(self, "shape", tuple(self.shape))
        _require_text(self.dtype, "source discovery dtype")
        if not isinstance(self.provenance, SourceRecordProvenance):
            raise TypeError("source provenance must be SourceRecordProvenance")
        if not isinstance(self.provenance_evidence, EvidenceSource):
            raise TypeError("source provenance evidence must be EvidenceSource")
        if not isinstance(self.source_mutability, SourceMutability):
            raise TypeError("source mutability must be SourceMutability")
        if not isinstance(self.mutability_evidence, EvidenceSource):
            raise TypeError("source mutability evidence must be EvidenceSource")
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in self.shape
        ):
            raise ValueError("source shape dimensions must be positive integers")
        is_absent = self.source_mutability == SourceMutability.ABSENT
        native_fields_absent = (
            self.source_native_name is None and self.source_native_owner_id is None
        )
        if is_absent and not native_fields_absent:
            raise ValueError("absent source record forbids native name and owner")
        if is_absent and self.provenance == SourceRecordProvenance.TIED_STORAGE:
            raise ValueError("absent source record cannot have tied-storage provenance")
        if not is_absent and native_fields_absent:
            raise ValueError("present source record requires native name and owner")
        if not is_absent and (
            self.source_native_name is None or self.source_native_owner_id is None
        ):
            raise ValueError("present source record requires both native fields")
        if is_absent:
            return
        _require_text(self.source_native_name, "source native name")
        _require_text(self.source_native_owner_id, "source native owner")


@dataclass(frozen=True, slots=True)
class SourceDiscoveryInventory:
    """Canonical raw discovery records for all explicitly declared graphs."""

    records: tuple[SourceDiscoveryRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, SourceDiscoveryRecord) for record in records):
            raise TypeError("source discovery inventory requires discovery records")
        record_ids = tuple(record.record_id for record in records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("duplicate source discovery record ID")
        present_native_names = tuple(
            (record.graph_instance_id, record.source_native_name)
            for record in records
            if record.source_mutability != SourceMutability.ABSENT
        )
        if len(present_native_names) != len(set(present_native_names)):
            raise ValueError("duplicate present source native name")
        object.__setattr__(
            self,
            "records",
            tuple(
                sorted(
                    records,
                    key=lambda record: (record.graph_instance_id, record.record_id),
                )
            ),
        )


@dataclass(frozen=True, slots=True, order=True)
class SourceIndexSpan:
    """Compact half-open arithmetic progression over one raw source axis."""

    start: int
    stop: int
    step: int = 1

    def __post_init__(self) -> None:
        _require_int(self.start, "source span start")
        _require_int(self.stop, "source span stop", minimum=1)
        _require_int(self.step, "source span step", minimum=1)
        if self.stop <= self.start:
            raise ValueError("source span stop must be greater than start")

    @property
    def cardinality(self) -> int:
        """Return selected ordinal count without materializing it."""
        return (self.stop - self.start + self.step - 1) // self.step


def _spans_intersect(left: SourceIndexSpan, right: SourceIndexSpan) -> bool:
    common_divisor = gcd(left.step, right.step)
    difference = right.start - left.start
    if difference % common_divisor:
        return False
    right_modulus = right.step // common_divisor
    if right_modulus == 1:
        multiplier = 0
    else:
        multiplier = (
            (difference // common_divisor)
            * pow(left.step // common_divisor, -1, right_modulus)
        ) % right_modulus
    first = left.start + left.step * multiplier
    period = left.step * right_modulus
    lower = max(left.start, right.start)
    if first < lower:
        first += ((lower - first + period - 1) // period) * period
    return first < min(left.stop, right.stop)


@dataclass(frozen=True, slots=True)
class SourceAxisSelection:
    """Disjoint compact spans selected from one raw source axis."""

    axis_index: int
    spans: tuple[SourceIndexSpan, ...]

    def __post_init__(self) -> None:
        _require_int(self.axis_index, "source selection axis_index")
        spans = tuple(self.spans)
        if not spans:
            raise ValueError("source axis selection requires non-empty spans")
        if any(not isinstance(span, SourceIndexSpan) for span in spans):
            raise TypeError("source axis selection requires SourceIndexSpan records")
        spans = tuple(sorted(spans))
        for index, left in enumerate(spans):
            if any(_spans_intersect(left, right) for right in spans[index + 1 :]):
                raise ValueError("source axis selection spans must be disjoint")
        object.__setattr__(self, "spans", spans)

    @property
    def cardinality(self) -> int:
        """Return the exact selected axis cardinality."""
        return sum(span.cardinality for span in self.spans)


@dataclass(frozen=True, slots=True)
class SourceRegion:
    """Compact Cartesian region over every raw source axis exactly once."""

    source_shape: tuple[int, ...]
    axis_selections: tuple[SourceAxisSelection, ...]

    def __post_init__(self) -> None:
        source_shape = tuple(self.source_shape)
        selections = tuple(self.axis_selections)
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in source_shape
        ):
            raise ValueError("source region shape dimensions must be positive integers")
        if any(not isinstance(item, SourceAxisSelection) for item in selections):
            raise TypeError("source region selections must be SourceAxisSelection")
        selections = tuple(sorted(selections, key=lambda item: item.axis_index))
        if tuple(item.axis_index for item in selections) != tuple(
            range(len(source_shape))
        ):
            raise ValueError("source region must select every source axis exactly once")
        for selection in selections:
            limit = source_shape[selection.axis_index]
            if any(span.stop > limit for span in selection.spans):
                raise ValueError("source region span exceeds its source axis")
        object.__setattr__(self, "source_shape", source_shape)
        object.__setattr__(self, "axis_selections", selections)

    @property
    def cardinality(self) -> int:
        """Return exact region cardinality without rendering source indices."""
        cardinality = 1
        for selection in self.axis_selections:
            cardinality *= selection.cardinality
        return cardinality


@dataclass(frozen=True, slots=True)
class SourceOrdinalMapSegment:
    """Compact affine mapping from raw source ordinals to target ordinals."""

    source_span: SourceIndexSpan
    target_ordinal_start: int
    target_ordinal_step: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.source_span, SourceIndexSpan):
            raise TypeError("ordinal map source_span must be SourceIndexSpan")
        _require_int(self.target_ordinal_start, "target ordinal start")
        if isinstance(self.target_ordinal_step, bool) or not isinstance(
            self.target_ordinal_step, int
        ):
            raise TypeError("target ordinal step must be an integer")
        if self.target_ordinal_step <= 0:
            raise ValueError("target ordinal step must be positive")


@dataclass(frozen=True, slots=True)
class FamilyIndexAxisTarget:
    """A mapped independent semantic-family axis."""

    axis_name: str

    def __post_init__(self) -> None:
        _require_text(self.axis_name, "family index target axis")


@dataclass(frozen=True, slots=True)
class LayerCoordinateTarget:
    """A mapped coordinate in the correlated semantic layer relation."""

    coordinate: Literal["global_decoder_layer", "moe_ordinal"]

    def __post_init__(self) -> None:
        if self.coordinate not in {"global_decoder_layer", "moe_ordinal"}:
            raise ValueError("unknown layer coordinate target")


@dataclass(frozen=True, slots=True)
class ComponentAxisTarget:
    """A mapped resolved axis for one ordered encoding component."""

    component_role: ComponentRole
    component_axis: str

    def __post_init__(self) -> None:
        _require_text(self.component_role, "logical component role")
        _require_text(self.component_axis, "component axis")


type SemanticAxisTarget = (
    FamilyIndexAxisTarget | LayerCoordinateTarget | ComponentAxisTarget
)


def _semantic_axis_target_key(target: SemanticAxisTarget) -> tuple[object, ...]:
    if isinstance(target, FamilyIndexAxisTarget):
        return (0, target.axis_name)
    if isinstance(target, LayerCoordinateTarget):
        return (1, target.coordinate)
    return (2, str(target.component_role), target.component_axis)


@dataclass(frozen=True, slots=True)
class FixedFamilyAxisCoordinate:
    """One fixed independent coordinate omitted from an output subdomain."""

    axis_name: str
    member: int | str

    def __post_init__(self) -> None:
        _require_text(self.axis_name, "fixed family axis")
        if isinstance(self.member, bool) or not isinstance(self.member, (int, str)):
            raise TypeError("fixed family-axis member must be int or str")
        if isinstance(self.member, str):
            _require_text(self.member, "fixed family-axis member")


@dataclass(frozen=True, slots=True)
class FixedLayerCoordinate:
    """One fixed correlated layer member omitted from an output subdomain."""

    member: LayerMember

    def __post_init__(self) -> None:
        if not isinstance(self.member, LayerMember):
            raise TypeError("fixed layer coordinate must be LayerMember")


type FixedMemberCoordinate = FixedFamilyAxisCoordinate | FixedLayerCoordinate


@dataclass(frozen=True, slots=True)
class OutputMemberTarget:
    """One exact compact semantic output-member subdomain."""

    inventory_entry_id: str
    member_domain: FamilyIndexDomain
    fixed_coordinates: tuple[FixedMemberCoordinate, ...]

    def __post_init__(self) -> None:
        _require_text(self.inventory_entry_id, "output inventory entry_id")
        if not isinstance(self.member_domain, FamilyIndexDomain):
            raise TypeError("output member_domain must be FamilyIndexDomain")
        coordinates = tuple(self.fixed_coordinates)
        if any(
            not isinstance(item, (FixedFamilyAxisCoordinate, FixedLayerCoordinate))
            for item in coordinates
        ):
            raise TypeError("output fixed coordinates must be typed records")
        fixed_axis_names = tuple(
            item.axis_name
            for item in coordinates
            if isinstance(item, FixedFamilyAxisCoordinate)
        )
        if len(fixed_axis_names) != len(set(fixed_axis_names)):
            raise ValueError("output target contains duplicate fixed family axes")
        if sum(isinstance(item, FixedLayerCoordinate) for item in coordinates) > 1:
            raise ValueError("output target contains duplicate fixed layer coordinates")
        object.__setattr__(
            self,
            "fixed_coordinates",
            tuple(
                sorted(
                    coordinates,
                    key=lambda item: (
                        0 if isinstance(item, FixedLayerCoordinate) else 1,
                        ""
                        if isinstance(item, FixedLayerCoordinate)
                        else item.axis_name,
                    ),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceToSemanticAxisMapping:
    """Map one selected raw axis to one typed semantic/output coordinate."""

    source_axis_index: int
    target: SemanticAxisTarget
    segments: tuple[SourceOrdinalMapSegment, ...]

    def __post_init__(self) -> None:
        _require_int(self.source_axis_index, "axis mapping source_axis_index")
        if not isinstance(
            self.target,
            (FamilyIndexAxisTarget, LayerCoordinateTarget, ComponentAxisTarget),
        ):
            raise TypeError("axis mapping target must be a typed semantic target")
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("axis mapping requires non-empty segments")
        if any(not isinstance(item, SourceOrdinalMapSegment) for item in segments):
            raise TypeError("axis mapping segments must be ordinal map segments")
        for index, left in enumerate(segments):
            if any(
                _spans_intersect(left.source_span, right.source_span)
                for right in segments[index + 1 :]
            ):
                raise ValueError("axis mapping source spans must be disjoint")
        object.__setattr__(
            self,
            "segments",
            tuple(sorted(segments, key=lambda item: item.source_span)),
        )


@dataclass(frozen=True, slots=True)
class CanonicalValueClassificationEdge:
    """A consuming raw-region edge to one canonical semantic component."""

    record_id: str
    source_region: SourceRegion
    output: OutputMemberTarget
    canonical_owner_family: OwnerFamilyReference
    component_role: ComponentRole
    axis_mappings: tuple[SourceToSemanticAxisMapping, ...]

    def __post_init__(self) -> None:
        _require_text(self.record_id, "classification edge record_id")
        if not isinstance(self.source_region, SourceRegion):
            raise TypeError("canonical edge source_region must be SourceRegion")
        if not isinstance(self.output, OutputMemberTarget):
            raise TypeError("canonical edge output must be OutputMemberTarget")
        if not isinstance(self.canonical_owner_family, OwnerFamilyReference):
            raise TypeError("canonical edge owner must be OwnerFamilyReference")
        _require_text(self.component_role, "canonical edge component role")
        mappings = tuple(self.axis_mappings)
        if any(not isinstance(item, SourceToSemanticAxisMapping) for item in mappings):
            raise TypeError("canonical edge mappings must be typed axis mappings")
        object.__setattr__(
            self,
            "axis_mappings",
            tuple(
                sorted(
                    mappings,
                    key=lambda item: (
                        _semantic_axis_target_key(item.target),
                        item.source_axis_index,
                        item.segments,
                    ),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class TiedAliasClassificationEdge:
    """A non-consuming tied-storage view resolved directly to canonical storage."""

    record_id: str
    aliased_source_region: SourceRegion
    alias_output: OutputMemberTarget
    canonical_owner_family: OwnerFamilyReference
    canonical_value_entry_id: str
    component_role: ComponentRole
    alias_to_canonical_axes: tuple[AxisProjection, ...]

    def __post_init__(self) -> None:
        _require_text(self.record_id, "tied edge record_id")
        if not isinstance(self.aliased_source_region, SourceRegion):
            raise TypeError("tied edge source_region must be SourceRegion")
        if not isinstance(self.alias_output, OutputMemberTarget):
            raise TypeError("tied edge output must be OutputMemberTarget")
        if not isinstance(self.canonical_owner_family, OwnerFamilyReference):
            raise TypeError("tied edge owner must be OwnerFamilyReference")
        _require_text(self.canonical_value_entry_id, "tied edge canonical value")
        _require_text(self.component_role, "tied edge component role")
        projections = tuple(self.alias_to_canonical_axes)
        if any(not isinstance(item, AxisProjection) for item in projections):
            raise TypeError("tied edge axes must be AxisProjection records")
        if len(projections) != len(set(projections)):
            raise ValueError("tied edge projection contains duplicates")
        object.__setattr__(
            self,
            "alias_to_canonical_axes",
            tuple(
                sorted(
                    projections,
                    key=lambda item: (item.member_axis, item.owner_axis),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class AbsentDiscoveryDispositionEdge:
    """The sole zero-output disposition for a raw record known to be absent."""

    record_id: str

    def __post_init__(self) -> None:
        _require_text(self.record_id, "absent disposition record_id")


type DiscoveryClassificationEdge = (
    CanonicalValueClassificationEdge
    | TiedAliasClassificationEdge
    | AbsentDiscoveryDispositionEdge
)


@dataclass(frozen=True, slots=True)
class RoleDefinitionContribution:
    """One adapter's independently derived schema-bound role domain."""

    schema_version: int
    role_name: str
    predicate: SemanticPredicate
    expected_inventory_entry_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_int(self.schema_version, "role contribution schema_version", minimum=1)
        _require_text(self.role_name, "role contribution name")
        if not isinstance(self.predicate, SemanticPredicate):
            raise TypeError("role contribution predicate must be SemanticPredicate")
        entry_ids = tuple(self.expected_inventory_entry_ids)
        if not entry_ids:
            raise ValueError("role contribution expected domain must be non-empty")
        for entry_id in entry_ids:
            _require_text(entry_id, "role contribution inventory entry")
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("role contribution expected domain contains duplicates")
        object.__setattr__(
            self, "expected_inventory_entry_ids", tuple(sorted(entry_ids))
        )


def _axis_member_sort_key(member: int | str) -> tuple[int, int | str]:
    return (0, member) if isinstance(member, int) else (1, member)


def _domain_sort_key(domain: FamilyIndexDomain) -> tuple[object, ...]:
    layer_key: tuple[object, ...]
    if domain.layer_domain is None:
        layer_key = (0,)
    else:
        layer_key = (
            1,
            tuple(
                (
                    member.global_decoder_layer,
                    -1 if member.moe_ordinal is None else member.moe_ordinal,
                )
                for member in domain.layer_domain.members
            ),
        )
    return (
        layer_key,
        tuple(
            (
                axis.name,
                tuple(_axis_member_sort_key(member) for member in axis.members),
            )
            for axis in domain.independent_axes
        ),
    )


def _region_sort_key(region: SourceRegion) -> tuple[object, ...]:
    return (
        region.source_shape,
        tuple(
            (
                selection.axis_index,
                tuple((span.start, span.stop, span.step) for span in selection.spans),
            )
            for selection in region.axis_selections
        ),
    )


def _output_target_sort_key(target: OutputMemberTarget) -> tuple[object, ...]:
    return (
        target.inventory_entry_id,
        _domain_sort_key(target.member_domain),
        tuple(
            (
                0,
                coordinate.member.global_decoder_layer,
                -1
                if coordinate.member.moe_ordinal is None
                else coordinate.member.moe_ordinal,
            )
            if isinstance(coordinate, FixedLayerCoordinate)
            else (
                1,
                coordinate.axis_name,
                _axis_member_sort_key(coordinate.member),
            )
            for coordinate in target.fixed_coordinates
        ),
    )


def _axis_mapping_sort_key(
    mapping: SourceToSemanticAxisMapping,
) -> tuple[object, ...]:
    return (
        mapping.source_axis_index,
        _semantic_axis_target_key(mapping.target),
        tuple(
            (
                segment.source_span.start,
                segment.source_span.stop,
                segment.source_span.step,
                segment.target_ordinal_start,
                segment.target_ordinal_step,
            )
            for segment in mapping.segments
        ),
    )


def _edge_sort_key(edge: DiscoveryClassificationEdge) -> tuple[object, ...]:
    if isinstance(edge, CanonicalValueClassificationEdge):
        return (
            edge.record_id,
            0,
            edge.output.inventory_entry_id,
            str(edge.component_role),
            _region_sort_key(edge.source_region),
            _output_target_sort_key(edge.output),
            edge.canonical_owner_family.graph_instance_id,
            edge.canonical_owner_family.owner_family_id,
            tuple(_axis_mapping_sort_key(mapping) for mapping in edge.axis_mappings),
        )
    if isinstance(edge, TiedAliasClassificationEdge):
        return (
            edge.record_id,
            1,
            edge.alias_output.inventory_entry_id,
            str(edge.component_role),
            _region_sort_key(edge.aliased_source_region),
            _output_target_sort_key(edge.alias_output),
            edge.canonical_owner_family.graph_instance_id,
            edge.canonical_owner_family.owner_family_id,
            edge.canonical_value_entry_id,
            tuple(
                (projection.member_axis, projection.owner_axis)
                for projection in edge.alias_to_canonical_axes
            ),
        )
    return (edge.record_id, 2)


@dataclass(frozen=True, slots=True)
class SemanticGraphBuildFragment:
    """One graph's complete classification result before atomic bundle exposure."""

    graph_instance_id: str
    classification_edges: tuple[DiscoveryClassificationEdge, ...]
    source_owners: tuple[SourceOwnerInventoryEntry, ...]
    inventory_entries: tuple[ParameterInventoryEntry, ...]
    manifest: SemanticGraphManifest
    role_contributions: tuple[RoleDefinitionContribution, ...]

    def __post_init__(self) -> None:
        _require_text(self.graph_instance_id, "fragment graph_instance_id")
        edges = tuple(self.classification_edges)
        if any(
            not isinstance(
                edge,
                (
                    CanonicalValueClassificationEdge,
                    TiedAliasClassificationEdge,
                    AbsentDiscoveryDispositionEdge,
                ),
            )
            for edge in edges
        ):
            raise TypeError("fragment classification edges must be typed records")
        owners = tuple(self.source_owners)
        if any(not isinstance(owner, SourceOwnerInventoryEntry) for owner in owners):
            raise TypeError("fragment source owners must be inventory owner records")
        entries = tuple(self.inventory_entries)
        if any(not isinstance(entry, ParameterInventoryEntry) for entry in entries):
            raise TypeError("fragment inventory entries must be typed records")
        if not isinstance(self.manifest, SemanticGraphManifest):
            raise TypeError("fragment manifest must be SemanticGraphManifest")
        contributions = tuple(self.role_contributions)
        if any(
            not isinstance(item, RoleDefinitionContribution) for item in contributions
        ):
            raise TypeError("fragment roles must be RoleDefinitionContribution records")
        object.__setattr__(
            self, "classification_edges", tuple(sorted(edges, key=_edge_sort_key))
        )
        object.__setattr__(
            self,
            "source_owners",
            tuple(
                sorted(
                    owners,
                    key=lambda item: (
                        item.owner_family.graph_instance_id,
                        item.owner_family.owner_family_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "inventory_entries",
            tuple(sorted(entries, key=lambda item: item.entry_id)),
        )
        object.__setattr__(
            self,
            "role_contributions",
            tuple(
                sorted(
                    contributions,
                    key=lambda item: (
                        item.schema_version,
                        item.role_name,
                        item.expected_inventory_entry_ids,
                    ),
                )
            ),
        )


def _axis_selection_intersection_cardinality(
    left: SourceAxisSelection,
    right: SourceAxisSelection,
) -> int:
    return sum(
        _span_intersection_cardinality(left_span, right_span)
        for left_span in left.spans
        for right_span in right.spans
    )


def _span_intersection_cardinality(
    left: SourceIndexSpan,
    right: SourceIndexSpan,
) -> int:
    common_divisor = gcd(left.step, right.step)
    difference = right.start - left.start
    if difference % common_divisor:
        return 0
    right_modulus = right.step // common_divisor
    if right_modulus == 1:
        multiplier = 0
    else:
        multiplier = (
            (difference // common_divisor)
            * pow(left.step // common_divisor, -1, right_modulus)
        ) % right_modulus
    first = left.start + left.step * multiplier
    period = left.step * right_modulus
    lower = max(left.start, right.start)
    if first < lower:
        first += ((lower - first + period - 1) // period) * period
    upper = min(left.stop, right.stop)
    if first >= upper:
        return 0
    return (upper - 1 - first) // period + 1


def _regions_intersect(left: SourceRegion, right: SourceRegion) -> bool:
    if left.source_shape != right.source_shape:
        return False
    return all(
        _axis_selection_intersection_cardinality(left_axis, right_axis) > 0
        for left_axis, right_axis in zip(
            left.axis_selections,
            right.axis_selections,
            strict=True,
        )
    )


def _validate_region_partition(
    record: SourceDiscoveryRecord,
    regions: tuple[SourceRegion, ...],
    *,
    tied: bool,
) -> None:
    for region in regions:
        if region.source_shape != record.shape:
            raise ValueError("classification source region shape mismatch")
    for index, left in enumerate(regions):
        if any(_regions_intersect(left, right) for right in regions[index + 1 :]):
            label = "tied source regions" if tied else "source regions"
            raise ValueError(f"overlapping {label} for record {record.record_id}")
    if sum(region.cardinality for region in regions) != prod(record.shape):
        label = "tied source region gap" if tied else "source region gap"
        raise ValueError(f"{label} for record {record.record_id}")


def _normalized_output_domain(
    target: OutputMemberTarget,
    entry_domain: FamilyIndexDomain,
) -> FamilyIndexDomain:
    fixed_layer = next(
        (
            item.member
            for item in target.fixed_coordinates
            if isinstance(item, FixedLayerCoordinate)
        ),
        None,
    )
    fixed_axes = {
        item.axis_name: item.member
        for item in target.fixed_coordinates
        if isinstance(item, FixedFamilyAxisCoordinate)
    }
    target_axes = {axis.name: axis for axis in target.member_domain.independent_axes}
    entry_axes = {axis.name: axis for axis in entry_domain.independent_axes}
    if set(target_axes) | set(fixed_axes) != set(entry_axes):
        raise ValueError("output target family coordinates must be disjoint and total")
    if set(target_axes) & set(fixed_axes):
        raise ValueError("output target family coordinates overlap")
    normalized_axes: list[AxisDomain] = []
    for axis_name, entry_axis in entry_axes.items():
        if axis_name in fixed_axes:
            member = fixed_axes[axis_name]
            if member not in entry_axis.members:
                raise ValueError(
                    "fixed output family coordinate is outside entry domain"
                )
            normalized_axes.append(AxisDomain(axis_name, (member,)))
            continue
        selected = target_axes[axis_name]
        if not set(selected.members).issubset(entry_axis.members):
            raise ValueError("output family subdomain is outside entry domain")
        normalized_axes.append(selected)
    if entry_domain.layer_domain is None:
        if target.member_domain.layer_domain is not None or fixed_layer is not None:
            raise ValueError("output target invents a layer coordinate")
        normalized_layer = None
    elif fixed_layer is not None:
        if target.member_domain.layer_domain is not None:
            raise ValueError("output target layer coordinates overlap")
        if fixed_layer not in entry_domain.layer_domain.members:
            raise ValueError("fixed output layer is outside entry domain")
        normalized_layer = LayerDomain((fixed_layer,))
    else:
        selected_layer = target.member_domain.layer_domain
        if selected_layer is None:
            raise ValueError("output target layer coordinates must be total")
        if not set(selected_layer.members).issubset(entry_domain.layer_domain.members):
            raise ValueError("output layer subdomain is outside entry domain")
        normalized_layer = selected_layer
    normalized = FamilyIndexDomain(normalized_layer, tuple(normalized_axes))
    if normalized.cardinality == 0:
        raise ValueError("output member target must be non-empty")
    return normalized


def _domain_intersection_cardinality(
    left: FamilyIndexDomain,
    right: FamilyIndexDomain,
) -> int:
    if left.axis_names != right.axis_names:
        return 0
    if left.layer_domain is None:
        layer_cardinality = 1
    else:
        assert right.layer_domain is not None
        layer_cardinality = len(
            set(left.layer_domain.members) & set(right.layer_domain.members)
        )
    left_axes = {axis.name: axis for axis in left.independent_axes}
    right_axes = {axis.name: axis for axis in right.independent_axes}
    return layer_cardinality * prod(
        len(set(left_axes[name].members) & set(right_axes[name].members))
        for name in left_axes
    )


def _domain_factor_members(
    domain: FamilyIndexDomain,
) -> tuple[tuple[str, tuple[LayerMember | int | str, ...]], ...]:
    factors: list[tuple[str, tuple[LayerMember | int | str, ...]]] = []
    if domain.layer_domain is not None:
        factors.append(("layer", domain.layer_domain.members))
    factors.extend(
        (f"axis:{axis.name}", axis.members) for axis in domain.independent_axes
    )
    return tuple(factors)


def _candidate_posting_union(
    postings: Mapping[LayerMember | int | str, set[int]],
    selected_members: tuple[LayerMember | int | str, ...],
) -> set[int]:
    candidates: set[int] = set()
    for member in selected_members:
        posting = postings.get(member)
        if posting is not None:
            candidates.update(posting)
    return candidates


def _validate_output_domain_partition(
    complete_domain: FamilyIndexDomain,
    claims: Sequence[FamilyIndexDomain],
) -> None:
    """Prove an exact product-free partition.

    Singleton claims use a linear factor-key hash. General rectangular claims
    use an exact output-sensitive search seeded by the rarest posting factor.
    """
    factors = tuple(name for name, _ in _domain_factor_members(complete_domain))
    claim_factor_records = tuple(_domain_factor_members(claim) for claim in claims)
    if any(
        tuple(name for name, _ in claim_factors) != factors
        for claim_factors in claim_factor_records
    ):
        raise ValueError("output member domain factors differ from entry domain")
    if all(
        all(len(selected_members) == 1 for _, selected_members in claim_factors)
        for claim_factors in claim_factor_records
    ):
        seen_singleton_keys: set[tuple[tuple[str, LayerMember | int | str], ...]] = (
            set()
        )
        for claim_factors in claim_factor_records:
            singleton_key = tuple(
                (factor_name, selected_members[0])
                for factor_name, selected_members in claim_factors
            )
            if singleton_key in seen_singleton_keys:
                raise ValueError("overlapping output member domains")
            seen_singleton_keys.add(singleton_key)
        if sum(claim.cardinality for claim in claims) != complete_domain.cardinality:
            raise ValueError("output member domain gap")
        return
    postings: dict[
        str,
        dict[LayerMember | int | str, set[int]],
    ] = {name: {} for name in factors}
    prior_claim_factors: list[dict[str, frozenset[LayerMember | int | str]]] = []
    for claim_id, claim_factors in enumerate(claim_factor_records):
        if not factors:
            if claim_id:
                raise ValueError("overlapping output member domains")
            continue
        claim_member_sets = {
            factor_name: frozenset(selected_members)
            for factor_name, selected_members in claim_factors
        }
        ranked_factors = sorted(
            claim_factors,
            key=lambda item: (
                sum(
                    len(posting)
                    for member in item[1]
                    if (posting := postings[item[0]].get(member)) is not None
                ),
                item[0],
            ),
        )
        rarest_name, rarest_members = ranked_factors[0]
        rarest_posting_cardinality = sum(
            len(posting)
            for member in rarest_members
            if (posting := postings[rarest_name].get(member)) is not None
        )
        if rarest_posting_cardinality:
            candidates = _candidate_posting_union(
                postings[rarest_name],
                rarest_members,
            )
            for candidate_id in candidates:
                prior_factors = prior_claim_factors[candidate_id]
                if all(
                    not claim_member_sets[factor_name].isdisjoint(
                        prior_factors[factor_name]
                    )
                    for factor_name in factors
                ):
                    raise ValueError("overlapping output member domains")
        prior_claim_factors.append(claim_member_sets)
        for factor_name, selected_members in claim_factors:
            factor_postings = postings[factor_name]
            for member in selected_members:
                factor_postings.setdefault(member, set()).add(claim_id)
    if sum(claim.cardinality for claim in claims) != complete_domain.cardinality:
        raise ValueError("output member domain gap")


def _mapped_target_spans(
    mapping: SourceToSemanticAxisMapping,
) -> tuple[SourceIndexSpan, ...]:
    return tuple(
        SourceIndexSpan(
            segment.target_ordinal_start,
            segment.target_ordinal_start
            + segment.target_ordinal_step * segment.source_span.cardinality,
            segment.target_ordinal_step,
        )
        for segment in mapping.segments
    )


def _span_contains(span: SourceIndexSpan, value: int) -> bool:
    return span.start <= value < span.stop and (value - span.start) % span.step == 0


def _mapping_matches_contiguous_size(
    mapping: SourceToSemanticAxisMapping,
    size: int,
) -> bool:
    target_spans = _mapped_target_spans(mapping)
    if any(
        _spans_intersect(left, right)
        for index, left in enumerate(target_spans)
        for right in target_spans[index + 1 :]
    ):
        return False
    return sum(span.cardinality for span in target_spans) == size and all(
        span.stop <= size for span in target_spans
    )


def _mapping_value_at(
    mapping: SourceToSemanticAxisMapping,
    source_ordinal: int,
) -> int:
    for segment in mapping.segments:
        if _span_contains(segment.source_span, source_ordinal):
            source_offset = (source_ordinal - segment.source_span.start) // (
                segment.source_span.step
            )
            return (
                segment.target_ordinal_start
                + source_offset * segment.target_ordinal_step
            )
    raise ValueError("layer mapping does not cover source ordinal")


def _validate_axis_mappings(
    edge: CanonicalValueClassificationEdge,
    entry: ParameterInventoryEntry,
    normalized_domain: FamilyIndexDomain,
    component_axes: tuple[tuple[str, int], ...],
) -> None:
    targets = tuple(
        _semantic_axis_target_key(item.target) for item in edge.axis_mappings
    )
    if len(targets) != len(set(targets)):
        raise ValueError("classification edge contains duplicate semantic axis targets")
    selections = {item.axis_index: item for item in edge.source_region.axis_selections}
    mappings_by_source_axis: dict[int, list[SourceToSemanticAxisMapping]] = {}
    for mapping in edge.axis_mappings:
        selection = selections.get(mapping.source_axis_index)
        if selection is None:
            raise ValueError("axis mapping references an unknown source axis")
        mappings_by_source_axis.setdefault(mapping.source_axis_index, []).append(
            mapping
        )
        mapped_source = SourceAxisSelection(
            mapping.source_axis_index,
            tuple(item.source_span for item in mapping.segments),
        )
        if (
            _axis_selection_intersection_cardinality(selection, mapped_source)
            != selection.cardinality
            or mapped_source.cardinality != selection.cardinality
        ):
            raise ValueError("axis mapping source spans must exactly cover edge region")
    if set(mappings_by_source_axis) != set(selections):
        raise ValueError("axis mappings must cover every raw source axis")
    allowed_layer_pair = {
        (1, "global_decoder_layer"),
        (1, "moe_ordinal"),
    }
    for mappings in mappings_by_source_axis.values():
        source_targets = {
            _semantic_axis_target_key(mapping.target) for mapping in mappings
        }
        if len(mappings) > 1 and source_targets != allowed_layer_pair:
            raise ValueError(
                "each raw source axis must feed one semantic target except the "
                "correlated layer pair"
            )
    expected_member_targets = {
        (0, name) if name not in {"global_decoder_layer", "moe_ordinal"} else (1, name)
        for name in edge.output.member_domain.axis_names
    }
    actual_member_targets = {
        _semantic_axis_target_key(mapping.target)
        for mapping in edge.axis_mappings
        if not isinstance(mapping.target, ComponentAxisTarget)
    }
    if actual_member_targets != expected_member_targets:
        raise ValueError(
            "mapped and fixed member coordinates must be disjoint and total"
        )
    expected_component_targets = {
        (2, str(edge.component_role), axis_name) for axis_name, _ in component_axes
    }
    actual_component_targets = {
        _semantic_axis_target_key(mapping.target)
        for mapping in edge.axis_mappings
        if isinstance(mapping.target, ComponentAxisTarget)
    }
    if actual_component_targets != expected_component_targets:
        raise ValueError("component axis mappings must be exact and total")
    component_sizes = dict(component_axes)
    family_axes = {
        axis.name: axis for axis in edge.output.member_domain.independent_axes
    }
    layer_domain = edge.output.member_domain.layer_domain
    for mapping in edge.axis_mappings:
        target = mapping.target
        if isinstance(target, FamilyIndexAxisTarget):
            axis = family_axes.get(target.axis_name)
            valid = axis is not None and _mapping_matches_contiguous_size(
                mapping,
                len(axis.members),
            )
        elif isinstance(target, LayerCoordinateTarget):
            valid = layer_domain is not None and _mapping_matches_contiguous_size(
                mapping,
                len(layer_domain.members),
            )
        else:
            valid = _mapping_matches_contiguous_size(
                mapping,
                component_sizes[target.component_axis],
            )
        if not valid:
            raise ValueError("axis mapping target ordinals do not match output domain")
    if layer_domain is not None and "moe_ordinal" in layer_domain.axis_names:
        layer_mappings = {
            mapping.target.coordinate: mapping
            for mapping in edge.axis_mappings
            if isinstance(mapping.target, LayerCoordinateTarget)
        }
        global_mapping = layer_mappings["global_decoder_layer"]
        moe_mapping = layer_mappings["moe_ordinal"]
        if global_mapping.source_axis_index != moe_mapping.source_axis_index:
            raise ValueError(
                "correlated layer relation must derive from one source axis"
            )
        selection = selections[global_mapping.source_axis_index]
        mapped_relation = tuple(
            LayerMember(
                layer_domain.members[
                    _mapping_value_at(global_mapping, source_ordinal)
                ].global_decoder_layer,
                layer_domain.members[
                    _mapping_value_at(moe_mapping, source_ordinal)
                ].moe_ordinal,
            )
            for span in selection.spans
            for source_ordinal in range(span.start, span.stop, span.step)
        )
        if len(mapped_relation) != len(set(mapped_relation)) or set(
            mapped_relation
        ) != set(layer_domain.members):
            raise ValueError("axis mappings violate correlated layer relation")
    expected_cardinality = normalized_domain.cardinality * prod(
        extent for _, extent in component_axes
    )
    if edge.source_region.cardinality != expected_cardinality:
        raise ValueError(
            "source region cardinality does not match output member/component domain"
        )


def validate_semantic_graph_build_fragment(
    schema_version: int,
    graph_input: GraphTopologyInput,
    source_records: tuple[SourceDiscoveryRecord, ...],
    fragment: SemanticGraphBuildFragment,
) -> None:
    """Validate one fragment's exact raw-to-semantic compact accounting."""
    _require_int(schema_version, "semantic schema_version", minimum=1)
    if not isinstance(graph_input, GraphTopologyInput):
        raise TypeError("graph_input must be GraphTopologyInput")
    if not isinstance(fragment, SemanticGraphBuildFragment):
        raise TypeError("fragment must be SemanticGraphBuildFragment")
    graph_id = graph_input.declaration.graph_instance_id
    if (
        fragment.graph_instance_id != graph_id
        or fragment.manifest.graph_instance_id != graph_id
    ):
        raise ValueError("fragment graph_instance_id must match its declaration")
    if fragment.manifest.lifecycle != graph_input.declaration.lifecycle:
        raise ValueError("fragment lifecycle must match its declaration")
    if fragment.manifest.model_revision != graph_input.resolved_model_revision:
        raise ValueError("fragment revision must match its topology input")
    fragment.manifest.validate_complete()
    records = tuple(source_records)
    if any(record.graph_instance_id != graph_id for record in records):
        raise ValueError("source record belongs to another graph")
    record_ids = tuple(record.record_id for record in records)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate source discovery record ID")
    records_by_id = {record.record_id: record for record in records}
    edges_by_record: dict[str, list[DiscoveryClassificationEdge]] = {
        record_id: [] for record_id in record_ids
    }
    for edge in fragment.classification_edges:
        edge_record_id = _require_text(edge.record_id, "classification edge record_id")
        if edge_record_id not in edges_by_record:
            raise ValueError("classification edge references unknown source record")
        edges_by_record[edge_record_id].append(edge)
    for record in records:
        edges = tuple(edges_by_record[record.record_id])
        if record.source_mutability == SourceMutability.ABSENT:
            if len(edges) != 1 or not isinstance(
                edges[0], AbsentDiscoveryDispositionEdge
            ):
                raise ValueError(
                    "absent record requires exactly one absent disposition"
                )
            continue
        if record.provenance == SourceRecordProvenance.TIED_STORAGE:
            if not edges or any(
                not isinstance(edge, TiedAliasClassificationEdge) for edge in edges
            ):
                raise ValueError("tied-storage record requires only tied alias edges")
            _validate_region_partition(
                record,
                tuple(edge.aliased_source_region for edge in edges),
                tied=True,
            )
            continue
        if not edges or any(
            not isinstance(edge, CanonicalValueClassificationEdge) for edge in edges
        ):
            raise ValueError(
                "present canonical record requires consuming canonical edges"
            )
        _validate_region_partition(
            record,
            tuple(edge.source_region for edge in edges),
            tied=False,
        )

    entries_by_id = {entry.entry_id: entry for entry in fragment.inventory_entries}
    if len(entries_by_id) != len(fragment.inventory_entries):
        raise ValueError("fragment contains duplicate semantic entry")
    if any(entry.graph_instance_id != graph_id for entry in fragment.inventory_entries):
        raise ValueError("fragment contains a foreign semantic entry")
    if fragment.manifest.inventory_entry_ids != tuple(sorted(entries_by_id)):
        raise ValueError("fragment manifest inventory accounting mismatch")
    owners_by_reference = {
        owner.owner_family: owner for owner in fragment.source_owners
    }
    if len(owners_by_reference) != len(fragment.source_owners):
        raise ValueError("fragment contains duplicate source owner")
    if any(
        owner.owner_family.graph_instance_id != graph_id
        for owner in fragment.source_owners
    ):
        raise ValueError("fragment contains a foreign source owner")
    output_claims: dict[tuple[str, ComponentRole], list[FamilyIndexDomain]] = {}
    justified_owners: set[OwnerFamilyReference] = set()
    raw_records_by_native_owner: dict[str, set[SourceDiscoveryRecord]] = {}
    canonical_owners_by_native_owner: dict[
        str,
        set[OwnerFamilyReference],
    ] = {}
    native_owners_by_direct_entry: dict[str, set[str]] = {}
    for edge in fragment.classification_edges:
        if isinstance(edge, AbsentDiscoveryDispositionEdge):
            continue
        record = records_by_id[edge.record_id]
        target = (
            edge.output
            if isinstance(edge, CanonicalValueClassificationEdge)
            else edge.alias_output
        )
        entry = entries_by_id.get(target.inventory_entry_id)
        if entry is None:
            raise ValueError("classification edge has unknown output inventory entry")
        components_by_role = {
            component.role: component for component in entry.member.format.components
        }
        component = components_by_role.get(edge.component_role)
        if component is None:
            raise ValueError("classification edge claims an unknown format component")
        if record.dtype != component.dtype:
            raise ValueError("raw dtype does not match claimed format component")
        component_axes = resolve_component_axes(
            component,
            logical_axes=entry.member.logical_axes,
            logical_shape=entry.member.logical_shape,
        )
        normalized = _normalized_output_domain(
            target, entry.member.ownership.binding.member_domain
        )
        output_claims.setdefault((entry.entry_id, edge.component_role), []).append(
            normalized
        )
        if isinstance(edge, CanonicalValueClassificationEdge):
            if entry.value_provenance == ValueProvenance.TIED_ALIAS:
                raise ValueError("canonical edge cannot justify a tied alias entry")
            if (
                graph_input.declaration.lifecycle.rollout_participation
                == RolloutParticipation.SERVED_FROM_CHECKPOINT
                and record.provenance == SourceRecordProvenance.TRAINING_RUNTIME
            ):
                raise ValueError(
                    "checkpoint-served graph cannot directly own training-runtime "
                    "authority"
                )
            expected_value_provenance = {
                SourceRecordProvenance.TRAINING_RUNTIME: ValueProvenance.TRAINING_PARAMETER,
                SourceRecordProvenance.CHECKPOINT_STORAGE: ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
                SourceRecordProvenance.BACKEND_DERIVED: ValueProvenance.BACKEND_DERIVED,
            }.get(record.provenance)
            if entry.value_provenance != expected_value_provenance:
                raise ValueError("value provenance is not backed by raw discovery")
            if (
                edge.canonical_owner_family
                != entry.member.ownership.binding.canonical_owner_family
            ):
                raise ValueError(
                    "canonical edge owner differs from semantic entry owner"
                )
            if edge.canonical_owner_family not in owners_by_reference:
                raise ValueError("canonical edge references an unknown source owner")
            assert record.source_native_owner_id is not None
            raw_records_by_native_owner.setdefault(
                record.source_native_owner_id,
                set(),
            ).add(record)
            canonical_owners_by_native_owner.setdefault(
                record.source_native_owner_id,
                set(),
            ).add(edge.canonical_owner_family)
            native_owners_by_direct_entry.setdefault(entry.entry_id, set()).add(
                record.source_native_owner_id
            )
            justified_owners.add(edge.canonical_owner_family)
            _validate_axis_mappings(edge, entry, normalized, component_axes)
        else:
            if entry.value_provenance != ValueProvenance.TIED_ALIAS:
                raise ValueError("tied edge requires a tied-alias semantic entry")
            binding = entry.member.ownership.binding
            if edge.canonical_owner_family != binding.canonical_owner_family:
                raise ValueError("tied edge owner differs from alias entry owner")
            if edge.canonical_value_entry_id != binding.canonical_value_entry_id:
                raise ValueError("tied edge target differs from alias entry target")
            if len(edge.alias_to_canonical_axes) != len(
                binding.member_to_value_axes
            ) or set(edge.alias_to_canonical_axes) != set(binding.member_to_value_axes):
                raise ValueError("tied edge projection differs from alias binding")
            expected_cardinality = normalized.cardinality * prod(
                extent for _, extent in component_axes
            )
            if edge.aliased_source_region.cardinality != expected_cardinality:
                raise ValueError(
                    "tied source cardinality does not match alias component domain"
                )
    for native_owner_id, raw_owner_records in raw_records_by_native_owner.items():
        raw_records_for_owner = tuple(raw_owner_records)
        first = raw_records_for_owner[0]
        if any(
            (
                record.provenance,
                record.provenance_evidence,
                record.source_mutability,
                record.mutability_evidence,
            )
            != (
                first.provenance,
                first.provenance_evidence,
                first.source_mutability,
                first.mutability_evidence,
            )
            for record in raw_records_for_owner[1:]
        ):
            raise ValueError(
                f"raw records for native owner {native_owner_id} disagree on authority"
            )
        canonical_owners = canonical_owners_by_native_owner[native_owner_id]
        if len(canonical_owners) != 1:
            raise ValueError("one native owner must resolve to one canonical owner")
        canonical_owner = next(iter(canonical_owners))
        source_owner = owners_by_reference[canonical_owner]
        if (
            source_owner.source_mutability != first.source_mutability
            or source_owner.mutability_evidence_source != first.mutability_evidence
        ):
            raise ValueError(
                "owner mutability evidence differs from raw discovery authority"
            )
    for edge in fragment.classification_edges:
        if not isinstance(edge, TiedAliasClassificationEdge):
            continue
        if edge.canonical_owner_family.graph_instance_id != graph_id:
            continue
        record = records_by_id[edge.record_id]
        alias_entry = entries_by_id[edge.alias_output.inventory_entry_id]
        direct_entry = entries_by_id.get(edge.canonical_value_entry_id)
        if direct_entry is None:
            raise ValueError("tied edge direct target is missing")
        if direct_entry.value_provenance == ValueProvenance.TIED_ALIAS:
            raise ValueError("tied edge direct target must not be an alias")
        direct_binding = direct_entry.member.ownership.binding
        if direct_binding.canonical_value_entry_id != direct_entry.entry_id:
            raise ValueError("tied edge direct target must bind directly")
        if direct_binding.canonical_owner_family != edge.canonical_owner_family:
            raise ValueError("tied edge direct target has a different owner")
        if (
            alias_entry.member.logical_dtype != direct_entry.member.logical_dtype
            or alias_entry.member.logical_shape != direct_entry.member.logical_shape
            or alias_entry.member.logical_axes != direct_entry.member.logical_axes
            or alias_entry.member.format != direct_entry.member.format
        ):
            raise ValueError("tied edge direct target is incompatible with alias")
        direct_native_owners = native_owners_by_direct_entry.get(
            direct_entry.entry_id,
            set(),
        )
        if record.source_native_owner_id not in direct_native_owners:
            raise ValueError("tied native owner differs from direct target")
        assert record.source_native_owner_id is not None
        canonical_raw_records = raw_records_by_native_owner[
            record.source_native_owner_id
        ]
        canonical_raw = next(iter(canonical_raw_records))
        if (
            record.source_mutability != canonical_raw.source_mutability
            or record.mutability_evidence != canonical_raw.mutability_evidence
        ):
            raise ValueError(
                "tied mutability evidence differs from canonical raw authority"
            )
    for entry in fragment.inventory_entries:
        for component in entry.member.format.components:
            claims = output_claims.get((entry.entry_id, component.role), [])
            if not claims:
                if not any(key[0] == entry.entry_id for key in output_claims):
                    raise ValueError(
                        f"semantic entry {entry.entry_id} has no classification edge"
                    )
                raise ValueError(
                    f"missing output component {component.role} for {entry.entry_id}"
                )
            _validate_output_domain_partition(
                entry.member.ownership.binding.member_domain,
                claims,
            )
    for owner_reference in owners_by_reference:
        if owner_reference not in justified_owners:
            raise ValueError(
                f"source owner {owner_reference} has no classification edge"
            )
    if (
        graph_input.declaration.lifecycle.rollout_participation
        == RolloutParticipation.SERVED_FROM_SOURCE
        and not any(
            isinstance(
                edge,
                (CanonicalValueClassificationEdge, TiedAliasClassificationEdge),
            )
            for edge in fragment.classification_edges
        )
    ):
        raise ValueError(
            "source-served graph requires a present canonical owner or tied alias"
        )
    for contribution in fragment.role_contributions:
        if contribution.schema_version != schema_version:
            raise ValueError("role contribution schema version mismatch")
        if not set(contribution.expected_inventory_entry_ids).issubset(entries_by_id):
            raise ValueError("role contribution references an unknown semantic entry")


class ModelTopologyAdapter(Protocol):
    """Pure family-specific classifier for one independently declared graph."""

    @property
    def adapter_id(self) -> str: ...

    def supports(self, model_config: Mapping[str, object]) -> bool: ...

    def classify_graph(
        self,
        schema_version: int,
        graph_input: GraphTopologyInput,
        source_records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment: ...


def _default_adapters() -> tuple[ModelTopologyAdapter, ...]:
    # Local import breaks the intentional topology/adapters registration cycle.
    from nemo_rl.precision_policy.adapters import BUILTIN_TOPOLOGY_ADAPTERS

    return BUILTIN_TOPOLOGY_ADAPTERS


def select_model_topology_adapter(
    model_config: Mapping[str, object],
    *,
    adapters: Sequence[ModelTopologyAdapter] | None = None,
) -> ModelTopologyAdapter:
    """Select exactly one deterministic family adapter or fail closed."""
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")
    candidates = tuple(_default_adapters() if adapters is None else adapters)
    adapter_ids = tuple(
        _require_text(item.adapter_id, "adapter_id") for item in candidates
    )
    if len(adapter_ids) != len(set(adapter_ids)):
        raise ValueError("topology adapter IDs must be unique")
    ordered = tuple(sorted(candidates, key=lambda item: item.adapter_id))
    matching = tuple(adapter for adapter in ordered if adapter.supports(model_config))
    if not matching:
        model_type = model_config.get("model_type", "<missing>")
        raise ValueError(f"unsupported model topology: model_type={model_type!r}")
    if len(matching) != 1:
        names = ", ".join(adapter.adapter_id for adapter in matching)
        raise ValueError(f"ambiguous model topology adapters: {names}")
    return matching[0]


def _graph_input_sort_key(graph_input: GraphTopologyInput) -> tuple[int, str]:
    graph_instance_id = graph_input.declaration.graph_instance_id
    return (0 if graph_instance_id == "main" else 1, graph_instance_id)


def _require_unique_fragment_outputs(
    fragments: tuple[SemanticGraphBuildFragment, ...],
) -> None:
    entry_ids = tuple(
        entry.entry_id for fragment in fragments for entry in fragment.inventory_entries
    )
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("duplicate inventory entry across topology fragments")
    owner_references = tuple(
        owner.owner_family for fragment in fragments for owner in fragment.source_owners
    )
    if len(owner_references) != len(set(owner_references)):
        raise ValueError("duplicate source owner across topology fragments")


def _source_region_is_subset(
    subset: SourceRegion,
    superset: SourceRegion,
) -> bool:
    if subset.source_shape != superset.source_shape:
        return False
    return all(
        _axis_selection_intersection_cardinality(subset_axis, superset_axis)
        == subset_axis.cardinality
        for subset_axis, superset_axis in zip(
            subset.axis_selections,
            superset.axis_selections,
            strict=True,
        )
    )


def _domain_relation_factors(
    domain: FamilyIndexDomain,
) -> tuple[
    tuple[tuple[str, ...], frozenset[tuple[int | str, ...]]],
    ...,
]:
    factors: list[tuple[tuple[str, ...], frozenset[tuple[int | str, ...]]]] = []
    if domain.layer_domain is not None:
        layer_axes = domain.layer_domain.axis_names
        factors.append(
            (
                layer_axes,
                frozenset(
                    tuple(
                        _layer_axis_value(member, axis_name) for axis_name in layer_axes
                    )
                    for member in domain.layer_domain.members
                ),
            )
        )
    factors.extend(
        (
            (axis.name,),
            frozenset((member,) for member in axis.members),
        )
        for axis in domain.independent_axes
    )
    return tuple(factors)


def _layer_axis_value(member: LayerMember, axis_name: str) -> int:
    if axis_name == "global_decoder_layer":
        return member.global_decoder_layer
    if axis_name == "moe_ordinal" and member.moe_ordinal is not None:
        return member.moe_ordinal
    raise ValueError(f"unknown or absent layer axis: {axis_name}")


def _domain_projection_is_subset(
    domain: FamilyIndexDomain,
    projected_axes: tuple[str, ...],
    allowed_points: frozenset[tuple[int | str, ...]],
) -> bool:
    factors = _domain_relation_factors(domain)
    locations = {
        axis_name: (factor_index, axis_index)
        for factor_index, (factor_axes, _) in enumerate(factors)
        for axis_index, axis_name in enumerate(factor_axes)
    }
    if any(axis_name not in locations for axis_name in projected_axes):
        return False
    selected_locations = tuple(locations[axis_name] for axis_name in projected_axes)
    factor_indices = {factor_index for factor_index, _ in selected_locations}
    if len(factor_indices) == 1:
        factor_index = selected_locations[0][0]
        factor_points = factors[factor_index][1]
        projected_points = frozenset(
            tuple(point[axis_index] for _, axis_index in selected_locations)
            for point in factor_points
        )
        return projected_points.issubset(allowed_points)

    selected_members: list[frozenset[int | str]] = []
    for factor_index, axis_index in selected_locations:
        selected_members.append(
            frozenset(point[axis_index] for point in factors[factor_index][1])
        )
    expected_cardinality = prod(len(members) for members in selected_members)
    covered_cardinality = sum(
        all(value in members for value, members in zip(point, selected_members))
        for point in allowed_points
    )
    return covered_cardinality == expected_cardinality


def _target_domain_is_subset_of_projected_source(
    source_domain: FamilyIndexDomain,
    target_domain: FamilyIndexDomain,
    projections: tuple[AxisProjection, ...],
) -> bool:
    source_axes = set(source_domain.axis_names)
    target_axes = set(target_domain.axis_names)
    source_to_target = {
        projection.member_axis: projection.owner_axis for projection in projections
    }
    if (
        len(source_to_target) != len(projections)
        or set(source_to_target) != source_axes
        or set(source_to_target.values()) != target_axes
        or len(set(source_to_target.values())) != len(projections)
    ):
        return False
    return all(
        _domain_projection_is_subset(
            target_domain,
            tuple(source_to_target[axis_name] for axis_name in factor_axes),
            allowed_points,
        )
        for factor_axes, allowed_points in _domain_relation_factors(source_domain)
    )


def _validate_global_canonical_native_authority(
    fragments: tuple[SemanticGraphBuildFragment, ...],
    records_by_id: Mapping[str, SourceDiscoveryRecord],
) -> None:
    owner_by_native_id: dict[str, OwnerFamilyReference] = {}
    record_by_native_id: dict[str, SourceDiscoveryRecord] = {}
    for fragment in fragments:
        for edge in fragment.classification_edges:
            if not isinstance(edge, CanonicalValueClassificationEdge):
                continue
            record = records_by_id[edge.record_id]
            assert record.source_native_owner_id is not None
            native_owner_id = record.source_native_owner_id
            prior_record = record_by_native_id.setdefault(native_owner_id, record)
            if (
                prior_record.provenance,
                prior_record.provenance_evidence,
                prior_record.source_mutability,
                prior_record.mutability_evidence,
            ) != (
                record.provenance,
                record.provenance_evidence,
                record.source_mutability,
                record.mutability_evidence,
            ):
                raise ValueError(
                    f"canonical native owner {native_owner_id} has inconsistent "
                    "authority evidence"
                )
            prior_owner = owner_by_native_id.setdefault(
                native_owner_id,
                edge.canonical_owner_family,
            )
            if prior_owner != edge.canonical_owner_family:
                raise ValueError(
                    f"canonical native owner {native_owner_id} resolves to multiple "
                    "owners"
                )


def _validate_cross_graph_tied_aliases(
    fragments: tuple[SemanticGraphBuildFragment, ...],
    records_by_id: Mapping[str, SourceDiscoveryRecord],
) -> None:
    entries_by_id = {
        entry.entry_id: entry
        for fragment in fragments
        for entry in fragment.inventory_entries
    }
    owners_by_reference = {
        owner.owner_family: owner
        for fragment in fragments
        for owner in fragment.source_owners
    }
    canonical_backings_by_component: dict[
        tuple[str, ComponentRole],
        list[tuple[CanonicalValueClassificationEdge, SourceDiscoveryRecord]],
    ] = {}
    for fragment in fragments:
        for edge in fragment.classification_edges:
            if not isinstance(edge, CanonicalValueClassificationEdge):
                continue
            canonical_backings_by_component.setdefault(
                (edge.output.inventory_entry_id, edge.component_role),
                [],
            ).append((edge, records_by_id[edge.record_id]))

    for fragment in fragments:
        for edge in fragment.classification_edges:
            if not isinstance(edge, TiedAliasClassificationEdge):
                continue
            alias_entry = entries_by_id[edge.alias_output.inventory_entry_id]
            direct_entry = entries_by_id.get(edge.canonical_value_entry_id)
            if direct_entry is None:
                raise ValueError("cross-graph tied alias direct target is missing")
            if direct_entry.value_provenance == ValueProvenance.TIED_ALIAS:
                raise ValueError("cross-graph tied alias target must be direct")
            direct_binding = direct_entry.member.ownership.binding
            if direct_binding.canonical_value_entry_id != direct_entry.entry_id:
                raise ValueError("cross-graph tied alias target must bind directly")
            if direct_binding.canonical_owner_family != edge.canonical_owner_family:
                raise ValueError("cross-graph tied alias canonical owner mismatch")
            if (
                alias_entry.member.logical_dtype != direct_entry.member.logical_dtype
                or alias_entry.member.logical_shape != direct_entry.member.logical_shape
                or alias_entry.member.logical_axes != direct_entry.member.logical_axes
                or alias_entry.member.format != direct_entry.member.format
            ):
                raise ValueError("cross-graph tied alias target is incompatible")

            owner = owners_by_reference.get(edge.canonical_owner_family)
            if owner is None:
                raise ValueError("cross-graph tied alias canonical owner is missing")
            tied_record = records_by_id[edge.record_id]
            canonical_backings = canonical_backings_by_component.get(
                (direct_entry.entry_id, edge.component_role),
                [],
            )
            matching_native_backings = tuple(
                (canonical_edge, record)
                for canonical_edge, record in canonical_backings
                if record.source_native_owner_id == tied_record.source_native_owner_id
            )
            if not matching_native_backings:
                raise ValueError(
                    "cross-graph tied native owner differs from direct target"
                )
            corresponding_backings: list[
                tuple[CanonicalValueClassificationEdge, SourceDiscoveryRecord]
            ] = []
            for canonical_edge, record in matching_native_backings:
                canonical_region = canonical_edge.source_region
                alias_region = edge.aliased_source_region
                if not _regions_intersect(canonical_region, alias_region):
                    continue
                if not _source_region_is_subset(canonical_region, alias_region):
                    raise ValueError(
                        "cross-graph tied alias subdomain has unaligned source regions"
                    )
                corresponding_backings.append((canonical_edge, record))
            corresponding_regions = tuple(
                canonical_edge.source_region
                for canonical_edge, _ in corresponding_backings
            )
            if (
                not corresponding_regions
                or any(
                    _regions_intersect(left, right)
                    for index, left in enumerate(corresponding_regions)
                    for right in corresponding_regions[index + 1 :]
                )
                or sum(region.cardinality for region in corresponding_regions)
                != edge.aliased_source_region.cardinality
            ):
                raise ValueError(
                    "cross-graph tied alias subdomain lacks exact canonical coverage"
                )
            alias_domain = _normalized_output_domain(
                edge.alias_output,
                alias_entry.member.ownership.binding.member_domain,
            )
            canonical_claims = tuple(
                _normalized_output_domain(
                    canonical_edge.output,
                    direct_entry.member.ownership.binding.member_domain,
                )
                for canonical_edge, _ in corresponding_backings
            )
            if sum(
                claim.cardinality for claim in canonical_claims
            ) != alias_domain.cardinality or any(
                not _target_domain_is_subset_of_projected_source(
                    alias_domain,
                    claim,
                    edge.alias_to_canonical_axes,
                )
                for claim in canonical_claims
            ):
                raise ValueError(
                    "cross-graph tied alias subdomain differs from canonical claims"
                )
            if (
                tied_record.source_mutability != owner.source_mutability
                or tied_record.mutability_evidence != owner.mutability_evidence_source
                or any(
                    record.source_mutability != tied_record.source_mutability
                    or record.mutability_evidence != tied_record.mutability_evidence
                    for _, record in corresponding_backings
                )
            ):
                raise ValueError(
                    "cross-graph tied mutability evidence differs from canonical owner"
                )


def _merge_role_contributions(
    schema_version: int,
    fragments: tuple[SemanticGraphBuildFragment, ...],
    entry_ids: frozenset[str],
) -> tuple[RoleDefinition, ...]:
    central_definitions = builtin_role_definitions(schema_version, {})
    central_predicates = {
        definition.role_name: definition.predicate for definition in central_definitions
    }
    predicates = dict(central_predicates)
    expected_ids_by_role: dict[str, set[str]] = {
        role_name: set() for role_name in central_predicates
    }
    contributions = sorted(
        (
            contribution
            for fragment in fragments
            for contribution in fragment.role_contributions
        ),
        key=lambda contribution: (
            contribution.schema_version,
            contribution.role_name,
            contribution.expected_inventory_entry_ids,
        ),
    )
    for contribution in contributions:
        if contribution.schema_version != schema_version:
            raise ValueError("role contribution schema version mismatch")
        unknown_entries = set(contribution.expected_inventory_entry_ids) - entry_ids
        if unknown_entries:
            raise ValueError(
                "role contribution references unknown inventory entry: "
                f"{sorted(unknown_entries)[0]}"
            )
        central_predicate = central_predicates.get(contribution.role_name)
        if (
            central_predicate is not None
            and contribution.predicate != central_predicate
        ):
            raise ValueError("adapter cannot replace a built-in role predicate")
        if central_predicate is None and "." not in contribution.role_name:
            raise ValueError("adapter role must be namespaced")
        existing_predicate = predicates.get(contribution.role_name)
        if (
            existing_predicate is not None
            and existing_predicate != contribution.predicate
        ):
            raise ValueError("role contributions have conflicting predicates")
        predicates[contribution.role_name] = contribution.predicate
        expected_ids = expected_ids_by_role.setdefault(contribution.role_name, set())
        overlap = expected_ids.intersection(contribution.expected_inventory_entry_ids)
        if overlap:
            raise ValueError(
                "role contributions have overlapping expected domains: "
                f"{sorted(overlap)[0]}"
            )
        expected_ids.update(contribution.expected_inventory_entry_ids)

    central_expected_domains = {
        role_name: RoleExpectedDomain(role_name, tuple(sorted(expected_ids)))
        for role_name, expected_ids in expected_ids_by_role.items()
        if role_name in central_predicates
    }
    definitions = list(
        builtin_role_definitions(schema_version, central_expected_domains)
    )
    definitions.extend(
        RoleDefinition(
            schema_version=schema_version,
            role_name=role_name,
            predicate=predicates[role_name],
            expected_domain=RoleExpectedDomain(
                role_name,
                tuple(sorted(expected_ids)),
            ),
        )
        for role_name, expected_ids in expected_ids_by_role.items()
        if role_name not in central_predicates
    )
    return tuple(sorted(definitions, key=lambda definition: definition.role_name))


def build_semantic_manifest_bundle(
    schema_version: int,
    graph_inputs: Sequence[GraphTopologyInput],
    source_discovery: SourceDiscoveryInventory,
) -> SemanticManifestBundle:
    """Classify and atomically validate every declared semantic graph."""
    _require_int(schema_version, "semantic schema_version", minimum=1)
    if not isinstance(source_discovery, SourceDiscoveryInventory):
        raise TypeError("source_discovery must be SourceDiscoveryInventory")
    inputs = tuple(graph_inputs)
    if any(not isinstance(graph_input, GraphTopologyInput) for graph_input in inputs):
        raise TypeError("graph_inputs must contain GraphTopologyInput records")
    graph_ids = tuple(
        graph_input.declaration.graph_instance_id for graph_input in inputs
    )
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("duplicate graph topology input declaration")
    declared_graph_ids = set(graph_ids)
    discovered_graph_ids = {
        record.graph_instance_id for record in source_discovery.records
    }
    undeclared = discovered_graph_ids - declared_graph_ids
    if undeclared:
        raise ValueError(f"undeclared source discovery graph: {sorted(undeclared)[0]}")
    missing = declared_graph_ids - discovered_graph_ids
    if missing:
        raise ValueError(f"missing source discovery graph: {sorted(missing)[0]}")

    records_by_graph: dict[str, list[SourceDiscoveryRecord]] = {
        graph_id: [] for graph_id in graph_ids
    }
    for record in source_discovery.records:
        records_by_graph[record.graph_instance_id].append(record)
    adapters = _default_adapters()
    fragments: list[SemanticGraphBuildFragment] = []
    for graph_input in sorted(inputs, key=_graph_input_sort_key):
        graph_id = graph_input.declaration.graph_instance_id
        records = tuple(records_by_graph[graph_id])
        adapter = select_model_topology_adapter(
            graph_input.model_config,
            adapters=adapters,
        )
        fragment = adapter.classify_graph(
            schema_version,
            graph_input,
            records,
        )
        validate_semantic_graph_build_fragment(
            schema_version,
            graph_input,
            records,
            fragment,
        )
        fragments.append(fragment)
    canonical_fragments = tuple(fragments)
    _require_unique_fragment_outputs(canonical_fragments)
    records_by_id = {record.record_id: record for record in source_discovery.records}
    _validate_global_canonical_native_authority(
        canonical_fragments,
        records_by_id,
    )
    _validate_cross_graph_tied_aliases(canonical_fragments, records_by_id)

    owners = tuple(
        owner for fragment in canonical_fragments for owner in fragment.source_owners
    )
    entries = tuple(
        entry
        for fragment in canonical_fragments
        for entry in fragment.inventory_entries
    )
    role_definitions = _merge_role_contributions(
        schema_version,
        canonical_fragments,
        frozenset(entry.entry_id for entry in entries),
    )
    bundle = SemanticManifestBundle(
        schema_version=schema_version,
        expected_graphs=tuple(
            graph_input.declaration
            for graph_input in sorted(inputs, key=_graph_input_sort_key)
        ),
        manifests=tuple(fragment.manifest for fragment in canonical_fragments),
        inventory=ParameterInventory(owners=owners, entries=entries),
        role_definitions=role_definitions,
    )
    bundle.validate_complete()
    return bundle
