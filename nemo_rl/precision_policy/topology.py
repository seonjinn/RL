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

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from heapq import heappop, heappush
from itertools import groupby
from math import gcd, prod
from typing import Literal, Protocol

from nemo_rl.precision_policy.semantic import (
    AxisDomain,
    AxisProjection,
    ComponentRole,
    EvidenceSource,
    FamilyIndexDomain,
    IdenticalStorageSourceAliasContract,
    IndexPathSegment,
    LayerDomain,
    LayerMember,
    OwnerFamilyReference,
    ParameterInventory,
    ParameterInventoryEntry,
    RoleDefinition,
    RoleExpectedDomain,
    RolloutParticipation,
    ResolvedSelectionTopology,
    SelectionTopologyEntry,
    SemanticTensor,
    SemanticTensorFamily,
    SemanticGraphManifest,
    SemanticManifestBundle,
    SemanticPredicate,
    SourceAliasContract,
    SourceMutability,
    SourceOwnerInventoryEntry,
    SourceReplicaSynchronizationEvidence,
    SourceSynchronizationBoundary,
    SynchronizedReplicaSourceAliasContract,
    ValueProvenance,
    _canonical_semantic_structure_value,
    _require_sha256_digest,
    builtin_role_definitions,
    resolve_component_axes,
)
from nemo_rl.precision_policy.source_discovery import (
    ExpectedContributorSet,
    GraphTopologyInput,
    RuntimeGraphSourceRequest,
    SourceDiscoveryInventory,
    SourceDiscoveryRecord,
    SourceRecordProvenance,
    _snapshot_sequence,
    validate_discovery_inventory,
    validate_runtime_discovery_inventory,
)
from nemo_rl.precision_policy.source_storage import (
    SourceDerivedRealization,
    SourceNormalizerManifest,
    SourceRealization,
    SourceStorageRealization,
)


type _SourceRecordKey = tuple[str, str]
type _SourceNativeOwnerKey = tuple[str, str]


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


@dataclass(frozen=True, slots=True, order=True)
class SourceIndexSpan:
    """Compact half-open progression over one normalized source-view axis."""

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
    """Disjoint compact spans selected from one normalized source-view axis."""

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
    """Compact Cartesian region over every normalized source-view axis once."""

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
    """Compact affine map from normalized source-view to target ordinals."""

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
    """Map one normalized source-view axis to a semantic/output coordinate."""

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
    """A consuming normalized-view edge to one canonical semantic component."""

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
class SynchronizedReplicaAliasClassificationEdge:
    """A non-consuming source replica resolved to canonical training authority."""

    record_id: str
    replica_source_region: SourceRegion
    alias_output: OutputMemberTarget
    canonical_record_id: str
    canonical_source_region: SourceRegion
    canonical_owner_family: OwnerFamilyReference
    canonical_value_entry_id: str
    component_role: ComponentRole
    alias_to_canonical_axes: tuple[AxisProjection, ...]
    synchronization: SourceReplicaSynchronizationEvidence

    def __post_init__(self) -> None:
        _require_text(self.record_id, "replica edge record_id")
        if not isinstance(self.replica_source_region, SourceRegion):
            raise TypeError("replica edge source_region must be SourceRegion")
        if not isinstance(self.alias_output, OutputMemberTarget):
            raise TypeError("replica edge output must be OutputMemberTarget")
        _require_text(self.canonical_record_id, "replica edge canonical record_id")
        if not isinstance(self.canonical_source_region, SourceRegion):
            raise TypeError("replica edge canonical source_region must be SourceRegion")
        if not isinstance(self.canonical_owner_family, OwnerFamilyReference):
            raise TypeError("replica edge owner must be OwnerFamilyReference")
        _require_text(self.canonical_value_entry_id, "replica edge canonical value")
        _require_text(self.component_role, "replica edge component role")
        projections = tuple(self.alias_to_canonical_axes)
        if any(not isinstance(item, AxisProjection) for item in projections):
            raise TypeError("replica edge axes must be AxisProjection records")
        if len(projections) != len(set(projections)):
            raise ValueError("replica edge projection contains duplicates")
        if not isinstance(
            self.synchronization,
            SourceReplicaSynchronizationEvidence,
        ):
            raise TypeError("replica edge synchronization must be typed")
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
    """The sole zero-output disposition for an absent normalized source view."""

    record_id: str

    def __post_init__(self) -> None:
        _require_text(self.record_id, "absent disposition record_id")


type DiscoveryClassificationEdge = (
    CanonicalValueClassificationEdge
    | TiedAliasClassificationEdge
    | SynchronizedReplicaAliasClassificationEdge
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
    if isinstance(edge, SynchronizedReplicaAliasClassificationEdge):
        return (
            edge.record_id,
            2,
            edge.alias_output.inventory_entry_id,
            str(edge.component_role),
            _region_sort_key(edge.replica_source_region),
            _output_target_sort_key(edge.alias_output),
            edge.canonical_record_id,
            _region_sort_key(edge.canonical_source_region),
            edge.canonical_owner_family.graph_instance_id,
            edge.canonical_owner_family.owner_family_id,
            edge.canonical_value_entry_id,
            tuple(
                (projection.member_axis, projection.owner_axis)
                for projection in edge.alias_to_canonical_axes
            ),
            edge.synchronization.replica_group_id,
            edge.synchronization.boundary.value,
            edge.synchronization.evidence_source.kind.value,
            edge.synchronization.evidence_source.locator,
            edge.synchronization.evidence_source.digest,
        )
    return (edge.record_id, 3)


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
                    SynchronizedReplicaAliasClassificationEdge,
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


@dataclass(frozen=True, slots=True)
class CanonicalSourceSemanticBinding:
    """Validated physical source evidence for one canonical classification edge."""

    graph_instance_id: str
    classification_edge: CanonicalValueClassificationEdge
    source_record: SourceDiscoveryRecord
    source_realizations: tuple[SourceRealization, ...]

    def __post_init__(self) -> None:
        _require_text(self.graph_instance_id, "source binding graph_instance_id")
        if not isinstance(self.classification_edge, CanonicalValueClassificationEdge):
            raise TypeError("source binding classification_edge must be canonical")
        if not isinstance(self.source_record, SourceDiscoveryRecord):
            raise TypeError("source binding record must be SourceDiscoveryRecord")
        realizations = tuple(self.source_realizations)
        if not realizations:
            raise ValueError("canonical source binding requires a realization")
        if any(
            not isinstance(
                realization,
                (SourceStorageRealization, SourceDerivedRealization),
            )
            for realization in realizations
        ):
            raise TypeError("source binding realizations must be typed")
        edge = self.classification_edge
        record = self.source_record
        if (
            record.graph_instance_id != self.graph_instance_id
            or edge.record_id != record.record_id
        ):
            raise ValueError("source binding edge and record identity differ")
        if edge.source_region.source_shape != record.shape:
            raise ValueError("source binding region shape differs from its record")
        for realization in realizations:
            if (
                realization.graph_instance_id != self.graph_instance_id
                or realization.output_record_id != record.record_id
            ):
                raise ValueError("source binding realization differs from its record")
            if (
                realization.output_dtype != record.dtype
                or realization.output_shape != record.shape
                or realization.output_numeric_encoding != record.numeric_encoding
            ):
                raise ValueError(
                    "source binding realization output differs from its record"
                )
        object.__setattr__(
            self,
            "source_realizations",
            tuple(sorted(realizations, key=lambda item: item.realization_id)),
        )


@dataclass(frozen=True, slots=True)
class GraphSemanticSourceBindings:
    """All canonical source classifications for one validated runtime graph."""

    graph_instance_id: str
    normalizer_manifest: SourceNormalizerManifest
    canonical_bindings: tuple[CanonicalSourceSemanticBinding, ...]

    def __post_init__(self) -> None:
        _require_text(self.graph_instance_id, "graph source bindings graph_instance_id")
        if not isinstance(self.normalizer_manifest, SourceNormalizerManifest):
            raise TypeError("normalizer_manifest must be SourceNormalizerManifest")
        bindings = tuple(self.canonical_bindings)
        if any(
            not isinstance(binding, CanonicalSourceSemanticBinding)
            for binding in bindings
        ):
            raise TypeError("canonical_bindings must contain typed bindings")
        if any(
            binding.graph_instance_id != self.graph_instance_id for binding in bindings
        ):
            raise ValueError("canonical source binding belongs to another graph")
        allowed_normalizers = frozenset(self.normalizer_manifest.contracts)
        if any(
            (
                realization.normalization
                if isinstance(realization, SourceStorageRealization)
                else realization.derivation
            )
            not in allowed_normalizers
            for binding in bindings
            for realization in binding.source_realizations
        ):
            raise ValueError("source binding normalizer is absent from its manifest")
        canonical = tuple(
            sorted(
                bindings,
                key=lambda binding: _edge_sort_key(binding.classification_edge),
            )
        )
        edge_keys = tuple(
            _edge_sort_key(binding.classification_edge) for binding in canonical
        )
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("duplicate canonical source classification binding")
        object.__setattr__(self, "canonical_bindings", canonical)


def _semantic_source_binding_digest(
    graph_bindings: tuple[GraphSemanticSourceBindings, ...],
) -> str:
    payload = _canonical_semantic_structure_value(graph_bindings)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SemanticSourceBindingInventory:
    """Canonical recoverable source-to-semantic evidence for all runtime graphs."""

    graph_bindings: tuple[GraphSemanticSourceBindings, ...]
    source_binding_digest: str = field(init=False)

    def __post_init__(self) -> None:
        bindings = tuple(self.graph_bindings)
        if any(
            not isinstance(binding, GraphSemanticSourceBindings) for binding in bindings
        ):
            raise TypeError("graph_bindings must contain typed graph bindings")
        canonical = tuple(
            sorted(
                bindings,
                key=lambda item: (
                    0 if item.graph_instance_id == "main" else 1,
                    item.graph_instance_id,
                ),
            )
        )
        graph_ids = tuple(item.graph_instance_id for item in canonical)
        if len(graph_ids) != len(set(graph_ids)):
            raise ValueError("duplicate semantic source binding graph")
        object.__setattr__(self, "graph_bindings", canonical)
        object.__setattr__(
            self,
            "source_binding_digest",
            _semantic_source_binding_digest(canonical),
        )


def validate_semantic_source_binding_inventory(
    inventory: SemanticSourceBindingInventory,
) -> SemanticSourceBindingInventory:
    """Replay canonical source-binding construction and its content digest."""
    if type(inventory) is not SemanticSourceBindingInventory:
        raise TypeError("source binding inventory must be exact")
    if type(inventory.graph_bindings) is not tuple:
        raise TypeError("source binding graph collection must be an exact tuple")
    canonical_graphs: list[GraphSemanticSourceBindings] = []
    for graph in inventory.graph_bindings:
        if type(graph) is not GraphSemanticSourceBindings:
            raise TypeError("source binding graphs must be exact records")
        if type(graph.normalizer_manifest) is not SourceNormalizerManifest:
            raise TypeError("source binding normalizer manifest must be exact")
        if type(graph.canonical_bindings) is not tuple:
            raise TypeError("canonical source bindings must be an exact tuple")
        canonical_bindings: list[CanonicalSourceSemanticBinding] = []
        for binding in graph.canonical_bindings:
            if type(binding) is not CanonicalSourceSemanticBinding:
                raise TypeError("canonical source bindings must be exact records")
            if (
                type(binding.classification_edge)
                is not CanonicalValueClassificationEdge
            ):
                raise TypeError("source classification edge must be exact")
            if type(binding.source_record) is not SourceDiscoveryRecord:
                raise TypeError("source discovery record must be exact")
            if type(binding.source_realizations) is not tuple or any(
                type(realization)
                not in (SourceStorageRealization, SourceDerivedRealization)
                for realization in binding.source_realizations
            ):
                raise TypeError("source realizations must be exact records")
            canonical_bindings.append(
                CanonicalSourceSemanticBinding(
                    graph_instance_id=binding.graph_instance_id,
                    classification_edge=binding.classification_edge,
                    source_record=binding.source_record,
                    source_realizations=binding.source_realizations,
                )
            )
        canonical_graphs.append(
            GraphSemanticSourceBindings(
                graph_instance_id=graph.graph_instance_id,
                normalizer_manifest=graph.normalizer_manifest,
                canonical_bindings=tuple(canonical_bindings),
            )
        )
    canonical = SemanticSourceBindingInventory(tuple(canonical_graphs))
    if _canonical_semantic_structure_value(inventory.graph_bindings) != (
        _canonical_semantic_structure_value(canonical.graph_bindings)
    ):
        raise ValueError("source binding inventory differs from canonical structure")
    if (
        type(inventory.source_binding_digest) is not str
        or inventory.source_binding_digest != canonical.source_binding_digest
    ):
        raise ValueError("source binding digest differs from canonical payload")
    return inventory


@dataclass(frozen=True, slots=True)
class RuntimeSourceProvenance:
    """Exact aggregate request/result identity retained with runtime topology."""

    selection_group_id: str
    request_digest: str
    result_digests: tuple[tuple[str, str], ...]
    provenance_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256_digest(self.selection_group_id, "selection_group_id")
        _require_sha256_digest(self.request_digest, "request_digest")
        result_digests = tuple(self.result_digests)
        if any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in result_digests
        ):
            raise TypeError(
                "result_digests must contain exact graph/digest string pairs"
            )
        for graph_instance_id, result_digest in result_digests:
            _require_text(graph_instance_id, "result graph_instance_id")
            _require_sha256_digest(result_digest, "result digest")
        canonical = tuple(
            sorted(
                result_digests,
                key=lambda item: (0 if item[0] == "main" else 1, item[0]),
            )
        )
        graph_ids = tuple(item[0] for item in canonical)
        if len(graph_ids) != len(set(graph_ids)):
            raise ValueError("result_digests contains duplicate graphs")
        object.__setattr__(self, "result_digests", canonical)
        payload = _canonical_semantic_structure_value(
            (
                self.selection_group_id,
                self.request_digest,
                canonical,
            )
        )
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        object.__setattr__(
            self,
            "provenance_digest",
            f"sha256:{sha256(encoded).hexdigest()}",
        )


@dataclass(frozen=True, slots=True)
class SemanticTopologyBuildResult:
    """Atomic semantic manifest plus the validated classifications that produced it."""

    manifest_bundle: SemanticManifestBundle
    source_bindings: SemanticSourceBindingInventory
    runtime_source_provenance: RuntimeSourceProvenance | None = None

    def __post_init__(self) -> None:
        if type(self.manifest_bundle) is not SemanticManifestBundle:
            raise TypeError("manifest_bundle must be SemanticManifestBundle")
        if type(self.source_bindings) is not SemanticSourceBindingInventory:
            raise TypeError("source_bindings must be SemanticSourceBindingInventory")
        validate_semantic_source_binding_inventory(self.source_bindings)
        if self.runtime_source_provenance is not None:
            if type(self.runtime_source_provenance) is not RuntimeSourceProvenance:
                raise TypeError(
                    "runtime_source_provenance must be RuntimeSourceProvenance or None"
                )
            canonical_provenance = RuntimeSourceProvenance(
                selection_group_id=(self.runtime_source_provenance.selection_group_id),
                request_digest=self.runtime_source_provenance.request_digest,
                result_digests=self.runtime_source_provenance.result_digests,
            )
            if _canonical_semantic_structure_value(
                self.runtime_source_provenance
            ) != _canonical_semantic_structure_value(canonical_provenance):
                raise ValueError(
                    "runtime source provenance differs from canonical payload"
                )
        self.manifest_bundle.validate_complete()
        expected_graph_ids = tuple(
            manifest.graph_instance_id for manifest in self.manifest_bundle.manifests
        )
        actual_graph_ids = tuple(
            binding.graph_instance_id for binding in self.source_bindings.graph_bindings
        )
        if actual_graph_ids != expected_graph_ids:
            raise ValueError(
                "semantic source bindings must cover every manifest graph exactly once"
            )


def _selection_entry_matches_runtime_member(
    selection_entry: SelectionTopologyEntry,
    runtime_entry: ParameterInventoryEntry,
) -> bool:
    member = runtime_entry.member
    if (
        runtime_entry.entry_id != selection_entry.entry_id
        or runtime_entry.graph_instance_id != selection_entry.graph_instance_id
        or member.logical_dtype != selection_entry.logical_dtype
        or member.logical_shape != selection_entry.logical_shape
        or member.logical_axes != selection_entry.logical_axes
    ):
        return False
    if isinstance(member, SemanticTensorFamily):
        return (
            member.pattern == selection_entry.pattern
            and member.domain == selection_entry.domain
        )
    if not isinstance(member, SemanticTensor):
        return False
    if selection_entry.domain.cardinality != 1:
        return False
    pattern = selection_entry.pattern
    address = member.address
    if (
        address.semantic_graph_path != pattern.semantic_graph_path
        or address.model_part != pattern.model_part
        or address.module_kind != pattern.module_kind
        or address.attributes != pattern.attributes
        or address.parameter_role != pattern.parameter_role
    ):
        return False
    coordinates: dict[str, int | str] = {}
    if selection_entry.domain.layer_domain is not None:
        layer_member = selection_entry.domain.layer_domain.members[0]
        coordinates["global_decoder_layer"] = layer_member.global_decoder_layer
        if layer_member.moe_ordinal is not None:
            coordinates["moe_ordinal"] = layer_member.moe_ordinal
    for axis in selection_entry.domain.independent_axes:
        coordinates[axis.name] = axis.members[0]
    suffix = ".".join(
        str(coordinates[segment.axis_name])
        if isinstance(segment, IndexPathSegment)
        else segment.value
        for segment in pattern.path_segments
    )
    return address.semantic_id == f"{pattern.semantic_graph_path}.{suffix}"


def validate_semantic_topology_projection(
    selection_topology: ResolvedSelectionTopology,
    result: SemanticTopologyBuildResult,
    runtime_graph_instance_ids: tuple[str, ...],
) -> SemanticTopologyBuildResult:
    """Prove Phase 2 added only source facts to the frozen Phase 1 topology."""
    if type(selection_topology) is not ResolvedSelectionTopology:
        raise TypeError("selection_topology must be exact ResolvedSelectionTopology")
    selection_topology.validate_complete()
    if type(result) is not SemanticTopologyBuildResult:
        raise TypeError("result must be exact SemanticTopologyBuildResult")
    graph_ids = tuple(runtime_graph_instance_ids)
    if any(type(graph_id) is not str for graph_id in graph_ids):
        raise TypeError("runtime graph IDs must be exact strings")
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("runtime graph IDs contain duplicates")
    selected_graphs = {
        graph.declaration.graph_instance_id: graph
        for graph in selection_topology.graphs
    }
    if any(graph_id not in selected_graphs for graph_id in graph_ids):
        raise ValueError("runtime topology contains an unknown Phase 1 graph")
    canonical_graph_ids = tuple(
        sorted(graph_ids, key=lambda item: (0 if item == "main" else 1, item))
    )
    manifests_by_graph = {
        manifest.graph_instance_id: manifest
        for manifest in result.manifest_bundle.manifests
    }
    if tuple(manifests_by_graph) != canonical_graph_ids:
        raise ValueError(
            "runtime semantic topology graph coverage differs from Phase 1"
        )
    runtime_entries_by_id = {
        entry.entry_id: entry for entry in result.manifest_bundle.inventory.entries
    }
    selected_runtime_entries = {
        entry.entry_id: entry
        for graph_id in canonical_graph_ids
        for entry in selected_graphs[graph_id].entries
    }
    if set(runtime_entries_by_id) != set(selected_runtime_entries):
        raise ValueError(
            "runtime semantic entry coverage differs from Phase 1 selection"
        )
    for graph_id in canonical_graph_ids:
        graph = selected_graphs[graph_id]
        manifest = manifests_by_graph[graph_id]
        if (
            manifest.model_family != graph.model_family
            or manifest.model_revision != graph.resolved_model_revision
            or manifest.lifecycle != graph.declaration.lifecycle
            or manifest.inventory_entry_ids
            != tuple(entry.entry_id for entry in graph.entries)
            or manifest.atomic_groups != graph.atomic_groups
        ):
            raise ValueError(
                f"runtime semantic graph differs from Phase 1 for {graph_id}"
            )
        if manifest.out_of_scope:
            raise ValueError(
                "runtime source classification cannot add out-of-scope selections"
            )
    for entry_id, selected_entry in selected_runtime_entries.items():
        if not _selection_entry_matches_runtime_member(
            selected_entry,
            runtime_entries_by_id[entry_id],
        ):
            raise ValueError(
                f"runtime semantic entry differs from Phase 1 for {entry_id}"
            )
    builtin_names = {
        definition.role_name
        for definition in builtin_role_definitions(
            selection_topology.schema_version, {}
        )
    }
    runtime_entry_ids = frozenset(selected_runtime_entries)
    expected_roles = {
        definition.role_name: (
            definition.predicate,
            tuple(
                entry_id
                for entry_id in definition.expected_domain.inventory_entry_ids
                if entry_id in runtime_entry_ids
            ),
        )
        for definition in selection_topology.role_definitions
        if definition.role_name in builtin_names
        or any(
            entry_id in runtime_entry_ids
            for entry_id in definition.expected_domain.inventory_entry_ids
        )
    }
    actual_roles = {
        definition.role_name: (
            definition.predicate,
            definition.expected_domain.inventory_entry_ids,
        )
        for definition in result.manifest_bundle.role_definitions
    }
    if actual_roles != expected_roles:
        raise ValueError("runtime semantic role registry differs from Phase 1")
    return result


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


def _singleton_region_key(region: SourceRegion) -> tuple[int, ...] | None:
    if any(selection.cardinality != 1 for selection in region.axis_selections):
        return None
    return tuple(selection.spans[0].start for selection in region.axis_selections)


def _selection_envelope(selection: SourceAxisSelection) -> tuple[int, int]:
    return (
        min(span.start for span in selection.spans),
        max(span.stop for span in selection.spans),
    )


def _tagged_spans_are_pairwise_disjoint(
    tagged_spans: Sequence[tuple[SourceIndexSpan, int]],
) -> bool:
    if len(tagged_spans) < 2:
        return True
    steps = {span.step for span, _ in tagged_spans}
    if len(steps) == 1:
        step = next(iter(steps))
        residue_groups: dict[int, list[tuple[SourceIndexSpan, int]]] = {}
        for span, owner_id in tagged_spans:
            residue_groups.setdefault(span.start % step, []).append((span, owner_id))
        groups = residue_groups.values()
        exact_intersection_required = False
    else:
        groups = (list(tagged_spans),)
        exact_intersection_required = True
    for group in groups:
        active_heap: list[tuple[int, int]] = []
        active: dict[int, tuple[SourceIndexSpan, int]] = {}
        for sequence_id, (span, owner_id) in enumerate(
            sorted(group, key=lambda item: (item[0].start, item[0].stop, item[1]))
        ):
            while active_heap and active_heap[0][0] <= span.start:
                _, expired_id = heappop(active_heap)
                active.pop(expired_id, None)
            for active_span, active_owner_id in active.values():
                if active_owner_id == owner_id:
                    continue
                if not exact_intersection_required or _spans_intersect(
                    active_span,
                    span,
                ):
                    return False
            active[sequence_id] = (span, owner_id)
            heappush(active_heap, (span.stop, sequence_id))
    return True


def _axis_selections_are_pairwise_disjoint(
    selections: Sequence[SourceAxisSelection],
) -> bool:
    singleton_coordinates: set[int] = set()
    all_singletons = True
    for selection in selections:
        if selection.cardinality != 1:
            all_singletons = False
            break
        coordinate = selection.spans[0].start
        if coordinate in singleton_coordinates:
            return False
        singleton_coordinates.add(coordinate)
    if all_singletons:
        return True
    return _tagged_spans_are_pairwise_disjoint(
        tuple(
            (span, selection_id)
            for selection_id, selection in enumerate(selections)
            for span in selection.spans
        )
    )


def _axis_envelope_candidate_count(
    regions: Sequence[SourceRegion],
    axis_index: int,
) -> int:
    envelopes = sorted(
        _selection_envelope(region.axis_selections[axis_index]) for region in regions
    )
    active_stops: list[int] = []
    candidate_count = 0
    for start, stop in envelopes:
        while active_stops and active_stops[0] <= start:
            heappop(active_stops)
        candidate_count += len(active_stops)
        heappush(active_stops, stop)
    return candidate_count


def _source_regions_overlap_fallback(
    regions: Sequence[SourceRegion],
    axis_indices: tuple[int, ...],
) -> bool:
    axis_index = min(
        axis_indices,
        key=lambda candidate: (
            _axis_envelope_candidate_count(regions, candidate),
            candidate,
        ),
    )
    ordered = sorted(
        enumerate(regions),
        key=lambda item: (
            *_selection_envelope(item[1].axis_selections[axis_index]),
            item[0],
        ),
    )
    active_heap: list[tuple[int, int]] = []
    active: dict[int, SourceRegion] = {}
    remaining_axes = tuple(index for index in axis_indices if index != axis_index)
    for region_id, region in ordered:
        selection = region.axis_selections[axis_index]
        start, stop = _selection_envelope(selection)
        while active_heap and active_heap[0][0] <= start:
            _, expired_id = heappop(active_heap)
            active.pop(expired_id, None)
        for candidate in active.values():
            if (
                _axis_selection_intersection_cardinality(
                    selection,
                    candidate.axis_selections[axis_index],
                )
                == 0
            ):
                continue
            if all(
                _axis_selection_intersection_cardinality(
                    region.axis_selections[other_axis],
                    candidate.axis_selections[other_axis],
                )
                > 0
                for other_axis in remaining_axes
            ):
                return True
        active[region_id] = region
        heappush(active_heap, (stop, region_id))
    return False


def _source_regions_overlap(
    regions: Sequence[SourceRegion],
    axis_indices: tuple[int, ...],
) -> bool:
    if len(regions) < 2:
        return False
    if not axis_indices:
        return True
    for axis_index in axis_indices:
        groups: dict[SourceAxisSelection, list[SourceRegion]] = {}
        for region in regions:
            groups.setdefault(region.axis_selections[axis_index], []).append(region)
        if len(groups) < 2 or not _axis_selections_are_pairwise_disjoint(tuple(groups)):
            continue
        remaining_axes = tuple(index for index in axis_indices if index != axis_index)
        return any(
            _source_regions_overlap(group, remaining_axes) for group in groups.values()
        )
    return _source_regions_overlap_fallback(regions, axis_indices)


def _validate_source_region_cover(
    complete_region: SourceRegion,
    claims: Sequence[SourceRegion],
    *,
    outside_message: str,
    overlap_message: str,
    gap_message: str,
) -> None:
    if any(
        claim.source_shape != complete_region.source_shape
        or not _source_region_is_subset(claim, complete_region)
        for claim in claims
    ):
        raise ValueError(outside_message)
    singleton_keys: set[tuple[int, ...]] = set()
    all_singletons = True
    for claim in claims:
        singleton_key = _singleton_region_key(claim)
        if singleton_key is None:
            all_singletons = False
            break
        if singleton_key in singleton_keys:
            raise ValueError(overlap_message)
        singleton_keys.add(singleton_key)
    if not all_singletons and _source_regions_overlap(
        claims,
        tuple(range(len(complete_region.source_shape))),
    ):
        raise ValueError(overlap_message)
    if sum(claim.cardinality for claim in claims) != complete_region.cardinality:
        raise ValueError(gap_message)


def _validate_region_partition(
    record: SourceDiscoveryRecord,
    regions: tuple[SourceRegion, ...],
    *,
    tied: bool,
) -> None:
    complete_region = SourceRegion(
        source_shape=record.shape,
        axis_selections=tuple(
            SourceAxisSelection(
                axis_index,
                (SourceIndexSpan(0, extent),),
            )
            for axis_index, extent in enumerate(record.shape)
        ),
    )
    overlap_label = "tied source regions" if tied else "source regions"
    gap_label = "tied source region gap" if tied else "source region gap"
    _validate_source_region_cover(
        complete_region,
        regions,
        outside_message="classification source region shape mismatch",
        overlap_message=f"overlapping {overlap_label} for record {record.record_id}",
        gap_message=f"{gap_label} for record {record.record_id}",
    )


def validate_source_region_partition(
    complete_region: SourceRegion,
    regions: tuple[SourceRegion, ...],
) -> None:
    """Require exact compact no-gap/no-overlap coverage of a source region."""
    if not isinstance(complete_region, SourceRegion):
        raise TypeError("complete_region must be SourceRegion")
    if any(not isinstance(region, SourceRegion) for region in regions):
        raise TypeError("regions must contain SourceRegion records")
    _validate_source_region_cover(
        complete_region,
        regions,
        outside_message="source slice is outside its classification region",
        overlap_message="source slices overlap",
        gap_message="source slices leave a gap",
    )


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


def resolve_output_member_domain(
    target: OutputMemberTarget,
    entry_domain: FamilyIndexDomain,
) -> FamilyIndexDomain:
    """Resolve one validated compact classification target into its exact domain."""
    if not isinstance(target, OutputMemberTarget):
        raise TypeError("target must be OutputMemberTarget")
    if not isinstance(entry_domain, FamilyIndexDomain):
        raise TypeError("entry_domain must be FamilyIndexDomain")
    return _normalized_output_domain(target, entry_domain)


def _compress_source_ordinals(ordinals: set[int]) -> tuple[SourceIndexSpan, ...]:
    values = tuple(sorted(ordinals))
    if not values:
        raise ValueError("source slice axis selection must be non-empty")
    spans: list[SourceIndexSpan] = []
    start = values[0]
    prior = start
    step: int | None = None
    for value in values[1:]:
        difference = value - prior
        if step is None:
            step = difference
        elif difference != step:
            spans.append(
                SourceIndexSpan(
                    start,
                    prior + 1,
                    step or 1,
                )
            )
            start = value
            step = None
        prior = value
    spans.append(
        SourceIndexSpan(
            start,
            prior + 1,
            step or 1,
        )
    )
    return tuple(spans)


def _source_ordinals_for_target_ordinals(
    mapping: SourceToSemanticAxisMapping,
    target_ordinals: frozenset[int],
) -> set[int]:
    source_ordinals: set[int] = set()
    for segment in mapping.segments:
        for target_ordinal in target_ordinals:
            delta = target_ordinal - segment.target_ordinal_start
            if delta < 0 or delta % segment.target_ordinal_step:
                continue
            source_offset = delta // segment.target_ordinal_step
            if source_offset >= segment.source_span.cardinality:
                continue
            source_ordinals.add(
                segment.source_span.start + source_offset * segment.source_span.step
            )
    return source_ordinals


def project_source_region_to_member_domain(
    edge: CanonicalValueClassificationEdge,
    entry_domain: FamilyIndexDomain,
    member_domain: FamilyIndexDomain,
) -> SourceRegion:
    """Invert validated member-axis mappings for one compact precision slice."""
    if not isinstance(edge, CanonicalValueClassificationEdge):
        raise TypeError("edge must be CanonicalValueClassificationEdge")
    if not isinstance(entry_domain, FamilyIndexDomain) or not isinstance(
        member_domain,
        FamilyIndexDomain,
    ):
        raise TypeError("entry and member domains must be FamilyIndexDomain")
    edge_domain = _normalized_output_domain(edge.output, entry_domain)
    if (
        _domain_intersection_cardinality(edge_domain, member_domain)
        != member_domain.cardinality
    ):
        raise ValueError("member slice is outside its classification domain")
    member_axes = {axis.name: axis for axis in member_domain.independent_axes}
    output_axes = {
        axis.name: axis for axis in edge.output.member_domain.independent_axes
    }
    restrictions: dict[int, list[set[int]]] = {}
    for mapping in edge.axis_mappings:
        target = mapping.target
        if isinstance(target, ComponentAxisTarget):
            continue
        if isinstance(target, FamilyIndexAxisTarget):
            output_axis = output_axes[target.axis_name]
            selected_members = frozenset(member_axes[target.axis_name].members)
            target_ordinals = frozenset(
                ordinal
                for ordinal, member in enumerate(output_axis.members)
                if member in selected_members
            )
        else:
            output_layers = edge.output.member_domain.layer_domain
            selected_layers = member_domain.layer_domain
            if output_layers is None or selected_layers is None:
                raise ValueError("mapped layer slice is missing its layer domain")
            selected_members = frozenset(selected_layers.members)
            target_ordinals = frozenset(
                ordinal
                for ordinal, member in enumerate(output_layers.members)
                if member in selected_members
            )
        restrictions.setdefault(mapping.source_axis_index, []).append(
            _source_ordinals_for_target_ordinals(mapping, target_ordinals)
        )
    selections: list[SourceAxisSelection] = []
    for selection in edge.source_region.axis_selections:
        axis_restrictions = restrictions.get(selection.axis_index)
        if axis_restrictions is None:
            selections.append(selection)
            continue
        selected_ordinals = set.intersection(*axis_restrictions)
        selections.append(
            SourceAxisSelection(
                selection.axis_index,
                _compress_source_ordinals(selected_ordinals),
            )
        )
    result = SourceRegion(edge.source_region.source_shape, tuple(selections))
    expected_cardinality, remainder = divmod(
        edge.source_region.cardinality * member_domain.cardinality,
        edge_domain.cardinality,
    )
    if remainder or result.cardinality != expected_cardinality:
        raise ValueError("source slice cardinality differs from its semantic domain")
    return result


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
        raise ValueError("axis mappings must cover every normalized source-view axis")
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
                "each normalized source-view axis must feed one semantic target "
                "except the correlated layer pair"
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


def _validate_record_classification_partition(
    record: SourceDiscoveryRecord,
    edges: tuple[DiscoveryClassificationEdge, ...],
) -> None:
    if record.source_mutability == SourceMutability.ABSENT:
        if len(edges) != 1 or not isinstance(edges[0], AbsentDiscoveryDispositionEdge):
            raise ValueError("absent record requires exactly one absent disposition")
        return
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
        return
    if record.provenance == SourceRecordProvenance.SYNCHRONIZED_REPLICA:
        if not edges or any(
            not isinstance(edge, SynchronizedReplicaAliasClassificationEdge)
            for edge in edges
        ):
            raise ValueError(
                "synchronized-replica record requires only replica alias edges"
            )
        _validate_region_partition(
            record,
            tuple(edge.replica_source_region for edge in edges),
            tied=True,
        )
        return
    if not edges or any(
        not isinstance(edge, CanonicalValueClassificationEdge) for edge in edges
    ):
        raise ValueError("present canonical record requires consuming canonical edges")
    _validate_region_partition(
        record,
        tuple(edge.source_region for edge in edges),
        tied=False,
    )


@dataclass(frozen=True, slots=True)
class _CanonicalNativeOwnerAuthority:
    provenance: SourceRecordProvenance
    provenance_evidence: EvidenceSource
    source_mutability: SourceMutability
    mutability_evidence: EvidenceSource
    canonical_owner_family: OwnerFamilyReference


def validate_semantic_graph_build_fragment(
    schema_version: int,
    graph_input: GraphTopologyInput,
    source_records: tuple[SourceDiscoveryRecord, ...],
    fragment: SemanticGraphBuildFragment,
) -> None:
    """Validate exact normalized-source-view-to-semantic compact accounting."""
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
    records_by_id: dict[str, SourceDiscoveryRecord] = {}
    for record in source_records:
        if record.graph_instance_id != graph_id:
            raise ValueError("source record belongs to another graph")
        if record.record_id in records_by_id:
            raise ValueError("duplicate source discovery record ID")
        records_by_id[record.record_id] = record
    edge_groups = iter(
        groupby(
            fragment.classification_edges,
            key=lambda edge: _require_text(
                edge.record_id,
                "classification edge record_id",
            ),
        )
    )
    current_group = next(edge_groups, None)
    for record_id in sorted(records_by_id):
        if current_group is not None and current_group[0] < record_id:
            raise ValueError("classification edge references unknown source record")
        if current_group is None or current_group[0] != record_id:
            edges: tuple[DiscoveryClassificationEdge, ...] = ()
        else:
            edges = tuple(current_group[1])
            current_group = next(edge_groups, None)
        _validate_record_classification_partition(records_by_id[record_id], edges)
    if current_group is not None:
        raise ValueError("classification edge references unknown source record")

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
    canonical_authorities_by_native_owner: dict[
        str,
        _CanonicalNativeOwnerAuthority,
    ] = {}
    local_tied_direct_requirements = {
        (
            edge.canonical_value_entry_id,
            records_by_id[edge.record_id].source_native_owner_id,
        )
        for edge in fragment.classification_edges
        if isinstance(edge, TiedAliasClassificationEdge)
        and edge.canonical_owner_family.graph_instance_id == graph_id
    }
    resolved_local_tied_direct_requirements: set[tuple[str, str | None]] = set()
    components_by_entry_id = {
        entry.entry_id: {
            component.role: component for component in entry.member.format.components
        }
        for entry in fragment.inventory_entries
    }
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
        component = components_by_entry_id[entry.entry_id].get(edge.component_role)
        if component is None:
            raise ValueError("classification edge claims an unknown format component")
        if record.dtype.value != component.dtype:
            raise ValueError(
                "normalized source view dtype does not match claimed format component"
            )
        if record.numeric_encoding != component.encoding:
            raise ValueError(
                "normalized source view encoding does not match claimed format "
                "component"
            )
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
            if entry.value_provenance == ValueProvenance.CANONICAL_ALIAS:
                raise ValueError(
                    "canonical edge cannot justify a canonical alias entry"
                )
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
                raise ValueError(
                    "value provenance is not backed by normalized source discovery"
                )
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
            authority = _CanonicalNativeOwnerAuthority(
                provenance=record.provenance,
                provenance_evidence=record.provenance_evidence,
                source_mutability=record.source_mutability,
                mutability_evidence=record.mutability_evidence,
                canonical_owner_family=edge.canonical_owner_family,
            )
            prior_authority = canonical_authorities_by_native_owner.setdefault(
                record.source_native_owner_id,
                authority,
            )
            if (
                prior_authority.provenance,
                prior_authority.provenance_evidence,
                prior_authority.source_mutability,
                prior_authority.mutability_evidence,
            ) != (
                authority.provenance,
                authority.provenance_evidence,
                authority.source_mutability,
                authority.mutability_evidence,
            ):
                raise ValueError(
                    "normalized source-view records for native owner "
                    f"{record.source_native_owner_id} disagree on authority"
                )
            if (
                prior_authority.canonical_owner_family
                != authority.canonical_owner_family
            ):
                raise ValueError("one native owner must resolve to one canonical owner")
            direct_native_owner = (entry.entry_id, record.source_native_owner_id)
            if direct_native_owner in local_tied_direct_requirements:
                resolved_local_tied_direct_requirements.add(direct_native_owner)
            justified_owners.add(edge.canonical_owner_family)
            _validate_axis_mappings(edge, entry, normalized, component_axes)
        else:
            alias_kind = (
                "tied" if isinstance(edge, TiedAliasClassificationEdge) else "replica"
            )
            if entry.value_provenance != ValueProvenance.CANONICAL_ALIAS:
                raise ValueError(
                    f"{alias_kind} edge requires a canonical-alias semantic entry"
                )
            binding = entry.member.ownership.binding
            if edge.canonical_owner_family != binding.canonical_owner_family:
                raise ValueError(
                    f"{alias_kind} edge owner differs from alias entry owner"
                )
            if edge.canonical_value_entry_id != binding.canonical_value_entry_id:
                raise ValueError(
                    f"{alias_kind} edge target differs from alias entry target"
                )
            if len(edge.alias_to_canonical_axes) != len(
                binding.member_to_value_axes
            ) or set(edge.alias_to_canonical_axes) != set(binding.member_to_value_axes):
                raise ValueError(
                    f"{alias_kind} edge projection differs from alias binding"
                )
            expected_cardinality = normalized.cardinality * prod(
                extent for _, extent in component_axes
            )
            alias_region = (
                edge.aliased_source_region
                if isinstance(edge, TiedAliasClassificationEdge)
                else edge.replica_source_region
            )
            if alias_region.cardinality != expected_cardinality:
                raise ValueError(
                    f"{alias_kind} source cardinality does not match alias component domain"
                )
    for authority in canonical_authorities_by_native_owner.values():
        source_owner = owners_by_reference[authority.canonical_owner_family]
        if (
            source_owner.source_mutability != authority.source_mutability
            or source_owner.mutability_evidence_source != authority.mutability_evidence
        ):
            raise ValueError(
                "owner mutability evidence differs from normalized source "
                "discovery authority"
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
        if direct_entry.value_provenance == ValueProvenance.CANONICAL_ALIAS:
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
        if (
            direct_entry.entry_id,
            record.source_native_owner_id,
        ) not in resolved_local_tied_direct_requirements:
            raise ValueError("tied native owner differs from direct target")
        assert record.source_native_owner_id is not None
        canonical_authority = canonical_authorities_by_native_owner[
            record.source_native_owner_id
        ]
        if (
            record.source_mutability != canonical_authority.source_mutability
            or record.mutability_evidence != canonical_authority.mutability_evidence
        ):
            raise ValueError(
                "tied mutability evidence differs from canonical source authority"
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
                (
                    CanonicalValueClassificationEdge,
                    TiedAliasClassificationEdge,
                    SynchronizedReplicaAliasClassificationEdge,
                ),
            )
            for edge in fragment.classification_edges
        )
    ):
        raise ValueError(
            "source-served graph requires a present canonical owner or source alias"
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


def select_model_topology_adapter_by_id(
    model_config: Mapping[str, object],
    adapter_id: str,
    *,
    adapters: Sequence[ModelTopologyAdapter] | None = None,
) -> ModelTopologyAdapter:
    """Resolve the exact Phase 1-selected adapter without reselecting a family."""
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")
    required_id = _require_text(adapter_id, "required adapter_id")
    candidates = tuple(_default_adapters() if adapters is None else adapters)
    candidate_ids = tuple(
        _require_text(item.adapter_id, "adapter_id") for item in candidates
    )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("topology adapter IDs must be unique")
    matching = tuple(
        adapter for adapter in candidates if adapter.adapter_id == required_id
    )
    if len(matching) != 1:
        raise ValueError(f"required topology adapter is unavailable: {required_id}")
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
    records_by_key: Mapping[_SourceRecordKey, SourceDiscoveryRecord],
) -> None:
    owner_by_native_key: dict[_SourceNativeOwnerKey, OwnerFamilyReference] = {}
    record_by_native_key: dict[_SourceNativeOwnerKey, SourceDiscoveryRecord] = {}
    for fragment in fragments:
        for edge in fragment.classification_edges:
            if not isinstance(edge, CanonicalValueClassificationEdge):
                continue
            record = records_by_key[(fragment.graph_instance_id, edge.record_id)]
            assert record.source_native_owner_id is not None
            native_owner_id = record.source_native_owner_id
            native_owner_key = (fragment.graph_instance_id, native_owner_id)
            prior_record = record_by_native_key.setdefault(native_owner_key, record)
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
            prior_owner = owner_by_native_key.setdefault(
                native_owner_key,
                edge.canonical_owner_family,
            )
            if prior_owner != edge.canonical_owner_family:
                raise ValueError(
                    f"canonical native owner {native_owner_id} resolves to multiple "
                    "owners"
                )


def _domain_axis_members(
    domain: FamilyIndexDomain,
    axis_name: str,
) -> frozenset[int | str]:
    if domain.layer_domain is not None and axis_name in domain.layer_domain.axis_names:
        return frozenset(
            _layer_axis_value(member, axis_name)
            for member in domain.layer_domain.members
        )
    for axis in domain.independent_axes:
        if axis.name == axis_name:
            return frozenset(axis.members)
    raise ValueError(f"unknown projected alias axis: {axis_name}")


@dataclass(slots=True)
class _AliasProjectionParentIndex:
    independent_members: dict[str, frozenset[int | str]]
    layer_postings: dict[str, dict[int, tuple[LayerMember, ...]]]


type _AliasProjectionParentIndexCache = dict[
    int,
    tuple[FamilyIndexDomain, _AliasProjectionParentIndex],
]


def _alias_projection_parent_index(
    domain: FamilyIndexDomain,
    cache: _AliasProjectionParentIndexCache,
) -> _AliasProjectionParentIndex:
    cached = cache.get(id(domain))
    if cached is not None and cached[0] is domain:
        return cached[1]
    independent_members = {
        axis.name: frozenset(axis.members) for axis in domain.independent_axes
    }
    layer_posting_lists: dict[str, dict[int, list[LayerMember]]] = {}
    if domain.layer_domain is not None:
        layer_posting_lists = {
            axis_name: {} for axis_name in domain.layer_domain.axis_names
        }
        for member in domain.layer_domain.members:
            for axis_name in domain.layer_domain.axis_names:
                value = _layer_axis_value(member, axis_name)
                layer_posting_lists[axis_name].setdefault(value, []).append(member)
    index = _AliasProjectionParentIndex(
        independent_members=independent_members,
        layer_postings={
            axis_name: {value: tuple(members) for value, members in postings.items()}
            for axis_name, postings in layer_posting_lists.items()
        },
    )
    cache[id(domain)] = (domain, index)
    return index


def _project_alias_domain(
    alias_domain: FamilyIndexDomain,
    canonical_complete_domain: FamilyIndexDomain,
    projections: tuple[AxisProjection, ...],
    *,
    parent_index_cache: _AliasProjectionParentIndexCache | None = None,
) -> FamilyIndexDomain:
    source_by_target = {
        projection.owner_axis: projection.member_axis for projection in projections
    }
    if (
        len(source_by_target) != len(projections)
        or set(source_by_target) != set(canonical_complete_domain.axis_names)
        or set(source_by_target.values()) != set(alias_domain.axis_names)
    ):
        raise ValueError("source alias projection is not exact and bijective")
    allowed_by_target = {
        target_axis: _domain_axis_members(alias_domain, source_axis)
        for target_axis, source_axis in source_by_target.items()
    }
    cache = {} if parent_index_cache is None else parent_index_cache
    parent_index = _alias_projection_parent_index(
        canonical_complete_domain,
        cache,
    )
    canonical_layer = canonical_complete_domain.layer_domain
    projected_layer = None
    if canonical_layer is not None:
        layer_values_by_axis = {
            axis_name: tuple(
                value
                for value in allowed_by_target[axis_name]
                if isinstance(value, int)
            )
            for axis_name in canonical_layer.axis_names
        }
        if any(
            len(layer_values_by_axis[axis_name]) != len(allowed_by_target[axis_name])
            for axis_name in canonical_layer.axis_names
        ):
            raise ValueError(
                "source alias projected layer coordinates must be integers"
            )
        empty_layer_members: tuple[LayerMember, ...] = ()
        rarest_axis = min(
            canonical_layer.axis_names,
            key=lambda axis_name: (
                sum(
                    len(
                        parent_index.layer_postings[axis_name].get(
                            value,
                            empty_layer_members,
                        )
                    )
                    for value in layer_values_by_axis[axis_name]
                ),
                axis_name,
            ),
        )
        candidate_members = {
            member
            for value in layer_values_by_axis[rarest_axis]
            for member in parent_index.layer_postings[rarest_axis].get(
                value,
                empty_layer_members,
            )
        }
        projected_layer = LayerDomain(
            tuple(
                member
                for member in candidate_members
                if all(
                    _layer_axis_value(member, axis_name) in allowed_by_target[axis_name]
                    for axis_name in canonical_layer.axis_names
                )
            )
        )
    projected_axes = tuple(
        AxisDomain(
            axis.name,
            tuple(allowed_by_target[axis.name]),
        )
        for axis in canonical_complete_domain.independent_axes
        if allowed_by_target[axis.name].issubset(
            parent_index.independent_members[axis.name]
        )
    )
    if len(projected_axes) != len(canonical_complete_domain.independent_axes):
        raise ValueError("source alias projected domain is outside canonical domain")
    projected = FamilyIndexDomain(projected_layer, projected_axes)
    if (
        projected.cardinality != alias_domain.cardinality
        or not _target_domain_is_subset_of_projected_source(
            alias_domain,
            projected,
            projections,
        )
    ):
        raise ValueError("source alias projected domain is not compactly representable")
    return projected


def _normalize_source_alias_contracts(
    fragments: tuple[SemanticGraphBuildFragment, ...],
    records_by_key: Mapping[_SourceRecordKey, SourceDiscoveryRecord],
) -> tuple[SourceAliasContract, ...]:
    alias_edges = tuple(
        (fragment.graph_instance_id, edge)
        for fragment in fragments
        for edge in fragment.classification_edges
        if isinstance(
            edge,
            (
                TiedAliasClassificationEdge,
                SynchronizedReplicaAliasClassificationEdge,
            ),
        )
    )
    if not alias_edges:
        return ()
    has_tied_aliases = any(
        isinstance(edge, TiedAliasClassificationEdge) for _, edge in alias_edges
    )
    has_replica_aliases = any(
        isinstance(edge, SynchronizedReplicaAliasClassificationEdge)
        for _, edge in alias_edges
    )
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
        tuple[str, ComponentRole, str],
        list[tuple[CanonicalValueClassificationEdge, SourceDiscoveryRecord]],
    ] = {}
    canonical_backing_by_exact_region: dict[
        tuple[_SourceRecordKey, str, ComponentRole, SourceRegion],
        tuple[CanonicalValueClassificationEdge, SourceDiscoveryRecord],
    ] = {}
    for fragment in fragments:
        for edge in fragment.classification_edges:
            if not isinstance(edge, CanonicalValueClassificationEdge):
                continue
            record_key = (fragment.graph_instance_id, edge.record_id)
            record = records_by_key[record_key]
            if has_tied_aliases:
                assert record.source_native_owner_id is not None
                canonical_backings_by_component.setdefault(
                    (
                        edge.output.inventory_entry_id,
                        edge.component_role,
                        record.source_native_owner_id,
                    ),
                    [],
                ).append((edge, record))
            if has_replica_aliases:
                exact_key = (
                    record_key,
                    edge.output.inventory_entry_id,
                    edge.component_role,
                    edge.source_region,
                )
                if exact_key in canonical_backing_by_exact_region:
                    raise ValueError("duplicate exact canonical source backing")
                canonical_backing_by_exact_region[exact_key] = (edge, record)

    if has_replica_aliases:
        replica_native_owner_keys: set[_SourceNativeOwnerKey] = set()
        for graph_instance_id, edge in alias_edges:
            if not isinstance(edge, SynchronizedReplicaAliasClassificationEdge):
                continue
            replica_record = records_by_key[(graph_instance_id, edge.record_id)]
            assert replica_record.source_native_owner_id is not None
            replica_native_owner_keys.add(
                (graph_instance_id, replica_record.source_native_owner_id)
            )
        for record in records_by_key.values():
            native_owner_id = record.source_native_owner_id
            if (
                native_owner_id is not None
                and (record.graph_instance_id, native_owner_id)
                in replica_native_owner_keys
                and record.provenance != SourceRecordProvenance.SYNCHRONIZED_REPLICA
            ):
                raise ValueError(
                    f"replica native owner {native_owner_id} cannot also be canonical "
                    "or tied authority"
                )

    if has_tied_aliases and not canonical_backings_by_component:
        if any(
            isinstance(edge, TiedAliasClassificationEdge) for _, edge in alias_edges
        ):
            raise ValueError("cross-graph tied native owner differs from direct target")

    contracts: list[SourceAliasContract] = []
    parent_index_cache: _AliasProjectionParentIndexCache = {}
    replica_relation_by_native_owner: dict[
        _SourceNativeOwnerKey,
        tuple[
            OwnerFamilyReference,
            str,
            SourceSynchronizationBoundary,
            EvidenceSource,
        ],
    ] = {}

    for graph_instance_id, edge in alias_edges:
        alias_entry = entries_by_id[edge.alias_output.inventory_entry_id]
        direct_entry = entries_by_id.get(edge.canonical_value_entry_id)
        if direct_entry is None:
            raise ValueError("source alias direct target is missing")
        if direct_entry.value_provenance == ValueProvenance.CANONICAL_ALIAS:
            raise ValueError("source alias target must be direct")
        direct_binding = direct_entry.member.ownership.binding
        if direct_binding.canonical_value_entry_id != direct_entry.entry_id:
            raise ValueError("source alias target must bind directly")
        if direct_binding.canonical_owner_family != edge.canonical_owner_family:
            raise ValueError("source alias canonical owner mismatch")
        if (
            alias_entry.member.logical_dtype != direct_entry.member.logical_dtype
            or alias_entry.member.logical_shape != direct_entry.member.logical_shape
            or alias_entry.member.logical_axes != direct_entry.member.logical_axes
            or alias_entry.member.format != direct_entry.member.format
        ):
            raise ValueError("source alias target is incompatible")

        owner = owners_by_reference.get(edge.canonical_owner_family)
        if owner is None:
            raise ValueError("source alias canonical owner is missing")
        alias_domain = _normalized_output_domain(
            edge.alias_output,
            alias_entry.member.ownership.binding.member_domain,
        )
        canonical_domain = _project_alias_domain(
            alias_domain,
            direct_binding.member_domain,
            edge.alias_to_canonical_axes,
            parent_index_cache=parent_index_cache,
        )

        if isinstance(edge, SynchronizedReplicaAliasClassificationEdge):
            replica_record_key = (graph_instance_id, edge.record_id)
            canonical_record_key = (
                edge.canonical_owner_family.graph_instance_id,
                edge.canonical_record_id,
            )
            replica_record = records_by_key[replica_record_key]
            canonical_record = records_by_key.get(canonical_record_key)
            if canonical_record is None:
                raise ValueError("replica canonical source record is missing")
            if replica_record_key == canonical_record_key:
                raise ValueError("replica and canonical record IDs must differ")
            if (
                replica_record.graph_instance_id,
                replica_record.source_native_owner_id,
            ) == (
                canonical_record.graph_instance_id,
                canonical_record.source_native_owner_id,
            ):
                raise ValueError("replica and canonical native owner IDs must differ")
            if (
                replica_record.dtype != canonical_record.dtype
                or replica_record.shape != canonical_record.shape
                or replica_record.numeric_encoding != canonical_record.numeric_encoding
            ):
                raise ValueError(
                    "replica normalized source-view dtype, shape, and encoding "
                    "must match canonical"
                )
            if edge.replica_source_region != edge.canonical_source_region:
                raise ValueError(
                    "replica source region must exactly match canonical region"
                )
            exact_key = (
                canonical_record_key,
                direct_entry.entry_id,
                edge.component_role,
                edge.canonical_source_region,
            )
            canonical_backing = canonical_backing_by_exact_region.get(exact_key)
            if canonical_backing is None:
                raise ValueError(
                    "replica lacks an exact consuming canonical source backing"
                )
            canonical_edge, canonical_record = canonical_backing
            if (
                canonical_record.provenance != SourceRecordProvenance.TRAINING_RUNTIME
                or direct_entry.value_provenance != ValueProvenance.TRAINING_PARAMETER
            ):
                raise ValueError(
                    "synchronized replica must target training-runtime parameter "
                    "authority"
                )
            if canonical_edge.canonical_owner_family != edge.canonical_owner_family:
                raise ValueError("replica canonical edge owner mismatch")
            canonical_claim = _normalized_output_domain(
                canonical_edge.output,
                direct_binding.member_domain,
            )
            if canonical_claim != canonical_domain:
                raise ValueError(
                    "replica semantic subdomain differs from canonical claim"
                )
            if (
                replica_record.source_mutability != owner.source_mutability
                or replica_record.mutability_evidence
                != owner.mutability_evidence_source
            ):
                raise ValueError(
                    "replica mutability evidence differs from canonical owner"
                )
            if (
                replica_record.provenance_evidence
                != edge.synchronization.evidence_source
            ):
                raise ValueError(
                    "replica source provenance evidence differs from synchronization"
                )
            assert replica_record.source_native_owner_id is not None
            relation = (
                edge.canonical_owner_family,
                edge.synchronization.replica_group_id,
                edge.synchronization.boundary,
                edge.synchronization.evidence_source,
            )
            prior_relation = replica_relation_by_native_owner.setdefault(
                (
                    replica_record.graph_instance_id,
                    replica_record.source_native_owner_id,
                ),
                relation,
            )
            if prior_relation != relation:
                raise ValueError("replica native owner has conflicting source relation")
            contracts.append(
                SynchronizedReplicaSourceAliasContract(
                    alias_entry_id=alias_entry.entry_id,
                    canonical_value_entry_id=direct_entry.entry_id,
                    canonical_owner_family=edge.canonical_owner_family,
                    component_role=edge.component_role,
                    alias_domain=alias_domain,
                    canonical_domain=canonical_domain,
                    alias_to_canonical_axes=edge.alias_to_canonical_axes,
                    synchronization=edge.synchronization,
                )
            )
            continue

        tied_record = records_by_key[(graph_instance_id, edge.record_id)]
        canonical_backings = canonical_backings_by_component.get(
            (
                direct_entry.entry_id,
                edge.component_role,
                tied_record.source_native_owner_id or "",
            ),
            [],
        )
        if not canonical_backings:
            raise ValueError("cross-graph tied native owner differs from direct target")
        corresponding_backings: list[
            tuple[CanonicalValueClassificationEdge, SourceDiscoveryRecord]
        ] = []
        for canonical_edge, record in canonical_backings:
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
            canonical_edge.source_region for canonical_edge, _ in corresponding_backings
        )
        if not corresponding_regions:
            raise ValueError(
                "cross-graph tied alias subdomain lacks exact canonical coverage"
            )
        _validate_source_region_cover(
            edge.aliased_source_region,
            corresponding_regions,
            outside_message=(
                "cross-graph tied alias subdomain has unaligned source regions"
            ),
            overlap_message=(
                "cross-graph tied alias subdomain lacks exact canonical coverage"
            ),
            gap_message=(
                "cross-graph tied alias subdomain lacks exact canonical coverage"
            ),
        )
        canonical_claims = tuple(
            _normalized_output_domain(
                canonical_edge.output,
                direct_entry.member.ownership.binding.member_domain,
            )
            for canonical_edge, _ in corresponding_backings
        )
        try:
            if any(
                _domain_intersection_cardinality(canonical_domain, claim)
                != claim.cardinality
                for claim in canonical_claims
            ):
                raise ValueError("canonical claim is outside projected subdomain")
            _validate_output_domain_partition(canonical_domain, canonical_claims)
        except ValueError as error:
            raise ValueError(
                "cross-graph tied alias subdomain differs from canonical claims"
            ) from error
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
        contracts.append(
            IdenticalStorageSourceAliasContract(
                alias_entry_id=alias_entry.entry_id,
                canonical_value_entry_id=direct_entry.entry_id,
                canonical_owner_family=edge.canonical_owner_family,
                component_role=edge.component_role,
                alias_domain=alias_domain,
                canonical_domain=canonical_domain,
                alias_to_canonical_axes=edge.alias_to_canonical_axes,
                storage_identity_evidence=tied_record.provenance_evidence,
            )
        )
    return tuple(contracts)


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


def _validate_native_component_sharing(
    source_discovery: SourceDiscoveryInventory,
    fragments: tuple[SemanticGraphBuildFragment, ...],
    records_by_key: Mapping[_SourceRecordKey, SourceDiscoveryRecord],
) -> None:
    shared_record_sets: list[set[_SourceRecordKey]] = []
    for partition in source_discovery.partitions:
        first_record_by_component: dict[str, _SourceRecordKey] = {}
        shared_records_by_component: dict[str, set[_SourceRecordKey]] = {}
        for realization in partition.storage_realizations.realizations:
            if not isinstance(realization, SourceStorageRealization):
                continue
            record_key = (
                partition.graph_instance_id,
                realization.output_record_id,
            )
            for component in realization.components:
                prior_record_key = first_record_by_component.setdefault(
                    component.native_component_id,
                    record_key,
                )
                if prior_record_key == record_key:
                    continue
                shared_records_by_component.setdefault(
                    component.native_component_id,
                    {prior_record_key},
                ).add(record_key)
        shared_record_sets.extend(shared_records_by_component.values())
    if not shared_record_sets:
        return

    canonical_records_by_target: dict[
        tuple[str, ComponentRole, str | None],
        list[tuple[_SourceRecordKey, SourceRegion]],
    ] = {}
    tied_edges: list[tuple[str, TiedAliasClassificationEdge]] = []
    for fragment in fragments:
        for edge in fragment.classification_edges:
            if isinstance(edge, CanonicalValueClassificationEdge):
                record_key = (fragment.graph_instance_id, edge.record_id)
                record = records_by_key[record_key]
                canonical_records_by_target.setdefault(
                    (
                        edge.output.inventory_entry_id,
                        edge.component_role,
                        record.source_native_owner_id,
                    ),
                    [],
                ).append((record_key, edge.source_region))
            elif isinstance(edge, TiedAliasClassificationEdge):
                tied_edges.append((fragment.graph_instance_id, edge))

    relation_parent: dict[_SourceRecordKey, _SourceRecordKey] = {}

    def find(record_key: _SourceRecordKey) -> _SourceRecordKey:
        parent = relation_parent.get(record_key)
        if parent is None:
            return record_key
        while parent != record_key:
            grandparent = relation_parent[parent]
            relation_parent[record_key] = grandparent
            record_key = parent
            parent = grandparent
        return record_key

    def union(
        left_record_key: _SourceRecordKey,
        right_record_key: _SourceRecordKey,
    ) -> None:
        left_root = find(left_record_key)
        right_root = find(right_record_key)
        if left_root == right_root:
            return
        canonical_root, alias_root = sorted((left_root, right_root))
        relation_parent.setdefault(canonical_root, canonical_root)
        relation_parent[alias_root] = canonical_root

    for graph_instance_id, edge in tied_edges:
        tied_record_key = (graph_instance_id, edge.record_id)
        tied_record = records_by_key[tied_record_key]
        candidates = canonical_records_by_target.get(
            (
                edge.canonical_value_entry_id,
                edge.component_role,
                tied_record.source_native_owner_id,
            ),
            (),
        )
        for canonical_record_key, canonical_region in candidates:
            if not _regions_intersect(edge.aliased_source_region, canonical_region):
                continue
            union(tied_record_key, canonical_record_key)

    for shared_record_keys in shared_record_sets:
        if any(
            records_by_key[record_key].provenance
            is SourceRecordProvenance.SYNCHRONIZED_REPLICA
            for record_key in shared_record_keys
        ):
            raise ValueError(
                "synchronized replicas require distinct native component identities"
            )
        relation_roots = {find(record_key) for record_key in shared_record_keys}
        if len(relation_roots) != 1:
            raise ValueError(
                "shared native component lacks a corresponding "
                "identical-storage relation"
            )


def _build_semantic_topology_result_from_validated_inventory(
    schema_version: int,
    graph_inputs: tuple[GraphTopologyInput, ...],
    source_discovery: SourceDiscoveryInventory,
    *,
    required_adapter_ids_by_graph: Mapping[str, str] | None = None,
    runtime_source_provenance: RuntimeSourceProvenance | None = None,
) -> SemanticTopologyBuildResult:
    inputs = graph_inputs
    partitions_by_graph = {
        partition.graph_instance_id: partition
        for partition in source_discovery.partitions
    }
    adapters = _default_adapters()
    required_adapter_ids: dict[str, str] | None = None
    if required_adapter_ids_by_graph is not None:
        if not isinstance(required_adapter_ids_by_graph, Mapping):
            raise TypeError("required_adapter_ids_by_graph must be a mapping")
        required_adapter_ids = {}
        for graph_id, adapter_id in tuple(required_adapter_ids_by_graph.items()):
            required_adapter_ids[_require_text(graph_id, "adapter graph ID")] = (
                _require_text(adapter_id, "required adapter ID")
            )
        expected_graph_ids = {
            graph_input.declaration.graph_instance_id for graph_input in inputs
        }
        if set(required_adapter_ids) != expected_graph_ids:
            raise ValueError("required adapter IDs must cover every graph exactly once")
    fragments: list[SemanticGraphBuildFragment] = []
    for graph_input in sorted(inputs, key=_graph_input_sort_key):
        graph_id = graph_input.declaration.graph_instance_id
        records = partitions_by_graph[graph_id].records
        adapter = (
            select_model_topology_adapter(
                graph_input.model_config,
                adapters=adapters,
            )
            if required_adapter_ids is None
            else select_model_topology_adapter_by_id(
                graph_input.model_config,
                required_adapter_ids[graph_id],
                adapters=adapters,
            )
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
    records_by_key = {
        (record.graph_instance_id, record.record_id): record
        for record in source_discovery.records
    }
    _validate_global_canonical_native_authority(
        canonical_fragments,
        records_by_key,
    )
    source_alias_contracts = _normalize_source_alias_contracts(
        canonical_fragments,
        records_by_key,
    )
    _validate_native_component_sharing(
        source_discovery,
        canonical_fragments,
        records_by_key,
    )

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
        source_alias_contracts=source_alias_contracts,
    )
    bundle.validate_complete()
    source_binding_graphs: list[GraphSemanticSourceBindings] = []
    fragments_by_graph = {
        fragment.graph_instance_id: fragment for fragment in canonical_fragments
    }
    for graph_input in sorted(inputs, key=_graph_input_sort_key):
        graph_id = graph_input.declaration.graph_instance_id
        partition = partitions_by_graph[graph_id]
        records_by_id = {record.record_id: record for record in partition.records}
        realizations_by_record: dict[str, list[SourceRealization]] = {}
        for realization in partition.storage_realizations.realizations:
            realizations_by_record.setdefault(
                realization.output_record_id,
                [],
            ).append(realization)
        canonical_bindings = tuple(
            CanonicalSourceSemanticBinding(
                graph_instance_id=graph_id,
                classification_edge=edge,
                source_record=records_by_id[edge.record_id],
                source_realizations=tuple(realizations_by_record[edge.record_id]),
            )
            for edge in fragments_by_graph[graph_id].classification_edges
            if isinstance(edge, CanonicalValueClassificationEdge)
        )
        source_binding_graphs.append(
            GraphSemanticSourceBindings(
                graph_instance_id=graph_id,
                normalizer_manifest=(
                    partition.storage_realizations.normalizer_manifest
                ),
                canonical_bindings=canonical_bindings,
            )
        )
    return SemanticTopologyBuildResult(
        manifest_bundle=bundle,
        source_bindings=SemanticSourceBindingInventory(tuple(source_binding_graphs)),
        runtime_source_provenance=runtime_source_provenance,
    )


def build_semantic_topology_result(
    schema_version: int,
    graph_inputs: Sequence[GraphTopologyInput],
    source_discovery: SourceDiscoveryInventory,
    expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
    *,
    required_adapter_ids_by_graph: Mapping[str, str] | None = None,
) -> SemanticTopologyBuildResult:
    """Classify graphs and retain the validated source-to-semantic evidence."""
    _require_int(schema_version, "semantic schema_version", minimum=1)
    inputs = _snapshot_sequence(graph_inputs, "graph inputs")
    validate_discovery_inventory(
        inputs,
        source_discovery,
        expected_contributors_by_graph,
    )
    return _build_semantic_topology_result_from_validated_inventory(
        schema_version,
        inputs,
        source_discovery,
        required_adapter_ids_by_graph=required_adapter_ids_by_graph,
    )


def build_runtime_semantic_topology_result(
    schema_version: int,
    runtime_requests: Sequence[RuntimeGraphSourceRequest],
    source_discovery: SourceDiscoveryInventory,
    expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
    *,
    required_adapter_ids_by_graph: Mapping[str, str],
) -> SemanticTopologyBuildResult:
    """Classify one fully validated Phase 2 runtime-source inventory."""
    _require_int(schema_version, "semantic schema_version", minimum=1)
    requests = _snapshot_sequence(runtime_requests, "runtime requests")
    if any(type(request) is not RuntimeGraphSourceRequest for request in requests):
        raise TypeError(
            "runtime_requests must contain exact RuntimeGraphSourceRequest records"
        )
    validate_runtime_discovery_inventory(
        requests,
        source_discovery,
        expected_contributors_by_graph,
    )
    return classify_validated_runtime_semantic_topology(
        schema_version,
        requests,
        source_discovery,
        required_adapter_ids_by_graph=required_adapter_ids_by_graph,
    )


def classify_validated_runtime_semantic_topology(
    schema_version: int,
    runtime_requests: Sequence[RuntimeGraphSourceRequest],
    validated_source_discovery: SourceDiscoveryInventory,
    *,
    required_adapter_ids_by_graph: Mapping[str, str],
    runtime_source_provenance: RuntimeSourceProvenance | None = None,
) -> SemanticTopologyBuildResult:
    """Classify a runtime inventory already validated by the Phase 2 binder."""
    _require_int(schema_version, "semantic schema_version", minimum=1)
    requests = _snapshot_sequence(runtime_requests, "runtime requests")
    if any(type(request) is not RuntimeGraphSourceRequest for request in requests):
        raise TypeError(
            "runtime_requests must contain exact RuntimeGraphSourceRequest records"
        )
    graph_inputs = tuple(
        GraphTopologyInput(
            declaration=request.declaration,
            model_config=request.model_config,
            resolved_model_revision=request.resolved_model_revision,
            source_producer_fingerprint=request.source_producer_fingerprint,
            expected_contributor_authority=request.expected_contributor_authority,
            source_identity=request.source_identity,
            artifact_identity=request.artifact_identity,
        )
        for request in requests
    )
    return _build_semantic_topology_result_from_validated_inventory(
        schema_version,
        graph_inputs,
        validated_source_discovery,
        required_adapter_ids_by_graph=required_adapter_ids_by_graph,
        runtime_source_provenance=runtime_source_provenance,
    )


def build_semantic_manifest_bundle(
    schema_version: int,
    graph_inputs: Sequence[GraphTopologyInput],
    source_discovery: SourceDiscoveryInventory,
    expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
) -> SemanticManifestBundle:
    """Compatibility wrapper returning only the validated semantic manifest."""
    return build_semantic_topology_result(
        schema_version,
        graph_inputs,
        source_discovery,
        expected_contributors_by_graph,
    ).manifest_bundle
