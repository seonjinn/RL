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

"""Compile semantic precision policy into immutable endpoint intent plans."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import Literal

from nemo_rl.precision_policy.config import (
    AdvancedMatchConfig,
    AtomicConflictMode,
    LayerSelectorConfig,
    PrecisionName,
    PrecisionPolicyConfig,
    PrecisionScopeConfig,
    SemanticAddressSelectorConfig,
)
from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    MXFP8_FORMAT,
    AtomicGroup,
    AxisDomain,
    AxisProjection,
    EvidenceSource,
    FamilyIndexDomain,
    FormatDescriptor,
    GraphLifecycle,
    IdenticalStorageSourceAliasContract,
    ImmutableAuxiliaryEvidence,
    IndexPathSegment,
    LayerDomain,
    LayerMember,
    LiteralComponentAxisSpec,
    LiteralPathSegment,
    LogicalComponentAxisSpec,
    OwnerFamilyReference,
    ParameterInventoryEntry,
    PredicateScalar,
    RefitRequirement,
    RolloutParticipation,
    SemanticAddress,
    SemanticAddressPattern,
    SemanticGraphManifest,
    SemanticInventoryMember,
    SemanticManifestBundle,
    SemanticTensor,
    SemanticTensorFamily,
    SourceAliasContract,
    SynchronizedReplicaSourceAliasContract,
    ValueProvenance,
    _derive_refit_requirements_unchecked,
    _source_required_entry_ids_unchecked,
)


class PrecisionPolicyError(ValueError):
    """Raised when a semantic precision policy cannot compile safely."""


class PrecisionEndpoint(StrEnum):
    """Logical endpoint configured by a compiled precision intent."""

    TRAINING = "training"
    ROLLOUT = "rollout"


_PRECISION_ENDPOINTS: tuple[PrecisionEndpoint, ...] = (
    PrecisionEndpoint.TRAINING,
    PrecisionEndpoint.ROLLOUT,
)


def _graph_sort_key(graph_instance_id: str) -> tuple[int, str]:
    return (0 if graph_instance_id == "main" else 1, graph_instance_id)


def _require_record_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _require_precision(
    value: object,
    field_name: str,
    *,
    optional: bool = False,
) -> PrecisionName | None:
    if value is None and optional:
        return None
    if value not in {"bf16", "mxfp8"}:
        suffix = " or None" if optional else ""
        raise TypeError(f"{field_name} must be bf16 or mxfp8{suffix}")
    return value  # type: ignore[return-value]


def _member_domain(member: SemanticInventoryMember) -> FamilyIndexDomain:
    if isinstance(member, SemanticTensorFamily):
        return member.domain
    return FamilyIndexDomain(None, ())


def _member_graph_path(member: SemanticInventoryMember) -> str:
    if isinstance(member, SemanticTensor):
        return member.address.semantic_graph_path
    return member.pattern.semantic_graph_path


def _member_model_part(member: SemanticInventoryMember) -> str:
    if isinstance(member, SemanticTensor):
        return member.address.model_part
    return member.pattern.model_part


def _member_module_kind(member: SemanticInventoryMember) -> str:
    if isinstance(member, SemanticTensor):
        return member.address.module_kind
    return member.pattern.module_kind


def _member_attributes(
    member: SemanticInventoryMember,
) -> tuple[tuple[str, PredicateScalar], ...]:
    if isinstance(member, SemanticTensor):
        return member.address.attributes
    return member.pattern.attributes


def _member_parameter_role(member: SemanticInventoryMember) -> str:
    if isinstance(member, SemanticTensor):
        return member.address.parameter_role
    return member.pattern.parameter_role


def _axis_key(value: int | str) -> tuple[int, int | str]:
    if isinstance(value, bool):
        raise TypeError("axis members cannot be bool")
    if isinstance(value, int):
        return (0, value)
    return (1, value)


def _scalar_key(value: PredicateScalar) -> tuple[int, object]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, float):
        if not isfinite(value):
            raise PrecisionPolicyError("semantic float values must be finite")
        return (2, 0.0 if value == 0.0 else value)
    if isinstance(value, str):
        return (3, value)
    raise TypeError("semantic scalar must be bool, int, float, or str")


def _scalar_payload(value: PredicateScalar) -> dict[str, object]:
    kind, canonical = _scalar_key(value)
    return {
        "type": ("bool", "int", "float", "str")[kind],
        "value": bool(canonical) if kind == 0 else canonical,
    }


def _domain_payload(domain: FamilyIndexDomain) -> dict[str, object]:
    return {
        "layers": None
        if domain.layer_domain is None
        else [
            {
                "global_decoder_layer": member.global_decoder_layer,
                "moe_ordinal": member.moe_ordinal,
            }
            for member in domain.layer_domain.members
        ],
        "independent_axes": [
            {
                "name": axis.name,
                "members": [_scalar_payload(member) for member in axis.members],
            }
            for axis in domain.independent_axes
        ],
    }


def _domain_sort_key(domain: FamilyIndexDomain) -> str:
    return json.dumps(
        _domain_payload(domain),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _format_payload(format_descriptor: FormatDescriptor) -> dict[str, object]:
    def component_axes_payload(
        component_axes: tuple[
            LogicalComponentAxisSpec | LiteralComponentAxisSpec,
            ...,
        ]
        | None,
    ) -> dict[str, object]:
        if component_axes is None:
            return {"kind": "identity"}
        return {
            "kind": "explicit",
            "axes": [
                {
                    "kind": "logical",
                    "logical_axis": axis.logical_axis,
                    "divisor": axis.divisor,
                    "rounding": axis.rounding.value,
                }
                if isinstance(axis, LogicalComponentAxisSpec)
                else {
                    "kind": "literal",
                    "axis_name": axis.axis_name,
                    "extent": axis.extent,
                }
                for axis in component_axes
            ],
        }

    return {
        "format_id": format_descriptor.format_id,
        "family": format_descriptor.family,
        "components": [
            {
                "role": component.role,
                "dtype": component.dtype,
                "encoding": component.encoding,
                "component_axes": component_axes_payload(component.component_axes),
            }
            for component in format_descriptor.components
        ],
    }


def _precision_format(precision: PrecisionName) -> FormatDescriptor:
    if precision == "bf16":
        return BF16_FORMAT
    if precision == "mxfp8":
        return MXFP8_FORMAT
    raise PrecisionPolicyError(f"unsupported precision profile: {precision}")


@dataclass(frozen=True, slots=True)
class CompactDomainSelection:
    """One graph-qualified compact subdomain of an inventory entry."""

    graph_instance_id: str
    semantic_graph_path: str
    inventory_entry_id: str
    member_domain: FamilyIndexDomain

    def __post_init__(self) -> None:
        _require_record_text(self.graph_instance_id, "graph_instance_id")
        _require_record_text(self.semantic_graph_path, "semantic_graph_path")
        _require_record_text(self.inventory_entry_id, "inventory_entry_id")
        if not isinstance(self.member_domain, FamilyIndexDomain):
            raise TypeError("member_domain must be FamilyIndexDomain")
        if self.member_domain.cardinality == 0:
            raise ValueError("compact selection member_domain must be non-empty")

    @property
    def logical_cardinality(self) -> int:
        """Return selected logical tensor count without member expansion."""
        return self.member_domain.cardinality


@dataclass(frozen=True, slots=True)
class LayerSelectionRecord:
    """One exact boundary decision over a canonical layer index space."""

    scope_id: str
    graph_instance_id: str
    semantic_graph_path: str
    index_space: Literal["global_decoder", "moe_ordinal"]
    universe_coordinates: tuple[int, ...]
    retained_coordinates: tuple[int, ...]
    selected_layer_members: tuple[LayerMember, ...]

    def __post_init__(self) -> None:
        _require_record_text(self.scope_id, "scope_id")
        _require_record_text(self.graph_instance_id, "graph_instance_id")
        _require_record_text(self.semantic_graph_path, "semantic_graph_path")
        if self.index_space not in {"global_decoder", "moe_ordinal"}:
            raise TypeError("index_space must be global_decoder or moe_ordinal")
        universe: tuple[int, ...] = tuple(self.universe_coordinates)
        retained: tuple[int, ...] = tuple(self.retained_coordinates)
        selected: tuple[LayerMember, ...] = tuple(self.selected_layer_members)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in universe
        ):
            raise TypeError("universe_coordinates must contain non-negative integers")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in retained
        ):
            raise TypeError("retained_coordinates must contain non-negative integers")
        if len(universe) != len(set(universe)):
            raise ValueError("universe_coordinates must contain unique coordinates")
        if len(retained) != len(set(retained)):
            raise ValueError("retained_coordinates must contain unique coordinates")
        if any(not isinstance(item, LayerMember) for item in selected):
            raise TypeError("selected_layer_members must contain LayerMember records")
        if len(selected) != len(set(selected)):
            raise ValueError("selected_layer_members must be unique")
        canonical_universe = tuple(sorted(universe))
        canonical_retained = tuple(sorted(retained))
        canonical_selected = tuple(sorted(selected))
        if not set(canonical_retained).issubset(canonical_universe):
            raise ValueError("retained layer coordinates must belong to the universe")
        selected_coordinates = {
            item.global_decoder_layer
            if self.index_space == "global_decoder"
            else item.moe_ordinal
            for item in canonical_selected
        }
        if None in selected_coordinates or not selected_coordinates.issubset(
            canonical_retained
        ):
            raise ValueError(
                "selected layer members must belong to retained coordinates"
            )
        object.__setattr__(self, "universe_coordinates", canonical_universe)
        object.__setattr__(self, "retained_coordinates", canonical_retained)
        object.__setattr__(self, "selected_layer_members", canonical_selected)


@dataclass(frozen=True, slots=True)
class CompiledScopeGraphResult:
    """Compact result of one scope within one graph instance."""

    scope_id: str
    graph_instance_id: str
    matched: tuple[CompactDomainSelection, ...]
    selected: tuple[CompactDomainSelection, ...]
    layer_selections: tuple[LayerSelectionRecord, ...]
    training_precision: PrecisionName | None
    rollout_precision: PrecisionName | None
    out_of_scope_matches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_record_text(self.scope_id, "scope_id")
        _require_record_text(self.graph_instance_id, "graph_instance_id")
        matched = tuple(self.matched)
        selected = tuple(self.selected)
        layer_selections = tuple(self.layer_selections)
        out_of_scope = tuple(self.out_of_scope_matches)
        for field_name, values, expected_type in (
            ("matched", matched, CompactDomainSelection),
            ("selected", selected, CompactDomainSelection),
            ("layer_selections", layer_selections, LayerSelectionRecord),
        ):
            if any(not isinstance(item, expected_type) for item in values):
                raise TypeError(f"{field_name} contains an invalid record")
        if any(
            item.graph_instance_id != self.graph_instance_id
            for item in (*matched, *selected, *layer_selections)
        ):
            raise ValueError("scope graph records must use the enclosing graph ID")
        if any(item.scope_id != self.scope_id for item in layer_selections):
            raise ValueError("layer selections must use the enclosing scope ID")
        if any(not isinstance(item, str) or not item for item in out_of_scope):
            raise TypeError("out_of_scope_matches must contain non-empty strings")
        canonical_matched = tuple(sorted(matched, key=_selection_sort_key))
        canonical_selected = tuple(sorted(selected, key=_selection_sort_key))
        canonical_layers = tuple(
            sorted(
                layer_selections,
                key=lambda item: (
                    item.semantic_graph_path,
                    item.index_space,
                    item.universe_coordinates,
                    item.retained_coordinates,
                    item.selected_layer_members,
                ),
            )
        )
        for field_name, values in (
            ("matched", canonical_matched),
            ("selected", canonical_selected),
        ):
            identities = tuple(_selection_sort_key(item) for item in values)
            if len(identities) != len(set(identities)):
                raise ValueError(f"{field_name} contains duplicate selections")
        if len(canonical_layers) != len(set(canonical_layers)):
            raise ValueError("layer_selections contains duplicates")
        if len(out_of_scope) != len(set(out_of_scope)):
            raise ValueError("out_of_scope_matches contains duplicates")
        _require_precision(self.training_precision, "training_precision", optional=True)
        _require_precision(self.rollout_precision, "rollout_precision", optional=True)
        object.__setattr__(self, "matched", canonical_matched)
        object.__setattr__(self, "selected", canonical_selected)
        object.__setattr__(self, "layer_selections", canonical_layers)
        object.__setattr__(self, "out_of_scope_matches", tuple(sorted(out_of_scope)))

    @property
    def matched_inventory_entry_ids(self) -> tuple[str, ...]:
        """Return canonical matched entry handles."""
        return tuple(
            sorted({selection.inventory_entry_id for selection in self.matched})
        )

    @property
    def selected_logical_cardinality(self) -> int:
        """Return post-boundary logical count."""
        return sum(item.logical_cardinality for item in self.selected)


@dataclass(frozen=True, slots=True)
class CompiledScopeResult:
    """Canonical result of one positive policy scope."""

    scope_id: str
    selector_kind: Literal["role", "advanced_match", "addresses"]
    atomic_conflict: AtomicConflictMode
    graph_results: tuple[CompiledScopeGraphResult, ...]

    def __post_init__(self) -> None:
        _require_record_text(self.scope_id, "scope_id")
        if self.selector_kind not in {"role", "advanced_match", "addresses"}:
            raise TypeError("selector_kind is invalid")
        if self.atomic_conflict not in {"error", "expand"}:
            raise TypeError("atomic_conflict must be error or expand")
        graph_results = tuple(self.graph_results)
        if any(
            not isinstance(item, CompiledScopeGraphResult) for item in graph_results
        ):
            raise TypeError(
                "graph_results must contain CompiledScopeGraphResult records"
            )
        if any(item.scope_id != self.scope_id for item in graph_results):
            raise ValueError("graph results must use the enclosing scope ID")
        graph_ids = tuple(item.graph_instance_id for item in graph_results)
        if len(graph_ids) != len(set(graph_ids)):
            raise ValueError("graph_results contains a duplicate graph")
        object.__setattr__(
            self,
            "graph_results",
            tuple(
                sorted(
                    graph_results,
                    key=lambda item: _graph_sort_key(item.graph_instance_id),
                )
            ),
        )

    def graph_result(self, graph_instance_id: str) -> CompiledScopeGraphResult:
        """Resolve a graph-specific scope result."""
        matches = tuple(
            result
            for result in self.graph_results
            if result.graph_instance_id == graph_instance_id
        )
        if len(matches) != 1:
            raise KeyError(
                f"scope {self.scope_id} has {len(matches)} results for "
                f"{graph_instance_id}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class CompactPrecisionAssignment:
    """One precision assignment over a compact semantic member domain."""

    graph_instance_id: str
    semantic_graph_path: str
    inventory_entry_id: str
    member_domain: FamilyIndexDomain
    precision: PrecisionName
    requested_format: FormatDescriptor

    def __post_init__(self) -> None:
        _require_record_text(self.graph_instance_id, "graph_instance_id")
        _require_record_text(self.semantic_graph_path, "semantic_graph_path")
        _require_record_text(self.inventory_entry_id, "inventory_entry_id")
        if not isinstance(self.member_domain, FamilyIndexDomain):
            raise TypeError("member_domain must be FamilyIndexDomain")
        if self.member_domain.cardinality == 0:
            raise ValueError("precision assignment member_domain must be non-empty")
        precision = _require_precision(self.precision, "precision")
        assert precision is not None
        if not isinstance(self.requested_format, FormatDescriptor):
            raise TypeError("requested_format must be FormatDescriptor")
        if self.requested_format != _precision_format(precision):
            raise ValueError("requested_format does not match assignment precision")


@dataclass(frozen=True, slots=True)
class EndpointPrecisionPlan:
    """Complete compact precision partition for one participating endpoint."""

    graph_instance_id: str
    endpoint: PrecisionEndpoint
    assignments: tuple[CompactPrecisionAssignment, ...]

    def __post_init__(self) -> None:
        _require_record_text(self.graph_instance_id, "graph_instance_id")
        if not isinstance(self.endpoint, PrecisionEndpoint):
            raise TypeError("endpoint must be PrecisionEndpoint")
        assignments = tuple(self.assignments)
        if any(
            not isinstance(item, CompactPrecisionAssignment) for item in assignments
        ):
            raise TypeError(
                "assignments must contain CompactPrecisionAssignment records"
            )
        if any(
            item.graph_instance_id != self.graph_instance_id for item in assignments
        ):
            raise ValueError("assignments must use the enclosing graph ID")
        canonical = tuple(
            sorted(
                assignments,
                key=lambda item: (
                    item.semantic_graph_path,
                    item.inventory_entry_id,
                    item.precision,
                    _domain_sort_key(item.member_domain),
                ),
            )
        )
        identities = tuple(
            (
                item.semantic_graph_path,
                item.inventory_entry_id,
                item.precision,
                _domain_sort_key(item.member_domain),
            )
            for item in canonical
        )
        if len(identities) != len(set(identities)):
            raise ValueError("assignments contains duplicates")
        object.__setattr__(self, "assignments", canonical)

    def precision_for(
        self,
        inventory_entry_id: str,
        *,
        global_decoder_layer: int | None = None,
        moe_ordinal: int | None = None,
        independent_axes: Mapping[str, int | str] | None = None,
    ) -> PrecisionName:
        """Resolve one diagnostic coordinate from the immutable compact plan."""
        coordinates = independent_axes or {}
        matches: tuple[PrecisionName, ...] = tuple(
            assignment.precision
            for assignment in self.assignments
            if assignment.inventory_entry_id == inventory_entry_id
            and _domain_contains_coordinate(
                assignment.member_domain,
                global_decoder_layer=global_decoder_layer,
                moe_ordinal=moe_ordinal,
                independent_axes=coordinates,
            )
        )
        if len(matches) != 1:
            raise KeyError(
                f"expected one precision assignment for {inventory_entry_id}, "
                f"got {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class AtomicExpansion:
    """Compact semantic members added by one atomic fixed-point step."""

    graph_instance_id: str
    endpoint: PrecisionEndpoint
    atomic_group_id: str
    triggering_scope_ids: tuple[str, ...]
    additions: tuple[CompactDomainSelection, ...]

    def __post_init__(self) -> None:
        _require_record_text(self.graph_instance_id, "graph_instance_id")
        if not isinstance(self.endpoint, PrecisionEndpoint):
            raise TypeError("endpoint must be PrecisionEndpoint")
        _require_record_text(self.atomic_group_id, "atomic_group_id")
        scope_ids = tuple(self.triggering_scope_ids)
        additions = tuple(self.additions)
        if any(not isinstance(item, str) or not item for item in scope_ids):
            raise TypeError("triggering_scope_ids must contain non-empty strings")
        if len(scope_ids) != len(set(scope_ids)):
            raise ValueError("triggering_scope_ids contains duplicates")
        if any(not isinstance(item, CompactDomainSelection) for item in additions):
            raise TypeError("additions must contain CompactDomainSelection records")
        if any(item.graph_instance_id != self.graph_instance_id for item in additions):
            raise ValueError("atomic additions must use the enclosing graph ID")
        canonical_additions = tuple(sorted(additions, key=_selection_sort_key))
        identities = tuple(_selection_sort_key(item) for item in canonical_additions)
        if len(identities) != len(set(identities)):
            raise ValueError("atomic additions contains duplicates")
        object.__setattr__(self, "triggering_scope_ids", tuple(sorted(scope_ids)))
        object.__setattr__(self, "additions", canonical_additions)


@dataclass(frozen=True, slots=True)
class OwnerRefitRequirements(Mapping[OwnerFamilyReference, RefitRequirement]):
    """Immutable mapping backed by canonical owner/requirement tuples."""

    entries: tuple[tuple[OwnerFamilyReference, RefitRequirement], ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], OwnerFamilyReference)
            or not isinstance(item[1], RefitRequirement)
            for item in entries
        ):
            raise TypeError(
                "entries must contain owner-reference/refit-requirement pairs"
            )
        canonical = tuple(
            sorted(
                entries,
                key=lambda item: (
                    _graph_sort_key(item[0].graph_instance_id),
                    item[0].owner_family_id,
                ),
            )
        )
        if len({reference for reference, _ in canonical}) != len(canonical):
            raise ValueError("owner refit requirements contain a duplicate owner")
        object.__setattr__(self, "entries", canonical)

    def __getitem__(self, key: OwnerFamilyReference) -> RefitRequirement:
        for reference, requirement in self.entries:
            if reference == key:
                return requirement
        raise KeyError(key)

    def __iter__(self) -> Iterator[OwnerFamilyReference]:
        return (reference for reference, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True, slots=True)
class OwnerRealizationRequest:
    """Canonical source owner needed at one semantic refit cadence."""

    owner_family: OwnerFamilyReference
    requirement: RefitRequirement
    member_graph_instance_ids: tuple[str, ...]
    inventory_entry_ids: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_family, OwnerFamilyReference):
            raise TypeError("owner_family must be OwnerFamilyReference")
        if not isinstance(self.requirement, RefitRequirement):
            raise TypeError("requirement must be RefitRequirement")
        if self.requirement == RefitRequirement.NONE:
            raise ValueError("source realization request cannot use NONE cadence")
        graph_ids = tuple(self.member_graph_instance_ids)
        inventory_ids = tuple(self.inventory_entry_ids)
        if any(not isinstance(item, str) or not item for item in graph_ids):
            raise TypeError("member_graph_instance_ids must contain non-empty strings")
        if len(graph_ids) != len(set(graph_ids)):
            raise ValueError("member_graph_instance_ids contains duplicates")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
            for item in inventory_ids
        ):
            raise TypeError("inventory_entry_ids must contain graph/entry string pairs")
        if len(inventory_ids) != len(set(inventory_ids)):
            raise ValueError("inventory_entry_ids contains duplicates")
        if any(graph_id not in graph_ids for graph_id, _ in inventory_ids):
            raise ValueError("inventory entry graph must be a member graph")
        object.__setattr__(
            self,
            "member_graph_instance_ids",
            tuple(sorted(graph_ids, key=_graph_sort_key)),
        )
        object.__setattr__(
            self,
            "inventory_entry_ids",
            tuple(
                sorted(
                    inventory_ids,
                    key=lambda item: (_graph_sort_key(item[0]), item[1]),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class CompiledGraphPrecisionIntent:
    """Immutable construction and cadence intent for one graph instance."""

    graph_instance_id: str
    model_family: str
    model_revision: str
    lifecycle: GraphLifecycle
    topology_digest: str
    policy_digest: str
    training_plan: EndpointPrecisionPlan | None
    rollout_plan: EndpointPrecisionPlan | None
    scope_results: tuple[CompiledScopeGraphResult, ...]
    atomic_expansions: tuple[AtomicExpansion, ...]
    owner_refit_requirements: OwnerRefitRequirements
    refit_requirement: RefitRequirement
    startup_owner_requests: tuple[OwnerFamilyReference, ...]
    every_version_owner_requests: tuple[OwnerFamilyReference, ...]
    immutable_checkpoint_evidence: ImmutableAuxiliaryEvidence | None
    out_of_scope_inventory_entry_ids: tuple[str, ...]
    intent_id: str

    def __post_init__(self) -> None:
        _require_record_text(self.graph_instance_id, "graph_instance_id")
        _require_record_text(self.model_family, "model_family")
        _require_record_text(self.model_revision, "model_revision")
        if not isinstance(self.lifecycle, GraphLifecycle):
            raise TypeError("lifecycle must be GraphLifecycle")
        _require_record_text(self.topology_digest, "topology_digest")
        _require_record_text(self.policy_digest, "policy_digest")
        for endpoint, plan in (
            (PrecisionEndpoint.TRAINING, self.training_plan),
            (PrecisionEndpoint.ROLLOUT, self.rollout_plan),
        ):
            if plan is not None and not isinstance(plan, EndpointPrecisionPlan):
                raise TypeError(f"{endpoint.value}_plan must be EndpointPrecisionPlan")
            if plan is not None and (
                plan.graph_instance_id != self.graph_instance_id
                or plan.endpoint != endpoint
            ):
                raise ValueError(
                    f"{endpoint.value}_plan must use the enclosing graph and endpoint"
                )
        scope_results = tuple(self.scope_results)
        atomic_expansions = tuple(self.atomic_expansions)
        startup_requests: tuple[OwnerFamilyReference, ...] = tuple(
            self.startup_owner_requests
        )
        every_version_requests: tuple[OwnerFamilyReference, ...] = tuple(
            self.every_version_owner_requests
        )
        out_of_scope = tuple(self.out_of_scope_inventory_entry_ids)
        if any(
            not isinstance(item, CompiledScopeGraphResult) for item in scope_results
        ):
            raise TypeError(
                "scope_results must contain CompiledScopeGraphResult records"
            )
        if any(
            item.graph_instance_id != self.graph_instance_id for item in scope_results
        ):
            raise ValueError("scope results must use the enclosing graph ID")
        if any(not isinstance(item, AtomicExpansion) for item in atomic_expansions):
            raise TypeError("atomic_expansions must contain AtomicExpansion records")
        if any(
            item.graph_instance_id != self.graph_instance_id
            for item in atomic_expansions
        ):
            raise ValueError("atomic expansions must use the enclosing graph ID")
        if not isinstance(self.owner_refit_requirements, OwnerRefitRequirements):
            raise TypeError("owner_refit_requirements must be OwnerRefitRequirements")
        if not isinstance(self.refit_requirement, RefitRequirement):
            raise TypeError("refit_requirement must be RefitRequirement")
        if any(not isinstance(item, OwnerFamilyReference) for item in startup_requests):
            raise TypeError(
                "startup_owner_requests must contain OwnerFamilyReference records"
            )
        if any(
            not isinstance(item, OwnerFamilyReference)
            for item in every_version_requests
        ):
            raise TypeError(
                "every_version_owner_requests must contain OwnerFamilyReference records"
            )
        if len(startup_requests) != len(set(startup_requests)):
            raise ValueError("startup_owner_requests contains duplicates")
        if len(every_version_requests) != len(set(every_version_requests)):
            raise ValueError("every_version_owner_requests contains duplicates")
        if self.immutable_checkpoint_evidence is not None and not isinstance(
            self.immutable_checkpoint_evidence, ImmutableAuxiliaryEvidence
        ):
            raise TypeError(
                "immutable_checkpoint_evidence must be ImmutableAuxiliaryEvidence"
            )
        if any(not isinstance(item, str) or not item for item in out_of_scope):
            raise TypeError(
                "out_of_scope_inventory_entry_ids must contain non-empty strings"
            )
        if len(out_of_scope) != len(set(out_of_scope)):
            raise ValueError("out_of_scope_inventory_entry_ids contains duplicates")
        if not isinstance(self.intent_id, str):
            raise TypeError("intent_id must be a string")
        scope_ids = tuple(item.scope_id for item in scope_results)
        if len(scope_ids) != len(set(scope_ids)):
            raise ValueError("scope_results contains duplicate scope IDs")
        owner_sort_key = lambda item: (
            _graph_sort_key(item.graph_instance_id),
            item.owner_family_id,
        )
        object.__setattr__(
            self,
            "scope_results",
            tuple(sorted(scope_results, key=lambda item: item.scope_id)),
        )
        object.__setattr__(
            self,
            "atomic_expansions",
            tuple(
                sorted(
                    atomic_expansions,
                    key=lambda item: (
                        item.endpoint.value,
                        item.atomic_group_id,
                        item.triggering_scope_ids,
                        tuple(_selection_sort_key(value) for value in item.additions),
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "startup_owner_requests",
            tuple(sorted(startup_requests, key=owner_sort_key)),
        )
        object.__setattr__(
            self,
            "every_version_owner_requests",
            tuple(sorted(every_version_requests, key=owner_sort_key)),
        )
        object.__setattr__(
            self, "out_of_scope_inventory_entry_ids", tuple(sorted(out_of_scope))
        )


@dataclass(frozen=True, slots=True)
class CompiledPrecisionIntentGroup:
    """Canonical immutable compiler output shared by all endpoints."""

    schema_version: int
    topology_digest: str
    policy_digest: str
    graph_intents: tuple[CompiledGraphPrecisionIntent, ...]
    scope_results: tuple[CompiledScopeResult, ...]
    atomic_expansions: tuple[AtomicExpansion, ...]
    startup_source_items: tuple[OwnerRealizationRequest, ...]
    every_version_source_items: tuple[OwnerRealizationRequest, ...]
    immutable_checkpoint_contexts: tuple[ImmutableAuxiliaryEvidence, ...]
    source_alias_contracts: tuple[SourceAliasContract, ...]
    intent_group_id: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema_version must be an integer")
        _require_record_text(self.topology_digest, "topology_digest")
        _require_record_text(self.policy_digest, "policy_digest")
        graph_intents = tuple(self.graph_intents)
        scope_results = tuple(self.scope_results)
        atomic_expansions = tuple(self.atomic_expansions)
        startup_items = tuple(self.startup_source_items)
        every_version_items = tuple(self.every_version_source_items)
        checkpoint_contexts = tuple(self.immutable_checkpoint_contexts)
        source_alias_contracts = tuple(self.source_alias_contracts)
        typed_collections: tuple[tuple[str, tuple[object, ...], type[object]], ...] = (
            ("graph_intents", graph_intents, CompiledGraphPrecisionIntent),
            ("scope_results", scope_results, CompiledScopeResult),
            ("atomic_expansions", atomic_expansions, AtomicExpansion),
            ("startup_source_items", startup_items, OwnerRealizationRequest),
            (
                "every_version_source_items",
                every_version_items,
                OwnerRealizationRequest,
            ),
            (
                "immutable_checkpoint_contexts",
                checkpoint_contexts,
                ImmutableAuxiliaryEvidence,
            ),
        )
        for field_name, values, expected_type in typed_collections:
            if any(not isinstance(item, expected_type) for item in values):
                raise TypeError(f"{field_name} contains an invalid record")
        if any(
            not isinstance(
                contract,
                (
                    IdenticalStorageSourceAliasContract,
                    SynchronizedReplicaSourceAliasContract,
                ),
            )
            for contract in source_alias_contracts
        ):
            raise TypeError("source_alias_contracts contains an invalid record")
        graph_ids = tuple(item.graph_instance_id for item in graph_intents)
        if len(graph_ids) != len(set(graph_ids)):
            raise ValueError("graph_intents contains duplicate graph IDs")
        scope_ids = tuple(item.scope_id for item in scope_results)
        if len(scope_ids) != len(set(scope_ids)):
            raise ValueError("scope_results contains duplicate scope IDs")
        for field_name, values in (
            ("startup_source_items", startup_items),
            ("every_version_source_items", every_version_items),
        ):
            owners = tuple(item.owner_family for item in values)
            if len(owners) != len(set(owners)):
                raise ValueError(f"{field_name} contains duplicate owners")
        evidence_graph_ids = tuple(
            item.graph_instance_id for item in checkpoint_contexts
        )
        if len(evidence_graph_ids) != len(set(evidence_graph_ids)):
            raise ValueError(
                "immutable_checkpoint_contexts contains duplicate graph IDs"
            )
        expansion_key = lambda item: (
            _graph_sort_key(item.graph_instance_id),
            item.endpoint.value,
            item.atomic_group_id,
            item.triggering_scope_ids,
            tuple(_selection_sort_key(value) for value in item.additions),
        )
        request_key = lambda item: (
            _graph_sort_key(item.owner_family.graph_instance_id),
            item.owner_family.owner_family_id,
        )
        object.__setattr__(
            self,
            "graph_intents",
            tuple(
                sorted(
                    graph_intents,
                    key=lambda item: _graph_sort_key(item.graph_instance_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "scope_results",
            tuple(sorted(scope_results, key=lambda item: item.scope_id)),
        )
        object.__setattr__(
            self,
            "atomic_expansions",
            tuple(sorted(atomic_expansions, key=expansion_key)),
        )
        object.__setattr__(
            self, "startup_source_items", tuple(sorted(startup_items, key=request_key))
        )
        object.__setattr__(
            self,
            "every_version_source_items",
            tuple(sorted(every_version_items, key=request_key)),
        )
        object.__setattr__(
            self,
            "immutable_checkpoint_contexts",
            tuple(
                sorted(
                    checkpoint_contexts,
                    key=lambda item: _graph_sort_key(item.graph_instance_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "source_alias_contracts",
            _canonical_source_alias_contracts(source_alias_contracts),
        )
        object.__setattr__(
            self,
            "intent_group_id",
            _digest(_compiled_group_payload(self, include_group_id=False)),
        )

    def graph_intent(self, graph_instance_id: str) -> CompiledGraphPrecisionIntent:
        """Resolve one compiled graph intent."""
        matches = tuple(
            intent
            for intent in self.graph_intents
            if intent.graph_instance_id == graph_instance_id
        )
        if len(matches) != 1:
            raise KeyError(f"expected one graph intent for {graph_instance_id}")
        return matches[0]

    def scope_result(self, scope_id: str) -> CompiledScopeResult:
        """Resolve one compiled policy-scope result."""
        matches = tuple(
            result for result in self.scope_results if result.scope_id == scope_id
        )
        if len(matches) != 1:
            raise KeyError(f"expected one scope result for {scope_id}")
        return matches[0]

    def to_wire_dict(self) -> dict[str, object]:
        """Serialize the canonical logical plan at an explicit process boundary."""
        return _compiled_group_payload(self, include_group_id=True)


@dataclass(frozen=True, slots=True)
class _DomainFactor:
    axes: tuple[str, ...]
    points: tuple[tuple[int | str, ...], ...]


def _domain_factors(domain: FamilyIndexDomain) -> tuple[_DomainFactor, ...]:
    factors: list[_DomainFactor] = []
    if domain.layer_domain is not None:
        layer_axes = domain.layer_domain.axis_names
        factors.append(
            _DomainFactor(
                axes=layer_axes,
                points=tuple(
                    tuple(
                        member.global_decoder_layer
                        if axis == "global_decoder_layer"
                        else _require_moe_ordinal(member)
                        for axis in layer_axes
                    )
                    for member in domain.layer_domain.members
                ),
            )
        )
    factors.extend(
        _DomainFactor((axis.name,), tuple((member,) for member in axis.members))
        for axis in domain.independent_axes
    )
    return tuple(factors)


def _require_moe_ordinal(member: LayerMember) -> int:
    if member.moe_ordinal is None:
        raise PrecisionPolicyError("moe_ordinal is absent from layer member")
    return member.moe_ordinal


def _domain_from_factors(
    template: FamilyIndexDomain,
    factors: tuple[_DomainFactor, ...],
) -> FamilyIndexDomain | None:
    if any(not factor.points for factor in factors):
        return None
    factor_by_axes = {factor.axes: factor for factor in factors}
    layer_domain: LayerDomain | None = None
    if template.layer_domain is not None:
        layer_axes = template.layer_domain.axis_names
        layer_factor = factor_by_axes[layer_axes]
        layer_domain = LayerDomain(
            tuple(
                LayerMember(
                    int(point[layer_axes.index("global_decoder_layer")]),
                    (
                        None
                        if "moe_ordinal" not in layer_axes
                        else int(point[layer_axes.index("moe_ordinal")])
                    ),
                )
                for point in layer_factor.points
            )
        )
    independent_axes = tuple(
        AxisDomain(
            axis.name,
            tuple(point[0] for point in factor_by_axes[(axis.name,)].points),
        )
        for axis in template.independent_axes
    )
    result = FamilyIndexDomain(layer_domain, independent_axes)
    return None if result.cardinality == 0 else result


def _factor_point_key(
    point: tuple[int | str, ...],
) -> tuple[tuple[int, int | str], ...]:
    return tuple(_axis_key(value) for value in point)


def _canonical_factor(
    axes: tuple[str, ...], points: set[tuple[int | str, ...]]
) -> _DomainFactor:
    return _DomainFactor(axes, tuple(sorted(points, key=_factor_point_key)))


def _same_domain_shape(left: FamilyIndexDomain, right: FamilyIndexDomain) -> bool:
    return tuple(factor.axes for factor in _domain_factors(left)) == tuple(
        factor.axes for factor in _domain_factors(right)
    )


def _domain_intersection(
    left: FamilyIndexDomain,
    right: FamilyIndexDomain,
) -> FamilyIndexDomain | None:
    if not _same_domain_shape(left, right):
        return None
    factors = tuple(
        _canonical_factor(
            left_factor.axes,
            set(left_factor.points).intersection(right_factor.points),
        )
        for left_factor, right_factor in zip(
            _domain_factors(left), _domain_factors(right), strict=True
        )
    )
    return _domain_from_factors(left, factors)


def _domain_difference(
    left: FamilyIndexDomain,
    right: FamilyIndexDomain,
) -> tuple[FamilyIndexDomain, ...]:
    intersection = _domain_intersection(left, right)
    if intersection is None:
        return (left,)
    left_factors = _domain_factors(left)
    intersection_factors = _domain_factors(intersection)
    pieces: list[FamilyIndexDomain] = []
    prefix: list[_DomainFactor] = []
    for index, (left_factor, intersection_factor) in enumerate(
        zip(left_factors, intersection_factors, strict=True)
    ):
        difference = set(left_factor.points).difference(intersection_factor.points)
        if difference:
            candidate_factors = (
                *prefix,
                _canonical_factor(left_factor.axes, difference),
                *left_factors[index + 1 :],
            )
            candidate = _domain_from_factors(left, tuple(candidate_factors))
            if candidate is not None:
                pieces.append(candidate)
        prefix.append(intersection_factor)
    return tuple(pieces)


def _subtract_regions(
    region: FamilyIndexDomain,
    subtractors: tuple[FamilyIndexDomain, ...],
) -> tuple[FamilyIndexDomain, ...]:
    pieces: tuple[FamilyIndexDomain, ...] = (region,)
    for subtractor in subtractors:
        pieces = tuple(
            child for piece in pieces for child in _domain_difference(piece, subtractor)
        )
        if not pieces:
            break
    return _canonical_regions(pieces)


def _try_merge_domains(
    left: FamilyIndexDomain,
    right: FamilyIndexDomain,
) -> FamilyIndexDomain | None:
    if not _same_domain_shape(left, right):
        return None
    left_factors = _domain_factors(left)
    right_factors = _domain_factors(right)
    differing = tuple(
        index
        for index, (left_factor, right_factor) in enumerate(
            zip(left_factors, right_factors, strict=True)
        )
        if left_factor.points != right_factor.points
    )
    if len(differing) > 1:
        return None
    if not differing:
        return left
    merge_index = differing[0]
    merged_factors = list(left_factors)
    merged_factors[merge_index] = _canonical_factor(
        left_factors[merge_index].axes,
        set(left_factors[merge_index].points).union(right_factors[merge_index].points),
    )
    return _domain_from_factors(left, tuple(merged_factors))


def _canonical_regions(
    regions: tuple[FamilyIndexDomain, ...] | list[FamilyIndexDomain],
) -> tuple[FamilyIndexDomain, ...]:
    disjoint: list[FamilyIndexDomain] = []
    for region in sorted(regions, key=_domain_sort_key):
        additions: tuple[FamilyIndexDomain, ...]
        if disjoint:
            additions = _subtract_regions(region, tuple(disjoint))
        else:
            additions = (region,)
        disjoint.extend(additions)
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(disjoint):
            for right_index, right in enumerate(disjoint):
                if right_index <= left_index:
                    continue
                merged = _try_merge_domains(left, right)
                if merged is None:
                    continue
                disjoint = [
                    item
                    for index, item in enumerate(disjoint)
                    if index not in {left_index, right_index}
                ]
                disjoint.append(merged)
                changed = True
                break
            if changed:
                break
    return tuple(sorted(disjoint, key=_domain_sort_key))


def _regions_intersect(
    left: tuple[FamilyIndexDomain, ...] | list[FamilyIndexDomain],
    right: tuple[FamilyIndexDomain, ...] | list[FamilyIndexDomain],
) -> bool:
    return any(
        _domain_intersection(left_region, right_region) is not None
        for left_region in left
        for right_region in right
    )


def _domain_contains_coordinate(
    domain: FamilyIndexDomain,
    *,
    global_decoder_layer: int | None,
    moe_ordinal: int | None,
    independent_axes: Mapping[str, int | str],
) -> bool:
    if domain.layer_domain is None:
        if global_decoder_layer is not None or moe_ordinal is not None:
            return False
    else:
        if not any(
            (
                global_decoder_layer is None
                or member.global_decoder_layer == global_decoder_layer
            )
            and (moe_ordinal is None or member.moe_ordinal == moe_ordinal)
            for member in domain.layer_domain.members
        ):
            return False
    axes = {axis.name: axis for axis in domain.independent_axes}
    if set(independent_axes) != set(axes):
        return False
    return all(
        any(
            _axis_key(member) == _axis_key(independent_axes[name])
            for member in axis.members
        )
        for name, axis in axes.items()
    )


@dataclass(frozen=True, slots=True)
class _Request:
    graph_instance_id: str
    semantic_graph_path: str
    inventory_entry_id: str
    member_domain: FamilyIndexDomain
    endpoint: PrecisionEndpoint
    precision: PrecisionName
    scope_id: str
    atomic_conflict: AtomicConflictMode


@dataclass(frozen=True, slots=True)
class _Fence:
    graph_instance_id: str
    inventory_entry_id: str
    member_domain: FamilyIndexDomain
    endpoint: PrecisionEndpoint
    scope_id: str


@dataclass(frozen=True, slots=True)
class _Indexes:
    entries_by_id: Mapping[str, ParameterInventoryEntry]
    manifests_by_graph: Mapping[str, SemanticGraphManifest]
    out_of_scope_by_graph: Mapping[str, frozenset[str]]
    global_layers: Mapping[tuple[str, str], tuple[int, ...]]
    moe_ordinals: Mapping[tuple[str, str], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class _ResolvedSelector:
    matched: tuple[CompactDomainSelection, ...]
    matched_by_role: tuple[tuple[str, tuple[CompactDomainSelection, ...]], ...] = ()


def _build_indexes(bundle: SemanticManifestBundle) -> _Indexes:
    entries_by_id = {entry.entry_id: entry for entry in bundle.inventory.entries}
    manifests_by_graph = {
        manifest.graph_instance_id: manifest for manifest in bundle.manifests
    }
    out_of_scope_by_graph = {
        manifest.graph_instance_id: frozenset(
            item.inventory_entry_id for item in manifest.out_of_scope
        )
        for manifest in bundle.manifests
    }
    global_layer_sets: dict[tuple[str, str], set[int]] = {}
    moe_ordinal_sets: dict[tuple[str, str], set[int]] = {}
    moe_by_global: dict[tuple[str, str], dict[int, int]] = {}
    global_by_moe: dict[tuple[str, str], dict[int, int]] = {}
    for entry in bundle.inventory.entries:
        key = (entry.graph_instance_id, _member_graph_path(entry.member))
        layer_members = _member_layer_members(entry.member)
        for member in layer_members:
            global_layer_sets.setdefault(key, set()).add(member.global_decoder_layer)
            if member.moe_ordinal is None:
                continue
            moe_ordinal_sets.setdefault(key, set()).add(member.moe_ordinal)
            existing_moe = moe_by_global.setdefault(key, {}).setdefault(
                member.global_decoder_layer, member.moe_ordinal
            )
            existing_global = global_by_moe.setdefault(key, {}).setdefault(
                member.moe_ordinal, member.global_decoder_layer
            )
            if (
                existing_moe != member.moe_ordinal
                or existing_global != member.global_decoder_layer
            ):
                raise PrecisionPolicyError(
                    f"ambiguous correlated layer coordinates for {key}"
                )
    return _Indexes(
        entries_by_id=entries_by_id,
        manifests_by_graph=manifests_by_graph,
        out_of_scope_by_graph=out_of_scope_by_graph,
        global_layers={
            key: tuple(sorted(values)) for key, values in global_layer_sets.items()
        },
        moe_ordinals={
            key: tuple(sorted(values)) for key, values in moe_ordinal_sets.items()
        },
    )


def _member_layer_members(member: SemanticInventoryMember) -> tuple[LayerMember, ...]:
    if isinstance(member, SemanticTensorFamily):
        if member.domain.layer_domain is None:
            return ()
        return member.domain.layer_domain.members
    address = member.address
    if address.global_decoder_layer is None:
        return ()
    return (LayerMember(address.global_decoder_layer, address.moe_ordinal),)


def _string_predicate_matches(
    predicate: str | list[str] | None,
    value: str,
) -> bool:
    if predicate is None:
        return True
    if isinstance(predicate, str):
        return value == predicate
    return value in predicate


def _attribute_predicate_matches(
    predicate: PredicateScalar | list[PredicateScalar],
    value: PredicateScalar,
) -> bool:
    allowed = predicate if isinstance(predicate, list) else [predicate]
    value_key = _scalar_key(value)
    return value_key in {_scalar_key(item) for item in allowed}


def _advanced_matches(
    matcher: AdvancedMatchConfig,
    entry: ParameterInventoryEntry,
) -> bool:
    member = entry.member
    if not _string_predicate_matches(
        matcher.graph_instance_id, entry.graph_instance_id
    ):
        return False
    if not _string_predicate_matches(
        matcher.semantic_graph_path, _member_graph_path(member)
    ):
        return False
    if not _string_predicate_matches(matcher.model_part, _member_model_part(member)):
        return False
    if not _string_predicate_matches(matcher.module_kind, _member_module_kind(member)):
        return False
    if not _string_predicate_matches(
        matcher.parameter_role, _member_parameter_role(member)
    ):
        return False
    attributes = dict(_member_attributes(member))
    return all(
        name in attributes and _attribute_predicate_matches(predicate, attributes[name])
        for name, predicate in matcher.attributes.items()
    )


def _selector_kind(
    scope: PrecisionScopeConfig,
) -> Literal["role", "advanced_match", "addresses"]:
    if scope.roles is not None:
        return "role"
    if scope.advanced_match is not None:
        return "advanced_match"
    return "addresses"


def _resolve_selector(
    policy: PrecisionPolicyConfig,
    scope: PrecisionScopeConfig,
    bundle: SemanticManifestBundle,
    indexes: _Indexes,
) -> _ResolvedSelector:
    if scope.roles is not None:
        matched_by_role: list[tuple[str, tuple[CompactDomainSelection, ...]]] = []
        combined: list[CompactDomainSelection] = []
        for role_name in scope.roles:
            try:
                role = bundle.role_definition(policy.schema_version, role_name)
            except ValueError as error:
                raise PrecisionPolicyError(f"unknown role: {role_name}") from error
            role.validate_expected_domain(bundle)
            role_matches = tuple(
                _whole_entry_selection(indexes.entries_by_id[entry_id])
                for entry_id in role.expected_domain.inventory_entry_ids
            )
            matched_by_role.append((role_name, role_matches))
            combined.extend(role_matches)
        return _ResolvedSelector(
            matched=_canonical_selections(combined),
            matched_by_role=tuple(matched_by_role),
        )
    if scope.advanced_match is not None:
        matcher = scope.advanced_match
        if matcher.graph_instance_id is None or matcher.semantic_graph_path is None:
            raise PrecisionPolicyError(
                f"advanced scope {scope.id} must qualify graph_instance_id and "
                "semantic_graph_path"
            )
        return _ResolvedSelector(
            matched=tuple(
                _whole_entry_selection(entry)
                for entry in bundle.inventory.entries
                if _advanced_matches(matcher, entry)
            )
        )
    if scope.addresses is None:
        raise PrecisionPolicyError(f"scope {scope.id} has no selector")
    resolved: list[CompactDomainSelection] = []
    for address in scope.addresses:
        matches = tuple(
            selection
            for entry in bundle.inventory.entries
            if entry.graph_instance_id == address.graph_instance_id
            for selection in _match_address(entry, address)
        )
        if len(matches) != 1:
            raise PrecisionPolicyError(
                f"address must resolve exactly once: "
                f"{address.graph_instance_id}:{address.semantic_id}; got {len(matches)}"
            )
        resolved.append(matches[0])
    return _ResolvedSelector(matched=_canonical_selections(resolved))


def _whole_entry_selection(entry: ParameterInventoryEntry) -> CompactDomainSelection:
    return CompactDomainSelection(
        graph_instance_id=entry.graph_instance_id,
        semantic_graph_path=_member_graph_path(entry.member),
        inventory_entry_id=entry.entry_id,
        member_domain=_member_domain(entry.member),
    )


def _match_address(
    entry: ParameterInventoryEntry,
    address: SemanticAddressSelectorConfig,
) -> tuple[CompactDomainSelection, ...]:
    member = entry.member
    if _member_graph_path(member) != address.semantic_graph_path:
        return ()
    if isinstance(member, SemanticTensor):
        if member.address.semantic_id != address.semantic_id:
            return ()
        return (_whole_entry_selection(entry),)
    singleton = _family_address_domain(member, address.semantic_id)
    if singleton is None:
        return ()
    return (
        CompactDomainSelection(
            graph_instance_id=entry.graph_instance_id,
            semantic_graph_path=member.pattern.semantic_graph_path,
            inventory_entry_id=entry.entry_id,
            member_domain=singleton,
        ),
    )


def _family_address_domain(
    family: SemanticTensorFamily,
    semantic_id: str,
) -> FamilyIndexDomain | None:
    prefix = family.pattern.semantic_graph_path.split(".")
    path = semantic_id.split(".")
    if path[: len(prefix)] != prefix:
        return None
    suffix = path[len(prefix) :]
    if len(suffix) != len(family.pattern.path_segments):
        return None
    rendered_coordinates: dict[str, str] = {}
    for segment, rendered in zip(family.pattern.path_segments, suffix, strict=True):
        if isinstance(segment, LiteralPathSegment):
            if segment.value != rendered:
                return None
        elif isinstance(segment, IndexPathSegment):
            rendered_coordinates[segment.axis_name] = rendered
        else:
            raise TypeError("unknown semantic path segment")
    layer_domain: LayerDomain | None = None
    if family.domain.layer_domain is not None:
        layer_members = tuple(
            member
            for member in family.domain.layer_domain.members
            if all(
                rendered_coordinates.get(axis)
                in {None, str(_layer_axis_value(member, axis))}
                for axis in family.domain.layer_domain.axis_names
            )
        )
        if len(layer_members) != 1:
            return None
        layer_domain = LayerDomain(layer_members)
    independent_axes: list[AxisDomain] = []
    for axis in family.domain.independent_axes:
        rendered = rendered_coordinates.get(axis.name)
        if rendered is None:
            return None
        members = tuple(member for member in axis.members if str(member) == rendered)
        if len(members) != 1:
            return None
        independent_axes.append(AxisDomain(axis.name, members))
    result = FamilyIndexDomain(layer_domain, tuple(independent_axes))
    return result if result.cardinality == 1 else None


def _layer_axis_value(member: LayerMember, axis: str) -> int | None:
    if axis == "global_decoder_layer":
        return member.global_decoder_layer
    if axis == "moe_ordinal":
        return member.moe_ordinal
    raise PrecisionPolicyError(f"unknown layer coordinate: {axis}")


def _selection_sort_key(selection: CompactDomainSelection) -> tuple[object, ...]:
    return (
        _graph_sort_key(selection.graph_instance_id),
        selection.semantic_graph_path,
        selection.inventory_entry_id,
        _domain_sort_key(selection.member_domain),
    )


def _canonical_selections(
    selections: tuple[CompactDomainSelection, ...] | list[CompactDomainSelection],
) -> tuple[CompactDomainSelection, ...]:
    grouped: dict[tuple[str, str, str], list[FamilyIndexDomain]] = {}
    for selection in selections:
        key = (
            selection.graph_instance_id,
            selection.semantic_graph_path,
            selection.inventory_entry_id,
        )
        grouped.setdefault(key, []).append(selection.member_domain)
    canonical = tuple(
        CompactDomainSelection(graph_id, graph_path, entry_id, domain)
        for (graph_id, graph_path, entry_id), domains in grouped.items()
        for domain in _canonical_regions(domains)
    )
    return tuple(sorted(canonical, key=_selection_sort_key))


def _endpoint_participates(
    lifecycle: GraphLifecycle,
    endpoint: PrecisionEndpoint,
) -> bool:
    if endpoint == PrecisionEndpoint.TRAINING:
        return lifecycle.graph_provenance.value == "training_runtime"
    return lifecycle.rollout_participation != RolloutParticipation.NOT_SERVED


def _scope_precision(
    scope: PrecisionScopeConfig,
    endpoint: PrecisionEndpoint,
) -> PrecisionName:
    requested = (
        scope.training if endpoint == PrecisionEndpoint.TRAINING else scope.rollout
    )
    return "bf16" if requested is None else requested


def _validate_coordinate_universe(
    coordinates: tuple[int, ...],
    *,
    index_space: Literal["global_decoder", "moe_ordinal"],
    key: tuple[str, str],
) -> None:
    if not coordinates:
        raise PrecisionPolicyError(
            f"layer selector {index_space} has no coordinates for {key}"
        )
    expected = tuple(range(coordinates[-1] + 1))
    if coordinates != expected:
        raise PrecisionPolicyError(
            f"layer selector {index_space} requires a complete zero-based "
            f"coordinate universe for {key}: got {coordinates}"
        )


def _filter_selection_layers(
    selection: CompactDomainSelection,
    entry: ParameterInventoryEntry,
    layer_selector: LayerSelectorConfig,
    indexes: _Indexes,
    scope_id: str,
) -> tuple[CompactDomainSelection | None, LayerSelectionRecord]:
    index_space = layer_selector.index_space
    coordinate_name = (
        "global_decoder_layer" if index_space == "global_decoder" else "moe_ordinal"
    )
    member_layers = _member_layer_members(entry.member)
    if not member_layers or any(
        _layer_axis_value(member, coordinate_name) is None for member in member_layers
    ):
        raise PrecisionPolicyError(
            f"entry {entry.entry_id} has no {index_space} layer coordinate"
        )
    key = (selection.graph_instance_id, selection.semantic_graph_path)
    universe = (
        indexes.global_layers.get(key, ())
        if index_space == "global_decoder"
        else indexes.moe_ordinals.get(key, ())
    )
    _validate_coordinate_universe(universe, index_space=index_space, key=key)
    if layer_selector.exclude_first + layer_selector.exclude_last >= len(universe):
        raise PrecisionPolicyError(
            f"scope layer exclusions consume the complete {index_space} universe "
            f"for {key}"
        )
    stop = len(universe) - layer_selector.exclude_last
    retained = universe[layer_selector.exclude_first : stop]
    retained_set = set(retained)
    selected_members = tuple(
        member
        for member in member_layers
        if _layer_axis_value(member, coordinate_name) in retained_set
    )
    selected: CompactDomainSelection | None
    if isinstance(entry.member, SemanticTensor):
        selected = selection if selected_members else None
    else:
        selection_layers = selection.member_domain.layer_domain
        if selection_layers is None:
            raise PrecisionPolicyError(
                f"entry {entry.entry_id} has no {index_space} layer domain"
            )
        selected_layer_set = set(selected_members)
        filtered_members = tuple(
            member
            for member in selection_layers.members
            if member in selected_layer_set
        )
        if not filtered_members:
            selected = None
        else:
            selected = CompactDomainSelection(
                selection.graph_instance_id,
                selection.semantic_graph_path,
                selection.inventory_entry_id,
                FamilyIndexDomain(
                    LayerDomain(filtered_members),
                    selection.member_domain.independent_axes,
                ),
            )
    return selected, LayerSelectionRecord(
        scope_id=scope_id,
        graph_instance_id=selection.graph_instance_id,
        semantic_graph_path=selection.semantic_graph_path,
        index_space=index_space,
        universe_coordinates=universe,
        retained_coordinates=retained,
        selected_layer_members=tuple(sorted(selected_members)),
    )


def _compile_scope(
    policy: PrecisionPolicyConfig,
    scope: PrecisionScopeConfig,
    bundle: SemanticManifestBundle,
    indexes: _Indexes,
) -> tuple[CompiledScopeResult, tuple[_Request, ...], tuple[_Fence, ...]]:
    selector = _resolve_selector(policy, scope, bundle, indexes)
    matched = selector.matched
    selected: list[CompactDomainSelection] = []
    layer_records: dict[tuple[str, str], LayerSelectionRecord] = {}
    for selection in matched:
        if scope.layers is None:
            selected.append(selection)
            continue
        filtered, record = _filter_selection_layers(
            selection,
            indexes.entries_by_id[selection.inventory_entry_id],
            scope.layers,
            indexes,
            scope.id,
        )
        record_key = (record.graph_instance_id, record.semantic_graph_path)
        existing = layer_records.get(record_key)
        if existing is None:
            layer_records[record_key] = record
        else:
            layer_records[record_key] = LayerSelectionRecord(
                scope_id=scope.id,
                graph_instance_id=record.graph_instance_id,
                semantic_graph_path=record.semantic_graph_path,
                index_space=record.index_space,
                universe_coordinates=record.universe_coordinates,
                retained_coordinates=record.retained_coordinates,
                selected_layer_members=tuple(
                    sorted(
                        set(existing.selected_layer_members).union(
                            record.selected_layer_members
                        )
                    )
                ),
            )
        if filtered is not None:
            selected.append(filtered)
    selected_canonical = _canonical_selections(selected)
    selected_count = sum(item.logical_cardinality for item in selected_canonical)
    if policy.require_match and selector.matched_by_role:
        selected_domains_by_identity: dict[
            tuple[str, str, str], list[FamilyIndexDomain]
        ] = {}
        for selection in selected_canonical:
            identity = (
                selection.graph_instance_id,
                selection.semantic_graph_path,
                selection.inventory_entry_id,
            )
            selected_domains_by_identity.setdefault(identity, []).append(
                selection.member_domain
            )
        for role_name, role_matches in selector.matched_by_role:
            role_selected = any(
                _domain_intersection(selection.member_domain, selected_domain)
                is not None
                for selection in role_matches
                for selected_domain in selected_domains_by_identity.get(
                    (
                        selection.graph_instance_id,
                        selection.semantic_graph_path,
                        selection.inventory_entry_id,
                    ),
                    (),
                )
            )
            if not role_selected:
                raise PrecisionPolicyError(
                    f"scope {scope.id} role {role_name} matched no semantic members "
                    "after layer filtering"
                )
    elif policy.require_match and selected_count == 0:
        raise PrecisionPolicyError(
            f"scope {scope.id} matched no semantic members after layer filtering"
        )
    effective_atomic = scope.atomic_conflict or policy.atomic_conflict
    by_graph_matched: dict[str, list[CompactDomainSelection]] = {}
    by_graph_selected: dict[str, list[CompactDomainSelection]] = {}
    for selection in matched:
        by_graph_matched.setdefault(selection.graph_instance_id, []).append(selection)
    for selection in selected_canonical:
        by_graph_selected.setdefault(selection.graph_instance_id, []).append(selection)

    out_of_scope_matches: dict[str, set[str]] = {}
    for selection in matched:
        if selection.inventory_entry_id in indexes.out_of_scope_by_graph.get(
            selection.graph_instance_id, frozenset()
        ):
            out_of_scope_matches.setdefault(selection.graph_instance_id, set()).add(
                selection.inventory_entry_id
            )
    if out_of_scope_matches:
        rendered = ", ".join(
            f"{graph}:{entry_id}"
            for graph in sorted(out_of_scope_matches, key=_graph_sort_key)
            for entry_id in sorted(out_of_scope_matches[graph])
        )
        raise PrecisionPolicyError(
            f"scope {scope.id} matched explicitly out-of-scope entries: {rendered}"
        )

    graph_results: list[CompiledScopeGraphResult] = []
    requests: list[_Request] = []
    fences: list[_Fence] = []
    changed_participating_endpoint = False
    for graph_id in sorted(indexes.manifests_by_graph, key=_graph_sort_key):
        manifest = indexes.manifests_by_graph[graph_id]
        graph_matched = _canonical_selections(by_graph_matched.get(graph_id, []))
        graph_selected = _canonical_selections(by_graph_selected.get(graph_id, []))
        training_precision: PrecisionName | None = None
        rollout_precision: PrecisionName | None = None
        for endpoint in _PRECISION_ENDPOINTS:
            if graph_matched and _endpoint_participates(manifest.lifecycle, endpoint):
                precision: PrecisionName = _scope_precision(scope, endpoint)
                if endpoint == PrecisionEndpoint.TRAINING:
                    training_precision = precision
                else:
                    rollout_precision = precision
                if precision == "mxfp8" and graph_selected:
                    changed_participating_endpoint = True
                requests.extend(
                    _Request(
                        selection.graph_instance_id,
                        selection.semantic_graph_path,
                        selection.inventory_entry_id,
                        selection.member_domain,
                        endpoint,
                        precision,
                        scope.id,
                        effective_atomic,
                    )
                    for selection in graph_selected
                )
                if scope.layers is not None:
                    selected_by_entry: dict[str, tuple[FamilyIndexDomain, ...]] = {}
                    for selection in graph_selected:
                        selected_by_entry.setdefault(selection.inventory_entry_id, ())
                        selected_by_entry[selection.inventory_entry_id] = (
                            *selected_by_entry[selection.inventory_entry_id],
                            selection.member_domain,
                        )
                    for matched_selection in graph_matched:
                        excluded = _subtract_regions(
                            matched_selection.member_domain,
                            selected_by_entry.get(
                                matched_selection.inventory_entry_id, ()
                            ),
                        )
                        fences.extend(
                            _Fence(
                                graph_id,
                                matched_selection.inventory_entry_id,
                                domain,
                                endpoint,
                                scope.id,
                            )
                            for domain in excluded
                        )
        graph_results.append(
            CompiledScopeGraphResult(
                scope_id=scope.id,
                graph_instance_id=graph_id,
                matched=graph_matched,
                selected=graph_selected,
                layer_selections=tuple(
                    sorted(
                        (
                            record
                            for key, record in layer_records.items()
                            if key[0] == graph_id
                        ),
                        key=lambda item: item.semantic_graph_path,
                    )
                ),
                training_precision=training_precision,
                rollout_precision=rollout_precision,
                out_of_scope_matches=tuple(
                    sorted(out_of_scope_matches.get(graph_id, ()))
                ),
            )
        )
    if selected_count and not changed_participating_endpoint:
        raise PrecisionPolicyError(
            f"scope {scope.id} does not request MXFP8 on a participating endpoint"
        )
    return (
        CompiledScopeResult(
            scope_id=scope.id,
            selector_kind=_selector_kind(scope),
            atomic_conflict=effective_atomic,
            graph_results=tuple(graph_results),
        ),
        tuple(requests),
        tuple(fences),
    )


def _validate_explicit_requests(
    requests: tuple[_Request, ...],
    fences: tuple[_Fence, ...],
) -> None:
    grouped: dict[tuple[str, PrecisionEndpoint, str], list[_Request]] = {}
    for request in requests:
        grouped.setdefault(
            (
                request.graph_instance_id,
                request.endpoint,
                request.inventory_entry_id,
            ),
            [],
        ).append(request)
    for key, values in grouped.items():
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                if left.precision == right.precision:
                    continue
                if (
                    _domain_intersection(left.member_domain, right.member_domain)
                    is not None
                ):
                    raise PrecisionPolicyError(
                        f"conflicting precision scopes {left.scope_id} and "
                        f"{right.scope_id} overlap {key}"
                    )
    for request in requests:
        if request.precision != "mxfp8":
            continue
        for fence in fences:
            if (
                request.graph_instance_id == fence.graph_instance_id
                and request.endpoint == fence.endpoint
                and request.inventory_entry_id == fence.inventory_entry_id
                and request.scope_id != fence.scope_id
                and _domain_intersection(request.member_domain, fence.member_domain)
                is not None
            ):
                raise PrecisionPolicyError(
                    f"scope {request.scope_id} crosses hard BF16 layer boundary "
                    f"from scope {fence.scope_id}"
                )


def _factor_index_for_axis(factors: tuple[_DomainFactor, ...], axis_name: str) -> int:
    matches = tuple(
        index for index, factor in enumerate(factors) if axis_name in factor.axes
    )
    if len(matches) != 1:
        raise PrecisionPolicyError(
            f"atomic projection expected one factor for axis {axis_name}"
        )
    return matches[0]


def _atomic_preimage(
    group_domain: FamilyIndexDomain,
    participant_domain: FamilyIndexDomain,
    projections: tuple[AxisProjection, ...],
    selected_participant_domain: FamilyIndexDomain,
) -> FamilyIndexDomain | None:
    selected = _domain_intersection(participant_domain, selected_participant_domain)
    if selected is None:
        return None
    group_factors = _domain_factors(group_domain)
    selected_factors = _domain_factors(selected)
    source_to_target = {
        projection.member_axis: projection.owner_axis for projection in projections
    }
    target_factor_source_indexes: dict[int, set[int]] = {}
    for source_axis, target_axis in source_to_target.items():
        source_index = _factor_index_for_axis(group_factors, source_axis)
        target_index = _factor_index_for_axis(selected_factors, target_axis)
        target_factor_source_indexes.setdefault(target_index, set()).add(source_index)
    for target_index, source_indexes in target_factor_source_indexes.items():
        if len(selected_factors[target_index].axes) > 1 and len(source_indexes) > 1:
            raise PrecisionPolicyError(
                "atomic projection cannot split a correlated target factor"
            )

    restricted_factors: list[_DomainFactor] = []
    for source_index, source_factor in enumerate(group_factors):
        mapped = tuple(
            (axis_index, source_to_target[axis_name])
            for axis_index, axis_name in enumerate(source_factor.axes)
            if axis_name in source_to_target
        )
        retained: set[tuple[int | str, ...]] = set()
        for point in source_factor.points:
            allowed = True
            for target_index, target_factor in enumerate(selected_factors):
                relevant = tuple(
                    (source_axis_index, target_axis)
                    for source_axis_index, target_axis in mapped
                    if target_axis in target_factor.axes
                )
                if not relevant:
                    continue
                projected_positions = tuple(
                    target_factor.axes.index(target_axis) for _, target_axis in relevant
                )
                projected_value = tuple(
                    point[source_axis_index] for source_axis_index, _ in relevant
                )
                allowed_projection = {
                    tuple(target_point[position] for position in projected_positions)
                    for target_point in target_factor.points
                }
                if projected_value not in allowed_projection:
                    allowed = False
                    break
            if allowed:
                retained.add(point)
        restricted_factors.append(_canonical_factor(source_factor.axes, retained))
    return _domain_from_factors(group_domain, tuple(restricted_factors))


def _atomic_project(
    group_subdomain: FamilyIndexDomain,
    participant_domain: FamilyIndexDomain,
    projections: tuple[AxisProjection, ...],
) -> FamilyIndexDomain | None:
    group_factors = _domain_factors(group_subdomain)
    participant_factors = _domain_factors(participant_domain)
    target_to_source = {
        projection.owner_axis: projection.member_axis for projection in projections
    }
    projected_factors: list[_DomainFactor] = []
    for target_factor in participant_factors:
        source_indexes = {
            _factor_index_for_axis(group_factors, target_to_source[target_axis])
            for target_axis in target_factor.axes
        }
        if len(source_indexes) != 1:
            raise PrecisionPolicyError(
                "atomic projection cannot synthesize a correlated target factor"
            )
        source_factor = group_factors[next(iter(source_indexes))]
        source_positions = tuple(
            source_factor.axes.index(target_to_source[target_axis])
            for target_axis in target_factor.axes
        )
        projected_points = {
            tuple(source_point[position] for position in source_positions)
            for source_point in source_factor.points
        }
        retained = set(target_factor.points).intersection(projected_points)
        projected_factors.append(_canonical_factor(target_factor.axes, retained))
    return _domain_from_factors(participant_domain, tuple(projected_factors))


def _atomic_missing_regions(
    requests: tuple[_Request, ...] | list[_Request],
    *,
    graph_instance_id: str,
    endpoint: PrecisionEndpoint,
    inventory_entry_id: str,
    required_domain: FamilyIndexDomain,
) -> tuple[FamilyIndexDomain, ...]:
    existing = _canonical_regions(
        [
            request.member_domain
            for request in _requests_for(
                requests,
                graph_instance_id=graph_instance_id,
                endpoint=endpoint,
                inventory_entry_id=inventory_entry_id,
                precision="mxfp8",
            )
        ]
    )
    return _subtract_regions(required_domain, existing)


def _atomic_missing_by_participant(
    requests: tuple[_Request, ...] | list[_Request],
    manifest: SemanticGraphManifest,
    endpoint: PrecisionEndpoint,
    group: AtomicGroup,
    group_selection: FamilyIndexDomain,
) -> tuple[tuple[str, FamilyIndexDomain], ...]:
    missing_by_participant: list[tuple[str, FamilyIndexDomain]] = []
    for participant in group.participants:
        required = _atomic_project(
            group_selection,
            participant.participant_domain,
            participant.group_to_participant_axes,
        )
        if required is None:
            continue
        missing_by_participant.extend(
            (participant.inventory_entry_id, missing)
            for missing in _atomic_missing_regions(
                requests,
                graph_instance_id=manifest.graph_instance_id,
                endpoint=endpoint,
                inventory_entry_id=participant.inventory_entry_id,
                required_domain=required,
            )
        )
    return tuple(missing_by_participant)


def _validate_atomic_error_requests(
    bundle: SemanticManifestBundle,
    requests: tuple[_Request, ...],
) -> None:
    """Reject every explicit partial error-mode trigger before expansion."""
    for manifest in sorted(
        bundle.manifests,
        key=lambda item: _graph_sort_key(item.graph_instance_id),
    ):
        for endpoint in _PRECISION_ENDPOINTS:
            if not _endpoint_participates(manifest.lifecycle, endpoint):
                continue
            for group in sorted(manifest.atomic_groups, key=lambda item: item.group_id):
                for triggering_participant in group.participants:
                    triggers = _requests_for(
                        requests,
                        graph_instance_id=manifest.graph_instance_id,
                        endpoint=endpoint,
                        inventory_entry_id=(triggering_participant.inventory_entry_id),
                        precision="mxfp8",
                    )
                    for trigger in triggers:
                        if trigger.atomic_conflict != "error":
                            continue
                        group_selection = _atomic_preimage(
                            group.group_domain,
                            triggering_participant.participant_domain,
                            triggering_participant.group_to_participant_axes,
                            trigger.member_domain,
                        )
                        if group_selection is None:
                            continue
                        if _atomic_missing_by_participant(
                            requests,
                            manifest,
                            endpoint,
                            group,
                            group_selection,
                        ):
                            raise PrecisionPolicyError(
                                f"atomic precision conflict in {group.group_id} "
                                f"at {endpoint.value} from scope {trigger.scope_id}"
                            )


def _apply_atomic_closure(
    bundle: SemanticManifestBundle,
    indexes: _Indexes,
    requests: tuple[_Request, ...],
    fences: tuple[_Fence, ...],
) -> tuple[tuple[_Request, ...], tuple[AtomicExpansion, ...]]:
    _validate_atomic_error_requests(bundle, requests)
    compiled_requests = list(requests)
    expansions: list[AtomicExpansion] = []
    changed = True
    while changed:
        changed = False
        for manifest in sorted(
            bundle.manifests,
            key=lambda item: _graph_sort_key(item.graph_instance_id),
        ):
            for endpoint in _PRECISION_ENDPOINTS:
                if not _endpoint_participates(manifest.lifecycle, endpoint):
                    continue
                for group in sorted(
                    manifest.atomic_groups, key=lambda item: item.group_id
                ):
                    for triggering_participant in group.participants:
                        triggers = tuple(
                            request
                            for request in _requests_for(
                                compiled_requests,
                                graph_instance_id=manifest.graph_instance_id,
                                endpoint=endpoint,
                                inventory_entry_id=(
                                    triggering_participant.inventory_entry_id
                                ),
                                precision="mxfp8",
                            )
                        )
                        for trigger in triggers:
                            if trigger.atomic_conflict != "expand":
                                continue
                            group_selection = _atomic_preimage(
                                group.group_domain,
                                triggering_participant.participant_domain,
                                triggering_participant.group_to_participant_axes,
                                trigger.member_domain,
                            )
                            if group_selection is None:
                                continue
                            missing_by_participant = _atomic_missing_by_participant(
                                compiled_requests,
                                manifest,
                                endpoint,
                                group,
                                group_selection,
                            )
                            if not missing_by_participant:
                                continue
                            additions: list[CompactDomainSelection] = []
                            for entry_id, missing in missing_by_participant:
                                if (
                                    entry_id
                                    in indexes.out_of_scope_by_graph[
                                        manifest.graph_instance_id
                                    ]
                                ):
                                    raise PrecisionPolicyError(
                                        f"atomic expansion from {trigger.scope_id} "
                                        "requires explicitly out-of-scope participant "
                                        f"{manifest.graph_instance_id}:{entry_id}"
                                    )
                                explicit_bf16 = tuple(
                                    request.member_domain
                                    for request in _requests_for(
                                        compiled_requests,
                                        graph_instance_id=manifest.graph_instance_id,
                                        endpoint=endpoint,
                                        inventory_entry_id=entry_id,
                                        precision="bf16",
                                    )
                                )
                                if _regions_intersect((missing,), explicit_bf16):
                                    raise PrecisionPolicyError(
                                        f"atomic expansion from {trigger.scope_id} "
                                        "conflicts with explicit BF16 precision"
                                    )
                                matching_fences = tuple(
                                    fence
                                    for fence in fences
                                    if fence.graph_instance_id
                                    == manifest.graph_instance_id
                                    and fence.endpoint == endpoint
                                    and fence.inventory_entry_id == entry_id
                                )
                                if _regions_intersect(
                                    (missing,),
                                    tuple(
                                        fence.member_domain for fence in matching_fences
                                    ),
                                ):
                                    raise PrecisionPolicyError(
                                        f"atomic expansion from {trigger.scope_id} "
                                        "crosses a hard BF16 layer boundary"
                                    )
                                entry = indexes.entries_by_id[entry_id]
                                compiled_requests.append(
                                    _Request(
                                        graph_instance_id=manifest.graph_instance_id,
                                        semantic_graph_path=_member_graph_path(
                                            entry.member
                                        ),
                                        inventory_entry_id=entry_id,
                                        member_domain=missing,
                                        endpoint=endpoint,
                                        precision="mxfp8",
                                        scope_id=trigger.scope_id,
                                        atomic_conflict="expand",
                                    )
                                )
                                additions.append(
                                    CompactDomainSelection(
                                        graph_instance_id=manifest.graph_instance_id,
                                        semantic_graph_path=_member_graph_path(
                                            entry.member
                                        ),
                                        inventory_entry_id=entry_id,
                                        member_domain=missing,
                                    )
                                )
                            expansions.append(
                                AtomicExpansion(
                                    graph_instance_id=manifest.graph_instance_id,
                                    endpoint=endpoint,
                                    atomic_group_id=group.group_id,
                                    triggering_scope_ids=(trigger.scope_id,),
                                    additions=_canonical_selections(additions),
                                )
                            )
                            changed = True
    return tuple(compiled_requests), tuple(
        sorted(
            expansions,
            key=lambda item: (
                _graph_sort_key(item.graph_instance_id),
                item.endpoint.value,
                item.atomic_group_id,
                item.triggering_scope_ids,
                tuple(_selection_sort_key(addition) for addition in item.additions),
            ),
        )
    )


def _requests_for(
    requests: tuple[_Request, ...] | list[_Request],
    *,
    graph_instance_id: str,
    endpoint: PrecisionEndpoint,
    inventory_entry_id: str,
    precision: PrecisionName | None = None,
) -> tuple[_Request, ...]:
    return tuple(
        request
        for request in requests
        if request.graph_instance_id == graph_instance_id
        and request.endpoint == endpoint
        and request.inventory_entry_id == inventory_entry_id
        and (precision is None or request.precision == precision)
    )


def _build_endpoint_plan(
    manifest: SemanticGraphManifest,
    endpoint: PrecisionEndpoint,
    indexes: _Indexes,
    requests: tuple[_Request, ...] | list[_Request],
) -> EndpointPrecisionPlan | None:
    if not _endpoint_participates(manifest.lifecycle, endpoint):
        return None
    assignments: list[CompactPrecisionAssignment] = []
    out_of_scope = indexes.out_of_scope_by_graph[manifest.graph_instance_id]
    for entry_id in manifest.inventory_entry_ids:
        if entry_id in out_of_scope:
            continue
        entry = indexes.entries_by_id[entry_id]
        full_domain = _member_domain(entry.member)
        mxfp8_regions = _canonical_regions(
            [
                request.member_domain
                for request in _requests_for(
                    requests,
                    graph_instance_id=manifest.graph_instance_id,
                    endpoint=endpoint,
                    inventory_entry_id=entry_id,
                    precision="mxfp8",
                )
            ]
        )
        bf16_regions = _subtract_regions(full_domain, mxfp8_regions)
        graph_path = _member_graph_path(entry.member)
        assignments.extend(
            CompactPrecisionAssignment(
                graph_instance_id=manifest.graph_instance_id,
                semantic_graph_path=graph_path,
                inventory_entry_id=entry_id,
                member_domain=domain,
                precision="bf16",
                requested_format=BF16_FORMAT,
            )
            for domain in bf16_regions
        )
        assignments.extend(
            CompactPrecisionAssignment(
                graph_instance_id=manifest.graph_instance_id,
                semantic_graph_path=graph_path,
                inventory_entry_id=entry_id,
                member_domain=domain,
                precision="mxfp8",
                requested_format=MXFP8_FORMAT,
            )
            for domain in mxfp8_regions
        )
    assignments.sort(
        key=lambda item: (
            item.semantic_graph_path,
            item.inventory_entry_id,
            item.precision,
            _domain_sort_key(item.member_domain),
        )
    )
    return EndpointPrecisionPlan(
        graph_instance_id=manifest.graph_instance_id,
        endpoint=endpoint,
        assignments=tuple(assignments),
    )


def _owner_requirements(
    manifest: SemanticGraphManifest,
    bundle: SemanticManifestBundle,
) -> tuple[OwnerRefitRequirements, RefitRequirement]:
    requirements, summary = _derive_refit_requirements_unchecked(
        bundle,
        manifest.graph_instance_id,
    )
    return OwnerRefitRequirements(requirements), summary


def _evidence_source_payload(source: EvidenceSource) -> dict[str, object]:
    return {
        "kind": source.kind.value,
        "locator": source.locator,
        "digest": source.digest,
    }


def _immutable_evidence_payload(
    evidence: ImmutableAuxiliaryEvidence,
) -> dict[str, object]:
    return {
        "graph_instance_id": evidence.graph_instance_id,
        "model_identity": evidence.model_identity,
        "pinned_checkpoint_revision": evidence.pinned_checkpoint_revision,
        "checkpoint_content_digest": evidence.checkpoint_content_digest,
        "model_config_digest": evidence.model_config_digest,
        "semantic_domain_digest": evidence.semantic_domain_digest,
        "evidence_source": _evidence_source_payload(evidence.evidence_source),
    }


def _source_alias_contract_payload(
    contract: SourceAliasContract,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "alias_entry_id": contract.alias_entry_id,
        "canonical_value_entry_id": contract.canonical_value_entry_id,
        "canonical_owner_family": _owner_reference_payload(
            contract.canonical_owner_family
        ),
        "component_role": str(contract.component_role),
        "alias_domain": _domain_payload(contract.alias_domain),
        "canonical_domain": _domain_payload(contract.canonical_domain),
        "alias_to_canonical_axes": [
            _projection_payload(projection)
            for projection in contract.alias_to_canonical_axes
        ],
    }
    if isinstance(contract, IdenticalStorageSourceAliasContract):
        payload.update(
            {
                "kind": "identical_storage",
                "storage_identity_evidence": _evidence_source_payload(
                    contract.storage_identity_evidence
                ),
            }
        )
    else:
        payload.update(
            {
                "kind": "synchronized_replica",
                "synchronization": {
                    "replica_group_id": contract.synchronization.replica_group_id,
                    "boundary": contract.synchronization.boundary.value,
                    "evidence_source": _evidence_source_payload(
                        contract.synchronization.evidence_source
                    ),
                },
            }
        )
    return payload


type _AxisStructuralKey = tuple[int, int | str]
type _DomainStructuralKey = tuple[
    bool,
    tuple[tuple[int, int], ...],
    tuple[tuple[str, tuple[_AxisStructuralKey, ...]], ...],
]
type _DomainStructuralKeyCache = dict[
    int,
    tuple[FamilyIndexDomain, _DomainStructuralKey],
]
type _SourceAliasRelationStructuralKey = tuple[int, str, str, str, str, str]
type _SourceAliasContractStructuralKey = tuple[
    str,
    str,
    _DomainStructuralKey,
    str,
    _DomainStructuralKey,
    str,
    str,
    tuple[tuple[str, str], ...],
    _SourceAliasRelationStructuralKey,
]


def _domain_structural_key(
    domain: FamilyIndexDomain,
    cache: _DomainStructuralKeyCache,
) -> _DomainStructuralKey:
    cached = cache.get(id(domain))
    if cached is not None and cached[0] is domain:
        return cached[1]
    layer_members = (
        ()
        if domain.layer_domain is None
        else tuple(
            (
                member.global_decoder_layer,
                -1 if member.moe_ordinal is None else member.moe_ordinal,
            )
            for member in domain.layer_domain.members
        )
    )
    key: _DomainStructuralKey = (
        domain.layer_domain is not None,
        layer_members,
        tuple(
            (
                axis.name,
                tuple(_axis_key(member) for member in axis.members),
            )
            for axis in domain.independent_axes
        ),
    )
    cache[id(domain)] = (domain, key)
    return key


def _source_alias_contract_structural_key(
    contract: SourceAliasContract,
    domain_key_cache: _DomainStructuralKeyCache,
) -> _SourceAliasContractStructuralKey:
    if isinstance(contract, IdenticalStorageSourceAliasContract):
        evidence = contract.storage_identity_evidence
        relation_key: _SourceAliasRelationStructuralKey = (
            0,
            "",
            "",
            evidence.kind.value,
            evidence.locator,
            evidence.digest,
        )
    else:
        evidence = contract.synchronization.evidence_source
        relation_key = (
            1,
            contract.synchronization.replica_group_id,
            contract.synchronization.boundary.value,
            evidence.kind.value,
            evidence.locator,
            evidence.digest,
        )
    return (
        contract.alias_entry_id,
        str(contract.component_role),
        _domain_structural_key(contract.alias_domain, domain_key_cache),
        contract.canonical_value_entry_id,
        _domain_structural_key(contract.canonical_domain, domain_key_cache),
        contract.canonical_owner_family.graph_instance_id,
        contract.canonical_owner_family.owner_family_id,
        tuple(
            (projection.member_axis, projection.owner_axis)
            for projection in contract.alias_to_canonical_axes
        ),
        relation_key,
    )


def _canonical_source_alias_contracts(
    contracts: tuple[SourceAliasContract, ...],
) -> tuple[SourceAliasContract, ...]:
    domain_key_cache: _DomainStructuralKeyCache = {}
    keyed = tuple(
        (
            _source_alias_contract_structural_key(contract, domain_key_cache),
            contract,
        )
        for contract in contracts
    )
    if all(keyed[index - 1][0] <= keyed[index][0] for index in range(1, len(keyed))):
        ordered = contracts
    else:
        ordered = tuple(
            contract for _, contract in sorted(keyed, key=lambda item: item[0])
        )
    if any(ordered[index - 1] == ordered[index] for index in range(1, len(ordered))):
        raise ValueError("duplicate source alias contract")
    return ordered


def _lifecycle_payload(lifecycle: GraphLifecycle) -> dict[str, object]:
    return {
        "graph_kind": lifecycle.graph_kind.value,
        "graph_provenance": lifecycle.graph_provenance.value,
        "rollout_participation": lifecycle.rollout_participation.value,
        "immutable_evidence": None
        if lifecycle.immutable_evidence is None
        else _immutable_evidence_payload(lifecycle.immutable_evidence),
    }


def _projection_payload(projection: AxisProjection) -> dict[str, object]:
    return {
        "member_axis": projection.member_axis,
        "owner_axis": projection.owner_axis,
    }


def _owner_reference_payload(reference: OwnerFamilyReference) -> dict[str, object]:
    return {
        "graph_instance_id": reference.graph_instance_id,
        "owner_family_id": reference.owner_family_id,
    }


def _member_fixed_payload(member: SemanticInventoryMember) -> dict[str, object]:
    if isinstance(member, SemanticTensor):
        address: SemanticAddress | SemanticAddressPattern = member.address
        identity: dict[str, object] = {
            "kind": "tensor",
            "semantic_id": member.address.semantic_id,
            "global_decoder_layer": member.address.global_decoder_layer,
            "moe_ordinal": member.address.moe_ordinal,
        }
    else:
        address = member.pattern
        identity = {
            "kind": "family",
            "path_segments": [
                {"kind": "literal", "value": segment.value}
                if isinstance(segment, LiteralPathSegment)
                else {"kind": "index", "axis_name": segment.axis_name}
                for segment in member.pattern.path_segments
            ],
            "domain": _domain_payload(member.domain),
        }
    binding = member.ownership.binding
    identity.update(
        {
            "semantic_graph_path": address.semantic_graph_path,
            "model_part": address.model_part,
            "module_kind": address.module_kind,
            "attributes": [
                {"name": name, "value": _scalar_payload(value)}
                for name, value in address.attributes
            ],
            "parameter_role": address.parameter_role,
            "format": _format_payload(member.format),
            "logical_dtype": member.logical_dtype,
            "logical_shape": list(member.logical_shape),
            "logical_axes": list(member.logical_axes),
            "ownership": {
                "canonical_owner_family": _owner_reference_payload(
                    binding.canonical_owner_family
                ),
                "canonical_value_entry_id": binding.canonical_value_entry_id,
                "member_domain": _domain_payload(binding.member_domain),
                "member_to_owner_axes": [
                    _projection_payload(item) for item in binding.member_to_owner_axes
                ],
                "member_to_value_axes": [
                    _projection_payload(item) for item in binding.member_to_value_axes
                ],
            },
        }
    )
    return identity


def _atomic_group_payload(group: AtomicGroup) -> dict[str, object]:
    return {
        "group_id": group.group_id,
        "graph_instance_id": group.graph_instance_id,
        "kind": group.kind.value,
        "group_domain": _domain_payload(group.group_domain),
        "participants": [
            {
                "inventory_entry_id": participant.inventory_entry_id,
                "participant_domain": _domain_payload(participant.participant_domain),
                "group_to_participant_axes": [
                    _projection_payload(projection)
                    for projection in participant.group_to_participant_axes
                ],
            }
            for participant in group.participants
        ],
    }


def _bundle_payload(bundle: SemanticManifestBundle) -> dict[str, object]:
    return {
        "schema_version": bundle.schema_version,
        "expected_graphs": [
            {
                "graph_instance_id": declaration.graph_instance_id,
                "model_identity": declaration.model_identity,
                "lifecycle": _lifecycle_payload(declaration.lifecycle),
            }
            for declaration in bundle.expected_graphs
        ],
        "manifests": [
            {
                "model_family": manifest.model_family,
                "model_revision": manifest.model_revision,
                "graph_instance_id": manifest.graph_instance_id,
                "lifecycle": _lifecycle_payload(manifest.lifecycle),
                "inventory_entry_ids": list(manifest.inventory_entry_ids),
                "atomic_groups": [
                    _atomic_group_payload(group) for group in manifest.atomic_groups
                ],
                "out_of_scope": [
                    {
                        "inventory_entry_id": item.inventory_entry_id,
                        "reason": item.reason.value,
                    }
                    for item in manifest.out_of_scope
                ],
            }
            for manifest in bundle.manifests
        ],
        "inventory": {
            "owners": [
                {
                    "owner_family": _owner_reference_payload(owner.owner_family),
                    "domain": _domain_payload(owner.domain),
                    "source_mutability": owner.source_mutability.value,
                    "mutability_evidence_source": _evidence_source_payload(
                        owner.mutability_evidence_source
                    ),
                }
                for owner in bundle.inventory.owners
            ],
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "graph_instance_id": entry.graph_instance_id,
                    "member": _member_fixed_payload(entry.member),
                    "value_provenance": entry.value_provenance.value,
                }
                for entry in bundle.inventory.entries
            ],
        },
        "role_definitions": [
            {
                "schema_version": role.schema_version,
                "role_name": role.role_name,
                "predicate": {
                    "graph_kinds": [kind.value for kind in role.predicate.graph_kinds],
                    "semantic_graph_paths": list(role.predicate.semantic_graph_paths),
                    "model_parts": list(role.predicate.model_parts),
                    "module_kinds": list(role.predicate.module_kinds),
                    "attributes": [
                        {
                            "name": attribute.name,
                            "allowed_values": [
                                _scalar_payload(value)
                                for value in attribute.allowed_values
                            ],
                        }
                        for attribute in role.predicate.attributes
                    ],
                    "parameter_roles": list(role.predicate.parameter_roles),
                },
                "expected_domain": {
                    "role_name": role.expected_domain.role_name,
                    "inventory_entry_ids": list(
                        role.expected_domain.inventory_entry_ids
                    ),
                },
            }
            for role in bundle.role_definitions
        ],
        "source_alias_contracts": [
            _source_alias_contract_payload(contract)
            for contract in bundle.source_alias_contracts
        ],
    }


def _predicate_config_payload(value: object) -> object:
    if isinstance(value, list):
        scalars = sorted(
            (_scalar_payload(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
        return scalars
    if isinstance(value, (str, int, float, bool)):
        return _scalar_payload(value)
    if value is None:
        return None
    raise TypeError("unsupported policy predicate value")


def _policy_payload(policy: PrecisionPolicyConfig) -> dict[str, object]:
    scopes: list[dict[str, object]] = []
    for scope in sorted(policy.scopes, key=lambda item: item.id):
        advanced: dict[str, object] | None = None
        if scope.advanced_match is not None:
            advanced = {
                "graph_instance_id": _predicate_config_payload(
                    scope.advanced_match.graph_instance_id
                ),
                "semantic_graph_path": _predicate_config_payload(
                    scope.advanced_match.semantic_graph_path
                ),
                "model_part": _predicate_config_payload(
                    scope.advanced_match.model_part
                ),
                "module_kind": _predicate_config_payload(
                    scope.advanced_match.module_kind
                ),
                "parameter_role": _predicate_config_payload(
                    scope.advanced_match.parameter_role
                ),
                "attributes": [
                    {
                        "name": name,
                        "predicate": _predicate_config_payload(predicate),
                    }
                    for name, predicate in sorted(
                        scope.advanced_match.attributes.items()
                    )
                ],
            }
        scopes.append(
            {
                "id": scope.id,
                "roles": None if scope.roles is None else sorted(scope.roles),
                "advanced_match": advanced,
                "addresses": None
                if scope.addresses is None
                else [
                    {
                        "graph_instance_id": address.graph_instance_id,
                        "semantic_graph_path": address.semantic_graph_path,
                        "semantic_id": address.semantic_id,
                    }
                    for address in sorted(
                        scope.addresses,
                        key=lambda item: (
                            _graph_sort_key(item.graph_instance_id),
                            item.semantic_graph_path,
                            item.semantic_id,
                        ),
                    )
                ],
                "layers": None
                if scope.layers is None
                else {
                    "index_space": scope.layers.index_space,
                    "exclude_first": scope.layers.exclude_first,
                    "exclude_last": scope.layers.exclude_last,
                },
                "training": scope.training,
                "rollout": scope.rollout,
                "atomic_conflict": scope.atomic_conflict,
            }
        )
    return {
        "schema_version": policy.schema_version,
        "default": policy.default,
        "require_match": policy.require_match,
        "atomic_conflict": policy.atomic_conflict,
        "scopes": scopes,
    }


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def _selection_payload(selection: CompactDomainSelection) -> dict[str, object]:
    return {
        "graph_instance_id": selection.graph_instance_id,
        "semantic_graph_path": selection.semantic_graph_path,
        "inventory_entry_id": selection.inventory_entry_id,
        "member_domain": _domain_payload(selection.member_domain),
        "logical_cardinality": selection.logical_cardinality,
    }


def _layer_selection_payload(record: LayerSelectionRecord) -> dict[str, object]:
    return {
        "scope_id": record.scope_id,
        "graph_instance_id": record.graph_instance_id,
        "semantic_graph_path": record.semantic_graph_path,
        "index_space": record.index_space,
        "universe_coordinates": list(record.universe_coordinates),
        "retained_coordinates": list(record.retained_coordinates),
        "selected_layer_members": [
            {
                "global_decoder_layer": item.global_decoder_layer,
                "moe_ordinal": item.moe_ordinal,
            }
            for item in record.selected_layer_members
        ],
    }


def _scope_graph_payload(result: CompiledScopeGraphResult) -> dict[str, object]:
    return {
        "scope_id": result.scope_id,
        "graph_instance_id": result.graph_instance_id,
        "matched": [_selection_payload(item) for item in result.matched],
        "selected": [_selection_payload(item) for item in result.selected],
        "layer_selections": [
            _layer_selection_payload(item) for item in result.layer_selections
        ],
        "training_precision": result.training_precision,
        "rollout_precision": result.rollout_precision,
        "out_of_scope_matches": list(result.out_of_scope_matches),
    }


def _scope_payload(result: CompiledScopeResult) -> dict[str, object]:
    return {
        "scope_id": result.scope_id,
        "selector_kind": result.selector_kind,
        "atomic_conflict": result.atomic_conflict,
        "graph_results": [_scope_graph_payload(item) for item in result.graph_results],
    }


def _assignment_payload(
    assignment: CompactPrecisionAssignment,
) -> dict[str, object]:
    return {
        "graph_instance_id": assignment.graph_instance_id,
        "semantic_graph_path": assignment.semantic_graph_path,
        "inventory_entry_id": assignment.inventory_entry_id,
        "member_domain": _domain_payload(assignment.member_domain),
        "precision": assignment.precision,
        "requested_format": _format_payload(assignment.requested_format),
    }


def _endpoint_plan_payload(plan: EndpointPrecisionPlan | None) -> object:
    if plan is None:
        return None
    return {
        "graph_instance_id": plan.graph_instance_id,
        "endpoint": plan.endpoint.value,
        "assignments": [_assignment_payload(item) for item in plan.assignments],
    }


def _atomic_expansion_payload(expansion: AtomicExpansion) -> dict[str, object]:
    return {
        "graph_instance_id": expansion.graph_instance_id,
        "endpoint": expansion.endpoint.value,
        "atomic_group_id": expansion.atomic_group_id,
        "triggering_scope_ids": list(expansion.triggering_scope_ids),
        "additions": [_selection_payload(item) for item in expansion.additions],
    }


def _graph_intent_payload(
    intent: CompiledGraphPrecisionIntent,
    *,
    include_intent_id: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "graph_instance_id": intent.graph_instance_id,
        "model_family": intent.model_family,
        "model_revision": intent.model_revision,
        "lifecycle": _lifecycle_payload(intent.lifecycle),
        "topology_digest": intent.topology_digest,
        "policy_digest": intent.policy_digest,
        "training_plan": _endpoint_plan_payload(intent.training_plan),
        "rollout_plan": _endpoint_plan_payload(intent.rollout_plan),
        "scope_results": [_scope_graph_payload(item) for item in intent.scope_results],
        "atomic_expansions": [
            _atomic_expansion_payload(item) for item in intent.atomic_expansions
        ],
        "owner_refit_requirements": [
            {
                "owner_family": _owner_reference_payload(reference),
                "requirement": requirement.value,
            }
            for reference, requirement in intent.owner_refit_requirements.entries
        ],
        "refit_requirement": intent.refit_requirement.value,
        "startup_owner_requests": [
            _owner_reference_payload(item) for item in intent.startup_owner_requests
        ],
        "every_version_owner_requests": [
            _owner_reference_payload(item)
            for item in intent.every_version_owner_requests
        ],
        "immutable_checkpoint_evidence": None
        if intent.immutable_checkpoint_evidence is None
        else _immutable_evidence_payload(intent.immutable_checkpoint_evidence),
        "out_of_scope_inventory_entry_ids": list(
            intent.out_of_scope_inventory_entry_ids
        ),
    }
    if include_intent_id:
        payload["intent_id"] = intent.intent_id
    return payload


def _owner_request_payload(request: OwnerRealizationRequest) -> dict[str, object]:
    return {
        "owner_family": _owner_reference_payload(request.owner_family),
        "requirement": request.requirement.value,
        "member_graph_instance_ids": list(request.member_graph_instance_ids),
        "inventory_entry_ids": [list(item) for item in request.inventory_entry_ids],
    }


def _compiled_group_payload(
    group: CompiledPrecisionIntentGroup,
    *,
    include_group_id: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": group.schema_version,
        "topology_digest": group.topology_digest,
        "policy_digest": group.policy_digest,
        "graph_intents": [
            _graph_intent_payload(intent, include_intent_id=True)
            for intent in group.graph_intents
        ],
        "scope_results": [_scope_payload(item) for item in group.scope_results],
        "atomic_expansions": [
            _atomic_expansion_payload(item) for item in group.atomic_expansions
        ],
        "startup_source_items": [
            _owner_request_payload(item) for item in group.startup_source_items
        ],
        "every_version_source_items": [
            _owner_request_payload(item) for item in group.every_version_source_items
        ],
        "immutable_checkpoint_contexts": [
            _immutable_evidence_payload(item)
            for item in group.immutable_checkpoint_contexts
        ],
        "source_alias_contracts": [
            _source_alias_contract_payload(contract)
            for contract in group.source_alias_contracts
        ],
    }
    if include_group_id:
        payload["intent_group_id"] = group.intent_group_id
    return payload


def _build_owner_realization_requests(
    graph_intents: tuple[CompiledGraphPrecisionIntent, ...],
    indexes: _Indexes,
    source_required_entry_ids_by_graph: Mapping[str, tuple[str, ...]],
    requirement: RefitRequirement,
) -> tuple[OwnerRealizationRequest, ...]:
    entries_by_owner: dict[OwnerFamilyReference, set[tuple[str, str]]] = {}
    for intent in graph_intents:
        requirements_by_owner: dict[OwnerFamilyReference, RefitRequirement] = {
            owner_family: owner_requirement
            for owner_family, owner_requirement in intent.owner_refit_requirements.entries
        }
        for entry_id in source_required_entry_ids_by_graph[intent.graph_instance_id]:
            entry = indexes.entries_by_id[entry_id]
            owner_family = entry.member.ownership.binding.canonical_owner_family
            if requirements_by_owner[owner_family] != requirement:
                continue
            entries_by_owner.setdefault(owner_family, set()).add(
                (intent.graph_instance_id, entry_id)
            )
            if entry.value_provenance == ValueProvenance.CANONICAL_ALIAS:
                target = indexes.entries_by_id[
                    entry.member.ownership.binding.canonical_value_entry_id
                ]
                entries_by_owner[owner_family].add(
                    (target.graph_instance_id, target.entry_id)
                )
    return tuple(
        OwnerRealizationRequest(
            owner_family=owner_family,
            requirement=requirement,
            member_graph_instance_ids=tuple(
                sorted(
                    {
                        graph_instance_id
                        for graph_instance_id, _ in entries_by_owner[owner_family]
                    },
                    key=_graph_sort_key,
                )
            ),
            inventory_entry_ids=tuple(
                sorted(
                    entries_by_owner[owner_family],
                    key=lambda item: (_graph_sort_key(item[0]), item[1]),
                )
            ),
        )
        for owner_family in sorted(
            entries_by_owner,
            key=lambda item: (
                _graph_sort_key(item.graph_instance_id),
                item.owner_family_id,
            ),
        )
    )


def compile_precision_policy(
    policy: PrecisionPolicyConfig,
    bundle: SemanticManifestBundle,
) -> CompiledPrecisionIntentGroup:
    """Compile a policy against one complete semantic bundle in explicit passes."""
    if policy.schema_version != bundle.schema_version:
        raise PrecisionPolicyError(
            "policy and semantic bundle schema versions differ: "
            f"{policy.schema_version} != {bundle.schema_version}"
        )

    bundle.validate_complete()
    indexes = _build_indexes(bundle)

    scope_results: list[CompiledScopeResult] = []
    requests: list[_Request] = []
    fences: list[_Fence] = []
    for scope in sorted(policy.scopes, key=lambda item: item.id):
        result, scope_requests, scope_fences = _compile_scope(
            policy, scope, bundle, indexes
        )
        scope_results.append(result)
        requests.extend(scope_requests)
        fences.extend(scope_fences)
    _validate_explicit_requests(tuple(requests), tuple(fences))
    compiled_requests, atomic_expansions = _apply_atomic_closure(
        bundle, indexes, tuple(requests), tuple(fences)
    )
    _validate_explicit_requests(compiled_requests, tuple(fences))
    topology_digest = _digest(_bundle_payload(bundle))
    policy_digest = _digest(_policy_payload(policy))
    graph_intents: list[CompiledGraphPrecisionIntent] = []
    for manifest in bundle.manifests:
        requirements, summary = _owner_requirements(manifest, bundle)
        graph_scope_results = tuple(
            graph_result
            for scope_result in scope_results
            for graph_result in scope_result.graph_results
            if graph_result.graph_instance_id == manifest.graph_instance_id
        )
        graph_expansions = tuple(
            item
            for item in atomic_expansions
            if item.graph_instance_id == manifest.graph_instance_id
        )
        provisional = CompiledGraphPrecisionIntent(
            graph_instance_id=manifest.graph_instance_id,
            model_family=manifest.model_family,
            model_revision=manifest.model_revision,
            lifecycle=manifest.lifecycle,
            topology_digest=topology_digest,
            policy_digest=policy_digest,
            training_plan=_build_endpoint_plan(
                manifest,
                PrecisionEndpoint.TRAINING,
                indexes,
                compiled_requests,
            ),
            rollout_plan=_build_endpoint_plan(
                manifest,
                PrecisionEndpoint.ROLLOUT,
                indexes,
                compiled_requests,
            ),
            scope_results=graph_scope_results,
            atomic_expansions=graph_expansions,
            owner_refit_requirements=requirements,
            refit_requirement=summary,
            startup_owner_requests=tuple(
                reference
                for reference, requirement in requirements.entries
                if requirement == RefitRequirement.INITIAL_ONLY
            ),
            every_version_owner_requests=tuple(
                reference
                for reference, requirement in requirements.entries
                if requirement == RefitRequirement.EVERY_VERSION
            ),
            immutable_checkpoint_evidence=manifest.lifecycle.immutable_evidence,
            out_of_scope_inventory_entry_ids=tuple(
                sorted(indexes.out_of_scope_by_graph[manifest.graph_instance_id])
            ),
            intent_id="",
        )
        graph_intents.append(
            replace(
                provisional,
                intent_id=_digest(
                    _graph_intent_payload(provisional, include_intent_id=False)
                ),
            )
        )
    canonical_graph_intents = tuple(
        sorted(
            graph_intents,
            key=lambda item: _graph_sort_key(item.graph_instance_id),
        )
    )
    source_required_entry_ids_by_graph = {
        manifest.graph_instance_id: _source_required_entry_ids_unchecked(
            bundle,
            manifest.graph_instance_id,
        )
        for manifest in bundle.manifests
    }
    startup_source_items = _build_owner_realization_requests(
        canonical_graph_intents,
        indexes,
        source_required_entry_ids_by_graph,
        RefitRequirement.INITIAL_ONLY,
    )
    every_version_source_items = _build_owner_realization_requests(
        canonical_graph_intents,
        indexes,
        source_required_entry_ids_by_graph,
        RefitRequirement.EVERY_VERSION,
    )
    checkpoint_contexts = tuple(
        intent.immutable_checkpoint_evidence
        for intent in canonical_graph_intents
        if intent.immutable_checkpoint_evidence is not None
    )
    return CompiledPrecisionIntentGroup(
        schema_version=policy.schema_version,
        topology_digest=topology_digest,
        policy_digest=policy_digest,
        graph_intents=canonical_graph_intents,
        scope_results=tuple(scope_results),
        atomic_expansions=atomic_expansions,
        startup_source_items=startup_source_items,
        every_version_source_items=every_version_source_items,
        immutable_checkpoint_contexts=checkpoint_contexts,
        source_alias_contracts=bundle.source_alias_contracts,
    )
