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

"""Backend-independent semantic model inventory and lifecycle contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from math import isfinite, prod
from typing import NewType


class GraphKind(StrEnum):
    """Declared semantic function of a graph instance."""

    MAIN = "main"
    MTP = "mtp"
    SPECULATIVE_DRAFTER = "speculative_drafter"


class GraphProvenance(StrEnum):
    """Authority that instantiated a graph."""

    TRAINING_RUNTIME = "training_runtime"
    MODEL_CHECKPOINT = "model_checkpoint"
    EXTERNAL_CHECKPOINT = "external_checkpoint"


class ValueProvenance(StrEnum):
    """Origin of one semantic inventory value."""

    TRAINING_PARAMETER = "training_parameter"
    CHECKPOINT_ENCODING_COMPONENT = "checkpoint_encoding_component"
    BACKEND_DERIVED = "backend_derived"
    TIED_ALIAS = "tied_alias"


class SourceMutability(StrEnum):
    """Source-discovery mutability of a canonical owner."""

    MUTABLE = "mutable"
    FROZEN = "frozen"
    ABSENT = "absent"


class RolloutParticipation(StrEnum):
    """How rollout obtains a graph."""

    NOT_SERVED = "not_served"
    SERVED_FROM_SOURCE = "served_from_source"
    SERVED_FROM_CHECKPOINT = "served_from_checkpoint"


class RefitRequirement(StrEnum):
    """Transiently derived refit cadence requirement."""

    NONE = "none"
    INITIAL_ONLY = "initial_only"
    EVERY_VERSION = "every_version"


class EvidenceSourceKind(StrEnum):
    """Typed authority for an evidence digest."""

    RUNTIME_INVENTORY = "runtime_inventory"
    PINNED_CHECKPOINT_MANIFEST = "pinned_checkpoint_manifest"
    CONTENT_ADDRESS = "content_address"


class OutOfScopeReason(StrEnum):
    """Closed reasons that a complete inventory entry is not refitted."""

    SOURCE_PROVEN_FROZEN = "source_proven_frozen"
    IMMUTABLE_AUXILIARY = "immutable_auxiliary"
    BACKEND_DERIVED_STATE = "backend_derived_state"


class AtomicGroupKind(StrEnum):
    """Semantic atomicity relation kind."""

    PRECISION = "precision"


type PredicateScalar = str | int | float | bool
type AxisMember = int | str
ComponentRole = NewType("ComponentRole", str)
LOGICAL_VALUES = ComponentRole("logical_values")
VALUES = ComponentRole("values")
BLOCK_SCALES = ComponentRole("block_scales")


_LOGICAL_AXES = frozenset(
    {
        "channels",
        "components",
        "embedding",
        "experts",
        "groups",
        "head_dim",
        "head_groups",
        "heads",
        "input_features",
        "intermediate_features",
        "kernel_height",
        "kernel_width",
        "layers",
        "ngram",
        "output_features",
        "scalar",
        "state",
        "tokens",
        "vocabulary",
    }
)
_LAYER_AXES = ("global_decoder_layer", "moe_ordinal")


def _require_enum(value: object, enum_type: type[StrEnum], name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be {enum_type.__name__}")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    if any(character.isspace() for character in value):
        raise ValueError(f"{name} must not contain whitespace")
    return value


def _is_atom_character(character: str) -> bool:
    return character.isascii() and (character.isalnum() or character in {"_", "-"})


def _require_atom(value: object, name: str) -> str:
    text = _require_text(value, name)
    if not all(_is_atom_character(character) for character in text):
        raise ValueError(f"{name} must be a canonical atom")
    return text


def _require_dotted_name(
    value: object,
    name: str,
    *,
    minimum_parts: int = 1,
) -> str:
    text = _require_text(value, name)
    parts = text.split(".")
    if len(parts) < minimum_parts or any(
        not part or not all(_is_atom_character(character) for character in part)
        for part in parts
    ):
        raise ValueError(f"{name} must be a canonical dotted name")
    return text


def _graph_sort_key(graph_instance_id: str) -> tuple[int, str]:
    return (0 if graph_instance_id == "main" else 1, graph_instance_id)


def _validate_graph_instance_id(
    graph_instance_id: str,
    graph_kind: GraphKind | None = None,
) -> None:
    _require_dotted_name(graph_instance_id, "graph_instance_id")
    if graph_instance_id == "main":
        if graph_kind is not None and graph_kind != GraphKind.MAIN:
            raise ValueError("graph_instance_id main requires GraphKind.MAIN")
        return
    prefix, separator, suffix = graph_instance_id.partition(".")
    if not separator or not suffix or prefix not in {"mtp", "draft"}:
        raise ValueError("graph_instance_id must be main, mtp.*, or draft.*")
    expected_kind = GraphKind.MTP if prefix == "mtp" else GraphKind.SPECULATIVE_DRAFTER
    if graph_kind is not None and graph_kind != expected_kind:
        raise ValueError("graph_instance_id does not match its graph kind")


def _typed_scalar_key(value: PredicateScalar) -> tuple[int, object]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("semantic float values must be finite")
        return (2, value)
    if isinstance(value, str):
        _require_text(value, "semantic scalar")
        return (3, value)
    raise TypeError("semantic attribute values must be str, int, float, or bool")


def _canonical_scalars(
    values: tuple[PredicateScalar, ...],
    name: str,
) -> tuple[PredicateScalar, ...]:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    keyed = [(_typed_scalar_key(value), value) for value in values]
    if len({key for key, _ in keyed}) != len(keyed):
        raise ValueError(f"{name} contains duplicate typed values")
    return tuple(value for _, value in sorted(keyed, key=lambda item: item[0]))


def _canonical_attributes(
    attributes: tuple[tuple[str, PredicateScalar], ...],
) -> tuple[tuple[str, PredicateScalar], ...]:
    canonical: list[tuple[str, PredicateScalar]] = []
    keys: set[str] = set()
    for key, value in attributes:
        _require_atom(key, "attribute name")
        if key in keys:
            raise ValueError(f"duplicate attribute: {key}")
        _typed_scalar_key(value)
        keys.add(key)
        canonical.append((key, value))
    return tuple(sorted(canonical, key=lambda item: item[0]))


def _validate_logical_shape(
    logical_shape: tuple[int, ...],
    logical_axes: tuple[str, ...],
) -> None:
    if len(logical_shape) != len(logical_axes):
        raise ValueError("logical shape and axes must have the same rank")
    if not logical_shape:
        raise ValueError("logical shape and axes must be non-empty")
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in logical_shape
    ):
        raise ValueError("logical shape dimensions must be positive integers")
    if len(set(logical_axes)) != len(logical_axes):
        raise ValueError("logical axes must be unique")
    for axis in logical_axes:
        _require_dotted_name(axis, "logical axis")
        if axis not in _LOGICAL_AXES and "." not in axis:
            raise ValueError(f"unknown logical axis: {axis}")


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    """One ordered canonical component of a logical encoding."""

    role: ComponentRole
    dtype: str
    encoding: str | None = None
    block_size: int | None = None
    block_axis: str | None = None

    def __post_init__(self) -> None:
        _require_atom(self.role, "component role")
        _require_atom(self.dtype, "component dtype")
        if self.encoding is not None:
            _require_atom(self.encoding, "component encoding")
        if (self.block_size is None) != (self.block_axis is None):
            raise ValueError("component block_size and block_axis must appear together")
        if self.block_size is not None and (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise ValueError("component block_size must be a positive integer")
        if self.block_axis is not None:
            _require_dotted_name(self.block_axis, "component block_axis")


@dataclass(frozen=True, slots=True)
class FormatDescriptor:
    """Logical encoding identity with ordered canonical components."""

    format_id: str
    family: str
    components: tuple[ComponentDescriptor, ...]

    def __post_init__(self) -> None:
        _require_dotted_name(self.format_id, "format_id")
        _require_dotted_name(self.family, "format family")
        if not self.components:
            raise ValueError("format descriptor must contain at least one component")
        if any(not isinstance(item, ComponentDescriptor) for item in self.components):
            raise TypeError("format components must be ComponentDescriptor records")
        roles = [item.role for item in self.components]
        if len(roles) != len(set(roles)):
            raise ValueError("format descriptor contains a duplicate component role")


BF16_FORMAT = FormatDescriptor(
    format_id="bf16.logical.v1",
    family="bf16",
    components=(ComponentDescriptor(role=LOGICAL_VALUES, dtype="bfloat16"),),
)
MXFP8_FORMAT = FormatDescriptor(
    format_id="mxfp8.e4m3-e8m0-block32-input-features.v1",
    family="mxfp8",
    components=(
        ComponentDescriptor(role=VALUES, dtype="e4m3"),
        ComponentDescriptor(
            role=BLOCK_SCALES,
            dtype="e8m0",
            encoding="mxfp8_scale",
            block_size=32,
            block_axis="input_features",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class AttributePredicate:
    """Exact finite predicate over one semantic attribute."""

    name: str
    allowed_values: tuple[PredicateScalar, ...]

    def __post_init__(self) -> None:
        _require_atom(self.name, "attribute predicate name")
        object.__setattr__(
            self,
            "allowed_values",
            _canonical_scalars(self.allowed_values, "attribute allowed_values"),
        )


@dataclass(frozen=True, slots=True)
class SemanticPredicate:
    """Structured, exact role predicate over fixed semantic facets."""

    graph_kinds: tuple[GraphKind, ...]
    semantic_graph_paths: tuple[str, ...]
    model_parts: tuple[str, ...]
    module_kinds: tuple[str, ...]
    attributes: tuple[AttributePredicate, ...]
    parameter_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        for graph_kind in self.graph_kinds:
            _require_enum(graph_kind, GraphKind, "predicate graph kind")
        object.__setattr__(
            self,
            "graph_kinds",
            tuple(sorted(set(self.graph_kinds), key=str)),
        )
        object.__setattr__(
            self,
            "semantic_graph_paths",
            _canonical_names(
                self.semantic_graph_paths,
                "semantic graph path",
                dotted=True,
            ),
        )
        object.__setattr__(
            self,
            "model_parts",
            _canonical_names(self.model_parts, "model part", dotted=True),
        )
        object.__setattr__(
            self,
            "module_kinds",
            _canonical_names(self.module_kinds, "module kind", dotted=True),
        )
        object.__setattr__(
            self,
            "parameter_roles",
            _canonical_names(self.parameter_roles, "parameter role", dotted=True),
        )
        if any(not isinstance(item, AttributePredicate) for item in self.attributes):
            raise TypeError("predicate attributes must be AttributePredicate records")
        names = [item.name for item in self.attributes]
        if len(names) != len(set(names)):
            raise ValueError("predicate contains a duplicate attribute")
        object.__setattr__(
            self,
            "attributes",
            tuple(sorted(self.attributes, key=lambda item: item.name)),
        )


def _canonical_names(
    values: tuple[str, ...],
    name: str,
    *,
    dotted: bool,
) -> tuple[str, ...]:
    validator = _require_dotted_name if dotted else _require_atom
    for value in values:
        validator(value, name)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} values must be unique")
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True, order=True)
class LayerMember:
    """One correlated global-decoder and optional MoE-layer coordinate."""

    global_decoder_layer: int
    moe_ordinal: int | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.global_decoder_layer, bool)
            or not isinstance(self.global_decoder_layer, int)
            or self.global_decoder_layer < 0
        ):
            raise ValueError("global_decoder_layer must be a non-negative integer")
        if self.moe_ordinal is not None and (
            isinstance(self.moe_ordinal, bool)
            or not isinstance(self.moe_ordinal, int)
            or self.moe_ordinal < 0
        ):
            raise ValueError("moe_ordinal must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class LayerDomain:
    """Finite correlated layer-coordinate relation."""

    members: tuple[LayerMember, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(item, LayerMember) for item in self.members):
            raise TypeError("layer-domain members must be LayerMember records")
        if len(self.members) != len(set(self.members)):
            raise ValueError("layer domain contains duplicate members")
        has_moe_ordinal = {item.moe_ordinal is not None for item in self.members}
        if len(has_moe_ordinal) > 1:
            raise ValueError(
                "layer domain must not mix missing and present moe_ordinal values"
            )
        object.__setattr__(self, "members", tuple(sorted(self.members)))

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Return the correlated coordinate names present in this domain."""
        if not self.members:
            return ("global_decoder_layer",)
        if self.members[0].moe_ordinal is None:
            return ("global_decoder_layer",)
        return _LAYER_AXES


def _axis_member_key(value: AxisMember) -> tuple[int, object]:
    if isinstance(value, bool):
        raise TypeError("axis-domain members must not be bool")
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, str):
        _require_atom(value, "axis-domain string member")
        return (1, value)
    raise TypeError("axis-domain members must be int or str")


@dataclass(frozen=True, slots=True)
class AxisDomain:
    """Finite genuinely independent family axis."""

    name: str
    members: tuple[AxisMember, ...]

    def __post_init__(self) -> None:
        _require_atom(self.name, "axis-domain name")
        if self.name == "projection":
            raise ValueError("projection must be fixed per family, not an axis")
        if self.name in _LAYER_AXES:
            raise ValueError(f"{self.name} belongs in LayerDomain")
        keyed = [(_axis_member_key(value), value) for value in self.members]
        rendered = [str(value) for _, value in keyed]
        if len(rendered) != len(set(rendered)):
            raise ValueError("axis-domain members must render uniquely")
        object.__setattr__(
            self,
            "members",
            tuple(value for _, value in sorted(keyed, key=lambda item: item[0])),
        )


@dataclass(frozen=True, slots=True)
class FamilyIndexDomain:
    """Compact factorized index domain with one optional layer relation."""

    layer_domain: LayerDomain | None
    independent_axes: tuple[AxisDomain, ...]

    def __post_init__(self) -> None:
        if self.layer_domain is not None and not isinstance(
            self.layer_domain, LayerDomain
        ):
            raise TypeError("layer_domain must be LayerDomain or None")
        if any(not isinstance(axis, AxisDomain) for axis in self.independent_axes):
            raise TypeError("independent_axes must contain AxisDomain records")
        names = [axis.name for axis in self.independent_axes]
        if len(names) != len(set(names)):
            raise ValueError("family domain contains duplicate independent axes")
        object.__setattr__(
            self,
            "independent_axes",
            tuple(sorted(self.independent_axes, key=lambda axis: axis.name)),
        )

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Return canonical member-axis names."""
        layer_axes = () if self.layer_domain is None else self.layer_domain.axis_names
        return layer_axes + tuple(axis.name for axis in self.independent_axes)

    @property
    def cardinality(self) -> int:
        """Return logical member count without materializing the Cartesian product."""
        layer_cardinality = (
            1 if self.layer_domain is None else len(self.layer_domain.members)
        )
        return layer_cardinality * prod(
            len(axis.members) for axis in self.independent_axes
        )


@dataclass(frozen=True, slots=True)
class LiteralPathSegment:
    """One canonical literal semantic-ID path segment."""

    value: str

    def __post_init__(self) -> None:
        _require_atom(self.value, "literal path segment")


@dataclass(frozen=True, slots=True)
class IndexPathSegment:
    """Reference to one typed finite family-domain axis."""

    axis_name: str

    def __post_init__(self) -> None:
        _require_atom(self.axis_name, "index path segment axis")


type SemanticPathSegment = LiteralPathSegment | IndexPathSegment


@dataclass(frozen=True, slots=True)
class SemanticAddress:
    """One explicit canonical logical tensor address."""

    semantic_id: str
    semantic_graph_path: str
    model_part: str
    module_kind: str
    attributes: tuple[tuple[str, PredicateScalar], ...]
    parameter_role: str
    global_decoder_layer: int | None
    moe_ordinal: int | None

    def __post_init__(self) -> None:
        _require_dotted_name(
            self.semantic_graph_path,
            "semantic_graph_path",
            minimum_parts=2,
        )
        _require_dotted_name(
            self.semantic_id,
            "semantic_id",
            minimum_parts=3,
        )
        if not self.semantic_id.startswith(f"{self.semantic_graph_path}."):
            raise ValueError("semantic_id must be a descendant of semantic_graph_path")
        _require_dotted_name(self.model_part, "model_part")
        _require_dotted_name(self.module_kind, "module_kind")
        _require_dotted_name(self.parameter_role, "parameter_role")
        if self.global_decoder_layer is not None and (
            isinstance(self.global_decoder_layer, bool)
            or not isinstance(self.global_decoder_layer, int)
            or self.global_decoder_layer < 0
        ):
            raise ValueError("global_decoder_layer must be non-negative or None")
        if self.moe_ordinal is not None and (
            isinstance(self.moe_ordinal, bool)
            or not isinstance(self.moe_ordinal, int)
            or self.moe_ordinal < 0
        ):
            raise ValueError("moe_ordinal must be non-negative or None")
        object.__setattr__(self, "attributes", _canonical_attributes(self.attributes))


@dataclass(frozen=True, slots=True)
class SemanticAddressPattern:
    """Structured compact semantic-ID family pattern."""

    semantic_graph_path: str
    path_segments: tuple[SemanticPathSegment, ...]
    model_part: str
    module_kind: str
    attributes: tuple[tuple[str, PredicateScalar], ...]
    parameter_role: str

    def __post_init__(self) -> None:
        _require_dotted_name(
            self.semantic_graph_path,
            "semantic_graph_path",
            minimum_parts=2,
        )
        if not self.path_segments:
            raise ValueError("semantic address pattern requires path segments")
        if any(
            not isinstance(segment, (LiteralPathSegment, IndexPathSegment))
            for segment in self.path_segments
        ):
            raise TypeError("path_segments must be typed semantic path segments")
        _require_dotted_name(self.model_part, "model_part")
        _require_dotted_name(self.module_kind, "module_kind")
        _require_dotted_name(self.parameter_role, "parameter_role")
        object.__setattr__(self, "attributes", _canonical_attributes(self.attributes))


@dataclass(frozen=True, slots=True)
class OwnerFamilyReference:
    """Globally qualified canonical source-owner family identity."""

    graph_instance_id: str
    owner_family_id: str

    def __post_init__(self) -> None:
        _validate_graph_instance_id(self.graph_instance_id)
        _require_dotted_name(self.owner_family_id, "owner_family_id")


@dataclass(frozen=True, slots=True)
class AxisProjection:
    """Map one member-domain axis onto one target-domain axis."""

    member_axis: str
    owner_axis: str

    def __post_init__(self) -> None:
        _require_atom(self.member_axis, "projection member_axis")
        _require_atom(self.owner_axis, "projection owner_axis")


@dataclass(frozen=True, slots=True)
class OwnerFamilyBinding:
    """Direct canonical owner and canonical semantic-value binding."""

    canonical_owner_family: OwnerFamilyReference
    canonical_value_entry_id: str
    member_domain: FamilyIndexDomain
    member_to_owner_axes: tuple[AxisProjection, ...]
    member_to_value_axes: tuple[AxisProjection, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_owner_family, OwnerFamilyReference):
            raise TypeError("canonical_owner_family must be OwnerFamilyReference")
        _require_dotted_name(self.canonical_value_entry_id, "canonical_value_entry_id")
        if not isinstance(self.member_domain, FamilyIndexDomain):
            raise TypeError("member_domain must be FamilyIndexDomain")
        for field_name in ("member_to_owner_axes", "member_to_value_axes"):
            projections = getattr(self, field_name)
            if any(not isinstance(item, AxisProjection) for item in projections):
                raise TypeError(f"{field_name} must contain AxisProjection records")
            object.__setattr__(
                self,
                field_name,
                tuple(
                    sorted(
                        projections,
                        key=lambda item: (item.owner_axis, item.member_axis),
                    )
                ),
            )


@dataclass(frozen=True, slots=True)
class SemanticOwnership:
    """Semantic member ownership declaration."""

    binding: OwnerFamilyBinding

    def __post_init__(self) -> None:
        if not isinstance(self.binding, OwnerFamilyBinding):
            raise TypeError("semantic ownership requires OwnerFamilyBinding")


def _validate_tensor_fields(
    format_descriptor: FormatDescriptor,
    logical_dtype: str,
    logical_shape: tuple[int, ...],
    logical_axes: tuple[str, ...],
    ownership: SemanticOwnership,
) -> None:
    if not isinstance(format_descriptor, FormatDescriptor):
        raise TypeError("format must be FormatDescriptor")
    _require_atom(logical_dtype, "logical_dtype")
    _validate_logical_shape(logical_shape, logical_axes)
    if not isinstance(ownership, SemanticOwnership):
        raise TypeError("ownership must be SemanticOwnership")


@dataclass(frozen=True, slots=True)
class SemanticTensor:
    """One explicit logical tensor with source ownership."""

    address: SemanticAddress
    format: FormatDescriptor
    logical_dtype: str
    logical_shape: tuple[int, ...]
    logical_axes: tuple[str, ...]
    ownership: SemanticOwnership

    def __post_init__(self) -> None:
        if not isinstance(self.address, SemanticAddress):
            raise TypeError("address must be SemanticAddress")
        _validate_tensor_fields(
            self.format,
            self.logical_dtype,
            self.logical_shape,
            self.logical_axes,
            self.ownership,
        )
        if self.ownership.binding.member_domain != FamilyIndexDomain(None, ()):
            raise ValueError(
                "an explicit semantic tensor requires a scalar member domain"
            )


@dataclass(frozen=True, slots=True)
class SemanticTensorFamily:
    """One homogeneous logical tensor family over a compact finite domain."""

    pattern: SemanticAddressPattern
    domain: FamilyIndexDomain
    format: FormatDescriptor
    logical_dtype: str
    logical_shape: tuple[int, ...]
    logical_axes: tuple[str, ...]
    ownership: SemanticOwnership

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, SemanticAddressPattern):
            raise TypeError("pattern must be SemanticAddressPattern")
        if not isinstance(self.domain, FamilyIndexDomain):
            raise TypeError("domain must be FamilyIndexDomain")
        _validate_tensor_fields(
            self.format,
            self.logical_dtype,
            self.logical_shape,
            self.logical_axes,
            self.ownership,
        )
        if self.ownership.binding.member_domain != self.domain:
            raise ValueError("family ownership member_domain must equal family domain")
        index_axes = tuple(
            segment.axis_name
            for segment in self.pattern.path_segments
            if isinstance(segment, IndexPathSegment)
        )
        if len(index_axes) != len(set(index_axes)):
            raise ValueError("each family index axis must occur exactly once")
        unknown = set(index_axes) - set(self.domain.axis_names)
        independent_names = {axis.name for axis in self.domain.independent_axes}
        missing_independent = independent_names - set(index_axes)
        if unknown or missing_independent:
            raise ValueError(
                f"family index axis mismatch: {sorted(unknown or missing_independent)}"
            )
        if self.domain.layer_domain is not None:
            rendered_layer_axes = tuple(
                axis for axis in index_axes if axis in _LAYER_AXES
            )
            if not rendered_layer_axes:
                raise ValueError("family path must identify its layer domain")
            rendered_layer_keys = tuple(
                tuple(_layer_value(member, axis) for axis in rendered_layer_axes)
                for member in self.domain.layer_domain.members
            )
            if len(rendered_layer_keys) != len(set(rendered_layer_keys)):
                raise ValueError(
                    "family path layer axes do not uniquely identify layer members"
                )

    def iter_semantic_ids(self) -> Iterator[str]:
        """Lazily render semantic IDs for local diagnostics or binding."""
        layer_members: tuple[LayerMember | None, ...]
        if self.domain.layer_domain is None:
            layer_members = (None,)
        else:
            layer_members = self.domain.layer_domain.members
        independent_members = tuple(
            axis.members for axis in self.domain.independent_axes
        )
        for layer_member in layer_members:
            for independent_values in product(*independent_members):
                coordinates: dict[str, AxisMember] = {}
                if layer_member is not None:
                    coordinates["global_decoder_layer"] = (
                        layer_member.global_decoder_layer
                    )
                    if layer_member.moe_ordinal is not None:
                        coordinates["moe_ordinal"] = layer_member.moe_ordinal
                coordinates.update(
                    {
                        axis.name: value
                        for axis, value in zip(
                            self.domain.independent_axes,
                            independent_values,
                            strict=True,
                        )
                    }
                )
                suffix = ".".join(
                    segment.value
                    if isinstance(segment, LiteralPathSegment)
                    else str(coordinates[segment.axis_name])
                    for segment in self.pattern.path_segments
                )
                yield f"{self.pattern.semantic_graph_path}.{suffix}"


type SemanticInventoryMember = SemanticTensor | SemanticTensorFamily


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """Typed source and digest for mutability or checkpoint evidence."""

    kind: EvidenceSourceKind
    locator: str
    digest: str

    def __post_init__(self) -> None:
        _require_enum(self.kind, EvidenceSourceKind, "evidence source kind")
        _require_text(self.locator, "evidence source locator")
        _require_text(self.digest, "evidence source digest")


@dataclass(frozen=True, slots=True)
class ParameterInventoryEntry:
    """One authoritative explicit member or complete family."""

    entry_id: str
    graph_instance_id: str
    member: SemanticInventoryMember
    value_provenance: ValueProvenance

    def __post_init__(self) -> None:
        _require_dotted_name(self.entry_id, "inventory entry_id")
        _validate_graph_instance_id(self.graph_instance_id)
        if not isinstance(self.member, (SemanticTensor, SemanticTensorFamily)):
            raise TypeError(
                "inventory member must be SemanticTensor or SemanticTensorFamily"
            )
        _require_enum(self.value_provenance, ValueProvenance, "value provenance")


@dataclass(frozen=True, slots=True)
class SourceOwnerInventoryEntry:
    """One canonical source-owner family and its evidenced mutability."""

    owner_family: OwnerFamilyReference
    domain: FamilyIndexDomain
    source_mutability: SourceMutability
    mutability_evidence_source: EvidenceSource

    def __post_init__(self) -> None:
        if not isinstance(self.owner_family, OwnerFamilyReference):
            raise TypeError("owner_family must be OwnerFamilyReference")
        if not isinstance(self.domain, FamilyIndexDomain):
            raise TypeError("owner domain must be FamilyIndexDomain")
        _require_enum(self.source_mutability, SourceMutability, "source mutability")
        if not isinstance(self.mutability_evidence_source, EvidenceSource):
            raise TypeError("mutability evidence must be EvidenceSource")


@dataclass(frozen=True, slots=True)
class ParameterInventory:
    """Canonical compact source-owner and semantic-member inventory."""

    owners: tuple[SourceOwnerInventoryEntry, ...]
    entries: tuple[ParameterInventoryEntry, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(item, SourceOwnerInventoryEntry) for item in self.owners):
            raise TypeError(
                "inventory owners must be SourceOwnerInventoryEntry records"
            )
        if any(not isinstance(item, ParameterInventoryEntry) for item in self.entries):
            raise TypeError("inventory entries must be ParameterInventoryEntry records")
        object.__setattr__(
            self,
            "owners",
            tuple(
                sorted(
                    self.owners,
                    key=lambda item: (
                        _graph_sort_key(item.owner_family.graph_instance_id),
                        item.owner_family.owner_family_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "entries",
            tuple(
                sorted(
                    self.entries,
                    key=lambda item: (
                        _graph_sort_key(item.graph_instance_id),
                        item.entry_id,
                    ),
                )
            ),
        )

    @property
    def logical_cardinality(self) -> int:
        """Return total logical members without storing expanded tensors."""
        return sum(_member_domain(entry.member).cardinality for entry in self.entries)

    def owner_family(
        self,
        graph_instance_id: str,
        owner_family_id: str,
    ) -> SourceOwnerInventoryEntry:
        """Resolve one globally qualified source-owner family."""
        reference = OwnerFamilyReference(graph_instance_id, owner_family_id)
        matches = tuple(
            owner for owner in self.owners if owner.owner_family == reference
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected one source owner for {reference}, got {len(matches)}"
            )
        return matches[0]


def _member_domain(member: SemanticInventoryMember) -> FamilyIndexDomain:
    if isinstance(member, SemanticTensorFamily):
        return member.domain
    return FamilyIndexDomain(None, ())


def _member_format(member: SemanticInventoryMember) -> FormatDescriptor:
    return member.format


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


@dataclass(frozen=True, slots=True)
class OutOfScopeTensor:
    """Typed whole-entry exclusion from policy-driven refit."""

    inventory_entry_id: str
    reason: OutOfScopeReason

    def __post_init__(self) -> None:
        _require_dotted_name(self.inventory_entry_id, "out-of-scope inventory entry")
        if not isinstance(self.reason, OutOfScopeReason):
            raise TypeError("reason must be OutOfScopeReason")


@dataclass(frozen=True, slots=True)
class RoleExpectedDomain:
    """Topology-advertised exact compact-entry domain for a role."""

    role_name: str
    inventory_entry_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_dotted_name(self.role_name, "role_name")
        if not self.inventory_entry_ids:
            raise ValueError("role expected domain must be non-empty")
        for entry_id in self.inventory_entry_ids:
            _require_dotted_name(entry_id, "role expected inventory entry")
        if len(self.inventory_entry_ids) != len(set(self.inventory_entry_ids)):
            raise ValueError("role expected domain contains duplicate entries")
        object.__setattr__(
            self,
            "inventory_entry_ids",
            tuple(sorted(self.inventory_entry_ids)),
        )


@dataclass(frozen=True, slots=True)
class RoleDefinition:
    """Schema-bound role predicate and independently advertised domain."""

    schema_version: int
    role_name: str
    predicate: SemanticPredicate
    expected_domain: RoleExpectedDomain

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("role schema_version must be an integer")
        _require_dotted_name(self.role_name, "role_name")
        if not isinstance(self.predicate, SemanticPredicate):
            raise TypeError("role predicate must be SemanticPredicate")
        if not isinstance(self.expected_domain, RoleExpectedDomain):
            raise TypeError("expected_domain must be RoleExpectedDomain")
        if self.expected_domain.role_name != self.role_name:
            raise ValueError("role expected domain name must match role definition")

    def matching_inventory_entry_ids(
        self,
        bundle: SemanticManifestBundle,
    ) -> tuple[str, ...]:
        """Match whole compact entries using complete bundle graph context."""
        lifecycle_by_graph = {
            manifest.graph_instance_id: manifest.lifecycle
            for manifest in bundle.manifests
        }
        matches: list[str] = []
        for entry in bundle.inventory.entries:
            lifecycle = lifecycle_by_graph.get(entry.graph_instance_id)
            if lifecycle is None:
                continue
            if _predicate_matches(self.predicate, lifecycle.graph_kind, entry.member):
                matches.append(entry.entry_id)
        return tuple(sorted(matches))

    def validate_expected_domain(self, bundle: SemanticManifestBundle) -> None:
        """Reject missing, extra, or orphaned compact role members."""
        actual = self.matching_inventory_entry_ids(bundle)
        if actual != self.expected_domain.inventory_entry_ids:
            raise ValueError(
                f"role {self.role_name} expected domain "
                f"{self.expected_domain.inventory_entry_ids}, matched {actual}"
            )


def _predicate_matches(
    predicate: SemanticPredicate,
    graph_kind: GraphKind,
    member: SemanticInventoryMember,
) -> bool:
    if predicate.graph_kinds and graph_kind not in predicate.graph_kinds:
        return False
    if (
        predicate.semantic_graph_paths
        and _member_graph_path(member) not in predicate.semantic_graph_paths
    ):
        return False
    if (
        predicate.model_parts
        and _member_model_part(member) not in predicate.model_parts
    ):
        return False
    if (
        predicate.module_kinds
        and _member_module_kind(member) not in predicate.module_kinds
    ):
        return False
    if (
        predicate.parameter_roles
        and _member_parameter_role(member) not in predicate.parameter_roles
    ):
        return False
    attributes = dict(_member_attributes(member))
    for attribute in predicate.attributes:
        value = attributes.get(attribute.name)
        if value is None:
            return False
        value_key = _typed_scalar_key(value)
        if value_key not in {
            _typed_scalar_key(allowed) for allowed in attribute.allowed_values
        }:
            return False
    return True


_BUILTIN_ROLE_PREDICATES: Mapping[str, SemanticPredicate] = {
    "attention.qkvo": SemanticPredicate(
        graph_kinds=(GraphKind.MAIN,),
        semantic_graph_paths=("text.decoder",),
        model_parts=("main",),
        module_kinds=("attention.projection",),
        attributes=(AttributePredicate("projection", ("q", "k", "v", "o")),),
        parameter_roles=("kernel",),
    ),
    "embedding.ngram": SemanticPredicate(
        graph_kinds=(GraphKind.MAIN,),
        semantic_graph_paths=("text.embedding",),
        model_parts=("main",),
        module_kinds=("embedding.ngram",),
        attributes=(),
        parameter_roles=("kernel",),
    ),
    "moe.routed_expert": SemanticPredicate(
        graph_kinds=(GraphKind.MAIN,),
        semantic_graph_paths=("text.decoder",),
        model_parts=("main",),
        module_kinds=("moe.expert_ffn",),
        attributes=(
            AttributePredicate("expert_kind", ("routed",)),
            AttributePredicate("projection", ("gate", "up", "down")),
        ),
        parameter_roles=("kernel",),
    ),
}


def builtin_role_definitions(
    schema_version: int,
    expected_domains: Mapping[str, RoleExpectedDomain],
) -> tuple[RoleDefinition, ...]:
    """Attach topology-advertised domains to fixed version-1 role predicates."""
    if schema_version != 1:
        raise ValueError(f"unsupported semantic schema version: {schema_version}")
    unknown = set(expected_domains) - set(_BUILTIN_ROLE_PREDICATES)
    if unknown:
        raise ValueError(f"unknown built-in role: {sorted(unknown)[0]}")
    definitions = tuple(
        RoleDefinition(
            schema_version=schema_version,
            role_name=role_name,
            predicate=_BUILTIN_ROLE_PREDICATES[role_name],
            expected_domain=expected_domain,
        )
        for role_name, expected_domain in expected_domains.items()
    )
    return tuple(sorted(definitions, key=lambda item: item.role_name))


@dataclass(frozen=True, slots=True)
class AtomicGroupParticipant:
    """One compact entry projection in a pointwise atomic group."""

    inventory_entry_id: str
    participant_domain: FamilyIndexDomain
    group_to_participant_axes: tuple[AxisProjection, ...]

    def __post_init__(self) -> None:
        _require_dotted_name(self.inventory_entry_id, "atomic participant entry")
        if not isinstance(self.participant_domain, FamilyIndexDomain):
            raise TypeError("participant_domain must be FamilyIndexDomain")
        if any(
            not isinstance(item, AxisProjection)
            for item in self.group_to_participant_axes
        ):
            raise TypeError("atomic participant projections must be AxisProjection")
        object.__setattr__(
            self,
            "group_to_participant_axes",
            tuple(
                sorted(
                    self.group_to_participant_axes,
                    key=lambda item: (item.owner_axis, item.member_axis),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class AtomicGroup:
    """Compact pointwise semantic precision-atomicity relation."""

    group_id: str
    graph_instance_id: str
    kind: AtomicGroupKind
    group_domain: FamilyIndexDomain
    participants: tuple[AtomicGroupParticipant, ...]

    def __post_init__(self) -> None:
        _require_dotted_name(self.group_id, "atomic group_id")
        _validate_graph_instance_id(self.graph_instance_id)
        _require_enum(self.kind, AtomicGroupKind, "atomic group kind")
        if not isinstance(self.group_domain, FamilyIndexDomain):
            raise TypeError("atomic group_domain must be FamilyIndexDomain")
        if any(
            not isinstance(item, AtomicGroupParticipant) for item in self.participants
        ):
            raise TypeError(
                "atomic participants must be AtomicGroupParticipant records"
            )
        object.__setattr__(
            self,
            "participants",
            tuple(sorted(self.participants, key=lambda item: item.inventory_entry_id)),
        )


@dataclass(frozen=True, slots=True)
class ImmutableAuxiliaryEvidence:
    """Pinned identity and content evidence for a checkpoint-served graph."""

    graph_instance_id: str
    model_identity: str
    pinned_checkpoint_revision: str
    checkpoint_content_digest: str
    model_config_digest: str
    semantic_domain_digest: str
    evidence_source: EvidenceSource

    def __post_init__(self) -> None:
        _validate_graph_instance_id(self.graph_instance_id)
        _require_text(self.model_identity, "immutable model identity")
        _require_text(self.pinned_checkpoint_revision, "pinned checkpoint revision")
        _require_text(self.checkpoint_content_digest, "checkpoint content digest")
        _require_text(self.model_config_digest, "model config digest")
        _require_text(self.semantic_domain_digest, "semantic domain digest")
        if not isinstance(self.evidence_source, EvidenceSource):
            raise TypeError("immutable evidence source must be EvidenceSource")


@dataclass(frozen=True, slots=True)
class GraphLifecycle:
    """Orthogonal graph facts, excluding source mutability and refit cadence."""

    graph_kind: GraphKind
    graph_provenance: GraphProvenance
    rollout_participation: RolloutParticipation
    immutable_evidence: ImmutableAuxiliaryEvidence | None = None

    def __post_init__(self) -> None:
        _require_enum(self.graph_kind, GraphKind, "graph kind")
        _require_enum(self.graph_provenance, GraphProvenance, "graph provenance")
        _require_enum(
            self.rollout_participation,
            RolloutParticipation,
            "rollout participation",
        )
        checkpoint_served = (
            self.rollout_participation == RolloutParticipation.SERVED_FROM_CHECKPOINT
        )
        checkpoint_provenance = self.graph_provenance in {
            GraphProvenance.MODEL_CHECKPOINT,
            GraphProvenance.EXTERNAL_CHECKPOINT,
        }
        if checkpoint_served != checkpoint_provenance:
            raise ValueError(
                "invalid graph lifecycle: checkpoint provenance and participation "
                "must agree"
            )
        if checkpoint_served and self.immutable_evidence is None:
            raise ValueError(
                "invalid graph lifecycle: checkpoint serving requires immutable evidence"
            )
        if not checkpoint_served and self.immutable_evidence is not None:
            raise ValueError(
                "invalid graph lifecycle: immutable evidence is checkpoint-only"
            )
        if self.graph_kind == GraphKind.MAIN and checkpoint_served:
            raise ValueError(
                "invalid graph lifecycle: MAIN cannot be checkpoint-served"
            )


@dataclass(frozen=True, slots=True)
class ExpectedGraphDeclaration:
    """Authoritative pre-adaptation graph-instance declaration."""

    graph_instance_id: str
    model_identity: str
    lifecycle: GraphLifecycle

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle, GraphLifecycle):
            raise TypeError("expected graph lifecycle must be GraphLifecycle")
        _validate_graph_instance_id(
            self.graph_instance_id,
            self.lifecycle.graph_kind,
        )
        _require_text(self.model_identity, "expected graph model identity")
        evidence = self.lifecycle.immutable_evidence
        if evidence is not None:
            if evidence.graph_instance_id != self.graph_instance_id:
                raise ValueError("immutable evidence graph_instance_id mismatch")
            if evidence.model_identity != self.model_identity:
                raise ValueError("immutable evidence model identity mismatch")


@dataclass(frozen=True, slots=True)
class AuxiliaryGraphDeclaration:
    """Topology-independent MTP or speculative-drafter declaration."""

    graph_instance_id: str
    model_identity: str
    lifecycle: GraphLifecycle

    def __post_init__(self) -> None:
        expected = ExpectedGraphDeclaration(
            self.graph_instance_id,
            self.model_identity,
            self.lifecycle,
        )
        if expected.lifecycle.graph_kind == GraphKind.MAIN:
            raise ValueError("AuxiliaryGraphDeclaration cannot declare MAIN")


@dataclass(frozen=True, slots=True)
class SemanticGraphManifest:
    """One graph instance and its complete compact accounting handles."""

    model_family: str
    model_revision: str
    graph_instance_id: str
    lifecycle: GraphLifecycle
    inventory_entry_ids: tuple[str, ...]
    atomic_groups: tuple[AtomicGroup, ...] = ()
    out_of_scope: tuple[OutOfScopeTensor, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.model_family, "manifest model_family")
        _require_text(self.model_revision, "manifest model_revision")
        if not isinstance(self.lifecycle, GraphLifecycle):
            raise TypeError("manifest lifecycle must be GraphLifecycle")
        _validate_graph_instance_id(
            self.graph_instance_id,
            self.lifecycle.graph_kind,
        )
        for entry_id in self.inventory_entry_ids:
            _require_dotted_name(entry_id, "manifest inventory entry")
        if any(not isinstance(item, AtomicGroup) for item in self.atomic_groups):
            raise TypeError("manifest atomic_groups must contain AtomicGroup records")
        if any(not isinstance(item, OutOfScopeTensor) for item in self.out_of_scope):
            raise TypeError(
                "manifest out_of_scope must contain OutOfScopeTensor records"
            )
        object.__setattr__(
            self,
            "inventory_entry_ids",
            tuple(sorted(self.inventory_entry_ids)),
        )
        object.__setattr__(
            self,
            "atomic_groups",
            tuple(sorted(self.atomic_groups, key=lambda item: item.group_id)),
        )
        object.__setattr__(
            self,
            "out_of_scope",
            tuple(sorted(self.out_of_scope, key=lambda item: item.inventory_entry_id)),
        )
        evidence = self.lifecycle.immutable_evidence
        if evidence is not None:
            if evidence.graph_instance_id != self.graph_instance_id:
                raise ValueError("immutable evidence graph_instance_id mismatch")
            if evidence.pinned_checkpoint_revision != self.model_revision:
                raise ValueError(
                    "manifest model revision must equal pinned_checkpoint_revision"
                )

    def validate_complete(self) -> None:
        """Validate graph-local accounting identities and semantic groups."""
        if len(self.inventory_entry_ids) != len(set(self.inventory_entry_ids)):
            raise ValueError("manifest contains a duplicate inventory entry")
        out_of_scope_ids = [item.inventory_entry_id for item in self.out_of_scope]
        if len(out_of_scope_ids) != len(set(out_of_scope_ids)):
            raise ValueError("manifest contains a duplicate out-of-scope claim")
        group_ids = [item.group_id for item in self.atomic_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("manifest contains a duplicate atomic group ID")
        for group in self.atomic_groups:
            if group.graph_instance_id != self.graph_instance_id:
                raise ValueError("atomic group graph does not match its manifest")


@dataclass(frozen=True, slots=True)
class SemanticManifestBundle:
    """Complete schema-bound logical inventory for all expected graph instances."""

    schema_version: int
    expected_graphs: tuple[ExpectedGraphDeclaration, ...]
    manifests: tuple[SemanticGraphManifest, ...]
    inventory: ParameterInventory
    role_definitions: tuple[RoleDefinition, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("semantic schema_version must be an integer")
        if any(
            not isinstance(item, ExpectedGraphDeclaration)
            for item in self.expected_graphs
        ):
            raise TypeError("expected_graphs must contain declarations")
        if any(not isinstance(item, SemanticGraphManifest) for item in self.manifests):
            raise TypeError("manifests must contain SemanticGraphManifest records")
        if not isinstance(self.inventory, ParameterInventory):
            raise TypeError("inventory must be ParameterInventory")
        if any(not isinstance(item, RoleDefinition) for item in self.role_definitions):
            raise TypeError("role_definitions must contain RoleDefinition records")
        object.__setattr__(
            self,
            "expected_graphs",
            tuple(
                sorted(
                    self.expected_graphs,
                    key=lambda item: _graph_sort_key(item.graph_instance_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "manifests",
            tuple(
                sorted(
                    self.manifests,
                    key=lambda item: _graph_sort_key(item.graph_instance_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "role_definitions",
            tuple(
                sorted(
                    self.role_definitions,
                    key=lambda item: (item.schema_version, item.role_name),
                )
            ),
        )

    def manifest(self, graph_instance_id: str) -> SemanticGraphManifest:
        """Resolve one manifest by runtime graph-instance identity."""
        matches = tuple(
            manifest
            for manifest in self.manifests
            if manifest.graph_instance_id == graph_instance_id
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected one manifest for {graph_instance_id}, got {len(matches)}"
            )
        return matches[0]

    def role_definition(self, schema_version: int, role_name: str) -> RoleDefinition:
        """Resolve one definition from the bundle-owned role registry."""
        matches = tuple(
            definition
            for definition in self.role_definitions
            if definition.schema_version == schema_version
            and definition.role_name == role_name
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected one role definition for {(schema_version, role_name)}, "
                f"got {len(matches)}"
            )
        return matches[0]

    def validate_complete(self) -> None:
        """Validate complete graph, inventory, owner, role, and lifecycle accounting."""
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported semantic schema version: {self.schema_version}"
            )
        for manifest in self.manifests:
            manifest.validate_complete()
        _validate_inventory_identities(self)
        _validate_graph_accounting(self)
        _validate_owner_bindings(self)
        _validate_out_of_scope(self)
        _validate_atomic_groups(self)
        _validate_role_registry(self)
        _validate_source_serving(self)

    def owner_refit_requirements(
        self,
        graph_instance_id: str,
    ) -> tuple[tuple[OwnerFamilyReference, RefitRequirement], ...]:
        """Derive per-owner cadence from lifecycle and evidenced mutability."""
        self.validate_complete()
        return _owner_refit_requirements_unchecked(self, graph_instance_id)

    def refit_requirement(self, graph_instance_id: str) -> RefitRequirement:
        """Derive the graph refit summary without storing an input cadence."""
        requirements = self.owner_refit_requirements(graph_instance_id)
        lifecycle = self.manifest(graph_instance_id).lifecycle
        if lifecycle.rollout_participation != RolloutParticipation.SERVED_FROM_SOURCE:
            return RefitRequirement.NONE
        if not requirements:
            raise ValueError(
                "served-from-source graph requires a non-empty semantic domain"
            )
        if any(
            requirement == RefitRequirement.EVERY_VERSION
            for _, requirement in requirements
        ):
            return RefitRequirement.EVERY_VERSION
        return RefitRequirement.INITIAL_ONLY


def _validate_unique_ids(values: tuple[str, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def _validate_graph_accounting(bundle: SemanticManifestBundle) -> None:
    expected_ids = tuple(item.graph_instance_id for item in bundle.expected_graphs)
    manifest_ids = tuple(item.graph_instance_id for item in bundle.manifests)
    _validate_unique_ids(expected_ids, "expected graph declaration")
    _validate_unique_ids(manifest_ids, "semantic graph manifest")
    main_ids = tuple(
        item.graph_instance_id
        for item in bundle.expected_graphs
        if item.lifecycle.graph_kind == GraphKind.MAIN
    )
    if main_ids != ("main",):
        raise ValueError(
            "semantic bundle requires exactly one MAIN instance named main"
        )
    if set(expected_ids) != set(manifest_ids):
        raise ValueError("expected graph and manifest sets must match bijectively")

    expected_by_id = {item.graph_instance_id: item for item in bundle.expected_graphs}
    manifest_by_id = {item.graph_instance_id: item for item in bundle.manifests}
    for graph_instance_id in expected_ids:
        expected = expected_by_id[graph_instance_id]
        manifest = manifest_by_id[graph_instance_id]
        if expected.lifecycle != manifest.lifecycle:
            difference = _lifecycle_difference(expected.lifecycle, manifest.lifecycle)
            raise ValueError(
                f"graph {graph_instance_id} lifecycle mismatch: {difference}"
            )
        evidence = expected.lifecycle.immutable_evidence
        if evidence is not None:
            if evidence.graph_instance_id != graph_instance_id:
                raise ValueError("immutable evidence graph_instance_id mismatch")
            if evidence.model_identity != expected.model_identity:
                raise ValueError("immutable evidence model identity mismatch")
            if evidence.pinned_checkpoint_revision != manifest.model_revision:
                raise ValueError(
                    "immutable evidence pinned checkpoint revision mismatch"
                )

    entries_by_id = {entry.entry_id: entry for entry in bundle.inventory.entries}
    entries_by_graph: dict[str, list[str]] = {graph_id: [] for graph_id in expected_ids}
    for entry in bundle.inventory.entries:
        if entry.graph_instance_id not in entries_by_graph:
            raise ValueError(
                f"inventory entry {entry.entry_id} belongs to undeclared graph"
            )
        entries_by_graph[entry.graph_instance_id].append(entry.entry_id)
    for graph_instance_id, manifest in manifest_by_id.items():
        for entry_id in manifest.inventory_entry_ids:
            entry = entries_by_id.get(entry_id)
            if entry is None:
                raise ValueError(
                    f"manifest inventory entry {entry_id} must resolve exactly once"
                )
            if entry.graph_instance_id != graph_instance_id:
                raise ValueError("manifest contains a foreign inventory entry")
        expected_entries = tuple(sorted(entries_by_graph[graph_instance_id]))
        if manifest.inventory_entry_ids != expected_entries:
            raise ValueError(
                f"graph {graph_instance_id} inventory accounting mismatch: "
                f"manifest={manifest.inventory_entry_ids}, inventory={expected_entries}"
            )
    for owner in bundle.inventory.owners:
        if owner.owner_family.graph_instance_id not in expected_by_id:
            raise ValueError("source owner belongs to an undeclared graph")


def _lifecycle_difference(left: GraphLifecycle, right: GraphLifecycle) -> str:
    for field_name in (
        "graph_kind",
        "graph_provenance",
        "rollout_participation",
    ):
        if getattr(left, field_name) != getattr(right, field_name):
            return field_name
    left_evidence = left.immutable_evidence
    right_evidence = right.immutable_evidence
    if left_evidence is None or right_evidence is None:
        return "immutable_evidence"
    for field_name in (
        "graph_instance_id",
        "model_identity",
        "pinned_checkpoint_revision",
        "checkpoint_content_digest",
        "model_config_digest",
        "semantic_domain_digest",
    ):
        if getattr(left_evidence, field_name) != getattr(right_evidence, field_name):
            return field_name
    if left_evidence.evidence_source != right_evidence.evidence_source:
        for field_name in ("kind", "locator", "digest"):
            if getattr(left_evidence.evidence_source, field_name) != getattr(
                right_evidence.evidence_source, field_name
            ):
                return f"evidence_source.{field_name}"
    return "unknown field"


def _validate_inventory_identities(bundle: SemanticManifestBundle) -> None:
    entry_ids = tuple(entry.entry_id for entry in bundle.inventory.entries)
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("duplicate inventory entry ID")
    owner_references = tuple(owner.owner_family for owner in bundle.inventory.owners)
    if len(owner_references) != len(set(owner_references)):
        raise ValueError("duplicate source owner family")

    formats_by_id: dict[str, FormatDescriptor] = {}
    for entry in bundle.inventory.entries:
        format_descriptor = _member_format(entry.member)
        existing = formats_by_id.get(format_descriptor.format_id)
        if existing is not None and existing != format_descriptor:
            raise ValueError(
                f"format_id {format_descriptor.format_id} has ambiguous descriptors"
            )
        formats_by_id[format_descriptor.format_id] = format_descriptor
        _validate_reserved_format(format_descriptor)

    entries = bundle.inventory.entries
    for index, left in enumerate(entries):
        for right in entries[index + 1 :]:
            if left.graph_instance_id != right.graph_instance_id:
                continue
            if _members_overlap(left.member, right.member):
                raise ValueError(
                    "duplicate canonical semantic identity across inventory entries "
                    f"{left.entry_id} and {right.entry_id}"
                )


def _validate_reserved_format(format_descriptor: FormatDescriptor) -> None:
    if format_descriptor.format_id == BF16_FORMAT.format_id:
        if format_descriptor != BF16_FORMAT:
            raise ValueError("reserved BF16 format_id has the wrong components")
    if format_descriptor.format_id == MXFP8_FORMAT.format_id:
        if format_descriptor != MXFP8_FORMAT:
            raise ValueError("reserved MXFP8 format_id has the wrong components")


def _members_overlap(
    left: SemanticInventoryMember,
    right: SemanticInventoryMember,
) -> bool:
    if isinstance(left, SemanticTensor) and isinstance(right, SemanticTensor):
        return left.address.semantic_id == right.address.semantic_id
    if isinstance(left, SemanticTensor) and isinstance(right, SemanticTensorFamily):
        return _family_contains_semantic_id(right, left.address.semantic_id)
    if isinstance(left, SemanticTensorFamily) and isinstance(right, SemanticTensor):
        return _family_contains_semantic_id(left, right.address.semantic_id)
    if isinstance(left, SemanticTensorFamily) and isinstance(
        right, SemanticTensorFamily
    ):
        return _families_overlap(left, right)
    raise TypeError("unknown semantic inventory member")


def _family_contains_semantic_id(
    family: SemanticTensorFamily,
    semantic_id: str,
) -> bool:
    prefix = f"{family.pattern.semantic_graph_path}."
    if not semantic_id.startswith(prefix):
        return False
    suffix = semantic_id[len(prefix) :].split(".")
    if len(suffix) != len(family.pattern.path_segments):
        return False
    constraints: dict[str, str] = {}
    for segment, value in zip(family.pattern.path_segments, suffix, strict=True):
        if isinstance(segment, LiteralPathSegment):
            if segment.value != value:
                return False
        else:
            constraints[segment.axis_name] = value
    return _domain_satisfies_rendered_constraints(family.domain, constraints)


def _domain_satisfies_rendered_constraints(
    domain: FamilyIndexDomain,
    constraints: Mapping[str, str],
) -> bool:
    if domain.cardinality == 0:
        return False
    if domain.layer_domain is not None:
        layer_match = any(
            constraints.get(
                "global_decoder_layer",
                str(member.global_decoder_layer),
            )
            == str(member.global_decoder_layer)
            and (
                member.moe_ordinal is None
                or constraints.get("moe_ordinal", str(member.moe_ordinal))
                == str(member.moe_ordinal)
            )
            for member in domain.layer_domain.members
        )
        if not layer_match:
            return False
    for axis in domain.independent_axes:
        required = constraints.get(axis.name)
        if required is not None and required not in {
            str(item) for item in axis.members
        }:
            return False
    return True


def _families_overlap(
    left: SemanticTensorFamily,
    right: SemanticTensorFamily,
) -> bool:
    if left.pattern.semantic_graph_path != right.pattern.semantic_graph_path:
        return False
    if len(left.pattern.path_segments) != len(right.pattern.path_segments):
        return False
    if left.pattern.path_segments == right.pattern.path_segments:
        return _domains_intersect(left.domain, right.domain)

    left_constraints: dict[str, set[str]] = {
        axis: _rendered_axis_values(left.domain, axis)
        for axis in left.domain.axis_names
    }
    right_constraints: dict[str, set[str]] = {
        axis: _rendered_axis_values(right.domain, axis)
        for axis in right.domain.axis_names
    }
    layer_links: list[tuple[str, str]] = []
    for left_segment, right_segment in zip(
        left.pattern.path_segments,
        right.pattern.path_segments,
        strict=True,
    ):
        if isinstance(left_segment, LiteralPathSegment) and isinstance(
            right_segment, LiteralPathSegment
        ):
            if left_segment.value != right_segment.value:
                return False
            continue
        if isinstance(left_segment, IndexPathSegment) and isinstance(
            right_segment, LiteralPathSegment
        ):
            allowed = left_constraints[left_segment.axis_name]
            allowed.intersection_update((right_segment.value,))
            if not allowed:
                return False
            continue
        if isinstance(left_segment, LiteralPathSegment) and isinstance(
            right_segment, IndexPathSegment
        ):
            allowed = right_constraints[right_segment.axis_name]
            allowed.intersection_update((left_segment.value,))
            if not allowed:
                return False
            continue
        if not isinstance(left_segment, IndexPathSegment) or not isinstance(
            right_segment, IndexPathSegment
        ):
            raise TypeError("unknown semantic path segment")
        shared = left_constraints[left_segment.axis_name].intersection(
            right_constraints[right_segment.axis_name]
        )
        if not shared:
            return False
        left_constraints[left_segment.axis_name] = set(shared)
        right_constraints[right_segment.axis_name] = set(shared)
        if (
            left_segment.axis_name in _LAYER_AXES
            and right_segment.axis_name in _LAYER_AXES
        ):
            layer_links.append((left_segment.axis_name, right_segment.axis_name))

    if not _layer_domain_has_allowed_member(left.domain, left_constraints):
        return False
    if not _layer_domain_has_allowed_member(right.domain, right_constraints):
        return False
    if layer_links:
        return _linked_layer_domains_overlap(
            left.domain,
            right.domain,
            left_constraints,
            right_constraints,
            layer_links,
        )
    return True


def _rendered_axis_values(domain: FamilyIndexDomain, axis_name: str) -> set[str]:
    if domain.layer_domain is not None:
        if axis_name == "global_decoder_layer":
            return {
                str(member.global_decoder_layer)
                for member in domain.layer_domain.members
            }
        if axis_name == "moe_ordinal":
            return {
                str(member.moe_ordinal)
                for member in domain.layer_domain.members
                if member.moe_ordinal is not None
            }
    for axis in domain.independent_axes:
        if axis.name == axis_name:
            return {str(member) for member in axis.members}
    raise ValueError(f"unknown domain axis: {axis_name}")


def _layer_value(member: LayerMember, axis_name: str) -> int | None:
    if axis_name == "global_decoder_layer":
        return member.global_decoder_layer
    if axis_name == "moe_ordinal":
        return member.moe_ordinal
    raise ValueError(f"unknown layer axis: {axis_name}")


def _layer_domain_has_allowed_member(
    domain: FamilyIndexDomain,
    constraints: Mapping[str, set[str]],
) -> bool:
    if domain.layer_domain is None:
        return True
    return any(
        all(
            str(_layer_value(member, axis_name)) in constraints[axis_name]
            for axis_name in domain.layer_domain.axis_names
        )
        for member in domain.layer_domain.members
    )


def _linked_layer_domains_overlap(
    left: FamilyIndexDomain,
    right: FamilyIndexDomain,
    left_constraints: Mapping[str, set[str]],
    right_constraints: Mapping[str, set[str]],
    links: list[tuple[str, str]],
) -> bool:
    if left.layer_domain is None or right.layer_domain is None:
        return False
    left_keys = {
        tuple(str(_layer_value(member, left_axis)) for left_axis, _ in links)
        for member in left.layer_domain.members
        if all(
            str(_layer_value(member, axis_name)) in left_constraints[axis_name]
            for axis_name in left.layer_domain.axis_names
        )
    }
    right_keys = {
        tuple(str(_layer_value(member, right_axis)) for _, right_axis in links)
        for member in right.layer_domain.members
        if all(
            str(_layer_value(member, axis_name)) in right_constraints[axis_name]
            for axis_name in right.layer_domain.axis_names
        )
    }
    return not left_keys.isdisjoint(right_keys)


def _domains_intersect(left: FamilyIndexDomain, right: FamilyIndexDomain) -> bool:
    if left.cardinality == 0 or right.cardinality == 0:
        return False
    if set(left.axis_names) != set(right.axis_names):
        return False
    if (left.layer_domain is None) != (right.layer_domain is None):
        return False
    if left.layer_domain is not None and right.layer_domain is not None:
        if not set(left.layer_domain.members).intersection(right.layer_domain.members):
            return False
    right_axes = {axis.name: axis for axis in right.independent_axes}
    for left_axis in left.independent_axes:
        right_axis = right_axes[left_axis.name]
        left_values = {_axis_member_key(item) for item in left_axis.members}
        right_values = {_axis_member_key(item) for item in right_axis.members}
        if not left_values.intersection(right_values):
            return False
    return True


def _domain_is_subset(
    subset: FamilyIndexDomain,
    superset: FamilyIndexDomain,
) -> bool:
    if set(subset.axis_names) != set(superset.axis_names):
        return False
    if (subset.layer_domain is None) != (superset.layer_domain is None):
        return False
    if subset.layer_domain is not None and superset.layer_domain is not None:
        if not set(subset.layer_domain.members).issubset(superset.layer_domain.members):
            return False
    super_axes = {axis.name: axis for axis in superset.independent_axes}
    for sub_axis in subset.independent_axes:
        super_axis = super_axes[sub_axis.name]
        sub_values = {_axis_member_key(item) for item in sub_axis.members}
        super_values = {_axis_member_key(item) for item in super_axis.members}
        if not sub_values.issubset(super_values):
            return False
    return True


def _axis_values(
    domain: FamilyIndexDomain,
    axis_name: str,
) -> set[tuple[int, object]]:
    if domain.layer_domain is not None:
        if axis_name == "global_decoder_layer":
            return {
                _axis_member_key(member.global_decoder_layer)
                for member in domain.layer_domain.members
            }
        if axis_name == "moe_ordinal":
            return {
                _axis_member_key(member.moe_ordinal)
                for member in domain.layer_domain.members
                if member.moe_ordinal is not None
            }
    for axis in domain.independent_axes:
        if axis.name == axis_name:
            return {_axis_member_key(member) for member in axis.members}
    raise ValueError(f"unknown axis {axis_name}")


def _validate_projection(
    source: FamilyIndexDomain,
    target: FamilyIndexDomain,
    projections: tuple[AxisProjection, ...],
    *,
    label: str,
    require_all_source_axes: bool,
) -> None:
    source_axes = set(source.axis_names)
    target_axes = set(target.axis_names)
    projected_source_axes = [item.member_axis for item in projections]
    projected_target_axes = [item.owner_axis for item in projections]
    if any(axis not in source_axes for axis in projected_source_axes):
        raise ValueError(f"{label} projection contains an unknown source axis")
    if any(axis not in target_axes for axis in projected_target_axes):
        raise ValueError(f"{label} projection contains an unknown target axis")
    if len(projected_source_axes) != len(set(projected_source_axes)):
        raise ValueError(f"{label} projection reuses a source axis ambiguously")
    if len(projected_target_axes) != len(set(projected_target_axes)):
        raise ValueError(f"{label} projection reuses a target axis ambiguously")
    if set(projected_target_axes) != target_axes:
        raise ValueError(f"{label} projection must cover every target axis")
    if require_all_source_axes and set(projected_source_axes) != source_axes:
        raise ValueError(f"{label} projection must cover every member axis")

    by_target = {item.owner_axis: item.member_axis for item in projections}
    if target.layer_domain is not None:
        layer_target_axes = target.layer_domain.axis_names
        if any(by_target[axis] not in _LAYER_AXES for axis in layer_target_axes):
            raise ValueError(f"{label} projection cannot invent layer correlation")
        projected_members: set[LayerMember] = set()
        if source.layer_domain is None:
            raise ValueError(f"{label} projection has no source layer domain")
        for source_member in source.layer_domain.members:
            global_value = _layer_value(
                source_member,
                by_target["global_decoder_layer"],
            )
            moe_value = (
                _layer_value(source_member, by_target["moe_ordinal"])
                if "moe_ordinal" in layer_target_axes
                else None
            )
            if global_value is None:
                raise ValueError(f"{label} projection lost global layer coordinate")
            projected_members.add(LayerMember(global_value, moe_value))
        if projected_members != set(target.layer_domain.members):
            raise ValueError(f"{label} projected domain does not match target domain")

    target_independent = {axis.name: axis for axis in target.independent_axes}
    for target_axis_name, target_axis in target_independent.items():
        source_axis_name = by_target[target_axis_name]
        source_values = _axis_values(source, source_axis_name)
        target_values = {_axis_member_key(item) for item in target_axis.members}
        if source_values != target_values:
            raise ValueError(f"{label} projected domain does not match target domain")


def _validate_owner_bindings(bundle: SemanticManifestBundle) -> None:
    owners_by_reference = {
        owner.owner_family: owner for owner in bundle.inventory.owners
    }
    entries_by_id = {entry.entry_id: entry for entry in bundle.inventory.entries}
    directly_referenced_owners: set[OwnerFamilyReference] = set()
    for entry in bundle.inventory.entries:
        member = entry.member
        binding = member.ownership.binding
        member_domain = _member_domain(member)
        if binding.member_domain != member_domain:
            raise ValueError(f"entry {entry.entry_id} ownership member domain mismatch")
        owner = owners_by_reference.get(binding.canonical_owner_family)
        if owner is None:
            raise ValueError(
                f"entry {entry.entry_id} has no canonical owner; "
                "canonical source owner missing: "
                f"{binding.canonical_owner_family}"
            )
        _validate_projection(
            member_domain,
            owner.domain,
            binding.member_to_owner_axes,
            label=f"entry {entry.entry_id} member-to-owner",
            require_all_source_axes=False,
        )

        if entry.value_provenance == ValueProvenance.TIED_ALIAS:
            target = entries_by_id.get(binding.canonical_value_entry_id)
            if target is None:
                raise ValueError(
                    f"alias {entry.entry_id} canonical value target is missing"
                )
            if target.value_provenance == ValueProvenance.TIED_ALIAS:
                raise ValueError(
                    f"alias-to-alias target is forbidden for {entry.entry_id}"
                )
            target_binding = target.member.ownership.binding
            if binding.canonical_owner_family != target_binding.canonical_owner_family:
                raise ValueError(
                    f"alias {entry.entry_id} canonical owner differs from its target"
                )
            if member.logical_shape != target.member.logical_shape:
                raise ValueError(f"alias {entry.entry_id} shape mismatch")
            if member.logical_axes != target.member.logical_axes:
                raise ValueError(f"alias {entry.entry_id} axes mismatch")
            if member.logical_dtype != target.member.logical_dtype:
                raise ValueError(f"alias {entry.entry_id} dtype mismatch")
            if member.format != target.member.format:
                raise ValueError(f"alias {entry.entry_id} format mismatch")
            _validate_projection(
                member_domain,
                _member_domain(target.member),
                binding.member_to_value_axes,
                label=f"alias {entry.entry_id} member-to-value",
                require_all_source_axes=True,
            )
            continue

        if binding.canonical_value_entry_id != entry.entry_id:
            raise ValueError(
                f"non-alias entry {entry.entry_id} must name itself as canonical value"
            )
        if binding.canonical_owner_family.graph_instance_id != entry.graph_instance_id:
            raise ValueError(
                f"direct entry {entry.entry_id} must own storage in its graph"
            )
        _validate_projection(
            member_domain,
            member_domain,
            binding.member_to_value_axes,
            label=f"entry {entry.entry_id} member-to-value",
            require_all_source_axes=True,
        )
        directly_referenced_owners.add(binding.canonical_owner_family)

    all_owner_references = set(owners_by_reference)
    unreferenced = all_owner_references - directly_referenced_owners
    if unreferenced:
        raise ValueError(
            f"unreferenced source owner: {sorted(map(str, unreferenced))[0]}"
        )


def _validate_out_of_scope(bundle: SemanticManifestBundle) -> None:
    entries_by_id = {entry.entry_id: entry for entry in bundle.inventory.entries}
    owners_by_reference = {
        owner.owner_family: owner for owner in bundle.inventory.owners
    }
    for manifest in bundle.manifests:
        for claim in manifest.out_of_scope:
            entry = entries_by_id.get(claim.inventory_entry_id)
            if (
                entry is None
                or claim.inventory_entry_id not in manifest.inventory_entry_ids
            ):
                raise ValueError(
                    "out-of-scope claim must name one whole manifest inventory entry"
                )
            owner = owners_by_reference.get(
                entry.member.ownership.binding.canonical_owner_family
            )
            if owner is None:
                raise ValueError("out-of-scope entry has no canonical source owner")
            if (
                manifest.lifecycle.graph_kind == GraphKind.MAIN
                and owner.source_mutability == SourceMutability.MUTABLE
            ):
                raise ValueError("mutable main-model tensor cannot be out of scope")
            if claim.reason == OutOfScopeReason.SOURCE_PROVEN_FROZEN:
                if owner.source_mutability != SourceMutability.FROZEN:
                    raise ValueError(
                        "source-proven frozen exclusion requires a frozen owner"
                    )
            elif claim.reason == OutOfScopeReason.IMMUTABLE_AUXILIARY:
                if (
                    manifest.lifecycle.graph_kind == GraphKind.MAIN
                    or manifest.lifecycle.rollout_participation
                    != RolloutParticipation.SERVED_FROM_CHECKPOINT
                    or manifest.lifecycle.immutable_evidence is None
                ):
                    raise ValueError(
                        "immutable auxiliary exclusion requires checkpoint evidence"
                    )
            elif claim.reason == OutOfScopeReason.BACKEND_DERIVED_STATE:
                if entry.value_provenance != ValueProvenance.BACKEND_DERIVED:
                    raise ValueError(
                        "backend-derived exclusion requires backend-derived value provenance"
                    )
            else:
                raise TypeError("out-of-scope reason must be OutOfScopeReason")


def _validate_atomic_groups(bundle: SemanticManifestBundle) -> None:
    entries_by_id = {entry.entry_id: entry for entry in bundle.inventory.entries}
    for manifest in bundle.manifests:
        for group in manifest.atomic_groups:
            if group.group_domain.cardinality == 0:
                raise ValueError("atomic group requires a non-empty group domain")
            if not group.participants:
                raise ValueError("atomic group requires non-empty participants")
            participant_ids = tuple(
                participant.inventory_entry_id for participant in group.participants
            )
            if len(participant_ids) != len(set(participant_ids)):
                raise ValueError("atomic group contains duplicate participants")
            for participant in group.participants:
                entry = entries_by_id.get(participant.inventory_entry_id)
                if entry is None:
                    raise ValueError(
                        "atomic group references an unknown inventory entry"
                    )
                if entry.graph_instance_id != manifest.graph_instance_id:
                    raise ValueError(
                        "atomic group participant belongs to another graph"
                    )
                if participant.participant_domain.cardinality == 0:
                    raise ValueError(
                        "atomic group participant requires a non-empty domain"
                    )
                if not _domain_is_subset(
                    participant.participant_domain,
                    _member_domain(entry.member),
                ):
                    raise ValueError(
                        "atomic participant domain is outside its inventory family"
                    )
                _validate_projection(
                    group.group_domain,
                    participant.participant_domain,
                    participant.group_to_participant_axes,
                    label=f"atomic participant {participant.inventory_entry_id}",
                    require_all_source_axes=False,
                )


def _validate_role_registry(bundle: SemanticManifestBundle) -> None:
    keys = tuple(
        (definition.schema_version, definition.role_name)
        for definition in bundle.role_definitions
    )
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate role definition key")
    for definition in bundle.role_definitions:
        if definition.schema_version != bundle.schema_version:
            raise ValueError("role definition schema version does not match bundle")
        builtin_predicate = _BUILTIN_ROLE_PREDICATES.get(definition.role_name)
        if builtin_predicate is not None and definition.predicate != builtin_predicate:
            raise ValueError("adapter cannot replace a built-in role predicate")
        definition.validate_expected_domain(bundle)


def _graph_owner_references(
    bundle: SemanticManifestBundle,
    graph_instance_id: str,
) -> tuple[OwnerFamilyReference, ...]:
    manifest = bundle.manifest(graph_instance_id)
    entries_by_id = {entry.entry_id: entry for entry in bundle.inventory.entries}
    references = {
        entries_by_id[entry_id].member.ownership.binding.canonical_owner_family
        for entry_id in manifest.inventory_entry_ids
        if _member_domain(entries_by_id[entry_id].member).cardinality > 0
    }
    return tuple(
        sorted(
            references,
            key=lambda item: (
                _graph_sort_key(item.graph_instance_id),
                item.owner_family_id,
            ),
        )
    )


def _owner_refit_requirements_unchecked(
    bundle: SemanticManifestBundle,
    graph_instance_id: str,
) -> tuple[tuple[OwnerFamilyReference, RefitRequirement], ...]:
    manifest = bundle.manifest(graph_instance_id)
    references = _graph_owner_references(bundle, graph_instance_id)
    owners_by_reference = {
        owner.owner_family: owner for owner in bundle.inventory.owners
    }
    if manifest.lifecycle.rollout_participation != (
        RolloutParticipation.SERVED_FROM_SOURCE
    ):
        return tuple((reference, RefitRequirement.NONE) for reference in references)
    requirements: list[tuple[OwnerFamilyReference, RefitRequirement]] = []
    for reference in references:
        owner = owners_by_reference[reference]
        if owner.source_mutability == SourceMutability.MUTABLE:
            requirement = RefitRequirement.EVERY_VERSION
        elif owner.source_mutability == SourceMutability.FROZEN:
            requirement = RefitRequirement.INITIAL_ONLY
        else:
            raise ValueError(
                f"served-from-source canonical owner {reference} is absent"
            )
        requirements.append((reference, requirement))
    return tuple(requirements)


def _validate_source_serving(bundle: SemanticManifestBundle) -> None:
    owners_by_reference = {
        owner.owner_family: owner for owner in bundle.inventory.owners
    }
    for manifest in bundle.manifests:
        references = _graph_owner_references(bundle, manifest.graph_instance_id)
        if (
            manifest.lifecycle.rollout_participation
            == RolloutParticipation.SERVED_FROM_SOURCE
        ):
            if not manifest.inventory_entry_ids or not references:
                raise ValueError(
                    "served-from-source graph requires a non-empty semantic domain "
                    "and canonical source owner"
                )
            if any(
                owners_by_reference[reference].source_mutability
                == SourceMutability.ABSENT
                for reference in references
            ):
                raise ValueError("served-from-source canonical owner cannot be absent")
            if (
                sum(
                    _member_domain(entry.member).cardinality
                    for entry in bundle.inventory.entries
                    if entry.graph_instance_id == manifest.graph_instance_id
                )
                == 0
            ):
                raise ValueError(
                    "served-from-source graph requires a non-empty semantic domain"
                )
        if (
            manifest.lifecycle.rollout_participation
            == RolloutParticipation.SERVED_FROM_CHECKPOINT
        ):
            if any(
                owners_by_reference[reference].source_mutability
                == SourceMutability.MUTABLE
                for reference in references
            ):
                raise ValueError(
                    "checkpoint-served graph cannot have a mutable source owner"
                )
