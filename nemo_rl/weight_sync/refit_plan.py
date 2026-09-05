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

"""Framework-free physical-format contracts for refit planning.

Logical precision is intentionally insufficient to authorize a transfer.  A
copy or transform is admitted only for one adjacent physical-stage pair whose
full ordered representations, exact route, and adapter-issued capability
proofs agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from hashlib import sha256
import hmac
import json
import re
import secrets
from threading import RLock
from typing import Self, TypeVar, cast

from nemo_rl.precision_policy.compiler import (
    CompiledPrecisionIntentGroup,
    CompiledPrecisionSelectionGroup,
    RuntimeSourceBindingSlice,
)
from nemo_rl.precision_policy.runtime_binding import (
    RuntimeSourceDiscoveryRequest,
    RuntimeSourceDiscoveryResult,
    validate_compiled_precision_intent_group,
)
from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    MXFP8_FORMAT,
    ComponentRole,
    FormatDescriptor,
    LiteralComponentAxisSpec,
    LogicalComponentAxisSpec,
    SelectionTopologyEntry,
)
from nemo_rl.precision_policy.source_discovery import SourceDiscoveryRecord
from nemo_rl.precision_policy.source_storage import (
    IDENTITY_PERMUTATION_ID,
    IDENTITY_SWIZZLE_ID,
    SourceLiteralAxisExtent,
    SourceNormalizedAxisExtent,
    SourceNormalizerManifest,
    SourcePaddingSemantics,
    SourcePhysicalAxisSpec,
    SourceStorageComponent,
    SourceStorageRealization,
    SourceStorageRealizationInventory,
    source_normalizer_manifest_digest,
    source_storage_inventory_digest,
    validate_source_storage_realization_inventory,
)
from nemo_rl.precision_policy.topology import (
    ComponentAxisTarget,
    FamilyIndexAxisTarget,
    LayerCoordinateTarget,
    SourceRegion,
    project_source_region_to_member_domain,
)


_SHA256_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_HMAC_SHA256_PATTERN = re.compile(r"hmac-sha256:[0-9a-f]{64}")
_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*")
_VERSIONED_SEMANTIC_ID_PATTERN = re.compile(
    r"[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*\.v(?:0|[1-9][0-9]*)"
)
_IMPLEMENTATION_VERSION_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){0,2}"
)
_AXIS_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_SCALAR_SEQUENCE_TYPES = (str, bytes, bytearray, memoryview)
_SequenceItemT = TypeVar("_SequenceItemT")
_ADAPTER_REGISTRY_FACTORY_TOKEN = object()
_FORMAT_SCHEMA_REGISTRY_FACTORY_TOKEN = object()
_REFIT_PLANNER_FACTORY_TOKEN = object()
_DESTINATION_BINDING_EVIDENCE_FACTORY_TOKEN = object()
_DESTINATION_REGISTRY_FACTORY_TOKEN = object()
_UNVALIDATED_SOURCE_EXTRACTION = object()


class TransformLocus(StrEnum):
    """Endpoint responsible for one selected physical transformation."""

    NONE = "none"
    SOURCE = "source"
    DESTINATION = "destination"
    DESTINATION_NATIVE_LOADER = "destination_native_loader"


class PhysicalFormatStage(StrEnum):
    """Ordered physical boundaries crossed by one refit component."""

    SOURCE_STORAGE = "source_storage"
    WIRE = "wire"
    DESTINATION_LOAD_API = "destination_load_api"
    DESTINATION_RUNTIME = "destination_runtime"


class PhysicalPaddingSemantics(StrEnum):
    """Observable guarantee for carrier capacity outside the logical view."""

    ZERO_FILLED = "zero_filled"
    UNSPECIFIED_IGNORED = "unspecified_ignored"


class SourceRegionTransformKind(StrEnum):
    """Exact operation class authorized for a compact source subregion."""

    GATHER_SPLIT = "gather_split"


_STAGE_ORDER = (
    PhysicalFormatStage.SOURCE_STORAGE,
    PhysicalFormatStage.WIRE,
    PhysicalFormatStage.DESTINATION_LOAD_API,
    PhysicalFormatStage.DESTINATION_RUNTIME,
)
_ALLOWED_TRANSFORM_LOCI = {
    (
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    ): frozenset({TransformLocus.SOURCE}),
    (
        PhysicalFormatStage.WIRE,
        PhysicalFormatStage.DESTINATION_LOAD_API,
    ): frozenset({TransformLocus.DESTINATION}),
    (
        PhysicalFormatStage.DESTINATION_LOAD_API,
        PhysicalFormatStage.DESTINATION_RUNTIME,
    ): frozenset(
        {
            TransformLocus.DESTINATION,
            TransformLocus.DESTINATION_NATIVE_LOADER,
        }
    ),
}


def _require_enum(value: object, enum_type: type[StrEnum], name: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be {enum_type.__name__}")
    member = cast(StrEnum, value)
    if type(member.value) is not str or str.__str__(member) != member.value:
        raise TypeError(
            f"{name} must be a registered {enum_type.__name__} with an exact "
            "canonical string value"
        )
    try:
        registered_member = enum_type(member.value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a registered {enum_type.__name__}") from error
    if registered_member is not value:
        raise TypeError(f"{name} must be a registered {enum_type.__name__}")


def _require_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if (
        not value
        or value != value.strip()
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be exact non-empty text without whitespace")
    return value


def _require_identifier(value: object, name: str) -> str:
    text = _require_text(value, name)
    if _IDENTIFIER_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase canonical identifier")
    return text


def _require_versioned_semantic_identifier(value: object, name: str) -> str:
    text = _require_text(value, name)
    if _VERSIONED_SEMANTIC_ID_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a versioned semantic identifier")
    return text


def _require_implementation_version(value: object, name: str) -> str:
    text = _require_text(value, name)
    if _IMPLEMENTATION_VERSION_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be an exact numeric implementation version")
    return text


def _require_axis_name(value: object, name: str) -> str:
    text = _require_text(value, name)
    if _AXIS_NAME_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase axis name")
    return text


def _require_sha256_digest(value: object, name: str) -> str:
    text = _require_text(value, name)
    if _SHA256_DIGEST_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a canonical SHA-256 digest")
    return text


def _require_exact_structural_match(
    candidate: object,
    expected: object,
    path: str,
    seen: set[tuple[int, int]] | None = None,
) -> None:
    """Compare transported artifacts without trusting user-defined equality."""
    if type(candidate) is not type(expected):
        raise ValueError(f"{path} has a non-exact runtime type")
    if seen is None:
        seen = set()
    identity = (id(candidate), id(expected))
    if identity in seen:
        return
    seen.add(identity)
    if is_dataclass(expected) and not isinstance(expected, type):
        for item in fields(expected):
            _require_exact_structural_match(
                getattr(candidate, item.name),
                getattr(expected, item.name),
                f"{path}.{item.name}",
                seen,
            )
        return
    if type(expected) is tuple:
        candidate_tuple = cast(tuple[object, ...], candidate)
        expected_tuple = cast(tuple[object, ...], expected)
        if len(candidate_tuple) != len(expected_tuple):
            raise ValueError(f"{path} has a different tuple length")
        for index, (candidate_item, expected_item) in enumerate(
            zip(candidate_tuple, expected_tuple, strict=True)
        ):
            _require_exact_structural_match(
                candidate_item,
                expected_item,
                f"{path}[{index}]",
                seen,
            )
        return
    if type(expected) is frozenset:
        candidate_items = sorted(
            cast(frozenset[object], candidate),
            key=repr,
        )
        expected_items = sorted(
            cast(frozenset[object], expected),
            key=repr,
        )
        _require_exact_structural_match(
            tuple(candidate_items),
            tuple(expected_items),
            path,
            seen,
        )
        return
    if type(expected) is dict:
        candidate_dict = cast(dict[object, object], candidate)
        expected_dict = cast(dict[object, object], expected)
        if type(candidate_dict.keys()) is not type(expected_dict.keys()):
            raise ValueError(f"{path} has a different mapping structure")
        if set(candidate_dict) != set(expected_dict):
            raise ValueError(f"{path} has different mapping keys")
        for key in sorted(expected_dict, key=repr):
            _require_exact_structural_match(
                candidate_dict[key],
                expected_dict[key],
                f"{path}[{key!r}]",
                seen,
            )
        return
    if candidate != expected:
        raise ValueError(f"{path} differs from its active canonical derivation")


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _snapshot_sequence(
    value: Sequence[_SequenceItemT],
    name: str,
) -> tuple[_SequenceItemT, ...]:
    if isinstance(value, _SCALAR_SEQUENCE_TYPES) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a non-scalar sequence")
    return tuple(value)


def _snapshot_axes(value: Sequence[str], name: str) -> tuple[str, ...]:
    axes = _snapshot_sequence(value, name)
    for axis in axes:
        _require_axis_name(axis, name)
    if len(axes) != len(set(axes)):
        raise ValueError(f"{name} must be duplicate-free")
    return axes


def _require_component_role(value: object) -> ComponentRole:
    text = _require_text(value, "component role")
    if not all(
        character.isascii() and (character.isalnum() or character in {"_", "-"})
        for character in text
    ):
        raise ValueError("component role must use the shared canonical atom syntax")
    return ComponentRole(text)


def _require_canonical_reserved_format(
    format_descriptor: FormatDescriptor,
) -> None:
    reserved_formats = {
        BF16_FORMAT.format_id: BF16_FORMAT,
        MXFP8_FORMAT.format_id: MXFP8_FORMAT,
    }
    reserved = reserved_formats.get(format_descriptor.format_id)
    if reserved is not None and format_descriptor != reserved:
        label = "BF16" if reserved is BF16_FORMAT else "MXFP8"
        raise ValueError(f"reserved {label} format_id has the wrong components")


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_digest(payload: object) -> str:
    return f"sha256:{sha256(_canonical_json_bytes(payload)).hexdigest()}"


def _require_adjacent_stage_pair(
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
) -> None:
    _require_enum(source_stage, PhysicalFormatStage, "source_stage")
    _require_enum(destination_stage, PhysicalFormatStage, "destination_stage")
    source_index = _STAGE_ORDER.index(source_stage)
    destination_index = _STAGE_ORDER.index(destination_stage)
    if destination_index != source_index + 1:
        raise ValueError("refit transforms require adjacent physical stages")


@dataclass(frozen=True, slots=True)
class PhysicalAxisMapping:
    """Mapping from one logical axis onto one or more carrier axes."""

    logical_axis: str
    physical_axes: tuple[str, ...]
    mapping_id: str

    def __post_init__(self) -> None:
        _require_axis_name(self.logical_axis, "logical_axis")
        physical_axes = _snapshot_axes(self.physical_axes, "physical_axes")
        if not physical_axes:
            raise ValueError("physical_axes must be non-empty")
        _require_versioned_semantic_identifier(self.mapping_id, "mapping_id")
        object.__setattr__(self, "physical_axes", physical_axes)


@dataclass(frozen=True, slots=True)
class PhysicalPadding:
    """Padding around one logical axis in a physical allocation."""

    logical_axis: str
    pad_before: int
    pad_after: int
    semantics: PhysicalPaddingSemantics
    fill_encoding: str | None

    def __post_init__(self) -> None:
        _require_axis_name(self.logical_axis, "padding logical_axis")
        pad_before = _require_nonnegative_int(self.pad_before, "padding pad_before")
        pad_after = _require_nonnegative_int(self.pad_after, "padding pad_after")
        if pad_before == 0 and pad_after == 0:
            raise ValueError("padding must add physical capacity")
        _require_enum(self.semantics, PhysicalPaddingSemantics, "padding semantics")
        if self.semantics is PhysicalPaddingSemantics.ZERO_FILLED:
            if self.fill_encoding is None:
                raise ValueError("ZERO_FILLED padding requires fill_encoding")
            _require_versioned_semantic_identifier(
                self.fill_encoding,
                "padding fill_encoding",
            )
        elif self.fill_encoding is not None:
            raise ValueError("UNSPECIFIED_IGNORED padding requires fill_encoding=None")


@dataclass(frozen=True, slots=True)
class PhysicalPermutation:
    """Named reordering from logical carrier axes to runtime storage axes."""

    permutation_id: str
    input_axis_order: tuple[str, ...]
    output_axis_order: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_versioned_semantic_identifier(
            self.permutation_id,
            "permutation_id",
        )
        input_axes = _snapshot_axes(self.input_axis_order, "input_axis_order")
        output_axes = _snapshot_axes(self.output_axis_order, "output_axis_order")
        if not input_axes or not output_axes:
            raise ValueError("permutation axis orders must be non-empty")
        if set(input_axes) != set(output_axes):
            raise ValueError("permutation input and output must contain the same axes")
        object.__setattr__(self, "input_axis_order", input_axes)
        object.__setattr__(self, "output_axis_order", output_axes)


@dataclass(frozen=True, slots=True)
class EndpointPlacement:
    """Rank-local location of one component representation."""

    rank: int
    device_type: str
    memory_space: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.rank, "endpoint rank")
        _require_identifier(self.device_type, "endpoint device_type")
        _require_identifier(self.memory_space, "endpoint memory_space")


@dataclass(frozen=True, slots=True)
class PhysicalRouteDescriptor:
    """Exact endpoint identities and capabilities for one adjacent route."""

    source_stage: PhysicalFormatStage
    destination_stage: PhysicalFormatStage
    route_id: str
    source_endpoint_instance_id: str
    destination_endpoint_instance_id: str
    source_endpoint_capability_fingerprint: str
    destination_endpoint_capability_fingerprint: str

    def __post_init__(self) -> None:
        _require_adjacent_stage_pair(self.source_stage, self.destination_stage)
        _require_text(self.route_id, "route_id")
        _require_text(
            self.source_endpoint_instance_id,
            "source_endpoint_instance_id",
        )
        _require_text(
            self.destination_endpoint_instance_id,
            "destination_endpoint_instance_id",
        )
        _require_sha256_digest(
            self.source_endpoint_capability_fingerprint,
            "source_endpoint_capability_fingerprint",
        )
        _require_sha256_digest(
            self.destination_endpoint_capability_fingerprint,
            "destination_endpoint_capability_fingerprint",
        )


@dataclass(frozen=True, slots=True)
class PhysicalLayoutDescriptor:
    """Complete carrier layout independent from endpoint placement."""

    axis_order: tuple[str, ...]
    logical_to_physical_axes: tuple[PhysicalAxisMapping, ...]
    padding: tuple[PhysicalPadding, ...]
    permutation: PhysicalPermutation | None
    storage_encoding: str
    swizzle_id: str = IDENTITY_SWIZZLE_ID

    def __post_init__(self) -> None:
        axis_order = _snapshot_axes(self.axis_order, "axis_order")
        mappings = _snapshot_sequence(
            self.logical_to_physical_axes,
            "logical_to_physical_axes",
        )
        if any(type(mapping) is not PhysicalAxisMapping for mapping in mappings):
            raise TypeError(
                "logical_to_physical_axes must contain PhysicalAxisMapping records"
            )
        logical_axes = tuple(mapping.logical_axis for mapping in mappings)
        if len(logical_axes) != len(set(logical_axes)):
            raise ValueError("logical axis mappings must be duplicate-free")
        mapped_physical_axes = tuple(
            physical_axis
            for mapping in mappings
            for physical_axis in mapping.physical_axes
        )
        if set(mapped_physical_axes) != set(axis_order):
            raise ValueError("logical axis mappings must cover every physical axis")

        padding = _snapshot_sequence(self.padding, "padding")
        if any(type(item) is not PhysicalPadding for item in padding):
            raise TypeError("padding must contain PhysicalPadding records")
        padding_axes = tuple(item.logical_axis for item in padding)
        if len(padding_axes) != len(set(padding_axes)):
            raise ValueError("padding logical axes must be duplicate-free")
        if any(axis not in set(logical_axes) for axis in padding_axes):
            raise ValueError("padding logical axes must have a physical axis mapping")

        if self.permutation is not None:
            if type(self.permutation) is not PhysicalPermutation:
                raise TypeError("permutation must be PhysicalPermutation or None")
            if self.permutation.output_axis_order != axis_order:
                raise ValueError("permutation output_axis_order must equal axis_order")
        elif not axis_order and (mappings or padding):
            raise ValueError("rank-zero layouts cannot contain mappings or padding")

        _require_versioned_semantic_identifier(
            self.storage_encoding,
            "storage_encoding",
        )
        _require_versioned_semantic_identifier(self.swizzle_id, "swizzle_id")
        object.__setattr__(self, "axis_order", axis_order)
        object.__setattr__(self, "logical_to_physical_axes", mappings)
        object.__setattr__(self, "padding", padding)


@dataclass(frozen=True, slots=True)
class PhysicalRepresentation:
    """One role's dtype, shape, and complete physical layout."""

    role: ComponentRole
    physical_dtype: str
    physical_shape: tuple[int, ...]
    layout: PhysicalLayoutDescriptor

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _require_component_role(self.role))
        _require_identifier(self.physical_dtype, "physical_dtype")
        physical_shape = _snapshot_sequence(self.physical_shape, "physical shape")
        for extent in physical_shape:
            if type(extent) is not int:
                raise TypeError("physical shape extents must be integers")
            if extent <= 0:
                raise ValueError("physical shape extents must be positive")
        if type(self.layout) is not PhysicalLayoutDescriptor:
            raise TypeError("layout must be PhysicalLayoutDescriptor")
        if len(physical_shape) != len(self.layout.axis_order):
            raise ValueError("physical shape rank must match layout rank")
        object.__setattr__(self, "physical_shape", physical_shape)


@dataclass(frozen=True, slots=True)
class PhysicalComponentDescriptor:
    """A physical representation paired with its endpoint placement."""

    representation: PhysicalRepresentation
    placement: EndpointPlacement
    source_storage_component: SourceStorageComponent | None = None

    def __post_init__(self) -> None:
        if type(self.representation) is not PhysicalRepresentation:
            raise TypeError("representation must be PhysicalRepresentation")
        if type(self.placement) is not EndpointPlacement:
            raise TypeError("placement must be EndpointPlacement")
        if (
            self.source_storage_component is not None
            and type(self.source_storage_component) is not SourceStorageComponent
        ):
            raise TypeError(
                "source_storage_component must be SourceStorageComponent or None"
            )


class FormatSchemaRegistry:
    """Immutable trusted authority for complete logical component schemas."""

    __slots__ = ("_formats",)
    _formats: frozenset[FormatDescriptor]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "FormatSchemaRegistry is created only inside a validated semantic "
            "or version-adapter trust boundary"
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("format schema registry is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("format schema registry is immutable")

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is FormatSchemaRegistry
            and self._formats == cast(FormatSchemaRegistry, other)._formats
        )

    def __hash__(self) -> int:
        return hash(self._formats)

    @classmethod
    def _create(
        cls,
        factory_token: object,
        formats: Sequence[FormatDescriptor],
    ) -> Self:
        if factory_token is not _FORMAT_SCHEMA_REGISTRY_FACTORY_TOKEN:
            raise TypeError("invalid format schema registry factory authority")
        captured_formats = _snapshot_sequence(formats, "format schemas")
        if not captured_formats:
            raise ValueError("format schemas must be non-empty")
        if any(type(item) is not FormatDescriptor for item in captured_formats):
            raise TypeError(
                "format schemas must contain exact FormatDescriptor records"
            )
        formats_by_id: dict[str, FormatDescriptor] = {}
        for format_descriptor in captured_formats:
            _require_canonical_reserved_format(format_descriptor)
            previous = formats_by_id.setdefault(
                format_descriptor.format_id,
                format_descriptor,
            )
            if previous != format_descriptor:
                raise ValueError(
                    f"format_id {format_descriptor.format_id} has ambiguous schemas"
                )
        registry = object.__new__(cls)
        object.__setattr__(registry, "_formats", frozenset(formats_by_id.values()))
        return registry

    def _require_registered(self, format_descriptor: FormatDescriptor) -> None:
        if type(format_descriptor) is not FormatDescriptor:
            raise TypeError("format schema must be an exact FormatDescriptor")
        if format_descriptor not in self._formats:
            raise ValueError(
                "format descriptor is not registered by the trusted schema authority"
            )

    def __reduce__(self) -> tuple[object, tuple[tuple[FormatDescriptor, ...]]]:
        """Restore through validation without carrying process-local authority."""
        ordered_formats = tuple(sorted(self._formats, key=lambda item: item.format_id))
        return (_restore_format_schema_registry, (ordered_formats,))


def _create_format_schema_registry(
    formats: Sequence[FormatDescriptor],
) -> FormatSchemaRegistry:
    """Create a schema authority inside semantic compiler or adapter code."""
    return FormatSchemaRegistry._create(
        _FORMAT_SCHEMA_REGISTRY_FACTORY_TOKEN,
        formats,
    )


def _restore_format_schema_registry(
    formats: tuple[FormatDescriptor, ...],
) -> FormatSchemaRegistry:
    return _create_format_schema_registry(formats)


@dataclass(frozen=True, slots=True)
class RealizedBindingFormat:
    """Ordered physical components and their atomic schemas at every boundary.

    Each stage's :class:`FormatDescriptor` is the semantic authority for one
    complete atomic component family.  Physical dtypes and layouts may differ
    from the logical descriptor, but the exact ordered role set may not be
    inferred, truncated, or extended by an adapter.
    """

    source_storage: tuple[PhysicalComponentDescriptor, ...]
    wire: tuple[PhysicalComponentDescriptor, ...]
    destination_load_api: tuple[PhysicalComponentDescriptor, ...]
    destination_runtime: tuple[PhysicalComponentDescriptor, ...]
    source_storage_format: FormatDescriptor
    wire_format: FormatDescriptor
    destination_load_api_format: FormatDescriptor
    destination_runtime_format: FormatDescriptor
    format_schema_registry: FormatSchemaRegistry
    routes: tuple[PhysicalRouteDescriptor, ...]

    def __post_init__(self) -> None:
        if type(self.format_schema_registry) is not FormatSchemaRegistry:
            raise TypeError(
                "format_schema_registry must be a trusted FormatSchemaRegistry"
            )
        stage_fields = (
            ("source_storage", "source_storage_format"),
            ("wire", "wire_format"),
            ("destination_load_api", "destination_load_api_format"),
            ("destination_runtime", "destination_runtime_format"),
        )
        for field_name, format_field_name in stage_fields:
            components = _snapshot_sequence(getattr(self, field_name), field_name)
            if not components:
                raise ValueError(f"{field_name} components must be non-empty")
            if any(
                type(component) is not PhysicalComponentDescriptor
                for component in components
            ):
                raise TypeError(
                    f"{field_name} must contain PhysicalComponentDescriptor records"
                )
            roles = tuple(component.representation.role for component in components)
            if len(roles) != len(set(roles)):
                raise ValueError(f"{field_name} contains a duplicate component role")
            format_descriptor = getattr(self, format_field_name)
            if type(format_descriptor) is not FormatDescriptor:
                raise TypeError(f"{format_field_name} must be FormatDescriptor")
            _require_canonical_reserved_format(format_descriptor)
            self.format_schema_registry._require_registered(format_descriptor)
            expected_roles = tuple(
                component.role for component in format_descriptor.components
            )
            if roles != expected_roles:
                raise ValueError(
                    f"{field_name} must realize the complete ordered component roles "
                    f"declared by {format_field_name}"
                )
            object.__setattr__(self, field_name, components)
        routes = _snapshot_sequence(self.routes, "routes")
        if any(type(route) is not PhysicalRouteDescriptor for route in routes):
            raise TypeError("routes must contain PhysicalRouteDescriptor records")
        route_pairs = tuple(
            (route.source_stage, route.destination_stage) for route in routes
        )
        expected_pairs = tuple(_ALLOWED_TRANSFORM_LOCI)
        if len(route_pairs) != len(set(route_pairs)):
            raise ValueError("routes must not duplicate an adjacent stage pair")
        if set(route_pairs) != set(expected_pairs):
            raise ValueError("routes must cover every adjacent physical stage pair")
        route_ids = tuple(route.route_id for route in routes)
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route_id values must be unique")
        ordered_routes = tuple(
            next(
                route
                for route in routes
                if (route.source_stage, route.destination_stage) == pair
            )
            for pair in expected_pairs
        )
        for previous, following in zip(ordered_routes, ordered_routes[1:]):
            if (
                previous.destination_endpoint_instance_id
                != following.source_endpoint_instance_id
                or previous.destination_endpoint_capability_fingerprint
                != following.source_endpoint_capability_fingerprint
            ):
                raise ValueError(
                    "adjacent routes must agree on their shared endpoint identity "
                    "and capability"
                )
        object.__setattr__(self, "routes", ordered_routes)

    def components_at(
        self,
        stage: PhysicalFormatStage,
    ) -> tuple[PhysicalComponentDescriptor, ...]:
        """Return the immutable ordered component tuple for one exact stage."""
        _require_enum(stage, PhysicalFormatStage, "stage")
        if stage is PhysicalFormatStage.SOURCE_STORAGE:
            return self.source_storage
        if stage is PhysicalFormatStage.WIRE:
            return self.wire
        if stage is PhysicalFormatStage.DESTINATION_LOAD_API:
            return self.destination_load_api
        return self.destination_runtime

    def route_between(
        self,
        source_stage: PhysicalFormatStage,
        destination_stage: PhysicalFormatStage,
    ) -> PhysicalRouteDescriptor:
        """Return the exact declared route for one adjacent stage pair."""
        _require_adjacent_stage_pair(source_stage, destination_stage)
        return next(
            route
            for route in self.routes
            if route.source_stage is source_stage
            and route.destination_stage is destination_stage
        )

    def format_at(self, stage: PhysicalFormatStage) -> FormatDescriptor:
        """Return the semantic atomic-component schema for one stage."""
        _require_enum(stage, PhysicalFormatStage, "stage")
        if stage is PhysicalFormatStage.SOURCE_STORAGE:
            return self.source_storage_format
        if stage is PhysicalFormatStage.WIRE:
            return self.wire_format
        if stage is PhysicalFormatStage.DESTINATION_LOAD_API:
            return self.destination_load_api_format
        return self.destination_runtime_format


def _component_axis_payload(
    axis: LogicalComponentAxisSpec | LiteralComponentAxisSpec,
) -> dict[str, object]:
    if isinstance(axis, LogicalComponentAxisSpec):
        return {
            "kind": "logical",
            "logical_axis": axis.logical_axis,
            "divisor": axis.divisor,
            "rounding": axis.rounding.value,
        }
    return {
        "kind": "literal",
        "axis_name": axis.axis_name,
        "extent": axis.extent,
    }


def _format_descriptor_payload(format_descriptor: FormatDescriptor) -> object:
    return {
        "format_id": format_descriptor.format_id,
        "family": format_descriptor.family,
        "components": [
            {
                "role": component.role,
                "dtype": component.dtype,
                "encoding": component.encoding,
                "component_axes": (
                    None
                    if component.component_axes is None
                    else [
                        _component_axis_payload(axis)
                        for axis in component.component_axes
                    ]
                ),
            }
            for component in format_descriptor.components
        ],
    }


def _layout_payload(layout: PhysicalLayoutDescriptor) -> object:
    return {
        "axis_order": list(layout.axis_order),
        "logical_to_physical_axes": [
            {
                "logical_axis": mapping.logical_axis,
                "physical_axes": list(mapping.physical_axes),
                "mapping_id": mapping.mapping_id,
            }
            for mapping in layout.logical_to_physical_axes
        ],
        "padding": [
            {
                "logical_axis": item.logical_axis,
                "pad_before": item.pad_before,
                "pad_after": item.pad_after,
                "semantics": item.semantics.value,
                "fill_encoding": item.fill_encoding,
            }
            for item in layout.padding
        ],
        "permutation": (
            None
            if layout.permutation is None
            else {
                "permutation_id": layout.permutation.permutation_id,
                "input_axis_order": list(layout.permutation.input_axis_order),
                "output_axis_order": list(layout.permutation.output_axis_order),
            }
        ),
        "storage_encoding": layout.storage_encoding,
        "swizzle_id": layout.swizzle_id,
    }


def _source_storage_component_payload(
    component: SourceStorageComponent,
) -> dict[str, object]:
    return {
        "graph_instance_id": component.graph_instance_id,
        "native_component_id": component.native_component_id,
        "source_native_name": component.source_native_name,
        "component_role": component.component_role,
        "carrier_dtype": component.carrier_dtype.value,
        "physical_shape": list(component.physical_shape),
        "physical_axes": [
            {
                "axis_name": axis.axis_name,
                "extent": (
                    {
                        "kind": "normalized",
                        "normalized_axis_indices": list(
                            axis.extent.normalized_axis_indices
                        ),
                        "divisor": axis.extent.divisor,
                        "rounding": axis.extent.rounding.value,
                        "alignment": axis.extent.alignment,
                    }
                    if isinstance(axis.extent, SourceNormalizedAxisExtent)
                    else {
                        "kind": "literal",
                        "extent": axis.extent.extent,
                    }
                ),
            }
            for axis in component.physical_axes
        ],
        "storage_encoding": component.storage_encoding,
        "padding_semantics": component.padding_semantics.value,
        "padding_fill_encoding": component.padding_fill_encoding,
        "permutation_id": component.permutation_id,
        "swizzle_id": component.swizzle_id,
    }


def physical_representation_digest(
    realized_format: RealizedBindingFormat,
    stage: PhysicalFormatStage,
) -> str:
    """Digest ordered representations at one stage, excluding placement."""
    if type(realized_format) is not RealizedBindingFormat:
        raise TypeError("realized_format must be RealizedBindingFormat")
    components = realized_format.components_at(stage)
    return _canonical_digest(
        {
            "atomic_component_schema": _format_descriptor_payload(
                realized_format.format_at(stage)
            ),
            "components": [
                {
                    "role": component.representation.role,
                    "physical_dtype": component.representation.physical_dtype,
                    "physical_shape": list(component.representation.physical_shape),
                    "layout": _layout_payload(component.representation.layout),
                    "source_storage_component": (
                        None
                        if component.source_storage_component is None
                        else _source_storage_component_payload(
                            component.source_storage_component
                        )
                    ),
                }
                for component in components
            ],
        }
    )


def endpoint_placement_digest(
    realized_format: RealizedBindingFormat,
    stage: PhysicalFormatStage,
) -> str:
    """Digest ordered role-to-placement assignments at one physical stage."""
    if type(realized_format) is not RealizedBindingFormat:
        raise TypeError("realized_format must be RealizedBindingFormat")
    components = realized_format.components_at(stage)
    return _canonical_digest(
        {
            "components": [
                {
                    "role": component.representation.role,
                    "rank": component.placement.rank,
                    "device_type": component.placement.device_type,
                    "memory_space": component.placement.memory_space,
                }
                for component in components
            ]
        }
    )


@dataclass(frozen=True, slots=True)
class SourceRegionExtractionCapability:
    """Typed adapter capability for one exact compact source extraction."""

    kind: SourceRegionTransformKind
    source_binding_set_digest: str
    source_region_digest: str
    selected_source_shapes: tuple[tuple[int, ...], ...]
    source_component_ids: tuple[str, ...]
    wire_component_roles: tuple[ComponentRole, ...]
    wire_component_shapes: tuple[tuple[int, ...], ...]
    wire_component_axis_orders: tuple[tuple[str, ...], ...]
    wire_representation_digest: str

    def __post_init__(self) -> None:
        _require_enum(self.kind, SourceRegionTransformKind, "source region transform")
        _require_sha256_digest(
            self.source_binding_set_digest,
            "source_binding_set_digest",
        )
        _require_sha256_digest(self.source_region_digest, "source_region_digest")
        selected_shapes = tuple(
            _snapshot_sequence(shape, "selected source shape")
            for shape in _snapshot_sequence(
                self.selected_source_shapes,
                "selected source shapes",
            )
        )
        component_ids = _snapshot_sequence(
            self.source_component_ids,
            "source component IDs",
        )
        roles = _snapshot_sequence(self.wire_component_roles, "wire component roles")
        wire_shapes = tuple(
            _snapshot_sequence(shape, "wire component shape")
            for shape in _snapshot_sequence(
                self.wire_component_shapes,
                "wire component shapes",
            )
        )
        axis_orders = tuple(
            _snapshot_axes(axis_order, "wire component axis order")
            for axis_order in _snapshot_sequence(
                self.wire_component_axis_orders,
                "wire component axis orders",
            )
        )
        count = len(roles)
        if count == 0 or any(
            len(items) != count
            for items in (selected_shapes, component_ids, wire_shapes, axis_orders)
        ):
            raise ValueError(
                "source region extraction must describe every atomic wire component"
            )
        if len(roles) != len(set(roles)):
            raise ValueError("source region extraction roles must be duplicate-free")
        for role in roles:
            _require_component_role(role)
        for component_id in component_ids:
            _require_text(component_id, "source component ID")
        for shape in (*selected_shapes, *wire_shapes):
            if not shape or any(
                type(extent) is not int or extent <= 0 for extent in shape
            ):
                raise ValueError("source region extraction shapes must be positive")
        _require_sha256_digest(
            self.wire_representation_digest,
            "wire_representation_digest",
        )
        object.__setattr__(self, "selected_source_shapes", selected_shapes)
        object.__setattr__(self, "source_component_ids", component_ids)
        object.__setattr__(self, "wire_component_roles", roles)
        object.__setattr__(self, "wire_component_shapes", wire_shapes)
        object.__setattr__(self, "wire_component_axis_orders", axis_orders)


def _source_region_extraction_payload(
    capability: SourceRegionExtractionCapability,
) -> dict[str, object]:
    return {
        "kind": capability.kind.value,
        "source_binding_set_digest": capability.source_binding_set_digest,
        "source_region_digest": capability.source_region_digest,
        "selected_source_shapes": [
            list(shape) for shape in capability.selected_source_shapes
        ],
        "source_component_ids": list(capability.source_component_ids),
        "wire_component_roles": list(capability.wire_component_roles),
        "wire_component_shapes": [
            list(shape) for shape in capability.wire_component_shapes
        ],
        "wire_component_axis_orders": [
            list(axis_order) for axis_order in capability.wire_component_axis_orders
        ],
        "wire_representation_digest": capability.wire_representation_digest,
    }


def _validate_source_region_extraction_capability(
    capability: SourceRegionExtractionCapability,
) -> SourceRegionExtractionCapability:
    if type(capability) is not SourceRegionExtractionCapability:
        raise TypeError("source region extraction capability must be exact")
    canonical = SourceRegionExtractionCapability(
        kind=capability.kind,
        source_binding_set_digest=capability.source_binding_set_digest,
        source_region_digest=capability.source_region_digest,
        selected_source_shapes=capability.selected_source_shapes,
        source_component_ids=capability.source_component_ids,
        wire_component_roles=capability.wire_component_roles,
        wire_component_shapes=capability.wire_component_shapes,
        wire_component_axis_orders=capability.wire_component_axis_orders,
        wire_representation_digest=capability.wire_representation_digest,
    )
    if capability != canonical:
        raise ValueError("source region extraction capability is not canonical")
    return capability


@dataclass(frozen=True, slots=True)
class AdapterOperationCapability:
    """One exact operation admitted by trusted version-adapter code.

    Constructing this inert descriptor grants no authority.  Only a registry
    created inside the adapter trust boundary can issue proofs for it.
    """

    source_stage: PhysicalFormatStage
    destination_stage: PhysicalFormatStage
    transform_locus: TransformLocus
    route_id: str
    source_endpoint_instance_id: str
    destination_endpoint_instance_id: str
    source_endpoint_capability_fingerprint: str
    destination_endpoint_capability_fingerprint: str
    source_representation_digest: str
    destination_representation_digest: str
    source_placement_digest: str
    destination_placement_digest: str
    implementation_id: str
    implementation_version: str
    source_region_extraction: SourceRegionExtractionCapability | None = None

    def __post_init__(self) -> None:
        _require_adjacent_stage_pair(self.source_stage, self.destination_stage)
        _require_enum(self.transform_locus, TransformLocus, "transform_locus")
        if (
            self.transform_locus is not TransformLocus.NONE
            and self.transform_locus
            not in _ALLOWED_TRANSFORM_LOCI[(self.source_stage, self.destination_stage)]
        ):
            raise ValueError("transform locus is invalid for the exact stage pair")
        _require_text(self.route_id, "route_id")
        _require_text(
            self.source_endpoint_instance_id,
            "source_endpoint_instance_id",
        )
        _require_text(
            self.destination_endpoint_instance_id,
            "destination_endpoint_instance_id",
        )
        for field_name in (
            "source_endpoint_capability_fingerprint",
            "destination_endpoint_capability_fingerprint",
            "source_representation_digest",
            "destination_representation_digest",
            "source_placement_digest",
            "destination_placement_digest",
        ):
            _require_sha256_digest(getattr(self, field_name), field_name)
        if (
            self.transform_locus is TransformLocus.NONE
            and self.source_representation_digest
            != self.destination_representation_digest
        ):
            raise ValueError(
                "direct-copy capability requires equal representation digests"
            )
        _require_identifier(self.implementation_id, "implementation_id")
        _require_implementation_version(
            self.implementation_version,
            "implementation_version",
        )
        extraction = self.source_region_extraction
        if extraction is not None:
            _validate_source_region_extraction_capability(extraction)
            if (
                self.source_stage is not PhysicalFormatStage.SOURCE_STORAGE
                or self.destination_stage is not PhysicalFormatStage.WIRE
                or self.transform_locus is not TransformLocus.SOURCE
            ):
                raise ValueError(
                    "source region extraction is valid only for a SOURCE storage-to-wire transform"
                )

    @classmethod
    def from_realized_format(
        cls,
        realized_format: RealizedBindingFormat,
        source_stage: PhysicalFormatStage,
        destination_stage: PhysicalFormatStage,
        *,
        transform_locus: TransformLocus,
        implementation_id: str,
        implementation_version: str,
        source_region_extraction: SourceRegionExtractionCapability | None = None,
    ) -> Self:
        """Describe an operation; a trusted registry must still admit it."""
        if type(realized_format) is not RealizedBindingFormat:
            raise TypeError("realized_format must be RealizedBindingFormat")
        route = realized_format.route_between(source_stage, destination_stage)
        return cls(
            source_stage=source_stage,
            destination_stage=destination_stage,
            transform_locus=transform_locus,
            route_id=route.route_id,
            source_endpoint_instance_id=route.source_endpoint_instance_id,
            destination_endpoint_instance_id=(route.destination_endpoint_instance_id),
            source_endpoint_capability_fingerprint=(
                route.source_endpoint_capability_fingerprint
            ),
            destination_endpoint_capability_fingerprint=(
                route.destination_endpoint_capability_fingerprint
            ),
            source_representation_digest=physical_representation_digest(
                realized_format,
                source_stage,
            ),
            destination_representation_digest=physical_representation_digest(
                realized_format,
                destination_stage,
            ),
            source_placement_digest=endpoint_placement_digest(
                realized_format,
                source_stage,
            ),
            destination_placement_digest=endpoint_placement_digest(
                realized_format,
                destination_stage,
            ),
            implementation_id=implementation_id,
            implementation_version=implementation_version,
            source_region_extraction=source_region_extraction,
        )


def _operation_capability_payload(
    capability: AdapterOperationCapability,
) -> dict[str, object]:
    return {
        "source_stage": capability.source_stage.value,
        "destination_stage": capability.destination_stage.value,
        "transform_locus": capability.transform_locus.value,
        "route_id": capability.route_id,
        "source_endpoint_instance_id": capability.source_endpoint_instance_id,
        "destination_endpoint_instance_id": (
            capability.destination_endpoint_instance_id
        ),
        "source_endpoint_capability_fingerprint": (
            capability.source_endpoint_capability_fingerprint
        ),
        "destination_endpoint_capability_fingerprint": (
            capability.destination_endpoint_capability_fingerprint
        ),
        "source_representation_digest": capability.source_representation_digest,
        "destination_representation_digest": (
            capability.destination_representation_digest
        ),
        "source_placement_digest": capability.source_placement_digest,
        "destination_placement_digest": capability.destination_placement_digest,
        "implementation_id": capability.implementation_id,
        "implementation_version": capability.implementation_version,
        "source_region_extraction": (
            None
            if capability.source_region_extraction is None
            else _source_region_extraction_payload(capability.source_region_extraction)
        ),
    }


def _public_proof_constructor_error() -> TypeError:
    return TypeError(
        "capability proofs must be issued by an adapter capability registry"
    )


@dataclass(frozen=True, slots=True, init=False)
class EndpointCapabilityProof:
    """Opaque endpoint attestation signed by an adapter capability registry."""

    stage: PhysicalFormatStage
    endpoint_instance_id: str
    capability_fingerprint: str
    ordered_representation_digest: str
    placement_digest: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise _public_proof_constructor_error()

    @classmethod
    def _issue(
        cls,
        factory_token: object,
        *,
        stage: PhysicalFormatStage,
        endpoint_instance_id: str,
        capability_fingerprint: str,
        ordered_representation_digest: str,
        placement_digest: str,
    ) -> Self:
        if factory_token is not _ADAPTER_REGISTRY_FACTORY_TOKEN:
            raise _public_proof_constructor_error()
        proof = object.__new__(cls)
        object.__setattr__(proof, "stage", stage)
        object.__setattr__(proof, "endpoint_instance_id", endpoint_instance_id)
        object.__setattr__(proof, "capability_fingerprint", capability_fingerprint)
        object.__setattr__(
            proof,
            "ordered_representation_digest",
            ordered_representation_digest,
        )
        object.__setattr__(proof, "placement_digest", placement_digest)
        return proof


def _validate_endpoint_capability_proof(proof: EndpointCapabilityProof) -> None:
    if type(proof) is not EndpointCapabilityProof:
        raise TypeError("endpoint capability must be EndpointCapabilityProof")
    _require_enum(proof.stage, PhysicalFormatStage, "endpoint capability stage")
    _require_text(proof.endpoint_instance_id, "endpoint_instance_id")
    for field_name in (
        "capability_fingerprint",
        "ordered_representation_digest",
        "placement_digest",
    ):
        _require_sha256_digest(getattr(proof, field_name), field_name)


def _validate_adapter_proof_fields(
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
    route_id: str,
    source_endpoint_capability: EndpointCapabilityProof,
    destination_endpoint_capability: EndpointCapabilityProof,
    implementation_id: str,
    implementation_version: str,
) -> None:
    _require_adjacent_stage_pair(source_stage, destination_stage)
    _require_text(route_id, "route_id")
    _validate_endpoint_capability_proof(source_endpoint_capability)
    _validate_endpoint_capability_proof(destination_endpoint_capability)
    if source_endpoint_capability.stage is not source_stage:
        raise ValueError("source endpoint capability stage mismatch")
    if destination_endpoint_capability.stage is not destination_stage:
        raise ValueError("destination endpoint capability stage mismatch")
    _require_identifier(implementation_id, "implementation_id")
    _require_implementation_version(
        implementation_version,
        "implementation_version",
    )


@dataclass(frozen=True, slots=True, init=False)
class DirectCopyCapabilityProof:
    """Opaque registry authorization for one exact adjacent physical copy."""

    source_stage: PhysicalFormatStage
    destination_stage: PhysicalFormatStage
    route_id: str
    source_endpoint_capability: EndpointCapabilityProof
    destination_endpoint_capability: EndpointCapabilityProof
    implementation_id: str
    implementation_version: str
    adapter_id: str
    adapter_version: str
    issuer_instance_id: str
    registry_digest: str
    signature: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise _public_proof_constructor_error()

    @classmethod
    def _issue(
        cls,
        factory_token: object,
        **fields: object,
    ) -> Self:
        if factory_token is not _ADAPTER_REGISTRY_FACTORY_TOKEN:
            raise _public_proof_constructor_error()
        proof = object.__new__(cls)
        for field_name, value in fields.items():
            object.__setattr__(proof, field_name, value)
        return proof


@dataclass(frozen=True, slots=True, init=False)
class TransformCapabilityProof:
    """Opaque registry authorization for one exact physical transform."""

    source_stage: PhysicalFormatStage
    destination_stage: PhysicalFormatStage
    transform_locus: TransformLocus
    route_id: str
    source_endpoint_capability: EndpointCapabilityProof
    destination_endpoint_capability: EndpointCapabilityProof
    implementation_id: str
    implementation_version: str
    source_region_extraction: SourceRegionExtractionCapability | None
    adapter_id: str
    adapter_version: str
    issuer_instance_id: str
    registry_digest: str
    signature: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise _public_proof_constructor_error()

    @classmethod
    def _issue(
        cls,
        factory_token: object,
        **fields: object,
    ) -> Self:
        if factory_token is not _ADAPTER_REGISTRY_FACTORY_TOKEN:
            raise _public_proof_constructor_error()
        proof = object.__new__(cls)
        for field_name, value in fields.items():
            object.__setattr__(proof, field_name, value)
        return proof


@dataclass(frozen=True, slots=True, init=False)
class DestinationBindingEvidence:
    """Opaque process-local discovery issued by a live destination registry."""

    destination_owner_instance_id: str
    destination_owner_capability_fingerprint: str
    finalizer_instance_id: str
    finalizer_capability_fingerprint: str
    destination_storage_generation: str
    destination_load_api_representation_digest: str
    destination_load_api_placement_digest: str
    destination_runtime_representation_digest: str
    destination_runtime_placement_digest: str
    adapter_id: str
    adapter_version: str
    destination_registry_instance_id: str
    destination_registry_generation: int
    evidence_id: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "destination binding evidence must be produced by live version-adapter discovery"
        )

    @classmethod
    def _issue(cls, factory_token: object, **fields: object) -> Self:
        if factory_token is not _DESTINATION_BINDING_EVIDENCE_FACTORY_TOKEN:
            raise TypeError(
                "destination binding evidence must be produced by live version-adapter discovery"
            )
        evidence = object.__new__(cls)
        for field_name, value in fields.items():
            object.__setattr__(evidence, field_name, value)
        return evidence

    def __reduce__(self) -> object:
        raise TypeError(
            "destination binding evidence is process-local; rediscover it from the live adapter"
        )


def _validate_destination_binding_evidence(
    evidence: DestinationBindingEvidence,
) -> DestinationBindingEvidence:
    if type(evidence) is not DestinationBindingEvidence:
        raise TypeError("destination binding evidence must be exact")
    for field_name in (
        "destination_owner_instance_id",
        "finalizer_instance_id",
        "destination_storage_generation",
    ):
        _require_text(getattr(evidence, field_name), field_name)
    _require_identifier(evidence.adapter_id, "destination evidence adapter_id")
    _require_implementation_version(
        evidence.adapter_version,
        "destination evidence adapter_version",
    )
    _require_text(
        evidence.destination_registry_instance_id,
        "destination_registry_instance_id",
    )
    if (
        type(evidence.destination_registry_generation) is not int
        or evidence.destination_registry_generation <= 0
    ):
        raise ValueError("destination registry generation must be a positive integer")
    for field_name in (
        "destination_owner_capability_fingerprint",
        "finalizer_capability_fingerprint",
        "destination_load_api_representation_digest",
        "destination_load_api_placement_digest",
        "destination_runtime_representation_digest",
        "destination_runtime_placement_digest",
        "evidence_id",
    ):
        _require_sha256_digest(getattr(evidence, field_name), field_name)
    return evidence


@dataclass(frozen=True, slots=True)
class _LiveDestinationBindingRecord:
    evidence: DestinationBindingEvidence
    snapshot: _DestinationEvidenceSnapshot
    snapshot_digest: str
    destination_owner: object
    finalizer: object
    storage_generation: object
    physical_digests: tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class _DestinationEvidenceSnapshot:
    destination_owner_instance_id: str
    destination_owner_capability_fingerprint: str
    finalizer_instance_id: str
    finalizer_capability_fingerprint: str
    destination_storage_generation: str
    destination_load_api_representation_digest: str
    destination_load_api_placement_digest: str
    destination_runtime_representation_digest: str
    destination_runtime_placement_digest: str
    adapter_id: str
    adapter_version: str
    destination_registry_instance_id: str
    destination_registry_generation: int
    evidence_id: str


def _destination_evidence_snapshot(
    evidence: DestinationBindingEvidence,
) -> _DestinationEvidenceSnapshot:
    validated = _validate_destination_binding_evidence(evidence)
    return _DestinationEvidenceSnapshot(
        **{
            field.name: getattr(validated, field.name)
            for field in fields(_DestinationEvidenceSnapshot)
        }
    )


def _destination_evidence_snapshot_digest(
    snapshot: _DestinationEvidenceSnapshot,
) -> str:
    return _canonical_digest(
        {
            "type": "destination_evidence_snapshot.v1",
            **{
                field.name: getattr(snapshot, field.name)
                for field in fields(_DestinationEvidenceSnapshot)
            },
        }
    )


@dataclass(frozen=True, slots=True, init=False)
class DestinationOwnerHandle:
    destination_registry_instance_id: str
    generation: int
    handle_id: str
    _live_identity: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "destination owner handles are issued by adapter mint authority"
        )

    def __reduce__(self) -> object:
        raise TypeError("destination owner handles are process-local")


@dataclass(frozen=True, slots=True, init=False)
class DestinationFinalizerHandle:
    destination_registry_instance_id: str
    generation: int
    handle_id: str
    _live_identity: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "destination finalizer handles are issued by adapter mint authority"
        )

    def __reduce__(self) -> object:
        raise TypeError("destination finalizer handles are process-local")


@dataclass(frozen=True, slots=True, init=False)
class DestinationStorageGeneration:
    """Opaque current-allocation lease issued by a live destination registry."""

    destination_registry_instance_id: str
    generation: int
    generation_id: str
    _storage_identity: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "destination storage generations must be issued by a live adapter registry"
        )

    @classmethod
    def _issue(
        cls,
        factory_token: object,
        *,
        destination_registry_instance_id: str,
        generation: int,
        generation_id: str,
        storage_identity: object,
    ) -> Self:
        if factory_token is not _DESTINATION_REGISTRY_FACTORY_TOKEN:
            raise TypeError("invalid destination storage generation authority")
        lease = object.__new__(cls)
        object.__setattr__(
            lease,
            "destination_registry_instance_id",
            destination_registry_instance_id,
        )
        object.__setattr__(lease, "generation", generation)
        object.__setattr__(lease, "generation_id", generation_id)
        object.__setattr__(lease, "_storage_identity", storage_identity)
        return lease

    def __reduce__(self) -> object:
        raise TypeError("destination storage generation leases are process-local")


@dataclass(frozen=True, slots=True, init=False)
class DestinationBindingMintAuthority:
    """Adapter-private authority; planning receives only the verifier registry."""

    _destination_registry: LiveDestinationBindingRegistry
    _capability_registry: AdapterCapabilityRegistry
    _adapter_instance: object
    _issuance_nonce: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("destination mint authority is issued only to a live adapter")

    @classmethod
    def _issue(
        cls,
        factory_token: object,
        *,
        destination_registry: LiveDestinationBindingRegistry,
        capability_registry: AdapterCapabilityRegistry,
        adapter_instance: object,
    ) -> Self:
        if factory_token is not _DESTINATION_REGISTRY_FACTORY_TOKEN:
            raise TypeError("invalid destination mint authority")
        authority = object.__new__(cls)
        object.__setattr__(authority, "_destination_registry", destination_registry)
        object.__setattr__(authority, "_capability_registry", capability_registry)
        object.__setattr__(authority, "_adapter_instance", adapter_instance)
        object.__setattr__(authority, "_issuance_nonce", object())
        return authority

    def __reduce__(self) -> object:
        raise TypeError("destination mint authorities are process-local")

    def begin_storage_generation(
        self,
        storage_generation: object,
    ) -> DestinationStorageGeneration:
        return self._destination_registry._begin_storage_generation(
            self,
            storage_generation,
        )

    def register_destination_owner(
        self,
        destination_owner: object,
    ) -> DestinationOwnerHandle:
        return self._destination_registry._register_destination_owner(
            self,
            destination_owner,
        )

    def register_finalizer(self, finalizer: object) -> DestinationFinalizerHandle:
        return self._destination_registry._register_finalizer(self, finalizer)

    def discover_destination_binding(
        self,
        realized_format: RealizedBindingFormat,
        *,
        destination_owner: DestinationOwnerHandle,
        finalizer: DestinationFinalizerHandle,
        storage_generation: DestinationStorageGeneration,
    ) -> DestinationBindingEvidence:
        return self._destination_registry._discover_destination_binding(
            self,
            realized_format,
            destination_owner=destination_owner,
            finalizer=finalizer,
            storage_generation=storage_generation,
        )

    def issue_destination_binding(
        self,
        realized_format: RealizedBindingFormat,
        *,
        evidence: DestinationBindingEvidence,
    ) -> DestinationBindingProof:
        return self._capability_registry._issue_destination_binding(
            self,
            realized_format,
            evidence=evidence,
        )


def _destination_physical_digests(
    realized_format: RealizedBindingFormat,
) -> tuple[str, str, str, str]:
    if type(realized_format) is not RealizedBindingFormat:
        raise TypeError("realized_format must be RealizedBindingFormat")
    return (
        physical_representation_digest(
            realized_format,
            PhysicalFormatStage.DESTINATION_LOAD_API,
        ),
        endpoint_placement_digest(
            realized_format,
            PhysicalFormatStage.DESTINATION_LOAD_API,
        ),
        physical_representation_digest(
            realized_format,
            PhysicalFormatStage.DESTINATION_RUNTIME,
        ),
        endpoint_placement_digest(
            realized_format,
            PhysicalFormatStage.DESTINATION_RUNTIME,
        ),
    )


def _require_live_identity(value: object, name: str) -> object:
    if (
        value is None
        or isinstance(value, _SCALAR_SEQUENCE_TYPES)
        or type(value)
        in {
            bool,
            int,
            float,
            complex,
        }
    ):
        raise TypeError(f"{name} must be a live adapter-owned object identity")
    return value


class LiveDestinationBindingRegistry:
    """Versioned process-local evidence registry owned by a live adapter."""

    __slots__ = (
        "_adapter_id",
        "_adapter_version",
        "_generation",
        "_finalizer_handles",
        "_instance_id",
        "_lock",
        "_mint_authority",
        "_owner_handles",
        "_records",
        "_records_by_identity",
        "_storage_generation",
        "_storage_generation_id",
        "_storage_generation_lease",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "LiveDestinationBindingRegistry is created by an adapter capability registry"
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("live destination binding registry state is adapter-owned")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("live destination binding registry state is adapter-owned")

    @classmethod
    def _create(
        cls,
        factory_token: object,
        *,
        adapter_id: str,
        adapter_version: str,
    ) -> Self:
        if factory_token is not _DESTINATION_REGISTRY_FACTORY_TOKEN:
            raise TypeError("invalid live destination registry factory authority")
        registry = object.__new__(cls)
        object.__setattr__(registry, "_adapter_id", adapter_id)
        object.__setattr__(registry, "_adapter_version", adapter_version)
        object.__setattr__(
            registry,
            "_instance_id",
            f"destination-registry-{secrets.token_hex(32)}",
        )
        object.__setattr__(registry, "_generation", 0)
        object.__setattr__(registry, "_mint_authority", None)
        object.__setattr__(registry, "_owner_handles", {})
        object.__setattr__(registry, "_finalizer_handles", {})
        object.__setattr__(registry, "_storage_generation", None)
        object.__setattr__(registry, "_storage_generation_id", None)
        object.__setattr__(registry, "_storage_generation_lease", None)
        object.__setattr__(registry, "_records", {})
        object.__setattr__(registry, "_records_by_identity", {})
        object.__setattr__(registry, "_lock", RLock())
        return registry

    def __reduce__(self) -> object:
        raise TypeError("live destination registries are process-local")

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def _issue_mint_authority(
        self,
        capability_registry: AdapterCapabilityRegistry,
        adapter_instance: object,
    ) -> DestinationBindingMintAuthority:
        live_adapter = _require_live_identity(adapter_instance, "adapter_instance")
        with self._lock:
            if self._mint_authority is not None:
                raise ValueError("destination mint authority was already issued")
            authority = DestinationBindingMintAuthority._issue(
                _DESTINATION_REGISTRY_FACTORY_TOKEN,
                destination_registry=self,
                capability_registry=capability_registry,
                adapter_instance=live_adapter,
            )
            object.__setattr__(self, "_mint_authority", authority)
            return authority

    def _require_mint_authority(
        self,
        authority: DestinationBindingMintAuthority,
    ) -> None:
        if (
            type(authority) is not DestinationBindingMintAuthority
            or authority is not self._mint_authority
            or authority._destination_registry is not self
        ):
            raise ValueError("destination operation requires adapter mint authority")

    def _begin_storage_generation(
        self,
        authority: DestinationBindingMintAuthority,
        storage_generation: object,
    ) -> DestinationStorageGeneration:
        """Advance once to a live allocation and return its non-replayable lease."""
        storage_object = _require_live_identity(
            storage_generation,
            "storage_generation",
        )
        with self._lock:
            self._require_mint_authority(authority)
            current = self._storage_generation_lease
            if (
                type(current) is DestinationStorageGeneration
                and current._storage_identity is storage_object
            ):
                return current
            object.__setattr__(self, "_storage_generation", storage_object)
            object.__setattr__(self, "_generation", self._generation + 1)
            generation_id = (
                f"storage-generation-{self._generation}-{secrets.token_hex(16)}"
            )
            object.__setattr__(self, "_storage_generation_id", generation_id)
            lease = DestinationStorageGeneration._issue(
                _DESTINATION_REGISTRY_FACTORY_TOKEN,
                destination_registry_instance_id=self._instance_id,
                generation=self._generation,
                generation_id=generation_id,
                storage_identity=storage_object,
            )
            object.__setattr__(self, "_storage_generation_lease", lease)
            self._records.clear()
            self._records_by_identity.clear()
            self._owner_handles.clear()
            self._finalizer_handles.clear()
            return lease

    def _register_destination_owner(
        self,
        authority: DestinationBindingMintAuthority,
        destination_owner: object,
    ) -> DestinationOwnerHandle:
        owner = _require_live_identity(destination_owner, "destination_owner")
        with self._lock:
            self._require_mint_authority(authority)
            if self._storage_generation_lease is None:
                raise ValueError("destination storage generation must begin first")
            handle = object.__new__(DestinationOwnerHandle)
            object.__setattr__(
                handle,
                "destination_registry_instance_id",
                self._instance_id,
            )
            object.__setattr__(handle, "generation", self._generation)
            object.__setattr__(handle, "handle_id", f"owner-{secrets.token_hex(32)}")
            object.__setattr__(handle, "_live_identity", owner)
            self._owner_handles[id(handle)] = handle
            return handle

    def _register_finalizer(
        self,
        authority: DestinationBindingMintAuthority,
        finalizer: object,
    ) -> DestinationFinalizerHandle:
        finalizer_object = _require_live_identity(finalizer, "finalizer")
        with self._lock:
            self._require_mint_authority(authority)
            if self._storage_generation_lease is None:
                raise ValueError("destination storage generation must begin first")
            handle = object.__new__(DestinationFinalizerHandle)
            object.__setattr__(
                handle,
                "destination_registry_instance_id",
                self._instance_id,
            )
            object.__setattr__(handle, "generation", self._generation)
            object.__setattr__(
                handle,
                "handle_id",
                f"finalizer-{secrets.token_hex(32)}",
            )
            object.__setattr__(handle, "_live_identity", finalizer_object)
            self._finalizer_handles[id(handle)] = handle
            return handle

    def _discover_destination_binding(
        self,
        authority: DestinationBindingMintAuthority,
        realized_format: RealizedBindingFormat,
        *,
        destination_owner: DestinationOwnerHandle,
        finalizer: DestinationFinalizerHandle,
        storage_generation: DestinationStorageGeneration,
    ) -> DestinationBindingEvidence:
        """Capture exact live owner/finalizer identities for one storage generation."""
        physical_digests = _destination_physical_digests(realized_format)
        with self._lock:
            self._require_mint_authority(authority)
            if (
                type(destination_owner) is not DestinationOwnerHandle
                or self._owner_handles.get(id(destination_owner))
                is not destination_owner
                or destination_owner.destination_registry_instance_id
                != self._instance_id
                or destination_owner.generation != self._generation
            ):
                raise TypeError(
                    "destination_owner must be an exact registered destination owner handle"
                )
            if (
                type(finalizer) is not DestinationFinalizerHandle
                or self._finalizer_handles.get(id(finalizer)) is not finalizer
                or finalizer.destination_registry_instance_id != self._instance_id
                or finalizer.generation != self._generation
            ):
                raise TypeError(
                    "finalizer must be an exact registered destination finalizer handle"
                )
            if (
                type(storage_generation) is not DestinationStorageGeneration
                or storage_generation is not self._storage_generation_lease
                or storage_generation.destination_registry_instance_id
                != self._instance_id
                or storage_generation.generation != self._generation
            ):
                raise ValueError("destination storage generation lease is stale")
            storage_object = storage_generation._storage_identity
            owner = destination_owner._live_identity
            finalizer_object = finalizer._live_identity
            identity_key = (
                destination_owner.handle_id,
                finalizer.handle_id,
                physical_digests,
            )
            existing_digest = self._records_by_identity.get(identity_key)
            if existing_digest is not None:
                existing = self._records[existing_digest]
                if (
                    existing.destination_owner is owner
                    and existing.finalizer is finalizer_object
                    and existing.storage_generation is storage_object
                ):
                    return existing.evidence
            evidence_id = f"sha256:{secrets.token_hex(32)}"
            generation_id = self._storage_generation_id
            if type(generation_id) is not str:
                raise RuntimeError("live destination storage generation is unavailable")
            evidence = DestinationBindingEvidence._issue(
                _DESTINATION_BINDING_EVIDENCE_FACTORY_TOKEN,
                destination_owner_instance_id=(
                    f"destination-owner-{secrets.token_hex(32)}"
                ),
                destination_owner_capability_fingerprint=(
                    f"sha256:{secrets.token_hex(32)}"
                ),
                finalizer_instance_id=f"destination-finalizer-{secrets.token_hex(32)}",
                finalizer_capability_fingerprint=f"sha256:{secrets.token_hex(32)}",
                destination_storage_generation=generation_id,
                destination_load_api_representation_digest=physical_digests[0],
                destination_load_api_placement_digest=physical_digests[1],
                destination_runtime_representation_digest=physical_digests[2],
                destination_runtime_placement_digest=physical_digests[3],
                adapter_id=self._adapter_id,
                adapter_version=self._adapter_version,
                destination_registry_instance_id=self._instance_id,
                destination_registry_generation=self._generation,
                evidence_id=evidence_id,
            )
            snapshot = _destination_evidence_snapshot(evidence)
            record = _LiveDestinationBindingRecord(
                evidence=evidence,
                snapshot=snapshot,
                snapshot_digest=_destination_evidence_snapshot_digest(snapshot),
                destination_owner=owner,
                finalizer=finalizer_object,
                storage_generation=storage_object,
                physical_digests=physical_digests,
            )
            self._records[evidence_id] = record
            self._records_by_identity[identity_key] = evidence_id
            return evidence

    def _require_evidence(
        self,
        evidence: DestinationBindingEvidence,
        realized_format: RealizedBindingFormat,
    ) -> _LiveDestinationBindingRecord:
        validated = _validate_destination_binding_evidence(evidence)
        with self._lock:
            record = self._records.get(validated.evidence_id)
            if record is None or record.evidence is not validated:
                raise ValueError(
                    "destination binding evidence is not live in this adapter registry"
                )
            current_snapshot = _destination_evidence_snapshot(validated)
            if (
                current_snapshot != record.snapshot
                or _destination_evidence_snapshot_digest(current_snapshot)
                != record.snapshot_digest
            ):
                raise ValueError(
                    "destination binding evidence differs from its immutable discovery snapshot"
                )
            if (
                validated.destination_registry_instance_id != self._instance_id
                or validated.destination_registry_generation != self._generation
                or record.storage_generation is not self._storage_generation
            ):
                raise ValueError(
                    "destination binding evidence storage generation is stale"
                )
            if record.physical_digests != _destination_physical_digests(
                realized_format
            ):
                raise ValueError(
                    "destination evidence differs from the realized destination"
                )
            return record

    def _require_proof_live(self, proof: DestinationBindingProof) -> int:
        with self._lock:
            if (
                proof.destination_registry_instance_id != self._instance_id
                or proof.destination_registry_generation != self._generation
            ):
                raise ValueError(
                    "destination binding proof storage generation is stale"
                )
            record = self._records.get(proof.destination_evidence_id)
            if (
                record is None
                or record.storage_generation is not self._storage_generation
            ):
                raise ValueError(
                    "destination binding proof lacks live adapter evidence"
                )
            evidence = record.snapshot
            if (
                _destination_evidence_snapshot_digest(evidence)
                != record.snapshot_digest
            ):
                raise ValueError("destination immutable discovery snapshot was mutated")
            if _destination_evidence_snapshot(record.evidence) != evidence:
                raise ValueError(
                    "destination binding evidence differs from its immutable discovery snapshot"
                )
            if (
                proof.destination_owner_instance_id
                != evidence.destination_owner_instance_id
                or proof.destination_owner_capability_fingerprint
                != evidence.destination_owner_capability_fingerprint
                or proof.finalizer_instance_id != evidence.finalizer_instance_id
                or proof.finalizer_capability_fingerprint
                != evidence.finalizer_capability_fingerprint
                or proof.destination_storage_generation
                != evidence.destination_storage_generation
                or proof.destination_load_api_representation_digest
                != evidence.destination_load_api_representation_digest
                or proof.destination_load_api_placement_digest
                != evidence.destination_load_api_placement_digest
                or proof.destination_runtime_representation_digest
                != evidence.destination_runtime_representation_digest
                or proof.destination_runtime_placement_digest
                != evidence.destination_runtime_placement_digest
                or proof.adapter_id != evidence.adapter_id
                or proof.adapter_version != evidence.adapter_version
                or proof.destination_evidence_id != evidence.evidence_id
            ):
                raise ValueError("destination binding proof differs from live evidence")
            return self._generation


@dataclass(frozen=True, slots=True, init=False)
class DestinationBindingProof:
    """Version-adapter attestation for the concrete destination owner pair."""

    destination_owner_instance_id: str
    destination_owner_capability_fingerprint: str
    finalizer_instance_id: str
    finalizer_capability_fingerprint: str
    destination_storage_generation: str
    destination_load_api_representation_digest: str
    destination_load_api_placement_digest: str
    destination_runtime_representation_digest: str
    destination_runtime_placement_digest: str
    adapter_id: str
    adapter_version: str
    issuer_instance_id: str
    registry_digest: str
    destination_registry_instance_id: str
    destination_registry_generation: int
    destination_evidence_id: str
    destination_binding_digest: str
    signature: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "destination binding proofs must be issued by an adapter capability "
            "registry"
        )

    @classmethod
    def _issue(cls, factory_token: object, **fields: object) -> Self:
        if factory_token is not _ADAPTER_REGISTRY_FACTORY_TOKEN:
            raise TypeError(
                "destination binding proofs must be issued by an adapter capability "
                "registry"
            )
        proof = object.__new__(cls)
        for field_name, value in fields.items():
            object.__setattr__(proof, field_name, value)
        return proof


def _destination_binding_payload(
    proof: DestinationBindingProof,
) -> dict[str, object]:
    return {
        "type": "destination_binding_proof.v2",
        "adapter_id": proof.adapter_id,
        "adapter_version": proof.adapter_version,
        "destination_registry_instance_id": proof.destination_registry_instance_id,
        "destination_registry_generation": proof.destination_registry_generation,
        "destination_evidence_id": proof.destination_evidence_id,
        "destination_owner_instance_id": proof.destination_owner_instance_id,
        "destination_owner_capability_fingerprint": (
            proof.destination_owner_capability_fingerprint
        ),
        "finalizer_instance_id": proof.finalizer_instance_id,
        "finalizer_capability_fingerprint": proof.finalizer_capability_fingerprint,
        "destination_storage_generation": proof.destination_storage_generation,
        "destination_load_api_representation_digest": (
            proof.destination_load_api_representation_digest
        ),
        "destination_load_api_placement_digest": (
            proof.destination_load_api_placement_digest
        ),
        "destination_runtime_representation_digest": (
            proof.destination_runtime_representation_digest
        ),
        "destination_runtime_placement_digest": (
            proof.destination_runtime_placement_digest
        ),
    }


@dataclass(frozen=True, slots=True, init=False)
class RefitOperationBaseProof:
    """Adapter authentication for one reusable context/format/destination base."""

    binding_context_digest: str
    binding_identity_digest: str
    source_storage_lowering_digest: str
    physical_binding_digest: str
    destination_binding_digest: str
    destination_storage_generation: str
    destination_registry_instance_id: str
    destination_registry_generation: int
    adapter_id: str
    adapter_version: str
    issuer_instance_id: str
    registry_digest: str
    base_digest: str
    signature: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "refit operation base proofs must be issued by an adapter capability registry"
        )

    @classmethod
    def _issue(cls, factory_token: object, **fields: object) -> Self:
        if factory_token is not _ADAPTER_REGISTRY_FACTORY_TOKEN:
            raise TypeError(
                "refit operation base proofs must be issued by an adapter capability registry"
            )
        proof = object.__new__(cls)
        for field_name, value in fields.items():
            object.__setattr__(proof, field_name, value)
        return proof


def _refit_operation_base_payload(
    proof: RefitOperationBaseProof,
) -> dict[str, object]:
    return {
        "type": "refit_operation_base.v1",
        "binding_context_digest": proof.binding_context_digest,
        "binding_identity_digest": proof.binding_identity_digest,
        "source_storage_lowering_digest": proof.source_storage_lowering_digest,
        "physical_binding_digest": proof.physical_binding_digest,
        "destination_binding_digest": proof.destination_binding_digest,
        "destination_storage_generation": proof.destination_storage_generation,
        "destination_registry_instance_id": proof.destination_registry_instance_id,
        "destination_registry_generation": proof.destination_registry_generation,
        "adapter_id": proof.adapter_id,
        "adapter_version": proof.adapter_version,
        "issuer_instance_id": proof.issuer_instance_id,
        "registry_digest": proof.registry_digest,
    }


def _source_region_payload(region: SourceRegion) -> dict[str, object]:
    return {
        "source_shape": list(region.source_shape),
        "axis_selections": [
            {
                "axis_index": selection.axis_index,
                "spans": [
                    {
                        "start": span.start,
                        "stop": span.stop,
                        "step": span.step,
                    }
                    for span in selection.spans
                ],
            }
            for selection in region.axis_selections
        ],
    }


def _source_region_is_complete(region: SourceRegion) -> bool:
    return all(
        len(selection.spans) == 1
        and selection.spans[0].start == 0
        and selection.spans[0].stop == region.source_shape[selection.axis_index]
        and selection.spans[0].step == 1
        for selection in region.axis_selections
    )


def _component_key_payload(
    source_slice: RuntimeSourceBindingSlice,
) -> dict[str, object]:
    key = source_slice.component_key
    domain = key.member_domain
    return {
        "graph_instance_id": key.graph_instance_id,
        "semantic_graph_path": key.semantic_graph_path,
        "inventory_entry_id": key.inventory_entry_id,
        "member_domain": {
            "layers": (
                None
                if domain.layer_domain is None
                else [
                    {
                        "global_decoder_layer": member.global_decoder_layer,
                        "moe_ordinal": member.moe_ordinal,
                    }
                    for member in domain.layer_domain.members
                ]
            ),
            "independent_axes": [
                {"name": axis.name, "members": list(axis.members)}
                for axis in domain.independent_axes
            ],
        },
        "component_role": key.component_role,
        "canonical_owner_family": {
            "graph_instance_id": key.canonical_owner_family.graph_instance_id,
            "owner_family_id": key.canonical_owner_family.owner_family_id,
        },
    }


def _source_binding_slice_digest(
    source_slice: RuntimeSourceBindingSlice,
    *,
    runtime_source_digest: str,
    component_key_digest: str,
) -> str:
    return _canonical_digest(
        {
            "type": "runtime_source_binding_slice.v1",
            "runtime_source_digest": runtime_source_digest,
            "component_key_digest": component_key_digest,
            "source_region": _source_region_payload(source_slice.source_region),
            "source_record_id": source_slice.source_binding.source_record.record_id,
            "training_precision": source_slice.training_assignment.precision,
            "training_format": _format_descriptor_payload(
                source_slice.training_assignment.requested_format
            ),
            "rollout_precision": source_slice.rollout_assignment.precision,
            "rollout_format": _format_descriptor_payload(
                source_slice.rollout_assignment.requested_format
            ),
        }
    )


def _binding_context_payload(context: RefitBindingContext) -> dict[str, object]:
    return {
        "atomic_source_bindings": [
            {
                "component_role": source_slice.component_key.component_role,
                "tensor_instance_id": (
                    source_slice.source_binding.source_record.source_native_owner_id
                ),
                "source_region": _source_region_payload(source_slice.source_region),
                "component_key_digest": component_key_digest,
                "source_binding_slice_digest": source_slice_digest,
                "source_record_id": source_slice.source_binding.source_record.record_id,
                "source_realization_id": realization.realization_id,
                "source_realization_digest": realization_digest,
            }
            for source_slice, realization, component_key_digest, source_slice_digest, realization_digest in zip(
                context.source_binding_slices,
                context.source_realizations,
                context.semantic_component_key_digests,
                context.source_binding_slice_digests,
                context.source_realization_digests,
                strict=True,
            )
        ],
        "graph_instance_id": context.graph_instance_id,
        "tensor_instance_id": context.tensor_instance_id,
        "semantic_graph_path": context.semantic_graph_path,
        "inventory_entry_id": context.inventory_entry_id,
        "semantic_component_key_digest": context.semantic_component_key_digest,
        "component_role": context.component_role,
        "canonical_owner_graph_instance_id": (
            context.canonical_owner_graph_instance_id
        ),
        "canonical_owner_family_id": context.canonical_owner_family_id,
        "selection_group_id": context.selection_group_id,
        "semantic_selection_digest": context.semantic_selection_digest,
        "intent_group_id": context.intent_group_id,
        "graph_intent_id": context.graph_intent_id,
        "runtime_source_result_digest": context.runtime_source_result_digest,
        "runtime_source_digest": context.runtime_source_digest,
        "source_binding_slice_digest": context.source_binding_slice_digest,
        "source_record_id": context.source_record_id,
        "source_region": _source_region_payload(context.source_region),
        "source_realization_digest": context.source_realization_digest,
        "source_normalizer_manifest_digest": (
            context.source_normalizer_manifest_digest
        ),
        "source_normalization_capability_id": (
            context.source_normalization_capability_id
        ),
        "source_normalization_kind": context.source_normalization_kind,
        "source_normalization_contract_digest": (
            context.source_normalization_contract_digest
        ),
        "source_component_ids": list(context.source_component_ids),
        "source_component_dtypes": list(context.source_component_dtypes),
        "source_component_shapes": [
            list(shape) for shape in context.source_component_shapes
        ],
        "source_component_encodings": list(context.source_component_encodings),
        "source_logical_axes": list(context.source_logical_axes),
        "source_logical_shape": list(context.source_logical_shape),
        "source_output_dtype": context.source_output_dtype,
        "source_output_shape": list(context.source_output_shape),
        "source_output_encoding": context.source_output_encoding,
        "source_format": _format_descriptor_payload(context.source_format),
        "destination_format": _format_descriptor_payload(context.destination_format),
    }


@dataclass(frozen=True, slots=True, init=False)
class RefitBindingContext:
    """Validated compiler/discovery binding used to compile physical operations."""

    graph_instance_id: str
    tensor_instance_id: str
    semantic_graph_path: str
    inventory_entry_id: str
    source_binding_slices: tuple[RuntimeSourceBindingSlice, ...]
    source_realizations: tuple[SourceStorageRealization, ...]
    semantic_component_key_digests: tuple[str, ...]
    source_binding_slice_digests: tuple[str, ...]
    source_realization_digests: tuple[str, ...]
    source_binding_slice: RuntimeSourceBindingSlice
    selection_entry: SelectionTopologyEntry
    source_region: SourceRegion
    component_role: ComponentRole
    canonical_owner_graph_instance_id: str
    canonical_owner_family_id: str
    selection_group_id: str
    semantic_selection_digest: str
    intent_group_id: str
    graph_intent_id: str
    runtime_source_result_digest: str
    runtime_source_digest: str
    semantic_component_key_digest: str
    source_binding_slice_digest: str
    source_record_id: str
    source_realization: SourceStorageRealization
    source_normalizer_manifest: SourceNormalizerManifest
    source_realization_digest: str
    source_normalizer_manifest_digest: str
    source_normalization_capability_id: str
    source_normalization_kind: str
    source_normalization_contract_digest: str
    source_component_ids: tuple[str, ...]
    source_component_dtypes: tuple[str, ...]
    source_component_shapes: tuple[tuple[int, ...], ...]
    source_component_encodings: tuple[str, ...]
    source_logical_axes: tuple[str, ...]
    source_logical_shape: tuple[int, ...]
    source_output_dtype: str
    source_output_shape: tuple[int, ...]
    source_output_encoding: str
    source_format: FormatDescriptor
    destination_format: FormatDescriptor
    binding_context_digest: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "RefitBindingContext must be derived from compiler and discovery artifacts"
        )

    @classmethod
    def _issue(
        cls,
        factory_token: object,
        **fields: object,
    ) -> Self:
        if factory_token is not _REFIT_PLANNER_FACTORY_TOKEN:
            raise TypeError("invalid refit planner authority")
        context = object.__new__(cls)
        for field_name, value in fields.items():
            object.__setattr__(context, field_name, value)
        object.__setattr__(
            context,
            "binding_context_digest",
            _canonical_digest(
                {
                    "type": "refit_binding_context.v1",
                    **_binding_context_payload(cast(RefitBindingContext, context)),
                }
            ),
        )
        return context


@dataclass(frozen=True, slots=True)
class _InstalledContextRecord:
    handle: InstalledRefitBindingContext
    context: RefitBindingContext
    export_context: RefitBindingContext
    payload_digest: str
    generation: int
    issuance_nonce: object


@dataclass(frozen=True, slots=True)
class _InstalledOperationBaseRecord:
    proof: RefitOperationBaseProof
    installed_context: InstalledRefitBindingContext
    destination_binding_proof: DestinationBindingProof
    generation: int
    issuance_nonce: object
    destination_registry: LiveDestinationBindingRegistry
    destination_generation: int


@dataclass(frozen=True, slots=True)
class _InstalledOperationRecord:
    handle: InstalledSelectedRefitOperation
    payload_digest: str
    executor_key: RefitExecutorKey
    installed_base: _InstalledOperationBaseRecord
    generation: int
    issuance_nonce: object
    destination_registry: LiveDestinationBindingRegistry
    destination_generation: int


class RefitPlanInstallationRegistry:
    """Process-local install session with exact-handle and generation authority."""

    __slots__ = (
        "_context_by_digest",
        "_contexts",
        "_generation",
        "_instance_id",
        "_lock",
        "_operation_bases",
        "_operation_by_digest",
        "_operations",
        "_runtime_generation_digest",
    )

    def __init__(self) -> None:
        object.__setattr__(self, "_lock", RLock())
        object.__setattr__(
            self, "_instance_id", f"refit-install-{secrets.token_hex(32)}"
        )
        object.__setattr__(self, "_generation", 0)
        object.__setattr__(self, "_runtime_generation_digest", None)
        object.__setattr__(self, "_contexts", {})
        object.__setattr__(self, "_context_by_digest", {})
        object.__setattr__(self, "_operation_bases", {})
        object.__setattr__(self, "_operations", {})
        object.__setattr__(self, "_operation_by_digest", {})

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("refit installation registry state is runtime-owned")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("refit installation registry state is runtime-owned")

    def __reduce__(self) -> object:
        raise TypeError("refit installation registries are process-local")

    def _activate(self, runtime_generation_digest: str) -> int:
        _require_sha256_digest(
            runtime_generation_digest,
            "runtime generation digest",
        )
        with self._lock:
            if self._runtime_generation_digest != runtime_generation_digest:
                object.__setattr__(
                    self,
                    "_runtime_generation_digest",
                    runtime_generation_digest,
                )
                object.__setattr__(self, "_generation", self._generation + 1)
                self._contexts.clear()
                self._context_by_digest.clear()
                self._operation_bases.clear()
                self._operations.clear()
                self._operation_by_digest.clear()
            return self._generation

    def _issue_context(
        self,
        factory_token: object,
        context: RefitBindingContext,
        *,
        runtime_generation_digest: str,
    ) -> InstalledRefitBindingContext:
        if factory_token is not _REFIT_PLANNER_FACTORY_TOKEN:
            raise TypeError("invalid refit installation authority")
        if type(context) is not RefitBindingContext:
            raise TypeError("installed context must be an exact RefitBindingContext")
        structural = deepcopy(context)
        payload_digest = _canonical_digest(
            {
                "type": "refit_binding_context.v1",
                **_binding_context_payload(structural),
            }
        )
        with self._lock:
            if self._runtime_generation_digest != runtime_generation_digest:
                raise ValueError("refit installation runtime generation changed")
            existing_id = self._context_by_digest.get(payload_digest)
            if existing_id is not None:
                existing = self._contexts[existing_id]
                _require_exact_structural_match(
                    structural,
                    existing.context,
                    "installed binding context",
                )
                return existing.handle
            handle = InstalledRefitBindingContext._issue(
                _REFIT_PLANNER_FACTORY_TOKEN,
                registry=self,
            )
            nonce = handle._issuance_nonce
            record = _InstalledContextRecord(
                handle=handle,
                context=structural,
                export_context=deepcopy(structural),
                payload_digest=payload_digest,
                generation=self._generation,
                issuance_nonce=nonce,
            )
            handle_id = id(handle)
            self._contexts[handle_id] = record
            self._context_by_digest[payload_digest] = handle_id
            return handle

    def _issue_operation_base(
        self,
        factory_token: object,
        proof: RefitOperationBaseProof,
        *,
        installed_context: InstalledRefitBindingContext,
        destination_binding_proof: DestinationBindingProof,
        destination_registry: LiveDestinationBindingRegistry,
    ) -> _InstalledOperationBaseRecord:
        if factory_token is not _REFIT_PLANNER_FACTORY_TOKEN:
            raise TypeError("invalid refit operation base installation authority")
        with self._lock:
            context = self._resolve_context(installed_context)
            if context.binding_context_digest != proof.binding_context_digest:
                raise ValueError("operation base binding context digest mismatch")
            if (
                destination_binding_proof.destination_binding_digest
                != proof.destination_binding_digest
            ):
                raise ValueError("operation base destination binding digest mismatch")
            destination_generation = destination_registry._require_proof_live(
                destination_binding_proof
            )
            existing = self._operation_bases.get(proof.base_digest)
            if existing is not None:
                if (
                    existing.proof != proof
                    or existing.installed_context is not installed_context
                    or existing.destination_binding_proof != destination_binding_proof
                    or existing.generation != self._generation
                    or existing.destination_registry is not destination_registry
                    or existing.destination_generation != destination_generation
                ):
                    raise ValueError("installed refit operation base mismatch")
                return existing
            record = _InstalledOperationBaseRecord(
                proof=deepcopy(proof),
                installed_context=installed_context,
                destination_binding_proof=deepcopy(destination_binding_proof),
                generation=self._generation,
                issuance_nonce=object(),
                destination_registry=destination_registry,
                destination_generation=destination_generation,
            )
            self._operation_bases[proof.base_digest] = record
            return record

    def _resolve_operation_base(
        self,
        proof: RefitOperationBaseProof,
        *,
        installed_context: InstalledRefitBindingContext,
        destination_registry: LiveDestinationBindingRegistry,
    ) -> _InstalledOperationBaseRecord | None:
        with self._lock:
            record = self._operation_bases.get(proof.base_digest)
            if record is None:
                return None
            live_destination_generation = destination_registry._require_proof_live(
                record.destination_binding_proof
            )
            if (
                record.proof != proof
                or record.installed_context is not installed_context
                or record.generation != self._generation
                or record.destination_registry is not destination_registry
                or record.destination_generation != live_destination_generation
            ):
                raise ValueError(
                    "installed refit operation base is stale or mismatched"
                )
            return record

    def _resolve_context(
        self,
        handle: InstalledRefitBindingContext,
    ) -> RefitBindingContext:
        with self._lock:
            record = self._contexts.get(id(handle))
            if (
                record is None
                or record.handle is not handle
                or handle._installation_registry is not self
                or handle._issuance_nonce is not record.issuance_nonce
            ):
                raise ValueError("binding context is not an issued installation handle")
            if record.generation != self._generation:
                raise ValueError("binding context installation generation is stale")
            return record.context

    def _export_context(
        self,
        handle: InstalledRefitBindingContext,
    ) -> RefitBindingContext:
        with self._lock:
            self._resolve_context(handle)
            return self._contexts[id(handle)].export_context

    def _issue_operation(
        self,
        factory_token: object,
        reference: _AuthenticatedSelectedOperationReference,
        *,
        installed_context: InstalledRefitBindingContext,
        installed_base: _InstalledOperationBaseRecord,
        destination_registry: LiveDestinationBindingRegistry,
    ) -> InstalledSelectedRefitOperation:
        if factory_token is not _REFIT_PLANNER_FACTORY_TOKEN:
            raise TypeError("invalid refit operation installation authority")
        payload_digest = _require_sha256_digest(
            reference.selected_operation_digest,
            "selected operation digest",
        )
        with self._lock:
            if self._runtime_generation_digest is None:
                raise ValueError("refit installation registry has no active runtime")
            active_context = self._resolve_context(installed_context)
            if (
                reference.operation_base_proof.binding_context_digest
                != active_context.binding_context_digest
            ):
                raise ValueError("selected operation binding context digest mismatch")
            destination_generation = destination_registry._require_proof_live(
                installed_base.destination_binding_proof
            )
            if (
                installed_base
                is not self._operation_bases.get(
                    reference.operation_base_proof.base_digest
                )
                or installed_base.installed_context is not installed_context
                or installed_base.generation != self._generation
                or installed_base.destination_registry is not destination_registry
                or installed_base.destination_generation != destination_generation
            ):
                raise ValueError("selected operation lacks an installed refit base")
            existing_id = self._operation_by_digest.get(payload_digest)
            if existing_id is not None:
                existing = self._operations[existing_id]
                if (
                    existing.destination_registry is not destination_registry
                    or existing.destination_generation != destination_generation
                ):
                    raise ValueError(
                        "installed operation belongs to another destination generation"
                    )
                return existing.handle
            handle = InstalledSelectedRefitOperation._issue(
                _REFIT_PLANNER_FACTORY_TOKEN,
                registry=self,
            )
            nonce = handle._issuance_nonce
            record = _InstalledOperationRecord(
                handle=handle,
                payload_digest=payload_digest,
                executor_key=deepcopy(reference.executor_key),
                installed_base=installed_base,
                generation=self._generation,
                issuance_nonce=nonce,
                destination_registry=destination_registry,
                destination_generation=destination_generation,
            )
            handle_id = id(handle)
            self._operations[handle_id] = record
            self._operation_by_digest[payload_digest] = handle_id
            return handle

    def _resolve_operation(
        self,
        handle: InstalledSelectedRefitOperation,
    ) -> _InstalledOperationRecord:
        with self._lock:
            record = self._operations.get(id(handle))
            if (
                record is None
                or record.handle is not handle
                or handle._installation_registry is not self
                or handle._issuance_nonce is not record.issuance_nonce
            ):
                raise ValueError("operation is not an issued installation handle")
            if record.generation != self._generation:
                raise ValueError("operation installation generation is stale")
            live_destination_generation = (
                record.destination_registry._require_proof_live(
                    record.installed_base.destination_binding_proof
                )
            )
            if record.destination_generation != live_destination_generation:
                raise ValueError("operation destination storage generation is stale")
            return record


@dataclass(frozen=True, slots=True, init=False)
class InstalledRefitBindingContext:
    """Process-local handle installed against the active runtime artifacts."""

    _installation_registry: RefitPlanInstallationRegistry
    _issuance_nonce: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "InstalledRefitBindingContext must be produced by active-artifact installation"
        )

    @classmethod
    def _issue(
        cls,
        factory_token: object,
        *,
        registry: RefitPlanInstallationRegistry,
    ) -> Self:
        if factory_token is not _REFIT_PLANNER_FACTORY_TOKEN:
            raise TypeError("invalid refit installation authority")
        installed = object.__new__(cls)
        object.__setattr__(installed, "_installation_registry", registry)
        object.__setattr__(installed, "_issuance_nonce", object())
        return installed

    def __reduce__(self) -> object:
        raise TypeError(
            "installed refit contexts are process-local; serialize binding_context and reinstall"
        )

    def _trusted_context(self) -> RefitBindingContext:
        return self._installation_registry._resolve_context(self)

    @property
    def binding_context(self) -> RefitBindingContext:
        return self._installation_registry._export_context(self)

    @property
    def graph_instance_id(self) -> str:
        return self._trusted_context().graph_instance_id

    @property
    def tensor_instance_id(self) -> str:
        return self._trusted_context().tensor_instance_id

    @property
    def inventory_entry_id(self) -> str:
        return self._trusted_context().inventory_entry_id

    @property
    def selection_group_id(self) -> str:
        return self._trusted_context().selection_group_id

    @property
    def intent_group_id(self) -> str:
        return self._trusted_context().intent_group_id

    @property
    def runtime_source_result_digest(self) -> str:
        return self._trusted_context().runtime_source_result_digest

    @property
    def source_format(self) -> FormatDescriptor:
        return deepcopy(self._trusted_context().source_format)

    @property
    def destination_format(self) -> FormatDescriptor:
        return deepcopy(self._trusted_context().destination_format)

    @property
    def source_record_id(self) -> str:
        return self._trusted_context().source_record_id

    @property
    def source_binding_slice_digest(self) -> str:
        return self._trusted_context().source_binding_slice_digest

    @property
    def source_region(self) -> SourceRegion:
        return deepcopy(self._trusted_context().source_region)

    @property
    def source_realization(self) -> SourceStorageRealization:
        return deepcopy(self._trusted_context().source_realization)

    @property
    def source_realization_digest(self) -> str:
        return self._trusted_context().source_realization_digest


def _require_installed_binding_context(
    context: object,
) -> RefitBindingContext:
    if type(context) is not InstalledRefitBindingContext:
        raise TypeError("binding_context must be an InstalledRefitBindingContext")
    installed = cast(InstalledRefitBindingContext, context)
    if type(installed._installation_registry) is not RefitPlanInstallationRegistry:
        raise ValueError("binding context installation registry mismatch")
    return installed._installation_registry._resolve_context(installed)


@dataclass(frozen=True, slots=True)
class _RefitContextIndex:
    slices: dict[str, RuntimeSourceBindingSlice]
    realizations: dict[tuple[str, str], SourceStorageRealization]
    graph_identity: dict[str, tuple[str, str]]
    selection_entries: dict[tuple[str, str], SelectionTopologyEntry]
    normalizer_manifests: dict[str, SourceNormalizerManifest]
    result_digests: dict[str, str]
    component_key_digests: dict[str, str]
    selection_group_id: str
    intent_group_id: str
    runtime_source_digest: str
    runtime_generation_digest: str


def _build_refit_context_index(
    intents: CompiledPrecisionIntentGroup,
) -> _RefitContextIndex:
    selection = intents.selection
    selection_group_id = intents.selection_group_id
    runtime_source_digest = intents.runtime_source_digest
    runtime_source_receipt = intents.runtime_source_receipt
    if (
        selection is None
        or selection_group_id is None
        or runtime_source_digest is None
        or runtime_source_receipt is None
    ):
        raise ValueError("runtime-bound intent identity is incomplete")
    slices: dict[str, RuntimeSourceBindingSlice] = {}
    realizations: dict[tuple[str, str], SourceStorageRealization] = {}
    graph_identity: dict[str, tuple[str, str]] = {}
    component_key_digests: dict[str, str] = {}
    for graph_intent in intents.graph_intents:
        if graph_intent.selection is None:
            raise ValueError("source slice graph has no validated Phase 1 selection")
        graph_id = graph_intent.graph_instance_id
        if graph_id in graph_identity:
            raise ValueError("source slice graph must occur exactly once in intents")
        graph_identity[graph_id] = (
            graph_intent.selection.selection_id,
            graph_intent.intent_id,
        )
        for source_slice in graph_intent.source_binding_slices:
            component_key_digest = _canonical_digest(
                {
                    "type": "semantic_component_key.v1",
                    **_component_key_payload(source_slice),
                }
            )
            source_slice_digest = _source_binding_slice_digest(
                source_slice,
                runtime_source_digest=runtime_source_digest,
                component_key_digest=component_key_digest,
            )
            if source_slice_digest in slices:
                raise ValueError(
                    "source binding slice must occur exactly once in runtime-bound "
                    "intents"
                )
            slices[source_slice_digest] = source_slice
            component_key_digests[source_slice_digest] = component_key_digest
            for realization in source_slice.source_binding.source_realizations:
                if type(realization) is not SourceStorageRealization:
                    continue
                realization_key = (source_slice_digest, realization.realization_id)
                if realization_key in realizations:
                    raise ValueError(
                        "source realization must occur exactly once in its source slice"
                    )
                realizations[realization_key] = realization
    selection_entries: dict[tuple[str, str], SelectionTopologyEntry] = {}
    for graph in selection.topology.graphs:
        graph_id = graph.declaration.graph_instance_id
        for entry in graph.entries:
            entry_key = (graph_id, entry.entry_id)
            if entry_key in selection_entries:
                raise ValueError(
                    "source slice entry must occur exactly once in topology"
                )
            selection_entries[entry_key] = entry
    normalizer_manifests: dict[str, SourceNormalizerManifest] = {}
    for graph_bindings in intents.source_bindings.graph_bindings:
        if graph_bindings.graph_instance_id in normalizer_manifests:
            raise ValueError("source normalizer manifest is ambiguous for the graph")
        normalizer_manifests[graph_bindings.graph_instance_id] = (
            graph_bindings.normalizer_manifest
        )
    result_digests: dict[str, str] = {}
    for graph_id, result_digest in runtime_source_receipt.result_digests:
        if graph_id in result_digests:
            raise ValueError(
                "runtime source result identity is ambiguous for the graph"
            )
        result_digests[graph_id] = result_digest
    return _RefitContextIndex(
        slices=slices,
        realizations=realizations,
        graph_identity=graph_identity,
        selection_entries=selection_entries,
        normalizer_manifests=normalizer_manifests,
        result_digests=result_digests,
        component_key_digests=component_key_digests,
        selection_group_id=selection_group_id,
        intent_group_id=intents.intent_group_id,
        runtime_source_digest=runtime_source_digest,
        runtime_generation_digest=_canonical_digest(
            {
                "type": "refit_runtime_generation.v1",
                "selection_group_id": selection_group_id,
                "intent_group_id": intents.intent_group_id,
                "runtime_source_digest": runtime_source_digest,
            }
        ),
    )


def _bind_refit_context_from_index(
    index: _RefitContextIndex,
    source_binding_slice: RuntimeSourceBindingSlice,
    source_realization: SourceStorageRealization,
) -> RefitBindingContext:
    if type(source_binding_slice) is not RuntimeSourceBindingSlice:
        raise TypeError(
            "source_binding_slice must be an exact RuntimeSourceBindingSlice"
        )
    if type(source_realization) is not SourceStorageRealization:
        raise TypeError("source_realization must be an exact SourceStorageRealization")
    component_key_digest = _canonical_digest(
        {
            "type": "semantic_component_key.v1",
            **_component_key_payload(source_binding_slice),
        }
    )
    source_binding_slice_digest = _source_binding_slice_digest(
        source_binding_slice,
        runtime_source_digest=index.runtime_source_digest,
        component_key_digest=component_key_digest,
    )
    source_slice = index.slices.get(source_binding_slice_digest)
    if source_slice is None:
        raise ValueError(
            "source binding slice must occur exactly once in runtime-bound intents"
        )
    try:
        _require_exact_structural_match(
            source_binding_slice,
            source_slice,
            "source binding slice",
        )
    except ValueError as error:
        raise ValueError(
            "source binding slice must occur exactly once in runtime-bound intents"
        ) from error
    graph_instance_id = source_slice.component_key.graph_instance_id
    graph_identity = index.graph_identity.get(graph_instance_id)
    if graph_identity is None:
        raise ValueError("source slice graph is absent from bound intents")
    selection_entry = index.selection_entries.get(
        (graph_instance_id, source_slice.component_key.inventory_entry_id)
    )
    if selection_entry is None:
        raise ValueError("source slice entry must occur exactly once in topology")
    selected_realization = index.realizations.get(
        (source_binding_slice_digest, source_realization.realization_id)
    )
    if selected_realization is None:
        raise ValueError(
            "source realization must occur exactly once in the bound source slice"
        )
    try:
        _require_exact_structural_match(
            source_realization,
            selected_realization,
            "source realization",
        )
    except ValueError as error:
        raise ValueError(
            "source realization must occur exactly once in the bound source slice"
        ) from error
    source_record = source_slice.source_binding.source_record
    if (
        source_record.graph_instance_id != graph_instance_id
        or selected_realization.graph_instance_id != graph_instance_id
        or selected_realization.output_record_id != source_record.record_id
        or source_slice.source_region.source_shape != source_record.shape
    ):
        raise ValueError("source realization differs from the classified source slice")
    if source_record.source_native_owner_id is None:
        raise ValueError("refit binding requires a present source-native owner")
    source_format = source_slice.training_assignment.requested_format
    destination_format = source_slice.rollout_assignment.requested_format
    _require_canonical_reserved_format(source_format)
    _require_canonical_reserved_format(destination_format)
    matching_component_descriptors = tuple(
        component
        for component in source_format.components
        if component.role == source_slice.component_key.component_role
    )
    if len(matching_component_descriptors) != 1:
        raise ValueError(
            "source format must contain the classified component role exactly once"
        )
    source_component_descriptor = matching_component_descriptors[0]
    if (
        source_component_descriptor.dtype != source_record.dtype.value
        or source_component_descriptor.encoding != source_record.numeric_encoding
        or selected_realization.output_dtype != source_record.dtype
        or selected_realization.output_shape != source_record.shape
        or selected_realization.output_numeric_encoding
        != source_record.numeric_encoding
    ):
        raise ValueError(
            "source record and realization do not lower the requested component format"
        )
    normalizer_manifest = index.normalizer_manifests.get(graph_instance_id)
    if normalizer_manifest is None:
        raise ValueError("source normalizer manifest is ambiguous for the graph")
    if selected_realization.normalization not in normalizer_manifest.contracts:
        raise ValueError("source realization normalizer is absent from its manifest")
    realization_inventory = SourceStorageRealizationInventory(
        graph_instance_id=graph_instance_id,
        normalizer_manifest=normalizer_manifest,
        realizations=(selected_realization,),
    )
    result_digest = index.result_digests.get(graph_instance_id)
    if result_digest is None:
        raise ValueError("runtime source result identity is ambiguous for the graph")
    if index.component_key_digests[source_binding_slice_digest] != component_key_digest:
        raise ValueError("source component key differs from its indexed slice")
    source_components = selected_realization.components
    realization_digest = source_storage_inventory_digest(realization_inventory)
    return RefitBindingContext._issue(
        _REFIT_PLANNER_FACTORY_TOKEN,
        graph_instance_id=graph_instance_id,
        tensor_instance_id=source_record.source_native_owner_id,
        semantic_graph_path=source_slice.component_key.semantic_graph_path,
        inventory_entry_id=source_slice.component_key.inventory_entry_id,
        source_binding_slices=(source_slice,),
        source_realizations=(selected_realization,),
        semantic_component_key_digests=(component_key_digest,),
        source_binding_slice_digests=(source_binding_slice_digest,),
        source_realization_digests=(realization_digest,),
        source_binding_slice=source_slice,
        selection_entry=selection_entry,
        source_region=source_slice.source_region,
        component_role=source_slice.component_key.component_role,
        canonical_owner_graph_instance_id=(
            source_slice.component_key.canonical_owner_family.graph_instance_id
        ),
        canonical_owner_family_id=(
            source_slice.component_key.canonical_owner_family.owner_family_id
        ),
        selection_group_id=index.selection_group_id,
        semantic_selection_digest=graph_identity[0],
        intent_group_id=index.intent_group_id,
        graph_intent_id=graph_identity[1],
        runtime_source_result_digest=result_digest,
        runtime_source_digest=index.runtime_source_digest,
        semantic_component_key_digest=component_key_digest,
        source_binding_slice_digest=source_binding_slice_digest,
        source_record_id=source_record.record_id,
        source_realization=selected_realization,
        source_normalizer_manifest=normalizer_manifest,
        source_realization_digest=realization_digest,
        source_normalizer_manifest_digest=source_normalizer_manifest_digest(
            normalizer_manifest
        ),
        source_normalization_capability_id=(
            selected_realization.normalization.capability_id
        ),
        source_normalization_kind=selected_realization.normalization.kind.value,
        source_normalization_contract_digest=(
            selected_realization.normalization.contract_digest
        ),
        source_component_ids=tuple(
            component.native_component_id for component in source_components
        ),
        source_component_dtypes=tuple(
            component.carrier_dtype.value for component in source_components
        ),
        source_component_shapes=tuple(
            component.physical_shape for component in source_components
        ),
        source_component_encodings=tuple(
            component.storage_encoding for component in source_components
        ),
        source_logical_axes=selection_entry.logical_axes,
        source_logical_shape=selection_entry.logical_shape,
        source_output_dtype=selected_realization.output_dtype.value,
        source_output_shape=selected_realization.output_shape,
        source_output_encoding=selected_realization.output_numeric_encoding,
        source_format=source_format,
        destination_format=destination_format,
    )


def _atomic_binding_group_key(source_slice: RuntimeSourceBindingSlice) -> str:
    component_payload = _component_key_payload(source_slice)
    component_payload.pop("component_role")
    component_payload.pop("canonical_owner_family")
    return _canonical_digest(
        {
            "type": "atomic_refit_binding_group.v1",
            "semantic_component": component_payload,
            "training_precision": source_slice.training_assignment.precision,
            "training_format": _format_descriptor_payload(
                source_slice.training_assignment.requested_format
            ),
            "rollout_precision": source_slice.rollout_assignment.precision,
            "rollout_format": _format_descriptor_payload(
                source_slice.rollout_assignment.requested_format
            ),
        }
    )


def _combine_refit_binding_contexts(
    contexts: Sequence[RefitBindingContext],
) -> RefitBindingContext:
    captured = _snapshot_sequence(contexts, "atomic binding contexts")
    if not captured:
        raise ValueError("atomic binding contexts must be non-empty")
    primary = captured[0]
    common_fields = (
        "graph_instance_id",
        "semantic_graph_path",
        "inventory_entry_id",
        "selection_entry",
        "selection_group_id",
        "semantic_selection_digest",
        "intent_group_id",
        "graph_intent_id",
        "runtime_source_result_digest",
        "runtime_source_digest",
        "source_normalizer_manifest",
        "source_normalizer_manifest_digest",
        "source_logical_axes",
        "source_logical_shape",
        "source_format",
        "destination_format",
    )
    if any(
        getattr(context, field_name) != getattr(primary, field_name)
        for context in captured[1:]
        for field_name in common_fields
    ):
        raise ValueError("atomic source bindings disagree on their transfer identity")
    expected_roles = tuple(
        component.role for component in primary.source_format.components
    )
    contexts_by_role = {context.component_role: context for context in captured}
    if len(contexts_by_role) != len(captured) or set(contexts_by_role) != set(
        expected_roles
    ):
        raise ValueError(
            "atomic source bindings must realize every source format role exactly once"
        )
    ordered = tuple(contexts_by_role[role] for role in expected_roles)
    if len(ordered) == 1:
        return ordered[0]
    primary = ordered[0]
    source_slices = tuple(context.source_binding_slice for context in ordered)
    source_realizations = tuple(context.source_realization for context in ordered)
    component_key_digests = tuple(
        context.semantic_component_key_digest for context in ordered
    )
    slice_digests = tuple(context.source_binding_slice_digest for context in ordered)
    realization_digests = tuple(
        context.source_realization_digest for context in ordered
    )
    atomic_component_key_digest = _canonical_digest(
        {
            "type": "atomic_semantic_component_keys.v1",
            "component_key_digests": list(component_key_digests),
        }
    )
    atomic_slice_digest = _canonical_digest(
        {
            "type": "atomic_source_binding_slices.v1",
            "source_binding_slice_digests": list(slice_digests),
        }
    )
    atomic_realization_digest = _canonical_digest(
        {
            "type": "atomic_source_realizations.v1",
            "source_realization_digests": list(realization_digests),
        }
    )
    all_source_components = tuple(
        component
        for context in ordered
        for component in context.source_realization.components
    )
    return RefitBindingContext._issue(
        _REFIT_PLANNER_FACTORY_TOKEN,
        graph_instance_id=primary.graph_instance_id,
        tensor_instance_id=primary.tensor_instance_id,
        semantic_graph_path=primary.semantic_graph_path,
        inventory_entry_id=primary.inventory_entry_id,
        source_binding_slices=source_slices,
        source_realizations=source_realizations,
        semantic_component_key_digests=component_key_digests,
        source_binding_slice_digests=slice_digests,
        source_realization_digests=realization_digests,
        source_binding_slice=primary.source_binding_slice,
        selection_entry=primary.selection_entry,
        source_region=primary.source_region,
        component_role=primary.component_role,
        canonical_owner_graph_instance_id=(primary.canonical_owner_graph_instance_id),
        canonical_owner_family_id=primary.canonical_owner_family_id,
        selection_group_id=primary.selection_group_id,
        semantic_selection_digest=primary.semantic_selection_digest,
        intent_group_id=primary.intent_group_id,
        graph_intent_id=primary.graph_intent_id,
        runtime_source_result_digest=primary.runtime_source_result_digest,
        runtime_source_digest=primary.runtime_source_digest,
        semantic_component_key_digest=atomic_component_key_digest,
        source_binding_slice_digest=atomic_slice_digest,
        source_record_id=primary.source_record_id,
        source_realization=primary.source_realization,
        source_normalizer_manifest=primary.source_normalizer_manifest,
        source_realization_digest=atomic_realization_digest,
        source_normalizer_manifest_digest=(primary.source_normalizer_manifest_digest),
        source_normalization_capability_id=(primary.source_normalization_capability_id),
        source_normalization_kind=primary.source_normalization_kind,
        source_normalization_contract_digest=(
            primary.source_normalization_contract_digest
        ),
        source_component_ids=tuple(
            component.native_component_id for component in all_source_components
        ),
        source_component_dtypes=tuple(
            component.carrier_dtype.value for component in all_source_components
        ),
        source_component_shapes=tuple(
            component.physical_shape for component in all_source_components
        ),
        source_component_encodings=tuple(
            component.storage_encoding for component in all_source_components
        ),
        source_logical_axes=primary.source_logical_axes,
        source_logical_shape=primary.source_logical_shape,
        source_output_dtype=primary.source_output_dtype,
        source_output_shape=primary.source_output_shape,
        source_output_encoding=primary.source_output_encoding,
        source_format=primary.source_format,
        destination_format=primary.destination_format,
    )


def bind_refit_contexts(
    *,
    intents: CompiledPrecisionIntentGroup,
    active_selection: CompiledPrecisionSelectionGroup,
    active_request: RuntimeSourceDiscoveryRequest,
    active_results: tuple[RuntimeSourceDiscoveryResult, ...],
    installation_registry: RefitPlanInstallationRegistry,
    bindings: Sequence[tuple[RuntimeSourceBindingSlice, SourceStorageRealization]],
) -> tuple[InstalledRefitBindingContext, ...]:
    """Bind many slices after one authoritative validation and index build."""
    validated_intents = validate_compiled_precision_intent_group(
        intents,
        active_selection=active_selection,
        active_request=active_request,
        active_results=active_results,
    )
    captured_bindings = _snapshot_sequence(bindings, "bindings")
    if not captured_bindings:
        raise ValueError("bindings must be non-empty")
    if any(
        type(binding) is not tuple or len(binding) != 2 for binding in captured_bindings
    ):
        raise TypeError("bindings must contain exact (slice, realization) tuples")
    if type(installation_registry) is not RefitPlanInstallationRegistry:
        raise TypeError("installation_registry must be RefitPlanInstallationRegistry")
    index = _build_refit_context_index(validated_intents)
    installation_registry._activate(index.runtime_generation_digest)
    single_contexts = tuple(
        _bind_refit_context_from_index(index, source_slice, source_realization)
        for source_slice, source_realization in captured_bindings
    )
    grouped: dict[str, list[RefitBindingContext]] = {}
    for context in single_contexts:
        grouped.setdefault(
            _atomic_binding_group_key(context.source_binding_slice),
            [],
        ).append(context)
    return tuple(
        installation_registry._issue_context(
            _REFIT_PLANNER_FACTORY_TOKEN,
            _combine_refit_binding_contexts(group_contexts),
            runtime_generation_digest=index.runtime_generation_digest,
        )
        for group_contexts in grouped.values()
    )


def bind_refit_context(
    *,
    intents: CompiledPrecisionIntentGroup,
    active_selection: CompiledPrecisionSelectionGroup,
    active_request: RuntimeSourceDiscoveryRequest,
    active_results: tuple[RuntimeSourceDiscoveryResult, ...],
    installation_registry: RefitPlanInstallationRegistry,
    source_binding_slice: RuntimeSourceBindingSlice,
    source_realization: SourceStorageRealization,
) -> InstalledRefitBindingContext:
    """Bind one exact Phase 2 component slice to its source realization."""
    return bind_refit_contexts(
        intents=intents,
        active_selection=active_selection,
        active_request=active_request,
        active_results=active_results,
        installation_registry=installation_registry,
        bindings=((source_binding_slice, source_realization),),
    )[0]


def install_refit_contexts(
    *,
    contexts: Sequence[RefitBindingContext],
    intents: CompiledPrecisionIntentGroup,
    active_selection: CompiledPrecisionSelectionGroup,
    active_request: RuntimeSourceDiscoveryRequest,
    active_results: tuple[RuntimeSourceDiscoveryResult, ...],
    installation_registry: RefitPlanInstallationRegistry,
) -> tuple[InstalledRefitBindingContext, ...]:
    """Reinstall serialized binding envelopes against one active artifact set."""
    captured_contexts = _snapshot_sequence(contexts, "contexts")
    if not captured_contexts:
        raise ValueError("contexts must be non-empty")
    if any(type(context) is not RefitBindingContext for context in captured_contexts):
        raise TypeError("contexts must contain exact RefitBindingContext envelopes")
    validated_intents = validate_compiled_precision_intent_group(
        intents,
        active_selection=active_selection,
        active_request=active_request,
        active_results=active_results,
    )
    if type(installation_registry) is not RefitPlanInstallationRegistry:
        raise TypeError("installation_registry must be RefitPlanInstallationRegistry")
    index = _build_refit_context_index(validated_intents)
    installation_registry._activate(index.runtime_generation_digest)
    installed: list[InstalledRefitBindingContext] = []
    installed_by_object_id: dict[
        int, tuple[RefitBindingContext, InstalledRefitBindingContext]
    ] = {}
    installed_by_digest: dict[
        str, tuple[RefitBindingContext, InstalledRefitBindingContext]
    ] = {}
    for context in captured_contexts:
        same_object = installed_by_object_id.get(id(context))
        if same_object is not None and same_object[0] is context:
            installed.append(same_object[1])
            continue
        _require_sha256_digest(
            context.binding_context_digest,
            "binding context digest",
        )
        duplicate = installed_by_digest.get(context.binding_context_digest)
        if duplicate is not None:
            _require_exact_structural_match(
                context,
                duplicate[0],
                "duplicate binding context",
            )
            installed.append(duplicate[1])
            installed_by_object_id[id(context)] = (context, duplicate[1])
            continue
        structural = _validate_binding_context(context)
        expected = _combine_refit_binding_contexts(
            tuple(
                _bind_refit_context_from_index(index, source_slice, realization)
                for source_slice, realization in zip(
                    structural.source_binding_slices,
                    structural.source_realizations,
                    strict=True,
                )
            )
        )
        try:
            _require_exact_structural_match(
                structural,
                expected,
                "binding context",
            )
        except ValueError as error:
            raise ValueError(
                "binding context differs from active runtime compiler artifacts"
            ) from error
        installed_context = installation_registry._issue_context(
            _REFIT_PLANNER_FACTORY_TOKEN,
            expected,
            runtime_generation_digest=index.runtime_generation_digest,
        )
        installed_by_digest[context.binding_context_digest] = (
            structural,
            installed_context,
        )
        installed_by_object_id[id(context)] = (context, installed_context)
        installed.append(installed_context)
    return tuple(installed)


def install_refit_context(
    *,
    context: RefitBindingContext,
    intents: CompiledPrecisionIntentGroup,
    active_selection: CompiledPrecisionSelectionGroup,
    active_request: RuntimeSourceDiscoveryRequest,
    active_results: tuple[RuntimeSourceDiscoveryResult, ...],
    installation_registry: RefitPlanInstallationRegistry,
) -> InstalledRefitBindingContext:
    """Install one serialized binding envelope against active artifacts."""
    return install_refit_contexts(
        contexts=(context,),
        intents=intents,
        active_selection=active_selection,
        active_request=active_request,
        active_results=active_results,
        installation_registry=installation_registry,
    )[0]


def _binding_identity_payload(identity: RefitBindingIdentity) -> dict[str, object]:
    return {
        "graph_instance_id": identity.graph_instance_id,
        "tensor_instance_id": identity.tensor_instance_id,
        "semantic_graph_path": identity.semantic_graph_path,
        "inventory_entry_id": identity.inventory_entry_id,
        "selection_group_id": identity.selection_group_id,
        "semantic_selection_digest": identity.semantic_selection_digest,
        "intent_group_id": identity.intent_group_id,
        "graph_intent_id": identity.graph_intent_id,
        "runtime_source_result_digest": identity.runtime_source_result_digest,
        "source_realization_digest": identity.source_realization_digest,
        "source_storage_lowering_digest": identity.source_storage_lowering_digest,
        "physical_binding_digest": identity.physical_binding_digest,
        "destination_binding_digest": identity.destination_binding_digest,
        "destination_storage_generation": identity.destination_storage_generation,
        "destination_owner_instance_id": identity.destination_owner_instance_id,
        "finalizer_instance_id": identity.finalizer_instance_id,
    }


def _validate_binding_context(context: object) -> RefitBindingContext:
    if type(context) is not RefitBindingContext:
        raise TypeError("binding_context must be a validated RefitBindingContext")
    typed_context = cast(RefitBindingContext, context)
    source_slices = typed_context.source_binding_slices
    source_realizations = typed_context.source_realizations
    component_key_digests = typed_context.semantic_component_key_digests
    slice_digests = typed_context.source_binding_slice_digests
    realization_digests = typed_context.source_realization_digests
    if any(
        type(items) is not tuple
        for items in (
            source_slices,
            source_realizations,
            component_key_digests,
            slice_digests,
            realization_digests,
        )
    ):
        raise TypeError("atomic binding context collections must be exact tuples")
    atomic_count = len(source_slices)
    if atomic_count == 0 or any(
        len(items) != atomic_count
        for items in (
            source_realizations,
            component_key_digests,
            slice_digests,
            realization_digests,
        )
    ):
        raise ValueError("atomic binding context collections are incomplete")
    if any(type(item) is not RuntimeSourceBindingSlice for item in source_slices):
        raise TypeError("atomic source slices must be exact")
    if any(type(item) is not SourceStorageRealization for item in source_realizations):
        raise TypeError("atomic source realizations must be exact")
    if (
        typed_context.source_binding_slice != source_slices[0]
        or typed_context.source_realization != source_realizations[0]
        or typed_context.source_region != source_slices[0].source_region
        or typed_context.component_role != source_slices[0].component_key.component_role
    ):
        raise ValueError("binding context primary aliases differ from atomic bindings")
    expected_roles = tuple(
        component.role for component in typed_context.source_format.components
    )
    actual_roles = tuple(
        source_slice.component_key.component_role for source_slice in source_slices
    )
    if actual_roles != expected_roles:
        raise ValueError("binding context atomic roles differ from source format")
    if atomic_count > 1:
        for field_name in (
            "selection_group_id",
            "semantic_selection_digest",
            "intent_group_id",
            "graph_intent_id",
            "runtime_source_result_digest",
            "runtime_source_digest",
            "semantic_component_key_digest",
            "source_binding_slice_digest",
            "source_realization_digest",
            "source_normalizer_manifest_digest",
            "binding_context_digest",
        ):
            _require_sha256_digest(getattr(typed_context, field_name), field_name)
        expected_digest = _canonical_digest(
            {
                "type": "refit_binding_context.v1",
                **_binding_context_payload(typed_context),
            }
        )
        if typed_context.binding_context_digest != expected_digest:
            raise ValueError("binding context differs from its canonical derivation")
        return typed_context
    if type(typed_context.source_binding_slice) is not RuntimeSourceBindingSlice:
        raise TypeError(
            "source_binding_slice must be an exact RuntimeSourceBindingSlice"
        )
    if type(typed_context.selection_entry) is not SelectionTopologyEntry:
        raise TypeError("selection_entry must be an exact SelectionTopologyEntry")
    if type(typed_context.source_region) is not SourceRegion:
        raise TypeError("source_region must be an exact SourceRegion")
    if type(typed_context.source_realization) is not SourceStorageRealization:
        raise TypeError("source_realization must be SourceStorageRealization")
    if type(typed_context.source_normalizer_manifest) is not SourceNormalizerManifest:
        raise TypeError("source_normalizer_manifest must be SourceNormalizerManifest")
    source_slice = typed_context.source_binding_slice
    canonical_slice = RuntimeSourceBindingSlice(
        component_key=source_slice.component_key,
        source_binding=source_slice.source_binding,
        source_region=source_slice.source_region,
        training_assignment=source_slice.training_assignment,
        rollout_assignment=source_slice.rollout_assignment,
    )
    if canonical_slice != source_slice:
        raise ValueError("binding context source slice is not canonical")
    selection_entry = typed_context.selection_entry
    canonical_entry = SelectionTopologyEntry(
        entry_id=selection_entry.entry_id,
        graph_instance_id=selection_entry.graph_instance_id,
        pattern=selection_entry.pattern,
        domain=selection_entry.domain,
        logical_dtype=selection_entry.logical_dtype,
        logical_shape=selection_entry.logical_shape,
        logical_axes=selection_entry.logical_axes,
    )
    if canonical_entry != selection_entry:
        raise ValueError("binding context selection entry is not canonical")
    key = source_slice.component_key
    source_record = source_slice.source_binding.source_record
    if type(source_record) is not SourceDiscoveryRecord:
        raise TypeError("source record must be an exact SourceDiscoveryRecord")
    canonical_record = SourceDiscoveryRecord(
        record_id=source_record.record_id,
        graph_instance_id=source_record.graph_instance_id,
        source_native_name=source_record.source_native_name,
        source_native_owner_id=source_record.source_native_owner_id,
        dtype=source_record.dtype,
        shape=source_record.shape,
        numeric_encoding=source_record.numeric_encoding,
        provenance=source_record.provenance,
        provenance_evidence=source_record.provenance_evidence,
        source_mutability=source_record.source_mutability,
        mutability_evidence=source_record.mutability_evidence,
    )
    if canonical_record != source_record:
        raise ValueError("binding context source record is not canonical")
    source_realization = typed_context.source_realization
    realization_inventory = SourceStorageRealizationInventory(
        graph_instance_id=key.graph_instance_id,
        normalizer_manifest=typed_context.source_normalizer_manifest,
        realizations=(source_realization,),
    )
    validate_source_storage_realization_inventory(realization_inventory)
    if (
        sum(
            candidate == source_realization
            for candidate in source_slice.source_binding.source_realizations
        )
        != 1
    ):
        raise ValueError("binding context realization differs from its source slice")
    expected_region = project_source_region_to_member_domain(
        source_slice.source_binding.classification_edge,
        selection_entry.domain,
        key.member_domain,
    )
    if (
        selection_entry.entry_id != key.inventory_entry_id
        or selection_entry.graph_instance_id != key.graph_instance_id
        or selection_entry.pattern.semantic_graph_path != key.semantic_graph_path
        or source_record.graph_instance_id != key.graph_instance_id
        or source_record.source_native_owner_id is None
        or source_realization.graph_instance_id != key.graph_instance_id
        or source_realization.output_record_id != source_record.record_id
        or source_realization.output_dtype != source_record.dtype
        or source_realization.output_shape != source_record.shape
        or source_realization.output_numeric_encoding != source_record.numeric_encoding
        or source_slice.source_region != expected_region
    ):
        raise ValueError("binding context embedded source facts disagree")
    source_format = source_slice.training_assignment.requested_format
    destination_format = source_slice.rollout_assignment.requested_format
    matching_components = tuple(
        component
        for component in source_format.components
        if component.role == key.component_role
    )
    if (
        len(matching_components) != 1
        or matching_components[0].dtype != source_record.dtype.value
        or matching_components[0].encoding != source_record.numeric_encoding
    ):
        raise ValueError("binding context source format disagrees with its record")
    source_components = source_realization.components
    expected_component_key_digest = _canonical_digest(
        {
            "type": "semantic_component_key.v1",
            **_component_key_payload(source_slice),
        }
    )
    expected_realization_digest = source_storage_inventory_digest(realization_inventory)
    expected_manifest_digest = source_normalizer_manifest_digest(
        typed_context.source_normalizer_manifest
    )
    expected_fields = {
        "graph_instance_id": key.graph_instance_id,
        "tensor_instance_id": source_record.source_native_owner_id,
        "semantic_graph_path": key.semantic_graph_path,
        "inventory_entry_id": key.inventory_entry_id,
        "source_region": expected_region,
        "component_role": key.component_role,
        "canonical_owner_graph_instance_id": (
            key.canonical_owner_family.graph_instance_id
        ),
        "canonical_owner_family_id": key.canonical_owner_family.owner_family_id,
        "semantic_component_key_digest": expected_component_key_digest,
        "source_record_id": source_record.record_id,
        "source_realization_digest": expected_realization_digest,
        "source_normalizer_manifest_digest": expected_manifest_digest,
        "source_normalization_capability_id": (
            source_realization.normalization.capability_id
        ),
        "source_normalization_kind": source_realization.normalization.kind.value,
        "source_normalization_contract_digest": (
            source_realization.normalization.contract_digest
        ),
        "source_component_ids": tuple(
            component.native_component_id for component in source_components
        ),
        "source_component_dtypes": tuple(
            component.carrier_dtype.value for component in source_components
        ),
        "source_component_shapes": tuple(
            component.physical_shape for component in source_components
        ),
        "source_component_encodings": tuple(
            component.storage_encoding for component in source_components
        ),
        "source_logical_axes": selection_entry.logical_axes,
        "source_logical_shape": selection_entry.logical_shape,
        "source_output_dtype": source_realization.output_dtype.value,
        "source_output_shape": source_realization.output_shape,
        "source_output_encoding": source_realization.output_numeric_encoding,
        "source_format": source_format,
        "destination_format": destination_format,
    }
    if any(
        getattr(typed_context, field_name) != expected_value
        for field_name, expected_value in expected_fields.items()
    ):
        raise ValueError("binding context cached fields differ from embedded authority")
    for field_name in (
        "selection_group_id",
        "semantic_selection_digest",
        "intent_group_id",
        "graph_intent_id",
        "runtime_source_result_digest",
        "runtime_source_digest",
        "semantic_component_key_digest",
        "source_binding_slice_digest",
        "source_realization_digest",
        "source_normalizer_manifest_digest",
        "source_normalization_contract_digest",
        "binding_context_digest",
    ):
        _require_sha256_digest(getattr(typed_context, field_name), field_name)
    if typed_context.source_binding_slice_digest != _source_binding_slice_digest(
        source_slice,
        runtime_source_digest=typed_context.runtime_source_digest,
        component_key_digest=expected_component_key_digest,
    ):
        raise ValueError("source binding slice digest differs from its source slice")
    for format_descriptor in (
        source_format,
        destination_format,
    ):
        if type(format_descriptor) is not FormatDescriptor:
            raise TypeError("binding context formats must be exact FormatDescriptor")
        _require_canonical_reserved_format(format_descriptor)
    expected_digest = _canonical_digest(
        {
            "type": "refit_binding_context.v1",
            **_binding_context_payload(typed_context),
        }
    )
    if typed_context.binding_context_digest != expected_digest:
        raise ValueError("binding context differs from its canonical derivation")
    return typed_context


def _source_target_axis_name(target: object) -> str | None:
    if isinstance(target, ComponentAxisTarget):
        return target.component_axis
    if isinstance(target, FamilyIndexAxisTarget):
        return target.axis_name
    if isinstance(target, LayerCoordinateTarget):
        return target.coordinate
    return None


def _source_axis_mapping_id(
    logical_axis: str,
    physical_axes: tuple[SourcePhysicalAxisSpec, ...],
) -> str:
    if all(
        isinstance(axis.extent, SourceNormalizedAxisExtent)
        and len(axis.extent.normalized_axis_indices) == 1
        and axis.extent.divisor == 1
        and axis.extent.rounding.value == "exact"
        and axis.extent.alignment == 1
        for axis in physical_axes
    ):
        return "identity.axis-map.v1"
    digest = _canonical_digest(
        {
            "type": "source_axis_mapping.v1",
            "logical_axis": logical_axis,
            "physical_axes": [
                {
                    "axis_name": axis.axis_name,
                    "extent": (
                        {
                            "kind": "normalized",
                            "normalized_axis_indices": list(
                                axis.extent.normalized_axis_indices
                            ),
                            "divisor": axis.extent.divisor,
                            "rounding": axis.extent.rounding.value,
                            "alignment": axis.extent.alignment,
                        }
                        if isinstance(axis.extent, SourceNormalizedAxisExtent)
                        else {"kind": "literal", "extent": axis.extent.extent}
                    ),
                }
                for axis in physical_axes
            ],
        }
    )
    return f"source.axis-map-{digest.removeprefix('sha256:')}.v1"


def _versioned_source_encoding(source_encoding: str) -> str:
    candidate = source_encoding.replace("_", "-")
    if _VERSIONED_SEMANTIC_ID_PATTERN.fullmatch(candidate) is not None:
        return candidate
    versioned = f"{candidate}.v1"
    if _VERSIONED_SEMANTIC_ID_PATTERN.fullmatch(versioned) is not None:
        return versioned
    digest = sha256(source_encoding.encode("utf-8")).hexdigest()
    return f"source.encoding-{digest}.v1"


def _source_padding(
    context: RefitBindingContext,
    source_component: SourceStorageComponent,
    *,
    normalized_shape: tuple[int, ...] | None = None,
    source_binding_slice: RuntimeSourceBindingSlice | None = None,
    source_realization: SourceStorageRealization | None = None,
    component_role: ComponentRole | None = None,
) -> tuple[PhysicalPadding, ...]:
    resolved_realization = source_realization or context.source_realization
    resolved_slice = source_binding_slice or context.source_binding_slice
    resolved_role = component_role or context.component_role
    output_shape = (
        resolved_realization.output_shape
        if normalized_shape is None
        else normalized_shape
    )
    edge = resolved_slice.source_binding.classification_edge
    padding_items: list[tuple[str, str, int]] = []
    for physical_axis in source_component.physical_axes:
        extent = physical_axis.extent
        if not isinstance(extent, SourceNormalizedAxisExtent):
            continue
        numerator = 1
        for axis_index in extent.normalized_axis_indices:
            numerator *= output_shape[axis_index]
        padding = physical_axis.resolve(output_shape) * extent.divisor - numerator
        if padding == 0:
            continue
        logical_axes = {
            axis_name
            for mapping in edge.axis_mappings
            if mapping.source_axis_index in extent.normalized_axis_indices
            if (axis_name := _source_target_axis_name(mapping.target)) is not None
            if not isinstance(mapping.target, ComponentAxisTarget)
            or mapping.target.component_role == resolved_role
        }
        logical_axis = (
            next(iter(logical_axes))
            if len(logical_axes) == 1
            else physical_axis.axis_name
        )
        padding_items.append((physical_axis.axis_name, logical_axis, padding))
    duplicated_axes = {
        logical_axis
        for _, logical_axis, _ in padding_items
        if sum(item[1] == logical_axis for item in padding_items) > 1
    }
    if not padding_items:
        if source_component.padding_semantics is not SourcePaddingSemantics.NO_PADDING:
            raise ValueError("source padding semantics require physical padding")
        return ()
    if source_component.padding_semantics is SourcePaddingSemantics.NO_PADDING:
        raise ValueError("source physical padding lacks attested padding semantics")
    padding_semantics = PhysicalPaddingSemantics(
        source_component.padding_semantics.value
    )
    return tuple(
        PhysicalPadding(
            logical_axis=(
                physical_axis_name if logical_axis in duplicated_axes else logical_axis
            ),
            pad_before=0,
            pad_after=padding,
            semantics=padding_semantics,
            fill_encoding=(
                None
                if source_component.padding_fill_encoding is None
                else _versioned_source_encoding(source_component.padding_fill_encoding)
            ),
        )
        for physical_axis_name, logical_axis, padding in padding_items
    )


def _source_layout_mapping_axes(
    context: RefitBindingContext,
    source_component: SourceStorageComponent,
    *,
    source_binding_slice: RuntimeSourceBindingSlice | None = None,
    component_role: ComponentRole | None = None,
) -> dict[str, frozenset[str]]:
    resolved_slice = source_binding_slice or context.source_binding_slice
    resolved_role = component_role or context.component_role
    edge = resolved_slice.source_binding.classification_edge
    mappings: dict[str, set[str]] = {}
    for mapping in edge.axis_mappings:
        target = mapping.target
        if (
            isinstance(target, ComponentAxisTarget)
            and target.component_role != resolved_role
        ):
            continue
        logical_axis = _source_target_axis_name(target)
        if logical_axis is None:
            continue
        for physical_axis in source_component.physical_axes:
            extent = physical_axis.extent
            if (
                isinstance(extent, SourceNormalizedAxisExtent)
                and mapping.source_axis_index in extent.normalized_axis_indices
            ):
                mappings.setdefault(logical_axis, set()).add(physical_axis.axis_name)
    for physical_axis in source_component.physical_axes:
        if isinstance(physical_axis.extent, SourceLiteralAxisExtent):
            mappings.setdefault(physical_axis.axis_name, set()).add(
                physical_axis.axis_name
            )
    return {
        logical_axis: frozenset(physical_axes)
        for logical_axis, physical_axes in mappings.items()
    }


def _derive_source_storage_layout(
    context: RefitBindingContext,
    source_component: SourceStorageComponent,
    *,
    normalized_shape: tuple[int, ...] | None = None,
    source_binding_slice: RuntimeSourceBindingSlice | None = None,
    source_realization: SourceStorageRealization | None = None,
    component_role: ComponentRole | None = None,
) -> PhysicalLayoutDescriptor:
    axis_order = tuple(axis.axis_name for axis in source_component.physical_axes)
    mapping_axes = _source_layout_mapping_axes(
        context,
        source_component,
        source_binding_slice=source_binding_slice,
        component_role=component_role,
    )
    padding = _source_padding(
        context,
        source_component,
        normalized_shape=normalized_shape,
        source_binding_slice=source_binding_slice,
        source_realization=source_realization,
        component_role=component_role,
    )
    for item in padding:
        if item.logical_axis not in mapping_axes:
            mapping_axes[item.logical_axis] = frozenset((item.logical_axis,))
    axes_by_name = {axis.axis_name: axis for axis in source_component.physical_axes}
    mappings = tuple(
        PhysicalAxisMapping(
            logical_axis=logical_axis,
            physical_axes=tuple(
                axis for axis in axis_order if axis in mapped_physical_axes
            ),
            mapping_id=_source_axis_mapping_id(
                logical_axis,
                tuple(
                    axes_by_name[axis]
                    for axis in axis_order
                    if axis in mapped_physical_axes
                ),
            ),
        )
        for logical_axis, mapped_physical_axes in sorted(mapping_axes.items())
    )
    permutation = (
        None
        if source_component.permutation_id == IDENTITY_PERMUTATION_ID
        else PhysicalPermutation(
            permutation_id=source_component.permutation_id,
            input_axis_order=axis_order,
            output_axis_order=axis_order,
        )
    )
    return PhysicalLayoutDescriptor(
        axis_order=axis_order,
        logical_to_physical_axes=mappings,
        padding=padding,
        permutation=permutation,
        storage_encoding=_versioned_source_encoding(source_component.storage_encoding),
        swizzle_id=source_component.swizzle_id,
    )


def derive_source_storage_layout(
    context: InstalledRefitBindingContext,
    source_component: SourceStorageComponent,
) -> PhysicalLayoutDescriptor:
    """Derive the only physical layout admitted for one attested source component."""
    validated_context = _require_installed_binding_context(context)
    matches = tuple(
        (source_slice, realization)
        for source_slice, realization in zip(
            validated_context.source_binding_slices,
            validated_context.source_realizations,
            strict=True,
        )
        if source_component in realization.components
    )
    if type(source_component) is not SourceStorageComponent or len(matches) != 1:
        raise ValueError("source component is absent or ambiguous in its realization")
    source_slice, realization = matches[0]
    return _derive_source_storage_layout(
        validated_context,
        source_component,
        source_binding_slice=source_slice,
        source_realization=realization,
        component_role=source_slice.component_key.component_role,
    )


def _selected_source_shape(region: SourceRegion) -> tuple[int, ...]:
    return tuple(selection.cardinality for selection in region.axis_selections)


def derive_source_region_extraction_capability(
    context: InstalledRefitBindingContext,
    realized_format: RealizedBindingFormat,
) -> SourceRegionExtractionCapability:
    """Derive the exact gather/split output admitted for a compact source region."""
    validated_context = _require_installed_binding_context(context)
    if type(realized_format) is not RealizedBindingFormat:
        raise TypeError("realized_format must be RealizedBindingFormat")
    if all(
        _source_region_is_complete(source_slice.source_region)
        for source_slice in validated_context.source_binding_slices
    ):
        raise ValueError("source region extraction requires a partial source region")
    _require_source_storage_lowering(validated_context, realized_format)
    if realized_format.wire_format != realized_format.source_storage_format:
        raise ValueError("source gather/split cannot change the atomic numeric format")
    wire = realized_format.wire
    selected_shapes: list[tuple[int, ...]] = []
    source_component_ids: list[str] = []
    for wire_component, source_slice, realization in zip(
        wire,
        validated_context.source_binding_slices,
        validated_context.source_realizations,
        strict=True,
    ):
        if len(realization.components) != 1:
            raise ValueError(
                "source gather/split requires one physical component per semantic role"
            )
        source_component = realization.components[0]
        selected_shape = _selected_source_shape(source_slice.source_region)
        expected_wire_shape = tuple(
            axis.resolve(selected_shape) for axis in source_component.physical_axes
        )
        expected_wire_layout = _derive_source_storage_layout(
            validated_context,
            source_component,
            normalized_shape=selected_shape,
            source_binding_slice=source_slice,
            source_realization=realization,
            component_role=source_slice.component_key.component_role,
        )
        wire_representation = wire_component.representation
        if (
            wire_component.source_storage_component is not None
            or wire_representation.role != source_slice.component_key.component_role
            or wire_representation.physical_dtype
            != source_component.carrier_dtype.value
            or wire_representation.physical_shape != expected_wire_shape
            or wire_representation.layout != expected_wire_layout
        ):
            raise ValueError(
                "source gather/split wire shape, order, dtype, or layout is not derived from the selected region"
            )
        selected_shapes.append(selected_shape)
        source_component_ids.append(source_component.native_component_id)
    region_digest = _canonical_digest(
        {
            "type": "source_region_set.v1",
            "regions": [
                _source_region_payload(source_slice.source_region)
                for source_slice in validated_context.source_binding_slices
            ],
        }
    )
    return SourceRegionExtractionCapability(
        kind=SourceRegionTransformKind.GATHER_SPLIT,
        source_binding_set_digest=validated_context.source_binding_slice_digest,
        source_region_digest=region_digest,
        selected_source_shapes=tuple(selected_shapes),
        source_component_ids=tuple(source_component_ids),
        wire_component_roles=tuple(component.representation.role for component in wire),
        wire_component_shapes=tuple(
            component.representation.physical_shape for component in wire
        ),
        wire_component_axis_orders=tuple(
            component.representation.layout.axis_order for component in wire
        ),
        wire_representation_digest=physical_representation_digest(
            realized_format,
            PhysicalFormatStage.WIRE,
        ),
    )


def _require_source_storage_lowering(
    context: RefitBindingContext,
    realized_format: RealizedBindingFormat,
) -> None:
    if realized_format.source_storage_format != context.source_format:
        raise ValueError("source physical binding format differs from compiler intent")
    if len(realized_format.source_storage) != len(context.source_realizations):
        raise ValueError(
            "source storage does not realize every atomic semantic component"
        )
    for physical_component, source_slice, realization in zip(
        realized_format.source_storage,
        context.source_binding_slices,
        context.source_realizations,
        strict=True,
    ):
        if len(realization.components) != 1:
            raise ValueError(
                "each atomic semantic role must bind exactly one native source component"
            )
        source_component = realization.components[0]
        if physical_component.source_storage_component != source_component:
            raise ValueError(
                "source storage shape or component differs from the selected "
                "realization"
            )
        representation = physical_component.representation
        expected_shape = tuple(
            axis.resolve(realization.output_shape)
            for axis in source_component.physical_axes
        )
        if (
            source_component.physical_shape != expected_shape
            or representation.physical_shape != expected_shape
            or representation.physical_dtype != source_component.carrier_dtype.value
        ):
            raise ValueError(
                "source storage physical shape or dtype differs from the "
                "selected realization"
            )
        if representation.layout != _derive_source_storage_layout(
            context,
            source_component,
            source_binding_slice=source_slice,
            source_realization=realization,
            component_role=source_slice.component_key.component_role,
        ):
            raise ValueError(
                "source storage layout differs from deterministic source realization"
            )


def _physical_binding_digest(realized_format: RealizedBindingFormat) -> str:
    return _canonical_digest(
        {
            "type": "realized_physical_binding.v1",
            "stages": [
                {
                    "stage": stage.value,
                    "representation_digest": physical_representation_digest(
                        realized_format,
                        stage,
                    ),
                    "placement_digest": endpoint_placement_digest(
                        realized_format,
                        stage,
                    ),
                }
                for stage in _STAGE_ORDER
            ],
            "routes": [
                {
                    "source_stage": route.source_stage.value,
                    "destination_stage": route.destination_stage.value,
                    "route_id": route.route_id,
                    "source_endpoint_instance_id": (route.source_endpoint_instance_id),
                    "destination_endpoint_instance_id": (
                        route.destination_endpoint_instance_id
                    ),
                    "source_endpoint_capability_fingerprint": (
                        route.source_endpoint_capability_fingerprint
                    ),
                    "destination_endpoint_capability_fingerprint": (
                        route.destination_endpoint_capability_fingerprint
                    ),
                }
                for route in realized_format.routes
            ],
        }
    )


def _source_storage_lowering_digest(
    context: RefitBindingContext,
    realized_format: RealizedBindingFormat,
) -> str:
    source_route = realized_format.route_between(
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    return _canonical_digest(
        {
            "type": "source_storage_lowering.v1",
            "binding_context_digest": context.binding_context_digest,
            "source_binding_slice_digest": context.source_binding_slice_digest,
            "source_realization_digest": context.source_realization_digest,
            "source_representation_digest": physical_representation_digest(
                realized_format,
                PhysicalFormatStage.SOURCE_STORAGE,
            ),
            "source_placement_digest": endpoint_placement_digest(
                realized_format,
                PhysicalFormatStage.SOURCE_STORAGE,
            ),
            "live_buffer_probe": {
                "endpoint_instance_id": source_route.source_endpoint_instance_id,
                "capability_fingerprint": (
                    source_route.source_endpoint_capability_fingerprint
                ),
            },
        }
    )


@dataclass(frozen=True, slots=True, init=False)
class RefitBindingIdentity:
    """Exact semantic, source, destination-owner, and finalizer identity."""

    graph_instance_id: str
    tensor_instance_id: str
    semantic_graph_path: str
    inventory_entry_id: str
    selection_group_id: str
    semantic_selection_digest: str
    intent_group_id: str
    graph_intent_id: str
    runtime_source_result_digest: str
    source_realization_digest: str
    source_storage_lowering_digest: str
    physical_binding_digest: str
    destination_binding_digest: str
    destination_storage_generation: str
    destination_owner_instance_id: str
    finalizer_instance_id: str
    binding_identity_digest: str = field(init=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "RefitBindingIdentity must be derived from a validated refit binding"
        )

    @classmethod
    def _issue(
        cls,
        factory_token: object,
        *,
        binding_context: RefitBindingContext,
        realized_format: RealizedBindingFormat,
        destination_binding_proof: DestinationBindingProof,
        validated_source_storage_lowering_digest: str | None = None,
        validated_physical_binding_digest: str | None = None,
    ) -> Self:
        if factory_token is not _REFIT_PLANNER_FACTORY_TOKEN:
            raise TypeError("invalid refit planner authority")
        if type(binding_context) is not RefitBindingContext:
            raise TypeError("binding_context must be a validated RefitBindingContext")
        validated_context = binding_context
        if type(realized_format) is not RealizedBindingFormat:
            raise TypeError("realized_format must be RealizedBindingFormat")
        if validated_source_storage_lowering_digest is None:
            _require_source_storage_lowering(validated_context, realized_format)
            source_storage_lowering_digest = _source_storage_lowering_digest(
                validated_context,
                realized_format,
            )
        else:
            source_storage_lowering_digest = _require_sha256_digest(
                validated_source_storage_lowering_digest,
                "validated_source_storage_lowering_digest",
            )
        if validated_physical_binding_digest is None:
            physical_binding_digest = _physical_binding_digest(realized_format)
        else:
            physical_binding_digest = _require_sha256_digest(
                validated_physical_binding_digest,
                "validated_physical_binding_digest",
            )
        if type(destination_binding_proof) is not DestinationBindingProof:
            raise TypeError("destination_binding_proof must be DestinationBindingProof")
        if (
            realized_format.destination_runtime_format
            != validated_context.destination_format
        ):
            raise ValueError(
                "destination physical binding format differs from compiler intent"
            )
        identity = object.__new__(cls)
        for field_name, value in (
            ("graph_instance_id", validated_context.graph_instance_id),
            ("tensor_instance_id", validated_context.tensor_instance_id),
            ("semantic_graph_path", validated_context.semantic_graph_path),
            ("inventory_entry_id", validated_context.inventory_entry_id),
            ("selection_group_id", validated_context.selection_group_id),
            (
                "semantic_selection_digest",
                validated_context.semantic_selection_digest,
            ),
            ("intent_group_id", validated_context.intent_group_id),
            ("graph_intent_id", validated_context.graph_intent_id),
            (
                "runtime_source_result_digest",
                validated_context.runtime_source_result_digest,
            ),
            (
                "source_realization_digest",
                validated_context.source_realization_digest,
            ),
            (
                "source_storage_lowering_digest",
                source_storage_lowering_digest,
            ),
            ("physical_binding_digest", physical_binding_digest),
            (
                "destination_binding_digest",
                destination_binding_proof.destination_binding_digest,
            ),
            (
                "destination_storage_generation",
                destination_binding_proof.destination_storage_generation,
            ),
            (
                "destination_owner_instance_id",
                destination_binding_proof.destination_owner_instance_id,
            ),
            (
                "finalizer_instance_id",
                destination_binding_proof.finalizer_instance_id,
            ),
        ):
            object.__setattr__(identity, field_name, value)
        identity.__post_init__()
        return identity

    def __post_init__(self) -> None:
        for field_name in (
            "graph_instance_id",
            "tensor_instance_id",
            "semantic_graph_path",
            "inventory_entry_id",
            "destination_owner_instance_id",
            "finalizer_instance_id",
            "destination_storage_generation",
        ):
            _require_text(getattr(self, field_name), field_name)
        for field_name in (
            "selection_group_id",
            "semantic_selection_digest",
            "intent_group_id",
            "graph_intent_id",
            "runtime_source_result_digest",
            "source_realization_digest",
            "source_storage_lowering_digest",
            "physical_binding_digest",
            "destination_binding_digest",
        ):
            _require_sha256_digest(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "binding_identity_digest",
            _canonical_digest(
                {
                    "type": "refit_binding_identity.v1",
                    **_binding_identity_payload(self),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RefitExecutorKey:
    """Exact versioned executor lookup key retained by a compiled operation."""

    adapter_id: str
    adapter_version: str
    implementation_id: str
    implementation_version: str
    transform_locus: TransformLocus

    def __post_init__(self) -> None:
        _require_identifier(self.adapter_id, "executor adapter_id")
        _require_implementation_version(
            self.adapter_version,
            "executor adapter_version",
        )
        _require_identifier(self.implementation_id, "executor implementation_id")
        _require_implementation_version(
            self.implementation_version,
            "executor implementation_version",
        )
        _require_enum(
            self.transform_locus,
            TransformLocus,
            "executor transform_locus",
        )

    @classmethod
    def _from_proof(
        cls,
        proof: DirectCopyCapabilityProof | TransformCapabilityProof,
    ) -> Self:
        return cls(
            adapter_id=proof.adapter_id,
            adapter_version=proof.adapter_version,
            implementation_id=proof.implementation_id,
            implementation_version=proof.implementation_version,
            transform_locus=_proof_locus(proof),
        )


@dataclass(frozen=True, slots=True, init=False)
class SelectedRefitOperation:
    """Opaque compiled operation consumed by execution dispatch.

    The adapter capability remains reusable and graph-neutral.  Selection adds
    the graph/tensor/semantic/source identity once, at plan compilation, and
    preserves the concrete executor key so dispatch never receives a bare
    transform locus.
    """

    binding_identity: RefitBindingIdentity
    binding_context: RefitBindingContext
    realized_format: RealizedBindingFormat
    destination_binding_proof: DestinationBindingProof
    source_stage: PhysicalFormatStage
    destination_stage: PhysicalFormatStage
    locus: TransformLocus
    executor_key: RefitExecutorKey
    capability_proof: DirectCopyCapabilityProof | TransformCapabilityProof
    operation_base_proof: RefitOperationBaseProof
    selected_operation_digest: str
    signature: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "SelectedRefitOperation must be produced by refit plan compilation"
        )

    @classmethod
    def _issue(
        cls,
        factory_token: object,
        *,
        binding_identity: RefitBindingIdentity,
        binding_context: RefitBindingContext,
        realized_format: RealizedBindingFormat,
        destination_binding_proof: DestinationBindingProof,
        capability_proof: DirectCopyCapabilityProof | TransformCapabilityProof,
        operation_base_proof: RefitOperationBaseProof,
    ) -> Self:
        if factory_token is not _REFIT_PLANNER_FACTORY_TOKEN:
            raise TypeError("invalid refit planner authority")
        if type(binding_identity) is not RefitBindingIdentity:
            raise TypeError("binding_identity must be RefitBindingIdentity")
        if type(binding_context) is not RefitBindingContext:
            raise TypeError("binding_context must be RefitBindingContext")
        if type(realized_format) is not RealizedBindingFormat:
            raise TypeError("realized_format must be RealizedBindingFormat")
        if type(destination_binding_proof) is not DestinationBindingProof:
            raise TypeError("destination_binding_proof must be DestinationBindingProof")
        if type(capability_proof) not in {
            DirectCopyCapabilityProof,
            TransformCapabilityProof,
        }:
            raise TypeError("capability_proof must be adapter-issued")
        if type(operation_base_proof) is not RefitOperationBaseProof:
            raise TypeError("operation_base_proof must be adapter-issued")
        selected = object.__new__(cls)
        object.__setattr__(selected, "binding_identity", binding_identity)
        object.__setattr__(selected, "binding_context", binding_context)
        object.__setattr__(selected, "realized_format", realized_format)
        object.__setattr__(
            selected,
            "destination_binding_proof",
            destination_binding_proof,
        )
        object.__setattr__(selected, "source_stage", capability_proof.source_stage)
        object.__setattr__(
            selected,
            "destination_stage",
            capability_proof.destination_stage,
        )
        object.__setattr__(selected, "locus", _proof_locus(capability_proof))
        object.__setattr__(
            selected,
            "executor_key",
            RefitExecutorKey._from_proof(capability_proof),
        )
        object.__setattr__(selected, "capability_proof", capability_proof)
        object.__setattr__(selected, "operation_base_proof", operation_base_proof)
        object.__setattr__(
            selected,
            "selected_operation_digest",
            _canonical_digest(
                _selected_operation_payload(cast(SelectedRefitOperation, selected))
            ),
        )
        object.__setattr__(selected, "signature", "hmac-sha256:" + "0" * 64)
        return selected


def _proof_locus(
    proof: DirectCopyCapabilityProof | TransformCapabilityProof,
) -> TransformLocus:
    if type(proof) is DirectCopyCapabilityProof:
        return TransformLocus.NONE
    return proof.transform_locus


def _proof_capability(
    proof: DirectCopyCapabilityProof | TransformCapabilityProof,
) -> AdapterOperationCapability:
    return AdapterOperationCapability(
        source_stage=proof.source_stage,
        destination_stage=proof.destination_stage,
        transform_locus=_proof_locus(proof),
        route_id=proof.route_id,
        source_endpoint_instance_id=(
            proof.source_endpoint_capability.endpoint_instance_id
        ),
        destination_endpoint_instance_id=(
            proof.destination_endpoint_capability.endpoint_instance_id
        ),
        source_endpoint_capability_fingerprint=(
            proof.source_endpoint_capability.capability_fingerprint
        ),
        destination_endpoint_capability_fingerprint=(
            proof.destination_endpoint_capability.capability_fingerprint
        ),
        source_representation_digest=(
            proof.source_endpoint_capability.ordered_representation_digest
        ),
        destination_representation_digest=(
            proof.destination_endpoint_capability.ordered_representation_digest
        ),
        source_placement_digest=proof.source_endpoint_capability.placement_digest,
        destination_placement_digest=(
            proof.destination_endpoint_capability.placement_digest
        ),
        implementation_id=proof.implementation_id,
        implementation_version=proof.implementation_version,
        source_region_extraction=(
            proof.source_region_extraction
            if type(proof) is TransformCapabilityProof
            else None
        ),
    )


def _proof_payload(
    proof: DirectCopyCapabilityProof | TransformCapabilityProof,
) -> dict[str, object]:
    extraction = (
        _validate_source_region_extraction_capability(proof.source_region_extraction)
        if type(proof) is TransformCapabilityProof
        and proof.source_region_extraction is not None
        else None
    )
    return {
        "type": "adapter_operation_proof.v2",
        "proof_kind": (
            "direct_copy" if type(proof) is DirectCopyCapabilityProof else "transform"
        ),
        "adapter_id": proof.adapter_id,
        "adapter_version": proof.adapter_version,
        "issuer_instance_id": proof.issuer_instance_id,
        "registry_digest": proof.registry_digest,
        "capability": {
            "source_stage": proof.source_stage.value,
            "destination_stage": proof.destination_stage.value,
            "transform_locus": _proof_locus(proof).value,
            "route_id": proof.route_id,
            "source_endpoint_instance_id": (
                proof.source_endpoint_capability.endpoint_instance_id
            ),
            "destination_endpoint_instance_id": (
                proof.destination_endpoint_capability.endpoint_instance_id
            ),
            "source_endpoint_capability_fingerprint": (
                proof.source_endpoint_capability.capability_fingerprint
            ),
            "destination_endpoint_capability_fingerprint": (
                proof.destination_endpoint_capability.capability_fingerprint
            ),
            "source_representation_digest": (
                proof.source_endpoint_capability.ordered_representation_digest
            ),
            "destination_representation_digest": (
                proof.destination_endpoint_capability.ordered_representation_digest
            ),
            "source_placement_digest": (
                proof.source_endpoint_capability.placement_digest
            ),
            "destination_placement_digest": (
                proof.destination_endpoint_capability.placement_digest
            ),
            "implementation_id": proof.implementation_id,
            "implementation_version": proof.implementation_version,
            "source_region_extraction": (
                None
                if extraction is None
                else _source_region_extraction_payload(extraction)
            ),
        },
    }


def _selected_operation_payload(
    selected: SelectedRefitOperation,
) -> dict[str, object]:
    proof = selected.capability_proof
    return {
        "type": "selected_refit_operation.v1",
        "binding_identity": _binding_identity_payload(selected.binding_identity),
        "binding_identity_digest": (selected.binding_identity.binding_identity_digest),
        "binding_context_digest": selected.binding_context.binding_context_digest,
        "physical_binding_digest": _physical_binding_digest(selected.realized_format),
        "destination_binding_digest": (
            selected.destination_binding_proof.destination_binding_digest
        ),
        "source_stage": selected.source_stage.value,
        "destination_stage": selected.destination_stage.value,
        "transform_locus": selected.locus.value,
        "executor_key": {
            "adapter_id": selected.executor_key.adapter_id,
            "adapter_version": selected.executor_key.adapter_version,
            "implementation_id": selected.executor_key.implementation_id,
            "implementation_version": selected.executor_key.implementation_version,
            "transform_locus": selected.executor_key.transform_locus.value,
        },
        "adapter_capability": {
            "proof_kind": (
                "direct_copy"
                if type(proof) is DirectCopyCapabilityProof
                else "transform"
            ),
            "adapter_id": proof.adapter_id,
            "adapter_version": proof.adapter_version,
            "capability": _operation_capability_payload(_proof_capability(proof)),
        },
    }


@dataclass(frozen=True, slots=True)
class _AuthenticatedSelectedOperationReference:
    selected_operation_digest: str
    binding_identity_digest: str
    binding_context_digest: str
    source_storage_lowering_digest: str
    physical_binding_digest: str
    destination_binding_digest: str
    destination_storage_generation: str
    destination_registry_instance_id: str
    destination_registry_generation: int
    source_stage: PhysicalFormatStage
    destination_stage: PhysicalFormatStage
    locus: TransformLocus
    executor_key: RefitExecutorKey
    capability_proof_kind: str
    capability_adapter_id: str
    capability_adapter_version: str
    capability_issuer_instance_id: str
    capability_registry_digest: str
    capability_signature: str
    capability_implementation_id: str
    capability_implementation_version: str
    operation_base_proof: RefitOperationBaseProof
    signature: str


def _capture_selected_operation_reference(
    selected: SelectedRefitOperation,
) -> _AuthenticatedSelectedOperationReference:
    if type(selected) is not SelectedRefitOperation:
        raise TypeError("selected must be SelectedRefitOperation")
    if type(selected.binding_identity) is not RefitBindingIdentity:
        raise TypeError("selected binding identity has a non-exact type")
    if type(selected.binding_context) is not RefitBindingContext:
        raise TypeError("selected binding context has a non-exact type")
    if type(selected.realized_format) is not RealizedBindingFormat:
        raise TypeError("selected realized format has a non-exact type")
    if type(selected.destination_binding_proof) is not DestinationBindingProof:
        raise TypeError("selected destination proof has a non-exact type")
    if type(selected.executor_key) is not RefitExecutorKey:
        raise TypeError("selected executor key has a non-exact type")
    proof = selected.capability_proof
    if type(proof) not in {
        DirectCopyCapabilityProof,
        TransformCapabilityProof,
    }:
        raise TypeError("selected capability proof has a non-exact type")
    if type(selected.operation_base_proof) is not RefitOperationBaseProof:
        raise TypeError("selected operation base proof has a non-exact type")
    return _AuthenticatedSelectedOperationReference(
        selected_operation_digest=selected.selected_operation_digest,
        binding_identity_digest=selected.binding_identity.binding_identity_digest,
        binding_context_digest=selected.binding_context.binding_context_digest,
        source_storage_lowering_digest=(
            selected.binding_identity.source_storage_lowering_digest
        ),
        physical_binding_digest=selected.binding_identity.physical_binding_digest,
        destination_binding_digest=(
            selected.destination_binding_proof.destination_binding_digest
        ),
        destination_storage_generation=(
            selected.destination_binding_proof.destination_storage_generation
        ),
        destination_registry_instance_id=(
            selected.destination_binding_proof.destination_registry_instance_id
        ),
        destination_registry_generation=(
            selected.destination_binding_proof.destination_registry_generation
        ),
        source_stage=selected.source_stage,
        destination_stage=selected.destination_stage,
        locus=selected.locus,
        executor_key=deepcopy(selected.executor_key),
        capability_proof_kind=(
            "direct_copy" if type(proof) is DirectCopyCapabilityProof else "transform"
        ),
        capability_adapter_id=proof.adapter_id,
        capability_adapter_version=proof.adapter_version,
        capability_issuer_instance_id=proof.issuer_instance_id,
        capability_registry_digest=proof.registry_digest,
        capability_signature=proof.signature,
        capability_implementation_id=proof.implementation_id,
        capability_implementation_version=proof.implementation_version,
        operation_base_proof=deepcopy(selected.operation_base_proof),
        signature=selected.signature,
    )


def _selected_operation_authentication_payload(
    reference: _AuthenticatedSelectedOperationReference,
) -> dict[str, object]:
    return {
        "type": "selected_refit_operation_authentication.v1",
        "selected_operation_digest": reference.selected_operation_digest,
        "binding_identity_digest": reference.binding_identity_digest,
        "binding_context_digest": reference.binding_context_digest,
        "operation_base_digest": reference.operation_base_proof.base_digest,
        "operation_base_signature": reference.operation_base_proof.signature,
        "physical_binding_digest": reference.physical_binding_digest,
        "destination_binding_digest": reference.destination_binding_digest,
        "source_stage": reference.source_stage.value,
        "destination_stage": reference.destination_stage.value,
        "transform_locus": reference.locus.value,
        "executor_key": {
            "adapter_id": reference.executor_key.adapter_id,
            "adapter_version": reference.executor_key.adapter_version,
            "implementation_id": reference.executor_key.implementation_id,
            "implementation_version": reference.executor_key.implementation_version,
            "transform_locus": reference.executor_key.transform_locus.value,
        },
        "adapter_capability": {
            "proof_kind": reference.capability_proof_kind,
            "adapter_id": reference.capability_adapter_id,
            "adapter_version": reference.capability_adapter_version,
            "issuer_instance_id": reference.capability_issuer_instance_id,
            "registry_digest": reference.capability_registry_digest,
            "signature": reference.capability_signature,
        },
    }


class AdapterCapabilityRegistry:
    """Trusted, immutable version-adapter capability registry and proof issuer."""

    __slots__ = (
        "_adapter_id",
        "_adapter_version",
        "_capabilities",
        "_destination_registry",
        "_issuer_instance_id",
        "_registry_digest",
        "_secret",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "AdapterCapabilityRegistry is created only inside the version-adapter "
            "trust boundary"
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("adapter capability registry is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("adapter capability registry is immutable")

    @classmethod
    def _create(
        cls,
        factory_token: object,
        *,
        adapter_id: str,
        adapter_version: str,
        capabilities: Sequence[AdapterOperationCapability],
    ) -> Self:
        if factory_token is not _ADAPTER_REGISTRY_FACTORY_TOKEN:
            raise TypeError("invalid adapter capability registry factory authority")
        _require_identifier(adapter_id, "adapter_id")
        _require_implementation_version(adapter_version, "adapter_version")
        captured_capabilities = _snapshot_sequence(capabilities, "capabilities")
        if not captured_capabilities:
            raise ValueError("capabilities must be non-empty")
        if any(
            type(capability) is not AdapterOperationCapability
            for capability in captured_capabilities
        ):
            raise TypeError(
                "capabilities must contain AdapterOperationCapability records"
            )
        if len(captured_capabilities) != len(set(captured_capabilities)):
            raise ValueError("capabilities must be duplicate-free")
        registry = object.__new__(cls)
        object.__setattr__(registry, "_adapter_id", adapter_id)
        object.__setattr__(registry, "_adapter_version", adapter_version)
        object.__setattr__(
            registry,
            "_capabilities",
            frozenset(captured_capabilities),
        )
        object.__setattr__(
            registry,
            "_destination_registry",
            LiveDestinationBindingRegistry._create(
                _DESTINATION_REGISTRY_FACTORY_TOKEN,
                adapter_id=adapter_id,
                adapter_version=adapter_version,
            ),
        )
        sorted_capability_payloads = sorted(
            (
                _operation_capability_payload(capability)
                for capability in captured_capabilities
            ),
            key=lambda payload: _canonical_json_bytes(payload),
        )
        object.__setattr__(
            registry,
            "_registry_digest",
            _canonical_digest(
                {
                    "adapter_id": adapter_id,
                    "adapter_version": adapter_version,
                    "capabilities": sorted_capability_payloads,
                }
            ),
        )
        object.__setattr__(
            registry,
            "_issuer_instance_id",
            f"issuer-{secrets.token_hex(32)}",
        )
        object.__setattr__(registry, "_secret", secrets.token_bytes(32))
        return registry

    @property
    def adapter_id(self) -> str:
        """Return the trusted version-adapter implementation identity."""
        return self._adapter_id

    @property
    def adapter_version(self) -> str:
        """Return the trusted version-adapter implementation version."""
        return self._adapter_version

    @property
    def issuer_instance_id(self) -> str:
        """Return this runtime issuer's unpredictable instance identity."""
        return self._issuer_instance_id

    @property
    def registry_digest(self) -> str:
        """Return the immutable admitted-capability set digest."""
        return self._registry_digest

    def _require_registered(
        self,
        capability: AdapterOperationCapability,
    ) -> None:
        if capability not in self._capabilities:
            raise ValueError(
                "adapter capability registry does not authorize the exact operation"
            )

    def _endpoint_proof(
        self,
        capability: AdapterOperationCapability,
        *,
        source: bool,
    ) -> EndpointCapabilityProof:
        return EndpointCapabilityProof._issue(
            _ADAPTER_REGISTRY_FACTORY_TOKEN,
            stage=(capability.source_stage if source else capability.destination_stage),
            endpoint_instance_id=(
                capability.source_endpoint_instance_id
                if source
                else capability.destination_endpoint_instance_id
            ),
            capability_fingerprint=(
                capability.source_endpoint_capability_fingerprint
                if source
                else capability.destination_endpoint_capability_fingerprint
            ),
            ordered_representation_digest=(
                capability.source_representation_digest
                if source
                else capability.destination_representation_digest
            ),
            placement_digest=(
                capability.source_placement_digest
                if source
                else capability.destination_placement_digest
            ),
        )

    def _sign(
        self,
        proof: DirectCopyCapabilityProof | TransformCapabilityProof,
    ) -> None:
        signature = hmac.new(
            self._secret,
            _canonical_json_bytes(_proof_payload(proof)),
            sha256,
        ).hexdigest()
        object.__setattr__(proof, "signature", f"hmac-sha256:{signature}")

    def _proof_fields(
        self,
        capability: AdapterOperationCapability,
    ) -> dict[str, object]:
        return {
            "source_stage": capability.source_stage,
            "destination_stage": capability.destination_stage,
            "route_id": capability.route_id,
            "source_endpoint_capability": self._endpoint_proof(
                capability,
                source=True,
            ),
            "destination_endpoint_capability": self._endpoint_proof(
                capability,
                source=False,
            ),
            "implementation_id": capability.implementation_id,
            "implementation_version": capability.implementation_version,
            "adapter_id": self._adapter_id,
            "adapter_version": self._adapter_version,
            "issuer_instance_id": self._issuer_instance_id,
            "registry_digest": self._registry_digest,
            "signature": "hmac-sha256:" + "0" * 64,
        }

    def issue_direct_copy(
        self,
        realized_format: RealizedBindingFormat,
        source_stage: PhysicalFormatStage,
        destination_stage: PhysicalFormatStage,
        *,
        implementation_id: str,
        implementation_version: str,
    ) -> DirectCopyCapabilityProof:
        """Issue a signed proof for one pre-registered exact copy."""
        capability = AdapterOperationCapability.from_realized_format(
            realized_format,
            source_stage,
            destination_stage,
            transform_locus=TransformLocus.NONE,
            implementation_id=implementation_id,
            implementation_version=implementation_version,
        )
        self._require_registered(capability)
        proof = DirectCopyCapabilityProof._issue(
            _ADAPTER_REGISTRY_FACTORY_TOKEN,
            **self._proof_fields(capability),
        )
        self._sign(proof)
        return proof

    def issue_transform(
        self,
        realized_format: RealizedBindingFormat,
        source_stage: PhysicalFormatStage,
        destination_stage: PhysicalFormatStage,
        *,
        transform_locus: TransformLocus,
        implementation_id: str,
        implementation_version: str,
        source_region_extraction: SourceRegionExtractionCapability | None = None,
    ) -> TransformCapabilityProof:
        """Issue a signed proof for one pre-registered exact transform."""
        if transform_locus is TransformLocus.NONE:
            raise ValueError("issue_transform cannot authorize direct copy")
        capability = AdapterOperationCapability.from_realized_format(
            realized_format,
            source_stage,
            destination_stage,
            transform_locus=transform_locus,
            implementation_id=implementation_id,
            implementation_version=implementation_version,
            source_region_extraction=source_region_extraction,
        )
        self._require_registered(capability)
        proof = TransformCapabilityProof._issue(
            _ADAPTER_REGISTRY_FACTORY_TOKEN,
            transform_locus=transform_locus,
            source_region_extraction=capability.source_region_extraction,
            **self._proof_fields(capability),
        )
        self._sign(proof)
        return proof

    def _issue_destination_binding(
        self,
        mint_authority: DestinationBindingMintAuthority,
        realized_format: RealizedBindingFormat,
        *,
        evidence: DestinationBindingEvidence,
    ) -> DestinationBindingProof:
        """Attest one identity-registered live destination discovery result."""
        if type(realized_format) is not RealizedBindingFormat:
            raise TypeError("realized_format must be RealizedBindingFormat")
        self._destination_registry._require_mint_authority(mint_authority)
        record = self._destination_registry._require_evidence(evidence, realized_format)
        validated_evidence = record.snapshot
        expected_physical_digests = record.physical_digests
        fields: dict[str, object] = {
            "destination_owner_instance_id": (
                validated_evidence.destination_owner_instance_id
            ),
            "destination_owner_capability_fingerprint": (
                validated_evidence.destination_owner_capability_fingerprint
            ),
            "finalizer_instance_id": validated_evidence.finalizer_instance_id,
            "finalizer_capability_fingerprint": (
                validated_evidence.finalizer_capability_fingerprint
            ),
            "destination_storage_generation": (
                validated_evidence.destination_storage_generation
            ),
            "destination_load_api_representation_digest": expected_physical_digests[0],
            "destination_load_api_placement_digest": expected_physical_digests[1],
            "destination_runtime_representation_digest": expected_physical_digests[2],
            "destination_runtime_placement_digest": expected_physical_digests[3],
            "adapter_id": self._adapter_id,
            "adapter_version": self._adapter_version,
            "issuer_instance_id": self._issuer_instance_id,
            "registry_digest": self._registry_digest,
            "destination_registry_instance_id": (
                validated_evidence.destination_registry_instance_id
            ),
            "destination_registry_generation": (
                validated_evidence.destination_registry_generation
            ),
            "destination_evidence_id": validated_evidence.evidence_id,
            "destination_binding_digest": "sha256:" + "0" * 64,
            "signature": "hmac-sha256:" + "0" * 64,
        }
        proof = DestinationBindingProof._issue(
            _ADAPTER_REGISTRY_FACTORY_TOKEN,
            **fields,
        )
        object.__setattr__(
            proof,
            "destination_binding_digest",
            _canonical_digest(_destination_binding_payload(proof)),
        )
        signature = hmac.new(
            self._secret,
            _canonical_json_bytes(
                {
                    **_destination_binding_payload(proof),
                    "issuer_instance_id": proof.issuer_instance_id,
                    "registry_digest": proof.registry_digest,
                    "destination_binding_digest": proof.destination_binding_digest,
                }
            ),
            sha256,
        ).hexdigest()
        object.__setattr__(proof, "signature", f"hmac-sha256:{signature}")
        return proof

    def _verify_destination_binding(
        self,
        proof: DestinationBindingProof,
        realized_format: RealizedBindingFormat,
    ) -> None:
        if type(proof) is not DestinationBindingProof:
            raise TypeError("destination_binding_proof must be DestinationBindingProof")
        for field_name in (
            "destination_owner_instance_id",
            "finalizer_instance_id",
            "destination_storage_generation",
            "issuer_instance_id",
            "destination_registry_instance_id",
        ):
            _require_text(getattr(proof, field_name), field_name)
        _require_identifier(proof.adapter_id, "destination proof adapter_id")
        _require_implementation_version(
            proof.adapter_version,
            "destination proof adapter_version",
        )
        for field_name in (
            "destination_owner_capability_fingerprint",
            "finalizer_capability_fingerprint",
            "destination_load_api_representation_digest",
            "destination_load_api_placement_digest",
            "destination_runtime_representation_digest",
            "destination_runtime_placement_digest",
            "registry_digest",
            "destination_evidence_id",
            "destination_binding_digest",
        ):
            _require_sha256_digest(getattr(proof, field_name), field_name)
        if (
            type(proof.destination_registry_generation) is not int
            or proof.destination_registry_generation <= 0
        ):
            raise ValueError(
                "destination registry generation must be a positive integer"
            )
        expected_digest = _canonical_digest(_destination_binding_payload(proof))
        if proof.destination_binding_digest != expected_digest:
            raise ValueError("destination binding proof digest mismatch")
        if (
            proof.adapter_id != self._adapter_id
            or proof.adapter_version != self._adapter_version
            or proof.issuer_instance_id != self._issuer_instance_id
            or proof.registry_digest != self._registry_digest
        ):
            raise ValueError("destination binding proof issuer identity mismatch")
        self._destination_registry._require_proof_live(proof)
        expected_signature = (
            "hmac-sha256:"
            + hmac.new(
                self._secret,
                _canonical_json_bytes(
                    {
                        **_destination_binding_payload(proof),
                        "issuer_instance_id": proof.issuer_instance_id,
                        "registry_digest": proof.registry_digest,
                        "destination_binding_digest": proof.destination_binding_digest,
                    }
                ),
                sha256,
            ).hexdigest()
        )
        if not hmac.compare_digest(proof.signature, expected_signature):
            raise ValueError("destination binding proof signature mismatch")
        expected_physical_digests = _destination_physical_digests(realized_format)
        proof_physical_digests = (
            proof.destination_load_api_representation_digest,
            proof.destination_load_api_placement_digest,
            proof.destination_runtime_representation_digest,
            proof.destination_runtime_placement_digest,
        )
        if proof_physical_digests != expected_physical_digests:
            raise ValueError(
                "destination binding proof differs from the realized destination"
            )

    def _issue_refit_operation_base_proof(
        self,
        factory_token: object,
        binding_identity: RefitBindingIdentity,
        binding_context: RefitBindingContext,
        destination_binding_proof: DestinationBindingProof,
    ) -> RefitOperationBaseProof:
        if factory_token is not _REFIT_PLANNER_FACTORY_TOKEN:
            raise TypeError("invalid refit operation base proof authority")
        if type(binding_identity) is not RefitBindingIdentity:
            raise TypeError("binding_identity must be RefitBindingIdentity")
        if type(binding_context) is not RefitBindingContext:
            raise TypeError("binding_context must be RefitBindingContext")
        if type(destination_binding_proof) is not DestinationBindingProof:
            raise TypeError("destination_binding_proof must be DestinationBindingProof")
        if (
            binding_identity.destination_binding_digest
            != destination_binding_proof.destination_binding_digest
            or binding_identity.destination_storage_generation
            != destination_binding_proof.destination_storage_generation
        ):
            raise ValueError("binding identity differs from destination proof")
        fields: dict[str, object] = {
            "binding_context_digest": binding_context.binding_context_digest,
            "binding_identity_digest": binding_identity.binding_identity_digest,
            "source_storage_lowering_digest": (
                binding_identity.source_storage_lowering_digest
            ),
            "physical_binding_digest": binding_identity.physical_binding_digest,
            "destination_binding_digest": (
                destination_binding_proof.destination_binding_digest
            ),
            "destination_storage_generation": (
                destination_binding_proof.destination_storage_generation
            ),
            "destination_registry_instance_id": (
                destination_binding_proof.destination_registry_instance_id
            ),
            "destination_registry_generation": (
                destination_binding_proof.destination_registry_generation
            ),
            "adapter_id": self._adapter_id,
            "adapter_version": self._adapter_version,
            "issuer_instance_id": self._issuer_instance_id,
            "registry_digest": self._registry_digest,
            "base_digest": "sha256:" + "0" * 64,
            "signature": "hmac-sha256:" + "0" * 64,
        }
        proof = RefitOperationBaseProof._issue(
            _ADAPTER_REGISTRY_FACTORY_TOKEN,
            **fields,
        )
        object.__setattr__(
            proof,
            "base_digest",
            _canonical_digest(_refit_operation_base_payload(proof)),
        )
        signature = hmac.new(
            self._secret,
            _canonical_json_bytes(
                {
                    **_refit_operation_base_payload(proof),
                    "base_digest": proof.base_digest,
                }
            ),
            sha256,
        ).hexdigest()
        object.__setattr__(proof, "signature", f"hmac-sha256:{signature}")
        return proof

    def _verify_refit_operation_base_proof(
        self,
        proof: RefitOperationBaseProof,
    ) -> None:
        if type(proof) is not RefitOperationBaseProof:
            raise TypeError("operation_base_proof must be RefitOperationBaseProof")
        for field_name in (
            "binding_context_digest",
            "binding_identity_digest",
            "source_storage_lowering_digest",
            "physical_binding_digest",
            "destination_binding_digest",
            "registry_digest",
            "base_digest",
        ):
            _require_sha256_digest(getattr(proof, field_name), field_name)
        _require_text(
            proof.destination_storage_generation,
            "destination_storage_generation",
        )
        _require_text(
            proof.destination_registry_instance_id,
            "destination_registry_instance_id",
        )
        if (
            type(proof.destination_registry_generation) is not int
            or proof.destination_registry_generation <= 0
        ):
            raise ValueError(
                "destination registry generation must be a positive integer"
            )
        _require_identifier(proof.adapter_id, "base proof adapter_id")
        _require_implementation_version(
            proof.adapter_version,
            "base proof adapter_version",
        )
        _require_text(proof.issuer_instance_id, "base proof issuer_instance_id")
        signature = _require_text(proof.signature, "base proof signature")
        if _HMAC_SHA256_PATTERN.fullmatch(signature) is None:
            raise ValueError(
                "base proof signature must be a canonical HMAC-SHA256 value"
            )
        if (
            proof.adapter_id != self._adapter_id
            or proof.adapter_version != self._adapter_version
            or proof.issuer_instance_id != self._issuer_instance_id
            or proof.registry_digest != self._registry_digest
        ):
            raise ValueError("refit operation base proof issuer identity mismatch")
        expected_digest = _canonical_digest(_refit_operation_base_payload(proof))
        if proof.base_digest != expected_digest:
            raise ValueError("refit operation base proof digest mismatch")
        expected_signature = (
            "hmac-sha256:"
            + hmac.new(
                self._secret,
                _canonical_json_bytes(
                    {
                        **_refit_operation_base_payload(proof),
                        "base_digest": proof.base_digest,
                    }
                ),
                sha256,
            ).hexdigest()
        )
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("refit operation base proof signature mismatch")

    def _authenticate_selected_operation(
        self,
        factory_token: object,
        selected: SelectedRefitOperation,
    ) -> None:
        if factory_token is not _REFIT_PLANNER_FACTORY_TOKEN:
            raise TypeError("invalid selected refit operation authority")
        reference = _capture_selected_operation_reference(selected)
        signature = hmac.new(
            self._secret,
            _canonical_json_bytes(
                _selected_operation_authentication_payload(reference)
            ),
            sha256,
        ).hexdigest()
        object.__setattr__(selected, "signature", f"hmac-sha256:{signature}")

    def _verify_selected_operation_authentication(
        self,
        selected: SelectedRefitOperation,
    ) -> _AuthenticatedSelectedOperationReference:
        reference = _capture_selected_operation_reference(selected)
        _require_enum(reference.source_stage, PhysicalFormatStage, "source_stage")
        _require_enum(
            reference.destination_stage,
            PhysicalFormatStage,
            "destination_stage",
        )
        _require_enum(reference.locus, TransformLocus, "transform_locus")
        _require_adjacent_stage_pair(
            reference.source_stage,
            reference.destination_stage,
        )
        _require_identifier(
            reference.capability_adapter_id,
            "capability proof adapter_id",
        )
        _require_implementation_version(
            reference.capability_adapter_version,
            "capability proof adapter_version",
        )
        _require_text(
            reference.capability_issuer_instance_id,
            "capability proof issuer_instance_id",
        )
        _require_sha256_digest(
            reference.capability_registry_digest,
            "capability proof registry_digest",
        )
        capability_signature = _require_text(
            reference.capability_signature,
            "capability proof signature",
        )
        if _HMAC_SHA256_PATTERN.fullmatch(capability_signature) is None:
            raise ValueError(
                "capability proof signature must be a canonical HMAC-SHA256 value"
            )
        expected_executor = RefitExecutorKey(
            adapter_id=reference.capability_adapter_id,
            adapter_version=reference.capability_adapter_version,
            implementation_id=reference.capability_implementation_id,
            implementation_version=reference.capability_implementation_version,
            transform_locus=reference.locus,
        )
        if reference.executor_key != expected_executor:
            raise ValueError("selected operation differs from its capability proof")
        if (
            reference.capability_proof_kind == "direct_copy"
            and reference.locus is not TransformLocus.NONE
        ) or (
            reference.capability_proof_kind == "transform"
            and reference.locus is TransformLocus.NONE
        ):
            raise ValueError("selected operation proof kind and locus disagree")
        base_proof = reference.operation_base_proof
        self._verify_refit_operation_base_proof(base_proof)
        if (
            base_proof.binding_context_digest != reference.binding_context_digest
            or base_proof.binding_identity_digest != reference.binding_identity_digest
            or base_proof.source_storage_lowering_digest
            != reference.source_storage_lowering_digest
            or base_proof.physical_binding_digest != reference.physical_binding_digest
            or base_proof.destination_binding_digest
            != reference.destination_binding_digest
            or base_proof.destination_storage_generation
            != reference.destination_storage_generation
            or base_proof.destination_registry_instance_id
            != reference.destination_registry_instance_id
            or base_proof.destination_registry_generation
            != reference.destination_registry_generation
        ):
            raise ValueError("selected operation differs from its authenticated base")
        if (
            reference.capability_adapter_id != self._adapter_id
            or reference.capability_adapter_version != self._adapter_version
            or reference.capability_issuer_instance_id != self._issuer_instance_id
            or reference.capability_registry_digest != self._registry_digest
        ):
            raise ValueError("selected capability proof issuer identity mismatch")
        _require_sha256_digest(
            reference.selected_operation_digest,
            "selected operation digest",
        )
        signature = _require_text(
            reference.signature,
            "selected operation signature",
        )
        if _HMAC_SHA256_PATTERN.fullmatch(signature) is None:
            raise ValueError(
                "selected operation signature must be a canonical HMAC-SHA256 value"
            )
        expected_signature = (
            "hmac-sha256:"
            + hmac.new(
                self._secret,
                _canonical_json_bytes(
                    _selected_operation_authentication_payload(reference)
                ),
                sha256,
            ).hexdigest()
        )
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("selected operation signature mismatch")
        return reference

    def _verify(
        self,
        proof: DirectCopyCapabilityProof | TransformCapabilityProof,
    ) -> None:
        if type(proof) not in {
            DirectCopyCapabilityProof,
            TransformCapabilityProof,
        }:
            raise TypeError("proof must be an adapter-issued capability proof")
        _validate_adapter_proof_fields(
            proof.source_stage,
            proof.destination_stage,
            proof.route_id,
            proof.source_endpoint_capability,
            proof.destination_endpoint_capability,
            proof.implementation_id,
            proof.implementation_version,
        )
        if type(proof) is TransformCapabilityProof:
            _require_enum(proof.transform_locus, TransformLocus, "transform_locus")
            if proof.transform_locus is TransformLocus.NONE:
                raise ValueError("transform proof cannot authorize direct copy")
        _require_identifier(proof.adapter_id, "proof adapter_id")
        _require_implementation_version(
            proof.adapter_version,
            "proof adapter_version",
        )
        _require_text(proof.issuer_instance_id, "proof issuer_instance_id")
        _require_sha256_digest(proof.registry_digest, "proof registry_digest")
        signature = _require_text(proof.signature, "proof signature")
        if _HMAC_SHA256_PATTERN.fullmatch(signature) is None:
            raise ValueError("proof signature must be a canonical HMAC-SHA256 value")
        if (
            proof.adapter_id != self._adapter_id
            or proof.adapter_version != self._adapter_version
            or proof.issuer_instance_id != self._issuer_instance_id
            or proof.registry_digest != self._registry_digest
        ):
            raise ValueError("capability proof issuer identity mismatch")
        expected_signature = (
            "hmac-sha256:"
            + hmac.new(
                self._secret,
                _canonical_json_bytes(_proof_payload(proof)),
                sha256,
            ).hexdigest()
        )
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("capability proof signature mismatch")
        self._require_registered(_proof_capability(proof))


def _create_adapter_capability_registry(
    *,
    adapter_id: str,
    adapter_version: str,
    capabilities: Sequence[AdapterOperationCapability],
    destination_adapter_instance: object,
) -> AdapterCapabilityRegistry:
    """Create a registry inside trusted version-adapter code.

    This package-private factory must never be exposed through recipe or model
    configuration.  The registry is immutable after construction.
    """
    registry = AdapterCapabilityRegistry._create(
        _ADAPTER_REGISTRY_FACTORY_TOKEN,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        capabilities=capabilities,
    )
    registry._destination_registry._issue_mint_authority(
        registry,
        destination_adapter_instance,
    )
    return registry


def _destination_binding_mint_authority_for_adapter(
    capability_registry: AdapterCapabilityRegistry,
    *,
    adapter_instance: object,
) -> DestinationBindingMintAuthority:
    """Return the atomically issued mint side to its exact adapter instance."""
    if type(capability_registry) is not AdapterCapabilityRegistry:
        raise TypeError("capability_registry must be AdapterCapabilityRegistry")
    authority = capability_registry._destination_registry._mint_authority
    if (
        type(authority) is not DestinationBindingMintAuthority
        or authority._adapter_instance is not adapter_instance
    ):
        raise ValueError("destination mint authority belongs to another adapter")
    return authority


def _require_endpoint_capability(
    realized_format: RealizedBindingFormat,
    route: PhysicalRouteDescriptor,
    proof: EndpointCapabilityProof,
    *,
    source: bool,
) -> None:
    expected_stage = route.source_stage if source else route.destination_stage
    endpoint_label = "source" if source else "destination"
    expected_instance_id = (
        route.source_endpoint_instance_id
        if source
        else route.destination_endpoint_instance_id
    )
    expected_fingerprint = (
        route.source_endpoint_capability_fingerprint
        if source
        else route.destination_endpoint_capability_fingerprint
    )
    if proof.stage is not expected_stage:
        raise ValueError(f"{endpoint_label} endpoint capability stage mismatch")
    if proof.endpoint_instance_id != expected_instance_id:
        raise ValueError(f"{endpoint_label} endpoint instance identity mismatch")
    if proof.capability_fingerprint != expected_fingerprint:
        raise ValueError(f"{endpoint_label} endpoint capability fingerprint mismatch")
    if proof.ordered_representation_digest != physical_representation_digest(
        realized_format,
        expected_stage,
    ):
        raise ValueError(f"{endpoint_label} endpoint ordered representation mismatch")
    if proof.placement_digest != endpoint_placement_digest(
        realized_format,
        expected_stage,
    ):
        raise ValueError(f"{endpoint_label} endpoint placement mismatch")


def _require_proof_route(
    realized_format: RealizedBindingFormat,
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
    proof: DirectCopyCapabilityProof | TransformCapabilityProof,
) -> None:
    if (
        proof.source_stage is not source_stage
        or proof.destination_stage is not destination_stage
    ):
        raise ValueError("capability proof does not bind the exact stage pair")
    route = realized_format.route_between(source_stage, destination_stage)
    if proof.route_id != route.route_id:
        raise ValueError("capability proof route identity mismatch")
    _require_endpoint_capability(
        realized_format,
        route,
        proof.source_endpoint_capability,
        source=True,
    )
    _require_endpoint_capability(
        realized_format,
        route,
        proof.destination_endpoint_capability,
        source=False,
    )


def require_direct_copy(
    realized_format: RealizedBindingFormat,
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
    *,
    capability_registry: AdapterCapabilityRegistry,
    proof: DirectCopyCapabilityProof | None,
) -> DirectCopyCapabilityProof:
    """Verify one copy while compiling an immutable refit plan."""
    if type(realized_format) is not RealizedBindingFormat:
        raise TypeError("realized_format must be RealizedBindingFormat")
    _require_adjacent_stage_pair(source_stage, destination_stage)
    source_components = realized_format.components_at(source_stage)
    destination_components = realized_format.components_at(destination_stage)
    source_representations = tuple(
        component.representation for component in source_components
    )
    destination_representations = tuple(
        component.representation for component in destination_components
    )
    if (
        source_representations != destination_representations
        or realized_format.format_at(source_stage)
        != realized_format.format_at(destination_stage)
    ):
        raise ValueError(
            "direct copy requires complete ordered physical descriptor equality"
        )
    if proof is None:
        raise ValueError("direct copy requires an adapter capability proof")
    if type(proof) is not DirectCopyCapabilityProof:
        raise TypeError("proof must be DirectCopyCapabilityProof or None")
    if type(capability_registry) is not AdapterCapabilityRegistry:
        raise TypeError("capability_registry must be AdapterCapabilityRegistry")
    capability_registry._verify(proof)
    _require_proof_route(
        realized_format,
        source_stage,
        destination_stage,
        proof,
    )
    return proof


def require_transform(
    realized_format: RealizedBindingFormat,
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
    *,
    capability_registry: AdapterCapabilityRegistry,
    transform_locus: TransformLocus,
    proof: TransformCapabilityProof | None,
) -> TransformCapabilityProof:
    """Verify one transform while compiling an immutable refit plan."""
    if type(realized_format) is not RealizedBindingFormat:
        raise TypeError("realized_format must be RealizedBindingFormat")
    _require_adjacent_stage_pair(source_stage, destination_stage)
    _require_enum(transform_locus, TransformLocus, "transform_locus")
    allowed_loci = _ALLOWED_TRANSFORM_LOCI[(source_stage, destination_stage)]
    if transform_locus not in allowed_loci:
        raise ValueError("transform locus is invalid for the exact adjacent stage pair")
    if proof is None:
        raise ValueError("transform requires an adapter transform capability proof")
    if type(proof) is not TransformCapabilityProof:
        raise TypeError("proof must be TransformCapabilityProof or None")
    if type(capability_registry) is not AdapterCapabilityRegistry:
        raise TypeError("capability_registry must be AdapterCapabilityRegistry")
    if proof.transform_locus is not transform_locus:
        raise ValueError("transform capability proof locus mismatch")
    capability_registry._verify(proof)
    _require_proof_route(
        realized_format,
        source_stage,
        destination_stage,
        proof,
    )
    return proof


def _issue_authenticated_selected_operation(
    *,
    validated_context: RefitBindingContext,
    realized_format: RealizedBindingFormat,
    destination_binding_proof: DestinationBindingProof,
    capability_proof: DirectCopyCapabilityProof | TransformCapabilityProof,
    capability_registry: AdapterCapabilityRegistry,
    validated_source_storage_lowering_digest: str | None,
    validated_physical_binding_digest: str | None,
) -> SelectedRefitOperation:
    binding_identity = RefitBindingIdentity._issue(
        _REFIT_PLANNER_FACTORY_TOKEN,
        binding_context=validated_context,
        realized_format=realized_format,
        destination_binding_proof=destination_binding_proof,
        validated_source_storage_lowering_digest=(
            validated_source_storage_lowering_digest
        ),
        validated_physical_binding_digest=validated_physical_binding_digest,
    )
    operation_base_proof = capability_registry._issue_refit_operation_base_proof(
        _REFIT_PLANNER_FACTORY_TOKEN,
        binding_identity,
        validated_context,
        destination_binding_proof,
    )
    selected = SelectedRefitOperation._issue(
        _REFIT_PLANNER_FACTORY_TOKEN,
        binding_identity=binding_identity,
        binding_context=deepcopy(validated_context),
        realized_format=realized_format,
        destination_binding_proof=destination_binding_proof,
        capability_proof=capability_proof,
        operation_base_proof=operation_base_proof,
    )
    capability_registry._authenticate_selected_operation(
        _REFIT_PLANNER_FACTORY_TOKEN,
        selected,
    )
    return selected


def _select_transform_impl(
    realized_format: RealizedBindingFormat,
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
    *,
    binding_context: InstalledRefitBindingContext,
    capability_registry: AdapterCapabilityRegistry,
    destination_binding_proof: DestinationBindingProof,
    transform_locus: TransformLocus,
    direct_copy_proof: DirectCopyCapabilityProof | None = None,
    transform_capability_proof: TransformCapabilityProof | None = None,
    validated_source_storage_lowering_digest: str | None = None,
    validated_physical_binding_digest: str | None = None,
    validated_required_extraction: object = _UNVALIDATED_SOURCE_EXTRACTION,
) -> SelectedRefitOperation:
    """Compile one fully bound operation after capability verification."""
    if type(realized_format) is not RealizedBindingFormat:
        raise TypeError("realized_format must be RealizedBindingFormat")
    if type(capability_registry) is not AdapterCapabilityRegistry:
        raise TypeError("capability_registry must be AdapterCapabilityRegistry")
    validated_context = _require_installed_binding_context(binding_context)
    capability_registry._verify_destination_binding(
        destination_binding_proof,
        realized_format,
    )
    _require_adjacent_stage_pair(source_stage, destination_stage)
    _require_enum(transform_locus, TransformLocus, "transform_locus")
    if validated_required_extraction is _UNVALIDATED_SOURCE_EXTRACTION:
        partial_source_region = (
            source_stage is PhysicalFormatStage.SOURCE_STORAGE
            and destination_stage is PhysicalFormatStage.WIRE
            and any(
                not _source_region_is_complete(source_slice.source_region)
                for source_slice in validated_context.source_binding_slices
            )
        )
        required_extraction: SourceRegionExtractionCapability | None = None
        if partial_source_region:
            if transform_locus is not TransformLocus.SOURCE:
                raise ValueError(
                    "partial source region requires an authenticated SOURCE gather/split transform"
                )
            required_extraction = derive_source_region_extraction_capability(
                binding_context,
                realized_format,
            )
    elif validated_required_extraction is None:
        required_extraction = None
    else:
        required_extraction = _validate_source_region_extraction_capability(
            cast(SourceRegionExtractionCapability, validated_required_extraction)
        )
    if direct_copy_proof is not None:
        if transform_capability_proof is not None:
            raise ValueError(
                "direct-copy and transform capability proofs are mutually exclusive"
            )
        if transform_locus is not TransformLocus.NONE:
            raise ValueError(
                "a direct-copy proof cannot be combined with a second transform"
            )
        validated_proof = require_direct_copy(
            realized_format,
            source_stage,
            destination_stage,
            capability_registry=capability_registry,
            proof=direct_copy_proof,
        )
        return _issue_authenticated_selected_operation(
            validated_context=validated_context,
            realized_format=realized_format,
            destination_binding_proof=destination_binding_proof,
            capability_proof=validated_proof,
            capability_registry=capability_registry,
            validated_source_storage_lowering_digest=(
                validated_source_storage_lowering_digest
            ),
            validated_physical_binding_digest=validated_physical_binding_digest,
        )
    if transform_locus is TransformLocus.NONE:
        if transform_capability_proof is not None:
            raise ValueError(
                "a transform capability proof cannot authorize direct copy"
            )
        require_direct_copy(
            realized_format,
            source_stage,
            destination_stage,
            capability_registry=capability_registry,
            proof=None,
        )
    validated_proof = require_transform(
        realized_format,
        source_stage,
        destination_stage,
        capability_registry=capability_registry,
        transform_locus=transform_locus,
        proof=transform_capability_proof,
    )
    if (
        source_stage is PhysicalFormatStage.SOURCE_STORAGE
        and destination_stage is PhysicalFormatStage.WIRE
        and validated_proof.source_region_extraction != required_extraction
    ):
        raise ValueError(
            "SOURCE transform proof does not authorize the exact source region extraction"
        )
    return _issue_authenticated_selected_operation(
        validated_context=validated_context,
        realized_format=realized_format,
        destination_binding_proof=destination_binding_proof,
        capability_proof=validated_proof,
        capability_registry=capability_registry,
        validated_source_storage_lowering_digest=(
            validated_source_storage_lowering_digest
        ),
        validated_physical_binding_digest=validated_physical_binding_digest,
    )


def select_transform(
    realized_format: RealizedBindingFormat,
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
    *,
    binding_context: InstalledRefitBindingContext,
    capability_registry: AdapterCapabilityRegistry,
    destination_binding_proof: DestinationBindingProof,
    transform_locus: TransformLocus,
    direct_copy_proof: DirectCopyCapabilityProof | None = None,
    transform_capability_proof: TransformCapabilityProof | None = None,
) -> SelectedRefitOperation:
    """Compile one fully bound operation after capability verification."""
    return _select_transform_impl(
        realized_format,
        source_stage,
        destination_stage,
        binding_context=binding_context,
        capability_registry=capability_registry,
        destination_binding_proof=destination_binding_proof,
        transform_locus=transform_locus,
        direct_copy_proof=direct_copy_proof,
        transform_capability_proof=transform_capability_proof,
    )


@dataclass(frozen=True, slots=True, init=False)
class InstalledSelectedRefitOperation:
    """Process-local trusted operation handle for O(1) hot dispatch."""

    _installation_registry: RefitPlanInstallationRegistry
    _issuance_nonce: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "InstalledSelectedRefitOperation must be produced by plan installation"
        )

    @classmethod
    def _issue(
        cls,
        factory_token: object,
        *,
        registry: RefitPlanInstallationRegistry,
    ) -> Self:
        if factory_token is not _REFIT_PLANNER_FACTORY_TOKEN:
            raise TypeError("invalid refit operation installation authority")
        installed = object.__new__(cls)
        object.__setattr__(installed, "_installation_registry", registry)
        object.__setattr__(installed, "_issuance_nonce", object())
        return installed

    def __reduce__(self) -> object:
        raise TypeError(
            "installed operations are process-local; serialize selected_operation and reinstall"
        )


def install_selected_refit_operations(
    *,
    selected_operations: Sequence[SelectedRefitOperation],
    intents: CompiledPrecisionIntentGroup,
    active_selection: CompiledPrecisionSelectionGroup,
    active_request: RuntimeSourceDiscoveryRequest,
    active_results: tuple[RuntimeSourceDiscoveryResult, ...],
    capability_registry: AdapterCapabilityRegistry,
    installation_registry: RefitPlanInstallationRegistry,
) -> tuple[InstalledSelectedRefitOperation, ...]:
    """Install a batch in O(A + E) from authenticated structural bases.

    ``A`` is the total size of distinct context/physical bases and ``E`` is the
    number of operation envelopes.  A base's embedded context and format are
    rederived once from active artifacts.  Later envelopes bearing the same
    adapter HMAC are treated only as compact references to that registry-owned
    canonical base; their repeated A-sized bodies are never executed or
    exported.  Pickle transport is supported within the issuing adapter
    registry's process lifetime.  A separately created registry has a fresh
    issuer identity and secret and therefore rejects the transported plan.
    """
    captured = _snapshot_sequence(selected_operations, "selected_operations")
    if not captured:
        raise ValueError("selected_operations must be non-empty")
    if any(type(selected) is not SelectedRefitOperation for selected in captured):
        raise TypeError(
            "selected_operations must contain exact SelectedRefitOperation envelopes"
        )
    if type(capability_registry) is not AdapterCapabilityRegistry:
        raise TypeError("capability_registry must be AdapterCapabilityRegistry")
    if type(installation_registry) is not RefitPlanInstallationRegistry:
        raise TypeError("installation_registry must be RefitPlanInstallationRegistry")
    context_representatives: dict[str, RefitBindingContext] = {}
    base_representatives: dict[
        str,
        tuple[
            SelectedRefitOperation,
            _AuthenticatedSelectedOperationReference,
        ],
    ] = {}
    authenticated_references: list[_AuthenticatedSelectedOperationReference] = []
    for selected in captured:
        reference = capability_registry._verify_selected_operation_authentication(
            selected
        )
        authenticated_references.append(reference)
        if reference.operation_base_proof.base_digest in base_representatives:
            continue
        representative = deepcopy(selected)
        representative_reference = (
            capability_registry._verify_selected_operation_authentication(
                representative
            )
        )
        if representative_reference != reference:
            raise ValueError("selected operation changed during authenticated capture")
        base_representatives[reference.operation_base_proof.base_digest] = (
            representative,
            representative_reference,
        )
        context_representatives.setdefault(
            reference.binding_context_digest,
            representative.binding_context,
        )
    installed_unique_contexts = install_refit_contexts(
        contexts=tuple(context_representatives.values()),
        intents=intents,
        active_selection=active_selection,
        active_request=active_request,
        active_results=active_results,
        installation_registry=installation_registry,
    )
    installed_context_by_digest = dict(
        zip(
            context_representatives,
            installed_unique_contexts,
            strict=True,
        )
    )
    installed_operations: list[InstalledSelectedRefitOperation] = []
    for reference in authenticated_references:
        installed_context = installed_context_by_digest[
            reference.binding_context_digest
        ]
        installed_base = installation_registry._resolve_operation_base(
            reference.operation_base_proof,
            installed_context=installed_context,
            destination_registry=capability_registry._destination_registry,
        )
        if installed_base is None:
            selected, representative_reference = base_representatives[
                reference.operation_base_proof.base_digest
            ]
            proof = selected.capability_proof
            expected = _select_transform_impl(
                selected.realized_format,
                selected.source_stage,
                selected.destination_stage,
                binding_context=installed_context,
                capability_registry=capability_registry,
                destination_binding_proof=selected.destination_binding_proof,
                transform_locus=selected.locus,
                direct_copy_proof=(
                    proof if type(proof) is DirectCopyCapabilityProof else None
                ),
                transform_capability_proof=(
                    proof if type(proof) is TransformCapabilityProof else None
                ),
            )
            expected_reference = (
                capability_registry._verify_selected_operation_authentication(expected)
            )
            if (
                representative_reference != expected_reference
                or _selected_operation_payload(selected)
                != _selected_operation_payload(expected)
            ):
                raise ValueError(
                    "selected refit operation differs from its active canonical derivation"
                )
            installed_base = installation_registry._issue_operation_base(
                _REFIT_PLANNER_FACTORY_TOKEN,
                expected.operation_base_proof,
                installed_context=installed_context,
                destination_binding_proof=expected.destination_binding_proof,
                destination_registry=capability_registry._destination_registry,
            )
        installed_operations.append(
            installation_registry._issue_operation(
                _REFIT_PLANNER_FACTORY_TOKEN,
                reference,
                installed_context=installed_context,
                installed_base=installed_base,
                destination_registry=capability_registry._destination_registry,
            )
        )
    return tuple(installed_operations)


def install_selected_refit_operation(
    *,
    selected_operation: SelectedRefitOperation,
    intents: CompiledPrecisionIntentGroup,
    active_selection: CompiledPrecisionSelectionGroup,
    active_request: RuntimeSourceDiscoveryRequest,
    active_results: tuple[RuntimeSourceDiscoveryResult, ...],
    capability_registry: AdapterCapabilityRegistry,
    installation_registry: RefitPlanInstallationRegistry,
) -> InstalledSelectedRefitOperation:
    """Install one serialized operation before hot dispatch."""
    return install_selected_refit_operations(
        selected_operations=(selected_operation,),
        intents=intents,
        active_selection=active_selection,
        active_request=active_request,
        active_results=active_results,
        capability_registry=capability_registry,
        installation_registry=installation_registry,
    )[0]


def execution_dispatch_key(
    selected_operation: InstalledSelectedRefitOperation,
) -> RefitExecutorKey:
    """Return the executor key from a one-time-validated process-local handle."""
    if type(selected_operation) is not InstalledSelectedRefitOperation:
        raise TypeError("execution dispatch requires InstalledSelectedRefitOperation")
    if (
        type(selected_operation._installation_registry)
        is not RefitPlanInstallationRegistry
    ):
        raise ValueError("installed operation registry mismatch")
    return selected_operation._installation_registry._resolve_operation(
        selected_operation
    ).executor_key


__all__ = [
    "AdapterCapabilityRegistry",
    "AdapterOperationCapability",
    "DirectCopyCapabilityProof",
    "DestinationBindingEvidence",
    "DestinationBindingProof",
    "EndpointCapabilityProof",
    "EndpointPlacement",
    "FormatSchemaRegistry",
    "InstalledRefitBindingContext",
    "InstalledSelectedRefitOperation",
    "PhysicalAxisMapping",
    "PhysicalComponentDescriptor",
    "PhysicalFormatStage",
    "PhysicalLayoutDescriptor",
    "PhysicalPadding",
    "PhysicalPaddingSemantics",
    "PhysicalPermutation",
    "PhysicalRepresentation",
    "PhysicalRouteDescriptor",
    "RealizedBindingFormat",
    "RefitBindingContext",
    "RefitBindingIdentity",
    "RefitExecutorKey",
    "RefitPlanInstallationRegistry",
    "SelectedRefitOperation",
    "SourceRegionExtractionCapability",
    "SourceRegionTransformKind",
    "TransformLocus",
    "TransformCapabilityProof",
    "bind_refit_context",
    "bind_refit_contexts",
    "derive_source_region_extraction_capability",
    "derive_source_storage_layout",
    "endpoint_placement_digest",
    "execution_dispatch_key",
    "install_refit_context",
    "install_refit_contexts",
    "install_selected_refit_operation",
    "install_selected_refit_operations",
    "physical_representation_digest",
    "require_direct_copy",
    "require_transform",
    "select_transform",
]
