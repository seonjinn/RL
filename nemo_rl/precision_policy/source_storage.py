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

"""Framework-free native source-storage realization contracts.

The records in this module describe how a producer-normalized metadata view can
be obtained from native carrier storage.  They do not materialize that view,
describe endpoint placement, or authorize a direct transfer.

Contract evolution is anchored by ``SourceNormalizerManifest.schema_version``
and versioned normalizer, permutation, and swizzle identities.  The inventory
intentionally has no redundant version field that could disagree with those
producer-fingerprint-bound authorities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from math import prod
import re
from typing import TypeVar

from nemo_rl.precision_policy.source_dtype import CanonicalSourceDType


_SHA256_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_VERSIONED_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+\.v[1-9][0-9]*")
_AXIS_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_SCALAR_OR_BUFFER_SEQUENCE_TYPES = (str, bytes, bytearray, memoryview)
_SequenceItemT = TypeVar("_SequenceItemT")

IDENTITY_PERMUTATION_ID = "identity.permutation.v1"
IDENTITY_SWIZZLE_ID = "identity.swizzle.v1"


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be exact non-empty text")
    return value


def _require_versioned_id(value: object, name: str) -> str:
    text = _require_text(value, name)
    if _VERSIONED_ID_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase versioned identifier")
    return text


def _require_sha256_digest(value: object, name: str) -> str:
    text = _require_text(value, name)
    if _SHA256_DIGEST_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a canonical SHA-256 digest")
    return text


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _snapshot_sequence(
    value: Sequence[_SequenceItemT],
    name: str,
) -> tuple[_SequenceItemT, ...]:
    if isinstance(value, _SCALAR_OR_BUFFER_SEQUENCE_TYPES) or not isinstance(
        value, Sequence
    ):
        raise TypeError(f"{name} must be a non-scalar sequence")
    return tuple(value)


def _snapshot_shape(value: Sequence[int], name: str) -> tuple[int, ...]:
    shape = _snapshot_sequence(value, name)
    if any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0
        for extent in shape
    ):
        raise ValueError(f"{name} extents must be positive integers")
    return shape


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_digest(payload: object) -> str:
    return f"sha256:{sha256(_canonical_json_bytes(payload)).hexdigest()}"


class SourceExtentRounding(StrEnum):
    """Division rule used before physical-axis alignment."""

    EXACT = "exact"
    CEIL = "ceil"


@dataclass(frozen=True, slots=True)
class SourceNormalizedAxisExtent:
    """Extent derived from a product of normalized-view axes."""

    normalized_axis_indices: tuple[int, ...]
    divisor: int
    rounding: SourceExtentRounding
    alignment: int

    def __post_init__(self) -> None:
        indices = _snapshot_sequence(
            self.normalized_axis_indices,
            "normalized axis indices",
        )
        if not indices:
            raise ValueError("normalized axis indices must be non-empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise ValueError("normalized axis indices must be non-negative integers")
        if len(indices) != len(set(indices)):
            raise ValueError("normalized axis indices must be duplicate-free")
        if not isinstance(self.rounding, SourceExtentRounding):
            raise TypeError("source extent rounding must be SourceExtentRounding")
        _require_positive_int(self.divisor, "source extent divisor")
        _require_positive_int(self.alignment, "source extent alignment")
        object.__setattr__(self, "normalized_axis_indices", indices)

    def unaligned_extent(self, normalized_shape: Sequence[int]) -> int:
        """Resolve the divided extent before alignment."""
        shape = _snapshot_shape(normalized_shape, "normalized output shape")
        if any(index >= len(shape) for index in self.normalized_axis_indices):
            raise ValueError("physical axis references an unknown normalized axis")
        numerator = prod(shape[index] for index in self.normalized_axis_indices)
        quotient, remainder = divmod(numerator, self.divisor)
        if self.rounding is SourceExtentRounding.EXACT:
            if remainder:
                raise ValueError("physical axis exact division has a remainder")
            return quotient
        return quotient + int(remainder != 0)

    def resolve(self, normalized_shape: Sequence[int]) -> int:
        """Resolve the aligned physical extent."""
        extent = self.unaligned_extent(normalized_shape)
        return ((extent + self.alignment - 1) // self.alignment) * self.alignment

    def has_padding(self, normalized_shape: Sequence[int]) -> bool:
        """Return whether division or alignment creates unused carrier capacity."""
        shape = _snapshot_shape(normalized_shape, "normalized output shape")
        if any(index >= len(shape) for index in self.normalized_axis_indices):
            raise ValueError("physical axis references an unknown normalized axis")
        numerator = prod(shape[index] for index in self.normalized_axis_indices)
        quotient, remainder = divmod(numerator, self.divisor)
        if self.rounding is SourceExtentRounding.EXACT:
            if remainder:
                raise ValueError("physical axis exact division has a remainder")
        else:
            quotient += int(remainder != 0)
        return bool(remainder) or quotient % self.alignment != 0


@dataclass(frozen=True, slots=True)
class SourceLiteralAxisExtent:
    """Positive physical extent independent from normalized-view axes."""

    extent: int

    def __post_init__(self) -> None:
        _require_positive_int(self.extent, "literal physical extent")

    def resolve(self, normalized_shape: Sequence[int]) -> int:
        """Return the literal extent after validating the source shape."""
        _snapshot_shape(normalized_shape, "normalized output shape")
        return self.extent


type SourceAxisExtent = SourceNormalizedAxisExtent | SourceLiteralAxisExtent


@dataclass(frozen=True, slots=True)
class SourcePhysicalAxisSpec:
    """One ordered physical axis and its exact extent formula."""

    axis_name: str
    extent: SourceAxisExtent

    def __post_init__(self) -> None:
        axis_name = _require_text(self.axis_name, "physical axis name")
        if _AXIS_NAME_PATTERN.fullmatch(axis_name) is None:
            raise ValueError("physical axis name must be a lowercase atom")
        if not isinstance(
            self.extent,
            (SourceNormalizedAxisExtent, SourceLiteralAxisExtent),
        ):
            raise TypeError("physical axis extent has an unsupported formula")

    def resolve(self, normalized_shape: Sequence[int]) -> int:
        """Resolve this axis for one normalized output shape."""
        return self.extent.resolve(normalized_shape)


class SourcePaddingSemantics(StrEnum):
    """Observable contract for bytes outside the normalized logical view."""

    NO_PADDING = "no_padding"
    ZERO_FILLED = "zero_filled"
    UNSPECIFIED_IGNORED = "unspecified_ignored"


class SourceNormalizationKind(StrEnum):
    """Coarse operation class of an evidence-bound normalizer capability."""

    IDENTITY = "identity"
    REINTERPRET = "reinterpret"
    CROP = "crop"
    UNFLATTEN = "unflatten"
    UNSWIZZLE = "unswizzle"
    REPACK = "repack"
    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"
    COMPOSITE = "composite"
    BACKEND_DERIVATION = "backend_derivation"


@dataclass(frozen=True, slots=True, order=True)
class SourceNormalizationContract:
    """One versioned normalizer capability admitted by a producer manifest."""

    capability_id: str
    kind: SourceNormalizationKind
    contract_digest: str

    def __post_init__(self) -> None:
        _require_versioned_id(
            self.capability_id,
            "normalization capability ID",
        )
        if not isinstance(self.kind, SourceNormalizationKind):
            raise TypeError("normalization kind must be SourceNormalizationKind")
        _require_sha256_digest(
            self.contract_digest,
            "normalization contract digest",
        )


def _normalization_contract_payload(
    contract: SourceNormalizationContract,
) -> dict[str, object]:
    return {
        "type": "source_normalization_contract",
        "capability_id": contract.capability_id,
        "kind": contract.kind.value,
        "contract_digest": contract.contract_digest,
    }


@dataclass(frozen=True, slots=True)
class SourceNormalizerManifest:
    """Canonical allowed-normalizer set committed by a producer fingerprint."""

    schema_version: int
    contracts: tuple[SourceNormalizationContract, ...]

    def __post_init__(self) -> None:
        _require_positive_int(self.schema_version, "normalizer manifest schema_version")
        contracts = _snapshot_sequence(
            self.contracts,
            "normalizer manifest contracts",
        )
        if not contracts:
            raise ValueError("normalizer manifest must contain at least one contract")
        if any(
            not isinstance(contract, SourceNormalizationContract)
            for contract in contracts
        ):
            raise TypeError(
                "normalizer manifest contracts must be SourceNormalizationContract"
            )
        capability_ids = tuple(contract.capability_id for contract in contracts)
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("normalizer manifest contains a duplicate capability ID")
        object.__setattr__(
            self,
            "contracts",
            tuple(sorted(contracts, key=lambda contract: contract.capability_id)),
        )


def _normalizer_manifest_payload(
    manifest: SourceNormalizerManifest,
) -> dict[str, object]:
    return {
        "type": "source_normalizer_manifest",
        "schema_version": manifest.schema_version,
        "contracts": [
            _normalization_contract_payload(contract) for contract in manifest.contracts
        ],
    }


def source_normalizer_manifest_digest(manifest: SourceNormalizerManifest) -> str:
    """Return the canonical content digest committed by a producer fingerprint."""
    if not isinstance(manifest, SourceNormalizerManifest):
        raise TypeError("manifest must be SourceNormalizerManifest")
    return _canonical_digest(_normalizer_manifest_payload(manifest))


@dataclass(frozen=True, slots=True)
class SourceStorageComponent:
    """One exact native carrier component of a storage realization."""

    graph_instance_id: str
    native_component_id: str
    source_native_name: str
    component_role: str
    carrier_dtype: CanonicalSourceDType
    physical_shape: tuple[int, ...]
    physical_axes: tuple[SourcePhysicalAxisSpec, ...]
    storage_encoding: str
    padding_semantics: SourcePaddingSemantics
    padding_fill_encoding: str | None
    permutation_id: str
    swizzle_id: str

    def __post_init__(self) -> None:
        _require_text(self.graph_instance_id, "storage component graph_instance_id")
        _require_text(self.native_component_id, "native component ID")
        _require_text(self.source_native_name, "storage component native name")
        _require_text(self.component_role, "storage component role")
        if not isinstance(self.carrier_dtype, CanonicalSourceDType):
            raise TypeError("carrier dtype must be CanonicalSourceDType")
        physical_shape = _snapshot_shape(self.physical_shape, "physical shape")
        physical_axes = _snapshot_sequence(self.physical_axes, "physical axes")
        if any(not isinstance(axis, SourcePhysicalAxisSpec) for axis in physical_axes):
            raise TypeError("physical axes must contain SourcePhysicalAxisSpec values")
        if len(physical_axes) != len(physical_shape):
            raise ValueError("physical axes must exactly describe the physical shape")
        axis_names = tuple(axis.axis_name for axis in physical_axes)
        if len(axis_names) != len(set(axis_names)):
            raise ValueError("physical axis names must be duplicate-free")
        _require_text(self.storage_encoding, "source storage encoding")
        if not isinstance(self.padding_semantics, SourcePaddingSemantics):
            raise TypeError("padding semantics must be SourcePaddingSemantics")
        if self.padding_semantics is SourcePaddingSemantics.ZERO_FILLED:
            if self.padding_fill_encoding is None:
                raise ValueError("zero-filled padding requires a fill encoding")
            _require_text(self.padding_fill_encoding, "padding fill encoding")
        elif self.padding_fill_encoding is not None:
            raise ValueError(
                "padding fill encoding is permitted only for zero-filled padding"
            )
        _require_versioned_id(self.permutation_id, "source permutation ID")
        _require_versioned_id(self.swizzle_id, "source swizzle ID")
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "physical_axes", physical_axes)


def _axis_extent_payload(extent: SourceAxisExtent) -> dict[str, object]:
    if isinstance(extent, SourceNormalizedAxisExtent):
        return {
            "kind": "normalized_axis_product",
            "normalized_axis_indices": list(extent.normalized_axis_indices),
            "divisor": extent.divisor,
            "rounding": extent.rounding.value,
            "alignment": extent.alignment,
        }
    return {"kind": "literal", "extent": extent.extent}


def _storage_component_payload(
    component: SourceStorageComponent,
    *,
    include_identity: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "source_storage_component",
        "component_role": component.component_role,
        "carrier_dtype": component.carrier_dtype.value,
        "physical_shape": list(component.physical_shape),
        "physical_axes": [
            {
                "axis_name": axis.axis_name,
                "extent": _axis_extent_payload(axis.extent),
            }
            for axis in component.physical_axes
        ],
        "storage_encoding": component.storage_encoding,
        "padding_semantics": component.padding_semantics.value,
        "padding_fill_encoding": component.padding_fill_encoding,
        "permutation_id": component.permutation_id,
        "swizzle_id": component.swizzle_id,
    }
    if include_identity:
        payload.update(
            {
                "graph_instance_id": component.graph_instance_id,
                "native_component_id": component.native_component_id,
                "source_native_name": component.source_native_name,
            }
        )
    return payload


def _resolved_component_shape(
    component: SourceStorageComponent,
    normalized_shape: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(axis.resolve(normalized_shape) for axis in component.physical_axes)


def _component_has_padding(
    component: SourceStorageComponent,
    normalized_shape: tuple[int, ...],
) -> bool:
    return any(
        isinstance(axis.extent, SourceNormalizedAxisExtent)
        and axis.extent.has_padding(normalized_shape)
        for axis in component.physical_axes
    )


def _component_is_identity_view(
    component: SourceStorageComponent,
    *,
    output_dtype: CanonicalSourceDType,
    output_shape: tuple[int, ...],
    output_numeric_encoding: str,
) -> bool:
    if (
        component.carrier_dtype != output_dtype
        or component.physical_shape != output_shape
        or component.storage_encoding != output_numeric_encoding
        or component.padding_semantics is not SourcePaddingSemantics.NO_PADDING
        or component.padding_fill_encoding is not None
        or component.permutation_id != IDENTITY_PERMUTATION_ID
        or component.swizzle_id != IDENTITY_SWIZZLE_ID
        or len(component.physical_axes) != len(output_shape)
    ):
        return False
    return all(
        isinstance(axis.extent, SourceNormalizedAxisExtent)
        and axis.extent.normalized_axis_indices == (axis_index,)
        and axis.extent.divisor == 1
        and axis.extent.rounding is SourceExtentRounding.EXACT
        and axis.extent.alignment == 1
        for axis_index, axis in enumerate(component.physical_axes)
    )


def _validate_output_facts(
    *,
    output_dtype: CanonicalSourceDType,
    output_shape: Sequence[int],
    output_numeric_encoding: str,
) -> tuple[int, ...]:
    if not isinstance(output_dtype, CanonicalSourceDType):
        raise TypeError("normalized output dtype must be CanonicalSourceDType")
    shape = _snapshot_shape(output_shape, "normalized output shape")
    _require_text(output_numeric_encoding, "normalized output numeric encoding")
    return shape


@dataclass(frozen=True, slots=True)
class SourceStorageRealization:
    """Native carrier storage and its exact normalized output contract."""

    realization_id: str
    graph_instance_id: str
    output_record_id: str
    components: tuple[SourceStorageComponent, ...]
    output_dtype: CanonicalSourceDType
    output_shape: tuple[int, ...]
    output_numeric_encoding: str
    normalization: SourceNormalizationContract

    def __post_init__(self) -> None:
        _require_text(self.realization_id, "source storage realization ID")
        _require_text(self.graph_instance_id, "realization graph_instance_id")
        _require_text(self.output_record_id, "realization output_record_id")
        components = _snapshot_sequence(self.components, "storage components")
        if not components:
            raise ValueError("storage realization must contain raw components")
        if any(not isinstance(item, SourceStorageComponent) for item in components):
            raise TypeError("storage realization components have an unsupported type")
        if any(item.graph_instance_id != self.graph_instance_id for item in components):
            raise ValueError("storage component belongs to another graph")
        component_ids = tuple(item.native_component_id for item in components)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("storage realization contains duplicate component IDs")
        output_shape = _validate_output_facts(
            output_dtype=self.output_dtype,
            output_shape=self.output_shape,
            output_numeric_encoding=self.output_numeric_encoding,
        )
        if not isinstance(self.normalization, SourceNormalizationContract):
            raise TypeError("normalization must be SourceNormalizationContract")
        if self.normalization.kind is SourceNormalizationKind.BACKEND_DERIVATION:
            raise ValueError("raw storage cannot use a backend-derivation contract")
        for component in components:
            if (
                _resolved_component_shape(component, output_shape)
                != component.physical_shape
            ):
                raise ValueError(
                    "physical shape does not match the normalized-axis formulas"
                )
            has_padding = _component_has_padding(component, output_shape)
            if has_padding == (
                component.padding_semantics is SourcePaddingSemantics.NO_PADDING
            ):
                raise ValueError(
                    "padding semantics do not match the resolved physical extent"
                )
        if self.normalization.kind is SourceNormalizationKind.IDENTITY and (
            len(components) != 1
            or not _component_is_identity_view(
                components[0],
                output_dtype=self.output_dtype,
                output_shape=output_shape,
                output_numeric_encoding=self.output_numeric_encoding,
            )
        ):
            raise ValueError(
                "identity normalization requires one exact native/view representation"
            )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "output_shape", output_shape)


@dataclass(frozen=True, slots=True)
class SourceDerivedRealization:
    """Zero-raw-component witness for a backend-derived normalized record."""

    realization_id: str
    graph_instance_id: str
    output_record_id: str
    output_dtype: CanonicalSourceDType
    output_shape: tuple[int, ...]
    output_numeric_encoding: str
    derivation: SourceNormalizationContract

    def __post_init__(self) -> None:
        _require_text(self.realization_id, "source derived realization ID")
        _require_text(self.graph_instance_id, "derived graph_instance_id")
        _require_text(self.output_record_id, "derived output_record_id")
        output_shape = _validate_output_facts(
            output_dtype=self.output_dtype,
            output_shape=self.output_shape,
            output_numeric_encoding=self.output_numeric_encoding,
        )
        if not isinstance(self.derivation, SourceNormalizationContract):
            raise TypeError("derivation must be SourceNormalizationContract")
        if self.derivation.kind is not SourceNormalizationKind.BACKEND_DERIVATION:
            raise ValueError(
                "source derived realization requires a backend-derivation contract"
            )
        object.__setattr__(self, "output_shape", output_shape)


type SourceRealization = SourceStorageRealization | SourceDerivedRealization


def _realization_normalization(
    realization: SourceRealization,
) -> SourceNormalizationContract:
    if isinstance(realization, SourceStorageRealization):
        return realization.normalization
    return realization.derivation


def _realization_output_facts(
    realization: SourceRealization,
) -> tuple[CanonicalSourceDType, tuple[int, ...], str]:
    return (
        realization.output_dtype,
        realization.output_shape,
        realization.output_numeric_encoding,
    )


@dataclass(frozen=True, slots=True)
class SourceStorageRealizationInventory:
    """Canonical graph-scoped witness set versioned by its bound manifest."""

    graph_instance_id: str
    normalizer_manifest: SourceNormalizerManifest
    realizations: tuple[SourceRealization, ...]

    def __post_init__(self) -> None:
        _require_text(
            self.graph_instance_id,
            "source realization inventory graph_instance_id",
        )
        if not isinstance(self.normalizer_manifest, SourceNormalizerManifest):
            raise TypeError("normalizer_manifest must be SourceNormalizerManifest")
        realizations = _snapshot_sequence(self.realizations, "source realizations")
        if any(
            not isinstance(item, (SourceStorageRealization, SourceDerivedRealization))
            for item in realizations
        ):
            raise TypeError("source realization inventory has an unsupported witness")
        if any(
            item.graph_instance_id != self.graph_instance_id for item in realizations
        ):
            raise ValueError("source realization belongs to another graph")
        realization_ids = tuple(item.realization_id for item in realizations)
        if len(realization_ids) != len(set(realization_ids)):
            raise ValueError("source realization IDs must be duplicate-free")
        allowed = frozenset(self.normalizer_manifest.contracts)
        if any(
            _realization_normalization(item) not in allowed for item in realizations
        ):
            raise ValueError(
                "source realization normalization contract is absent from the manifest"
            )
        output_facts: dict[
            str,
            tuple[CanonicalSourceDType, tuple[int, ...], str],
        ] = {}
        component_by_id: dict[str, SourceStorageComponent] = {}
        for realization in realizations:
            facts = _realization_output_facts(realization)
            prior_facts = output_facts.setdefault(realization.output_record_id, facts)
            if prior_facts != facts:
                raise ValueError(
                    "alternative realizations for one record disagree on output facts"
                )
            if not isinstance(realization, SourceStorageRealization):
                continue
            for component in realization.components:
                prior_component = component_by_id.setdefault(
                    component.native_component_id,
                    component,
                )
                if prior_component != component:
                    raise ValueError(
                        "one native component ID has conflicting storage metadata"
                    )
        object.__setattr__(
            self,
            "realizations",
            tuple(
                sorted(
                    realizations,
                    key=lambda item: (item.output_record_id, item.realization_id),
                )
            ),
        )


def _validated_extent_copy(extent: object) -> SourceAxisExtent:
    if isinstance(extent, SourceNormalizedAxisExtent):
        return SourceNormalizedAxisExtent(
            normalized_axis_indices=extent.normalized_axis_indices,
            divisor=extent.divisor,
            rounding=extent.rounding,
            alignment=extent.alignment,
        )
    if isinstance(extent, SourceLiteralAxisExtent):
        return SourceLiteralAxisExtent(extent=extent.extent)
    raise TypeError("physical axis extent has an unsupported formula")


def _validated_axis_copy(axis: object) -> SourcePhysicalAxisSpec:
    if not isinstance(axis, SourcePhysicalAxisSpec):
        raise TypeError("physical axes must contain SourcePhysicalAxisSpec values")
    return SourcePhysicalAxisSpec(
        axis_name=axis.axis_name,
        extent=_validated_extent_copy(axis.extent),
    )


def _validated_contract_copy(contract: object) -> SourceNormalizationContract:
    if not isinstance(contract, SourceNormalizationContract):
        raise TypeError("normalization contract has an unsupported type")
    return SourceNormalizationContract(
        capability_id=contract.capability_id,
        kind=contract.kind,
        contract_digest=contract.contract_digest,
    )


def _validated_manifest_copy(manifest: object) -> SourceNormalizerManifest:
    if not isinstance(manifest, SourceNormalizerManifest):
        raise TypeError("normalizer_manifest must be SourceNormalizerManifest")
    contracts = _snapshot_sequence(
        manifest.contracts,
        "normalizer manifest contracts",
    )
    return SourceNormalizerManifest(
        schema_version=manifest.schema_version,
        contracts=tuple(_validated_contract_copy(contract) for contract in contracts),
    )


def _validated_component_copy(component: object) -> SourceStorageComponent:
    if not isinstance(component, SourceStorageComponent):
        raise TypeError("storage realization components have an unsupported type")
    physical_axes = _snapshot_sequence(component.physical_axes, "physical axes")
    return SourceStorageComponent(
        graph_instance_id=component.graph_instance_id,
        native_component_id=component.native_component_id,
        source_native_name=component.source_native_name,
        component_role=component.component_role,
        carrier_dtype=component.carrier_dtype,
        physical_shape=component.physical_shape,
        physical_axes=tuple(_validated_axis_copy(axis) for axis in physical_axes),
        storage_encoding=component.storage_encoding,
        padding_semantics=component.padding_semantics,
        padding_fill_encoding=component.padding_fill_encoding,
        permutation_id=component.permutation_id,
        swizzle_id=component.swizzle_id,
    )


def _validated_realization_copy(realization: object) -> SourceRealization:
    if isinstance(realization, SourceStorageRealization):
        components = _snapshot_sequence(realization.components, "storage components")
        return SourceStorageRealization(
            realization_id=realization.realization_id,
            graph_instance_id=realization.graph_instance_id,
            output_record_id=realization.output_record_id,
            components=tuple(
                _validated_component_copy(component) for component in components
            ),
            output_dtype=realization.output_dtype,
            output_shape=realization.output_shape,
            output_numeric_encoding=realization.output_numeric_encoding,
            normalization=_validated_contract_copy(realization.normalization),
        )
    if isinstance(realization, SourceDerivedRealization):
        return SourceDerivedRealization(
            realization_id=realization.realization_id,
            graph_instance_id=realization.graph_instance_id,
            output_record_id=realization.output_record_id,
            output_dtype=realization.output_dtype,
            output_shape=realization.output_shape,
            output_numeric_encoding=realization.output_numeric_encoding,
            derivation=_validated_contract_copy(realization.derivation),
        )
    raise TypeError("source realization inventory has an unsupported witness")


def validate_source_storage_realization_inventory(
    inventory: SourceStorageRealizationInventory,
) -> None:
    """Revalidate the complete immutable structure without trusting construction.

    Discovery calls this immediately before adapter selection.  Rebuilding the
    metadata-only records replays every nested constructor invariant and also
    rejects noncanonical tuple/order mutations.  It never reads or transforms a
    tensor payload.
    """
    if not isinstance(inventory, SourceStorageRealizationInventory):
        raise TypeError("inventory must be SourceStorageRealizationInventory")
    realizations = _snapshot_sequence(inventory.realizations, "source realizations")
    validated = SourceStorageRealizationInventory(
        graph_instance_id=inventory.graph_instance_id,
        normalizer_manifest=_validated_manifest_copy(inventory.normalizer_manifest),
        realizations=tuple(
            _validated_realization_copy(realization) for realization in realizations
        ),
    )
    if validated != inventory:
        raise ValueError("source realization inventory is not canonical")


def _storage_realization_payload(
    realization: SourceStorageRealization,
) -> dict[str, object]:
    return {
        "type": "source_storage_realization",
        "realization_id": realization.realization_id,
        "graph_instance_id": realization.graph_instance_id,
        "output_record_id": realization.output_record_id,
        "components": [
            _storage_component_payload(component, include_identity=True)
            for component in realization.components
        ],
        "output_dtype": realization.output_dtype.value,
        "output_shape": list(realization.output_shape),
        "output_numeric_encoding": realization.output_numeric_encoding,
        "normalization": _normalization_contract_payload(realization.normalization),
    }


def _derived_realization_payload(
    realization: SourceDerivedRealization,
) -> dict[str, object]:
    return {
        "type": "source_derived_realization",
        "realization_id": realization.realization_id,
        "graph_instance_id": realization.graph_instance_id,
        "output_record_id": realization.output_record_id,
        "output_dtype": realization.output_dtype.value,
        "output_shape": list(realization.output_shape),
        "output_numeric_encoding": realization.output_numeric_encoding,
        "derivation": _normalization_contract_payload(realization.derivation),
    }


def source_storage_inventory_digest(
    inventory: SourceStorageRealizationInventory,
) -> str:
    """Stream the canonical digest of one complete realization inventory."""
    if not isinstance(inventory, SourceStorageRealizationInventory):
        raise TypeError("inventory must be SourceStorageRealizationInventory")
    digest = sha256()
    digest.update(b'{"graph_instance_id":')
    digest.update(_canonical_json_bytes(inventory.graph_instance_id))
    digest.update(b',"normalizer_manifest":')
    digest.update(
        _canonical_json_bytes(
            _normalizer_manifest_payload(inventory.normalizer_manifest)
        )
    )
    digest.update(b',"realizations":[')
    for realization_index, realization in enumerate(inventory.realizations):
        if realization_index:
            digest.update(b",")
        payload = (
            _storage_realization_payload(realization)
            if isinstance(realization, SourceStorageRealization)
            else _derived_realization_payload(realization)
        )
        digest.update(_canonical_json_bytes(payload))
    digest.update(b'],"type":"source_storage_realization_inventory"}')
    return f"sha256:{digest.hexdigest()}"


def source_realization_is_wire_eligible(realization: SourceRealization) -> bool:
    """Return whether the witness has raw components that may enter planning.

    A true result is not direct-copy authority.  Task 7 must still re-probe the
    live endpoint and prove exact adjacent-stage compatibility.
    """
    if not isinstance(
        realization,
        (SourceStorageRealization, SourceDerivedRealization),
    ):
        raise TypeError("realization has an unsupported type")
    return isinstance(realization, SourceStorageRealization)


def source_realizations_have_exact_physical_representation(
    left: SourceRealization,
    right: SourceRealization,
) -> bool:
    """Test necessary physical equality without granting transfer authority.

    Native identities and names are intentionally excluded because endpoints
    can use different handles.  Numeric output facts, normalizer provenance,
    ordered component roles, and every physical layout fact are included.
    """
    if not isinstance(
        left,
        (SourceStorageRealization, SourceDerivedRealization),
    ) or not isinstance(
        right,
        (SourceStorageRealization, SourceDerivedRealization),
    ):
        raise TypeError("realizations have unsupported types")
    if not isinstance(left, SourceStorageRealization) or not isinstance(
        right, SourceStorageRealization
    ):
        return False
    return (
        _realization_output_facts(left) == _realization_output_facts(right)
        and left.normalization == right.normalization
        and tuple(
            _storage_component_payload(component, include_identity=False)
            for component in left.components
        )
        == tuple(
            _storage_component_payload(component, include_identity=False)
            for component in right.components
        )
    )
