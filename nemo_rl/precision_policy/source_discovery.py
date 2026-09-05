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

"""Framework-free, graph-scoped source-discovery trust boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Any, TypeVar, cast

from nemo_rl.precision_policy.semantic import (
    EvidenceSource,
    EvidenceSourceKind,
    ExpectedGraphDeclaration,
    ImmutableAuxiliaryEvidence,
    ResolvedGraphTopology,
    SourceMutability,
    canonical_resolved_graph_topology_payload,
)
from nemo_rl.precision_policy.source_dtype import CanonicalSourceDType
from nemo_rl.precision_policy.source_storage import (
    SourceDerivedRealization,
    SourceExtentRounding,
    SourceLiteralAxisExtent,
    SourceNormalizationContract,
    SourceNormalizationKind,
    SourceNormalizedAxisExtent,
    SourceNormalizerManifest,
    SourcePaddingSemantics,
    SourcePhysicalAxisSpec,
    SourceStorageRealization,
    SourceStorageRealizationInventory,
    SourceStorageComponent,
    source_normalizer_manifest_digest,
    source_storage_inventory_digest,
    validate_source_storage_realization_inventory,
)


_SOURCE_SCHEMA_PATTERN = re.compile(r"[a-z][a-z0-9-]*(?:\.[a-z0-9-]+)+\.v[1-9][0-9]*")
_IMMUTABLE_REVISION_PATTERN = re.compile(
    r"(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256:[0-9a-f]{64})"
)
_SHA256_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
EXPECTED_CONTRIBUTOR_AUTHORITY_LOCATOR = (
    "precision-policy.expected-contributor-authority.v1"
)
_SCALAR_OR_BUFFER_SEQUENCE_TYPES = (str, bytes, bytearray, memoryview)
_SequenceItemT = TypeVar("_SequenceItemT")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be exact non-empty text")
    return value


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


def _evidence_payload(evidence: EvidenceSource) -> dict[str, object]:
    return {
        "type": "evidence_source",
        "kind": evidence.kind.value,
        "locator": evidence.locator,
        "digest": evidence.digest,
    }


def _immutable_evidence_payload(
    evidence: ImmutableAuxiliaryEvidence,
) -> dict[str, object]:
    return {
        "type": "immutable_auxiliary_evidence",
        "graph_instance_id": evidence.graph_instance_id,
        "model_identity": evidence.model_identity,
        "pinned_checkpoint_revision": evidence.pinned_checkpoint_revision,
        "checkpoint_content_digest": evidence.checkpoint_content_digest,
        "model_config_digest": evidence.model_config_digest,
        "semantic_domain_digest": evidence.semantic_domain_digest,
        "evidence_source": _evidence_payload(evidence.evidence_source),
    }


def _declaration_payload(
    declaration: ExpectedGraphDeclaration,
) -> dict[str, object]:
    lifecycle = declaration.lifecycle
    return {
        "type": "expected_graph_declaration",
        "graph_instance_id": declaration.graph_instance_id,
        "model_identity": declaration.model_identity,
        "lifecycle": {
            "type": "graph_lifecycle",
            "graph_kind": lifecycle.graph_kind.value,
            "graph_provenance": lifecycle.graph_provenance.value,
            "rollout_participation": lifecycle.rollout_participation.value,
            "immutable_evidence": (
                None
                if lifecycle.immutable_evidence is None
                else _immutable_evidence_payload(lifecycle.immutable_evidence)
            ),
        },
    }


def _typed_config_payload(value: object) -> dict[str, object]:
    if value is None:
        return {"type": "null", "value": None}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if not isfinite(value):  # pragma: no cover - rejected during snapshot
            raise ValueError("configuration floats must be finite")
        return {"type": "float", "value": 0.0 if value == 0.0 else value}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is _FrozenConfigMapping:
        return {
            "type": "mapping",
            "entries": [
                {"key": key, "value": _typed_config_payload(item)}
                for key, item in value.entries
            ],
        }
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("configuration mapping keys must be exact strings")
        return {
            "type": "mapping",
            "entries": [
                {"key": key, "value": _typed_config_payload(value[key])}
                for key in sorted(value)
            ],
        }
    if type(value) is tuple:
        return {
            "type": "sequence",
            "items": [_typed_config_payload(item) for item in value],
        }
    raise TypeError("configuration contains an unsupported value")


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True, order=True)
class SourceSchemaId:
    """Exact producer metadata schema identity."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("source schema ID must be a string")
        if _SOURCE_SCHEMA_PATTERN.fullmatch(self.value) is None:
            raise ValueError("source schema ID must be lowercase and versioned")


HF_SAFETENSORS_HEADER_V1 = SourceSchemaId("hf.safetensors.header.v1")
MEGATRON_BRIDGE_STATE_DICT_V1 = SourceSchemaId("megatron.bridge.state-dict.v1")
NEMO_AUTOMODEL_STATE_DICT_V1 = SourceSchemaId("nemo-automodel.state-dict.v1")
TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1 = SourceSchemaId(
    "transformer-engine.quantized-storage.v1"
)


@dataclass(frozen=True, slots=True)
class SourceProducerFingerprint:
    """Immutable producer implementation and normalization identity."""

    schema_id: SourceSchemaId
    producer_implementation_id: str
    producer_revision: str
    normalization_contract_digest: str
    evidence: EvidenceSource

    def __post_init__(self) -> None:
        if not isinstance(self.schema_id, SourceSchemaId):
            raise TypeError("producer schema_id must be SourceSchemaId")
        _require_text(
            self.producer_implementation_id,
            "producer implementation ID",
        )
        if not isinstance(self.producer_revision, str):
            raise TypeError("producer revision must be a string")
        if _IMMUTABLE_REVISION_PATTERN.fullmatch(self.producer_revision) is None:
            raise ValueError("producer revision must be an immutable content identity")
        _require_sha256_digest(
            self.normalization_contract_digest,
            "normalization contract digest",
        )
        if not isinstance(self.evidence, EvidenceSource):
            raise TypeError("producer evidence must be EvidenceSource")


def _fingerprint_payload(
    fingerprint: SourceProducerFingerprint,
) -> dict[str, object]:
    return {
        "type": "source_producer_fingerprint",
        "schema_id": fingerprint.schema_id.value,
        "producer_implementation_id": fingerprint.producer_implementation_id,
        "producer_revision": fingerprint.producer_revision,
        "normalization_contract_digest": fingerprint.normalization_contract_digest,
        "evidence": _evidence_payload(fingerprint.evidence),
    }


def _fingerprint_digest(fingerprint: SourceProducerFingerprint) -> str:
    return _canonical_digest(_fingerprint_payload(fingerprint))


@dataclass(frozen=True, slots=True)
class ExpectedContributorAuthority:
    """ID-free commitment to one trusted expected contributor set."""

    contributor_set_digest: str
    contributor_count: int
    authority: EvidenceSource

    def __post_init__(self) -> None:
        _require_sha256_digest(
            self.contributor_set_digest,
            "contributor set digest",
        )
        _require_positive_int(self.contributor_count, "contributor count")
        if not isinstance(self.authority, EvidenceSource):
            raise TypeError("contributor authority must be EvidenceSource")
        if self.authority.kind != EvidenceSourceKind.CONTENT_ADDRESS:
            raise ValueError("contributor authority must use content-address evidence")
        if self.authority.locator != EXPECTED_CONTRIBUTOR_AUTHORITY_LOCATOR:
            raise ValueError("contributor authority must use the canonical locator")
        _require_sha256_digest(
            self.authority.digest,
            "contributor authority digest",
        )


def _authority_payload(
    authority: ExpectedContributorAuthority,
) -> dict[str, object]:
    return {
        "type": "expected_contributor_authority",
        "contributor_set_digest": authority.contributor_set_digest,
        "contributor_count": authority.contributor_count,
        "authority": _evidence_payload(authority.authority),
    }


def _authority_evidence_commitment(evidence: EvidenceSource) -> EvidenceSource:
    return EvidenceSource(
        kind=EvidenceSourceKind.CONTENT_ADDRESS,
        locator=EXPECTED_CONTRIBUTOR_AUTHORITY_LOCATOR,
        digest=_canonical_digest(
            {
                "type": "expected_contributor_authority_evidence",
                "evidence": _evidence_payload(evidence),
            }
        ),
    )


def _contributor_set_digest(contributor_ids: tuple[str, ...]) -> str:
    return _canonical_digest(
        {
            "type": "expected_contributor_set",
            "contributor_ids": list(contributor_ids),
        }
    )


@dataclass(frozen=True, slots=True)
class ExpectedContributorSet:
    """Trusted opaque contributors retained only through core validation."""

    contributor_ids: tuple[str, ...]
    authority: EvidenceSource

    def __post_init__(self) -> None:
        contributor_items = _snapshot_sequence(
            self.contributor_ids,
            "expected contributor IDs",
        )
        contributor_ids = tuple(
            _require_text(contributor_id, "contributor ID")
            for contributor_id in contributor_items
        )
        if not contributor_ids:
            raise ValueError("expected contributor set must be non-empty")
        if len(contributor_ids) != len(set(contributor_ids)):
            raise ValueError("expected contributor IDs must be duplicate-free")
        if not isinstance(self.authority, EvidenceSource):
            raise TypeError("expected contributor authority must be EvidenceSource")
        object.__setattr__(self, "contributor_ids", tuple(sorted(contributor_ids)))

    def to_authority(self) -> ExpectedContributorAuthority:
        """Return the ID-free canonical commitment to this trusted set."""
        return ExpectedContributorAuthority(
            contributor_set_digest=_contributor_set_digest(self.contributor_ids),
            contributor_count=len(self.contributor_ids),
            authority=_authority_evidence_commitment(self.authority),
        )


class SourceRecordProvenance(StrEnum):
    """Source authority recorded before semantic classification."""

    TRAINING_RUNTIME = "training_runtime"
    CHECKPOINT_STORAGE = "checkpoint_storage"
    BACKEND_DERIVED = "backend_derived"
    TIED_STORAGE = "tied_storage"
    SYNCHRONIZED_REPLICA = "synchronized_replica"


class DiscoveryRequestKind(StrEnum):
    """Exact request contract committed by a discovery receipt."""

    GRAPH_TOPOLOGY_INPUT = "graph_topology_input"
    RUNTIME_GRAPH_SOURCE_REQUEST = "runtime_graph_source_request"


@dataclass(frozen=True, slots=True)
class SourceDiscoveryRecord:
    """Frozen producer-normalized component view with native provenance."""

    record_id: str
    graph_instance_id: str
    source_native_name: str | None
    source_native_owner_id: str | None
    dtype: CanonicalSourceDType
    shape: tuple[int, ...]
    numeric_encoding: str
    provenance: SourceRecordProvenance
    provenance_evidence: EvidenceSource
    source_mutability: SourceMutability
    mutability_evidence: EvidenceSource

    def __post_init__(self) -> None:
        _require_text(self.record_id, "source discovery record_id")
        _require_text(self.graph_instance_id, "source discovery graph_instance_id")
        shape = _snapshot_sequence(self.shape, "source shape")
        object.__setattr__(self, "shape", shape)
        if not isinstance(self.dtype, CanonicalSourceDType):
            raise TypeError("source discovery dtype must be CanonicalSourceDType")
        _require_text(self.numeric_encoding, "normalized source numeric encoding")
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
        if is_absent and self.provenance == SourceRecordProvenance.SYNCHRONIZED_REPLICA:
            raise ValueError(
                "absent source record cannot have synchronized-replica provenance"
            )
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


def _record_payload(record: SourceDiscoveryRecord) -> dict[str, object]:
    return {
        "type": "source_discovery_record",
        "record_id": record.record_id,
        "graph_instance_id": record.graph_instance_id,
        "source_native_name": record.source_native_name,
        "source_native_owner_id": record.source_native_owner_id,
        "dtype": record.dtype.value,
        "shape": list(record.shape),
        "numeric_encoding": record.numeric_encoding,
        "provenance": record.provenance.value,
        "provenance_evidence": _evidence_payload(record.provenance_evidence),
        "source_mutability": record.source_mutability.value,
        "mutability_evidence": _evidence_payload(record.mutability_evidence),
    }


def _source_identity_payload(record: SourceDiscoveryRecord) -> dict[str, object]:
    return {
        "type": "source_identity",
        "record_id": record.record_id,
        "graph_instance_id": record.graph_instance_id,
        "source_native_name": record.source_native_name,
        "source_native_owner_id": record.source_native_owner_id,
    }


@dataclass(frozen=True, slots=True)
class DiscoveryContribution:
    """One opaque contributor's normalized records before trust validation."""

    contributor_id: str
    graph_instance_id: str
    producer_fingerprint: SourceProducerFingerprint
    records: tuple[SourceDiscoveryRecord, ...]
    storage_realizations: SourceStorageRealizationInventory

    def __post_init__(self) -> None:
        _require_text(self.contributor_id, "contributor ID")
        _require_text(self.graph_instance_id, "contribution graph_instance_id")
        if not isinstance(self.producer_fingerprint, SourceProducerFingerprint):
            raise TypeError(
                "contribution fingerprint must be SourceProducerFingerprint"
            )
        records = _snapshot_sequence(self.records, "contribution records")
        if any(not isinstance(record, SourceDiscoveryRecord) for record in records):
            raise TypeError("contribution records must be SourceDiscoveryRecord values")
        if not isinstance(
            self.storage_realizations,
            SourceStorageRealizationInventory,
        ):
            raise TypeError(
                "contribution storage_realizations must be "
                "SourceStorageRealizationInventory"
            )
        if self.storage_realizations.graph_instance_id != self.graph_instance_id:
            raise ValueError("contribution realization inventory graph mismatch")
        if (
            source_normalizer_manifest_digest(
                self.storage_realizations.normalizer_manifest
            )
            != self.producer_fingerprint.normalization_contract_digest
        ):
            raise ValueError(
                "contribution normalizer manifest differs from producer fingerprint"
            )
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


@dataclass(frozen=True, slots=True)
class DiscoveryCompletenessReceipt:
    """Assembly-derived commitment revalidated before classification."""

    graph_instance_id: str
    producer_fingerprint_digest: str
    observed_contributor_set_digest: str
    observed_contributor_count: int
    source_set_digest: str
    source_count: int
    canonical_records_digest: str
    storage_realization_set_digest: str
    storage_realization_count: int
    request_kind: DiscoveryRequestKind
    request_digest: str

    def __post_init__(self) -> None:
        _require_text(self.graph_instance_id, "receipt graph_instance_id")
        for field_name in (
            "producer_fingerprint_digest",
            "observed_contributor_set_digest",
            "source_set_digest",
            "canonical_records_digest",
            "storage_realization_set_digest",
        ):
            _require_sha256_digest(
                getattr(self, field_name),
                field_name.replace("_", " "),
            )
        _require_registered_enum_member(
            self.request_kind,
            DiscoveryRequestKind,
            "receipt request_kind",
        )
        if type(self.request_digest) is not str:
            raise TypeError("receipt request_digest must be an exact string")
        _require_sha256_digest(self.request_digest, "request digest")
        _require_positive_int(
            self.observed_contributor_count,
            "observed contributor count",
        )
        _require_positive_int(self.source_count, "source count")
        if isinstance(self.storage_realization_count, bool) or not isinstance(
            self.storage_realization_count,
            int,
        ):
            raise TypeError("storage realization count must be an integer")
        if self.storage_realization_count < 0:
            raise ValueError("storage realization count must be non-negative")


@dataclass(frozen=True, slots=True)
class GraphDiscoveryPartition:
    """ID-free, canonical source universe for exactly one graph."""

    graph_instance_id: str
    producer_fingerprint: SourceProducerFingerprint
    expected_contributor_authority: ExpectedContributorAuthority
    records: tuple[SourceDiscoveryRecord, ...]
    storage_realizations: SourceStorageRealizationInventory
    completeness_receipt: DiscoveryCompletenessReceipt

    def __post_init__(self) -> None:
        _require_text(self.graph_instance_id, "partition graph_instance_id")
        if not isinstance(self.producer_fingerprint, SourceProducerFingerprint):
            raise TypeError("partition fingerprint must be SourceProducerFingerprint")
        if not isinstance(
            self.expected_contributor_authority,
            ExpectedContributorAuthority,
        ):
            raise TypeError(
                "partition expected authority must be ExpectedContributorAuthority"
            )
        records = _snapshot_sequence(self.records, "partition records")
        if any(not isinstance(record, SourceDiscoveryRecord) for record in records):
            raise TypeError("partition records must be SourceDiscoveryRecord values")
        if not isinstance(
            self.storage_realizations,
            SourceStorageRealizationInventory,
        ):
            raise TypeError(
                "partition storage_realizations must be "
                "SourceStorageRealizationInventory"
            )
        if self.storage_realizations.graph_instance_id != self.graph_instance_id:
            raise ValueError("partition realization inventory graph mismatch")
        if not isinstance(self.completeness_receipt, DiscoveryCompletenessReceipt):
            raise TypeError("partition receipt must be DiscoveryCompletenessReceipt")
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


def _graph_sort_key(graph_instance_id: str) -> tuple[int, str]:
    return (0 if graph_instance_id == "main" else 1, graph_instance_id)


@dataclass(frozen=True, slots=True)
class SourceDiscoveryInventory:
    """Canonical graph-partitioned source discovery inventory."""

    partitions: tuple[GraphDiscoveryPartition, ...]

    def __post_init__(self) -> None:
        partitions = _snapshot_sequence(self.partitions, "inventory partitions")
        if any(
            not isinstance(partition, GraphDiscoveryPartition)
            for partition in partitions
        ):
            raise TypeError("source discovery inventory requires graph partitions")
        object.__setattr__(
            self,
            "partitions",
            tuple(
                sorted(
                    partitions,
                    key=lambda partition: _graph_sort_key(partition.graph_instance_id),
                )
            ),
        )

    @property
    def records(self) -> tuple[SourceDiscoveryRecord, ...]:
        """Return a read-only flattened view for topology-internal migration."""
        return tuple(
            record for partition in self.partitions for record in partition.records
        )


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
        if type(other) is _FrozenConfigMapping:
            return self.entries == other.entries
        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        return all(key in other and value == other[key] for key, value in self.entries)


def _freeze_config_value(
    value: object,
    path: str,
    active_ids: set[int],
) -> object:
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("model_config must not contain cycles")
        active_ids.add(identity)
        try:
            frozen: dict[str, object] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path} keys must be exact strings")
                frozen[key] = _freeze_config_value(
                    item,
                    f"{path}.{key}",
                    active_ids,
                )
            return _FrozenConfigMapping(tuple(sorted(frozen.items())))
        finally:
            active_ids.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("model_config must not contain cycles")
        active_ids.add(identity)
        try:
            return tuple(
                _freeze_config_value(
                    item,
                    f"{path}[{index}]",
                    active_ids,
                )
                for index, item in enumerate(value)
            )
        finally:
            active_ids.remove(identity)
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path} floats must be finite")
        return value
    if value is None or type(value) in {str, int, bool}:
        return value
    raise TypeError(f"{path} must contain only exact JSON scalar values")


def _freeze_model_config(config: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_config_value(config, "model_config", set())
    if not isinstance(frozen, Mapping):  # pragma: no cover - fixed by input type
        raise AssertionError("model_config snapshot must be a mapping")
    return frozen


def _require_exact_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return _require_text(value, name)


def _require_exact_sha256_digest(value: object, name: str) -> str:
    _require_exact_text(value, name)
    return _require_sha256_digest(value, name)


def _require_registered_enum_member(
    value: object,
    enum_type: type[StrEnum],
    name: str,
) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be an exact {enum_type.__name__}")
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


def _validate_exact_evidence_source(value: object, name: str) -> None:
    if type(value) is not EvidenceSource:
        raise TypeError(f"{name} must be an exact EvidenceSource")
    _require_registered_enum_member(
        value.kind,
        EvidenceSourceKind,
        f"{name}.kind",
    )
    _require_exact_text(value.locator, f"{name}.locator")
    _require_exact_text(value.digest, f"{name}.digest")
    value.__post_init__()


def _validate_exact_producer_fingerprint(value: object) -> None:
    if type(value) is not SourceProducerFingerprint:
        raise TypeError(
            "source_producer_fingerprint must be an exact SourceProducerFingerprint"
        )
    if type(value.schema_id) is not SourceSchemaId:
        raise TypeError("producer schema_id must be an exact SourceSchemaId")
    _require_exact_text(value.schema_id.value, "source schema")
    value.schema_id.__post_init__()
    _require_exact_text(
        value.producer_implementation_id,
        "producer implementation ID",
    )
    _require_exact_text(value.producer_revision, "producer revision")
    _require_exact_sha256_digest(
        value.normalization_contract_digest,
        "normalization contract digest",
    )
    _validate_exact_evidence_source(value.evidence, "producer evidence")
    value.__post_init__()


def _validate_exact_contributor_authority(value: object) -> None:
    if type(value) is not ExpectedContributorAuthority:
        raise TypeError(
            "expected_contributor_authority must be an exact "
            "ExpectedContributorAuthority"
        )
    _require_exact_sha256_digest(
        value.contributor_set_digest,
        "contributor set digest",
    )
    if type(value.contributor_count) is not int:
        raise TypeError("contributor count must be an exact integer")
    _validate_exact_evidence_source(value.authority, "contributor authority")
    value.__post_init__()


def _validate_exact_expected_contributor_set(value: object) -> None:
    if type(value) is not ExpectedContributorSet:
        raise TypeError("expected_contributors must be an exact ExpectedContributorSet")
    if type(value.contributor_ids) is not tuple:
        raise TypeError("expected contributor IDs must be an exact tuple")
    contributor_ids = tuple(
        _require_exact_text(contributor_id, "contributor ID")
        for contributor_id in value.contributor_ids
    )
    if not contributor_ids:
        raise ValueError("expected contributor set must be non-empty")
    if len(contributor_ids) != len(set(contributor_ids)):
        raise ValueError("expected contributor IDs must be duplicate-free")
    if contributor_ids != tuple(sorted(contributor_ids)):
        raise ValueError("expected contributor IDs must use canonical order")
    _validate_exact_evidence_source(value.authority, "expected contributor authority")


def _authority_from_exact_expected_contributor_set(
    value: ExpectedContributorSet,
) -> ExpectedContributorAuthority:
    _validate_exact_expected_contributor_set(value)
    return ExpectedContributorAuthority(
        contributor_set_digest=_contributor_set_digest(value.contributor_ids),
        contributor_count=len(value.contributor_ids),
        authority=_authority_evidence_commitment(value.authority),
    )


def _validate_exact_frozen_config(
    value: object,
    path: str,
    active_ids: set[int] | None = None,
    completed_ids: set[int] | None = None,
) -> None:
    if active_ids is None:
        active_ids = set()
    if completed_ids is None:
        completed_ids = set()
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not isfinite(cast(float, value)):
            raise ValueError(f"{path} floats must be finite")
        return
    if value_type in {tuple, _FrozenConfigMapping}:
        identity = id(value)
        if identity in active_ids:
            raise ValueError(f"{path} must not contain cycles")
        if identity in completed_ids:
            return
        active_ids.add(identity)
        try:
            if value_type is tuple:
                for index, item in enumerate(cast(tuple[object, ...], value)):
                    _validate_exact_frozen_config(
                        item,
                        f"{path}[{index}]",
                        active_ids,
                        completed_ids,
                    )
                return
            mapping = cast(_FrozenConfigMapping, value)
            if type(mapping.entries) is not tuple:
                raise TypeError(f"{path} entries must be an exact tuple")
            keys: list[str] = []
            for index, entry in enumerate(mapping.entries):
                if type(entry) is not tuple or len(entry) != 2:
                    raise TypeError(
                        f"{path} entry {index} must be an exact key/value tuple"
                    )
                key, item = entry
                keys.append(_require_exact_text(key, f"{path} key"))
                _validate_exact_frozen_config(
                    item,
                    f"{path}.{key}",
                    active_ids,
                    completed_ids,
                )
            if keys != sorted(set(keys)):
                raise ValueError(f"{path} entries must have unique canonical key order")
            return
        finally:
            active_ids.remove(identity)
            completed_ids.add(identity)
    raise TypeError(f"{path} contains a non-exact frozen value")


def _validate_runtime_source_request_fields(
    runtime_request: object,
) -> tuple[RuntimeGraphSourceRequest, dict[str, object]]:
    if type(runtime_request) is not RuntimeGraphSourceRequest:
        raise TypeError("runtime_request must be an exact RuntimeGraphSourceRequest")
    request = cast(RuntimeGraphSourceRequest, runtime_request)
    resolved_graph_payload = canonical_resolved_graph_topology_payload(
        request.resolved_graph
    )
    if request.declaration is not request.resolved_graph.declaration:
        raise ValueError(
            "runtime request declaration is not the resolved graph snapshot"
        )
    _require_exact_sha256_digest(
        request.semantic_structure_digest,
        "semantic_structure_digest",
    )
    _require_exact_sha256_digest(
        request.selection_group_id,
        "selection_group_id",
    )
    _validate_exact_frozen_config(request.model_config, "model_config")
    _require_exact_text(
        request.resolved_model_revision,
        "resolved_model_revision",
    )
    if (
        request.resolved_model_revision
        != request.resolved_graph.resolved_model_revision
    ):
        raise ValueError("resolved graph revision mismatch")
    _validate_exact_producer_fingerprint(request.source_producer_fingerprint)
    _validate_exact_contributor_authority(request.expected_contributor_authority)
    _validate_exact_evidence_source(request.source_identity, "source_identity")
    _validate_exact_evidence_source(
        request.artifact_identity,
        "artifact_identity",
    )
    _require_exact_text(
        request.source_allocation_generation,
        "source_allocation_generation",
    )
    return request, resolved_graph_payload


def _runtime_source_request_identity_digest_unchecked(
    runtime_request: RuntimeGraphSourceRequest,
    resolved_graph_payload: dict[str, object],
) -> str:
    return _canonical_digest(
        {
            "type": "runtime_graph_source_request",
            "declaration": _declaration_payload(runtime_request.declaration),
            "resolved_graph": resolved_graph_payload,
            "semantic_structure_digest": runtime_request.semantic_structure_digest,
            "selection_group_id": runtime_request.selection_group_id,
            "model_config": _typed_config_payload(runtime_request.model_config),
            "resolved_model_revision": runtime_request.resolved_model_revision,
            "source_producer_fingerprint": _fingerprint_payload(
                runtime_request.source_producer_fingerprint
            ),
            "expected_contributor_authority": _authority_payload(
                runtime_request.expected_contributor_authority
            ),
            "source_identity": _evidence_payload(runtime_request.source_identity),
            "artifact_identity": _evidence_payload(runtime_request.artifact_identity),
            "source_allocation_generation": (
                runtime_request.source_allocation_generation
            ),
        }
    )


def _validate_runtime_source_request_snapshot(
    runtime_request: object,
) -> RuntimeGraphSourceRequest:
    request, resolved_graph_payload = _validate_runtime_source_request_fields(
        runtime_request
    )
    _require_exact_sha256_digest(
        request.runtime_source_request_digest,
        "runtime_source_request_digest",
    )
    derived = _runtime_source_request_identity_digest_unchecked(
        request,
        resolved_graph_payload,
    )
    if request.runtime_source_request_digest != derived:
        raise ValueError("runtime source request digest mismatch")
    return request


@dataclass(frozen=True, slots=True)
class RuntimeGraphSourceRequest:
    """Phase 2 source request bound to one immutable Phase 1 graph selection."""

    declaration: ExpectedGraphDeclaration
    resolved_graph: ResolvedGraphTopology
    semantic_structure_digest: str
    selection_group_id: str
    model_config: Mapping[str, object]
    resolved_model_revision: str
    source_producer_fingerprint: SourceProducerFingerprint
    expected_contributor_authority: ExpectedContributorAuthority
    source_identity: EvidenceSource
    artifact_identity: EvidenceSource
    source_allocation_generation: str
    runtime_source_request_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.declaration) is not ExpectedGraphDeclaration:
            raise TypeError("declaration must be an exact ExpectedGraphDeclaration")
        if type(self.resolved_graph) is not ResolvedGraphTopology:
            raise TypeError("resolved_graph must be an exact ResolvedGraphTopology")
        self.resolved_graph.validate_complete()
        if self.resolved_graph.declaration != self.declaration:
            raise ValueError("resolved graph declaration mismatch")
        object.__setattr__(self, "declaration", self.resolved_graph.declaration)
        if not isinstance(self.model_config, Mapping):
            raise TypeError("model_config must be a mapping")
        object.__setattr__(
            self,
            "model_config",
            _freeze_model_config(self.model_config),
        )
        _, resolved_graph_payload = _validate_runtime_source_request_fields(self)
        object.__setattr__(
            self,
            "runtime_source_request_digest",
            _runtime_source_request_identity_digest_unchecked(
                self,
                resolved_graph_payload,
            ),
        )


def runtime_source_request_identity_digest(
    runtime_request: RuntimeGraphSourceRequest,
) -> str:
    """Return the canonical identity of one Phase 1-bound runtime request."""
    return _validate_runtime_source_request_snapshot(
        runtime_request
    ).runtime_source_request_digest


@dataclass(frozen=True, slots=True)
class GraphTopologyInput:
    """One declared graph paired with immutable discovery identities."""

    declaration: ExpectedGraphDeclaration
    model_config: Mapping[str, object]
    resolved_model_revision: str
    source_producer_fingerprint: SourceProducerFingerprint
    expected_contributor_authority: ExpectedContributorAuthority
    source_identity: EvidenceSource
    artifact_identity: EvidenceSource

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, ExpectedGraphDeclaration):
            raise TypeError("declaration must be ExpectedGraphDeclaration")
        if not isinstance(self.model_config, Mapping):
            raise TypeError("model_config must be a mapping")
        _require_text(self.resolved_model_revision, "resolved_model_revision")
        if not isinstance(
            self.source_producer_fingerprint,
            SourceProducerFingerprint,
        ):
            raise TypeError(
                "source_producer_fingerprint must be SourceProducerFingerprint"
            )
        if not isinstance(
            self.expected_contributor_authority,
            ExpectedContributorAuthority,
        ):
            raise TypeError(
                "expected_contributor_authority must be ExpectedContributorAuthority"
            )
        if not isinstance(self.source_identity, EvidenceSource):
            raise TypeError("source_identity must be EvidenceSource")
        if not isinstance(self.artifact_identity, EvidenceSource):
            raise TypeError("artifact_identity must be EvidenceSource")
        object.__setattr__(
            self,
            "model_config",
            _freeze_model_config(self.model_config),
        )


def graph_input_identity_digest(graph_input: GraphTopologyInput) -> str:
    """Return the typed canonical identity of one discovery graph input."""
    if not isinstance(graph_input, GraphTopologyInput):
        raise TypeError("graph_input must be GraphTopologyInput")
    return _canonical_digest(
        {
            "type": "graph_topology_input",
            "declaration": _declaration_payload(graph_input.declaration),
            "model_config": _typed_config_payload(graph_input.model_config),
            "resolved_model_revision": graph_input.resolved_model_revision,
            "source_producer_fingerprint": _fingerprint_payload(
                graph_input.source_producer_fingerprint
            ),
            "expected_contributor_authority": _authority_payload(
                graph_input.expected_contributor_authority
            ),
            "source_identity": _evidence_payload(graph_input.source_identity),
            "artifact_identity": _evidence_payload(graph_input.artifact_identity),
        }
    )


def _validate_record_universe(
    graph_instance_id: str,
    records: tuple[SourceDiscoveryRecord, ...],
) -> tuple[SourceDiscoveryRecord, ...]:
    if not records:
        raise ValueError("complete graph source universe cannot be empty")
    if any(record.graph_instance_id != graph_instance_id for record in records):
        raise ValueError("source record belongs to another graph")
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
    return tuple(
        sorted(
            records,
            key=lambda record: (record.graph_instance_id, record.record_id),
        )
    )


def _source_set_digest(records: tuple[SourceDiscoveryRecord, ...]) -> str:
    return _canonical_digest(
        {
            "type": "source_identity_set",
            "sources": [_source_identity_payload(record) for record in records],
        }
    )


def _records_digest(records: tuple[SourceDiscoveryRecord, ...]) -> str:
    return _canonical_digest(
        {
            "type": "canonical_source_records",
            "records": [_record_payload(record) for record in records],
        }
    )


def _merge_storage_realization_inventories(
    graph_instance_id: str,
    contributions: tuple[DiscoveryContribution, ...],
) -> SourceStorageRealizationInventory:
    manifests = tuple(
        contribution.storage_realizations.normalizer_manifest
        for contribution in contributions
    )
    manifest = manifests[0]
    if any(candidate != manifest for candidate in manifests[1:]):
        raise ValueError("discovery contributions use different normalizer manifests")
    inventory = SourceStorageRealizationInventory(
        graph_instance_id=graph_instance_id,
        normalizer_manifest=manifest,
        realizations=tuple(
            realization
            for contribution in contributions
            for realization in contribution.storage_realizations.realizations
        ),
    )
    validate_source_storage_realization_inventory(inventory)
    return inventory


def _validate_storage_realization_coverage(
    graph_instance_id: str,
    records: tuple[SourceDiscoveryRecord, ...],
    storage_realizations: SourceStorageRealizationInventory,
) -> None:
    if storage_realizations.graph_instance_id != graph_instance_id:
        raise ValueError("storage realization inventory belongs to another graph")
    records_by_id = {record.record_id: record for record in records}
    realizations_by_record: dict[
        str,
        list[SourceStorageRealization | SourceDerivedRealization],
    ] = {}
    for realization in storage_realizations.realizations:
        output_record_id = (
            realization.output_record_id
            if isinstance(realization, SourceStorageRealization)
            else realization.output_record_id
        )
        if output_record_id not in records_by_id:
            raise ValueError("storage realization references an unknown output record")
        record_realizations = realizations_by_record.get(output_record_id)
        if record_realizations is None:
            record_realizations = []
            realizations_by_record[output_record_id] = record_realizations
        record_realizations.append(realization)
    for record in records:
        realizations = tuple(realizations_by_record.get(record.record_id, ()))
        if record.source_mutability == SourceMutability.ABSENT:
            if realizations:
                raise ValueError("absent record cannot have a storage realization")
            continue
        if record.provenance == SourceRecordProvenance.BACKEND_DERIVED:
            if len(realizations) != 1 or not isinstance(
                realizations[0], SourceDerivedRealization
            ):
                raise ValueError(
                    "backend-derived record requires one zero-raw derivation witness"
                )
        elif not realizations or any(
            not isinstance(realization, SourceStorageRealization)
            for realization in realizations
        ):
            raise ValueError(
                "present source record requires a native storage realization"
            )
        for realization in realizations:
            if (
                realization.output_dtype != record.dtype
                or realization.output_shape != record.shape
                or realization.output_numeric_encoding != record.numeric_encoding
            ):
                raise ValueError(
                    "storage realization output differs from normalized source view"
                )


_RUNTIME_DISCOVERY_RECORD_TYPES: frozenset[type[object]] = frozenset(
    {
        DiscoveryCompletenessReceipt,
        DiscoveryContribution,
        EvidenceSource,
        ExpectedContributorAuthority,
        GraphDiscoveryPartition,
        SourceDerivedRealization,
        SourceDiscoveryInventory,
        SourceDiscoveryRecord,
        SourceLiteralAxisExtent,
        SourceNormalizationContract,
        SourceNormalizedAxisExtent,
        SourceNormalizerManifest,
        SourcePhysicalAxisSpec,
        SourceProducerFingerprint,
        SourceSchemaId,
        SourceStorageComponent,
        SourceStorageRealization,
        SourceStorageRealizationInventory,
    }
)
_RUNTIME_DISCOVERY_ENUM_TYPES: frozenset[type[StrEnum]] = frozenset(
    {
        CanonicalSourceDType,
        DiscoveryRequestKind,
        EvidenceSourceKind,
        SourceExtentRounding,
        SourceMutability,
        SourceNormalizationKind,
        SourcePaddingSemantics,
        SourceRecordProvenance,
    }
)
_RUNTIME_DISCOVERY_SCALAR_TYPES: frozenset[type[object]] = frozenset(
    {bool, float, int, str, type(None)}
)
_RUNTIME_DISCOVERY_CONTAINER_TYPES: frozenset[type[object]] = frozenset(
    {
        DiscoveryContribution,
        GraphDiscoveryPartition,
        SourceDiscoveryInventory,
    }
)


def _validate_exact_runtime_discovery_tree(value: object) -> None:
    active_ids: set[int] = set()
    completed_ids: set[int] = set()

    def validate(item: object) -> None:
        item_type = type(item)
        if item_type in _RUNTIME_DISCOVERY_SCALAR_TYPES:
            if item_type is float and not isfinite(cast(float, item)):
                raise ValueError("runtime discovery floats must be finite")
            return
        if item_type in _RUNTIME_DISCOVERY_ENUM_TYPES:
            _require_registered_enum_member(
                item,
                cast(type[StrEnum], item_type),
                "runtime discovery enum",
            )
            return
        is_tuple = item_type is tuple
        is_record = is_dataclass(item) and not isinstance(item, type)
        if is_record:
            if item_type not in _RUNTIME_DISCOVERY_RECORD_TYPES:
                raise TypeError(
                    "runtime discovery contains a non-exact runtime discovery "
                    f"record: {item_type.__name__}"
                )
        elif not is_tuple:
            raise TypeError(
                "runtime discovery contains a non-exact runtime discovery "
                f"value: {item_type.__name__}"
            )

        identity = id(item)
        if identity in active_ids:
            raise ValueError("runtime discovery transport tree contains a cycle")
        if identity in completed_ids:
            return
        active_ids.add(identity)
        try:
            children = (
                tuple.__iter__(cast(tuple[object, ...], item))
                if is_tuple
                else (
                    getattr(item, record_field.name)
                    for record_field in fields(cast(Any, item))
                )
            )
            for child in children:
                validate(child)
            if item_type is EvidenceSource:
                cast(EvidenceSource, item).__post_init__()
            elif item_type is SourceSchemaId:
                cast(SourceSchemaId, item).__post_init__()
            elif item_type is SourceProducerFingerprint:
                _validate_exact_producer_fingerprint(item)
            elif item_type is ExpectedContributorAuthority:
                _validate_exact_contributor_authority(item)
            elif item_type is SourceDiscoveryRecord:
                cast(SourceDiscoveryRecord, item).__post_init__()
            elif item_type is DiscoveryCompletenessReceipt:
                cast(DiscoveryCompletenessReceipt, item).__post_init__()
            elif item_type in _RUNTIME_DISCOVERY_CONTAINER_TYPES:
                record_fields = fields(cast(Any, item))
                reconstructed = cast(Any, item_type)(
                    **{
                        record_field.name: getattr(item, record_field.name)
                        for record_field in record_fields
                        if record_field.init
                    }
                )
                if reconstructed != item:
                    raise ValueError(
                        "runtime discovery contains a noncanonical "
                        f"{item_type.__name__}"
                    )
        finally:
            active_ids.remove(identity)
            completed_ids.add(identity)

    validate(value)


def _select_partition_request(
    graph_input: GraphTopologyInput | None,
    runtime_request: RuntimeGraphSourceRequest | None,
) -> GraphTopologyInput | RuntimeGraphSourceRequest:
    if (graph_input is None) == (runtime_request is None):
        raise ValueError(
            "partition assembly requires exactly one graph_input or runtime_request"
        )
    if runtime_request is not None:
        return _validate_runtime_source_request_snapshot(runtime_request)
    if not isinstance(graph_input, GraphTopologyInput):
        raise TypeError("graph_input must be GraphTopologyInput")
    return graph_input


def _request_receipt_identity(
    request: GraphTopologyInput | RuntimeGraphSourceRequest,
) -> tuple[DiscoveryRequestKind, str]:
    if isinstance(request, RuntimeGraphSourceRequest):
        verified = _validate_runtime_source_request_snapshot(request)
        return (
            DiscoveryRequestKind.RUNTIME_GRAPH_SOURCE_REQUEST,
            verified.runtime_source_request_digest,
        )
    if not isinstance(request, GraphTopologyInput):  # pragma: no cover - typed union
        raise TypeError("source request has an unsupported type")
    return (
        DiscoveryRequestKind.GRAPH_TOPOLOGY_INPUT,
        graph_input_identity_digest(request),
    )


def _assemble_graph_discovery_partition(
    *,
    graph_input: GraphTopologyInput | None = None,
    runtime_request: RuntimeGraphSourceRequest | None = None,
    expected_contributors: ExpectedContributorSet,
    contributions: Sequence[DiscoveryContribution],
) -> GraphDiscoveryPartition:
    """Validate a complete contribution union and strip contributor identities."""
    request = _select_partition_request(graph_input, runtime_request)
    if type(request) is RuntimeGraphSourceRequest:
        _validate_exact_expected_contributor_set(expected_contributors)
    elif not isinstance(expected_contributors, ExpectedContributorSet):
        raise TypeError("expected_contributors must be ExpectedContributorSet")
    contribution_tuple = _snapshot_sequence(
        contributions,
        "discovery contributions",
    )
    if type(request) is RuntimeGraphSourceRequest:
        if any(
            type(contribution) is not DiscoveryContribution
            for contribution in contribution_tuple
        ):
            raise TypeError(
                "runtime contributions must contain exact DiscoveryContribution records"
            )
        for contribution in contribution_tuple:
            _validate_exact_runtime_discovery_tree(contribution)
            validate_source_storage_realization_inventory(
                contribution.storage_realizations
            )
    elif any(
        not isinstance(contribution, DiscoveryContribution)
        for contribution in contribution_tuple
    ):
        raise TypeError("contributions must contain DiscoveryContribution records")
    expected_authority = (
        _authority_from_exact_expected_contributor_set(expected_contributors)
        if type(request) is RuntimeGraphSourceRequest
        else expected_contributors.to_authority()
    )
    if request.expected_contributor_authority != expected_authority:
        label = (
            "runtime request"
            if type(request) is RuntimeGraphSourceRequest
            else "graph input"
        )
        raise ValueError(f"{label} expected contributor authority mismatch")

    contributor_ids = tuple(
        contribution.contributor_id for contribution in contribution_tuple
    )
    if len(contributor_ids) != len(set(contributor_ids)):
        raise ValueError("duplicate discovery contributor")
    observed = set(contributor_ids)
    expected = set(expected_contributors.contributor_ids)
    missing = expected - observed
    if missing:
        raise ValueError(f"missing discovery contributor: {sorted(missing)[0]}")
    unexpected = observed - expected
    if unexpected:
        raise ValueError(f"unexpected discovery contributor: {sorted(unexpected)[0]}")

    graph_id = request.declaration.graph_instance_id
    if any(
        contribution.graph_instance_id != graph_id
        for contribution in contribution_tuple
    ):
        raise ValueError("discovery contribution graph mismatch")
    if any(
        contribution.producer_fingerprint != request.source_producer_fingerprint
        for contribution in contribution_tuple
    ):
        raise ValueError("discovery contribution producer fingerprint mismatch")
    records = _validate_record_universe(
        graph_id,
        tuple(
            record
            for contribution in contribution_tuple
            for record in contribution.records
        ),
    )
    storage_realizations = _merge_storage_realization_inventories(
        graph_id,
        contribution_tuple,
    )
    if (
        source_normalizer_manifest_digest(storage_realizations.normalizer_manifest)
        != request.source_producer_fingerprint.normalization_contract_digest
    ):
        raise ValueError(
            "storage normalizer manifest differs from producer fingerprint"
        )
    _validate_storage_realization_coverage(
        graph_id,
        records,
        storage_realizations,
    )
    request_kind, request_digest = _request_receipt_identity(request)
    receipt = DiscoveryCompletenessReceipt(
        graph_instance_id=graph_id,
        producer_fingerprint_digest=_fingerprint_digest(
            request.source_producer_fingerprint
        ),
        observed_contributor_set_digest=_contributor_set_digest(
            tuple(sorted(contributor_ids))
        ),
        observed_contributor_count=len(contributor_ids),
        source_set_digest=_source_set_digest(records),
        source_count=len(records),
        canonical_records_digest=_records_digest(records),
        storage_realization_set_digest=source_storage_inventory_digest(
            storage_realizations
        ),
        storage_realization_count=len(storage_realizations.realizations),
        request_kind=request_kind,
        request_digest=request_digest,
    )
    return GraphDiscoveryPartition(
        graph_instance_id=graph_id,
        producer_fingerprint=request.source_producer_fingerprint,
        expected_contributor_authority=expected_authority,
        records=records,
        storage_realizations=storage_realizations,
        completeness_receipt=receipt,
    )


def assemble_graph_discovery_partition(
    *,
    graph_input: GraphTopologyInput,
    expected_contributors: ExpectedContributorSet,
    contributions: Sequence[DiscoveryContribution],
) -> GraphDiscoveryPartition:
    """Compatibility assembly for the pre-Phase-1-bound discovery path."""
    return _assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected_contributors,
        contributions=contributions,
    )


def assemble_runtime_graph_discovery_partition(
    *,
    runtime_request: RuntimeGraphSourceRequest,
    expected_contributors: ExpectedContributorSet,
    contributions: Sequence[DiscoveryContribution],
) -> GraphDiscoveryPartition:
    """Assemble one exact Phase-1-bound runtime discovery partition."""
    return _assemble_graph_discovery_partition(
        runtime_request=runtime_request,
        expected_contributors=expected_contributors,
        contributions=contributions,
    )


def _validate_discovery_inventory(
    graph_inputs: Sequence[GraphTopologyInput | RuntimeGraphSourceRequest],
    source_discovery: SourceDiscoveryInventory,
    expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
) -> SourceDiscoveryInventory:
    """Revalidate every independent discovery commitment before classification."""
    inputs = _snapshot_sequence(graph_inputs, "graph inputs")
    if not inputs:
        raise ValueError("source request set must not be empty")
    if any(
        not isinstance(graph_input, (GraphTopologyInput, RuntimeGraphSourceRequest))
        for graph_input in inputs
    ):
        raise TypeError("graph_inputs must contain source request records")
    runtime_input_count = sum(
        isinstance(graph_input, RuntimeGraphSourceRequest) for graph_input in inputs
    )
    if runtime_input_count not in (0, len(inputs)):
        raise ValueError("source request set must not mix legacy and runtime requests")
    if not isinstance(expected_contributors_by_graph, Mapping):
        raise TypeError("expected_contributors_by_graph must be a mapping")
    trusted_contributors = (
        dict(expected_contributors_by_graph)
        if runtime_input_count
        else expected_contributors_by_graph
    )
    if runtime_input_count and any(
        type(graph_id) is not str for graph_id in trusted_contributors
    ):
        raise TypeError("trusted runtime graph IDs must be exact strings")
    for graph_input in inputs:
        if isinstance(graph_input, RuntimeGraphSourceRequest):
            _select_partition_request(None, graph_input)
    if not isinstance(source_discovery, SourceDiscoveryInventory):
        raise TypeError("source_discovery must be SourceDiscoveryInventory")
    if runtime_input_count:
        _validate_exact_runtime_discovery_tree(source_discovery)
        canonical_partition_order = tuple(
            sorted(
                source_discovery.partitions,
                key=lambda partition: _graph_sort_key(partition.graph_instance_id),
            )
        )
        if source_discovery.partitions != canonical_partition_order:
            raise ValueError("runtime discovery partitions are not canonically ordered")
        selection_identities = {
            (
                graph_input.semantic_structure_digest,
                graph_input.selection_group_id,
            )
            for graph_input in inputs
            if type(graph_input) is RuntimeGraphSourceRequest
        }
        if len(selection_identities) != 1:
            raise ValueError(
                "runtime requests must share one Phase 1 selection identity"
            )
    graph_ids = tuple(
        graph_input.declaration.graph_instance_id for graph_input in inputs
    )
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("duplicate graph topology input declaration")
    declared = set(graph_ids)

    trusted_graph_ids = set(trusted_contributors)
    missing_trusted = declared - trusted_graph_ids
    if missing_trusted:
        raise ValueError(
            f"missing trusted expected contributor set: {sorted(missing_trusted)[0]}"
        )
    undeclared_trusted = trusted_graph_ids - declared
    if undeclared_trusted:
        raise ValueError(
            "undeclared trusted expected contributor set: "
            f"{sorted(undeclared_trusted)[0]}"
        )
    if any(
        not isinstance(trusted_contributors[graph_id], ExpectedContributorSet)
        for graph_id in graph_ids
    ):
        raise TypeError("trusted mapping values must be ExpectedContributorSet")

    partition_graph_ids = tuple(
        partition.graph_instance_id for partition in source_discovery.partitions
    )
    if len(partition_graph_ids) != len(set(partition_graph_ids)):
        raise ValueError("duplicate source discovery graph partition")
    discovered = set(partition_graph_ids)
    missing_partitions = declared - discovered
    if missing_partitions:
        raise ValueError(
            f"missing source discovery graph partition: {sorted(missing_partitions)[0]}"
        )
    undeclared_partitions = discovered - declared
    if undeclared_partitions:
        raise ValueError(
            "undeclared source discovery graph partition: "
            f"{sorted(undeclared_partitions)[0]}"
        )
    record_ids = tuple(
        record.record_id
        for partition in source_discovery.partitions
        for record in partition.records
    )
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate source discovery record ID across graph partitions")

    inputs_by_graph = {
        graph_input.declaration.graph_instance_id: graph_input for graph_input in inputs
    }
    partitions_by_graph = {
        partition.graph_instance_id: partition
        for partition in source_discovery.partitions
    }
    for graph_id in sorted(declared, key=_graph_sort_key):
        graph_input = inputs_by_graph[graph_id]
        partition = partitions_by_graph[graph_id]
        expected_set = trusted_contributors[graph_id]
        if runtime_input_count:
            _validate_exact_expected_contributor_set(expected_set)
        expected_authority = (
            _authority_from_exact_expected_contributor_set(expected_set)
            if runtime_input_count
            else expected_set.to_authority()
        )
        if graph_input.expected_contributor_authority != expected_authority:
            raise ValueError(
                "graph input differs from trusted expected contributor authority"
            )
        if partition.expected_contributor_authority != expected_authority:
            raise ValueError(
                "partition differs from trusted expected contributor authority"
            )
        if partition.producer_fingerprint != graph_input.source_producer_fingerprint:
            raise ValueError("partition producer fingerprint mismatch")
        records = _validate_record_universe(graph_id, partition.records)
        if runtime_input_count and partition.records != records:
            raise ValueError("runtime discovery records are not canonically ordered")
        storage_realizations = partition.storage_realizations
        validate_source_storage_realization_inventory(storage_realizations)
        if (
            source_normalizer_manifest_digest(storage_realizations.normalizer_manifest)
            != graph_input.source_producer_fingerprint.normalization_contract_digest
        ):
            raise ValueError(
                "partition normalizer manifest differs from producer fingerprint"
            )
        _validate_storage_realization_coverage(
            graph_id,
            records,
            storage_realizations,
        )
        receipt = partition.completeness_receipt
        receipt.__post_init__()
        if receipt.graph_instance_id != graph_id:
            raise ValueError("receipt graph_instance_id mismatch")
        if receipt.producer_fingerprint_digest != _fingerprint_digest(
            partition.producer_fingerprint
        ):
            raise ValueError("receipt producer fingerprint digest mismatch")
        if (
            receipt.observed_contributor_set_digest
            != expected_authority.contributor_set_digest
            or receipt.observed_contributor_count
            != expected_authority.contributor_count
        ):
            raise ValueError("receipt observed contributor authority mismatch")
        if receipt.source_count != len(records):
            raise ValueError("receipt source count mismatch")
        if receipt.source_set_digest != _source_set_digest(records):
            raise ValueError("receipt source set digest mismatch")
        if receipt.canonical_records_digest != _records_digest(records):
            raise ValueError("receipt canonical records digest mismatch")
        if receipt.storage_realization_count != len(storage_realizations.realizations):
            raise ValueError("receipt storage realization count mismatch")
        if receipt.storage_realization_set_digest != source_storage_inventory_digest(
            storage_realizations
        ):
            raise ValueError("receipt storage realization set digest mismatch")
        request_kind, request_digest = _request_receipt_identity(graph_input)
        if receipt.request_kind is not request_kind:
            raise ValueError("receipt source request kind mismatch")
        if receipt.request_digest != request_digest:
            label = (
                "runtime source request"
                if request_kind is DiscoveryRequestKind.RUNTIME_GRAPH_SOURCE_REQUEST
                else "graph input"
            )
            raise ValueError(f"receipt {label} digest mismatch")
    return source_discovery


def validate_discovery_inventory(
    graph_inputs: Sequence[GraphTopologyInput],
    source_discovery: SourceDiscoveryInventory,
    expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
) -> SourceDiscoveryInventory:
    """Compatibility validation for the pre-Phase-1-bound discovery path."""
    inputs = _snapshot_sequence(graph_inputs, "graph inputs")
    if any(not isinstance(item, GraphTopologyInput) for item in inputs):
        raise TypeError("graph_inputs must contain GraphTopologyInput records")
    return _validate_discovery_inventory(
        inputs,
        source_discovery,
        expected_contributors_by_graph,
    )


def validate_runtime_discovery_inventory(
    runtime_requests: Sequence[RuntimeGraphSourceRequest],
    source_discovery: SourceDiscoveryInventory,
    expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
) -> SourceDiscoveryInventory:
    """Validate and return one exact Phase-1-bound runtime inventory."""
    requests = _snapshot_sequence(runtime_requests, "runtime requests")
    if any(type(item) is not RuntimeGraphSourceRequest for item in requests):
        raise TypeError(
            "runtime_requests must contain exact RuntimeGraphSourceRequest records"
        )
    return _validate_discovery_inventory(
        requests,
        source_discovery,
        expected_contributors_by_graph,
    )
