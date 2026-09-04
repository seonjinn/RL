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
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from math import isfinite
import re

from nemo_rl.precision_policy.semantic import (
    EvidenceSource,
    ExpectedGraphDeclaration,
    ImmutableAuxiliaryEvidence,
    SourceMutability,
)
from nemo_rl.precision_policy.source_dtype import CanonicalSourceDType


_SOURCE_SCHEMA_PATTERN = re.compile(r"[a-z][a-z0-9-]*(?:\.[a-z0-9-]+)+\.v[1-9][0-9]*")
_IMMUTABLE_REVISION_PATTERN = re.compile(
    r"(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256:[0-9a-f]{64})"
)
_SHA256_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


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
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not isfinite(value):  # pragma: no cover - rejected during snapshot
            raise ValueError("configuration floats must be finite")
        return {"type": "float", "value": 0.0 if value == 0.0 else value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "entries": [
                {"key": key, "value": _typed_config_payload(value[key])}
                for key in sorted(value)
            ],
        }
    if isinstance(value, tuple):
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


def _authority_payload(
    authority: ExpectedContributorAuthority,
) -> dict[str, object]:
    return {
        "type": "expected_contributor_authority",
        "contributor_set_digest": authority.contributor_set_digest,
        "contributor_count": authority.contributor_count,
        "authority": _evidence_payload(authority.authority),
    }


def _validate_authority_evidence_is_id_free(
    contributor_ids: tuple[str, ...],
    evidence: EvidenceSource,
) -> None:
    for contributor_id in contributor_ids:
        if contributor_id in evidence.locator or contributor_id in evidence.digest:
            raise ValueError("authority evidence must not contain a contributor ID")


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
        contributor_ids = tuple(self.contributor_ids)
        if not contributor_ids:
            raise ValueError("expected contributor set must be non-empty")
        for contributor_id in contributor_ids:
            _require_text(contributor_id, "contributor ID")
        if len(contributor_ids) != len(set(contributor_ids)):
            raise ValueError("expected contributor IDs must be duplicate-free")
        if not isinstance(self.authority, EvidenceSource):
            raise TypeError("expected contributor authority must be EvidenceSource")
        object.__setattr__(self, "contributor_ids", tuple(sorted(contributor_ids)))

    def to_authority(self) -> ExpectedContributorAuthority:
        """Return the ID-free canonical commitment to this trusted set."""
        _validate_authority_evidence_is_id_free(
            self.contributor_ids,
            self.authority,
        )
        return ExpectedContributorAuthority(
            contributor_set_digest=_contributor_set_digest(self.contributor_ids),
            contributor_count=len(self.contributor_ids),
            authority=self.authority,
        )


class SourceRecordProvenance(StrEnum):
    """Raw source authority recorded before semantic classification."""

    TRAINING_RUNTIME = "training_runtime"
    CHECKPOINT_STORAGE = "checkpoint_storage"
    BACKEND_DERIVED = "backend_derived"
    TIED_STORAGE = "tied_storage"
    SYNCHRONIZED_REPLICA = "synchronized_replica"


@dataclass(frozen=True, slots=True)
class SourceDiscoveryRecord:
    """Frozen native tensor metadata with no semantic or runtime binding."""

    record_id: str
    graph_instance_id: str
    source_native_name: str | None
    source_native_owner_id: str | None
    dtype: CanonicalSourceDType
    shape: tuple[int, ...]
    provenance: SourceRecordProvenance
    provenance_evidence: EvidenceSource
    source_mutability: SourceMutability
    mutability_evidence: EvidenceSource

    def __post_init__(self) -> None:
        _require_text(self.record_id, "source discovery record_id")
        _require_text(self.graph_instance_id, "source discovery graph_instance_id")
        object.__setattr__(self, "shape", tuple(self.shape))
        if not isinstance(self.dtype, CanonicalSourceDType):
            raise TypeError("source discovery dtype must be CanonicalSourceDType")
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

    def __post_init__(self) -> None:
        _require_text(self.contributor_id, "contributor ID")
        _require_text(self.graph_instance_id, "contribution graph_instance_id")
        if not isinstance(self.producer_fingerprint, SourceProducerFingerprint):
            raise TypeError(
                "contribution fingerprint must be SourceProducerFingerprint"
            )
        records = tuple(self.records)
        if any(not isinstance(record, SourceDiscoveryRecord) for record in records):
            raise TypeError("contribution records must be SourceDiscoveryRecord values")
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
    graph_input_digest: str

    def __post_init__(self) -> None:
        _require_text(self.graph_instance_id, "receipt graph_instance_id")
        for field_name in (
            "producer_fingerprint_digest",
            "observed_contributor_set_digest",
            "source_set_digest",
            "canonical_records_digest",
            "graph_input_digest",
        ):
            _require_sha256_digest(
                getattr(self, field_name),
                field_name.replace("_", " "),
            )
        _require_positive_int(
            self.observed_contributor_count,
            "observed contributor count",
        )
        _require_positive_int(self.source_count, "source count")


@dataclass(frozen=True, slots=True)
class GraphDiscoveryPartition:
    """ID-free, canonical source universe for exactly one graph."""

    graph_instance_id: str
    producer_fingerprint: SourceProducerFingerprint
    expected_contributor_authority: ExpectedContributorAuthority
    records: tuple[SourceDiscoveryRecord, ...]
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
        records = tuple(self.records)
        if any(not isinstance(record, SourceDiscoveryRecord) for record in records):
            raise TypeError("partition records must be SourceDiscoveryRecord values")
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
        partitions = tuple(self.partitions)
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


def assemble_graph_discovery_partition(
    *,
    graph_input: GraphTopologyInput,
    expected_contributors: ExpectedContributorSet,
    contributions: Sequence[DiscoveryContribution],
) -> GraphDiscoveryPartition:
    """Validate a complete contribution union and strip contributor identities."""
    if not isinstance(graph_input, GraphTopologyInput):
        raise TypeError("graph_input must be GraphTopologyInput")
    if not isinstance(expected_contributors, ExpectedContributorSet):
        raise TypeError("expected_contributors must be ExpectedContributorSet")
    contribution_tuple = tuple(contributions)
    if any(
        not isinstance(contribution, DiscoveryContribution)
        for contribution in contribution_tuple
    ):
        raise TypeError("contributions must contain DiscoveryContribution records")
    expected_authority = expected_contributors.to_authority()
    if graph_input.expected_contributor_authority != expected_authority:
        raise ValueError("graph input expected contributor authority mismatch")

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

    graph_id = graph_input.declaration.graph_instance_id
    if any(
        contribution.graph_instance_id != graph_id
        for contribution in contribution_tuple
    ):
        raise ValueError("discovery contribution graph mismatch")
    if any(
        contribution.producer_fingerprint != graph_input.source_producer_fingerprint
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
    receipt = DiscoveryCompletenessReceipt(
        graph_instance_id=graph_id,
        producer_fingerprint_digest=_fingerprint_digest(
            graph_input.source_producer_fingerprint
        ),
        observed_contributor_set_digest=_contributor_set_digest(
            tuple(sorted(contributor_ids))
        ),
        observed_contributor_count=len(contributor_ids),
        source_set_digest=_source_set_digest(records),
        source_count=len(records),
        canonical_records_digest=_records_digest(records),
        graph_input_digest=graph_input_identity_digest(graph_input),
    )
    return GraphDiscoveryPartition(
        graph_instance_id=graph_id,
        producer_fingerprint=graph_input.source_producer_fingerprint,
        expected_contributor_authority=expected_authority,
        records=records,
        completeness_receipt=receipt,
    )


def validate_discovery_inventory(
    graph_inputs: Sequence[GraphTopologyInput],
    source_discovery: SourceDiscoveryInventory,
    expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
) -> None:
    """Revalidate every independent discovery commitment before classification."""
    inputs = tuple(graph_inputs)
    if any(not isinstance(graph_input, GraphTopologyInput) for graph_input in inputs):
        raise TypeError("graph_inputs must contain GraphTopologyInput records")
    if not isinstance(source_discovery, SourceDiscoveryInventory):
        raise TypeError("source_discovery must be SourceDiscoveryInventory")
    if not isinstance(expected_contributors_by_graph, Mapping):
        raise TypeError("expected_contributors_by_graph must be a mapping")
    graph_ids = tuple(
        graph_input.declaration.graph_instance_id for graph_input in inputs
    )
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("duplicate graph topology input declaration")
    declared = set(graph_ids)

    trusted_graph_ids = set(expected_contributors_by_graph)
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
        not isinstance(expected_contributors_by_graph[graph_id], ExpectedContributorSet)
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
        expected_set = expected_contributors_by_graph[graph_id]
        expected_authority = expected_set.to_authority()
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
        receipt = partition.completeness_receipt
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
        if receipt.graph_input_digest != graph_input_identity_digest(graph_input):
            raise ValueError("receipt graph input digest mismatch")
