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

"""Selection-bound Phase 2 runtime source discovery contracts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TypeVar, cast

from nemo_rl.precision_policy.compiler import (
    ActiveRuntimeSourceProvenanceAnchor,
    CompiledPrecisionIntentGroup,
    CompiledPrecisionSelectionGroup,
    _bind_compiled_precision_intents,
    _issue_runtime_source_evidence_receipt,
    validate_compiled_precision_selection_group,
)
from nemo_rl.precision_policy.discovery_producers import SourceMetadataProducer
from nemo_rl.precision_policy.semantic import (
    EvidenceSource,
    GraphProvenance,
    ResolvedGraphTopology,
    canonical_model_config_digest,
)
from nemo_rl.precision_policy.source_discovery import (
    DiscoveryCompletenessReceipt,
    DiscoveryContribution,
    ExpectedContributorSet,
    GraphDiscoveryPartition,
    RuntimeGraphSourceRequest,
    SourceDiscoveryInventory,
    SourceProducerFingerprint,
    SourceSchemaId,
    assemble_runtime_graph_discovery_partition,
    derive_expected_contributor_authority,
    runtime_source_request_identity_digest,
    source_producer_fingerprint_identity_digest,
    validate_runtime_discovery_inventory,
)
from nemo_rl.precision_policy.topology import (
    RuntimeSourceProvenance,
    classify_validated_runtime_semantic_topology,
)

_SequenceItemT = TypeVar("_SequenceItemT")
_SCALAR_SEQUENCE_TYPES = (str, bytes, bytearray, memoryview)


def _graph_sort_key(graph_instance_id: str) -> tuple[int, str]:
    return (0 if graph_instance_id == "main" else 1, graph_instance_id)


def _snapshot_sequence(
    values: Sequence[_SequenceItemT],
    label: str,
) -> tuple[_SequenceItemT, ...]:
    if isinstance(values, _SCALAR_SEQUENCE_TYPES) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a non-scalar sequence")
    return tuple(values)


_DiscoverContributions = Callable[
    [RuntimeGraphSourceRequest, ExpectedContributorSet],
    Sequence[DiscoveryContribution],
]


@dataclass(frozen=True, slots=True)
class _RuntimeSourceProducerSnapshot:
    producer: SourceMetadataProducer
    producer_id: str
    schema_id: SourceSchemaId
    fingerprint_digest: str
    discover_contributions: _DiscoverContributions


def _snapshot_runtime_source_producer_mapping(
    producers_by_graph: Mapping[str, SourceMetadataProducer],
) -> tuple[tuple[str, SourceMetadataProducer], ...]:
    if not isinstance(producers_by_graph, Mapping):
        raise TypeError("producers_by_graph must be a mapping")
    entries = tuple(producers_by_graph.items())
    normalized: list[tuple[str, SourceMetadataProducer]] = []
    for entry in entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise TypeError("producer bindings must be exact graph/producer tuples")
        graph_instance_id, producer = entry
        if type(graph_instance_id) is not str:
            raise TypeError("producer binding graph IDs must be exact strings")
        if not graph_instance_id or graph_instance_id != graph_instance_id.strip():
            raise ValueError("producer binding graph IDs must be exact non-empty text")
        normalized.append((graph_instance_id, producer))
    graph_ids = tuple(graph_instance_id for graph_instance_id, _ in normalized)
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("duplicate runtime source producer graph binding")
    return tuple(sorted(normalized, key=lambda item: _graph_sort_key(item[0])))


def _snapshot_runtime_source_producer(
    producer: SourceMetadataProducer,
) -> _RuntimeSourceProducerSnapshot:
    try:
        producer_id = producer.producer_id
        schema_id = producer.schema_id
        fingerprint_factory = producer.fingerprint
        discover_contributions = producer.discover_contributions
    except AttributeError as error:
        raise TypeError("producer does not implement SourceMetadataProducer") from error
    if type(producer_id) is not str:
        raise TypeError("producer_id must be an exact string")
    if not producer_id or producer_id != producer_id.strip():
        raise ValueError("producer_id must be exact non-empty text")
    if type(schema_id) is not SourceSchemaId:
        raise TypeError("schema_id must be an exact SourceSchemaId")
    if type(schema_id.value) is not str:
        raise TypeError("schema_id.value must be an exact string")
    schema_id.__post_init__()
    if not callable(fingerprint_factory):
        raise TypeError("producer fingerprint must be callable")
    if not callable(discover_contributions):
        raise TypeError("producer discover_contributions must be callable")
    fingerprint = fingerprint_factory()
    fingerprint_digest = source_producer_fingerprint_identity_digest(fingerprint)
    if producer_id != fingerprint.producer_implementation_id:
        raise ValueError("producer_id differs from producer fingerprint")
    if schema_id != fingerprint.schema_id:
        raise ValueError("schema_id differs from producer fingerprint")
    return _RuntimeSourceProducerSnapshot(
        producer=producer,
        producer_id=producer_id,
        schema_id=schema_id,
        fingerprint_digest=fingerprint_digest,
        discover_contributions=discover_contributions,
    )


@dataclass(frozen=True, slots=True)
class RuntimeGraphSourceContext:
    """Ephemeral caller-supplied runtime facts for one graph."""

    graph_instance_id: str
    model_config: Mapping[str, object]
    source_producer_fingerprint: SourceProducerFingerprint
    expected_contributors: ExpectedContributorSet
    source_identity: EvidenceSource
    artifact_identity: EvidenceSource
    source_allocation_generation: str

    def __post_init__(self) -> None:
        if type(self.graph_instance_id) is not str:
            raise TypeError("graph_instance_id must be an exact string")
        if not self.graph_instance_id or self.graph_instance_id != (
            self.graph_instance_id.strip()
        ):
            raise ValueError("graph_instance_id must be exact non-empty text")
        if not isinstance(self.model_config, Mapping):
            raise TypeError("model_config must be a mapping")
        if type(self.source_producer_fingerprint) is not SourceProducerFingerprint:
            raise TypeError(
                "source_producer_fingerprint must be an exact SourceProducerFingerprint"
            )
        derive_expected_contributor_authority(self.expected_contributors)
        for value, label in (
            (self.source_identity, "source_identity"),
            (self.artifact_identity, "artifact_identity"),
        ):
            if type(value) is not EvidenceSource:
                raise TypeError(f"{label} must be an exact EvidenceSource")
        if type(self.source_allocation_generation) is not str:
            raise TypeError("source_allocation_generation must be an exact string")
        if not self.source_allocation_generation or (
            self.source_allocation_generation
            != self.source_allocation_generation.strip()
        ):
            raise ValueError(
                "source_allocation_generation must be exact non-empty text"
            )


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def _authority_payload(expected: ExpectedContributorSet) -> dict[str, object]:
    authority = derive_expected_contributor_authority(expected)
    return {
        "contributor_set_digest": authority.contributor_set_digest,
        "contributor_count": authority.contributor_count,
        "authority": {
            "kind": authority.authority.kind.value,
            "locator": authority.authority.locator,
            "digest": authority.authority.digest,
        },
    }


def _runtime_graph_id(request: RuntimeGraphSourceRequest) -> str:
    return request.declaration.graph_instance_id


def _snapshot_trusted_mapping(
    values: Mapping[str, ExpectedContributorSet],
) -> tuple[tuple[str, ExpectedContributorSet], ...]:
    if not isinstance(values, Mapping):
        raise TypeError("trusted_expected_contributors must be a mapping")
    entries = tuple(values.items())
    return _canonical_trusted_entries(entries)


def _canonical_trusted_entries(
    values: Sequence[tuple[str, ExpectedContributorSet]],
) -> tuple[tuple[str, ExpectedContributorSet], ...]:
    entries = _snapshot_sequence(values, "trusted expected contributor entries")
    normalized: list[tuple[str, ExpectedContributorSet]] = []
    for entry in entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise TypeError(
                "trusted expected contributor entries must be exact graph/set tuples"
            )
        graph_instance_id, expected = entry
        if type(graph_instance_id) is not str:
            raise TypeError("trusted runtime graph IDs must be exact strings")
        derive_expected_contributor_authority(expected)
        normalized.append((graph_instance_id, expected))
    graph_ids = tuple(graph_instance_id for graph_instance_id, _ in normalized)
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("duplicate trusted runtime graph ID")
    return tuple(sorted(normalized, key=lambda item: _graph_sort_key(item[0])))


def _aggregate_request_digest(
    *,
    graph_requests: tuple[RuntimeGraphSourceRequest, ...],
    trusted_expected_contributors: tuple[tuple[str, ExpectedContributorSet], ...],
    semantic_structure_digest: str,
    selection_group_id: str,
) -> str:
    trusted_by_graph = dict(trusted_expected_contributors)
    return _canonical_digest(
        {
            "type": "runtime_source_discovery_request",
            "semantic_structure_digest": semantic_structure_digest,
            "selection_group_id": selection_group_id,
            "graph_requests": [
                {
                    "graph_instance_id": _runtime_graph_id(request),
                    "runtime_source_request_digest": (
                        request.runtime_source_request_digest
                    ),
                    "expected_contributor_authority": _authority_payload(
                        trusted_by_graph[_runtime_graph_id(request)]
                    ),
                }
                for request in graph_requests
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class RuntimeSourceDiscoveryRequest:
    """Complete runtime-source request set bound to one Phase 1 selection."""

    graph_requests: tuple[RuntimeGraphSourceRequest, ...]
    trusted_expected_contributors: tuple[tuple[str, ExpectedContributorSet], ...]
    semantic_structure_digest: str = field(init=False)
    selection_group_id: str = field(init=False)
    request_digest: str = field(init=False)

    def __post_init__(self) -> None:
        graph_requests = _snapshot_sequence(
            self.graph_requests,
            "runtime graph requests",
        )
        if not graph_requests:
            raise ValueError("runtime graph request set must not be empty")
        if any(
            type(request) is not RuntimeGraphSourceRequest for request in graph_requests
        ):
            raise TypeError(
                "graph_requests must contain exact RuntimeGraphSourceRequest records"
            )
        for request in graph_requests:
            runtime_source_request_identity_digest(request)
        graph_ids = tuple(_runtime_graph_id(request) for request in graph_requests)
        if len(graph_ids) != len(set(graph_ids)):
            raise ValueError("duplicate runtime graph request")
        trusted_entries = _canonical_trusted_entries(self.trusted_expected_contributors)
        trusted_by_graph = dict(trusted_entries)
        if set(graph_ids) != set(trusted_by_graph):
            raise ValueError(
                "runtime graph requests and trusted contributor sets differ"
            )
        for request in graph_requests:
            graph_id = _runtime_graph_id(request)
            expected_authority = derive_expected_contributor_authority(
                trusted_by_graph[graph_id]
            )
            if request.expected_contributor_authority != expected_authority:
                raise ValueError(
                    "runtime request differs from trusted contributor authority"
                )
        selection_identities = {
            (request.semantic_structure_digest, request.selection_group_id)
            for request in graph_requests
        }
        if len(selection_identities) != 1:
            raise ValueError("runtime graph requests must share one selection identity")
        semantic_structure_digest, selection_group_id = next(iter(selection_identities))
        canonical_requests = tuple(
            sorted(
                graph_requests,
                key=lambda request: _graph_sort_key(_runtime_graph_id(request)),
            )
        )
        object.__setattr__(self, "graph_requests", canonical_requests)
        object.__setattr__(self, "trusted_expected_contributors", trusted_entries)
        object.__setattr__(
            self,
            "semantic_structure_digest",
            semantic_structure_digest,
        )
        object.__setattr__(self, "selection_group_id", selection_group_id)
        object.__setattr__(
            self,
            "request_digest",
            _aggregate_request_digest(
                graph_requests=canonical_requests,
                trusted_expected_contributors=trusted_entries,
                semantic_structure_digest=semantic_structure_digest,
                selection_group_id=selection_group_id,
            ),
        )


def _validate_aggregate_request_snapshot(
    request: object,
) -> RuntimeSourceDiscoveryRequest:
    if type(request) is not RuntimeSourceDiscoveryRequest:
        raise TypeError("request must be an exact RuntimeSourceDiscoveryRequest")
    typed_request = cast(RuntimeSourceDiscoveryRequest, request)
    if type(typed_request.graph_requests) is not tuple:
        raise TypeError("graph_requests must be an exact tuple")
    if type(typed_request.trusted_expected_contributors) is not tuple:
        raise TypeError("trusted_expected_contributors must be an exact tuple")
    expected = RuntimeSourceDiscoveryRequest(
        graph_requests=typed_request.graph_requests,
        trusted_expected_contributors=typed_request.trusted_expected_contributors,
    )
    if typed_request.graph_requests != expected.graph_requests:
        raise ValueError("graph_requests are not in canonical order")
    if (
        typed_request.trusted_expected_contributors
        != expected.trusted_expected_contributors
    ):
        raise ValueError("trusted_expected_contributors are not canonical")
    for field_name in (
        "semantic_structure_digest",
        "selection_group_id",
        "request_digest",
    ):
        if type(getattr(typed_request, field_name)) is not str:
            raise TypeError(f"{field_name} must be an exact string")
        if getattr(typed_request, field_name) != getattr(expected, field_name):
            raise ValueError(f"{field_name} differs from its canonical derivation")
    return typed_request


def _validate_request_against_selection(
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
    *,
    graphs_by_id: Mapping[str, ResolvedGraphTopology] | None = None,
    request_snapshot_validated: bool = False,
    config_digests_validated: bool = False,
) -> RuntimeSourceDiscoveryRequest:
    if not request_snapshot_validated:
        request = _validate_aggregate_request_snapshot(request)
    if graphs_by_id is None:
        graphs_by_id = {
            graph.declaration.graph_instance_id: graph
            for graph in selection.topology.graphs
        }
    required_graph_ids = tuple(
        sorted(
            (
                graph_id
                for graph_id, graph in graphs_by_id.items()
                if graph.declaration.lifecycle.graph_provenance
                is GraphProvenance.TRAINING_RUNTIME
            ),
            key=_graph_sort_key,
        )
    )
    requests_by_graph = {
        _runtime_graph_id(graph_request): graph_request
        for graph_request in request.graph_requests
    }
    if tuple(requests_by_graph) != required_graph_ids:
        raise ValueError(
            "runtime graph request coverage differs from the Phase 1 lifecycle"
        )
    trusted_by_graph = dict(request.trusted_expected_contributors)
    if tuple(trusted_by_graph) != required_graph_ids:
        raise ValueError(
            "trusted runtime graph coverage differs from the Phase 1 lifecycle"
        )
    if request.semantic_structure_digest != selection.semantic_structure_digest:
        raise ValueError("semantic_structure_digest differs from selection")
    if request.selection_group_id != selection.selection_group_id:
        raise ValueError("selection_group_id differs from selection")
    for graph_id in required_graph_ids:
        graph = graphs_by_id[graph_id]
        graph_request = requests_by_graph[graph_id]
        if graph_request.resolved_graph != graph:
            raise ValueError(
                f"runtime request resolved graph differs from Phase 1 for {graph_id}"
            )
        if graph_request.declaration != graph.declaration:
            raise ValueError(
                f"runtime request declaration differs from Phase 1 for {graph_id}"
            )
        if graph_request.resolved_model_revision != graph.resolved_model_revision:
            raise ValueError(
                f"runtime request revision differs from Phase 1 for {graph_id}"
            )
        if (
            graph_request.semantic_structure_digest
            != selection.semantic_structure_digest
        ):
            raise ValueError("runtime request semantic_structure_digest mismatch")
        if graph_request.selection_group_id != selection.selection_group_id:
            raise ValueError("runtime request selection_group_id mismatch")
        expected_config_digest = graph.effective_model_config_digest
        if expected_config_digest is None:
            raise ValueError(
                "runtime graph request requires a Phase 1 effective model config digest"
            )
        if not config_digests_validated and (
            canonical_model_config_digest(graph_request.model_config)
            != expected_config_digest
        ):
            raise ValueError(
                f"runtime effective model config digest differs from Phase 1 for {graph_id}"
            )
        if graph_request.expected_contributor_authority != (
            derive_expected_contributor_authority(trusted_by_graph[graph_id])
        ):
            raise ValueError(
                f"runtime request authority differs from trusted set for {graph_id}"
            )
    return request


def validate_runtime_source_discovery_request(
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
) -> RuntimeSourceDiscoveryRequest:
    """Revalidate an aggregate runtime request against its exact selection."""
    validated_selection = validate_compiled_precision_selection_group(selection)
    return _validate_request_against_selection(validated_selection, request)


def build_runtime_source_discovery_request(
    *,
    selection: CompiledPrecisionSelectionGroup,
    graph_requests: Sequence[RuntimeGraphSourceRequest],
    trusted_expected_contributors: Mapping[str, ExpectedContributorSet],
) -> RuntimeSourceDiscoveryRequest:
    """Build the complete canonical runtime request set for one selection."""
    validated_selection = validate_compiled_precision_selection_group(selection)
    graph_request_snapshot = _snapshot_sequence(
        graph_requests,
        "runtime graph requests",
    )
    trusted_snapshot = _snapshot_trusted_mapping(trusted_expected_contributors)
    request = RuntimeSourceDiscoveryRequest(
        graph_requests=graph_request_snapshot,
        trusted_expected_contributors=trusted_snapshot,
    )
    return _validate_request_against_selection(
        validated_selection,
        request,
        request_snapshot_validated=True,
    )


def _snapshot_runtime_contexts(
    contexts: Mapping[str, RuntimeGraphSourceContext],
) -> tuple[tuple[str, RuntimeGraphSourceContext], ...]:
    if not isinstance(contexts, Mapping):
        raise TypeError("contexts must be a mapping")
    entries = tuple(contexts.items())
    normalized: list[tuple[str, RuntimeGraphSourceContext]] = []
    for graph_id, context in entries:
        if type(graph_id) is not str:
            raise TypeError("runtime context graph IDs must be exact strings")
        if type(context) is not RuntimeGraphSourceContext:
            raise TypeError(
                "contexts must contain exact RuntimeGraphSourceContext records"
            )
        if context.graph_instance_id != graph_id:
            raise ValueError("runtime context key differs from graph_instance_id")
        normalized.append((graph_id, context))
    graph_ids = tuple(graph_id for graph_id, _ in normalized)
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("duplicate runtime context graph ID")
    return tuple(sorted(normalized, key=lambda item: _graph_sort_key(item[0])))


def _build_graph_request_from_validated_selection(
    *,
    selection: CompiledPrecisionSelectionGroup,
    graph: ResolvedGraphTopology,
    context: RuntimeGraphSourceContext,
) -> RuntimeGraphSourceRequest:
    if (
        graph.declaration.lifecycle.graph_provenance
        is not GraphProvenance.TRAINING_RUNTIME
    ):
        raise ValueError("runtime source requests require a training-runtime graph")
    expected_config_digest = graph.effective_model_config_digest
    if expected_config_digest is None:
        raise ValueError(
            "runtime graph request requires a Phase 1 effective model config digest"
        )
    candidate = RuntimeGraphSourceRequest(
        declaration=graph.declaration,
        resolved_graph=graph,
        semantic_structure_digest=selection.semantic_structure_digest,
        selection_group_id=selection.selection_group_id,
        model_config=context.model_config,
        resolved_model_revision=graph.resolved_model_revision,
        source_producer_fingerprint=context.source_producer_fingerprint,
        expected_contributor_authority=derive_expected_contributor_authority(
            context.expected_contributors
        ),
        source_identity=context.source_identity,
        artifact_identity=context.artifact_identity,
        source_allocation_generation=context.source_allocation_generation,
    )
    if canonical_model_config_digest(candidate.model_config) != expected_config_digest:
        raise ValueError("runtime effective model config digest differs from Phase 1")
    return candidate


def build_runtime_source_discovery_request_from_contexts(
    *,
    selection: CompiledPrecisionSelectionGroup,
    contexts: Mapping[str, RuntimeGraphSourceContext],
) -> RuntimeSourceDiscoveryRequest:
    """Build all runtime graph requests with one Phase 1 validation pass.

    This is the startup-only production construction path. Per-version refit code
    consumes the returned immutable identities and never repeats topology binding.
    """
    context_entries = _snapshot_runtime_contexts(contexts)
    validated_selection = validate_compiled_precision_selection_group(selection)
    graphs_by_id = {
        graph.declaration.graph_instance_id: graph
        for graph in validated_selection.topology.graphs
    }
    required_graph_ids = tuple(
        sorted(
            (
                graph_id
                for graph_id, graph in graphs_by_id.items()
                if graph.declaration.lifecycle.graph_provenance
                is GraphProvenance.TRAINING_RUNTIME
            ),
            key=_graph_sort_key,
        )
    )
    contexts_by_graph = dict(context_entries)
    if tuple(contexts_by_graph) != required_graph_ids:
        raise ValueError("runtime context coverage differs from the Phase 1 lifecycle")
    graph_requests = tuple(
        _build_graph_request_from_validated_selection(
            selection=validated_selection,
            graph=graphs_by_id[graph_id],
            context=contexts_by_graph[graph_id],
        )
        for graph_id in required_graph_ids
    )
    request = RuntimeSourceDiscoveryRequest(
        graph_requests=graph_requests,
        trusted_expected_contributors=tuple(
            (graph_id, contexts_by_graph[graph_id].expected_contributors)
            for graph_id in required_graph_ids
        ),
    )
    return _validate_request_against_selection(
        validated_selection,
        request,
        graphs_by_id=graphs_by_id,
        request_snapshot_validated=True,
        config_digests_validated=True,
    )


def _receipt_payload(receipt: DiscoveryCompletenessReceipt) -> dict[str, object]:
    return {
        "graph_instance_id": receipt.graph_instance_id,
        "producer_fingerprint_digest": receipt.producer_fingerprint_digest,
        "observed_contributor_set_digest": (receipt.observed_contributor_set_digest),
        "observed_contributor_count": receipt.observed_contributor_count,
        "source_set_digest": receipt.source_set_digest,
        "source_count": receipt.source_count,
        "canonical_records_digest": receipt.canonical_records_digest,
        "storage_realization_set_digest": (receipt.storage_realization_set_digest),
        "storage_realization_count": receipt.storage_realization_count,
        "request_kind": receipt.request_kind.value,
        "request_digest": receipt.request_digest,
    }


def _result_digest(
    graph_request: RuntimeGraphSourceRequest,
    partition: GraphDiscoveryPartition,
) -> str:
    authority = partition.expected_contributor_authority
    return _canonical_digest(
        {
            "type": "runtime_source_discovery_result",
            "graph_instance_id": _runtime_graph_id(graph_request),
            "runtime_source_request_digest": (
                graph_request.runtime_source_request_digest
            ),
            "semantic_structure_digest": graph_request.semantic_structure_digest,
            "selection_group_id": graph_request.selection_group_id,
            "producer_fingerprint_digest": (
                partition.completeness_receipt.producer_fingerprint_digest
            ),
            "expected_contributor_authority": {
                "contributor_set_digest": authority.contributor_set_digest,
                "contributor_count": authority.contributor_count,
                "authority": {
                    "kind": authority.authority.kind.value,
                    "locator": authority.authority.locator,
                    "digest": authority.authority.digest,
                },
            },
            "completeness_receipt": _receipt_payload(partition.completeness_receipt),
        }
    )


@dataclass(frozen=True, slots=True)
class RuntimeSourceDiscoveryResult:
    """One graph partition and its immutable aggregate binding identity."""

    graph_request: RuntimeGraphSourceRequest
    partition: GraphDiscoveryPartition
    graph_instance_id: str = field(init=False)
    runtime_source_request_digest: str = field(init=False)
    semantic_structure_digest: str = field(init=False)
    selection_group_id: str = field(init=False)
    producer_fingerprint: SourceProducerFingerprint = field(init=False)
    result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.graph_request) is not RuntimeGraphSourceRequest:
            raise TypeError("graph_request must be an exact RuntimeGraphSourceRequest")
        if type(self.partition) is not GraphDiscoveryPartition:
            raise TypeError("partition must be an exact GraphDiscoveryPartition")
        graph_instance_id = _runtime_graph_id(self.graph_request)
        if self.partition.graph_instance_id != graph_instance_id:
            raise ValueError("result partition graph differs from its request")
        if (
            self.partition.producer_fingerprint
            != self.graph_request.source_producer_fingerprint
        ):
            raise ValueError("result partition producer differs from its request")
        if (
            self.partition.expected_contributor_authority
            != self.graph_request.expected_contributor_authority
        ):
            raise ValueError("result partition authority differs from its request")
        object.__setattr__(self, "graph_instance_id", graph_instance_id)
        object.__setattr__(
            self,
            "runtime_source_request_digest",
            self.graph_request.runtime_source_request_digest,
        )
        object.__setattr__(
            self,
            "semantic_structure_digest",
            self.graph_request.semantic_structure_digest,
        )
        object.__setattr__(
            self,
            "selection_group_id",
            self.graph_request.selection_group_id,
        )
        object.__setattr__(
            self,
            "producer_fingerprint",
            self.graph_request.source_producer_fingerprint,
        )
        object.__setattr__(
            self,
            "result_digest",
            _result_digest(self.graph_request, self.partition),
        )


def _validate_result_snapshot(result: object) -> RuntimeSourceDiscoveryResult:
    if type(result) is not RuntimeSourceDiscoveryResult:
        raise TypeError("result must be an exact RuntimeSourceDiscoveryResult")
    typed_result = cast(RuntimeSourceDiscoveryResult, result)
    if type(typed_result.graph_request) is not RuntimeGraphSourceRequest:
        raise TypeError("graph_request must be an exact RuntimeGraphSourceRequest")
    if type(typed_result.partition) is not GraphDiscoveryPartition:
        raise TypeError("partition must be an exact GraphDiscoveryPartition")
    expected_fields: tuple[tuple[str, object], ...] = (
        ("graph_instance_id", _runtime_graph_id(typed_result.graph_request)),
        (
            "runtime_source_request_digest",
            typed_result.graph_request.runtime_source_request_digest,
        ),
        (
            "semantic_structure_digest",
            typed_result.graph_request.semantic_structure_digest,
        ),
        ("selection_group_id", typed_result.graph_request.selection_group_id),
        (
            "result_digest",
            _result_digest(typed_result.graph_request, typed_result.partition),
        ),
    )
    for field_name, expected_value in expected_fields:
        actual_value = getattr(typed_result, field_name)
        if type(actual_value) is not str:
            raise TypeError(f"{field_name} must be an exact string")
        if actual_value != expected_value:
            raise ValueError(f"{field_name} differs from its canonical derivation")
    if type(typed_result.producer_fingerprint) is not SourceProducerFingerprint:
        raise TypeError(
            "producer_fingerprint must be an exact SourceProducerFingerprint"
        )
    if (
        typed_result.producer_fingerprint
        != typed_result.graph_request.source_producer_fingerprint
    ):
        raise ValueError("producer_fingerprint differs from its canonical derivation")
    return typed_result


def build_runtime_source_discovery_result(
    *,
    request: RuntimeSourceDiscoveryRequest,
    graph_request: RuntimeGraphSourceRequest,
    partition: GraphDiscoveryPartition,
) -> RuntimeSourceDiscoveryResult:
    """Bind one producer partition to its exact aggregate graph request."""
    validated_request = _validate_aggregate_request_snapshot(request)
    if type(graph_request) is not RuntimeGraphSourceRequest:
        raise TypeError("graph_request must be an exact RuntimeGraphSourceRequest")
    runtime_source_request_identity_digest(graph_request)
    graph_id = _runtime_graph_id(graph_request)
    requests_by_graph = {
        _runtime_graph_id(item): item for item in validated_request.graph_requests
    }
    canonical_request = requests_by_graph.get(graph_id)
    if canonical_request is None or canonical_request != graph_request:
        raise ValueError(
            "graph_request is not an exact member of the aggregate request"
        )
    trusted_by_graph = dict(validated_request.trusted_expected_contributors)
    validated_inventory = validate_runtime_discovery_inventory(
        (canonical_request,),
        SourceDiscoveryInventory((partition,)),
        {graph_id: trusted_by_graph[graph_id]},
    )
    return RuntimeSourceDiscoveryResult(
        graph_request=canonical_request,
        partition=validated_inventory.partitions[0],
    )


def build_runtime_source_discovery_results(
    *,
    request: RuntimeSourceDiscoveryRequest,
    partitions: Sequence[GraphDiscoveryPartition],
) -> tuple[RuntimeSourceDiscoveryResult, ...]:
    """Bind all producer partitions with one aggregate request validation pass."""
    validated_request = _validate_aggregate_request_snapshot(request)
    partition_snapshot = _snapshot_sequence(partitions, "runtime graph partitions")
    if any(
        type(partition) is not GraphDiscoveryPartition
        for partition in partition_snapshot
    ):
        raise TypeError("partitions must contain exact GraphDiscoveryPartition records")
    partition_graph_ids = tuple(
        partition.graph_instance_id for partition in partition_snapshot
    )
    if any(type(graph_id) is not str for graph_id in partition_graph_ids):
        raise TypeError("partition graph_instance_id must be an exact string")
    requests_by_graph = {
        _runtime_graph_id(graph_request): graph_request
        for graph_request in validated_request.graph_requests
    }
    if len(partition_graph_ids) != len(set(partition_graph_ids)) or set(
        partition_graph_ids
    ) != set(requests_by_graph):
        raise ValueError(
            "runtime partition coverage must contain every requested graph exactly once"
        )
    partitions_by_graph = {
        partition.graph_instance_id: partition for partition in partition_snapshot
    }
    inventory = validate_runtime_discovery_inventory(
        validated_request.graph_requests,
        SourceDiscoveryInventory(
            tuple(partitions_by_graph[graph_id] for graph_id in requests_by_graph)
        ),
        dict(validated_request.trusted_expected_contributors),
    )
    validated_partitions_by_graph = {
        partition.graph_instance_id: partition for partition in inventory.partitions
    }
    return tuple(
        RuntimeSourceDiscoveryResult(
            graph_request=graph_request,
            partition=validated_partitions_by_graph[graph_id],
        )
        for graph_id, graph_request in requests_by_graph.items()
    )


def produce_runtime_source_discovery_results(
    *,
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
    producers_by_graph: Mapping[str, SourceMetadataProducer],
) -> tuple[RuntimeSourceDiscoveryResult, ...]:
    """Run explicitly graph-bound runtime producers and publish atomically."""
    producer_entries = _snapshot_runtime_source_producer_mapping(producers_by_graph)
    validated_selection = validate_compiled_precision_selection_group(selection)
    selection_policy_snapshot = validated_selection.policy_snapshot
    selection_topology = validated_selection.topology
    selection_policy_digest = validated_selection.policy_digest
    selection_semantic_structure_digest = validated_selection.semantic_structure_digest
    selection_group_id = validated_selection.selection_group_id
    graphs_by_id = {
        graph.declaration.graph_instance_id: graph
        for graph in validated_selection.topology.graphs
    }
    validated_request = _validate_request_against_selection(
        validated_selection,
        request,
        graphs_by_id=graphs_by_id,
    )
    request_digest = validated_request.request_digest
    graph_requests = validated_request.graph_requests
    trusted_expected_contributors = validated_request.trusted_expected_contributors
    graph_request_objects = tuple(graph_requests)
    trusted_expected_contributor_objects = tuple(
        expected for _, expected in trusted_expected_contributors
    )
    requested_graph_ids = tuple(
        _runtime_graph_id(graph_request) for graph_request in graph_requests
    )
    bound_graph_ids = tuple(graph_id for graph_id, _ in producer_entries)
    if bound_graph_ids != requested_graph_ids:
        raise ValueError(
            "runtime source producer coverage differs from the runtime request"
        )

    producer_snapshots_by_identity: dict[int, _RuntimeSourceProducerSnapshot] = {}
    for _, producer in producer_entries:
        producer_identity = id(producer)
        snapshot = producer_snapshots_by_identity.get(producer_identity)
        if snapshot is None:
            snapshot = _snapshot_runtime_source_producer(producer)
            producer_snapshots_by_identity[producer_identity] = snapshot
        elif snapshot.producer is not producer:  # pragma: no cover - live ID collision
            raise RuntimeError("runtime source producer identity collision")

    producers_by_graph_snapshot = dict(producer_entries)
    for graph_request in validated_request.graph_requests:
        graph_id = _runtime_graph_id(graph_request)
        producer = producers_by_graph_snapshot[graph_id]
        producer_snapshot = producer_snapshots_by_identity[id(producer)]
        request_fingerprint_digest = source_producer_fingerprint_identity_digest(
            graph_request.source_producer_fingerprint
        )
        if producer_snapshot.fingerprint_digest != request_fingerprint_digest:
            raise ValueError(
                f"runtime source producer fingerprint differs for {graph_id}"
            )

    revalidated_selection = validate_compiled_precision_selection_group(
        validated_selection
    )
    if revalidated_selection is not validated_selection:
        raise ValueError(
            "compiled precision selection identity changed during preflight"
        )
    if revalidated_selection.policy_snapshot is not selection_policy_snapshot:
        raise ValueError("precision policy snapshot identity changed during preflight")
    if revalidated_selection.topology is not selection_topology:
        raise ValueError("selection topology identity changed during preflight")
    if revalidated_selection.policy_digest != selection_policy_digest:
        raise ValueError("precision policy digest changed during preflight")
    if (
        revalidated_selection.semantic_structure_digest
        != selection_semantic_structure_digest
    ):
        raise ValueError("semantic structure digest changed during preflight")
    if revalidated_selection.selection_group_id != selection_group_id:
        raise ValueError("selection group identity changed during preflight")
    revalidated_graphs_by_id = {
        graph.declaration.graph_instance_id: graph
        for graph in revalidated_selection.topology.graphs
    }
    revalidated_request = _validate_request_against_selection(
        revalidated_selection,
        validated_request,
        graphs_by_id=revalidated_graphs_by_id,
        config_digests_validated=True,
    )
    if revalidated_request is not validated_request:
        raise ValueError("runtime source request identity changed during preflight")
    if revalidated_request.request_digest != request_digest:
        raise ValueError("runtime source request digest changed during preflight")
    if revalidated_request.graph_requests is not graph_requests:
        raise ValueError("runtime graph request collection changed during preflight")
    if (
        revalidated_request.trusted_expected_contributors
        is not trusted_expected_contributors
    ):
        raise ValueError(
            "trusted expected contributor collection changed during preflight"
        )
    if any(
        current is not original
        for current, original in zip(
            revalidated_request.graph_requests,
            graph_request_objects,
            strict=True,
        )
    ):
        raise ValueError("runtime graph request identity changed during preflight")
    if any(
        current is not original
        for current, original in zip(
            (
                expected
                for _, expected in revalidated_request.trusted_expected_contributors
            ),
            trusted_expected_contributor_objects,
            strict=True,
        )
    ):
        raise ValueError(
            "trusted expected contributor identity changed during preflight"
        )

    trusted_by_graph = dict(validated_request.trusted_expected_contributors)
    partitions: list[GraphDiscoveryPartition] = []
    for graph_request in validated_request.graph_requests:
        graph_id = _runtime_graph_id(graph_request)
        producer = producers_by_graph_snapshot[graph_id]
        producer_snapshot = producer_snapshots_by_identity[id(producer)]
        trusted_expected_contributors = trusted_by_graph[graph_id]
        contributions = producer_snapshot.discover_contributions(
            graph_request,
            trusted_expected_contributors,
        )
        partitions.append(
            assemble_runtime_graph_discovery_partition(
                runtime_request=graph_request,
                expected_contributors=trusted_expected_contributors,
                contributions=contributions,
            )
        )
    return build_runtime_source_discovery_results(
        request=validated_request,
        partitions=partitions,
    )


def _validate_runtime_source_discovery_results(
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
    results: Sequence[RuntimeSourceDiscoveryResult],
) -> tuple[
    CompiledPrecisionSelectionGroup,
    RuntimeSourceDiscoveryRequest,
    SourceDiscoveryInventory,
    dict[str, RuntimeSourceDiscoveryResult],
]:
    validated_selection = validate_compiled_precision_selection_group(selection)
    validated_request = _validate_request_against_selection(
        validated_selection,
        request,
    )
    result_snapshot = _snapshot_sequence(results, "runtime discovery results")
    if any(
        type(result) is not RuntimeSourceDiscoveryResult for result in result_snapshot
    ):
        raise TypeError(
            "results must contain exact RuntimeSourceDiscoveryResult records"
        )
    for result in result_snapshot:
        if type(result.graph_request) is not RuntimeGraphSourceRequest:
            raise TypeError("graph_request must be an exact RuntimeGraphSourceRequest")
        if type(result.partition) is not GraphDiscoveryPartition:
            raise TypeError("partition must be an exact GraphDiscoveryPartition")
        if type(result.partition.graph_instance_id) is not str:
            raise TypeError("partition graph_instance_id must be an exact string")
        runtime_source_request_identity_digest(result.graph_request)
        if result.partition.graph_instance_id != _runtime_graph_id(
            result.graph_request
        ):
            raise ValueError("result partition graph differs from its graph request")
    expected_requests_by_graph = {
        _runtime_graph_id(item): item for item in validated_request.graph_requests
    }
    result_graph_ids = tuple(
        _runtime_graph_id(result.graph_request) for result in result_snapshot
    )
    if len(result_graph_ids) != len(set(result_graph_ids)) or set(
        result_graph_ids
    ) != set(expected_requests_by_graph):
        raise ValueError(
            "runtime result coverage must contain every requested graph exactly once"
        )
    results_by_graph = {
        _runtime_graph_id(result.graph_request): result for result in result_snapshot
    }
    for graph_id, expected_request in expected_requests_by_graph.items():
        result = results_by_graph[graph_id]
        if result.graph_request != expected_request:
            raise ValueError(f"runtime result graph request is stale for {graph_id}")
    inventory = SourceDiscoveryInventory(
        tuple(
            results_by_graph[graph_id].partition
            for graph_id in expected_requests_by_graph
        )
    )
    trusted_by_graph = dict(validated_request.trusted_expected_contributors)
    validated_inventory = validate_runtime_discovery_inventory(
        validated_request.graph_requests,
        inventory,
        trusted_by_graph,
    )
    for graph_id in expected_requests_by_graph:
        _validate_result_snapshot(results_by_graph[graph_id])
    return (
        validated_selection,
        validated_request,
        validated_inventory,
        results_by_graph,
    )


def validate_runtime_source_discovery_results(
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
    results: Sequence[RuntimeSourceDiscoveryResult],
) -> SourceDiscoveryInventory:
    """Validate one complete result set atomically and return its inventory."""
    _, _, validated_inventory, _ = _validate_runtime_source_discovery_results(
        selection,
        request,
        results,
    )
    return validated_inventory


def bind_runtime_source_intents(
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
    results: tuple[RuntimeSourceDiscoveryResult, ...],
) -> CompiledPrecisionIntentGroup:
    """Atomically bind complete Phase 2 sources to the exact Phase 1 selection."""
    if type(results) is not tuple:
        raise TypeError("results must be an exact tuple")
    (
        validated_selection,
        validated_request,
        validated_inventory,
        results_by_graph,
    ) = _validate_runtime_source_discovery_results(
        selection,
        request,
        results,
    )
    required_adapter_ids = {
        graph.declaration.graph_instance_id: graph.adapter_id
        for graph in validated_selection.topology.graphs
        if graph.declaration.lifecycle.graph_provenance
        is GraphProvenance.TRAINING_RUNTIME
    }
    source_provenance = RuntimeSourceProvenance(
        selection_group_id=validated_selection.selection_group_id,
        request_digest=validated_request.request_digest,
        result_digests=tuple(
            (graph_id, results_by_graph[graph_id].result_digest)
            for graph_id in required_adapter_ids
        ),
    )
    source_topology = classify_validated_runtime_semantic_topology(
        validated_selection.schema_version,
        validated_request.graph_requests,
        validated_inventory,
        required_adapter_ids_by_graph=required_adapter_ids,
        runtime_source_provenance=source_provenance,
    )
    receipt = _issue_runtime_source_evidence_receipt(
        source_provenance=source_provenance,
        source_binding_digest=(source_topology.source_bindings.source_binding_digest),
    )
    return _bind_compiled_precision_intents(
        validated_selection,
        source_topology,
        receipt,
    )


def derive_active_runtime_source_provenance(
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
    results: tuple[RuntimeSourceDiscoveryResult, ...],
) -> ActiveRuntimeSourceProvenanceAnchor:
    """Derive consumer authority from the independently held active artifacts."""
    if type(results) is not tuple:
        raise TypeError("results must be an exact tuple")
    (
        validated_selection,
        validated_request,
        _,
        results_by_graph,
    ) = _validate_runtime_source_discovery_results(
        selection,
        request,
        results,
    )
    runtime_graph_ids = tuple(
        graph.declaration.graph_instance_id
        for graph in validated_selection.topology.graphs
        if graph.declaration.lifecycle.graph_provenance
        is GraphProvenance.TRAINING_RUNTIME
    )
    source_provenance = RuntimeSourceProvenance(
        selection_group_id=validated_selection.selection_group_id,
        request_digest=validated_request.request_digest,
        result_digests=tuple(
            (graph_id, results_by_graph[graph_id].result_digest)
            for graph_id in runtime_graph_ids
        ),
    )
    anchor = object.__new__(ActiveRuntimeSourceProvenanceAnchor)
    object.__setattr__(anchor, "source_provenance", source_provenance)
    object.__setattr__(
        anchor,
        "anchor_digest",
        _canonical_digest(
            {
                "type": "active_runtime_source_provenance_anchor",
                "source_provenance_digest": source_provenance.provenance_digest,
            }
        ),
    )
    return anchor


def build_runtime_graph_source_request(
    *,
    selection: CompiledPrecisionSelectionGroup,
    graph_instance_id: str,
    model_config: Mapping[str, object],
    source_producer_fingerprint: SourceProducerFingerprint,
    expected_contributors: ExpectedContributorSet,
    source_identity: EvidenceSource,
    artifact_identity: EvidenceSource,
    source_allocation_generation: str,
) -> RuntimeGraphSourceRequest:
    """Bind runtime-only source context to one exact Phase 1 graph."""
    validated_selection = validate_compiled_precision_selection_group(selection)
    context = RuntimeGraphSourceContext(
        graph_instance_id=graph_instance_id,
        model_config=model_config,
        source_producer_fingerprint=source_producer_fingerprint,
        expected_contributors=expected_contributors,
        source_identity=source_identity,
        artifact_identity=artifact_identity,
        source_allocation_generation=source_allocation_generation,
    )
    graphs_by_id = {
        graph.declaration.graph_instance_id: graph
        for graph in validated_selection.topology.graphs
    }
    try:
        graph = graphs_by_id[graph_instance_id]
    except KeyError as error:
        raise ValueError(f"unknown Phase 1 graph {graph_instance_id!r}") from error
    return _build_graph_request_from_validated_selection(
        selection=validated_selection,
        graph=graph,
        context=context,
    )
