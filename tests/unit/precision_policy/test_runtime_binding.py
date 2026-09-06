from __future__ import annotations

import os
import pickle
import subprocess
import sys
from collections.abc import Callable, ItemsView, Iterator, Mapping, Sequence
from copy import copy
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from typing import TypeVar, cast

import pytest

import nemo_rl.precision_policy.compiler as compiler_module
import nemo_rl.precision_policy.runtime_binding as runtime_binding_module
import nemo_rl.precision_policy.topology as topology_module
from nemo_rl.precision_policy.compiler import (
    CompiledGraphPrecisionIntent,
    CompiledPrecisionIntentGroup,
    CompiledPrecisionSelectionGroup,
    EndpointPrecisionPlan,
    RuntimeSourceEvidenceReceipt,
    compile_precision_selection,
)
from nemo_rl.precision_policy.config import PrecisionPolicyConfig
from nemo_rl.precision_policy.discovery_producers import SourceMetadataProducer
from nemo_rl.precision_policy.runtime_binding import (
    RuntimeGraphSourceContext,
    RuntimeSourceDiscoveryRequest,
    RuntimeSourceDiscoveryResult,
    build_runtime_graph_source_request,
    build_runtime_source_discovery_request,
    build_runtime_source_discovery_request_from_contexts,
    build_runtime_source_discovery_result,
    build_runtime_source_discovery_results,
    bind_runtime_source_intents,
    produce_runtime_source_discovery_results,
    validate_compiled_precision_intent_group,
    validate_runtime_source_discovery_request,
    validate_runtime_source_discovery_results,
)
from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    AxisDomain,
    AxisProjection,
    DecoderLayerUniverse,
    EvidenceSource,
    EvidenceSourceKind,
    ExpectedGraphDeclaration,
    FamilyIndexDomain,
    GraphKind,
    GraphLifecycle,
    GraphProvenance,
    ImmutableAuxiliaryEvidence,
    IndexPathSegment,
    LayerDomain,
    LayerMember,
    LiteralPathSegment,
    LOGICAL_VALUES,
    OwnerFamilyBinding,
    OwnerFamilyReference,
    ParameterInventoryEntry,
    RoleExpectedDomain,
    ResolvedGraphTopology,
    ResolvedSelectionTopology,
    RolloutParticipation,
    SelectionTopologyEntry,
    SemanticAddressPattern,
    SemanticGraphManifest,
    SemanticOwnership,
    SemanticTensorFamily,
    SourceMutability,
    SourceOwnerInventoryEntry,
    ValueProvenance,
    _compute_semantic_structure_digest,
    _merge_selection_role_definitions,
    builtin_role_definitions,
    canonical_model_config_digest,
)
from nemo_rl.precision_policy.source_discovery import (
    HF_SAFETENSORS_HEADER_V1,
    DiscoveryContribution,
    ExpectedContributorSet,
    GraphDiscoveryPartition,
    GraphTopologyInput,
    RuntimeGraphSourceRequest,
    SourceDiscoveryInventory,
    SourceDiscoveryRecord,
    SourceProducerFingerprint,
    SourceRecordProvenance,
    SourceSchemaId,
    assemble_runtime_graph_discovery_partition,
    derive_expected_contributor_authority,
    source_producer_fingerprint_identity_digest,
    validate_source_producer_fingerprint,
)
from nemo_rl.precision_policy.topology import (
    CanonicalSourceSemanticBinding,
    CanonicalValueClassificationEdge,
    ComponentAxisTarget,
    FamilyIndexAxisTarget,
    FixedLayerCoordinate,
    GraphSemanticSourceBindings,
    LayerCoordinateTarget,
    ModelTopologyAdapter,
    OutputMemberTarget,
    RoleDefinitionContribution,
    SemanticGraphBuildFragment,
    SemanticSourceBindingInventory,
    SourceAxisSelection,
    SourceIndexSpan,
    SourceOrdinalMapSegment,
    SourceRegion,
    SourceToSemanticAxisMapping,
)
from nemo_rl.precision_policy.topology_resolver import (
    GraphTopologyResolutionRequest,
    freeze_phase1_requests_by_graph,
)
from nemo_rl.precision_policy.source_dtype import CanonicalSourceDType
from nemo_rl.precision_policy.source_storage import (
    IDENTITY_PERMUTATION_ID,
    IDENTITY_SWIZZLE_ID,
    SourceExtentRounding,
    SourceNormalizationContract,
    SourceNormalizationKind,
    SourceNormalizedAxisExtent,
    SourceNormalizerManifest,
    SourcePaddingSemantics,
    SourcePhysicalAxisSpec,
    SourceStorageComponent,
    SourceStorageRealization,
    SourceStorageRealizationInventory,
    source_normalizer_manifest_digest,
)

_ValueT = TypeVar("_ValueT")


class _ForgedLegacyAnchor:
    source_provenance: object
    anchor_digest: str


class _OneShotMapping(Mapping[str, ExpectedContributorSet]):
    def __init__(
        self,
        values: Mapping[str, ExpectedContributorSet],
    ) -> None:
        self._values: dict[str, ExpectedContributorSet] = {
            key: value for key, value in values.items()
        }
        self.items_calls = 0

    def __getitem__(self, key: str) -> ExpectedContributorSet:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def items(self) -> ItemsView[str, ExpectedContributorSet]:
        self.items_calls += 1
        if self.items_calls > 1:
            raise AssertionError("caller mapping was traversed more than once")
        return self._values.items()


class _OneShotSequence(list[_ValueT]):
    def __init__(self, values: Sequence[_ValueT]) -> None:
        super().__init__(values)
        self.iter_calls = 0

    def __iter__(self) -> Iterator[_ValueT]:
        self.iter_calls += 1
        if self.iter_calls > 1:
            raise AssertionError("caller sequence was traversed more than once")
        return super().__iter__()


class _TwoViewConfig(Mapping[str, object]):
    def __init__(
        self,
        first: Mapping[str, object],
        second: Mapping[str, object],
    ) -> None:
        self._views = (dict(first), dict(second))
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        return self._views[min(self.items_calls, 1)][key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._views[min(self.items_calls, 1)])

    def __len__(self) -> int:
        return len(self._views[min(self.items_calls, 1)])

    def items(self) -> ItemsView[str, object]:
        view = self._views[min(self.items_calls, 1)]
        self.items_calls += 1
        return view.items()


class _OneShotProducerMapping(Mapping[str, object]):
    def __init__(self, entries: Sequence[tuple[str, object]]) -> None:
        self._entries = tuple(entries)
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        for candidate, producer in self._entries:
            if candidate == key:
                return producer
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (graph_id for graph_id, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self):
        self.items_calls += 1
        if self.items_calls > 1:
            raise AssertionError("producer mapping was traversed more than once")
        return self._entries


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _evidence(name: str, character: str) -> EvidenceSource:
    return EvidenceSource(
        kind=EvidenceSourceKind.RUNTIME_INVENTORY,
        locator=f"runtime://{name}",
        digest=_digest(character),
    )


def _model_config(graph_instance_id: str) -> dict[str, object]:
    return {
        "architectures": ["TestForCausalLM"],
        "graph_instance_id": graph_instance_id,
        "model_type": "test",
    }


def _lifecycle(graph_instance_id: str) -> GraphLifecycle:
    if graph_instance_id == "main":
        return GraphLifecycle(
            graph_kind=GraphKind.MAIN,
            graph_provenance=GraphProvenance.TRAINING_RUNTIME,
            rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
        )
    if graph_instance_id.startswith("mtp."):
        return GraphLifecycle(
            graph_kind=GraphKind.MTP,
            graph_provenance=GraphProvenance.TRAINING_RUNTIME,
            rollout_participation=RolloutParticipation.NOT_SERVED,
        )
    evidence = ImmutableAuxiliaryEvidence(
        graph_instance_id=graph_instance_id,
        model_identity="test/static-draft",
        pinned_checkpoint_revision="static-draft-revision",
        checkpoint_content_digest=_digest("d"),
        model_config_digest=_digest("e"),
        semantic_domain_digest=_digest("f"),
        evidence_source=EvidenceSource(
            kind=EvidenceSourceKind.PINNED_CHECKPOINT_MANIFEST,
            locator="checkpoint://static-draft",
            digest=_digest("9"),
        ),
    )
    return GraphLifecycle(
        graph_kind=GraphKind.SPECULATIVE_DRAFTER,
        graph_provenance=GraphProvenance.EXTERNAL_CHECKPOINT,
        rollout_participation=RolloutParticipation.SERVED_FROM_CHECKPOINT,
        immutable_evidence=evidence,
    )


def _entry(graph_instance_id: str) -> SelectionTopologyEntry:
    if graph_instance_id == "main":
        graph_path, model_part = "text.decoder", "main"
    elif graph_instance_id.startswith("mtp."):
        graph_path, model_part = "auxiliary.mtp", "mtp"
    else:
        graph_path, model_part = "draft.decoder", "draft"
    return SelectionTopologyEntry(
        entry_id=f"{graph_instance_id}.dense.weight",
        graph_instance_id=graph_instance_id,
        pattern=SemanticAddressPattern(
            semantic_graph_path=graph_path,
            path_segments=(
                LiteralPathSegment("layer"),
                IndexPathSegment("global_decoder_layer"),
                LiteralPathSegment("weight"),
            ),
            model_part=model_part,
            module_kind="ffn.dense",
            attributes=(),
            parameter_role="kernel",
        ),
        domain=FamilyIndexDomain(
            layer_domain=LayerDomain((LayerMember(0, None),)),
            independent_axes=(),
        ),
        logical_dtype="bfloat16",
        logical_shape=(8, 8),
        logical_axes=("output_features", "input_features"),
    )


def _selection_fixture() -> tuple[
    CompiledPrecisionSelectionGroup,
    dict[str, dict[str, object]],
]:
    return _selection_for_graph_ids(("main", "mtp.aux", "draft.static"))


def _selection_for_graph_ids(
    graph_instance_ids: tuple[str, ...],
    *,
    model_configs: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[
    CompiledPrecisionSelectionGroup,
    dict[str, dict[str, object]],
]:
    configs = (
        {
            graph_instance_id: _model_config(graph_instance_id)
            for graph_instance_id in graph_instance_ids
        }
        if model_configs is None
        else {
            graph_instance_id: dict(model_configs[graph_instance_id])
            for graph_instance_id in graph_instance_ids
        }
    )
    graphs = []
    for graph_instance_id in graph_instance_ids:
        lifecycle = _lifecycle(graph_instance_id)
        evidence = lifecycle.immutable_evidence
        declaration = ExpectedGraphDeclaration(
            graph_instance_id=graph_instance_id,
            model_identity=(
                evidence.model_identity
                if evidence is not None
                else f"test/{graph_instance_id}"
            ),
            lifecycle=lifecycle,
        )
        graphs.append(
            ResolvedGraphTopology(
                declaration=declaration,
                model_family=f"family-{graph_instance_id}",
                resolved_model_revision=(
                    evidence.pinned_checkpoint_revision
                    if evidence is not None
                    else f"revision-{graph_instance_id}"
                ),
                adapter_id=f"adapter-{graph_instance_id}.v1",
                decoder_layer_universe=DecoderLayerUniverse((0,), ()),
                entries=(_entry(graph_instance_id),),
                role_definitions=(),
                atomic_groups=(),
                effective_model_config_digest=canonical_model_config_digest(
                    configs[graph_instance_id]
                ),
            )
        )
    canonical_graphs = tuple(
        sorted(
            graphs,
            key=lambda graph: (
                0 if graph.declaration.graph_instance_id == "main" else 1,
                graph.declaration.graph_instance_id,
            ),
        )
    )
    role_definitions = _merge_selection_role_definitions(canonical_graphs, 1)
    topology = ResolvedSelectionTopology(
        schema_version=1,
        graphs=canonical_graphs,
        role_definitions=role_definitions,
        semantic_structure_digest=_compute_semantic_structure_digest(
            schema_version=1,
            graphs=canonical_graphs,
            role_definitions=role_definitions,
        ),
    )
    selection = compile_precision_selection(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        topology,
    )
    return selection, configs


def _normalizer_manifest() -> SourceNormalizerManifest:
    return SourceNormalizerManifest(
        schema_version=1,
        contracts=(
            SourceNormalizationContract(
                capability_id="test.identity.v1",
                kind=SourceNormalizationKind.IDENTITY,
                contract_digest=_digest("1"),
            ),
        ),
    )


def _fingerprint() -> SourceProducerFingerprint:
    manifest = _normalizer_manifest()
    return SourceProducerFingerprint(
        schema_id=HF_SAFETENSORS_HEADER_V1,
        producer_implementation_id="test.runtime-metadata",
        producer_revision="a" * 40,
        normalization_contract_digest=source_normalizer_manifest_digest(manifest),
        evidence=_evidence("producer", "2"),
    )


def _expected(graph_instance_id: str) -> ExpectedContributorSet:
    return ExpectedContributorSet(
        contributor_ids=(f"{graph_instance_id}-rank-0",),
        authority=_evidence(f"{graph_instance_id}-contributors", "3"),
    )


def _build_graph_request(
    selection: CompiledPrecisionSelectionGroup,
    configs: Mapping[str, Mapping[str, object]],
    graph_instance_id: str,
):
    return build_runtime_graph_source_request(
        selection=selection,
        graph_instance_id=graph_instance_id,
        model_config=configs[graph_instance_id],
        source_producer_fingerprint=_fingerprint(),
        expected_contributors=_expected(graph_instance_id),
        source_identity=_evidence(f"{graph_instance_id}-source", "4"),
        artifact_identity=_evidence(f"{graph_instance_id}-artifact", "5"),
        source_allocation_generation="allocation-1",
    )


def _manual_graph_request(
    selection: CompiledPrecisionSelectionGroup,
    configs: Mapping[str, Mapping[str, object]],
    graph_instance_id: str,
    *,
    graph: ResolvedGraphTopology | None = None,
    semantic_structure_digest: str | None = None,
    selection_group_id: str | None = None,
    allocation_generation: str = "allocation-1",
) -> RuntimeGraphSourceRequest:
    resolved_graph = graph or next(
        item
        for item in selection.topology.graphs
        if item.declaration.graph_instance_id == graph_instance_id
    )
    return RuntimeGraphSourceRequest(
        declaration=resolved_graph.declaration,
        resolved_graph=resolved_graph,
        semantic_structure_digest=(
            selection.semantic_structure_digest
            if semantic_structure_digest is None
            else semantic_structure_digest
        ),
        selection_group_id=(
            selection.selection_group_id
            if selection_group_id is None
            else selection_group_id
        ),
        model_config=configs[graph_instance_id],
        resolved_model_revision=resolved_graph.resolved_model_revision,
        source_producer_fingerprint=_fingerprint(),
        expected_contributor_authority=derive_expected_contributor_authority(
            _expected(graph_instance_id)
        ),
        source_identity=_evidence(f"{graph_instance_id}-source", "4"),
        artifact_identity=_evidence(f"{graph_instance_id}-artifact", "5"),
        source_allocation_generation=allocation_generation,
    )


def _runtime_requests(
    selection: CompiledPrecisionSelectionGroup,
    configs: Mapping[str, Mapping[str, object]],
) -> tuple[RuntimeGraphSourceRequest, RuntimeGraphSourceRequest]:
    return (
        _build_graph_request(selection, configs, "main"),
        _build_graph_request(selection, configs, "mtp.aux"),
    )


def _runtime_context(
    graph_instance_id: str,
    model_config: Mapping[str, object],
) -> RuntimeGraphSourceContext:
    return RuntimeGraphSourceContext(
        graph_instance_id=graph_instance_id,
        model_config=model_config,
        source_producer_fingerprint=_fingerprint(),
        expected_contributors=_expected(graph_instance_id),
        source_identity=_evidence(f"{graph_instance_id}-source", "4"),
        artifact_identity=_evidence(f"{graph_instance_id}-artifact", "5"),
        source_allocation_generation="allocation-1",
    )


def _source_record(graph_instance_id: str) -> SourceDiscoveryRecord:
    native_name = f"{graph_instance_id}.model.weight"
    return SourceDiscoveryRecord(
        record_id=f"{graph_instance_id}.dense.weight",
        graph_instance_id=graph_instance_id,
        source_native_name=native_name,
        source_native_owner_id=native_name,
        dtype=CanonicalSourceDType.BFLOAT16,
        shape=(8, 8),
        numeric_encoding="plain_bfloat16",
        provenance=SourceRecordProvenance.TRAINING_RUNTIME,
        provenance_evidence=_evidence(f"{graph_instance_id}-provenance", "6"),
        source_mutability=SourceMutability.MUTABLE,
        mutability_evidence=_evidence(f"{graph_instance_id}-mutability", "7"),
    )


def _storage_realizations(
    record: SourceDiscoveryRecord,
) -> SourceStorageRealizationInventory:
    manifest = _normalizer_manifest()
    return SourceStorageRealizationInventory(
        graph_instance_id=record.graph_instance_id,
        normalizer_manifest=manifest,
        realizations=(
            SourceStorageRealization(
                realization_id=f"{record.record_id}.identity",
                graph_instance_id=record.graph_instance_id,
                output_record_id=record.record_id,
                components=(
                    SourceStorageComponent(
                        graph_instance_id=record.graph_instance_id,
                        native_component_id=f"{record.record_id}.component",
                        source_native_name=record.source_native_name or "",
                        component_role="normalized_values",
                        carrier_dtype=record.dtype,
                        physical_shape=record.shape,
                        physical_axes=tuple(
                            SourcePhysicalAxisSpec(
                                axis_name=f"axis_{axis_index}",
                                extent=SourceNormalizedAxisExtent(
                                    normalized_axis_indices=(axis_index,),
                                    divisor=1,
                                    rounding=SourceExtentRounding.EXACT,
                                    alignment=1,
                                ),
                            )
                            for axis_index in range(len(record.shape))
                        ),
                        storage_encoding=record.numeric_encoding,
                        padding_semantics=SourcePaddingSemantics.NO_PADDING,
                        padding_fill_encoding=None,
                        permutation_id=IDENTITY_PERMUTATION_ID,
                        swizzle_id=IDENTITY_SWIZZLE_ID,
                    ),
                ),
                output_dtype=record.dtype,
                output_shape=record.shape,
                output_numeric_encoding=record.numeric_encoding,
                normalization=manifest.contracts[0],
            ),
        ),
    )


def _partition(
    graph_request: RuntimeGraphSourceRequest,
    expected: ExpectedContributorSet,
) -> GraphDiscoveryPartition:
    graph_instance_id = graph_request.declaration.graph_instance_id
    record = _source_record(graph_instance_id)
    return assemble_runtime_graph_discovery_partition(
        runtime_request=graph_request,
        expected_contributors=expected,
        contributions=(
            DiscoveryContribution(
                contributor_id=expected.contributor_ids[0],
                graph_instance_id=graph_instance_id,
                producer_fingerprint=graph_request.source_producer_fingerprint,
                records=(record,),
                storage_realizations=_storage_realizations(record),
            ),
        ),
    )


_DEFAULT_PRODUCER_FIELD = object()


class _TestSourceMetadataProducer:
    def __init__(
        self,
        fingerprint: SourceProducerFingerprint,
        *,
        producer_id: object = _DEFAULT_PRODUCER_FIELD,
        schema_id: object = _DEFAULT_PRODUCER_FIELD,
        fingerprint_result: object = _DEFAULT_PRODUCER_FIELD,
        fail_graph: str | None = None,
        empty_graph: str | None = None,
        events: list[str] | None = None,
    ) -> None:
        self._producer_id = (
            fingerprint.producer_implementation_id
            if producer_id is _DEFAULT_PRODUCER_FIELD
            else producer_id
        )
        self._schema_id = (
            fingerprint.schema_id if schema_id is _DEFAULT_PRODUCER_FIELD else schema_id
        )
        self._fingerprint_result = (
            fingerprint
            if fingerprint_result is _DEFAULT_PRODUCER_FIELD
            else fingerprint_result
        )
        self._fail_graph = fail_graph
        self._empty_graph = empty_graph
        self._events = events
        self.producer_id_reads = 0
        self.schema_id_reads = 0
        self.fingerprint_calls = 0
        self.discover_method_reads = 0
        self.discover_graph_ids: list[str] = []
        self.received_requests: list[RuntimeGraphSourceRequest] = []
        self.received_expected_contributors: list[ExpectedContributorSet] = []

    def __getattribute__(self, name: str):
        if name == "discover_contributions":
            reads = object.__getattribute__(self, "discover_method_reads") + 1
            object.__setattr__(self, "discover_method_reads", reads)
            if reads > 1:
                raise AssertionError("discover_contributions was read more than once")
        return object.__getattribute__(self, name)

    @property
    def producer_id(self) -> str:
        self.producer_id_reads += 1
        if self.producer_id_reads > 1:
            raise AssertionError("producer_id was read more than once")
        return cast(str, self._producer_id)

    @property
    def schema_id(self) -> SourceSchemaId:
        self.schema_id_reads += 1
        if self.schema_id_reads > 1:
            raise AssertionError("schema_id was read more than once")
        return cast(SourceSchemaId, self._schema_id)

    def fingerprint(self) -> SourceProducerFingerprint:
        self.fingerprint_calls += 1
        if self.fingerprint_calls > 1:
            raise AssertionError("fingerprint was called more than once")
        return cast(SourceProducerFingerprint, self._fingerprint_result)

    def discover_contributions(
        self,
        request: RuntimeGraphSourceRequest,
        trusted_expected_contributors: ExpectedContributorSet,
    ) -> tuple[DiscoveryContribution, ...]:
        graph_instance_id = request.declaration.graph_instance_id
        self.discover_graph_ids.append(graph_instance_id)
        self.received_requests.append(request)
        self.received_expected_contributors.append(trusted_expected_contributors)
        if self._events is not None:
            self._events.append(f"discover:{graph_instance_id}")
        if graph_instance_id == self._fail_graph:
            raise RuntimeError(f"discovery failed for {graph_instance_id}")
        if graph_instance_id == self._empty_graph:
            return ()
        record = _source_record(graph_instance_id)
        return (
            DiscoveryContribution(
                contributor_id=trusted_expected_contributors.contributor_ids[0],
                graph_instance_id=graph_instance_id,
                producer_fingerprint=request.source_producer_fingerprint,
                records=(record,),
                storage_realizations=_storage_realizations(record),
            ),
        )


def _aggregate_fixture() -> tuple[
    CompiledPrecisionSelectionGroup,
    RuntimeSourceDiscoveryRequest,
    tuple[RuntimeSourceDiscoveryResult, ...],
]:
    selection, configs = _selection_fixture()
    main, mtp = _runtime_requests(selection, configs)
    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=(main, mtp),
        trusted_expected_contributors={
            "main": _expected("main"),
            "mtp.aux": _expected("mtp.aux"),
        },
    )
    results = tuple(
        build_runtime_source_discovery_result(
            request=request,
            graph_request=graph_request,
            partition=_partition(
                graph_request,
                _expected(graph_request.declaration.graph_instance_id),
            ),
        )
        for graph_request in request.graph_requests
    )
    return selection, request, results


def _generation_fixture(
    selection: CompiledPrecisionSelectionGroup,
    configs: Mapping[str, Mapping[str, object]],
    allocation_generation: str,
) -> tuple[
    RuntimeSourceDiscoveryRequest,
    tuple[RuntimeSourceDiscoveryResult, ...],
]:
    graph_requests = tuple(
        _manual_graph_request(
            selection,
            configs,
            graph_id,
            allocation_generation=allocation_generation,
        )
        for graph_id in ("main", "mtp.aux")
    )
    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=graph_requests,
        trusted_expected_contributors={
            graph_id: _expected(graph_id) for graph_id in ("main", "mtp.aux")
        },
    )
    results = tuple(
        build_runtime_source_discovery_result(
            request=request,
            graph_request=graph_request,
            partition=_partition(
                graph_request,
                _expected(graph_request.declaration.graph_instance_id),
            ),
        )
        for graph_request in request.graph_requests
    )
    return request, results


@dataclass(frozen=True)
class _RuntimeTopologyAdapter:
    adapter_id: str
    graph: ResolvedGraphTopology

    def supports(self, model_config: Mapping[str, object]) -> bool:
        return model_config.get("graph_instance_id") == (
            self.graph.declaration.graph_instance_id
        )

    def classify_graph(
        self,
        schema_version: int,
        graph_input: GraphTopologyInput,
        source_records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment:
        assert schema_version == 1
        assert graph_input.declaration == self.graph.declaration
        assert len(source_records) == 1
        record = source_records[0]
        selection_entry = self.graph.entries[0]
        owner_reference = OwnerFamilyReference(
            self.graph.declaration.graph_instance_id,
            "source.dense.weight",
        )
        owner_axes = tuple(
            AxisProjection(axis_name, axis_name)
            for axis_name in selection_entry.domain.axis_names
        )
        member = SemanticTensorFamily(
            pattern=selection_entry.pattern,
            domain=selection_entry.domain,
            format=BF16_FORMAT,
            logical_dtype=selection_entry.logical_dtype,
            logical_shape=selection_entry.logical_shape,
            logical_axes=selection_entry.logical_axes,
            ownership=SemanticOwnership(
                OwnerFamilyBinding(
                    canonical_owner_family=owner_reference,
                    canonical_value_entry_id=selection_entry.entry_id,
                    member_domain=selection_entry.domain,
                    member_to_owner_axes=owner_axes,
                    member_to_value_axes=owner_axes,
                )
            ),
        )
        entry = ParameterInventoryEntry(
            entry_id=selection_entry.entry_id,
            graph_instance_id=self.graph.declaration.graph_instance_id,
            member=member,
            value_provenance=ValueProvenance.TRAINING_PARAMETER,
        )
        source_region = SourceRegion(
            source_shape=record.shape,
            axis_selections=tuple(
                SourceAxisSelection(
                    axis_index=axis_index,
                    spans=(SourceIndexSpan(0, extent),),
                )
                for axis_index, extent in enumerate(record.shape)
            ),
        )
        layer_domain = selection_entry.domain.layer_domain
        if layer_domain is None:
            output_member_domain = selection_entry.domain
            fixed_coordinates = ()
        else:
            output_member_domain = FamilyIndexDomain(None, ())
            fixed_coordinates = (FixedLayerCoordinate(layer_domain.members[0]),)
        edge = CanonicalValueClassificationEdge(
            record_id=record.record_id,
            source_region=source_region,
            output=OutputMemberTarget(
                inventory_entry_id=selection_entry.entry_id,
                member_domain=output_member_domain,
                fixed_coordinates=fixed_coordinates,
            ),
            canonical_owner_family=owner_reference,
            component_role=LOGICAL_VALUES,
            axis_mappings=tuple(
                SourceToSemanticAxisMapping(
                    source_axis_index=axis_index,
                    target=ComponentAxisTarget(LOGICAL_VALUES, logical_axis),
                    segments=(
                        SourceOrdinalMapSegment(
                            SourceIndexSpan(0, extent),
                            0,
                        ),
                    ),
                )
                for axis_index, (extent, logical_axis) in enumerate(
                    zip(record.shape, selection_entry.logical_axes, strict=True)
                )
            ),
        )
        return SemanticGraphBuildFragment(
            graph_instance_id=self.graph.declaration.graph_instance_id,
            classification_edges=(edge,),
            source_owners=(
                SourceOwnerInventoryEntry(
                    owner_family=owner_reference,
                    domain=selection_entry.domain,
                    source_mutability=record.source_mutability,
                    mutability_evidence_source=record.mutability_evidence,
                ),
            ),
            inventory_entries=(entry,),
            manifest=SemanticGraphManifest(
                model_family=self.graph.model_family,
                model_revision=self.graph.resolved_model_revision,
                graph_instance_id=self.graph.declaration.graph_instance_id,
                lifecycle=self.graph.declaration.lifecycle,
                inventory_entry_ids=(entry.entry_id,),
                atomic_groups=self.graph.atomic_groups,
            ),
            role_contributions=(),
        )


def _install_runtime_topology_adapters(
    monkeypatch: pytest.MonkeyPatch,
    selection: CompiledPrecisionSelectionGroup,
) -> None:
    adapters: tuple[ModelTopologyAdapter, ...] = tuple(
        _RuntimeTopologyAdapter(graph.adapter_id, graph)
        for graph in selection.topology.graphs
        if graph.declaration.lifecycle.graph_provenance
        is GraphProvenance.TRAINING_RUNTIME
    )
    monkeypatch.setattr(topology_module, "_default_adapters", lambda: adapters)


def _explicit_runtime_adapter_authority(
    selection: CompiledPrecisionSelectionGroup,
    configs: Mapping[str, Mapping[str, object]],
) -> tuple[
    dict[str, ModelTopologyAdapter],
    Mapping[str, GraphTopologyResolutionRequest],
]:
    runtime_adapters: dict[str, ModelTopologyAdapter] = {
        graph.adapter_id: _RuntimeTopologyAdapter(graph.adapter_id, graph)
        for graph in selection.topology.graphs
    }
    return runtime_adapters, _retained_phase1_requests(selection, configs)


def _retained_phase1_requests(
    selection: CompiledPrecisionSelectionGroup,
    configs: Mapping[str, Mapping[str, object]],
) -> Mapping[str, GraphTopologyResolutionRequest]:
    return freeze_phase1_requests_by_graph(
        tuple(
            GraphTopologyResolutionRequest(
                declaration=graph.declaration,
                effective_model_config=configs[graph.declaration.graph_instance_id],
                resolved_model_revision=graph.resolved_model_revision,
                decoder_layer_universe=graph.decoder_layer_universe,
            )
            for graph in selection.topology.graphs
        )
    )


def _axisless_runtime_fixture() -> tuple[
    CompiledPrecisionSelectionGroup,
    RuntimeSourceDiscoveryRequest,
    tuple[RuntimeSourceDiscoveryResult, ...],
]:
    config = _model_config("main")
    declaration = ExpectedGraphDeclaration(
        "main",
        "test/main",
        _lifecycle("main"),
    )
    entry = SelectionTopologyEntry(
        entry_id="main.global.weight",
        graph_instance_id="main",
        pattern=SemanticAddressPattern(
            semantic_graph_path="text.decoder",
            path_segments=(
                LiteralPathSegment("global"),
                LiteralPathSegment("weight"),
            ),
            model_part="main",
            module_kind="embedding.global",
            attributes=(),
            parameter_role="kernel",
        ),
        domain=FamilyIndexDomain(None, ()),
        logical_dtype="bfloat16",
        logical_shape=(8, 8),
        logical_axes=("output_features", "input_features"),
    )
    graph = ResolvedGraphTopology(
        declaration=declaration,
        model_family="test-global",
        resolved_model_revision="revision-main",
        adapter_id="test.axisless-runtime.v1",
        decoder_layer_universe=DecoderLayerUniverse((0,), ()),
        entries=(entry,),
        role_definitions=(),
        atomic_groups=(),
        effective_model_config_digest=canonical_model_config_digest(config),
    )
    role_definitions = _merge_selection_role_definitions((graph,), 1)
    topology = ResolvedSelectionTopology(
        schema_version=1,
        graphs=(graph,),
        role_definitions=role_definitions,
        semantic_structure_digest=_compute_semantic_structure_digest(
            schema_version=1,
            graphs=(graph,),
            role_definitions=role_definitions,
        ),
    )
    selection = compile_precision_selection(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        topology,
    )
    graph_request = _build_graph_request(selection, {"main": config}, "main")
    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=(graph_request,),
        trusted_expected_contributors={"main": _expected("main")},
    )
    result = build_runtime_source_discovery_result(
        request=request,
        graph_request=graph_request,
        partition=_partition(graph_request, _expected("main")),
    )
    return selection, request, (result,)


def test_phase_two_binder_retains_exact_selection_and_physical_source_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)

    intents = bind_runtime_source_intents(selection, request, results)

    assert isinstance(intents, CompiledPrecisionIntentGroup)
    assert intents.selection is selection
    assert intents.semantic_structure_digest == selection.semantic_structure_digest
    assert intents.selection_group_id == selection.selection_group_id
    assert intents.runtime_source_digest.startswith("sha256:")
    assert tuple(intent.selection for intent in intents.graph_intents) == (
        *selection.graph_selections,
    )
    binding = intents.source_bindings.graph_bindings[0].canonical_bindings[0]
    assert binding.source_record.source_native_owner_id == "main.model.weight"
    assert binding.source_realizations[0].components[0].physical_shape == (8, 8)


def test_explicit_phase_two_adapter_authority_never_reads_global_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, configs = _selection_fixture()
    request, results = _generation_fixture(selection, configs, "allocation-a")
    runtime_adapters, phase1_requests = _explicit_runtime_adapter_authority(
        selection,
        configs,
    )

    def poison_default_adapters() -> tuple[ModelTopologyAdapter, ...]:
        raise AssertionError("explicit Phase 2 authority consulted global defaults")

    monkeypatch.setattr(topology_module, "_default_adapters", poison_default_adapters)

    intents = bind_runtime_source_intents(
        selection,
        request,
        results,
        runtime_adapters_by_id=runtime_adapters,
        phase1_requests_by_graph=phase1_requests,
    )

    assert intents.selection is selection


class _ObservingRuntimeTopologyAdapter:
    def __init__(
        self,
        delegate: _RuntimeTopologyAdapter,
        *,
        supported: bool = True,
    ) -> None:
        self.adapter_id = delegate.adapter_id
        self._delegate = delegate
        self._supported = supported
        self.seen_configs: list[Mapping[str, object]] = []

    def supports(self, model_config: Mapping[str, object]) -> bool:
        self.seen_configs.append(model_config)
        return self._supported and self._delegate.supports(model_config)

    def classify_graph(
        self,
        schema_version: int,
        graph_input: GraphTopologyInput,
        source_records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment:
        if not self._supported:
            raise AssertionError("unsupported adapter reached source classification")
        return self._delegate.classify_graph(
            schema_version,
            graph_input,
            source_records,
        )


class _SharedRuntimeTopologyAdapter:
    adapter_id = "shared.adapter.v1"

    def __init__(self, graphs: tuple[ResolvedGraphTopology, ...]) -> None:
        self._delegates = {
            graph.declaration.graph_instance_id: _RuntimeTopologyAdapter(
                self.adapter_id,
                graph,
            )
            for graph in graphs
        }
        self.seen_configs: list[Mapping[str, object]] = []

    def supports(self, model_config: Mapping[str, object]) -> bool:
        self.seen_configs.append(model_config)
        graph_id = model_config.get("graph_instance_id")
        return isinstance(graph_id, str) and graph_id in self._delegates

    def classify_graph(
        self,
        schema_version: int,
        graph_input: GraphTopologyInput,
        source_records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment:
        graph_id = graph_input.declaration.graph_instance_id
        return self._delegates[graph_id].classify_graph(
            schema_version,
            graph_input,
            source_records,
        )


def test_explicit_phase_two_rechecks_each_retained_graph_specific_config() -> None:
    selection, configs = _selection_fixture()
    request, results = _generation_fixture(selection, configs, "allocation-a")
    base_adapters, phase1_requests = _explicit_runtime_adapter_authority(
        selection,
        configs,
    )
    adapters = {
        adapter_id: _ObservingRuntimeTopologyAdapter(
            cast(_RuntimeTopologyAdapter, item)
        )
        for adapter_id, item in base_adapters.items()
    }
    runtime_adapters: dict[str, ModelTopologyAdapter] = {
        adapter_id: adapter for adapter_id, adapter in adapters.items()
    }

    bind_runtime_source_intents(
        selection,
        request,
        results,
        runtime_adapters_by_id=runtime_adapters,
        phase1_requests_by_graph=phase1_requests,
    )

    graphs_by_adapter_id = {
        graph.adapter_id: graph for graph in selection.topology.graphs
    }
    for adapter_id, adapter in adapters.items():
        graph_id = graphs_by_adapter_id[adapter_id].declaration.graph_instance_id
        assert adapter.seen_configs == [
            phase1_requests[graph_id].effective_model_config
        ]
        assert adapter.seen_configs[0] is (
            phase1_requests[graph_id].effective_model_config
        )


def test_shared_runtime_adapter_rechecks_every_graph_specific_phase_one_config() -> (
    None
):
    base_selection, configs = _selection_for_graph_ids(("main", "mtp.aux"))
    graphs = tuple(
        replace(graph, adapter_id="shared.adapter.v1")
        for graph in base_selection.topology.graphs
    )
    roles = _merge_selection_role_definitions(graphs, 1)
    topology = ResolvedSelectionTopology(
        schema_version=1,
        graphs=graphs,
        role_definitions=roles,
        semantic_structure_digest=_compute_semantic_structure_digest(
            schema_version=1,
            graphs=graphs,
            role_definitions=roles,
        ),
    )
    selection = compile_precision_selection(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        topology,
    )
    request, results = _generation_fixture(selection, configs, "allocation-a")
    phase1_requests = _retained_phase1_requests(selection, configs)
    adapter = _SharedRuntimeTopologyAdapter(graphs)

    bind_runtime_source_intents(
        selection,
        request,
        results,
        runtime_adapters_by_id={adapter.adapter_id: adapter},
        phase1_requests_by_graph=phase1_requests,
    )

    assert adapter.seen_configs == [
        phase1_requests["main"].effective_model_config,
        phase1_requests["mtp.aux"].effective_model_config,
    ]
    assert adapter.seen_configs[0] is phase1_requests["main"].effective_model_config
    assert adapter.seen_configs[1] is (
        phase1_requests["mtp.aux"].effective_model_config
    )


def test_explicit_phase_two_rejects_adapter_that_no_longer_supports_phase_one() -> None:
    selection, configs = _selection_fixture()
    request, results = _generation_fixture(selection, configs, "allocation-a")
    base_adapters, phase1_requests = _explicit_runtime_adapter_authority(
        selection,
        configs,
    )
    adapters = dict(base_adapters)
    main_adapter_id = selection.topology.graphs[0].adapter_id
    adapters[main_adapter_id] = _ObservingRuntimeTopologyAdapter(
        cast(_RuntimeTopologyAdapter, base_adapters[main_adapter_id]),
        supported=False,
    )

    with pytest.raises(ValueError, match="does not support retained Phase 1 config"):
        bind_runtime_source_intents(
            selection,
            request,
            results,
            runtime_adapters_by_id=adapters,
            phase1_requests_by_graph=phase1_requests,
        )


@pytest.mark.parametrize("coverage", ("missing", "extra"))
def test_explicit_phase_two_requires_exact_selected_adapter_id_coverage(
    coverage: str,
) -> None:
    selection, configs = _selection_fixture()
    request, results = _generation_fixture(selection, configs, "allocation-a")
    runtime_adapters, phase1_requests = _explicit_runtime_adapter_authority(
        selection,
        configs,
    )
    adapters = dict(runtime_adapters)
    if coverage == "missing":
        adapters.pop(selection.topology.graphs[-1].adapter_id)
    else:
        adapters["unused.adapter.v1"] = _RuntimeTopologyAdapter(
            "unused.adapter.v1",
            selection.topology.graphs[0],
        )

    with pytest.raises(ValueError, match="exactly cover selected adapter IDs"):
        bind_runtime_source_intents(
            selection,
            request,
            results,
            runtime_adapters_by_id=adapters,
            phase1_requests_by_graph=phase1_requests,
        )


def test_explicit_phase_two_rejects_mapping_subclasses_before_traversal() -> None:
    selection, configs = _selection_fixture()
    request, results = _generation_fixture(selection, configs, "allocation-a")
    runtime_adapters, phase1_requests = _explicit_runtime_adapter_authority(
        selection,
        configs,
    )

    class PoisonMapping(dict[str, ModelTopologyAdapter]):
        def items(self):
            raise AssertionError("non-exact mapping was traversed")

    with pytest.raises(TypeError, match="exact dictionary"):
        bind_runtime_source_intents(
            selection,
            request,
            results,
            runtime_adapters_by_id=PoisonMapping(runtime_adapters),
            phase1_requests_by_graph=phase1_requests,
        )


def test_explicit_phase_two_rejects_non_exact_mapping_key() -> None:
    selection, configs = _selection_fixture()
    request, results = _generation_fixture(selection, configs, "allocation-a")
    runtime_adapters, phase1_requests = _explicit_runtime_adapter_authority(
        selection,
        configs,
    )

    class AdapterIdSubclass(str):
        pass

    adapters = dict(runtime_adapters)
    adapter_id, adapter = adapters.popitem()
    adapters[AdapterIdSubclass(adapter_id)] = adapter

    with pytest.raises(ValueError, match="canonical text"):
        bind_runtime_source_intents(
            selection,
            request,
            results,
            runtime_adapters_by_id=adapters,
            phase1_requests_by_graph=phase1_requests,
        )


def test_explicit_phase_two_rejects_non_exact_adapter_id_type() -> None:
    selection, configs = _selection_fixture()
    request, results = _generation_fixture(selection, configs, "allocation-a")
    runtime_adapters, phase1_requests = _explicit_runtime_adapter_authority(
        selection,
        configs,
    )

    class AdapterIdSubclass(str):
        pass

    adapters = dict(runtime_adapters)
    adapter_id, adapter = adapters.popitem()
    adapters[adapter_id] = replace(
        cast(_RuntimeTopologyAdapter, adapter),
        adapter_id=AdapterIdSubclass(adapter_id),
    )

    with pytest.raises(TypeError, match="adapter_id must be an exact string"):
        bind_runtime_source_intents(
            selection,
            request,
            results,
            runtime_adapters_by_id=adapters,
            phase1_requests_by_graph=phase1_requests,
        )


@pytest.mark.parametrize("missing", ("adapters", "requests"))
def test_phase_two_rejects_partial_explicit_adapter_authority(missing: str) -> None:
    selection, configs = _selection_fixture()
    request, results = _generation_fixture(selection, configs, "allocation-a")
    runtime_adapters, phase1_requests = _explicit_runtime_adapter_authority(
        selection,
        configs,
    )
    with pytest.raises(TypeError, match="must be supplied together"):
        if missing == "adapters":
            bind_runtime_source_intents(
                selection,
                request,
                results,
                phase1_requests_by_graph=phase1_requests,
            )
        else:
            bind_runtime_source_intents(
                selection,
                request,
                results,
                runtime_adapters_by_id=runtime_adapters,
            )


def test_explicit_runtime_bound_intents_revalidate_without_global_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, configs = _selection_fixture()
    request, results = _generation_fixture(selection, configs, "allocation-a")
    runtime_adapters, phase1_requests = _explicit_runtime_adapter_authority(
        selection,
        configs,
    )
    intents = bind_runtime_source_intents(
        selection,
        request,
        results,
        runtime_adapters_by_id=runtime_adapters,
        phase1_requests_by_graph=phase1_requests,
    )

    def poison_default_adapters() -> tuple[ModelTopologyAdapter, ...]:
        raise AssertionError("explicit validation consulted global defaults")

    monkeypatch.setattr(topology_module, "_default_adapters", poison_default_adapters)

    assert (
        validate_compiled_precision_intent_group(
            intents,
            active_selection=selection,
            active_request=request,
            active_results=results,
            runtime_adapters_by_id=runtime_adapters,
            phase1_requests_by_graph=phase1_requests,
        )
        is intents
    )


def test_phase_two_binder_validates_selection_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)
    calls = [0]
    real_validator = compiler_module.validate_compiled_precision_selection_group

    def counting_validator(
        candidate: CompiledPrecisionSelectionGroup,
    ) -> CompiledPrecisionSelectionGroup:
        calls[0] += 1
        return real_validator(candidate)

    monkeypatch.setattr(
        runtime_binding_module,
        "validate_compiled_precision_selection_group",
        counting_validator,
    )
    monkeypatch.setattr(
        compiler_module,
        "validate_compiled_precision_selection_group",
        counting_validator,
    )

    bind_runtime_source_intents(selection, request, results)

    assert calls == [1]


def test_runtime_bound_intents_validate_after_pickle_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)
    intents = bind_runtime_source_intents(selection, request, results)

    restored = pickle.loads(pickle.dumps(intents))
    restored_selection = pickle.loads(pickle.dumps(selection))
    restored_request = pickle.loads(pickle.dumps(request))
    restored_results = pickle.loads(pickle.dumps(results))

    assert restored == intents
    assert (
        validate_compiled_precision_intent_group(
            restored,
            active_selection=restored_selection,
            active_request=restored_request,
            active_results=restored_results,
        )
        is restored
    )


def test_runtime_source_evidence_receipt_cannot_be_caller_issued() -> None:
    with pytest.raises(TypeError, match="issued only by the Phase 2 binder"):
        RuntimeSourceEvidenceReceipt(
            selection_group_id=_digest("1"),
            request_digest=_digest("2"),
            result_digests=(("main", _digest("3")),),
            source_binding_digest=_digest("4"),
            source_provenance_digest=_digest("5"),
        )


def test_phase_two_binder_rejects_reconstructed_runtime_source_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)
    forged_record = replace(
        results[0].partition.records[0],
        source_native_owner_id="invented.tensor",
    )
    forged_result = RuntimeSourceDiscoveryResult(
        graph_request=results[0].graph_request,
        partition=replace(results[0].partition, records=(forged_record,)),
    )

    with pytest.raises(ValueError, match="receipt source set digest mismatch"):
        bind_runtime_source_intents(
            selection,
            request,
            (forged_result, results[1]),
        )


def test_runtime_bound_intent_validator_rejects_replaced_compiler_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)
    intents = bind_runtime_source_intents(selection, request, results)
    forged_graph_intent = replace(
        intents.graph_intents[0],
        out_of_scope_inventory_entry_ids=("invented.owner",),
        intent_id="",
    )
    forged = replace(
        intents,
        graph_intents=(forged_graph_intent, *intents.graph_intents[1:]),
    )
    with pytest.raises(ValueError, match="runtime-bound compiler output"):
        validate_compiled_precision_intent_group(
            forged,
            active_selection=selection,
            active_request=request,
            active_results=results,
        )


def test_runtime_bound_intent_rejects_cross_generation_receipt_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, configs = _selection_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)
    request_a, results_a = _generation_fixture(selection, configs, "allocation-a")
    request_b, results_b = _generation_fixture(selection, configs, "allocation-b")
    intents_a = bind_runtime_source_intents(selection, request_a, results_a)
    intents_b = bind_runtime_source_intents(selection, request_b, results_b)
    assert intents_a.source_bindings == intents_b.source_bindings
    assert intents_a.runtime_source_receipt != intents_b.runtime_source_receipt
    with pytest.raises(ValueError, match="runtime source provenance"):
        forged = replace(
            intents_a,
            runtime_source_receipt=intents_b.runtime_source_receipt,
            runtime_source_digest=intents_b.runtime_source_digest,
        )
        validate_compiled_precision_intent_group(
            forged,
            active_selection=selection,
            active_request=request_a,
            active_results=results_a,
        )


def test_runtime_bound_intent_requires_independent_active_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, configs = _selection_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)
    request_a, results_a = _generation_fixture(selection, configs, "allocation-a")
    intents_a = bind_runtime_source_intents(selection, request_a, results_a)
    restored_a = pickle.loads(pickle.dumps(intents_a))

    assert (
        validate_compiled_precision_intent_group(
            restored_a,
            active_selection=pickle.loads(pickle.dumps(selection)),
            active_request=pickle.loads(pickle.dumps(request_a)),
            active_results=pickle.loads(pickle.dumps(results_a)),
        )
        is restored_a
    )
    validator = cast(Callable[..., object], validate_compiled_precision_intent_group)
    with pytest.raises(TypeError):
        validator(restored_a)
    assert restored_a.source_topology is not None
    with pytest.raises(TypeError):
        validator(
            restored_a,
            active_selection=selection,
            active_request=restored_a.source_topology.runtime_source_provenance,
            active_results=results_a,
        )


def test_crafted_legacy_anchor_cannot_authorize_runtime_intents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)
    intents = bind_runtime_source_intents(selection, request, results)
    assert intents.source_topology is not None
    group_provenance = intents.source_topology.runtime_source_provenance
    assert group_provenance is not None

    forged_anchor = object.__new__(_ForgedLegacyAnchor)
    forged_anchor.source_provenance = group_provenance
    forged_anchor.anchor_digest = group_provenance.provenance_digest
    transported_anchor = pickle.loads(pickle.dumps(forged_anchor))
    validator = cast(Callable[..., object], validate_compiled_precision_intent_group)

    with pytest.raises(TypeError):
        validator(
            intents,
            active_selection=selection,
            active_request=request,
            active_results=results,
            expected_source_provenance=transported_anchor,
        )


def test_runtime_bound_intent_rejects_full_cross_generation_transplant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, configs = _selection_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)
    request_a, results_a = _generation_fixture(selection, configs, "allocation-a")
    request_b, results_b = _generation_fixture(selection, configs, "allocation-b")
    intents_a = bind_runtime_source_intents(selection, request_a, results_a)
    intents_b = bind_runtime_source_intents(selection, request_b, results_b)
    transplanted = replace(
        pickle.loads(pickle.dumps(intents_a)),
        runtime_source_receipt=intents_b.runtime_source_receipt,
        runtime_source_digest=intents_b.runtime_source_digest,
        source_topology=intents_b.source_topology,
    )
    assert transplanted == intents_b

    with pytest.raises(ValueError, match="runtime-bound compiler output"):
        validate_compiled_precision_intent_group(
            transplanted,
            active_selection=selection,
            active_request=request_a,
            active_results=results_a,
        )


def test_binder_rederives_source_binding_inventory_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)
    intents = bind_runtime_source_intents(selection, request, results)
    assert intents.source_topology is not None
    source_topology = intents.source_topology
    inventory = source_topology.source_bindings
    graph = inventory.graph_bindings[0]
    binding = graph.canonical_bindings[0]
    forged_record = replace(
        binding.source_record,
        source_native_owner_id="invented.tensor",
    )
    forged_binding = replace(binding, source_record=forged_record)
    forged_graph = GraphSemanticSourceBindings(
        graph_instance_id=graph.graph_instance_id,
        normalizer_manifest=graph.normalizer_manifest,
        canonical_bindings=(forged_binding, *graph.canonical_bindings[1:]),
    )
    forged_inventory = SemanticSourceBindingInventory(
        (forged_graph, *inventory.graph_bindings[1:])
    )
    object.__setattr__(
        forged_inventory,
        "source_binding_digest",
        inventory.source_binding_digest,
    )
    forged_topology = copy(source_topology)
    object.__setattr__(forged_topology, "source_bindings", forged_inventory)
    assert intents.runtime_source_receipt is not None

    with pytest.raises(ValueError, match="source binding digest"):
        compiler_module._bind_compiled_precision_intents(
            selection,
            forged_topology,
            intents.runtime_source_receipt,
        )


def test_runtime_validator_rejects_nested_graph_intent_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)
    intents = bind_runtime_source_intents(selection, request, results)
    original = intents.graph_intents[0]

    class ForgedGraphIntent(CompiledGraphPrecisionIntent):
        def __eq__(self, other: object) -> bool:
            return True

    forged_graph_intent = object.__new__(ForgedGraphIntent)
    for item in fields(CompiledGraphPrecisionIntent):
        object.__setattr__(
            forged_graph_intent,
            item.name,
            getattr(original, item.name),
        )
    with pytest.raises(TypeError, match="graph_intents"):
        replace(
            intents,
            graph_intents=(forged_graph_intent, *intents.graph_intents[1:]),
        )

    forged_intents = copy(intents)
    object.__setattr__(
        forged_intents,
        "graph_intents",
        (forged_graph_intent, *intents.graph_intents[1:]),
    )
    with pytest.raises(TypeError, match="exact runtime-bound intent structure"):
        validate_compiled_precision_intent_group(
            forged_intents,
            active_selection=selection,
            active_request=request,
            active_results=results,
        )


@dataclass(frozen=True)
class _GroupedRuntimeTopologyAdapter:
    adapter_id: str
    graph: ResolvedGraphTopology
    use_runtime_logical_shape: bool = False

    def supports(self, model_config: Mapping[str, object]) -> bool:
        raise AssertionError("Phase 2 must not rerun supports-based adapter selection")

    def classify_graph(
        self,
        schema_version: int,
        graph_input: GraphTopologyInput,
        source_records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment:
        assert schema_version == 1
        assert graph_input.declaration == self.graph.declaration
        assert len(source_records) == 1
        record = source_records[0]
        selection_entry = self.graph.entries[0]
        logical_shape = (
            record.shape[2:]
            if self.use_runtime_logical_shape
            else selection_entry.logical_shape
        )
        owner_reference = OwnerFamilyReference("main", "source.moe.routed.gate")
        owner_axes = tuple(
            AxisProjection(axis_name, axis_name)
            for axis_name in selection_entry.domain.axis_names
        )
        member = SemanticTensorFamily(
            pattern=selection_entry.pattern,
            domain=selection_entry.domain,
            format=BF16_FORMAT,
            logical_dtype=selection_entry.logical_dtype,
            logical_shape=logical_shape,
            logical_axes=selection_entry.logical_axes,
            ownership=SemanticOwnership(
                OwnerFamilyBinding(
                    canonical_owner_family=owner_reference,
                    canonical_value_entry_id=selection_entry.entry_id,
                    member_domain=selection_entry.domain,
                    member_to_owner_axes=owner_axes,
                    member_to_value_axes=owner_axes,
                )
            ),
        )
        entry = ParameterInventoryEntry(
            entry_id=selection_entry.entry_id,
            graph_instance_id="main",
            member=member,
            value_provenance=ValueProvenance.TRAINING_PARAMETER,
        )
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
        layer_segment = (
            SourceOrdinalMapSegment(SourceIndexSpan(0, record.shape[0]), 0),
        )
        edge = CanonicalValueClassificationEdge(
            record_id=record.record_id,
            source_region=complete_region,
            output=OutputMemberTarget(
                inventory_entry_id=entry.entry_id,
                member_domain=selection_entry.domain,
                fixed_coordinates=(),
            ),
            canonical_owner_family=owner_reference,
            component_role=LOGICAL_VALUES,
            axis_mappings=(
                SourceToSemanticAxisMapping(
                    0,
                    LayerCoordinateTarget("global_decoder_layer"),
                    layer_segment,
                ),
                SourceToSemanticAxisMapping(
                    0,
                    LayerCoordinateTarget("moe_ordinal"),
                    layer_segment,
                ),
                SourceToSemanticAxisMapping(
                    1,
                    FamilyIndexAxisTarget("expert"),
                    (
                        SourceOrdinalMapSegment(
                            SourceIndexSpan(0, record.shape[1]),
                            0,
                        ),
                    ),
                ),
                SourceToSemanticAxisMapping(
                    2,
                    ComponentAxisTarget(LOGICAL_VALUES, "output_features"),
                    (
                        SourceOrdinalMapSegment(
                            SourceIndexSpan(0, record.shape[2]),
                            0,
                        ),
                    ),
                ),
                SourceToSemanticAxisMapping(
                    3,
                    ComponentAxisTarget(LOGICAL_VALUES, "input_features"),
                    (
                        SourceOrdinalMapSegment(
                            SourceIndexSpan(0, record.shape[3]),
                            0,
                        ),
                    ),
                ),
            ),
        )
        definition = next(
            definition
            for definition in self.graph.role_definitions
            if definition.role_name == "moe.routed_expert"
        )
        return SemanticGraphBuildFragment(
            graph_instance_id="main",
            classification_edges=(edge,),
            source_owners=(
                SourceOwnerInventoryEntry(
                    owner_family=owner_reference,
                    domain=selection_entry.domain,
                    source_mutability=record.source_mutability,
                    mutability_evidence_source=record.mutability_evidence,
                ),
            ),
            inventory_entries=(entry,),
            manifest=SemanticGraphManifest(
                model_family=self.graph.model_family,
                model_revision=self.graph.resolved_model_revision,
                graph_instance_id="main",
                lifecycle=self.graph.declaration.lifecycle,
                inventory_entry_ids=(entry.entry_id,),
            ),
            role_contributions=(
                RoleDefinitionContribution(
                    schema_version=1,
                    role_name=definition.role_name,
                    predicate=definition.predicate,
                    expected_inventory_entry_ids=(entry.entry_id,),
                ),
            ),
        )


def _mixed_boundary_fixture(
    *,
    source_shape: tuple[int, ...] | None = None,
    layer_count: int = 3,
) -> tuple[
    CompiledPrecisionSelectionGroup,
    RuntimeSourceDiscoveryRequest,
    tuple[RuntimeSourceDiscoveryResult, ...],
    _GroupedRuntimeTopologyAdapter,
]:
    if source_shape is None:
        source_shape = (layer_count, 2, 8, 8)
    config = _model_config("main")
    lifecycle = _lifecycle("main")
    declaration = ExpectedGraphDeclaration("main", "test/main", lifecycle)
    domain = FamilyIndexDomain(
        layer_domain=LayerDomain(
            tuple(LayerMember(index, index) for index in range(layer_count))
        ),
        independent_axes=(AxisDomain("expert", (0, 1)),),
    )
    entry = SelectionTopologyEntry(
        entry_id="main.moe.routed.gate",
        graph_instance_id="main",
        pattern=SemanticAddressPattern(
            semantic_graph_path="text.decoder",
            path_segments=(
                LiteralPathSegment("layer"),
                IndexPathSegment("global_decoder_layer"),
                LiteralPathSegment("expert"),
                IndexPathSegment("expert"),
                LiteralPathSegment("gate"),
            ),
            model_part="main",
            module_kind="moe.expert_ffn",
            attributes=(("expert_kind", "routed"), ("projection", "gate")),
            parameter_role="kernel",
        ),
        domain=domain,
        logical_dtype="bfloat16",
        logical_shape=(8, 8),
        logical_axes=("output_features", "input_features"),
    )
    roles = builtin_role_definitions(
        1,
        {
            "moe.routed_expert": RoleExpectedDomain(
                "moe.routed_expert",
                (entry.entry_id,),
            )
        },
    )
    graph = ResolvedGraphTopology(
        declaration=declaration,
        model_family="test-moe",
        resolved_model_revision="revision-main",
        adapter_id="test.grouped-runtime.v1",
        decoder_layer_universe=DecoderLayerUniverse(
            tuple(range(layer_count)),
            tuple(range(layer_count)),
        ),
        entries=(entry,),
        role_definitions=roles,
        atomic_groups=(),
        effective_model_config_digest=canonical_model_config_digest(config),
    )
    topology = ResolvedSelectionTopology(
        schema_version=1,
        graphs=(graph,),
        role_definitions=roles,
        semantic_structure_digest=_compute_semantic_structure_digest(
            schema_version=1,
            graphs=(graph,),
            role_definitions=roles,
        ),
    )
    selection = compile_precision_selection(
        PrecisionPolicyConfig.model_validate(
            {
                "scopes": [
                    {
                        "id": "middle",
                        "roles": ["moe.routed_expert"],
                        "layers": {
                            "index_space": "global_decoder",
                            "exclude_first": 1,
                            "exclude_last": 1,
                        },
                        "training": "bf16",
                        "rollout": "mxfp8",
                    }
                ]
            }
        ),
        topology,
    )
    graph_request = _build_graph_request(selection, {"main": config}, "main")
    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=(graph_request,),
        trusted_expected_contributors={"main": _expected("main")},
    )
    record = replace(
        _source_record("main"),
        record_id="main.moe.routed.gate.source",
        shape=source_shape,
    )
    partition = assemble_runtime_graph_discovery_partition(
        runtime_request=graph_request,
        expected_contributors=_expected("main"),
        contributions=(
            DiscoveryContribution(
                contributor_id="main-rank-0",
                graph_instance_id="main",
                producer_fingerprint=graph_request.source_producer_fingerprint,
                records=(record,),
                storage_realizations=_storage_realizations(record),
            ),
        ),
    )
    result = build_runtime_source_discovery_result(
        request=request,
        graph_request=graph_request,
        partition=partition,
    )
    return (
        selection,
        request,
        (result,),
        _GroupedRuntimeTopologyAdapter(graph.adapter_id, graph),
    )


def test_grouped_source_is_partitioned_by_bf16_boundaries_without_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results, adapter = _mixed_boundary_fixture()
    monkeypatch.setattr(topology_module, "_default_adapters", lambda: (adapter,))

    intents = bind_runtime_source_intents(selection, request, results)

    slices = intents.graph_intent("main").source_binding_slices
    assert len(slices) == 2
    precisions_by_layers = {
        tuple(
            member.global_decoder_layer
            for member in binding.component_key.member_domain.layer_domain.members
        ): (
            binding.training_assignment.precision,
            binding.rollout_assignment.precision,
        )
        for binding in slices
        if binding.component_key.member_domain.layer_domain is not None
    }
    assert precisions_by_layers == {
        (0, 2): ("bf16", "bf16"),
        (1,): ("bf16", "mxfp8"),
    }
    assert all(
        binding.component_key.member_domain.independent_axes[0].members == (0, 1)
        for binding in slices
    )
    source_layer_spans = {
        binding.rollout_assignment.precision: tuple(
            (span.start, span.stop, span.step)
            for span in binding.source_region.axis_selections[0].spans
        )
        for binding in slices
    }
    assert source_layer_spans == {
        "bf16": ((0, 3, 2),),
        "mxfp8": ((1, 2, 1),),
    }
    assert len({id(binding.source_binding) for binding in slices}) == 1


def test_source_slice_assignment_lookup_is_linear_in_edges_plus_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results, adapter = _mixed_boundary_fixture()
    monkeypatch.setattr(topology_module, "_default_adapters", lambda: (adapter,))
    intents = bind_runtime_source_intents(selection, request, results)
    source_topology = intents.source_topology
    assert source_topology is not None
    base_graph_bindings = source_topology.source_bindings.graph_bindings[0]
    base_binding = base_graph_bindings.canonical_bindings[0]
    edge_count = 64
    canonical_bindings: list[CanonicalSourceSemanticBinding] = []
    for edge_index in range(edge_count):
        record_id = f"main.moe.routed.gate.source.{edge_index}"
        native_name = f"model.layers.grouped_gate_{edge_index}.weight"
        record = replace(
            base_binding.source_record,
            record_id=record_id,
            source_native_name=native_name,
            source_native_owner_id=native_name,
        )
        realizations = tuple(
            replace(
                realization,
                realization_id=f"{record_id}.identity",
                output_record_id=record_id,
                components=tuple(
                    replace(
                        component,
                        native_component_id=f"{record_id}.component.{component_index}",
                        source_native_name=native_name,
                    )
                    for component_index, component in enumerate(realization.components)
                ),
            )
            for realization in base_binding.source_realizations
        )
        canonical_bindings.append(
            replace(
                base_binding,
                classification_edge=replace(
                    base_binding.classification_edge,
                    record_id=record_id,
                ),
                source_record=record,
                source_realizations=realizations,
            )
        )
    source_bindings = replace(
        source_topology.source_bindings,
        graph_bindings=(
            replace(
                base_graph_bindings,
                canonical_bindings=tuple(canonical_bindings),
            ),
        ),
    )
    large_source_topology = replace(
        source_topology,
        source_bindings=source_bindings,
    )

    class EntryIdProbe(str):
        comparisons = 0

        def __eq__(self, other: object) -> bool:
            type(self).comparisons += 1
            return super().__eq__(other)

        __hash__ = str.__hash__

    graph_selection = selection.graph_selections[0]
    decoy_count = 256

    def with_decoy_assignments(
        plan: EndpointPrecisionPlan | None,
    ) -> EndpointPrecisionPlan:
        assert plan is not None
        template = plan.assignments[0]
        decoys = tuple(
            replace(
                template,
                inventory_entry_id=EntryIdProbe(f"unused.entry.{index}"),
            )
            for index in range(decoy_count)
        )
        retained = tuple(
            replace(
                assignment,
                inventory_entry_id=EntryIdProbe(assignment.inventory_entry_id),
            )
            for assignment in plan.assignments
        )
        return replace(plan, assignments=(*decoys, *retained))

    large_graph_selection = replace(
        graph_selection,
        training_plan=with_decoy_assignments(graph_selection.training_plan),
        rollout_plan=with_decoy_assignments(graph_selection.rollout_plan),
    )
    EntryIdProbe.comparisons = 0

    slices = compiler_module._runtime_source_binding_slices_for_graph(
        large_graph_selection,
        selection.topology.graphs[0],
        large_source_topology,
    )

    assert len(slices) == edge_count * 2
    assert large_graph_selection.training_plan is not None
    assert large_graph_selection.rollout_plan is not None
    assignment_count = len(large_graph_selection.training_plan.assignments) + len(
        large_graph_selection.rollout_plan.assignments
    )
    assert EntryIdProbe.comparisons <= assignment_count + edge_count * 2


def test_precision_partition_overlay_avoids_pairwise_domain_intersections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_count = 32
    selection, request, results, adapter = _mixed_boundary_fixture(
        layer_count=layer_count
    )
    monkeypatch.setattr(topology_module, "_default_adapters", lambda: (adapter,))
    intents = bind_runtime_source_intents(selection, request, results)
    source_topology = intents.source_topology
    assert source_topology is not None
    graph_selection = selection.graph_selections[0]
    entry = selection.topology.graphs[0].entries[0]
    base_graph_bindings = source_topology.source_bindings.graph_bindings[0]
    base_binding = base_graph_bindings.canonical_bindings[0]
    source_bindings: list[CanonicalSourceSemanticBinding] = []
    for layer_index in range(layer_count):
        record_id = f"main.moe.routed.gate.layer-{layer_index}"
        native_name = f"model.layers.{layer_index}.grouped_gate.weight"
        record = replace(
            base_binding.source_record,
            record_id=record_id,
            source_native_name=native_name,
            source_native_owner_id=native_name,
        )
        layer_span = SourceIndexSpan(layer_index, layer_index + 1)
        source_region = replace(
            base_binding.classification_edge.source_region,
            axis_selections=tuple(
                replace(selection, spans=(layer_span,))
                if selection.axis_index == 0
                else selection
                for selection in (
                    base_binding.classification_edge.source_region.axis_selections
                )
            ),
        )
        member_domain = FamilyIndexDomain(
            layer_domain=LayerDomain((LayerMember(layer_index, layer_index),)),
            independent_axes=entry.domain.independent_axes,
        )
        edge = replace(
            base_binding.classification_edge,
            record_id=record_id,
            source_region=source_region,
            output=replace(
                base_binding.classification_edge.output,
                member_domain=member_domain,
            ),
            axis_mappings=tuple(
                replace(
                    mapping,
                    segments=(SourceOrdinalMapSegment(layer_span, 0),),
                )
                if mapping.source_axis_index == 0
                else mapping
                for mapping in base_binding.classification_edge.axis_mappings
            ),
        )
        realizations = tuple(
            replace(
                realization,
                realization_id=f"{record_id}.identity",
                output_record_id=record_id,
                components=tuple(
                    replace(
                        component,
                        native_component_id=f"{record_id}.component.{component_index}",
                        source_native_name=native_name,
                    )
                    for component_index, component in enumerate(realization.components)
                ),
            )
            for realization in base_binding.source_realizations
        )
        source_bindings.append(
            replace(
                base_binding,
                classification_edge=edge,
                source_record=record,
                source_realizations=realizations,
            )
        )
    partitioned_source_topology = replace(
        source_topology,
        source_bindings=replace(
            source_topology.source_bindings,
            graph_bindings=(
                replace(
                    base_graph_bindings,
                    canonical_bindings=tuple(source_bindings),
                ),
            ),
        ),
    )

    def singleton_partition(
        plan: EndpointPrecisionPlan | None,
    ) -> EndpointPrecisionPlan:
        assert plan is not None
        assignments = []
        for layer_index in range(layer_count):
            member_domain = FamilyIndexDomain(
                layer_domain=LayerDomain((LayerMember(layer_index, layer_index),)),
                independent_axes=entry.domain.independent_axes,
            )
            precision = plan.precision_for(
                entry.entry_id,
                global_decoder_layer=layer_index,
                moe_ordinal=layer_index,
                independent_axes={"expert": 0},
            )
            template = next(
                assignment
                for assignment in plan.assignments
                if assignment.precision == precision
            )
            assignments.append(replace(template, member_domain=member_domain))
        return replace(plan, assignments=tuple(assignments))

    partitioned_selection = replace(
        graph_selection,
        training_plan=singleton_partition(graph_selection.training_plan),
        rollout_plan=singleton_partition(graph_selection.rollout_plan),
    )
    intersection_calls = [0]
    real_intersection = compiler_module._domain_intersection

    def counting_intersection(
        left: FamilyIndexDomain,
        right: FamilyIndexDomain,
    ) -> FamilyIndexDomain | None:
        intersection_calls[0] += 1
        return real_intersection(left, right)

    monkeypatch.setattr(
        compiler_module,
        "_domain_intersection",
        counting_intersection,
    )

    slices = compiler_module._runtime_source_binding_slices_for_graph(
        partitioned_selection,
        selection.topology.graphs[0],
        partitioned_source_topology,
    )

    assert len(slices) == layer_count
    assert intersection_calls[0] <= layer_count * 6


def test_axisless_domain_is_indexed_by_the_precision_overlay() -> None:
    domain = FamilyIndexDomain(None, ())

    postings = compiler_module._factor_postings_for_domains((domain,))

    assert compiler_module._overlapping_domain_ids(domain, postings) == {0}


def test_axisless_global_component_binds_one_complete_source_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _axisless_runtime_fixture()
    _install_runtime_topology_adapters(monkeypatch, selection)

    intents = bind_runtime_source_intents(selection, request, results)

    slices = intents.graph_intent("main").source_binding_slices
    assert len(slices) == 1
    assert slices[0].component_key.member_domain == FamilyIndexDomain(None, ())
    assert slices[0].source_region == SourceRegion(
        source_shape=(8, 8),
        axis_selections=(
            SourceAxisSelection(0, (SourceIndexSpan(0, 8),)),
            SourceAxisSelection(1, (SourceIndexSpan(0, 8),)),
        ),
    )


def test_phase_two_rejects_runtime_semantic_shape_that_differs_from_phase_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results, adapter = _mixed_boundary_fixture(
        source_shape=(3, 2, 4, 16)
    )
    adapter = replace(adapter, use_runtime_logical_shape=True)
    monkeypatch.setattr(topology_module, "_default_adapters", lambda: (adapter,))

    with pytest.raises(
        ValueError,
        match="runtime semantic entry differs from Phase 1",
    ):
        bind_runtime_source_intents(selection, request, results)


def test_graph_request_derives_every_phase_one_identity_from_selection() -> None:
    selection, configs = _selection_fixture()

    request = _build_graph_request(selection, configs, "main")
    graph = selection.topology.graphs[0]

    assert request.declaration is graph.declaration
    assert request.resolved_graph is graph
    assert request.resolved_model_revision == graph.resolved_model_revision
    assert request.resolved_graph.adapter_id == "adapter-main.v1"
    assert request.semantic_structure_digest == selection.semantic_structure_digest
    assert request.selection_group_id == selection.selection_group_id
    assert request.expected_contributor_authority == _expected("main").to_authority()


def test_graph_request_rejects_runtime_config_that_differs_from_phase_one() -> None:
    selection, configs = _selection_fixture()
    changed = dict(configs["main"])
    changed["hidden_size"] = 16

    with pytest.raises(ValueError, match="effective model config digest"):
        build_runtime_graph_source_request(
            selection=selection,
            graph_instance_id="main",
            model_config=changed,
            source_producer_fingerprint=_fingerprint(),
            expected_contributors=_expected("main"),
            source_identity=_evidence("main-source", "4"),
            artifact_identity=_evidence("main-artifact", "5"),
            source_allocation_generation="allocation-1",
        )


def test_graph_request_snapshots_stateful_model_config_exactly_once() -> None:
    selection, configs = _selection_fixture()
    changed = dict(configs["main"])
    changed["hidden_size"] = 16
    stateful = _TwoViewConfig(configs["main"], changed)

    request = build_runtime_graph_source_request(
        selection=selection,
        graph_instance_id="main",
        model_config=stateful,
        source_producer_fingerprint=_fingerprint(),
        expected_contributors=_expected("main"),
        source_identity=_evidence("main-source", "4"),
        artifact_identity=_evidence("main-artifact", "5"),
        source_allocation_generation="allocation-1",
    )

    assert stateful.items_calls == 1
    assert canonical_model_config_digest(request.model_config) == (
        selection.topology.graphs[0].effective_model_config_digest
    )


def test_graph_request_rejects_static_checkpoint_graph() -> None:
    selection, configs = _selection_fixture()

    with pytest.raises(ValueError, match="training-runtime"):
        _build_graph_request(selection, configs, "draft.static")


def test_graph_request_rejects_a_selection_with_forged_identity() -> None:
    selection, configs = _selection_fixture()
    forged = replace(selection, topology=selection.topology)
    object.__setattr__(forged, "selection_group_id", _digest("0"))

    with pytest.raises(ValueError, match="selection_group_id"):
        _build_graph_request(forged, configs, "main")


def test_aggregate_request_includes_training_only_graph_and_omits_static_graph() -> (
    None
):
    selection, configs = _selection_fixture()
    main, mtp = _runtime_requests(selection, configs)

    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=(mtp, main),
        trusted_expected_contributors={
            "mtp.aux": _expected("mtp.aux"),
            "main": _expected("main"),
        },
    )

    assert tuple(
        item.declaration.graph_instance_id for item in request.graph_requests
    ) == ("main", "mtp.aux")
    assert tuple(graph_id for graph_id, _ in request.trusted_expected_contributors) == (
        "main",
        "mtp.aux",
    )
    assert request.semantic_structure_digest == selection.semantic_structure_digest
    assert request.selection_group_id == selection.selection_group_id
    assert tuple(
        graph.declaration.graph_instance_id for graph in selection.topology.graphs
    ) == ("main", "draft.static", "mtp.aux")


@pytest.mark.parametrize("case", ("missing", "duplicate", "static"))
def test_aggregate_request_rejects_inexact_runtime_graph_coverage(case: str) -> None:
    selection, configs = _selection_fixture()
    main, mtp = _runtime_requests(selection, configs)
    graph_requests = {
        "missing": (main,),
        "duplicate": (main, main, mtp),
        "static": (
            main,
            mtp,
            _manual_graph_request(selection, configs, "draft.static"),
        ),
    }[case]
    trusted = {
        item.declaration.graph_instance_id: _expected(
            item.declaration.graph_instance_id
        )
        for item in graph_requests
    }

    with pytest.raises(ValueError, match="runtime graph"):
        build_runtime_source_discovery_request(
            selection=selection,
            graph_requests=graph_requests,
            trusted_expected_contributors=trusted,
        )


@pytest.mark.parametrize("mutation", ("adapter", "revision", "shape"))
def test_aggregate_request_rejects_self_consistent_changed_graph(
    mutation: str,
) -> None:
    selection, configs = _selection_fixture()
    main, mtp = _runtime_requests(selection, configs)
    graph = main.resolved_graph
    if mutation == "adapter":
        changed_graph = replace(graph, adapter_id="forged.adapter.v1")
    elif mutation == "revision":
        changed_graph = replace(graph, resolved_model_revision="forged-revision")
    else:
        changed_graph = replace(
            graph,
            entries=(replace(graph.entries[0], logical_shape=(16, 8)),),
        )
    changed_main = _manual_graph_request(
        selection,
        configs,
        "main",
        graph=changed_graph,
    )

    with pytest.raises(ValueError, match="resolved graph"):
        build_runtime_source_discovery_request(
            selection=selection,
            graph_requests=(changed_main, mtp),
            trusted_expected_contributors={
                "main": _expected("main"),
                "mtp.aux": _expected("mtp.aux"),
            },
        )


def test_aggregate_validator_rejects_coordinated_invented_selection_ids() -> None:
    selection, configs = _selection_fixture()
    invented_semantic_digest = _digest("7")
    invented_selection_id = _digest("8")
    graph_requests = tuple(
        _manual_graph_request(
            selection,
            configs,
            graph_instance_id,
            semantic_structure_digest=invented_semantic_digest,
            selection_group_id=invented_selection_id,
        )
        for graph_instance_id in ("main", "mtp.aux")
    )
    request = RuntimeSourceDiscoveryRequest(
        graph_requests=graph_requests,
        trusted_expected_contributors=tuple(
            (graph_instance_id, _expected(graph_instance_id))
            for graph_instance_id in ("main", "mtp.aux")
        ),
    )

    assert request.semantic_structure_digest == invented_semantic_digest
    assert request.selection_group_id == invented_selection_id
    with pytest.raises(ValueError, match="semantic_structure_digest"):
        validate_runtime_source_discovery_request(selection, request)


def test_aggregate_request_digest_is_derived_and_revalidated() -> None:
    selection, configs = _selection_fixture()
    main, mtp = _runtime_requests(selection, configs)
    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=(main, mtp),
        trusted_expected_contributors={
            "main": _expected("main"),
            "mtp.aux": _expected("mtp.aux"),
        },
    )
    forged = copy(request)
    object.__setattr__(forged, "request_digest", _digest("0"))

    assert {item.name: item.init for item in fields(RuntimeSourceDiscoveryRequest)} == {
        "graph_requests": True,
        "trusted_expected_contributors": True,
        "semantic_structure_digest": False,
        "selection_group_id": False,
        "request_digest": False,
    }
    with pytest.raises(ValueError, match="request_digest"):
        validate_runtime_source_discovery_request(selection, forged)


def test_aggregate_request_is_order_independent_and_pickle_stable() -> None:
    selection, configs = _selection_fixture()
    main, mtp = _runtime_requests(selection, configs)
    first = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=(main, mtp),
        trusted_expected_contributors={
            "main": _expected("main"),
            "mtp.aux": _expected("mtp.aux"),
        },
    )
    second = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=(mtp, main),
        trusted_expected_contributors={
            "mtp.aux": _expected("mtp.aux"),
            "main": _expected("main"),
        },
    )
    restored = pickle.loads(pickle.dumps(first))

    assert second == first
    assert second.request_digest == first.request_digest
    assert restored == first
    assert validate_runtime_source_discovery_request(selection, restored) is restored


def test_bulk_request_factory_validates_selection_once_for_many_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_graph_ids = ("main", *(f"mtp.{index:02d}" for index in range(8)))
    selection, configs = _selection_for_graph_ids((*runtime_graph_ids, "draft.static"))
    contexts = {
        graph_id: _runtime_context(graph_id, configs[graph_id])
        for graph_id in reversed(runtime_graph_ids)
    }
    validation_calls = [0]
    request_identity_calls = [0]
    real_validator = runtime_binding_module.validate_compiled_precision_selection_group
    real_request_identity = (
        runtime_binding_module.runtime_source_request_identity_digest
    )

    def counting_validator(
        candidate: CompiledPrecisionSelectionGroup,
    ) -> CompiledPrecisionSelectionGroup:
        validation_calls[0] += 1
        return real_validator(candidate)

    def counting_request_identity(candidate: RuntimeGraphSourceRequest) -> str:
        request_identity_calls[0] += 1
        return real_request_identity(candidate)

    monkeypatch.setattr(
        runtime_binding_module,
        "validate_compiled_precision_selection_group",
        counting_validator,
    )
    monkeypatch.setattr(
        runtime_binding_module,
        "runtime_source_request_identity_digest",
        counting_request_identity,
    )

    request = build_runtime_source_discovery_request_from_contexts(
        selection=selection,
        contexts=contexts,
    )

    assert validation_calls == [1]
    assert request_identity_calls == [len(runtime_graph_ids)]
    assert (
        tuple(
            graph_request.declaration.graph_instance_id
            for graph_request in request.graph_requests
        )
        == runtime_graph_ids
    )


def test_bulk_request_factory_snapshots_each_runtime_config_once() -> None:
    selection, configs = _selection_fixture()
    changed = dict(configs["main"])
    changed["hidden_size"] = 16
    stateful = _TwoViewConfig(configs["main"], changed)
    contexts = {
        "mtp.aux": _runtime_context("mtp.aux", configs["mtp.aux"]),
        "main": _runtime_context("main", stateful),
    }

    request = build_runtime_source_discovery_request_from_contexts(
        selection=selection,
        contexts=contexts,
    )

    assert stateful.items_calls == 1
    assert canonical_model_config_digest(request.graph_requests[0].model_config) == (
        selection.topology.graphs[0].effective_model_config_digest
    )


def test_bulk_request_semantic_config_digest_streams_frozen_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    large_config: dict[str, object] = {
        "architectures": ["TestForCausalLM"],
        "graph_instance_id": "main",
        "model_type": "test",
        **{f"key-{index:05d}": index for index in range(10_000)},
    }
    selection, configs = _selection_for_graph_ids(
        ("main",),
        model_configs={"main": large_config},
    )
    probe = _manual_graph_request(selection, configs, "main")
    frozen_mapping_type = type(probe.model_config)

    def fail_quadratic_lookup(_self: object, _key: str) -> object:
        raise AssertionError("semantic config digest must stream frozen entries")

    monkeypatch.setattr(frozen_mapping_type, "__getitem__", fail_quadratic_lookup)

    request = build_runtime_source_discovery_request_from_contexts(
        selection=selection,
        contexts={"main": _runtime_context("main", configs["main"])},
    )

    assert len(request.graph_requests[0].model_config) == len(large_config)


def test_discovery_result_derives_request_and_partition_identity() -> None:
    selection, request, results = _aggregate_fixture()
    result = results[0]

    assert result.graph_request is request.graph_requests[0]
    assert result.partition.graph_instance_id == "main"
    assert result.graph_instance_id == "main"
    assert (
        result.runtime_source_request_digest
        == request.graph_requests[0].runtime_source_request_digest
    )
    assert result.semantic_structure_digest == selection.semantic_structure_digest
    assert result.selection_group_id == selection.selection_group_id
    assert (
        result.producer_fingerprint
        == request.graph_requests[0].source_producer_fingerprint
    )
    assert result.result_digest.startswith("sha256:")


def test_bulk_result_factory_snapshots_aggregate_once_for_many_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_graph_ids = ("main", *(f"mtp.{index:02d}" for index in range(8)))
    selection, configs = _selection_for_graph_ids((*runtime_graph_ids, "draft.static"))
    request = build_runtime_source_discovery_request_from_contexts(
        selection=selection,
        contexts={
            graph_id: _runtime_context(graph_id, configs[graph_id])
            for graph_id in runtime_graph_ids
        },
    )
    partitions = tuple(
        _partition(
            graph_request, _expected(graph_request.declaration.graph_instance_id)
        )
        for graph_request in reversed(request.graph_requests)
    )
    snapshot_calls = [0]
    inventory_calls = [0]
    request_identity_calls = [0]
    real_snapshot = runtime_binding_module._validate_aggregate_request_snapshot
    real_inventory_validator = (
        runtime_binding_module.validate_runtime_discovery_inventory
    )
    real_request_identity = (
        runtime_binding_module.runtime_source_request_identity_digest
    )

    def counting_snapshot(candidate: object) -> RuntimeSourceDiscoveryRequest:
        snapshot_calls[0] += 1
        return real_snapshot(candidate)

    def counting_inventory_validator(
        runtime_requests: Sequence[RuntimeGraphSourceRequest],
        source_discovery: SourceDiscoveryInventory,
        expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
    ) -> SourceDiscoveryInventory:
        inventory_calls[0] += 1
        return real_inventory_validator(
            runtime_requests,
            source_discovery,
            expected_contributors_by_graph,
        )

    def counting_request_identity(candidate: RuntimeGraphSourceRequest) -> str:
        request_identity_calls[0] += 1
        return real_request_identity(candidate)

    monkeypatch.setattr(
        runtime_binding_module,
        "_validate_aggregate_request_snapshot",
        counting_snapshot,
    )
    monkeypatch.setattr(
        runtime_binding_module,
        "validate_runtime_discovery_inventory",
        counting_inventory_validator,
    )
    monkeypatch.setattr(
        runtime_binding_module,
        "runtime_source_request_identity_digest",
        counting_request_identity,
    )

    results = build_runtime_source_discovery_results(
        request=request,
        partitions=partitions,
    )

    assert snapshot_calls == [1]
    assert inventory_calls == [1]
    assert request_identity_calls == [len(runtime_graph_ids)]
    assert tuple(result.graph_instance_id for result in results) == runtime_graph_ids


@pytest.mark.parametrize("builder", ("single", "bulk"))
@pytest.mark.parametrize("mutation", ("receipt", "record"))
def test_result_builders_do_not_publish_unvalidated_partitions(
    builder: str,
    mutation: str,
) -> None:
    selection, configs = _selection_fixture()
    request = build_runtime_source_discovery_request_from_contexts(
        selection=selection,
        contexts={
            graph_id: _runtime_context(graph_id, configs[graph_id])
            for graph_id in ("main", "mtp.aux")
        },
    )
    main_request, mtp_request = request.graph_requests
    bad_partition = copy(_partition(main_request, _expected("main")))
    if mutation == "receipt":
        bad_receipt = copy(bad_partition.completeness_receipt)
        object.__setattr__(bad_receipt, "request_digest", _digest("0"))
        object.__setattr__(bad_partition, "completeness_receipt", bad_receipt)
    else:
        bad_record = copy(bad_partition.records[0])
        object.__setattr__(bad_record, "numeric_encoding", "forged_bfloat16")
        object.__setattr__(bad_partition, "records", (bad_record,))

    with pytest.raises(ValueError):
        if builder == "single":
            build_runtime_source_discovery_result(
                request=request,
                graph_request=main_request,
                partition=bad_partition,
            )
        else:
            build_runtime_source_discovery_results(
                request=request,
                partitions=(
                    bad_partition,
                    _partition(mtp_request, _expected("mtp.aux")),
                ),
            )


def test_complete_results_are_validated_once_as_one_canonical_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    calls = [0]
    real_validator = runtime_binding_module.validate_runtime_discovery_inventory

    def counting_validator(
        runtime_requests: Sequence[RuntimeGraphSourceRequest],
        source_discovery: SourceDiscoveryInventory,
        expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
    ) -> SourceDiscoveryInventory:
        calls[0] += 1
        return real_validator(
            runtime_requests,
            source_discovery,
            expected_contributors_by_graph,
        )

    monkeypatch.setattr(
        runtime_binding_module,
        "validate_runtime_discovery_inventory",
        counting_validator,
    )

    inventory = validate_runtime_source_discovery_results(
        selection,
        request,
        tuple(reversed(results)),
    )

    assert calls == [1]
    assert tuple(partition.graph_instance_id for partition in inventory.partitions) == (
        "main",
        "mtp.aux",
    )


@pytest.mark.parametrize("case", ("missing", "duplicate", "extra", "stale"))
def test_missing_extra_duplicate_or_stale_results_fail_atomically(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    main_result, mtp_result = results
    if case == "missing":
        candidate = (main_result,)
    elif case == "duplicate":
        candidate = (main_result, main_result, mtp_result)
    elif case == "extra":
        _, configs = _selection_fixture()
        static_request = _manual_graph_request(
            selection,
            configs,
            "draft.static",
        )
        static_result = RuntimeSourceDiscoveryResult(
            graph_request=static_request,
            partition=_partition(static_request, _expected("draft.static")),
        )
        candidate = (*results, static_result)
    else:
        stale_graph_requests = tuple(
            _manual_graph_request(
                selection,
                {graph_id: _model_config(graph_id) for graph_id in ("main", "mtp.aux")},
                graph_id,
                allocation_generation="allocation-2",
            )
            for graph_id in ("main", "mtp.aux")
        )
        stale_request = build_runtime_source_discovery_request(
            selection=selection,
            graph_requests=stale_graph_requests,
            trusted_expected_contributors={
                "main": _expected("main"),
                "mtp.aux": _expected("mtp.aux"),
            },
        )
        stale_main = stale_request.graph_requests[0]
        candidate = (
            build_runtime_source_discovery_result(
                request=stale_request,
                graph_request=stale_main,
                partition=_partition(stale_main, _expected("main")),
            ),
            mtp_result,
        )
    calls = [0]

    def should_not_run(
        runtime_requests: Sequence[RuntimeGraphSourceRequest],
        source_discovery: SourceDiscoveryInventory,
        expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
    ) -> SourceDiscoveryInventory:
        del runtime_requests, source_discovery, expected_contributors_by_graph
        calls[0] += 1
        raise AssertionError("whole-inventory validation must follow result preflight")

    monkeypatch.setattr(
        runtime_binding_module,
        "validate_runtime_discovery_inventory",
        should_not_run,
    )

    with pytest.raises(ValueError, match="runtime result"):
        validate_runtime_source_discovery_results(selection, request, candidate)
    assert calls == [0]


def test_result_factory_rejects_partition_for_another_graph() -> None:
    _, request, _ = _aggregate_fixture()
    main, mtp = request.graph_requests

    with pytest.raises(ValueError, match="partition"):
        build_runtime_source_discovery_result(
            request=request,
            graph_request=main,
            partition=_partition(mtp, _expected("mtp.aux")),
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "graph_instance_id",
        "runtime_source_request_digest",
        "semantic_structure_digest",
        "selection_group_id",
        "producer_fingerprint",
        "result_digest",
    ),
)
def test_result_validator_rejects_forged_derived_fields(field_name: str) -> None:
    selection, request, results = _aggregate_fixture()
    forged = copy(results[0])
    replacement: object = (
        replace(
            _fingerprint(),
            producer_implementation_id="forged.runtime-metadata",
        )
        if field_name == "producer_fingerprint"
        else ("forged" if field_name == "graph_instance_id" else _digest("0"))
    )
    object.__setattr__(forged, field_name, replacement)

    with pytest.raises(ValueError, match=field_name):
        validate_runtime_source_discovery_results(
            selection,
            request,
            (forged, results[1]),
        )


def test_result_validator_accepts_structurally_equal_transported_fingerprint() -> None:
    selection, request, results = _aggregate_fixture()
    forged = copy(results[0])
    replacement = replace(forged.producer_fingerprint)
    assert replacement == forged.producer_fingerprint
    assert replacement is not forged.producer_fingerprint
    object.__setattr__(forged, "producer_fingerprint", replacement)

    validate_runtime_source_discovery_results(
        selection,
        request,
        (forged, results[1]),
    )


def test_result_validator_rejects_mutated_result_roots_before_field_access() -> None:
    selection, request, results = _aggregate_fixture()
    forged_request = copy(results[0])
    forged_partition = copy(results[0])
    forged_partition_graph = copy(results[0])
    object.__setattr__(forged_request, "graph_request", object())
    object.__setattr__(forged_partition, "partition", object())
    mutated_partition = copy(forged_partition_graph.partition)
    object.__setattr__(mutated_partition, "graph_instance_id", object())
    object.__setattr__(forged_partition_graph, "partition", mutated_partition)

    with pytest.raises(TypeError, match="graph_request"):
        validate_runtime_source_discovery_results(
            selection,
            request,
            (forged_request, results[1]),
        )
    with pytest.raises(TypeError, match="partition"):
        validate_runtime_source_discovery_results(
            selection,
            request,
            (forged_partition, results[1]),
        )
    with pytest.raises(TypeError, match="partition graph_instance_id"):
        validate_runtime_source_discovery_results(
            selection,
            request,
            (forged_partition_graph, results[1]),
        )


def test_result_validator_validates_partition_tree_before_deriving_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, results = _aggregate_fixture()
    forged = copy(results[0])
    bad_partition = copy(forged.partition)
    bad_receipt = copy(bad_partition.completeness_receipt)
    object.__setattr__(bad_receipt, "request_digest", _digest("0"))
    object.__setattr__(bad_partition, "completeness_receipt", bad_receipt)
    object.__setattr__(forged, "partition", bad_partition)
    digest_calls = [0]

    def should_not_run(
        graph_request: RuntimeGraphSourceRequest,
        partition: GraphDiscoveryPartition,
    ) -> str:
        del graph_request, partition
        digest_calls[0] += 1
        raise AssertionError("result digest must follow exact inventory validation")

    monkeypatch.setattr(runtime_binding_module, "_result_digest", should_not_run)

    with pytest.raises(ValueError, match="receipt runtime source request digest"):
        validate_runtime_source_discovery_results(
            selection,
            request,
            (forged, results[1]),
        )
    assert digest_calls == [0]


def test_result_derived_fields_are_not_constructor_inputs_and_pickle_is_stable() -> (
    None
):
    selection, request, results = _aggregate_fixture()
    restored = pickle.loads(pickle.dumps(results))

    assert {item.name: item.init for item in fields(RuntimeSourceDiscoveryResult)} == {
        "graph_request": True,
        "partition": True,
        "graph_instance_id": False,
        "runtime_source_request_digest": False,
        "semantic_structure_digest": False,
        "selection_group_id": False,
        "producer_fingerprint": False,
        "result_digest": False,
    }
    assert restored == results
    assert validate_runtime_source_discovery_results(
        selection,
        request,
        restored,
    ) == SourceDiscoveryInventory(tuple(result.partition for result in results))


def test_public_factories_snapshot_caller_mappings_and_sequences_once() -> None:
    selection, configs = _selection_fixture()
    main, mtp = _runtime_requests(selection, configs)
    graph_requests = _OneShotSequence((mtp, main))
    trusted = _OneShotMapping(
        {
            "mtp.aux": _expected("mtp.aux"),
            "main": _expected("main"),
        }
    )

    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=graph_requests,
        trusted_expected_contributors=trusted,
    )
    partitions = _OneShotSequence(
        tuple(
            _partition(
                graph_request,
                _expected(graph_request.declaration.graph_instance_id),
            )
            for graph_request in reversed(request.graph_requests)
        )
    )
    results = build_runtime_source_discovery_results(
        request=request,
        partitions=partitions,
    )
    result_sequence = _OneShotSequence(tuple(reversed(results)))

    validate_runtime_source_discovery_results(
        selection,
        request,
        result_sequence,
    )

    assert graph_requests.iter_calls == 1
    assert trusted.items_calls == 1
    assert partitions.iter_calls == 1
    assert result_sequence.iter_calls == 1


def test_runtime_binding_rejects_transport_subclasses_and_mutated_containers() -> None:
    selection, request, results = _aggregate_fixture()

    class RequestSubclass(RuntimeSourceDiscoveryRequest):
        pass

    class ResultSubclass(RuntimeSourceDiscoveryResult):
        pass

    request_subclass = RequestSubclass(
        graph_requests=request.graph_requests,
        trusted_expected_contributors=request.trusted_expected_contributors,
    )
    result_subclass = ResultSubclass(
        graph_request=results[0].graph_request,
        partition=results[0].partition,
    )
    mutated_request = copy(request)
    object.__setattr__(mutated_request, "graph_requests", list(request.graph_requests))

    with pytest.raises(TypeError, match="exact RuntimeSourceDiscoveryRequest"):
        validate_runtime_source_discovery_request(selection, request_subclass)
    with pytest.raises(TypeError, match="exact RuntimeSourceDiscoveryResult"):
        validate_runtime_source_discovery_results(
            selection,
            request,
            (result_subclass, results[1]),
        )
    with pytest.raises(TypeError, match="exact tuple"):
        validate_runtime_source_discovery_request(selection, mutated_request)


def test_runtime_graph_context_is_an_exact_frozen_ephemeral_envelope() -> None:
    context = _runtime_context("main", _model_config("main"))

    assert {item.name: item.init for item in fields(RuntimeGraphSourceContext)} == {
        "graph_instance_id": True,
        "model_config": True,
        "source_producer_fingerprint": True,
        "expected_contributors": True,
        "source_identity": True,
        "artifact_identity": True,
        "source_allocation_generation": True,
    }
    with pytest.raises(FrozenInstanceError):
        setattr(context, "graph_instance_id", "mtp.forged")

    class ContextSubclass(RuntimeGraphSourceContext):
        pass

    selection, configs = _selection_fixture()
    subclass = ContextSubclass(
        graph_instance_id="main",
        model_config=configs["main"],
        source_producer_fingerprint=_fingerprint(),
        expected_contributors=_expected("main"),
        source_identity=_evidence("main-source", "4"),
        artifact_identity=_evidence("main-artifact", "5"),
        source_allocation_generation="allocation-1",
    )
    with pytest.raises(TypeError, match="exact RuntimeGraphSourceContext"):
        build_runtime_source_discovery_request_from_contexts(
            selection=selection,
            contexts={
                "main": subclass,
                "mtp.aux": _runtime_context("mtp.aux", configs["mtp.aux"]),
            },
        )


def test_runtime_binding_imports_without_training_or_generation_frameworks() -> None:
    script = r"""
import importlib.abc
import sys
from pathlib import Path
from types import ModuleType

blocked = (
    "torch",
    "ray",
    "megatron",
    "nemo_automodel",
    "nemo_gym",
    "transformer_engine",
    "transformers",
    "vllm",
)

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"forbidden framework import: {fullname}")
        return None

# Isolate this module-graph contract from unrelated process-wide bootstraps in
# nemo_rl.__init__, which may intentionally initialize optional frameworks.
nemo_rl_package = ModuleType("nemo_rl")
nemo_rl_package.__package__ = "nemo_rl"
nemo_rl_package.__path__ = [str(Path.cwd() / "nemo_rl")]
sys.modules["nemo_rl"] = nemo_rl_package
sys.meta_path.insert(0, Blocker())
import nemo_rl.precision_policy.runtime_binding
import nemo_rl.precision_policy.discovery_producers
assert not any(name.split(".", 1)[0] in blocked for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_runtime_source_dispatcher_contract_has_lazy_public_exports() -> None:
    import nemo_rl.precision_policy as precision_policy

    assert precision_policy.bind_runtime_source_intents is bind_runtime_source_intents
    assert precision_policy.SourceMetadataProducer is SourceMetadataProducer
    assert (
        precision_policy.produce_runtime_source_discovery_results
        is produce_runtime_source_discovery_results
    )
    assert (
        precision_policy.source_producer_fingerprint_identity_digest
        is source_producer_fingerprint_identity_digest
    )
    assert (
        precision_policy.validate_source_producer_fingerprint
        is validate_source_producer_fingerprint
    )
    assert (
        precision_policy.validate_compiled_precision_intent_group
        is validate_compiled_precision_intent_group
    )


def test_source_producer_fingerprint_public_identity_is_exact_and_pickle_stable() -> (
    None
):
    fingerprint = _fingerprint()
    restored = pickle.loads(pickle.dumps(fingerprint))

    assert validate_source_producer_fingerprint(fingerprint) is fingerprint
    assert validate_source_producer_fingerprint(restored) is restored
    assert source_producer_fingerprint_identity_digest(fingerprint) == (
        source_producer_fingerprint_identity_digest(restored)
    )


def test_source_producer_fingerprint_public_identity_rejects_forged_subclass() -> None:
    fingerprint = _fingerprint()

    class FingerprintSubclass(SourceProducerFingerprint):
        pass

    subclass = FingerprintSubclass(
        schema_id=fingerprint.schema_id,
        producer_implementation_id=fingerprint.producer_implementation_id,
        producer_revision=fingerprint.producer_revision,
        normalization_contract_digest=fingerprint.normalization_contract_digest,
        evidence=fingerprint.evidence,
    )

    with pytest.raises(TypeError, match="exact SourceProducerFingerprint"):
        validate_source_producer_fingerprint(subclass)
    with pytest.raises(TypeError, match="exact SourceProducerFingerprint"):
        source_producer_fingerprint_identity_digest(object())


@pytest.mark.parametrize("mutation", ("schema", "digest", "evidence"))
def test_source_producer_fingerprint_public_identity_rejects_mutation(
    mutation: str,
) -> None:
    fingerprint = copy(_fingerprint())
    if mutation == "schema":
        schema = copy(fingerprint.schema_id)
        object.__setattr__(schema, "value", "forged")
        object.__setattr__(fingerprint, "schema_id", schema)
    elif mutation == "digest":
        object.__setattr__(fingerprint, "normalization_contract_digest", object())
    else:
        evidence = copy(fingerprint.evidence)
        object.__setattr__(evidence, "locator", " forged ")
        object.__setattr__(fingerprint, "evidence", evidence)

    with pytest.raises((TypeError, ValueError)):
        validate_source_producer_fingerprint(fingerprint)


def test_runtime_source_dispatcher_covers_only_runtime_graphs_in_canonical_order() -> (
    None
):
    selection, request, _ = _aggregate_fixture()
    producer = _TestSourceMetadataProducer(_fingerprint())
    bindings = _OneShotProducerMapping((("mtp.aux", producer), ("main", producer)))

    results = produce_runtime_source_discovery_results(
        selection=selection,
        request=request,
        producers_by_graph=bindings,
    )

    assert bindings.items_calls == 1
    assert tuple(result.graph_instance_id for result in results) == (
        "main",
        "mtp.aux",
    )
    assert tuple(
        result.graph_request is graph_request
        for result, graph_request in zip(results, request.graph_requests, strict=True)
    ) == (True, True)
    assert producer.discover_graph_ids == ["main", "mtp.aux"]
    assert tuple(
        received is expected
        for received, expected in zip(
            producer.received_requests,
            request.graph_requests,
            strict=True,
        )
    ) == (True, True)
    trusted_by_graph = dict(request.trusted_expected_contributors)
    assert tuple(
        received is trusted_by_graph[graph_id]
        for graph_id, received in zip(
            ("main", "mtp.aux"),
            producer.received_expected_contributors,
            strict=True,
        )
    ) == (True, True)
    assert "draft.static" not in producer.discover_graph_ids


@pytest.mark.parametrize("coverage", ("missing", "extra"))
def test_runtime_source_dispatcher_rejects_non_exact_graph_bindings_before_discovery(
    coverage: str,
) -> None:
    selection, request, _ = _aggregate_fixture()
    producer = _TestSourceMetadataProducer(_fingerprint())
    bindings = (
        {"main": producer}
        if coverage == "missing"
        else {
            "main": producer,
            "mtp.aux": producer,
            "draft.static": producer,
        }
    )

    with pytest.raises(ValueError, match="producer.*coverage"):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph=bindings,
        )

    assert producer.discover_graph_ids == []
    assert producer.producer_id_reads == 0
    assert producer.schema_id_reads == 0
    assert producer.fingerprint_calls == 0


def test_runtime_source_dispatcher_rejects_non_exact_graph_id_before_discovery() -> (
    None
):
    selection, request, _ = _aggregate_fixture()
    producer = _TestSourceMetadataProducer(_fingerprint())

    class GraphIdSubclass(str):
        pass

    bindings = _OneShotProducerMapping(
        ((GraphIdSubclass("main"), producer), ("mtp.aux", producer))
    )

    with pytest.raises(TypeError, match="graph IDs must be exact strings"):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph=bindings,
        )

    assert producer.discover_graph_ids == []
    assert producer.producer_id_reads == 0


def test_runtime_source_dispatcher_rejects_duplicate_mapping_items_before_preflight() -> (
    None
):
    selection, request, _ = _aggregate_fixture()
    producer = _TestSourceMetadataProducer(_fingerprint())
    bindings = _OneShotProducerMapping(
        (("main", producer), ("main", producer), ("mtp.aux", producer))
    )

    with pytest.raises(ValueError, match="duplicate.*graph binding"):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph=bindings,
        )

    assert bindings.items_calls == 1
    assert producer.producer_id_reads == 0
    assert producer.schema_id_reads == 0
    assert producer.fingerprint_calls == 0
    assert producer.discover_method_reads == 0
    assert producer.discover_graph_ids == []


@pytest.mark.parametrize(
    ("field_name", "replacement", "expected_error"),
    (
        ("producer_id", object(), TypeError),
        ("producer_id", "test.wrong-producer", ValueError),
        ("schema_id", object(), TypeError),
        ("schema_id", SourceSchemaId("test.other.v1"), ValueError),
        ("fingerprint_result", object(), TypeError),
        (
            "fingerprint_result",
            replace(_fingerprint(), producer_revision="b" * 40),
            ValueError,
        ),
    ),
)
def test_runtime_source_dispatcher_rejects_wrong_producer_identity_preflight(
    field_name: str,
    replacement: object,
    expected_error: type[Exception],
) -> None:
    selection, request, _ = _aggregate_fixture()
    bad = _TestSourceMetadataProducer(_fingerprint(), **{field_name: replacement})
    good = _TestSourceMetadataProducer(_fingerprint())

    with pytest.raises(expected_error):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph={"main": good, "mtp.aux": bad},
        )

    assert good.discover_graph_ids == []
    assert bad.discover_graph_ids == []


def test_runtime_source_dispatcher_rejects_mutated_fingerprint_before_discovery() -> (
    None
):
    selection, request, _ = _aggregate_fixture()
    mutated = copy(_fingerprint())
    object.__setattr__(mutated, "normalization_contract_digest", object())
    producer = _TestSourceMetadataProducer(
        _fingerprint(),
        fingerprint_result=mutated,
    )

    with pytest.raises(TypeError, match="normalization contract digest"):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph={"main": producer, "mtp.aux": producer},
        )

    assert producer.discover_graph_ids == []


def test_runtime_source_dispatcher_revalidates_request_after_producer_preflight() -> (
    None
):
    selection, request, _ = _aggregate_fixture()

    class _RequestMutatingProducer(_TestSourceMetadataProducer):
        def fingerprint(self) -> SourceProducerFingerprint:
            fingerprint = super().fingerprint()
            object.__setattr__(
                request.graph_requests[0],
                "source_identity",
                _evidence("mutated-during-producer-preflight", "0"),
            )
            return fingerprint

    producer = _RequestMutatingProducer(_fingerprint())

    with pytest.raises(ValueError, match="runtime source request digest mismatch"):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph={"main": producer, "mtp.aux": producer},
        )

    assert producer.discover_graph_ids == []


def test_runtime_source_dispatcher_revalidates_selection_after_producer_preflight() -> (
    None
):
    selection, request, _ = _aggregate_fixture()

    class _SelectionMutatingProducer(_TestSourceMetadataProducer):
        def fingerprint(self) -> SourceProducerFingerprint:
            fingerprint = super().fingerprint()
            object.__setattr__(selection, "policy_digest", _digest("0"))
            return fingerprint

    producer = _SelectionMutatingProducer(_fingerprint())

    with pytest.raises(ValueError, match="policy_digest differs"):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph={"main": producer, "mtp.aux": producer},
        )

    assert producer.discover_graph_ids == []


@pytest.mark.parametrize(
    "collection_name",
    ("graph_requests", "trusted_expected_contributors"),
)
def test_runtime_source_dispatcher_rejects_equal_replaced_request_collections(
    collection_name: str,
) -> None:
    selection, request, _ = _aggregate_fixture()
    original = getattr(request, collection_name)
    replacement = tuple([*original])
    assert replacement == original
    assert replacement is not original

    class _RequestCollectionReplacingProducer(_TestSourceMetadataProducer):
        def fingerprint(self) -> SourceProducerFingerprint:
            fingerprint = super().fingerprint()
            object.__setattr__(request, collection_name, replacement)
            return fingerprint

    producer = _RequestCollectionReplacingProducer(_fingerprint())

    with pytest.raises(ValueError, match="collection changed during preflight"):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph={"main": producer, "mtp.aux": producer},
        )

    assert producer.discover_graph_ids == []


@pytest.mark.parametrize("shared_object", (True, False))
def test_runtime_source_dispatcher_preflights_each_unique_producer_once(
    shared_object: bool,
) -> None:
    selection, request, _ = _aggregate_fixture()
    main_producer = _TestSourceMetadataProducer(_fingerprint())
    mtp_producer = (
        main_producer
        if shared_object
        else _TestSourceMetadataProducer(pickle.loads(pickle.dumps(_fingerprint())))
    )

    produce_runtime_source_discovery_results(
        selection=selection,
        request=request,
        producers_by_graph={"main": main_producer, "mtp.aux": mtp_producer},
    )

    unique_producers = {
        id(main_producer): main_producer,
        id(mtp_producer): mtp_producer,
    }
    assert all(
        producer.producer_id_reads == 1 for producer in unique_producers.values()
    )
    assert all(producer.schema_id_reads == 1 for producer in unique_producers.values())
    assert all(
        producer.fingerprint_calls == 1 for producer in unique_producers.values()
    )
    assert all(
        producer.discover_method_reads == 1 for producer in unique_producers.values()
    )
    assert main_producer.discover_graph_ids == (
        ["main", "mtp.aux"] if shared_object else ["main"]
    )
    if not shared_object:
        assert mtp_producer.discover_graph_ids == ["mtp.aux"]


def test_runtime_source_dispatcher_validates_all_producers_before_any_discovery() -> (
    None
):
    selection, request, _ = _aggregate_fixture()
    good = _TestSourceMetadataProducer(_fingerprint())
    bad = _TestSourceMetadataProducer(
        _fingerprint(),
        fingerprint_result=replace(_fingerprint(), producer_revision="b" * 40),
    )

    with pytest.raises(ValueError, match="fingerprint"):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph={"main": good, "mtp.aux": bad},
        )

    assert good.discover_graph_ids == []
    assert bad.discover_graph_ids == []


def test_runtime_source_dispatcher_rejects_later_noncallable_discovery_preflight() -> (
    None
):
    selection, request, _ = _aggregate_fixture()
    main = _TestSourceMetadataProducer(_fingerprint())
    mtp = _TestSourceMetadataProducer(_fingerprint())
    object.__setattr__(mtp, "discover_contributions", object())

    with pytest.raises(TypeError, match="discover_contributions must be callable"):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph={"main": main, "mtp.aux": mtp},
        )

    assert main.discover_graph_ids == []
    assert mtp.discover_graph_ids == []
    assert main.discover_method_reads == 1
    assert mtp.discover_method_reads == 1


def test_runtime_source_dispatcher_validates_request_before_producer_preflight() -> (
    None
):
    selection, request, _ = _aggregate_fixture()
    forged = copy(request)
    object.__setattr__(forged, "request_digest", _digest("0"))
    producer = _TestSourceMetadataProducer(_fingerprint())

    with pytest.raises(ValueError, match="request_digest"):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=forged,
            producers_by_graph={"main": producer, "mtp.aux": producer},
        )

    assert producer.producer_id_reads == 0
    assert producer.schema_id_reads == 0
    assert producer.fingerprint_calls == 0
    assert producer.discover_graph_ids == []


@pytest.mark.parametrize("failure_kind", ("discover", "assembly"))
@pytest.mark.parametrize("failure_graph", ("main", "mtp.aux"))
def test_runtime_source_dispatcher_stops_after_first_graph_failure_without_results(
    failure_kind: str,
    failure_graph: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, _ = _aggregate_fixture()
    events: list[str] = []
    main = _TestSourceMetadataProducer(
        _fingerprint(),
        fail_graph=(
            failure_graph
            if failure_graph == "main" and failure_kind == "discover"
            else None
        ),
        empty_graph=(
            failure_graph
            if failure_graph == "main" and failure_kind == "assembly"
            else None
        ),
        events=events,
    )
    mtp = _TestSourceMetadataProducer(
        _fingerprint(),
        fail_graph=(
            failure_graph
            if failure_graph == "mtp.aux" and failure_kind == "discover"
            else None
        ),
        empty_graph=(
            failure_graph
            if failure_graph == "mtp.aux" and failure_kind == "assembly"
            else None
        ),
        events=events,
    )
    publication_calls = [0]

    def should_not_publish(
        **_kwargs: object,
    ) -> tuple[RuntimeSourceDiscoveryResult, ...]:
        publication_calls[0] += 1
        raise AssertionError("failed discovery must not publish partial results")

    monkeypatch.setattr(
        runtime_binding_module,
        "build_runtime_source_discovery_results",
        should_not_publish,
    )

    with pytest.raises((RuntimeError, ValueError)):
        produce_runtime_source_discovery_results(
            selection=selection,
            request=request,
            producers_by_graph={"mtp.aux": mtp, "main": main},
        )

    expected_events = ["discover:main"]
    if failure_graph == "mtp.aux":
        expected_events.append("discover:mtp.aux")
    assert events == expected_events
    assert mtp.discover_graph_ids == ([] if failure_graph == "main" else ["mtp.aux"])
    assert publication_calls == [0]


def test_runtime_source_dispatcher_revalidates_selection_and_uses_one_bulk_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, request, _ = _aggregate_fixture()
    producer = _TestSourceMetadataProducer(_fingerprint())
    request_validation_calls = [0]
    selection_validation_calls = [0]
    bulk_result_calls = [0]
    config_digest_calls = [0]
    real_request_validator = runtime_binding_module._validate_request_against_selection
    real_selection_validator = (
        runtime_binding_module.validate_compiled_precision_selection_group
    )
    real_bulk_builder = runtime_binding_module.build_runtime_source_discovery_results
    real_config_digest = runtime_binding_module.canonical_model_config_digest

    def counting_request_validator(
        candidate_selection: CompiledPrecisionSelectionGroup,
        candidate_request: RuntimeSourceDiscoveryRequest,
        **kwargs: object,
    ) -> RuntimeSourceDiscoveryRequest:
        request_validation_calls[0] += 1
        return real_request_validator(candidate_selection, candidate_request, **kwargs)

    def counting_selection_validator(
        candidate: CompiledPrecisionSelectionGroup,
    ) -> CompiledPrecisionSelectionGroup:
        selection_validation_calls[0] += 1
        return real_selection_validator(candidate)

    def counting_bulk_builder(
        *,
        request: RuntimeSourceDiscoveryRequest,
        partitions: Sequence[GraphDiscoveryPartition],
    ) -> tuple[RuntimeSourceDiscoveryResult, ...]:
        bulk_result_calls[0] += 1
        return real_bulk_builder(request=request, partitions=partitions)

    def counting_config_digest(model_config: Mapping[str, object]) -> str:
        config_digest_calls[0] += 1
        return real_config_digest(model_config)

    def forbidden_path(**_kwargs: object) -> object:
        raise AssertionError("dispatcher must not rebuild graph requests")

    monkeypatch.setattr(
        runtime_binding_module,
        "_validate_request_against_selection",
        counting_request_validator,
    )
    monkeypatch.setattr(
        runtime_binding_module,
        "validate_compiled_precision_selection_group",
        counting_selection_validator,
    )
    monkeypatch.setattr(
        runtime_binding_module,
        "build_runtime_source_discovery_results",
        counting_bulk_builder,
    )
    monkeypatch.setattr(
        runtime_binding_module,
        "canonical_model_config_digest",
        counting_config_digest,
    )
    monkeypatch.setattr(
        runtime_binding_module,
        "build_runtime_source_discovery_result",
        forbidden_path,
    )
    monkeypatch.setattr(
        runtime_binding_module,
        "build_runtime_graph_source_request",
        forbidden_path,
    )
    monkeypatch.setattr(
        runtime_binding_module,
        "build_runtime_source_discovery_request_from_contexts",
        forbidden_path,
    )

    results = produce_runtime_source_discovery_results(
        selection=selection,
        request=request,
        producers_by_graph={"mtp.aux": producer, "main": producer},
    )

    assert tuple(result.graph_instance_id for result in results) == (
        "main",
        "mtp.aux",
    )
    assert request_validation_calls == [2]
    assert selection_validation_calls == [2]
    assert bulk_result_calls == [1]
    assert config_digest_calls == [2]
