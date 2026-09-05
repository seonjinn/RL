from __future__ import annotations

import os
import pickle
import subprocess
import sys
from collections.abc import ItemsView, Iterator, Mapping, Sequence
from copy import copy
from dataclasses import FrozenInstanceError, fields, replace
from typing import TypeVar

import pytest

import nemo_rl.precision_policy.runtime_binding as runtime_binding_module
from nemo_rl.precision_policy.compiler import (
    CompiledPrecisionSelectionGroup,
    compile_precision_selection,
)
from nemo_rl.precision_policy.config import PrecisionPolicyConfig
from nemo_rl.precision_policy.runtime_binding import (
    RuntimeGraphSourceContext,
    RuntimeSourceDiscoveryRequest,
    RuntimeSourceDiscoveryResult,
    build_runtime_graph_source_request,
    build_runtime_source_discovery_request,
    build_runtime_source_discovery_request_from_contexts,
    build_runtime_source_discovery_result,
    build_runtime_source_discovery_results,
    validate_runtime_source_discovery_request,
    validate_runtime_source_discovery_results,
)
from nemo_rl.precision_policy.semantic import (
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
    ResolvedGraphTopology,
    ResolvedSelectionTopology,
    RolloutParticipation,
    SelectionTopologyEntry,
    SemanticAddressPattern,
    SourceMutability,
    _compute_semantic_structure_digest,
    _merge_selection_role_definitions,
    canonical_model_config_digest,
)
from nemo_rl.precision_policy.source_discovery import (
    HF_SAFETENSORS_HEADER_V1,
    DiscoveryContribution,
    ExpectedContributorSet,
    GraphDiscoveryPartition,
    RuntimeGraphSourceRequest,
    SourceDiscoveryInventory,
    SourceDiscoveryRecord,
    SourceProducerFingerprint,
    SourceRecordProvenance,
    assemble_runtime_graph_discovery_partition,
    derive_expected_contributor_authority,
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


def test_result_validator_rejects_equal_but_replaced_derived_fingerprint() -> None:
    selection, request, results = _aggregate_fixture()
    forged = copy(results[0])
    replacement = replace(forged.producer_fingerprint)
    assert replacement == forged.producer_fingerprint
    assert replacement is not forged.producer_fingerprint
    object.__setattr__(forged, "producer_fingerprint", replacement)

    with pytest.raises(ValueError, match="producer_fingerprint"):
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

blocked = ("torch", "megatron", "nemo_automodel", "transformer_engine", "vllm")

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"forbidden framework import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import nemo_rl.precision_policy.runtime_binding
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
