from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, asdict, dataclass, fields, replace
from math import inf, nan
from pickle import dumps, loads
import subprocess
import sys
from typing import Never

import pytest

import nemo_rl.precision_policy as precision_policy
import nemo_rl.precision_policy.topology as topology_module
from nemo_rl.precision_policy.source_dtype import (
    CanonicalSourceDType,
    normalize_safetensors_dtype,
    normalize_torch_dtype,
)
from nemo_rl.precision_policy.source_discovery import (
    HF_SAFETENSORS_HEADER_V1,
    DiscoveryContribution,
    ExpectedContributorSet,
    SourceProducerFingerprint,
    assemble_graph_discovery_partition,
)
from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    LOGICAL_VALUES,
    AttributePredicate,
    AxisDomain,
    AxisExtentRounding,
    AxisProjection,
    ComponentDescriptor,
    ComponentRole,
    EvidenceSource,
    EvidenceSourceKind,
    ExpectedGraphDeclaration,
    FamilyIndexDomain,
    FormatDescriptor,
    GraphKind,
    GraphLifecycle,
    GraphProvenance,
    IdenticalStorageSourceAliasContract,
    ImmutableAuxiliaryEvidence,
    IndexPathSegment,
    LayerDomain,
    LayerMember,
    LiteralComponentAxisSpec,
    LiteralPathSegment,
    LogicalComponentAxisSpec,
    OwnerFamilyBinding,
    OwnerFamilyReference,
    ParameterInventoryEntry,
    RoleExpectedDomain,
    RolloutParticipation,
    SemanticAddressPattern,
    SemanticGraphManifest,
    SemanticManifestBundle,
    SemanticOwnership,
    SemanticPredicate,
    SemanticTensorFamily,
    SourceMutability,
    SourceOwnerInventoryEntry,
    SourceReplicaSynchronizationEvidence,
    SourceSynchronizationBoundary,
    SynchronizedReplicaSourceAliasContract,
    ValueProvenance,
    builtin_role_definitions,
)
from nemo_rl.precision_policy.topology import (
    AbsentDiscoveryDispositionEdge,
    CanonicalValueClassificationEdge,
    ComponentAxisTarget,
    FamilyIndexAxisTarget,
    GraphTopologyInput,
    LayerCoordinateTarget,
    OutputMemberTarget,
    RoleDefinitionContribution,
    SemanticGraphBuildFragment,
    SourceAxisSelection,
    SourceDiscoveryInventory,
    SourceDiscoveryRecord,
    SourceIndexSpan,
    SourceOrdinalMapSegment,
    SourceRecordProvenance,
    SourceRegion,
    SourceToSemanticAxisMapping,
    SynchronizedReplicaAliasClassificationEdge,
    TiedAliasClassificationEdge,
    build_semantic_manifest_bundle,
    resolve_text_config,
    select_model_topology_adapter,
    validate_semantic_graph_build_fragment,
)


def _discovery_evidence(name: str, character: str) -> EvidenceSource:
    return EvidenceSource(
        kind=EvidenceSourceKind.RUNTIME_INVENTORY,
        locator=f"runtime://{name}",
        digest=f"sha256:{character * 64}",
    )


def _test_expected_contributors(graph_instance_id: str) -> ExpectedContributorSet:
    return ExpectedContributorSet(
        contributor_ids=(f"{graph_instance_id}:complete",),
        authority=_discovery_evidence(f"{graph_instance_id}-membership", "1"),
    )


def _test_source_fingerprint() -> SourceProducerFingerprint:
    return SourceProducerFingerprint(
        schema_id=HF_SAFETENSORS_HEADER_V1,
        producer_implementation_id="test-source-producer",
        producer_revision="a" * 40,
        normalization_contract_digest=f"sha256:{'2' * 64}",
        evidence=_discovery_evidence("test-source-producer", "3"),
    )


def _graph_discovery_fields(graph_instance_id: str) -> dict[str, object]:
    expected = _test_expected_contributors(graph_instance_id)
    return {
        "source_producer_fingerprint": _test_source_fingerprint(),
        "expected_contributor_authority": expected.to_authority(),
        "source_identity": _discovery_evidence(
            f"{graph_instance_id}-source",
            "4",
        ),
        "artifact_identity": _discovery_evidence(
            f"{graph_instance_id}-artifact",
            "5",
        ),
    }


def test_resolve_text_config_preserves_nested_decoder_config() -> None:
    nested = {
        "model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 40,
        "layer_types": ["linear_attention", "full_attention"],
    }

    resolved = resolve_text_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": nested,
        }
    )

    assert isinstance(resolved, Mapping)
    assert resolved == nested


def test_graph_topology_input_snapshots_nested_caller_config() -> None:
    from nemo_rl.precision_policy.semantic import (
        ExpectedGraphDeclaration,
        GraphKind,
        GraphLifecycle,
        GraphProvenance,
        RolloutParticipation,
    )

    layer_types = ["linear_attention", "full_attention"]
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "layer_types": layer_types,
        },
    }
    graph_input = GraphTopologyInput(
        declaration=ExpectedGraphDeclaration(
            graph_instance_id="main",
            model_identity="Qwen/Qwen3.5-35B-A3B",
            lifecycle=GraphLifecycle(
                graph_kind=GraphKind.MAIN,
                graph_provenance=GraphProvenance.TRAINING_RUNTIME,
                rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
            ),
        ),
        model_config=config,
        resolved_model_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        **_graph_discovery_fields("main"),
    )

    layer_types.append("mutated")

    assert resolve_text_config(graph_input.model_config)["layer_types"] == (
        "linear_attention",
        "full_attention",
    )


def test_graph_topology_input_snapshot_is_process_serializable() -> None:
    from nemo_rl.precision_policy.semantic import (
        ExpectedGraphDeclaration,
        GraphKind,
        GraphLifecycle,
        GraphProvenance,
        RolloutParticipation,
    )

    graph_input = GraphTopologyInput(
        declaration=ExpectedGraphDeclaration(
            graph_instance_id="main",
            model_identity="Qwen/Qwen3.5-35B-A3B",
            lifecycle=GraphLifecycle(
                graph_kind=GraphKind.MAIN,
                graph_provenance=GraphProvenance.TRAINING_RUNTIME,
                rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
            ),
        ),
        model_config={
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "layer_types": ["linear_attention", "full_attention"],
            },
        },
        resolved_model_revision="revision:content-addressed-v1",
        **_graph_discovery_fields("main"),
    )

    restored = loads(dumps(graph_input))

    assert restored == graph_input
    assert resolve_text_config(restored.model_config)["layer_types"] == (
        "linear_attention",
        "full_attention",
    )


def test_graph_topology_input_snapshot_is_canonical_and_preserves_scalars() -> None:
    from nemo_rl.precision_policy.semantic import (
        ExpectedGraphDeclaration,
        GraphKind,
        GraphLifecycle,
        GraphProvenance,
        RolloutParticipation,
    )

    declaration = ExpectedGraphDeclaration(
        graph_instance_id="main",
        model_identity="test/model",
        lifecycle=GraphLifecycle(
            graph_kind=GraphKind.MAIN,
            graph_provenance=GraphProvenance.TRAINING_RUNTIME,
            rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
        ),
    )
    graph_input = GraphTopologyInput(
        declaration=declaration,
        model_config={
            "z": None,
            "a": [True, 7, 2.5, "value"],
        },
        resolved_model_revision="content-addressed:test",
        **_graph_discovery_fields("main"),
    )

    assert tuple(graph_input.model_config) == ("a", "z")
    assert tuple(graph_input.model_config.keys()) == ("a", "z")
    assert tuple(graph_input.model_config.values()) == (
        (True, 7, 2.5, "value"),
        None,
    )
    assert tuple(graph_input.model_config.items()) == (
        ("a", (True, 7, 2.5, "value")),
        ("z", None),
    )
    assert graph_input.model_config.get("missing", "fallback") == "fallback"
    assert graph_input.model_config["a"] == (True, 7, 2.5, "value")
    assert graph_input.model_config == {
        "a": (True, 7, 2.5, "value"),
        "z": None,
    }
    assert replace(graph_input) == graph_input
    assert graph_input == loads(dumps(graph_input))


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_graph_topology_input_rejects_non_finite_config_floats(value: float) -> None:
    from nemo_rl.precision_policy.semantic import (
        ExpectedGraphDeclaration,
        GraphKind,
        GraphLifecycle,
        GraphProvenance,
        RolloutParticipation,
    )

    with pytest.raises(ValueError, match="finite"):
        GraphTopologyInput(
            declaration=ExpectedGraphDeclaration(
                graph_instance_id="main",
                model_identity="test/model",
                lifecycle=GraphLifecycle(
                    graph_kind=GraphKind.MAIN,
                    graph_provenance=GraphProvenance.TRAINING_RUNTIME,
                    rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
                ),
            ),
            model_config={"bad": value},
            resolved_model_revision="content-addressed:test",
            **_graph_discovery_fields("main"),
        )


def _evidence(name: str):
    from nemo_rl.precision_policy.semantic import EvidenceSource, EvidenceSourceKind

    return EvidenceSource(
        kind=EvidenceSourceKind.RUNTIME_INVENTORY,
        locator=f"runtime://{name}",
        digest=f"sha256:{name}",
    )


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("BF16", CanonicalSourceDType.BFLOAT16),
        ("F16", CanonicalSourceDType.FLOAT16),
        ("F32", CanonicalSourceDType.FLOAT32),
        ("F8_E4M3", CanonicalSourceDType.E4M3),
        ("F8_E8M0", CanonicalSourceDType.E8M0),
        ("U8", CanonicalSourceDType.UINT8),
        ("I32", CanonicalSourceDType.INT32),
        ("I64", CanonicalSourceDType.INT64),
    ],
)
def test_safetensors_dtype_normalizer_maps_only_exact_metadata_tokens(
    token: str,
    expected: CanonicalSourceDType,
) -> None:
    assert normalize_safetensors_dtype(token) is expected


@pytest.mark.parametrize(
    "token",
    ["bf16", " BF16", "BF16 ", "half", "E4M3", "", "F8_E4M3FN"],
)
def test_safetensors_dtype_normalizer_rejects_noncanonical_metadata_tokens(
    token: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported safetensors dtype"):
        normalize_safetensors_dtype(token)


@pytest.mark.parametrize("value", [b"BF16", None, True])
def test_safetensors_dtype_normalizer_rejects_non_string_metadata_tokens(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="safetensors dtype must be a string"):
        normalize_safetensors_dtype(value)


def test_torch_dtype_normalizer_maps_only_supported_dtype_singletons() -> None:
    torch = pytest.importorskip("torch")
    cases = [
        (torch.bfloat16, CanonicalSourceDType.BFLOAT16),
        (torch.float16, CanonicalSourceDType.FLOAT16),
        (torch.float32, CanonicalSourceDType.FLOAT32),
        (torch.uint8, CanonicalSourceDType.UINT8),
        (torch.int32, CanonicalSourceDType.INT32),
        (torch.int64, CanonicalSourceDType.INT64),
    ]
    if hasattr(torch, "float8_e4m3fn"):
        cases.append((torch.float8_e4m3fn, CanonicalSourceDType.E4M3))
    if hasattr(torch, "float8_e8m0fnu"):
        cases.append((torch.float8_e8m0fnu, CanonicalSourceDType.E8M0))

    for dtype, expected in cases:
        assert normalize_torch_dtype(dtype) is expected


class _EqualityLyingDTypeWrapper:
    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 0


@pytest.mark.parametrize(
    "value",
    ["torch.float16", _EqualityLyingDTypeWrapper(), object()],
)
def test_torch_dtype_normalizer_rejects_non_dtype_objects(value: object) -> None:
    with pytest.raises(TypeError, match="torch dtype"):
        normalize_torch_dtype(value)


def test_torch_dtype_normalizer_rejects_supported_torch_but_unsupported_dtype() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="unsupported torch dtype"):
        normalize_torch_dtype(torch.float64)


def test_source_dtype_boundary_exports_are_deterministic() -> None:
    assert tuple(member.value for member in CanonicalSourceDType) == (
        "bfloat16",
        "float16",
        "float32",
        "e4m3",
        "e8m0",
        "uint8",
        "int32",
        "int64",
    )
    assert precision_policy.CanonicalSourceDType is CanonicalSourceDType
    assert precision_policy.normalize_safetensors_dtype is normalize_safetensors_dtype
    assert precision_policy.normalize_torch_dtype is normalize_torch_dtype


def test_precision_policy_imports_without_torch() -> None:
    code = """
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch imports are blocked')
        return None

sys.meta_path.insert(0, BlockTorch())
import nemo_rl.precision_policy
import nemo_rl.precision_policy.source_dtype
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _source_record(
    *,
    record_id: str = "main.experts.gate",
    graph_instance_id: str = "main",
    native_name: str | None = "model.layers.mlp.experts.gate_proj.weight",
    native_owner: str | None = "model.layers.mlp.experts.gate_proj",
    dtype: CanonicalSourceDType = CanonicalSourceDType.BFLOAT16,
    shape: tuple[int, ...] = (2, 4, 8, 8),
    provenance: SourceRecordProvenance = SourceRecordProvenance.TRAINING_RUNTIME,
    source_mutability=None,
) -> SourceDiscoveryRecord:
    from nemo_rl.precision_policy.semantic import SourceMutability

    mutability = source_mutability or SourceMutability.MUTABLE
    return SourceDiscoveryRecord(
        record_id=record_id,
        graph_instance_id=graph_instance_id,
        source_native_name=native_name,
        source_native_owner_id=native_owner,
        dtype=dtype,
        shape=shape,
        provenance=provenance,
        provenance_evidence=_evidence(f"{record_id}-provenance"),
        source_mutability=mutability,
        mutability_evidence=_evidence(f"{record_id}-mutability"),
    )


def _partitioned_discovery(
    graph_inputs: tuple[GraphTopologyInput, ...],
    records: tuple[SourceDiscoveryRecord, ...],
) -> tuple[SourceDiscoveryInventory, dict[str, ExpectedContributorSet]]:
    expected_by_graph: dict[str, ExpectedContributorSet] = {}
    partitions = []
    for graph_input in graph_inputs:
        graph_id = graph_input.declaration.graph_instance_id
        expected = _test_expected_contributors(graph_id)
        expected_by_graph[graph_id] = expected
        contribution = DiscoveryContribution(
            contributor_id=expected.contributor_ids[0],
            graph_instance_id=graph_id,
            producer_fingerprint=graph_input.source_producer_fingerprint,
            records=tuple(
                record for record in records if record.graph_instance_id == graph_id
            ),
        )
        partitions.append(
            assemble_graph_discovery_partition(
                graph_input=graph_input,
                expected_contributors=expected,
                contributions=(contribution,),
            )
        )
    return SourceDiscoveryInventory(tuple(partitions)), expected_by_graph


def _build_semantic_bundle(
    schema_version: int,
    graph_inputs: tuple[GraphTopologyInput, ...],
    records: tuple[SourceDiscoveryRecord, ...],
) -> SemanticManifestBundle:
    source_discovery, expected_by_graph = _partitioned_discovery(
        graph_inputs,
        records,
    )
    return build_semantic_manifest_bundle(
        schema_version,
        graph_inputs,
        source_discovery,
        expected_by_graph,
    )


def test_source_discovery_record_is_raw_frozen_metadata() -> None:
    record = _source_record()

    assert record.shape == (2, 4, 8, 8)
    assert tuple(field.name for field in fields(SourceDiscoveryRecord)) == (
        "record_id",
        "graph_instance_id",
        "source_native_name",
        "source_native_owner_id",
        "dtype",
        "shape",
        "provenance",
        "provenance_evidence",
        "source_mutability",
        "mutability_evidence",
    )
    assert not {
        "source_axes",
        "semantic_address",
        "role",
        "module_kind",
        "policy",
        "endpoint",
    } & {field.name for field in fields(SourceDiscoveryRecord)}
    with pytest.raises(FrozenInstanceError):
        record.dtype = CanonicalSourceDType.FLOAT32  # type: ignore[misc]


def test_source_discovery_record_requires_canonical_source_dtype() -> None:
    with pytest.raises(TypeError, match="CanonicalSourceDType"):
        _source_record(dtype="bfloat16")  # type: ignore[arg-type]

    record = _source_record(dtype=CanonicalSourceDType.BFLOAT16)

    assert record.dtype is CanonicalSourceDType.BFLOAT16


@pytest.mark.parametrize(
    ("native_name", "native_owner"),
    [(None, "owner"), ("name", None), (None, None)],
)
def test_present_source_record_requires_both_native_fields(
    native_name: str | None,
    native_owner: str | None,
) -> None:
    with pytest.raises(ValueError, match="present source record"):
        _source_record(native_name=native_name, native_owner=native_owner)


@pytest.mark.parametrize(
    ("native_name", "native_owner"),
    [("name", "owner"), ("name", None), (None, "owner")],
)
def test_absent_source_record_forbids_both_native_fields(
    native_name: str | None,
    native_owner: str | None,
) -> None:
    from nemo_rl.precision_policy.semantic import SourceMutability

    with pytest.raises(ValueError, match="absent source record"):
        _source_record(
            native_name=native_name,
            native_owner=native_owner,
            source_mutability=SourceMutability.ABSENT,
        )


def test_absent_source_record_retains_only_expected_raw_shape_and_axes() -> None:
    from nemo_rl.precision_policy.semantic import SourceMutability

    record = _source_record(
        native_name=None,
        native_owner=None,
        source_mutability=SourceMutability.ABSENT,
    )

    assert record.source_native_name is None
    assert record.source_native_owner_id is None
    assert record.shape == (2, 4, 8, 8)


def test_absent_source_record_rejects_tied_storage_provenance() -> None:
    with pytest.raises(ValueError, match="absent.*tied"):
        _source_record(
            native_name=None,
            native_owner=None,
            provenance=SourceRecordProvenance.TIED_STORAGE,
            source_mutability=SourceMutability.ABSENT,
        )


def test_absent_source_record_rejects_synchronized_replica_provenance() -> None:
    with pytest.raises(ValueError, match="absent.*synchronized-replica"):
        _source_record(
            native_name=None,
            native_owner=None,
            provenance=SourceRecordProvenance.SYNCHRONIZED_REPLICA,
            source_mutability=SourceMutability.ABSENT,
        )


@pytest.mark.parametrize("shape", [(2, 0), (2, -1)])
def test_source_discovery_record_rejects_invalid_raw_shape(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="positive"):
        _source_record(shape=shape)


def test_source_discovery_inventory_is_canonical_and_rejects_duplicates() -> None:
    later = _source_record(record_id="main.z", native_name="model.z.weight")
    earlier = _source_record(record_id="main.a", native_name="model.a.weight")

    inventory, _ = _partitioned_discovery(
        (_main_graph_input(),),
        (later, earlier),
    )

    assert tuple(record.record_id for record in inventory.records) == (
        "main.a",
        "main.z",
    )
    with pytest.raises(ValueError, match="duplicate source discovery record"):
        _partitioned_discovery(
            (_main_graph_input(),),
            (earlier, earlier),
        )


def test_source_discovery_inventory_rejects_duplicate_present_native_name() -> None:
    first = _source_record(record_id="main.values")
    duplicate = replace(first, record_id="main.duplicate")

    with pytest.raises(ValueError, match="duplicate present source native name"):
        _partitioned_discovery(
            (_main_graph_input(),),
            (first, duplicate),
        )


def test_source_discovery_inventory_allows_component_records_to_share_owner() -> None:
    values = _source_record(
        record_id="main.values",
        native_name="model.layers.mlp.experts.gate_proj.weight",
        native_owner="model.layers.mlp.experts.gate_proj",
    )
    scales = _source_record(
        record_id="main.scales",
        native_name="model.layers.mlp.experts.gate_proj.weight_scale",
        native_owner="model.layers.mlp.experts.gate_proj",
    )

    inventory, _ = _partitioned_discovery(
        (_main_graph_input(),),
        (scales, values),
    )

    assert tuple(record.record_id for record in inventory.records) == (
        "main.scales",
        "main.values",
    )


def test_partitioned_inventory_rejects_duplicate_record_ids_across_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_input = _main_graph_input()
    draft_input = replace(
        main_input,
        declaration=ExpectedGraphDeclaration(
            graph_instance_id="draft.external",
            model_identity="test/draft",
            lifecycle=GraphLifecycle(
                graph_kind=GraphKind.SPECULATIVE_DRAFTER,
                graph_provenance=GraphProvenance.TRAINING_RUNTIME,
                rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
            ),
        ),
        **_graph_discovery_fields("draft.external"),
    )
    records = (
        _source_record(record_id="duplicate.record"),
        _source_record(
            record_id="duplicate.record",
            graph_instance_id="draft.external",
            native_name="draft.weight",
            native_owner="draft.weight",
        ),
    )
    source_discovery, expected = _partitioned_discovery(
        (main_input, draft_input),
        records,
    )

    monkeypatch.setattr(
        topology_module,
        "_default_adapters",
        lambda: (_ for _ in ()).throw(
            AssertionError("adapter selection ran before duplicate-ID preflight")
        ),
    )
    with pytest.raises(ValueError, match="duplicate source discovery record ID"):
        build_semantic_manifest_bundle(
            1,
            (main_input, draft_input),
            source_discovery,
            expected,
        )


def test_source_regions_keep_strided_compact_spans_without_enumeration() -> None:
    region = SourceRegion(
        source_shape=(8, 12),
        axis_selections=(
            SourceAxisSelection(
                axis_index=1,
                spans=(SourceIndexSpan(1, 12, 2),),
            ),
            SourceAxisSelection(
                axis_index=0,
                spans=(SourceIndexSpan(0, 4), SourceIndexSpan(4, 8)),
            ),
        ),
    )

    assert region.axis_selections[0].axis_index == 0
    assert region.cardinality == 8 * 6
    assert region.axis_selections[1].spans == (SourceIndexSpan(1, 12, 2),)


def test_scalar_source_record_and_region_have_cardinality_one() -> None:
    record = _source_record(shape=())
    region = SourceRegion(source_shape=(), axis_selections=())

    assert record.shape == ()
    assert region.cardinality == 1


def test_region_partition_is_linear_for_ten_thousand_singletons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _source_record(shape=(10_000,))
    regions = tuple(
        SourceRegion(
            source_shape=record.shape,
            axis_selections=(
                SourceAxisSelection(0, (SourceIndexSpan(index, index + 1),)),
            ),
        )
        for index in range(10_000)
    )
    intersection_calls = 0
    original = topology_module._regions_intersect

    def bounded_intersection(left: SourceRegion, right: SourceRegion) -> bool:
        nonlocal intersection_calls
        intersection_calls += 1
        if intersection_calls > 20_000:
            raise AssertionError("performed pairwise source-region comparisons")
        return original(left, right)

    monkeypatch.setattr(
        topology_module,
        "_regions_intersect",
        bounded_intersection,
    )

    topology_module._validate_region_partition(record, regions, tied=False)

    assert intersection_calls <= 20_000


@pytest.mark.parametrize(
    "spans",
    [
        (SourceIndexSpan(0, 5, 2), SourceIndexSpan(2, 6, 2)),
        (),
    ],
)
def test_source_axis_selection_rejects_overlap_or_empty(
    spans: tuple[SourceIndexSpan, ...],
) -> None:
    with pytest.raises(ValueError):
        SourceAxisSelection(0, spans)


@pytest.mark.parametrize(
    "region",
    [
        ((8, 8), (SourceAxisSelection(0, (SourceIndexSpan(0, 8),)),)),
        (
            (8,),
            (
                SourceAxisSelection(0, (SourceIndexSpan(0, 8),)),
                SourceAxisSelection(1, (SourceIndexSpan(0, 1),)),
            ),
        ),
        ((8,), (SourceAxisSelection(0, (SourceIndexSpan(0, 9),)),)),
    ],
)
def test_source_region_requires_every_in_bounds_axis_once(
    region: tuple[tuple[int, ...], tuple[SourceAxisSelection, ...]],
) -> None:
    with pytest.raises(ValueError):
        SourceRegion(source_shape=region[0], axis_selections=region[1])


@dataclass(frozen=True)
class _SelectionAdapter:
    adapter_id: str
    supported_model_type: str

    def supports(self, model_config: Mapping[str, object]) -> bool:
        return model_config.get("model_type") == self.supported_model_type

    def classify_graph(
        self,
        schema_version: int,
        graph_input: GraphTopologyInput,
        source_records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment:
        raise AssertionError("selection tests must not classify")


def test_adapter_selection_requires_exactly_one_support_match() -> None:
    selected = select_model_topology_adapter(
        {"model_type": "chosen"},
        adapters=(
            _SelectionAdapter("z-adapter", "other"),
            _SelectionAdapter("a-adapter", "chosen"),
        ),
    )

    assert selected.adapter_id == "a-adapter"


def test_adapter_selection_rejects_unsupported_and_ambiguous_configs() -> None:
    with pytest.raises(ValueError, match="unsupported model topology"):
        select_model_topology_adapter(
            {"model_type": "unknown"},
            adapters=(_SelectionAdapter("qwen", "qwen"),),
        )

    with pytest.raises(ValueError, match="ambiguous.*a-adapter.*z-adapter"):
        select_model_topology_adapter(
            {"model_type": "chosen"},
            adapters=(
                _SelectionAdapter("z-adapter", "chosen"),
                _SelectionAdapter("a-adapter", "chosen"),
            ),
        )


def _main_graph_input() -> GraphTopologyInput:
    return GraphTopologyInput(
        declaration=ExpectedGraphDeclaration(
            graph_instance_id="main",
            model_identity="test/routed-model",
            lifecycle=GraphLifecycle(
                graph_kind=GraphKind.MAIN,
                graph_provenance=GraphProvenance.TRAINING_RUNTIME,
                rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
            ),
        ),
        model_config={"model_type": "test_routed", "num_hidden_layers": 2},
        resolved_model_revision="content-addressed:test-routed-v1",
        **_graph_discovery_fields("main"),
    )


def _layer_expert_domain(
    layers: tuple[LayerMember, ...] = (LayerMember(0, 0), LayerMember(1, 1)),
    experts: tuple[int, ...] = (0, 1, 2, 3),
) -> FamilyIndexDomain:
    return FamilyIndexDomain(
        layer_domain=LayerDomain(layers),
        independent_axes=(AxisDomain("expert", experts),),
    )


def _identity_axes(domain: FamilyIndexDomain) -> tuple[AxisProjection, ...]:
    return tuple(AxisProjection(axis, axis) for axis in domain.axis_names)


def _routed_entry(
    *,
    entry_id: str = "main.moe.routed.gate",
    projection: str = "gate",
    domain: FamilyIndexDomain | None = None,
    owner_family: OwnerFamilyReference | None = None,
) -> ParameterInventoryEntry:
    family_domain = domain or _layer_expert_domain()
    canonical_owner = owner_family or OwnerFamilyReference(
        "main", "source.moe.routed.gate"
    )
    return ParameterInventoryEntry(
        entry_id=entry_id,
        graph_instance_id="main",
        member=SemanticTensorFamily(
            pattern=SemanticAddressPattern(
                semantic_graph_path="text.decoder",
                path_segments=(
                    LiteralPathSegment("layer"),
                    IndexPathSegment("global_decoder_layer"),
                    LiteralPathSegment("moe"),
                    LiteralPathSegment("routed"),
                    IndexPathSegment("expert"),
                    LiteralPathSegment(projection),
                ),
                model_part="main",
                module_kind="moe.expert_ffn",
                attributes=(
                    ("expert_kind", "routed"),
                    ("projection", projection),
                ),
                parameter_role="kernel",
            ),
            domain=family_domain,
            format=BF16_FORMAT,
            logical_dtype="bfloat16",
            logical_shape=(8, 8),
            logical_axes=("output_features", "input_features"),
            ownership=SemanticOwnership(
                OwnerFamilyBinding(
                    canonical_owner_family=canonical_owner,
                    canonical_value_entry_id=entry_id,
                    member_domain=family_domain,
                    member_to_owner_axes=_identity_axes(family_domain),
                    member_to_value_axes=_identity_axes(family_domain),
                )
            ),
        ),
        value_provenance=ValueProvenance.TRAINING_PARAMETER,
    )


def _whole_region(shape: tuple[int, ...]) -> SourceRegion:
    return SourceRegion(
        source_shape=shape,
        axis_selections=tuple(
            SourceAxisSelection(index, (SourceIndexSpan(0, size),))
            for index, size in enumerate(shape)
        ),
    )


def test_synchronized_replica_edge_has_distinct_typed_relation_evidence() -> None:
    domain = _layer_expert_domain()
    synchronization = SourceReplicaSynchronizationEvidence(
        replica_group_id="replicas.mtp.0",
        boundary=SourceSynchronizationBoundary.SOURCE_VERSION_READY,
        evidence_source=_evidence("replicas-mtp-0"),
    )

    edge = SynchronizedReplicaAliasClassificationEdge(
        record_id="mtp.0.replica.gate",
        replica_source_region=_whole_region((2, 4, 8, 8)),
        alias_output=OutputMemberTarget("mtp.0.moe.routed.gate", domain, ()),
        canonical_record_id="main.experts.gate",
        canonical_source_region=_whole_region((2, 4, 8, 8)),
        canonical_owner_family=OwnerFamilyReference("main", "source.moe.routed.gate"),
        canonical_value_entry_id="main.moe.routed.gate",
        component_role=LOGICAL_VALUES,
        alias_to_canonical_axes=tuple(reversed(_identity_axes(domain))),
        synchronization=synchronization,
    )

    assert SourceRecordProvenance.SYNCHRONIZED_REPLICA.value == ("synchronized_replica")
    assert edge.alias_to_canonical_axes == tuple(
        sorted(
            _identity_axes(domain),
            key=lambda projection: (
                projection.member_axis,
                projection.owner_axis,
            ),
        )
    )
    assert edge.synchronization is synchronization
    with pytest.raises(FrozenInstanceError):
        edge.canonical_record_id = "replacement"  # type: ignore[misc]


def _whole_routed_edge(
    record: SourceDiscoveryRecord,
    entry: ParameterInventoryEntry,
    *,
    source_region: SourceRegion | None = None,
    output: OutputMemberTarget | None = None,
) -> CanonicalValueClassificationEdge:
    domain = entry.member.domain
    return CanonicalValueClassificationEdge(
        record_id=record.record_id,
        source_region=source_region or _whole_region(record.shape),
        output=output or OutputMemberTarget(entry.entry_id, domain, ()),
        canonical_owner_family=entry.member.ownership.binding.canonical_owner_family,
        component_role=LOGICAL_VALUES,
        axis_mappings=(
            SourceToSemanticAxisMapping(
                0,
                LayerCoordinateTarget("global_decoder_layer"),
                (SourceOrdinalMapSegment(SourceIndexSpan(0, 2), 0),),
            ),
            SourceToSemanticAxisMapping(
                0,
                LayerCoordinateTarget("moe_ordinal"),
                (SourceOrdinalMapSegment(SourceIndexSpan(0, 2), 0),),
            ),
            SourceToSemanticAxisMapping(
                1,
                FamilyIndexAxisTarget("expert"),
                (SourceOrdinalMapSegment(SourceIndexSpan(0, 4), 0),),
            ),
            SourceToSemanticAxisMapping(
                2,
                ComponentAxisTarget(LOGICAL_VALUES, "output_features"),
                (SourceOrdinalMapSegment(SourceIndexSpan(0, 8), 0),),
            ),
            SourceToSemanticAxisMapping(
                3,
                ComponentAxisTarget(LOGICAL_VALUES, "input_features"),
                (SourceOrdinalMapSegment(SourceIndexSpan(0, 8), 0),),
            ),
        ),
    )


def _routed_role_contribution(entry_id: str) -> RoleDefinitionContribution:
    definition = next(
        definition
        for definition in builtin_role_definitions(
            1,
            {"moe.routed_expert": RoleExpectedDomain("moe.routed_expert", (entry_id,))},
        )
        if definition.role_name == "moe.routed_expert"
    )
    return RoleDefinitionContribution(
        schema_version=1,
        role_name=definition.role_name,
        predicate=definition.predicate,
        expected_inventory_entry_ids=(entry_id,),
    )


def _valid_routed_fragment() -> tuple[
    GraphTopologyInput,
    tuple[SourceDiscoveryRecord, ...],
    SemanticGraphBuildFragment,
]:
    graph_input = _main_graph_input()
    record = _source_record()
    entry = _routed_entry()
    owner = SourceOwnerInventoryEntry(
        owner_family=entry.member.ownership.binding.canonical_owner_family,
        domain=entry.member.domain,
        source_mutability=SourceMutability.MUTABLE,
        mutability_evidence_source=record.mutability_evidence,
    )
    fragment = SemanticGraphBuildFragment(
        graph_instance_id="main",
        classification_edges=(_whole_routed_edge(record, entry),),
        source_owners=(owner,),
        inventory_entries=(entry,),
        manifest=SemanticGraphManifest(
            model_family="test-routed",
            model_revision=graph_input.resolved_model_revision,
            graph_instance_id="main",
            lifecycle=graph_input.declaration.lifecycle,
            inventory_entry_ids=(entry.entry_id,),
        ),
        role_contributions=(_routed_role_contribution(entry.entry_id),),
    )
    return graph_input, (record,), fragment


def test_fragment_accepts_one_complete_compact_canonical_edge() -> None:
    graph_input, records, fragment = _valid_routed_fragment()

    validate_semantic_graph_build_fragment(1, graph_input, records, fragment)


def test_fragment_validation_does_not_retain_full_records_per_native_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_input, base_records, base_fragment = _valid_routed_fragment()
    base_record = base_records[0]
    base_entry = base_fragment.inventory_entries[0]
    owner_reference = base_entry.member.ownership.binding.canonical_owner_family
    entries = tuple(
        _routed_entry(
            entry_id=f"main.moe.routed.projection-{index}",
            projection=f"projection-{index}",
            owner_family=owner_reference,
        )
        for index in range(256)
    )
    records = tuple(
        replace(
            base_record,
            record_id=f"main.experts.projection-{index}",
            source_native_name=f"model.experts.projection-{index}.weight",
        )
        for index in range(256)
    )
    fragment = replace(
        base_fragment,
        classification_edges=tuple(
            _whole_routed_edge(record, entry)
            for record, entry in zip(records, entries, strict=True)
        ),
        inventory_entries=entries,
        manifest=replace(
            base_fragment.manifest,
            inventory_entry_ids=tuple(entry.entry_id for entry in entries),
        ),
        role_contributions=(
            replace(
                base_fragment.role_contributions[0],
                expected_inventory_entry_ids=tuple(entry.entry_id for entry in entries),
            ),
        ),
    )

    def fail_full_record_hash(_record: SourceDiscoveryRecord) -> int:
        raise AssertionError("retained full source records in a native-owner set")

    monkeypatch.setattr(SourceDiscoveryRecord, "__hash__", fail_full_record_hash)

    validate_semantic_graph_build_fragment(1, graph_input, records, fragment)


def test_canonical_only_fragment_does_not_index_every_direct_native_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    entry = fragment.inventory_entries[0]
    record = records[0]
    forbidden_pair = (entry.entry_id, record.source_native_owner_id)

    class TrackingSet(set[object]):
        def add(self, value: object) -> None:
            if value == forbidden_pair:
                raise AssertionError("indexed an unreferenced direct native owner")
            super().add(value)

    monkeypatch.setattr(topology_module, "set", TrackingSet, raising=False)

    validate_semantic_graph_build_fragment(1, graph_input, records, fragment)


def _valid_scalar_component_fragment() -> tuple[
    GraphTopologyInput,
    tuple[SourceDiscoveryRecord, ...],
    SemanticGraphBuildFragment,
]:
    from nemo_rl.precision_policy.semantic import (
        ComponentDescriptor,
        ComponentRole,
        FormatDescriptor,
        SemanticAddress,
        SemanticTensor,
    )

    graph_input = _main_graph_input()
    domain = FamilyIndexDomain(None, ())
    component_role = ComponentRole("weight_scale_2")
    format_descriptor = FormatDescriptor(
        "test.scalar-metadata.v1",
        "test.scalar-metadata",
        (
            ComponentDescriptor(
                component_role,
                "float32",
                encoding="scalar_metadata",
                component_axes=(),
            ),
        ),
    )
    owner_reference = OwnerFamilyReference("main", "source.scalar-metadata")
    entry = ParameterInventoryEntry(
        entry_id="main.scalar-metadata",
        graph_instance_id="main",
        member=SemanticTensor(
            address=SemanticAddress(
                semantic_id="text.decoder.scalar-metadata.value",
                semantic_graph_path="text.decoder",
                model_part="main",
                module_kind="metadata.scalar",
                attributes=(),
                parameter_role="value",
                global_decoder_layer=None,
                moe_ordinal=None,
            ),
            format=format_descriptor,
            logical_dtype="float32",
            logical_shape=(1,),
            logical_axes=("scalar",),
            ownership=SemanticOwnership(
                OwnerFamilyBinding(
                    canonical_owner_family=owner_reference,
                    canonical_value_entry_id="main.scalar-metadata",
                    member_domain=domain,
                    member_to_owner_axes=(),
                    member_to_value_axes=(),
                )
            ),
        ),
        value_provenance=ValueProvenance.TRAINING_PARAMETER,
    )
    record = _source_record(
        record_id="main.scalar-metadata",
        native_name="model.scalar_metadata",
        native_owner="model.scalar_metadata",
        dtype=CanonicalSourceDType.FLOAT32,
        shape=(),
    )
    edge = CanonicalValueClassificationEdge(
        record_id=record.record_id,
        source_region=SourceRegion((), ()),
        output=OutputMemberTarget(entry.entry_id, domain, ()),
        canonical_owner_family=owner_reference,
        component_role=component_role,
        axis_mappings=(),
    )
    fragment = SemanticGraphBuildFragment(
        graph_instance_id="main",
        classification_edges=(edge,),
        source_owners=(
            SourceOwnerInventoryEntry(
                owner_family=owner_reference,
                domain=domain,
                source_mutability=record.source_mutability,
                mutability_evidence_source=record.mutability_evidence,
            ),
        ),
        inventory_entries=(entry,),
        manifest=SemanticGraphManifest(
            model_family="test-scalar",
            model_revision=graph_input.resolved_model_revision,
            graph_instance_id="main",
            lifecycle=graph_input.declaration.lifecycle,
            inventory_entry_ids=(entry.entry_id,),
        ),
        role_contributions=(),
    )
    return graph_input, (record,), fragment


def test_scalar_component_fragment_partitions_rank_zero_source() -> None:
    graph_input, records, fragment = _valid_scalar_component_fragment()

    validate_semantic_graph_build_fragment(1, graph_input, records, fragment)


def test_scalar_component_fragment_rejects_duplicate_scalar_region() -> None:
    graph_input, records, fragment = _valid_scalar_component_fragment()
    edge = fragment.classification_edges[0]
    broken = replace(fragment, classification_edges=(edge, edge))

    with pytest.raises(ValueError, match="overlapping source regions"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken)


def test_edge_axis_mapping_order_is_canonical() -> None:
    _, _, fragment = _valid_routed_fragment()
    edge = fragment.classification_edges[0]
    assert isinstance(edge, CanonicalValueClassificationEdge)

    reversed_edge = replace(edge, axis_mappings=tuple(reversed(edge.axis_mappings)))

    assert reversed_edge == edge


def _one_layer_edge(
    edge: CanonicalValueClassificationEdge,
    layer_member: LayerMember,
    source_ordinal: int,
) -> CanonicalValueClassificationEdge:
    assert edge.output.member_domain.layer_domain is not None
    expert_axes = edge.output.member_domain.independent_axes
    source_region = replace(
        edge.source_region,
        axis_selections=(
            SourceAxisSelection(
                0,
                (SourceIndexSpan(source_ordinal, source_ordinal + 1),),
            ),
            *edge.source_region.axis_selections[1:],
        ),
    )
    output = replace(
        edge.output,
        member_domain=FamilyIndexDomain(
            layer_domain=LayerDomain((layer_member,)),
            independent_axes=expert_axes,
        ),
    )
    mappings = tuple(
        replace(
            mapping,
            segments=(
                SourceOrdinalMapSegment(
                    SourceIndexSpan(source_ordinal, source_ordinal + 1),
                    0,
                ),
            ),
        )
        if isinstance(mapping.target, LayerCoordinateTarget)
        else mapping
        for mapping in edge.axis_mappings
    )
    return replace(
        edge,
        source_region=source_region,
        output=output,
        axis_mappings=mappings,
    )


def test_fragment_edge_order_is_canonical_for_colliding_prefix_keys() -> None:
    _, _, fragment = _valid_routed_fragment()
    edge = fragment.classification_edges[0]
    assert isinstance(edge, CanonicalValueClassificationEdge)
    layer_domain = edge.output.member_domain.layer_domain
    assert layer_domain is not None
    first = _one_layer_edge(edge, layer_domain.members[0], 0)
    second = _one_layer_edge(edge, layer_domain.members[1], 1)

    forward = replace(fragment, classification_edges=(first, second))
    reversed_fragment = replace(fragment, classification_edges=(second, first))

    assert forward == reversed_fragment


def test_fragment_rejects_marginally_equal_but_swapped_layer_relation() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    edge = fragment.classification_edges[0]
    assert isinstance(edge, CanonicalValueClassificationEdge)
    swapped = tuple(
        replace(
            mapping,
            segments=(
                SourceOrdinalMapSegment(SourceIndexSpan(0, 1), 1),
                SourceOrdinalMapSegment(SourceIndexSpan(1, 2), 0),
            ),
        )
        if mapping.target == LayerCoordinateTarget("moe_ordinal")
        else mapping
        for mapping in edge.axis_mappings
    )
    broken = replace(
        fragment,
        classification_edges=(replace(edge, axis_mappings=swapped),),
    )

    with pytest.raises(ValueError, match="correlated layer relation"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken)


def test_axis_mappings_use_ordinals_for_noncontiguous_layer_members() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    domain = _layer_expert_domain((LayerMember(10, 0), LayerMember(20, 1)))
    entry = _routed_entry(domain=domain)
    owner = replace(fragment.source_owners[0], domain=domain)
    edge = _whole_routed_edge(records[0], entry)
    valid = replace(
        fragment,
        classification_edges=(edge,),
        source_owners=(owner,),
        inventory_entries=(entry,),
    )

    validate_semantic_graph_build_fragment(1, graph_input, records, valid)


def test_axis_mappings_use_ordinals_for_string_family_members() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    domain = FamilyIndexDomain(
        layer_domain=LayerDomain((LayerMember(0, 0), LayerMember(1, 1))),
        independent_axes=(AxisDomain("expert", ("east", "west")),),
    )
    record = replace(records[0], shape=(2, 2, 8, 8))
    entry = _routed_entry(domain=domain)
    owner = replace(fragment.source_owners[0], domain=domain)
    edge = _whole_routed_edge(record, entry)
    edge = replace(
        edge,
        axis_mappings=tuple(
            replace(
                mapping,
                segments=(SourceOrdinalMapSegment(SourceIndexSpan(0, 2), 0),),
            )
            if mapping.target == FamilyIndexAxisTarget("expert")
            else mapping
            for mapping in edge.axis_mappings
        ),
    )
    valid = replace(
        fragment,
        classification_edges=(edge,),
        source_owners=(owner,),
        inventory_entries=(entry,),
    )

    validate_semantic_graph_build_fragment(1, graph_input, (record,), valid)


def test_fragment_rejects_unmapped_raw_source_axis() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    record = replace(records[0], shape=records[0].shape + (1,))
    edge = fragment.classification_edges[0]
    assert isinstance(edge, CanonicalValueClassificationEdge)
    edge = replace(edge, source_region=_whole_region(record.shape))
    broken = replace(fragment, classification_edges=(edge,))

    with pytest.raises(ValueError, match="every raw source axis"):
        validate_semantic_graph_build_fragment(1, graph_input, (record,), broken)


def test_fragment_rejects_non_layer_source_axis_fanout() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    record = replace(records[0], shape=(2, 4, 8))
    edge = fragment.classification_edges[0]
    assert isinstance(edge, CanonicalValueClassificationEdge)
    mappings = tuple(
        replace(mapping, source_axis_index=2)
        if mapping.target == ComponentAxisTarget(LOGICAL_VALUES, "input_features")
        else mapping
        for mapping in edge.axis_mappings
    )
    broken = replace(
        fragment,
        classification_edges=(
            replace(
                edge,
                source_region=_whole_region(record.shape),
                axis_mappings=mappings,
            ),
        ),
    )

    with pytest.raises(ValueError, match="one semantic target"):
        validate_semantic_graph_build_fragment(1, graph_input, (record,), broken)


def test_logical_axis_mapping_validation_is_compact_for_huge_extent() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    huge = 10**12
    record = replace(records[0], shape=(2, 4, huge, 8))
    entry = replace(
        fragment.inventory_entries[0],
        member=replace(
            fragment.inventory_entries[0].member,
            logical_shape=(huge, 8),
        ),
    )
    edge = _whole_routed_edge(record, entry)
    edge = replace(
        edge,
        axis_mappings=tuple(
            replace(
                mapping,
                segments=(SourceOrdinalMapSegment(SourceIndexSpan(0, huge), 0),),
            )
            if mapping.target == ComponentAxisTarget(LOGICAL_VALUES, "output_features")
            else mapping
            for mapping in edge.axis_mappings
        ),
    )
    valid = replace(
        fragment,
        classification_edges=(edge,),
        inventory_entries=(entry,),
    )

    validate_semantic_graph_build_fragment(1, graph_input, (record,), valid)


def test_explicit_component_mapping_uses_resolved_axis_extent() -> None:
    from nemo_rl.precision_policy.semantic import (
        BLOCK_SCALES,
        ComponentDescriptor,
        FormatDescriptor,
    )

    graph_input, records, fragment = _valid_routed_fragment()
    block_format = FormatDescriptor(
        format_id="test.block-scales.v1",
        family="test.block-scales",
        components=(
            ComponentDescriptor(
                role=BLOCK_SCALES,
                dtype="e8m0",
                encoding="block_scale",
                component_axes=(
                    LogicalComponentAxisSpec("output_features"),
                    LogicalComponentAxisSpec(
                        "input_features",
                        divisor=4,
                        rounding=AxisExtentRounding.CEIL,
                    ),
                ),
            ),
        ),
    )
    record = replace(
        records[0],
        dtype=CanonicalSourceDType.E8M0,
        shape=(2, 4, 8, 2),
        provenance=SourceRecordProvenance.CHECKPOINT_STORAGE,
    )
    old_entry = fragment.inventory_entries[0]
    entry = replace(
        old_entry,
        member=replace(old_entry.member, format=block_format),
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    edge = _whole_routed_edge(record, entry)
    edge = replace(
        edge,
        component_role=BLOCK_SCALES,
        axis_mappings=tuple(
            replace(
                mapping,
                target=ComponentAxisTarget(
                    BLOCK_SCALES,
                    mapping.target.component_axis,
                ),
                segments=(SourceOrdinalMapSegment(SourceIndexSpan(0, 2), 0),),
            )
            if isinstance(mapping.target, ComponentAxisTarget)
            and mapping.target.component_axis == "input_features"
            else replace(
                mapping,
                target=ComponentAxisTarget(
                    BLOCK_SCALES,
                    mapping.target.component_axis,
                ),
            )
            if isinstance(mapping.target, ComponentAxisTarget)
            else mapping
            for mapping in edge.axis_mappings
        ),
    )
    valid = replace(
        fragment,
        classification_edges=(edge,),
        inventory_entries=(entry,),
    )

    validate_semantic_graph_build_fragment(1, graph_input, (record,), valid)


def test_kimi_k25_style_components_use_only_declared_component_axes() -> None:
    packed_role = ComponentRole("packed_values")
    scales_role = ComponentRole("weight_scale")
    shape_role = ComponentRole("weight_shape")
    format_descriptor = FormatDescriptor(
        "test.kimi-k25-packed.v1",
        "test.kimi-k25-packed",
        (
            ComponentDescriptor(
                packed_role,
                "int32",
                encoding="packed",
                component_axes=(
                    LogicalComponentAxisSpec("output_features"),
                    LogicalComponentAxisSpec("input_features", divisor=8),
                ),
            ),
            ComponentDescriptor(
                scales_role,
                "bfloat16",
                encoding="scales",
                component_axes=(
                    LogicalComponentAxisSpec("output_features"),
                    LogicalComponentAxisSpec(
                        "input_features",
                        divisor=32,
                        rounding=AxisExtentRounding.CEIL,
                    ),
                ),
            ),
            ComponentDescriptor(
                shape_role,
                "int32",
                encoding="shape_metadata",
                component_axes=(LiteralComponentAxisSpec("shape_field", 2),),
            ),
        ),
    )
    graph_input, records, fragment = _valid_routed_fragment()
    base_record = records[0]
    entry = replace(
        fragment.inventory_entries[0],
        member=replace(
            fragment.inventory_entries[0].member,
            format=format_descriptor,
            logical_shape=(8, 64),
        ),
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    component_records = (
        replace(
            base_record,
            record_id="main.experts.gate.packed",
            source_native_name="model.layers.mlp.experts.gate_proj.weight_packed",
            dtype=CanonicalSourceDType.INT32,
            shape=(2, 4, 8, 8),
            provenance=SourceRecordProvenance.CHECKPOINT_STORAGE,
        ),
        replace(
            base_record,
            record_id="main.experts.gate.scales",
            source_native_name="model.layers.mlp.experts.gate_proj.weight_scale",
            dtype=CanonicalSourceDType.BFLOAT16,
            shape=(2, 4, 8, 2),
            provenance=SourceRecordProvenance.CHECKPOINT_STORAGE,
        ),
        replace(
            base_record,
            record_id="main.experts.gate.shape",
            source_native_name="model.layers.mlp.experts.gate_proj.weight_shape",
            dtype=CanonicalSourceDType.INT32,
            shape=(2, 4, 2),
            provenance=SourceRecordProvenance.CHECKPOINT_STORAGE,
        ),
    )

    def component_edge(
        record: SourceDiscoveryRecord,
        role: ComponentRole,
        axes: tuple[tuple[str, int], ...],
    ) -> CanonicalValueClassificationEdge:
        mappings: list[SourceToSemanticAxisMapping] = [
            SourceToSemanticAxisMapping(
                0,
                LayerCoordinateTarget("global_decoder_layer"),
                (SourceOrdinalMapSegment(SourceIndexSpan(0, 2), 0),),
            ),
            SourceToSemanticAxisMapping(
                0,
                LayerCoordinateTarget("moe_ordinal"),
                (SourceOrdinalMapSegment(SourceIndexSpan(0, 2), 0),),
            ),
            SourceToSemanticAxisMapping(
                1,
                FamilyIndexAxisTarget("expert"),
                (SourceOrdinalMapSegment(SourceIndexSpan(0, 4), 0),),
            ),
        ]
        mappings.extend(
            SourceToSemanticAxisMapping(
                source_axis_index,
                ComponentAxisTarget(role, axis_name),
                (SourceOrdinalMapSegment(SourceIndexSpan(0, extent), 0),),
            )
            for source_axis_index, (axis_name, extent) in enumerate(axes, start=2)
        )
        return CanonicalValueClassificationEdge(
            record_id=record.record_id,
            source_region=_whole_region(record.shape),
            output=OutputMemberTarget(entry.entry_id, entry.member.domain, ()),
            canonical_owner_family=entry.member.ownership.binding.canonical_owner_family,
            component_role=role,
            axis_mappings=tuple(mappings),
        )

    edges = (
        component_edge(
            component_records[0],
            packed_role,
            (("output_features", 8), ("input_features", 8)),
        ),
        component_edge(
            component_records[1],
            scales_role,
            (("output_features", 8), ("input_features", 2)),
        ),
        component_edge(
            component_records[2],
            shape_role,
            (("shape_field", 2),),
        ),
    )
    valid = replace(
        fragment,
        classification_edges=edges,
        inventory_entries=(entry,),
    )

    validate_semantic_graph_build_fragment(
        1,
        graph_input,
        component_records,
        valid,
    )


def test_fragment_rejects_raw_dtype_incompatible_with_component() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    record = replace(records[0], dtype=CanonicalSourceDType.FLOAT32)

    with pytest.raises(ValueError, match="raw dtype.*format component"):
        validate_semantic_graph_build_fragment(1, graph_input, (record,), fragment)


def test_fragment_does_not_interpret_uint8_as_e8m0_component() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    e8m0_format = FormatDescriptor(
        "test.e8m0.v1",
        "test.e8m0",
        (ComponentDescriptor(LOGICAL_VALUES, "e8m0"),),
    )
    record = replace(records[0], dtype=CanonicalSourceDType.UINT8)
    broken = replace(
        fragment,
        inventory_entries=(
            replace(
                fragment.inventory_entries[0],
                member=replace(
                    fragment.inventory_entries[0].member,
                    format=e8m0_format,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="raw dtype.*format component"):
        validate_semantic_graph_build_fragment(1, graph_input, (record,), broken)


def test_fragment_rejects_source_region_gap() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    edge = fragment.classification_edges[0]
    assert isinstance(edge, CanonicalValueClassificationEdge)
    incomplete = SourceRegion(
        source_shape=records[0].shape,
        axis_selections=(
            SourceAxisSelection(0, (SourceIndexSpan(0, 2),)),
            SourceAxisSelection(1, (SourceIndexSpan(0, 4),)),
            SourceAxisSelection(2, (SourceIndexSpan(0, 7),)),
            SourceAxisSelection(3, (SourceIndexSpan(0, 8),)),
        ),
    )
    broken = replace(
        fragment,
        classification_edges=(replace(edge, source_region=incomplete),),
    )

    with pytest.raises(ValueError, match="source region gap"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken)


def test_fragment_rejects_overlapping_source_regions() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    edge = fragment.classification_edges[0]
    broken = replace(fragment, classification_edges=(edge, edge))

    with pytest.raises(ValueError, match="overlapping source regions"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken)


def test_fragment_rejects_missing_format_component_output() -> None:
    from nemo_rl.precision_policy.semantic import (
        BLOCK_SCALES,
        VALUES,
        ComponentDescriptor,
        FormatDescriptor,
    )

    graph_input, records, fragment = _valid_routed_fragment()
    old_entry = fragment.inventory_entries[0]
    encoded_format = FormatDescriptor(
        format_id="test.block.v1",
        family="test.block",
        components=(
            ComponentDescriptor(VALUES, "e4m3"),
            ComponentDescriptor(BLOCK_SCALES, "float32"),
        ),
    )
    entry = replace(old_entry, member=replace(old_entry.member, format=encoded_format))
    record = replace(records[0], dtype=CanonicalSourceDType.E4M3)
    edge = _whole_routed_edge(record, entry)
    value_mappings = tuple(
        replace(
            mapping,
            target=ComponentAxisTarget(VALUES, mapping.target.component_axis),
        )
        if isinstance(mapping.target, ComponentAxisTarget)
        else mapping
        for mapping in edge.axis_mappings
    )
    broken = replace(
        fragment,
        classification_edges=(
            replace(edge, component_role=VALUES, axis_mappings=value_mappings),
        ),
        inventory_entries=(entry,),
    )

    with pytest.raises(ValueError, match="missing output.*block_scales"):
        validate_semantic_graph_build_fragment(1, graph_input, (record,), broken)


def test_fragment_rejects_edge_output_omitted_from_fragment() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    edge = fragment.classification_edges[0]
    assert isinstance(edge, CanonicalValueClassificationEdge)
    broken = replace(
        fragment,
        classification_edges=(
            replace(
                edge,
                output=replace(edge.output, inventory_entry_id="main.missing.entry"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown output inventory entry"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken)


def test_fragment_rejects_semantic_entry_or_owner_invented_without_edge() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    extra_entry = _routed_entry(
        entry_id="main.moe.routed.up",
        projection="up",
        owner_family=OwnerFamilyReference("main", "source.moe.routed.up"),
    )
    extra_owner = SourceOwnerInventoryEntry(
        owner_family=extra_entry.member.ownership.binding.canonical_owner_family,
        domain=extra_entry.member.domain,
        source_mutability=SourceMutability.MUTABLE,
        mutability_evidence_source=_evidence("extra-owner"),
    )
    broken_entry = replace(
        fragment,
        inventory_entries=fragment.inventory_entries + (extra_entry,),
        manifest=replace(
            fragment.manifest,
            inventory_entry_ids=fragment.manifest.inventory_entry_ids
            + (extra_entry.entry_id,),
        ),
    )
    broken_owner = replace(
        fragment,
        source_owners=fragment.source_owners + (extra_owner,),
    )

    with pytest.raises(ValueError, match="semantic entry.*no classification edge"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken_entry)
    with pytest.raises(ValueError, match="source owner.*no classification edge"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken_owner)


def test_fragment_rejects_owner_mutability_or_evidence_not_backed_by_raw_record() -> (
    None
):
    graph_input, records, fragment = _valid_routed_fragment()
    fabricated = replace(
        fragment.source_owners[0],
        source_mutability=SourceMutability.FROZEN,
        mutability_evidence_source=_evidence("fabricated-freeze"),
    )
    broken = replace(fragment, source_owners=(fabricated,))

    with pytest.raises(ValueError, match="owner mutability evidence.*raw discovery"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken)


def _slice_routed_edge(
    record: SourceDiscoveryRecord,
    entry: ParameterInventoryEntry,
    *,
    source_start: int,
    source_stop: int,
) -> CanonicalValueClassificationEdge:
    edge = _whole_routed_edge(record, entry)
    region = replace(
        edge.source_region,
        axis_selections=(
            SourceAxisSelection(0, (SourceIndexSpan(0, 2),)),
            SourceAxisSelection(1, (SourceIndexSpan(0, 4),)),
            SourceAxisSelection(2, (SourceIndexSpan(source_start, source_stop),)),
            SourceAxisSelection(3, (SourceIndexSpan(0, 8),)),
        ),
    )
    mappings = tuple(
        replace(
            mapping,
            segments=(
                SourceOrdinalMapSegment(
                    SourceIndexSpan(source_start, source_stop),
                    0,
                ),
            ),
        )
        if mapping.target == ComponentAxisTarget(LOGICAL_VALUES, "output_features")
        else mapping
        for mapping in edge.axis_mappings
    )
    return replace(edge, source_region=region, axis_mappings=mappings)


def test_one_fused_raw_owner_cannot_invent_two_canonical_owners() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    record = records[0]
    gate = replace(
        fragment.inventory_entries[0],
        member=replace(fragment.inventory_entries[0].member, logical_shape=(4, 8)),
    )
    up = _routed_entry(
        entry_id="main.moe.routed.up",
        projection="up",
        owner_family=OwnerFamilyReference("main", "source.moe.routed.up"),
    )
    up = replace(up, member=replace(up.member, logical_shape=(4, 8)))
    gate_owner = replace(fragment.source_owners[0], domain=gate.member.domain)
    up_owner = SourceOwnerInventoryEntry(
        owner_family=up.member.ownership.binding.canonical_owner_family,
        domain=up.member.domain,
        source_mutability=record.source_mutability,
        mutability_evidence_source=record.mutability_evidence,
    )
    broken = replace(
        fragment,
        classification_edges=(
            _slice_routed_edge(record, gate, source_start=0, source_stop=4),
            _slice_routed_edge(record, up, source_start=4, source_stop=8),
        ),
        source_owners=(gate_owner, up_owner),
        inventory_entries=(gate, up),
        manifest=replace(
            fragment.manifest,
            inventory_entry_ids=(gate.entry_id, up.entry_id),
        ),
        role_contributions=(
            replace(
                fragment.role_contributions[0],
                expected_inventory_entry_ids=(gate.entry_id, up.entry_id),
            ),
        ),
    )

    with pytest.raises(ValueError, match="native owner.*one canonical owner"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken)


def test_component_records_for_one_native_owner_must_share_authority() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    second_record = replace(
        records[0],
        record_id="main.experts.gate.scales",
        source_native_name="model.layers.mlp.experts.gate_proj.weight_scale",
        mutability_evidence=_evidence("different-mutability-authority"),
    )
    second_edge = replace(
        fragment.classification_edges[0],
        record_id=second_record.record_id,
    )
    broken = replace(
        fragment,
        classification_edges=fragment.classification_edges + (second_edge,),
    )

    with pytest.raises(ValueError, match="native owner.*disagree on authority"):
        validate_semantic_graph_build_fragment(
            1,
            graph_input,
            records + (second_record,),
            broken,
        )


def test_component_records_for_one_native_owner_can_share_canonical_owner() -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    direct = fragment.inventory_entries[0]
    owner_reference = direct.member.ownership.binding.canonical_owner_family
    up = _routed_entry(
        entry_id="main.moe.routed.up",
        projection="up",
        owner_family=owner_reference,
    )
    up_record = replace(
        records[0],
        record_id="main.experts.up",
        source_native_name="model.layers.mlp.experts.up_proj.weight",
    )
    valid = replace(
        fragment,
        classification_edges=fragment.classification_edges
        + (_whole_routed_edge(up_record, up),),
        inventory_entries=(direct, up),
        manifest=replace(
            fragment.manifest,
            inventory_entry_ids=(direct.entry_id, up.entry_id),
        ),
        role_contributions=(
            replace(
                fragment.role_contributions[0],
                expected_inventory_entry_ids=(direct.entry_id, up.entry_id),
            ),
        ),
    )

    validate_semantic_graph_build_fragment(
        1,
        graph_input,
        records + (up_record,),
        valid,
    )


@pytest.mark.parametrize(
    ("provenance", "value_provenance"),
    [
        (
            SourceRecordProvenance.TRAINING_RUNTIME,
            ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
        ),
        (SourceRecordProvenance.CHECKPOINT_STORAGE, ValueProvenance.TRAINING_PARAMETER),
        (SourceRecordProvenance.BACKEND_DERIVED, ValueProvenance.TRAINING_PARAMETER),
    ],
)
def test_fragment_rejects_value_provenance_not_backed_by_raw_record(
    provenance: SourceRecordProvenance,
    value_provenance: ValueProvenance,
) -> None:
    graph_input, records, fragment = _valid_routed_fragment()
    record = replace(records[0], provenance=provenance)
    entry = replace(fragment.inventory_entries[0], value_provenance=value_provenance)
    broken = replace(
        fragment,
        inventory_entries=(entry,),
        classification_edges=(_whole_routed_edge(record, entry),),
    )

    with pytest.raises(ValueError, match="value provenance.*raw discovery"):
        validate_semantic_graph_build_fragment(1, graph_input, (record,), broken)


def _valid_tied_fragment() -> tuple[
    GraphTopologyInput,
    tuple[SourceDiscoveryRecord, ...],
    SemanticGraphBuildFragment,
]:
    graph_input, records, fragment = _valid_routed_fragment()
    direct = fragment.inventory_entries[0]
    alias = _routed_entry(
        entry_id="main.moe.routed.up",
        projection="up",
        owner_family=direct.member.ownership.binding.canonical_owner_family,
    )
    alias = replace(
        alias,
        member=replace(
            alias.member,
            ownership=SemanticOwnership(
                replace(
                    alias.member.ownership.binding,
                    canonical_value_entry_id=direct.entry_id,
                )
            ),
        ),
        value_provenance=ValueProvenance.CANONICAL_ALIAS,
    )
    tied_record = replace(
        records[0],
        record_id="main.tied.up",
        source_native_name="model.layers.mlp.experts.up_proj.tied_weight",
        provenance=SourceRecordProvenance.TIED_STORAGE,
    )
    tied_edge = TiedAliasClassificationEdge(
        record_id=tied_record.record_id,
        aliased_source_region=_whole_region(tied_record.shape),
        alias_output=OutputMemberTarget(alias.entry_id, alias.member.domain, ()),
        canonical_owner_family=direct.member.ownership.binding.canonical_owner_family,
        canonical_value_entry_id=direct.entry_id,
        component_role=LOGICAL_VALUES,
        alias_to_canonical_axes=alias.member.ownership.binding.member_to_value_axes,
    )
    tied_fragment = replace(
        fragment,
        classification_edges=fragment.classification_edges + (tied_edge,),
        inventory_entries=(direct, alias),
        manifest=replace(
            fragment.manifest,
            inventory_entry_ids=(direct.entry_id, alias.entry_id),
        ),
        role_contributions=(
            replace(
                fragment.role_contributions[0],
                expected_inventory_entry_ids=(direct.entry_id, alias.entry_id),
            ),
        ),
    )
    return graph_input, records + (tied_record,), tied_fragment


def test_tied_alias_edge_resolves_direct_owner_without_consuming_it_twice() -> None:
    graph_input, records, fragment = _valid_tied_fragment()

    validate_semantic_graph_build_fragment(1, graph_input, records, fragment)

    assert len(fragment.source_owners) == 1


@pytest.mark.parametrize(
    "projections",
    [(), (AxisProjection("expert", "global_decoder_layer"),)],
)
def test_tied_alias_edge_requires_exact_alias_projection(
    projections: tuple[AxisProjection, ...],
) -> None:
    graph_input, records, fragment = _valid_tied_fragment()
    tied_edge = next(
        edge
        for edge in fragment.classification_edges
        if isinstance(edge, TiedAliasClassificationEdge)
    )
    broken = replace(
        fragment,
        classification_edges=tuple(
            replace(edge, alias_to_canonical_axes=projections)
            if edge is tied_edge
            else edge
            for edge in fragment.classification_edges
        ),
    )

    with pytest.raises(ValueError, match="tied edge projection"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken)


def test_tied_alias_edge_requires_existing_direct_target() -> None:
    graph_input, records, fragment = _valid_tied_fragment()
    alias = next(
        entry
        for entry in fragment.inventory_entries
        if entry.value_provenance == ValueProvenance.CANONICAL_ALIAS
    )
    alias = replace(
        alias,
        member=replace(
            alias.member,
            ownership=SemanticOwnership(
                replace(
                    alias.member.ownership.binding,
                    canonical_value_entry_id="main.missing.direct",
                )
            ),
        ),
    )
    broken = replace(
        fragment,
        inventory_entries=tuple(
            alias
            if entry.value_provenance == ValueProvenance.CANONICAL_ALIAS
            else entry
            for entry in fragment.inventory_entries
        ),
        classification_edges=tuple(
            replace(edge, canonical_value_entry_id="main.missing.direct")
            if isinstance(edge, TiedAliasClassificationEdge)
            else edge
            for edge in fragment.classification_edges
        ),
    )

    with pytest.raises(ValueError, match="tied edge direct target is missing"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken)


def test_cross_graph_tied_alias_defers_direct_target_resolution_to_bundle() -> None:
    _, records, fragment = _valid_tied_fragment()
    alias = next(
        entry
        for entry in fragment.inventory_entries
        if entry.value_provenance == ValueProvenance.CANONICAL_ALIAS
    )
    tied_record = next(
        record
        for record in records
        if record.provenance == SourceRecordProvenance.TIED_STORAGE
    )
    tied_edge = next(
        edge
        for edge in fragment.classification_edges
        if isinstance(edge, TiedAliasClassificationEdge)
    )
    graph_id = "mtp.0"
    graph_input = GraphTopologyInput(
        declaration=ExpectedGraphDeclaration(
            graph_instance_id=graph_id,
            model_identity="test/tied-mtp",
            lifecycle=GraphLifecycle(
                graph_kind=GraphKind.MTP,
                graph_provenance=GraphProvenance.TRAINING_RUNTIME,
                rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
            ),
        ),
        model_config={"model_type": "test_routed"},
        resolved_model_revision="content-addressed:tied-mtp",
        **_graph_discovery_fields(graph_id),
    )
    alias = replace(
        alias,
        entry_id="mtp.0.moe.routed.up",
        graph_instance_id=graph_id,
        member=replace(
            alias.member,
            pattern=replace(alias.member.pattern, model_part="mtp"),
        ),
    )
    tied_record = replace(
        tied_record,
        record_id="mtp.0.tied.up",
        graph_instance_id=graph_id,
    )
    tied_edge = replace(
        tied_edge,
        record_id=tied_record.record_id,
        alias_output=replace(
            tied_edge.alias_output,
            inventory_entry_id=alias.entry_id,
        ),
    )
    alias_fragment = SemanticGraphBuildFragment(
        graph_instance_id=graph_id,
        classification_edges=(tied_edge,),
        source_owners=(),
        inventory_entries=(alias,),
        manifest=SemanticGraphManifest(
            model_family="test-routed",
            model_revision=graph_input.resolved_model_revision,
            graph_instance_id=graph_id,
            lifecycle=graph_input.declaration.lifecycle,
            inventory_entry_ids=(alias.entry_id,),
        ),
        role_contributions=(),
    )

    validate_semantic_graph_build_fragment(
        1,
        graph_input,
        (tied_record,),
        alias_fragment,
    )


def test_tied_alias_edge_requires_same_underlying_native_owner() -> None:
    graph_input, records, fragment = _valid_tied_fragment()
    tied_record = next(
        record
        for record in records
        if record.provenance == SourceRecordProvenance.TIED_STORAGE
    )
    wrong_record = replace(
        tied_record,
        source_native_owner_id="different.native.owner",
    )

    with pytest.raises(ValueError, match="tied native owner.*direct target"):
        validate_semantic_graph_build_fragment(
            1,
            graph_input,
            tuple(
                wrong_record if record is tied_record else record for record in records
            ),
            fragment,
        )


def test_tied_alias_edge_requires_canonical_owner_mutability_evidence() -> None:
    graph_input, records, fragment = _valid_tied_fragment()
    tied_record = next(
        record
        for record in records
        if record.provenance == SourceRecordProvenance.TIED_STORAGE
    )
    wrong_record = replace(
        tied_record,
        source_mutability=SourceMutability.FROZEN,
        mutability_evidence=_evidence("fabricated-tied-freeze"),
    )

    with pytest.raises(ValueError, match="tied mutability evidence.*canonical raw"):
        validate_semantic_graph_build_fragment(
            1,
            graph_input,
            tuple(
                wrong_record if record is tied_record else record for record in records
            ),
            fragment,
        )


def test_tied_alias_edge_requires_exact_raw_view_cardinality() -> None:
    graph_input, records, fragment = _valid_tied_fragment()
    tied_record = next(
        record
        for record in records
        if record.provenance == SourceRecordProvenance.TIED_STORAGE
    )
    oversized_record = replace(tied_record, shape=tied_record.shape + (2,))
    broken = replace(
        fragment,
        classification_edges=tuple(
            replace(
                edge,
                aliased_source_region=_whole_region(oversized_record.shape),
            )
            if isinstance(edge, TiedAliasClassificationEdge)
            else edge
            for edge in fragment.classification_edges
        ),
    )

    with pytest.raises(ValueError, match="tied source cardinality"):
        validate_semantic_graph_build_fragment(
            1,
            graph_input,
            tuple(
                oversized_record if record is tied_record else record
                for record in records
            ),
            broken,
        )


def test_tied_alias_edge_rejects_mixed_consuming_edge_variant() -> None:
    graph_input, records, fragment = _valid_tied_fragment()
    tied_record = next(
        record
        for record in records
        if record.provenance == SourceRecordProvenance.TIED_STORAGE
    )
    direct = fragment.inventory_entries[0]
    mixed_edge = replace(
        _whole_routed_edge(tied_record, direct),
        record_id=tied_record.record_id,
    )
    broken = replace(
        fragment,
        classification_edges=fragment.classification_edges + (mixed_edge,),
    )

    with pytest.raises(ValueError, match="tied-storage record requires only tied"):
        validate_semantic_graph_build_fragment(1, graph_input, records, broken)


def _absent_auxiliary_fragment(
    participation: RolloutParticipation,
) -> tuple[GraphTopologyInput, SourceDiscoveryRecord, SemanticGraphBuildFragment]:
    graph_input = GraphTopologyInput(
        declaration=ExpectedGraphDeclaration(
            graph_instance_id="mtp.absent",
            model_identity="test/absent-mtp",
            lifecycle=GraphLifecycle(
                graph_kind=GraphKind.MTP,
                graph_provenance=GraphProvenance.TRAINING_RUNTIME,
                rollout_participation=participation,
            ),
        ),
        model_config={"model_type": "test_routed", "num_hidden_layers": 1},
        resolved_model_revision="content-addressed:absent-mtp",
        **_graph_discovery_fields("mtp.absent"),
    )
    record = _source_record(
        record_id="mtp.absent.expected",
        graph_instance_id="mtp.absent",
        native_name=None,
        native_owner=None,
        source_mutability=SourceMutability.ABSENT,
    )
    fragment = SemanticGraphBuildFragment(
        graph_instance_id="mtp.absent",
        classification_edges=(AbsentDiscoveryDispositionEdge(record.record_id),),
        source_owners=(),
        inventory_entries=(),
        manifest=SemanticGraphManifest(
            model_family="test-routed",
            model_revision=graph_input.resolved_model_revision,
            graph_instance_id="mtp.absent",
            lifecycle=graph_input.declaration.lifecycle,
            inventory_entry_ids=(),
        ),
        role_contributions=(),
    )
    return graph_input, record, fragment


def test_absent_disposition_is_the_only_valid_zero_output_edge() -> None:
    graph_input, record, fragment = _absent_auxiliary_fragment(
        RolloutParticipation.NOT_SERVED
    )

    validate_semantic_graph_build_fragment(1, graph_input, (record,), fragment)


def test_absent_disposition_cannot_justify_source_served_graph() -> None:
    graph_input, record, fragment = _absent_auxiliary_fragment(
        RolloutParticipation.SERVED_FROM_SOURCE
    )

    with pytest.raises(
        ValueError, match="source-served graph.*present canonical owner"
    ):
        validate_semantic_graph_build_fragment(1, graph_input, (record,), fragment)


def test_output_partition_singleton_hash_path_avoids_candidate_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nemo_rl.precision_policy.topology as topology

    layer_members = tuple(LayerMember(index, index) for index in range(100))
    expert_members = tuple(range(100))
    complete = FamilyIndexDomain(
        layer_domain=LayerDomain(layer_members),
        independent_axes=(AxisDomain("expert", expert_members),),
    )
    claims = tuple(
        FamilyIndexDomain(
            layer_domain=LayerDomain((layer,)),
            independent_axes=(AxisDomain("expert", (expert,)),),
        )
        for layer in layer_members
        for expert in expert_members
    )

    def fail_candidate_probe(*args: object) -> set[int]:
        raise AssertionError("singleton partition proof must use direct hashing")

    monkeypatch.setattr(topology, "_candidate_posting_union", fail_candidate_probe)

    topology._validate_output_domain_partition(complete, claims)
    with pytest.raises(ValueError, match="overlapping output member domains"):
        topology._validate_output_domain_partition(complete, claims + (claims[0],))


def test_output_partition_uses_rarest_factor_before_ubiquitous_posting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nemo_rl.precision_policy.topology as topology

    claim_count = 10_000
    complete = FamilyIndexDomain(
        layer_domain=None,
        independent_axes=(
            AxisDomain("shared", (0, 1)),
            AxisDomain("unique", tuple(range(claim_count))),
        ),
    )
    claims = tuple(
        FamilyIndexDomain(
            layer_domain=None,
            independent_axes=(
                AxisDomain("shared", (0, 1)),
                AxisDomain("unique", (index,)),
            ),
        )
        for index in range(claim_count)
    )
    original = topology._candidate_posting_union
    scanned_posting_ids = [0]

    def counted_union(
        postings: Mapping[LayerMember | int | str, set[int]],
        selected_members: tuple[LayerMember | int | str, ...],
    ) -> set[int]:
        for member in selected_members:
            posting = postings.get(member)
            if posting is not None:
                scanned_posting_ids[0] += len(posting)
        return original(postings, selected_members)

    monkeypatch.setattr(topology, "_candidate_posting_union", counted_union)

    topology._validate_output_domain_partition(complete, claims)
    with pytest.raises(ValueError, match="overlapping output member domains"):
        topology._validate_output_domain_partition(complete, claims + (claims[0],))
    assert scanned_posting_ids[0] <= 1


@dataclass(frozen=True)
class _BundleAdapter:
    adapter_id: str
    supported_model_type: str
    fragments: Mapping[str, SemanticGraphBuildFragment]

    def supports(self, model_config: Mapping[str, object]) -> bool:
        return model_config.get("model_type") == self.supported_model_type

    def classify_graph(
        self,
        schema_version: int,
        graph_input: GraphTopologyInput,
        source_records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment:
        return self.fragments[graph_input.declaration.graph_instance_id]


def _direct_graph_fixture(
    *,
    graph_instance_id: str,
    graph_kind: GraphKind,
    model_type: str,
    model_family: str,
    semantic_graph_path: str,
    model_part: str,
    namespaced_role: str | None = None,
) -> tuple[
    GraphTopologyInput,
    SourceDiscoveryRecord,
    SemanticGraphBuildFragment,
]:
    graph_input = GraphTopologyInput(
        declaration=ExpectedGraphDeclaration(
            graph_instance_id=graph_instance_id,
            model_identity=f"test/{model_family}",
            lifecycle=GraphLifecycle(
                graph_kind=graph_kind,
                graph_provenance=GraphProvenance.TRAINING_RUNTIME,
                rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
            ),
        ),
        model_config={"model_type": model_type},
        resolved_model_revision=f"content-addressed:{model_family}",
        **_graph_discovery_fields(graph_instance_id),
    )
    domain = _layer_expert_domain()
    entry_id = f"{graph_instance_id}.moe.routed.gate"
    owner_reference = OwnerFamilyReference(
        graph_instance_id,
        "source.moe.routed.gate",
    )
    base_entry = _routed_entry()
    binding = OwnerFamilyBinding(
        canonical_owner_family=owner_reference,
        canonical_value_entry_id=entry_id,
        member_domain=domain,
        member_to_owner_axes=_identity_axes(domain),
        member_to_value_axes=_identity_axes(domain),
    )
    entry = replace(
        base_entry,
        entry_id=entry_id,
        graph_instance_id=graph_instance_id,
        member=replace(
            base_entry.member,
            pattern=replace(
                base_entry.member.pattern,
                semantic_graph_path=semantic_graph_path,
                model_part=model_part,
            ),
            ownership=SemanticOwnership(binding),
        ),
    )
    record = _source_record(
        record_id=f"{graph_instance_id}.experts.gate",
        graph_instance_id=graph_instance_id,
        native_name=f"{graph_instance_id}.model.experts.gate.weight",
        native_owner=f"{graph_instance_id}.model.experts.gate",
    )
    owner = SourceOwnerInventoryEntry(
        owner_family=owner_reference,
        domain=domain,
        source_mutability=record.source_mutability,
        mutability_evidence_source=record.mutability_evidence,
    )
    if graph_kind == GraphKind.MAIN:
        contributions = (_routed_role_contribution(entry_id),)
    elif namespaced_role is None:
        contributions = ()
    else:
        contributions = (
            RoleDefinitionContribution(
                schema_version=1,
                role_name=namespaced_role,
                predicate=SemanticPredicate(
                    graph_kinds=(graph_kind,),
                    semantic_graph_paths=(semantic_graph_path,),
                    model_parts=(model_part,),
                    module_kinds=("moe.expert_ffn",),
                    attributes=(
                        AttributePredicate("expert_kind", ("routed",)),
                        AttributePredicate("projection", ("gate",)),
                    ),
                    parameter_roles=("kernel",),
                ),
                expected_inventory_entry_ids=(entry_id,),
            ),
        )
    fragment = SemanticGraphBuildFragment(
        graph_instance_id=graph_instance_id,
        classification_edges=(_whole_routed_edge(record, entry),),
        source_owners=(owner,),
        inventory_entries=(entry,),
        manifest=SemanticGraphManifest(
            model_family=model_family,
            model_revision=graph_input.resolved_model_revision,
            graph_instance_id=graph_instance_id,
            lifecycle=graph_input.declaration.lifecycle,
            inventory_entry_ids=(entry_id,),
        ),
        role_contributions=contributions,
    )
    return graph_input, record, fragment


def _install_bundle_adapters(
    monkeypatch: pytest.MonkeyPatch,
    *adapters: _BundleAdapter,
) -> None:
    import nemo_rl.precision_policy.topology as topology

    monkeypatch.setattr(topology, "_default_adapters", lambda: adapters)


def test_bundle_reconciles_graph_inputs_and_raw_discovery_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_input, main_record, main_fragment = _direct_graph_fixture(
        graph_instance_id="main",
        graph_kind=GraphKind.MAIN,
        model_type="main_family",
        model_family="main-family",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter("main-adapter", "main_family", {"main": main_fragment}),
    )
    source_discovery, expected = _partitioned_discovery(
        (main_input,),
        (main_record,),
    )

    with pytest.raises(
        ValueError, match="missing source discovery graph partition.*main"
    ):
        build_semantic_manifest_bundle(
            1,
            (main_input,),
            SourceDiscoveryInventory(()),
            expected,
        )
    extra_partition = replace(
        source_discovery.partitions[0],
        graph_instance_id="draft.extra",
    )
    with pytest.raises(
        ValueError, match="undeclared source discovery graph partition.*draft.extra"
    ):
        build_semantic_manifest_bundle(
            1,
            (main_input,),
            SourceDiscoveryInventory((*source_discovery.partitions, extra_partition)),
            expected,
        )
    with pytest.raises(ValueError, match="duplicate graph topology input"):
        build_semantic_manifest_bundle(
            1,
            (main_input, main_input),
            source_discovery,
            expected,
        )


def test_bundle_selects_main_and_drafter_adapters_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_input, main_record, main_fragment = _direct_graph_fixture(
        graph_instance_id="main",
        graph_kind=GraphKind.MAIN,
        model_type="main_family",
        model_family="main-family",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    draft_input, draft_record, draft_fragment = _direct_graph_fixture(
        graph_instance_id="draft.external",
        graph_kind=GraphKind.SPECULATIVE_DRAFTER,
        model_type="draft_family",
        model_family="draft-family",
        semantic_graph_path="draft.decoder",
        model_part="draft",
    )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter(
            "draft-adapter", "draft_family", {"draft.external": draft_fragment}
        ),
        _BundleAdapter("main-adapter", "main_family", {"main": main_fragment}),
    )

    bundle = _build_semantic_bundle(
        1,
        (draft_input, main_input),
        (draft_record, main_record),
    )

    assert tuple(manifest.graph_instance_id for manifest in bundle.manifests) == (
        "main",
        "draft.external",
    )
    assert bundle.manifest("main").model_family == "main-family"
    assert bundle.manifest("draft.external").model_family == "draft-family"
    assert tuple(role.role_name for role in bundle.role_definitions) == (
        "attention.qkvo",
        "embedding.ngram",
        "moe.routed_expert",
    )


def test_successful_adapter_boundary_strips_contributor_and_placement_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_input, record, fragment = _direct_graph_fixture(
        graph_instance_id="main",
        graph_kind=GraphKind.MAIN,
        model_type="main_family",
        model_family="main-family",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    placement_id = "private-pp7-tp3-ep2"
    expected = ExpectedContributorSet(
        contributor_ids=(placement_id,),
        authority=_discovery_evidence("trusted-membership", "8"),
    )
    bound_input = replace(
        graph_input,
        expected_contributor_authority=expected.to_authority(),
    )
    contribution = DiscoveryContribution(
        contributor_id=placement_id,
        graph_instance_id="main",
        producer_fingerprint=bound_input.source_producer_fingerprint,
        records=(record,),
    )
    partition = assemble_graph_discovery_partition(
        graph_input=bound_input,
        expected_contributors=expected,
        contributions=(contribution,),
    )
    captured_calls: list[
        tuple[int, GraphTopologyInput, tuple[SourceDiscoveryRecord, ...]]
    ] = []

    class CapturingAdapter:
        adapter_id = "capturing-adapter"

        def supports(self, model_config: Mapping[str, object]) -> bool:
            return model_config.get("model_type") == "main_family"

        def classify_graph(
            self,
            schema_version: int,
            adapter_graph_input: GraphTopologyInput,
            source_records: tuple[SourceDiscoveryRecord, ...],
        ) -> SemanticGraphBuildFragment:
            captured_calls.append((schema_version, adapter_graph_input, source_records))
            return fragment

    monkeypatch.setattr(
        topology_module,
        "_default_adapters",
        lambda: (CapturingAdapter(),),
    )
    bundle = build_semantic_manifest_bundle(
        1,
        (bound_input,),
        SourceDiscoveryInventory((partition,)),
        {"main": expected},
    )

    assert captured_calls == [(1, bound_input, (record,))]
    assert tuple(manifest.graph_instance_id for manifest in bundle.manifests) == (
        "main",
    )
    assert bundle.manifest("main").inventory_entry_ids == ("main.moe.routed.gate",)
    exposed = repr(
        (
            asdict(captured_calls[0][1]),
            tuple(asdict(item) for item in captured_calls[0][2]),
            asdict(bundle),
        )
    )
    for private_marker in (placement_id, "pp7", "tp3", "ep2"):
        assert private_marker not in exposed


def test_bundle_rejects_checkpoint_graph_direct_training_runtime_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_input, main_record, main_fragment = _direct_graph_fixture(
        graph_instance_id="main",
        graph_kind=GraphKind.MAIN,
        model_type="main_family",
        model_family="main-family",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    draft_input, draft_record, draft_fragment = _direct_graph_fixture(
        graph_instance_id="draft.static",
        graph_kind=GraphKind.SPECULATIVE_DRAFTER,
        model_type="draft_family",
        model_family="draft-family",
        semantic_graph_path="draft.decoder",
        model_part="draft",
    )
    evidence = ImmutableAuxiliaryEvidence(
        graph_instance_id="draft.static",
        model_identity=draft_input.declaration.model_identity,
        pinned_checkpoint_revision=draft_input.resolved_model_revision,
        checkpoint_content_digest="sha256:draft-static-checkpoint",
        model_config_digest="sha256:draft-static-config",
        semantic_domain_digest="sha256:draft-static-domain",
        evidence_source=EvidenceSource(
            kind=EvidenceSourceKind.PINNED_CHECKPOINT_MANIFEST,
            locator="checkpoint://draft.static/resolved",
            digest="sha256:draft-static-manifest",
        ),
    )
    lifecycle = GraphLifecycle(
        graph_kind=GraphKind.SPECULATIVE_DRAFTER,
        graph_provenance=GraphProvenance.EXTERNAL_CHECKPOINT,
        rollout_participation=RolloutParticipation.SERVED_FROM_CHECKPOINT,
        immutable_evidence=evidence,
    )
    draft_input = replace(
        draft_input,
        declaration=replace(draft_input.declaration, lifecycle=lifecycle),
    )
    draft_record = replace(
        draft_record,
        source_mutability=SourceMutability.FROZEN,
    )
    draft_fragment = replace(
        draft_fragment,
        source_owners=(
            replace(
                draft_fragment.source_owners[0],
                source_mutability=SourceMutability.FROZEN,
            ),
        ),
        manifest=replace(draft_fragment.manifest, lifecycle=lifecycle),
    )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter(
            "draft-adapter",
            "draft_family",
            {"draft.static": draft_fragment},
        ),
        _BundleAdapter("main-adapter", "main_family", {"main": main_fragment}),
    )

    with pytest.raises(ValueError, match="checkpoint-served.*training-runtime"):
        _build_semantic_bundle(
            1,
            (main_input, draft_input),
            (main_record, draft_record),
        )


def test_bundle_rejects_cross_graph_canonical_native_owner_split_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_input, main_record, main_fragment = _direct_graph_fixture(
        graph_instance_id="main",
        graph_kind=GraphKind.MAIN,
        model_type="main_family",
        model_family="main-family",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    draft_input, draft_record, draft_fragment = _direct_graph_fixture(
        graph_instance_id="draft.external",
        graph_kind=GraphKind.SPECULATIVE_DRAFTER,
        model_type="draft_family",
        model_family="draft-family",
        semantic_graph_path="draft.decoder",
        model_part="draft",
    )
    draft_record = replace(
        draft_record,
        source_native_owner_id=main_record.source_native_owner_id,
        provenance=main_record.provenance,
        provenance_evidence=main_record.provenance_evidence,
        source_mutability=main_record.source_mutability,
        mutability_evidence=main_record.mutability_evidence,
    )
    draft_fragment = replace(
        draft_fragment,
        source_owners=(
            replace(
                draft_fragment.source_owners[0],
                source_mutability=main_record.source_mutability,
                mutability_evidence_source=main_record.mutability_evidence,
            ),
        ),
    )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter(
            "draft-adapter", "draft_family", {"draft.external": draft_fragment}
        ),
        _BundleAdapter("main-adapter", "main_family", {"main": main_fragment}),
    )

    with pytest.raises(ValueError, match="canonical native owner.*multiple owners"):
        _build_semantic_bundle(
            1,
            (main_input, draft_input),
            (main_record, draft_record),
        )


@pytest.mark.parametrize(
    "authority_field",
    ("provenance", "provenance_evidence", "source_mutability", "mutability_evidence"),
)
def test_bundle_rejects_cross_graph_canonical_native_owner_authority_conflict(
    monkeypatch: pytest.MonkeyPatch,
    authority_field: str,
) -> None:
    main_input, main_record, main_fragment = _direct_graph_fixture(
        graph_instance_id="main",
        graph_kind=GraphKind.MAIN,
        model_type="main_family",
        model_family="main-family",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    draft_input, draft_record, draft_fragment = _direct_graph_fixture(
        graph_instance_id="draft.external",
        graph_kind=GraphKind.SPECULATIVE_DRAFTER,
        model_type="draft_family",
        model_family="draft-family",
        semantic_graph_path="draft.decoder",
        model_part="draft",
    )
    draft_record = replace(
        draft_record,
        source_native_owner_id=main_record.source_native_owner_id,
        provenance=main_record.provenance,
        provenance_evidence=main_record.provenance_evidence,
        source_mutability=main_record.source_mutability,
        mutability_evidence=main_record.mutability_evidence,
    )
    draft_fragment = replace(
        draft_fragment,
        source_owners=(
            replace(
                draft_fragment.source_owners[0],
                source_mutability=main_record.source_mutability,
                mutability_evidence_source=main_record.mutability_evidence,
            ),
        ),
    )
    if authority_field == "provenance":
        draft_record = replace(
            draft_record,
            provenance=SourceRecordProvenance.CHECKPOINT_STORAGE,
        )
        draft_fragment = replace(
            draft_fragment,
            inventory_entries=(
                replace(
                    draft_fragment.inventory_entries[0],
                    value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
                ),
            ),
        )
    elif authority_field == "provenance_evidence":
        draft_record = replace(
            draft_record,
            provenance_evidence=_evidence("conflicting-cross-graph-provenance"),
        )
    elif authority_field == "source_mutability":
        draft_record = replace(
            draft_record,
            source_mutability=SourceMutability.FROZEN,
        )
        draft_fragment = replace(
            draft_fragment,
            source_owners=(
                replace(
                    draft_fragment.source_owners[0],
                    source_mutability=SourceMutability.FROZEN,
                    mutability_evidence_source=main_record.mutability_evidence,
                ),
            ),
        )
    else:
        draft_record = replace(
            draft_record,
            mutability_evidence=_evidence("conflicting-cross-graph-mutability"),
        )
        draft_fragment = replace(
            draft_fragment,
            source_owners=(
                replace(
                    draft_fragment.source_owners[0],
                    mutability_evidence_source=draft_record.mutability_evidence,
                ),
            ),
        )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter(
            "draft-adapter", "draft_family", {"draft.external": draft_fragment}
        ),
        _BundleAdapter("main-adapter", "main_family", {"main": main_fragment}),
    )

    with pytest.raises(ValueError, match="canonical native owner.*authority evidence"):
        _build_semantic_bundle(
            1,
            (main_input, draft_input),
            (main_record, draft_record),
        )


def test_bundle_merges_equal_namespaced_role_contributions_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_input, main_record, main_fragment = _direct_graph_fixture(
        graph_instance_id="main",
        graph_kind=GraphKind.MAIN,
        model_type="main_family",
        model_family="main-family",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    draft_a = _direct_graph_fixture(
        graph_instance_id="draft.a",
        graph_kind=GraphKind.SPECULATIVE_DRAFTER,
        model_type="draft_family",
        model_family="draft-family-a",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        namespaced_role="test.draft_expert",
    )
    draft_b = _direct_graph_fixture(
        graph_instance_id="draft.b",
        graph_kind=GraphKind.SPECULATIVE_DRAFTER,
        model_type="draft_family",
        model_family="draft-family-b",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        namespaced_role="test.draft_expert",
    )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter(
            "draft-adapter",
            "draft_family",
            {"draft.a": draft_a[2], "draft.b": draft_b[2]},
        ),
        _BundleAdapter("main-adapter", "main_family", {"main": main_fragment}),
    )

    forward = _build_semantic_bundle(
        1,
        (main_input, draft_a[0], draft_b[0]),
        (main_record, draft_a[1], draft_b[1]),
    )
    reverse = _build_semantic_bundle(
        1,
        (draft_b[0], draft_a[0], main_input),
        (draft_b[1], draft_a[1], main_record),
    )

    definition = forward.role_definition(1, "test.draft_expert")
    assert definition.expected_domain.inventory_entry_ids == (
        "draft.a.moe.routed.gate",
        "draft.b.moe.routed.gate",
    )
    assert reverse == forward


def _cross_graph_alias_fixture() -> tuple[
    tuple[GraphTopologyInput, ...],
    tuple[SourceDiscoveryRecord, ...],
    tuple[SemanticGraphBuildFragment, ...],
]:
    main_input, main_record, main_fragment = _direct_graph_fixture(
        graph_instance_id="main",
        graph_kind=GraphKind.MAIN,
        model_type="main_family",
        model_family="main-family",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    direct = main_fragment.inventory_entries[0]
    alias_graph_id = "mtp.0"
    alias_input = GraphTopologyInput(
        declaration=ExpectedGraphDeclaration(
            graph_instance_id=alias_graph_id,
            model_identity="test/mtp-family",
            lifecycle=GraphLifecycle(
                graph_kind=GraphKind.MTP,
                graph_provenance=GraphProvenance.TRAINING_RUNTIME,
                rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
            ),
        ),
        model_config={"model_type": "mtp_family"},
        resolved_model_revision="content-addressed:mtp-family",
        **_graph_discovery_fields(alias_graph_id),
    )
    alias_entry_id = "mtp.0.moe.routed.gate"
    direct_binding = direct.member.ownership.binding
    alias_entry = replace(
        direct,
        entry_id=alias_entry_id,
        graph_instance_id=alias_graph_id,
        member=replace(
            direct.member,
            pattern=replace(
                direct.member.pattern,
                semantic_graph_path="auxiliary.mtp",
                model_part="mtp",
            ),
            ownership=SemanticOwnership(
                replace(
                    direct_binding,
                    canonical_value_entry_id=direct.entry_id,
                )
            ),
        ),
        value_provenance=ValueProvenance.CANONICAL_ALIAS,
    )
    tied_record = _source_record(
        record_id="mtp.0.tied.gate",
        graph_instance_id=alias_graph_id,
        native_name="mtp.0.model.experts.gate.tied_weight",
        native_owner=main_record.source_native_owner_id,
        provenance=SourceRecordProvenance.TIED_STORAGE,
    )
    tied_record = replace(
        tied_record,
        source_mutability=main_record.source_mutability,
        mutability_evidence=main_record.mutability_evidence,
    )
    alias_fragment = SemanticGraphBuildFragment(
        graph_instance_id=alias_graph_id,
        classification_edges=(
            TiedAliasClassificationEdge(
                record_id=tied_record.record_id,
                aliased_source_region=_whole_region(tied_record.shape),
                alias_output=OutputMemberTarget(
                    alias_entry.entry_id,
                    alias_entry.member.domain,
                    (),
                ),
                canonical_owner_family=direct_binding.canonical_owner_family,
                canonical_value_entry_id=direct.entry_id,
                component_role=LOGICAL_VALUES,
                alias_to_canonical_axes=alias_entry.member.ownership.binding.member_to_value_axes,
            ),
        ),
        source_owners=(),
        inventory_entries=(alias_entry,),
        manifest=SemanticGraphManifest(
            model_family="mtp-family",
            model_revision=alias_input.resolved_model_revision,
            graph_instance_id=alias_graph_id,
            lifecycle=alias_input.declaration.lifecycle,
            inventory_entry_ids=(alias_entry.entry_id,),
        ),
        role_contributions=(),
    )
    return (
        (main_input, alias_input),
        (main_record, tied_record),
        (main_fragment, alias_fragment),
    )


def test_alias_normalization_fast_path_does_not_index_direct_only_graphs() -> None:
    _, records, fragment = _valid_routed_fragment()
    canonical_edge = fragment.classification_edges[0]

    class RepeatedCanonicalEdges:
        def __init__(
            self,
            edge: CanonicalValueClassificationEdge,
            count: int,
        ) -> None:
            self.edge = edge
            self.count = count
            self.iterations = 0

        def __iter__(self) -> Iterator[CanonicalValueClassificationEdge]:
            for _ in range(self.count):
                self.iterations += 1
                yield self.edge

    class InaccessibleRecords(dict[str, SourceDiscoveryRecord]):
        def __getitem__(self, key: str) -> SourceDiscoveryRecord:
            raise AssertionError(f"allocated canonical indexes for {key}")

        def values(self) -> Never:
            raise AssertionError("allocated native-owner indexes")

    repeated = RepeatedCanonicalEdges(canonical_edge, 300_000)
    object.__setattr__(fragment, "classification_edges", repeated)

    contracts = topology_module._normalize_source_alias_contracts(
        (fragment,),
        InaccessibleRecords({records[0].record_id: records[0]}),
    )

    assert contracts == ()
    assert repeated.iterations == 300_000


def test_alias_projection_reuses_large_canonical_parent_index() -> None:
    iterations = 0

    class CountingMembers(tuple[int, ...]):
        def __iter__(self) -> Iterator[int]:
            nonlocal iterations
            for member in super().__iter__():
                iterations += 1
                if iterations > 30_000:
                    raise AssertionError("rescanned the canonical parent domain")
                yield member

    canonical_axis = AxisDomain("canonical_expert", tuple(range(10_000)))
    object.__setattr__(
        canonical_axis,
        "members",
        CountingMembers(canonical_axis.members),
    )
    canonical_domain = FamilyIndexDomain(None, (canonical_axis,))
    projections = (AxisProjection("alias_expert", "canonical_expert"),)
    parent_index_cache: topology_module._AliasProjectionParentIndexCache = {}

    for expert in range(10_000):
        alias_domain = FamilyIndexDomain(
            None,
            (AxisDomain("alias_expert", (expert,)),),
        )
        projected = topology_module._project_alias_domain(
            alias_domain,
            canonical_domain,
            projections,
            parent_index_cache=parent_index_cache,
        )
        assert projected.cardinality == 1

    assert iterations <= 30_000


def test_bundle_resolves_cross_graph_mtp_alias_to_one_native_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_alias_fixture()
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter("main-adapter", "main_family", {"main": fragments[0]}),
        _BundleAdapter("mtp-adapter", "mtp_family", {"mtp.0": fragments[1]}),
    )

    bundle = _build_semantic_bundle(
        1,
        tuple(reversed(graph_inputs)),
        tuple(reversed(records)),
    )

    assert len(bundle.inventory.owners) == 1
    assert (
        bundle.inventory.entries[1].value_provenance == ValueProvenance.CANONICAL_ALIAS
    )
    assert len(bundle.source_alias_contracts) == 1
    assert isinstance(
        bundle.source_alias_contracts[0],
        IdenticalStorageSourceAliasContract,
    )
    assert (
        bundle.source_alias_contracts[0].storage_identity_evidence
        == records[1].provenance_evidence
    )
    assert bundle.owner_refit_requirements("mtp.0") == (
        (
            OwnerFamilyReference("main", "source.moe.routed.gate"),
            bundle.owner_refit_requirements("main")[0][1],
        ),
    )


def _cross_graph_replica_fixture() -> tuple[
    tuple[GraphTopologyInput, ...],
    tuple[SourceDiscoveryRecord, ...],
    tuple[SemanticGraphBuildFragment, ...],
]:
    graph_inputs, records, fragments = _cross_graph_alias_fixture()
    main_record, tied_record = records
    tied_edge = fragments[1].classification_edges[0]
    assert isinstance(tied_edge, TiedAliasClassificationEdge)
    replica_record = replace(
        tied_record,
        source_native_name="mtp.0.model.experts.gate.replica_weight",
        source_native_owner_id="mtp.0.model.experts.gate.replica",
        provenance=SourceRecordProvenance.SYNCHRONIZED_REPLICA,
        provenance_evidence=_evidence("mtp-0-replica-synchronization"),
    )
    synchronization = SourceReplicaSynchronizationEvidence(
        replica_group_id="replicas.mtp.0",
        boundary=SourceSynchronizationBoundary.SOURCE_VERSION_READY,
        evidence_source=replica_record.provenance_evidence,
    )
    replica_edge = SynchronizedReplicaAliasClassificationEdge(
        record_id=replica_record.record_id,
        replica_source_region=tied_edge.aliased_source_region,
        alias_output=tied_edge.alias_output,
        canonical_record_id=main_record.record_id,
        canonical_source_region=tied_edge.aliased_source_region,
        canonical_owner_family=tied_edge.canonical_owner_family,
        canonical_value_entry_id=tied_edge.canonical_value_entry_id,
        component_role=tied_edge.component_role,
        alias_to_canonical_axes=tied_edge.alias_to_canonical_axes,
        synchronization=synchronization,
    )
    replica_fragment = replace(
        fragments[1],
        classification_edges=(replica_edge,),
    )
    return (
        graph_inputs,
        (main_record, replica_record),
        (
            fragments[0],
            replica_fragment,
        ),
    )


def test_bundle_persists_cross_graph_synchronized_replica_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter("main-adapter", "main_family", {"main": fragments[0]}),
        _BundleAdapter("mtp-adapter", "mtp_family", {"mtp.0": fragments[1]}),
    )

    bundle = _build_semantic_bundle(
        1,
        graph_inputs,
        records,
    )

    assert len(bundle.inventory.owners) == 1
    assert len(bundle.source_alias_contracts) == 1
    contract = bundle.source_alias_contracts[0]
    assert isinstance(contract, SynchronizedReplicaSourceAliasContract)
    assert contract.synchronization.evidence_source == records[1].provenance_evidence
    assert bundle.owner_refit_requirements("mtp.0") == (
        (
            OwnerFamilyReference("main", "source.moe.routed.gate"),
            bundle.owner_refit_requirements("main")[0][1],
        ),
    )


@pytest.mark.parametrize(
    "provenance",
    [
        SourceRecordProvenance.TRAINING_RUNTIME,
        SourceRecordProvenance.CHECKPOINT_STORAGE,
        SourceRecordProvenance.BACKEND_DERIVED,
        SourceRecordProvenance.TIED_STORAGE,
    ],
)
def test_replica_alias_edge_requires_synchronized_replica_raw_provenance(
    provenance: SourceRecordProvenance,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    wrong_record = replace(records[1], provenance=provenance)

    with pytest.raises(ValueError, match="record requires"):
        validate_semantic_graph_build_fragment(
            1,
            graph_inputs[1],
            (wrong_record,),
            fragments[1],
        )


def test_synchronized_replica_record_rejects_mixed_alias_edge_variants() -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    replica_edge = fragments[1].classification_edges[0]
    assert isinstance(replica_edge, SynchronizedReplicaAliasClassificationEdge)
    tied_edge = TiedAliasClassificationEdge(
        record_id=replica_edge.record_id,
        aliased_source_region=replica_edge.replica_source_region,
        alias_output=replica_edge.alias_output,
        canonical_owner_family=replica_edge.canonical_owner_family,
        canonical_value_entry_id=replica_edge.canonical_value_entry_id,
        component_role=replica_edge.component_role,
        alias_to_canonical_axes=replica_edge.alias_to_canonical_axes,
    )
    broken = replace(
        fragments[1],
        classification_edges=(replica_edge, tied_edge),
    )

    with pytest.raises(ValueError, match="requires only replica alias edges"):
        validate_semantic_graph_build_fragment(
            1,
            graph_inputs[1],
            (records[1],),
            broken,
        )


def _build_replica_fixture(
    monkeypatch: pytest.MonkeyPatch,
    graph_inputs: tuple[GraphTopologyInput, ...],
    records: tuple[SourceDiscoveryRecord, ...],
    fragments: tuple[SemanticGraphBuildFragment, ...],
) -> SemanticManifestBundle:
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter("main-adapter", "main_family", {"main": fragments[0]}),
        _BundleAdapter("mtp-adapter", "mtp_family", {"mtp.0": fragments[1]}),
    )
    return _build_semantic_bundle(
        1,
        graph_inputs,
        records,
    )


def test_replica_native_owner_must_be_distinct_from_canonical_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    wrong_replica = replace(
        records[1],
        source_native_owner_id=records[0].source_native_owner_id,
    )

    with pytest.raises(ValueError, match="replica native owner.*canonical"):
        _build_replica_fixture(
            monkeypatch,
            graph_inputs,
            (records[0], wrong_replica),
            fragments,
        )


@pytest.mark.parametrize("canonical_record_id", ["missing.record", "mtp.0.tied.gate"])
def test_replica_edge_requires_distinct_existing_canonical_record(
    monkeypatch: pytest.MonkeyPatch,
    canonical_record_id: str,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    edge = fragments[1].classification_edges[0]
    assert isinstance(edge, SynchronizedReplicaAliasClassificationEdge)
    broken = replace(
        fragments[1],
        classification_edges=(replace(edge, canonical_record_id=canonical_record_id),),
    )

    with pytest.raises(ValueError, match="replica canonical source record|must differ"):
        _build_replica_fixture(
            monkeypatch,
            graph_inputs,
            records,
            (fragments[0], broken),
        )


def test_replica_edge_requires_exact_corresponding_source_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    edge = fragments[1].classification_edges[0]
    assert isinstance(edge, SynchronizedReplicaAliasClassificationEdge)
    broken = replace(
        fragments[1],
        classification_edges=(
            replace(
                edge,
                canonical_source_region=_layer_slice_region(
                    edge.canonical_source_region.source_shape,
                    0,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="exactly match canonical region"):
        _build_replica_fixture(
            monkeypatch,
            graph_inputs,
            records,
            (fragments[0], broken),
        )


def test_replica_edge_requires_canonical_owner_mutability_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    wrong_replica = replace(
        records[1],
        source_mutability=SourceMutability.FROZEN,
        mutability_evidence=_evidence("fabricated-replica-freeze"),
    )

    with pytest.raises(ValueError, match="replica mutability evidence"):
        _build_replica_fixture(
            monkeypatch,
            graph_inputs,
            (records[0], wrong_replica),
            fragments,
        )


def test_replica_edge_requires_raw_synchronization_evidence_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    edge = fragments[1].classification_edges[0]
    assert isinstance(edge, SynchronizedReplicaAliasClassificationEdge)
    broken = replace(
        fragments[1],
        classification_edges=(
            replace(
                edge,
                synchronization=replace(
                    edge.synchronization,
                    evidence_source=_evidence("fabricated-replica-sync"),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="raw provenance evidence"):
        _build_replica_fixture(
            monkeypatch,
            graph_inputs,
            records,
            (fragments[0], broken),
        )


def test_replica_edge_rejects_checkpoint_canonical_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    checkpoint_record = replace(
        records[0],
        provenance=SourceRecordProvenance.CHECKPOINT_STORAGE,
    )
    checkpoint_entry = replace(
        fragments[0].inventory_entries[0],
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    checkpoint_fragment = replace(
        fragments[0],
        inventory_entries=(checkpoint_entry,),
    )

    with pytest.raises(ValueError, match="training-runtime parameter authority"):
        _build_replica_fixture(
            monkeypatch,
            graph_inputs,
            (checkpoint_record, records[1]),
            (checkpoint_fragment, fragments[1]),
        )


def test_replica_edge_rejects_raw_dtype_mismatch() -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    wrong_record = replace(records[1], dtype=CanonicalSourceDType.FLOAT16)

    with pytest.raises(ValueError, match="raw dtype"):
        validate_semantic_graph_build_fragment(
            1,
            graph_inputs[1],
            (wrong_record,),
            fragments[1],
        )


def test_replica_edge_rejects_equal_cardinality_raw_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    edge = fragments[1].classification_edges[0]
    assert isinstance(edge, SynchronizedReplicaAliasClassificationEdge)
    wrong_shape = (2, 4, 4, 16)
    wrong_record = replace(records[1], shape=wrong_shape)
    broken = replace(
        fragments[1],
        classification_edges=(
            replace(edge, replica_source_region=_whole_region(wrong_shape)),
        ),
    )

    with pytest.raises(ValueError, match="raw dtype and shape"):
        _build_replica_fixture(
            monkeypatch,
            graph_inputs,
            (records[0], wrong_record),
            (fragments[0], broken),
        )


@pytest.mark.parametrize("failure", ["component", "owner", "projection"])
def test_replica_edge_requires_exact_semantic_binding(
    failure: str,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    edge = fragments[1].classification_edges[0]
    assert isinstance(edge, SynchronizedReplicaAliasClassificationEdge)
    if failure == "component":
        broken_edge = replace(edge, component_role=ComponentRole("unknown"))
        error = "unknown format component"
    elif failure == "owner":
        broken_edge = replace(
            edge,
            canonical_owner_family=OwnerFamilyReference(
                "main", "source.moe.routed.other"
            ),
        )
        error = "replica edge owner"
    else:
        broken_edge = replace(edge, alias_to_canonical_axes=())
        error = "replica edge projection"
    broken = replace(fragments[1], classification_edges=(broken_edge,))

    with pytest.raises(ValueError, match=error):
        validate_semantic_graph_build_fragment(
            1,
            graph_inputs[1],
            (records[1],),
            broken,
        )


@pytest.mark.parametrize("coverage", ["gap", "overlap"])
def test_replica_record_requires_exact_nonoverlapping_edge_coverage(
    coverage: str,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    edge = fragments[1].classification_edges[0]
    assert isinstance(edge, SynchronizedReplicaAliasClassificationEdge)
    alias_entry = fragments[1].inventory_entries[0]
    first = replace(
        edge,
        replica_source_region=_layer_slice_region(
            edge.replica_source_region.source_shape,
            0,
        ),
        alias_output=_layer_slice_target(alias_entry, 0),
    )
    edges = (first,) if coverage == "gap" else (edge, edge)
    broken = replace(fragments[1], classification_edges=edges)

    with pytest.raises(ValueError, match="source region gap|overlapping tied source"):
        validate_semantic_graph_build_fragment(
            1,
            graph_inputs[1],
            (records[1],),
            broken,
        )


def test_replica_edges_cannot_swap_semantic_and_canonical_subdomains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    direct_entry = fragments[0].inventory_entries[0]
    alias_entry = fragments[1].inventory_entries[0]
    canonical_edge = fragments[0].classification_edges[0]
    replica_edge = fragments[1].classification_edges[0]
    assert isinstance(canonical_edge, CanonicalValueClassificationEdge)
    assert isinstance(replica_edge, SynchronizedReplicaAliasClassificationEdge)
    split_main = replace(
        fragments[0],
        classification_edges=tuple(
            _slice_canonical_edge(canonical_edge, direct_entry, layer)
            for layer in range(2)
        ),
    )
    swapped_replica = replace(
        fragments[1],
        classification_edges=tuple(
            replace(
                replica_edge,
                replica_source_region=_layer_slice_region(
                    replica_edge.replica_source_region.source_shape,
                    source_layer,
                ),
                canonical_source_region=_layer_slice_region(
                    replica_edge.canonical_source_region.source_shape,
                    source_layer,
                ),
                alias_output=_layer_slice_target(alias_entry, alias_layer),
            )
            for source_layer, alias_layer in ((0, 1), (1, 0))
        ),
    )

    with pytest.raises(ValueError, match="semantic subdomain"):
        _build_replica_fixture(
            monkeypatch,
            graph_inputs,
            records,
            (split_main, swapped_replica),
        )


def _add_second_replica_alias(
    records: tuple[SourceDiscoveryRecord, ...],
    fragments: tuple[SemanticGraphBuildFragment, ...],
    *,
    native_owner: str,
    replica_group_id: str,
    evidence_name: str,
) -> tuple[
    tuple[SourceDiscoveryRecord, ...],
    tuple[SemanticGraphBuildFragment, ...],
]:
    alias = fragments[1].inventory_entries[0]
    assert isinstance(alias.member, SemanticTensorFamily)
    pattern = alias.member.pattern
    second_alias = replace(
        alias,
        entry_id="mtp.0.moe.routed.up",
        member=replace(
            alias.member,
            pattern=replace(
                pattern,
                path_segments=pattern.path_segments[:-1] + (LiteralPathSegment("up"),),
                attributes=tuple(
                    (name, "up" if name == "projection" else value)
                    for name, value in pattern.attributes
                ),
            ),
        ),
    )
    first_edge = fragments[1].classification_edges[0]
    assert isinstance(first_edge, SynchronizedReplicaAliasClassificationEdge)
    evidence = _evidence(evidence_name)
    second_record = replace(
        records[1],
        record_id="mtp.0.replica.up",
        source_native_name="mtp.0.model.experts.up.replica_weight",
        source_native_owner_id=native_owner,
        provenance_evidence=evidence,
    )
    second_edge = replace(
        first_edge,
        record_id=second_record.record_id,
        alias_output=replace(
            first_edge.alias_output,
            inventory_entry_id=second_alias.entry_id,
        ),
        synchronization=replace(
            first_edge.synchronization,
            replica_group_id=replica_group_id,
            evidence_source=evidence,
        ),
    )
    second_fragment = replace(
        fragments[1],
        classification_edges=(first_edge, second_edge),
        inventory_entries=(alias, second_alias),
        manifest=replace(
            fragments[1].manifest,
            inventory_entry_ids=(alias.entry_id, second_alias.entry_id),
        ),
    )
    return records + (second_record,), (fragments[0], second_fragment)


def test_one_replica_native_owner_requires_one_relation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    expanded_records, expanded_fragments = _add_second_replica_alias(
        records,
        fragments,
        native_owner=records[1].source_native_owner_id or "unreachable",
        replica_group_id="replicas.mtp.conflicting",
        evidence_name="conflicting-native-relation",
    )

    with pytest.raises(ValueError, match="replica native owner.*conflicting"):
        _build_replica_fixture(
            monkeypatch,
            graph_inputs,
            expanded_records,
            expanded_fragments,
        )


def test_one_replica_group_requires_one_boundary_evidence_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    first_edge = fragments[1].classification_edges[0]
    assert isinstance(first_edge, SynchronizedReplicaAliasClassificationEdge)
    expanded_records, expanded_fragments = _add_second_replica_alias(
        records,
        fragments,
        native_owner="mtp.0.model.experts.up.replica",
        replica_group_id=first_edge.synchronization.replica_group_id,
        evidence_name="conflicting-group-evidence",
    )

    with pytest.raises(ValueError, match="replica group.*conflicting"):
        _build_replica_fixture(
            monkeypatch,
            graph_inputs,
            expanded_records,
            expanded_fragments,
        )


def test_one_replica_native_owner_can_cover_multiple_alias_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    first_edge = fragments[1].classification_edges[0]
    assert isinstance(first_edge, SynchronizedReplicaAliasClassificationEdge)
    expanded_records, expanded_fragments = _add_second_replica_alias(
        records,
        fragments,
        native_owner=records[1].source_native_owner_id or "unreachable",
        replica_group_id=first_edge.synchronization.replica_group_id,
        evidence_name="mtp-0-replica-synchronization",
    )

    bundle = _build_replica_fixture(
        monkeypatch,
        graph_inputs,
        expanded_records,
        expanded_fragments,
    )

    assert len(bundle.source_alias_contracts) == 2
    assert all(
        isinstance(contract, SynchronizedReplicaSourceAliasContract)
        for contract in bundle.source_alias_contracts
    )
    assert len(bundle.inventory.owners) == 1


def _canonical_edge_for_component(
    record: SourceDiscoveryRecord,
    entry: ParameterInventoryEntry,
    component_role: ComponentRole,
) -> CanonicalValueClassificationEdge:
    edge = _whole_routed_edge(record, entry)
    return replace(
        edge,
        component_role=component_role,
        axis_mappings=tuple(
            replace(
                mapping,
                target=ComponentAxisTarget(
                    component_role,
                    mapping.target.component_axis,
                ),
            )
            if isinstance(mapping.target, ComponentAxisTarget)
            else mapping
            for mapping in edge.axis_mappings
        ),
    )


def test_replica_bundle_normalizes_every_multicomponent_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_replica_fixture()
    values_role = ComponentRole("values")
    scales_role = ComponentRole("scales")
    source_format = FormatDescriptor(
        "test.replica-components.v1",
        "test.replica-components",
        (
            ComponentDescriptor(values_role, "e4m3"),
            ComponentDescriptor(scales_role, "e8m0"),
        ),
    )
    direct = replace(
        fragments[0].inventory_entries[0],
        member=replace(fragments[0].inventory_entries[0].member, format=source_format),
    )
    alias = replace(
        fragments[1].inventory_entries[0],
        member=replace(fragments[1].inventory_entries[0].member, format=source_format),
    )
    canonical_records = tuple(
        replace(
            records[0],
            record_id=f"main.experts.gate.{role}",
            source_native_name=f"main.model.experts.gate.{role}",
            dtype=dtype,
        )
        for role, dtype in (
            (values_role, CanonicalSourceDType.E4M3),
            (scales_role, CanonicalSourceDType.E8M0),
        )
    )
    replica_records = tuple(
        replace(
            records[1],
            record_id=f"mtp.0.replica.gate.{role}",
            source_native_name=f"mtp.0.model.experts.gate.replica_{role}",
            dtype=dtype,
        )
        for role, dtype in (
            (values_role, CanonicalSourceDType.E4M3),
            (scales_role, CanonicalSourceDType.E8M0),
        )
    )
    canonical_edges = tuple(
        _canonical_edge_for_component(record, direct, role)
        for record, role in zip(
            canonical_records,
            (values_role, scales_role),
            strict=True,
        )
    )
    first_replica_edge = fragments[1].classification_edges[0]
    assert isinstance(
        first_replica_edge,
        SynchronizedReplicaAliasClassificationEdge,
    )
    replica_edges = tuple(
        replace(
            first_replica_edge,
            record_id=replica_record.record_id,
            alias_output=replace(
                first_replica_edge.alias_output,
                inventory_entry_id=alias.entry_id,
            ),
            canonical_record_id=canonical_record.record_id,
            component_role=role,
        )
        for replica_record, canonical_record, role in zip(
            replica_records,
            canonical_records,
            (values_role, scales_role),
            strict=True,
        )
    )
    main_fragment = replace(
        fragments[0],
        classification_edges=canonical_edges,
        inventory_entries=(direct,),
    )
    replica_fragment = replace(
        fragments[1],
        classification_edges=replica_edges,
        inventory_entries=(alias,),
    )

    bundle = _build_replica_fixture(
        monkeypatch,
        graph_inputs,
        canonical_records + replica_records,
        (main_fragment, replica_fragment),
    )

    assert tuple(
        contract.component_role for contract in bundle.source_alias_contracts
    ) == (scales_role, values_role)
    assert all(
        isinstance(contract, SynchronizedReplicaSourceAliasContract)
        for contract in bundle.source_alias_contracts
    )
    assert len(bundle.inventory.owners) == 1


def _layer_slice_target(
    entry: ParameterInventoryEntry,
    layer_index: int,
) -> OutputMemberTarget:
    domain = entry.member.domain
    assert domain.layer_domain is not None
    return OutputMemberTarget(
        entry.entry_id,
        FamilyIndexDomain(
            LayerDomain((domain.layer_domain.members[layer_index],)),
            domain.independent_axes,
        ),
        (),
    )


def _layer_slice_region(shape: tuple[int, ...], layer_index: int) -> SourceRegion:
    return SourceRegion(
        shape,
        (
            SourceAxisSelection(0, (SourceIndexSpan(layer_index, layer_index + 1),)),
            *(
                SourceAxisSelection(index, (SourceIndexSpan(0, extent),))
                for index, extent in enumerate(shape[1:], start=1)
            ),
        ),
    )


def _slice_canonical_edge(
    edge: CanonicalValueClassificationEdge,
    entry: ParameterInventoryEntry,
    layer_index: int,
) -> CanonicalValueClassificationEdge:
    layer_span = SourceIndexSpan(layer_index, layer_index + 1)
    return replace(
        edge,
        source_region=_layer_slice_region(edge.source_region.source_shape, layer_index),
        output=_layer_slice_target(entry, layer_index),
        axis_mappings=tuple(
            replace(
                mapping,
                segments=(SourceOrdinalMapSegment(layer_span, 0),),
            )
            if isinstance(mapping.target, LayerCoordinateTarget)
            else mapping
            for mapping in edge.axis_mappings
        ),
    )


def test_bundle_matches_cross_graph_alias_edge_subdomains_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_alias_fixture()
    direct_entry = fragments[0].inventory_entries[0]
    canonical_edge = fragments[0].classification_edges[0]
    assert isinstance(canonical_edge, CanonicalValueClassificationEdge)
    split_main = replace(
        fragments[0],
        classification_edges=tuple(
            _slice_canonical_edge(canonical_edge, direct_entry, index)
            for index in range(2)
        ),
    )
    alias_entry = fragments[1].inventory_entries[0]
    tied_edge = fragments[1].classification_edges[0]
    assert isinstance(tied_edge, TiedAliasClassificationEdge)
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter("main-adapter", "main_family", {"main": split_main}),
        _BundleAdapter("mtp-adapter", "mtp_family", {"mtp.0": fragments[1]}),
    )

    _build_semantic_bundle(
        1,
        graph_inputs,
        records,
    )

    swapped_alias = replace(
        fragments[1],
        classification_edges=tuple(
            replace(
                tied_edge,
                aliased_source_region=_layer_slice_region(
                    tied_edge.aliased_source_region.source_shape,
                    source_layer,
                ),
                alias_output=_layer_slice_target(alias_entry, output_layer),
            )
            for source_layer, output_layer in ((0, 1), (1, 0))
        ),
    )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter("main-adapter", "main_family", {"main": split_main}),
        _BundleAdapter("mtp-adapter", "mtp_family", {"mtp.0": swapped_alias}),
    )

    with pytest.raises(ValueError, match="cross-graph tied.*subdomain"):
        _build_semantic_bundle(
            1,
            graph_inputs,
            records,
        )


def test_bundle_rejects_cross_graph_alias_with_wrong_native_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_alias_fixture()
    wrong_tied_record = replace(
        records[1],
        source_native_owner_id="different.native.owner",
    )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter("main-adapter", "main_family", {"main": fragments[0]}),
        _BundleAdapter("mtp-adapter", "mtp_family", {"mtp.0": fragments[1]}),
    )

    with pytest.raises(ValueError, match="cross-graph tied native owner"):
        _build_semantic_bundle(
            1,
            graph_inputs,
            (records[0], wrong_tied_record),
        )


def test_bundle_rejects_cross_graph_alias_with_fabricated_freeze_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_inputs, records, fragments = _cross_graph_alias_fixture()
    wrong_tied_record = replace(
        records[1],
        source_mutability=SourceMutability.FROZEN,
        mutability_evidence=_evidence("fabricated-cross-graph-freeze"),
    )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter("main-adapter", "main_family", {"main": fragments[0]}),
        _BundleAdapter("mtp-adapter", "mtp_family", {"mtp.0": fragments[1]}),
    )

    with pytest.raises(ValueError, match="cross-graph tied mutability evidence"):
        _build_semantic_bundle(
            1,
            graph_inputs,
            (records[0], wrong_tied_record),
        )


@pytest.mark.parametrize(
    ("conflict", "error"),
    [
        ("predicate", "conflicting predicates"),
        ("overlap", "overlapping expected domains"),
    ],
)
def test_bundle_rejects_namespaced_role_contribution_merge_conflicts(
    monkeypatch: pytest.MonkeyPatch,
    conflict: str,
    error: str,
) -> None:
    main_input, main_record, main_fragment = _direct_graph_fixture(
        graph_instance_id="main",
        graph_kind=GraphKind.MAIN,
        model_type="main_family",
        model_family="main-family",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    draft_a = _direct_graph_fixture(
        graph_instance_id="draft.a",
        graph_kind=GraphKind.SPECULATIVE_DRAFTER,
        model_type="draft_family",
        model_family="draft-family-a",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        namespaced_role="test.draft_expert",
    )
    draft_b = _direct_graph_fixture(
        graph_instance_id="draft.b",
        graph_kind=GraphKind.SPECULATIVE_DRAFTER,
        model_type="draft_family",
        model_family="draft-family-b",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        namespaced_role="test.draft_expert",
    )
    contribution = draft_b[2].role_contributions[0]
    if conflict == "predicate":
        contribution = replace(
            contribution,
            predicate=replace(contribution.predicate, module_kinds=("ffn.dense",)),
        )
        broken_b = replace(draft_b[2], role_contributions=(contribution,))
    else:
        broken_b = replace(
            draft_b[2],
            role_contributions=(contribution, contribution),
        )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter(
            "draft-adapter",
            "draft_family",
            {"draft.a": draft_a[2], "draft.b": broken_b},
        ),
        _BundleAdapter("main-adapter", "main_family", {"main": main_fragment}),
    )

    with pytest.raises(ValueError, match=error):
        _build_semantic_bundle(
            1,
            (main_input, draft_a[0], draft_b[0]),
            (main_record, draft_a[1], draft_b[1]),
        )


def test_bundle_rejects_mutated_builtin_predicate_and_domain_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_input, main_record, main_fragment = _direct_graph_fixture(
        graph_instance_id="main",
        graph_kind=GraphKind.MAIN,
        model_type="main_family",
        model_family="main-family",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    contribution = main_fragment.role_contributions[0]
    changed_predicate = replace(
        main_fragment,
        role_contributions=(
            replace(
                contribution,
                predicate=replace(contribution.predicate, module_kinds=("ffn.dense",)),
            ),
        ),
    )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter(
            "main-adapter",
            "main_family",
            {"main": changed_predicate},
        ),
    )
    with pytest.raises(ValueError, match="cannot replace a built-in role predicate"):
        _build_semantic_bundle(
            1,
            (main_input,),
            (main_record,),
        )

    changed_member = replace(
        main_fragment.inventory_entries[0],
        member=replace(
            main_fragment.inventory_entries[0].member,
            pattern=replace(
                main_fragment.inventory_entries[0].member.pattern,
                module_kind="ffn.dense",
            ),
        ),
    )
    mismatched_domain = replace(
        main_fragment,
        inventory_entries=(changed_member,),
    )
    _install_bundle_adapters(
        monkeypatch,
        _BundleAdapter(
            "main-adapter",
            "main_family",
            {"main": mismatched_domain},
        ),
    )
    with pytest.raises(ValueError, match="role moe.routed_expert expected domain"):
        _build_semantic_bundle(
            1,
            (main_input,),
            (main_record,),
        )
