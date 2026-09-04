from collections.abc import Mapping
from math import inf, nan
from pickle import dumps, loads
from dataclasses import FrozenInstanceError, dataclass, fields, replace

import pytest

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
    TiedAliasClassificationEdge,
    resolve_text_config,
    select_model_topology_adapter,
    validate_semantic_graph_build_fragment,
)
from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    LOGICAL_VALUES,
    AxisExtentRounding,
    AxisDomain,
    AxisProjection,
    ComponentDescriptor,
    ComponentRole,
    ExpectedGraphDeclaration,
    FamilyIndexDomain,
    FormatDescriptor,
    GraphKind,
    GraphLifecycle,
    GraphProvenance,
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
    SemanticOwnership,
    SemanticTensorFamily,
    SourceMutability,
    SourceOwnerInventoryEntry,
    ValueProvenance,
    builtin_role_definitions,
)


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
        )


def _evidence(name: str):
    from nemo_rl.precision_policy.semantic import EvidenceSource, EvidenceSourceKind

    return EvidenceSource(
        kind=EvidenceSourceKind.RUNTIME_INVENTORY,
        locator=f"runtime://{name}",
        digest=f"sha256:{name}",
    )


def _source_record(
    *,
    record_id: str = "main.experts.gate",
    graph_instance_id: str = "main",
    native_name: str | None = "model.layers.mlp.experts.gate_proj.weight",
    native_owner: str | None = "model.layers.mlp.experts.gate_proj",
    dtype: str = "bfloat16",
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
        record.dtype = "float32"  # type: ignore[misc]


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


@pytest.mark.parametrize("shape", [(2, 0), (2, -1)])
def test_source_discovery_record_rejects_invalid_raw_shape(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="positive"):
        _source_record(shape=shape)


def test_source_discovery_inventory_is_canonical_and_rejects_duplicates() -> None:
    later = _source_record(record_id="main.z", native_name="model.z.weight")
    earlier = _source_record(record_id="main.a", native_name="model.a.weight")

    inventory = SourceDiscoveryInventory((later, earlier))

    assert tuple(record.record_id for record in inventory.records) == (
        "main.a",
        "main.z",
    )
    with pytest.raises(ValueError, match="duplicate source discovery record"):
        SourceDiscoveryInventory((earlier, earlier))


def test_source_discovery_inventory_rejects_duplicate_present_native_name() -> None:
    first = _source_record(record_id="main.values")
    duplicate = replace(first, record_id="main.duplicate")

    with pytest.raises(ValueError, match="duplicate present source native name"):
        SourceDiscoveryInventory((first, duplicate))


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

    inventory = SourceDiscoveryInventory((scales, values))

    assert tuple(record.record_id for record in inventory.records) == (
        "main.scales",
        "main.values",
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
    definition = builtin_role_definitions(
        1,
        {"moe.routed_expert": RoleExpectedDomain("moe.routed_expert", (entry_id,))},
    )[0]
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
        dtype="float32",
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
        dtype="e8m0",
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
                "float16",
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
                "int64",
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
            dtype="int32",
            shape=(2, 4, 8, 8),
            provenance=SourceRecordProvenance.CHECKPOINT_STORAGE,
        ),
        replace(
            base_record,
            record_id="main.experts.gate.scales",
            source_native_name="model.layers.mlp.experts.gate_proj.weight_scale",
            dtype="float16",
            shape=(2, 4, 8, 2),
            provenance=SourceRecordProvenance.CHECKPOINT_STORAGE,
        ),
        replace(
            base_record,
            record_id="main.experts.gate.shape",
            source_native_name="model.layers.mlp.experts.gate_proj.weight_shape",
            dtype="int64",
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
    record = replace(records[0], dtype="float32")

    with pytest.raises(ValueError, match="raw dtype.*format component"):
        validate_semantic_graph_build_fragment(1, graph_input, (record,), fragment)


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
    record = replace(records[0], dtype="e4m3")
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
        value_provenance=ValueProvenance.TIED_ALIAS,
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
        if entry.value_provenance == ValueProvenance.TIED_ALIAS
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
            alias if entry.value_provenance == ValueProvenance.TIED_ALIAS else entry
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
        if entry.value_provenance == ValueProvenance.TIED_ALIAS
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
