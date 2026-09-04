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

from dataclasses import FrozenInstanceError, replace

import pytest

import nemo_rl.precision_policy.semantic as semantic_module
from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    BLOCK_SCALES,
    LOGICAL_VALUES,
    MXFP8_FORMAT,
    VALUES,
    AtomicGroup,
    AtomicGroupKind,
    AtomicGroupParticipant,
    AttributePredicate,
    AuxiliaryGraphDeclaration,
    AxisExtentRounding,
    AxisDomain,
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
    ImmutableAuxiliaryEvidence,
    IndexPathSegment,
    LayerDomain,
    LayerMember,
    LiteralComponentAxisSpec,
    LiteralPathSegment,
    LogicalComponentAxisSpec,
    OutOfScopeReason,
    OutOfScopeTensor,
    OwnerFamilyBinding,
    OwnerFamilyReference,
    ParameterInventory,
    ParameterInventoryEntry,
    RefitRequirement,
    RoleDefinition,
    RoleExpectedDomain,
    RolloutParticipation,
    SemanticAddress,
    SemanticAddressPattern,
    SemanticGraphManifest,
    SemanticManifestBundle,
    SemanticOwnership,
    SemanticPredicate,
    SemanticTensor,
    SemanticTensorFamily,
    SourceMutability,
    SourceOwnerInventoryEntry,
    ValueProvenance,
    builtin_role_definitions,
    resolve_component_axes,
)


def _evidence(name: str) -> EvidenceSource:
    return EvidenceSource(
        kind=EvidenceSourceKind.RUNTIME_INVENTORY,
        locator=f"runtime://{name}",
        digest=f"sha256:{name}",
    )


def _scalar_domain() -> FamilyIndexDomain:
    return FamilyIndexDomain(layer_domain=None, independent_axes=())


def _layer_expert_domain(
    layers: tuple[int, ...] = (0, 1),
    experts: tuple[int, ...] = (0, 1),
    *,
    moe_ordinals: tuple[int, ...] | None = None,
) -> FamilyIndexDomain:
    if moe_ordinals is None:
        members = tuple(LayerMember(layer, None) for layer in layers)
    else:
        members = tuple(
            LayerMember(layer, moe_ordinal)
            for layer, moe_ordinal in zip(layers, moe_ordinals, strict=True)
        )
    return FamilyIndexDomain(
        layer_domain=LayerDomain(members),
        independent_axes=(AxisDomain("expert", experts),),
    )


def _layer_only_domain(layers: tuple[int, ...]) -> FamilyIndexDomain:
    return FamilyIndexDomain(
        layer_domain=LayerDomain(tuple(LayerMember(layer, None) for layer in layers)),
        independent_axes=(),
    )


def _identity_projections(domain: FamilyIndexDomain) -> tuple[AxisProjection, ...]:
    return tuple(AxisProjection(name, name) for name in domain.axis_names)


def _direct_binding(
    entry_id: str,
    graph_instance_id: str,
    domain: FamilyIndexDomain,
    *,
    owner_family_id: str | None = None,
    owner_axes: tuple[str, ...] | None = None,
) -> OwnerFamilyBinding:
    owner_axis_names = domain.axis_names if owner_axes is None else owner_axes
    return OwnerFamilyBinding(
        canonical_owner_family=OwnerFamilyReference(
            graph_instance_id,
            owner_family_id or f"owner-{entry_id}",
        ),
        canonical_value_entry_id=entry_id,
        member_domain=domain,
        member_to_owner_axes=tuple(
            AxisProjection(axis_name, axis_name) for axis_name in owner_axis_names
        ),
        member_to_value_axes=_identity_projections(domain),
    )


def _address(
    semantic_id: str,
    *,
    semantic_graph_path: str = "text.decoder",
    model_part: str = "main",
    module_kind: str = "ffn.dense",
    attributes: tuple[tuple[str, str | int | float | bool], ...] = (),
    parameter_role: str = "kernel",
    global_decoder_layer: int | None = None,
    moe_ordinal: int | None = None,
) -> SemanticAddress:
    return SemanticAddress(
        semantic_id=semantic_id,
        semantic_graph_path=semantic_graph_path,
        model_part=model_part,
        module_kind=module_kind,
        attributes=attributes,
        parameter_role=parameter_role,
        global_decoder_layer=global_decoder_layer,
        moe_ordinal=moe_ordinal,
    )


def _tensor_entry(
    entry_id: str,
    graph_instance_id: str,
    semantic_id: str,
    *,
    semantic_graph_path: str = "text.decoder",
    model_part: str = "main",
    module_kind: str = "ffn.dense",
    attributes: tuple[tuple[str, str | int | float | bool], ...] = (),
    parameter_role: str = "kernel",
    global_decoder_layer: int | None = None,
    moe_ordinal: int | None = None,
    format: FormatDescriptor = BF16_FORMAT,
    logical_dtype: str = "bfloat16",
    logical_shape: tuple[int, ...] = (8, 8),
    logical_axes: tuple[str, ...] = ("output_features", "input_features"),
    binding: OwnerFamilyBinding | None = None,
    value_provenance: ValueProvenance = ValueProvenance.TRAINING_PARAMETER,
) -> ParameterInventoryEntry:
    domain = _scalar_domain()
    return ParameterInventoryEntry(
        entry_id=entry_id,
        graph_instance_id=graph_instance_id,
        member=SemanticTensor(
            address=_address(
                semantic_id,
                semantic_graph_path=semantic_graph_path,
                model_part=model_part,
                module_kind=module_kind,
                attributes=attributes,
                parameter_role=parameter_role,
                global_decoder_layer=global_decoder_layer,
                moe_ordinal=moe_ordinal,
            ),
            format=format,
            logical_dtype=logical_dtype,
            logical_shape=logical_shape,
            logical_axes=logical_axes,
            ownership=SemanticOwnership(
                binding=binding or _direct_binding(entry_id, graph_instance_id, domain)
            ),
        ),
        value_provenance=value_provenance,
    )


def _family_entry(
    entry_id: str,
    graph_instance_id: str,
    projection: str,
    domain: FamilyIndexDomain,
    *,
    semantic_graph_path: str = "text.decoder",
    model_part: str = "main",
    module_kind: str = "moe.expert_ffn",
    expert_kind: str = "routed",
    format: FormatDescriptor = BF16_FORMAT,
    logical_dtype: str = "bfloat16",
    logical_shape: tuple[int, ...] = (8, 8),
    logical_axes: tuple[str, ...] = ("output_features", "input_features"),
    binding: OwnerFamilyBinding | None = None,
    value_provenance: ValueProvenance = ValueProvenance.TRAINING_PARAMETER,
) -> ParameterInventoryEntry:
    return ParameterInventoryEntry(
        entry_id=entry_id,
        graph_instance_id=graph_instance_id,
        member=SemanticTensorFamily(
            pattern=SemanticAddressPattern(
                semantic_graph_path=semantic_graph_path,
                path_segments=(
                    LiteralPathSegment("layer"),
                    IndexPathSegment("global_decoder_layer"),
                    LiteralPathSegment("expert"),
                    IndexPathSegment("expert"),
                    LiteralPathSegment(projection),
                ),
                model_part=model_part,
                module_kind=module_kind,
                attributes=(
                    ("expert_kind", expert_kind),
                    ("projection", projection),
                ),
                parameter_role="kernel",
            ),
            domain=domain,
            format=format,
            logical_dtype=logical_dtype,
            logical_shape=logical_shape,
            logical_axes=logical_axes,
            ownership=SemanticOwnership(
                binding=binding or _direct_binding(entry_id, graph_instance_id, domain)
            ),
        ),
        value_provenance=value_provenance,
    )


def _attention_family_entry(
    entry_id: str,
    projection: str,
    domain: FamilyIndexDomain,
) -> ParameterInventoryEntry:
    return ParameterInventoryEntry(
        entry_id=entry_id,
        graph_instance_id="main",
        member=SemanticTensorFamily(
            pattern=SemanticAddressPattern(
                semantic_graph_path="text.decoder",
                path_segments=(
                    LiteralPathSegment("layer"),
                    IndexPathSegment("global_decoder_layer"),
                    LiteralPathSegment("attention"),
                    LiteralPathSegment(projection),
                ),
                model_part="main",
                module_kind="attention.projection",
                attributes=(("projection", projection),),
                parameter_role="kernel",
            ),
            domain=domain,
            format=BF16_FORMAT,
            logical_dtype="bfloat16",
            logical_shape=(8, 8),
            logical_axes=("output_features", "input_features"),
            ownership=SemanticOwnership(_direct_binding(entry_id, "main", domain)),
        ),
        value_provenance=ValueProvenance.TRAINING_PARAMETER,
    )


def _pattern_family_entry(
    entry_id: str,
    graph_instance_id: str,
    semantic_graph_path: str,
    path_segments: tuple[LiteralPathSegment | IndexPathSegment, ...],
    domain: FamilyIndexDomain,
    *,
    binding: OwnerFamilyBinding | None = None,
    format: FormatDescriptor = BF16_FORMAT,
    logical_axes: tuple[str, ...] = ("output_features", "input_features"),
) -> ParameterInventoryEntry:
    return ParameterInventoryEntry(
        entry_id=entry_id,
        graph_instance_id=graph_instance_id,
        member=SemanticTensorFamily(
            pattern=SemanticAddressPattern(
                semantic_graph_path=semantic_graph_path,
                path_segments=path_segments,
                model_part="main",
                module_kind="ffn.dense",
                attributes=(),
                parameter_role="kernel",
            ),
            domain=domain,
            format=format,
            logical_dtype="bfloat16",
            logical_shape=(8, 8),
            logical_axes=logical_axes,
            ownership=SemanticOwnership(
                binding or _direct_binding(entry_id, graph_instance_id, domain)
            ),
        ),
        value_provenance=ValueProvenance.TRAINING_PARAMETER,
    )


def _owner(
    entry: ParameterInventoryEntry,
    source_mutability: SourceMutability = SourceMutability.MUTABLE,
    *,
    domain: FamilyIndexDomain | None = None,
) -> SourceOwnerInventoryEntry:
    return SourceOwnerInventoryEntry(
        owner_family=entry.member.ownership.binding.canonical_owner_family,
        domain=domain or entry.member.ownership.binding.member_domain,
        source_mutability=source_mutability,
        mutability_evidence_source=_evidence(entry.entry_id),
    )


def _runtime_lifecycle(
    graph_kind: GraphKind,
    rollout_participation: RolloutParticipation,
) -> GraphLifecycle:
    return GraphLifecycle(
        graph_kind=graph_kind,
        graph_provenance=GraphProvenance.TRAINING_RUNTIME,
        rollout_participation=rollout_participation,
    )


def _static_evidence(
    graph_instance_id: str,
    model_identity: str,
    revision: str,
) -> ImmutableAuxiliaryEvidence:
    return ImmutableAuxiliaryEvidence(
        graph_instance_id=graph_instance_id,
        model_identity=model_identity,
        pinned_checkpoint_revision=revision,
        checkpoint_content_digest=f"sha256:{graph_instance_id}-checkpoint",
        model_config_digest=f"sha256:{graph_instance_id}-config",
        semantic_domain_digest=f"sha256:{graph_instance_id}-domain",
        evidence_source=EvidenceSource(
            kind=EvidenceSourceKind.PINNED_CHECKPOINT_MANIFEST,
            locator=f"checkpoint://{graph_instance_id}/{revision}",
            digest=f"sha256:{graph_instance_id}-manifest",
        ),
    )


def _static_lifecycle(
    graph_instance_id: str,
    graph_kind: GraphKind,
    model_identity: str,
    revision: str,
    *,
    provenance: GraphProvenance = GraphProvenance.EXTERNAL_CHECKPOINT,
) -> GraphLifecycle:
    return GraphLifecycle(
        graph_kind=graph_kind,
        graph_provenance=provenance,
        rollout_participation=RolloutParticipation.SERVED_FROM_CHECKPOINT,
        immutable_evidence=_static_evidence(
            graph_instance_id,
            model_identity,
            revision,
        ),
    )


def _bundle(
    entries: tuple[ParameterInventoryEntry, ...],
    owners: tuple[SourceOwnerInventoryEntry, ...],
    *,
    lifecycles: dict[str, GraphLifecycle] | None = None,
    model_identities: dict[str, str] | None = None,
    manifests: tuple[SemanticGraphManifest, ...] | None = None,
    expected_graphs: tuple[ExpectedGraphDeclaration, ...] | None = None,
    role_definitions: tuple[RoleDefinition, ...] = (),
    atomic_groups: dict[str, tuple[AtomicGroup, ...]] | None = None,
    out_of_scope: dict[str, tuple[OutOfScopeTensor, ...]] | None = None,
) -> SemanticManifestBundle:
    if lifecycles is None:
        lifecycles = {
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.SERVED_FROM_SOURCE,
            )
        }
    graph_ids = list(lifecycles)
    for entry in entries:
        if entry.graph_instance_id not in graph_ids:
            graph_ids.append(entry.graph_instance_id)
    model_identities = model_identities or {}
    atomic_groups = atomic_groups or {}
    out_of_scope = out_of_scope or {}

    if manifests is None:
        manifests = tuple(
            SemanticGraphManifest(
                model_family=model_identities.get(graph_id, f"{graph_id}-model"),
                model_revision=(
                    lifecycles[graph_id].immutable_evidence.pinned_checkpoint_revision
                    if lifecycles[graph_id].immutable_evidence is not None
                    else f"{graph_id}-revision"
                ),
                graph_instance_id=graph_id,
                lifecycle=lifecycles[graph_id],
                inventory_entry_ids=tuple(
                    entry.entry_id
                    for entry in entries
                    if entry.graph_instance_id == graph_id
                ),
                atomic_groups=atomic_groups.get(graph_id, ()),
                out_of_scope=out_of_scope.get(graph_id, ()),
            )
            for graph_id in graph_ids
        )
    if expected_graphs is None:
        expected_graphs = tuple(
            ExpectedGraphDeclaration(
                graph_instance_id=graph_id,
                model_identity=model_identities.get(graph_id, f"{graph_id}-model"),
                lifecycle=lifecycles[graph_id],
            )
            for graph_id in graph_ids
        )
    inventory = ParameterInventory(owners=owners, entries=entries)
    provisional = SemanticManifestBundle(
        schema_version=1,
        expected_graphs=expected_graphs,
        manifests=manifests,
        inventory=inventory,
        role_definitions=role_definitions,
    )
    existing_role_names = {role.role_name for role in role_definitions}
    central_templates = builtin_role_definitions(1, {})
    inferred_central = builtin_role_definitions(
        1,
        {
            role.role_name: RoleExpectedDomain(
                role.role_name,
                role.matching_inventory_entry_ids(provisional),
            )
            for role in central_templates
        },
    )
    return replace(
        provisional,
        role_definitions=role_definitions
        + tuple(
            role
            for role in inferred_central
            if role.role_name not in existing_role_names
        ),
    )


def test_bf16_descriptor_is_one_logical_bfloat16_component() -> None:
    assert BF16_FORMAT.format_id == "bf16.logical.v1"
    assert tuple(
        (component.role, component.dtype) for component in BF16_FORMAT.components
    ) == (("logical_values", "bfloat16"),)

    assert resolve_component_axes(
        BF16_FORMAT.components[0],
        logical_axes=("output_features", "input_features"),
        logical_shape=(64, 128),
    ) == (("output_features", 64), ("input_features", 128))


def test_explicit_component_axes_cover_packed_scaled_metadata_and_scalar_shapes() -> (
    None
):
    packed = ComponentDescriptor(
        ComponentRole("packed_values"),
        "uint8",
        component_axes=(
            LogicalComponentAxisSpec("output_features"),
            LogicalComponentAxisSpec(
                "input_features",
                divisor=8,
                rounding=AxisExtentRounding.EXACT,
            ),
        ),
    )
    scales = ComponentDescriptor(
        ComponentRole("block_scales"),
        "e8m0",
        component_axes=(
            LogicalComponentAxisSpec("output_features"),
            LogicalComponentAxisSpec(
                "input_features",
                divisor=32,
                rounding=AxisExtentRounding.CEIL,
            ),
        ),
    )
    shape_metadata = ComponentDescriptor(
        ComponentRole("weight_shape"),
        "int64",
        component_axes=(LiteralComponentAxisSpec("shape_field", 2),),
    )
    scalar = ComponentDescriptor(
        ComponentRole("weight_scale_2"),
        "float32",
        component_axes=(),
    )

    logical_axes = ("output_features", "input_features")
    logical_shape = (96, 130)
    assert resolve_component_axes(
        packed,
        logical_axes=logical_axes,
        logical_shape=(96, 128),
    ) == (("output_features", 96), ("input_features", 16))
    assert resolve_component_axes(
        scales,
        logical_axes=logical_axes,
        logical_shape=logical_shape,
    ) == (("output_features", 96), ("input_features", 5))
    assert resolve_component_axes(
        shape_metadata,
        logical_axes=logical_axes,
        logical_shape=logical_shape,
    ) == (("shape_field", 2),)
    assert (
        resolve_component_axes(
            scalar,
            logical_axes=logical_axes,
            logical_shape=logical_shape,
        )
        == ()
    )


def test_component_axis_order_is_preserved() -> None:
    descriptor = ComponentDescriptor(
        ComponentRole("transposed_metadata"),
        "int64",
        component_axes=(
            LiteralComponentAxisSpec("columns", 3),
            LiteralComponentAxisSpec("rows", 2),
        ),
    )

    assert resolve_component_axes(
        descriptor,
        logical_axes=("output_features", "input_features"),
        logical_shape=(8, 8),
    ) == (("columns", 3), ("rows", 2))


def test_component_axis_resolver_rejects_missing_or_nondivisible_logical_axis() -> None:
    missing = ComponentDescriptor(
        ComponentRole("missing"),
        "uint8",
        component_axes=(LogicalComponentAxisSpec("channels"),),
    )
    exact = ComponentDescriptor(
        ComponentRole("packed"),
        "uint8",
        component_axes=(LogicalComponentAxisSpec("input_features", divisor=8),),
    )

    with pytest.raises(ValueError, match="component logical axis.*member logical axes"):
        resolve_component_axes(
            missing,
            logical_axes=("output_features", "input_features"),
            logical_shape=(8, 8),
        )
    with pytest.raises(ValueError, match="exactly divisible"):
        resolve_component_axes(
            exact,
            logical_axes=("output_features", "input_features"),
            logical_shape=(8, 10),
        )


@pytest.mark.parametrize(
    "component_axes",
    [
        (LogicalComponentAxisSpec("input_features"),) * 2,
        (
            LogicalComponentAxisSpec("input_features"),
            LiteralComponentAxisSpec("input_features", 2),
        ),
    ],
)
def test_component_descriptor_rejects_duplicate_component_axis_names(
    component_axes: tuple[LogicalComponentAxisSpec | LiteralComponentAxisSpec, ...],
) -> None:
    with pytest.raises(ValueError, match="component axis names must be unique"):
        ComponentDescriptor(
            ComponentRole("duplicate_axes"),
            "uint8",
            component_axes=component_axes,
        )


@pytest.mark.parametrize(
    ("instance", "field_name"),
    [
        (LogicalComponentAxisSpec("input_features"), "logical_axis"),
        (LiteralComponentAxisSpec("shape_field", 2), "axis_name"),
    ],
)
def test_component_axis_spec_records_are_frozen_and_slotted(
    instance: LogicalComponentAxisSpec | LiteralComponentAxisSpec,
    field_name: str,
) -> None:
    assert not hasattr(instance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field_name, "mutated")


def test_mxfp8_descriptor_has_ordered_values_and_block_scales() -> None:
    assert MXFP8_FORMAT.format_id == ("mxfp8.e4m3-e8m0-block32-input-features.v1")
    assert tuple(
        (
            component.role,
            component.dtype,
            component.encoding,
            component.component_axes,
        )
        for component in MXFP8_FORMAT.components
    ) == (
        (VALUES, "e4m3", None, None),
        (
            BLOCK_SCALES,
            "e8m0",
            "mxfp8_scale",
            (
                LogicalComponentAxisSpec("output_features"),
                LogicalComponentAxisSpec(
                    "input_features",
                    divisor=32,
                    rounding=AxisExtentRounding.CEIL,
                ),
            ),
        ),
    )
    assert resolve_component_axes(
        MXFP8_FORMAT.components[1],
        logical_axes=("output_features", "input_features"),
        logical_shape=(96, 130),
    ) == (("output_features", 96), ("input_features", 5))
    assert LOGICAL_VALUES == "logical_values"


def test_component_logical_axis_must_exist_in_member_logical_axes() -> None:
    with pytest.raises(ValueError, match="component logical axis.*logical axes"):
        _tensor_entry(
            "bad-mxfp8-axis",
            "main",
            "text.decoder.layer.0.ffn.kernel",
            format=MXFP8_FORMAT,
            logical_axes=("output_features", "channels"),
        )


def test_adapter_formats_remain_complete_and_distinct() -> None:
    formats = (
        FormatDescriptor(
            format_id="adapter.block_fp8.e4m3-scale-inverse.v2",
            family="adapter.block_fp8",
            components=(
                ComponentDescriptor(ComponentRole("values"), "e4m3"),
                ComponentDescriptor(
                    ComponentRole("scale_inverse"),
                    "float32",
                    encoding="inverse_scale",
                    component_axes=(
                        LogicalComponentAxisSpec("output_features"),
                        LogicalComponentAxisSpec(
                            "input_features",
                            divisor=128,
                            rounding=AxisExtentRounding.CEIL,
                        ),
                    ),
                ),
            ),
        ),
        FormatDescriptor(
            format_id="adapter.nvfp4.e2m1-scales.v3",
            family="adapter.nvfp4",
            components=(
                ComponentDescriptor(
                    ComponentRole("packed_values"),
                    "uint8",
                    component_axes=(
                        LogicalComponentAxisSpec("output_features"),
                        LogicalComponentAxisSpec("input_features", divisor=2),
                    ),
                ),
                ComponentDescriptor(
                    ComponentRole("block_scales"),
                    "float8_e4m3fn",
                    component_axes=(
                        LogicalComponentAxisSpec("output_features"),
                        LogicalComponentAxisSpec(
                            "input_features",
                            divisor=16,
                            rounding=AxisExtentRounding.CEIL,
                        ),
                    ),
                ),
                ComponentDescriptor(
                    ComponentRole("weight_scale_2"),
                    "float32",
                    component_axes=(),
                ),
            ),
        ),
        FormatDescriptor(
            format_id="adapter.mxfp4.e2m1-e8m0.v1",
            family="adapter.mxfp4",
            components=(
                ComponentDescriptor(
                    ComponentRole("packed_values"),
                    "uint8",
                    component_axes=(
                        LogicalComponentAxisSpec("output_features"),
                        LogicalComponentAxisSpec("input_features", divisor=2),
                    ),
                ),
                ComponentDescriptor(
                    ComponentRole("block_scales"),
                    "e8m0",
                    component_axes=(
                        LogicalComponentAxisSpec("output_features"),
                        LogicalComponentAxisSpec(
                            "input_features",
                            divisor=32,
                            rounding=AxisExtentRounding.CEIL,
                        ),
                    ),
                ),
            ),
        ),
    )

    assert tuple(item.format_id for item in formats) == (
        "adapter.block_fp8.e4m3-scale-inverse.v2",
        "adapter.nvfp4.e2m1-scales.v3",
        "adapter.mxfp4.e2m1-e8m0.v1",
    )
    assert tuple(item.family for item in formats) == (
        "adapter.block_fp8",
        "adapter.nvfp4",
        "adapter.mxfp4",
    )
    assert tuple(
        resolve_component_axes(
            component,
            logical_axes=("output_features", "input_features"),
            logical_shape=(8, 64),
        )
        for format_descriptor in formats
        for component in format_descriptor.components
    ) == (
        (("output_features", 8), ("input_features", 64)),
        (("output_features", 8), ("input_features", 1)),
        (("output_features", 8), ("input_features", 32)),
        (("output_features", 8), ("input_features", 4)),
        (),
        (("output_features", 8), ("input_features", 32)),
        (("output_features", 8), ("input_features", 2)),
    )
    entries = tuple(
        _tensor_entry(
            f"format-{index}",
            "main",
            f"text.decoder.format.{index}.kernel",
            format=format_descriptor,
        )
        for index, format_descriptor in enumerate(formats)
    )
    bundle = _bundle(entries, tuple(_owner(entry) for entry in entries))
    bundle.validate_complete()


def test_format_descriptor_rejects_empty_duplicate_or_ambiguous_identity() -> None:
    with pytest.raises(ValueError, match="component"):
        FormatDescriptor("adapter.empty.v1", "adapter.empty", ())
    with pytest.raises(ValueError, match="duplicate component role"):
        FormatDescriptor(
            "adapter.duplicate.v1",
            "adapter.duplicate",
            (
                ComponentDescriptor(ComponentRole("values"), "e4m3"),
                ComponentDescriptor(ComponentRole("values"), "float32"),
            ),
        )

    first_format = FormatDescriptor(
        "adapter.same-id.v1",
        "adapter.first",
        (ComponentDescriptor(ComponentRole("values"), "e4m3"),),
    )
    second_format = FormatDescriptor(
        "adapter.same-id.v1",
        "adapter.second",
        (ComponentDescriptor(ComponentRole("packed_values"), "uint8"),),
    )
    first = _tensor_entry(
        "first-format",
        "main",
        "text.decoder.format.first.kernel",
        format=first_format,
    )
    second = _tensor_entry(
        "second-format",
        "main",
        "text.decoder.format.second.kernel",
        format=second_format,
    )
    bundle = _bundle((first, second), (_owner(first), _owner(second)))
    with pytest.raises(ValueError, match="format_id"):
        bundle.validate_complete()


@pytest.mark.parametrize(
    "descriptor",
    [
        FormatDescriptor,
        ComponentDescriptor,
    ],
)
def test_descriptor_records_are_frozen_and_slotted(descriptor: type[object]) -> None:
    instance: object
    if descriptor is FormatDescriptor:
        instance = BF16_FORMAT
    else:
        instance = BF16_FORMAT.components[0]
    assert not hasattr(instance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(instance, "dtype", "float32")


def test_collection_fields_snapshot_mutable_inputs() -> None:
    component_axes = [LogicalComponentAxisSpec("output_features")]
    component = ComponentDescriptor(
        VALUES,
        "e4m3",
        component_axes=component_axes,  # type: ignore[arg-type]
    )
    component_axes.append(LogicalComponentAxisSpec("input_features"))
    assert component.component_axes == (LogicalComponentAxisSpec("output_features"),)

    components = [ComponentDescriptor(LOGICAL_VALUES, "bfloat16")]
    format_descriptor = FormatDescriptor(
        "adapter.snapshot.v1",
        "adapter.snapshot",
        components,  # type: ignore[arg-type]
    )
    components.append(ComponentDescriptor(VALUES, "e4m3"))
    assert format_descriptor.components == (
        ComponentDescriptor(LOGICAL_VALUES, "bfloat16"),
    )

    path_segments = [LiteralPathSegment("kernel")]
    pattern = SemanticAddressPattern(
        semantic_graph_path="text.decoder",
        path_segments=path_segments,  # type: ignore[arg-type]
        model_part="main",
        module_kind="ffn.dense",
        attributes=(),
        parameter_role="kernel",
    )
    path_segments.append(LiteralPathSegment("mutated"))
    assert pattern.path_segments == (LiteralPathSegment("kernel"),)

    logical_shape = [8, 8]
    logical_axes = ["output_features", "input_features"]
    tensor = SemanticTensor(
        address=_address("text.decoder.snapshot.kernel"),
        format=BF16_FORMAT,
        logical_dtype="bfloat16",
        logical_shape=logical_shape,  # type: ignore[arg-type]
        logical_axes=logical_axes,  # type: ignore[arg-type]
        ownership=SemanticOwnership(
            _direct_binding("snapshot", "main", _scalar_domain())
        ),
    )
    logical_shape[0] = 16
    logical_axes[0] = "channels"
    assert tensor.logical_shape == (8, 8)
    assert tensor.logical_axes == ("output_features", "input_features")

    family_shape = [8, 8]
    family_axes = ["output_features", "input_features"]
    family_domain = _layer_only_domain((0,))
    family = SemanticTensorFamily(
        pattern=SemanticAddressPattern(
            semantic_graph_path="text.decoder",
            path_segments=(IndexPathSegment("global_decoder_layer"),),
            model_part="main",
            module_kind="ffn.dense",
            attributes=(),
            parameter_role="kernel",
        ),
        domain=family_domain,
        format=BF16_FORMAT,
        logical_dtype="bfloat16",
        logical_shape=family_shape,  # type: ignore[arg-type]
        logical_axes=family_axes,  # type: ignore[arg-type]
        ownership=SemanticOwnership(
            _direct_binding("snapshot-family", "main", family_domain)
        ),
    )
    family_shape[0] = 16
    family_axes[0] = "channels"
    assert family.logical_shape == (8, 8)
    assert family.logical_axes == ("output_features", "input_features")


def test_graph_lifecycle_rejects_untyped_immutable_evidence_immediately() -> None:
    with pytest.raises(TypeError, match="immutable_evidence"):
        GraphLifecycle(
            graph_kind=GraphKind.MTP,
            graph_provenance=GraphProvenance.MODEL_CHECKPOINT,
            rollout_participation=RolloutParticipation.SERVED_FROM_CHECKPOINT,
            immutable_evidence=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("role", "dtype"),
    [
        (ComponentRole(""), "float32"),
        (ComponentRole("values"), ""),
    ],
)
def test_malformed_component_descriptor_is_rejected(
    role: ComponentRole,
    dtype: str,
) -> None:
    with pytest.raises(ValueError):
        ComponentDescriptor(role, dtype)


def test_component_descriptor_rejects_untyped_component_axis() -> None:
    with pytest.raises(TypeError, match="typed component-axis specs"):
        ComponentDescriptor(
            ComponentRole("values"),
            "e4m3",
            component_axes=(object(),),  # type: ignore[arg-type]
        )


def test_component_axis_extents_reject_bool_nonpositive_and_untyped_rounding() -> None:
    with pytest.raises(ValueError, match="divisor"):
        LogicalComponentAxisSpec("input_features", divisor=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="divisor"):
        LogicalComponentAxisSpec("input_features", divisor=0)
    with pytest.raises(TypeError, match="rounding"):
        LogicalComponentAxisSpec(
            "input_features",
            rounding="ceil",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="extent"):
        LiteralComponentAxisSpec("shape_field", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="extent"):
        LiteralComponentAxisSpec("shape_field", 0)


def test_builtin_roles_include_only_exact_main_text_components() -> None:
    main_routed = tuple(
        _tensor_entry(
            f"main-routed-{projection}",
            "main",
            f"text.decoder.layer.1.moe.routed.0.{projection}.kernel",
            module_kind="moe.expert_ffn",
            attributes=(("expert_kind", "routed"), ("projection", projection)),
            global_decoder_layer=1,
            moe_ordinal=0,
        )
        for projection in ("gate", "up", "down")
    )
    main_shared = _tensor_entry(
        "main-shared-gate",
        "main",
        "text.decoder.layer.1.moe.shared.gate.kernel",
        module_kind="moe.expert_ffn",
        attributes=(("expert_kind", "shared"), ("projection", "gate")),
        global_decoder_layer=1,
    )
    main_router = _tensor_entry(
        "main-router",
        "main",
        "text.decoder.layer.1.moe.router.kernel",
        module_kind="moe.router",
        global_decoder_layer=1,
    )
    main_bias = _tensor_entry(
        "main-routed-bias",
        "main",
        "text.decoder.layer.1.moe.routed.0.gate.bias",
        module_kind="moe.expert_ffn",
        attributes=(("expert_kind", "routed"), ("projection", "gate")),
        parameter_role="bias",
        global_decoder_layer=1,
        moe_ordinal=0,
    )
    draft_routed = _tensor_entry(
        "draft-routed-gate",
        "draft.0",
        "text.decoder.layer.1.moe.routed.0.gate.kernel",
        module_kind="moe.expert_ffn",
        attributes=(("expert_kind", "routed"), ("projection", "gate")),
        global_decoder_layer=1,
        moe_ordinal=0,
    )
    mtp_routed = _tensor_entry(
        "mtp-routed-gate",
        "mtp.0",
        "text.decoder.layer.1.moe.routed.0.gate.kernel",
        module_kind="moe.expert_ffn",
        attributes=(("expert_kind", "routed"), ("projection", "gate")),
        global_decoder_layer=1,
        moe_ordinal=0,
    )
    main_qkvo = tuple(
        _tensor_entry(
            f"main-{projection}",
            "main",
            f"text.decoder.layer.1.attention.{projection}.kernel",
            module_kind="attention.projection",
            attributes=(("projection", projection),),
            global_decoder_layer=1,
        )
        for projection in ("q", "k", "v", "o")
    )
    main_mla = _tensor_entry(
        "main-mla",
        "main",
        "text.decoder.layer.1.attention.mla.q_a.kernel",
        module_kind="attention.mla",
        attributes=(("projection", "q_a"),),
        global_decoder_layer=1,
    )
    ngram = _tensor_entry(
        "main-ngram",
        "main",
        "text.embedding.ngram.kernel",
        semantic_graph_path="text.embedding",
        module_kind="embedding.ngram",
    )
    entries = (
        main_bias,
        main_mla,
        main_router,
        main_shared,
        ngram,
        draft_routed,
        mtp_routed,
        *main_routed,
        *main_qkvo,
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "draft.0": _runtime_lifecycle(
            GraphKind.SPECULATIVE_DRAFTER,
            RolloutParticipation.NOT_SERVED,
        ),
        "mtp.0": _runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.NOT_SERVED,
        ),
    }
    role_definitions = builtin_role_definitions(
        1,
        {
            "moe.routed_expert": RoleExpectedDomain(
                "moe.routed_expert",
                ("main-routed-gate", "main-routed-up", "main-routed-down"),
            ),
            "attention.qkvo": RoleExpectedDomain(
                "attention.qkvo",
                ("main-q", "main-k", "main-v", "main-o"),
            ),
            "embedding.ngram": RoleExpectedDomain(
                "embedding.ngram",
                ("main-ngram",),
            ),
        },
    )
    bundle = _bundle(
        entries,
        tuple(_owner(entry) for entry in entries),
        lifecycles=lifecycles,
        role_definitions=role_definitions,
    )

    bundle.validate_complete()
    assert bundle.role_definition(1, "moe.routed_expert").matching_inventory_entry_ids(
        bundle
    ) == (
        "main-routed-down",
        "main-routed-gate",
        "main-routed-up",
    )
    assert bundle.role_definition(1, "attention.qkvo").matching_inventory_entry_ids(
        bundle
    ) == (
        "main-k",
        "main-o",
        "main-q",
        "main-v",
    )
    assert bundle.role_definition(1, "embedding.ngram").matching_inventory_entry_ids(
        bundle
    ) == ("main-ngram",)


def test_role_expected_domain_must_equal_compact_matches() -> None:
    routed = _tensor_entry(
        "main-routed-gate",
        "main",
        "text.decoder.layer.1.moe.routed.0.gate.kernel",
        module_kind="moe.expert_ffn",
        attributes=(("expert_kind", "routed"), ("projection", "gate")),
    )
    roles = builtin_role_definitions(
        1,
        {
            "moe.routed_expert": RoleExpectedDomain(
                "moe.routed_expert",
                ("missing-routed-up",),
            )
        },
    )
    bundle = _bundle((routed,), (_owner(routed),), role_definitions=roles)

    with pytest.raises(ValueError, match="expected domain"):
        bundle.validate_complete()


def test_routed_expert_expected_domain_accepts_non_gated_topology() -> None:
    entries = tuple(
        _family_entry(
            f"main-routed-{projection}",
            "main",
            projection,
            _layer_expert_domain(layers=(0,), experts=(0, 1)),
        )
        for projection in ("up", "down")
    )
    roles = builtin_role_definitions(
        1,
        {
            "moe.routed_expert": RoleExpectedDomain(
                "moe.routed_expert",
                tuple(entry.entry_id for entry in entries),
            )
        },
    )
    bundle = _bundle(
        entries,
        tuple(_owner(entry) for entry in entries),
        role_definitions=roles,
    )

    bundle.validate_complete()


def _adapter_ffn_role(schema_version: int = 1) -> RoleDefinition:
    return RoleDefinition(
        schema_version=schema_version,
        role_name="adapter.ffn",
        predicate=SemanticPredicate(
            graph_kinds=(GraphKind.MAIN,),
            semantic_graph_paths=("text.decoder",),
            model_parts=("main",),
            module_kinds=("ffn.dense",),
            attributes=(),
            parameter_roles=("kernel",),
        ),
        expected_domain=RoleExpectedDomain("adapter.ffn", ("main-kernel",)),
    )


def test_role_registry_rejects_wrong_schema() -> None:
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    bundle = _bundle(
        (entry,),
        (_owner(entry),),
        role_definitions=(_adapter_ffn_role(schema_version=2),),
    )

    with pytest.raises(ValueError, match="schema version"):
        bundle.validate_complete()


def test_role_registry_rejects_duplicate_keys() -> None:
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    role = _adapter_ffn_role()
    bundle = _bundle(
        (entry,),
        (_owner(entry),),
        role_definitions=(role, replace(role)),
    )

    with pytest.raises(ValueError, match="duplicate role definition"):
        bundle.validate_complete()


def test_builtin_role_definitions_emit_every_central_role_with_empty_defaults() -> None:
    with pytest.raises(ValueError, match="schema version"):
        builtin_role_definitions(2, {})
    with pytest.raises(ValueError, match="unknown built-in role"):
        builtin_role_definitions(
            1,
            {"adapter.unknown": RoleExpectedDomain("adapter.unknown", ("x",))},
        )

    definitions = builtin_role_definitions(1, {})

    assert tuple(definition.role_name for definition in definitions) == (
        "attention.qkvo",
        "embedding.ngram",
        "moe.routed_expert",
    )
    assert all(
        definition.expected_domain.inventory_entry_ids == ()
        for definition in definitions
    )


def test_role_registry_rejects_empty_namespaced_domain_and_missing_builtin() -> None:
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    empty_namespaced = replace(
        _adapter_ffn_role(),
        expected_domain=RoleExpectedDomain("adapter.ffn", ()),
    )
    with pytest.raises(ValueError, match="namespaced role.*non-empty"):
        _bundle(
            (entry,),
            (_owner(entry),),
            role_definitions=(empty_namespaced,),
        ).validate_complete()

    complete = _bundle((entry,), (_owner(entry),))
    missing_builtin = replace(
        complete,
        role_definitions=tuple(
            role
            for role in complete.role_definitions
            if role.role_name != "attention.qkvo"
        ),
    )
    with pytest.raises(ValueError, match="missing built-in role.*attention.qkvo"):
        missing_builtin.validate_complete()


def test_correlated_layer_members_and_independent_experts_stay_compact() -> None:
    domain = _layer_expert_domain(
        layers=(10, 12, 13),
        experts=(0, 1, 2, 3),
        moe_ordinals=(0, 1, 2),
    )

    assert domain.axis_names == (
        "global_decoder_layer",
        "moe_ordinal",
        "expert",
    )
    assert domain.cardinality == 12
    assert domain.layer_domain is not None
    assert domain.layer_domain.members == (
        LayerMember(10, 0),
        LayerMember(12, 1),
        LayerMember(13, 2),
    )


def test_empty_domain_is_generic_but_family_and_owner_must_be_nonempty() -> None:
    empty_domain = FamilyIndexDomain(
        layer_domain=LayerDomain((LayerMember(0, None),)),
        independent_axes=(AxisDomain("expert", ()),),
    )
    assert empty_domain.cardinality == 0

    with pytest.raises(ValueError, match="semantic family.*non-empty"):
        _family_entry(
            "empty-family",
            "main",
            "gate",
            empty_domain,
        )

    with pytest.raises(ValueError, match="owner domain.*non-empty"):
        SourceOwnerInventoryEntry(
            owner_family=OwnerFamilyReference("main", "empty-owner"),
            domain=empty_domain,
            source_mutability=SourceMutability.FROZEN,
            mutability_evidence_source=_evidence("empty-owner"),
        )


def test_kimi_expert_inventory_stays_compact_during_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = _layer_expert_domain(
        layers=tuple(range(60)),
        experts=tuple(range(384)),
        moe_ordinals=tuple(range(60)),
    )
    entries = tuple(
        _family_entry(f"main-{projection}", "main", projection, domain)
        for projection in ("gate", "up", "down")
    )
    role_definitions = builtin_role_definitions(
        1,
        {
            "moe.routed_expert": RoleExpectedDomain(
                "moe.routed_expert",
                ("main-gate", "main-up", "main-down"),
            )
        },
    )
    bundle = _bundle(
        entries,
        tuple(_owner(entry) for entry in entries),
        role_definitions=role_definitions,
    )

    def fail_materialization(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validation materialized a semantic family")

    monkeypatch.setattr(
        SemanticTensorFamily,
        "iter_semantic_ids",
        fail_materialization,
    )
    bundle.validate_complete()

    assert len(bundle.inventory.entries) == 3
    assert bundle.inventory.logical_cardinality == 60 * 384 * 3
    assert all(
        isinstance(entry.member, SemanticTensorFamily)
        for entry in bundle.inventory.entries
    )
    assert bundle.role_definition(1, "moe.routed_expert").matching_inventory_entry_ids(
        bundle
    ) == (
        "main-down",
        "main-gate",
        "main-up",
    )


def test_family_renderer_is_lazy_and_not_persisted() -> None:
    domain = _layer_expert_domain(layers=(2,), experts=(7, 9))
    entry = _family_entry("main-gate", "main", "gate", domain)
    assert isinstance(entry.member, SemanticTensorFamily)

    rendered = entry.member.iter_semantic_ids()

    assert iter(rendered) is rendered
    assert tuple(rendered) == (
        "text.decoder.layer.2.expert.7.gate",
        "text.decoder.layer.2.expert.9.gate",
    )
    assert not hasattr(entry.member, "tensors")


def test_fixed_projection_families_are_not_a_projection_axis() -> None:
    with pytest.raises(ValueError, match="projection"):
        AxisDomain("projection", ("gate", "up"))


def test_unknown_index_axis_and_free_form_template_are_rejected() -> None:
    domain = _layer_expert_domain(layers=(0,), experts=(0,))
    binding = _direct_binding("bad", "main", domain)

    with pytest.raises(ValueError, match="index axis"):
        SemanticTensorFamily(
            pattern=SemanticAddressPattern(
                semantic_graph_path="text.decoder",
                path_segments=(
                    LiteralPathSegment("layer"),
                    IndexPathSegment("missing"),
                ),
                model_part="main",
                module_kind="ffn.dense",
                attributes=(),
                parameter_role="kernel",
            ),
            domain=domain,
            format=BF16_FORMAT,
            logical_dtype="bfloat16",
            logical_shape=(8, 8),
            logical_axes=("output_features", "input_features"),
            ownership=SemanticOwnership(binding),
        )
    with pytest.raises(ValueError, match="canonical atom"):
        LiteralPathSegment("{expert}")


@pytest.mark.parametrize(
    ("graph_instance_id", "graph_kind"),
    [
        ("main.0", GraphKind.MAIN),
        ("mtp.", GraphKind.MTP),
        ("draft", GraphKind.SPECULATIVE_DRAFTER),
        ("mtp.bad value", GraphKind.MTP),
    ],
)
def test_graph_instance_grammar_is_kind_specific(
    graph_instance_id: str,
    graph_kind: GraphKind,
) -> None:
    lifecycle = _runtime_lifecycle(graph_kind, RolloutParticipation.NOT_SERVED)
    with pytest.raises(ValueError, match="graph_instance_id"):
        ExpectedGraphDeclaration(graph_instance_id, "fixture", lifecycle)


def test_semantic_id_must_be_descendant_of_semantic_graph_path() -> None:
    with pytest.raises(ValueError, match="semantic_id"):
        _address(
            "draft.decoder.layer.0.kernel",
            semantic_graph_path="text.decoder",
        )


def test_shape_axis_and_logical_axis_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="shape.*axes"):
        _tensor_entry(
            "bad-rank",
            "main",
            "text.decoder.bad_rank.kernel",
            logical_shape=(8,),
            logical_axes=("output_features", "input_features"),
        )
    with pytest.raises(ValueError, match="logical axis"):
        _tensor_entry(
            "bad-axis",
            "main",
            "text.decoder.bad_axis.kernel",
            logical_axes=("output_features", "banana"),
        )


def test_explicit_tensor_colliding_with_family_is_rejected_without_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = _layer_expert_domain(layers=(2,), experts=(7, 9))
    family = _family_entry("main-gate-family", "main", "gate", domain)
    explicit = _tensor_entry(
        "main-gate-explicit",
        "main",
        "text.decoder.layer.2.expert.7.gate",
        module_kind="moe.expert_ffn",
        attributes=(("expert_kind", "routed"), ("projection", "gate")),
        global_decoder_layer=2,
    )
    bundle = _bundle(
        (family, explicit),
        (_owner(family), _owner(explicit)),
    )

    def fail_materialization(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("collision validation expanded a semantic family")

    monkeypatch.setattr(
        SemanticTensorFamily,
        "iter_semantic_ids",
        fail_materialization,
    )

    with pytest.raises(ValueError, match="duplicate canonical semantic identity"):
        bundle.validate_complete()


def test_overlapping_families_are_rejected_without_rendering() -> None:
    left = _family_entry(
        "left-gate",
        "main",
        "gate",
        _layer_expert_domain(layers=(0, 1), experts=(0, 1)),
    )
    right = _family_entry(
        "right-gate",
        "main",
        "gate",
        _layer_expert_domain(layers=(1, 2), experts=(1, 2)),
    )
    bundle = _bundle((left, right), (_owner(left), _owner(right)))

    with pytest.raises(ValueError, match="duplicate canonical semantic identity"):
        bundle.validate_complete()


def test_family_overlap_uses_complete_rendered_id_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_left = _family_entry(
        "hidden-left",
        "main",
        "gate",
        _layer_expert_domain(
            layers=(1,),
            experts=(0,),
            moe_ordinals=(0,),
        ),
    )
    hidden_right = _family_entry(
        "hidden-right",
        "main",
        "gate",
        _layer_expert_domain(
            layers=(1,),
            experts=(0,),
            moe_ordinals=(9,),
        ),
    )

    rendered_int = _family_entry(
        "rendered-int",
        "main",
        "gate",
        _layer_expert_domain(layers=(2,), experts=(1,)),
    )
    rendered_string = _family_entry(
        "rendered-string",
        "main",
        "gate",
        FamilyIndexDomain(
            layer_domain=LayerDomain((LayerMember(2, None),)),
            independent_axes=(AxisDomain("expert", ("1",)),),
        ),
    )

    left_path = _pattern_family_entry(
        "nested-left",
        "main",
        "text.decoder",
        (
            LiteralPathSegment("layer"),
            IndexPathSegment("global_decoder_layer"),
            LiteralPathSegment("kernel"),
        ),
        _layer_only_domain((3,)),
    )
    right_path = _pattern_family_entry(
        "nested-right",
        "main",
        "text.decoder.layer",
        (
            IndexPathSegment("global_decoder_layer"),
            LiteralPathSegment("kernel"),
        ),
        _layer_only_domain((3,)),
    )

    def fail_materialization(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("overlap validation expanded a semantic family")

    monkeypatch.setattr(
        SemanticTensorFamily,
        "iter_semantic_ids",
        fail_materialization,
    )
    for entries in (
        (hidden_left, hidden_right),
        (rendered_int, rendered_string),
        (left_path, right_path),
    ):
        with pytest.raises(
            ValueError,
            match="duplicate canonical semantic identity",
        ):
            _bundle(
                entries,
                tuple(_owner(entry) for entry in entries),
            ).validate_complete()


def test_ragged_disjoint_families_form_a_complete_compact_union() -> None:
    first = _family_entry(
        "layers-0-1-gate",
        "main",
        "gate",
        _layer_expert_domain(layers=(0, 1), experts=(0, 1, 2)),
    )
    second = _family_entry(
        "layer-2-gate",
        "main",
        "gate",
        _layer_expert_domain(layers=(2,), experts=(0,)),
    )
    bundle = _bundle((second, first), (_owner(second), _owner(first)))

    bundle.validate_complete()

    assert bundle.inventory.logical_cardinality == 7
    assert tuple(entry.entry_id for entry in bundle.inventory.entries) == (
        "layer-2-gate",
        "layers-0-1-gate",
    )


def test_same_semantic_id_on_distinct_graph_instances_is_not_a_collision() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    draft = _tensor_entry(
        "draft-kernel",
        "draft.0",
        "text.decoder.layer.0.ffn.kernel",
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "draft.0": _runtime_lifecycle(
            GraphKind.SPECULATIVE_DRAFTER,
            RolloutParticipation.NOT_SERVED,
        ),
    }
    bundle = _bundle(
        (main, draft),
        (_owner(main), _owner(draft)),
        lifecycles=lifecycles,
    )

    bundle.validate_complete()

    assert bundle.inventory.logical_cardinality == 2


def test_fused_direct_values_can_share_one_canonical_owner() -> None:
    domain = _layer_expert_domain(layers=(0, 1), experts=(0, 1))
    owner_reference = OwnerFamilyReference("main", "fused-gate-up")
    gate_binding = replace(
        _direct_binding("gate", "main", domain),
        canonical_owner_family=owner_reference,
    )
    up_binding = replace(
        _direct_binding("up", "main", domain),
        canonical_owner_family=owner_reference,
    )
    gate = _family_entry("gate", "main", "gate", domain, binding=gate_binding)
    up = _family_entry("up", "main", "up", domain, binding=up_binding)
    owner = SourceOwnerInventoryEntry(
        owner_family=owner_reference,
        domain=domain,
        source_mutability=SourceMutability.MUTABLE,
        mutability_evidence_source=_evidence("fused-gate-up"),
    )
    bundle = _bundle((gate, up), (owner,))

    bundle.validate_complete()

    assert bundle.refit_requirement("main") == RefitRequirement.EVERY_VERSION
    assert bundle.owner_refit_requirements("main") == (
        (owner_reference, RefitRequirement.EVERY_VERSION),
    )


def test_one_canonical_owner_cannot_mix_direct_value_authorities() -> None:
    domain = _scalar_domain()
    training = _tensor_entry(
        "training-value",
        "main",
        "text.decoder.training.kernel",
    )
    owner = training.member.ownership.binding.canonical_owner_family
    checkpoint_binding = replace(
        _direct_binding("checkpoint-value", "main", domain),
        canonical_owner_family=owner,
    )
    checkpoint = _tensor_entry(
        "checkpoint-value",
        "main",
        "text.decoder.checkpoint.kernel",
        binding=checkpoint_binding,
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    bundle = _bundle((training, checkpoint), (_owner(training),))

    with pytest.raises(ValueError, match="canonical owner.*mixed value provenance"):
        bundle.validate_complete()


def test_packed_owner_projection_can_intentionally_omit_expert_axis() -> None:
    member_domain = _layer_expert_domain(layers=(0, 1), experts=(0, 1, 2))
    owner_domain = FamilyIndexDomain(
        layer_domain=LayerDomain((LayerMember(0, None), LayerMember(1, None))),
        independent_axes=(),
    )
    binding = _direct_binding(
        "packed-experts",
        "main",
        member_domain,
        owner_family_id="packed-experts",
        owner_axes=("global_decoder_layer",),
    )
    entry = _family_entry(
        "packed-experts",
        "main",
        "gate",
        member_domain,
        binding=binding,
    )
    owner = _owner(entry, domain=owner_domain)
    bundle = _bundle((entry,), (owner,))

    bundle.validate_complete()

    assert bundle.inventory.logical_cardinality == 6


def test_projection_rejects_marginally_equal_but_correlated_domain() -> None:
    member_domain = FamilyIndexDomain(
        layer_domain=LayerDomain(
            (
                LayerMember(0, 0),
                LayerMember(1, 1),
            )
        ),
        independent_axes=(),
    )
    owner_domain = FamilyIndexDomain(
        layer_domain=None,
        independent_axes=(
            AxisDomain("owner_layer", (0, 1)),
            AxisDomain("owner_ordinal", (0, 1)),
        ),
    )
    binding = OwnerFamilyBinding(
        canonical_owner_family=OwnerFamilyReference("main", "diagonal-owner"),
        canonical_value_entry_id="diagonal-family",
        member_domain=member_domain,
        member_to_owner_axes=(
            AxisProjection("global_decoder_layer", "owner_layer"),
            AxisProjection("moe_ordinal", "owner_ordinal"),
        ),
        member_to_value_axes=_identity_projections(member_domain),
    )
    entry = _pattern_family_entry(
        "diagonal-family",
        "main",
        "text.decoder",
        (
            LiteralPathSegment("layer"),
            IndexPathSegment("global_decoder_layer"),
            LiteralPathSegment("kernel"),
        ),
        member_domain,
        binding=binding,
    )
    bundle = _bundle(
        (entry,),
        (_owner(entry, domain=owner_domain),),
    )

    with pytest.raises(ValueError, match="projected domain"):
        bundle.validate_complete()


def test_projection_preserves_exact_relation_across_axis_renames() -> None:
    member_domain = FamilyIndexDomain(
        layer_domain=None,
        independent_axes=(
            AxisDomain("row", (0, 1)),
            AxisDomain("ordinal", (0, 1)),
        ),
    )
    owner_domain = FamilyIndexDomain(
        layer_domain=LayerDomain(
            (
                LayerMember(0, 0),
                LayerMember(0, 1),
                LayerMember(1, 0),
                LayerMember(1, 1),
            )
        ),
        independent_axes=(),
    )
    binding = OwnerFamilyBinding(
        canonical_owner_family=OwnerFamilyReference("main", "renamed-owner"),
        canonical_value_entry_id="renamed-family",
        member_domain=member_domain,
        member_to_owner_axes=(
            AxisProjection("row", "global_decoder_layer"),
            AxisProjection("ordinal", "moe_ordinal"),
        ),
        member_to_value_axes=_identity_projections(member_domain),
    )
    entry = _pattern_family_entry(
        "renamed-family",
        "main",
        "text.decoder",
        (
            LiteralPathSegment("matrix"),
            IndexPathSegment("row"),
            IndexPathSegment("ordinal"),
        ),
        member_domain,
        binding=binding,
    )

    _bundle(
        (entry,),
        (_owner(entry, domain=owner_domain),),
    ).validate_complete()

    correlated_domain = FamilyIndexDomain(
        layer_domain=LayerDomain(
            (
                LayerMember(0, 1),
                LayerMember(1, 2),
            )
        ),
        independent_axes=(),
    )
    swapped_owner_domain = FamilyIndexDomain(
        layer_domain=LayerDomain(
            (
                LayerMember(1, 0),
                LayerMember(2, 1),
            )
        ),
        independent_axes=(),
    )
    swapped_binding = OwnerFamilyBinding(
        canonical_owner_family=OwnerFamilyReference("main", "swapped-owner"),
        canonical_value_entry_id="swapped-family",
        member_domain=correlated_domain,
        member_to_owner_axes=(
            AxisProjection("global_decoder_layer", "moe_ordinal"),
            AxisProjection("moe_ordinal", "global_decoder_layer"),
        ),
        member_to_value_axes=_identity_projections(correlated_domain),
    )
    swapped_entry = _pattern_family_entry(
        "swapped-family",
        "main",
        "text.decoder",
        (
            LiteralPathSegment("layer"),
            IndexPathSegment("global_decoder_layer"),
            LiteralPathSegment("kernel"),
        ),
        correlated_domain,
        binding=swapped_binding,
    )

    _bundle(
        (swapped_entry,),
        (_owner(swapped_entry, domain=swapped_owner_domain),),
    ).validate_complete()


def test_large_projection_validation_never_expands_cartesian_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = FamilyIndexDomain(
        layer_domain=None,
        independent_axes=(
            AxisDomain("row", tuple(range(4096))),
            AxisDomain("column", tuple(range(4096))),
        ),
    )
    entry = _pattern_family_entry(
        "large-family",
        "main",
        "text.decoder",
        (
            LiteralPathSegment("matrix"),
            IndexPathSegment("row"),
            IndexPathSegment("column"),
        ),
        domain,
    )

    def fail_expansion(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("projection validation expanded a Cartesian domain")

    monkeypatch.setattr(semantic_module, "product", fail_expansion)
    monkeypatch.setattr(
        SemanticTensorFamily,
        "iter_semantic_ids",
        fail_expansion,
    )

    _bundle((entry,), (_owner(entry),)).validate_complete()

    assert domain.cardinality == 4096 * 4096


def test_duplicate_or_unreferenced_source_owner_is_rejected() -> None:
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    owner = _owner(entry)
    duplicate_bundle = _bundle((entry,), (owner, replace(owner)))
    with pytest.raises(ValueError, match="duplicate source owner"):
        duplicate_bundle.validate_complete()

    unused = replace(
        owner,
        owner_family=OwnerFamilyReference("main", "invented-owner"),
    )
    unused_bundle = _bundle((entry,), (owner, unused))
    with pytest.raises(ValueError, match="unreferenced source owner"):
        unused_bundle.validate_complete()


def test_alias_only_mtp_resolves_directly_to_main_owner() -> None:
    main_domain = _layer_expert_domain(layers=(0, 1), experts=(0, 1))
    main = _family_entry("main-head", "main", "gate", main_domain)
    alias_binding = OwnerFamilyBinding(
        canonical_owner_family=(main.member.ownership.binding.canonical_owner_family),
        canonical_value_entry_id="main-head",
        member_domain=main_domain,
        member_to_owner_axes=_identity_projections(main_domain),
        member_to_value_axes=_identity_projections(main_domain),
    )
    alias = _family_entry(
        "mtp-tied-head",
        "mtp.0",
        "gate",
        main_domain,
        semantic_graph_path="auxiliary.mtp",
        model_part="auxiliary",
        module_kind="auxiliary.mtp",
        binding=alias_binding,
        value_provenance=ValueProvenance.TIED_ALIAS,
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "mtp.0": _runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
    }
    bundle = _bundle(
        (alias, main),
        (_owner(main),),
        lifecycles=lifecycles,
    )

    bundle.validate_complete()

    assert bundle.refit_requirement("mtp.0") == RefitRequirement.EVERY_VERSION
    assert bundle.owner_refit_requirements("mtp.0") == (
        (
            main.member.ownership.binding.canonical_owner_family,
            RefitRequirement.EVERY_VERSION,
        ),
    )


def _alias_bundle_with_changes(
    *,
    alias_domain: FamilyIndexDomain | None = None,
    alias_shape: tuple[int, ...] = (8, 8),
    alias_axes: tuple[str, ...] = ("output_features", "input_features"),
    alias_dtype: str = "bfloat16",
    alias_format: FormatDescriptor = BF16_FORMAT,
    target_provenance: ValueProvenance = ValueProvenance.TRAINING_PARAMETER,
    target_entry_id: str = "main-head",
    canonical_owner_family: OwnerFamilyReference | None = None,
    owner_projections: tuple[AxisProjection, ...] | None = None,
    value_projections: tuple[AxisProjection, ...] | None = None,
    target_mutability: SourceMutability = SourceMutability.MUTABLE,
    main_participation: RolloutParticipation = RolloutParticipation.SERVED_FROM_SOURCE,
    alias_lifecycle: GraphLifecycle | None = None,
) -> SemanticManifestBundle:
    target_domain = _layer_expert_domain(layers=(0, 1), experts=(0, 1))
    main = _family_entry(
        "main-head",
        "main",
        "gate",
        target_domain,
        value_provenance=target_provenance,
    )
    alias_domain = alias_domain or target_domain
    binding = OwnerFamilyBinding(
        canonical_owner_family=(
            canonical_owner_family
            or main.member.ownership.binding.canonical_owner_family
        ),
        canonical_value_entry_id=target_entry_id,
        member_domain=alias_domain,
        member_to_owner_axes=(
            _identity_projections(alias_domain)
            if owner_projections is None
            else owner_projections
        ),
        member_to_value_axes=(
            _identity_projections(alias_domain)
            if value_projections is None
            else value_projections
        ),
    )
    alias = _family_entry(
        "mtp-tied-head",
        "mtp.0",
        "gate",
        alias_domain,
        semantic_graph_path="auxiliary.mtp",
        model_part="auxiliary",
        module_kind="auxiliary.mtp",
        logical_shape=alias_shape,
        logical_axes=alias_axes,
        logical_dtype=alias_dtype,
        format=alias_format,
        binding=binding,
        value_provenance=ValueProvenance.TIED_ALIAS,
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            main_participation,
        ),
        "mtp.0": (
            _runtime_lifecycle(
                GraphKind.MTP,
                RolloutParticipation.SERVED_FROM_SOURCE,
            )
            if alias_lifecycle is None
            else alias_lifecycle
        ),
    }
    owners = [_owner(main, target_mutability)]
    if (
        canonical_owner_family is not None
        and canonical_owner_family
        != main.member.ownership.binding.canonical_owner_family
    ):
        owners.append(
            SourceOwnerInventoryEntry(
                owner_family=canonical_owner_family,
                domain=target_domain,
                source_mutability=target_mutability,
                mutability_evidence_source=_evidence("alternate-owner"),
            )
        )
    return _bundle(
        (main, alias),
        tuple(owners),
        lifecycles=lifecycles,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"target_entry_id": "missing"}, "canonical value target"),
        (
            {"target_provenance": ValueProvenance.TIED_ALIAS},
            "alias-to-alias",
        ),
        (
            {
                "alias_domain": _layer_expert_domain(
                    layers=(4, 5),
                    experts=(0, 1),
                )
            },
            "projected domain",
        ),
        ({"alias_shape": (16, 8)}, "shape"),
        (
            {"alias_axes": ("input_features", "output_features")},
            "axes",
        ),
        ({"alias_dtype": "float32"}, "dtype"),
        ({"alias_format": MXFP8_FORMAT}, "format"),
        (
            {"canonical_owner_family": OwnerFamilyReference("main", "wrong-owner")},
            "canonical owner differs from its target",
        ),
    ],
)
def test_tied_alias_rejects_incompatible_direct_target(
    changes: dict[str, object],
    message: str,
) -> None:
    bundle = _alias_bundle_with_changes(**changes)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        bundle.validate_complete()


@pytest.mark.parametrize(
    "changes",
    [
        {
            "owner_projections": (
                AxisProjection(
                    "global_decoder_layer",
                    "global_decoder_layer",
                ),
            )
        },
        {
            "owner_projections": (
                AxisProjection("global_decoder_layer", "expert"),
                AxisProjection("expert", "expert"),
            )
        },
        {"owner_projections": (AxisProjection("missing", "expert"),)},
        {
            "value_projections": (
                AxisProjection(
                    "global_decoder_layer",
                    "global_decoder_layer",
                ),
            )
        },
        {
            "value_projections": (
                AxisProjection("global_decoder_layer", "expert"),
                AxisProjection("expert", "expert"),
            )
        },
        {"value_projections": (AxisProjection("missing", "expert"),)},
    ],
)
def test_tied_alias_rejects_incomplete_ambiguous_or_unknown_axis_projection(
    changes: dict[str, tuple[AxisProjection, ...]],
) -> None:
    bundle = _alias_bundle_with_changes(**changes)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="projection"):
        bundle.validate_complete()


def test_tied_alias_owner_and_value_mappings_must_commute() -> None:
    bundle = _alias_bundle_with_changes(
        value_projections=(
            AxisProjection("global_decoder_layer", "expert"),
            AxisProjection("expert", "global_decoder_layer"),
        )
    )

    with pytest.raises(ValueError, match="mapping.*commute"):
        bundle.validate_complete()


def test_non_alias_self_value_mapping_must_preserve_member_axes() -> None:
    domain = _layer_expert_domain(layers=(0, 1), experts=(0, 1))
    binding = replace(
        _direct_binding("main-gate", "main", domain),
        member_to_value_axes=(
            AxisProjection("global_decoder_layer", "expert"),
            AxisProjection("expert", "global_decoder_layer"),
        ),
    )
    entry = _family_entry(
        "main-gate",
        "main",
        "gate",
        domain,
        binding=binding,
    )

    with pytest.raises(ValueError, match="self-value.*identity"):
        _bundle((entry,), (_owner(entry),)).validate_complete()


def test_alias_only_graph_over_frozen_owner_derives_initial_only() -> None:
    bundle = _alias_bundle_with_changes(target_mutability=SourceMutability.FROZEN)

    bundle.validate_complete()

    assert bundle.refit_requirement("mtp.0") == RefitRequirement.INITIAL_ONLY


@pytest.mark.parametrize(
    ("mutability", "expected"),
    [
        (SourceMutability.MUTABLE, RefitRequirement.EVERY_VERSION),
        (SourceMutability.FROZEN, RefitRequirement.INITIAL_ONLY),
    ],
)
def test_checkpoint_served_cross_graph_training_alias_inherits_owner_cadence(
    mutability: SourceMutability,
    expected: RefitRequirement,
) -> None:
    bundle = _alias_bundle_with_changes(
        target_mutability=mutability,
        alias_lifecycle=_static_lifecycle(
            "mtp.0",
            GraphKind.MTP,
            "mtp.0-model",
            "resolved-mtp-012345",
        ),
    )
    owner = OwnerFamilyReference("main", "owner-main-head")

    bundle.validate_complete()

    assert bundle.owner_refit_requirements("mtp.0") == ((owner, expected),)
    assert bundle.refit_requirement("mtp.0") == expected


@pytest.mark.parametrize(
    "target_provenance",
    (
        ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
        ValueProvenance.BACKEND_DERIVED,
    ),
)
def test_checkpoint_served_alias_does_not_request_nontraining_authority(
    target_provenance: ValueProvenance,
) -> None:
    bundle = _alias_bundle_with_changes(
        target_provenance=target_provenance,
        target_mutability=SourceMutability.FROZEN,
        main_participation=RolloutParticipation.NOT_SERVED,
        alias_lifecycle=_static_lifecycle(
            "mtp.0",
            GraphKind.MTP,
            "mtp.0-model",
            "resolved-mtp-012345",
        ),
    )

    bundle.validate_complete()

    assert bundle.owner_refit_requirements("mtp.0") == (
        (OwnerFamilyReference("main", "owner-main-head"), RefitRequirement.NONE),
    )
    assert bundle.refit_requirement("mtp.0") == RefitRequirement.NONE


@pytest.mark.parametrize(
    "target_provenance",
    (
        ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
        ValueProvenance.BACKEND_DERIVED,
    ),
)
def test_source_served_alias_only_nontraining_authority_is_rejected(
    target_provenance: ValueProvenance,
) -> None:
    bundle = _alias_bundle_with_changes(
        target_provenance=target_provenance,
        target_mutability=SourceMutability.FROZEN,
        main_participation=RolloutParticipation.NOT_SERVED,
    )
    with pytest.raises(ValueError, match="must reach.*training parameter"):
        bundle.validate_complete()


@pytest.mark.parametrize(
    "nontraining_provenance",
    (
        ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
        ValueProvenance.BACKEND_DERIVED,
    ),
)
def test_mixed_source_alias_graph_only_refits_training_authority(
    nontraining_provenance: ValueProvenance,
) -> None:
    domain = _scalar_domain()
    training = _tensor_entry(
        "main-training",
        "main",
        "text.decoder.layer.0.training.kernel",
    )
    nontraining = _tensor_entry(
        "main-nontraining",
        "main",
        "text.decoder.layer.0.nontraining.kernel",
        value_provenance=nontraining_provenance,
    )

    def alias(
        target: ParameterInventoryEntry, entry_id: str
    ) -> ParameterInventoryEntry:
        binding = OwnerFamilyBinding(
            canonical_owner_family=(
                target.member.ownership.binding.canonical_owner_family
            ),
            canonical_value_entry_id=target.entry_id,
            member_domain=domain,
            member_to_owner_axes=(),
            member_to_value_axes=(),
        )
        return _tensor_entry(
            entry_id,
            "mtp.0",
            f"auxiliary.mtp.{entry_id}.kernel",
            semantic_graph_path="auxiliary.mtp",
            model_part="auxiliary",
            module_kind="auxiliary.mtp",
            binding=binding,
            value_provenance=ValueProvenance.TIED_ALIAS,
        )

    training_alias = alias(training, "training-alias")
    nontraining_alias = alias(nontraining, "nontraining-alias")
    training_owner = training.member.ownership.binding.canonical_owner_family
    nontraining_owner = nontraining.member.ownership.binding.canonical_owner_family
    bundle = _bundle(
        (training, nontraining, training_alias, nontraining_alias),
        (
            _owner(training, SourceMutability.MUTABLE),
            _owner(nontraining, SourceMutability.FROZEN),
        ),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.NOT_SERVED,
            ),
            "mtp.0": _runtime_lifecycle(
                GraphKind.MTP,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
        },
    )

    bundle.validate_complete()

    requirements: dict[OwnerFamilyReference, RefitRequirement] = {
        owner: requirement
        for owner, requirement in bundle.owner_refit_requirements("mtp.0")
    }
    assert requirements[training_owner] == RefitRequirement.EVERY_VERSION
    assert requirements[nontraining_owner] == RefitRequirement.NONE
    assert bundle.refit_requirement("mtp.0") == RefitRequirement.EVERY_VERSION


def test_not_served_alias_never_requests_its_canonical_source() -> None:
    bundle = _alias_bundle_with_changes(
        alias_lifecycle=_runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.NOT_SERVED,
        ),
    )

    bundle.validate_complete()

    assert bundle.owner_refit_requirements("mtp.0") == (
        (OwnerFamilyReference("main", "owner-main-head"), RefitRequirement.NONE),
    )
    assert bundle.refit_requirement("mtp.0") == RefitRequirement.NONE


def test_checkpoint_served_direct_body_stays_none_alongside_training_alias() -> None:
    lifecycle = _static_lifecycle(
        "mtp.0",
        GraphKind.MTP,
        "mtp.0-model",
        "resolved-mtp-012345",
    )
    alias_bundle = _alias_bundle_with_changes(alias_lifecycle=lifecycle)
    body = _tensor_entry(
        "mtp-checkpoint-body",
        "mtp.0",
        "auxiliary.mtp.body.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="auxiliary",
        module_kind="auxiliary.mtp",
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    bundle = _bundle(
        alias_bundle.inventory.entries + (body,),
        alias_bundle.inventory.owners + (_owner(body, SourceMutability.FROZEN),),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
            "mtp.0": lifecycle,
        },
    )

    bundle.validate_complete()

    assert bundle.owner_refit_requirements("mtp.0") == (
        (
            OwnerFamilyReference("main", "owner-main-head"),
            RefitRequirement.EVERY_VERSION,
        ),
        (
            OwnerFamilyReference("mtp.0", "owner-mtp-checkpoint-body"),
            RefitRequirement.NONE,
        ),
    )
    assert bundle.refit_requirement("mtp.0") == RefitRequirement.EVERY_VERSION


def test_non_alias_entry_must_name_itself_as_canonical_value() -> None:
    domain = _scalar_domain()
    binding = replace(
        _direct_binding("main-kernel", "main", domain),
        canonical_value_entry_id="other-entry",
    )
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
        binding=binding,
    )
    bundle = _bundle((entry,), (_owner(entry),))

    with pytest.raises(ValueError, match="must name itself"):
        bundle.validate_complete()


def test_mutable_training_only_mtp_is_complete_but_requires_no_refit() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    mtp = _tensor_entry(
        "mtp-head",
        "mtp.0",
        "auxiliary.mtp.head.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="auxiliary",
        module_kind="auxiliary.mtp",
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "mtp.0": _runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.NOT_SERVED,
        ),
    }
    bundle = _bundle(
        (mtp, main),
        (_owner(mtp), _owner(main)),
        lifecycles=lifecycles,
    )

    bundle.validate_complete()

    assert (
        bundle.inventory.owner_family("mtp.0", "owner-mtp-head").source_mutability
        == SourceMutability.MUTABLE
    )
    assert bundle.refit_requirement("mtp.0") == RefitRequirement.NONE
    assert bundle.owner_refit_requirements("mtp.0") == (
        (
            OwnerFamilyReference("mtp.0", "owner-mtp-head"),
            RefitRequirement.NONE,
        ),
    )


def test_independent_eagle_drafter_owner_refits_every_version() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    draft = _tensor_entry(
        "eagle-head",
        "draft.eagle",
        "draft.decoder.head.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        module_kind="draft.decoder",
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "draft.eagle": _runtime_lifecycle(
            GraphKind.SPECULATIVE_DRAFTER,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
    }
    bundle = _bundle(
        (main, draft),
        (_owner(main), _owner(draft)),
        lifecycles=lifecycles,
        model_identities={"main": "qwen", "draft.eagle": "eagle-other-family"},
    )

    bundle.validate_complete()

    assert bundle.refit_requirement("draft.eagle") == (RefitRequirement.EVERY_VERSION)
    assert bundle.owner_refit_requirements("draft.eagle") == (
        (
            OwnerFamilyReference("draft.eagle", "owner-eagle-head"),
            RefitRequirement.EVERY_VERSION,
        ),
    )


def test_mutable_different_family_training_only_drafter_requires_no_refit() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    draft = _tensor_entry(
        "draft-kernel",
        "draft.other",
        "draft.decoder.head.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        module_kind="draft.decoder",
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "draft.other": _runtime_lifecycle(
            GraphKind.SPECULATIVE_DRAFTER,
            RolloutParticipation.NOT_SERVED,
        ),
    }
    bundle = _bundle(
        (main, draft),
        (_owner(main), _owner(draft)),
        lifecycles=lifecycles,
        model_identities={
            "main": "qwen-main-family",
            "draft.other": "different-drafter-family",
        },
    )

    bundle.validate_complete()

    assert bundle.refit_requirement("draft.other") == RefitRequirement.NONE


@pytest.mark.parametrize(
    ("graph_instance_id", "graph_kind", "provenance", "participation", "evidence"),
    [
        (
            "mtp.0",
            GraphKind.MTP,
            GraphProvenance.TRAINING_RUNTIME,
            RolloutParticipation.NOT_SERVED,
            _static_evidence(
                "mtp.0",
                "mtp-model",
                "resolved-mtp-012345",
            ),
        ),
        (
            "mtp.0",
            GraphKind.MTP,
            GraphProvenance.TRAINING_RUNTIME,
            RolloutParticipation.SERVED_FROM_CHECKPOINT,
            _static_evidence(
                "mtp.0",
                "mtp-model",
                "resolved-mtp-012345",
            ),
        ),
        (
            "mtp.0",
            GraphKind.MTP,
            GraphProvenance.MODEL_CHECKPOINT,
            RolloutParticipation.SERVED_FROM_SOURCE,
            None,
        ),
        (
            "draft.external",
            GraphKind.SPECULATIVE_DRAFTER,
            GraphProvenance.EXTERNAL_CHECKPOINT,
            RolloutParticipation.NOT_SERVED,
            None,
        ),
    ],
)
def test_inconsistent_lifecycle_combinations_are_rejected(
    graph_instance_id: str,
    graph_kind: GraphKind,
    provenance: GraphProvenance,
    participation: RolloutParticipation,
    evidence: ImmutableAuxiliaryEvidence | None,
) -> None:
    with pytest.raises(ValueError, match="lifecycle"):
        lifecycle = GraphLifecycle(
            graph_kind=graph_kind,
            graph_provenance=provenance,
            rollout_participation=participation,
            immutable_evidence=evidence,
        )
        ExpectedGraphDeclaration(graph_instance_id, "mtp-model", lifecycle)


def test_mixed_mutable_and_frozen_owners_derive_distinct_cadences() -> None:
    mutable = _tensor_entry(
        "main-mutable",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    frozen = _tensor_entry(
        "main-frozen",
        "main",
        "text.decoder.layer.0.norm.kernel",
        module_kind="normalization",
        logical_shape=(8,),
        logical_axes=("output_features",),
    )
    bundle = _bundle(
        (frozen, mutable),
        (
            _owner(frozen, SourceMutability.FROZEN),
            _owner(mutable, SourceMutability.MUTABLE),
        ),
    )

    bundle.validate_complete()

    assert bundle.refit_requirement("main") == RefitRequirement.EVERY_VERSION
    assert bundle.owner_refit_requirements("main") == (
        (
            OwnerFamilyReference("main", "owner-main-frozen"),
            RefitRequirement.INITIAL_ONLY,
        ),
        (
            OwnerFamilyReference("main", "owner-main-mutable"),
            RefitRequirement.EVERY_VERSION,
        ),
    )


def test_all_frozen_nonempty_source_graph_derives_initial_only() -> None:
    frozen = _tensor_entry(
        "main-frozen",
        "main",
        "text.decoder.layer.0.norm.kernel",
        module_kind="normalization",
        logical_shape=(8,),
        logical_axes=("output_features",),
    )
    bundle = _bundle(
        (frozen,),
        (_owner(frozen, SourceMutability.FROZEN),),
    )

    bundle.validate_complete()

    assert bundle.refit_requirement("main") == RefitRequirement.INITIAL_ONLY


def test_empty_source_served_graph_never_vacuously_derives_initial_only() -> None:
    lifecycle = _runtime_lifecycle(
        GraphKind.MAIN,
        RolloutParticipation.SERVED_FROM_SOURCE,
    )
    bundle = _bundle((), (), lifecycles={"main": lifecycle})

    with pytest.raises(ValueError, match="non-empty semantic domain"):
        bundle.validate_complete()
    with pytest.raises(ValueError, match="non-empty semantic domain"):
        bundle.refit_requirement("main")


def test_source_served_graph_rejects_missing_or_absent_canonical_owner() -> None:
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    missing = _bundle((entry,), ())
    with pytest.raises(ValueError, match="canonical source owner"):
        missing.validate_complete()

    absent = _bundle((entry,), (_owner(entry, SourceMutability.ABSENT),))
    with pytest.raises(ValueError, match="absent"):
        absent.validate_complete()


def test_source_served_graph_requires_in_scope_training_authority() -> None:
    entry = _tensor_entry(
        "main-frozen",
        "main",
        "text.decoder.layer.0.frozen.kernel",
    )
    bundle = _bundle(
        (entry,),
        (_owner(entry, SourceMutability.FROZEN),),
        out_of_scope={
            "main": (
                OutOfScopeTensor(
                    "main-frozen",
                    OutOfScopeReason.SOURCE_PROVEN_FROZEN,
                ),
            )
        },
    )

    with pytest.raises(ValueError, match="must reach.*in-scope training parameter"):
        bundle.validate_complete()


@pytest.mark.parametrize(
    "value_provenance",
    (
        ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
        ValueProvenance.BACKEND_DERIVED,
    ),
)
def test_source_served_direct_only_nontraining_authority_is_rejected(
    value_provenance: ValueProvenance,
) -> None:
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
        value_provenance=value_provenance,
    )
    bundle = _bundle(
        (entry,),
        (_owner(entry, SourceMutability.FROZEN),),
    )

    with pytest.raises(ValueError, match="must reach.*training parameter"):
        bundle.validate_complete()


def test_checkpoint_served_direct_training_parameter_is_rejected() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    draft = _tensor_entry(
        "draft-kernel",
        "draft.external",
        "draft.decoder.head.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        module_kind="draft.decoder",
        value_provenance=ValueProvenance.TRAINING_PARAMETER,
    )
    lifecycle = _static_lifecycle(
        "draft.external",
        GraphKind.SPECULATIVE_DRAFTER,
        "draft-model",
        "resolved-draft-012345",
    )
    bundle = _bundle(
        (main, draft),
        (_owner(main), _owner(draft, SourceMutability.FROZEN)),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
            "draft.external": lifecycle,
        },
        model_identities={"draft.external": "draft-model"},
    )

    with pytest.raises(ValueError, match="checkpoint-served.*training parameter"):
        bundle.validate_complete()


def test_checkpoint_served_direct_backend_authority_requires_no_source_wire() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    draft = _tensor_entry(
        "draft-derived",
        "draft.external",
        "draft.decoder.derived.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        module_kind="draft.decoder",
        value_provenance=ValueProvenance.BACKEND_DERIVED,
    )
    owner = draft.member.ownership.binding.canonical_owner_family
    lifecycle = _static_lifecycle(
        "draft.external",
        GraphKind.SPECULATIVE_DRAFTER,
        "draft-model",
        "resolved-draft-012345",
    )
    bundle = _bundle(
        (main, draft),
        (_owner(main), _owner(draft, SourceMutability.FROZEN)),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
            "draft.external": lifecycle,
        },
        model_identities={"draft.external": "draft-model"},
    )

    bundle.validate_complete()

    assert bundle.owner_refit_requirements("draft.external") == (
        (owner, RefitRequirement.NONE),
    )
    assert bundle.refit_requirement("draft.external") == RefitRequirement.NONE


@pytest.mark.parametrize(
    ("graph_instance_id", "graph_kind", "provenance"),
    [
        ("mtp.static", GraphKind.MTP, GraphProvenance.MODEL_CHECKPOINT),
        (
            "draft.external",
            GraphKind.SPECULATIVE_DRAFTER,
            GraphProvenance.EXTERNAL_CHECKPOINT,
        ),
    ],
)
def test_static_auxiliary_evidence_is_complete_and_requires_no_refit(
    graph_instance_id: str,
    graph_kind: GraphKind,
    provenance: GraphProvenance,
) -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    model_identity = f"model-{graph_instance_id}"
    revision = f"resolved-{graph_instance_id}-012345"
    static = _tensor_entry(
        f"{graph_instance_id}-kernel",
        graph_instance_id,
        (
            "auxiliary.mtp.head.kernel"
            if graph_kind == GraphKind.MTP
            else "draft.decoder.head.kernel"
        ),
        semantic_graph_path=(
            "auxiliary.mtp" if graph_kind == GraphKind.MTP else "draft.decoder"
        ),
        model_part="auxiliary",
        module_kind=(
            "auxiliary.mtp" if graph_kind == GraphKind.MTP else "draft.decoder"
        ),
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    lifecycle = _static_lifecycle(
        graph_instance_id,
        graph_kind,
        model_identity,
        revision,
        provenance=provenance,
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        graph_instance_id: lifecycle,
    }
    bundle = _bundle(
        (main, static),
        (_owner(main), _owner(static, SourceMutability.FROZEN)),
        lifecycles=lifecycles,
        model_identities={graph_instance_id: model_identity},
    )

    bundle.validate_complete()

    assert bundle.refit_requirement(graph_instance_id) == RefitRequirement.NONE


def _static_draft_bundle_with_lifecycles(
    expected_lifecycle: GraphLifecycle,
    manifest_lifecycle: GraphLifecycle,
) -> SemanticManifestBundle:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    static = _tensor_entry(
        "draft-kernel",
        "draft.external",
        "draft.decoder.head.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        module_kind="draft.decoder",
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    main_lifecycle = _runtime_lifecycle(
        GraphKind.MAIN,
        RolloutParticipation.SERVED_FROM_SOURCE,
    )
    expected_graphs = (
        ExpectedGraphDeclaration("main", "main-model", main_lifecycle),
        ExpectedGraphDeclaration(
            "draft.external",
            "draft-model",
            expected_lifecycle,
        ),
    )
    manifests = (
        SemanticGraphManifest(
            "main-model",
            "main-revision",
            "main",
            main_lifecycle,
            ("main-kernel",),
        ),
        SemanticGraphManifest(
            "draft-model",
            "resolved-draft-012345",
            "draft.external",
            manifest_lifecycle,
            ("draft-kernel",),
        ),
    )
    return _bundle(
        (main, static),
        (_owner(main), _owner(static, SourceMutability.FROZEN)),
        lifecycles={
            "main": main_lifecycle,
            "draft.external": manifest_lifecycle,
        },
        manifests=manifests,
        expected_graphs=expected_graphs,
    )


def test_checkpoint_served_graph_rejects_missing_evidence() -> None:
    with pytest.raises(ValueError, match="immutable evidence"):
        GraphLifecycle(
            graph_kind=GraphKind.SPECULATIVE_DRAFTER,
            graph_provenance=GraphProvenance.EXTERNAL_CHECKPOINT,
            rollout_participation=RolloutParticipation.SERVED_FROM_CHECKPOINT,
        )


def _mutate_static_evidence(
    evidence: ImmutableAuxiliaryEvidence,
    field_name: str,
) -> ImmutableAuxiliaryEvidence:
    if field_name == "evidence_source.kind":
        return replace(
            evidence,
            evidence_source=replace(
                evidence.evidence_source,
                kind=EvidenceSourceKind.CONTENT_ADDRESS,
            ),
        )
    if field_name == "evidence_source.locator":
        return replace(
            evidence,
            evidence_source=replace(
                evidence.evidence_source,
                locator="checkpoint://different/location",
            ),
        )
    if field_name == "evidence_source.digest":
        return replace(
            evidence,
            evidence_source=replace(
                evidence.evidence_source,
                digest="sha256:different-evidence-source",
            ),
        )
    replacement_values = {
        "graph_instance_id": "draft.different",
        "model_identity": "different-model",
        "pinned_checkpoint_revision": "resolved-different-987654",
        "checkpoint_content_digest": "sha256:different-checkpoint",
        "model_config_digest": "sha256:different-config",
        "semantic_domain_digest": "sha256:different-domain",
    }
    return replace(evidence, **{field_name: replacement_values[field_name]})


@pytest.mark.parametrize(
    "field_name",
    [
        "graph_instance_id",
        "model_identity",
        "pinned_checkpoint_revision",
        "checkpoint_content_digest",
        "model_config_digest",
        "semantic_domain_digest",
        "evidence_source.kind",
        "evidence_source.locator",
        "evidence_source.digest",
    ],
)
def test_checkpoint_served_graph_rejects_each_evidence_mismatch(
    field_name: str,
) -> None:
    expected_lifecycle = _static_lifecycle(
        "draft.external",
        GraphKind.SPECULATIVE_DRAFTER,
        "draft-model",
        "resolved-draft-012345",
    )
    assert expected_lifecycle.immutable_evidence is not None
    manifest_lifecycle = replace(
        expected_lifecycle,
        immutable_evidence=_mutate_static_evidence(
            expected_lifecycle.immutable_evidence,
            field_name,
        ),
    )

    with pytest.raises(ValueError, match=field_name.replace(".", "\\.")):
        bundle = _static_draft_bundle_with_lifecycles(
            expected_lifecycle,
            manifest_lifecycle,
        )
        bundle.validate_complete()


@pytest.mark.parametrize(
    "field_name",
    [
        "model_identity",
        "pinned_checkpoint_revision",
        "checkpoint_content_digest",
        "model_config_digest",
        "semantic_domain_digest",
    ],
)
def test_immutable_evidence_rejects_each_blank_required_field(
    field_name: str,
) -> None:
    evidence = _static_evidence(
        "draft.external",
        "draft-model",
        "resolved-draft-012345",
    )
    with pytest.raises(ValueError, match=field_name.replace("_", " ")):
        replace(evidence, **{field_name: ""})


@pytest.mark.parametrize("field_name", ["locator", "digest"])
def test_evidence_source_rejects_each_blank_required_field(field_name: str) -> None:
    source = _evidence("source")
    with pytest.raises(ValueError, match=field_name):
        replace(source, **{field_name: ""})


def test_checkpoint_served_graph_cannot_claim_mutable_source_owner() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    draft = _tensor_entry(
        "draft-kernel",
        "draft.external",
        "draft.decoder.head.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        module_kind="draft.decoder",
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    lifecycle = _static_lifecycle(
        "draft.external",
        GraphKind.SPECULATIVE_DRAFTER,
        "draft-model",
        "resolved-draft-012345",
    )
    bundle = _bundle(
        (main, draft),
        (_owner(main), _owner(draft, SourceMutability.MUTABLE)),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
            "draft.external": lifecycle,
        },
        model_identities={"draft.external": "draft-model"},
    )

    with pytest.raises(ValueError, match="checkpoint-served.*mutable"):
        bundle.validate_complete()


def test_auxiliary_declaration_preserves_lifecycle_without_rank_state() -> None:
    lifecycle = _runtime_lifecycle(GraphKind.MTP, RolloutParticipation.NOT_SERVED)
    declaration = AuxiliaryGraphDeclaration(
        graph_instance_id="mtp.0",
        model_identity="qwen-mtp",
        lifecycle=lifecycle,
    )

    assert declaration.lifecycle == lifecycle
    assert not hasattr(declaration, "rank")
    assert not hasattr(declaration, "destination_layout")


def test_expected_graph_manifest_and_inventory_sets_must_be_bijective() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    mtp_lifecycle = _runtime_lifecycle(
        GraphKind.MTP,
        RolloutParticipation.NOT_SERVED,
    )
    main_lifecycle = _runtime_lifecycle(
        GraphKind.MAIN,
        RolloutParticipation.SERVED_FROM_SOURCE,
    )
    main_manifest = SemanticGraphManifest(
        model_family="main-model",
        model_revision="main-revision",
        graph_instance_id="main",
        lifecycle=main_lifecycle,
        inventory_entry_ids=("main-kernel",),
    )
    expected = (
        ExpectedGraphDeclaration("main", "main-model", main_lifecycle),
        ExpectedGraphDeclaration("mtp.0", "mtp-model", mtp_lifecycle),
    )
    bundle = _bundle(
        (main,),
        (_owner(main),),
        lifecycles={"main": main_lifecycle, "mtp.0": mtp_lifecycle},
        manifests=(main_manifest,),
        expected_graphs=expected,
    )

    with pytest.raises(ValueError, match="expected graph.*manifest"):
        bundle.validate_complete()


def test_extra_undeclared_manifest_is_rejected() -> None:
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    main_lifecycle = _runtime_lifecycle(
        GraphKind.MAIN,
        RolloutParticipation.SERVED_FROM_SOURCE,
    )
    mtp_lifecycle = _runtime_lifecycle(
        GraphKind.MTP,
        RolloutParticipation.NOT_SERVED,
    )
    manifests = (
        SemanticGraphManifest(
            "main-model",
            "main-revision",
            "main",
            main_lifecycle,
            ("main-kernel",),
        ),
        SemanticGraphManifest(
            "mtp-model",
            "mtp-revision",
            "mtp.0",
            mtp_lifecycle,
            (),
        ),
    )
    bundle = _bundle(
        (entry,),
        (_owner(entry),),
        lifecycles={"main": main_lifecycle},
        manifests=manifests,
        expected_graphs=(
            ExpectedGraphDeclaration("main", "main-model", main_lifecycle),
        ),
    )

    with pytest.raises(ValueError, match="match bijectively"):
        bundle.validate_complete()


def test_duplicate_expected_graph_and_manifest_ids_are_rejected() -> None:
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    lifecycle = _runtime_lifecycle(
        GraphKind.MAIN,
        RolloutParticipation.SERVED_FROM_SOURCE,
    )
    declaration = ExpectedGraphDeclaration("main", "main-model", lifecycle)
    manifest = SemanticGraphManifest(
        "main-model",
        "main-revision",
        "main",
        lifecycle,
        ("main-kernel",),
    )

    duplicate_expected = _bundle(
        (entry,),
        (_owner(entry),),
        manifests=(manifest,),
        expected_graphs=(declaration, replace(declaration)),
    )
    with pytest.raises(ValueError, match="duplicate expected graph"):
        duplicate_expected.validate_complete()

    duplicate_manifest = _bundle(
        (entry,),
        (_owner(entry),),
        manifests=(manifest, replace(manifest)),
        expected_graphs=(declaration,),
    )
    with pytest.raises(ValueError, match="duplicate semantic graph manifest"):
        duplicate_manifest.validate_complete()


def test_declaration_and_manifest_lifecycle_must_match_exactly() -> None:
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    expected_lifecycle = _runtime_lifecycle(
        GraphKind.MAIN,
        RolloutParticipation.SERVED_FROM_SOURCE,
    )
    manifest_lifecycle = _runtime_lifecycle(
        GraphKind.MAIN,
        RolloutParticipation.NOT_SERVED,
    )
    bundle = _bundle(
        (entry,),
        (_owner(entry),),
        manifests=(
            SemanticGraphManifest(
                "main-model",
                "main-revision",
                "main",
                manifest_lifecycle,
                ("main-kernel",),
            ),
        ),
        expected_graphs=(
            ExpectedGraphDeclaration(
                "main",
                "main-model",
                expected_lifecycle,
            ),
        ),
    )

    with pytest.raises(ValueError, match="lifecycle mismatch.*participation"):
        bundle.validate_complete()


def test_manifest_rejects_inventory_entry_from_another_graph() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    mtp = _tensor_entry(
        "mtp-head",
        "mtp.0",
        "auxiliary.mtp.head.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="auxiliary",
        module_kind="auxiliary.mtp",
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "mtp.0": _runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.NOT_SERVED,
        ),
    }
    manifests = (
        SemanticGraphManifest(
            "main-model",
            "main-revision",
            "main",
            lifecycles["main"],
            ("main-kernel", "mtp-head"),
        ),
        SemanticGraphManifest(
            "mtp-model",
            "mtp-revision",
            "mtp.0",
            lifecycles["mtp.0"],
            (),
        ),
    )
    bundle = _bundle(
        (main, mtp),
        (_owner(main), _owner(mtp)),
        lifecycles=lifecycles,
        manifests=manifests,
    )

    with pytest.raises(ValueError, match="foreign inventory entry"):
        bundle.validate_complete()


def test_manifest_must_account_for_every_whole_inventory_entry_exactly_once() -> None:
    first = _tensor_entry(
        "main-first",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    second = _tensor_entry(
        "main-second",
        "main",
        "text.decoder.layer.1.ffn.kernel",
    )
    lifecycle = _runtime_lifecycle(
        GraphKind.MAIN,
        RolloutParticipation.SERVED_FROM_SOURCE,
    )
    manifest = SemanticGraphManifest(
        model_family="main-model",
        model_revision="main-revision",
        graph_instance_id="main",
        lifecycle=lifecycle,
        inventory_entry_ids=("main-first",),
    )
    bundle = _bundle(
        (first, second),
        (_owner(first), _owner(second)),
        lifecycles={"main": lifecycle},
        manifests=(manifest,),
    )

    with pytest.raises(ValueError, match="inventory accounting"):
        bundle.validate_complete()


def test_duplicate_global_inventory_entry_id_is_rejected() -> None:
    first = _tensor_entry(
        "duplicate",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    draft = _tensor_entry(
        "duplicate",
        "draft.0",
        "draft.decoder.layer.0.ffn.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        module_kind="draft.decoder",
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "draft.0": _runtime_lifecycle(
            GraphKind.SPECULATIVE_DRAFTER,
            RolloutParticipation.NOT_SERVED,
        ),
    }
    bundle = _bundle(
        (first, draft),
        (_owner(first), _owner(draft)),
        lifecycles=lifecycles,
    )

    with pytest.raises(ValueError, match="duplicate inventory entry"):
        bundle.validate_complete()


def test_exactly_one_main_graph_is_required() -> None:
    mtp = _tensor_entry(
        "mtp-head",
        "mtp.0",
        "auxiliary.mtp.head.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="auxiliary",
        module_kind="auxiliary.mtp",
    )
    lifecycle = _runtime_lifecycle(GraphKind.MTP, RolloutParticipation.NOT_SERVED)
    no_main = _bundle(
        (mtp,),
        (_owner(mtp),),
        lifecycles={"mtp.0": lifecycle},
    )
    with pytest.raises(ValueError, match="exactly one MAIN"):
        no_main.validate_complete()


def test_mutable_main_tensor_cannot_hide_out_of_scope() -> None:
    entry = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    bundle = _bundle(
        (entry,),
        (_owner(entry),),
        out_of_scope={
            "main": (
                OutOfScopeTensor(
                    "main-kernel",
                    OutOfScopeReason.SOURCE_PROVEN_FROZEN,
                ),
            )
        },
    )

    with pytest.raises(ValueError, match="mutable main-model"):
        bundle.validate_complete()


def test_whole_frozen_entry_can_be_out_of_scope_with_typed_reason() -> None:
    active = _tensor_entry(
        "main-active",
        "main",
        "text.decoder.layer.0.dense.kernel",
    )
    frozen = _family_entry(
        "main-frozen-family",
        "main",
        "gate",
        _layer_expert_domain(layers=(0, 1), experts=(0, 1)),
    )
    bundle = _bundle(
        (active, frozen),
        (
            _owner(active, SourceMutability.MUTABLE),
            _owner(frozen, SourceMutability.FROZEN),
        ),
        out_of_scope={
            "main": (
                OutOfScopeTensor(
                    "main-frozen-family",
                    OutOfScopeReason.SOURCE_PROVEN_FROZEN,
                ),
            )
        },
    )

    bundle.validate_complete()

    manifest = bundle.manifest("main")
    assert bundle.owner_refit_requirements("main") == (
        (
            active.member.ownership.binding.canonical_owner_family,
            RefitRequirement.EVERY_VERSION,
        ),
        (
            frozen.member.ownership.binding.canonical_owner_family,
            RefitRequirement.NONE,
        ),
    )
    assert manifest.out_of_scope == (
        OutOfScopeTensor(
            "main-frozen-family",
            OutOfScopeReason.SOURCE_PROVEN_FROZEN,
        ),
    )
    with pytest.raises(TypeError):
        OutOfScopeTensor(
            inventory_entry_id="main-frozen-family",
            reason=OutOfScopeReason.SOURCE_PROVEN_FROZEN,
            domain=_scalar_domain(),  # type: ignore[call-arg]
        )


def test_checkpoint_auxiliary_can_be_out_of_scope_with_immutable_reason() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    draft = _tensor_entry(
        "draft-kernel",
        "draft.external",
        "draft.decoder.head.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        module_kind="draft.decoder",
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "draft.external": _static_lifecycle(
            "draft.external",
            GraphKind.SPECULATIVE_DRAFTER,
            "draft-model",
            "resolved-draft-revision",
        ),
    }
    bundle = _bundle(
        (main, draft),
        (_owner(main), _owner(draft, SourceMutability.FROZEN)),
        lifecycles=lifecycles,
        model_identities={"draft.external": "draft-model"},
        out_of_scope={
            "draft.external": (
                OutOfScopeTensor(
                    "draft-kernel",
                    OutOfScopeReason.IMMUTABLE_AUXILIARY,
                ),
            )
        },
    )

    bundle.validate_complete()


def test_runtime_auxiliary_cannot_use_immutable_out_of_scope_reason() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    mtp = _tensor_entry(
        "mtp-head",
        "mtp.0",
        "auxiliary.mtp.head.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="auxiliary",
        module_kind="auxiliary.mtp",
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "mtp.0": _runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.NOT_SERVED,
        ),
    }
    bundle = _bundle(
        (main, mtp),
        (_owner(main), _owner(mtp, SourceMutability.FROZEN)),
        lifecycles=lifecycles,
        out_of_scope={
            "mtp.0": (
                OutOfScopeTensor(
                    "mtp-head",
                    OutOfScopeReason.IMMUTABLE_AUXILIARY,
                ),
            )
        },
    )

    with pytest.raises(ValueError, match="checkpoint evidence"):
        bundle.validate_complete()


def test_backend_derived_entry_can_use_backend_derived_out_of_scope_reason() -> None:
    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    mtp = _tensor_entry(
        "mtp-derived-cache",
        "mtp.0",
        "auxiliary.mtp.derived.cache",
        semantic_graph_path="auxiliary.mtp",
        model_part="auxiliary",
        module_kind="auxiliary.mtp",
        value_provenance=ValueProvenance.BACKEND_DERIVED,
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "mtp.0": _runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.NOT_SERVED,
        ),
    }
    bundle = _bundle(
        (main, mtp),
        (_owner(main), _owner(mtp, SourceMutability.FROZEN)),
        lifecycles=lifecycles,
        out_of_scope={
            "mtp.0": (
                OutOfScopeTensor(
                    "mtp-derived-cache",
                    OutOfScopeReason.BACKEND_DERIVED_STATE,
                ),
            )
        },
    )

    bundle.validate_complete()


def test_out_of_scope_rejects_unknown_duplicate_untyped_or_false_reason() -> None:
    entry = _tensor_entry(
        "main-frozen",
        "main",
        "text.decoder.layer.0.norm.kernel",
        module_kind="normalization",
    )
    unknown = _bundle(
        (entry,),
        (_owner(entry, SourceMutability.FROZEN),),
        out_of_scope={
            "main": (
                OutOfScopeTensor(
                    "missing",
                    OutOfScopeReason.SOURCE_PROVEN_FROZEN,
                ),
            )
        },
    )
    with pytest.raises(ValueError, match="out-of-scope.*inventory entry"):
        unknown.validate_complete()

    claim = OutOfScopeTensor(
        "main-frozen",
        OutOfScopeReason.SOURCE_PROVEN_FROZEN,
    )
    duplicate = _bundle(
        (entry,),
        (_owner(entry, SourceMutability.FROZEN),),
        out_of_scope={"main": (claim, claim)},
    )
    with pytest.raises(ValueError, match="duplicate out-of-scope"):
        duplicate.validate_complete()

    with pytest.raises(TypeError, match="OutOfScopeReason"):
        OutOfScopeTensor("main-frozen", "frozen-ish")  # type: ignore[arg-type]

    main = _tensor_entry(
        "main-kernel",
        "main",
        "text.decoder.layer.0.ffn.kernel",
    )
    mtp = _tensor_entry(
        "mtp-mutable",
        "mtp.0",
        "auxiliary.mtp.head.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="auxiliary",
        module_kind="auxiliary.mtp",
    )
    false_reason = _bundle(
        (main, mtp),
        (_owner(main), _owner(mtp, SourceMutability.MUTABLE)),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
            "mtp.0": _runtime_lifecycle(
                GraphKind.MTP,
                RolloutParticipation.NOT_SERVED,
            ),
        },
        out_of_scope={
            "mtp.0": (
                OutOfScopeTensor(
                    "mtp-mutable",
                    OutOfScopeReason.SOURCE_PROVEN_FROZEN,
                ),
            )
        },
    )
    with pytest.raises(ValueError, match="source-proven frozen"):
        false_reason.validate_complete()


def test_backend_derived_out_of_scope_requires_backend_value_provenance() -> None:
    entry = _tensor_entry(
        "derived-cache",
        "main",
        "text.decoder.derived.cache",
        value_provenance=ValueProvenance.TRAINING_PARAMETER,
    )
    bundle = _bundle(
        (entry,),
        (_owner(entry, SourceMutability.FROZEN),),
        out_of_scope={
            "main": (
                OutOfScopeTensor(
                    "derived-cache",
                    OutOfScopeReason.BACKEND_DERIVED_STATE,
                ),
            )
        },
    )

    with pytest.raises(ValueError, match="backend-derived"):
        bundle.validate_complete()


def _pointwise_expert_group_fixture() -> tuple[
    tuple[ParameterInventoryEntry, ...], AtomicGroup
]:
    domain = _layer_expert_domain(
        layers=(0, 2),
        experts=(0, 1, 2),
        moe_ordinals=(0, 1),
    )
    entries = tuple(
        _family_entry(f"main-{projection}", "main", projection, domain)
        for projection in ("gate", "up", "down")
    )
    group = AtomicGroup(
        group_id="expert-gate-up-down",
        graph_instance_id="main",
        kind=AtomicGroupKind.PRECISION,
        group_domain=domain,
        participants=tuple(
            AtomicGroupParticipant(
                inventory_entry_id=entry.entry_id,
                participant_domain=domain,
                group_to_participant_axes=_identity_projections(domain),
            )
            for entry in entries
        ),
    )
    return entries, group


def test_atomic_group_expresses_pointwise_experts_without_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries, group = _pointwise_expert_group_fixture()
    bundle = _bundle(
        entries,
        tuple(_owner(entry) for entry in entries),
        atomic_groups={"main": (group,)},
    )

    def fail_materialization(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("atomic validation expanded a semantic family")

    monkeypatch.setattr(
        SemanticTensorFamily,
        "iter_semantic_ids",
        fail_materialization,
    )
    bundle.validate_complete()

    assert group.group_domain.cardinality == 6
    assert tuple(item.inventory_entry_id for item in group.participants) == (
        "main-down",
        "main-gate",
        "main-up",
    )


def test_atomic_group_can_address_a_complete_subdomain() -> None:
    full_domain = _layer_only_domain((0, 1))
    point_domain = _layer_only_domain((0,))
    entries = tuple(
        _attention_family_entry(f"main-{projection}", projection, full_domain)
        for projection in ("q", "k", "v", "o")
    )
    group = AtomicGroup(
        group_id="qkvo-layer-zero",
        graph_instance_id="main",
        kind=AtomicGroupKind.PRECISION,
        group_domain=point_domain,
        participants=tuple(
            AtomicGroupParticipant(
                entry.entry_id,
                point_domain,
                _identity_projections(point_domain),
            )
            for entry in entries
        ),
    )
    bundle = _bundle(
        entries,
        tuple(_owner(entry) for entry in entries),
        atomic_groups={"main": (group,)},
    )

    bundle.validate_complete()

    assert group.group_domain.cardinality == 1


def test_atomic_groups_reject_empty_group_or_participant_domain() -> None:
    entries, valid = _pointwise_expert_group_fixture()
    empty_domain = FamilyIndexDomain(
        layer_domain=LayerDomain(()),
        independent_axes=(),
    )
    empty = replace(valid, group_domain=empty_domain)
    empty_bundle = _bundle(
        entries,
        tuple(_owner(entry) for entry in entries),
        atomic_groups={"main": (empty,)},
    )
    with pytest.raises(ValueError, match="non-empty.*group domain"):
        empty_bundle.validate_complete()

    empty_participant = replace(
        valid,
        participants=(replace(valid.participants[0], participant_domain=empty_domain),),
    )
    participant_bundle = _bundle(
        entries,
        tuple(_owner(entry) for entry in entries),
        atomic_groups={"main": (empty_participant,)},
    )
    with pytest.raises(ValueError, match="participant.*non-empty"):
        participant_bundle.validate_complete()


def test_atomic_group_rejects_actual_cross_graph_participant() -> None:
    entries, valid = _pointwise_expert_group_fixture()
    draft = _family_entry(
        "draft-gate",
        "draft.0",
        "gate",
        valid.group_domain,
        semantic_graph_path="draft.decoder",
        model_part="draft",
        module_kind="draft.decoder",
    )
    cross = replace(
        valid,
        participants=(replace(valid.participants[0], inventory_entry_id="draft-gate"),),
    )
    cross_bundle = _bundle(
        (*entries, draft),
        (*tuple(_owner(entry) for entry in entries), _owner(draft)),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
            "draft.0": _runtime_lifecycle(
                GraphKind.SPECULATIVE_DRAFTER,
                RolloutParticipation.NOT_SERVED,
            ),
        },
        atomic_groups={"main": (cross,)},
    )
    with pytest.raises(ValueError, match="participant.*another graph"):
        cross_bundle.validate_complete()


def test_atomic_group_rejects_unknown_duplicate_or_out_of_family_participant() -> None:
    entries, valid = _pointwise_expert_group_fixture()
    owners = tuple(_owner(entry) for entry in entries)
    first = valid.participants[0]

    unknown = replace(
        valid,
        participants=(replace(first, inventory_entry_id="missing-entry"),),
    )
    with pytest.raises(ValueError, match="unknown inventory entry"):
        _bundle(
            entries,
            owners,
            atomic_groups={"main": (unknown,)},
        ).validate_complete()

    duplicate = replace(valid, participants=(first, first))
    with pytest.raises(ValueError, match="duplicate participants"):
        _bundle(
            entries,
            owners,
            atomic_groups={"main": (duplicate,)},
        ).validate_complete()

    outside_domain = _layer_expert_domain(
        layers=(99,),
        experts=(0, 1, 2),
        moe_ordinals=(99,),
    )
    outside = replace(
        valid,
        group_domain=outside_domain,
        participants=(
            replace(
                first,
                participant_domain=outside_domain,
                group_to_participant_axes=_identity_projections(outside_domain),
            ),
        ),
    )
    with pytest.raises(ValueError, match="outside its inventory family"):
        _bundle(
            entries,
            owners,
            atomic_groups={"main": (outside,)},
        ).validate_complete()


def test_atomic_group_rejects_duplicate_group_id() -> None:
    entries, valid = _pointwise_expert_group_fixture()
    bundle = _bundle(
        entries,
        tuple(_owner(entry) for entry in entries),
        atomic_groups={"main": (valid, replace(valid))},
    )

    with pytest.raises(ValueError, match="duplicate atomic group"):
        bundle.validate_complete()


@pytest.mark.parametrize(
    "projections",
    [
        (
            AxisProjection(
                "global_decoder_layer",
                "global_decoder_layer",
            ),
        ),
        (
            AxisProjection(
                "global_decoder_layer",
                "global_decoder_layer",
            ),
            AxisProjection("moe_ordinal", "expert"),
            AxisProjection("expert", "expert"),
        ),
        (
            AxisProjection("missing", "global_decoder_layer"),
            AxisProjection("moe_ordinal", "moe_ordinal"),
            AxisProjection("expert", "expert"),
        ),
        (
            AxisProjection("global_decoder_layer", "missing"),
            AxisProjection("moe_ordinal", "moe_ordinal"),
            AxisProjection("expert", "expert"),
        ),
        (
            AxisProjection("expert", "global_decoder_layer"),
            AxisProjection("expert", "moe_ordinal"),
            AxisProjection("global_decoder_layer", "expert"),
        ),
    ],
)
def test_atomic_group_rejects_incomplete_unknown_or_ambiguous_projection(
    projections: tuple[AxisProjection, ...],
) -> None:
    entries, valid = _pointwise_expert_group_fixture()
    invalid = replace(
        valid,
        participants=(
            replace(
                valid.participants[0],
                group_to_participant_axes=projections,
            ),
        ),
    )
    bundle = _bundle(
        entries,
        tuple(_owner(entry) for entry in entries),
        atomic_groups={"main": (invalid,)},
    )

    with pytest.raises(ValueError, match="projection"):
        bundle.validate_complete()


def test_manifest_local_validation_rejects_duplicate_accounting_handles() -> None:
    lifecycle = _runtime_lifecycle(
        GraphKind.MAIN,
        RolloutParticipation.SERVED_FROM_SOURCE,
    )
    manifest = SemanticGraphManifest(
        model_family="fixture",
        model_revision="revision",
        graph_instance_id="main",
        lifecycle=lifecycle,
        inventory_entry_ids=("same", "same"),
    )

    with pytest.raises(ValueError, match="duplicate inventory entry"):
        manifest.validate_complete()


def test_all_collection_records_have_deterministic_canonical_ordering() -> None:
    domain = FamilyIndexDomain(
        layer_domain=LayerDomain(
            (
                LayerMember(3, 2),
                LayerMember(1, 0),
                LayerMember(2, 1),
            )
        ),
        independent_axes=(AxisDomain("expert", (3, 1, 2)),),
    )
    up = _family_entry("z-up", "main", "up", domain)
    gate = _family_entry("a-gate", "main", "gate", domain)
    draft = _tensor_entry(
        "draft-head",
        "draft.z",
        "draft.decoder.head.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        module_kind="draft.decoder",
    )
    mtp = _tensor_entry(
        "mtp-head",
        "mtp.a",
        "auxiliary.mtp.head.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="auxiliary",
        module_kind="auxiliary.mtp",
    )
    lifecycles = {
        "draft.z": _runtime_lifecycle(
            GraphKind.SPECULATIVE_DRAFTER,
            RolloutParticipation.NOT_SERVED,
        ),
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "mtp.a": _runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.NOT_SERVED,
        ),
    }
    bundle = _bundle(
        (up, draft, gate, mtp),
        (_owner(up), _owner(draft), _owner(gate), _owner(mtp)),
        lifecycles=lifecycles,
    )

    bundle.validate_complete()

    assert tuple(item.graph_instance_id for item in bundle.expected_graphs) == (
        "main",
        "draft.z",
        "mtp.a",
    )
    assert tuple(item.graph_instance_id for item in bundle.manifests) == (
        "main",
        "draft.z",
        "mtp.a",
    )
    assert tuple(item.entry_id for item in bundle.inventory.entries) == (
        "a-gate",
        "z-up",
        "draft-head",
        "mtp-head",
    )
    assert domain.layer_domain is not None
    assert domain.layer_domain.members == (
        LayerMember(1, 0),
        LayerMember(2, 1),
        LayerMember(3, 2),
    )
    assert domain.independent_axes[0].members == (1, 2, 3)


def test_attributes_are_typed_unique_and_canonical() -> None:
    address = _address(
        "text.decoder.layer.0.ffn.kernel",
        attributes=(
            ("z_flag", True),
            ("expert_index", 0),
            ("alpha", 0.5),
        ),
    )
    assert address.attributes == (
        ("alpha", 0.5),
        ("expert_index", 0),
        ("z_flag", True),
    )

    with pytest.raises(ValueError, match="duplicate attribute"):
        _address(
            "text.decoder.layer.0.ffn.kernel",
            attributes=(("projection", "q"), ("projection", "k")),
        )
    with pytest.raises(TypeError, match="attribute"):
        _address(
            "text.decoder.layer.0.ffn.kernel",
            attributes=(("bad", object()),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_semantic_attributes_and_predicates_reject_non_finite_floats(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        _address(
            "text.decoder.layer.0.ffn.kernel",
            attributes=(("coefficient", value),),
        )
    with pytest.raises(ValueError, match="finite"):
        AttributePredicate("coefficient", (value,))


def test_predicate_scalar_identity_keeps_bool_distinct_from_int() -> None:
    predicate = AttributePredicate("flag_or_index", (False, 0, True, 1))

    assert predicate.allowed_values == (False, True, 0, 1)


def test_semantic_record_equality_and_hash_use_typed_scalar_identity() -> None:
    false_predicate = AttributePredicate("flag", (False,))
    zero_predicate = AttributePredicate("flag", (0,))
    assert false_predicate != zero_predicate
    assert len({false_predicate, zero_predicate}) == 2

    false_address = _address(
        "text.decoder.flag.kernel",
        attributes=(("flag", False),),
    )
    zero_address = _address(
        "text.decoder.flag.kernel",
        attributes=(("flag", 0),),
    )
    assert false_address != zero_address
    assert len({false_address, zero_address}) == 2

    false_pattern = SemanticAddressPattern(
        semantic_graph_path="text.decoder",
        path_segments=(LiteralPathSegment("flag"),),
        model_part="main",
        module_kind="ffn.dense",
        attributes=(("flag", False),),
        parameter_role="kernel",
    )
    zero_pattern = replace(false_pattern, attributes=(("flag", 0),))
    assert false_pattern != zero_pattern
    assert len({false_pattern, zero_pattern}) == 2

    false_role_predicate = SemanticPredicate(
        graph_kinds=(GraphKind.MAIN,),
        semantic_graph_paths=("text.decoder",),
        model_parts=("main",),
        module_kinds=("ffn.dense",),
        attributes=(false_predicate,),
        parameter_roles=("kernel",),
    )
    zero_role_predicate = replace(
        false_role_predicate,
        attributes=(zero_predicate,),
    )
    assert false_role_predicate != zero_role_predicate
    assert len({false_role_predicate, zero_role_predicate}) == 2


@pytest.mark.parametrize("value", [False, True])
def test_axis_domain_rejects_bool_members(value: bool) -> None:
    with pytest.raises(TypeError, match="bool"):
        AxisDomain("expert", (value,))  # type: ignore[arg-type]
