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

"""Contract tests for deterministic semantic precision compilation."""

import pickle
from collections.abc import Iterator
from copy import copy
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from math import copysign
from typing import cast

import pytest

import nemo_rl.precision_policy.compiler as compiler_module
from nemo_rl.precision_policy.compiler import (
    AtomicExpansion,
    CanonicalPrecisionPolicySnapshot,
    CompiledGraphPrecisionSelection,
    CompiledPrecisionIntentGroup,
    CompiledPrecisionSelectionGroup,
    CompiledSelectionScopeResult,
    PrecisionBoundaryFence,
    PrecisionPolicyError,
    compile_precision_policy,
    compile_precision_selection,
)
from nemo_rl.precision_policy.config import PrecisionPolicyConfig
from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    MXFP8_FORMAT,
    AtomicGroup,
    AtomicGroupKind,
    AtomicGroupParticipant,
    AttributePredicate,
    AxisDomain,
    AxisExtentRounding,
    AxisProjection,
    ComponentDescriptor,
    ComponentRole,
    DecoderLayerUniverse,
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
    OutOfScopeReason,
    OutOfScopeTensor,
    OwnerFamilyBinding,
    OwnerFamilyReference,
    ParameterInventory,
    ParameterInventoryEntry,
    PredicateScalar,
    RefitRequirement,
    ResolvedGraphTopology,
    ResolvedSelectionTopology,
    RoleDefinition,
    RoleExpectedDomain,
    RolloutParticipation,
    SelectionTopologyEntry,
    SemanticAddress,
    SemanticAddressPattern,
    SemanticGraphManifest,
    SemanticManifestBundle,
    SemanticOwnership,
    SemanticPredicate,
    SemanticTensor,
    SemanticTensorFamily,
    SourceAliasContract,
    SourceMutability,
    SourceOwnerInventoryEntry,
    SourceReplicaSynchronizationEvidence,
    SourceSynchronizationBoundary,
    SynchronizedReplicaSourceAliasContract,
    ValueProvenance,
    _compute_semantic_structure_digest,
    _merge_selection_role_definitions,
    _predicate_matches,
    builtin_role_definitions,
)


def _evidence(name: str) -> EvidenceSource:
    return EvidenceSource(
        kind=EvidenceSourceKind.RUNTIME_INVENTORY,
        locator=f"runtime://{name}",
        digest=f"sha256:{name}",
    )


def _scalar_domain() -> FamilyIndexDomain:
    return FamilyIndexDomain(layer_domain=None, independent_axes=())


def _layer_domain(
    layers: tuple[int, ...],
    *,
    moe_ordinals: tuple[int, ...] | None = None,
    experts: tuple[int, ...] | None = None,
) -> FamilyIndexDomain:
    if moe_ordinals is None:
        members = tuple(LayerMember(layer, None) for layer in layers)
    else:
        members = tuple(
            LayerMember(layer, ordinal)
            for layer, ordinal in zip(layers, moe_ordinals, strict=True)
        )
    axes = () if experts is None else (AxisDomain("expert", experts),)
    return FamilyIndexDomain(LayerDomain(members), axes)


def _identity_axes(domain: FamilyIndexDomain) -> tuple[AxisProjection, ...]:
    return tuple(AxisProjection(axis, axis) for axis in domain.axis_names)


def _binding(
    entry_id: str,
    graph_instance_id: str,
    domain: FamilyIndexDomain,
    *,
    owner_family: OwnerFamilyReference | None = None,
    canonical_value_entry_id: str | None = None,
    member_to_owner_axes: tuple[AxisProjection, ...] | None = None,
    member_to_value_axes: tuple[AxisProjection, ...] | None = None,
) -> OwnerFamilyBinding:
    return OwnerFamilyBinding(
        canonical_owner_family=owner_family
        or OwnerFamilyReference(graph_instance_id, f"owner-{entry_id}"),
        canonical_value_entry_id=canonical_value_entry_id or entry_id,
        member_domain=domain,
        member_to_owner_axes=(
            _identity_axes(domain)
            if member_to_owner_axes is None
            else member_to_owner_axes
        ),
        member_to_value_axes=(
            _identity_axes(domain)
            if member_to_value_axes is None
            else member_to_value_axes
        ),
    )


def _family_entry(
    entry_id: str,
    projection: str,
    domain: FamilyIndexDomain,
    *,
    graph_instance_id: str = "main",
    semantic_graph_path: str = "text.decoder",
    model_part: str = "main",
    module_kind: str = "moe.expert_ffn",
    expert_kind: str = "routed",
    binding: OwnerFamilyBinding | None = None,
    value_provenance: ValueProvenance = ValueProvenance.TRAINING_PARAMETER,
) -> ParameterInventoryEntry:
    segments: list[LiteralPathSegment | IndexPathSegment] = [
        LiteralPathSegment("layer"),
        IndexPathSegment("global_decoder_layer"),
    ]
    if "expert" in domain.axis_names:
        segments.extend((LiteralPathSegment("expert"), IndexPathSegment("expert")))
    segments.append(LiteralPathSegment(projection))
    return ParameterInventoryEntry(
        entry_id=entry_id,
        graph_instance_id=graph_instance_id,
        member=SemanticTensorFamily(
            pattern=SemanticAddressPattern(
                semantic_graph_path=semantic_graph_path,
                path_segments=tuple(segments),
                model_part=model_part,
                module_kind=module_kind,
                attributes=(
                    ("expert_kind", expert_kind),
                    ("projection", projection),
                ),
                parameter_role="kernel",
            ),
            domain=domain,
            format=BF16_FORMAT,
            logical_dtype="bfloat16",
            logical_shape=(8, 8),
            logical_axes=("output_features", "input_features"),
            ownership=SemanticOwnership(
                binding or _binding(entry_id, graph_instance_id, domain)
            ),
        ),
        value_provenance=value_provenance,
    )


def _attention_entry(
    entry_id: str,
    projection: str,
    domain: FamilyIndexDomain,
) -> ParameterInventoryEntry:
    entry = _family_entry(
        entry_id,
        projection,
        domain,
        module_kind="attention.projection",
    )
    assert isinstance(entry.member, SemanticTensorFamily)
    return replace(
        entry,
        member=replace(
            entry.member,
            pattern=replace(
                entry.member.pattern,
                path_segments=(
                    LiteralPathSegment("layer"),
                    IndexPathSegment("global_decoder_layer"),
                    LiteralPathSegment("attention"),
                    LiteralPathSegment(projection),
                ),
                attributes=(("projection", projection),),
            ),
        ),
    )


def _explicit_entry(
    entry_id: str,
    graph_instance_id: str,
    semantic_id: str,
    *,
    semantic_graph_path: str,
    model_part: str,
    module_kind: str = "ffn.dense",
    global_decoder_layer: int | None = None,
    binding: OwnerFamilyBinding | None = None,
    value_provenance: ValueProvenance = ValueProvenance.TRAINING_PARAMETER,
) -> ParameterInventoryEntry:
    domain = _scalar_domain()
    return ParameterInventoryEntry(
        entry_id=entry_id,
        graph_instance_id=graph_instance_id,
        member=SemanticTensor(
            address=SemanticAddress(
                semantic_id=semantic_id,
                semantic_graph_path=semantic_graph_path,
                model_part=model_part,
                module_kind=module_kind,
                attributes=(),
                parameter_role="kernel",
                global_decoder_layer=global_decoder_layer,
                moe_ordinal=None,
            ),
            format=BF16_FORMAT,
            logical_dtype="bfloat16",
            logical_shape=(8, 8),
            logical_axes=("output_features", "input_features"),
            ownership=SemanticOwnership(
                binding or _binding(entry_id, graph_instance_id, domain)
            ),
        ),
        value_provenance=value_provenance,
    )


def _owner(
    entry: ParameterInventoryEntry,
    mutability: SourceMutability = SourceMutability.MUTABLE,
) -> SourceOwnerInventoryEntry:
    return SourceOwnerInventoryEntry(
        owner_family=entry.member.ownership.binding.canonical_owner_family,
        domain=entry.member.ownership.binding.member_domain,
        source_mutability=mutability,
        mutability_evidence_source=_evidence(entry.entry_id),
    )


def _runtime_lifecycle(
    kind: GraphKind,
    participation: RolloutParticipation,
) -> GraphLifecycle:
    return GraphLifecycle(
        graph_kind=kind,
        graph_provenance=GraphProvenance.TRAINING_RUNTIME,
        rollout_participation=participation,
    )


def _checkpoint_lifecycle(
    graph_instance_id: str,
    kind: GraphKind,
    model_identity: str,
    revision: str,
) -> GraphLifecycle:
    evidence = ImmutableAuxiliaryEvidence(
        graph_instance_id=graph_instance_id,
        model_identity=model_identity,
        pinned_checkpoint_revision=revision,
        checkpoint_content_digest=f"sha256:{graph_instance_id}:checkpoint",
        model_config_digest=f"sha256:{graph_instance_id}:config",
        semantic_domain_digest=f"sha256:{graph_instance_id}:domain",
        evidence_source=EvidenceSource(
            kind=EvidenceSourceKind.PINNED_CHECKPOINT_MANIFEST,
            locator=f"checkpoint://{graph_instance_id}/{revision}",
            digest=f"sha256:{graph_instance_id}:manifest",
        ),
    )
    return GraphLifecycle(
        graph_kind=kind,
        graph_provenance=GraphProvenance.EXTERNAL_CHECKPOINT,
        rollout_participation=RolloutParticipation.SERVED_FROM_CHECKPOINT,
        immutable_evidence=evidence,
    )


def _bundle(
    entries: tuple[ParameterInventoryEntry, ...],
    *,
    lifecycles: dict[str, GraphLifecycle] | None = None,
    owners: tuple[SourceOwnerInventoryEntry, ...] | None = None,
    mutabilities: dict[str, SourceMutability] | None = None,
    role_definitions: tuple[RoleDefinition, ...] = (),
    atomic_groups: dict[str, tuple[AtomicGroup, ...]] | None = None,
    model_families: dict[str, str] | None = None,
    out_of_scope: dict[str, tuple[OutOfScopeTensor, ...]] | None = None,
    source_alias_contracts: tuple[SourceAliasContract, ...] = (),
) -> SemanticManifestBundle:
    lifecycles = lifecycles or {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        )
    }
    mutabilities = mutabilities or {}
    atomic_groups = atomic_groups or {}
    model_families = model_families or {}
    out_of_scope = out_of_scope or {}
    if owners is None:
        direct_owner_entries: dict[OwnerFamilyReference, ParameterInventoryEntry] = {}
        for entry in entries:
            if entry.value_provenance != ValueProvenance.CANONICAL_ALIAS:
                direct_owner_entries[
                    entry.member.ownership.binding.canonical_owner_family
                ] = entry
        owners = tuple(
            _owner(
                entry,
                mutabilities.get(entry.entry_id, SourceMutability.MUTABLE),
            )
            for entry in direct_owner_entries.values()
        )
    manifests = tuple(
        SemanticGraphManifest(
            model_family=model_families.get(graph_id, f"family-{graph_id}"),
            model_revision=(
                lifecycle.immutable_evidence.pinned_checkpoint_revision
                if lifecycle.immutable_evidence is not None
                else f"revision-{graph_id}"
            ),
            graph_instance_id=graph_id,
            lifecycle=lifecycle,
            inventory_entry_ids=tuple(
                entry.entry_id
                for entry in entries
                if entry.graph_instance_id == graph_id
            ),
            atomic_groups=atomic_groups.get(graph_id, ()),
            out_of_scope=out_of_scope.get(graph_id, ()),
        )
        for graph_id, lifecycle in lifecycles.items()
    )
    expected = tuple(
        ExpectedGraphDeclaration(
            graph_instance_id=graph_id,
            model_identity=f"model-{graph_id}",
            lifecycle=lifecycle,
        )
        for graph_id, lifecycle in lifecycles.items()
    )
    inventory = ParameterInventory(owners=owners, entries=entries)
    provisional = SemanticManifestBundle(
        schema_version=1,
        expected_graphs=expected,
        manifests=manifests,
        inventory=inventory,
        role_definitions=role_definitions,
        source_alias_contracts=source_alias_contracts,
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


def _identical_alias_contracts(
    alias: ParameterInventoryEntry,
    canonical: ParameterInventoryEntry,
) -> tuple[IdenticalStorageSourceAliasContract, ...]:
    binding = alias.member.ownership.binding
    return tuple(
        IdenticalStorageSourceAliasContract(
            alias_entry_id=alias.entry_id,
            canonical_value_entry_id=binding.canonical_value_entry_id,
            canonical_owner_family=binding.canonical_owner_family,
            component_role=component.role,
            alias_domain=binding.member_domain,
            canonical_domain=canonical.member.ownership.binding.member_domain,
            alias_to_canonical_axes=binding.member_to_value_axes,
            storage_identity_evidence=_evidence(f"{alias.entry_id}-storage"),
        )
        for component in alias.member.format.components
    )


def _replica_alias_contracts(
    alias: ParameterInventoryEntry,
    canonical: ParameterInventoryEntry,
    *,
    replica_group_id: str = "replicas.mtp",
    evidence_name: str = "mtp-replica-synchronization",
) -> tuple[SynchronizedReplicaSourceAliasContract, ...]:
    binding = alias.member.ownership.binding
    synchronization = SourceReplicaSynchronizationEvidence(
        replica_group_id=replica_group_id,
        boundary=SourceSynchronizationBoundary.SOURCE_VERSION_READY,
        evidence_source=_evidence(evidence_name),
    )
    return tuple(
        SynchronizedReplicaSourceAliasContract(
            alias_entry_id=alias.entry_id,
            canonical_value_entry_id=binding.canonical_value_entry_id,
            canonical_owner_family=binding.canonical_owner_family,
            component_role=component.role,
            alias_domain=binding.member_domain,
            canonical_domain=canonical.member.ownership.binding.member_domain,
            alias_to_canonical_axes=binding.member_to_value_axes,
            synchronization=synchronization,
        )
        for component in alias.member.format.components
    )


def _sparse_moe_bundle(
    *,
    projections: tuple[str, ...] = ("gate", "up", "down"),
    atomic_groups: tuple[AtomicGroup, ...] = (),
) -> SemanticManifestBundle:
    routed_domain = _layer_domain(
        (1, 2, 4, 5), moe_ordinals=(0, 1, 2, 3), experts=(0, 1)
    )
    entries = tuple(
        _family_entry(f"routed-{projection}", projection, routed_domain)
        for projection in projections
    ) + (
        _family_entry(
            "dense-markers",
            "dense",
            _layer_domain((0, 3)),
            module_kind="ffn.dense",
            expert_kind="dense",
        ),
    )
    roles = builtin_role_definitions(
        1,
        {
            "moe.routed_expert": RoleExpectedDomain(
                "moe.routed_expert",
                tuple(f"routed-{projection}" for projection in projections),
            )
        },
    )
    return _bundle(
        entries,
        role_definitions=roles,
        atomic_groups={"main": atomic_groups},
    )


def _policy(scope: dict[str, object], **overrides: object) -> PrecisionPolicyConfig:
    raw: dict[str, object] = {"scopes": [{"id": "scope", **scope}], **overrides}
    return PrecisionPolicyConfig.model_validate(raw)


def _selection_entry(
    entry_id: str,
    projection: str,
    domain: FamilyIndexDomain,
    *,
    graph_instance_id: str = "main",
    semantic_graph_path: str | None = None,
    model_part: str | None = None,
    module_kind: str = "moe.expert_ffn",
    expert_kind: str = "routed",
) -> SelectionTopologyEntry:
    if semantic_graph_path is None:
        semantic_graph_path = (
            "text.decoder"
            if graph_instance_id == "main"
            else (
                "auxiliary.mtp"
                if graph_instance_id.startswith("mtp.")
                else "draft.decoder"
            )
        )
    if model_part is None:
        model_part = (
            "main"
            if graph_instance_id == "main"
            else ("mtp" if graph_instance_id.startswith("mtp.") else "draft")
        )
    segments: list[LiteralPathSegment | IndexPathSegment] = []
    if domain.layer_domain is not None:
        segments.extend(
            (
                LiteralPathSegment("layer"),
                IndexPathSegment("global_decoder_layer"),
            )
        )
    if "expert" in domain.axis_names:
        segments.extend((LiteralPathSegment("expert"), IndexPathSegment("expert")))
    if module_kind == "attention.projection":
        segments.append(LiteralPathSegment("attention"))
    segments.append(LiteralPathSegment(projection))
    if module_kind == "moe.expert_ffn":
        attributes: tuple[tuple[str, PredicateScalar], ...] = (
            ("expert_kind", expert_kind),
            ("projection", projection),
        )
    elif module_kind == "attention.projection":
        attributes = (("projection", projection),)
    else:
        attributes = ()
    return SelectionTopologyEntry(
        entry_id=entry_id,
        graph_instance_id=graph_instance_id,
        pattern=SemanticAddressPattern(
            semantic_graph_path=semantic_graph_path,
            path_segments=tuple(segments),
            model_part=model_part,
            module_kind=module_kind,
            attributes=attributes,
            parameter_role="kernel",
        ),
        domain=domain,
        logical_dtype="bfloat16",
        logical_shape=(8, 8),
        logical_axes=("output_features", "input_features"),
    )


def _selection_topology(
    entries: tuple[SelectionTopologyEntry, ...],
    *,
    lifecycles: dict[str, GraphLifecycle] | None = None,
    universes: dict[str, DecoderLayerUniverse] | None = None,
    atomic_groups: dict[str, tuple[AtomicGroup, ...]] | None = None,
    extra_roles: dict[str, tuple[RoleDefinition, ...]] | None = None,
    model_families: dict[str, str] | None = None,
) -> ResolvedSelectionTopology:
    lifecycles = lifecycles or {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        )
    }
    universes = universes or {}
    atomic_groups = atomic_groups or {}
    extra_roles = extra_roles or {}
    model_families = model_families or {}
    builtin_templates = builtin_role_definitions(1, {})
    graphs: list[ResolvedGraphTopology] = []
    for graph_instance_id, lifecycle in lifecycles.items():
        graph_entries = tuple(
            entry for entry in entries if entry.graph_instance_id == graph_instance_id
        )
        layer_members = tuple(
            layer_member
            for entry in graph_entries
            if entry.domain.layer_domain is not None
            for layer_member in entry.domain.layer_domain.members
        )
        max_global_layer = max(
            (member.global_decoder_layer for member in layer_members),
            default=0,
        )
        moe_layers_by_ordinal = {
            member.moe_ordinal: member.global_decoder_layer
            for member in layer_members
            if member.moe_ordinal is not None
        }
        inferred_universe = DecoderLayerUniverse(
            global_decoder_layers=tuple(range(max_global_layer + 1)),
            moe_global_decoder_layers_by_ordinal=tuple(
                moe_layers_by_ordinal[index]
                for index in range(len(moe_layers_by_ordinal))
            ),
        )
        graph_roles = tuple(
            replace(
                template,
                expected_domain=RoleExpectedDomain(
                    template.role_name,
                    tuple(
                        entry.entry_id
                        for entry in graph_entries
                        if _predicate_matches(
                            template.predicate,
                            lifecycle.graph_kind,
                            entry,
                        )
                    ),
                ),
            )
            for template in builtin_templates
            if any(
                _predicate_matches(
                    template.predicate,
                    lifecycle.graph_kind,
                    entry,
                )
                for entry in graph_entries
            )
        ) + extra_roles.get(graph_instance_id, ())
        evidence = lifecycle.immutable_evidence
        declaration = ExpectedGraphDeclaration(
            graph_instance_id=graph_instance_id,
            model_identity=(
                evidence.model_identity
                if evidence is not None
                else f"model-{graph_instance_id}"
            ),
            lifecycle=lifecycle,
        )
        graphs.append(
            ResolvedGraphTopology(
                declaration=declaration,
                model_family=model_families.get(
                    graph_instance_id,
                    f"family-{graph_instance_id}",
                ),
                resolved_model_revision=(
                    evidence.pinned_checkpoint_revision
                    if evidence is not None
                    else f"revision-{graph_instance_id}"
                ),
                adapter_id=f"adapter-{graph_instance_id}",
                decoder_layer_universe=universes.get(
                    graph_instance_id,
                    inferred_universe,
                ),
                entries=graph_entries,
                role_definitions=graph_roles,
                atomic_groups=atomic_groups.get(graph_instance_id, ()),
            )
        )
    canonical_graphs = tuple(
        sorted(
            graphs,
            key=lambda graph: compiler_module._graph_sort_key(
                graph.declaration.graph_instance_id
            ),
        )
    )
    role_definitions = _merge_selection_role_definitions(canonical_graphs, 1)
    digest = _compute_semantic_structure_digest(
        schema_version=1,
        graphs=canonical_graphs,
        role_definitions=role_definitions,
    )
    return ResolvedSelectionTopology(
        schema_version=1,
        graphs=canonical_graphs,
        role_definitions=role_definitions,
        semantic_structure_digest=digest,
    )


def _validate_selection_group_artifacts(
    group: CompiledPrecisionSelectionGroup,
    *,
    policy_digest: str | None = None,
    graph_selections: tuple[CompiledGraphPrecisionSelection, ...] | None = None,
    scope_results: tuple[CompiledSelectionScopeResult, ...] | None = None,
    bf16_fences: tuple[PrecisionBoundaryFence, ...] | None = None,
    atomic_expansions: tuple[AtomicExpansion, ...] | None = None,
) -> None:
    compiler_module._validate_compiled_selection_group(
        topology=group.topology,
        policy_digest=(group.policy_digest if policy_digest is None else policy_digest),
        graph_selections=(
            group.graph_selections if graph_selections is None else graph_selections
        ),
        scope_results=(group.scope_results if scope_results is None else scope_results),
        fences=group.bf16_fences if bf16_fences is None else bf16_fences,
        expansions=(
            group.atomic_expansions if atomic_expansions is None else atomic_expansions
        ),
    )


def test_source_neutral_selection_uses_declared_global_decoder_boundaries() -> None:
    domain = _layer_domain(
        (1, 2, 4, 5),
        moe_ordinals=(0, 1, 2, 3),
        experts=(0, 1),
    )
    topology = _selection_topology(
        tuple(
            _selection_entry(f"routed-{projection}", projection, domain)
            for projection in ("gate", "up", "down")
        ),
        universes={
            "main": DecoderLayerUniverse(
                global_decoder_layers=tuple(range(6)),
                moe_global_decoder_layers_by_ordinal=(1, 2, 4, 5),
            )
        },
    )
    result = compile_precision_selection(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {
                    "index_space": "global_decoder",
                    "exclude_first": 2,
                    "exclude_last": 1,
                },
                "rollout": "mxfp8",
            }
        ),
        topology,
    )

    assert isinstance(result, CompiledPrecisionSelectionGroup)
    assert result.topology is topology
    graph = result.graph_selection("main")
    assert isinstance(graph, CompiledGraphPrecisionSelection)
    assert graph.rollout_plan is not None
    assert (
        graph.rollout_plan.precision_for(
            "routed-gate",
            global_decoder_layer=1,
            moe_ordinal=0,
            independent_axes={"expert": 0},
        )
        == "bf16"
    )
    assert (
        graph.rollout_plan.precision_for(
            "routed-gate",
            global_decoder_layer=2,
            moe_ordinal=1,
            independent_axes={"expert": 0},
        )
        == "mxfp8"
    )
    assert (
        graph.rollout_plan.precision_for(
            "routed-gate",
            global_decoder_layer=4,
            moe_ordinal=2,
            independent_axes={"expert": 0},
        )
        == "mxfp8"
    )
    assert (
        graph.rollout_plan.precision_for(
            "routed-gate",
            global_decoder_layer=5,
            moe_ordinal=3,
            independent_axes={"expert": 0},
        )
        == "bf16"
    )
    assert graph.training_plan is not None
    assert all(
        assignment.requested_format == BF16_FORMAT
        for assignment in graph.training_plan.assignments
    )
    assert {fence.endpoint for fence in graph.bf16_fences} == {
        compiler_module.PrecisionEndpoint.TRAINING,
        compiler_module.PrecisionEndpoint.ROLLOUT,
    }
    assert all(isinstance(fence, PrecisionBoundaryFence) for fence in graph.bf16_fences)
    assert all(
        fence.bf16_layer_members == (LayerMember(1, 0), LayerMember(5, 3))
        for fence in graph.bf16_fences
    )


def test_source_neutral_plural_roles_share_global_decoder_boundaries() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain(
                    (0, 1, 2),
                    moe_ordinals=(0, 1, 2),
                    experts=(0,),
                ),
            ),
            _selection_entry(
                "attention-q",
                "q",
                _layer_domain((0, 1, 2)),
                module_kind="attention.projection",
            ),
        )
    )

    result = compile_precision_selection(
        _policy(
            {
                "roles": ["attention.qkvo", "moe.routed_expert"],
                "layers": {
                    "index_space": "global_decoder",
                    "exclude_first": 1,
                    "exclude_last": 1,
                },
                "rollout": "mxfp8",
            }
        ),
        topology,
    )

    rollout = result.graph_selection("main").rollout_plan
    assert rollout is not None
    assert rollout.precision_for("attention-q", global_decoder_layer=0) == "bf16"
    assert rollout.precision_for("attention-q", global_decoder_layer=1) == "mxfp8"
    assert rollout.precision_for("attention-q", global_decoder_layer=2) == "bf16"
    assert (
        rollout.precision_for(
            "routed-up",
            global_decoder_layer=0,
            moe_ordinal=0,
            independent_axes={"expert": 0},
        )
        == "bf16"
    )
    assert (
        rollout.precision_for(
            "routed-up",
            global_decoder_layer=1,
            moe_ordinal=1,
            independent_axes={"expert": 0},
        )
        == "mxfp8"
    )
    assert (
        rollout.precision_for(
            "routed-up",
            global_decoder_layer=2,
            moe_ordinal=2,
            independent_axes={"expert": 0},
        )
        == "bf16"
    )


def test_source_neutral_moe_ordinal_boundaries_use_declared_ordinal_mapping() -> None:
    domain = _layer_domain(
        (1, 2, 4, 5),
        moe_ordinals=(0, 1, 2, 3),
        experts=(0,),
    )
    topology = _selection_topology(
        (_selection_entry("routed-up", "up", domain),),
        universes={"main": DecoderLayerUniverse(tuple(range(6)), (1, 2, 4, 5))},
    )
    global_selection = compile_precision_selection(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {
                    "index_space": "global_decoder",
                    "exclude_first": 1,
                },
                "rollout": "mxfp8",
            }
        ),
        topology,
    )
    ordinal_selection = compile_precision_selection(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {
                    "index_space": "moe_ordinal",
                    "exclude_first": 1,
                },
                "rollout": "mxfp8",
            }
        ),
        topology,
    )
    global_rollout = global_selection.graph_selection("main").rollout_plan
    ordinal_rollout = ordinal_selection.graph_selection("main").rollout_plan
    assert global_rollout is not None
    assert ordinal_rollout is not None
    assert (
        global_rollout.precision_for(
            "routed-up",
            global_decoder_layer=1,
            moe_ordinal=0,
            independent_axes={"expert": 0},
        )
        == "mxfp8"
    )
    assert (
        ordinal_rollout.precision_for(
            "routed-up",
            global_decoder_layer=1,
            moe_ordinal=0,
            independent_axes={"expert": 0},
        )
        == "bf16"
    )
    assert (
        ordinal_rollout.precision_for(
            "routed-up",
            global_decoder_layer=2,
            moe_ordinal=1,
            independent_axes={"expert": 0},
        )
        == "mxfp8"
    )


def test_source_neutral_supports_mxfp8_training_and_rollout_together() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    result = compile_precision_selection(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "training": "mxfp8",
                "rollout": "mxfp8",
            }
        ),
        topology,
    )
    graph = result.graph_selection("main")
    assert graph.training_plan is not None
    assert graph.rollout_plan is not None
    for plan in (graph.training_plan, graph.rollout_plan):
        assert {assignment.precision for assignment in plan.assignments} == {"mxfp8"}
        assert all(
            assignment.requested_format == MXFP8_FORMAT
            for assignment in plan.assignments
        )


def test_source_neutral_schema_mismatch_precedes_topology_validation() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "dense",
                "weight",
                _layer_domain((0,)),
                module_kind="ffn.dense",
            ),
        )
    )
    object.__setattr__(topology, "role_definitions", ())
    object.__setattr__(topology, "semantic_structure_digest", "sha256:" + "0" * 64)
    policy = PrecisionPolicyConfig.model_validate({"scopes": []})
    object.__setattr__(policy, "schema_version", 2)

    with pytest.raises(PrecisionPolicyError, match="schema versions differ"):
        compile_precision_selection(policy, topology)


def test_source_neutral_plural_roles_require_each_role_after_filtering() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
            _selection_entry(
                "attention-q",
                "q",
                _layer_domain((1,)),
                module_kind="attention.projection",
            ),
        ),
        universes={"main": DecoderLayerUniverse((0, 1), (0,))},
    )
    scope: dict[str, object] = {
        "roles": ["moe.routed_expert", "attention.qkvo"],
        "layers": {"exclude_first": 1},
        "rollout": "mxfp8",
    }
    with pytest.raises(
        PrecisionPolicyError,
        match="moe[.]routed_expert.*matched no semantic members",
    ):
        compile_precision_selection(_policy(scope), topology)

    first = compile_precision_selection(
        _policy(scope, require_match=False),
        topology,
    )
    second = compile_precision_selection(
        _policy(
            {**scope, "roles": ["attention.qkvo", "moe.routed_expert"]},
            require_match=False,
        ),
        topology,
    )
    assert first.selection_group_id == second.selection_group_id
    assert first.to_wire_dict() == second.to_wire_dict()
    selected = first.scope_result("scope").graph_result("main").selected
    assert tuple(item.inventory_entry_id for item in selected) == ("attention-q",)


def test_source_neutral_rejects_unknown_and_incorrect_advertised_roles() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    with pytest.raises(PrecisionPolicyError, match="unknown role"):
        compile_precision_selection(
            _policy({"roles": ["adapter.unknown"], "rollout": "mxfp8"}),
            topology,
        )

    graph = topology.graphs[0]
    routed_role = next(
        role for role in graph.role_definitions if role.role_name == "moe.routed_expert"
    )
    invalid_role = replace(
        routed_role,
        expected_domain=RoleExpectedDomain("moe.routed_expert", ()),
    )
    invalid_graph_roles = tuple(
        invalid_role if role.role_name == "moe.routed_expert" else role
        for role in graph.role_definitions
    )
    object.__setattr__(graph, "role_definitions", invalid_graph_roles)
    merged = _merge_selection_role_definitions(topology.graphs, 1)
    object.__setattr__(topology, "role_definitions", merged)
    with pytest.raises(ValueError, match="expected domain"):
        compile_precision_selection(
            _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
            topology,
        )


def test_source_neutral_advanced_and_address_selectors_are_graph_qualified() -> None:
    domain = _layer_domain((0, 1))
    topology = _selection_topology(
        tuple(
            _selection_entry(
                f"attention-{projection}",
                projection,
                domain,
                module_kind="attention.projection",
            )
            for projection in ("q", "k")
        )
    )
    advanced = compile_precision_selection(
        _policy(
            {
                "advanced_match": {
                    "graph_instance_id": "main",
                    "semantic_graph_path": "text.decoder",
                    "module_kind": "attention.projection",
                    "attributes": {"projection": "q"},
                },
                "rollout": "mxfp8",
            }
        ),
        topology,
    )
    address = compile_precision_selection(
        _policy(
            {
                "addresses": [
                    {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "semantic_id": "text.decoder.layer.1.attention.k",
                    }
                ],
                "rollout": "mxfp8",
            }
        ),
        topology,
    )
    advanced_rollout = advanced.graph_selection("main").rollout_plan
    address_rollout = address.graph_selection("main").rollout_plan
    assert advanced_rollout is not None
    assert address_rollout is not None
    assert (
        advanced_rollout.precision_for("attention-q", global_decoder_layer=0) == "mxfp8"
    )
    assert (
        advanced_rollout.precision_for("attention-k", global_decoder_layer=0) == "bf16"
    )
    assert (
        address_rollout.precision_for("attention-k", global_decoder_layer=0) == "bf16"
    )
    assert (
        address_rollout.precision_for("attention-k", global_decoder_layer=1) == "mxfp8"
    )

    with pytest.raises(PrecisionPolicyError, match="must qualify"):
        compile_precision_selection(
            _policy(
                {
                    "advanced_match": {"module_kind": "attention.projection"},
                    "rollout": "mxfp8",
                }
            ),
            topology,
        )
    with pytest.raises(PrecisionPolicyError, match="resolve exactly once"):
        compile_precision_selection(
            _policy(
                {
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.9.attention.k",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ),
            topology,
        )


def test_source_neutral_rejects_conflicting_scopes_per_endpoint() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    address = {
        "graph_instance_id": "main",
        "semantic_graph_path": "text.decoder",
        "semantic_id": "text.decoder.layer.0.expert.0.up",
    }
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "mxfp8",
                    "addresses": [address],
                    "training": "mxfp8",
                    "rollout": "mxfp8",
                },
                {
                    "id": "bf16",
                    "addresses": [address],
                    "training": "mxfp8",
                    "rollout": "bf16",
                },
            ]
        }
    )
    with pytest.raises(
        PrecisionPolicyError,
        match="conflicting precision scopes.*rollout",
    ):
        compile_precision_selection(policy, topology)


def test_source_neutral_full_exclusion_and_unlayered_filter_fail_closed() -> None:
    routed = _selection_entry(
        "routed-up",
        "up",
        _layer_domain((1,), moe_ordinals=(0,), experts=(0,)),
    )
    sparse = _selection_topology(
        (routed,),
        universes={"main": DecoderLayerUniverse((0, 1), (1,))},
    )
    with pytest.raises(PrecisionPolicyError, match="consume the complete"):
        compile_precision_selection(
            _policy(
                {
                    "roles": ["moe.routed_expert"],
                    "layers": {"exclude_first": 2},
                    "rollout": "mxfp8",
                }
            ),
            sparse,
        )

    scalar = _selection_topology(
        (
            _selection_entry(
                "scalar",
                "weight",
                _scalar_domain(),
                module_kind="ffn.dense",
            ),
        )
    )
    with pytest.raises(PrecisionPolicyError, match="has no global_decoder"):
        compile_precision_selection(
            _policy(
                {
                    "advanced_match": {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "module_kind": "ffn.dense",
                    },
                    "layers": {"exclude_last": 0},
                    "rollout": "mxfp8",
                }
            ),
            scalar,
        )


def _selection_atomic_topology(
    projections: tuple[str, ...],
    group_members: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    attention: bool = False,
) -> ResolvedSelectionTopology:
    domain = _layer_domain(
        (0, 1),
        **({} if attention else {"moe_ordinals": (0, 1), "experts": (0, 1)}),
    )
    entries = tuple(
        _selection_entry(
            f"weight-{projection}",
            projection,
            domain,
            module_kind=("attention.projection" if attention else "moe.expert_ffn"),
        )
        for projection in projections
    )
    groups = tuple(
        AtomicGroup(
            group_id=group_id,
            graph_instance_id="main",
            kind=AtomicGroupKind.PRECISION,
            group_domain=domain,
            participants=tuple(
                _atomic_participant(f"weight-{projection}", domain)
                for projection in participants
            ),
        )
        for group_id, participants in group_members
    )
    return _selection_topology(entries, atomic_groups={"main": groups})


def test_source_neutral_atomic_error_expand_and_fixed_point_are_compact() -> None:
    qkv = _selection_atomic_topology(
        ("q", "k", "v"),
        (("attention.qkv", ("q", "k", "v")),),
        attention=True,
    )
    q_address = {
        "graph_instance_id": "main",
        "semantic_graph_path": "text.decoder",
        "semantic_id": "text.decoder.layer.0.attention.q",
    }
    with pytest.raises(PrecisionPolicyError, match="atomic precision conflict"):
        compile_precision_selection(
            _policy({"addresses": [q_address], "rollout": "mxfp8"}),
            qkv,
        )

    expanded = compile_precision_selection(
        _policy(
            {
                "addresses": [q_address],
                "rollout": "mxfp8",
                "atomic_conflict": "expand",
            }
        ),
        qkv,
    )
    rollout = expanded.graph_selection("main").rollout_plan
    assert rollout is not None
    for projection in ("q", "k", "v"):
        assert (
            rollout.precision_for(f"weight-{projection}", global_decoder_layer=0)
            == "mxfp8"
        )
        assert (
            rollout.precision_for(f"weight-{projection}", global_decoder_layer=1)
            == "bf16"
        )
    assert (
        sum(
            addition.logical_cardinality
            for expansion in expanded.atomic_expansions
            for addition in expansion.additions
        )
        == 2
    )

    chained = _selection_atomic_topology(
        ("gate", "up", "down"),
        (
            ("moe.gate-up", ("gate", "up")),
            ("moe.up-down", ("up", "down")),
        ),
    )
    fixed_point = compile_precision_selection(
        _policy(
            {
                "addresses": [
                    {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "semantic_id": "text.decoder.layer.0.expert.0.gate",
                    }
                ],
                "rollout": "mxfp8",
                "atomic_conflict": "expand",
            }
        ),
        chained,
    )
    fixed_rollout = fixed_point.graph_selection("main").rollout_plan
    assert fixed_rollout is not None
    for projection in ("gate", "up", "down"):
        assert (
            fixed_rollout.precision_for(
                f"weight-{projection}",
                global_decoder_layer=0,
                moe_ordinal=0,
                independent_axes={"expert": 0},
            )
            == "mxfp8"
        )
    assert {item.atomic_group_id for item in fixed_point.atomic_expansions} == {
        "moe.gate-up",
        "moe.up-down",
    }


def test_source_neutral_atomic_expansion_cannot_cross_bf16_fence() -> None:
    topology = _selection_atomic_topology(
        ("q", "k", "v"),
        (("attention.qkv", ("q", "k", "v")),),
        attention=True,
    )
    policy = PrecisionPolicyConfig.model_validate(
        {
            "atomic_conflict": "expand",
            "scopes": [
                {
                    "id": "kv-after-first",
                    "advanced_match": {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "module_kind": "attention.projection",
                        "attributes": {"projection": ["k", "v"]},
                    },
                    "layers": {"exclude_first": 1},
                    "rollout": "mxfp8",
                },
                {
                    "id": "q-first",
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.attention.q",
                        }
                    ],
                    "rollout": "mxfp8",
                },
            ],
        }
    )
    with pytest.raises(PrecisionPolicyError, match="hard BF16 layer boundary"):
        compile_precision_selection(policy, topology)


def test_source_neutral_preserves_every_graph_and_lifecycle_endpoint_matrix() -> None:
    lifecycles = {
        "draft.static": _checkpoint_lifecycle(
            "draft.static",
            GraphKind.SPECULATIVE_DRAFTER,
            "model-draft.static",
            "revision-draft.static",
        ),
        "mtp.train": _runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.NOT_SERVED,
        ),
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
    }
    entries = (
        _selection_entry(
            "main-dense",
            "weight",
            _layer_domain((0, 1)),
            module_kind="ffn.dense",
        ),
        _selection_entry(
            "mtp-dense",
            "weight",
            _layer_domain((0,)),
            graph_instance_id="mtp.train",
            module_kind="ffn.dense",
        ),
        _selection_entry(
            "draft-dense",
            "weight",
            _layer_domain((0, 1, 2)),
            graph_instance_id="draft.static",
            module_kind="ffn.dense",
        ),
    )
    topology = _selection_topology(entries, lifecycles=lifecycles)
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "main",
                    "advanced_match": {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "module_kind": "ffn.dense",
                    },
                    "rollout": "mxfp8",
                },
                {
                    "id": "mtp",
                    "advanced_match": {
                        "graph_instance_id": "mtp.train",
                        "semantic_graph_path": "auxiliary.mtp",
                        "module_kind": "ffn.dense",
                    },
                    "training": "mxfp8",
                },
                {
                    "id": "draft",
                    "advanced_match": {
                        "graph_instance_id": "draft.static",
                        "semantic_graph_path": "draft.decoder",
                        "module_kind": "ffn.dense",
                    },
                    "rollout": "mxfp8",
                },
            ]
        }
    )
    result = compile_precision_selection(policy, topology)
    assert tuple(graph.graph_instance_id for graph in result.graph_selections) == (
        "main",
        "draft.static",
        "mtp.train",
    )
    main = result.graph_selection("main")
    mtp = result.graph_selection("mtp.train")
    draft = result.graph_selection("draft.static")
    assert main.training_plan is not None and main.rollout_plan is not None
    assert mtp.training_plan is not None and mtp.rollout_plan is None
    assert draft.training_plan is None and draft.rollout_plan is not None
    assert draft.immutable_checkpoint_evidence is not None
    assert main.decoder_layer_universe.global_decoder_layers == (0, 1)
    assert mtp.decoder_layer_universe.global_decoder_layers == (0,)
    assert draft.decoder_layer_universe.global_decoder_layers == (0, 1, 2)
    assert all(len(scope.graph_results) == 3 for scope in result.scope_results)
    wire_graph_ids = {
        graph["declaration"]["graph_instance_id"]
        for graph in result.to_wire_dict()["topology"]["graphs"]
    }
    assert wire_graph_ids == {"main", "mtp.train", "draft.static"}


def test_source_neutral_revalidates_immutable_auxiliary_identity() -> None:
    lifecycle = _checkpoint_lifecycle(
        "draft.static",
        GraphKind.SPECULATIVE_DRAFTER,
        "model-draft.static",
        "revision-draft.static",
    )
    topology = _selection_topology(
        (
            _selection_entry(
                "main-dense",
                "weight",
                _layer_domain((0,)),
                module_kind="ffn.dense",
            ),
            _selection_entry(
                "draft-dense",
                "weight",
                _layer_domain((0,)),
                graph_instance_id="draft.static",
                module_kind="ffn.dense",
            ),
        ),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
            "draft.static": lifecycle,
        },
    )
    draft = next(
        graph
        for graph in topology.graphs
        if graph.declaration.graph_instance_id == "draft.static"
    )
    object.__setattr__(draft, "resolved_model_revision", "wrong-revision")

    with pytest.raises(ValueError, match="pinned checkpoint revision"):
        compile_precision_selection(
            PrecisionPolicyConfig.model_validate({"scopes": []}),
            topology,
        )


def test_source_neutral_builtin_main_role_never_selects_auxiliary_graph() -> None:
    main = _selection_entry(
        "main-dense",
        "weight",
        _layer_domain((0,)),
        module_kind="ffn.dense",
    )
    auxiliary = _selection_entry(
        "mtp-routed",
        "up",
        _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
        graph_instance_id="mtp.train",
    )
    topology = _selection_topology(
        (main, auxiliary),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
            "mtp.train": _runtime_lifecycle(
                GraphKind.MTP,
                RolloutParticipation.NOT_SERVED,
            ),
        },
    )
    builtin = compile_precision_selection(
        _policy(
            {"roles": ["moe.routed_expert"], "training": "mxfp8"},
            require_match=False,
        ),
        topology,
    )
    mtp_training = builtin.graph_selection("mtp.train").training_plan
    assert mtp_training is not None
    assert (
        mtp_training.precision_for(
            "mtp-routed",
            global_decoder_layer=0,
            moe_ordinal=0,
            independent_axes={"expert": 0},
        )
        == "bf16"
    )

    qualified = compile_precision_selection(
        _policy(
            {
                "advanced_match": {
                    "graph_instance_id": "mtp.train",
                    "semantic_graph_path": "auxiliary.mtp",
                    "module_kind": "moe.expert_ffn",
                },
                "training": "mxfp8",
            }
        ),
        topology,
    )
    qualified_training = qualified.graph_selection("mtp.train").training_plan
    assert qualified_training is not None
    assert (
        qualified_training.precision_for(
            "mtp-routed",
            global_decoder_layer=0,
            moe_ordinal=0,
            independent_axes={"expert": 0},
        )
        == "mxfp8"
    )


def test_source_neutral_ids_wire_and_pickle_are_canonical() -> None:
    domain = _layer_domain((0, 1))
    entries = tuple(
        _selection_entry(
            f"attention-{projection}",
            projection,
            domain,
            module_kind="attention.projection",
        )
        for projection in ("q", "k")
    )
    first_topology = _selection_topology(entries)
    second_topology = _selection_topology(tuple(reversed(entries)))
    scopes = [
        {
            "id": projection,
            "addresses": [
                {
                    "graph_instance_id": "main",
                    "semantic_graph_path": "text.decoder",
                    "semantic_id": f"text.decoder.layer.0.attention.{projection}",
                }
            ],
            "rollout": "mxfp8",
        }
        for projection in ("q", "k")
    ]
    first = compile_precision_selection(
        PrecisionPolicyConfig.model_validate({"scopes": scopes}),
        first_topology,
    )
    second = compile_precision_selection(
        PrecisionPolicyConfig.model_validate({"scopes": list(reversed(scopes))}),
        second_topology,
    )
    assert first.selection_group_id == second.selection_group_id
    assert (
        first.graph_selection("main").selection_id
        == second.graph_selection("main").selection_id
    )
    assert first.to_wire_dict() == second.to_wire_dict()
    assert pickle.loads(pickle.dumps(first)).to_wire_dict() == first.to_wire_dict()
    assert first.topology is first_topology
    topology_payload = first.to_wire_dict()["topology"]
    assert isinstance(topology_payload, dict)
    assert "graphs" in topology_payload
    assert (
        next(
            item
            for item in fields(CompiledGraphPrecisionSelection)
            if item.name == "selection_id"
        ).init
        is False
    )
    assert (
        next(
            item
            for item in fields(CompiledPrecisionSelectionGroup)
            if item.name == "selection_group_id"
        ).init
        is False
    )


@pytest.mark.parametrize(
    ("field_name", "expected_init"),
    (
        ("policy_snapshot", True),
        ("topology", True),
        ("schema_version", False),
        ("semantic_structure_digest", False),
        ("policy_digest", False),
        ("graph_selections", False),
        ("scope_results", False),
        ("bf16_fences", False),
        ("atomic_expansions", False),
        ("selection_group_id", False),
    ),
)
def test_source_neutral_group_exposes_only_canonical_inputs_to_replace(
    field_name: str,
    expected_init: bool,
) -> None:
    group_fields = {item.name: item for item in fields(CompiledPrecisionSelectionGroup)}

    assert group_fields[field_name].init is expected_init


def test_source_neutral_group_rejects_coordinated_all_bf16_replacement() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    mxfp8 = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    )
    bf16 = compile_precision_selection(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        topology,
    )
    mxfp8_graph = mxfp8.graph_selection("main")
    bf16_graph = bf16.graph_selection("main")

    with pytest.raises(TypeError, match="init=False"):
        replace(
            mxfp8,
            graph_selections=(
                replace(
                    mxfp8_graph,
                    training_plan=bf16_graph.training_plan,
                    rollout_plan=bf16_graph.rollout_plan,
                    scope_results=bf16_graph.scope_results,
                    bf16_fences=bf16_graph.bf16_fences,
                    atomic_expansions=bf16_graph.atomic_expansions,
                ),
            ),
            scope_results=bf16.scope_results,
            bf16_fences=bf16.bf16_fences,
            atomic_expansions=bf16.atomic_expansions,
        )


def test_source_neutral_group_policy_digest_cannot_be_supplied_by_replace() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    result = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    )

    with pytest.raises(TypeError, match="init=False"):
        replace(result, policy_digest=result.policy_digest)


def test_source_neutral_group_replacing_policy_snapshot_rederives_exactly() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    mxfp8 = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    )
    bf16 = compile_precision_selection(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        topology,
    )

    rederived = replace(mxfp8, policy_snapshot=bf16.policy_snapshot)

    assert rederived == bf16


def test_source_neutral_group_retains_frozen_policy_snapshot() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    result = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    )

    with pytest.raises(FrozenInstanceError):
        setattr(
            result.policy_snapshot,
            "canonical_json",
            result.policy_snapshot.canonical_json,
        )


def test_policy_snapshot_is_detached_from_deep_input_model_mutation() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    policy = _policy(
        {
            "advanced_match": {
                "graph_instance_id": "main",
                "semantic_graph_path": "text.decoder",
                "attributes": {"projection": ["up"]},
            },
            "rollout": "mxfp8",
        }
    )
    result = compile_precision_selection(policy, topology)
    before = (
        result.policy_snapshot.canonical_json,
        result.to_wire_dict(),
        result.selection_group_id,
    )
    advanced_match = policy.scopes[0].advanced_match
    assert advanced_match is not None
    projection = advanced_match.attributes["projection"]
    assert isinstance(projection, list)

    projection.append("gate")

    assert (
        result.policy_snapshot.canonical_json,
        result.to_wire_dict(),
        result.selection_group_id,
    ) == before


def test_policy_snapshot_revalidates_extras_mutated_after_model_creation() -> None:
    policy = _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"})
    setattr(policy.scopes[0], "undocumented_override", "mxfp8")

    with pytest.raises(ValueError, match="Undocumented precision policy field"):
        CanonicalPrecisionPolicySnapshot.from_policy(policy)


def test_policy_snapshot_rejects_precision_policy_subclass() -> None:
    class PrecisionPolicySubclass(PrecisionPolicyConfig):
        pass

    policy = PrecisionPolicySubclass.model_validate({"scopes": []})

    with pytest.raises(TypeError, match="policy must be PrecisionPolicyConfig"):
        CanonicalPrecisionPolicySnapshot.from_policy(policy)


def test_selection_group_rejects_policy_snapshot_subclass() -> None:
    class PolicySnapshotSubclass(CanonicalPrecisionPolicySnapshot):
        pass

    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    canonical = CanonicalPrecisionPolicySnapshot.from_policy(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"})
    )
    subclass = PolicySnapshotSubclass(canonical.canonical_json)

    with pytest.raises(
        TypeError,
        match="policy_snapshot must be CanonicalPrecisionPolicySnapshot",
    ):
        CompiledPrecisionSelectionGroup(
            policy_snapshot=subclass,
            topology=topology,
        )


def test_selection_group_rejects_topology_subclass() -> None:
    class SelectionTopologySubclass(ResolvedSelectionTopology):
        pass

    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    subclass = SelectionTopologySubclass(
        schema_version=topology.schema_version,
        graphs=topology.graphs,
        role_definitions=topology.role_definitions,
        semantic_structure_digest=topology.semantic_structure_digest,
    )

    with pytest.raises(TypeError, match="topology must be ResolvedSelectionTopology"):
        CompiledPrecisionSelectionGroup(
            policy_snapshot=CanonicalPrecisionPolicySnapshot.from_policy(
                _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"})
            ),
            topology=subclass,
        )


def test_selection_group_rejects_nested_graph_subclass_validation_bypass() -> None:
    class UncheckedGraph(ResolvedGraphTopology):
        def validate_complete(self) -> None:
            pass

    base = _selection_topology(
        (
            _selection_entry(
                "main-dense",
                "weight",
                _layer_domain((0,)),
                module_kind="ffn.dense",
            ),
        )
    )
    base_graph = base.graphs[0]
    foreign_entry = replace(
        base_graph.entries[0],
        graph_instance_id="draft.evil",
    )
    unchecked_graph = UncheckedGraph(
        declaration=base_graph.declaration,
        model_family=base_graph.model_family,
        resolved_model_revision=base_graph.resolved_model_revision,
        adapter_id=base_graph.adapter_id,
        decoder_layer_universe=base_graph.decoder_layer_universe,
        entries=(foreign_entry,),
        role_definitions=base_graph.role_definitions,
        atomic_groups=base_graph.atomic_groups,
    )
    with pytest.raises(
        TypeError,
        match="non-exact source-neutral record: UncheckedGraph",
    ):
        ResolvedSelectionTopology(
            schema_version=base.schema_version,
            graphs=(unchecked_graph,),
            role_definitions=base.role_definitions,
            semantic_structure_digest=_compute_semantic_structure_digest(
                schema_version=base.schema_version,
                graphs=(unchecked_graph,),
                role_definitions=base.role_definitions,
            ),
        )

    corrupted = copy(base)
    object.__setattr__(corrupted, "graphs", (unchecked_graph,))
    object.__setattr__(
        corrupted,
        "semantic_structure_digest",
        _compute_semantic_structure_digest(
            schema_version=base.schema_version,
            graphs=(unchecked_graph,),
            role_definitions=base.role_definitions,
        ),
    )
    with pytest.raises(
        TypeError,
        match="non-exact source-neutral record: UncheckedGraph",
    ):
        compile_precision_selection(
            PrecisionPolicyConfig.model_validate({"scopes": []}),
            corrupted,
        )


def test_selection_topology_rejects_role_scalar_equality_override() -> None:
    class EqualToEveryString(str):
        def __eq__(self, other: object) -> bool:
            return isinstance(other, str)

        def __ne__(self, other: object) -> bool:
            return not self.__eq__(other)

        __hash__ = str.__hash__

    domain = _layer_domain((0,), moe_ordinals=(0,), experts=(0,))
    base = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                domain,
            ),
            _selection_entry(
                "shared-up",
                "shared-up",
                domain,
                expert_kind="shared",
            ),
        )
    )
    base_graph = base.graphs[0]
    builtin = next(
        definition
        for definition in base.role_definitions
        if definition.role_name == "moe.routed_expert"
    )
    deceptive_predicate = SemanticPredicate(
        graph_kinds=builtin.predicate.graph_kinds,
        semantic_graph_paths=builtin.predicate.semantic_graph_paths,
        model_parts=builtin.predicate.model_parts,
        module_kinds=builtin.predicate.module_kinds,
        attributes=(
            AttributePredicate(
                "expert_kind",
                (EqualToEveryString("shared"),),
            ),
            AttributePredicate(
                "projection",
                tuple(
                    EqualToEveryString(value)
                    for value in ("shared-down", "shared-gate", "shared-up")
                ),
            ),
        ),
        parameter_roles=builtin.predicate.parameter_roles,
    )
    deceptive_role = RoleDefinition(
        schema_version=base.schema_version,
        role_name="moe.routed_expert",
        predicate=deceptive_predicate,
        expected_domain=RoleExpectedDomain("moe.routed_expert", ("shared-up",)),
    )
    altered_graph = replace(base_graph, role_definitions=(deceptive_role,))
    altered_roles = _merge_selection_role_definitions(
        (altered_graph,),
        base.schema_version,
    )

    with pytest.raises(
        TypeError,
        match="non-exact source-neutral value: EqualToEveryString",
    ):
        ResolvedSelectionTopology(
            schema_version=base.schema_version,
            graphs=(altered_graph,),
            role_definitions=altered_roles,
            semantic_structure_digest=_compute_semantic_structure_digest(
                schema_version=base.schema_version,
                graphs=(altered_graph,),
                role_definitions=altered_roles,
            ),
        )


def test_selection_group_rejects_tuple_subclass_behavior() -> None:
    class BehavioralTuple(tuple[LayerMember, ...]):
        pass

    topology = _selection_topology(
        (
            _selection_entry(
                "main-dense",
                "weight",
                _layer_domain((0,)),
                module_kind="ffn.dense",
            ),
        )
    )
    layer_domain = topology.graphs[0].entries[0].domain.layer_domain
    assert layer_domain is not None
    object.__setattr__(
        layer_domain,
        "members",
        BehavioralTuple(layer_domain.members),
    )

    with pytest.raises(
        TypeError,
        match="non-exact source-neutral value: BehavioralTuple",
    ):
        compile_precision_selection(
            PrecisionPolicyConfig.model_validate({"scopes": []}),
            topology,
        )


@pytest.mark.parametrize(
    ("canonical_json", "expected_exception", "expected_message"),
    (
        pytest.param(
            '{"default":"bf16","atomic_conflict":"error",'
            '"require_match":true,"schema_version":1,"scopes":[]}',
            ValueError,
            "not the canonical policy encoding",
            id="noncanonical-key-order",
        ),
        pytest.param(
            '{"atomic_conflict":"error","atomic_conflict":"error",'
            '"default":"bf16","require_match":true,"schema_version":1,'
            '"scopes":[]}',
            ValueError,
            "duplicate key",
            id="duplicate-json-key",
        ),
        pytest.param(
            '{"atomic_conflict":"error","default":"bf16",'
            '"require_match":true,"schema_version":1,"scopes":'
            '[{"addresses":null,"advanced_match":{"attributes":'
            '[{"name":"marker","predicate":{"type":"int",'
            '"value":false}}],"graph_instance_id":null,"model_part":null,'
            '"module_kind":null,"parameter_role":null,'
            '"semantic_graph_path":null},"atomic_conflict":null,'
            '"id":"scope","layers":null,"roles":null,'
            '"rollout":"mxfp8","training":null}]}',
            TypeError,
            "type tag does not match",
            id="scalar-type-tag-mismatch",
        ),
        pytest.param(
            '{"atomic_conflict":"error","default":"bf16",'
            '"require_match":true,"schema_version":NaN,"scopes":[]}',
            ValueError,
            "non-finite value",
            id="nonfinite-json-number",
        ),
    ),
)
def test_policy_snapshot_rejects_noncanonical_or_ambiguous_json(
    canonical_json: str,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    with pytest.raises(expected_exception, match=expected_message):
        CanonicalPrecisionPolicySnapshot(canonical_json)


def test_policy_snapshot_rejects_string_subclass_comparison_override() -> None:
    class NonCanonicalString(str):
        def __ne__(self, other: object) -> bool:
            return False

    noncanonical = NonCanonicalString(
        '{"default":"bf16","atomic_conflict":"error",'
        '"require_match":true,"schema_version":1,"scopes":[]}'
    )

    with pytest.raises(TypeError, match="canonical_json must be an exact string"):
        CanonicalPrecisionPolicySnapshot(noncanonical)


@pytest.mark.parametrize(
    ("value", "expected_identity"),
    (
        pytest.param(False, ("bool", False, None), id="bool-false"),
        pytest.param(0, ("int", 0, None), id="integer-zero"),
        pytest.param(0.0, ("float", 0.0, 1.0), id="positive-float-zero"),
        pytest.param(-0.0, ("float", 0.0, 1.0), id="negative-float-zero"),
    ),
)
def test_policy_snapshot_roundtrip_preserves_scalar_identity(
    value: bool | int | float,
    expected_identity: tuple[str, bool | int | float, float | None],
) -> None:
    policy = _policy(
        {
            "advanced_match": {"attributes": {"marker": value}},
            "rollout": "mxfp8",
        }
    )
    snapshot = CanonicalPrecisionPolicySnapshot.from_policy(policy)
    restored = CanonicalPrecisionPolicySnapshot(snapshot.canonical_json).to_policy()
    advanced_match = restored.scopes[0].advanced_match
    assert advanced_match is not None
    restored_value = advanced_match.attributes["marker"]
    zero_sign = (
        copysign(1.0, restored_value) if isinstance(restored_value, float) else None
    )

    assert (
        type(restored_value).__name__,
        restored_value,
        zero_sign,
    ) == expected_identity


def test_policy_snapshot_wire_dict_is_deeply_detached() -> None:
    snapshot = CanonicalPrecisionPolicySnapshot.from_policy(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"})
    )
    wire = snapshot.to_wire_dict()
    scopes = cast(list[dict[str, object]], wire["scopes"])

    scopes[0]["id"] = "tampered"

    fresh_scopes = cast(list[dict[str, object]], snapshot.to_wire_dict()["scopes"])
    assert fresh_scopes[0]["id"] == "scope"


def test_source_neutral_fences_participate_in_graph_and_group_identity() -> None:
    domain = _layer_domain((0, 1), moe_ordinals=(0, 1), experts=(0,))
    topology = _selection_topology((_selection_entry("routed-up", "up", domain),))
    result = compile_precision_selection(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {"exclude_first": 1},
                "rollout": "mxfp8",
            }
        ),
        topology,
    )
    graph = result.graph_selection("main")
    reordered = replace(graph, bf16_fences=tuple(reversed(graph.bf16_fences)))
    fewer = replace(graph, bf16_fences=graph.bf16_fences[:-1])
    assert reordered.selection_id == graph.selection_id
    assert fewer.selection_id != graph.selection_id
    assert result.to_wire_dict()["bf16_fences"]


def test_source_neutral_group_rejects_noncanonical_child_aggregates() -> None:
    domain = _layer_domain((0, 1), moe_ordinals=(0, 1), experts=(0,))
    topology = _selection_topology((_selection_entry("routed-up", "up", domain),))
    result = compile_precision_selection(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {"exclude_first": 1},
                "rollout": "mxfp8",
                "atomic_conflict": "expand",
            }
        ),
        topology,
    )
    graph = result.graph_selection("main")
    with pytest.raises(ValueError, match="scope result aggregate"):
        _validate_selection_group_artifacts(result, scope_results=())
    with pytest.raises(ValueError, match="BF16 fence aggregate"):
        _validate_selection_group_artifacts(
            result,
            bf16_fences=result.bf16_fences[:-1],
        )
    with pytest.raises(ValueError, match="topology metadata"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(replace(graph, model_family="wrong"),),
        )

    atomic = _selection_atomic_topology(
        ("q", "k", "v"),
        (("attention.qkv", ("q", "k", "v")),),
        attention=True,
    )
    expanded = compile_precision_selection(
        _policy(
            {
                "addresses": [
                    {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "semantic_id": "text.decoder.layer.0.attention.q",
                    }
                ],
                "rollout": "mxfp8",
                "atomic_conflict": "expand",
            }
        ),
        atomic,
    )
    with pytest.raises(ValueError, match="atomic expansion aggregate"):
        _validate_selection_group_artifacts(expanded, atomic_expansions=())


def test_source_neutral_group_requires_exact_graph_selection_coverage() -> None:
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "draft.static": _checkpoint_lifecycle(
            "draft.static",
            GraphKind.SPECULATIVE_DRAFTER,
            "model-draft.static",
            "revision-draft.static",
        ),
    }
    topology = _selection_topology(
        (
            _selection_entry(
                "main-dense",
                "weight",
                _layer_domain((0,)),
                module_kind="ffn.dense",
            ),
            _selection_entry(
                "draft-dense",
                "weight",
                _layer_domain((0,)),
                graph_instance_id="draft.static",
                module_kind="ffn.dense",
            ),
        ),
        lifecycles=lifecycles,
    )
    result = compile_precision_selection(
        _policy(
            {
                "advanced_match": {
                    "graph_instance_id": "main",
                    "semantic_graph_path": "text.decoder",
                    "module_kind": "ffn.dense",
                },
                "rollout": "mxfp8",
            }
        ),
        topology,
    )
    main = result.graph_selection("main")

    for invalid_graphs in ((main,), (main, main)):
        with pytest.raises(
            ValueError,
            match="graph selection aggregate must cover every topology graph exactly once",
        ):
            _validate_selection_group_artifacts(
                result,
                graph_selections=invalid_graphs,
            )


def test_source_neutral_group_rejects_forged_child_selection_id() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    result = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    )
    forged_graph = copy(result.graph_selection("main"))
    object.__setattr__(forged_graph, "selection_id", f"sha256:{'0' * 64}")

    with pytest.raises(ValueError, match="selection_id"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(forged_graph,),
        )


@pytest.mark.parametrize(
    "invalid_digest",
    (
        "not-a-digest",
        f"sha256:{'A' * 64}",
        f"sha256:{'0' * 63}",
        f"sha256:{'g' * 64}",
    ),
)
def test_source_neutral_graph_rejects_noncanonical_policy_digest(
    invalid_digest: str,
) -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    graph = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    ).graph_selection("main")

    with pytest.raises(ValueError, match="canonical SHA-256"):
        replace(graph, policy_digest=invalid_digest)


@pytest.mark.parametrize(
    "invalid_digest",
    (
        "not-a-digest",
        f"sha256:{'A' * 64}",
        f"sha256:{'0' * 63}",
        f"sha256:{'g' * 64}",
    ),
)
def test_source_neutral_group_rejects_noncanonical_policy_digest(
    invalid_digest: str,
) -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    result = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    )

    with pytest.raises(ValueError, match="canonical SHA-256"):
        _validate_selection_group_artifacts(
            result,
            policy_digest=invalid_digest,
        )


def test_source_neutral_group_rejects_incomplete_or_overlapping_partitions() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    result = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    )
    graph = result.graph_selection("main")
    rollout = graph.rollout_plan
    assert rollout is not None
    assert len(rollout.assignments) == 1

    incomplete = replace(rollout, assignments=())
    with pytest.raises(ValueError, match="complete.*partition"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(replace(graph, rollout_plan=incomplete),),
        )

    assignment = rollout.assignments[0]
    overlap = compiler_module.CompactPrecisionAssignment(
        graph_instance_id=assignment.graph_instance_id,
        semantic_graph_path=assignment.semantic_graph_path,
        inventory_entry_id=assignment.inventory_entry_id,
        member_domain=assignment.member_domain,
        precision="bf16",
        requested_format=BF16_FORMAT,
    )
    overlapping = replace(
        rollout,
        assignments=(*rollout.assignments, overlap),
    )
    with pytest.raises(ValueError, match="disjoint.*partition"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(replace(graph, rollout_plan=overlapping),),
        )


def test_source_neutral_group_cross_checks_fences_and_layer_records() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0, 1), moe_ordinals=(0, 1), experts=(0,)),
            ),
        )
    )
    result = compile_precision_selection(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {"index_space": "moe_ordinal", "exclude_first": 1},
                "rollout": "mxfp8",
            }
        ),
        topology,
    )
    graph = result.graph_selection("main")
    bad_fence = replace(
        graph.bf16_fences[0],
        bf16_layer_members=(LayerMember(999, 99),),
    )
    bad_graph = replace(
        graph,
        bf16_fences=(bad_fence, *graph.bf16_fences[1:]),
    )
    with pytest.raises(ValueError, match="BF16 fence.*scope result"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(bad_graph,),
            bf16_fences=(bad_fence, *result.bf16_fences[1:]),
        )

    scope = result.scope_results[0]
    graph_result = scope.graph_results[0]
    layer_record = graph_result.layer_selections[0]
    bad_layer_record = replace(
        layer_record,
        universe_coordinates=(*layer_record.universe_coordinates, 2),
    )
    bad_graph_result = replace(
        graph_result,
        layer_selections=(bad_layer_record,),
    )
    bad_scope = replace(scope, graph_results=(bad_graph_result,))
    bad_graph = replace(graph, scope_results=(bad_graph_result,))
    with pytest.raises(ValueError, match="layer universe differs"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(bad_graph,),
            scope_results=(bad_scope,),
        )

    unexplained_graph_result = replace(graph_result, layer_selections=())
    unexplained_scope = replace(
        scope,
        graph_results=(unexplained_graph_result,),
    )
    unexplained_graph = replace(
        graph,
        scope_results=(unexplained_graph_result,),
        bf16_fences=(),
    )
    with pytest.raises(ValueError, match="must explain matched/selected"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(unexplained_graph,),
            scope_results=(unexplained_scope,),
            bf16_fences=(),
        )

    inconsistent_record = replace(
        layer_record,
        retained_coordinates=layer_record.universe_coordinates,
    )
    inconsistent_graph_result = replace(
        graph_result,
        layer_selections=(inconsistent_record,),
    )
    inconsistent_scope = replace(
        scope,
        graph_results=(inconsistent_graph_result,),
    )
    inconsistent_graph = replace(
        graph,
        scope_results=(inconsistent_graph_result,),
    )
    with pytest.raises(ValueError, match="do not explain matched/selected"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(inconsistent_graph,),
            scope_results=(inconsistent_scope,),
        )


def test_source_neutral_group_cross_checks_atomic_expansion_topology() -> None:
    topology = _selection_atomic_topology(
        ("q", "k", "v"),
        (("attention.qkv", ("q", "k", "v")),),
        attention=True,
    )
    result = compile_precision_selection(
        _policy(
            {
                "addresses": [
                    {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "semantic_id": "text.decoder.layer.0.attention.q",
                    }
                ],
                "rollout": "mxfp8",
                "atomic_conflict": "expand",
            }
        ),
        topology,
    )
    graph = result.graph_selection("main")
    expansion = result.atomic_expansions[0]
    bad_expansion = replace(expansion, atomic_group_id="attention.unknown")
    bad_graph = replace(graph, atomic_expansions=(bad_expansion,))
    with pytest.raises(ValueError, match="unknown topology atomic group"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(bad_graph,),
            atomic_expansions=(bad_expansion,),
        )

    partial_expansion = replace(expansion, additions=expansion.additions[:1])
    partial_graph = replace(graph, atomic_expansions=(partial_expansion,))
    with pytest.raises(ValueError, match="exact fixed-point closure"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(partial_graph,),
            atomic_expansions=(partial_expansion,),
        )


def test_source_neutral_group_rejects_forged_endpoint_semantics() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "dense",
                "weight",
                _layer_domain((0,)),
                module_kind="ffn.dense",
            ),
        )
    )
    result = compile_precision_selection(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        topology,
    )
    graph = result.graph_selection("main")
    rollout = graph.rollout_plan
    assert rollout is not None
    forged_assignment = replace(
        rollout.assignments[0],
        precision="mxfp8",
        requested_format=MXFP8_FORMAT,
    )
    forged_rollout = replace(rollout, assignments=(forged_assignment,))
    forged_graph = replace(graph, rollout_plan=forged_rollout)
    with pytest.raises(ValueError, match="exact compiled endpoint plan"):
        _validate_selection_group_artifacts(
            result,
            graph_selections=(forged_graph,),
        )


def test_phase_one_selection_contains_no_source_or_runtime_binding_fields() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )
    result = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    )
    forbidden = {
        "source_format",
        "source_mutability",
        "native_storage",
        "producer_fingerprint",
        "source_alias_contracts",
        "owner_refit_requirements",
        "refit_requirement",
        "startup_owner_requests",
        "every_version_owner_requests",
        "startup_source_items",
        "every_version_source_items",
        "schedule",
        "runtime_handle",
        "expanded_members",
        "intent_id",
        "intent_group_id",
        "out_of_scope_matches",
    }

    def dataclass_field_names(value: object) -> set[str]:
        if not is_dataclass(value) or isinstance(value, type):
            if isinstance(value, (tuple, list)):
                return set().union(*(dataclass_field_names(item) for item in value))
            return set()
        names = {item.name for item in fields(value)}
        return names.union(
            *(
                dataclass_field_names(getattr(value, item.name))
                for item in fields(value)
            )
        )

    def wire_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(wire_keys(item) for item in value.values()))
        if isinstance(value, (tuple, list)):
            return set().union(*(wire_keys(item) for item in value))
        return set()

    assert dataclass_field_names(result).isdisjoint(forbidden)
    assert wire_keys(result.to_wire_dict()).isdisjoint(forbidden)


def test_phase_one_selection_uses_dedicated_scope_result_records() -> None:
    topology = _selection_topology(
        (
            _selection_entry(
                "routed-up",
                "up",
                _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
            ),
        )
    )

    result = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    )
    scope_result_type = compiler_module.CompiledSelectionScopeResult
    graph_result_type = compiler_module.CompiledSelectionScopeGraphResult
    scope_result = result.scope_result("scope")
    graph_result = scope_result.graph_result("main")

    assert type(scope_result) is scope_result_type
    assert type(graph_result) is graph_result_type
    assert "out_of_scope_matches" not in {
        item.name for item in fields(graph_result_type)
    }
    assert type(result.graph_selection("main").scope_results[0]) is graph_result_type
    assert (
        "out_of_scope_matches"
        not in result.to_wire_dict()["scope_results"][0]["graph_results"][0]
    )


def test_large_source_neutral_selection_stays_compact() -> None:
    domain = _layer_domain(
        tuple(range(60)),
        moe_ordinals=tuple(range(60)),
        experts=tuple(range(384)),
    )
    topology = _selection_topology(
        tuple(
            _selection_entry(f"kimi-{projection}", projection, domain)
            for projection in ("gate", "up", "down")
        )
    )

    result = compile_precision_selection(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        topology,
    )
    graph_result = result.scope_result("scope").graph_result("main")
    assert graph_result.selected_logical_cardinality == 60 * 384 * 3
    rollout = result.graph_selection("main").rollout_plan
    training = result.graph_selection("main").training_plan
    assert rollout is not None and training is not None
    assert len(rollout.assignments) == 3
    assert len(training.assignments) == 3


def test_source_neutral_compile_work_does_not_scale_with_cartesian_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_domain = _layer_domain(
        (0, 1),
        moe_ordinals=(0, 1),
        experts=(0, 1),
    )
    large_domain = _layer_domain(
        tuple(range(60)),
        moe_ordinals=tuple(range(60)),
        experts=tuple(range(384)),
    )
    small_topology = _selection_topology(
        tuple(
            _selection_entry(f"small-{projection}", projection, small_domain)
            for projection in ("gate", "up", "down")
        )
    )
    large_topology = _selection_topology(
        tuple(
            _selection_entry(f"large-{projection}", projection, large_domain)
            for projection in ("gate", "up", "down")
        )
    )
    calls = {"selection": 0, "assignment": 0, "domain": 0}
    original_selection = compiler_module.CompactDomainSelection.__post_init__
    original_assignment = compiler_module.CompactPrecisionAssignment.__post_init__
    original_domain = FamilyIndexDomain.__post_init__

    def counted_selection(value: compiler_module.CompactDomainSelection) -> None:
        calls["selection"] += 1
        original_selection(value)

    def counted_assignment(value: compiler_module.CompactPrecisionAssignment) -> None:
        calls["assignment"] += 1
        original_assignment(value)

    def counted_domain(value: FamilyIndexDomain) -> None:
        calls["domain"] += 1
        original_domain(value)

    monkeypatch.setattr(
        compiler_module.CompactDomainSelection,
        "__post_init__",
        counted_selection,
    )
    monkeypatch.setattr(
        compiler_module.CompactPrecisionAssignment,
        "__post_init__",
        counted_assignment,
    )
    monkeypatch.setattr(FamilyIndexDomain, "__post_init__", counted_domain)

    def compile_and_count(
        topology: ResolvedSelectionTopology,
    ) -> tuple[CompiledPrecisionSelectionGroup, dict[str, int]]:
        for name in calls:
            calls[name] = 0
        result = compile_precision_selection(
            _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
            topology,
        )
        return result, dict(calls)

    small_result, small_calls = compile_and_count(small_topology)
    large_result, large_calls = compile_and_count(large_topology)

    assert (
        small_calls
        == large_calls
        == {
            "selection": 21,
            "assignment": 12,
            "domain": 28,
        }
    )
    for result in (small_result, large_result):
        graph = result.graph_selection("main")
        assert graph.training_plan is not None
        assert graph.rollout_plan is not None
        assert len(graph.training_plan.assignments) == 3
        assert len(graph.rollout_plan.assignments) == 3


def _multi_role_bundle(
    *,
    routed_layers: tuple[int, ...] = (0, 1),
    attention_layers: tuple[int, ...] = (0, 1),
    attention_expected: tuple[str, ...] = ("attention-q",),
) -> SemanticManifestBundle:
    routed = _family_entry(
        "routed-up",
        "up",
        _layer_domain(
            routed_layers,
            moe_ordinals=tuple(range(len(routed_layers))),
            experts=(0,),
        ),
    )
    attention = _attention_entry(
        "attention-q",
        "q",
        _layer_domain(attention_layers),
    )
    return _bundle(
        (routed, attention),
        role_definitions=builtin_role_definitions(
            1,
            {
                "attention.qkvo": RoleExpectedDomain(
                    "attention.qkvo", attention_expected
                ),
                "moe.routed_expert": RoleExpectedDomain(
                    "moe.routed_expert", ("routed-up",)
                ),
            },
        ),
    )


def test_multi_role_scope_selects_each_advertised_role() -> None:
    plan = compile_precision_policy(
        _policy(
            {
                "roles": ["moe.routed_expert", "attention.qkvo"],
                "rollout": "mxfp8",
            }
        ),
        _multi_role_bundle(),
    )

    graph_result = plan.scope_result("scope").graph_result("main")
    assert graph_result.matched_inventory_entry_ids == (
        "attention-q",
        "routed-up",
    )
    rollout = plan.graph_intent("main").rollout_plan
    assert rollout is not None
    assert rollout.precision_for("attention-q", global_decoder_layer=0) == "mxfp8"
    assert (
        rollout.precision_for(
            "routed-up", global_decoder_layer=0, independent_axes={"expert": 0}
        )
        == "mxfp8"
    )


def test_legacy_plural_roles_share_global_decoder_boundaries() -> None:
    plan = compile_precision_policy(
        _policy(
            {
                "roles": ["attention.qkvo", "moe.routed_expert"],
                "layers": {
                    "index_space": "global_decoder",
                    "exclude_first": 1,
                    "exclude_last": 1,
                },
                "rollout": "mxfp8",
            }
        ),
        _multi_role_bundle(
            routed_layers=(0, 1, 2),
            attention_layers=(0, 1, 2),
        ),
    )

    rollout = plan.graph_intent("main").rollout_plan
    assert rollout is not None
    assert rollout.precision_for("attention-q", global_decoder_layer=0) == "bf16"
    assert rollout.precision_for("attention-q", global_decoder_layer=1) == "mxfp8"
    assert rollout.precision_for("attention-q", global_decoder_layer=2) == "bf16"
    assert (
        rollout.precision_for(
            "routed-up",
            global_decoder_layer=0,
            moe_ordinal=0,
            independent_axes={"expert": 0},
        )
        == "bf16"
    )
    assert (
        rollout.precision_for(
            "routed-up",
            global_decoder_layer=1,
            moe_ordinal=1,
            independent_axes={"expert": 0},
        )
        == "mxfp8"
    )
    assert (
        rollout.precision_for(
            "routed-up",
            global_decoder_layer=2,
            moe_ordinal=2,
            independent_axes={"expert": 0},
        )
        == "bf16"
    )


def test_role_list_order_does_not_change_policy_or_intent_digest() -> None:
    bundle = _multi_role_bundle()
    first = compile_precision_policy(
        _policy(
            {
                "roles": ["moe.routed_expert", "attention.qkvo"],
                "rollout": "mxfp8",
            }
        ),
        bundle,
    )
    second = compile_precision_policy(
        _policy(
            {
                "roles": ["attention.qkvo", "moe.routed_expert"],
                "rollout": "mxfp8",
            }
        ),
        bundle,
    )

    assert first.policy_digest == second.policy_digest
    assert first.intent_group_id == second.intent_group_id
    assert first.to_wire_dict() == second.to_wire_dict()


def test_multi_role_require_match_is_enforced_per_role_after_layer_filtering() -> None:
    bundle = _multi_role_bundle(routed_layers=(0,), attention_layers=(1,))

    with pytest.raises(
        PrecisionPolicyError,
        match="moe[.]routed_expert.*matched no semantic members after layer filtering",
    ):
        compile_precision_policy(
            _policy(
                {
                    "roles": ["attention.qkvo", "moe.routed_expert"],
                    "layers": {"exclude_first": 1},
                    "rollout": "mxfp8",
                }
            ),
            bundle,
        )


def test_multi_role_validates_advertised_domain_for_each_role() -> None:
    with pytest.raises(ValueError, match="role attention[.]qkvo expected domain"):
        compile_precision_policy(
            _policy(
                {
                    "roles": ["moe.routed_expert", "attention.qkvo"],
                    "rollout": "mxfp8",
                }
            ),
            _multi_role_bundle(attention_expected=()),
        )


def test_multi_role_require_match_rejects_one_known_absent_role() -> None:
    with pytest.raises(
        PrecisionPolicyError,
        match="attention[.]qkvo.*matched no semantic members after layer filtering",
    ):
        compile_precision_policy(
            _policy(
                {
                    "roles": ["moe.routed_expert", "attention.qkvo"],
                    "rollout": "mxfp8",
                }
            ),
            _sparse_moe_bundle(),
        )


def test_global_decoder_boundaries_keep_sparse_first_and_last_moe_layers_bf16() -> None:
    """Counting only MoE layers for global boundaries would quantize layer 1."""
    plan = compile_precision_policy(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {"exclude_first": 2, "exclude_last": 1},
                "rollout": "mxfp8",
            }
        ),
        _sparse_moe_bundle(),
    )
    rollout = plan.graph_intent("main").rollout_plan
    training = plan.graph_intent("main").training_plan
    assert rollout is not None
    assert training is not None
    assert (
        rollout.precision_for(
            "routed-gate", global_decoder_layer=1, independent_axes={"expert": 0}
        )
        == "bf16"
    )
    assert (
        rollout.precision_for(
            "routed-gate", global_decoder_layer=2, independent_axes={"expert": 0}
        )
        == "mxfp8"
    )
    assert (
        rollout.precision_for(
            "routed-gate", global_decoder_layer=4, independent_axes={"expert": 1}
        )
        == "mxfp8"
    )
    assert (
        rollout.precision_for(
            "routed-gate", global_decoder_layer=5, independent_axes={"expert": 1}
        )
        == "bf16"
    )
    assert (
        training.precision_for(
            "routed-gate", global_decoder_layer=2, independent_axes={"expert": 0}
        )
        == "bf16"
    )
    assert {
        assignment.requested_format
        for assignment in training.assignments
        if assignment.inventory_entry_id == "routed-gate"
    } == {BF16_FORMAT}
    assert {
        assignment.precision: assignment.requested_format
        for assignment in rollout.assignments
        if assignment.inventory_entry_id == "routed-gate"
    } == {"bf16": BF16_FORMAT, "mxfp8": MXFP8_FORMAT}
    graph_result = plan.scope_result("scope").graph_result("main")
    assert graph_result.selected_logical_cardinality == 2 * 2 * 3
    assert graph_result.layer_selections[0].universe_coordinates == tuple(range(6))
    assert graph_result.layer_selections[0].retained_coordinates == (2, 3, 4)


def test_moe_ordinal_boundaries_count_only_moe_bearing_layers() -> None:
    """Using decoder coordinates for moe_ordinal would retain routed layer 1."""
    plan = compile_precision_policy(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {
                    "index_space": "moe_ordinal",
                    "exclude_first": 1,
                    "exclude_last": 1,
                },
                "rollout": "mxfp8",
            }
        ),
        _sparse_moe_bundle(),
    )
    rollout = plan.graph_intent("main").rollout_plan
    assert rollout is not None
    assert (
        rollout.precision_for(
            "routed-up", global_decoder_layer=1, independent_axes={"expert": 0}
        )
        == "bf16"
    )
    assert (
        rollout.precision_for(
            "routed-up", global_decoder_layer=2, independent_axes={"expert": 0}
        )
        == "mxfp8"
    )
    assert (
        rollout.precision_for(
            "routed-up", global_decoder_layer=4, independent_axes={"expert": 0}
        )
        == "mxfp8"
    )
    assert (
        rollout.precision_for(
            "routed-up", global_decoder_layer=5, independent_axes={"expert": 0}
        )
        == "bf16"
    )
    layer_selection = (
        plan.scope_result("scope").graph_result("main").layer_selections[0]
    )
    assert layer_selection.universe_coordinates == (0, 1, 2, 3)
    assert layer_selection.retained_coordinates == (1, 2)


def test_same_first_exclusion_distinguishes_global_from_moe_ordinal() -> None:
    """A dense prefix layer counts globally but not in the MoE ordinal space."""
    global_plan = compile_precision_policy(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {"exclude_first": 1},
                "rollout": "mxfp8",
            }
        ),
        _sparse_moe_bundle(),
    )
    ordinal_plan = compile_precision_policy(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {
                    "index_space": "moe_ordinal",
                    "exclude_first": 1,
                },
                "rollout": "mxfp8",
            }
        ),
        _sparse_moe_bundle(),
    )
    global_rollout = global_plan.graph_intent("main").rollout_plan
    ordinal_rollout = ordinal_plan.graph_intent("main").rollout_plan
    assert global_rollout is not None
    assert ordinal_rollout is not None
    assert (
        global_rollout.precision_for(
            "routed-up", global_decoder_layer=1, independent_axes={"expert": 0}
        )
        == "mxfp8"
    )
    assert (
        ordinal_rollout.precision_for(
            "routed-up", global_decoder_layer=1, independent_axes={"expert": 0}
        )
        == "bf16"
    )


def test_same_last_exclusion_distinguishes_global_from_moe_ordinal() -> None:
    """A dense suffix counts globally but not in the MoE ordinal space."""
    routed_domain = _layer_domain((1, 2, 4), moe_ordinals=(0, 1, 2), experts=(0,))
    routed = _family_entry("routed-up", "up", routed_domain)
    dense = _family_entry(
        "dense-markers",
        "dense",
        _layer_domain((0, 3, 5)),
        module_kind="ffn.dense",
        expert_kind="dense",
    )
    bundle = _bundle(
        (routed, dense),
        role_definitions=builtin_role_definitions(
            1,
            {
                "moe.routed_expert": RoleExpectedDomain(
                    "moe.routed_expert", ("routed-up",)
                )
            },
        ),
    )
    global_plan = compile_precision_policy(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {"exclude_last": 1},
                "rollout": "mxfp8",
            }
        ),
        bundle,
    )
    ordinal_plan = compile_precision_policy(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "layers": {
                    "index_space": "moe_ordinal",
                    "exclude_last": 1,
                },
                "rollout": "mxfp8",
            }
        ),
        bundle,
    )
    global_rollout = global_plan.graph_intent("main").rollout_plan
    ordinal_rollout = ordinal_plan.graph_intent("main").rollout_plan
    assert global_rollout is not None
    assert ordinal_rollout is not None
    assert (
        global_rollout.precision_for(
            "routed-up", global_decoder_layer=4, independent_axes={"expert": 0}
        )
        == "mxfp8"
    )
    assert (
        ordinal_rollout.precision_for(
            "routed-up", global_decoder_layer=4, independent_axes={"expert": 0}
        )
        == "bf16"
    )


def test_source_served_role_can_request_mxfp8_on_both_endpoints() -> None:
    """Training and rollout assignments must be independently materialized."""
    plan = compile_precision_policy(
        _policy(
            {
                "roles": ["moe.routed_expert"],
                "training": "mxfp8",
                "rollout": "mxfp8",
            }
        ),
        _sparse_moe_bundle(),
    )
    intent = plan.graph_intent("main")
    assert intent.training_plan is not None
    assert intent.rollout_plan is not None
    for endpoint_plan in (intent.training_plan, intent.rollout_plan):
        assert (
            endpoint_plan.precision_for(
                "routed-down",
                global_decoder_layer=4,
                independent_axes={"expert": 1},
            )
            == "mxfp8"
        )
        assert MXFP8_FORMAT in {
            assignment.requested_format
            for assignment in endpoint_plan.assignments
            if assignment.inventory_entry_id == "routed-down"
        }


def test_non_gated_routed_role_uses_topology_expected_domain() -> None:
    """Assuming gate/up/down cardinality would reject valid non-gated Nemotron."""
    plan = compile_precision_policy(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        _sparse_moe_bundle(projections=("up", "down")),
    )
    result = plan.scope_result("scope").graph_result("main")
    assert result.matched_inventory_entry_ids == ("routed-down", "routed-up")
    assert result.selected_logical_cardinality == 4 * 2 * 2


def test_role_is_validated_before_layer_filtering() -> None:
    """Layer filtering must not hide an incompletely advertised role."""
    retained_domain = _layer_domain((1,), moe_ordinals=(1,), experts=(0,))
    excluded_domain = _layer_domain((0,), moe_ordinals=(0,), experts=(0,))
    entries = (
        _family_entry("routed-gate", "gate", retained_domain),
        _family_entry("routed-up", "up", retained_domain),
        _family_entry("routed-down", "down", excluded_domain),
    )
    broken = _bundle(
        entries,
        role_definitions=builtin_role_definitions(
            1,
            {
                "moe.routed_expert": RoleExpectedDomain(
                    "moe.routed_expert", ("routed-gate", "routed-up")
                )
            },
        ),
    )
    with pytest.raises(ValueError, match="expected domain"):
        compile_precision_policy(
            _policy(
                {
                    "roles": ["moe.routed_expert"],
                    "layers": {"exclude_first": 1},
                    "rollout": "mxfp8",
                }
            ),
            broken,
        )


def test_unknown_role_and_required_zero_match_fail_closed() -> None:
    """A misspelled or inapplicable required selector cannot silently stay BF16."""
    with pytest.raises(PrecisionPolicyError, match="unknown role"):
        compile_precision_policy(
            _policy({"roles": ["moe.unknown"], "rollout": "mxfp8"}),
            _sparse_moe_bundle(),
        )
    dense_only = _family_entry(
        "dense-only",
        "dense",
        _layer_domain((0,)),
        module_kind="ffn.dense",
        expert_kind="dense",
    )
    with pytest.raises(PrecisionPolicyError, match="matched no semantic members"):
        compile_precision_policy(
            _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
            _bundle(
                (dense_only,),
                role_definitions=builtin_role_definitions(1, {}),
            ),
        )
    with pytest.raises(PrecisionPolicyError, match="matched no semantic members"):
        compile_precision_policy(
            _policy(
                {
                    "advanced_match": {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "module_kind": "embedding.token",
                    },
                    "rollout": "mxfp8",
                }
            ),
            _sparse_moe_bundle(),
        )


def test_optional_advanced_zero_match_keeps_participating_defaults() -> None:
    """An explicitly optional qualified selector may compile to an empty scope."""
    plan = compile_precision_policy(
        _policy(
            {
                "advanced_match": {
                    "graph_instance_id": "main",
                    "semantic_graph_path": "text.decoder",
                    "module_kind": "embedding.token",
                },
                "rollout": "mxfp8",
            },
            require_match=False,
        ),
        _sparse_moe_bundle(),
    )
    result = plan.scope_result("scope").graph_result("main")
    assert result.matched == ()
    assert result.selected == ()
    rollout = plan.graph_intent("main").rollout_plan
    assert rollout is not None
    assert (
        rollout.precision_for(
            "routed-gate", global_decoder_layer=2, independent_axes={"expert": 0}
        )
        == "bf16"
    )


def test_layer_selector_rejects_full_exclusion_and_unlayered_targets() -> None:
    """A boundary typo or layer selector on an embedding must fail at startup."""
    with pytest.raises(PrecisionPolicyError, match="consume the complete"):
        compile_precision_policy(
            _policy(
                {
                    "roles": ["moe.routed_expert"],
                    "layers": {"exclude_first": 6},
                    "rollout": "mxfp8",
                }
            ),
            _sparse_moe_bundle(),
        )
    with pytest.raises(PrecisionPolicyError, match="consume the complete"):
        compile_precision_policy(
            _policy(
                {
                    "roles": ["moe.routed_expert"],
                    "layers": {"exclude_first": 3, "exclude_last": 3},
                    "rollout": "mxfp8",
                }
            ),
            _sparse_moe_bundle(),
        )
    embedding = _explicit_entry(
        "embedding",
        "main",
        "text.embedding.ngram.kernel",
        semantic_graph_path="text.embedding",
        model_part="main",
        module_kind="embedding.ngram",
    )
    roles = builtin_role_definitions(
        1,
        {"embedding.ngram": RoleExpectedDomain("embedding.ngram", ("embedding",))},
    )
    bundle = _bundle((embedding,), role_definitions=roles)
    with pytest.raises(PrecisionPolicyError, match="has no moe_ordinal"):
        compile_precision_policy(
            _policy(
                {
                    "roles": ["embedding.ngram"],
                    "layers": {"index_space": "moe_ordinal"},
                    "rollout": "mxfp8",
                }
            ),
            bundle,
        )


def test_explicit_empty_layer_selector_rejects_unlayered_target() -> None:
    """Materializing layers={} as omitted would hide an invalid selector."""
    embedding = _explicit_entry(
        "embedding",
        "main",
        "text.embedding.ngram.kernel",
        semantic_graph_path="text.embedding",
        model_part="main",
        module_kind="embedding.ngram",
    )
    roles = builtin_role_definitions(
        1,
        {"embedding.ngram": RoleExpectedDomain("embedding.ngram", ("embedding",))},
    )
    with pytest.raises(PrecisionPolicyError, match="has no global_decoder"):
        compile_precision_policy(
            _policy(
                {
                    "roles": ["embedding.ngram"],
                    "layers": {},
                    "rollout": "mxfp8",
                }
            ),
            _bundle((embedding,), role_definitions=roles),
        )


def test_omitted_layer_selector_accepts_unlayered_target() -> None:
    """Treating omitted layers like explicit {} would reject valid embeddings."""
    embedding = _explicit_entry(
        "embedding",
        "main",
        "text.embedding.ngram.kernel",
        semantic_graph_path="text.embedding",
        model_part="main",
        module_kind="embedding.ngram",
    )
    bundle = _bundle(
        (embedding,),
        role_definitions=builtin_role_definitions(
            1,
            {"embedding.ngram": RoleExpectedDomain("embedding.ngram", ("embedding",))},
        ),
    )
    plan = compile_precision_policy(
        _policy({"roles": ["embedding.ngram"], "rollout": "mxfp8"}), bundle
    )
    rollout = plan.graph_intent("main").rollout_plan
    assert rollout is not None
    assert rollout.precision_for("embedding") == "mxfp8"


def test_required_scope_fails_when_boundary_filters_every_role_member() -> None:
    """Pre-filter matches must not satisfy require_match after filtering."""
    routed = _family_entry(
        "routed-up",
        "up",
        _layer_domain((0,), moe_ordinals=(0,), experts=(0,)),
    )
    dense = _family_entry(
        "dense-layer",
        "dense",
        _layer_domain((1,)),
        module_kind="ffn.dense",
        expert_kind="dense",
    )
    bundle = _bundle(
        (routed, dense),
        role_definitions=builtin_role_definitions(
            1,
            {
                "moe.routed_expert": RoleExpectedDomain(
                    "moe.routed_expert", ("routed-up",)
                )
            },
        ),
    )
    with pytest.raises(PrecisionPolicyError, match="after layer filtering"):
        compile_precision_policy(
            _policy(
                {
                    "roles": ["moe.routed_expert"],
                    "layers": {"exclude_first": 1},
                    "rollout": "mxfp8",
                }
            ),
            bundle,
        )


def test_qualified_advanced_and_address_selectors_are_exact() -> None:
    """Graph/path qualification must prevent accidental auxiliary selection."""
    entries = _sparse_moe_bundle().inventory.entries
    draft = _family_entry(
        "draft-up",
        "up",
        _layer_domain((0, 1), experts=(0, 1)),
        graph_instance_id="draft.external",
        semantic_graph_path="draft.decoder",
        model_part="draft",
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN, RolloutParticipation.SERVED_FROM_SOURCE
        ),
        "draft.external": _runtime_lifecycle(
            GraphKind.SPECULATIVE_DRAFTER,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
    }
    bundle = _bundle(
        (*entries, draft),
        lifecycles=lifecycles,
        model_families={"draft.external": "other-family"},
    )
    advanced = compile_precision_policy(
        _policy(
            {
                "advanced_match": {
                    "graph_instance_id": "draft.external",
                    "semantic_graph_path": "draft.decoder",
                    "module_kind": "moe.expert_ffn",
                    "attributes": {"projection": "up"},
                },
                "training": "mxfp8",
                "rollout": "mxfp8",
            }
        ),
        bundle,
    )
    draft_intent = advanced.graph_intent("draft.external")
    assert draft_intent.training_plan is not None
    assert draft_intent.rollout_plan is not None
    assert (
        draft_intent.training_plan.precision_for(
            "draft-up", global_decoder_layer=0, independent_axes={"expert": 0}
        )
        == "mxfp8"
    )
    main_rollout = advanced.graph_intent("main").rollout_plan
    assert main_rollout is not None
    assert (
        main_rollout.precision_for(
            "routed-up", global_decoder_layer=1, independent_axes={"expert": 0}
        )
        == "bf16"
    )

    address = compile_precision_policy(
        _policy(
            {
                "addresses": [
                    {
                        "graph_instance_id": "draft.external",
                        "semantic_graph_path": "draft.decoder",
                        "semantic_id": "draft.decoder.layer.0.expert.0.up",
                    }
                ],
                "rollout": "mxfp8",
            }
        ),
        bundle,
    )
    rollout = address.graph_intent("draft.external").rollout_plan
    assert rollout is not None
    assert (
        address.scope_result("scope")
        .graph_result("draft.external")
        .selected_logical_cardinality
        == 1
    )
    assert (
        rollout.precision_for(
            "draft-up", global_decoder_layer=0, independent_axes={"expert": 0}
        )
        == "mxfp8"
    )
    assert (
        rollout.precision_for(
            "draft-up", global_decoder_layer=0, independent_axes={"expert": 1}
        )
        == "bf16"
    )
    assert (
        rollout.precision_for(
            "draft-up", global_decoder_layer=1, independent_axes={"expert": 0}
        )
        == "bf16"
    )


@pytest.mark.parametrize(
    "advanced_match",
    [
        {"semantic_graph_path": "text.decoder", "module_kind": "moe.expert_ffn"},
        {"graph_instance_id": "main", "module_kind": "moe.expert_ffn"},
    ],
)
def test_advanced_selector_requires_graph_and_semantic_path_qualification(
    advanced_match: dict[str, object],
) -> None:
    """Dropping either identity facet would make future auxiliary matches ambiguous."""
    with pytest.raises(PrecisionPolicyError, match="must qualify"):
        compile_precision_policy(
            _policy({"advanced_match": advanced_match, "rollout": "mxfp8"}),
            _sparse_moe_bundle(),
        )


def test_exact_address_must_resolve_once_even_when_optional_matching_is_enabled() -> (
    None
):
    """An explicit canonical address is never an optional wildcard."""
    policy = _policy(
        {
            "addresses": [
                {
                    "graph_instance_id": "main",
                    "semantic_graph_path": "text.decoder",
                    "semantic_id": "text.decoder.layer.99.expert.0.gate",
                }
            ],
            "rollout": "mxfp8",
        },
        require_match=False,
    )
    with pytest.raises(PrecisionPolicyError, match="address must resolve exactly once"):
        compile_precision_policy(policy, _sparse_moe_bundle())


def test_default_only_schema_mismatch_is_rejected_before_validation_or_hashing() -> (
    None
):
    """A no-scope policy must not bypass the schema compatibility gate."""
    policy = PrecisionPolicyConfig.model_construct(schema_version=2, scopes=[])
    bundle = replace(_sparse_moe_bundle(), manifests=())
    with pytest.raises(PrecisionPolicyError, match="schema versions differ"):
        compile_precision_policy(policy, bundle)


def test_builtin_role_requires_main_graph_kind_and_exact_semantic_path() -> None:
    """Main-like auxiliary facets must not leak into a built-in role."""
    domain = _layer_domain((0,), moe_ordinals=(0,), experts=(0,))
    true_main = _family_entry("main-up", "up", domain)
    draft_masquerade = _family_entry(
        "draft-up",
        "up",
        domain,
        graph_instance_id="draft.fake",
        semantic_graph_path="text.decoder",
        model_part="main",
    )
    wrong_path = _family_entry(
        "main-mtp-up",
        "up",
        domain,
        semantic_graph_path="auxiliary.mtp",
        model_part="main",
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN, RolloutParticipation.SERVED_FROM_SOURCE
        ),
        "draft.fake": _runtime_lifecycle(
            GraphKind.SPECULATIVE_DRAFTER,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
    }
    bundle = _bundle(
        (draft_masquerade, wrong_path, true_main),
        lifecycles=lifecycles,
        role_definitions=builtin_role_definitions(
            1,
            {
                "moe.routed_expert": RoleExpectedDomain(
                    "moe.routed_expert", ("main-up",)
                )
            },
        ),
    )
    plan = compile_precision_policy(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}), bundle
    )
    result = plan.scope_result("scope").graph_result("main")
    assert result.matched_inventory_entry_ids == ("main-up",)
    main_rollout = plan.graph_intent("main").rollout_plan
    draft_rollout = plan.graph_intent("draft.fake").rollout_plan
    assert main_rollout is not None
    assert draft_rollout is not None
    assert (
        main_rollout.precision_for(
            "main-up", global_decoder_layer=0, independent_axes={"expert": 0}
        )
        == "mxfp8"
    )
    assert (
        main_rollout.precision_for(
            "main-mtp-up", global_decoder_layer=0, independent_axes={"expert": 0}
        )
        == "bf16"
    )
    assert (
        draft_rollout.precision_for(
            "draft-up", global_decoder_layer=0, independent_axes={"expert": 0}
        )
        == "bf16"
    )


def test_compiled_records_are_frozen_and_have_canonical_graph_order() -> None:
    """Mutable or input-ordered plans would make cross-rank identity unstable."""
    main_entry = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    draft_entry = _explicit_entry(
        "draft-weight",
        "draft.z",
        "draft.decoder.layer.0.dense.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        global_decoder_layer=0,
    )
    mtp_entry = _explicit_entry(
        "mtp-weight",
        "mtp.a",
        "auxiliary.mtp.layer.0.dense.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="mtp",
        global_decoder_layer=0,
    )
    lifecycles = {
        "draft.z": _runtime_lifecycle(
            GraphKind.SPECULATIVE_DRAFTER, RolloutParticipation.NOT_SERVED
        ),
        "mtp.a": _runtime_lifecycle(GraphKind.MTP, RolloutParticipation.NOT_SERVED),
        "main": _runtime_lifecycle(
            GraphKind.MAIN, RolloutParticipation.SERVED_FROM_SOURCE
        ),
    }
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        _bundle((draft_entry, mtp_entry, main_entry), lifecycles=lifecycles),
    )
    assert tuple(item.graph_instance_id for item in plan.graph_intents) == (
        "main",
        "draft.z",
        "mtp.a",
    )
    with pytest.raises(FrozenInstanceError):
        plan.intent_group_id = "changed"  # type: ignore[misc]


def test_training_only_mtp_and_different_family_draft_have_no_rollout_or_refit() -> (
    None
):
    """Training-only auxiliaries must not synthesize rollout owners or requests."""
    main = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    mtp = _explicit_entry(
        "mtp-weight",
        "mtp.train",
        "auxiliary.mtp.layer.0.dense.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="mtp",
        global_decoder_layer=0,
    )
    draft = _explicit_entry(
        "draft-weight",
        "draft.other",
        "draft.decoder.layer.0.dense.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        global_decoder_layer=0,
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN, RolloutParticipation.SERVED_FROM_SOURCE
        ),
        "mtp.train": _runtime_lifecycle(GraphKind.MTP, RolloutParticipation.NOT_SERVED),
        "draft.other": _runtime_lifecycle(
            GraphKind.SPECULATIVE_DRAFTER, RolloutParticipation.NOT_SERVED
        ),
    }
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "mtp",
                    "advanced_match": {
                        "graph_instance_id": "mtp.train",
                        "semantic_graph_path": "auxiliary.mtp",
                    },
                    "training": "mxfp8",
                },
                {
                    "id": "draft",
                    "addresses": [
                        {
                            "graph_instance_id": "draft.other",
                            "semantic_graph_path": "draft.decoder",
                            "semantic_id": "draft.decoder.layer.0.dense.kernel",
                        }
                    ],
                    "training": "mxfp8",
                },
            ]
        }
    )
    plan = compile_precision_policy(
        policy,
        _bundle(
            (draft, mtp, main),
            lifecycles=lifecycles,
            model_families={"draft.other": "unrelated-drafter-family"},
        ),
    )
    for graph_id, entry_id in (
        ("mtp.train", "mtp-weight"),
        ("draft.other", "draft-weight"),
    ):
        intent = plan.graph_intent(graph_id)
        assert intent.training_plan is not None
        assert intent.training_plan.precision_for(entry_id) == "mxfp8"
        assert intent.rollout_plan is None
        assert intent.refit_requirement == RefitRequirement.NONE
        assert tuple(intent.owner_refit_requirements.values()) == (
            RefitRequirement.NONE,
        )
        assert intent.startup_owner_requests == ()
        assert intent.every_version_owner_requests == ()
    assert all(
        "mtp.train" not in item.member_graph_instance_ids
        and "draft.other" not in item.member_graph_instance_ids
        for item in (*plan.startup_source_items, *plan.every_version_source_items)
    )
    for scope_id in ("mtp", "draft"):
        scope_result = plan.scope_result(scope_id)
        assert tuple(item.graph_instance_id for item in scope_result.graph_results) == (
            "main",
            "draft.other",
            "mtp.train",
        )
        assert scope_result.graph_result("main").selected == ()


def test_rollout_request_for_training_only_auxiliary_is_rejected() -> None:
    """A non-served graph cannot satisfy a rollout-only MXFP8 request."""
    main = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    mtp = _explicit_entry(
        "mtp-weight",
        "mtp.train",
        "auxiliary.mtp.layer.0.dense.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="mtp",
        global_decoder_layer=0,
    )
    bundle = _bundle(
        (main, mtp),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN, RolloutParticipation.SERVED_FROM_SOURCE
            ),
            "mtp.train": _runtime_lifecycle(
                GraphKind.MTP, RolloutParticipation.NOT_SERVED
            ),
        },
    )
    with pytest.raises(
        PrecisionPolicyError,
        match="does not request MXFP8 on a participating endpoint",
    ):
        compile_precision_policy(
            _policy(
                {
                    "advanced_match": {
                        "graph_instance_id": "mtp.train",
                        "semantic_graph_path": "auxiliary.mtp",
                    },
                    "rollout": "mxfp8",
                }
            ),
            bundle,
        )


def test_training_request_for_checkpoint_only_auxiliary_is_rejected() -> None:
    """A checkpoint-only graph cannot satisfy a training-only MXFP8 request."""
    main = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    draft = _explicit_entry(
        "draft-weight",
        "draft.static",
        "draft.decoder.layer.0.dense.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        global_decoder_layer=0,
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    bundle = _bundle(
        (main, draft),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN, RolloutParticipation.SERVED_FROM_SOURCE
            ),
            "draft.static": _checkpoint_lifecycle(
                "draft.static",
                GraphKind.SPECULATIVE_DRAFTER,
                "model-draft.static",
                "rev-1",
            ),
        },
        mutabilities={"draft-weight": SourceMutability.FROZEN},
    )
    with pytest.raises(
        PrecisionPolicyError,
        match="does not request MXFP8 on a participating endpoint",
    ):
        compile_precision_policy(
            _policy(
                {
                    "advanced_match": {
                        "graph_instance_id": "draft.static",
                        "semantic_graph_path": "draft.decoder",
                    },
                    "training": "mxfp8",
                }
            ),
            bundle,
        )


def test_source_served_empty_graph_is_rejected_during_compilation() -> None:
    """Compilation must run the complete source-serving semantic gate."""
    with pytest.raises(ValueError, match="non-empty semantic domain"):
        compile_precision_policy(
            PrecisionPolicyConfig.model_validate({"scopes": []}),
            _bundle(()),
        )


def test_source_served_absent_owner_is_rejected_during_compilation() -> None:
    """An ABSENT owner cannot produce a startup or every-version request."""
    entry = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    with pytest.raises(ValueError, match="canonical owner cannot be absent"):
        compile_precision_policy(
            PrecisionPolicyConfig.model_validate({"scopes": []}),
            _bundle((entry,), owners=(_owner(entry, SourceMutability.ABSENT),)),
        )


def test_source_served_duplicate_mixed_mutability_owner_is_rejected() -> None:
    """One owner identity cannot make cadence depend on inventory order."""
    entry = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    owners = (
        _owner(entry, SourceMutability.MUTABLE),
        _owner(entry, SourceMutability.FROZEN),
    )
    with pytest.raises(ValueError, match="duplicate source owner family"):
        compile_precision_policy(
            PrecisionPolicyConfig.model_validate({"scopes": []}),
            _bundle((entry,), owners=owners),
        )


def test_frozen_source_graph_is_startup_only() -> None:
    """Treating a frozen served owner as mutable would retransmit it every update."""
    entry = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        _bundle(
            (entry,),
            mutabilities={"main-weight": SourceMutability.FROZEN},
        ),
    )
    intent = plan.graph_intent("main")
    owner = entry.member.ownership.binding.canonical_owner_family
    assert intent.refit_requirement == RefitRequirement.INITIAL_ONLY
    assert intent.owner_refit_requirements[owner] == RefitRequirement.INITIAL_ONLY
    assert tuple(intent.owner_refit_requirements.items()) == (
        (owner, RefitRequirement.INITIAL_ONLY),
    )
    assert tuple(intent.owner_refit_requirements.keys()) == (owner,)
    assert tuple(intent.owner_refit_requirements.values()) == (
        RefitRequirement.INITIAL_ONLY,
    )
    assert intent.startup_owner_requests == (owner,)
    assert intent.every_version_owner_requests == ()
    assert tuple(item.owner_family for item in plan.startup_source_items) == (owner,)
    assert plan.every_version_source_items == ()


def test_mixed_mutability_retains_per_owner_startup_and_every_version_requests() -> (
    None
):
    """A mutable graph summary must not promote its independent frozen owner."""
    frozen = _explicit_entry(
        "frozen-weight",
        "main",
        "text.decoder.layer.0.frozen.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    mutable = _explicit_entry(
        "mutable-weight",
        "main",
        "text.decoder.layer.0.mutable.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        _bundle(
            (mutable, frozen),
            mutabilities={"frozen-weight": SourceMutability.FROZEN},
        ),
    )
    intent = plan.graph_intent("main")
    frozen_owner = frozen.member.ownership.binding.canonical_owner_family
    mutable_owner = mutable.member.ownership.binding.canonical_owner_family
    assert intent.refit_requirement == RefitRequirement.EVERY_VERSION
    assert (
        intent.owner_refit_requirements[frozen_owner] == RefitRequirement.INITIAL_ONLY
    )
    assert (
        intent.owner_refit_requirements[mutable_owner] == RefitRequirement.EVERY_VERSION
    )
    assert intent.startup_owner_requests == (frozen_owner,)
    assert intent.every_version_owner_requests == (mutable_owner,)
    assert tuple(item.owner_family for item in plan.startup_source_items) == (
        frozen_owner,
    )
    assert tuple(item.owner_family for item in plan.every_version_source_items) == (
        mutable_owner,
    )


@pytest.mark.parametrize(
    "mutability,expected_requirement,request_collection",
    [
        (
            SourceMutability.MUTABLE,
            RefitRequirement.EVERY_VERSION,
            "every_version_source_items",
        ),
        (
            SourceMutability.FROZEN,
            RefitRequirement.INITIAL_ONLY,
            "startup_source_items",
        ),
    ],
)
def test_independent_source_served_mtp_derives_owner_cadence(
    mutability: SourceMutability,
    expected_requirement: RefitRequirement,
    request_collection: str,
) -> None:
    """MTP graph kind must not be special-cased out of source cadence."""
    main = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    mtp = _explicit_entry(
        "mtp-weight",
        "mtp.served",
        "auxiliary.mtp.layer.0.dense.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="mtp",
        global_decoder_layer=0,
    )
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        _bundle(
            (main, mtp),
            lifecycles={
                "main": _runtime_lifecycle(
                    GraphKind.MAIN, RolloutParticipation.SERVED_FROM_SOURCE
                ),
                "mtp.served": _runtime_lifecycle(
                    GraphKind.MTP, RolloutParticipation.SERVED_FROM_SOURCE
                ),
            },
            mutabilities={"mtp-weight": mutability},
        ),
    )
    owner = mtp.member.ownership.binding.canonical_owner_family
    intent = plan.graph_intent("mtp.served")
    assert intent.refit_requirement == expected_requirement
    assert intent.owner_refit_requirements[owner] == expected_requirement
    requests = getattr(plan, request_collection)
    matching_requests = tuple(
        request for request in requests if request.owner_family == owner
    )
    assert len(matching_requests) == 1
    assert matching_requests[0].member_graph_instance_ids == ("mtp.served",)
    other_requests = (
        plan.startup_source_items
        if request_collection == "every_version_source_items"
        else plan.every_version_source_items
    )
    assert owner not in {request.owner_family for request in other_requests}


def test_out_of_scope_entry_is_accounted_but_never_assigned_or_selected() -> None:
    """Typed frozen exclusions remain visible without leaking into endpoint plans."""
    active = _explicit_entry(
        "active-weight",
        "main",
        "text.decoder.layer.0.active.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    frozen = _explicit_entry(
        "frozen-weight",
        "main",
        "text.decoder.layer.0.frozen.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    bundle = _bundle(
        (active, frozen),
        mutabilities={"frozen-weight": SourceMutability.FROZEN},
        out_of_scope={
            "main": (
                OutOfScopeTensor(
                    "frozen-weight", OutOfScopeReason.SOURCE_PROVEN_FROZEN
                ),
            )
        },
    )
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}), bundle
    )
    intent = plan.graph_intent("main")
    active_owner = active.member.ownership.binding.canonical_owner_family
    frozen_owner = frozen.member.ownership.binding.canonical_owner_family
    assert intent.out_of_scope_inventory_entry_ids == ("frozen-weight",)
    assert intent.owner_refit_requirements[active_owner] == (
        RefitRequirement.EVERY_VERSION
    )
    assert intent.owner_refit_requirements[frozen_owner] == RefitRequirement.NONE
    assert plan.startup_source_items == ()
    assert tuple(
        request.owner_family for request in plan.every_version_source_items
    ) == (active_owner,)
    assert plan.every_version_source_items[0].inventory_entry_ids == (
        ("main", "active-weight"),
    )
    assert intent.training_plan is not None
    assert intent.rollout_plan is not None
    assert {
        assignment.inventory_entry_id for assignment in intent.training_plan.assignments
    } == {"active-weight"}
    assert {
        assignment.inventory_entry_id for assignment in intent.rollout_plan.assignments
    } == {"active-weight"}
    with pytest.raises(PrecisionPolicyError, match="explicitly out-of-scope"):
        compile_precision_policy(
            _policy(
                {
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.frozen.kernel",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ),
            bundle,
        )


def test_owner_request_excludes_out_of_scope_fused_sibling() -> None:
    """One physical owner does not make every semantic member a destination."""
    active = _explicit_entry(
        "active-weight",
        "main",
        "text.decoder.layer.0.active.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    owner = active.member.ownership.binding.canonical_owner_family
    excluded = _explicit_entry(
        "excluded-weight",
        "main",
        "text.decoder.layer.0.excluded.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
        binding=_binding(
            "excluded-weight",
            "main",
            _scalar_domain(),
            owner_family=owner,
        ),
    )
    bundle = _bundle(
        (active, excluded),
        owners=(_owner(active, SourceMutability.FROZEN),),
        out_of_scope={
            "main": (
                OutOfScopeTensor(
                    "excluded-weight",
                    OutOfScopeReason.SOURCE_PROVEN_FROZEN,
                ),
            )
        },
    )

    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}), bundle
    )

    assert plan.every_version_source_items == ()
    assert len(plan.startup_source_items) == 1
    request = plan.startup_source_items[0]
    assert request.owner_family == owner
    assert request.member_graph_instance_ids == ("main",)
    assert request.inventory_entry_ids == (("main", "active-weight"),)


def test_checkpoint_served_draft_has_rollout_context_but_no_source_request() -> None:
    """Static checkpoint rollout must not be turned into a source refit."""
    main = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    draft = _explicit_entry(
        "draft-weight",
        "draft.static",
        "draft.decoder.layer.0.dense.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        global_decoder_layer=0,
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    lifecycle = _checkpoint_lifecycle(
        "draft.static", GraphKind.SPECULATIVE_DRAFTER, "model-draft.static", "rev-1"
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN, RolloutParticipation.SERVED_FROM_SOURCE
        ),
        "draft.static": lifecycle,
    }
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        _bundle(
            (main, draft),
            lifecycles=lifecycles,
            mutabilities={"draft-weight": SourceMutability.FROZEN},
        ),
    )
    intent = plan.graph_intent("draft.static")
    assert intent.training_plan is None
    assert intent.rollout_plan is not None
    assert intent.rollout_plan.precision_for("draft-weight") == "bf16"
    assert intent.refit_requirement == RefitRequirement.NONE
    assert intent.immutable_checkpoint_evidence == lifecycle.immutable_evidence
    assert plan.immutable_checkpoint_contexts == (lifecycle.immutable_evidence,)
    assert all(
        item.owner_family.graph_instance_id != "draft.static"
        for item in (*plan.startup_source_items, *plan.every_version_source_items)
    )


def test_eagle_copy_head_remains_an_independent_direct_owner_request() -> None:
    main = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    eagle = _explicit_entry(
        "eagle-head-weight",
        "draft.eagle",
        "draft.decoder.layer.0.head.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        global_decoder_layer=0,
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

    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        _bundle((main, eagle), lifecycles=lifecycles),
    )

    assert plan.source_alias_contracts == ()
    assert tuple(
        request.owner_family for request in plan.every_version_source_items
    ) == tuple(
        sorted(
            (
                main.member.ownership.binding.canonical_owner_family,
                eagle.member.ownership.binding.canonical_owner_family,
            ),
            key=lambda owner: (
                0 if owner.graph_instance_id == "main" else 1,
                owner.graph_instance_id,
                owner.owner_family_id,
            ),
        )
    )


def _alias_bundle(
    *,
    canonical_value_entry_id: str = "main-weight",
    target_provenance: ValueProvenance = ValueProvenance.TRAINING_PARAMETER,
    target_mutability: SourceMutability = SourceMutability.MUTABLE,
    main_participation: RolloutParticipation = RolloutParticipation.SERVED_FROM_SOURCE,
    alias_lifecycle: GraphLifecycle | None = None,
) -> tuple[SemanticManifestBundle, OwnerFamilyReference]:
    main = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
        value_provenance=target_provenance,
    )
    owner = main.member.ownership.binding.canonical_owner_family
    alias_binding = _binding(
        "mtp-alias",
        "mtp.tied",
        _scalar_domain(),
        owner_family=owner,
        canonical_value_entry_id=canonical_value_entry_id,
    )
    alias = _explicit_entry(
        "mtp-alias",
        "mtp.tied",
        "auxiliary.mtp.layer.0.tied.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="mtp",
        global_decoder_layer=0,
        binding=alias_binding,
        value_provenance=ValueProvenance.CANONICAL_ALIAS,
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            main_participation,
        ),
        "mtp.tied": (
            _runtime_lifecycle(GraphKind.MTP, RolloutParticipation.SERVED_FROM_SOURCE)
            if alias_lifecycle is None
            else alias_lifecycle
        ),
    }
    return (
        _bundle(
            (alias, main),
            lifecycles=lifecycles,
            owners=(_owner(main, target_mutability),),
            source_alias_contracts=_identical_alias_contracts(alias, main),
        ),
        owner,
    )


def test_alias_only_auxiliary_reuses_one_canonical_owner_request() -> None:
    """A tied MTP graph must not duplicate transfer ownership."""
    bundle, owner = _alias_bundle()
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}), bundle
    )
    assert plan.graph_intent("mtp.tied").owner_refit_requirements[owner] == (
        RefitRequirement.EVERY_VERSION
    )
    assert tuple(item.owner_family for item in plan.every_version_source_items) == (
        owner,
    )
    request = plan.every_version_source_items[0]
    assert request.member_graph_instance_ids == ("main", "mtp.tied")
    assert request.inventory_entry_ids == (
        ("main", "main-weight"),
        ("mtp.tied", "mtp-alias"),
    )


def test_compiled_group_retains_typed_source_alias_contract_payload() -> None:
    identical_bundle, _ = _alias_bundle()
    entries = {entry.entry_id: entry for entry in identical_bundle.inventory.entries}
    alias = entries["mtp-alias"]
    canonical = entries["main-weight"]
    replica_bundle = replace(
        identical_bundle,
        source_alias_contracts=_replica_alias_contracts(alias, canonical),
    )

    identical_plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        identical_bundle,
    )
    replica_plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        replica_bundle,
    )

    assert identical_plan.source_alias_contracts == (
        identical_bundle.source_alias_contracts
    )
    assert replica_plan.source_alias_contracts == replica_bundle.source_alias_contracts
    identical_payload = identical_plan.to_wire_dict()["source_alias_contracts"]
    replica_payload = replica_plan.to_wire_dict()["source_alias_contracts"]
    assert isinstance(identical_payload, list)
    assert isinstance(replica_payload, list)
    assert identical_payload[0]["kind"] == "identical_storage"
    assert "storage_identity_evidence" in identical_payload[0]
    assert replica_payload[0]["kind"] == "synchronized_replica"
    assert replica_payload[0]["synchronization"] == {
        "replica_group_id": "replicas.mtp",
        "boundary": "source_version_ready",
        "evidence_source": {
            "kind": "runtime_inventory",
            "locator": "runtime://mtp-replica-synchronization",
            "digest": "sha256:mtp-replica-synchronization",
        },
    }
    assert len(replica_plan.every_version_source_items) == 1
    assert replica_plan.every_version_source_items[0].inventory_entry_ids == (
        ("main", "main-weight"),
        ("mtp.tied", "mtp-alias"),
    )


def test_source_alias_relation_changes_every_derived_topology_identity() -> None:
    identical_bundle, _ = _alias_bundle()
    entries = {entry.entry_id: entry for entry in identical_bundle.inventory.entries}
    alias = entries["mtp-alias"]
    canonical = entries["main-weight"]
    replica_a = replace(
        identical_bundle,
        source_alias_contracts=_replica_alias_contracts(alias, canonical),
    )
    replica_b = replace(
        identical_bundle,
        source_alias_contracts=_replica_alias_contracts(
            alias,
            canonical,
            replica_group_id="replicas.mtp.changed",
            evidence_name="mtp-replica-synchronization-changed",
        ),
    )
    policy = PrecisionPolicyConfig.model_validate({"scopes": []})

    plans = tuple(
        compile_precision_policy(policy, bundle)
        for bundle in (identical_bundle, replica_a, replica_b)
    )

    assert len({plan.topology_digest for plan in plans}) == 3
    for graph_id in ("main", "mtp.tied"):
        assert len({plan.graph_intent(graph_id).intent_id for plan in plans}) == 3
    assert len({plan.intent_group_id for plan in plans}) == 3


def test_source_alias_contract_order_is_canonical_in_digest_and_wire_payload() -> None:
    base, _ = _alias_bundle()
    entries = {entry.entry_id: entry for entry in base.inventory.entries}
    canonical = replace(
        entries["main-weight"],
        member=replace(entries["main-weight"].member, format=MXFP8_FORMAT),
    )
    alias = replace(
        entries["mtp-alias"],
        member=replace(entries["mtp-alias"].member, format=MXFP8_FORMAT),
    )
    contracts = _replica_alias_contracts(alias, canonical)
    forward_bundle = replace(
        base,
        inventory=ParameterInventory(
            owners=base.inventory.owners,
            entries=(canonical, alias),
        ),
        source_alias_contracts=contracts,
    )
    reverse_bundle = replace(
        forward_bundle,
        source_alias_contracts=tuple(reversed(contracts)),
    )
    policy = PrecisionPolicyConfig.model_validate({"scopes": []})

    forward = compile_precision_policy(policy, forward_bundle)
    reverse = compile_precision_policy(policy, reverse_bundle)

    assert forward.topology_digest == reverse.topology_digest
    assert forward.intent_group_id == reverse.intent_group_id
    assert forward.to_wire_dict() == reverse.to_wire_dict()
    payload = forward.to_wire_dict()["source_alias_contracts"]
    assert isinstance(payload, list)
    assert [item["component_role"] for item in payload] == [
        "block_scales",
        "values",
    ]


def test_compiled_alias_order_does_not_serialize_contracts_for_sorting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _ = _alias_bundle()
    entries = {entry.entry_id: entry for entry in base.inventory.entries}
    canonical = replace(
        entries["main-weight"],
        member=replace(entries["main-weight"].member, format=MXFP8_FORMAT),
    )
    alias = replace(
        entries["mtp-alias"],
        member=replace(entries["mtp-alias"].member, format=MXFP8_FORMAT),
    )
    contracts = _replica_alias_contracts(alias, canonical)
    bundle = replace(
        base,
        inventory=ParameterInventory(
            owners=base.inventory.owners,
            entries=(canonical, alias),
        ),
        source_alias_contracts=contracts,
    )
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        bundle,
    )

    def fail_payload(_contract: SourceAliasContract) -> dict[str, object]:
        raise AssertionError("serialized a source alias contract to sort it")

    monkeypatch.setattr(
        compiler_module,
        "_source_alias_contract_payload",
        fail_payload,
    )

    rebuilt = compiler_module._canonical_source_alias_contracts(
        tuple(reversed(contracts))
    )

    assert rebuilt == plan.source_alias_contracts


def test_compile_constructs_final_intent_group_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ = _alias_bundle()
    construction_count = 0
    original_post_init = CompiledPrecisionIntentGroup.__post_init__

    def count_construction(candidate: CompiledPrecisionIntentGroup) -> None:
        nonlocal construction_count
        construction_count += 1
        original_post_init(candidate)

    monkeypatch.setattr(
        CompiledPrecisionIntentGroup,
        "__post_init__",
        count_construction,
    )

    compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        bundle,
    )

    assert construction_count == 1


def test_compiled_alias_sort_caches_large_shared_domain_keys() -> None:
    bundle, _ = _alias_bundle()
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        bundle,
    )
    iterations = 0

    class CountingMembers(tuple[int, ...]):
        def __iter__(self) -> Iterator[int]:
            nonlocal iterations
            for member in super().__iter__():
                iterations += 1
                if iterations > 20_000:
                    raise AssertionError("rebuilt a shared compact-domain sort key")
                yield member

    axis = AxisDomain("expert", tuple(range(10_000)))
    object.__setattr__(axis, "members", CountingMembers(axis.members))
    shared_domain = FamilyIndexDomain(None, (axis,))
    base_contract = plan.source_alias_contracts[0]
    contracts = tuple(
        replace(
            base_contract,
            component_role=ComponentRole(f"component-{index:05d}"),
            alias_domain=shared_domain,
            canonical_domain=shared_domain,
            alias_to_canonical_axes=(AxisProjection("expert", "expert"),),
        )
        for index in reversed(range(10_000))
    )

    rebuilt = compiler_module._canonical_source_alias_contracts(contracts)

    assert str(rebuilt[0].component_role) == "component-00000"
    assert iterations <= 20_000


def test_renamed_alias_projection_order_is_canonical_in_digest_and_wire() -> None:
    alias_domain = FamilyIndexDomain(
        layer_domain=None,
        independent_axes=(
            AxisDomain("alias_alpha", (0, 1)),
            AxisDomain("alias_zeta", (10, 11, 12)),
        ),
    )
    canonical_domain = FamilyIndexDomain(
        layer_domain=None,
        independent_axes=(
            AxisDomain("canonical_alpha", (10, 11, 12)),
            AxisDomain("canonical_zeta", (0, 1)),
        ),
    )
    projections = (
        AxisProjection("alias_zeta", "canonical_alpha"),
        AxisProjection("alias_alpha", "canonical_zeta"),
    )

    def family_entry(
        entry_id: str,
        graph_instance_id: str,
        graph_path: str,
        domain: FamilyIndexDomain,
        binding: OwnerFamilyBinding,
        provenance: ValueProvenance,
    ) -> ParameterInventoryEntry:
        return ParameterInventoryEntry(
            entry_id=entry_id,
            graph_instance_id=graph_instance_id,
            member=SemanticTensorFamily(
                pattern=SemanticAddressPattern(
                    semantic_graph_path=graph_path,
                    path_segments=(
                        LiteralPathSegment("matrix"),
                        *(IndexPathSegment(axis) for axis in domain.axis_names),
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
            ),
            value_provenance=provenance,
        )

    canonical_binding = _binding(
        "main-renamed-family",
        "main",
        canonical_domain,
    )
    canonical = family_entry(
        "main-renamed-family",
        "main",
        "text.decoder",
        canonical_domain,
        canonical_binding,
        ValueProvenance.TRAINING_PARAMETER,
    )
    lifecycles = {
        "main": _runtime_lifecycle(
            GraphKind.MAIN,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
        "mtp.renamed": _runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
    }

    def make_bundle(reverse: bool) -> SemanticManifestBundle:
        ordered = tuple(reversed(projections)) if reverse else projections
        alias_binding = _binding(
            "mtp-renamed-family",
            "mtp.renamed",
            alias_domain,
            owner_family=canonical_binding.canonical_owner_family,
            canonical_value_entry_id=canonical.entry_id,
            member_to_owner_axes=ordered,
            member_to_value_axes=ordered,
        )
        alias = family_entry(
            "mtp-renamed-family",
            "mtp.renamed",
            "auxiliary.mtp",
            alias_domain,
            alias_binding,
            ValueProvenance.CANONICAL_ALIAS,
        )
        contract = IdenticalStorageSourceAliasContract(
            alias_entry_id=alias.entry_id,
            canonical_value_entry_id=canonical.entry_id,
            canonical_owner_family=canonical_binding.canonical_owner_family,
            component_role=ComponentRole("logical_values"),
            alias_domain=alias_domain,
            canonical_domain=canonical_domain,
            alias_to_canonical_axes=tuple(reversed(ordered)),
            storage_identity_evidence=_evidence("renamed-storage"),
        )
        return _bundle(
            (canonical, alias),
            lifecycles=lifecycles,
            owners=(_owner(canonical),),
            source_alias_contracts=(contract,),
        )

    policy = PrecisionPolicyConfig.model_validate({"scopes": []})
    forward = compile_precision_policy(policy, make_bundle(False))
    reverse = compile_precision_policy(policy, make_bundle(True))

    assert forward.topology_digest == reverse.topology_digest
    assert forward.intent_group_id == reverse.intent_group_id
    assert forward.to_wire_dict() == reverse.to_wire_dict()
    payload = forward.to_wire_dict()["source_alias_contracts"]
    assert isinstance(payload, list)
    assert payload[0]["alias_to_canonical_axes"] == [
        {"member_axis": "alias_alpha", "owner_axis": "canonical_zeta"},
        {"member_axis": "alias_zeta", "owner_axis": "canonical_alpha"},
    ]


def test_compiled_group_rejects_duplicate_source_alias_contracts() -> None:
    bundle, _ = _alias_bundle()
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        bundle,
    )
    contract = plan.source_alias_contracts[0]

    with pytest.raises(ValueError, match="duplicate source alias contract"):
        replace(plan, source_alias_contracts=(contract, contract))


def test_replacing_compiled_group_semantics_recomputes_group_id() -> None:
    bundle, _ = _alias_bundle()
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        bundle,
    )
    contract = plan.source_alias_contracts[0]
    assert isinstance(contract, IdenticalStorageSourceAliasContract)

    changed = replace(
        plan,
        source_alias_contracts=(
            replace(
                contract,
                storage_identity_evidence=_evidence("changed-storage-identity"),
            ),
        ),
    )

    assert changed.intent_group_id != plan.intent_group_id


def test_alias_request_keeps_exact_locally_excluded_canonical_source() -> None:
    """Local exclusion removes a destination, not an alias's source descriptor."""
    base, owner = _alias_bundle(
        target_mutability=SourceMutability.FROZEN,
        main_participation=RolloutParticipation.NOT_SERVED,
    )
    sibling = _explicit_entry(
        "main-sibling",
        "main",
        "text.decoder.layer.0.sibling.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
        binding=_binding(
            "main-sibling",
            "main",
            _scalar_domain(),
            owner_family=owner,
        ),
    )
    bundle = _bundle(
        base.inventory.entries + (sibling,),
        lifecycles={
            manifest.graph_instance_id: manifest.lifecycle
            for manifest in base.manifests
        },
        owners=base.inventory.owners,
        out_of_scope={
            "main": (
                OutOfScopeTensor(
                    "main-weight",
                    OutOfScopeReason.SOURCE_PROVEN_FROZEN,
                ),
                OutOfScopeTensor(
                    "main-sibling",
                    OutOfScopeReason.SOURCE_PROVEN_FROZEN,
                ),
            )
        },
        source_alias_contracts=base.source_alias_contracts,
    )

    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}), bundle
    )

    assert plan.every_version_source_items == ()
    assert len(plan.startup_source_items) == 1
    request = plan.startup_source_items[0]
    assert request.owner_family == owner
    assert request.member_graph_instance_ids == ("main", "mtp.tied")
    assert request.inventory_entry_ids == (
        ("main", "main-weight"),
        ("mtp.tied", "mtp-alias"),
    )


@pytest.mark.parametrize(
    ("mutability", "expected", "request_collection"),
    [
        (
            SourceMutability.MUTABLE,
            RefitRequirement.EVERY_VERSION,
            "every_version_source_items",
        ),
        (
            SourceMutability.FROZEN,
            RefitRequirement.INITIAL_ONLY,
            "startup_source_items",
        ),
    ],
)
def test_checkpoint_served_training_alias_uses_semantic_owner_cadence(
    mutability: SourceMutability,
    expected: RefitRequirement,
    request_collection: str,
) -> None:
    lifecycle = _checkpoint_lifecycle(
        "mtp.tied",
        GraphKind.MTP,
        "model-mtp.tied",
        "rev-mtp-tied",
    )
    bundle, owner = _alias_bundle(
        target_mutability=mutability,
        alias_lifecycle=lifecycle,
    )

    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        bundle,
    )

    intent = plan.graph_intent("mtp.tied")
    assert intent.owner_refit_requirements[owner] == expected
    assert intent.refit_requirement == expected
    requests = getattr(plan, request_collection)
    assert len(requests) == 1
    assert requests[0].owner_family == owner
    assert requests[0].member_graph_instance_ids == ("main", "mtp.tied")
    assert requests[0].inventory_entry_ids == (
        ("main", "main-weight"),
        ("mtp.tied", "mtp-alias"),
    )
    assert intent.immutable_checkpoint_evidence == lifecycle.immutable_evidence
    assert plan.immutable_checkpoint_contexts == (lifecycle.immutable_evidence,)


def test_checkpoint_served_checkpoint_authority_alias_adds_no_source_request() -> None:
    lifecycle = _checkpoint_lifecycle(
        "mtp.tied",
        GraphKind.MTP,
        "model-mtp.tied",
        "rev-mtp-tied",
    )
    bundle, owner = _alias_bundle(
        target_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
        target_mutability=SourceMutability.FROZEN,
        main_participation=RolloutParticipation.NOT_SERVED,
        alias_lifecycle=lifecycle,
    )

    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        bundle,
    )

    intent = plan.graph_intent("mtp.tied")
    assert intent.owner_refit_requirements[owner] == RefitRequirement.NONE
    assert intent.refit_requirement == RefitRequirement.NONE
    assert plan.startup_source_items == ()
    assert plan.every_version_source_items == ()
    assert plan.immutable_checkpoint_contexts == (lifecycle.immutable_evidence,)


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
    bundle, _ = _alias_bundle(
        target_provenance=target_provenance,
        target_mutability=SourceMutability.FROZEN,
        main_participation=RolloutParticipation.NOT_SERVED,
    )

    with pytest.raises(ValueError, match="must reach.*training parameter"):
        compile_precision_policy(
            PrecisionPolicyConfig.model_validate({"scopes": []}),
            bundle,
        )


@pytest.mark.parametrize(
    "nontraining_provenance",
    (
        ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
        ValueProvenance.BACKEND_DERIVED,
    ),
)
def test_mixed_source_alias_graph_requests_only_training_authority(
    nontraining_provenance: ValueProvenance,
) -> None:
    domain = _scalar_domain()
    training = _explicit_entry(
        "main-training",
        "main",
        "text.decoder.layer.0.training.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    nontraining = _explicit_entry(
        "main-nontraining",
        "main",
        "text.decoder.layer.0.nontraining.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
        value_provenance=nontraining_provenance,
    )

    def alias(
        target: ParameterInventoryEntry, entry_id: str
    ) -> ParameterInventoryEntry:
        binding = _binding(
            entry_id,
            "mtp.tied",
            domain,
            owner_family=target.member.ownership.binding.canonical_owner_family,
            canonical_value_entry_id=target.entry_id,
        )
        return _explicit_entry(
            entry_id,
            "mtp.tied",
            f"auxiliary.mtp.{entry_id}.kernel",
            semantic_graph_path="auxiliary.mtp",
            model_part="mtp",
            global_decoder_layer=0,
            binding=binding,
            value_provenance=ValueProvenance.CANONICAL_ALIAS,
        )

    training_alias = alias(training, "training-alias")
    nontraining_alias = alias(nontraining, "nontraining-alias")
    training_owner = training.member.ownership.binding.canonical_owner_family
    nontraining_owner = nontraining.member.ownership.binding.canonical_owner_family
    bundle = _bundle(
        (training, nontraining, training_alias, nontraining_alias),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.NOT_SERVED,
            ),
            "mtp.tied": _runtime_lifecycle(
                GraphKind.MTP,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
        },
        mutabilities={"main-nontraining": SourceMutability.FROZEN},
        source_alias_contracts=(
            *_identical_alias_contracts(training_alias, training),
            *_identical_alias_contracts(nontraining_alias, nontraining),
        ),
    )

    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        bundle,
    )

    intent = plan.graph_intent("mtp.tied")
    assert intent.owner_refit_requirements[training_owner] == (
        RefitRequirement.EVERY_VERSION
    )
    assert intent.owner_refit_requirements[nontraining_owner] == RefitRequirement.NONE
    assert intent.refit_requirement == RefitRequirement.EVERY_VERSION
    assert plan.startup_source_items == ()
    assert len(plan.every_version_source_items) == 1
    request = plan.every_version_source_items[0]
    assert request.owner_family == training_owner
    assert request.member_graph_instance_ids == ("main", "mtp.tied")
    assert request.inventory_entry_ids == (
        ("main", "main-training"),
        ("mtp.tied", "training-alias"),
    )


def test_compile_validates_semantic_bundle_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ = _alias_bundle()
    validation_calls = [0]
    validate_complete = SemanticManifestBundle.validate_complete

    def count_validation(candidate: SemanticManifestBundle) -> None:
        validation_calls[0] += 1
        validate_complete(candidate)

    monkeypatch.setattr(
        SemanticManifestBundle,
        "validate_complete",
        count_validation,
    )

    compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        bundle,
    )

    assert validation_calls == [1]


def test_not_served_alias_is_excluded_from_its_source_owner_request() -> None:
    bundle, owner = _alias_bundle(
        alias_lifecycle=_runtime_lifecycle(
            GraphKind.MTP,
            RolloutParticipation.NOT_SERVED,
        )
    )

    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        bundle,
    )

    alias_intent = plan.graph_intent("mtp.tied")
    assert alias_intent.owner_refit_requirements[owner] == RefitRequirement.NONE
    assert alias_intent.refit_requirement == RefitRequirement.NONE
    assert plan.startup_source_items == ()
    assert len(plan.every_version_source_items) == 1
    assert plan.every_version_source_items[0].member_graph_instance_ids == ("main",)
    assert plan.every_version_source_items[0].inventory_entry_ids == (
        ("main", "main-weight"),
    )


def test_compile_rejects_mutable_direct_checkpoint_body() -> None:
    main = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    body = _explicit_entry(
        "mtp-body",
        "mtp.static",
        "auxiliary.mtp.body.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="mtp",
        global_decoder_layer=0,
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    lifecycle = _checkpoint_lifecycle(
        "mtp.static",
        GraphKind.MTP,
        "model-mtp.static",
        "rev-mtp-static",
    )
    bundle = _bundle(
        (main, body),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN,
                RolloutParticipation.SERVED_FROM_SOURCE,
            ),
            "mtp.static": lifecycle,
        },
    )

    with pytest.raises(ValueError, match="checkpoint-served.*mutable direct"):
        compile_precision_policy(
            PrecisionPolicyConfig.model_validate({"scopes": []}),
            bundle,
        )


def test_compile_revalidates_unresolved_alias_before_emitting_intent() -> None:
    """A stale alias target must fail before owner requests are produced."""
    bundle, _ = _alias_bundle(canonical_value_entry_id="missing-target")
    with pytest.raises(ValueError, match="canonical value target is missing"):
        compile_precision_policy(
            PrecisionPolicyConfig.model_validate({"scopes": []}), bundle
        )


def test_compile_revalidates_alias_to_alias_target() -> None:
    """A direct-target contract cannot degrade into a hidden alias chain."""
    bundle, owner = _alias_bundle()
    first_alias = next(
        entry for entry in bundle.inventory.entries if entry.entry_id == "mtp-alias"
    )
    second_alias_binding = _binding(
        "mtp-second-alias",
        "mtp.tied",
        _scalar_domain(),
        owner_family=owner,
        canonical_value_entry_id="main-weight",
    )
    second_alias = _explicit_entry(
        "mtp-second-alias",
        "mtp.tied",
        "auxiliary.mtp.layer.0.second-tied.kernel",
        semantic_graph_path="auxiliary.mtp",
        model_part="mtp",
        global_decoder_layer=0,
        binding=second_alias_binding,
        value_provenance=ValueProvenance.CANONICAL_ALIAS,
    )
    chained_first = replace(
        first_alias,
        member=replace(
            first_alias.member,
            ownership=SemanticOwnership(
                replace(
                    first_alias.member.ownership.binding,
                    canonical_value_entry_id="mtp-second-alias",
                )
            ),
        ),
    )
    main = next(
        entry for entry in bundle.inventory.entries if entry.entry_id == "main-weight"
    )
    invalid = _bundle(
        (chained_first, second_alias, main),
        lifecycles={
            manifest.graph_instance_id: manifest.lifecycle
            for manifest in bundle.manifests
        },
        owners=bundle.inventory.owners,
    )
    with pytest.raises(ValueError, match="alias-to-alias target is forbidden"):
        compile_precision_policy(
            PrecisionPolicyConfig.model_validate({"scopes": []}), invalid
        )


def test_compile_revalidates_incompatible_alias_format() -> None:
    """Matching owner identity cannot excuse an incompatible alias encoding."""
    bundle, _ = _alias_bundle()
    alias = next(
        entry for entry in bundle.inventory.entries if entry.entry_id == "mtp-alias"
    )
    incompatible_alias = replace(
        alias,
        member=replace(alias.member, format=MXFP8_FORMAT),
    )
    main = next(
        entry for entry in bundle.inventory.entries if entry.entry_id == "main-weight"
    )
    invalid = _bundle(
        (incompatible_alias, main),
        lifecycles={
            manifest.graph_instance_id: manifest.lifecycle
            for manifest in bundle.manifests
        },
        owners=bundle.inventory.owners,
    )
    with pytest.raises(ValueError, match="format mismatch"):
        compile_precision_policy(
            PrecisionPolicyConfig.model_validate({"scopes": []}), invalid
        )


def test_invalid_checkpoint_evidence_relation_fails_before_intent() -> None:
    """A declaration/manifest revision mismatch cannot enter serving context."""
    main = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    draft = _explicit_entry(
        "draft-weight",
        "draft.static",
        "draft.decoder.layer.0.dense.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        global_decoder_layer=0,
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    static = _checkpoint_lifecycle(
        "draft.static", GraphKind.SPECULATIVE_DRAFTER, "model-draft.static", "rev-1"
    )
    bundle = _bundle(
        (main, draft),
        lifecycles={
            "main": _runtime_lifecycle(
                GraphKind.MAIN, RolloutParticipation.SERVED_FROM_SOURCE
            ),
            "draft.static": static,
        },
        mutabilities={"draft-weight": SourceMutability.FROZEN},
    )
    static_manifest = copy(bundle.manifest("draft.static"))
    object.__setattr__(static_manifest, "model_revision", "wrong-revision")
    bad_manifests = tuple(
        static_manifest if manifest.graph_instance_id == "draft.static" else manifest
        for manifest in bundle.manifests
    )
    with pytest.raises(ValueError, match="pinned checkpoint revision"):
        compile_precision_policy(
            PrecisionPolicyConfig.model_validate({"scopes": []}),
            replace(bundle, manifests=bad_manifests),
        )


def _atomic_participant(
    entry_id: str,
    domain: FamilyIndexDomain,
    *,
    projections: tuple[AxisProjection, ...] | None = None,
) -> AtomicGroupParticipant:
    return AtomicGroupParticipant(
        inventory_entry_id=entry_id,
        participant_domain=domain,
        group_to_participant_axes=projections or _identity_axes(domain),
    )


def _chained_expert_atomic_bundle() -> SemanticManifestBundle:
    domain = _layer_domain((0, 1), moe_ordinals=(0, 1), experts=(0, 1))
    entries = tuple(
        _family_entry(f"routed-{projection}", projection, domain)
        for projection in ("gate", "up", "down")
    )
    groups = (
        AtomicGroup(
            group_id="moe.gate-up",
            graph_instance_id="main",
            kind=AtomicGroupKind.PRECISION,
            group_domain=domain,
            participants=(
                _atomic_participant("routed-gate", domain),
                _atomic_participant("routed-up", domain),
            ),
        ),
        AtomicGroup(
            group_id="moe.up-down",
            graph_instance_id="main",
            kind=AtomicGroupKind.PRECISION,
            group_domain=domain,
            participants=(
                _atomic_participant("routed-up", domain),
                _atomic_participant("routed-down", domain),
            ),
        ),
    )
    return _bundle(entries, atomic_groups={"main": groups})


def _qkv_atomic_bundle() -> SemanticManifestBundle:
    domain = _layer_domain((0, 1))
    entries = tuple(
        _attention_entry(f"attention-{projection}", projection, domain)
        for projection in ("q", "k", "v")
    )
    group = AtomicGroup(
        group_id="attention.qkv",
        graph_instance_id="main",
        kind=AtomicGroupKind.PRECISION,
        group_domain=domain,
        participants=tuple(
            _atomic_participant(f"attention-{projection}", domain)
            for projection in ("q", "k", "v")
        ),
    )
    return _bundle(entries, atomic_groups={"main": (group,)})


@pytest.mark.parametrize(
    "second_endpoint_values, expected_endpoint",
    [
        ({"training": "bf16", "rollout": "mxfp8"}, "training"),
        ({"training": "mxfp8", "rollout": "bf16"}, "rollout"),
    ],
)
def test_conflicting_scopes_fail_independently_on_each_participating_endpoint(
    second_endpoint_values: dict[str, str], expected_endpoint: str
) -> None:
    """Checking conflicts on only one endpoint would silently accept the other."""
    address = {
        "graph_instance_id": "main",
        "semantic_graph_path": "text.decoder",
        "semantic_id": "text.decoder.layer.1.expert.0.gate",
    }
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "both-mxfp8",
                    "addresses": [address],
                    "training": "mxfp8",
                    "rollout": "mxfp8",
                },
                {
                    "id": "one-bf16",
                    "addresses": [address],
                    **second_endpoint_values,
                },
            ]
        }
    )
    with pytest.raises(
        PrecisionPolicyError,
        match=f"conflicting precision scopes.*{expected_endpoint}",
    ):
        compile_precision_policy(policy, _sparse_moe_bundle())


@pytest.mark.parametrize("endpoint", ["training", "rollout"])
def test_atomic_error_rejects_partial_fused_qkv_selection(endpoint: str) -> None:
    """Selecting Q alone must not silently realize an inseparable QKV owner."""
    with pytest.raises(PrecisionPolicyError, match="atomic precision conflict"):
        compile_precision_policy(
            _policy(
                {
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.attention.q",
                        }
                    ],
                    endpoint: "mxfp8",
                }
            ),
            _qkv_atomic_bundle(),
        )


@pytest.mark.parametrize("endpoint", ["training", "rollout"])
def test_atomic_expand_reaches_pointwise_fixed_point_without_neighbor_expansion(
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping after one group or expanding a whole family corrupts precision scope."""

    def fail_renderer(self: SemanticTensorFamily) -> None:
        raise AssertionError("atomic closure materialized semantic family members")

    monkeypatch.setattr(SemanticTensorFamily, "iter_semantic_ids", fail_renderer)
    plan = compile_precision_policy(
        _policy(
            {
                "addresses": [
                    {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "semantic_id": "text.decoder.layer.0.expert.0.gate",
                    }
                ],
                endpoint: "mxfp8",
                "atomic_conflict": "expand",
            }
        ),
        _chained_expert_atomic_bundle(),
    )
    intent = plan.graph_intent("main")
    endpoint_plan = (
        intent.training_plan if endpoint == "training" else intent.rollout_plan
    )
    other_plan = intent.rollout_plan if endpoint == "training" else intent.training_plan
    assert endpoint_plan is not None
    assert other_plan is not None
    for entry_id in ("routed-gate", "routed-up", "routed-down"):
        assert (
            endpoint_plan.precision_for(
                entry_id, global_decoder_layer=0, independent_axes={"expert": 0}
            )
            == "mxfp8"
        )
        assert (
            endpoint_plan.precision_for(
                entry_id, global_decoder_layer=0, independent_axes={"expert": 1}
            )
            == "bf16"
        )
        assert (
            endpoint_plan.precision_for(
                entry_id, global_decoder_layer=1, independent_axes={"expert": 0}
            )
            == "bf16"
        )
        assert (
            other_plan.precision_for(
                entry_id, global_decoder_layer=0, independent_axes={"expert": 0}
            )
            == "bf16"
        )
    assert (
        sum(
            addition.logical_cardinality
            for expansion in plan.atomic_expansions
            for addition in expansion.additions
        )
        == 2
    )
    assert {item.atomic_group_id for item in plan.atomic_expansions} == {
        "moe.gate-up",
        "moe.up-down",
    }


def test_scope_atomic_override_beats_policy_inherited_expand() -> None:
    """Materializing the policy default into every scope would erase local error."""
    raw_scope: dict[str, object] = {
        "addresses": [
            {
                "graph_instance_id": "main",
                "semantic_graph_path": "text.decoder",
                "semantic_id": "text.decoder.layer.0.attention.q",
            }
        ],
        "rollout": "mxfp8",
    }
    inherited = compile_precision_policy(
        _policy(raw_scope, atomic_conflict="expand"), _qkv_atomic_bundle()
    )
    rollout = inherited.graph_intent("main").rollout_plan
    assert rollout is not None
    assert rollout.precision_for("attention-k", global_decoder_layer=0) == "mxfp8"
    with pytest.raises(PrecisionPolicyError, match="atomic precision conflict"):
        compile_precision_policy(
            _policy(
                {**raw_scope, "atomic_conflict": "error"},
                atomic_conflict="expand",
            ),
            _qkv_atomic_bundle(),
        )


@pytest.mark.parametrize(
    "expand_scope_id,error_scope_id",
    [("a-expand", "z-error"), ("z-expand", "a-error")],
)
def test_atomic_error_is_strictest_independent_of_lexical_scope_order(
    expand_scope_id: str,
    error_scope_id: str,
) -> None:
    """An earlier expansion must not hide a later partial error-mode trigger."""
    address = {
        "graph_instance_id": "main",
        "semantic_graph_path": "text.decoder",
        "semantic_id": "text.decoder.layer.0.attention.q",
    }
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": expand_scope_id,
                    "addresses": [address],
                    "rollout": "mxfp8",
                    "atomic_conflict": "expand",
                },
                {
                    "id": error_scope_id,
                    "addresses": [address],
                    "rollout": "mxfp8",
                    "atomic_conflict": "error",
                },
            ]
        }
    )
    with pytest.raises(PrecisionPolicyError, match="atomic precision conflict"):
        compile_precision_policy(policy, _qkv_atomic_bundle())


def test_atomic_expansion_digest_is_independent_of_scope_input_order() -> None:
    """Canonical expansion identity must not depend on mapping/list insertion."""
    address = {
        "graph_instance_id": "main",
        "semantic_graph_path": "text.decoder",
        "semantic_id": "text.decoder.layer.0.attention.q",
    }
    scopes: list[dict[str, object]] = [
        {
            "id": scope_id,
            "addresses": [address],
            "rollout": "mxfp8",
            "atomic_conflict": "expand",
        }
        for scope_id in ("first", "second")
    ]
    first = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": scopes}),
        _qkv_atomic_bundle(),
    )
    second = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": list(reversed(scopes))}),
        _qkv_atomic_bundle(),
    )
    assert first.policy_digest == second.policy_digest
    assert first.intent_group_id == second.intent_group_id
    assert first.to_wire_dict() == second.to_wire_dict()


def test_atomic_expansion_rejects_required_out_of_scope_participant() -> None:
    """A successful closure cannot omit a participant from endpoint assignments."""
    base = _qkv_atomic_bundle()
    bundle = _bundle(
        base.inventory.entries,
        owners=tuple(
            replace(
                owner,
                source_mutability=(
                    SourceMutability.FROZEN
                    if owner.owner_family.owner_family_id == "owner-attention-k"
                    else owner.source_mutability
                ),
            )
            for owner in base.inventory.owners
        ),
        atomic_groups={"main": base.manifests[0].atomic_groups},
        out_of_scope={
            "main": (
                OutOfScopeTensor("attention-k", OutOfScopeReason.SOURCE_PROVEN_FROZEN),
            )
        },
    )
    with pytest.raises(
        PrecisionPolicyError,
        match="requires explicitly out-of-scope participant.*attention-k",
    ):
        compile_precision_policy(
            _policy(
                {
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.attention.q",
                        }
                    ],
                    "rollout": "mxfp8",
                    "atomic_conflict": "expand",
                }
            ),
            bundle,
        )


def test_atomic_expansion_rejects_new_conflict_with_explicit_bf16() -> None:
    """Closure must recheck conflicts that were absent from raw selections."""
    q_address = {
        "graph_instance_id": "main",
        "semantic_graph_path": "text.decoder",
        "semantic_id": "text.decoder.layer.0.attention.q",
    }
    k_address = {
        "graph_instance_id": "main",
        "semantic_graph_path": "text.decoder",
        "semantic_id": "text.decoder.layer.0.attention.k",
    }
    policy = PrecisionPolicyConfig.model_validate(
        {
            "atomic_conflict": "expand",
            "scopes": [
                {
                    "id": "q-mxfp8",
                    "addresses": [q_address],
                    "training": "mxfp8",
                    "rollout": "mxfp8",
                },
                {
                    "id": "k-rollout-bf16",
                    "addresses": [k_address],
                    "training": "mxfp8",
                    "rollout": "bf16",
                },
            ],
        }
    )
    with pytest.raises(PrecisionPolicyError, match="explicit BF16 precision"):
        compile_precision_policy(policy, _qkv_atomic_bundle())


def test_atomic_expansion_cannot_cross_another_scopes_bf16_boundary() -> None:
    """Atomic closure must preserve a hard first-layer BF16 fence."""
    policy = PrecisionPolicyConfig.model_validate(
        {
            "atomic_conflict": "expand",
            "scopes": [
                {
                    "id": "kv-middle",
                    "advanced_match": {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "module_kind": "attention.projection",
                        "attributes": {"projection": ["k", "v"]},
                    },
                    "layers": {"exclude_first": 1},
                    "rollout": "mxfp8",
                },
                {
                    "id": "q-first",
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.attention.q",
                        }
                    ],
                    "rollout": "mxfp8",
                },
            ],
        }
    )
    with pytest.raises(PrecisionPolicyError, match="hard BF16 layer boundary"):
        compile_precision_policy(policy, _qkv_atomic_bundle())


def _attribute_entry(
    entry_id: str,
    semantic_id: str,
    attribute_name: str,
    attribute_value: str | int | float | bool,
) -> ParameterInventoryEntry:
    entry = _explicit_entry(
        entry_id,
        "main",
        semantic_id,
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    assert isinstance(entry.member, SemanticTensor)
    return replace(
        entry,
        member=replace(
            entry.member,
            address=replace(
                entry.member.address,
                attributes=((attribute_name, attribute_value),),
            ),
        ),
    )


def test_policy_and_intent_digests_ignore_scope_and_mapping_input_order() -> None:
    """Input insertion order must not become a cross-rank plan identity."""
    bundle = _sparse_moe_bundle()
    scopes = [
        {
            "id": "gate",
            "advanced_match": {
                "graph_instance_id": "main",
                "semantic_graph_path": "text.decoder",
                "attributes": {
                    "projection": "gate",
                    "expert_kind": "routed",
                },
            },
            "rollout": "mxfp8",
        },
        {
            "id": "up",
            "advanced_match": {
                "attributes": {
                    "expert_kind": "routed",
                    "projection": "up",
                },
                "semantic_graph_path": "text.decoder",
                "graph_instance_id": "main",
            },
            "rollout": "mxfp8",
        },
    ]
    reordered_scopes: list[dict[str, object]] = []
    for scope in reversed(scopes):
        advanced = scope["advanced_match"]
        assert isinstance(advanced, dict)
        attributes = advanced["attributes"]
        assert isinstance(attributes, dict)
        reordered_advanced: dict[str, object] = dict(reversed(tuple(advanced.items())))
        reordered_advanced["attributes"] = dict(reversed(tuple(attributes.items())))
        reordered_scope: dict[str, object] = dict(reversed(tuple(scope.items())))
        reordered_scope["advanced_match"] = reordered_advanced
        reordered_scopes.append(reordered_scope)
    first = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": scopes}), bundle
    )
    second = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": reordered_scopes}),
        replace(
            bundle,
            inventory=ParameterInventory(
                owners=tuple(reversed(bundle.inventory.owners)),
                entries=tuple(reversed(bundle.inventory.entries)),
            ),
            manifests=tuple(reversed(bundle.manifests)),
            expected_graphs=tuple(reversed(bundle.expected_graphs)),
            role_definitions=tuple(reversed(bundle.role_definitions)),
        ),
    )
    assert first.topology_digest == second.topology_digest
    assert first.policy_digest == second.policy_digest
    assert tuple(item.intent_id for item in first.graph_intents) == tuple(
        item.intent_id for item in second.graph_intents
    )
    assert first.intent_group_id == second.intent_group_id
    assert first.to_wire_dict() == second.to_wire_dict()


def test_signed_zero_normalizes_but_bool_int_and_float_remain_distinct() -> None:
    """Untyped JSON scalar hashing would collapse semantically distinct matches."""
    zero = _attribute_entry(
        "zero-float", "text.decoder.layer.0.zero-float.kernel", "marker", 0.0
    )
    false = _attribute_entry(
        "false-bool", "text.decoder.layer.0.false-bool.kernel", "marker", False
    )
    integer = _attribute_entry(
        "zero-int", "text.decoder.layer.0.zero-int.kernel", "marker", 0
    )
    bundle = _bundle((zero, false, integer))

    def compile_marker(value: float | int | bool) -> CompiledPrecisionIntentGroup:
        return compile_precision_policy(
            _policy(
                {
                    "advanced_match": {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "attributes": {"marker": value},
                    },
                    "rollout": "mxfp8",
                }
            ),
            bundle,
        )

    negative_zero = compile_marker(-0.0)
    positive_zero = compile_marker(0.0)
    false_plan = compile_marker(False)
    integer_plan = compile_marker(0)
    assert negative_zero.policy_digest == positive_zero.policy_digest
    assert (
        negative_zero.graph_intent("main").intent_id
        == positive_zero.graph_intent("main").intent_id
    )
    assert negative_zero.intent_group_id == positive_zero.intent_group_id
    assert negative_zero.scope_result("scope").graph_result(
        "main"
    ).matched_inventory_entry_ids == ("zero-float",)
    assert false_plan.scope_result("scope").graph_result(
        "main"
    ).matched_inventory_entry_ids == ("false-bool",)
    assert integer_plan.scope_result("scope").graph_result(
        "main"
    ).matched_inventory_entry_ids == ("zero-int",)
    assert (
        len(
            {
                negative_zero.policy_digest,
                false_plan.policy_digest,
                integer_plan.policy_digest,
            }
        )
        == 3
    )
    assert (
        len(
            {
                negative_zero.intent_group_id,
                false_plan.intent_group_id,
                integer_plan.intent_group_id,
            }
        )
        == 3
    )


def test_topology_revision_and_lifecycle_change_all_derived_identity() -> None:
    """Caller-supplied/stale topology IDs would miss semantic topology changes."""
    entry = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    source_bundle = _bundle((entry,))
    not_served_lifecycle = _runtime_lifecycle(
        GraphKind.MAIN, RolloutParticipation.NOT_SERVED
    )
    not_served_bundle = _bundle((entry,), lifecycles={"main": not_served_lifecycle})
    revision_bundle = replace(
        source_bundle,
        manifests=(
            replace(source_bundle.manifests[0], model_revision="different-revision"),
        ),
    )
    policy = PrecisionPolicyConfig.model_validate({"scopes": []})
    plans = tuple(
        compile_precision_policy(policy, bundle)
        for bundle in (source_bundle, not_served_bundle, revision_bundle)
    )
    assert len({plan.topology_digest for plan in plans}) == 3
    assert len({plan.graph_intents[0].intent_id for plan in plans}) == 3
    assert len({plan.intent_group_id for plan in plans}) == 3


def test_component_axis_shape_changes_topology_digest_and_payload() -> None:
    base = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )

    def encoded_format(divisor: int) -> FormatDescriptor:
        return FormatDescriptor(
            "test.packed.v1",
            "test.packed",
            (
                ComponentDescriptor(
                    ComponentRole("packed_values"),
                    "uint8",
                    encoding="packed",
                    component_axes=(
                        LogicalComponentAxisSpec("output_features"),
                        LogicalComponentAxisSpec(
                            "input_features",
                            divisor=divisor,
                            rounding=AxisExtentRounding.EXACT,
                        ),
                        LiteralComponentAxisSpec("metadata", 2),
                    ),
                ),
            ),
        )

    first_format = encoded_format(2)
    second_format = encoded_format(4)
    first = replace(base, member=replace(base.member, format=first_format))
    second = replace(base, member=replace(base.member, format=second_format))
    policy = PrecisionPolicyConfig.model_validate({"scopes": []})

    first_plan = compile_precision_policy(policy, _bundle((first,)))
    second_plan = compile_precision_policy(policy, _bundle((second,)))

    assert first_plan.topology_digest != second_plan.topology_digest
    assert compiler_module._format_payload(first_format) == {
        "format_id": "test.packed.v1",
        "family": "test.packed",
        "components": [
            {
                "role": "packed_values",
                "dtype": "uint8",
                "encoding": "packed",
                "component_axes": {
                    "kind": "explicit",
                    "axes": [
                        {
                            "kind": "logical",
                            "logical_axis": "output_features",
                            "divisor": 1,
                            "rounding": "exact",
                        },
                        {
                            "kind": "logical",
                            "logical_axis": "input_features",
                            "divisor": 2,
                            "rounding": "exact",
                        },
                        {
                            "kind": "literal",
                            "axis_name": "metadata",
                            "extent": 2,
                        },
                    ],
                },
            }
        ],
    }


def test_builtin_format_wire_payloads_use_canonical_encodings() -> None:
    assert compiler_module._format_payload(BF16_FORMAT) == {
        "format_id": "bf16.logical.v1",
        "family": "bf16",
        "components": [
            {
                "role": "logical_values",
                "dtype": "bfloat16",
                "encoding": "plain_bfloat16",
                "component_axes": {"kind": "identity"},
            }
        ],
    }
    assert compiler_module._format_payload(MXFP8_FORMAT) == {
        "format_id": "mxfp8.e4m3-e8m0-block32-input-features.v1",
        "family": "mxfp8",
        "components": [
            {
                "role": "values",
                "dtype": "e4m3",
                "encoding": "mxfp8_e4m3_values",
                "component_axes": {"kind": "identity"},
            },
            {
                "role": "block_scales",
                "dtype": "e8m0",
                "encoding": "mxfp8_e8m0_scale",
                "component_axes": {
                    "kind": "explicit",
                    "axes": [
                        {
                            "kind": "logical",
                            "logical_axis": "output_features",
                            "divisor": 1,
                            "rounding": "exact",
                        },
                        {
                            "kind": "logical",
                            "logical_axis": "input_features",
                            "divisor": 32,
                            "rounding": "ceil",
                        },
                    ],
                },
            },
        ],
    }


def test_checkpoint_evidence_content_changes_topology_intent_and_group_identity() -> (
    None
):
    """Checkpoint content evidence must participate in every serving identity."""
    main = _explicit_entry(
        "main-weight",
        "main",
        "text.decoder.layer.0.dense.kernel",
        semantic_graph_path="text.decoder",
        model_part="main",
        global_decoder_layer=0,
    )
    draft = _explicit_entry(
        "draft-weight",
        "draft.static",
        "draft.decoder.layer.0.dense.kernel",
        semantic_graph_path="draft.decoder",
        model_part="draft",
        global_decoder_layer=0,
        value_provenance=ValueProvenance.CHECKPOINT_ENCODING_COMPONENT,
    )
    first_lifecycle = _checkpoint_lifecycle(
        "draft.static", GraphKind.SPECULATIVE_DRAFTER, "model-draft.static", "rev-1"
    )
    assert first_lifecycle.immutable_evidence is not None
    second_lifecycle = replace(
        first_lifecycle,
        immutable_evidence=replace(
            first_lifecycle.immutable_evidence,
            checkpoint_content_digest="sha256:different-checkpoint-bytes",
        ),
    )
    main_lifecycle = _runtime_lifecycle(
        GraphKind.MAIN, RolloutParticipation.SERVED_FROM_SOURCE
    )
    bundles = tuple(
        _bundle(
            (main, draft),
            lifecycles={"main": main_lifecycle, "draft.static": lifecycle},
            mutabilities={"draft-weight": SourceMutability.FROZEN},
        )
        for lifecycle in (first_lifecycle, second_lifecycle)
    )
    policy = PrecisionPolicyConfig.model_validate({"scopes": []})
    first, second = (compile_precision_policy(policy, bundle) for bundle in bundles)
    assert first.topology_digest != second.topology_digest
    assert (
        first.graph_intent("draft.static").intent_id
        != second.graph_intent("draft.static").intent_id
    )
    assert first.intent_group_id != second.intent_group_id


def test_public_compiled_collections_snapshot_mutable_inputs() -> None:
    """Frozen records must not retain caller-owned mutable list aliases."""
    plan = compile_precision_policy(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}),
        _sparse_moe_bundle(),
    )
    graph_intents = list(plan.graph_intents)
    scope_results = list(plan.scope_results)
    every_version = list(plan.every_version_source_items)
    rebuilt = replace(
        plan,
        graph_intents=graph_intents,  # type: ignore[arg-type]
        scope_results=scope_results,  # type: ignore[arg-type]
        every_version_source_items=every_version,  # type: ignore[arg-type]
    )
    endpoint = rebuilt.graph_intent("main").rollout_plan
    assert endpoint is not None
    assignments = list(endpoint.assignments)
    rebuilt_endpoint = replace(
        endpoint,
        assignments=assignments,  # type: ignore[arg-type]
    )
    requirements = list(rebuilt.graph_intent("main").owner_refit_requirements.items())
    rebuilt_requirements = replace(
        rebuilt.graph_intent("main").owner_refit_requirements,
        entries=requirements,  # type: ignore[arg-type]
    )
    graph_intents.clear()
    scope_results.clear()
    every_version.clear()
    assignments.clear()
    requirements.clear()
    assert isinstance(rebuilt.graph_intents, tuple)
    assert isinstance(rebuilt.scope_results, tuple)
    assert isinstance(rebuilt.every_version_source_items, tuple)
    assert isinstance(rebuilt_endpoint.assignments, tuple)
    assert isinstance(rebuilt_requirements.entries, tuple)
    assert rebuilt.to_wire_dict() == plan.to_wire_dict()


def test_public_compiled_collections_reject_wrong_element_types() -> None:
    """A malformed frozen plan must fail at construction, not serialization."""
    plan = compile_precision_policy(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        _sparse_moe_bundle(),
    )
    with pytest.raises(TypeError, match="graph_intents"):
        replace(plan, graph_intents=("not-an-intent",))  # type: ignore[arg-type]


def test_large_kimi_style_domains_compile_without_rendering_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Role and exact-address compilation must never enumerate 69,120 IDs."""
    domain = _layer_domain(
        tuple(range(60)),
        moe_ordinals=tuple(range(60)),
        experts=tuple(range(384)),
    )
    entries = tuple(
        _family_entry(f"kimi-{projection}", projection, domain)
        for projection in ("gate", "up", "down")
    )
    bundle = _bundle(
        entries,
        role_definitions=builtin_role_definitions(
            1,
            {
                "moe.routed_expert": RoleExpectedDomain(
                    "moe.routed_expert", ("kimi-gate", "kimi-up", "kimi-down")
                )
            },
        ),
    )

    def fail_renderer(self: SemanticTensorFamily) -> None:
        raise AssertionError("compiler materialized semantic family members")

    monkeypatch.setattr(SemanticTensorFamily, "iter_semantic_ids", fail_renderer)
    role_plan = compile_precision_policy(
        _policy({"roles": ["moe.routed_expert"], "rollout": "mxfp8"}), bundle
    )
    assert (
        role_plan.scope_result("scope")
        .graph_result("main")
        .selected_logical_cardinality
        == 60 * 384 * 3
    )
    role_rollout = role_plan.graph_intent("main").rollout_plan
    assert role_rollout is not None
    assert len(role_rollout.assignments) == 3

    address_plan = compile_precision_policy(
        _policy(
            {
                "addresses": [
                    {
                        "graph_instance_id": "main",
                        "semantic_graph_path": "text.decoder",
                        "semantic_id": "text.decoder.layer.59.expert.383.down",
                    }
                ],
                "rollout": "mxfp8",
            }
        ),
        bundle,
    )
    assert (
        address_plan.scope_result("scope")
        .graph_result("main")
        .selected_logical_cardinality
        == 1
    )
    address_rollout = address_plan.graph_intent("main").rollout_plan
    assert address_rollout is not None
    assert len(address_rollout.assignments) <= 7
