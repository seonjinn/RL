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

from copy import copy
from dataclasses import FrozenInstanceError, replace

import pytest

import nemo_rl.precision_policy.compiler as compiler_module
from nemo_rl.precision_policy.compiler import (
    CompiledPrecisionIntentGroup,
    PrecisionPolicyError,
    compile_precision_policy,
)
from nemo_rl.precision_policy.config import PrecisionPolicyConfig
from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    MXFP8_FORMAT,
    AtomicGroup,
    AtomicGroupKind,
    AtomicGroupParticipant,
    AxisExtentRounding,
    AxisDomain,
    AxisProjection,
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
    SemanticTensor,
    SemanticTensorFamily,
    SourceMutability,
    SourceOwnerInventoryEntry,
    ValueProvenance,
    builtin_role_definitions,
    ComponentDescriptor,
    ComponentRole,
    FormatDescriptor,
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
            if entry.value_provenance != ValueProvenance.TIED_ALIAS:
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


def test_global_decoder_boundaries_keep_sparse_first_and_last_moe_layers_bf16() -> None:
    """Counting only MoE layers for global boundaries would quantize layer 1."""
    plan = compile_precision_policy(
        _policy(
            {
                "role": "moe.routed_expert",
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
                "role": "moe.routed_expert",
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
                "role": "moe.routed_expert",
                "layers": {"exclude_first": 1},
                "rollout": "mxfp8",
            }
        ),
        _sparse_moe_bundle(),
    )
    ordinal_plan = compile_precision_policy(
        _policy(
            {
                "role": "moe.routed_expert",
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
                "role": "moe.routed_expert",
                "layers": {"exclude_last": 1},
                "rollout": "mxfp8",
            }
        ),
        bundle,
    )
    ordinal_plan = compile_precision_policy(
        _policy(
            {
                "role": "moe.routed_expert",
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
                "role": "moe.routed_expert",
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
        _policy({"role": "moe.routed_expert", "rollout": "mxfp8"}),
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
                    "role": "moe.routed_expert",
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
            _policy({"role": "moe.unknown", "rollout": "mxfp8"}),
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
            _policy({"role": "moe.routed_expert", "rollout": "mxfp8"}),
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
                    "role": "moe.routed_expert",
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
                    "role": "moe.routed_expert",
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
                    "role": "embedding.ngram",
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
                    "role": "embedding.ngram",
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
        _policy({"role": "embedding.ngram", "rollout": "mxfp8"}), bundle
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
                    "role": "moe.routed_expert",
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
        _policy({"role": "moe.routed_expert", "rollout": "mxfp8"}), bundle
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
        value_provenance=ValueProvenance.TIED_ALIAS,
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
            value_provenance=ValueProvenance.TIED_ALIAS,
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
        value_provenance=ValueProvenance.TIED_ALIAS,
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
        _policy({"role": "moe.routed_expert", "rollout": "mxfp8"}),
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
        _policy({"role": "moe.routed_expert", "rollout": "mxfp8"}), bundle
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
