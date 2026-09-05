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

from __future__ import annotations

import pickle
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, fields, replace

import pytest

from nemo_rl.precision_policy.semantic import (
    AtomicGroup,
    DecoderLayerUniverse,
    ExpectedGraphDeclaration,
    FamilyIndexDomain,
    GraphKind,
    GraphLifecycle,
    GraphProvenance,
    IndexPathSegment,
    LayerDomain,
    LayerMember,
    LiteralPathSegment,
    ResolvedGraphTopology,
    ResolvedSelectionTopology,
    RoleDefinition,
    RoleExpectedDomain,
    RolloutParticipation,
    SelectionTopologyEntry,
    SemanticAddressPattern,
    SemanticPredicate,
    _compute_semantic_structure_digest,
    builtin_role_definitions,
)
from nemo_rl.precision_policy.topology_resolver import (
    GraphTopologyResolutionRequest,
    SelectionTopologyAdapter,
    resolve_selection_topology,
)


class _FakeAdapter:
    def __init__(
        self,
        *,
        adapter_id: str,
        model_type: str,
        builder: Callable[[GraphTopologyResolutionRequest, str], ResolvedGraphTopology],
    ) -> None:
        self.adapter_id = adapter_id
        self._model_type = model_type
        self._builder = builder

    def supports(self, model_config: Mapping[str, object]) -> bool:
        return model_config.get("model_type") == self._model_type

    def resolve_graph(
        self,
        request: GraphTopologyResolutionRequest,
    ) -> ResolvedGraphTopology:
        return self._builder(request, self.adapter_id)


def _declaration(
    graph_instance_id: str = "main",
    graph_kind: GraphKind = GraphKind.MAIN,
) -> ExpectedGraphDeclaration:
    return ExpectedGraphDeclaration(
        graph_instance_id=graph_instance_id,
        model_identity=f"test/{graph_instance_id}",
        lifecycle=GraphLifecycle(
            graph_kind=graph_kind,
            graph_provenance=GraphProvenance.TRAINING_RUNTIME,
            rollout_participation=(
                RolloutParticipation.SERVED_FROM_SOURCE
                if graph_kind == GraphKind.MAIN
                else RolloutParticipation.NOT_SERVED
            ),
        ),
    )


def _request(
    graph_instance_id: str = "main",
    graph_kind: GraphKind = GraphKind.MAIN,
    *,
    model_type: str = "test_model",
    universe: DecoderLayerUniverse | None = None,
) -> GraphTopologyResolutionRequest:
    return GraphTopologyResolutionRequest(
        declaration=_declaration(graph_instance_id, graph_kind),
        effective_model_config={
            "model_type": model_type,
            "text_config": {"architectures": ["TestForCausalLM"]},
        },
        resolved_model_revision=f"revision-{graph_instance_id}",
        decoder_layer_universe=universe
        or DecoderLayerUniverse(
            global_decoder_layers=(0, 1, 2, 3),
            moe_global_decoder_layers_by_ordinal=(),
        ),
    )


def _entry(
    graph_instance_id: str,
    entry_id: str,
    *,
    global_layers: tuple[int, ...],
    moe_ordinals: tuple[int, ...] | None = None,
    routed_expert: bool = False,
) -> SelectionTopologyEntry:
    if moe_ordinals is None:
        layer_members = tuple(LayerMember(layer, None) for layer in global_layers)
    else:
        layer_members = tuple(
            LayerMember(layer, ordinal)
            for layer, ordinal in zip(global_layers, moe_ordinals, strict=True)
        )
    graph_path = {
        "main": "text.decoder",
        "mtp.aux": "auxiliary.mtp",
        "draft.aux": "draft.decoder",
    }[graph_instance_id]
    model_part = {
        "main": "main",
        "mtp.aux": "mtp",
        "draft.aux": "draft",
    }[graph_instance_id]
    return SelectionTopologyEntry(
        entry_id=entry_id,
        graph_instance_id=graph_instance_id,
        pattern=SemanticAddressPattern(
            semantic_graph_path=graph_path,
            path_segments=(
                LiteralPathSegment("layer"),
                IndexPathSegment("global_decoder_layer"),
                LiteralPathSegment("weight"),
            ),
            model_part=model_part,
            module_kind="moe.expert_ffn" if routed_expert else "ffn.dense",
            attributes=(
                (("expert_kind", "routed"), ("projection", "gate"))
                if routed_expert
                else ()
            ),
            parameter_role="kernel",
        ),
        domain=FamilyIndexDomain(
            layer_domain=LayerDomain(layer_members),
            independent_axes=(),
        ),
        logical_dtype="bfloat16",
        logical_shape=(8, 8),
        logical_axes=("output_features", "input_features"),
    )


def _role_definitions(
    entries: tuple[SelectionTopologyEntry, ...],
    *,
    extra: tuple[RoleDefinition, ...] = (),
) -> tuple[RoleDefinition, ...]:
    routed = tuple(
        entry.entry_id
        for entry in entries
        if entry.graph_instance_id == "main"
        and entry.pattern.module_kind == "moe.expert_ffn"
    )
    expected = (
        {"moe.routed_expert": RoleExpectedDomain("moe.routed_expert", routed)}
        if routed
        else {}
    )
    return builtin_role_definitions(1, expected) + extra


def _graph_builder(
    entries_by_graph: Mapping[str, tuple[SelectionTopologyEntry, ...]],
    *,
    roles_by_graph: Mapping[str, tuple[RoleDefinition, ...]] | None = None,
    universe_by_graph: Mapping[str, DecoderLayerUniverse] | None = None,
    atomic_groups_by_graph: Mapping[str, tuple[AtomicGroup, ...]] | None = None,
) -> Callable[[GraphTopologyResolutionRequest, str], ResolvedGraphTopology]:
    def build(
        request: GraphTopologyResolutionRequest,
        adapter_id: str,
    ) -> ResolvedGraphTopology:
        entries = entries_by_graph[request.declaration.graph_instance_id]
        return ResolvedGraphTopology(
            declaration=request.declaration,
            model_family="test_family",
            resolved_model_revision=request.resolved_model_revision,
            adapter_id=adapter_id,
            decoder_layer_universe=(universe_by_graph or {}).get(
                request.declaration.graph_instance_id,
                request.decoder_layer_universe,
            ),
            entries=entries,
            role_definitions=(
                (roles_by_graph or {}).get(request.declaration.graph_instance_id)
                or _role_definitions(entries)
            ),
            atomic_groups=(atomic_groups_by_graph or {}).get(
                request.declaration.graph_instance_id, ()
            ),
        )

    return build


def _adapter(
    entries_by_graph: Mapping[str, tuple[SelectionTopologyEntry, ...]],
    *,
    adapter_id: str = "test.adapter.v1",
    model_type: str = "test_model",
    roles_by_graph: Mapping[str, tuple[RoleDefinition, ...]] | None = None,
    universe_by_graph: Mapping[str, DecoderLayerUniverse] | None = None,
) -> SelectionTopologyAdapter:
    return _FakeAdapter(
        adapter_id=adapter_id,
        model_type=model_type,
        builder=_graph_builder(
            entries_by_graph,
            roles_by_graph=roles_by_graph,
            universe_by_graph=universe_by_graph,
        ),
    )


def test_global_boundary_uses_declared_decoder_universe_without_dense_marker() -> None:
    request = _request(
        universe=DecoderLayerUniverse(tuple(range(4)), (1, 3)),
    )
    expert = _entry(
        "main",
        "main.routed.gate",
        global_layers=(1, 3),
        moe_ordinals=(0, 1),
        routed_expert=True,
    )

    topology = resolve_selection_topology(
        (request,),
        1,
        adapters=(_adapter({"main": (expert,)}),),
    )

    assert topology.graphs[0].decoder_layer_universe.global_decoder_layers == (
        0,
        1,
        2,
        3,
    )
    assert tuple(
        member.global_decoder_layer
        for member in topology.graphs[0].entries[0].domain.layer_domain.members
    ) == (1, 3)


def test_moe_ordinal_universe_is_exact_contiguous_one_to_one_mapping() -> None:
    universe = DecoderLayerUniverse(
        global_decoder_layers=tuple(range(8)),
        moe_global_decoder_layers_by_ordinal=(1, 4, 7),
    )
    assert tuple(enumerate(universe.moe_global_decoder_layers_by_ordinal)) == (
        (0, 1),
        (1, 4),
        (2, 7),
    )

    invalid = (
        ((0, 2), ()),
        (tuple(range(4)), (1, 1)),
        (tuple(range(5)), (4, 1)),
        (tuple(range(4)), (1, 4)),
        (tuple(range(4)), (True,)),
    )
    for global_layers, moe_layers in invalid:
        with pytest.raises(ValueError):
            DecoderLayerUniverse(global_layers, moe_layers)


def test_main_mtp_and_draft_layer_universes_are_independent_and_zero_based() -> None:
    main = _request(
        universe=DecoderLayerUniverse(tuple(range(4)), (1, 3)),
    )
    mtp = _request(
        "mtp.aux",
        GraphKind.MTP,
        universe=DecoderLayerUniverse((0,), ()),
    )
    draft = _request(
        "draft.aux",
        GraphKind.SPECULATIVE_DRAFTER,
        universe=DecoderLayerUniverse((0, 1), (0,)),
    )
    entries = {
        "main": (
            _entry(
                "main",
                "main.routed",
                global_layers=(1, 3),
                moe_ordinals=(0, 1),
                routed_expert=True,
            ),
        ),
        "mtp.aux": (_entry("mtp.aux", "mtp.aux.dense", global_layers=(0,)),),
        "draft.aux": (
            _entry(
                "draft.aux",
                "draft.aux.routed",
                global_layers=(0,),
                moe_ordinals=(0,),
                routed_expert=True,
            ),
        ),
    }

    topology = resolve_selection_topology(
        (draft, main, mtp),
        1,
        adapters=(_adapter(entries),),
    )

    assert tuple(graph.declaration.graph_instance_id for graph in topology.graphs) == (
        "main",
        "draft.aux",
        "mtp.aux",
    )
    universes = {
        graph.declaration.graph_instance_id: graph.decoder_layer_universe
        for graph in topology.graphs
    }
    assert universes["main"].global_decoder_layers == (0, 1, 2, 3)
    assert universes["mtp.aux"].global_decoder_layers == (0,)
    assert universes["draft.aux"].global_decoder_layers == (0, 1)
    with pytest.raises(ValueError, match="zero-based"):
        DecoderLayerUniverse((1, 2), ())


@pytest.mark.parametrize(
    ("graph_instance_id", "graph_kind", "semantic_graph_path", "model_part"),
    (
        ("main", GraphKind.MAIN, "auxiliary.mtp", "mtp"),
        ("main", GraphKind.MAIN, "draft.decoder", "draft"),
        ("main", GraphKind.MAIN, "text.decoder", "mtp"),
        ("mtp.aux", GraphKind.MTP, "text.decoder", "main"),
        ("mtp.aux", GraphKind.MTP, "draft.decoder", "draft"),
        ("mtp.aux", GraphKind.MTP, "mtp.decoder", "mtp"),
        ("mtp.aux", GraphKind.MTP, "auxiliary.mtp", "auxiliary"),
        ("draft.aux", GraphKind.SPECULATIVE_DRAFTER, "text.decoder", "main"),
        (
            "draft.aux",
            GraphKind.SPECULATIVE_DRAFTER,
            "auxiliary.mtp",
            "mtp",
        ),
        (
            "draft.aux",
            GraphKind.SPECULATIVE_DRAFTER,
            "draft.decoder",
            "mtp",
        ),
    ),
)
def test_layered_decoder_entries_require_graph_kind_specific_address_pair(
    graph_instance_id: str,
    graph_kind: GraphKind,
    semantic_graph_path: str,
    model_part: str,
) -> None:
    universe = DecoderLayerUniverse((0,), ())
    request = _request(
        graph_instance_id,
        graph_kind,
        universe=universe,
    )
    entry = _entry(
        graph_instance_id,
        f"{graph_instance_id}.dense",
        global_layers=(0,),
    )
    invalid = replace(
        entry,
        pattern=replace(
            entry.pattern,
            semantic_graph_path=semantic_graph_path,
            model_part=model_part,
        ),
    )

    with pytest.raises(ValueError, match="layered decoder entry address"):
        resolve_selection_topology(
            (request,) if graph_kind == GraphKind.MAIN else (_request(), request),
            1,
            adapters=(
                _adapter(
                    {
                        "main": (_entry("main", "main.dense", global_layers=(0,)),),
                        graph_instance_id: (invalid,),
                    }
                ),
            ),
        )


def test_phase_one_selection_contains_no_source_mutability_alias_or_cadence() -> None:
    request = _request()
    entry = _entry("main", "main.dense", global_layers=(0, 1, 2, 3))
    topology = resolve_selection_topology(
        (request,),
        1,
        adapters=(_adapter({"main": (entry,)}),),
    )

    record_types = (
        DecoderLayerUniverse,
        SelectionTopologyEntry,
        ResolvedGraphTopology,
        ResolvedSelectionTopology,
        GraphTopologyResolutionRequest,
    )
    forbidden = (
        "format",
        "source",
        "native",
        "fingerprint",
        "alias",
        "mutability",
        "cadence",
    )
    for record_type in record_types:
        names = tuple(field.name for field in fields(record_type))
        assert not any(token in name for token in forbidden for name in names)

    with pytest.raises(FrozenInstanceError):
        topology.graphs = ()  # type: ignore[misc]
    nested = request.effective_model_config["text_config"]
    assert isinstance(nested, Mapping)
    with pytest.raises(TypeError):
        nested["architectures"] = ()  # type: ignore[index]


def test_request_recursively_snapshots_plain_configuration() -> None:
    model_config: dict[str, object] = {
        "model_type": "test_model",
        "nested": {"layers": [1, 2]},
    }
    request = GraphTopologyResolutionRequest(
        declaration=_declaration(),
        effective_model_config=model_config,
        resolved_model_revision="revision-main",
        decoder_layer_universe=DecoderLayerUniverse((0, 1), (1,)),
    )
    model_config["nested"] = {"layers": [99]}

    nested = request.effective_model_config["nested"]
    assert isinstance(nested, Mapping)
    assert nested["layers"] == (1, 2)
    restored = pickle.loads(pickle.dumps(request))
    assert restored == request
    assert restored.effective_model_config == request.effective_model_config
    with pytest.raises(TypeError, match="plain configuration"):
        replace(request, effective_model_config={"bad": object()})


def test_recursively_frozen_request_config_has_structural_mapping_equality() -> None:
    request = _request()
    equivalent = {
        "model_type": "test_model",
        "text_config": {"architectures": ("TestForCausalLM",)},
    }
    different = {
        "model_type": "different_model",
        "text_config": {"architectures": ("TestForCausalLM",)},
    }

    assert request.effective_model_config == equivalent
    assert equivalent == request.effective_model_config
    assert request.effective_model_config != different
    assert different != request.effective_model_config


def test_resolver_rejects_adapter_universe_disagreement_atomically() -> None:
    request = _request()
    entry = _entry("main", "main.dense", global_layers=(0, 1, 2, 3))
    mismatched = DecoderLayerUniverse(tuple(range(5)), ())

    with pytest.raises(ValueError, match="decoder layer universe mismatch"):
        resolve_selection_topology(
            (request,),
            1,
            adapters=(
                _adapter(
                    {"main": (entry,)},
                    universe_by_graph={"main": mismatched},
                ),
            ),
        )


def test_resolver_requires_exactly_one_adapter_and_a_complete_graph_set() -> None:
    request = _request()
    entry = _entry("main", "main.dense", global_layers=(0, 1, 2, 3))
    matching = _adapter({"main": (entry,)})

    with pytest.raises(ValueError, match="exactly one selection topology adapter"):
        resolve_selection_topology((request,), 1, adapters=())
    with pytest.raises(ValueError, match="exactly one selection topology adapter"):
        resolve_selection_topology(
            (request,),
            1,
            adapters=(
                matching,
                _adapter(
                    {"main": (entry,)},
                    adapter_id="test.second-adapter.v1",
                ),
            ),
        )
    with pytest.raises(ValueError, match="exactly one MAIN"):
        resolve_selection_topology(
            (
                _request(
                    "mtp.aux",
                    GraphKind.MTP,
                    universe=DecoderLayerUniverse((0,), ()),
                ),
            ),
            1,
            adapters=(
                _adapter(
                    {
                        "mtp.aux": (
                            _entry(
                                "mtp.aux",
                                "mtp.aux.dense",
                                global_layers=(0,),
                            ),
                        )
                    }
                ),
            ),
        )


def test_resolver_rejects_duplicate_adapter_ids_before_support_dispatch() -> None:
    request = _request()
    entry = _entry("main", "main.dense", global_layers=(0, 1, 2, 3))
    first = _adapter({"main": (entry,)}, adapter_id="duplicate.adapter.v1")
    unrelated = _adapter(
        {"main": (entry,)},
        adapter_id="duplicate.adapter.v1",
        model_type="unrelated_model",
    )

    with pytest.raises(ValueError, match="duplicate selection topology adapter_id"):
        resolve_selection_topology(
            (request,),
            1,
            adapters=(first, unrelated),
        )


def test_resolver_merges_roles_and_hashes_canonical_graph_order() -> None:
    main = _request(universe=DecoderLayerUniverse((0, 1), ()))
    mtp = _request(
        "mtp.aux",
        GraphKind.MTP,
        universe=DecoderLayerUniverse((0,), ()),
    )
    main_entry = _entry("main", "main.dense", global_layers=(0, 1))
    mtp_entry = _entry("mtp.aux", "mtp.aux.dense", global_layers=(0,))
    shared_predicate = SemanticPredicate(
        graph_kinds=(),
        semantic_graph_paths=(),
        model_parts=(),
        module_kinds=("ffn.dense",),
        attributes=(),
        parameter_roles=("kernel",),
    )
    main_extra = RoleDefinition(
        1,
        "test.dense",
        shared_predicate,
        RoleExpectedDomain("test.dense", (main_entry.entry_id,)),
    )
    mtp_extra = RoleDefinition(
        1,
        "test.dense",
        shared_predicate,
        RoleExpectedDomain("test.dense", (mtp_entry.entry_id,)),
    )
    entries = {"main": (main_entry,), "mtp.aux": (mtp_entry,)}
    roles = {
        "main": _role_definitions((main_entry,), extra=(main_extra,)),
        "mtp.aux": _role_definitions((mtp_entry,), extra=(mtp_extra,)),
    }

    forward = resolve_selection_topology(
        (main, mtp),
        1,
        adapters=(_adapter(entries, roles_by_graph=roles),),
    )
    reverse = resolve_selection_topology(
        (mtp, main),
        1,
        adapters=(_adapter(entries, roles_by_graph=roles),),
    )

    assert forward == reverse
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", forward.semantic_structure_digest)
    assert forward.role_definition(1, "test.dense").expected_domain == (
        RoleExpectedDomain("test.dense", ("main.dense", "mtp.aux.dense"))
    )


def test_resolver_rejects_conflicting_role_contributions() -> None:
    main = _request(universe=DecoderLayerUniverse((0,), ()))
    mtp = _request(
        "mtp.aux",
        GraphKind.MTP,
        universe=DecoderLayerUniverse((0,), ()),
    )
    main_entry = _entry("main", "main.dense", global_layers=(0,))
    mtp_entry = _entry("mtp.aux", "mtp.aux.dense", global_layers=(0,))
    first = RoleDefinition(
        1,
        "test.dense",
        SemanticPredicate((), (), (), ("ffn.dense",), (), ("kernel",)),
        RoleExpectedDomain("test.dense", (main_entry.entry_id,)),
    )
    conflicting = RoleDefinition(
        1,
        "test.dense",
        SemanticPredicate((GraphKind.MTP,), (), (), ("ffn.dense",), (), ("kernel",)),
        RoleExpectedDomain("test.dense", (mtp_entry.entry_id,)),
    )

    with pytest.raises(ValueError, match="conflicting role predicate"):
        resolve_selection_topology(
            (main, mtp),
            1,
            adapters=(
                _adapter(
                    {"main": (main_entry,), "mtp.aux": (mtp_entry,)},
                    roles_by_graph={
                        "main": _role_definitions((main_entry,), extra=(first,)),
                        "mtp.aux": _role_definitions(
                            (mtp_entry,), extra=(conflicting,)
                        ),
                    },
                ),
            ),
        )


def test_resolver_rejects_overlapping_role_contributions() -> None:
    main = _request(universe=DecoderLayerUniverse((0,), ()))
    mtp = _request(
        "mtp.aux",
        GraphKind.MTP,
        universe=DecoderLayerUniverse((0,), ()),
    )
    main_entry = _entry("main", "shared.dense", global_layers=(0,))
    mtp_entry = _entry("mtp.aux", "shared.dense", global_layers=(0,))
    predicate = SemanticPredicate((), (), (), ("ffn.dense",), (), ("kernel",))
    main_definition = RoleDefinition(
        1,
        "test.dense",
        predicate,
        RoleExpectedDomain("test.dense", (main_entry.entry_id,)),
    )
    mtp_definition = RoleDefinition(
        1,
        "test.dense",
        predicate,
        RoleExpectedDomain("test.dense", (mtp_entry.entry_id,)),
    )

    with pytest.raises(ValueError, match="overlapping role contribution"):
        resolve_selection_topology(
            (main, mtp),
            1,
            adapters=(
                _adapter(
                    {"main": (main_entry,), "mtp.aux": (mtp_entry,)},
                    roles_by_graph={
                        "main": _role_definitions(
                            (main_entry,), extra=(main_definition,)
                        ),
                        "mtp.aux": _role_definitions(
                            (mtp_entry,), extra=(mtp_definition,)
                        ),
                    },
                ),
            ),
        )


def test_resolved_graph_rejects_duplicate_role_definition_keys_before_merge() -> None:
    request = _request(universe=DecoderLayerUniverse((0,), ()))
    entry = _entry("main", "main.dense", global_layers=(0,))
    roles = _role_definitions((entry,))

    with pytest.raises(ValueError, match="duplicate resolved graph role definition"):
        resolve_selection_topology(
            (request,),
            1,
            adapters=(
                _adapter(
                    {"main": (entry,)},
                    roles_by_graph={"main": roles + (roles[0],)},
                ),
            ),
        )


def test_resolver_rejects_entry_layer_coordinates_outside_declared_universe() -> None:
    request = _request(universe=DecoderLayerUniverse((0, 1), (1,)))
    out_of_range = _entry("main", "main.dense", global_layers=(0, 2))

    with pytest.raises(ValueError, match="outside decoder layer universe"):
        resolve_selection_topology(
            (request,),
            1,
            adapters=(_adapter({"main": (out_of_range,)}),),
        )


def test_resolver_rejects_entry_moe_coordinates_that_disagree_with_universe() -> None:
    request = _request(universe=DecoderLayerUniverse((0, 1), (1,)))
    wrong_ordinal = _entry(
        "main",
        "main.routed.gate",
        global_layers=(1,),
        moe_ordinals=(1,),
        routed_expert=True,
    )

    with pytest.raises(ValueError, match="MoE ordinal mapping"):
        resolve_selection_topology(
            (request,),
            1,
            adapters=(_adapter({"main": (wrong_ordinal,)}),),
        )
    with pytest.raises(ValueError, match="global_decoder_layer"):
        LayerMember(True, None)


def test_routed_expert_entries_require_complete_moe_ordinal_coordinates() -> None:
    request = _request(universe=DecoderLayerUniverse((0, 1, 2), (1, 2)))
    missing_ordinals = _entry(
        "main",
        "main.routed.gate",
        global_layers=(1, 2),
        routed_expert=True,
    )

    with pytest.raises(ValueError, match="routed expert.*moe_ordinal"):
        resolve_selection_topology(
            (request,),
            1,
            adapters=(_adapter({"main": (missing_ordinals,)}),),
        )


def test_split_routed_expert_families_collectively_cover_moe_universe() -> None:
    request = _request(universe=DecoderLayerUniverse((0, 1, 2), (1, 2)))
    first = _entry(
        "main",
        "main.routed.first",
        global_layers=(1,),
        moe_ordinals=(0,),
        routed_expert=True,
    )
    last = _entry(
        "main",
        "main.routed.last",
        global_layers=(2,),
        moe_ordinals=(1,),
        routed_expert=True,
    )

    topology = resolve_selection_topology(
        (request,),
        1,
        adapters=(_adapter({"main": (last, first)}),),
    )

    assert topology.role_definition(
        1, "moe.routed_expert"
    ).expected_domain.inventory_entry_ids == ("main.routed.first", "main.routed.last")


def test_routed_expert_family_union_must_cover_moe_universe() -> None:
    request = _request(universe=DecoderLayerUniverse((0, 1, 2), (1, 2)))
    partial = _entry(
        "main",
        "main.routed.partial",
        global_layers=(1,),
        moe_ordinals=(0,),
        routed_expert=True,
    )

    with pytest.raises(ValueError, match="layer union"):
        resolve_selection_topology(
            (request,),
            1,
            adapters=(_adapter({"main": (partial,)}),),
        )


def test_moe_universe_requires_a_routed_expert_family() -> None:
    request = _request(universe=DecoderLayerUniverse((0, 1), (1,)))
    dense = _entry("main", "main.dense", global_layers=(0, 1))

    with pytest.raises(ValueError, match="layer union"):
        resolve_selection_topology(
            (request,),
            1,
            adapters=(_adapter({"main": (dense,)}),),
        )


def test_semantic_structure_digest_binds_logical_shape() -> None:
    request = _request(universe=DecoderLayerUniverse((0,), ()))
    entry = _entry("main", "main.dense", global_layers=(0,))
    changed = replace(entry, logical_shape=(16, 8))

    baseline = resolve_selection_topology(
        (request,),
        1,
        adapters=(_adapter({"main": (entry,)}),),
    )
    reshaped = resolve_selection_topology(
        (request,),
        1,
        adapters=(_adapter({"main": (changed,)}),),
    )

    assert baseline.semantic_structure_digest != reshaped.semantic_structure_digest


def test_semantic_structure_digest_normalizes_signed_zero_and_preserves_types() -> None:
    request = _request(universe=DecoderLayerUniverse((0,), ()))

    def resolve_marker(marker: float | int | bool) -> ResolvedSelectionTopology:
        base = _entry("main", "main.dense", global_layers=(0,))
        entry = replace(
            base,
            pattern=replace(base.pattern, attributes=(("marker", marker),)),
        )
        return resolve_selection_topology(
            (request,),
            1,
            adapters=(_adapter({"main": (entry,)}),),
        )

    positive_zero = resolve_marker(0.0)
    negative_zero = resolve_marker(-0.0)
    false_value = resolve_marker(False)
    integer_zero = resolve_marker(0)

    assert positive_zero.graphs == negative_zero.graphs
    assert positive_zero == negative_zero
    assert positive_zero.semantic_structure_digest == (
        negative_zero.semantic_structure_digest
    )
    assert positive_zero.graphs != false_value.graphs
    assert positive_zero.graphs != integer_zero.graphs
    assert false_value.graphs != integer_zero.graphs
    assert (
        len(
            {
                positive_zero.semantic_structure_digest,
                false_value.semantic_structure_digest,
                integer_zero.semantic_structure_digest,
            }
        )
        == 3
    )


def test_resolved_selection_topology_rejects_forged_digest() -> None:
    request = _request()
    entries = (_entry("main", "main.dense", global_layers=(0, 1, 2, 3)),)
    topology = resolve_selection_topology(
        (request,),
        1,
        adapters=(_adapter({"main": entries}),),
    )

    with pytest.raises(ValueError, match="semantic_structure_digest mismatch"):
        replace(
            topology,
            semantic_structure_digest="sha256:" + "0" * 64,
        )
    with pytest.raises(ValueError, match="semantic_structure_digest mismatch"):
        replace(
            topology,
            graphs=(replace(topology.graphs[0], model_family="changed_family"),),
        )


@pytest.mark.parametrize("mutation", ("missing", "injected"))
def test_resolved_selection_topology_requires_exact_merged_graph_role_registry(
    mutation: str,
) -> None:
    request = _request(universe=DecoderLayerUniverse((0,), ()))
    entry = _entry("main", "main.dense", global_layers=(0,))
    predicate = SemanticPredicate((), (), (), ("ffn.dense",), (), ("kernel",))
    contributed = RoleDefinition(
        1,
        "test.dense",
        predicate,
        RoleExpectedDomain("test.dense", (entry.entry_id,)),
    )
    topology = resolve_selection_topology(
        (request,),
        1,
        adapters=(
            _adapter(
                {"main": (entry,)},
                roles_by_graph={
                    "main": _role_definitions((entry,), extra=(contributed,))
                },
            ),
        ),
    )
    if mutation == "missing":
        changed_roles = tuple(
            definition
            for definition in topology.role_definitions
            if definition.role_name != "test.dense"
        )
    else:
        changed_roles = topology.role_definitions + (
            RoleDefinition(
                1,
                "test.injected",
                predicate,
                RoleExpectedDomain("test.injected", (entry.entry_id,)),
            ),
        )
    changed_digest = _compute_semantic_structure_digest(
        schema_version=topology.schema_version,
        graphs=topology.graphs,
        role_definitions=changed_roles,
    )

    with pytest.raises(ValueError, match="canonical graph role merge"):
        ResolvedSelectionTopology(
            schema_version=topology.schema_version,
            graphs=topology.graphs,
            role_definitions=changed_roles,
            semantic_structure_digest=changed_digest,
        )


def test_phase_one_module_does_not_import_source_discovery_or_frameworks() -> None:
    code = r"""
import builtins
import importlib
import sys

blocked = (
    "nemo_rl.precision_policy.config",
    "nemo_rl.precision_policy.source_discovery",
    "nemo_rl.precision_policy.source_dtype",
    "nemo_rl.precision_policy.source_storage",
    "pydantic",
    "torch",
    "megatron",
    "nemo_automodel",
    "transformer_engine",
    "vllm",
)
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise AssertionError(f"Phase 1 imported forbidden module: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
importlib.import_module("nemo_rl.precision_policy.topology_resolver")
unexpected = tuple(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)
)
if unexpected:
    raise AssertionError(f"Phase 1 retained forbidden modules: {unexpected}")
print("phase-one-import-ok")
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "."},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "phase-one-import-ok"


def test_lazy_precision_policy_exports_preserve_the_public_surface() -> None:
    code = r"""
import nemo_rl.precision_policy as precision_policy

for name in precision_policy.__all__:
    getattr(precision_policy, name)
if not set(precision_policy.__all__).issubset(dir(precision_policy)):
    raise AssertionError("lazy public exports are missing from dir()")
print("lazy-public-exports-ok")
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "."},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy-public-exports-ok"
