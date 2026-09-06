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

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from unittest.mock import Mock

import cloudpickle
import pytest

import nemo_rl.precision_policy.materialize as materialize
import nemo_rl.precision_policy.topology as topology_module
from nemo_rl.precision_policy.adapters import (
    PrecisionTopologyAdapterBundle,
    validate_precision_adapter_bundles,
)
from nemo_rl.precision_policy.compiler import (
    CompiledPrecisionSelectionGroup,
    compile_precision_selection,
)
from nemo_rl.precision_policy.config import PrecisionPolicyConfig
from nemo_rl.precision_policy.materialize import (
    BoundSemanticPrecisionRuntimeContext,
    BoundSemanticPrecisionWorkerProjection,
    RuntimeAdapterAuthority,
    SemanticPrecisionBootstrap,
)
from nemo_rl.precision_policy.runtime_binding import (
    RuntimeSourceDiscoveryRequest,
    RuntimeSourceDiscoveryResult,
    build_runtime_source_discovery_request,
    build_runtime_source_discovery_result,
)
from nemo_rl.precision_policy.semantic import (
    DecoderLayerUniverse,
    GraphKind,
    ResolvedGraphTopology,
)
from nemo_rl.precision_policy.source_discovery import (
    GraphTopologyInput,
    SourceDiscoveryRecord,
)
from nemo_rl.precision_policy.topology import SemanticGraphBuildFragment
from nemo_rl.precision_policy.topology_resolver import (
    GraphTopologyResolutionRequest,
    freeze_phase1_requests_by_graph,
    phase1_request_identity_digest,
    phase1_request_set_digest,
    resolve_selection_topology,
    validate_graph_topology_resolution_request,
    validate_phase1_request_retention,
)
from tests.unit.precision_policy import test_topology_resolver
from tests.unit.precision_policy import test_runtime_binding as runtime_binding_fixtures


_IMPLEMENTATION_FINGERPRINT = "sha256:" + "a" * 64


@dataclass(frozen=True)
class _RuntimeAdapter:
    adapter_id: str
    graph: ResolvedGraphTopology | None = None
    observed_configs: list[Mapping[str, object]] | None = None
    implementation_fingerprint: str = _IMPLEMENTATION_FINGERPRINT

    def supports(self, model_config: Mapping[str, object]) -> bool:
        if self.observed_configs is not None:
            self.observed_configs.append(model_config)
        return model_config.get("model_type") == "test_model"

    def classify_graph(
        self,
        schema_version: int,
        graph_input: GraphTopologyInput,
        source_records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment:
        if self.graph is None:
            raise AssertionError("registry validation must not classify a graph")
        return runtime_binding_fixtures._RuntimeTopologyAdapter(
            self.adapter_id,
            self.graph,
        ).classify_graph(schema_version, graph_input, source_records)


def _enabled_policy_config() -> dict[str, object]:
    return {
        "model_name": "test/main",
        "generation": {"backend": "vllm", "model_name": "test/main"},
        "precision_policy": {"schema_version": 1, "scopes": []},
    }


def _legacy_policy_config_without_precision_policy() -> dict[str, object]:
    return {
        "model_name": "test/main",
        "generation": {
            "backend": "sglang",
            "model_path": "test/legacy-main",
        },
    }


def _main_request(
    *,
    revision: str = "revision-main",
    layer_count: int = 2,
    model_type: str = "test_model",
) -> GraphTopologyResolutionRequest:
    base = test_topology_resolver._request(
        universe=DecoderLayerUniverse(tuple(range(layer_count)), ()),
    )
    return GraphTopologyResolutionRequest(
        declaration=base.declaration,
        effective_model_config={
            "architectures": ["TestForCausalLM"],
            "graph_instance_id": "main",
            "model_type": model_type,
            "num_hidden_layers": layer_count,
        },
        resolved_model_revision=revision,
        decoder_layer_universe=DecoderLayerUniverse(tuple(range(layer_count)), ()),
    )


def _bundle_for_request(
    request: GraphTopologyResolutionRequest,
    *,
    adapter_id: str = "test.adapter.v1",
    observed_configs: list[Mapping[str, object]] | None = None,
) -> PrecisionTopologyAdapterBundle:
    graph_id = request.declaration.graph_instance_id
    entries = (
        test_topology_resolver._entry(
            graph_id,
            f"{graph_id}.dense",
            global_layers=request.decoder_layer_universe.global_decoder_layers,
        ),
    )
    selection = test_topology_resolver._adapter(
        {graph_id: entries},
        adapter_id=adapter_id,
        model_type=str(request.effective_model_config["model_type"]),
    )
    graph = selection.resolve_graph(request)
    return PrecisionTopologyAdapterBundle(
        adapter_id=adapter_id,
        selection=selection,
        runtime=_RuntimeAdapter(
            adapter_id,
            graph,
            observed_configs,
        ),
    )


def _materialized_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    *,
    request: GraphTopologyResolutionRequest | None = None,
    bundles: tuple[PrecisionTopologyAdapterBundle, ...] | None = None,
    policy_config: dict[str, object] | None = None,
) -> tuple[
    SemanticPrecisionBootstrap,
    GraphTopologyResolutionRequest,
    CompiledPrecisionSelectionGroup,
]:
    selected_request = request or _main_request()
    selected_bundles = (
        (_bundle_for_request(selected_request),) if bundles is None else bundles
    )
    monkeypatch.setattr(
        materialize,
        "build_graph_topology_resolution_requests",
        Mock(return_value=(selected_request,)),
    )
    bootstrap = SemanticPrecisionBootstrap(adapter_bundles=selected_bundles)
    selection = bootstrap.materialize(policy_config or _enabled_policy_config())
    assert selection is not None
    return bootstrap, selected_request, selection


def _phase2_artifacts(
    selection: CompiledPrecisionSelectionGroup,
    phase1_request: GraphTopologyResolutionRequest,
) -> tuple[RuntimeSourceDiscoveryRequest, tuple[RuntimeSourceDiscoveryResult, ...]]:
    configs = {"main": phase1_request.effective_model_config}
    graph_request = runtime_binding_fixtures._build_graph_request(
        selection,
        configs,
        "main",
    )
    expected = runtime_binding_fixtures._expected("main")
    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=(graph_request,),
        trusted_expected_contributors={"main": expected},
    )
    result = build_runtime_source_discovery_result(
        request=request,
        graph_request=graph_request,
        partition=runtime_binding_fixtures._partition(graph_request, expected),
    )
    return request, (result,)


def _phase1_fixture() -> tuple[
    CompiledPrecisionSelectionGroup,
    tuple[GraphTopologyResolutionRequest, ...],
]:
    main = test_topology_resolver._request(
        universe=DecoderLayerUniverse((0,), ()),
    )
    mtp = test_topology_resolver._request(
        "mtp.aux",
        GraphKind.MTP,
        universe=DecoderLayerUniverse((0,), ()),
    )
    selection_adapter = test_topology_resolver._adapter(
        {
            "main": (
                test_topology_resolver._entry(
                    "main",
                    "main.dense",
                    global_layers=(0,),
                ),
            ),
            "mtp.aux": (
                test_topology_resolver._entry(
                    "mtp.aux",
                    "mtp.aux.dense",
                    global_layers=(0,),
                ),
            ),
        }
    )
    topology = resolve_selection_topology(
        (main, mtp),
        1,
        adapters=(selection_adapter,),
    )
    selection = compile_precision_selection(
        PrecisionPolicyConfig.model_validate({"scopes": []}),
        topology,
    )
    return selection, (main, mtp)


@pytest.mark.parametrize("wrong_half", ("selection", "runtime"))
def test_precision_adapter_bundle_requires_both_exact_matching_ids(
    wrong_half: str,
) -> None:
    selection = test_topology_resolver._adapter(
        {},
        adapter_id=("other.adapter.v1" if wrong_half == "selection" else "adapter.v1"),
    )
    runtime = _RuntimeAdapter(
        "other.adapter.v1" if wrong_half == "runtime" else "adapter.v1"
    )

    with pytest.raises(ValueError, match=f"{wrong_half} adapter ID"):
        PrecisionTopologyAdapterBundle(
            adapter_id="adapter.v1",
            selection=selection,
            runtime=runtime,
        )


def test_precision_adapter_registry_is_unique_and_canonically_ordered() -> None:
    selection_a = test_topology_resolver._adapter({}, adapter_id="family.a.v1")
    selection_b = test_topology_resolver._adapter({}, adapter_id="family.b.v1")
    bundle_a = PrecisionTopologyAdapterBundle(
        "family.a.v1",
        selection_a,
        _RuntimeAdapter("family.a.v1"),
    )
    bundle_b = PrecisionTopologyAdapterBundle(
        "family.b.v1",
        selection_b,
        _RuntimeAdapter("family.b.v1"),
    )

    assert validate_precision_adapter_bundles((bundle_b, bundle_a)) == (
        bundle_a,
        bundle_b,
    )
    with pytest.raises(ValueError, match="duplicate precision topology adapter ID"):
        validate_precision_adapter_bundles((bundle_a, bundle_a))


def test_precision_adapter_registry_rejects_non_exact_records_and_ids() -> None:
    class AdapterIdSubclass(str):
        pass

    selection = test_topology_resolver._adapter({}, adapter_id="family.a.v1")
    with pytest.raises(ValueError, match="canonical non-empty text"):
        PrecisionTopologyAdapterBundle(
            AdapterIdSubclass("family.a.v1"),
            selection,
            _RuntimeAdapter("family.a.v1"),
        )
    with pytest.raises(TypeError, match="exact tuple"):
        validate_precision_adapter_bundles([])  # type: ignore[arg-type]


def test_phase1_request_retention_freezes_exact_canonical_graph_authority() -> None:
    selection, requests = _phase1_fixture()

    retained = freeze_phase1_requests_by_graph(requests)
    digest = phase1_request_set_digest(retained)

    assert type(retained) is MappingProxyType
    assert tuple(retained) == ("main", "mtp.aux")
    assert retained["main"] is requests[0]
    assert retained["mtp.aux"] is requests[1]
    assert digest.startswith("sha256:")
    assert validate_phase1_request_retention(selection, retained, digest) is retained


def test_phase1_request_identity_revalidates_derived_digest() -> None:
    _, requests = _phase1_fixture()
    request = requests[0]

    assert validate_graph_topology_resolution_request(request) is request
    assert phase1_request_identity_digest(request) == request.phase1_request_digest

    object.__setattr__(request, "phase1_request_digest", "sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="Phase 1 request digest mismatch"):
        phase1_request_identity_digest(request)


def test_phase1_request_retention_rejects_graph_and_aggregate_drift() -> None:
    selection, requests = _phase1_fixture()
    changed_requests = (
        replace(requests[0], resolved_model_revision="changed-revision"),
        requests[1],
    )
    retained = freeze_phase1_requests_by_graph(changed_requests)
    digest = phase1_request_set_digest(retained)

    with pytest.raises(ValueError, match="resolved model revision"):
        validate_phase1_request_retention(selection, retained, digest)

    original = freeze_phase1_requests_by_graph(requests)
    with pytest.raises(ValueError, match="Phase 1 request set digest mismatch"):
        validate_phase1_request_retention(
            selection,
            original,
            "sha256:" + "0" * 64,
        )


def test_phase1_request_retention_rejects_noncanonical_or_incomplete_mapping() -> None:
    selection, requests = _phase1_fixture()

    with pytest.raises(ValueError, match="canonical graph order"):
        freeze_phase1_requests_by_graph(tuple(reversed(requests)))

    retained = MappingProxyType({"main": requests[0]})
    with pytest.raises(ValueError, match="selection graph coverage"):
        validate_phase1_request_retention(
            selection,
            retained,
            phase1_request_set_digest(retained),
        )


def test_absent_policy_is_byte_for_byte_noop_and_skips_request_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _legacy_policy_config_without_precision_policy()
    before = json.dumps(config, sort_keys=False, separators=(",", ":")).encode()
    request_builder = Mock(side_effect=AssertionError("request builder called"))
    monkeypatch.setattr(
        materialize,
        "build_graph_topology_resolution_requests",
        request_builder,
    )

    bootstrap = SemanticPrecisionBootstrap(
        adapter_bundles=(_bundle_for_request(_main_request()),)
    )

    assert bootstrap.materialize(config) is None
    assert bootstrap.selection is None
    assert bootstrap.context is None
    assert json.dumps(config, sort_keys=False, separators=(",", ":")).encode() == before
    request_builder.assert_not_called()


def test_materialize_returns_the_same_frozen_selection_on_identical_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _main_request()
    request_builder = Mock(return_value=(request,))
    monkeypatch.setattr(
        materialize,
        "build_graph_topology_resolution_requests",
        request_builder,
    )
    bootstrap = SemanticPrecisionBootstrap(
        adapter_bundles=(_bundle_for_request(request),)
    )

    first = bootstrap.materialize(_enabled_policy_config())
    second = bootstrap.materialize(_enabled_policy_config())

    assert first is not None
    assert second is first
    assert bootstrap.selection is first


def test_materialize_rejects_policy_or_topology_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_request = _main_request()
    changed_request = _main_request(revision="revision-main-changed")
    monkeypatch.setattr(
        materialize,
        "build_graph_topology_resolution_requests",
        Mock(side_effect=((first_request,), (changed_request,))),
    )
    bootstrap = SemanticPrecisionBootstrap(
        adapter_bundles=(_bundle_for_request(first_request),)
    )
    assert bootstrap.materialize(_enabled_policy_config()) is not None

    with pytest.raises(ValueError, match="materialized Phase 1 input changed"):
        bootstrap.materialize(_enabled_policy_config())


def test_enabled_policy_with_no_matching_adapter_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _main_request(model_type="unknown_family")
    known_bundle = _bundle_for_request(_main_request())
    monkeypatch.setattr(
        materialize,
        "build_graph_topology_resolution_requests",
        Mock(return_value=(request,)),
    )

    with pytest.raises(ValueError, match="exactly one selection topology adapter"):
        SemanticPrecisionBootstrap(adapter_bundles=(known_bundle,)).materialize(
            _enabled_policy_config()
        )


def test_enabled_policy_with_ambiguous_adapters_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _main_request()
    adapter_ids = ("test.family-a.v1", "test.family-b.v1")
    bundles = tuple(
        _bundle_for_request(request, adapter_id=adapter_id)
        for adapter_id in adapter_ids
    )
    monkeypatch.setattr(
        materialize,
        "build_graph_topology_resolution_requests",
        Mock(return_value=(request,)),
    )

    with pytest.raises(ValueError) as error:
        SemanticPrecisionBootstrap(adapter_bundles=bundles).materialize(
            _enabled_policy_config()
        )

    message = str(error.value)
    assert adapter_ids[0] in message
    assert adapter_ids[1] in message


def test_bootstrap_owns_immutable_graph_keyed_phase1_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _main_request()
    config = _enabled_policy_config()
    before = json.dumps(config, sort_keys=False, separators=(",", ":")).encode()
    bootstrap, retained_request, selection = _materialized_bootstrap(
        monkeypatch,
        request=request,
        policy_config=config,
    )

    assert type(bootstrap.phase1_requests_by_graph) is MappingProxyType
    assert tuple(bootstrap.phase1_requests_by_graph) == ("main",)
    assert bootstrap.phase1_requests_by_graph["main"] is retained_request
    assert (
        retained_request.effective_model_config_digest
        == selection.topology.graphs[0].effective_model_config_digest
    )
    assert json.dumps(config, sort_keys=False, separators=(",", ":")).encode() == before

    config["model_name"] = "mutated/after-materialize"
    config["generation"] = {"backend": "dynamo"}
    assert retained_request.effective_model_config["model_type"] == "test_model"
    assert retained_request.resolved_model_revision == "revision-main"


def test_bootstrap_filters_registry_to_selected_runtime_adapter_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_request = _main_request()
    unused_request = _main_request(model_type="unused_family")
    selected = _bundle_for_request(
        selected_request,
        adapter_id="test.selected.v1",
    )
    unused = _bundle_for_request(
        unused_request,
        adapter_id="test.unused.v1",
    )
    bootstrap, _, _ = _materialized_bootstrap(
        monkeypatch,
        request=selected_request,
        bundles=(unused, selected),
    )

    assert tuple(bootstrap.runtime_adapters_by_id) == ("test.selected.v1",)
    assert bootstrap.runtime_adapters_by_id["test.selected.v1"] is selected.runtime
    assert bootstrap.worker_adapter_authority.adapter_ids == ("test.selected.v1",)
    assert bootstrap.worker_adapter_authority.implementation_fingerprints == (
        _IMPLEMENTATION_FINGERPRINT,
    )


def test_bind_preserves_complete_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap, phase1_request, selection = _materialized_bootstrap(monkeypatch)
    request, results = _phase2_artifacts(selection, phase1_request)
    monkeypatch.setattr(
        topology_module,
        "_default_adapters",
        Mock(side_effect=AssertionError("global adapter lookup used")),
    )

    context = bootstrap.bind_runtime_sources(request, results)

    assert type(context) is BoundSemanticPrecisionRuntimeContext
    assert context is bootstrap.context
    assert context.selection is bootstrap.selection
    assert context.source_request is request
    assert len(context.source_results) == len(results)
    assert all(
        retained is supplied
        for retained, supplied in zip(context.source_results, results, strict=True)
    )
    assert context.intents.selection is context.selection
    assert context.runtime_adapters_by_id is bootstrap.runtime_adapters_by_id
    assert context.adapter_authority is bootstrap.worker_adapter_authority
    assert context.selection.selection_group_id == selection.selection_group_id
    assert context.source_request.request_digest == request.request_digest
    assert context.runtime_context_id.startswith("sha256:")


def test_phase2_supports_uses_retained_graph_keyed_phase1_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _main_request()
    observed_configs: list[Mapping[str, object]] = []
    bundle = _bundle_for_request(request, observed_configs=observed_configs)
    caller_config = _enabled_policy_config()
    bootstrap, retained_request, selection = _materialized_bootstrap(
        monkeypatch,
        request=request,
        bundles=(bundle,),
        policy_config=caller_config,
    )
    caller_config["model_name"] = "mutated/after-materialize"
    caller_config["generation"] = {"backend": "dynamo"}
    runtime_request, results = _phase2_artifacts(selection, retained_request)

    bootstrap.bind_runtime_sources(runtime_request, results)

    assert observed_configs
    assert all(
        observed is retained_request.effective_model_config
        for observed in observed_configs
    )


def test_bind_cannot_run_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap, phase1_request, selection = _materialized_bootstrap(monkeypatch)
    request, results = _phase2_artifacts(selection, phase1_request)
    assert bootstrap.bind_runtime_sources(request, results) is not None

    with pytest.raises(RuntimeError, match="bind exactly once"):
        bootstrap.bind_runtime_sources(request, results)


def test_bound_context_publication_is_transactional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap, phase1_request, selection = _materialized_bootstrap(monkeypatch)
    request, results = _phase2_artifacts(selection, phase1_request)

    with monkeypatch.context() as patch:
        patch.setattr(
            materialize,
            "validate_worker_projection_round_trip",
            Mock(side_effect=ValueError("projection validation failed")),
        )
        with pytest.raises(ValueError, match="projection validation failed"):
            bootstrap.bind_runtime_sources(request, results)

    assert bootstrap.context is None
    assert bootstrap.bind_runtime_sources(request, results) is not None


def test_worker_adapter_authority_is_scalar_selected_and_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap, _, _ = _materialized_bootstrap(monkeypatch)
    authority = bootstrap.worker_adapter_authority
    payload = authority.to_wire_dict()

    assert payload == {
        "adapter_ids": ["test.adapter.v1"],
        "implementation_fingerprints": [_IMPLEMENTATION_FINGERPRINT],
        "authority_digest": authority.authority_digest,
    }
    assert RuntimeAdapterAuthority.from_wire_dict(payload) == authority
    assert "_RuntimeAdapter" not in json.dumps(payload, sort_keys=True)


def test_worker_projection_excludes_controller_trust_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap, phase1_request, selection = _materialized_bootstrap(monkeypatch)
    request, results = _phase2_artifacts(selection, phase1_request)
    context = bootstrap.bind_runtime_sources(request, results)
    assert context is not None
    projection = context.to_worker_projection(
        binding_phase="phase2_bound",
        bound_plan_group_ids=(),
    )
    payload = projection.to_wire_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert "main-rank-0" not in encoded
    assert "runtime://main-contributors" not in encoded
    assert "_RuntimeAdapter" not in encoded
    decoded = BoundSemanticPrecisionWorkerProjection.from_wire_dict(
        payload,
        expected_adapter_authority=bootstrap.worker_adapter_authority,
    )
    assert decoded.to_wire_dict() == payload
    with pytest.raises(TypeError, match="expected_adapter_authority"):
        BoundSemanticPrecisionWorkerProjection.from_wire_dict(  # type: ignore[call-arg]
            payload
        )

    foreign_authority = RuntimeAdapterAuthority(
        adapter_ids=("test.foreign.v1",),
        implementation_fingerprints=("sha256:" + "f" * 64,),
    )
    with pytest.raises(ValueError, match="adapter authority"):
        BoundSemanticPrecisionWorkerProjection.from_wire_dict(
            payload,
            expected_adapter_authority=foreign_authority,
        )


def test_full_runtime_context_is_not_serializable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap, phase1_request, selection = _materialized_bootstrap(monkeypatch)
    request, results = _phase2_artifacts(selection, phase1_request)
    context = bootstrap.bind_runtime_sources(request, results)
    assert context is not None

    for serializer in (pickle.dumps, cloudpickle.dumps):
        with pytest.raises(
            TypeError,
            match="bound semantic precision runtime context is controller-local",
        ):
            serializer(context)

    assert not hasattr(context, "to_wire_dict")
    projection = context.to_worker_projection(
        binding_phase="phase2_bound",
        bound_plan_group_ids=(),
    )
    assert pickle.loads(pickle.dumps(projection)) == projection
    assert cloudpickle.loads(cloudpickle.dumps(projection)) == projection
