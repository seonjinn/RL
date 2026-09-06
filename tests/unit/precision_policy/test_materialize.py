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

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import pytest

from nemo_rl.precision_policy.adapters import (
    PrecisionTopologyAdapterBundle,
    validate_precision_adapter_bundles,
)
from nemo_rl.precision_policy.compiler import (
    CompiledPrecisionSelectionGroup,
    compile_precision_selection,
)
from nemo_rl.precision_policy.config import PrecisionPolicyConfig
from nemo_rl.precision_policy.semantic import DecoderLayerUniverse, GraphKind
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


@dataclass(frozen=True)
class _RuntimeAdapter:
    adapter_id: str

    def supports(self, model_config: Mapping[str, object]) -> bool:
        return bool(model_config)

    def classify_graph(
        self,
        schema_version: int,
        graph_input: GraphTopologyInput,
        source_records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment:
        raise AssertionError("registry validation must not classify a graph")


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
