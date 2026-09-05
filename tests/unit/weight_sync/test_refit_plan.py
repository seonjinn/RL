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

import importlib.util
import pickle
import sys
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from pathlib import Path
from types import ModuleType

import pytest

from nemo_rl.precision_policy.compiler import (
    CompiledPrecisionIntentGroup,
    CompiledPrecisionSelectionGroup,
    RuntimeSourceBindingSlice,
    compile_precision_selection,
)
from nemo_rl.precision_policy.config import PrecisionPolicyConfig
from nemo_rl.precision_policy.runtime_binding import (
    RuntimeSourceDiscoveryRequest,
    RuntimeSourceDiscoveryResult,
    bind_runtime_source_intents,
    build_runtime_source_discovery_request,
    build_runtime_source_discovery_result,
)
from nemo_rl.precision_policy.source_discovery import (
    DiscoveryContribution,
    GraphTopologyInput,
    SourceDiscoveryRecord,
    assemble_runtime_graph_discovery_partition,
)
from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    BLOCK_SCALES,
    LOGICAL_VALUES,
    MXFP8_FORMAT,
    VALUES,
    AxisProjection,
    ComponentDescriptor,
    ComponentRole,
    FormatDescriptor,
    OwnerFamilyBinding,
    OwnerFamilyReference,
    ParameterInventoryEntry,
    ResolvedGraphTopology,
    SemanticGraphManifest,
    SemanticOwnership,
    SemanticTensorFamily,
    SourceOwnerInventoryEntry,
    ValueProvenance,
)
from nemo_rl.precision_policy.source_dtype import CanonicalSourceDType
from nemo_rl.precision_policy.source_storage import (
    IDENTITY_PERMUTATION_ID,
    IDENTITY_SWIZZLE_ID,
    SourceExtentRounding,
    SourceNormalizationContract,
    SourceNormalizationKind,
    SourceNormalizerManifest,
    SourceNormalizedAxisExtent,
    SourcePaddingSemantics,
    SourcePhysicalAxisSpec,
    SourceStorageComponent,
    SourceStorageRealization,
    SourceStorageRealizationInventory,
    source_normalizer_manifest_digest,
)
from nemo_rl.precision_policy.topology import (
    CanonicalValueClassificationEdge,
    ComponentAxisTarget,
    LayerCoordinateTarget,
    OutputMemberTarget,
    RoleDefinitionContribution,
    SemanticGraphBuildFragment,
    SourceAxisSelection,
    SourceIndexSpan,
    SourceOrdinalMapSegment,
    SourceRegion,
    SourceToSemanticAxisMapping,
)


def _load_refit_plan() -> ModuleType:
    module_path = Path(__file__).parents[3] / "nemo_rl/weight_sync/refit_plan.py"
    spec = importlib.util.spec_from_file_location("_refit_plan_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_REFIT_PLAN = _load_refit_plan()
AdapterCapabilityRegistry = _REFIT_PLAN.AdapterCapabilityRegistry
AdapterOperationCapability = _REFIT_PLAN.AdapterOperationCapability
DirectCopyCapabilityProof = _REFIT_PLAN.DirectCopyCapabilityProof
DestinationBindingEvidence = _REFIT_PLAN.DestinationBindingEvidence
DestinationBindingProof = _REFIT_PLAN.DestinationBindingProof
EndpointCapabilityProof = _REFIT_PLAN.EndpointCapabilityProof
EndpointPlacement = _REFIT_PLAN.EndpointPlacement
PhysicalAxisMapping = _REFIT_PLAN.PhysicalAxisMapping
PhysicalComponentDescriptor = _REFIT_PLAN.PhysicalComponentDescriptor
PhysicalFormatStage = _REFIT_PLAN.PhysicalFormatStage
PhysicalLayoutDescriptor = _REFIT_PLAN.PhysicalLayoutDescriptor
PhysicalPadding = _REFIT_PLAN.PhysicalPadding
PhysicalPaddingSemantics = _REFIT_PLAN.PhysicalPaddingSemantics
PhysicalPermutation = _REFIT_PLAN.PhysicalPermutation
PhysicalRepresentation = _REFIT_PLAN.PhysicalRepresentation
PhysicalRouteDescriptor = _REFIT_PLAN.PhysicalRouteDescriptor
RealizedBindingFormat = _REFIT_PLAN.RealizedBindingFormat
RefitBindingIdentity = _REFIT_PLAN.RefitBindingIdentity
RefitExecutorKey = _REFIT_PLAN.RefitExecutorKey
RefitPlanInstallationRegistry = _REFIT_PLAN.RefitPlanInstallationRegistry
SelectedRefitOperation = _REFIT_PLAN.SelectedRefitOperation
TransformCapabilityProof = _REFIT_PLAN.TransformCapabilityProof
TransformLocus = _REFIT_PLAN.TransformLocus
endpoint_placement_digest = _REFIT_PLAN.endpoint_placement_digest
execution_dispatch_key = _REFIT_PLAN.execution_dispatch_key
physical_representation_digest = _REFIT_PLAN.physical_representation_digest
require_direct_copy = _REFIT_PLAN.require_direct_copy
_select_transform = _REFIT_PLAN.select_transform


_OTHER_CAPABILITY_FINGERPRINT = f"sha256:{'b' * 64}"
_STAGE_CAPABILITY_FINGERPRINTS = (
    f"sha256:{'1' * 64}",
    f"sha256:{'2' * 64}",
    f"sha256:{'3' * 64}",
    f"sha256:{'4' * 64}",
)
_TEST_REGISTRIES: dict[str, AdapterCapabilityRegistry] = {}
_TEST_DESTINATION_EVIDENCE: dict[str, DestinationBindingEvidence] = {}
_TEST_DESTINATION_LIVE_OBJECTS: dict[str, tuple[object, object, object]] = {}
_TEST_DESTINATION_AUTHORITIES: dict[str, object] = {}


def _adapter_registry(
    *,
    adapter_id: str,
    adapter_version: str,
    capabilities: tuple[AdapterOperationCapability, ...],
) -> AdapterCapabilityRegistry:
    adapter_instance = object()
    registry = _REFIT_PLAN._create_adapter_capability_registry(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        capabilities=capabilities,
        destination_adapter_instance=adapter_instance,
    )
    _TEST_DESTINATION_AUTHORITIES[registry.issuer_instance_id] = (
        _REFIT_PLAN._destination_binding_mint_authority_for_adapter(
            registry,
            adapter_instance=adapter_instance,
        )
    )
    return registry


def _destination_evidence(
    registry: AdapterCapabilityRegistry,
    realized_format: RealizedBindingFormat,
    *,
    storage_generation: object | None = None,
) -> DestinationBindingEvidence:
    destination_owner = object()
    finalizer = object()
    generation = object() if storage_generation is None else storage_generation
    authority = _TEST_DESTINATION_AUTHORITIES.get(registry.issuer_instance_id)
    assert authority is not None
    generation_lease = authority.begin_storage_generation(generation)
    owner_handle = authority.register_destination_owner(destination_owner)
    finalizer_handle = authority.register_finalizer(finalizer)
    evidence = authority.discover_destination_binding(
        realized_format,
        destination_owner=owner_handle,
        finalizer=finalizer_handle,
        storage_generation=generation_lease,
    )
    _TEST_DESTINATION_LIVE_OBJECTS[registry.issuer_instance_id] = (
        destination_owner,
        finalizer,
        generation,
    )
    return evidence


def _routes() -> tuple[PhysicalRouteDescriptor, ...]:
    endpoint_ids = (
        "source-endpoint-0",
        "wire-endpoint-1",
        "load-api-endpoint-2",
        "runtime-endpoint-3",
    )
    stages = tuple(PhysicalFormatStage)
    return tuple(
        PhysicalRouteDescriptor(
            source_stage=source_stage,
            destination_stage=destination_stage,
            route_id=f"route-{source_stage.value}-{destination_stage.value}",
            source_endpoint_instance_id=endpoint_ids[index],
            destination_endpoint_instance_id=endpoint_ids[index + 1],
            source_endpoint_capability_fingerprint=(
                _STAGE_CAPABILITY_FINGERPRINTS[index]
            ),
            destination_endpoint_capability_fingerprint=(
                _STAGE_CAPABILITY_FINGERPRINTS[index + 1]
            ),
        )
        for index, (source_stage, destination_stage) in enumerate(
            zip(stages, stages[1:])
        )
    )


def _identity_layout(
    *axis_order: str, storage_encoding: str
) -> PhysicalLayoutDescriptor:
    return PhysicalLayoutDescriptor(
        axis_order=axis_order,
        logical_to_physical_axes=tuple(
            PhysicalAxisMapping(
                logical_axis=axis,
                physical_axes=(axis,),
                mapping_id="identity.axis-map.v1",
            )
            for axis in axis_order
        ),
        padding=(),
        permutation=None,
        storage_encoding=storage_encoding,
    )


def _component(
    role: ComponentRole,
    *,
    dtype: str = "bfloat16",
    shape: tuple[int, ...] = (8, 8),
    layout: PhysicalLayoutDescriptor | None = None,
    rank: int = 0,
    device_type: str = "cuda",
    memory_space: str = "device",
    source_component_id: str = "main.dense.weight.component",
    source_native_name: str = "main.model.weight",
    source_component_role: str = "normalized_values",
    source_storage_encoding: str = "plain_bfloat16",
) -> PhysicalComponentDescriptor:
    resolved_layout = layout or PhysicalLayoutDescriptor(
        axis_order=("axis_0", "axis_1"),
        logical_to_physical_axes=(
            PhysicalAxisMapping("output_features", ("axis_0",), "identity.axis-map.v1"),
            PhysicalAxisMapping("input_features", ("axis_1",), "identity.axis-map.v1"),
        ),
        padding=(),
        permutation=None,
        storage_encoding="plain-bfloat16.v1",
    )
    source_component = SourceStorageComponent(
        graph_instance_id="main",
        native_component_id=source_component_id,
        source_native_name=source_native_name,
        component_role=source_component_role,
        carrier_dtype=CanonicalSourceDType(dtype),
        physical_shape=shape,
        physical_axes=tuple(
            SourcePhysicalAxisSpec(
                axis_name=axis_name,
                extent=SourceNormalizedAxisExtent(
                    normalized_axis_indices=(axis_index,),
                    divisor=1,
                    rounding=SourceExtentRounding.EXACT,
                    alignment=1,
                ),
            )
            for axis_index, axis_name in enumerate(resolved_layout.axis_order)
        ),
        storage_encoding=source_storage_encoding,
        padding_semantics=SourcePaddingSemantics.NO_PADDING,
        padding_fill_encoding=None,
        permutation_id=(
            IDENTITY_PERMUTATION_ID
            if resolved_layout.permutation is None
            else resolved_layout.permutation.permutation_id
        ),
        swizzle_id=resolved_layout.swizzle_id or IDENTITY_SWIZZLE_ID,
    )
    return PhysicalComponentDescriptor(
        representation=PhysicalRepresentation(
            role=role,
            physical_dtype=dtype,
            physical_shape=shape,
            layout=resolved_layout,
        ),
        placement=EndpointPlacement(
            rank=rank,
            device_type=device_type,
            memory_space=memory_space,
        ),
        source_storage_component=source_component,
    )


def _format(
    *,
    source_storage: tuple[PhysicalComponentDescriptor, ...],
    wire: tuple[PhysicalComponentDescriptor, ...] | None = None,
    destination_load_api: tuple[PhysicalComponentDescriptor, ...] | None = None,
    destination_runtime: tuple[PhysicalComponentDescriptor, ...] | None = None,
    source_storage_format: FormatDescriptor | None = None,
    wire_format: FormatDescriptor | None = None,
    destination_load_api_format: FormatDescriptor | None = None,
    destination_runtime_format: FormatDescriptor | None = None,
    routes: tuple[PhysicalRouteDescriptor, ...] | None = None,
) -> RealizedBindingFormat:
    resolved_wire = wire or source_storage
    resolved_load_api = destination_load_api or resolved_wire
    resolved_runtime = destination_runtime or resolved_load_api
    resolved_source_format = source_storage_format or BF16_FORMAT
    resolved_wire_format = wire_format or resolved_source_format
    resolved_load_api_format = destination_load_api_format or resolved_wire_format
    resolved_runtime_format = destination_runtime_format or resolved_load_api_format
    format_schema_registry = _REFIT_PLAN._create_format_schema_registry(
        (
            resolved_source_format,
            resolved_wire_format,
            resolved_load_api_format,
            resolved_runtime_format,
        )
    )
    return RealizedBindingFormat(
        source_storage=source_storage,
        wire=resolved_wire,
        destination_load_api=resolved_load_api,
        destination_runtime=resolved_runtime,
        source_storage_format=resolved_source_format,
        wire_format=resolved_wire_format,
        destination_load_api_format=resolved_load_api_format,
        destination_runtime_format=resolved_runtime_format,
        format_schema_registry=format_schema_registry,
        routes=_routes() if routes is None else routes,
    )


def _binding_context(
    *,
    allocation_generation: str = "allocation-1",
    destination_precision: str = "bf16",
    padded_source: bool = False,
    installation_registry: object | None = None,
) -> object:
    return _bind_refit_context_inputs(
        _binding_context_inputs(
            allocation_generation=allocation_generation,
            destination_precision=destination_precision,
            padded_source=padded_source,
        ),
        installation_registry=installation_registry,
    )


def _bind_refit_context_inputs(
    inputs: dict[str, object],
    *,
    installation_registry: object | None = None,
) -> object:
    from tests.unit.precision_policy import test_runtime_binding

    kwargs = dict(inputs)
    kwargs["installation_registry"] = (
        RefitPlanInstallationRegistry()
        if installation_registry is None
        else installation_registry
    )
    adapters = kwargs.pop("_runtime_topology_adapters")
    assert isinstance(adapters, tuple)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            test_runtime_binding.topology_module,
            "_default_adapters",
            lambda: adapters,
        )
        return _REFIT_PLAN.bind_refit_context(**kwargs)


def _install_refit_context_inputs(
    context: object,
    inputs: dict[str, object],
    *,
    installation_registry: object | None = None,
) -> object:
    from tests.unit.precision_policy import test_runtime_binding

    kwargs = dict(inputs)
    kwargs["installation_registry"] = (
        RefitPlanInstallationRegistry()
        if installation_registry is None
        else installation_registry
    )
    adapters = kwargs.pop("_runtime_topology_adapters")
    kwargs.pop("source_binding_slice")
    kwargs.pop("source_realization")
    assert isinstance(adapters, tuple)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            test_runtime_binding.topology_module,
            "_default_adapters",
            lambda: adapters,
        )
        return _REFIT_PLAN.install_refit_context(context=context, **kwargs)


def _binding_context_inputs(
    *,
    allocation_generation: str = "allocation-1",
    destination_precision: str = "bf16",
    padded_source: bool = False,
) -> dict[str, object]:
    from tests.unit.precision_policy import test_runtime_binding

    selection, configs = test_runtime_binding._selection_fixture()
    if destination_precision == "mxfp8":
        selection = compile_precision_selection(
            PrecisionPolicyConfig.model_validate(
                {
                    "scopes": [
                        {
                            "id": "dense-rollout",
                            "advanced_match": {
                                "graph_instance_id": "main",
                                "semantic_graph_path": "text.decoder",
                                "module_kind": "ffn.dense",
                            },
                            "rollout": "mxfp8",
                        }
                    ]
                }
            ),
            selection.topology,
        )
    else:
        assert destination_precision == "bf16"
    graph_requests = tuple(
        replace(
            graph_request,
            source_allocation_generation=allocation_generation,
        )
        for graph_request in test_runtime_binding._runtime_requests(selection, configs)
    )
    padded_inventory = None
    if padded_source:
        normalization = SourceNormalizationContract(
            capability_id="test.padded-crop.v1",
            kind=SourceNormalizationKind.CROP,
            contract_digest=f"sha256:{'c' * 64}",
        )
        manifest = SourceNormalizerManifest(
            schema_version=1,
            contracts=(normalization,),
        )
        graph_requests = tuple(
            replace(
                graph_request,
                source_producer_fingerprint=replace(
                    graph_request.source_producer_fingerprint,
                    normalization_contract_digest=source_normalizer_manifest_digest(
                        manifest
                    ),
                ),
            )
            if graph_request.declaration.graph_instance_id == "main"
            else graph_request
            for graph_request in graph_requests
        )
        record = test_runtime_binding._source_record("main")
        padded_inventory = SourceStorageRealizationInventory(
            graph_instance_id="main",
            normalizer_manifest=manifest,
            realizations=(
                SourceStorageRealization(
                    realization_id="main.dense.weight.padded",
                    graph_instance_id="main",
                    output_record_id=record.record_id,
                    components=(
                        SourceStorageComponent(
                            graph_instance_id="main",
                            native_component_id="main.dense.weight.padded.component",
                            source_native_name=record.source_native_name or "",
                            component_role="normalized_values",
                            carrier_dtype=record.dtype,
                            physical_shape=(8, 16),
                            physical_axes=(
                                SourcePhysicalAxisSpec(
                                    "axis_0",
                                    SourceNormalizedAxisExtent(
                                        (0,),
                                        1,
                                        SourceExtentRounding.EXACT,
                                        1,
                                    ),
                                ),
                                SourcePhysicalAxisSpec(
                                    "axis_1",
                                    SourceNormalizedAxisExtent(
                                        (1,),
                                        1,
                                        SourceExtentRounding.EXACT,
                                        16,
                                    ),
                                ),
                            ),
                            storage_encoding="padded_bfloat16",
                            padding_semantics=SourcePaddingSemantics.ZERO_FILLED,
                            padding_fill_encoding="zero_bfloat16",
                            permutation_id=IDENTITY_PERMUTATION_ID,
                            swizzle_id=IDENTITY_SWIZZLE_ID,
                        ),
                    ),
                    output_dtype=record.dtype,
                    output_shape=record.shape,
                    output_numeric_encoding=record.numeric_encoding,
                    normalization=normalization,
                ),
            ),
        )
    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=graph_requests,
        trusted_expected_contributors={
            graph_request.declaration.graph_instance_id: (
                test_runtime_binding._expected(
                    graph_request.declaration.graph_instance_id
                )
            )
            for graph_request in graph_requests
        },
    )
    results = []
    for graph_request in request.graph_requests:
        graph_id = graph_request.declaration.graph_instance_id
        expected = test_runtime_binding._expected(graph_id)
        if graph_id == "main" and padded_inventory is not None:
            record = test_runtime_binding._source_record(graph_id)
            partition = assemble_runtime_graph_discovery_partition(
                runtime_request=graph_request,
                expected_contributors=expected,
                contributions=(
                    DiscoveryContribution(
                        contributor_id=expected.contributor_ids[0],
                        graph_instance_id=graph_id,
                        producer_fingerprint=(
                            graph_request.source_producer_fingerprint
                        ),
                        records=(record,),
                        storage_realizations=padded_inventory,
                    ),
                ),
            )
        else:
            partition = test_runtime_binding._partition(graph_request, expected)
        results.append(
            build_runtime_source_discovery_result(
                request=request,
                graph_request=graph_request,
                partition=partition,
            )
        )
    result_tuple = tuple(results)
    with pytest.MonkeyPatch.context() as monkeypatch:
        test_runtime_binding._install_runtime_topology_adapters(
            monkeypatch,
            selection,
        )
        intents = bind_runtime_source_intents(selection, request, result_tuple)
        runtime_topology_adapters = tuple(
            test_runtime_binding._RuntimeTopologyAdapter(graph.adapter_id, graph)
            for graph in selection.topology.graphs
            if graph.declaration.lifecycle.graph_provenance.value == "training_runtime"
        )
    matching_slices = tuple(
        source_slice
        for source_slice in intents.graph_intent("main").source_binding_slices
        if source_slice.rollout_assignment.precision == destination_precision
    )
    assert len(matching_slices) == 1
    source_binding_slice = matching_slices[0]
    return {
        "intents": intents,
        "active_selection": selection,
        "active_request": request,
        "active_results": result_tuple,
        "_runtime_topology_adapters": runtime_topology_adapters,
        "source_binding_slice": source_binding_slice,
        "source_realization": source_binding_slice.source_binding.source_realizations[
            0
        ],
    }


@dataclass(frozen=True)
class _NativeMxfp8RuntimeTopologyAdapter:
    adapter_id: str
    graph: ResolvedGraphTopology

    def supports(self, _model_config: Mapping[str, object]) -> bool:
        raise AssertionError("Phase 2 must select the resolved adapter by identity")

    def classify_graph(
        self,
        schema_version: int,
        graph_input: GraphTopologyInput,
        source_records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment:
        from tests.unit.precision_policy import test_runtime_binding

        records_by_encoding = {
            record.numeric_encoding: record for record in source_records
        }
        values_record = records_by_encoding["mxfp8_e4m3_values"]
        scales_record = records_by_encoding["mxfp8_e8m0_scale"]
        base = test_runtime_binding._RuntimeTopologyAdapter(
            self.adapter_id,
            self.graph,
        ).classify_graph(schema_version, graph_input, (values_record,))
        base_entry = base.inventory_entries[0]
        entry = replace(
            base_entry,
            member=replace(base_entry.member, format=MXFP8_FORMAT),
        )
        base_edge = base.classification_edges[0]
        values_mappings = tuple(
            replace(
                mapping,
                target=ComponentAxisTarget(
                    VALUES,
                    mapping.target.component_axis,
                ),
            )
            if isinstance(mapping.target, ComponentAxisTarget)
            else mapping
            for mapping in base_edge.axis_mappings
        )
        values_edge = replace(
            base_edge,
            component_role=VALUES,
            axis_mappings=values_mappings,
        )
        scales_edge = replace(
            base_edge,
            record_id=scales_record.record_id,
            source_region=SourceRegion(
                source_shape=scales_record.shape,
                axis_selections=tuple(
                    SourceAxisSelection(
                        axis_index,
                        (SourceIndexSpan(0, extent),),
                    )
                    for axis_index, extent in enumerate(scales_record.shape)
                ),
            ),
            component_role=BLOCK_SCALES,
            axis_mappings=tuple(
                SourceToSemanticAxisMapping(
                    source_axis_index=axis_index,
                    target=ComponentAxisTarget(BLOCK_SCALES, logical_axis),
                    segments=(
                        SourceOrdinalMapSegment(
                            SourceIndexSpan(0, extent),
                            0,
                        ),
                    ),
                )
                for axis_index, (extent, logical_axis) in enumerate(
                    zip(
                        scales_record.shape,
                        ("output_features", "input_features"),
                        strict=True,
                    )
                )
            ),
        )
        return replace(
            base,
            classification_edges=(values_edge, scales_edge),
            inventory_entries=(entry,),
        )


def _native_mxfp8_binding_inputs() -> dict[str, object]:
    from tests.unit.precision_policy import test_runtime_binding

    base_selection, configs = test_runtime_binding._selection_fixture()
    selection = compile_precision_selection(
        PrecisionPolicyConfig.model_validate(
            {
                "scopes": [
                    {
                        "id": "native-mxfp8",
                        "advanced_match": {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "module_kind": "ffn.dense",
                        },
                        "training": "mxfp8",
                        "rollout": "mxfp8",
                    }
                ]
            }
        ),
        base_selection.topology,
    )
    graph_requests = test_runtime_binding._runtime_requests(selection, configs)
    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=graph_requests,
        trusted_expected_contributors={
            graph_request.declaration.graph_instance_id: (
                test_runtime_binding._expected(
                    graph_request.declaration.graph_instance_id
                )
            )
            for graph_request in graph_requests
        },
    )
    results = []
    for graph_request in request.graph_requests:
        graph_id = graph_request.declaration.graph_instance_id
        expected = test_runtime_binding._expected(graph_id)
        if graph_id != "main":
            partition = test_runtime_binding._partition(graph_request, expected)
        else:
            values_record = replace(
                test_runtime_binding._source_record(graph_id),
                record_id="main.dense.weight.values",
                dtype=CanonicalSourceDType.E4M3,
                numeric_encoding="mxfp8_e4m3_values",
            )
            scales_record = replace(
                values_record,
                record_id="main.dense.weight.block-scales",
                source_native_name="main.model.weight_scale_inv",
                dtype=CanonicalSourceDType.E8M0,
                shape=(8, 1),
                numeric_encoding="mxfp8_e8m0_scale",
            )
            values_inventory = test_runtime_binding._storage_realizations(values_record)
            scales_inventory = test_runtime_binding._storage_realizations(scales_record)
            values_realization = replace(
                values_inventory.realizations[0],
                components=(
                    replace(
                        values_inventory.realizations[0].components[0],
                        component_role=VALUES.value,
                    ),
                ),
            )
            scales_realization = replace(
                scales_inventory.realizations[0],
                components=(
                    replace(
                        scales_inventory.realizations[0].components[0],
                        component_role=BLOCK_SCALES.value,
                    ),
                ),
            )
            inventory = SourceStorageRealizationInventory(
                graph_instance_id=graph_id,
                normalizer_manifest=values_inventory.normalizer_manifest,
                realizations=(values_realization, scales_realization),
            )
            partition = assemble_runtime_graph_discovery_partition(
                runtime_request=graph_request,
                expected_contributors=expected,
                contributions=(
                    DiscoveryContribution(
                        contributor_id=expected.contributor_ids[0],
                        graph_instance_id=graph_id,
                        producer_fingerprint=graph_request.source_producer_fingerprint,
                        records=(values_record, scales_record),
                        storage_realizations=inventory,
                    ),
                ),
            )
        results.append(
            build_runtime_source_discovery_result(
                request=request,
                graph_request=graph_request,
                partition=partition,
            )
        )
    active_results = tuple(results)
    adapters = tuple(
        (
            _NativeMxfp8RuntimeTopologyAdapter(graph.adapter_id, graph)
            if graph.declaration.graph_instance_id == "main"
            else test_runtime_binding._RuntimeTopologyAdapter(graph.adapter_id, graph)
        )
        for graph in selection.topology.graphs
        if graph.declaration.lifecycle.graph_provenance.value == "training_runtime"
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            test_runtime_binding.topology_module,
            "_default_adapters",
            lambda: adapters,
        )
        intents = bind_runtime_source_intents(selection, request, active_results)
    slices = intents.graph_intent("main").source_binding_slices
    return {
        "intents": intents,
        "active_selection": selection,
        "active_request": request,
        "active_results": active_results,
        "_runtime_topology_adapters": adapters,
        "bindings": tuple(
            (source_slice, source_slice.source_binding.source_realizations[0])
            for source_slice in slices
        ),
    }


@dataclass(frozen=True)
class _FusedQkvRuntimeTopologyAdapter:
    adapter_id: str
    graph: ResolvedGraphTopology

    def supports(self, _model_config: Mapping[str, object]) -> bool:
        raise AssertionError("Phase 2 must select the resolved adapter by identity")

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
        assert record.shape[0] == len(self.graph.entries)
        owner_reference = OwnerFamilyReference("main", "source.attention.qkv")
        inventory_entries = []
        classification_edges = []
        for projection_index, selection_entry in enumerate(self.graph.entries):
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
            inventory_entries.append(
                ParameterInventoryEntry(
                    entry_id=selection_entry.entry_id,
                    graph_instance_id="main",
                    member=member,
                    value_provenance=ValueProvenance.TRAINING_PARAMETER,
                )
            )
            classification_edges.append(
                CanonicalValueClassificationEdge(
                    record_id=record.record_id,
                    source_region=SourceRegion(
                        source_shape=record.shape,
                        axis_selections=(
                            SourceAxisSelection(
                                0,
                                (
                                    SourceIndexSpan(
                                        projection_index,
                                        projection_index + 1,
                                    ),
                                ),
                            ),
                            *tuple(
                                SourceAxisSelection(
                                    axis_index,
                                    (SourceIndexSpan(0, extent),),
                                )
                                for axis_index, extent in enumerate(
                                    record.shape[1:],
                                    start=1,
                                )
                            ),
                        ),
                    ),
                    output=OutputMemberTarget(
                        inventory_entry_id=selection_entry.entry_id,
                        member_domain=selection_entry.domain,
                        fixed_coordinates=(),
                    ),
                    canonical_owner_family=owner_reference,
                    component_role=LOGICAL_VALUES,
                    axis_mappings=(
                        SourceToSemanticAxisMapping(
                            1,
                            LayerCoordinateTarget("global_decoder_layer"),
                            (
                                SourceOrdinalMapSegment(
                                    SourceIndexSpan(0, record.shape[1]),
                                    0,
                                ),
                            ),
                        ),
                        *tuple(
                            SourceToSemanticAxisMapping(
                                axis_index,
                                ComponentAxisTarget(LOGICAL_VALUES, logical_axis),
                                (
                                    SourceOrdinalMapSegment(
                                        SourceIndexSpan(0, record.shape[axis_index]),
                                        0,
                                    ),
                                ),
                            )
                            for axis_index, logical_axis in zip(
                                (2, 3),
                                selection_entry.logical_axes,
                                strict=True,
                            )
                        ),
                    ),
                )
            )
        attention_definition = next(
            definition
            for definition in self.graph.role_definitions
            if definition.role_name == "attention.qkvo"
        )
        return SemanticGraphBuildFragment(
            graph_instance_id="main",
            classification_edges=tuple(classification_edges),
            source_owners=(
                SourceOwnerInventoryEntry(
                    owner_family=owner_reference,
                    domain=self.graph.entries[0].domain,
                    source_mutability=record.source_mutability,
                    mutability_evidence_source=record.mutability_evidence,
                ),
            ),
            inventory_entries=tuple(inventory_entries),
            manifest=SemanticGraphManifest(
                model_family=self.graph.model_family,
                model_revision=self.graph.resolved_model_revision,
                graph_instance_id="main",
                lifecycle=self.graph.declaration.lifecycle,
                inventory_entry_ids=tuple(
                    entry.entry_id for entry in inventory_entries
                ),
                atomic_groups=self.graph.atomic_groups,
            ),
            role_contributions=(
                RoleDefinitionContribution(
                    schema_version=1,
                    role_name=attention_definition.role_name,
                    predicate=attention_definition.predicate,
                    expected_inventory_entry_ids=tuple(
                        entry.entry_id for entry in inventory_entries
                    ),
                ),
            ),
        )


def _fused_qkv_binding_inputs() -> dict[str, object]:
    from tests.unit.precision_policy import test_compiler, test_runtime_binding

    config = test_runtime_binding._model_config("main")
    topology = test_compiler._selection_atomic_topology(
        ("q", "k", "v"),
        (("attention.qkv", ("q", "k", "v")),),
        attention=True,
    )
    graph = replace(
        topology.graphs[0],
        effective_model_config_digest=test_runtime_binding.canonical_model_config_digest(
            config
        ),
    )
    topology = replace(
        topology,
        graphs=(graph,),
        semantic_structure_digest=(
            test_runtime_binding._compute_semantic_structure_digest(
                schema_version=1,
                graphs=(graph,),
                role_definitions=topology.role_definitions,
            )
        ),
    )
    selection = compile_precision_selection(
        PrecisionPolicyConfig.model_validate(
            {
                "scopes": [
                    {
                        "id": "qkv-bf16",
                        "roles": ["attention.qkvo"],
                        "training": "bf16",
                        "rollout": "bf16",
                    }
                ]
            }
        ),
        topology,
    )
    graph_request = test_runtime_binding._build_graph_request(
        selection,
        {"main": config},
        "main",
    )
    request = build_runtime_source_discovery_request(
        selection=selection,
        graph_requests=(graph_request,),
        trusted_expected_contributors={"main": test_runtime_binding._expected("main")},
    )
    record = replace(
        test_runtime_binding._source_record("main"),
        record_id="main.attention.qkv.fused",
        source_native_name="main.model.attention.qkv.weight",
        source_native_owner_id="main.model.attention.qkv",
        shape=(3, 2, 8, 8),
    )
    partition = assemble_runtime_graph_discovery_partition(
        runtime_request=graph_request,
        expected_contributors=test_runtime_binding._expected("main"),
        contributions=(
            DiscoveryContribution(
                contributor_id="main-rank-0",
                graph_instance_id="main",
                producer_fingerprint=graph_request.source_producer_fingerprint,
                records=(record,),
                storage_realizations=test_runtime_binding._storage_realizations(record),
            ),
        ),
    )
    results = (
        build_runtime_source_discovery_result(
            request=request,
            graph_request=graph_request,
            partition=partition,
        ),
    )
    adapter = _FusedQkvRuntimeTopologyAdapter(graph.adapter_id, graph)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            test_runtime_binding.topology_module,
            "_default_adapters",
            lambda: (adapter,),
        )
        intents = bind_runtime_source_intents(selection, request, results)
    q_slice = next(
        source_slice
        for source_slice in intents.graph_intent("main").source_binding_slices
        if source_slice.component_key.inventory_entry_id == "weight-q"
    )
    return {
        "intents": intents,
        "active_selection": selection,
        "active_request": request,
        "active_results": results,
        "_runtime_topology_adapters": (adapter,),
        "source_binding_slice": q_slice,
        "source_realization": q_slice.source_binding.source_realizations[0],
    }


def _mixed_boundary_binding_inputs() -> tuple[
    CompiledPrecisionIntentGroup,
    CompiledPrecisionSelectionGroup,
    RuntimeSourceDiscoveryRequest,
    tuple[RuntimeSourceDiscoveryResult, ...],
    object,
    tuple[RuntimeSourceBindingSlice, ...],
]:
    from tests.unit.precision_policy import test_runtime_binding

    selection, request, results, adapter = (
        test_runtime_binding._mixed_boundary_fixture()
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            test_runtime_binding.topology_module,
            "_default_adapters",
            lambda: (adapter,),
        )
        intents = bind_runtime_source_intents(selection, request, results)
    return (
        intents,
        selection,
        request,
        results,
        adapter,
        intents.graph_intent("main").source_binding_slices,
    )


def _bind_mixed_refit_contexts(
    *,
    intents: CompiledPrecisionIntentGroup,
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
    results: tuple[RuntimeSourceDiscoveryResult, ...],
    adapter: object,
    bindings: tuple[tuple[RuntimeSourceBindingSlice, SourceStorageRealization], ...],
    installation_registry: object | None = None,
) -> tuple[object, ...]:
    from tests.unit.precision_policy import test_runtime_binding

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            test_runtime_binding.topology_module,
            "_default_adapters",
            lambda: (adapter,),
        )
        return _REFIT_PLAN.bind_refit_contexts(
            intents=intents,
            active_selection=selection,
            active_request=request,
            active_results=results,
            installation_registry=(
                RefitPlanInstallationRegistry()
                if installation_registry is None
                else installation_registry
            ),
            bindings=bindings,
        )


def _proof(
    realized_format: RealizedBindingFormat,
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
) -> DirectCopyCapabilityProof:
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        source_stage,
        destination_stage,
        transform_locus=TransformLocus.NONE,
        implementation_id="nemo.direct-copy",
        implementation_version="1.0.0",
    )
    registry = _adapter_registry(
        adapter_id="test.refit-adapter",
        adapter_version="1.0.0",
        capabilities=(capability,),
    )
    destination_evidence = _destination_evidence(registry, realized_format)
    proof = registry.issue_direct_copy(
        realized_format,
        source_stage,
        destination_stage,
        implementation_id=capability.implementation_id,
        implementation_version=capability.implementation_version,
    )
    _TEST_REGISTRIES[proof.issuer_instance_id] = registry
    _TEST_DESTINATION_EVIDENCE[proof.issuer_instance_id] = destination_evidence
    return proof


def _transform_proof(
    realized_format: RealizedBindingFormat,
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
    transform_locus: TransformLocus,
    *,
    binding_context: object | None = None,
) -> TransformCapabilityProof:
    source_region_extraction = (
        None
        if binding_context is None
        else _REFIT_PLAN.derive_source_region_extraction_capability(
            binding_context,
            realized_format,
        )
    )
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        source_stage,
        destination_stage,
        transform_locus=transform_locus,
        implementation_id="nemo.physical-transform",
        implementation_version="1.0.0",
        source_region_extraction=source_region_extraction,
    )
    registry = _adapter_registry(
        adapter_id="test.refit-adapter",
        adapter_version="1.0.0",
        capabilities=(capability,),
    )
    destination_evidence = _destination_evidence(registry, realized_format)
    proof = registry.issue_transform(
        realized_format,
        source_stage,
        destination_stage,
        transform_locus=capability.transform_locus,
        implementation_id=capability.implementation_id,
        implementation_version=capability.implementation_version,
        source_region_extraction=source_region_extraction,
    )
    _TEST_REGISTRIES[proof.issuer_instance_id] = registry
    _TEST_DESTINATION_EVIDENCE[proof.issuer_instance_id] = destination_evidence
    return proof


def _registry_for(
    proof: DirectCopyCapabilityProof | TransformCapabilityProof,
) -> AdapterCapabilityRegistry:
    return _TEST_REGISTRIES[proof.issuer_instance_id]


def _destination_binding_proof(
    registry: AdapterCapabilityRegistry,
    realized_format: RealizedBindingFormat,
) -> DestinationBindingProof:
    authority = _TEST_DESTINATION_AUTHORITIES[registry.issuer_instance_id]
    return authority.issue_destination_binding(
        realized_format,
        evidence=_TEST_DESTINATION_EVIDENCE[registry.issuer_instance_id],
    )


def select_transform(
    realized_format: RealizedBindingFormat,
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
    **kwargs: object,
) -> SelectedRefitOperation:
    registry = kwargs.get("capability_registry")
    if isinstance(registry, AdapterCapabilityRegistry):
        kwargs.setdefault(
            "destination_binding_proof",
            _destination_binding_proof(registry, realized_format),
        )
    return _select_transform(
        realized_format,
        source_stage,
        destination_stage,
        **kwargs,
    )


def _install_selected_operation(
    selected: SelectedRefitOperation,
    registry: AdapterCapabilityRegistry,
    *,
    inputs: dict[str, object] | None = None,
    installation_registry: object | None = None,
) -> object:
    return _install_selected_operations(
        (selected,),
        registry,
        inputs=inputs,
        installation_registry=installation_registry,
    )[0]


def _install_selected_operations(
    selected: tuple[SelectedRefitOperation, ...],
    registry: AdapterCapabilityRegistry,
    *,
    inputs: dict[str, object] | None = None,
    installation_registry: object | None = None,
) -> tuple[object, ...]:
    from tests.unit.precision_policy import test_runtime_binding

    active_inputs = _binding_context_inputs() if inputs is None else dict(inputs)
    adapters = active_inputs.pop("_runtime_topology_adapters")
    active_inputs.pop("source_binding_slice")
    active_inputs.pop("source_realization")
    assert isinstance(adapters, tuple)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            test_runtime_binding.topology_module,
            "_default_adapters",
            lambda: adapters,
        )
        return _REFIT_PLAN.install_selected_refit_operations(
            selected_operations=selected,
            capability_registry=registry,
            installation_registry=(
                RefitPlanInstallationRegistry()
                if installation_registry is None
                else installation_registry
            ),
            **active_inputs,
        )


def _trtllm_runtime_layout() -> PhysicalLayoutDescriptor:
    return PhysicalLayoutDescriptor(
        axis_order=(
            "experts",
            "input_feature_blocks",
            "intermediate_features_padded",
            "input_feature_block",
        ),
        logical_to_physical_axes=(
            PhysicalAxisMapping(
                logical_axis="experts",
                physical_axes=("experts",),
                mapping_id="identity.axis-map.v1",
            ),
            PhysicalAxisMapping(
                logical_axis="intermediate_features",
                physical_axes=("intermediate_features_padded",),
                mapping_id="pad-to-1024.axis-map.v1",
            ),
            PhysicalAxisMapping(
                logical_axis="input_features",
                physical_axes=(
                    "input_feature_blocks",
                    "input_feature_block",
                ),
                mapping_id="split-block64.axis-map.v1",
            ),
        ),
        padding=(
            PhysicalPadding(
                logical_axis="intermediate_features",
                pad_before=0,
                pad_after=96,
                semantics=PhysicalPaddingSemantics.ZERO_FILLED,
                fill_encoding="bfloat16.zero.v1",
            ),
        ),
        permutation=PhysicalPermutation(
            permutation_id="trtllm.expert-blocked.permutation.v1",
            input_axis_order=(
                "experts",
                "intermediate_features_padded",
                "input_feature_blocks",
                "input_feature_block",
            ),
            output_axis_order=(
                "experts",
                "input_feature_blocks",
                "intermediate_features_padded",
                "input_feature_block",
            ),
        ),
        storage_encoding="trtllm.expert-blocked-bf16.v1",
    )


def test_equal_bf16_dtype_does_not_authorize_logical_to_trtllm_runtime_copy() -> None:
    logical = _component(LOGICAL_VALUES)
    load_api = replace(logical, placement=replace(logical.placement, rank=1))
    runtime = _component(
        LOGICAL_VALUES,
        shape=(128, 42, 1024, 64),
        layout=_trtllm_runtime_layout(),
        rank=1,
    )
    realized_format = _format(
        source_storage=(logical,),
        destination_load_api=(load_api,),
        destination_runtime=(runtime,),
    )

    assert load_api.representation.physical_dtype == "bfloat16"
    assert runtime.representation.physical_dtype == "bfloat16"
    supporting_proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    with pytest.raises(ValueError, match="physical descriptor"):
        require_direct_copy(
            realized_format,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            PhysicalFormatStage.DESTINATION_RUNTIME,
            capability_registry=_registry_for(supporting_proof),
            proof=None,
        )
    transform_proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.DESTINATION_LOAD_API,
        PhysicalFormatStage.DESTINATION_RUNTIME,
        TransformLocus.DESTINATION_NATIVE_LOADER,
    )
    assert (
        select_transform(
            realized_format,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            PhysicalFormatStage.DESTINATION_RUNTIME,
            binding_context=_binding_context(),
            capability_registry=_registry_for(transform_proof),
            transform_locus=TransformLocus.DESTINATION_NATIVE_LOADER,
            transform_capability_proof=transform_proof,
        ).locus
        is TransformLocus.DESTINATION_NATIVE_LOADER
    )


def test_direct_copy_rejects_non_adjacent_stage_skip() -> None:
    component = _component(LOGICAL_VALUES)
    realized_format = _format(source_storage=(component,))

    with pytest.raises(ValueError, match="adjacent physical stages"):
        proof = _proof(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
        )
        require_direct_copy(
            realized_format,
            PhysicalFormatStage.WIRE,
            PhysicalFormatStage.DESTINATION_RUNTIME,
            capability_registry=_registry_for(proof),
            proof=None,
        )


def test_source_to_wire_copy_allows_different_ranks_with_exact_route_proof() -> None:
    source = _component(LOGICAL_VALUES, rank=3)
    wire = replace(source, placement=replace(source.placement, rank=11))
    realized_format = _format(source_storage=(source,), wire=(wire,))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    assert source.placement != wire.placement
    assert (
        require_direct_copy(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            capability_registry=_registry_for(proof),
            proof=proof,
        )
        is proof
    )
    selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=_binding_context(),
        capability_registry=_registry_for(proof),
        direct_copy_proof=proof,
        transform_locus=TransformLocus.NONE,
    )
    assert type(selected) is SelectedRefitOperation
    assert selected.locus is TransformLocus.NONE
    assert selected.capability_proof is proof
    assert selected.executor_key.implementation_id == "nemo.direct-copy"
    installed = _install_selected_operation(selected, _registry_for(proof))
    assert execution_dispatch_key(installed) == selected.executor_key


def test_direct_copy_does_not_erase_atomic_format_schema_identity() -> None:
    component = _component(LOGICAL_VALUES)
    alternate_format = FormatDescriptor(
        format_id="adapter.alternate-bf16.v2",
        family="adapter.alternate-bf16",
        components=(ComponentDescriptor(LOGICAL_VALUES, "bfloat16"),),
    )
    realized_format = _format(
        source_storage=(component,),
        wire=(component,),
        source_storage_format=BF16_FORMAT,
        wire_format=alternate_format,
    )

    assert physical_representation_digest(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
    ) != physical_representation_digest(
        realized_format,
        PhysicalFormatStage.WIRE,
    )
    with pytest.raises(ValueError, match="physical descriptor"):
        require_direct_copy(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            capability_registry=object(),  # type: ignore[arg-type]
            proof=None,
        )


def test_every_adjacent_stage_can_be_selected_without_skipping_runtime_loader() -> None:
    source = _component(LOGICAL_VALUES)
    wire = _component(
        LOGICAL_VALUES,
        dtype="e4m3",
        shape=(8, 8),
        layout=_identity_layout(
            "output_features",
            "input_features",
            storage_encoding="mxfp8.e4m3-values.v1",
        ),
    )
    load_api = replace(wire, placement=replace(wire.placement, rank=1))
    runtime = replace(
        load_api,
        representation=replace(
            load_api.representation,
            layout=replace(
                load_api.representation.layout,
                storage_encoding="mxfp8.runtime-packed.v1",
            ),
        ),
    )
    realized_format = _format(
        source_storage=(source,),
        wire=(wire,),
        destination_load_api=(load_api,),
        destination_runtime=(runtime,),
    )

    source_transform_proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        TransformLocus.SOURCE,
    )
    assert (
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=_binding_context(),
            capability_registry=_registry_for(source_transform_proof),
            transform_locus=TransformLocus.SOURCE,
            transform_capability_proof=source_transform_proof,
        ).locus
        is TransformLocus.SOURCE
    )
    route_proof = _proof(
        realized_format,
        PhysicalFormatStage.WIRE,
        PhysicalFormatStage.DESTINATION_LOAD_API,
    )
    assert (
        select_transform(
            realized_format,
            PhysicalFormatStage.WIRE,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            binding_context=_binding_context(),
            capability_registry=_registry_for(route_proof),
            direct_copy_proof=route_proof,
            transform_locus=TransformLocus.NONE,
        ).locus
        is TransformLocus.NONE
    )
    runtime_transform_proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.DESTINATION_LOAD_API,
        PhysicalFormatStage.DESTINATION_RUNTIME,
        TransformLocus.DESTINATION_NATIVE_LOADER,
    )
    assert (
        select_transform(
            realized_format,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            PhysicalFormatStage.DESTINATION_RUNTIME,
            binding_context=_binding_context(),
            capability_registry=_registry_for(runtime_transform_proof),
            transform_locus=TransformLocus.DESTINATION_NATIVE_LOADER,
            transform_capability_proof=runtime_transform_proof,
        ).locus
        is TransformLocus.DESTINATION_NATIVE_LOADER
    )


def test_native_mxfp8_direct_copy_preserves_values_then_block_scales_order() -> None:
    values = _component(
        VALUES,
        dtype="uint8",
        shape=(128, 928, 2688),
        layout=_identity_layout(
            "experts",
            "intermediate_features",
            "input_features",
            storage_encoding="mxfp8.e4m3-carrier.v1",
        ),
    )
    scales = _component(
        BLOCK_SCALES,
        dtype="uint8",
        shape=(128, 928, 84),
        layout=_identity_layout(
            "experts",
            "intermediate_features",
            "input_feature_blocks",
            storage_encoding="mxfp8.e8m0-carrier.v1",
        ),
    )
    wire = (values, scales)
    destination = tuple(
        replace(component, placement=replace(component.placement, rank=1))
        for component in wire
    )
    realized_format = _format(
        source_storage=wire,
        wire=wire,
        destination_load_api=destination,
        destination_runtime=destination,
        source_storage_format=MXFP8_FORMAT,
        wire_format=MXFP8_FORMAT,
        destination_load_api_format=MXFP8_FORMAT,
        destination_runtime_format=MXFP8_FORMAT,
    )
    proof = _proof(
        realized_format,
        PhysicalFormatStage.WIRE,
        PhysicalFormatStage.DESTINATION_LOAD_API,
    )

    assert tuple(
        component.representation.role
        for component in realized_format.destination_load_api
    ) == (VALUES, BLOCK_SCALES)
    require_direct_copy(
        realized_format,
        PhysicalFormatStage.WIRE,
        PhysicalFormatStage.DESTINATION_LOAD_API,
        capability_registry=_registry_for(proof),
        proof=proof,
    )


def test_component_roles_are_open_for_future_quantization_metadata() -> None:
    for future_role in (
        ComponentRole("qkvo_packed_shape"),
        ComponentRole("2bit_scales"),
    ):
        future_component = _component(
            future_role,
            dtype="int32",
            shape=(4,),
            layout=_identity_layout("components", storage_encoding="shape-vector.v1"),
        )
        future_format = FormatDescriptor(
            format_id=f"adapter.future-{future_role}.v1",
            family="adapter.future-metadata",
            components=(ComponentDescriptor(future_role, "int32"),),
        )
        realized_format = _format(
            source_storage=(future_component,),
            source_storage_format=future_format,
        )

        assert realized_format.source_storage[0].representation.role == future_role


def test_many_logical_axes_can_share_one_flattened_physical_axis() -> None:
    flattened_layout = PhysicalLayoutDescriptor(
        axis_order=("experts_and_intermediate", "input_features"),
        logical_to_physical_axes=(
            PhysicalAxisMapping(
                logical_axis="experts",
                physical_axes=("experts_and_intermediate",),
                mapping_id="flatten-expert-intermediate.axis-map.v1",
            ),
            PhysicalAxisMapping(
                logical_axis="intermediate_features",
                physical_axes=("experts_and_intermediate",),
                mapping_id="flatten-expert-intermediate.axis-map.v1",
            ),
            PhysicalAxisMapping(
                logical_axis="input_features",
                physical_axes=("input_features",),
                mapping_id="identity.axis-map.v1",
            ),
        ),
        padding=(),
        permutation=None,
        storage_encoding="flattened-bfloat16.v1",
    )

    representation = PhysicalRepresentation(
        role=LOGICAL_VALUES,
        physical_dtype="bfloat16",
        physical_shape=(128 * 928, 2688),
        layout=flattened_layout,
    )

    assert representation.layout.logical_to_physical_axes[0].physical_axes == (
        "experts_and_intermediate",
    )


def test_bf16_wire_to_mxfp8_load_uses_a_destination_transform() -> None:
    bf16 = _component(LOGICAL_VALUES)
    mxfp8_values = _component(
        VALUES,
        dtype="uint8",
        layout=replace(
            bf16.representation.layout,
            storage_encoding="mxfp8.e4m3-carrier.v1",
        ),
        rank=1,
    )
    mxfp8_scales = _component(
        BLOCK_SCALES,
        dtype="uint8",
        shape=(128, 928, 84),
        layout=_identity_layout(
            "experts",
            "intermediate_features",
            "input_feature_blocks",
            storage_encoding="mxfp8.e8m0-carrier.v1",
        ),
        rank=1,
    )
    realized_format = _format(
        source_storage=(bf16,),
        wire=(bf16,),
        destination_load_api=(mxfp8_values, mxfp8_scales),
        destination_runtime=(mxfp8_values, mxfp8_scales),
        destination_load_api_format=MXFP8_FORMAT,
        destination_runtime_format=MXFP8_FORMAT,
    )

    transform_proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.WIRE,
        PhysicalFormatStage.DESTINATION_LOAD_API,
        TransformLocus.DESTINATION,
    )
    assert (
        select_transform(
            realized_format,
            PhysicalFormatStage.WIRE,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            binding_context=_binding_context(destination_precision="mxfp8"),
            capability_registry=_registry_for(transform_proof),
            transform_locus=TransformLocus.DESTINATION,
            transform_capability_proof=transform_proof,
        ).locus
        is TransformLocus.DESTINATION
    )


@pytest.mark.parametrize("mutation", ["missing_scales", "role", "shape"])
def test_transform_proof_fails_closed_when_realized_components_change(
    mutation: str,
) -> None:
    bf16 = _component(LOGICAL_VALUES)
    values = _component(
        VALUES,
        dtype="uint8",
        layout=replace(
            bf16.representation.layout,
            storage_encoding="mxfp8.e4m3-carrier.v1",
        ),
        rank=1,
    )
    scales = _component(
        BLOCK_SCALES,
        dtype="uint8",
        shape=(128, 928, 84),
        layout=_identity_layout(
            "experts",
            "intermediate_features",
            "input_feature_blocks",
            storage_encoding="mxfp8.e8m0-carrier.v1",
        ),
        rank=1,
    )
    realized_format = _format(
        source_storage=(bf16,),
        wire=(bf16,),
        destination_load_api=(values, scales),
        destination_runtime=(values, scales),
        destination_load_api_format=MXFP8_FORMAT,
        destination_runtime_format=MXFP8_FORMAT,
    )
    proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.WIRE,
        PhysicalFormatStage.DESTINATION_LOAD_API,
        TransformLocus.DESTINATION,
    )
    if mutation == "missing_scales":
        changed_components = (values,)
    elif mutation == "role":
        changed_components = (
            replace(
                values,
                representation=replace(
                    values.representation,
                    role=ComponentRole("future_quantized_values"),
                ),
            ),
            scales,
        )
    else:
        changed_components = (
            replace(
                values,
                representation=replace(
                    values.representation,
                    physical_shape=(7, 8),
                ),
            ),
            scales,
        )
    if mutation in {"missing_scales", "role"}:
        with pytest.raises(ValueError, match="complete ordered component roles"):
            replace(
                realized_format,
                destination_load_api=changed_components,
            )
        return
    changed_format = replace(realized_format, destination_load_api=changed_components)

    with pytest.raises(ValueError, match="ordered representation"):
        select_transform(
            changed_format,
            PhysicalFormatStage.WIRE,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            binding_context=_binding_context(destination_precision="mxfp8"),
            capability_registry=_registry_for(proof),
            transform_locus=TransformLocus.DESTINATION,
            transform_capability_proof=proof,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "role",
        "order",
        "dtype",
        "shape",
        "axis_order",
        "axis_mapping",
        "padding",
        "permutation",
        "storage_encoding",
    ],
)
def test_direct_copy_checks_every_physical_representation_dimension(
    mutation: str,
) -> None:
    source_format = FormatDescriptor(
        format_id="test.physical-pair.v1",
        family="test.physical-pair",
        components=(
            ComponentDescriptor(VALUES, "bfloat16"),
            ComponentDescriptor(ComponentRole("metadata"), "bfloat16"),
        ),
    )
    first = _component(VALUES)
    second = _component(ComponentRole("metadata"))
    source = (first, second)
    destination = source
    if mutation == "role":
        destination = (
            replace(
                first,
                representation=replace(
                    first.representation, role=ComponentRole("other_values")
                ),
            ),
            second,
        )
    elif mutation == "order":
        destination = tuple(reversed(source))
    elif mutation == "dtype":
        destination = (
            replace(
                first,
                representation=replace(first.representation, physical_dtype="float16"),
            ),
            second,
        )
    elif mutation == "shape":
        destination = (
            replace(
                first,
                representation=replace(first.representation, physical_shape=(7, 8)),
            ),
            second,
        )
    else:
        layout = first.representation.layout
        if mutation == "axis_order":
            changed_layout = PhysicalLayoutDescriptor(
                axis_order=tuple(reversed(layout.axis_order)),
                logical_to_physical_axes=layout.logical_to_physical_axes,
                padding=layout.padding,
                permutation=PhysicalPermutation(
                    permutation_id="reverse.permutation.v1",
                    input_axis_order=layout.axis_order,
                    output_axis_order=tuple(reversed(layout.axis_order)),
                ),
                storage_encoding=layout.storage_encoding,
            )
        elif mutation == "axis_mapping":
            changed_mapping = replace(
                layout.logical_to_physical_axes[0],
                mapping_id="alternate.axis-map.v1",
            )
            changed_layout = replace(
                layout,
                logical_to_physical_axes=(
                    changed_mapping,
                    *layout.logical_to_physical_axes[1:],
                ),
            )
        elif mutation == "padding":
            changed_layout = replace(
                layout,
                padding=(
                    PhysicalPadding(
                        logical_axis="output_features",
                        pad_before=0,
                        pad_after=1,
                        semantics=PhysicalPaddingSemantics.ZERO_FILLED,
                        fill_encoding="bfloat16.zero.v1",
                    ),
                ),
            )
        elif mutation == "permutation":
            changed_layout = replace(
                layout,
                permutation=PhysicalPermutation(
                    permutation_id="identity-but-attested.permutation.v1",
                    input_axis_order=layout.axis_order,
                    output_axis_order=layout.axis_order,
                ),
            )
        else:
            changed_layout = replace(layout, storage_encoding="different-storage.v1")
        destination = (
            replace(
                first,
                representation=replace(first.representation, layout=changed_layout),
            ),
            second,
        )
    if mutation == "role":
        destination_format = FormatDescriptor(
            format_id="test.physical-pair-role-changed.v1",
            family="test.physical-pair",
            components=(
                ComponentDescriptor(ComponentRole("other_values"), "bfloat16"),
                ComponentDescriptor(ComponentRole("metadata"), "bfloat16"),
            ),
        )
    elif mutation == "order":
        destination_format = FormatDescriptor(
            format_id="test.physical-pair-order-changed.v1",
            family="test.physical-pair",
            components=tuple(reversed(source_format.components)),
        )
    else:
        destination_format = source_format
    realized_format = _format(
        source_storage=source,
        wire=destination,
        source_storage_format=source_format,
        wire_format=destination_format,
    )
    authorized_format = _format(
        source_storage=source,
        wire=source,
        source_storage_format=source_format,
        wire_format=source_format,
    )
    proof = _proof(
        authorized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    with pytest.raises(ValueError, match="physical descriptor"):
        require_direct_copy(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            capability_registry=_registry_for(proof),
            proof=proof,
        )


@pytest.mark.parametrize(
    "proof_mutation",
    [
        "source_representation",
        "destination_representation",
        "source_placement",
        "destination_placement",
        "source_instance",
        "destination_instance",
        "source_capability",
        "destination_capability",
        "route",
    ],
)
@pytest.mark.parametrize("proof_kind", ["direct", "transform"])
def test_signed_adapter_proofs_reject_field_tampering(
    proof_mutation: str,
    proof_kind: str,
) -> None:
    source = _component(LOGICAL_VALUES, rank=0)
    destination = replace(source, placement=replace(source.placement, rank=1))
    realized_format = _format(source_storage=(source,), wire=(destination,))
    if proof_kind == "direct":
        proof = _proof(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
        )
    else:
        proof = _transform_proof(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            TransformLocus.SOURCE,
        )
    registry = _registry_for(proof)
    tampered_proof = copy(proof)
    if proof_mutation == "route":
        object.__setattr__(tampered_proof, "route_id", "different-route")
    elif proof_mutation.startswith("source_"):
        field_name = {
            "source_representation": "ordered_representation_digest",
            "source_placement": "placement_digest",
            "source_instance": "endpoint_instance_id",
            "source_capability": "capability_fingerprint",
        }[proof_mutation]
        bad_value = (
            "different-source-instance"
            if field_name == "endpoint_instance_id"
            else _OTHER_CAPABILITY_FINGERPRINT
        )
        tampered_endpoint = copy(proof.source_endpoint_capability)
        object.__setattr__(tampered_endpoint, field_name, bad_value)
        object.__setattr__(
            tampered_proof,
            "source_endpoint_capability",
            tampered_endpoint,
        )
    else:
        field_name = {
            "destination_representation": "ordered_representation_digest",
            "destination_placement": "placement_digest",
            "destination_instance": "endpoint_instance_id",
            "destination_capability": "capability_fingerprint",
        }[proof_mutation]
        bad_value = (
            "different-destination-instance"
            if field_name == "endpoint_instance_id"
            else _OTHER_CAPABILITY_FINGERPRINT
        )
        tampered_endpoint = copy(proof.destination_endpoint_capability)
        object.__setattr__(tampered_endpoint, field_name, bad_value)
        object.__setattr__(
            tampered_proof,
            "destination_endpoint_capability",
            tampered_endpoint,
        )

    with pytest.raises(ValueError, match="signature mismatch"):
        if proof_kind == "direct":
            require_direct_copy(
                realized_format,
                PhysicalFormatStage.SOURCE_STORAGE,
                PhysicalFormatStage.WIRE,
                capability_registry=registry,
                proof=tampered_proof,
            )
        else:
            select_transform(
                realized_format,
                PhysicalFormatStage.SOURCE_STORAGE,
                PhysicalFormatStage.WIRE,
                binding_context=_binding_context(),
                capability_registry=registry,
                transform_locus=TransformLocus.SOURCE,
                transform_capability_proof=tampered_proof,
            )


def test_stage_pair_and_capability_proof_are_both_required_for_direct_copy() -> None:
    component = _component(LOGICAL_VALUES)
    realized_format = _format(source_storage=(component,))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    registry = _registry_for(proof)

    with pytest.raises(ValueError, match="capability proof"):
        require_direct_copy(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            capability_registry=registry,
            proof=None,
        )
    with pytest.raises(ValueError, match="exact stage pair"):
        require_direct_copy(
            realized_format,
            PhysicalFormatStage.WIRE,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            capability_registry=registry,
            proof=proof,
        )


@pytest.mark.parametrize(
    "source_stage, destination_stage, locus",
    [
        (
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            TransformLocus.DESTINATION,
        ),
        (
            PhysicalFormatStage.WIRE,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            TransformLocus.SOURCE,
        ),
        (
            PhysicalFormatStage.DESTINATION_LOAD_API,
            PhysicalFormatStage.DESTINATION_RUNTIME,
            TransformLocus.SOURCE,
        ),
    ],
)
def test_transform_locus_must_belong_to_the_exact_adjacent_stage(
    source_stage: PhysicalFormatStage,
    destination_stage: PhysicalFormatStage,
    locus: TransformLocus,
) -> None:
    source = _component(LOGICAL_VALUES)
    changed = replace(
        source,
        representation=replace(source.representation, physical_dtype="float16"),
    )
    stages = {
        PhysicalFormatStage.SOURCE_STORAGE: (source,),
        PhysicalFormatStage.WIRE: (source,),
        PhysicalFormatStage.DESTINATION_LOAD_API: (source,),
        PhysicalFormatStage.DESTINATION_RUNTIME: (source,),
    }
    stages[destination_stage] = (changed,)
    realized_format = _format(
        source_storage=stages[PhysicalFormatStage.SOURCE_STORAGE],
        wire=stages[PhysicalFormatStage.WIRE],
        destination_load_api=stages[PhysicalFormatStage.DESTINATION_LOAD_API],
        destination_runtime=stages[PhysicalFormatStage.DESTINATION_RUNTIME],
        routes=_routes(),
    )
    support_format = _format(source_storage=(source,))
    support_proof = _proof(
        support_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    with pytest.raises(ValueError, match="transform locus"):
        select_transform(
            realized_format,
            source_stage,
            destination_stage,
            binding_context=_binding_context(),
            capability_registry=_registry_for(support_proof),
            transform_locus=locus,
        )


@pytest.mark.parametrize(
    "bad_shape, expected_exception",
    [
        ((0,), ValueError),
        ((-1,), ValueError),
        ((True,), TypeError),
        ((1.0,), TypeError),
    ],
)
def test_physical_shape_extents_are_exact_positive_integers(
    bad_shape: tuple[object, ...],
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception, match="physical shape"):
        PhysicalRepresentation(
            role=LOGICAL_VALUES,
            physical_dtype="bfloat16",
            physical_shape=bad_shape,  # type: ignore[arg-type]
            layout=_identity_layout("x", storage_encoding="plain.v1"),
        )


def test_rank_zero_scalar_component_is_a_valid_physical_representation() -> None:
    scalar = PhysicalRepresentation(
        role=ComponentRole("global_scale"),
        physical_dtype="float32",
        physical_shape=(),
        layout=PhysicalLayoutDescriptor(
            axis_order=(),
            logical_to_physical_axes=(),
            padding=(),
            permutation=None,
            storage_encoding="scalar.float32.v1",
        ),
    )

    assert scalar.physical_shape == ()


@pytest.mark.parametrize(
    "bad_rank, expected_exception",
    [(-1, ValueError), (True, TypeError), (1.0, TypeError)],
)
def test_endpoint_rank_is_an_exact_nonnegative_integer(
    bad_rank: object,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception, match="rank"):
        EndpointPlacement(
            rank=bad_rank,  # type: ignore[arg-type]
            device_type="cuda",
            memory_space="device",
        )


@pytest.mark.parametrize(
    "pad_before, pad_after, expected_exception",
    [
        (-1, 1, ValueError),
        (1, -1, ValueError),
        (True, 1, TypeError),
        (1, True, TypeError),
        (1.0, 1, TypeError),
        (1, 1.0, TypeError),
        (0, 0, ValueError),
    ],
)
def test_padding_is_exact_nonnegative_and_describes_real_capacity(
    pad_before: object,
    pad_after: object,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception, match="padding"):
        PhysicalPadding(
            logical_axis="input_features",
            pad_before=pad_before,  # type: ignore[arg-type]
            pad_after=pad_after,  # type: ignore[arg-type]
            semantics=PhysicalPaddingSemantics.ZERO_FILLED,
            fill_encoding="zero.v1",
        )


@pytest.mark.parametrize(
    "semantics, fill_encoding",
    [
        (PhysicalPaddingSemantics.ZERO_FILLED, None),
        (PhysicalPaddingSemantics.ZERO_FILLED, ""),
        (PhysicalPaddingSemantics.UNSPECIFIED_IGNORED, "zero.v1"),
    ],
)
def test_padding_fill_encoding_is_conditioned_on_semantics(
    semantics: PhysicalPaddingSemantics, fill_encoding: str | None
) -> None:
    with pytest.raises(ValueError, match="fill_encoding"):
        PhysicalPadding(
            logical_axis="input_features",
            pad_before=0,
            pad_after=1,
            semantics=semantics,
            fill_encoding=fill_encoding,
        )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: PhysicalAxisMapping(
            logical_axis="x",
            physical_axes=("x",),
            mapping_id="identity.axis-map",
        ),
        lambda: PhysicalPermutation(
            permutation_id="identity.permutation",
            input_axis_order=("x",),
            output_axis_order=("x",),
        ),
        lambda: PhysicalLayoutDescriptor(
            axis_order=("x",),
            logical_to_physical_axes=(
                PhysicalAxisMapping(
                    logical_axis="x",
                    physical_axes=("x",),
                    mapping_id="identity.axis-map.v1",
                ),
            ),
            padding=(),
            permutation=None,
            storage_encoding="plain-bfloat16",
        ),
        lambda: PhysicalPadding(
            logical_axis="x",
            pad_before=0,
            pad_after=1,
            semantics=PhysicalPaddingSemantics.ZERO_FILLED,
            fill_encoding="bfloat16.zero",
        ),
    ],
)
def test_layout_semantic_identifiers_require_explicit_versions(
    constructor: object,
) -> None:
    with pytest.raises(ValueError, match="versioned semantic identifier"):
        constructor()  # type: ignore[operator]


@pytest.mark.parametrize(
    "constructor, expected",
    [
        (
            lambda: PhysicalAxisMapping(
                logical_axis="x",
                physical_axes=(),
                mapping_id="identity.axis-map.v1",
            ),
            "physical_axes",
        ),
        (
            lambda: PhysicalAxisMapping(
                logical_axis="x",
                physical_axes=("x", "x"),
                mapping_id="identity.axis-map.v1",
            ),
            "physical_axes",
        ),
        (
            lambda: PhysicalPermutation(
                permutation_id="bad.permutation.v1",
                input_axis_order=("x", "y"),
                output_axis_order=("x", "z"),
            ),
            "same axes",
        ),
        (
            lambda: PhysicalLayoutDescriptor(
                axis_order=("x", "x"),
                logical_to_physical_axes=(
                    PhysicalAxisMapping(
                        logical_axis="x",
                        physical_axes=("x",),
                        mapping_id="identity.axis-map.v1",
                    ),
                ),
                padding=(),
                permutation=None,
                storage_encoding="plain.v1",
            ),
            "axis_order",
        ),
        (
            lambda: PhysicalLayoutDescriptor(
                axis_order=("x", "y"),
                logical_to_physical_axes=(
                    PhysicalAxisMapping(
                        logical_axis="x",
                        physical_axes=("x",),
                        mapping_id="identity.axis-map.v1",
                    ),
                ),
                padding=(),
                permutation=None,
                storage_encoding="plain.v1",
            ),
            "cover every physical axis",
        ),
        (
            lambda: PhysicalLayoutDescriptor(
                axis_order=("x",),
                logical_to_physical_axes=(
                    PhysicalAxisMapping(
                        logical_axis="x",
                        physical_axes=("x",),
                        mapping_id="identity.axis-map.v1",
                    ),
                    PhysicalAxisMapping(
                        logical_axis="x",
                        physical_axes=("x",),
                        mapping_id="alternate.axis-map.v1",
                    ),
                ),
                padding=(),
                permutation=None,
                storage_encoding="plain.v1",
            ),
            "logical axis mappings",
        ),
    ],
)
def test_axis_and_layout_records_reject_ambiguous_physical_descriptions(
    constructor: object, expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        constructor()  # type: ignore[operator]


def test_physical_shape_rank_must_match_layout_rank() -> None:
    with pytest.raises(ValueError, match="rank must match"):
        PhysicalRepresentation(
            role=LOGICAL_VALUES,
            physical_dtype="bfloat16",
            physical_shape=(1, 2),
            layout=_identity_layout("x", storage_encoding="plain.v1"),
        )


@pytest.mark.parametrize(
    "stage_name",
    [
        "source_storage",
        "wire",
        "destination_load_api",
        "destination_runtime",
    ],
)
def test_every_realized_stage_requires_nonempty_unique_typed_components(
    stage_name: str,
) -> None:
    component = _component(LOGICAL_VALUES)
    kwargs: dict[str, object] = {
        "source_storage": (component,),
        "wire": (component,),
        "destination_load_api": (component,),
        "destination_runtime": (component,),
        "source_storage_format": BF16_FORMAT,
        "wire_format": BF16_FORMAT,
        "destination_load_api_format": BF16_FORMAT,
        "destination_runtime_format": BF16_FORMAT,
        "format_schema_registry": _REFIT_PLAN._create_format_schema_registry(
            (BF16_FORMAT,)
        ),
        "routes": _routes(),
    }
    kwargs[stage_name] = ()
    with pytest.raises(ValueError, match=f"{stage_name}.*non-empty"):
        RealizedBindingFormat(**kwargs)  # type: ignore[arg-type]

    kwargs[stage_name] = (component, component)
    with pytest.raises(ValueError, match=f"{stage_name}.*duplicate component role"):
        RealizedBindingFormat(**kwargs)  # type: ignore[arg-type]

    kwargs[stage_name] = ("not-a-component",)
    with pytest.raises(TypeError, match=f"{stage_name}.*PhysicalComponentDescriptor"):
        RealizedBindingFormat(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "routes, expected",
    [
        ((), "cover every adjacent"),
        ((_routes()[0], _routes()[0], _routes()[2]), "duplicate"),
        (
            (
                _routes()[0],
                replace(_routes()[1], route_id=_routes()[0].route_id),
                _routes()[2],
            ),
            "route_id.*unique",
        ),
        (
            (
                _routes()[0],
                replace(
                    _routes()[1],
                    source_endpoint_instance_id="different-wire-endpoint",
                ),
                _routes()[2],
            ),
            "shared endpoint identity",
        ),
    ],
)
def test_realized_format_requires_exact_continuous_adjacent_routes(
    routes: tuple[PhysicalRouteDescriptor, ...],
    expected: str,
) -> None:
    component = _component(LOGICAL_VALUES)

    with pytest.raises(ValueError, match=expected):
        _format(source_storage=(component,), routes=routes)


def test_records_reject_untyped_enum_strings() -> None:
    with pytest.raises(TypeError):
        PhysicalPadding(
            logical_axis="x",
            pad_before=0,
            pad_after=1,
            semantics="zero_filled",  # type: ignore[arg-type]
            fill_encoding="zero.v1",
        )

    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.NONE,
        implementation_id=proof.implementation_id,
        implementation_version=proof.implementation_version,
    )
    with pytest.raises(TypeError, match="PhysicalFormatStage"):
        replace(capability, source_stage="source_storage")  # type: ignore[arg-type]


def test_records_reject_forged_enum_members_with_registered_values() -> None:
    forged_stage = str.__new__(PhysicalFormatStage, "source_storage")
    forged_stage._name_ = "SOURCE_STORAGE"
    forged_stage._value_ = "source_storage"
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.NONE,
        implementation_id="nemo.direct-copy",
        implementation_version="1.0.0",
    )

    with pytest.raises(TypeError, match="registered PhysicalFormatStage"):
        replace(capability, source_stage=forged_stage)


def test_records_are_frozen_and_snapshot_mutable_sequences() -> None:
    axes = ["x"]
    mapping = PhysicalAxisMapping(
        logical_axis="x",
        physical_axes=axes,  # type: ignore[arg-type]
        mapping_id="identity.axis-map.v1",
    )
    axes.append("y")

    assert mapping.physical_axes == ("x",)
    with pytest.raises(FrozenInstanceError):
        mapping.mapping_id = "changed.axis-map.v1"  # type: ignore[misc]


def test_malformed_digests_and_fingerprints_fail_at_record_construction() -> None:
    route = _routes()[0]
    with pytest.raises(ValueError, match="canonical SHA-256"):
        replace(
            route,
            source_endpoint_capability_fingerprint="vllm-0.25.1",
        )

    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.NONE,
        implementation_id="nemo.direct-copy",
        implementation_version="1.0.0",
    )
    with pytest.raises(ValueError, match="canonical SHA-256"):
        replace(
            capability,
            source_representation_digest="bad",
        )


@pytest.mark.parametrize("proof_kind", ["direct", "transform"])
@pytest.mark.parametrize(
    "field_name, bad_value, expected",
    [
        ("implementation_id", "quantize$mxfp8", "canonical identifier"),
        ("implementation_version", "latest", "numeric implementation version"),
    ],
)
def test_adapter_capabilities_require_exact_implementation_identity_and_version(
    proof_kind: str,
    field_name: str,
    bad_value: str,
    expected: str,
) -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    if proof_kind == "direct":
        transform_locus = TransformLocus.NONE
    else:
        transform_locus = TransformLocus.SOURCE
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=transform_locus,
        implementation_id="nemo.operation",
        implementation_version="1.0.0",
    )

    with pytest.raises(ValueError, match=expected):
        replace(capability, **{field_name: bad_value})


def test_transform_selection_rejects_direct_proof_plus_a_second_transform() -> None:
    component = _component(LOGICAL_VALUES)
    realized_format = _format(source_storage=(component,))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=_binding_context(),
            capability_registry=_registry_for(proof),
            direct_copy_proof=proof,
            transform_locus=TransformLocus.SOURCE,
        )


def test_transform_selection_rejects_none_without_a_direct_copy_proof() -> None:
    component = _component(LOGICAL_VALUES)
    realized_format = _format(source_storage=(component,))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    with pytest.raises(ValueError, match="capability proof"):
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=_binding_context(),
            capability_registry=_registry_for(proof),
            transform_locus=TransformLocus.NONE,
        )


def test_non_direct_transform_requires_an_adapter_capability_proof() -> None:
    source = _component(LOGICAL_VALUES)
    destination = replace(
        source,
        representation=replace(source.representation, physical_dtype="float16"),
    )
    realized_format = _format(source_storage=(source,), wire=(destination,))
    proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        TransformLocus.SOURCE,
    )

    with pytest.raises(ValueError, match="transform capability proof"):
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=_binding_context(),
            capability_registry=_registry_for(proof),
            transform_locus=TransformLocus.SOURCE,
        )


def test_non_direct_transform_rejects_an_untyped_capability_proof() -> None:
    source = _component(LOGICAL_VALUES)
    destination = replace(
        source,
        representation=replace(source.representation, physical_dtype="float16"),
    )
    realized_format = _format(source_storage=(source,), wire=(destination,))
    proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        TransformLocus.SOURCE,
    )

    with pytest.raises(TypeError, match="TransformCapabilityProof"):
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=_binding_context(),
            capability_registry=_registry_for(proof),
            transform_locus=TransformLocus.SOURCE,
            transform_capability_proof=object(),
        )


def test_adapter_transform_proof_binds_route_endpoints_and_implementation() -> None:
    source = _component(LOGICAL_VALUES, rank=0)
    wire = replace(
        source,
        representation=replace(source.representation, physical_dtype="float16"),
        placement=replace(source.placement, rank=1),
    )
    load_api = replace(wire, placement=replace(wire.placement, rank=2))
    runtime = replace(load_api, placement=replace(load_api.placement, rank=3))
    routes = (
        PhysicalRouteDescriptor(
            source_stage=PhysicalFormatStage.SOURCE_STORAGE,
            destination_stage=PhysicalFormatStage.WIRE,
            route_id="route-source-wire-7",
            source_endpoint_instance_id="training-rank-0",
            destination_endpoint_instance_id="wire-rank-1",
            source_endpoint_capability_fingerprint=f"sha256:{'1' * 64}",
            destination_endpoint_capability_fingerprint=f"sha256:{'2' * 64}",
        ),
        PhysicalRouteDescriptor(
            source_stage=PhysicalFormatStage.WIRE,
            destination_stage=PhysicalFormatStage.DESTINATION_LOAD_API,
            route_id="route-wire-loader-8",
            source_endpoint_instance_id="wire-rank-1",
            destination_endpoint_instance_id="loader-rank-2",
            source_endpoint_capability_fingerprint=f"sha256:{'2' * 64}",
            destination_endpoint_capability_fingerprint=f"sha256:{'3' * 64}",
        ),
        PhysicalRouteDescriptor(
            source_stage=PhysicalFormatStage.DESTINATION_LOAD_API,
            destination_stage=PhysicalFormatStage.DESTINATION_RUNTIME,
            route_id="route-loader-runtime-9",
            source_endpoint_instance_id="loader-rank-2",
            destination_endpoint_instance_id="runtime-rank-3",
            source_endpoint_capability_fingerprint=f"sha256:{'3' * 64}",
            destination_endpoint_capability_fingerprint=f"sha256:{'4' * 64}",
        ),
    )
    source_format = BF16_FORMAT
    wire_format = BF16_FORMAT
    load_api_format = BF16_FORMAT
    runtime_format = BF16_FORMAT
    realized_format = RealizedBindingFormat(
        source_storage=(source,),
        wire=(wire,),
        destination_load_api=(load_api,),
        destination_runtime=(runtime,),
        source_storage_format=source_format,
        wire_format=wire_format,
        destination_load_api_format=load_api_format,
        destination_runtime_format=runtime_format,
        format_schema_registry=_REFIT_PLAN._create_format_schema_registry(
            (source_format, wire_format, load_api_format, runtime_format)
        ),
        routes=routes,
    )
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.SOURCE,
        implementation_id="nemo.quantize-mxfp8",
        implementation_version="1.0.0",
    )
    registry = _adapter_registry(
        adapter_id="vllm.refit-adapter",
        adapter_version="0.25.1",
        capabilities=(capability,),
    )
    destination_evidence = _destination_evidence(registry, realized_format)
    _TEST_DESTINATION_EVIDENCE[registry.issuer_instance_id] = destination_evidence
    proof = registry.issue_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.SOURCE,
        implementation_id=capability.implementation_id,
        implementation_version=capability.implementation_version,
    )

    assert proof.adapter_id == "vllm.refit-adapter"
    assert proof.adapter_version == "0.25.1"
    assert proof.issuer_instance_id == registry.issuer_instance_id
    assert proof.registry_digest == registry.registry_digest
    assert (
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=_binding_context(),
            capability_registry=registry,
            transform_locus=TransformLocus.SOURCE,
            transform_capability_proof=proof,
        ).locus
        is TransformLocus.SOURCE
    )


def test_capability_authority_cannot_be_created_by_public_constructors() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        TransformLocus.SOURCE,
    )

    with pytest.raises(TypeError, match="adapter capability registry"):
        EndpointCapabilityProof()
    with pytest.raises(TypeError, match="adapter capability registry"):
        DirectCopyCapabilityProof()
    with pytest.raises(TypeError, match="adapter capability registry"):
        TransformCapabilityProof(
            source_stage=proof.source_stage,
            destination_stage=proof.destination_stage,
            transform_locus=proof.transform_locus,
            route_id=proof.route_id,
            source_endpoint_capability=proof.source_endpoint_capability,
            destination_endpoint_capability=proof.destination_endpoint_capability,
            implementation_id=proof.implementation_id,
            implementation_version=proof.implementation_version,
        )
    with pytest.raises(TypeError, match="version-adapter trust boundary"):
        AdapterCapabilityRegistry()


def test_adapter_registry_is_immutable_after_trusted_construction() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    registry = _registry_for(proof)

    with pytest.raises(AttributeError, match="immutable"):
        registry._adapter_id = "caller.forged-adapter"  # type: ignore[misc]


def test_identical_capability_registry_instances_do_not_share_authority() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.SOURCE,
        implementation_id="nemo.physical-transform",
        implementation_version="1.0.0",
    )
    first_registry = _adapter_registry(
        adapter_id="vllm.refit-adapter",
        adapter_version="0.25.1",
        capabilities=(capability,),
    )
    second_registry = _adapter_registry(
        adapter_id="vllm.refit-adapter",
        adapter_version="0.25.1",
        capabilities=(capability,),
    )
    first_evidence = _destination_evidence(first_registry, realized_format)
    second_evidence = _destination_evidence(second_registry, realized_format)
    _TEST_DESTINATION_EVIDENCE[first_registry.issuer_instance_id] = first_evidence
    _TEST_DESTINATION_EVIDENCE[second_registry.issuer_instance_id] = second_evidence
    proof = first_registry.issue_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.SOURCE,
        implementation_id=capability.implementation_id,
        implementation_version=capability.implementation_version,
    )
    second_proof = second_registry.issue_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.SOURCE,
        implementation_id=capability.implementation_id,
        implementation_version=capability.implementation_version,
    )

    assert first_registry.registry_digest == second_registry.registry_digest
    assert first_registry.issuer_instance_id != second_registry.issuer_instance_id
    with pytest.raises(ValueError, match="not live"):
        _TEST_DESTINATION_AUTHORITIES[
            second_registry.issuer_instance_id
        ].issue_destination_binding(
            realized_format,
            evidence=first_evidence,
        )
    with pytest.raises(ValueError, match="issuer identity mismatch"):
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=_binding_context(),
            capability_registry=second_registry,
            transform_locus=TransformLocus.SOURCE,
            transform_capability_proof=proof,
        )
    binding_context = _binding_context()
    first_selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=binding_context,
        capability_registry=first_registry,
        transform_locus=TransformLocus.SOURCE,
        transform_capability_proof=proof,
    )
    second_selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=binding_context,
        capability_registry=second_registry,
        transform_locus=TransformLocus.SOURCE,
        transform_capability_proof=second_proof,
    )
    assert first_selected.selected_operation_digest == (
        second_selected.selected_operation_digest
    )


def test_trusted_registry_cannot_issue_or_accept_missing_mxfp8_scales() -> None:
    bf16 = _component(LOGICAL_VALUES)
    values = _component(
        VALUES,
        dtype="uint8",
        layout=replace(
            bf16.representation.layout,
            storage_encoding="mxfp8.e4m3-carrier.v1",
        ),
        rank=1,
    )
    scales = _component(
        BLOCK_SCALES,
        dtype="uint8",
        shape=(128, 928, 84),
        layout=_identity_layout(
            "experts",
            "intermediate_features",
            "input_feature_blocks",
            storage_encoding="mxfp8.e8m0-carrier.v1",
        ),
        rank=1,
    )
    complete_format = _format(
        source_storage=(bf16,),
        wire=(bf16,),
        destination_load_api=(values, scales),
        destination_runtime=(values, scales),
        destination_load_api_format=MXFP8_FORMAT,
        destination_runtime_format=MXFP8_FORMAT,
    )
    capability = AdapterOperationCapability.from_realized_format(
        complete_format,
        PhysicalFormatStage.WIRE,
        PhysicalFormatStage.DESTINATION_LOAD_API,
        transform_locus=TransformLocus.DESTINATION,
        implementation_id="nemo.quantize-mxfp8",
        implementation_version="1.0.0",
    )
    registry = _adapter_registry(
        adapter_id="vllm.refit-adapter",
        adapter_version="0.25.1",
        capabilities=(capability,),
    )
    complete_proof = registry.issue_transform(
        complete_format,
        PhysicalFormatStage.WIRE,
        PhysicalFormatStage.DESTINATION_LOAD_API,
        transform_locus=TransformLocus.DESTINATION,
        implementation_id="nemo.quantize-mxfp8",
        implementation_version="1.0.0",
    )
    with pytest.raises(ValueError, match="complete ordered component roles"):
        replace(
            complete_format,
            destination_load_api=(values,),
            destination_runtime=(values,),
        )

    assert complete_proof.destination_endpoint_capability.ordered_representation_digest
    assert registry.registry_digest


def test_trusted_registry_rejects_unregistered_implementation_identity() -> None:
    source = _component(LOGICAL_VALUES)
    wire = replace(
        source,
        representation=replace(source.representation, physical_dtype="float16"),
    )
    realized_format = _format(source_storage=(source,), wire=(wire,))
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.SOURCE,
        implementation_id="nemo.cast-float16",
        implementation_version="1.0.0",
    )
    registry = _adapter_registry(
        adapter_id="vllm.refit-adapter",
        adapter_version="0.25.1",
        capabilities=(capability,),
    )

    with pytest.raises(ValueError, match="does not authorize"):
        registry.issue_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            transform_locus=TransformLocus.SOURCE,
            implementation_id="caller.chosen-transform",
            implementation_version="999.0.0",
        )


def test_transform_capability_rejects_a_locus_invalid_for_its_stage_pair() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.SOURCE,
        implementation_id="nemo.physical-transform",
        implementation_version="1.0.0",
    )

    with pytest.raises(ValueError, match="locus.*stage pair"):
        replace(capability, transform_locus=TransformLocus.DESTINATION)


def test_transform_selection_requires_the_proofs_exact_stage_pair_and_locus() -> None:
    component = _component(LOGICAL_VALUES)
    realized_format = _format(source_storage=(component,))
    wrong_stage_proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.WIRE,
        PhysicalFormatStage.DESTINATION_LOAD_API,
        TransformLocus.DESTINATION,
    )
    wrong_locus_proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.DESTINATION_LOAD_API,
        PhysicalFormatStage.DESTINATION_RUNTIME,
        TransformLocus.DESTINATION_NATIVE_LOADER,
    )

    with pytest.raises(ValueError, match="exact stage pair"):
        select_transform(
            realized_format,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            PhysicalFormatStage.DESTINATION_RUNTIME,
            binding_context=_binding_context(),
            capability_registry=_registry_for(wrong_stage_proof),
            transform_locus=TransformLocus.DESTINATION,
            transform_capability_proof=wrong_stage_proof,
        )
    with pytest.raises(ValueError, match="locus mismatch"):
        select_transform(
            realized_format,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            PhysicalFormatStage.DESTINATION_RUNTIME,
            binding_context=_binding_context(),
            capability_registry=_registry_for(wrong_locus_proof),
            transform_locus=TransformLocus.DESTINATION,
            transform_capability_proof=wrong_locus_proof,
        )


@pytest.mark.parametrize("proof_kind", ["direct", "transform"])
def test_adapter_proofs_reject_endpoint_attestations_for_another_stage(
    proof_kind: str,
) -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    if proof_kind == "direct":
        proof = _proof(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
        )
    else:
        proof = _transform_proof(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            TransformLocus.SOURCE,
        )

    tampered_proof = copy(proof)
    tampered_endpoint = copy(proof.source_endpoint_capability)
    object.__setattr__(tampered_endpoint, "stage", PhysicalFormatStage.WIRE)
    object.__setattr__(
        tampered_proof,
        "source_endpoint_capability",
        tampered_endpoint,
    )
    with pytest.raises(ValueError, match="source endpoint capability stage"):
        _registry_for(proof)._verify(tampered_proof)


def test_transform_selection_rejects_an_untyped_realized_format() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        TransformLocus.SOURCE,
    )
    with pytest.raises(TypeError, match="RealizedBindingFormat"):
        select_transform(
            "not-a-realized-format",  # type: ignore[arg-type]
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=_binding_context(),
            capability_registry=_registry_for(proof),
            transform_locus=TransformLocus.SOURCE,
        )


def test_selected_operation_retains_executor_and_binding_identity() -> None:
    source = _component(LOGICAL_VALUES)
    wire = replace(
        source,
        representation=replace(source.representation, physical_dtype="float16"),
    )
    realized_format = _format(source_storage=(source,), wire=(wire,))
    capabilities = tuple(
        AdapterOperationCapability.from_realized_format(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            transform_locus=TransformLocus.SOURCE,
            implementation_id=implementation_id,
            implementation_version=implementation_version,
        )
        for implementation_id, implementation_version in (
            ("nemo.cast-float16", "1.0.0"),
            ("nemo.fused-cast-float16", "2.0.0"),
        )
    )
    registry = _adapter_registry(
        adapter_id="vllm.refit-adapter",
        adapter_version="0.25.1",
        capabilities=capabilities,
    )
    destination_evidence = _destination_evidence(registry, realized_format)
    _TEST_DESTINATION_EVIDENCE[registry.issuer_instance_id] = destination_evidence
    binding_context = _binding_context()
    selected_operations: list[SelectedRefitOperation] = []
    proofs: list[TransformCapabilityProof] = []
    for capability in capabilities:
        proof = registry.issue_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            transform_locus=TransformLocus.SOURCE,
            implementation_id=capability.implementation_id,
            implementation_version=capability.implementation_version,
        )
        proofs.append(proof)
        selected_operations.append(
            select_transform(
                realized_format,
                PhysicalFormatStage.SOURCE_STORAGE,
                PhysicalFormatStage.WIRE,
                binding_context=binding_context,
                capability_registry=registry,
                transform_locus=TransformLocus.SOURCE,
                transform_capability_proof=proof,
            )
        )

    first, second = selected_operations
    assert first.binding_identity == second.binding_identity
    assert first.capability_proof.implementation_id == "nemo.cast-float16"
    assert second.capability_proof.implementation_id == "nemo.fused-cast-float16"
    assert first.executor_key != second.executor_key
    assert first.selected_operation_digest != second.selected_operation_digest
    installed_first = _install_selected_operation(first, registry)
    assert execution_dispatch_key(installed_first) == first.executor_key
    rebound = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=_binding_context(allocation_generation="allocation-2"),
        capability_registry=registry,
        transform_locus=TransformLocus.SOURCE,
        transform_capability_proof=proofs[0],
    )
    assert rebound.capability_proof is first.capability_proof
    assert rebound.executor_key == first.executor_key
    assert rebound.selected_operation_digest != first.selected_operation_digest
    with pytest.raises(FrozenInstanceError):
        first.locus = TransformLocus.NONE  # type: ignore[misc]
    with pytest.raises(TypeError, match="plan compilation"):
        SelectedRefitOperation()
    with pytest.raises(TypeError, match="SelectedRefitOperation"):
        execution_dispatch_key(TransformLocus.SOURCE)  # type: ignore[arg-type]


def test_deserialized_selected_operation_requires_one_time_installation() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=_binding_context(),
        capability_registry=_registry_for(proof),
        transform_locus=TransformLocus.NONE,
        direct_copy_proof=proof,
    )
    registry = _registry_for(proof)
    restored = pickle.loads(pickle.dumps(selected))

    assert restored.selected_operation_digest == selected.selected_operation_digest
    assert restored.operation_base_proof == selected.operation_base_proof
    assert restored.signature == selected.signature
    with pytest.raises(TypeError, match="InstalledSelectedRefitOperation"):
        execution_dispatch_key(restored)
    installed = _install_selected_operation(restored, registry)
    assert execution_dispatch_key(installed) == selected.executor_key
    other_registry = _adapter_registry(
        adapter_id=registry.adapter_id,
        adapter_version=registry.adapter_version,
        capabilities=tuple(registry._capabilities),
    )
    with pytest.raises(ValueError, match="issuer identity"):
        _install_selected_operation(restored, other_registry)


@pytest.mark.parametrize("mutation", ("binding_identity", "executor_key"))
def test_selected_operation_install_rejects_rechecksummed_pickle_tamper(
    mutation: str,
) -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    registry = _registry_for(proof)
    selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=_binding_context(),
        capability_registry=registry,
        transform_locus=TransformLocus.NONE,
        direct_copy_proof=proof,
    )
    forged = pickle.loads(pickle.dumps(selected))
    if mutation == "binding_identity":
        identity = copy(forged.binding_identity)
        object.__setattr__(identity, "tensor_instance_id", "invented.tensor")
        object.__setattr__(
            identity,
            "binding_identity_digest",
            _REFIT_PLAN._canonical_digest(
                {
                    "type": "refit_binding_identity.v1",
                    **_REFIT_PLAN._binding_identity_payload(identity),
                }
            ),
        )
        object.__setattr__(forged, "binding_identity", identity)
    else:
        assert mutation == "executor_key"
        object.__setattr__(
            forged,
            "executor_key",
            replace(
                forged.executor_key,
                implementation_id="invented.executor",
            ),
        )
    object.__setattr__(
        forged,
        "selected_operation_digest",
        _REFIT_PLAN._canonical_digest(_REFIT_PLAN._selected_operation_payload(forged)),
    )

    with pytest.raises(
        ValueError,
        match="authenticated base|capability proof|signature mismatch",
    ):
        _install_selected_operation(forged, registry)


def test_realized_binding_format_and_schema_registry_survive_pickle() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))

    restored = pickle.loads(pickle.dumps(realized_format))

    assert restored == realized_format
    restored.format_schema_registry._require_registered(BF16_FORMAT)


def test_refit_binding_identity_cannot_be_self_attested_by_a_caller() -> None:
    with pytest.raises(TypeError, match="validated refit binding"):
        RefitBindingIdentity(
            graph_instance_id="invented.graph",
            tensor_instance_id="invented.tensor",
            semantic_graph_path="text.decoder",
            semantic_id="text.decoder.invented.weight",
            selection_group_id="not-a-real-selection-id",
            semantic_selection_digest=f"sha256:{'7' * 64}",
            source_realization_digest=f"sha256:{'8' * 64}",
        )


def test_refit_binding_context_is_derived_from_compiler_and_discovery_artifacts() -> (
    None
):
    context = _binding_context()

    assert context.graph_instance_id == "main"
    assert context.inventory_entry_id == "main.dense.weight"
    assert context.tensor_instance_id == "main.model.weight"
    assert context.selection_group_id.startswith("sha256:")
    assert context.intent_group_id.startswith("sha256:")
    assert context.runtime_source_result_digest.startswith("sha256:")
    assert context.source_format is BF16_FORMAT
    assert context.destination_format is BF16_FORMAT


def test_refit_binding_preserves_compact_grouped_source_boundary_slices() -> None:
    intents, selection, request, results, adapter, slices = (
        _mixed_boundary_binding_inputs()
    )
    slices_by_precision = {
        source_slice.rollout_assignment.precision: source_slice
        for source_slice in slices
    }

    contexts = {
        context.destination_format.family: context
        for context in _bind_mixed_refit_contexts(
            intents=intents,
            selection=selection,
            request=request,
            results=results,
            adapter=adapter,
            bindings=tuple(
                (source_slice, source_slice.source_binding.source_realizations[0])
                for source_slice in slices_by_precision.values()
            ),
        )
    }

    assert set(contexts) == {"bf16", "mxfp8"}
    assert contexts["bf16"].inventory_entry_id == "main.moe.routed.gate"
    assert contexts["bf16"].source_record_id == "main.moe.routed.gate.source"
    assert contexts["bf16"].source_binding_slice_digest != (
        contexts["mxfp8"].source_binding_slice_digest
    )
    assert tuple(
        (span.start, span.stop, span.step)
        for span in contexts["bf16"].source_region.axis_selections[0].spans
    ) == ((0, 3, 2),)
    assert tuple(
        (span.start, span.stop, span.step)
        for span in contexts["mxfp8"].source_region.axis_selections[0].spans
    ) == ((1, 2, 1),)


def test_native_mxfp8_values_and_scales_bind_and_install_as_one_atomic_set() -> None:
    from tests.unit.precision_policy import test_runtime_binding

    inputs = _native_mxfp8_binding_inputs()
    adapters = inputs["_runtime_topology_adapters"]
    assert isinstance(adapters, tuple)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            test_runtime_binding.topology_module,
            "_default_adapters",
            lambda: adapters,
        )
        (context,) = _REFIT_PLAN.bind_refit_contexts(
            intents=inputs["intents"],
            active_selection=inputs["active_selection"],
            active_request=inputs["active_request"],
            active_results=inputs["active_results"],
            installation_registry=RefitPlanInstallationRegistry(),
            bindings=inputs["bindings"],
        )
    envelope = context.binding_context
    assert tuple(
        source_slice.component_key.component_role
        for source_slice in envelope.source_binding_slices
    ) == (VALUES, BLOCK_SCALES)
    source_storage = tuple(
        PhysicalComponentDescriptor(
            representation=PhysicalRepresentation(
                role=source_slice.component_key.component_role,
                physical_dtype=realization.components[0].carrier_dtype.value,
                physical_shape=realization.components[0].physical_shape,
                layout=_REFIT_PLAN.derive_source_storage_layout(
                    context,
                    realization.components[0],
                ),
            ),
            placement=EndpointPlacement(0, "cuda", "device"),
            source_storage_component=realization.components[0],
        )
        for source_slice, realization in zip(
            envelope.source_binding_slices,
            envelope.source_realizations,
            strict=True,
        )
    )
    realized_format = _format(
        source_storage=source_storage,
        source_storage_format=MXFP8_FORMAT,
        wire_format=MXFP8_FORMAT,
        destination_load_api_format=MXFP8_FORMAT,
        destination_runtime_format=MXFP8_FORMAT,
    )
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    registry = _registry_for(proof)
    selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=context,
        capability_registry=registry,
        transform_locus=TransformLocus.NONE,
        direct_copy_proof=proof,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            test_runtime_binding.topology_module,
            "_default_adapters",
            lambda: adapters,
        )
        installed = _REFIT_PLAN.install_selected_refit_operation(
            selected_operation=pickle.loads(pickle.dumps(selected)),
            intents=inputs["intents"],
            active_selection=inputs["active_selection"],
            active_request=inputs["active_request"],
            active_results=inputs["active_results"],
            capability_registry=registry,
            installation_registry=RefitPlanInstallationRegistry(),
        )

    assert execution_dispatch_key(installed) == selected.executor_key
    assert selected.binding_identity.source_realization_digest == (
        envelope.source_realization_digest
    )


def test_refit_binding_rejects_a_boundary_assignment_swapped_between_slices() -> None:
    intents, selection, request, results, adapter, slices = (
        _mixed_boundary_binding_inputs()
    )
    slices_by_precision = {
        source_slice.rollout_assignment.precision: source_slice
        for source_slice in slices
    }
    bf16_slice = slices_by_precision["bf16"]
    mxfp8_slice = slices_by_precision["mxfp8"]
    bf16_realization = bf16_slice.source_binding.source_realizations[0]
    assert isinstance(bf16_realization, SourceStorageRealization)
    swapped = replace(
        bf16_slice,
        rollout_assignment=mxfp8_slice.rollout_assignment,
    )

    with pytest.raises(ValueError, match="occur exactly once"):
        _bind_mixed_refit_contexts(
            intents=intents,
            selection=selection,
            request=request,
            results=results,
            adapter=adapter,
            bindings=((swapped, bf16_realization),),
        )


def test_refit_binding_accepts_independently_deserialized_owned_artifacts() -> None:
    inputs = _binding_context_inputs()
    for field_name in (
        "intents",
        "active_selection",
        "active_request",
        "active_results",
        "source_binding_slice",
        "source_realization",
    ):
        inputs[field_name] = pickle.loads(pickle.dumps(inputs[field_name]))

    context = _bind_refit_context_inputs(inputs)

    assert context.inventory_entry_id == "main.dense.weight"
    assert context.tensor_instance_id == "main.model.weight"


def test_refit_binding_rejects_cross_generation_source_transplant() -> None:
    active_inputs = _binding_context_inputs(allocation_generation="allocation-a")
    transplanted_inputs = _binding_context_inputs(allocation_generation="allocation-b")
    transplanted_inputs["intents"] = pickle.loads(
        pickle.dumps(transplanted_inputs["intents"])
    )
    transplanted_inputs["source_binding_slice"] = pickle.loads(
        pickle.dumps(transplanted_inputs["source_binding_slice"])
    )
    transplanted_inputs["source_realization"] = pickle.loads(
        pickle.dumps(transplanted_inputs["source_realization"])
    )
    for field_name in ("active_selection", "active_request", "active_results"):
        transplanted_inputs[field_name] = pickle.loads(
            pickle.dumps(active_inputs[field_name])
        )

    with pytest.raises(ValueError, match="runtime-bound compiler output"):
        _bind_refit_context_inputs(transplanted_inputs)


def test_batch_refit_binding_validates_the_intent_group_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intents, selection, request, results, adapter, slices = (
        _mixed_boundary_binding_inputs()
    )
    calls = [0]
    validate = _REFIT_PLAN.validate_compiled_precision_intent_group

    def counting_validate(
        group: CompiledPrecisionIntentGroup,
        *,
        active_selection: CompiledPrecisionSelectionGroup,
        active_request: RuntimeSourceDiscoveryRequest,
        active_results: tuple[RuntimeSourceDiscoveryResult, ...],
    ) -> object:
        calls[0] += 1
        return validate(
            group,
            active_selection=active_selection,
            active_request=active_request,
            active_results=active_results,
        )

    monkeypatch.setattr(
        _REFIT_PLAN,
        "validate_compiled_precision_intent_group",
        counting_validate,
    )
    assert hasattr(_REFIT_PLAN, "bind_refit_contexts")
    contexts = _bind_mixed_refit_contexts(
        intents=intents,
        selection=selection,
        request=request,
        results=results,
        adapter=adapter,
        bindings=tuple(
            (source_slice, source_slice.source_binding.source_realizations[0])
            for source_slice in slices
        ),
    )

    assert len(contexts) == len(slices)
    assert calls == [1]


def test_refit_binding_context_rejects_an_invented_selection_identity() -> None:
    inputs = _binding_context_inputs()
    intents = copy(inputs["intents"])
    object.__setattr__(
        intents,
        "selection_group_id",
        f"sha256:{'0' * 64}",
    )
    inputs["intents"] = intents

    with pytest.raises(ValueError, match="runtime-bound compiler output"):
        _bind_refit_context_inputs(inputs)


def test_refit_binding_rejects_reconstructed_runtime_source_owner() -> None:
    inputs = _binding_context_inputs()
    source_slice = inputs["source_binding_slice"]
    assert isinstance(source_slice, RuntimeSourceBindingSlice)
    forged_record = replace(
        source_slice.source_binding.source_record,
        source_native_owner_id="invented.tensor",
    )
    inputs["source_binding_slice"] = replace(
        source_slice,
        source_binding=replace(
            source_slice.source_binding,
            source_record=forged_record,
        ),
    )

    with pytest.raises(ValueError, match="occur exactly once"):
        _bind_refit_context_inputs(inputs)


def test_refit_binding_rejects_self_consistent_forged_intent_metadata() -> None:
    inputs = _binding_context_inputs()
    intents = inputs["intents"]
    assert isinstance(intents, CompiledPrecisionIntentGroup)
    forged_graph_intent = replace(
        intents.graph_intents[0],
        out_of_scope_inventory_entry_ids=("invented.owner",),
    )
    inputs["intents"] = replace(
        intents,
        graph_intents=(forged_graph_intent, *intents.graph_intents[1:]),
    )

    with pytest.raises(ValueError, match="runtime-bound compiler output"):
        _bind_refit_context_inputs(inputs)


def test_refit_binding_context_rejects_a_noncanonical_graph_intent_id() -> None:
    inputs = _binding_context_inputs()
    intents = inputs["intents"]
    assert isinstance(intents, CompiledPrecisionIntentGroup)
    forged_graph_intent = replace(
        intents.graph_intents[0],
        intent_id=f"sha256:{'f' * 64}",
    )
    inputs["intents"] = replace(
        intents,
        graph_intents=(forged_graph_intent, *intents.graph_intents[1:]),
    )

    with pytest.raises(ValueError, match="runtime-bound compiler output"):
        _bind_refit_context_inputs(inputs)


def test_transform_selection_rejects_source_shape_not_realized_by_discovery() -> None:
    context = _binding_context()
    realized_format = _format(
        source_storage=(_component(LOGICAL_VALUES, shape=(7, 8)),)
    )
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    with pytest.raises(ValueError, match="source storage.*shape"):
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=context,
            capability_registry=_registry_for(proof),
            transform_locus=TransformLocus.NONE,
            direct_copy_proof=proof,
        )


def test_transform_selection_rejects_equal_cardinality_forged_source_layout() -> None:
    context = _binding_context()
    forged = _component(
        LOGICAL_VALUES,
        shape=(4, 16),
        layout=_identity_layout(
            "output_features",
            "input_features",
            storage_encoding="forged-source.v1",
        ),
    )
    realized_format = _format(source_storage=(forged,))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    with pytest.raises(ValueError, match="source storage.*realization"):
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=context,
            capability_registry=_registry_for(proof),
            transform_locus=TransformLocus.NONE,
            direct_copy_proof=proof,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "shape",
        "dtype",
        "axes",
        "mapping",
        "encoding",
        "padding",
        "permutation",
        "swizzle",
    ),
)
def test_transform_selection_rejects_unattested_source_storage_facts(
    mutation: str,
) -> None:
    context = _binding_context()
    attested = _component(LOGICAL_VALUES)
    representation = attested.representation
    layout = representation.layout
    if mutation == "shape":
        representation = replace(representation, physical_shape=(4, 16))
    elif mutation == "dtype":
        representation = replace(representation, physical_dtype="float16")
    elif mutation == "axes":
        layout = replace(
            layout,
            axis_order=("axis_1", "axis_0"),
            logical_to_physical_axes=(
                PhysicalAxisMapping(
                    "output_features", ("axis_1",), "identity.axis-map.v1"
                ),
                PhysicalAxisMapping(
                    "input_features", ("axis_0",), "identity.axis-map.v1"
                ),
            ),
        )
    elif mutation == "mapping":
        layout = replace(
            layout,
            logical_to_physical_axes=(
                PhysicalAxisMapping(
                    "output_features", ("axis_1",), "identity.axis-map.v1"
                ),
                PhysicalAxisMapping(
                    "input_features", ("axis_0",), "identity.axis-map.v1"
                ),
            ),
        )
    elif mutation == "encoding":
        layout = replace(layout, storage_encoding="forged-source.v1")
    elif mutation == "padding":
        layout = replace(
            layout,
            padding=(
                PhysicalPadding(
                    logical_axis="output_features",
                    pad_before=0,
                    pad_after=1,
                    semantics=PhysicalPaddingSemantics.UNSPECIFIED_IGNORED,
                    fill_encoding=None,
                ),
            ),
        )
    elif mutation == "permutation":
        layout = replace(
            layout,
            permutation=PhysicalPermutation(
                permutation_id="forged-permutation.v1",
                input_axis_order=("axis_1", "axis_0"),
                output_axis_order=("axis_0", "axis_1"),
            ),
        )
    else:
        assert mutation == "swizzle"
        layout = replace(layout, swizzle_id="forged-swizzle.v1")
    if mutation not in {"shape", "dtype"}:
        representation = replace(representation, layout=layout)
    forged = replace(attested, representation=representation)
    realized_format = _format(source_storage=(forged,))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    with pytest.raises(ValueError, match="source storage"):
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=context,
            capability_registry=_registry_for(proof),
            transform_locus=TransformLocus.NONE,
            direct_copy_proof=proof,
        )


def test_grouped_boundary_refit_uses_attested_full_native_source_storage() -> None:
    intents, selection, request, results, adapter, slices = (
        _mixed_boundary_binding_inputs()
    )
    source_slice = next(
        item for item in slices if item.rollout_assignment.precision == "bf16"
    )
    source_realization = source_slice.source_binding.source_realizations[0]
    assert isinstance(source_realization, SourceStorageRealization)
    (context,) = _bind_mixed_refit_contexts(
        intents=intents,
        selection=selection,
        request=request,
        results=results,
        adapter=adapter,
        bindings=((source_slice, source_realization),),
    )
    native = _component(
        LOGICAL_VALUES,
        shape=(3, 2, 8, 8),
        source_component_id="main.moe.routed.gate.source.component",
        layout=PhysicalLayoutDescriptor(
            axis_order=("axis_0", "axis_1", "axis_2", "axis_3"),
            logical_to_physical_axes=(
                PhysicalAxisMapping(
                    "global_decoder_layer", ("axis_0",), "identity.axis-map.v1"
                ),
                PhysicalAxisMapping("moe_ordinal", ("axis_0",), "identity.axis-map.v1"),
                PhysicalAxisMapping("expert", ("axis_1",), "identity.axis-map.v1"),
                PhysicalAxisMapping(
                    "output_features", ("axis_2",), "identity.axis-map.v1"
                ),
                PhysicalAxisMapping(
                    "input_features", ("axis_3",), "identity.axis-map.v1"
                ),
            ),
            padding=(),
            permutation=None,
            storage_encoding="plain-bfloat16.v1",
        ),
    )
    assert native.source_storage_component is not None
    native = replace(
        native,
        representation=replace(
            native.representation,
            layout=_REFIT_PLAN.derive_source_storage_layout(
                context,
                native.source_storage_component,
            ),
        ),
    )
    direct_format = _format(source_storage=(native,))
    direct_proof = _proof(
        direct_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    with pytest.raises(ValueError, match="partial source region.*SOURCE"):
        select_transform(
            direct_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=context,
            capability_registry=_registry_for(direct_proof),
            transform_locus=TransformLocus.NONE,
            direct_copy_proof=direct_proof,
        )

    compact = replace(
        _component(
            LOGICAL_VALUES,
            shape=(2, 2, 8, 8),
            layout=PhysicalLayoutDescriptor(
                axis_order=("axis_0", "axis_1", "axis_2", "axis_3"),
                logical_to_physical_axes=native.representation.layout.logical_to_physical_axes,
                padding=(),
                permutation=None,
                storage_encoding="plain-bfloat16.v1",
            ),
        ),
        source_storage_component=None,
    )
    transformed_format = _format(
        source_storage=(native,),
        wire=(compact,),
        destination_load_api=(compact,),
        destination_runtime=(compact,),
    )
    unrelated_transform_proof = _transform_proof(
        transformed_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        TransformLocus.SOURCE,
    )
    with pytest.raises(ValueError, match="exact source region extraction"):
        select_transform(
            transformed_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=context,
            capability_registry=_registry_for(unrelated_transform_proof),
            transform_locus=TransformLocus.SOURCE,
            transform_capability_proof=unrelated_transform_proof,
        )
    transform_proof = _transform_proof(
        transformed_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        TransformLocus.SOURCE,
        binding_context=context,
    )
    selected = select_transform(
        transformed_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=context,
        capability_registry=_registry_for(transform_proof),
        transform_locus=TransformLocus.SOURCE,
        transform_capability_proof=transform_proof,
    )

    assert selected.binding_identity.source_realization_digest == (
        context.source_realization_digest
    )


def test_fused_qkv_subregion_requires_a_source_split_transform() -> None:
    inputs = _fused_qkv_binding_inputs()
    context = _bind_refit_context_inputs(inputs)
    q_projection = context.source_region
    source_component = context.source_realization.components[0]
    native_layout = _REFIT_PLAN.derive_source_storage_layout(
        context,
        source_component,
    )
    native = PhysicalComponentDescriptor(
        representation=PhysicalRepresentation(
            role=LOGICAL_VALUES,
            physical_dtype=source_component.carrier_dtype.value,
            physical_shape=source_component.physical_shape,
            layout=native_layout,
        ),
        placement=EndpointPlacement(0, "cuda", "device"),
        source_storage_component=source_component,
    )
    selected_shape = tuple(
        selection.cardinality for selection in q_projection.axis_selections
    )
    compact = PhysicalComponentDescriptor(
        representation=PhysicalRepresentation(
            role=LOGICAL_VALUES,
            physical_dtype=source_component.carrier_dtype.value,
            physical_shape=tuple(
                axis.resolve(selected_shape) for axis in source_component.physical_axes
            ),
            layout=_REFIT_PLAN._derive_source_storage_layout(
                context.binding_context,
                source_component,
                normalized_shape=selected_shape,
                source_binding_slice=context.binding_context.source_binding_slice,
                source_realization=context.binding_context.source_realization,
                component_role=LOGICAL_VALUES,
            ),
        ),
        placement=EndpointPlacement(0, "cuda", "device"),
        source_storage_component=None,
    )
    realized_format = _format(
        source_storage=(native,),
        wire=(compact,),
        destination_load_api=(compact,),
        destination_runtime=(compact,),
    )
    unrelated_proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        TransformLocus.SOURCE,
    )
    with pytest.raises(ValueError, match="exact source region extraction"):
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=context,
            capability_registry=_registry_for(unrelated_proof),
            transform_locus=TransformLocus.SOURCE,
            transform_capability_proof=unrelated_proof,
        )
    proof = _transform_proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        TransformLocus.SOURCE,
        binding_context=context,
    )
    selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=context,
        capability_registry=_registry_for(proof),
        transform_locus=TransformLocus.SOURCE,
        transform_capability_proof=proof,
    )

    assert not _REFIT_PLAN._source_region_is_complete(q_projection)
    assert tuple(
        (span.start, span.stop) for span in q_projection.axis_selections[0].spans
    ) == ((0, 1),)
    assert selected.capability_proof.source_region_extraction is not None
    assert selected.capability_proof.source_region_extraction.wire_component_shapes == (
        selected_shape,
    )


@pytest.mark.parametrize("mutation", ("amount", "axis", "mapping_formula"))
def test_padded_source_layout_is_deterministically_derived(mutation: str) -> None:
    context = _binding_context(padded_source=True)
    source_component = context.source_realization.components[0]
    layout = _REFIT_PLAN.derive_source_storage_layout(context, source_component)
    assert layout.padding[0].logical_axis == "input_features"
    assert layout.padding[0].pad_before == 0
    assert layout.padding[0].pad_after == 8
    if mutation == "amount":
        layout = replace(
            layout,
            padding=(replace(layout.padding[0], pad_after=1),),
        )
    elif mutation == "axis":
        layout = replace(
            layout,
            padding=(replace(layout.padding[0], logical_axis="output_features"),),
        )
    else:
        assert mutation == "mapping_formula"
        layout = replace(
            layout,
            logical_to_physical_axes=(
                replace(
                    layout.logical_to_physical_axes[0],
                    mapping_id="forged.axis-map.v1",
                ),
                *layout.logical_to_physical_axes[1:],
            ),
        )
    physical = PhysicalComponentDescriptor(
        representation=PhysicalRepresentation(
            role=LOGICAL_VALUES,
            physical_dtype=source_component.carrier_dtype.value,
            physical_shape=source_component.physical_shape,
            layout=layout,
        ),
        placement=EndpointPlacement(0, "cuda", "device"),
        source_storage_component=source_component,
    )
    realized_format = _format(source_storage=(physical,))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    with pytest.raises(ValueError, match="source storage"):
        select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=context,
            capability_registry=_registry_for(proof),
            transform_locus=TransformLocus.NONE,
            direct_copy_proof=proof,
        )


@pytest.mark.parametrize(
    ("field_name", "invented_value"),
    (
        ("tensor_instance_id", "invented.tensor"),
        ("source_component_dtypes", ("float16",)),
        ("source_output_dtype", "float16"),
        ("source_output_shape", (4, 16)),
        ("source_output_encoding", "invented_encoding"),
        ("selection_group_id", f"sha256:{'a' * 64}"),
        ("semantic_selection_digest", f"sha256:{'b' * 64}"),
        ("intent_group_id", f"sha256:{'c' * 64}"),
        ("graph_intent_id", f"sha256:{'d' * 64}"),
        ("runtime_source_result_digest", f"sha256:{'e' * 64}"),
        ("runtime_source_digest", f"sha256:{'f' * 64}"),
    ),
)
def test_binding_context_rejects_self_consistent_cached_field_tamper(
    field_name: str,
    invented_value: object,
) -> None:
    inputs = _binding_context_inputs()
    installed = _bind_refit_context_inputs(inputs)
    context = copy(installed.binding_context)
    object.__setattr__(context, field_name, invented_value)
    if field_name == "runtime_source_digest":
        slice_digest = _REFIT_PLAN._source_binding_slice_digest(
            context.source_binding_slice,
            runtime_source_digest=context.runtime_source_digest,
            component_key_digest=context.semantic_component_key_digest,
        )
        object.__setattr__(context, "source_binding_slice_digest", slice_digest)
        object.__setattr__(context, "source_binding_slice_digests", (slice_digest,))
    object.__setattr__(
        context,
        "binding_context_digest",
        _REFIT_PLAN._canonical_digest(
            {
                "type": "refit_binding_context.v1",
                **_REFIT_PLAN._binding_context_payload(context),
            }
        ),
    )

    with pytest.raises(ValueError, match="active runtime compiler artifacts"):
        _install_refit_context_inputs(context, inputs)


def test_transform_selection_derives_identity_from_the_validated_context() -> None:
    context = _binding_context()
    component = _component(LOGICAL_VALUES)
    realized_format = _format(source_storage=(component,))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )

    selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=context,
        capability_registry=_registry_for(proof),
        transform_locus=TransformLocus.NONE,
        direct_copy_proof=proof,
    )
    destination_evidence = _TEST_DESTINATION_EVIDENCE[proof.issuer_instance_id]

    assert selected.binding_identity.graph_instance_id == context.graph_instance_id
    assert selected.binding_identity.inventory_entry_id == context.inventory_entry_id
    assert selected.binding_identity.selection_group_id == context.selection_group_id
    assert selected.binding_identity.intent_group_id == context.intent_group_id
    assert selected.binding_identity.runtime_source_result_digest == (
        context.runtime_source_result_digest
    )
    assert (
        selected.binding_identity.destination_owner_instance_id
        == destination_evidence.destination_owner_instance_id
    )
    assert (
        selected.binding_identity.destination_owner_instance_id
        != _routes()[1].destination_endpoint_instance_id
    )
    assert (
        selected.binding_identity.finalizer_instance_id
        == destination_evidence.finalizer_instance_id
    )
    assert (
        selected.binding_identity.finalizer_instance_id
        != _routes()[2].destination_endpoint_instance_id
    )


def test_destination_binding_proof_is_sealed_and_pickle_safe() -> None:
    with pytest.raises(TypeError, match="issued by an adapter"):
        DestinationBindingProof()

    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    operation_proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    registry = _registry_for(operation_proof)
    proof = _destination_binding_proof(registry, realized_format)

    assert pickle.loads(pickle.dumps(proof)) == proof


def test_destination_binding_cannot_be_minted_from_caller_owner_strings() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    operation_proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    registry = _registry_for(operation_proof)

    with pytest.raises(TypeError, match="live version-adapter discovery"):
        DestinationBindingEvidence(
            destination_owner_instance_id="caller.owner",
            finalizer_instance_id="caller.finalizer",
        )
    assert not hasattr(registry, "destination_registry")
    assert not hasattr(registry, "issue_destination_binding")
    with pytest.raises(ValueError, match="another adapter"):
        _REFIT_PLAN._destination_binding_mint_authority_for_adapter(
            registry,
            adapter_instance=object(),
        )
    authority = _TEST_DESTINATION_AUTHORITIES[registry.issuer_instance_id]
    with pytest.raises(TypeError, match="live adapter-owned object identity"):
        authority.begin_storage_generation("caller.generation")


def test_destination_binding_proof_rejects_another_physical_destination() -> None:
    component = _component(LOGICAL_VALUES)
    realized_format = _format(source_storage=(component,))
    capability = AdapterOperationCapability.from_realized_format(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.NONE,
        implementation_id="nemo.direct-copy",
        implementation_version="1.0.0",
    )
    moved = replace(component, placement=replace(component.placement, rank=1))
    other_destination = _format(
        source_storage=(component,),
        destination_load_api=(moved,),
        destination_runtime=(moved,),
    )
    registry = _adapter_registry(
        adapter_id="test.refit-adapter",
        adapter_version="1.0.0",
        capabilities=(capability,),
    )
    storage_generation = object()
    realized_evidence = _destination_evidence(
        registry,
        realized_format,
        storage_generation=storage_generation,
    )
    other_evidence = _destination_evidence(
        registry,
        other_destination,
        storage_generation=storage_generation,
    )
    operation_proof = registry.issue_direct_copy(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        implementation_id="nemo.direct-copy",
        implementation_version="1.0.0",
    )
    destination_proof = _TEST_DESTINATION_AUTHORITIES[
        registry.issuer_instance_id
    ].issue_destination_binding(
        other_destination,
        evidence=other_evidence,
    )

    with pytest.raises(ValueError, match="differs from the realized destination"):
        _select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=_binding_context(),
            capability_registry=registry,
            destination_binding_proof=destination_proof,
            transform_locus=TransformLocus.NONE,
            direct_copy_proof=operation_proof,
        )


def test_selection_accepts_independently_deserialized_destination_authority() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    operation_proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    registry = _registry_for(operation_proof)
    destination_proof = _destination_binding_proof(registry, realized_format)
    inputs = _binding_context_inputs()
    context = _bind_refit_context_inputs(inputs)
    restored_context = _install_refit_context_inputs(
        pickle.loads(pickle.dumps(context.binding_context)),
        inputs,
    )

    selected = _select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=restored_context,
        capability_registry=registry,
        destination_binding_proof=pickle.loads(pickle.dumps(destination_proof)),
        transform_locus=TransformLocus.NONE,
        direct_copy_proof=operation_proof,
    )

    assert selected.destination_binding_proof == destination_proof


def test_installed_context_requires_exact_issued_object_identity() -> None:
    context = _binding_context()
    forged = object.__new__(type(context))
    object.__setattr__(
        forged,
        "_installation_registry",
        context._installation_registry,
    )
    object.__setattr__(forged, "_issuance_nonce", context._issuance_nonce)

    with pytest.raises(ValueError, match="issued installation handle"):
        _ = forged.tensor_instance_id
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(context)


def test_installed_context_rejects_post_install_payload_mutation() -> None:
    inputs = _binding_context_inputs()
    context = _bind_refit_context_inputs(inputs)
    trusted_tensor_id = context.tensor_instance_id
    raw_context = context.binding_context
    object.__setattr__(raw_context, "tensor_instance_id", "invented.tensor")

    assert context.tensor_instance_id == trusted_tensor_id
    with pytest.raises(ValueError, match="canonical derivation|active runtime"):
        _install_refit_context_inputs(raw_context, inputs)


def test_runtime_generation_rollover_revokes_installed_context() -> None:
    installation_registry = RefitPlanInstallationRegistry()
    first = _binding_context(
        allocation_generation="allocation-a",
        installation_registry=installation_registry,
    )
    _binding_context(
        allocation_generation="allocation-b",
        installation_registry=installation_registry,
    )

    with pytest.raises(
        ValueError, match="issued installation handle|generation is stale"
    ):
        _ = first.tensor_instance_id


def test_context_batch_install_deduplicates_canonical_derivation() -> None:
    from tests.unit.precision_policy import test_runtime_binding

    inputs = _binding_context_inputs()
    context = _bind_refit_context_inputs(inputs)
    raw_context = pickle.loads(pickle.dumps(context.binding_context))
    duplicate = pickle.loads(pickle.dumps(raw_context))
    adapters = inputs["_runtime_topology_adapters"]
    assert isinstance(adapters, tuple)
    calls = [0]
    original = _REFIT_PLAN._bind_refit_context_from_index

    def counted_bind(*args: object, **kwargs: object) -> object:
        calls[0] += 1
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            test_runtime_binding.topology_module,
            "_default_adapters",
            lambda: adapters,
        )
        monkeypatch.setattr(
            _REFIT_PLAN,
            "_bind_refit_context_from_index",
            counted_bind,
        )
        installed = _REFIT_PLAN.install_refit_contexts(
            contexts=(raw_context, duplicate),
            intents=inputs["intents"],
            active_selection=inputs["active_selection"],
            active_request=inputs["active_request"],
            active_results=inputs["active_results"],
            installation_registry=RefitPlanInstallationRegistry(),
        )

    assert calls[0] == len(raw_context.source_binding_slices)
    assert installed[0] is installed[1]


def test_operation_batch_install_derives_large_binding_base_once() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    registry = _registry_for(proof)
    selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=_binding_context(),
        capability_registry=registry,
        transform_locus=TransformLocus.NONE,
        direct_copy_proof=proof,
    )
    heavy_operations = (
        "_require_exact_structural_match",
        "_validate_binding_context",
        "_source_storage_lowering_digest",
        "_physical_binding_digest",
        "physical_representation_digest",
        "endpoint_placement_digest",
        "_require_proof_route",
        "_select_transform_impl",
    )

    def install_with_counts(
        operations: tuple[SelectedRefitOperation, ...],
    ) -> tuple[tuple[object, ...], dict[str, int]]:
        counts = dict.fromkeys(heavy_operations, 0)
        with pytest.MonkeyPatch.context() as monkeypatch:
            for operation_name in heavy_operations:
                original = getattr(_REFIT_PLAN, operation_name)

                def counted(
                    *args: object,
                    __name: str = operation_name,
                    __original: Callable[..., object] = original,
                    **kwargs: object,
                ) -> object:
                    counts[__name] += 1
                    return __original(*args, **kwargs)

                monkeypatch.setattr(_REFIT_PLAN, operation_name, counted)
            installed = _install_selected_operations(operations, registry)
        return installed, counts

    raw = pickle.loads(pickle.dumps(selected))
    independently_deserialized = tuple(
        pickle.loads(pickle.dumps(raw)) for _ in range(16)
    )
    ignored_duplicate_body = pickle.loads(pickle.dumps(raw))
    object.__setattr__(
        ignored_duplicate_body.realized_format.source_storage[0].representation,
        "physical_shape",
        (4, 16),
    )
    _, single_counts = install_with_counts((raw,))
    installed, batch_counts = install_with_counts(
        (*independently_deserialized, ignored_duplicate_body)
    )

    assert batch_counts == single_counts
    assert all(handle is installed[0] for handle in installed)

    forged_reference = pickle.loads(pickle.dumps(raw))
    object.__setattr__(
        forged_reference.operation_base_proof,
        "physical_binding_digest",
        f"sha256:{'f' * 64}",
    )
    object.__setattr__(
        forged_reference.operation_base_proof,
        "base_digest",
        _REFIT_PLAN._canonical_digest(
            _REFIT_PLAN._refit_operation_base_payload(
                forged_reference.operation_base_proof
            )
        ),
    )
    object.__setattr__(
        forged_reference,
        "selected_operation_digest",
        _REFIT_PLAN._canonical_digest(
            _REFIT_PLAN._selected_operation_payload(forged_reference)
        ),
    )
    with pytest.raises(ValueError, match="base proof signature mismatch"):
        _install_selected_operation(forged_reference, registry)


def test_installed_operation_requires_exact_issued_object_identity() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=_binding_context(),
        capability_registry=_registry_for(proof),
        transform_locus=TransformLocus.NONE,
        direct_copy_proof=proof,
    )
    installed = _install_selected_operation(selected, _registry_for(proof))
    forged = object.__new__(type(installed))
    object.__setattr__(
        forged,
        "_installation_registry",
        installed._installation_registry,
    )
    object.__setattr__(forged, "_issuance_nonce", installed._issuance_nonce)

    with pytest.raises(ValueError, match="issued installation handle"):
        execution_dispatch_key(forged)
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(installed)
    object.__setattr__(installed, "_issuance_nonce", object())
    with pytest.raises(ValueError, match="issued installation handle"):
        execution_dispatch_key(installed)


def test_source_generation_rollover_revokes_installed_operation() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    selected = select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=_binding_context(),
        capability_registry=_registry_for(proof),
        transform_locus=TransformLocus.NONE,
        direct_copy_proof=proof,
    )
    installation_registry = RefitPlanInstallationRegistry()
    installed = _install_selected_operation(
        selected,
        _registry_for(proof),
        installation_registry=installation_registry,
    )
    _binding_context(
        allocation_generation="allocation-2",
        installation_registry=installation_registry,
    )

    with pytest.raises(
        ValueError, match="issued installation handle|generation is stale"
    ):
        execution_dispatch_key(installed)


def test_complete_source_region_rejects_stale_extraction_and_hmac_morph() -> None:
    context = _binding_context()
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    extraction = _REFIT_PLAN.SourceRegionExtractionCapability(
        kind=_REFIT_PLAN.SourceRegionTransformKind.GATHER_SPLIT,
        source_binding_set_digest=f"sha256:{'a' * 64}",
        source_region_digest=f"sha256:{'b' * 64}",
        selected_source_shapes=((8, 8),),
        source_component_ids=("main.dense.weight.component",),
        wire_component_roles=(LOGICAL_VALUES,),
        wire_component_shapes=((8, 8),),
        wire_component_axis_orders=(("axis_0", "axis_1"),),
        wire_representation_digest=physical_representation_digest(
            realized_format,
            PhysicalFormatStage.WIRE,
        ),
    )
    capabilities = tuple(
        AdapterOperationCapability.from_realized_format(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            transform_locus=TransformLocus.SOURCE,
            implementation_id="nemo.physical-transform",
            implementation_version="1.0.0",
            source_region_extraction=registered_extraction,
        )
        for registered_extraction in (None, extraction)
    )
    registry = _adapter_registry(
        adapter_id="test.refit-adapter",
        adapter_version="1.0.0",
        capabilities=capabilities,
    )
    destination_evidence = _destination_evidence(registry, realized_format)
    plain_proof = registry.issue_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.SOURCE,
        implementation_id="nemo.physical-transform",
        implementation_version="1.0.0",
    )
    stale_extraction_proof = registry.issue_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.SOURCE,
        implementation_id="nemo.physical-transform",
        implementation_version="1.0.0",
        source_region_extraction=extraction,
    )
    morphed = pickle.loads(pickle.dumps(plain_proof))
    object.__setattr__(morphed, "source_region_extraction", extraction)

    with pytest.raises(ValueError, match="signature mismatch"):
        registry._verify(morphed)
    with pytest.raises(ValueError, match="exact source region extraction"):
        _select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=context,
            capability_registry=registry,
            destination_binding_proof=_TEST_DESTINATION_AUTHORITIES[
                registry.issuer_instance_id
            ].issue_destination_binding(
                realized_format,
                evidence=destination_evidence,
            ),
            transform_locus=TransformLocus.SOURCE,
            transform_capability_proof=stale_extraction_proof,
        )


def test_partial_atomic_mxfp8_extraction_hmac_binds_values_and_scales() -> None:
    values = _component(
        VALUES,
        dtype="uint8",
        shape=(4, 8),
        layout=_identity_layout(
            "output_features",
            "input_features",
            storage_encoding="mxfp8.e4m3-carrier.v1",
        ),
    )
    scales = _component(
        BLOCK_SCALES,
        dtype="uint8",
        shape=(4, 1),
        layout=_identity_layout(
            "output_features",
            "input_feature_blocks",
            storage_encoding="mxfp8.e8m0-carrier.v1",
        ),
    )
    realized_format = _format(
        source_storage=(values, scales),
        source_storage_format=MXFP8_FORMAT,
        wire_format=MXFP8_FORMAT,
        destination_load_api_format=MXFP8_FORMAT,
        destination_runtime_format=MXFP8_FORMAT,
    )
    base_fields = {
        "kind": _REFIT_PLAN.SourceRegionTransformKind.GATHER_SPLIT,
        "source_binding_set_digest": f"sha256:{'c' * 64}",
        "source_region_digest": f"sha256:{'d' * 64}",
        "selected_source_shapes": ((4, 8), (4, 1)),
        "source_component_ids": ("weight.values", "weight.block-scales"),
        "wire_component_roles": (VALUES, BLOCK_SCALES),
        "wire_component_shapes": ((4, 8), (4, 1)),
        "wire_component_axis_orders": (
            ("output_features", "input_features"),
            ("output_features", "input_feature_blocks"),
        ),
        "wire_representation_digest": physical_representation_digest(
            realized_format,
            PhysicalFormatStage.WIRE,
        ),
    }
    exact = _REFIT_PLAN.SourceRegionExtractionCapability(**base_fields)
    other = _REFIT_PLAN.SourceRegionExtractionCapability(
        **{
            **base_fields,
            "source_region_digest": f"sha256:{'e' * 64}",
        }
    )
    capabilities = tuple(
        AdapterOperationCapability.from_realized_format(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            transform_locus=TransformLocus.SOURCE,
            implementation_id="nemo.gather-split-mxfp8",
            implementation_version="1.0.0",
            source_region_extraction=extraction,
        )
        for extraction in (exact, other)
    )
    registry = _adapter_registry(
        adapter_id="test.refit-adapter",
        adapter_version="1.0.0",
        capabilities=capabilities,
    )
    proof = registry.issue_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        transform_locus=TransformLocus.SOURCE,
        implementation_id="nemo.gather-split-mxfp8",
        implementation_version="1.0.0",
        source_region_extraction=exact,
    )
    morphed = pickle.loads(pickle.dumps(proof))
    object.__setattr__(morphed, "source_region_extraction", other)

    assert exact.wire_component_roles == (VALUES, BLOCK_SCALES)
    with pytest.raises(ValueError, match="signature mismatch"):
        registry._verify(morphed)


def test_destination_evidence_requires_live_registry_identity() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    registry = _registry_for(proof)
    evidence = _TEST_DESTINATION_EVIDENCE[proof.issuer_instance_id]
    authority = _TEST_DESTINATION_AUTHORITIES[proof.issuer_instance_id]
    storage_identity = _TEST_DESTINATION_LIVE_OBJECTS[proof.issuer_instance_id][2]
    storage_lease = authority.begin_storage_generation(storage_identity)
    forged = object.__new__(DestinationBindingEvidence)
    for evidence_field in fields(evidence):
        object.__setattr__(
            forged, evidence_field.name, getattr(evidence, evidence_field.name)
        )

    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(evidence)
    with pytest.raises(TypeError, match="process-local"):
        copy(evidence)
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(storage_lease)
    with pytest.raises(ValueError, match="not live"):
        authority.issue_destination_binding(realized_format, evidence=forged)
    with pytest.raises(TypeError, match="registered destination owner handle"):
        authority.discover_destination_binding(
            realized_format,
            destination_owner=object(),
            finalizer=object(),
            storage_generation=storage_lease,
        )
    owner_handle = authority.register_destination_owner(object())
    finalizer_handle = authority.register_finalizer(object())
    copied_owner_handle = object.__new__(type(owner_handle))
    for handle_field in fields(owner_handle):
        object.__setattr__(
            copied_owner_handle,
            handle_field.name,
            getattr(owner_handle, handle_field.name),
        )
    with pytest.raises(TypeError, match="registered destination owner handle"):
        authority.discover_destination_binding(
            realized_format,
            destination_owner=copied_owner_handle,
            finalizer=finalizer_handle,
            storage_generation=storage_lease,
        )
    destination_proof = authority.issue_destination_binding(
        realized_format,
        evidence=evidence,
    )
    object.__setattr__(
        evidence,
        "destination_owner_instance_id",
        "forged.destination-owner",
    )
    with pytest.raises(ValueError, match="immutable discovery snapshot"):
        authority.issue_destination_binding(realized_format, evidence=evidence)
    with pytest.raises(ValueError, match="immutable discovery snapshot"):
        registry._verify_destination_binding(destination_proof, realized_format)


def test_destination_storage_rollover_rejects_proof_and_installed_operation() -> None:
    realized_format = _format(source_storage=(_component(LOGICAL_VALUES),))
    operation_proof = _proof(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
    )
    registry = _registry_for(operation_proof)
    stale_evidence = _TEST_DESTINATION_EVIDENCE[operation_proof.issuer_instance_id]
    stale_storage_identity = _TEST_DESTINATION_LIVE_OBJECTS[
        operation_proof.issuer_instance_id
    ][2]
    authority = _TEST_DESTINATION_AUTHORITIES[operation_proof.issuer_instance_id]
    stale_storage_lease = authority.begin_storage_generation(stale_storage_identity)
    destination_proof = _destination_binding_proof(registry, realized_format)
    selected = _select_transform(
        realized_format,
        PhysicalFormatStage.SOURCE_STORAGE,
        PhysicalFormatStage.WIRE,
        binding_context=_binding_context(),
        capability_registry=registry,
        destination_binding_proof=destination_proof,
        transform_locus=TransformLocus.NONE,
        direct_copy_proof=operation_proof,
    )
    installed = _install_selected_operation(selected, registry)
    next_generation = authority.begin_storage_generation(object())
    authority.discover_destination_binding(
        realized_format,
        destination_owner=authority.register_destination_owner(object()),
        finalizer=authority.register_finalizer(object()),
        storage_generation=next_generation,
    )

    with pytest.raises(ValueError, match="not live|storage generation is stale"):
        authority.issue_destination_binding(
            realized_format,
            evidence=stale_evidence,
        )
    with pytest.raises(ValueError, match="generation lease is stale"):
        authority.discover_destination_binding(
            realized_format,
            destination_owner=authority.register_destination_owner(object()),
            finalizer=authority.register_finalizer(object()),
            storage_generation=stale_storage_lease,
        )
    with pytest.raises(ValueError, match="storage generation is stale"):
        _select_transform(
            realized_format,
            PhysicalFormatStage.SOURCE_STORAGE,
            PhysicalFormatStage.WIRE,
            binding_context=_binding_context(),
            capability_registry=registry,
            destination_binding_proof=destination_proof,
            transform_locus=TransformLocus.NONE,
            direct_copy_proof=operation_proof,
        )
    with pytest.raises(ValueError, match="storage generation is stale"):
        execution_dispatch_key(installed)


@pytest.mark.parametrize(
    "stage_name",
    (
        "source_storage",
        "wire",
        "destination_load_api",
        "destination_runtime",
    ),
)
def test_realized_format_enforces_atomic_format_component_completeness(
    stage_name: str,
) -> None:
    bf16 = _component(LOGICAL_VALUES)
    mxfp8_values = _component(
        VALUES,
        dtype="uint8",
        layout=replace(
            bf16.representation.layout,
            storage_encoding="mxfp8.e4m3-carrier.v1",
        ),
    )
    mxfp8_scales = _component(
        BLOCK_SCALES,
        dtype="uint8",
        shape=(128, 928, 84),
        layout=_identity_layout(
            "experts",
            "intermediate_features",
            "input_feature_blocks",
            storage_encoding="mxfp8.e8m0-carrier.v1",
        ),
    )
    complete = (mxfp8_values, mxfp8_scales)
    kwargs: dict[str, object] = {
        "source_storage": complete,
        "wire": complete,
        "destination_load_api": complete,
        "destination_runtime": complete,
        "source_storage_format": MXFP8_FORMAT,
        "wire_format": MXFP8_FORMAT,
        "destination_load_api_format": MXFP8_FORMAT,
        "destination_runtime_format": MXFP8_FORMAT,
        "format_schema_registry": _REFIT_PLAN._create_format_schema_registry(
            (MXFP8_FORMAT,)
        ),
        "routes": _routes(),
    }
    kwargs[stage_name] = (mxfp8_values,)

    with pytest.raises(
        ValueError,
        match=rf"{stage_name}.*complete ordered component roles",
    ):
        RealizedBindingFormat(**kwargs)  # type: ignore[arg-type]


def test_atomic_component_completeness_is_format_driven_for_future_encodings() -> None:
    roles = (
        ComponentRole("packed_values"),
        ComponentRole("group_scales"),
        ComponentRole("logical_shape"),
    )
    future_format = FormatDescriptor(
        format_id="adapter.future-quantized.v7",
        family="adapter.future-quantized",
        components=tuple(
            ComponentDescriptor(role=role, dtype="uint8") for role in roles
        ),
    )
    complete = tuple(_component(role, dtype="uint8") for role in roles)
    incomplete = (complete[0], complete[2])

    with pytest.raises(ValueError, match="complete ordered component roles"):
        _format(
            source_storage=complete,
            wire=incomplete,
            source_storage_format=future_format,
            wire_format=future_format,
            destination_load_api_format=future_format,
            destination_runtime_format=future_format,
        )


def test_reserved_mxfp8_schema_cannot_be_redeclared_as_values_only() -> None:
    forged_mxfp8 = FormatDescriptor(
        format_id=MXFP8_FORMAT.format_id,
        family=MXFP8_FORMAT.family,
        components=(MXFP8_FORMAT.components[0],),
    )
    values = _component(
        VALUES,
        dtype="uint8",
        shape=(128, 928, 2688),
        layout=_identity_layout(
            "experts",
            "intermediate_features",
            "input_features",
            storage_encoding="mxfp8.e4m3-carrier.v1",
        ),
    )

    with pytest.raises(ValueError, match="reserved MXFP8 format_id"):
        format_schema_registry = _REFIT_PLAN._create_format_schema_registry(
            (forged_mxfp8,)
        )
        RealizedBindingFormat(
            source_storage=(values,),
            wire=(values,),
            destination_load_api=(values,),
            destination_runtime=(values,),
            source_storage_format=forged_mxfp8,
            wire_format=forged_mxfp8,
            destination_load_api_format=forged_mxfp8,
            destination_runtime_format=forged_mxfp8,
            format_schema_registry=format_schema_registry,
            routes=_routes(),
        )


def test_realized_format_requires_a_trusted_format_schema_registry() -> None:
    future_format = FormatDescriptor(
        format_id="adapter.future-quantized.v7",
        family="adapter.future-quantized",
        components=(ComponentDescriptor(ComponentRole("packed_values"), "uint8"),),
    )
    component = _component(ComponentRole("packed_values"), dtype="uint8")

    with pytest.raises(TypeError, match="format_schema_registry"):
        RealizedBindingFormat(
            source_storage=(component,),
            wire=(component,),
            destination_load_api=(component,),
            destination_runtime=(component,),
            source_storage_format=future_format,
            wire_format=future_format,
            destination_load_api_format=future_format,
            destination_runtime_format=future_format,
            routes=_routes(),
        )
