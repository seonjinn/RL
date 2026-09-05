from collections import UserDict, UserList
from collections.abc import Callable, Iterator, Mapping
from dataclasses import FrozenInstanceError, asdict, dataclass, fields, replace
from enum import StrEnum
import os
from pickle import dumps, loads
import re
import subprocess
import sys
from typing import cast

import pytest

import nemo_rl.precision_policy.source_discovery as source_discovery_module
from nemo_rl.precision_policy.semantic import (
    DecoderLayerUniverse,
    EvidenceSource,
    EvidenceSourceKind,
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
    RolloutParticipation,
    SelectionTopologyEntry,
    SemanticAddressPattern,
    SourceMutability,
    builtin_role_definitions,
    canonical_model_config_digest,
)
from nemo_rl.precision_policy.source_discovery import (
    HF_SAFETENSORS_HEADER_V1,
    MEGATRON_BRIDGE_STATE_DICT_V1,
    NEMO_AUTOMODEL_STATE_DICT_V1,
    TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1,
    DiscoveryCompletenessReceipt,
    DiscoveryContribution,
    DiscoveryRequestKind,
    ExpectedContributorAuthority,
    ExpectedContributorSet,
    GraphDiscoveryPartition,
    GraphTopologyInput,
    RuntimeGraphSourceRequest,
    SourceDiscoveryInventory,
    SourceDiscoveryRecord,
    SourceProducerFingerprint,
    SourceRecordProvenance,
    SourceSchemaId,
    assemble_graph_discovery_partition,
    assemble_runtime_graph_discovery_partition,
    graph_input_identity_digest,
    runtime_source_request_identity_digest,
    validate_discovery_inventory,
    validate_runtime_discovery_inventory,
)
from nemo_rl.precision_policy.source_dtype import CanonicalSourceDType
from nemo_rl.precision_policy.source_storage import (
    IDENTITY_PERMUTATION_ID,
    IDENTITY_SWIZZLE_ID,
    SourceDerivedRealization,
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


EXPECTED_CONTRIBUTOR_AUTHORITY_LOCATOR = (
    "precision-policy.expected-contributor-authority.v1"
)


class _TextSubclass(str):
    pass


class _IntSubclass(int):
    pass


class _ComparisonBypassText(str):
    def __ne__(self, other: object) -> bool:
        return False


class _ComparisonBypassInt(int):
    def __ne__(self, other: object) -> bool:
        return False


def _unregistered_enum_member(
    enum_type: type[StrEnum],
    *,
    underlying_value: str,
    reported_value: str,
) -> StrEnum:
    member = cast(StrEnum, str.__new__(enum_type, underlying_value))
    object.__setattr__(member, "_name_", "UNREGISTERED")
    object.__setattr__(member, "_value_", reported_value)
    return member


@dataclass(frozen=True, slots=True)
class _SelectionTopologyEntrySubclass(SelectionTopologyEntry):
    pass


@dataclass(frozen=True, slots=True)
class _EvidenceSourceSubclass(EvidenceSource):
    hidden_state: str = "hidden"


@dataclass(frozen=True, slots=True)
class _SourceProducerFingerprintSubclass(SourceProducerFingerprint):
    hidden_state: str = "hidden"


@dataclass(frozen=True, slots=True)
class _ExpectedContributorAuthoritySubclass(ExpectedContributorAuthority):
    hidden_state: str = "hidden"


@dataclass(frozen=True, slots=True)
class _ExpectedContributorSetSubclass(ExpectedContributorSet):
    hidden_state: str = "hidden"


@dataclass(frozen=True, slots=True)
class _SpoofedExpectedContributorSet(ExpectedContributorSet):
    spoofed_authority: ExpectedContributorAuthority | None = None

    def to_authority(self) -> ExpectedContributorAuthority:
        assert self.spoofed_authority is not None
        return self.spoofed_authority


@dataclass(frozen=True, slots=True)
class _DiscoveryCompletenessReceiptSubclass(DiscoveryCompletenessReceipt):
    hidden_state: str = "hidden"


@dataclass(frozen=True, slots=True)
class _GraphDiscoveryPartitionSubclass(GraphDiscoveryPartition):
    hidden_state: str = "hidden"


@dataclass(frozen=True, slots=True)
class _SourceDiscoveryInventorySubclass(SourceDiscoveryInventory):
    hidden_state: str = "hidden"


@dataclass(frozen=True, slots=True)
class _SourceDiscoveryRecordSubclass(SourceDiscoveryRecord):
    hidden_state: str = "hidden"


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _identity_contract(character: str = "1") -> SourceNormalizationContract:
    return SourceNormalizationContract(
        capability_id="test.identity.v1",
        kind=SourceNormalizationKind.IDENTITY,
        contract_digest=_digest(character),
    )


def _identity_manifest(character: str = "1") -> SourceNormalizerManifest:
    return SourceNormalizerManifest(
        schema_version=1,
        contracts=(_identity_contract(character),),
    )


_NORMALIZER_MANIFESTS_BY_DIGEST: dict[str, SourceNormalizerManifest] = {}


def _identity_storage_for_record(
    record: SourceDiscoveryRecord,
    *,
    manifest: SourceNormalizerManifest | None = None,
) -> SourceStorageRealizationInventory:
    normalizers = manifest or _identity_manifest()
    contract = normalizers.contracts[0]
    if record.source_mutability == SourceMutability.ABSENT:
        realizations: tuple[SourceStorageRealization, ...] = ()
    else:
        assert record.source_native_name is not None
        realizations = (
            SourceStorageRealization(
                realization_id=f"{record.record_id}.identity",
                graph_instance_id=record.graph_instance_id,
                output_record_id=record.record_id,
                components=(
                    SourceStorageComponent(
                        graph_instance_id=record.graph_instance_id,
                        native_component_id=f"{record.record_id}.component",
                        source_native_name=record.source_native_name,
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
                normalization=contract,
            ),
        )
    return SourceStorageRealizationInventory(
        graph_instance_id=record.graph_instance_id,
        normalizer_manifest=normalizers,
        realizations=realizations,
    )


def _identity_storage_for_records(
    graph_instance_id: str,
    records: tuple[SourceDiscoveryRecord, ...],
    manifest: SourceNormalizerManifest,
) -> SourceStorageRealizationInventory:
    return SourceStorageRealizationInventory(
        graph_instance_id=graph_instance_id,
        normalizer_manifest=manifest,
        realizations=tuple(
            realization
            for record in records
            for realization in _identity_storage_for_record(
                record,
                manifest=manifest,
            ).realizations
        ),
    )


def _evidence(
    name: str,
    character: str,
    *,
    kind: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_INVENTORY,
) -> EvidenceSource:
    return EvidenceSource(
        kind=kind,
        locator=f"runtime://{name}",
        digest=_digest(character),
    )


def _fingerprint(
    *,
    schema_id: SourceSchemaId = HF_SAFETENSORS_HEADER_V1,
    implementation_id: str = "checkpoint-header-reader",
    revision: str = "a" * 40,
    character: str = "1",
) -> SourceProducerFingerprint:
    manifest = _identity_manifest(character)
    normalization_contract_digest = source_normalizer_manifest_digest(manifest)
    _NORMALIZER_MANIFESTS_BY_DIGEST[normalization_contract_digest] = manifest
    return SourceProducerFingerprint(
        schema_id=schema_id,
        producer_implementation_id=implementation_id,
        producer_revision=revision,
        normalization_contract_digest=normalization_contract_digest,
        evidence=_evidence(f"producer-{character}", character),
    )


def _expected(
    contributor_ids: tuple[str, ...] = ("checkpoint-index",),
    *,
    character: str = "2",
) -> ExpectedContributorSet:
    return ExpectedContributorSet(
        contributor_ids=contributor_ids,
        authority=_evidence("trusted-membership", character),
    )


def _declaration(graph_instance_id: str = "main") -> ExpectedGraphDeclaration:
    graph_kind = (
        GraphKind.MAIN if graph_instance_id == "main" else GraphKind.SPECULATIVE_DRAFTER
    )
    return ExpectedGraphDeclaration(
        graph_instance_id=graph_instance_id,
        model_identity=f"test/{graph_instance_id}",
        lifecycle=GraphLifecycle(
            graph_kind=graph_kind,
            graph_provenance=GraphProvenance.TRAINING_RUNTIME,
            rollout_participation=RolloutParticipation.SERVED_FROM_SOURCE,
        ),
    )


def _graph_input(
    graph_instance_id: str = "main",
    *,
    fingerprint: SourceProducerFingerprint | None = None,
    expected: ExpectedContributorSet | None = None,
    config: Mapping[str, object] | None = None,
    revision: str = "b" * 40,
    source_character: str = "3",
    artifact_character: str = "4",
) -> GraphTopologyInput:
    trusted = expected or _expected()
    return GraphTopologyInput(
        declaration=_declaration(graph_instance_id),
        model_config=config or {"model_type": "test", "layers": [0, 1]},
        resolved_model_revision=revision,
        source_producer_fingerprint=fingerprint or _fingerprint(),
        expected_contributor_authority=trusted.to_authority(),
        source_identity=_evidence("source-identity", source_character),
        artifact_identity=_evidence("artifact-identity", artifact_character),
    )


def _resolved_graph(
    graph_instance_id: str = "main",
    *,
    revision: str = "b" * 40,
    adapter_id: str = "test.adapter.v1",
    logical_shape: tuple[int, ...] = (8, 8),
) -> ResolvedGraphTopology:
    declaration = _declaration(graph_instance_id)
    is_main = graph_instance_id == "main"
    entry = SelectionTopologyEntry(
        entry_id=f"{graph_instance_id}.dense.weight",
        graph_instance_id=graph_instance_id,
        pattern=SemanticAddressPattern(
            semantic_graph_path="text.decoder" if is_main else "draft.decoder",
            path_segments=(
                LiteralPathSegment("layer"),
                IndexPathSegment("global_decoder_layer"),
                LiteralPathSegment("weight"),
            ),
            model_part="main" if is_main else "draft",
            module_kind="ffn.dense",
            attributes=(),
            parameter_role="kernel",
        ),
        domain=FamilyIndexDomain(
            layer_domain=LayerDomain((LayerMember(0, None),)),
            independent_axes=(),
        ),
        logical_dtype="bfloat16",
        logical_shape=logical_shape,
        logical_axes=("output_features", "input_features"),
    )
    return ResolvedGraphTopology(
        declaration=declaration,
        model_family="test_family",
        resolved_model_revision=revision,
        adapter_id=adapter_id,
        decoder_layer_universe=DecoderLayerUniverse((0,), ()),
        entries=(entry,),
        role_definitions=builtin_role_definitions(1, {}),
        atomic_groups=(),
    )


def _runtime_request(
    graph_instance_id: str = "main",
    *,
    resolved_graph: ResolvedGraphTopology | None = None,
    fingerprint: SourceProducerFingerprint | None = None,
    expected: ExpectedContributorSet | None = None,
    config: Mapping[str, object] | None = None,
    revision: str = "b" * 40,
    semantic_character: str = "a",
    selection_character: str = "b",
    source_character: str = "3",
    artifact_character: str = "4",
    allocation_generation: str = "allocation-1",
) -> RuntimeGraphSourceRequest:
    trusted = expected or _expected()
    graph = resolved_graph or _resolved_graph(
        graph_instance_id,
        revision=revision,
    )
    return RuntimeGraphSourceRequest(
        declaration=graph.declaration,
        resolved_graph=graph,
        semantic_structure_digest=_digest(semantic_character),
        selection_group_id=_digest(selection_character),
        model_config=config or {"model_type": "test", "layers": [0, 1]},
        resolved_model_revision=revision,
        source_producer_fingerprint=fingerprint or _fingerprint(),
        expected_contributor_authority=trusted.to_authority(),
        source_identity=_evidence("source-identity", source_character),
        artifact_identity=_evidence("artifact-identity", artifact_character),
        source_allocation_generation=allocation_generation,
    )


def _record(
    record_id: str = "main.weight",
    *,
    graph_instance_id: str = "main",
    native_name: str | None = "model.weight",
    native_owner: str | None = "model.weight",
    source_mutability: SourceMutability = SourceMutability.MUTABLE,
    numeric_encoding: str = "plain_bfloat16",
) -> SourceDiscoveryRecord:
    return SourceDiscoveryRecord(
        record_id=record_id,
        graph_instance_id=graph_instance_id,
        source_native_name=native_name,
        source_native_owner_id=native_owner,
        dtype=CanonicalSourceDType.BFLOAT16,
        shape=(8, 8),
        numeric_encoding=numeric_encoding,
        provenance=SourceRecordProvenance.TRAINING_RUNTIME,
        provenance_evidence=_evidence(f"{record_id}-provenance", "5"),
        source_mutability=source_mutability,
        mutability_evidence=_evidence(f"{record_id}-mutability", "6"),
    )


def _contribution(
    contributor_id: str,
    records: tuple[SourceDiscoveryRecord, ...],
    *,
    graph_instance_id: str = "main",
    fingerprint: SourceProducerFingerprint | None = None,
) -> DiscoveryContribution:
    producer = fingerprint or _fingerprint()
    manifest = _NORMALIZER_MANIFESTS_BY_DIGEST[producer.normalization_contract_digest]
    return DiscoveryContribution(
        contributor_id=contributor_id,
        graph_instance_id=graph_instance_id,
        producer_fingerprint=producer,
        records=records,
        storage_realizations=_identity_storage_for_records(
            graph_instance_id,
            records,
            manifest,
        ),
    )


def _complete_pair(
    graph_instance_id: str = "main",
    *,
    fingerprint: SourceProducerFingerprint | None = None,
    expected: ExpectedContributorSet | None = None,
) -> tuple[GraphTopologyInput, ExpectedContributorSet, GraphDiscoveryPartition]:
    trusted = expected or _expected()
    producer = fingerprint or _fingerprint()
    graph_input = _graph_input(
        graph_instance_id,
        fingerprint=producer,
        expected=trusted,
    )
    record = _record(
        f"{graph_instance_id}.weight",
        graph_instance_id=graph_instance_id,
        native_name=f"{graph_instance_id}.model.weight",
        native_owner=f"{graph_instance_id}.model.weight",
    )
    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=trusted,
        contributions=(
            _contribution(
                trusted.contributor_ids[0],
                (record,),
                graph_instance_id=graph_instance_id,
                fingerprint=producer,
            ),
        ),
    )
    return graph_input, trusted, partition


def _complete_runtime_pair(
    graph_instance_id: str = "main",
    *,
    fingerprint: SourceProducerFingerprint | None = None,
    expected: ExpectedContributorSet | None = None,
    semantic_character: str = "a",
    selection_character: str = "b",
) -> tuple[
    RuntimeGraphSourceRequest,
    ExpectedContributorSet,
    GraphDiscoveryPartition,
]:
    trusted = expected or _expected()
    producer = fingerprint or _fingerprint()
    runtime_request = _runtime_request(
        graph_instance_id,
        fingerprint=producer,
        expected=trusted,
        semantic_character=semantic_character,
        selection_character=selection_character,
    )
    record = _record(
        f"{graph_instance_id}.weight",
        graph_instance_id=graph_instance_id,
        native_name=f"{graph_instance_id}.model.weight",
        native_owner=f"{graph_instance_id}.model.weight",
    )
    partition = assemble_runtime_graph_discovery_partition(
        runtime_request=runtime_request,
        expected_contributors=trusted,
        contributions=(
            _contribution(
                trusted.contributor_ids[0],
                (record,),
                graph_instance_id=graph_instance_id,
                fingerprint=producer,
            ),
        ),
    )
    return runtime_request, trusted, partition


@pytest.mark.parametrize(
    "value",
    [
        "hf.safetensors.header.v1",
        "megatron.bridge.state-dict.v1",
        "nemo-automodel.state-dict.v1",
        "transformer-engine.quantized-storage.v1",
        "a.b.v12",
    ],
)
def test_source_schema_id_accepts_only_exact_namespaced_versioned_atoms(
    value: str,
) -> None:
    assert SourceSchemaId(value).value == value


@pytest.mark.parametrize(
    "value",
    [
        "hf.v1",
        "HF.safetensors.header.v1",
        " hf.safetensors.header.v1",
        "hf.safetensors.header.v1 ",
        "hf.safetensors.header.v0",
        "hf.safetensors.header.v01",
        "hf.safetensors.header.1",
        "hf..header.v1",
        "hf.header_thing.v1",
        "1hf.header.v1",
        "",
    ],
)
def test_source_schema_id_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError, match="source schema"):
        SourceSchemaId(value)


def test_initial_source_schema_constants_are_exact() -> None:
    assert (
        HF_SAFETENSORS_HEADER_V1.value,
        MEGATRON_BRIDGE_STATE_DICT_V1.value,
        NEMO_AUTOMODEL_STATE_DICT_V1.value,
        TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1.value,
    ) == (
        "hf.safetensors.header.v1",
        "megatron.bridge.state-dict.v1",
        "nemo-automodel.state-dict.v1",
        "transformer-engine.quantized-storage.v1",
    )


@pytest.mark.parametrize("revision", ["main", "latest", "v1.2.3", "refs/heads/x"])
def test_producer_fingerprint_requires_an_immutable_revision(revision: str) -> None:
    with pytest.raises(ValueError, match="immutable"):
        _fingerprint(revision=revision)


def test_producer_fingerprint_accepts_non_git_content_identity() -> None:
    fingerprint = _fingerprint(revision=_digest("a"))

    assert fingerprint.producer_revision == _digest("a")


@pytest.mark.parametrize(
    "revision",
    [
        "a" * 39,
        "A" * 40,
        "a" * 65,
        "sha256:" + "a" * 63,
        "sha256:" + "A" * 64,
        "sha256:" + "g" * 64,
        " sha256:" + "a" * 64,
        "sha256:" + "a" * 64 + " ",
    ],
)
def test_producer_fingerprint_rejects_malformed_content_identity(
    revision: str,
) -> None:
    with pytest.raises(ValueError, match="immutable"):
        _fingerprint(revision=revision)


@pytest.mark.parametrize("revision", [None, True, 40, b"a" * 40])
def test_producer_fingerprint_rejects_non_string_content_identity(
    revision: object,
) -> None:
    with pytest.raises(TypeError, match="revision.*string"):
        _fingerprint(revision=revision)  # type: ignore[arg-type]


@pytest.mark.parametrize("implementation_id", ["", " producer", "producer "])
def test_producer_fingerprint_rejects_malformed_implementation_id(
    implementation_id: str,
) -> None:
    with pytest.raises(ValueError, match="implementation ID"):
        _fingerprint(implementation_id=implementation_id)


@pytest.mark.parametrize("implementation_id", [None, True, 7, b"producer"])
def test_producer_fingerprint_rejects_non_string_implementation_id(
    implementation_id: object,
) -> None:
    with pytest.raises(TypeError, match="implementation ID.*string"):
        _fingerprint(implementation_id=implementation_id)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "normalization_digest",
    [
        "a" * 64,
        "sha256:" + "a" * 63,
        "sha256:" + "A" * 64,
        "sha256:" + "g" * 64,
        " sha256:" + "a" * 64,
        "sha256:" + "a" * 64 + " ",
    ],
)
def test_producer_fingerprint_rejects_malformed_normalization_digest(
    normalization_digest: str,
) -> None:
    with pytest.raises(ValueError, match="normalization contract digest"):
        SourceProducerFingerprint(
            schema_id=HF_SAFETENSORS_HEADER_V1,
            producer_implementation_id="checkpoint-header-reader",
            producer_revision="a" * 40,
            normalization_contract_digest=normalization_digest,
            evidence=_evidence("producer", "1"),
        )


@pytest.mark.parametrize("normalization_digest", [None, True, 7, b"digest"])
def test_producer_fingerprint_rejects_non_string_normalization_digest(
    normalization_digest: object,
) -> None:
    with pytest.raises(TypeError, match="normalization contract digest.*string"):
        SourceProducerFingerprint(
            schema_id=HF_SAFETENSORS_HEADER_V1,
            producer_implementation_id="checkpoint-header-reader",
            producer_revision="a" * 40,
            normalization_contract_digest=normalization_digest,  # type: ignore[arg-type]
            evidence=_evidence("producer", "1"),
        )


def test_producer_fingerprint_requires_typed_schema_and_evidence() -> None:
    with pytest.raises(TypeError, match="SourceSchemaId"):
        _fingerprint(schema_id="hf.safetensors.header.v1")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="EvidenceSource"):
        SourceProducerFingerprint(
            schema_id=HF_SAFETENSORS_HEADER_V1,
            producer_implementation_id="checkpoint-header-reader",
            producer_revision="a" * 40,
            normalization_contract_digest=_digest("1"),
            evidence={"kind": "runtime"},  # type: ignore[arg-type]
        )


def test_expected_contributor_authority_is_canonical_and_id_free() -> None:
    contributor_ids = ["shard-b", "shard-a"]
    trusted = ExpectedContributorSet(
        contributor_ids=contributor_ids,  # type: ignore[arg-type]
        authority=_evidence("trusted-membership", "2"),
    )
    reverse = ExpectedContributorSet(
        contributor_ids=("shard-a", "shard-b"),
        authority=_evidence("trusted-membership", "2"),
    )
    contributor_ids.append("mutated")

    assert trusted.contributor_ids == ("shard-a", "shard-b")
    assert trusted.to_authority() == reverse.to_authority()
    authority_payload = asdict(trusted.to_authority())
    assert authority_payload["contributor_count"] == 2
    assert "shard-a" not in repr(authority_payload)
    assert "shard-b" not in repr(authority_payload)


def test_exact_expected_contributor_authority_derivation_fails_closed() -> None:
    expected = _expected(("rank-b", "rank-a"))

    authority = source_discovery_module.derive_expected_contributor_authority(expected)

    assert authority == expected.to_authority()
    malformed = loads(dumps(expected))
    object.__setattr__(malformed, "contributor_ids", ("rank-b", "rank-a"))
    with pytest.raises(ValueError, match="canonical order"):
        source_discovery_module.derive_expected_contributor_authority(malformed)


@pytest.mark.parametrize(
    "contributor_ids",
    ["0", b"0", bytearray(b"0"), memoryview(b"0")],
    ids=("str", "bytes", "bytearray", "memoryview"),
)
def test_expected_contributor_ids_reject_scalar_or_buffer_outer_values(
    contributor_ids: object,
) -> None:
    with pytest.raises(TypeError, match="contributor IDs.*sequence"):
        ExpectedContributorSet(
            contributor_ids=contributor_ids,  # type: ignore[arg-type]
            authority=_evidence("trusted-membership", "2"),
        )


def test_expected_contributor_ids_reject_unsupported_generator() -> None:
    contributor_ids = (item for item in ("rank-a", "rank-b"))

    with pytest.raises(TypeError, match="contributor IDs.*sequence"):
        ExpectedContributorSet(
            contributor_ids=contributor_ids,  # type: ignore[arg-type]
            authority=_evidence("trusted-membership", "2"),
        )


@pytest.mark.parametrize(
    "contributor_ids",
    [
        ("rank-b", "rank-a"),
        ["rank-b", "rank-a"],
        UserList(["rank-b", "rank-a"]),
    ],
    ids=("tuple", "list", "sequence"),
)
def test_expected_contributor_ids_snapshot_supported_sequences(
    contributor_ids: object,
) -> None:
    trusted = ExpectedContributorSet(
        contributor_ids=contributor_ids,  # type: ignore[arg-type]
        authority=_evidence("trusted-membership", "2"),
    )

    if isinstance(contributor_ids, (list, UserList)):
        contributor_ids.append("mutated")
    assert trusted.contributor_ids == ("rank-a", "rank-b")


@pytest.mark.parametrize(
    "contributor_ids",
    [(), ("shard-a", "shard-a"), ("",), (" shard-a",)],
)
def test_expected_contributor_set_rejects_invalid_opaque_ids(
    contributor_ids: tuple[str, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _expected(contributor_ids)


def test_structural_authority_hides_trusted_evidence_and_placement_ids() -> None:
    trusted = ExpectedContributorSet(
        contributor_ids=("0", "private-pp7-tp3-ep2"),
        authority=EvidenceSource(
            kind=EvidenceSourceKind.RUNTIME_INVENTORY,
            locator="runtime://membership/private-pp7-tp3-ep2",
            digest="membership-proof:pp7/tp3/ep2/0000",
        ),
    )
    authority = trusted.to_authority()
    payload = repr(asdict(authority))

    assert authority.authority.kind == EvidenceSourceKind.CONTENT_ADDRESS
    assert authority.authority.locator == EXPECTED_CONTRIBUTOR_AUTHORITY_LOCATOR
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", authority.authority.digest)
    for private_value in (
        "private-pp7-tp3-ep2",
        "runtime://membership/private-pp7-tp3-ep2",
        "membership-proof:pp7/tp3/ep2/0000",
        "pp7",
        "tp3",
        "ep2",
    ):
        assert private_value not in payload


def test_single_character_contributor_id_does_not_scan_digest_substrings() -> None:
    trusted = ExpectedContributorSet(
        contributor_ids=("0",),
        authority=_evidence("membership", "0"),
    )

    authority = trusted.to_authority()

    assert authority.contributor_count == 1
    assert authority.authority.locator == EXPECTED_CONTRIBUTOR_AUTHORITY_LOCATOR


def test_original_authority_evidence_fields_change_opaque_commitment() -> None:
    original = _evidence("trusted-membership", "2")
    variants = (
        original,
        replace(original, kind=EvidenceSourceKind.PINNED_CHECKPOINT_MANIFEST),
        replace(original, locator="runtime://different-membership"),
        replace(original, digest=_digest("7")),
    )
    commitments = tuple(
        ExpectedContributorSet(("rank-a",), evidence).to_authority().authority.digest
        for evidence in variants
    )

    assert len(set(commitments)) == len(variants)


@pytest.mark.parametrize(
    "authority",
    [
        EvidenceSource(
            kind=EvidenceSourceKind.RUNTIME_INVENTORY,
            locator=EXPECTED_CONTRIBUTOR_AUTHORITY_LOCATOR,
            digest=_digest("2"),
        ),
        EvidenceSource(
            kind=EvidenceSourceKind.CONTENT_ADDRESS,
            locator=f"{EXPECTED_CONTRIBUTOR_AUTHORITY_LOCATOR}.pp7-tp3-ep2",
            digest=_digest("2"),
        ),
        EvidenceSource(
            kind=EvidenceSourceKind.CONTENT_ADDRESS,
            locator=EXPECTED_CONTRIBUTOR_AUTHORITY_LOCATOR,
            digest="not-a-canonical-digest",
        ),
    ],
    ids=("wrong-kind", "coordinate-bearing-locator", "noncanonical-digest"),
)
def test_direct_expected_authority_requires_structural_commitment(
    authority: EvidenceSource,
) -> None:
    with pytest.raises(ValueError, match="contributor authority"):
        ExpectedContributorAuthority(
            contributor_set_digest=_digest("1"),
            contributor_count=1,
            authority=authority,
        )


def test_graph_input_snapshot_and_digest_are_canonical_and_serializable() -> None:
    config = {"z": None, "a": [True, 7, 2.5, {"b": "value"}]}
    graph_input = _graph_input(config=config)
    reordered = _graph_input(config={"a": (True, 7, 2.5, {"b": "value"}), "z": None})
    config["a"].append("mutated")  # type: ignore[union-attr]

    assert tuple(graph_input.model_config) == ("a", "z")
    assert graph_input.model_config["a"] == (
        True,
        7,
        2.5,
        {"b": "value"},
    )
    assert graph_input_identity_digest(graph_input) == graph_input_identity_digest(
        reordered
    )
    assert loads(dumps(graph_input)) == graph_input


@pytest.mark.parametrize(
    "mutation",
    [
        "config",
        "revision",
        "source_identity",
        "artifact_identity",
        "fingerprint",
        "authority",
        "graph",
    ],
)
def test_graph_input_digest_binds_every_discovery_identity(mutation: str) -> None:
    graph_input = _graph_input()
    if mutation == "config":
        changed = _graph_input(config={"model_type": "changed"})
    elif mutation == "revision":
        changed = _graph_input(revision="c" * 40)
    elif mutation == "source_identity":
        changed = _graph_input(source_character="7")
    elif mutation == "artifact_identity":
        changed = _graph_input(artifact_character="8")
    elif mutation == "fingerprint":
        changed = _graph_input(fingerprint=_fingerprint(character="9"))
    elif mutation == "authority":
        changed = _graph_input(expected=_expected(("other-shard",)))
    else:
        changed = _graph_input("draft.external")

    assert graph_input_identity_digest(changed) != graph_input_identity_digest(
        graph_input
    )


def test_runtime_source_request_snapshot_and_digest_are_canonical_and_serializable() -> (
    None
):
    config = {"z": None, "a": [True, 7, 2.5, {"b": "value"}]}
    request = _runtime_request(config=config)
    reordered = _runtime_request(
        config={"a": (True, 7, 2.5, {"b": "value"}), "z": None}
    )
    config["a"].append("mutated")  # type: ignore[union-attr]

    assert tuple(request.model_config) == ("a", "z")
    assert request.model_config["a"] == (True, 7, 2.5, {"b": "value"})
    assert request.runtime_source_request_digest == (
        runtime_source_request_identity_digest(request)
    )
    assert request.runtime_source_request_digest == (
        reordered.runtime_source_request_digest
    )
    assert loads(dumps(request)) == request


@pytest.mark.parametrize(
    "mutation",
    [
        "resolved_graph",
        "semantic_structure_digest",
        "selection_group_id",
        "config",
        "revision",
        "source_identity",
        "artifact_identity",
        "fingerprint",
        "authority",
        "allocation_generation",
    ],
)
def test_runtime_source_request_digest_binds_every_runtime_and_phase_one_identity(
    mutation: str,
) -> None:
    request = _runtime_request()
    if mutation == "resolved_graph":
        changed = _runtime_request(
            resolved_graph=_resolved_graph(adapter_id="test.adapter.v2")
        )
    elif mutation == "semantic_structure_digest":
        changed = _runtime_request(semantic_character="c")
    elif mutation == "selection_group_id":
        changed = _runtime_request(selection_character="d")
    elif mutation == "config":
        changed = _runtime_request(config={"model_type": "changed"})
    elif mutation == "revision":
        changed = _runtime_request(revision="c" * 40)
    elif mutation == "source_identity":
        changed = _runtime_request(source_character="7")
    elif mutation == "artifact_identity":
        changed = _runtime_request(artifact_character="8")
    elif mutation == "fingerprint":
        changed = _runtime_request(fingerprint=_fingerprint(character="9"))
    elif mutation == "authority":
        changed = _runtime_request(expected=_expected(("other-shard",)))
    else:
        changed = _runtime_request(allocation_generation="allocation-2")

    assert changed.runtime_source_request_digest != (
        request.runtime_source_request_digest
    )


@pytest.mark.parametrize("mismatch", ["declaration", "revision"])
def test_runtime_source_request_rejects_resolved_graph_identity_mismatch(
    mismatch: str,
) -> None:
    request = _runtime_request()
    kwargs = {
        item.name: getattr(request, item.name)
        for item in fields(RuntimeGraphSourceRequest)
        if item.init
    }
    if mismatch == "declaration":
        kwargs["declaration"] = _declaration("draft.external")
    else:
        kwargs["resolved_model_revision"] = "c" * 40

    with pytest.raises(ValueError, match=mismatch):
        RuntimeGraphSourceRequest(**kwargs)


def test_runtime_source_request_digest_is_derived_not_caller_supplied() -> None:
    request = _runtime_request()
    kwargs = {
        item.name: getattr(request, item.name)
        for item in fields(RuntimeGraphSourceRequest)
        if item.init
    }
    kwargs["runtime_source_request_digest"] = _digest("f")

    with pytest.raises(TypeError, match="runtime_source_request_digest"):
        RuntimeGraphSourceRequest(**kwargs)

    digest_field = next(
        item
        for item in fields(RuntimeGraphSourceRequest)
        if item.name == "runtime_source_request_digest"
    )
    assert digest_field.init is False


@pytest.mark.parametrize("mutation", ["record_subclass", "scalar_subclass"])
def test_runtime_source_request_rejects_non_exact_resolved_graph_tree(
    mutation: str,
) -> None:
    graph = _resolved_graph()
    entry = graph.entries[0]
    if mutation == "record_subclass":
        entry_kwargs = {
            item.name: getattr(entry, item.name)
            for item in fields(SelectionTopologyEntry)
            if item.init
        }
        changed_entry = _SelectionTopologyEntrySubclass(**entry_kwargs)
    else:
        changed_entry = replace(
            entry,
            logical_dtype=_TextSubclass(entry.logical_dtype),
        )
    changed_graph = replace(graph, entries=(changed_entry,))

    with pytest.raises(TypeError, match="non-exact source-neutral"):
        _runtime_request(resolved_graph=changed_graph)


def test_runtime_source_request_rejects_unregistered_exact_enum_member() -> None:
    graph = loads(dumps(_resolved_graph()))
    forged_graph_kind = cast(
        GraphKind,
        _unregistered_enum_member(
            GraphKind,
            underlying_value=GraphKind.MAIN.value,
            reported_value=GraphKind.MTP.value,
        ),
    )
    object.__setattr__(graph.declaration.lifecycle, "graph_kind", forged_graph_kind)

    with pytest.raises(TypeError, match="registered GraphKind"):
        _runtime_request(resolved_graph=graph)


def test_runtime_assembly_rejects_unregistered_dtype_with_bf16_storage() -> None:
    expected = _expected()
    runtime_request = _runtime_request(expected=expected)
    contribution = loads(
        dumps(_contribution(expected.contributor_ids[0], (_record(),)))
    )
    forged_dtype = cast(
        CanonicalSourceDType,
        _unregistered_enum_member(
            CanonicalSourceDType,
            underlying_value=CanonicalSourceDType.BFLOAT16.value,
            reported_value=CanonicalSourceDType.E4M3.value,
        ),
    )
    object.__setattr__(contribution.records[0], "dtype", forged_dtype)

    with pytest.raises(TypeError, match="registered CanonicalSourceDType"):
        assemble_runtime_graph_discovery_partition(
            runtime_request=runtime_request,
            expected_contributors=expected,
            contributions=(contribution,),
        )


def test_runtime_inventory_rejects_poisoned_registered_enum_value_type() -> None:
    code = """
from tests.unit.precision_policy.test_source_discovery import _complete_runtime_pair
from nemo_rl.precision_policy.source_discovery import (
    SourceDiscoveryInventory,
    validate_runtime_discovery_inventory,
)
from nemo_rl.precision_policy.source_dtype import CanonicalSourceDType

class TextSubclass(str):
    pass

request, expected, partition = _complete_runtime_pair()
object.__setattr__(
    CanonicalSourceDType.BFLOAT16,
    '_value_',
    TextSubclass(CanonicalSourceDType.BFLOAT16.value),
)
try:
    validate_runtime_discovery_inventory(
        (request,), SourceDiscoveryInventory((partition,)), {'main': expected}
    )
except TypeError as error:
    if 'exact canonical' in str(error) and 'string value' in str(error):
        raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "config",
    [
        {"value": _TextSubclass("text")},
        {"value": _IntSubclass(7)},
        {_TextSubclass("value"): 7},
    ],
    ids=("str-subclass", "int-subclass", "key-subclass"),
)
def test_runtime_source_request_rejects_scalar_subclasses_in_model_config(
    config: Mapping[str, object],
) -> None:
    with pytest.raises(TypeError, match="exact JSON scalar|keys"):
        _runtime_request(config=config)


def test_runtime_source_request_rejects_cyclic_model_config() -> None:
    config: dict[str, object] = {}
    config["cycle"] = config

    with pytest.raises(ValueError, match="cycles"):
        _runtime_request(config=config)


@pytest.mark.parametrize(
    ("mutation", "error_type", "match"),
    (
        ("source-identity", ValueError, "non-empty"),
        ("semantic-digest", ValueError, "canonical SHA-256"),
        ("config-cycle", ValueError, "cycle"),
        ("stored-digest-subclass", TypeError, "exact string"),
    ),
)
def test_runtime_source_request_public_digest_revalidates_frozen_snapshot(
    mutation: str,
    error_type: type[Exception],
    match: str,
) -> None:
    request = loads(dumps(_runtime_request()))
    if mutation == "source-identity":
        object.__setattr__(request.source_identity, "locator", "")
    elif mutation == "semantic-digest":
        object.__setattr__(request, "semantic_structure_digest", "bogus")
    elif mutation == "config-cycle":
        object.__setattr__(
            request.model_config,
            "entries",
            (("cycle", request.model_config),),
        )
    else:
        object.__setattr__(
            request,
            "runtime_source_request_digest",
            _ComparisonBypassText(request.runtime_source_request_digest),
        )

    with pytest.raises(error_type, match=match):
        runtime_source_request_identity_digest(request)


def test_runtime_source_request_digest_streams_canonical_config_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _runtime_request(
        config={f"key-{index:05d}": index for index in range(10_000)}
    )
    frozen_mapping_type = type(request.model_config)

    def fail_quadratic_lookup(_self: object, _key: str) -> object:
        raise AssertionError("canonical frozen config must not perform key lookup")

    monkeypatch.setattr(frozen_mapping_type, "__getitem__", fail_quadratic_lookup)

    assert runtime_source_request_identity_digest(request) == (
        request.runtime_source_request_digest
    )


def test_runtime_request_config_snapshot_streams_shared_semantic_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _runtime_request(
        config={f"key-{index:05d}": index for index in range(10_000)}
    )
    frozen_mapping_type = type(request.model_config)

    def fail_quadratic_lookup(_self: object, _key: str) -> object:
        raise AssertionError("semantic config digest must stream frozen entries")

    monkeypatch.setattr(frozen_mapping_type, "__getitem__", fail_quadratic_lookup)

    assert canonical_model_config_digest(request.model_config).startswith("sha256:")


def test_runtime_source_request_equality_streams_canonical_config_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _runtime_request(
        config={f"key-{index:05d}": index for index in range(10_000)}
    )
    roundtrip = loads(dumps(request))
    frozen_mapping_type = type(request.model_config)

    def fail_quadratic_lookup(_self: object, _key: str) -> object:
        raise AssertionError("canonical frozen config must not perform key lookup")

    monkeypatch.setattr(frozen_mapping_type, "__getitem__", fail_quadratic_lookup)

    assert request == roundtrip


@pytest.mark.parametrize(
    "mutation",
    [
        "fingerprint_subclass",
        "fingerprint_nested_evidence_subclass",
        "authority_subclass",
        "authority_nested_evidence_subclass",
        "source_evidence_subclass",
        "artifact_evidence_subclass",
        "digest_string_subclass",
        "revision_string_subclass",
        "allocation_string_subclass",
    ],
)
def test_runtime_source_request_rejects_identity_record_or_scalar_subclasses(
    mutation: str,
) -> None:
    request = _runtime_request()
    kwargs = {
        item.name: getattr(request, item.name)
        for item in fields(RuntimeGraphSourceRequest)
        if item.init
    }
    fingerprint = request.source_producer_fingerprint
    authority = request.expected_contributor_authority
    if mutation == "fingerprint_subclass":
        kwargs["source_producer_fingerprint"] = _SourceProducerFingerprintSubclass(
            **{
                item.name: getattr(fingerprint, item.name)
                for item in fields(SourceProducerFingerprint)
                if item.init
            }
        )
    elif mutation == "fingerprint_nested_evidence_subclass":
        kwargs["source_producer_fingerprint"] = replace(
            fingerprint,
            evidence=_EvidenceSourceSubclass(
                fingerprint.evidence.kind,
                fingerprint.evidence.locator,
                fingerprint.evidence.digest,
            ),
        )
    elif mutation == "authority_subclass":
        kwargs["expected_contributor_authority"] = (
            _ExpectedContributorAuthoritySubclass(
                **{
                    item.name: getattr(authority, item.name)
                    for item in fields(ExpectedContributorAuthority)
                    if item.init
                }
            )
        )
    elif mutation == "authority_nested_evidence_subclass":
        kwargs["expected_contributor_authority"] = replace(
            authority,
            authority=_EvidenceSourceSubclass(
                authority.authority.kind,
                authority.authority.locator,
                authority.authority.digest,
            ),
        )
    elif mutation == "source_evidence_subclass":
        source = request.source_identity
        kwargs["source_identity"] = _EvidenceSourceSubclass(
            source.kind,
            source.locator,
            source.digest,
        )
    elif mutation == "artifact_evidence_subclass":
        artifact = request.artifact_identity
        kwargs["artifact_identity"] = _EvidenceSourceSubclass(
            artifact.kind,
            artifact.locator,
            artifact.digest,
        )
    elif mutation == "digest_string_subclass":
        kwargs["semantic_structure_digest"] = _TextSubclass(
            request.semantic_structure_digest
        )
    elif mutation == "revision_string_subclass":
        kwargs["resolved_model_revision"] = _TextSubclass(
            request.resolved_model_revision
        )
    else:
        kwargs["source_allocation_generation"] = _TextSubclass(
            request.source_allocation_generation
        )

    with pytest.raises(TypeError, match="exact"):
        RuntimeGraphSourceRequest(**kwargs)


def test_runtime_partition_receipt_binds_phase_one_request_and_returns_inventory() -> (
    None
):
    runtime_request, expected, partition = _complete_runtime_pair()
    inventory = SourceDiscoveryInventory((partition,))

    assert partition.completeness_receipt.request_kind == (
        DiscoveryRequestKind.RUNTIME_GRAPH_SOURCE_REQUEST
    )
    assert partition.completeness_receipt.request_digest == (
        runtime_request.runtime_source_request_digest
    )
    assert (
        validate_runtime_discovery_inventory(
            (runtime_request,),
            inventory,
            {"main": expected},
        )
        is inventory
    )


def test_receipt_uses_one_derived_tagged_request_identity_per_api() -> None:
    graph_input, _, legacy_partition = _complete_pair()
    runtime_request, _, runtime_partition = _complete_runtime_pair()
    receipt_fields = {item.name for item in fields(DiscoveryCompletenessReceipt)}

    assert receipt_fields >= {"request_kind", "request_digest"}
    assert "graph_input_digest" not in receipt_fields
    assert "runtime_source_request_digest" not in receipt_fields
    assert legacy_partition.completeness_receipt.request_kind == (
        DiscoveryRequestKind.GRAPH_TOPOLOGY_INPUT
    )
    assert legacy_partition.completeness_receipt.request_digest == (
        graph_input_identity_digest(graph_input)
    )
    assert runtime_partition.completeness_receipt.request_kind == (
        DiscoveryRequestKind.RUNTIME_GRAPH_SOURCE_REQUEST
    )
    assert runtime_partition.completeness_receipt.request_digest == (
        runtime_request.runtime_source_request_digest
    )
    obsolete_kwargs = {
        item.name: getattr(legacy_partition.completeness_receipt, item.name)
        for item in fields(DiscoveryCompletenessReceipt)
        if item.init
    }
    obsolete_kwargs.pop("request_kind")
    legacy_digest = obsolete_kwargs.pop("request_digest")
    obsolete_kwargs["graph_input_digest"] = legacy_digest

    with pytest.raises(TypeError, match="unexpected keyword"):
        DiscoveryCompletenessReceipt(**obsolete_kwargs)


@pytest.mark.parametrize(
    "mutation",
    (
        "runtime-cross-mode",
        "legacy-cross-mode",
        "raw-string-kind",
        "legacy-raw-string-kind",
        "wrong-enum-kind",
        "unregistered-kind",
        "legacy-unregistered-kind",
        "comparison-digest",
        "legacy-comparison-digest",
    ),
)
def test_receipt_request_identity_fails_closed_across_modes(mutation: str) -> None:
    graph_input, legacy_expected, legacy_partition = _complete_pair()
    runtime_request, runtime_expected, runtime_partition = _complete_runtime_pair()
    if mutation == "runtime-cross-mode":
        forged = replace(
            runtime_partition,
            completeness_receipt=replace(
                runtime_partition.completeness_receipt,
                request_kind=DiscoveryRequestKind.GRAPH_TOPOLOGY_INPUT,
                request_digest=legacy_partition.completeness_receipt.request_digest,
            ),
        )
        validate = lambda: validate_runtime_discovery_inventory(  # noqa: E731
            (runtime_request,),
            SourceDiscoveryInventory((forged,)),
            {"main": runtime_expected},
        )
        error = ValueError
        match = "request kind"
    elif mutation == "legacy-cross-mode":
        forged = replace(
            legacy_partition,
            completeness_receipt=replace(
                legacy_partition.completeness_receipt,
                request_kind=DiscoveryRequestKind.RUNTIME_GRAPH_SOURCE_REQUEST,
                request_digest=runtime_partition.completeness_receipt.request_digest,
            ),
        )
        validate = lambda: validate_discovery_inventory(  # noqa: E731
            (graph_input,),
            SourceDiscoveryInventory((forged,)),
            {"main": legacy_expected},
        )
        error = ValueError
        match = "request kind"
    else:
        legacy_mutation = mutation.startswith("legacy-")
        forged = loads(
            dumps(legacy_partition if legacy_mutation else runtime_partition)
        )
        receipt = forged.completeness_receipt
        if mutation in {"raw-string-kind", "legacy-raw-string-kind"}:
            object.__setattr__(
                receipt,
                "request_kind",
                receipt.request_kind.value,
            )
        elif mutation == "wrong-enum-kind":
            object.__setattr__(
                receipt,
                "request_kind",
                SourceRecordProvenance.TRAINING_RUNTIME,
            )
        elif mutation in {"unregistered-kind", "legacy-unregistered-kind"}:
            object.__setattr__(
                receipt,
                "request_kind",
                _unregistered_enum_member(
                    DiscoveryRequestKind,
                    underlying_value=DiscoveryRequestKind.RUNTIME_GRAPH_SOURCE_REQUEST.value,
                    reported_value=DiscoveryRequestKind.GRAPH_TOPOLOGY_INPUT.value,
                ),
            )
        else:
            object.__setattr__(
                receipt,
                "request_digest",
                _ComparisonBypassText(receipt.request_digest),
            )
        if legacy_mutation:
            validate = lambda: validate_discovery_inventory(  # noqa: E731
                (graph_input,),
                SourceDiscoveryInventory((forged,)),
                {"main": legacy_expected},
            )
        else:
            validate = lambda: validate_runtime_discovery_inventory(  # noqa: E731
                (runtime_request,),
                SourceDiscoveryInventory((forged,)),
                {"main": runtime_expected},
            )
        error = TypeError
        match = (
            "registered DiscoveryRequestKind"
            if mutation in {"unregistered-kind", "legacy-unregistered-kind"}
            else "exact"
        )

    with pytest.raises(error, match=match):
        validate()


@pytest.mark.parametrize(
    "changed_request",
    [
        pytest.param(
            lambda: _runtime_request(selection_character="c"),
            id="selection",
        ),
        pytest.param(
            lambda: _runtime_request(allocation_generation="allocation-2"),
            id="allocation",
        ),
        pytest.param(
            lambda: _runtime_request(
                resolved_graph=_resolved_graph(adapter_id="test.adapter.v2")
            ),
            id="adapter",
        ),
    ],
)
def test_runtime_partition_replay_against_changed_phase_one_request_fails(
    changed_request: Callable[[], RuntimeGraphSourceRequest],
) -> None:
    _, expected, partition = _complete_runtime_pair()
    replacement = changed_request()

    with pytest.raises(ValueError, match="runtime source request digest"):
        validate_runtime_discovery_inventory(
            (replacement,),
            SourceDiscoveryInventory((partition,)),
            {"main": expected},
        )


def test_legacy_and_runtime_partition_assembly_apis_are_disjoint() -> None:
    runtime_request, expected, _ = _complete_runtime_pair()
    graph_input = _graph_input(expected=expected)
    record = _record()
    contribution = _contribution(
        expected.contributor_ids[0],
        (record,),
        fingerprint=runtime_request.source_producer_fingerprint,
    )

    with pytest.raises(TypeError, match="unexpected keyword"):
        assemble_graph_discovery_partition(
            graph_input=graph_input,
            runtime_request=runtime_request,
            expected_contributors=expected,
            contributions=(contribution,),
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        assemble_runtime_graph_discovery_partition(
            graph_input=graph_input,
            runtime_request=runtime_request,
            expected_contributors=expected,
            contributions=(contribution,),
        )
    with pytest.raises(TypeError, match="required keyword-only argument"):
        assemble_runtime_graph_discovery_partition(
            expected_contributors=expected,
            contributions=(contribution,),
        )


def test_runtime_inventory_rejects_mixed_legacy_and_phase_one_bound_requests() -> None:
    runtime_request, runtime_expected, runtime_partition = _complete_runtime_pair()
    legacy_input, legacy_expected, legacy_partition = _complete_pair(
        "draft.external",
        expected=_expected(("draft-rank",), character="7"),
    )

    with pytest.raises(TypeError, match="exact RuntimeGraphSourceRequest"):
        validate_runtime_discovery_inventory(
            (runtime_request, legacy_input),
            SourceDiscoveryInventory((runtime_partition, legacy_partition)),
            {"main": runtime_expected, "draft.external": legacy_expected},
        )
    with pytest.raises(TypeError, match="GraphTopologyInput"):
        validate_discovery_inventory(
            (runtime_request,),
            SourceDiscoveryInventory((runtime_partition,)),
            {"main": runtime_expected},
        )


def test_runtime_boundary_rejects_expected_contributor_set_subclasses() -> None:
    runtime_request, expected, partition = _complete_runtime_pair()
    subclass = _ExpectedContributorSetSubclass(
        expected.contributor_ids,
        expected.authority,
    )
    spoofed = _SpoofedExpectedContributorSet(
        ("not-the-observed-contributor",),
        expected.authority,
        spoofed_authority=expected.to_authority(),
    )

    with pytest.raises(TypeError, match="exact ExpectedContributorSet"):
        assemble_runtime_graph_discovery_partition(
            runtime_request=runtime_request,
            expected_contributors=subclass,
            contributions=(),
        )
    with pytest.raises(TypeError, match="exact ExpectedContributorSet"):
        validate_runtime_discovery_inventory(
            (runtime_request,),
            SourceDiscoveryInventory((partition,)),
            {"main": spoofed},
        )


@pytest.mark.parametrize(
    ("semantic_character", "selection_character"),
    (("c", "b"), ("a", "c")),
    ids=("semantic-structure", "selection-group"),
)
def test_runtime_inventory_requires_one_phase_one_selection_identity(
    semantic_character: str,
    selection_character: str,
) -> None:
    main_request, main_expected, main_partition = _complete_runtime_pair()
    draft_request, draft_expected, draft_partition = _complete_runtime_pair(
        "draft.external",
        expected=_expected(("draft-rank",), character="7"),
        semantic_character=semantic_character,
        selection_character=selection_character,
    )

    with pytest.raises(ValueError, match="one Phase 1 selection identity"):
        validate_runtime_discovery_inventory(
            (main_request, draft_request),
            SourceDiscoveryInventory((main_partition, draft_partition)),
            {"main": main_expected, "draft.external": draft_expected},
        )


def test_runtime_inventory_rejects_forged_request_digest_subclass() -> None:
    runtime_request, expected, partition = _complete_runtime_pair()
    forged_request = loads(dumps(runtime_request))
    object.__setattr__(
        forged_request,
        "runtime_source_request_digest",
        _ComparisonBypassText(_digest("9")),
    )

    with pytest.raises(TypeError, match="exact string"):
        validate_runtime_discovery_inventory(
            (forged_request,),
            SourceDiscoveryInventory((partition,)),
            {"main": expected},
        )


def test_runtime_inventory_rejects_coordinated_phase_one_receipt_forgery() -> None:
    _, expected, partition = _complete_runtime_pair()
    forged_request = loads(dumps(_runtime_request(selection_character="c")))
    forged_partition = loads(dumps(partition))
    forged_digest = _ComparisonBypassText(_digest("9"))
    object.__setattr__(
        forged_request,
        "runtime_source_request_digest",
        forged_digest,
    )
    object.__setattr__(
        forged_partition.completeness_receipt,
        "request_digest",
        forged_digest,
    )

    with pytest.raises(TypeError, match="exact string"):
        validate_runtime_discovery_inventory(
            (forged_request,),
            SourceDiscoveryInventory((forged_partition,)),
            {"main": expected},
        )


def test_runtime_inventory_replays_resolved_graph_constructor_invariants() -> None:
    runtime_request, _, _ = _complete_runtime_pair()
    object.__setattr__(runtime_request.resolved_graph, "model_family", "")

    with pytest.raises(ValueError, match="model_family.*non-empty"):
        runtime_source_request_identity_digest(runtime_request)


@pytest.mark.parametrize("mutation", ("digest", "count"))
def test_runtime_inventory_rejects_receipt_scalar_subclasses(mutation: str) -> None:
    runtime_request, expected, partition = _complete_runtime_pair()
    receipt = partition.completeness_receipt
    forged_receipt = loads(dumps(receipt))
    if mutation == "digest":
        object.__setattr__(
            forged_receipt,
            "request_digest",
            _ComparisonBypassText(_digest("9")),
        )
    else:
        object.__setattr__(
            forged_receipt,
            "source_count",
            _ComparisonBypassInt(999),
        )

    with pytest.raises(TypeError, match="exact"):
        validate_runtime_discovery_inventory(
            (runtime_request,),
            SourceDiscoveryInventory(
                (replace(partition, completeness_receipt=forged_receipt),)
            ),
            {"main": expected},
        )


@pytest.mark.parametrize(
    "container_kind",
    ("inventory", "partition", "receipt", "record"),
)
def test_runtime_inventory_rejects_record_and_container_subclasses(
    container_kind: str,
) -> None:
    runtime_request, expected, partition = _complete_runtime_pair()
    if container_kind == "inventory":
        inventory = _SourceDiscoveryInventorySubclass((partition,))
    elif container_kind == "partition":
        partition_kwargs = {
            item.name: getattr(partition, item.name)
            for item in fields(GraphDiscoveryPartition)
            if item.init
        }
        inventory = SourceDiscoveryInventory(
            (_GraphDiscoveryPartitionSubclass(**partition_kwargs),)
        )
    elif container_kind == "receipt":
        receipt = partition.completeness_receipt
        receipt_kwargs = {
            item.name: getattr(receipt, item.name)
            for item in fields(DiscoveryCompletenessReceipt)
            if item.init
        }
        inventory = SourceDiscoveryInventory(
            (
                replace(
                    partition,
                    completeness_receipt=_DiscoveryCompletenessReceiptSubclass(
                        **receipt_kwargs
                    ),
                ),
            )
        )
    else:
        record = partition.records[0]
        record_kwargs = {
            item.name: getattr(record, item.name)
            for item in fields(SourceDiscoveryRecord)
            if item.init
        }
        inventory = SourceDiscoveryInventory(
            (
                replace(
                    partition,
                    records=(_SourceDiscoveryRecordSubclass(**record_kwargs),),
                ),
            )
        )

    with pytest.raises(TypeError, match="exact runtime discovery"):
        validate_runtime_discovery_inventory(
            (runtime_request,),
            inventory,
            {"main": expected},
        )


@pytest.mark.parametrize("mutation", ("evidence", "native-name"))
def test_runtime_discovery_replays_nested_record_invariants(mutation: str) -> None:
    runtime_request, expected, _ = _complete_runtime_pair()
    contribution = loads(
        dumps(
            _contribution(
                expected.contributor_ids[0],
                (_record(),),
                fingerprint=runtime_request.source_producer_fingerprint,
            )
        )
    )
    if mutation == "evidence":
        object.__setattr__(contribution.records[0].provenance_evidence, "locator", "")
        error = "non-empty"
    else:
        object.__setattr__(contribution.records[0], "source_native_name", None)
        error = "both native fields"

    with pytest.raises(ValueError, match=error):
        assemble_runtime_graph_discovery_partition(
            runtime_request=runtime_request,
            expected_contributors=expected,
            contributions=(contribution,),
        )


def test_runtime_partition_rejects_mutated_contribution_storage_inventory() -> None:
    runtime_request, expected, _ = _complete_runtime_pair()
    contribution = loads(
        dumps(
            _contribution(
                expected.contributor_ids[0],
                (_record(),),
                fingerprint=runtime_request.source_producer_fingerprint,
            )
        )
    )
    object.__setattr__(
        contribution.storage_realizations,
        "graph_instance_id",
        "draft.external",
    )

    with pytest.raises(ValueError, match="realization inventory graph mismatch"):
        assemble_runtime_graph_discovery_partition(
            runtime_request=runtime_request,
            expected_contributors=expected,
            contributions=(contribution,),
        )


@pytest.mark.parametrize(
    "location",
    ("inventory", "contribution", "contribution-root"),
)
def test_runtime_boundary_rejects_exact_records_in_wrong_container_fields(
    location: str,
) -> None:
    runtime_request, expected, partition = _complete_runtime_pair()
    if location == "inventory":
        inventory = SourceDiscoveryInventory((partition,))
        object.__setattr__(inventory, "partitions", (_record(),))
        with pytest.raises(TypeError, match="graph partitions"):
            validate_runtime_discovery_inventory(
                (runtime_request,),
                inventory,
                {"main": expected},
            )
    elif location == "contribution":
        contribution = _contribution(
            expected.contributor_ids[0],
            (_record(),),
            fingerprint=runtime_request.source_producer_fingerprint,
        )
        object.__setattr__(
            contribution,
            "records",
            (_evidence("wrong-field", "7"),),
        )
        with pytest.raises(TypeError, match="SourceDiscoveryRecord"):
            assemble_runtime_graph_discovery_partition(
                runtime_request=runtime_request,
                expected_contributors=expected,
                contributions=(contribution,),
            )
    else:
        with pytest.raises(TypeError, match="exact DiscoveryContribution"):
            assemble_runtime_graph_discovery_partition(
                runtime_request=runtime_request,
                expected_contributors=expected,
                contributions=(partition,),  # type: ignore[arg-type]
            )


def test_runtime_validation_snapshots_mapping_before_validating_inventory() -> None:
    runtime_request, expected, partition = _complete_runtime_pair()
    inventory = SourceDiscoveryInventory((partition,))

    class MutatingMapping(Mapping[str, ExpectedContributorSet]):
        def __getitem__(self, key: str) -> ExpectedContributorSet:
            if key != "main":
                raise KeyError(key)
            return expected

        def __iter__(self) -> Iterator[str]:
            object.__setattr__(
                partition.completeness_receipt,
                "source_count",
                _ComparisonBypassInt(999),
            )
            return iter(("main",))

        def __len__(self) -> int:
            return 1

    with pytest.raises(TypeError, match="exact runtime discovery"):
        validate_runtime_discovery_inventory(
            (runtime_request,),
            inventory,
            MutatingMapping(),
        )


def test_runtime_inventory_rejects_cyclic_exact_transport_tree() -> None:
    code = """
from tests.unit.precision_policy.test_source_discovery import _complete_runtime_pair
from nemo_rl.precision_policy.source_discovery import (
    SourceDiscoveryInventory,
    validate_runtime_discovery_inventory,
)

request, expected, partition = _complete_runtime_pair()
inventory = SourceDiscoveryInventory((partition,))
object.__setattr__(inventory, 'partitions', (inventory,))
try:
    validate_runtime_discovery_inventory((request,), inventory, {'main': expected})
except ValueError as error:
    if 'cycle' in str(error):
        raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_inventory_rejects_cyclic_frozen_request_config() -> None:
    runtime_request, expected, partition = _complete_runtime_pair()
    model_config = runtime_request.model_config
    object.__setattr__(model_config, "entries", (("cycle", model_config),))

    with pytest.raises(ValueError, match="cycle"):
        validate_runtime_discovery_inventory(
            (runtime_request,),
            SourceDiscoveryInventory((partition,)),
            {"main": expected},
        )


def test_runtime_inventory_rejects_cyclic_resolved_topology_tree() -> None:
    code = """
from tests.unit.precision_policy.test_source_discovery import _complete_runtime_pair
from nemo_rl.precision_policy.source_discovery import (
    SourceDiscoveryInventory,
    validate_runtime_discovery_inventory,
)

request, expected, partition = _complete_runtime_pair()
object.__setattr__(request.resolved_graph, 'entries', (request.resolved_graph,))
try:
    validate_runtime_discovery_inventory(
        (request,), SourceDiscoveryInventory((partition,)), {'main': expected}
    )
except ValueError as error:
    if 'cycle' in str(error):
        raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_inventory_rejects_empty_and_duplicate_request_sets() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_runtime_discovery_inventory((), SourceDiscoveryInventory(()), {})

    runtime_request, expected, partition = _complete_runtime_pair()
    with pytest.raises(ValueError, match="duplicate graph topology input"):
        validate_runtime_discovery_inventory(
            (runtime_request, runtime_request),
            SourceDiscoveryInventory((partition,)),
            {"main": expected},
        )


def test_runtime_inventory_accepts_benign_mapping_and_pickle_roundtrip() -> None:
    main_request, main_expected, main_partition = _complete_runtime_pair()
    draft_request, draft_expected, draft_partition = _complete_runtime_pair(
        "draft.external",
        expected=_expected(("draft-rank",), character="7"),
    )
    requests, inventory, trusted = loads(
        dumps(
            (
                (draft_request, main_request),
                SourceDiscoveryInventory((draft_partition, main_partition)),
                UserDict(
                    {
                        "draft.external": draft_expected,
                        "main": main_expected,
                    }
                ),
            )
        )
    )

    assert (
        validate_runtime_discovery_inventory(requests, inventory, trusted) is inventory
    )


def test_runtime_inventory_rejects_noncanonical_partition_order() -> None:
    main_request, main_expected, main_partition = _complete_runtime_pair()
    draft_request, draft_expected, draft_partition = _complete_runtime_pair(
        "draft.external",
        expected=_expected(("draft-rank",), character="7"),
    )
    inventory = SourceDiscoveryInventory((main_partition, draft_partition))
    object.__setattr__(
        inventory,
        "partitions",
        tuple(reversed(inventory.partitions)),
    )

    with pytest.raises(ValueError, match="noncanonical|canonically ordered"):
        validate_runtime_discovery_inventory(
            (main_request, draft_request),
            inventory,
            {"main": main_expected, "draft.external": draft_expected},
        )


def test_public_canonical_digests_use_lowercase_sha256_grammar() -> None:
    graph_input, expected, partition = _complete_pair()
    receipt = partition.completeness_receipt
    canonical_digest = re.compile(r"sha256:[0-9a-f]{64}").fullmatch
    digests = (
        graph_input_identity_digest(graph_input),
        expected.to_authority().contributor_set_digest,
        receipt.producer_fingerprint_digest,
        receipt.observed_contributor_set_digest,
        receipt.source_set_digest,
        receipt.canonical_records_digest,
        receipt.storage_realization_set_digest,
        receipt.request_digest,
    )

    assert all(canonical_digest(digest) is not None for digest in digests)


def test_canonical_digests_are_stable_across_python_hash_seeds() -> None:
    code = """
from nemo_rl.precision_policy.semantic import (
    EvidenceSource, EvidenceSourceKind, ExpectedGraphDeclaration, GraphKind,
    GraphLifecycle, GraphProvenance, RolloutParticipation, SourceMutability,
)
from nemo_rl.precision_policy.source_discovery import (
    HF_SAFETENSORS_HEADER_V1, DiscoveryContribution, ExpectedContributorSet,
    GraphTopologyInput, SourceDiscoveryRecord, SourceProducerFingerprint,
    SourceRecordProvenance, assemble_graph_discovery_partition,
    graph_input_identity_digest,
)
from nemo_rl.precision_policy.source_dtype import CanonicalSourceDType
from nemo_rl.precision_policy.source_storage import (
    IDENTITY_PERMUTATION_ID, IDENTITY_SWIZZLE_ID, SourceExtentRounding,
    SourceNormalizationContract, SourceNormalizationKind,
    SourceNormalizerManifest, SourceNormalizedAxisExtent,
    SourcePaddingSemantics, SourcePhysicalAxisSpec, SourceStorageComponent,
    SourceStorageRealization, SourceStorageRealizationInventory,
    source_normalizer_manifest_digest,
)

def evidence(name, character):
    return EvidenceSource(
        kind=EvidenceSourceKind.RUNTIME_INVENTORY,
        locator=f'runtime://{name}',
        digest=f'sha256:{character * 64}',
    )

expected = ExpectedContributorSet(('rank-b', 'rank-a'), evidence('membership', '1'))
normalization = SourceNormalizationContract(
    'test.identity.v1', SourceNormalizationKind.IDENTITY, 'sha256:' + '2' * 64,
)
manifest = SourceNormalizerManifest(1, (normalization,))
fingerprint = SourceProducerFingerprint(
    HF_SAFETENSORS_HEADER_V1,
    'checkpoint-header-reader',
    'a' * 40,
    source_normalizer_manifest_digest(manifest),
    evidence('producer', '3'),
)
graph_input = GraphTopologyInput(
    ExpectedGraphDeclaration(
        'main',
        'test/main',
        GraphLifecycle(
            GraphKind.MAIN,
            GraphProvenance.TRAINING_RUNTIME,
            RolloutParticipation.SERVED_FROM_SOURCE,
        ),
    ),
    {'z': None, 'a': [True, 7, 2.5, {'b': 'value'}]},
    'b' * 40,
    fingerprint,
    expected.to_authority(),
    evidence('source', '4'),
    evidence('artifact', '5'),
)
record = SourceDiscoveryRecord(
    'main.weight', 'main', 'model.weight', 'model.weight',
    CanonicalSourceDType.BFLOAT16, (8, 8), 'plain_bfloat16',
    SourceRecordProvenance.TRAINING_RUNTIME, evidence('provenance', '6'),
    SourceMutability.MUTABLE, evidence('mutability', '7'),
)
component = SourceStorageComponent(
    'main', 'main.weight.component', 'model.weight', 'logical_values',
    CanonicalSourceDType.BFLOAT16, (8, 8),
    (
        SourcePhysicalAxisSpec(
            'axis_0', SourceNormalizedAxisExtent(
                (0,), 1, SourceExtentRounding.EXACT, 1,
            ),
        ),
        SourcePhysicalAxisSpec(
            'axis_1', SourceNormalizedAxisExtent(
                (1,), 1, SourceExtentRounding.EXACT, 1,
            ),
        ),
    ),
    'plain_bfloat16', SourcePaddingSemantics.NO_PADDING, None,
    IDENTITY_PERMUTATION_ID, IDENTITY_SWIZZLE_ID,
)
realization = SourceStorageRealization(
    'main.weight.identity', 'main', 'main.weight', (component,),
    CanonicalSourceDType.BFLOAT16, (8, 8), 'plain_bfloat16', normalization,
)
empty_storage = SourceStorageRealizationInventory('main', manifest, ())
weight_storage = SourceStorageRealizationInventory(
    'main', manifest, (realization,),
)
partition = assemble_graph_discovery_partition(
    graph_input=graph_input,
    expected_contributors=expected,
    contributions=(
        DiscoveryContribution('rank-b', 'main', fingerprint, (), empty_storage),
        DiscoveryContribution(
            'rank-a', 'main', fingerprint, (record,), weight_storage,
        ),
    ),
)
receipt = partition.completeness_receipt
print('|'.join((
    graph_input_identity_digest(graph_input),
    expected.to_authority().contributor_set_digest,
    receipt.producer_fingerprint_digest,
    receipt.source_set_digest,
    receipt.canonical_records_digest,
)))
"""
    outputs = []
    for seed in ("1", "8675309"):
        result = subprocess.run(
            (sys.executable, "-c", code),
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "."},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())

    assert outputs[0] == outputs[1]


def test_canonical_digests_do_not_use_repr_or_dataclass_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_input, expected, partition = _complete_pair()

    def forbidden(*_args: object) -> object:
        raise AssertionError("repr/dataclass hash entered canonical digest path")

    monkeypatch.setattr(SourceDiscoveryRecord, "__repr__", forbidden)
    monkeypatch.setattr(SourceDiscoveryRecord, "__hash__", forbidden)
    monkeypatch.setattr(SourceProducerFingerprint, "__repr__", forbidden)
    monkeypatch.setattr(SourceProducerFingerprint, "__hash__", forbidden)

    rebuilt = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(
            _contribution(
                "checkpoint-index",
                partition.records,
                fingerprint=graph_input.source_producer_fingerprint,
            ),
        ),
    )

    assert rebuilt.completeness_receipt == partition.completeness_receipt


def _canonical_records_digest_for(record: SourceDiscoveryRecord) -> str:
    expected = _expected()
    fingerprint = _fingerprint()
    graph_input = _graph_input(
        record.graph_instance_id,
        expected=expected,
        fingerprint=fingerprint,
    )
    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(
            _contribution(
                "checkpoint-index",
                (record,),
                graph_instance_id=record.graph_instance_id,
                fingerprint=fingerprint,
            ),
        ),
    )
    return partition.completeness_receipt.canonical_records_digest


@pytest.mark.parametrize(
    "changed",
    [
        _record("main.other"),
        _record(
            "draft.external.weight",
            graph_instance_id="draft.external",
            native_name="model.weight",
            native_owner="model.weight",
        ),
        _record(native_name="model.other"),
        _record(native_owner="model.other"),
        replace(_record(), dtype=CanonicalSourceDType.FLOAT16),
        replace(_record(), shape=(4, 16)),
        replace(_record(), numeric_encoding="different_encoding"),
        replace(
            _record(),
            provenance=SourceRecordProvenance.CHECKPOINT_STORAGE,
        ),
        replace(
            _record(),
            provenance_evidence=_evidence("different-provenance", "7"),
        ),
        replace(_record(), source_mutability=SourceMutability.FROZEN),
        replace(
            _record(),
            mutability_evidence=_evidence("different-mutability", "8"),
        ),
    ],
    ids=(
        "record-id",
        "graph-instance-id",
        "native-name",
        "native-owner",
        "dtype",
        "shape",
        "numeric-encoding",
        "provenance",
        "provenance-evidence",
        "mutability",
        "mutability-evidence",
    ),
)
def test_canonical_record_digest_binds_every_normalized_record_field(
    changed: SourceDiscoveryRecord,
) -> None:
    assert _canonical_records_digest_for(changed) != _canonical_records_digest_for(
        _record()
    )


def _fingerprint_digests(
    fingerprint: SourceProducerFingerprint,
) -> tuple[str, str]:
    expected = _expected()
    graph_input = _graph_input(expected=expected, fingerprint=fingerprint)
    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(
            _contribution(
                "checkpoint-index",
                (_record(),),
                fingerprint=fingerprint,
            ),
        ),
    )
    return (
        partition.completeness_receipt.producer_fingerprint_digest,
        partition.completeness_receipt.request_digest,
    )


@pytest.mark.parametrize(
    "changed",
    [
        _fingerprint(schema_id=MEGATRON_BRIDGE_STATE_DICT_V1),
        _fingerprint(implementation_id="different-reader"),
        _fingerprint(revision="b" * 40),
        _fingerprint(character="7"),
        replace(_fingerprint(), evidence=_evidence("different-producer", "8")),
    ],
    ids=("schema", "implementation", "revision", "normalization", "evidence"),
)
def test_canonical_fingerprint_digest_binds_every_fingerprint_field(
    changed: SourceProducerFingerprint,
) -> None:
    baseline = _fingerprint_digests(_fingerprint())
    changed_digests = _fingerprint_digests(changed)

    assert changed_digests[0] != baseline[0]
    assert changed_digests[1] != baseline[1]


def test_one_fingerprint_is_stored_once_per_complete_graph_partition() -> None:
    fingerprint = _fingerprint()
    expected = _expected(("checkpoint-index",))
    graph_input = _graph_input(fingerprint=fingerprint, expected=expected)
    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(
            _contribution(
                "checkpoint-index",
                (
                    _record("main.z", native_name="model.z"),
                    _record("main.a", native_name="model.a"),
                ),
                fingerprint=fingerprint,
            ),
        ),
    )

    assert partition.producer_fingerprint == fingerprint
    assert partition.expected_contributor_authority == expected.to_authority()
    assert partition.completeness_receipt.observed_contributor_count == 1
    assert partition.completeness_receipt.source_count == 2
    assert tuple(record.record_id for record in partition.records) == (
        "main.a",
        "main.z",
    )
    assert all(
        "fingerprint" not in {field.name for field in fields(record)}
        for record in partition.records
    )
    assert "contributor_id" not in {field.name for field in fields(partition)}
    assert "checkpoint-index" not in repr(asdict(partition))


def test_reordered_contributions_and_records_assemble_identically() -> None:
    expected = _expected(("rank-b", "rank-a"))
    graph_input = _graph_input(expected=expected)
    first = _contribution(
        "rank-a",
        (
            _record("main.b", native_name="model.b"),
            _record("main.a", native_name="model.a"),
        ),
    )
    second = _contribution(
        "rank-b",
        (_record("main.c", native_name="model.c"),),
    )

    forward = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(first, second),
    )
    reverse = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(
            replace(second, records=tuple(reversed(second.records))),
            replace(first, records=tuple(reversed(first.records))),
        ),
    )

    assert reverse == forward
    assert loads(dumps(forward)) == forward


def test_contributions_cannot_self_certify_and_private_placement_is_stripped() -> None:
    expected = _expected(("private-pp0-tp1-ep2", "private-pp1-tp0-ep3"))
    graph_input = _graph_input(expected=expected)
    caller_records = [_record()]
    first = _contribution(
        "private-pp0-tp1-ep2",
        caller_records,  # type: ignore[arg-type]
    )
    caller_records.clear()
    second = _contribution("private-pp1-tp0-ep3", ())

    assert tuple(field.name for field in fields(DiscoveryContribution)) == (
        "contributor_id",
        "graph_instance_id",
        "producer_fingerprint",
        "records",
        "storage_realizations",
    )
    assert first.records == (_record(),)
    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(first, second),
    )
    partition_payload = repr(asdict(partition))
    assert "private-pp0-tp1-ep2" not in partition_payload
    assert "private-pp1-tp0-ep3" not in partition_payload
    assert "pp0" not in partition_payload
    assert "tp1" not in partition_payload
    assert "ep2" not in partition_payload


@pytest.mark.parametrize(
    ("contributions", "error"),
    [
        (("rank-a",), "missing"),
        (("rank-a", "rank-a", "rank-b"), "duplicate"),
        (("rank-a", "rank-b", "rank-c"), "unexpected"),
    ],
)
def test_partition_assembly_requires_exact_contributor_equality(
    contributions: tuple[str, ...],
    error: str,
) -> None:
    expected = _expected(("rank-a", "rank-b"))
    graph_input = _graph_input(expected=expected)

    with pytest.raises(ValueError, match=error):
        assemble_graph_discovery_partition(
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=tuple(
                _contribution(
                    contributor_id,
                    (
                        _record(
                            f"main.{index}",
                            native_name=f"model.{index}",
                        ),
                    ),
                )
                for index, contributor_id in enumerate(contributions)
            ),
        )


def test_partition_assembly_rejects_wrong_graph_and_mixed_fingerprints() -> None:
    expected = _expected()
    graph_input = _graph_input(expected=expected)
    record = _record()

    with pytest.raises(ValueError, match="graph"):
        assemble_graph_discovery_partition(
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=(
                _contribution(
                    "checkpoint-index",
                    (replace(record, graph_instance_id="draft.external"),),
                ),
            ),
        )
    with pytest.raises(ValueError, match="fingerprint"):
        assemble_graph_discovery_partition(
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=(
                _contribution(
                    "checkpoint-index",
                    (record,),
                    fingerprint=_fingerprint(character="9"),
                ),
            ),
        )


def test_partition_assembly_rejects_wrong_contribution_graph() -> None:
    expected = _expected()
    graph_input = _graph_input(expected=expected)

    with pytest.raises(ValueError, match="graph"):
        assemble_graph_discovery_partition(
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=(
                _contribution(
                    "checkpoint-index",
                    (_record(),),
                    graph_instance_id="draft.external",
                ),
            ),
        )


def test_partition_assembly_rejects_true_two_contributor_mixed_fingerprints() -> None:
    expected = _expected(("rank-a", "rank-b"))
    graph_input = _graph_input(expected=expected)

    with pytest.raises(ValueError, match="fingerprint"):
        assemble_graph_discovery_partition(
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=(
                _contribution(
                    "rank-a",
                    (_record("main.a", native_name="model.a"),),
                ),
                _contribution(
                    "rank-b",
                    (_record("main.b", native_name="model.b"),),
                    fingerprint=_fingerprint(character="9"),
                ),
            ),
        )


def test_partition_assembly_rejects_graph_input_authority_mismatch() -> None:
    expected = _expected()
    graph_input = _graph_input(expected=expected)
    other_authority = _expected(("other",)).to_authority()

    with pytest.raises(ValueError, match="authority"):
        assemble_graph_discovery_partition(
            graph_input=replace(
                graph_input,
                expected_contributor_authority=other_authority,
            ),
            expected_contributors=expected,
            contributions=(_contribution("checkpoint-index", (_record(),)),),
        )


def test_partition_assembly_binds_authority_evidence_not_only_id_digest() -> None:
    expected = _expected(("checkpoint-index",), character="2")
    same_ids_different_evidence = _expected(
        ("checkpoint-index",),
        character="7",
    )
    graph_input = _graph_input(
        expected=same_ids_different_evidence,
    )

    assert (
        expected.to_authority().contributor_set_digest
        == same_ids_different_evidence.to_authority().contributor_set_digest
    )
    assert (
        expected.to_authority().contributor_count
        == same_ids_different_evidence.to_authority().contributor_count
    )
    assert (
        expected.to_authority().authority
        != same_ids_different_evidence.to_authority().authority
    )
    with pytest.raises(ValueError, match="authority"):
        assemble_graph_discovery_partition(
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=(_contribution("checkpoint-index", (_record(),)),),
        )


@pytest.mark.parametrize("duplicate_kind", ["record_id", "native_name"])
def test_duplicate_sources_fail_across_contributor_boundaries(
    duplicate_kind: str,
) -> None:
    expected = _expected(("rank-a", "rank-b"))
    graph_input = _graph_input(expected=expected)
    first = _record("main.first", native_name="model.first")
    second = _record("main.second", native_name="model.second")
    if duplicate_kind == "record_id":
        second = replace(second, record_id=first.record_id)
    else:
        second = replace(second, source_native_name=first.source_native_name)

    with pytest.raises(ValueError, match="duplicate"):
        assemble_graph_discovery_partition(
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=(
                _contribution("rank-a", (first,)),
                _contribution("rank-b", (second,)),
            ),
        )


def test_expected_empty_contributor_is_valid_but_empty_universe_is_not() -> None:
    expected = _expected(("rank-a", "rank-b"))
    graph_input = _graph_input(expected=expected)

    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(
            _contribution("rank-a", (_record(),)),
            _contribution("rank-b", ()),
        ),
    )

    assert partition.completeness_receipt.observed_contributor_count == 2
    assert partition.completeness_receipt.source_count == 1
    with pytest.raises(ValueError, match="source universe.*empty"):
        assemble_graph_discovery_partition(
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=(
                _contribution("rank-a", ()),
                _contribution("rank-b", ()),
            ),
        )


def test_typed_absent_record_makes_the_complete_universe_nonempty() -> None:
    expected = _expected()
    graph_input = _graph_input(expected=expected)
    absent = _record(
        native_name=None,
        native_owner=None,
        source_mutability=SourceMutability.ABSENT,
    )

    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(_contribution("checkpoint-index", (absent,)),),
    )

    assert partition.records == (absent,)


def _validate_complete_pair(
    graph_input: GraphTopologyInput,
    expected: ExpectedContributorSet,
    partition: GraphDiscoveryPartition,
) -> None:
    validate_discovery_inventory(
        (graph_input,),
        SourceDiscoveryInventory((partition,)),
        {graph_input.declaration.graph_instance_id: expected},
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "partition_fingerprint",
        "partition_authority",
        "partition_graph",
        "records",
        "receipt_graph",
        "receipt_fingerprint",
        "receipt_observed_count",
        "receipt_observed_digest",
        "receipt_source_count",
        "receipt_source_digest",
        "receipt_records_digest",
        "receipt_storage_count",
        "receipt_storage_digest",
        "receipt_request_digest",
    ],
)
def test_inventory_validation_rejects_forged_or_replaced_partition_fields(
    mutation: str,
) -> None:
    graph_input, expected, partition = _complete_pair()
    receipt = partition.completeness_receipt
    if mutation == "partition_fingerprint":
        forged = replace(partition, producer_fingerprint=_fingerprint(character="9"))
    elif mutation == "partition_authority":
        forged = replace(
            partition,
            expected_contributor_authority=_expected(("other",)).to_authority(),
        )
    elif mutation == "partition_graph":
        forged = loads(dumps(partition))
        object.__setattr__(forged, "graph_instance_id", "draft.external")
    elif mutation == "records":
        forged = replace(
            partition,
            records=(replace(partition.records[0], shape=(7, 8)),),
        )
    elif mutation == "receipt_graph":
        forged = replace(
            partition,
            completeness_receipt=replace(
                receipt,
                graph_instance_id="draft.external",
            ),
        )
    elif mutation == "receipt_fingerprint":
        forged = replace(
            partition,
            completeness_receipt=replace(
                receipt,
                producer_fingerprint_digest=_digest("9"),
            ),
        )
    elif mutation == "receipt_observed_count":
        forged = replace(
            partition,
            completeness_receipt=replace(
                receipt,
                observed_contributor_count=2,
            ),
        )
    elif mutation == "receipt_observed_digest":
        forged = replace(
            partition,
            completeness_receipt=replace(
                receipt,
                observed_contributor_set_digest=_digest("9"),
            ),
        )
    elif mutation == "receipt_source_count":
        forged = replace(
            partition,
            completeness_receipt=replace(receipt, source_count=2),
        )
    elif mutation == "receipt_source_digest":
        forged = replace(
            partition,
            completeness_receipt=replace(receipt, source_set_digest=_digest("9")),
        )
    elif mutation == "receipt_records_digest":
        forged = replace(
            partition,
            completeness_receipt=replace(
                receipt,
                canonical_records_digest=_digest("9"),
            ),
        )
    elif mutation == "receipt_storage_count":
        forged = replace(
            partition,
            completeness_receipt=replace(
                receipt,
                storage_realization_count=2,
            ),
        )
    elif mutation == "receipt_storage_digest":
        forged = replace(
            partition,
            completeness_receipt=replace(
                receipt,
                storage_realization_set_digest=_digest("9"),
            ),
        )
    else:
        forged = replace(
            partition,
            completeness_receipt=replace(
                receipt,
                request_digest=_digest("9"),
            ),
        )

    with pytest.raises(ValueError):
        _validate_complete_pair(graph_input, expected, forged)


@pytest.mark.parametrize(
    "changed_input",
    [
        _graph_input(config={"model_type": "changed"}),
        _graph_input(revision="c" * 40),
        _graph_input(source_character="7"),
        _graph_input(artifact_character="8"),
        _graph_input(fingerprint=_fingerprint(character="9")),
        _graph_input(expected=_expected(("other",))),
    ],
    ids=["config", "revision", "source", "artifact", "fingerprint", "authority"],
)
def test_partition_replay_against_changed_graph_input_fails(
    changed_input: GraphTopologyInput,
) -> None:
    _, expected, partition = _complete_pair()

    with pytest.raises(ValueError):
        _validate_complete_pair(changed_input, expected, partition)


def test_partition_replay_under_another_graph_id_fails() -> None:
    _, _, partition = _complete_pair()
    draft_expected = _expected(("draft-rank",), character="7")
    draft_input = _graph_input("draft.external", expected=draft_expected)

    with pytest.raises(ValueError, match="graph partition"):
        validate_discovery_inventory(
            (draft_input,),
            SourceDiscoveryInventory((partition,)),
            {"draft.external": draft_expected},
        )


def test_coordinated_authority_replacement_fails_against_independent_mapping() -> None:
    graph_input, expected, partition = _complete_pair()
    replacement = _expected(("replacement-shard",), character="7")
    replacement_input = replace(
        graph_input,
        expected_contributor_authority=replacement.to_authority(),
    )
    replacement_receipt = replace(
        partition.completeness_receipt,
        observed_contributor_set_digest=(
            replacement.to_authority().contributor_set_digest
        ),
        observed_contributor_count=replacement.to_authority().contributor_count,
        request_digest=graph_input_identity_digest(replacement_input),
    )
    replacement_partition = replace(
        partition,
        expected_contributor_authority=replacement.to_authority(),
        completeness_receipt=replacement_receipt,
    )

    with pytest.raises(ValueError, match="trusted expected contributor authority"):
        _validate_complete_pair(replacement_input, expected, replacement_partition)


def test_coordinated_authority_evidence_replacement_with_same_ids_fails() -> None:
    graph_input, expected, partition = _complete_pair()
    replacement = _expected(expected.contributor_ids, character="7")
    replacement_authority = replacement.to_authority()
    replacement_input = replace(
        graph_input,
        expected_contributor_authority=replacement_authority,
    )
    replacement_partition = replace(
        partition,
        expected_contributor_authority=replacement_authority,
        completeness_receipt=replace(
            partition.completeness_receipt,
            observed_contributor_set_digest=(
                replacement_authority.contributor_set_digest
            ),
            observed_contributor_count=replacement_authority.contributor_count,
            request_digest=graph_input_identity_digest(replacement_input),
        ),
    )

    assert (
        replacement_authority.contributor_set_digest
        == expected.to_authority().contributor_set_digest
    )
    assert replacement_authority.authority != expected.to_authority().authority
    with pytest.raises(ValueError, match="trusted expected contributor authority"):
        _validate_complete_pair(replacement_input, expected, replacement_partition)


def test_inventory_requires_exactly_one_partition_and_trusted_set_per_graph() -> None:
    main_input, main_expected, main_partition = _complete_pair()
    draft_input, draft_expected, draft_partition = _complete_pair(
        "draft.external",
        fingerprint=_fingerprint(
            schema_id=NEMO_AUTOMODEL_STATE_DICT_V1,
            implementation_id="automodel-state-dict-reader",
            revision="c" * 40,
            character="7",
        ),
        expected=_expected(("draft-rank",), character="8"),
    )
    inputs = (main_input, draft_input)
    trusted = {"main": main_expected, "draft.external": draft_expected}

    validate_discovery_inventory(
        inputs,
        SourceDiscoveryInventory((draft_partition, main_partition)),
        trusted,
    )
    cases = (
        (SourceDiscoveryInventory((main_partition,)), trusted, "missing"),
        (
            SourceDiscoveryInventory((main_partition, main_partition, draft_partition)),
            trusted,
            "duplicate",
        ),
        (
            SourceDiscoveryInventory((main_partition, draft_partition)),
            {"main": main_expected},
            "missing.*trusted",
        ),
        (
            SourceDiscoveryInventory((main_partition, draft_partition)),
            {**trusted, "draft.extra": draft_expected},
            "undeclared.*trusted",
        ),
    )
    for inventory, mapping, error in cases:
        with pytest.raises(ValueError, match=error):
            validate_discovery_inventory(inputs, inventory, mapping)


def test_public_inventory_validator_rejects_undeclared_partition() -> None:
    main_input, main_expected, main_partition = _complete_pair()
    _, _, draft_partition = _complete_pair(
        "draft.external",
        expected=_expected(("draft-rank",), character="7"),
    )

    with pytest.raises(ValueError, match="undeclared source discovery graph partition"):
        validate_discovery_inventory(
            (main_input,),
            SourceDiscoveryInventory((main_partition, draft_partition)),
            {"main": main_expected},
        )


def test_public_inventory_validator_rejects_duplicate_graph_inputs() -> None:
    main_input, main_expected, main_partition = _complete_pair()

    with pytest.raises(ValueError, match="duplicate graph topology input"):
        validate_discovery_inventory(
            (main_input, main_input),
            SourceDiscoveryInventory((main_partition,)),
            {"main": main_expected},
        )


def test_native_name_and_owner_uniqueness_is_graph_scoped() -> None:
    main_expected = _expected(("main-rank",), character="2")
    draft_expected = _expected(("draft-rank",), character="7")
    main_input = _graph_input("main", expected=main_expected)
    draft_input = _graph_input("draft.external", expected=draft_expected)
    native_name = "shared.model.weight"
    native_owner = "shared.model"
    main_record = _record(
        "main.weight",
        native_name=native_name,
        native_owner=native_owner,
    )
    draft_record = _record(
        "draft.external.weight",
        graph_instance_id="draft.external",
        native_name=native_name,
        native_owner=native_owner,
    )
    main_partition = assemble_graph_discovery_partition(
        graph_input=main_input,
        expected_contributors=main_expected,
        contributions=(_contribution("main-rank", (main_record,)),),
    )
    draft_partition = assemble_graph_discovery_partition(
        graph_input=draft_input,
        expected_contributors=draft_expected,
        contributions=(
            _contribution(
                "draft-rank",
                (draft_record,),
                graph_instance_id="draft.external",
            ),
        ),
    )

    validate_discovery_inventory(
        (main_input, draft_input),
        SourceDiscoveryInventory((main_partition, draft_partition)),
        {"main": main_expected, "draft.external": draft_expected},
    )


def test_legacy_record_id_uniqueness_is_graph_scoped() -> None:
    main_expected = _expected(("main-rank",), character="2")
    draft_expected = _expected(("draft-rank",), character="7")
    main_fingerprint = _fingerprint(character="1")
    draft_fingerprint = _fingerprint(
        schema_id=TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1,
        implementation_id="te-storage-reader",
        revision="d" * 40,
        character="8",
    )
    main_input = _graph_input(
        "main",
        expected=main_expected,
        fingerprint=main_fingerprint,
    )
    draft_input = _graph_input(
        "draft.external",
        expected=draft_expected,
        fingerprint=draft_fingerprint,
    )
    local_record_id = "model.weight"
    main_record = _record(local_record_id)
    draft_record = _record(
        local_record_id,
        graph_instance_id="draft.external",
        native_name="draft.model.weight",
        native_owner="draft.model.weight",
    )
    main_partition = assemble_graph_discovery_partition(
        graph_input=main_input,
        expected_contributors=main_expected,
        contributions=(
            _contribution(
                "main-rank",
                (main_record,),
                fingerprint=main_fingerprint,
            ),
        ),
    )
    draft_partition = assemble_graph_discovery_partition(
        graph_input=draft_input,
        expected_contributors=draft_expected,
        contributions=(
            _contribution(
                "draft-rank",
                (draft_record,),
                graph_instance_id="draft.external",
                fingerprint=draft_fingerprint,
            ),
        ),
    )
    inventory = SourceDiscoveryInventory((draft_partition, main_partition))

    validated = validate_discovery_inventory(
        (draft_input, main_input),
        inventory,
        {"draft.external": draft_expected, "main": main_expected},
    )

    assert tuple(
        (record.graph_instance_id, record.record_id) for record in validated.records
    ) == (("main", local_record_id), ("draft.external", local_record_id))


def test_runtime_record_id_uniqueness_is_graph_scoped() -> None:
    main_expected = _expected(("main-rank",), character="2")
    draft_expected = _expected(("draft-rank",), character="7")
    main_fingerprint = _fingerprint(character="1")
    draft_fingerprint = _fingerprint(
        schema_id=TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1,
        implementation_id="te-storage-reader",
        revision="d" * 40,
        character="8",
    )
    main_request = _runtime_request(
        "main",
        expected=main_expected,
        fingerprint=main_fingerprint,
    )
    draft_request = _runtime_request(
        "draft.external",
        expected=draft_expected,
        fingerprint=draft_fingerprint,
    )
    local_record_id = "model.weight"
    main_record = _record(local_record_id)
    draft_record = _record(
        local_record_id,
        graph_instance_id="draft.external",
        native_name="draft.model.weight",
        native_owner="draft.model.weight",
    )
    main_partition = assemble_runtime_graph_discovery_partition(
        runtime_request=main_request,
        expected_contributors=main_expected,
        contributions=(
            _contribution(
                "main-rank",
                (main_record,),
                fingerprint=main_fingerprint,
            ),
        ),
    )
    draft_partition = assemble_runtime_graph_discovery_partition(
        runtime_request=draft_request,
        expected_contributors=draft_expected,
        contributions=(
            _contribution(
                "draft-rank",
                (draft_record,),
                graph_instance_id="draft.external",
                fingerprint=draft_fingerprint,
            ),
        ),
    )
    inventory = SourceDiscoveryInventory((draft_partition, main_partition))

    validated = validate_runtime_discovery_inventory(
        (draft_request, main_request),
        inventory,
        {"draft.external": draft_expected, "main": main_expected},
    )

    assert tuple(
        (record.graph_instance_id, record.record_id) for record in validated.records
    ) == (("main", local_record_id), ("draft.external", local_record_id))


def test_main_and_different_family_draft_partitions_remain_isolated() -> None:
    main_input, main_expected, main_partition = _complete_pair()
    draft_fingerprint = _fingerprint(
        schema_id=TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1,
        implementation_id="te-storage-reader",
        revision="d" * 40,
        character="8",
    )
    draft_expected = _expected(("draft-rank",), character="9")
    draft_input, _, draft_partition = _complete_pair(
        "draft.external",
        fingerprint=draft_fingerprint,
        expected=draft_expected,
    )
    inventory = SourceDiscoveryInventory((draft_partition, main_partition))

    validate_discovery_inventory(
        (draft_input, main_input),
        inventory,
        {"draft.external": draft_expected, "main": main_expected},
    )

    assert tuple(partition.graph_instance_id for partition in inventory.partitions) == (
        "main",
        "draft.external",
    )
    assert inventory.partitions[0].producer_fingerprint != draft_fingerprint
    assert all(
        record.graph_instance_id == partition.graph_instance_id
        for partition in inventory.partitions
        for record in partition.records
    )


def test_public_discovery_containers_are_deeply_frozen() -> None:
    graph_input, expected, partition = _complete_pair()
    partitions = [partition]
    inventory = SourceDiscoveryInventory(partitions)  # type: ignore[arg-type]
    partitions.clear()

    assert inventory.partitions == (partition,)
    assert inventory.records == partition.records
    assert loads(dumps(inventory)) == inventory
    with pytest.raises(FrozenInstanceError):
        partition.graph_instance_id = "draft.external"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        expected.contributor_ids = ("other",)  # type: ignore[misc]
    _validate_complete_pair(graph_input, expected, partition)


def test_discovery_boundary_is_strict_and_exactly_serializable() -> None:
    graph_input, expected, partition = _complete_pair()
    inventory = SourceDiscoveryInventory((partition,))

    with pytest.raises(TypeError, match="requires graph partitions"):
        SourceDiscoveryInventory((_record(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GraphTopologyInput(
            declaration=_declaration(),
            model_config={"model_type": "legacy"},
            resolved_model_revision="b" * 40,
        )  # type: ignore[call-arg]

    assert tuple(field.name for field in fields(GraphTopologyInput)) == (
        "declaration",
        "model_config",
        "resolved_model_revision",
        "source_producer_fingerprint",
        "expected_contributor_authority",
        "source_identity",
        "artifact_identity",
    )
    assert tuple(field.name for field in fields(ExpectedContributorAuthority)) == (
        "contributor_set_digest",
        "contributor_count",
        "authority",
    )
    assert tuple(field.name for field in fields(DiscoveryCompletenessReceipt)) == (
        "graph_instance_id",
        "producer_fingerprint_digest",
        "observed_contributor_set_digest",
        "observed_contributor_count",
        "source_set_digest",
        "source_count",
        "canonical_records_digest",
        "storage_realization_set_digest",
        "storage_realization_count",
        "request_kind",
        "request_digest",
    )
    assert tuple(field.name for field in fields(GraphDiscoveryPartition)) == (
        "graph_instance_id",
        "producer_fingerprint",
        "expected_contributor_authority",
        "records",
        "storage_realizations",
        "completeness_receipt",
    )
    assert tuple(field.name for field in fields(SourceDiscoveryInventory)) == (
        "partitions",
    )
    assert loads(dumps((graph_input, expected, partition, inventory))) == (
        graph_input,
        expected,
        partition,
        inventory,
    )


class _CountedText(str):
    operation_count = 0

    def __hash__(self) -> int:
        type(self).operation_count += 1
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        type(self).operation_count += 1
        return super().__eq__(other)

    def __lt__(self, other: str) -> bool:
        type(self).operation_count += 1
        return super().__lt__(other)


def test_ten_thousand_record_and_contributor_dedup_is_not_pairwise() -> None:
    size = 10_000
    _CountedText.operation_count = 0
    contributor_ids = tuple(
        _CountedText(f"rank-{index:05d}") for index in reversed(range(size))
    )
    expected = ExpectedContributorSet(
        contributor_ids=contributor_ids,
        authority=_evidence("large-membership", "2"),
    )
    fingerprint = _fingerprint()
    graph_input = _graph_input(expected=expected, fingerprint=fingerprint)
    contributions = tuple(
        _contribution(
            contributor_id,
            (
                _record(
                    _CountedText(f"main.record-{index:05d}"),
                    native_name=_CountedText(f"model.record-{index:05d}"),
                    native_owner="model",
                ),
            ),
            fingerprint=fingerprint,
        )
        for index, contributor_id in enumerate(contributor_ids)
    )

    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=contributions,
    )

    assert len(partition.records) == size
    assert _CountedText.operation_count < size * 200


@pytest.mark.parametrize(
    "outer_value",
    ["", b"", bytearray(), memoryview(b"")],
    ids=("str", "bytes", "bytearray", "memoryview"),
)
@pytest.mark.parametrize(
    "boundary",
    ("contribution", "partition", "inventory", "assembly", "validator"),
)
def test_analogous_tuple_boundaries_reject_scalar_or_buffer_outer_values(
    boundary: str,
    outer_value: object,
) -> None:
    graph_input, expected, partition = _complete_pair()
    if boundary == "contribution":
        construct = lambda: replace(  # noqa: E731
            _contribution("checkpoint-index", (_record(),)),
            records=outer_value,
        )
    elif boundary == "partition":
        construct = lambda: replace(partition, records=outer_value)  # noqa: E731
    elif boundary == "inventory":
        construct = lambda: SourceDiscoveryInventory(outer_value)  # noqa: E731
    elif boundary == "assembly":
        construct = lambda: assemble_graph_discovery_partition(  # noqa: E731
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=outer_value,  # type: ignore[arg-type]
        )
    else:
        construct = lambda: validate_discovery_inventory(  # noqa: E731
            outer_value,  # type: ignore[arg-type]
            SourceDiscoveryInventory(()),
            {},
        )

    with pytest.raises(TypeError, match="non-scalar sequence"):
        construct()


@pytest.mark.parametrize(
    "boundary",
    ("contribution", "partition", "inventory", "assembly", "validator"),
)
def test_analogous_tuple_boundaries_reject_generators(boundary: str) -> None:
    graph_input, expected, partition = _complete_pair()
    contribution = _contribution("checkpoint-index", (_record(),))
    if boundary == "contribution":
        construct = lambda: replace(  # noqa: E731
            contribution,
            records=(record for record in contribution.records),
        )
    elif boundary == "partition":
        construct = lambda: replace(  # noqa: E731
            partition,
            records=(record for record in partition.records),
        )
    elif boundary == "inventory":
        construct = lambda: SourceDiscoveryInventory(  # noqa: E731
            item for item in (partition,)
        )
    elif boundary == "assembly":
        construct = lambda: assemble_graph_discovery_partition(  # noqa: E731
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=(item for item in (contribution,)),
        )
    else:
        construct = lambda: validate_discovery_inventory(  # noqa: E731
            (item for item in (graph_input,)),
            SourceDiscoveryInventory((partition,)),
            {"main": expected},
        )

    with pytest.raises(TypeError, match="sequence"):
        construct()


def test_topology_reexports_one_source_record_type_identity() -> None:
    from nemo_rl.precision_policy import topology

    assert topology.SourceDiscoveryRecord is SourceDiscoveryRecord
    assert topology.SourceRecordProvenance is SourceRecordProvenance
    assert topology.GraphTopologyInput is GraphTopologyInput
    assert topology.SourceDiscoveryInventory is SourceDiscoveryInventory


def test_precision_policy_imports_source_discovery_without_frameworks() -> None:
    code = """
import importlib.abc
import sys
from pathlib import Path
from types import ModuleType

BLOCKED = (
    'torch',
    'megatron',
    'nemo_automodel',
    'transformer_engine',
    'vllm',
    'nemo_rl.precision_policy.compiler',
)

class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(f'{name}.') for name in BLOCKED):
            raise ImportError(f'{fullname} imports are blocked')
        return None

nemo_rl_package = ModuleType('nemo_rl')
nemo_rl_package.__package__ = 'nemo_rl'
nemo_rl_package.__path__ = [str(Path.cwd() / 'nemo_rl')]
sys.modules['nemo_rl'] = nemo_rl_package
sys.meta_path.insert(0, BlockFrameworks())
import nemo_rl.precision_policy
import nemo_rl.precision_policy.source_discovery
import nemo_rl.precision_policy.topology
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_bundle_preflight_mismatch_does_not_select_an_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.precision_policy import topology

    graph_input, expected, partition = _complete_pair()
    forged = replace(
        partition,
        completeness_receipt=replace(
            partition.completeness_receipt,
            request_digest=_digest("9"),
        ),
    )

    def fail_adapter_selection():
        raise AssertionError("adapter selection ran before discovery preflight")

    monkeypatch.setattr(topology, "_default_adapters", fail_adapter_selection)
    with pytest.raises(ValueError, match="graph input digest"):
        topology.build_semantic_manifest_bundle(
            1,
            (graph_input,),
            SourceDiscoveryInventory((forged,)),
            {"main": expected},
        )


def test_whole_inventory_preflight_rejects_later_draft_before_any_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.precision_policy import topology

    main_input, main_expected, main_partition = _complete_pair()
    draft_expected = _expected(("draft-rank",), character="7")
    draft_input, _, draft_partition = _complete_pair(
        "draft.external",
        expected=draft_expected,
    )
    forged_draft = replace(
        draft_partition,
        completeness_receipt=replace(
            draft_partition.completeness_receipt,
            canonical_records_digest=_digest("9"),
        ),
    )
    adapter_requests = []

    def fail_adapter_selection() -> tuple[object, ...]:
        adapter_requests.append("requested")
        raise AssertionError("adapter selection ran before whole-inventory preflight")

    monkeypatch.setattr(topology, "_default_adapters", fail_adapter_selection)
    with pytest.raises(ValueError, match="canonical records digest"):
        topology.build_semantic_manifest_bundle(
            1,
            (main_input, draft_input),
            SourceDiscoveryInventory((main_partition, forged_draft)),
            {"main": main_expected, "draft.external": draft_expected},
        )

    assert adapter_requests == []


def _normalized_record_for_storage(
    *,
    record_id: str = "main.weight",
    dtype: CanonicalSourceDType = CanonicalSourceDType.BFLOAT16,
    shape: tuple[int, ...] = (8, 8),
    numeric_encoding: str = "plain_bfloat16",
    provenance: SourceRecordProvenance = SourceRecordProvenance.TRAINING_RUNTIME,
    source_mutability: SourceMutability = SourceMutability.MUTABLE,
) -> SourceDiscoveryRecord:
    return SourceDiscoveryRecord(
        record_id=record_id,
        graph_instance_id="main",
        source_native_name=(
            None if source_mutability is SourceMutability.ABSENT else "model.weight"
        ),
        source_native_owner_id=(
            None if source_mutability is SourceMutability.ABSENT else "model.weight"
        ),
        dtype=dtype,
        shape=shape,
        numeric_encoding=numeric_encoding,
        provenance=provenance,
        provenance_evidence=_evidence(f"{record_id}-provenance", "5"),
        source_mutability=source_mutability,
        mutability_evidence=_evidence(f"{record_id}-mutability", "6"),
    )


def _storage_bound_graph_input(
    manifest: SourceNormalizerManifest,
) -> tuple[GraphTopologyInput, ExpectedContributorSet]:
    expected = _expected()
    fingerprint = _fingerprint(character="1")
    fingerprint = replace(
        fingerprint,
        normalization_contract_digest=source_normalizer_manifest_digest(manifest),
    )
    return (
        _graph_input(fingerprint=fingerprint, expected=expected),
        expected,
    )


def test_partition_receipt_commits_normalized_views_and_native_realizations() -> None:
    record = _normalized_record_for_storage(
        record_id="main.rowwise-scale",
        dtype=CanonicalSourceDType.E8M0,
        shape=(3, 32, 29),
        numeric_encoding="mxfp8_e8m0_scale",
        provenance=SourceRecordProvenance.CHECKPOINT_STORAGE,
    )
    normalizer = SourceNormalizationContract(
        capability_id="te.mxfp8.rowwise-crop.v1",
        kind=SourceNormalizationKind.CROP,
        contract_digest=_digest("7"),
    )
    manifest = SourceNormalizerManifest(schema_version=1, contracts=(normalizer,))
    graph_input, expected = _storage_bound_graph_input(manifest)
    realization_inventory = SourceStorageRealizationInventory(
        graph_instance_id="main",
        normalizer_manifest=manifest,
        realizations=(
            SourceStorageRealization(
                realization_id="main.rowwise-scale.compact",
                graph_instance_id="main",
                output_record_id=record.record_id,
                components=(
                    SourceStorageComponent(
                        graph_instance_id="main",
                        native_component_id="main.rowwise-scale.raw",
                        source_native_name="model.weight_rowwise_scale_inv",
                        component_role="block_scales",
                        carrier_dtype=CanonicalSourceDType.UINT8,
                        physical_shape=(128, 32),
                        physical_axes=(
                            SourcePhysicalAxisSpec(
                                "flattened_m",
                                SourceNormalizedAxisExtent(
                                    normalized_axis_indices=(0, 1),
                                    divisor=1,
                                    rounding=SourceExtentRounding.EXACT,
                                    alignment=128,
                                ),
                            ),
                            SourcePhysicalAxisSpec(
                                "scale_k",
                                SourceNormalizedAxisExtent(
                                    normalized_axis_indices=(2,),
                                    divisor=1,
                                    rounding=SourceExtentRounding.EXACT,
                                    alignment=4,
                                ),
                            ),
                        ),
                        storage_encoding="uint8-carried-e8m0",
                        padding_semantics=SourcePaddingSemantics.UNSPECIFIED_IGNORED,
                        padding_fill_encoding=None,
                        permutation_id=IDENTITY_PERMUTATION_ID,
                        swizzle_id=IDENTITY_SWIZZLE_ID,
                    ),
                ),
                output_dtype=record.dtype,
                output_shape=record.shape,
                output_numeric_encoding=record.numeric_encoding,
                normalization=normalizer,
            ),
        ),
    )
    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(
            DiscoveryContribution(
                contributor_id="checkpoint-index",
                graph_instance_id="main",
                producer_fingerprint=graph_input.source_producer_fingerprint,
                records=(record,),
                storage_realizations=realization_inventory,
            ),
        ),
    )

    assert partition.records == (record,)
    assert partition.storage_realizations == realization_inventory
    assert partition.completeness_receipt.storage_realization_count == 1
    assert partition.completeness_receipt.storage_realization_set_digest.startswith(
        "sha256:"
    )
    validate_discovery_inventory(
        (graph_input,),
        SourceDiscoveryInventory((partition,)),
        {"main": expected},
    )


def test_present_record_without_native_realization_fails_during_assembly() -> None:
    record = _normalized_record_for_storage()
    manifest = _identity_manifest()
    graph_input, expected = _storage_bound_graph_input(manifest)
    empty = SourceStorageRealizationInventory(
        graph_instance_id="main",
        normalizer_manifest=manifest,
        realizations=(),
    )

    with pytest.raises(ValueError, match="present.*storage realization"):
        assemble_graph_discovery_partition(
            graph_input=graph_input,
            expected_contributors=expected,
            contributions=(
                DiscoveryContribution(
                    contributor_id="checkpoint-index",
                    graph_instance_id="main",
                    producer_fingerprint=graph_input.source_producer_fingerprint,
                    records=(record,),
                    storage_realizations=empty,
                ),
            ),
        )


def test_realization_for_unknown_or_absent_record_fails_during_assembly() -> None:
    present = _normalized_record_for_storage()
    manifest = _identity_manifest()
    graph_input, expected = _storage_bound_graph_input(manifest)
    identity = _identity_storage_for_record(present, manifest=manifest).realizations[0]
    assert isinstance(identity, SourceStorageRealization)
    absent = _normalized_record_for_storage(
        record_id="main.absent",
        source_mutability=SourceMutability.ABSENT,
    )

    for record, output_record_id, error in (
        (present, "main.unknown", "unknown output record"),
        (absent, absent.record_id, "absent record.*storage realization"),
    ):
        inventory = SourceStorageRealizationInventory(
            graph_instance_id="main",
            normalizer_manifest=manifest,
            realizations=(
                replace(
                    identity,
                    realization_id=f"{output_record_id}.identity",
                    output_record_id=output_record_id,
                ),
            ),
        )
        with pytest.raises(ValueError, match=error):
            assemble_graph_discovery_partition(
                graph_input=graph_input,
                expected_contributors=expected,
                contributions=(
                    DiscoveryContribution(
                        contributor_id="checkpoint-index",
                        graph_instance_id="main",
                        producer_fingerprint=graph_input.source_producer_fingerprint,
                        records=(record,),
                        storage_realizations=inventory,
                    ),
                ),
            )


def test_contribution_rejects_uncommitted_normalizer_manifest() -> None:
    record = _normalized_record_for_storage()
    committed_manifest = _identity_manifest()
    graph_input, _ = _storage_bound_graph_input(committed_manifest)
    other_contract = SourceNormalizationContract(
        capability_id="test.other-identity.v1",
        kind=SourceNormalizationKind.IDENTITY,
        contract_digest=_digest("9"),
    )
    other_manifest = SourceNormalizerManifest(
        schema_version=1,
        contracts=(other_contract,),
    )
    identity = _identity_storage_for_record(
        record,
        manifest=committed_manifest,
    ).realizations[0]
    assert isinstance(identity, SourceStorageRealization)
    inventory = SourceStorageRealizationInventory(
        graph_instance_id="main",
        normalizer_manifest=other_manifest,
        realizations=(replace(identity, normalization=other_contract),),
    )

    with pytest.raises(ValueError, match="manifest.*producer fingerprint"):
        DiscoveryContribution(
            contributor_id="checkpoint-index",
            graph_instance_id="main",
            producer_fingerprint=graph_input.source_producer_fingerprint,
            records=(record,),
            storage_realizations=inventory,
        )


def test_backend_derived_record_requires_one_zero_raw_derivation_witness() -> None:
    record = _normalized_record_for_storage(
        record_id="main.derived",
        provenance=SourceRecordProvenance.BACKEND_DERIVED,
    )
    derivation = SourceNormalizationContract(
        capability_id="backend.derive-cache.v1",
        kind=SourceNormalizationKind.BACKEND_DERIVATION,
        contract_digest=_digest("8"),
    )
    manifest = SourceNormalizerManifest(schema_version=1, contracts=(derivation,))
    graph_input, expected = _storage_bound_graph_input(manifest)
    inventory = SourceStorageRealizationInventory(
        graph_instance_id="main",
        normalizer_manifest=manifest,
        realizations=(
            SourceDerivedRealization(
                realization_id="main.derived.witness",
                graph_instance_id="main",
                output_record_id=record.record_id,
                output_dtype=record.dtype,
                output_shape=record.shape,
                output_numeric_encoding=record.numeric_encoding,
                derivation=derivation,
            ),
        ),
    )

    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(
            DiscoveryContribution(
                contributor_id="checkpoint-index",
                graph_instance_id="main",
                producer_fingerprint=graph_input.source_producer_fingerprint,
                records=(record,),
                storage_realizations=inventory,
            ),
        ),
    )

    assert isinstance(
        partition.storage_realizations.realizations[0],
        SourceDerivedRealization,
    )


def test_receipt_revalidation_rejects_realization_inventory_mutation() -> None:
    record = _normalized_record_for_storage()
    manifest = _identity_manifest()
    graph_input, expected = _storage_bound_graph_input(manifest)
    inventory = _identity_storage_for_record(record, manifest=manifest)
    partition = assemble_graph_discovery_partition(
        graph_input=graph_input,
        expected_contributors=expected,
        contributions=(
            DiscoveryContribution(
                contributor_id="checkpoint-index",
                graph_instance_id="main",
                producer_fingerprint=graph_input.source_producer_fingerprint,
                records=(record,),
                storage_realizations=inventory,
            ),
        ),
    )
    realization = inventory.realizations[0]
    assert isinstance(realization, SourceStorageRealization)
    changed_inventory = SourceStorageRealizationInventory(
        graph_instance_id="main",
        normalizer_manifest=manifest,
        realizations=(
            replace(realization, realization_id="main.weight.changed-identity"),
        ),
    )
    forged = replace(partition, storage_realizations=changed_inventory)

    with pytest.raises(ValueError, match="storage realization.*digest"):
        validate_discovery_inventory(
            (graph_input,),
            SourceDiscoveryInventory((forged,)),
            {"main": expected},
        )
