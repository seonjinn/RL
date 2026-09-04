from collections.abc import Mapping
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pickle import dumps, loads
import subprocess
import sys

import pytest

from nemo_rl.precision_policy.semantic import (
    EvidenceSource,
    EvidenceSourceKind,
    ExpectedGraphDeclaration,
    GraphKind,
    GraphLifecycle,
    GraphProvenance,
    RolloutParticipation,
    SourceMutability,
)
from nemo_rl.precision_policy.source_discovery import (
    HF_SAFETENSORS_HEADER_V1,
    MEGATRON_BRIDGE_STATE_DICT_V1,
    NEMO_AUTOMODEL_STATE_DICT_V1,
    TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1,
    DiscoveryContribution,
    ExpectedContributorSet,
    GraphDiscoveryPartition,
    GraphTopologyInput,
    SourceDiscoveryInventory,
    SourceDiscoveryRecord,
    SourceProducerFingerprint,
    SourceRecordProvenance,
    SourceSchemaId,
    assemble_graph_discovery_partition,
    graph_input_identity_digest,
    validate_discovery_inventory,
)
from nemo_rl.precision_policy.source_dtype import CanonicalSourceDType


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


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
    return SourceProducerFingerprint(
        schema_id=schema_id,
        producer_implementation_id=implementation_id,
        producer_revision=revision,
        normalization_contract_digest=_digest(character),
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


def _record(
    record_id: str = "main.weight",
    *,
    graph_instance_id: str = "main",
    native_name: str | None = "model.weight",
    native_owner: str | None = "model.weight",
    source_mutability: SourceMutability = SourceMutability.MUTABLE,
) -> SourceDiscoveryRecord:
    return SourceDiscoveryRecord(
        record_id=record_id,
        graph_instance_id=graph_instance_id,
        source_native_name=native_name,
        source_native_owner_id=native_owner,
        dtype=CanonicalSourceDType.BFLOAT16,
        shape=(8, 8),
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
    return DiscoveryContribution(
        contributor_id=contributor_id,
        graph_instance_id=graph_instance_id,
        producer_fingerprint=fingerprint or _fingerprint(),
        records=records,
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


@pytest.mark.parametrize(
    "contributor_ids",
    [(), ("shard-a", "shard-a"), ("",), (" shard-a",)],
)
def test_expected_contributor_set_rejects_invalid_opaque_ids(
    contributor_ids: tuple[str, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _expected(contributor_ids)


def test_expected_contributor_authority_rejects_evidence_that_leaks_ids() -> None:
    trusted = ExpectedContributorSet(
        contributor_ids=("private-pp0-tp1",),
        authority=EvidenceSource(
            kind=EvidenceSourceKind.RUNTIME_INVENTORY,
            locator="runtime://membership/private-pp0-tp1",
            digest=_digest("2"),
        ),
    )

    with pytest.raises(ValueError, match="contributor ID"):
        trusted.to_authority()


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
        "receipt_graph_input_digest",
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
        forged = replace(partition, graph_instance_id="draft.external")
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
    else:
        forged = replace(
            partition,
            completeness_receipt=replace(
                receipt,
                graph_input_digest=_digest("9"),
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
        graph_input_digest=graph_input_identity_digest(replacement_input),
    )
    replacement_partition = replace(
        partition,
        expected_contributor_authority=replacement.to_authority(),
        completeness_receipt=replacement_receipt,
    )

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

BLOCKED = ('torch', 'megatron', 'nemo_automodel', 'transformer_engine', 'vllm')

class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(f'{name}.') for name in BLOCKED):
            raise ImportError(f'{fullname} imports are blocked')
        return None

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
            graph_input_digest=_digest("9"),
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
