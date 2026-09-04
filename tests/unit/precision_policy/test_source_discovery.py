from collections.abc import Mapping
from dataclasses import FrozenInstanceError, asdict, fields, replace
import os
from pickle import dumps, loads
import re
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
    DiscoveryCompletenessReceipt,
    DiscoveryContribution,
    ExpectedContributorAuthority,
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
        receipt.graph_input_digest,
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

def evidence(name, character):
    return EvidenceSource(
        kind=EvidenceSourceKind.RUNTIME_INVENTORY,
        locator=f'runtime://{name}',
        digest=f'sha256:{character * 64}',
    )

expected = ExpectedContributorSet(('rank-b', 'rank-a'), evidence('membership', '1'))
fingerprint = SourceProducerFingerprint(
    HF_SAFETENSORS_HEADER_V1,
    'checkpoint-header-reader',
    'a' * 40,
    'sha256:' + '2' * 64,
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
    CanonicalSourceDType.BFLOAT16, (8, 8),
    SourceRecordProvenance.TRAINING_RUNTIME, evidence('provenance', '6'),
    SourceMutability.MUTABLE, evidence('mutability', '7'),
)
partition = assemble_graph_discovery_partition(
    graph_input=graph_input,
    expected_contributors=expected,
    contributions=(
        DiscoveryContribution('rank-b', 'main', fingerprint, ()),
        DiscoveryContribution('rank-a', 'main', fingerprint, (record,)),
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
        "provenance",
        "provenance-evidence",
        "mutability",
        "mutability-evidence",
    ),
)
def test_canonical_record_digest_binds_every_raw_record_field(
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
        partition.completeness_receipt.graph_input_digest,
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
            graph_input_digest=graph_input_identity_digest(replacement_input),
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
        "graph_input_digest",
    )
    assert tuple(field.name for field in fields(GraphDiscoveryPartition)) == (
        "graph_instance_id",
        "producer_fingerprint",
        "expected_contributor_authority",
        "records",
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
