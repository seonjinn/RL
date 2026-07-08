# pyright: reportMissingImports=false

import dataclasses
import hashlib
import json

import pytest
import torch

from nemo_rl.algorithms.sft_validation_artifact import (
    MemoryBudget,
    PrecomputedValidationEvent,
    ValidationArtifactFingerprint,
    clone_validation_event_data,
    load_validation_event,
    save_validation_event,
    tensor_content_sha256,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _fingerprint() -> ValidationArtifactFingerprint:
    return ValidationArtifactFingerprint(
        dataset_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        preprocessing_sha256="c" * 64,
        nemo_rl_commit="d" * 40,
        submodule_commits=(("Megatron-LM", "e" * 40),),
        container_sha256="f" * 64,
    )


def _event_fixture() -> PrecomputedValidationEvent:
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(6, dtype=torch.int64).reshape(2, 3),
            "loss_mask": torch.tensor([[True, True, False], [True, False, False]]),
        }
    )
    return PrecomputedValidationEvent(
        data=data,
        num_valid_tokens=(2, 1, 0, 3),
        payload_digest=hashlib.sha256(b"fixture").hexdigest(),
        retained_bytes=sum(tensor.nbytes for tensor in data.values()),
    )


def _memory_budget() -> MemoryBudget:
    return MemoryBudget(available_bytes=1_000_000)


def test_validation_artifact_round_trip_preserves_tensor_contract(tmp_path) -> None:
    event = _event_fixture()

    manifest = save_validation_event(tmp_path, event, _fingerprint())
    loaded = load_validation_event(manifest, _fingerprint(), _memory_budget())

    assert loaded.num_valid_tokens == event.num_valid_tokens
    assert loaded.payload_digest == event.payload_digest
    for key in event.data:
        assert torch.equal(loaded.data[key], event.data[key])


def test_validation_artifact_rejects_unknown_non_tensor_value(tmp_path) -> None:
    event = _event_fixture()
    event.data["messages"] = ["unsupported"]

    with pytest.raises(TypeError, match="tensor-only"):
        save_validation_event(tmp_path, event, _fingerprint())


@pytest.mark.parametrize(
    "field", ["dataset_sha256", "tokenizer_sha256", "preprocessing_sha256"]
)
def test_load_fails_closed_on_fingerprint_mismatch(tmp_path, field: str) -> None:
    fingerprint = _fingerprint()
    manifest = save_validation_event(tmp_path, _event_fixture(), fingerprint)
    changed = dataclasses.replace(fingerprint, **{field: "f" * 64})

    with pytest.raises(ValueError, match=field):
        load_validation_event(manifest, changed, _memory_budget())


def test_load_rejects_corrupted_tensor_bytes(tmp_path) -> None:
    manifest = save_validation_event(tmp_path, _event_fixture(), _fingerprint())
    tensor_path = manifest.parent / "validation.safetensors"
    content = bytearray(tensor_path.read_bytes())
    content[-1] ^= 1
    tensor_path.write_bytes(content)

    with pytest.raises(ValueError, match="SHA-256"):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


def test_save_rejects_non_cpu_tensor(tmp_path) -> None:
    event = _event_fixture()
    event.data["input_ids"] = torch.empty(1, device="meta")

    with pytest.raises(ValueError, match="CPU tensors only"):
        save_validation_event(tmp_path, event, _fingerprint())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_save_rejects_cuda_tensor(tmp_path) -> None:
    event = _event_fixture()
    event.data["input_ids"] = torch.zeros(1, device="cuda")

    with pytest.raises(ValueError, match="CPU tensors only"):
        save_validation_event(tmp_path, event, _fingerprint())


def test_load_enforces_three_copy_memory_headroom(tmp_path) -> None:
    manifest = save_validation_event(tmp_path, _event_fixture(), _fingerprint())

    with pytest.raises(MemoryError, match="three-copy headroom"):
        load_validation_event(manifest, _fingerprint(), MemoryBudget(available_bytes=1))


def test_submission_clone_cannot_mutate_canonical_event() -> None:
    canonical = _event_fixture()
    submitted = clone_validation_event_data(canonical.data)
    submitted["input_ids"][0, 0] = -1

    assert canonical.data["input_ids"][0, 0].item() == 0


def test_tensor_content_hash_is_independent_of_tensor_layout() -> None:
    contiguous = torch.arange(12, dtype=torch.int64).reshape(3, 4)

    assert tensor_content_sha256(contiguous) == tensor_content_sha256(
        contiguous.transpose(0, 1).contiguous().transpose(0, 1)
    )


def test_load_rejects_unknown_manifest_key(tmp_path) -> None:
    manifest = save_validation_event(tmp_path, _event_fixture(), _fingerprint())
    content = json.loads(manifest.read_text())
    content["unexpected"] = True
    manifest.write_text(json.dumps(content))

    with pytest.raises(ValueError, match="unknown keys"):
        load_validation_event(manifest, _fingerprint(), _memory_budget())
