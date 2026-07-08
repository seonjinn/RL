# pyright: reportMissingImports=false

import dataclasses
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from examples import run_sft
from examples.prepare_sft_validation_event import (
    build_precomputed_validation_event,
    derive_validation_artifact_eligibility,
    digest_validation_event_data,
)
import nemo_rl.algorithms.sft as run_sft_sft
import nemo_rl.algorithms.sft_validation_artifact as artifact_module
from nemo_rl.algorithms.sft_validation_artifact import (
    MemoryBudget,
    PrecomputedValidationEvent,
    ValidationArtifactEligibility,
    ValidationArtifactFingerprint,
    clone_validation_event_data,
    load_validation_event,
    save_validation_event,
    tensor_content_sha256,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.data.megatron_sft_packed import megatron_sft_packed_preprocessor


class _ResponseDatasetFixture:
    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        self.task_spec = None
        self.dataset = [{"task_name": task_name}]
        self.preprocessor = None

    def processor(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {}


def _data_config_fixture() -> dict[str, object]:
    return {
        "train": {"dataset_name": "train"},
        "validation": {"dataset_name": "validation"},
        "add_bos": False,
        "add_eos": True,
        "add_generation_prompt": False,
        "max_input_seq_length": 16,
    }


def test_setup_data_skips_configured_validation_when_requested() -> None:
    loaded_config_names: list[str] = []

    def load_dataset(config: dict[str, str]) -> _ResponseDatasetFixture:
        dataset_name = config["dataset_name"]
        loaded_config_names.append(dataset_name)
        return _ResponseDatasetFixture(dataset_name)

    with patch.object(run_sft, "load_response_dataset", side_effect=load_dataset):
        dataset, val_dataset = run_sft.setup_data(
            tokenizer=object(),
            data_config=_data_config_fixture(),
            load_validation=False,
        )

    assert dataset is not None
    assert val_dataset is None
    assert loaded_config_names == ["train"]


def test_setup_data_loads_configured_validation_by_default() -> None:
    loaded_config_names: list[str] = []

    def load_dataset(config: dict[str, str]) -> _ResponseDatasetFixture:
        dataset_name = config["dataset_name"]
        loaded_config_names.append(dataset_name)
        return _ResponseDatasetFixture(dataset_name)

    with patch.object(run_sft, "load_response_dataset", side_effect=load_dataset):
        dataset, val_dataset = run_sft.setup_data(
            tokenizer=object(),
            data_config=_data_config_fixture(),
        )

    assert dataset is not None
    assert val_dataset is not None
    assert loaded_config_names == ["train", "validation"]


def _packed_validation_batch(batch_index: int) -> BatchedDataDict:
    row_ids = torch.arange(batch_index * 64, (batch_index + 1) * 64)
    input_ids = torch.stack((row_ids, row_ids + 1000), dim=1)
    return BatchedDataDict(
        input_ids=input_ids,
        target_ids=input_ids + 1,
        token_mask=torch.ones((64, 2), dtype=torch.float32),
        position_ids=torch.tensor([0, 1], dtype=torch.int64).repeat(64, 1),
        input_lengths=torch.full((64,), 2, dtype=torch.int64),
        processed_token_counts=torch.full((64,), 2, dtype=torch.int64),
        sample_mask=torch.ones(64, dtype=torch.float32),
        packed_cu_seqlens=torch.tensor([[0, 2]], dtype=torch.int32).repeat(64, 1),
        packed_cu_seqlens_lengths=torch.full((64,), 2, dtype=torch.int64),
        packed_max_seqlens=torch.full((64,), 2, dtype=torch.int64),
        idx=row_ids.tolist(),
        task_name=["megatron_sft_packed"] * 64,
    )


def _producer_config_fixture() -> SimpleNamespace:
    return SimpleNamespace(
        data={
            "validation": {"dataset_name": "megatron_sft_packed"},
            "shuffle": False,
            "num_workers": 0,
        },
        policy={
            "dynamic_batching": {"enabled": False},
            "megatron_cfg": {"enabled": True, "prepacked_sft_loss_mode": "labels"},
            "sequence_packing": {"enabled": True},
        },
        sft=SimpleNamespace(
            val_batches=4,
            val_global_batch_size=64,
            val_micro_batch_size=1,
        ),
    )


def _producer_dataset_fixture(batches: list[BatchedDataDict]) -> SimpleNamespace:
    return SimpleNamespace(
        batches=batches,
        task_data_processors={
            "megatron_sft_packed": (
                None,
                partial(megatron_sft_packed_preprocessor, prompt_format="identity"),
            )
        },
        task_data_preprocessors={},
    )


class _FixtureDataLoader:
    def __init__(self, dataset, **kwargs: object) -> None:
        assert kwargs["batch_size"] == 64
        assert kwargs["shuffle"] is False
        assert kwargs["drop_last"] is True
        self._batches = dataset.batches

    def __iter__(self):
        return iter(self._batches)


def _producer_event(
    batches: list[BatchedDataDict],
) -> PrecomputedValidationEvent:
    with patch(
        "examples.prepare_sft_validation_event.StatefulDataLoader",
        _FixtureDataLoader,
    ):
        return build_precomputed_validation_event(
            _producer_config_fixture(),
            SimpleNamespace(pad_token_id=0),
            _producer_dataset_fixture(batches),
        )


def test_producer_matches_live_packed_event_combination() -> None:
    batches = [_packed_validation_batch(batch_index) for batch_index in range(4)]
    expected_token_counts = (128, 128, 128, 128)
    live_batches = list(batches)

    produced = _producer_event(batches)
    live = run_sft_sft._combine_validation_event_batches(
        live_batches,
        global_batch_size=64,
        pad_token_id=0,
    )

    assert produced.num_valid_tokens == expected_token_counts
    assert produced.payload_digest == digest_validation_event_data(live)
    for key, value in produced.data.items():
        assert torch.equal(value, live[key])
    assert produced.data["input_ids"][:, 0].tolist() == list(range(256))


def test_producer_rejects_unknown_validation_dataset_contract() -> None:
    config = _producer_config_fixture()
    config.data["validation"] = {"dataset_name": "unknown"}

    with pytest.raises(ValueError, match="megatron_sft_packed"):
        derive_validation_artifact_eligibility(
            config,
            _producer_dataset_fixture([]),
        )


def test_repeated_production_preserves_rng_and_serialized_artifact(tmp_path) -> None:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    first = _producer_event([_packed_validation_batch(index) for index in range(4)])
    second = _producer_event([_packed_validation_batch(index) for index in range(4)])

    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)

    first_manifest = save_validation_event(
        tmp_path / "first", first, _fingerprint(), _supported_eligibility()
    )
    second_manifest = save_validation_event(
        tmp_path / "second", second, _fingerprint(), _supported_eligibility()
    )

    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    first_tensor = (
        first_manifest.parent / _manifest_content(first_manifest)["tensor_file"]
    )
    second_tensor = (
        second_manifest.parent / _manifest_content(second_manifest)["tensor_file"]
    )
    assert first_tensor.read_bytes() == second_tensor.read_bytes()


def _fingerprint() -> ValidationArtifactFingerprint:
    return ValidationArtifactFingerprint(
        dataset_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        preprocessing_sha256="c" * 64,
        nemo_rl_commit="d" * 40,
        submodule_commits=(("Megatron-LM", "e" * 40),),
        container_sha256="f" * 64,
    )


def _event_fixture(*, offset: int = 0) -> PrecomputedValidationEvent:
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(offset, offset + 6, dtype=torch.int64).reshape(
                2, 3
            ),
            "input_lengths": torch.tensor([3, 2], dtype=torch.int64),
            "token_mask": torch.tensor([[True, True, False], [True, False, False]]),
            "sample_mask": torch.ones(2, dtype=torch.float32),
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


def _supported_eligibility() -> ValidationArtifactEligibility:
    return ValidationArtifactEligibility.from_producer_facts(
        prepacked_input=True,
        raw_online_packing=False,
        stochastic_preprocessing=False,
        dynamic_batching=False,
        multimodal_data=False,
    )


def _manifest_content(manifest) -> dict[str, Any]:
    return json.loads(manifest.read_text())


def _write_manifest(manifest, content: dict[str, Any]) -> None:
    manifest.write_text(json.dumps(content))


def test_validation_artifact_round_trip_preserves_tensor_contract(tmp_path) -> None:
    event = _event_fixture()

    manifest = save_validation_event(
        tmp_path, event, _fingerprint(), _supported_eligibility()
    )
    loaded = load_validation_event(manifest, _fingerprint(), _memory_budget())

    assert loaded.num_valid_tokens == event.num_valid_tokens
    assert loaded.payload_digest == event.payload_digest
    for key in event.data:
        assert torch.equal(loaded.data[key], event.data[key])
    content = _manifest_content(manifest)
    assert content["tensor_file"].startswith("validation-")
    assert content["tensor_file"].endswith(".safetensors")


def test_validation_artifact_rejects_unknown_non_tensor_value(tmp_path) -> None:
    event = _event_fixture()
    event.data["messages"] = ["unsupported"]

    with pytest.raises(TypeError, match="tensor-only"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


@pytest.mark.parametrize(
    "field", ["dataset_sha256", "tokenizer_sha256", "preprocessing_sha256"]
)
def test_load_fails_closed_on_fingerprint_mismatch(tmp_path, field: str) -> None:
    fingerprint = _fingerprint()
    manifest = save_validation_event(
        tmp_path, _event_fixture(), fingerprint, _supported_eligibility()
    )
    changed = dataclasses.replace(fingerprint, **{field: "f" * 64})

    with pytest.raises(ValueError, match=field):
        load_validation_event(manifest, changed, _memory_budget())


def test_load_rejects_corrupted_tensor_bytes(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    tensor_path = manifest.parent / _manifest_content(manifest)["tensor_file"]
    content = bytearray(tensor_path.read_bytes())
    content[-1] ^= 1
    tensor_path.write_bytes(content)

    with pytest.raises(ValueError, match="SHA-256"):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


def test_save_rejects_non_cpu_tensor(tmp_path) -> None:
    event = _event_fixture()
    event.data["input_ids"] = torch.empty(1, device="meta")

    with pytest.raises(ValueError, match="CPU tensors only"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_save_rejects_cuda_tensor(tmp_path) -> None:
    event = _event_fixture()
    event.data["input_ids"] = torch.zeros(1, device="cuda")

    with pytest.raises(ValueError, match="CPU tensors only"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


def test_load_enforces_three_copy_memory_headroom(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )

    with pytest.raises(MemoryError, match="three-copy headroom"):
        load_validation_event(manifest, _fingerprint(), MemoryBudget(available_bytes=1))


def test_submission_clone_cannot_mutate_canonical_event() -> None:
    canonical = _event_fixture()
    submitted = clone_validation_event_data(canonical.data)
    submitted["input_ids"][0, 0] = -1

    assert canonical.data["input_ids"][0, 0].item() == 0


def test_submission_clone_rejects_unknown_sft_tensor_key() -> None:
    canonical = _event_fixture()
    canonical.data["pixel_values"] = torch.zeros((2, 3, 4, 4))

    with pytest.raises(ValueError, match="unknown SFT tensor keys.*pixel_values"):
        clone_validation_event_data(canonical.data)


def test_tensor_content_hash_is_independent_of_tensor_layout() -> None:
    contiguous = torch.arange(12, dtype=torch.int64).reshape(3, 4)

    assert tensor_content_sha256(contiguous) == tensor_content_sha256(
        contiguous.transpose(0, 1).contiguous().transpose(0, 1)
    )


def test_load_rejects_unknown_manifest_key(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content["unexpected"] = True
    _write_manifest(manifest, content)

    with pytest.raises(ValueError, match="unknown keys"):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


def test_interrupted_publish_preserves_previous_artifact_pair(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )

    with patch.object(
        artifact_module,
        "_atomic_write",
        side_effect=OSError("interrupted manifest publish"),
    ):
        with pytest.raises(OSError, match="interrupted"):
            save_validation_event(
                tmp_path,
                _event_fixture(offset=100),
                _fingerprint(),
                _supported_eligibility(),
            )

    loaded = load_validation_event(manifest, _fingerprint(), _memory_budget())
    assert loaded.data["input_ids"][0, 0].item() == 0


def test_concurrent_writers_are_serialized(tmp_path) -> None:
    original_save = artifact_module.save_safetensors_file
    state_lock = threading.Lock()
    active_writers = 0
    maximum_active_writers = 0

    def observed_save(*args, **kwargs) -> None:
        nonlocal active_writers, maximum_active_writers
        with state_lock:
            active_writers += 1
            maximum_active_writers = max(maximum_active_writers, active_writers)
        try:
            time.sleep(0.05)
            original_save(*args, **kwargs)
        finally:
            with state_lock:
                active_writers -= 1

    events = [_event_fixture(), _event_fixture(offset=100)]
    with patch.object(artifact_module, "save_safetensors_file", observed_save):
        with ThreadPoolExecutor(max_workers=2) as executor:
            manifests = list(
                executor.map(
                    lambda event: save_validation_event(
                        tmp_path, event, _fingerprint(), _supported_eligibility()
                    ),
                    events,
                )
            )

    assert maximum_active_writers == 1
    assert manifests[0] == manifests[1]
    loaded = load_validation_event(manifests[0], _fingerprint(), _memory_budget())
    assert loaded.data["input_ids"][0, 0].item() in {0, 100}


def test_forged_small_manifest_fails_memory_check_before_tensor_load(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content["retained_bytes"] = 0
    for record in content["tensors"].values():
        record["nbytes"] = 0
    _write_manifest(manifest, content)

    with patch.object(artifact_module, "load_safetensors_file") as tensor_loader:
        with pytest.raises(MemoryError, match="three-copy headroom"):
            load_validation_event(
                manifest,
                _fingerprint(),
                MemoryBudget(available_bytes=1),
            )

    tensor_loader.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("dataset_sha256", "A" * 64, id="uppercase-dataset"),
        pytest.param("tokenizer_sha256", "b" * 63, id="short-tokenizer"),
        pytest.param("preprocessing_sha256", "z" * 64, id="nonhex-preprocessing"),
        pytest.param("container_sha256", "", id="empty-container"),
    ],
)
def test_save_rejects_malformed_sha256_fingerprint(
    tmp_path, field: str, value: str
) -> None:
    fingerprint = dataclasses.replace(_fingerprint(), **{field: value})

    with pytest.raises(ValueError, match=field):
        save_validation_event(
            tmp_path, _event_fixture(), fingerprint, _supported_eligibility()
        )


@pytest.mark.parametrize("commit", ["", "g" * 40, "a" * 39])
def test_save_rejects_malformed_nemo_rl_commit(tmp_path, commit: str) -> None:
    fingerprint = dataclasses.replace(_fingerprint(), nemo_rl_commit=commit)

    with pytest.raises(ValueError, match="nemo_rl_commit"):
        save_validation_event(
            tmp_path, _event_fixture(), fingerprint, _supported_eligibility()
        )


@pytest.mark.parametrize(
    "submodules",
    [
        pytest.param((), id="empty"),
        pytest.param((("a", "e" * 40), ("a", "f" * 40)), id="duplicate-path"),
        pytest.param((("z", "e" * 40), ("a", "f" * 40)), id="unsorted"),
        pytest.param((("../escape", "e" * 40),), id="parent-path"),
        pytest.param((("/absolute", "e" * 40),), id="absolute-path"),
        pytest.param(((".", "e" * 40),), id="current-directory-path"),
        pytest.param((("module", "g" * 40),), id="invalid-commit"),
    ],
)
def test_save_rejects_invalid_recursive_submodule_entries(
    tmp_path, submodules: tuple[tuple[str, str], ...]
) -> None:
    fingerprint = dataclasses.replace(_fingerprint(), submodule_commits=submodules)

    with pytest.raises(ValueError, match="submodule_commits"):
        save_validation_event(
            tmp_path, _event_fixture(), fingerprint, _supported_eligibility()
        )


@pytest.mark.parametrize(
    "num_valid_tokens",
    [
        pytest.param((True, 1, 2, 3), id="bool"),
        pytest.param((-1, 1, 2, 3), id="negative"),
    ],
)
def test_save_rejects_nonexact_token_counts(tmp_path, num_valid_tokens) -> None:
    event = dataclasses.replace(_event_fixture(), num_valid_tokens=num_valid_tokens)

    with pytest.raises(ValueError, match="non-negative integers"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


@pytest.mark.parametrize("retained_bytes", [True, -1])
def test_save_rejects_nonexact_retained_bytes(tmp_path, retained_bytes) -> None:
    event = dataclasses.replace(_event_fixture(), retained_bytes=retained_bytes)

    with pytest.raises(ValueError, match="retained_bytes"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


@pytest.mark.parametrize(
    "budget",
    [
        pytest.param(MemoryBudget(available_bytes=True), id="bool-available"),
        pytest.param(MemoryBudget(available_bytes=-1), id="negative-available"),
        pytest.param(
            MemoryBudget(available_bytes=1_000_000, required_copy_count=True),
            id="bool-copies",
        ),
        pytest.param(
            MemoryBudget(available_bytes=1_000_000, required_copy_count=0),
            id="zero-copies",
        ),
    ],
)
def test_load_rejects_nonexact_memory_budget(tmp_path, budget: MemoryBudget) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )

    with pytest.raises(ValueError, match="MemoryBudget"):
        load_validation_event(manifest, _fingerprint(), budget)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("artifact_version", True, id="bool-version"),
        pytest.param("retained_bytes", True, id="bool-retained-bytes"),
        pytest.param("retained_bytes", -1, id="negative-retained-bytes"),
    ],
)
def test_load_rejects_nonexact_manifest_integers(
    tmp_path, field: str, value: object
) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content[field] = value
    _write_manifest(manifest, content)

    with pytest.raises(ValueError, match=field):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("nbytes", True, id="bool-bytes"),
        pytest.param("nbytes", -1, id="negative-bytes"),
        pytest.param("shape", [2, -1], id="negative-dimension"),
        pytest.param("shape", [2, True], id="bool-dimension"),
    ],
)
def test_load_rejects_nonexact_tensor_record_integers(
    tmp_path, field: str, value: object
) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content["tensors"]["input_ids"][field] = value
    _write_manifest(manifest, content)

    with patch.object(artifact_module, "load_safetensors_file") as tensor_loader:
        with pytest.raises(ValueError, match=field):
            load_validation_event(manifest, _fingerprint(), _memory_budget())

    tensor_loader.assert_not_called()


def test_save_rejects_unknown_sft_tensor_key(tmp_path) -> None:
    event = _event_fixture()
    event.data["pixel_values"] = torch.zeros((2, 3, 4, 4))

    with pytest.raises(ValueError, match="unknown SFT tensor keys.*pixel_values"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


def test_save_requires_complete_sft_tensor_schema(tmp_path) -> None:
    event = _event_fixture()
    del event.data["sample_mask"]

    with pytest.raises(ValueError, match="missing required SFT tensor keys"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


def test_save_requires_complete_packed_metadata_group(tmp_path) -> None:
    event = _event_fixture()
    event.data["packed_cu_seqlens"] = torch.tensor([[0, 3], [0, 2]], dtype=torch.int32)

    with pytest.raises(ValueError, match="packed metadata must include"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


def test_supported_producer_eligibility_round_trips(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )

    assert _manifest_content(manifest)["eligibility"] == {
        "prepacked_input": True,
        "dynamic_batching": False,
        "multimodal_data": False,
        "raw_online_packing": False,
        "stochastic_preprocessing": False,
    }
    load_validation_event(manifest, _fingerprint(), _memory_budget())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("prepacked_input", False, id="not-prepacked"),
        pytest.param("raw_online_packing", True, id="online-packing"),
        pytest.param("stochastic_preprocessing", True, id="stochastic"),
        pytest.param("dynamic_batching", True, id="dynamic-batching"),
        pytest.param("multimodal_data", True, id="multimodal"),
    ],
)
def test_save_rejects_unsupported_producer_fact_before_publication(
    tmp_path, field: str, value: bool
) -> None:
    artifact_directory = tmp_path / "artifact"
    eligibility = dataclasses.replace(_supported_eligibility(), **{field: value})

    with pytest.raises(ValueError, match="producer eligibility"):
        save_validation_event(
            artifact_directory, _event_fixture(), _fingerprint(), eligibility
        )

    assert not artifact_directory.exists()


def test_save_requires_explicit_producer_eligibility_before_publication(
    tmp_path,
) -> None:
    artifact_directory = tmp_path / "artifact"
    untyped_save: Any = save_validation_event

    with pytest.raises(TypeError, match="eligibility"):
        untyped_save(artifact_directory, _event_fixture(), _fingerprint())

    assert not artifact_directory.exists()


@pytest.mark.parametrize(
    "field",
    [
        "prepacked_input",
        "raw_online_packing",
        "stochastic_preprocessing",
        "dynamic_batching",
        "multimodal_data",
    ],
)
def test_producer_eligibility_requires_exact_booleans(field: str) -> None:
    facts: dict[str, object] = {
        "prepacked_input": True,
        "raw_online_packing": False,
        "stochastic_preprocessing": False,
        "dynamic_batching": False,
        "multimodal_data": False,
    }
    facts[field] = 1
    untyped_factory: Any = ValidationArtifactEligibility.from_producer_facts

    with pytest.raises(TypeError, match=field):
        untyped_factory(**facts)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("prepacked_input", False, id="not-prepacked"),
        pytest.param("raw_online_packing", True, id="online-packing"),
        pytest.param("stochastic_preprocessing", True, id="stochastic"),
        pytest.param("dynamic_batching", True, id="dynamic-batching"),
        pytest.param("multimodal_data", True, id="multimodal"),
    ],
)
def test_load_rejects_ineligible_artifact(tmp_path, field: str, value: object) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content["eligibility"][field] = value
    _write_manifest(manifest, content)

    with pytest.raises(ValueError, match="eligibility"):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


def test_tensor_content_sha256_handles_scalar_tensor() -> None:
    scalar = torch.tensor(7, dtype=torch.int64)

    assert (
        tensor_content_sha256(scalar)
        == hashlib.sha256(scalar.numpy().tobytes()).hexdigest()
    )
