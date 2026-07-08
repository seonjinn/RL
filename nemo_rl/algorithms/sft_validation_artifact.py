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
"""Tensor-only storage for precomputed SFT validation events."""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_ARTIFACT_VERSION = 1
_MANIFEST_FILE_NAME = "validation.manifest.json"
_TENSOR_FILE_NAME = "validation.safetensors"
_MANIFEST_KEYS = frozenset(
    {
        "artifact_version",
        "fingerprint",
        "num_valid_tokens",
        "payload_digest",
        "retained_bytes",
        "tensor_file",
        "tensor_file_sha256",
        "tensors",
    }
)
_FINGERPRINT_KEYS = frozenset(
    {
        "container_sha256",
        "dataset_sha256",
        "nemo_rl_commit",
        "preprocessing_sha256",
        "submodule_commits",
        "tokenizer_sha256",
    }
)
_TENSOR_RECORD_KEYS = frozenset({"dtype", "nbytes", "sha256", "shape"})


@dataclass(frozen=True)
class ValidationArtifactFingerprint:
    dataset_sha256: str
    tokenizer_sha256: str
    preprocessing_sha256: str
    nemo_rl_commit: str
    submodule_commits: tuple[tuple[str, str], ...]
    container_sha256: str


@dataclass(frozen=True)
class PrecomputedValidationEvent:
    data: BatchedDataDict[Mapping[str, torch.Tensor]]
    num_valid_tokens: tuple[int, int, int, int]
    payload_digest: str
    retained_bytes: int


@dataclass(frozen=True)
class MemoryBudget:
    available_bytes: int
    required_copy_count: int = 3


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Return the SHA-256 hash of contiguous CPU tensor content bytes."""
    _require_cpu_tensor(tensor)
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def save_validation_event(
    artifact_directory: Path,
    event: PrecomputedValidationEvent,
    fingerprint: ValidationArtifactFingerprint,
) -> Path:
    """Atomically persist a tensor-only validation event and its manifest."""
    artifact_directory.mkdir(parents=True, exist_ok=True)
    tensors = _event_tensors(event.data)
    retained_bytes = sum(tensor.nbytes for tensor in tensors.values())
    _validate_event_metadata(event, retained_bytes)

    tensor_path = artifact_directory / _TENSOR_FILE_NAME
    _atomic_save_safetensors(tensor_path, tensors)
    manifest: dict[str, object] = {
        "artifact_version": _ARTIFACT_VERSION,
        "fingerprint": _fingerprint_as_manifest(fingerprint),
        "num_valid_tokens": list(event.num_valid_tokens),
        "payload_digest": event.payload_digest,
        "retained_bytes": retained_bytes,
        "tensor_file": _TENSOR_FILE_NAME,
        "tensor_file_sha256": _file_sha256(tensor_path),
        "tensors": {key: _tensor_record(tensor) for key, tensor in tensors.items()},
    }
    manifest_path = artifact_directory / _MANIFEST_FILE_NAME
    _atomic_write(manifest_path, _canonical_json_bytes(manifest))
    return manifest_path


def load_validation_event(
    manifest_path: Path,
    fingerprint: ValidationArtifactFingerprint,
    memory_budget: MemoryBudget | None = None,
) -> PrecomputedValidationEvent:
    """Load a verified validation event with owning CPU tensor copies."""
    manifest = _load_manifest(manifest_path)
    _validate_fingerprint(manifest["fingerprint"], fingerprint)
    _validate_memory_budget(
        manifest["retained_bytes"],
        memory_budget or MemoryBudget(psutil.virtual_memory().available),
    )

    tensor_path = manifest_path.parent / _TENSOR_FILE_NAME
    if not tensor_path.is_file():
        raise ValueError(f"Validation artifact tensor file is missing: {tensor_path}")
    if _file_sha256(tensor_path) != manifest["tensor_file_sha256"]:
        raise ValueError(
            "Validation artifact tensor file SHA-256 does not match manifest"
        )
    try:
        tensors = load_safetensors_file(str(tensor_path), device="cpu")
    except Exception as error:
        raise ValueError(
            "Validation artifact tensor file could not be loaded"
        ) from error

    data = _validated_loaded_data(tensors, manifest["tensors"])
    retained_bytes = sum(tensor.nbytes for tensor in data.values())
    if retained_bytes != manifest["retained_bytes"]:
        raise ValueError(
            "Validation artifact retained_bytes does not match tensor data"
        )
    num_valid_tokens = manifest["num_valid_tokens"]
    if not isinstance(num_valid_tokens, list):
        raise ValueError("Validation artifact num_valid_tokens must be a list")
    payload_digest = manifest["payload_digest"]
    if not isinstance(payload_digest, str):
        raise ValueError("Validation artifact payload_digest must be a string")
    return PrecomputedValidationEvent(
        data=data,
        num_valid_tokens=tuple(num_valid_tokens),
        payload_digest=payload_digest,
        retained_bytes=retained_bytes,
    )


def clone_validation_event_data(
    data: BatchedDataDict[Mapping[str, torch.Tensor]],
) -> BatchedDataDict[Mapping[str, torch.Tensor]]:
    """Clone event data so validation submission cannot mutate its canonical cache."""
    cloned = BatchedDataDict[Mapping[str, torch.Tensor]]()
    for key, value in data.items():
        _require_named_cpu_tensor(key, value)
        cloned[key] = _clone_tensor(value)
    return cloned


def _event_tensors(
    data: BatchedDataDict[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in sorted(data.items()):
        _require_named_cpu_tensor(key, value)
        tensors[key] = _clone_tensor(value)
    return tensors


def _validate_event_metadata(
    event: PrecomputedValidationEvent, retained_bytes: int
) -> None:
    if len(event.num_valid_tokens) != 4 or any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in event.num_valid_tokens
    ):
        raise ValueError("num_valid_tokens must contain exactly four integer counts")
    if not isinstance(event.payload_digest, str):
        raise TypeError("payload_digest must be a string")
    if event.retained_bytes != retained_bytes:
        raise ValueError("retained_bytes does not match tensor payload bytes")


def _require_named_cpu_tensor(key: object, value: object) -> None:
    if not isinstance(key, str):
        raise TypeError("Validation artifact tensor keys must be strings")
    _require_cpu_tensor(value)


def _require_cpu_tensor(value: object) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError("Validation artifact data must be tensor-only")
    if value.device.type != "cpu":
        raise ValueError("Validation artifact supports CPU tensors only")


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().clone()


def _tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "dtype": str(tensor.dtype),
        "nbytes": tensor.nbytes,
        "sha256": tensor_content_sha256(tensor),
        "shape": list(tensor.shape),
    }


def _fingerprint_as_manifest(
    fingerprint: ValidationArtifactFingerprint,
) -> dict[str, object]:
    return {
        "container_sha256": fingerprint.container_sha256,
        "dataset_sha256": fingerprint.dataset_sha256,
        "nemo_rl_commit": fingerprint.nemo_rl_commit,
        "preprocessing_sha256": fingerprint.preprocessing_sha256,
        "submodule_commits": [
            list(commit) for commit in sorted(fingerprint.submodule_commits)
        ],
        "tokenizer_sha256": fingerprint.tokenizer_sha256,
    }


def _atomic_save_safetensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(file_descriptor)
    try:
        save_safetensors_file(dict(tensors), temporary_path)
        with open(temporary_path, "rb") as file_handle:
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_write(path: Path, content: bytes) -> None:
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "wb") as file_handle:
            file_handle.write(content)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Validation artifact manifest could not be read") from error
    if not isinstance(manifest, dict):
        raise ValueError("Validation artifact manifest must be a JSON object")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    if manifest["artifact_version"] != _ARTIFACT_VERSION:
        raise ValueError("Unsupported validation artifact version")
    if manifest["tensor_file"] != _TENSOR_FILE_NAME:
        raise ValueError("Validation artifact tensor file name is invalid")
    _validate_manifest_metadata(manifest)
    return manifest


def _require_exact_keys(
    value: Mapping[str, object], expected_keys: frozenset[str], subject: str
) -> None:
    actual_keys = set(value)
    if actual_keys != expected_keys:
        unknown_keys = sorted(actual_keys - expected_keys)
        missing_keys = sorted(expected_keys - actual_keys)
        raise ValueError(
            f"Validation artifact {subject} has unknown keys {unknown_keys} "
            f"or missing keys {missing_keys}"
        )


def _validate_manifest_metadata(manifest: Mapping[str, object]) -> None:
    fingerprint = manifest["fingerprint"]
    if not isinstance(fingerprint, Mapping):
        raise ValueError("Validation artifact fingerprint must be an object")
    _require_exact_keys(fingerprint, _FINGERPRINT_KEYS, "fingerprint")
    token_counts = manifest["num_valid_tokens"]
    if not isinstance(token_counts, list) or len(token_counts) != 4:
        raise ValueError(
            "Validation artifact num_valid_tokens must contain four values"
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in token_counts
    ):
        raise ValueError("Validation artifact num_valid_tokens must contain integers")
    if not isinstance(manifest["payload_digest"], str):
        raise ValueError("Validation artifact payload_digest must be a string")
    retained_bytes = manifest["retained_bytes"]
    if not isinstance(retained_bytes, int) or retained_bytes < 0:
        raise ValueError("Validation artifact retained_bytes must be non-negative")
    if not isinstance(manifest["tensor_file_sha256"], str):
        raise ValueError("Validation artifact tensor_file_sha256 must be a string")
    tensor_records = manifest["tensors"]
    if not isinstance(tensor_records, Mapping):
        raise ValueError("Validation artifact tensors must be an object")
    for key, record in tensor_records.items():
        if not isinstance(key, str) or not isinstance(record, Mapping):
            raise ValueError("Validation artifact tensor records must be named objects")
        _require_exact_keys(record, _TENSOR_RECORD_KEYS, f"tensor record {key!r}")


def _validate_fingerprint(
    saved_fingerprint: object, expected_fingerprint: ValidationArtifactFingerprint
) -> None:
    if not isinstance(saved_fingerprint, Mapping):
        raise ValueError("Validation artifact fingerprint must be an object")
    for key, expected_value in _fingerprint_as_manifest(expected_fingerprint).items():
        if saved_fingerprint[key] != expected_value:
            raise ValueError(f"Validation artifact fingerprint mismatch for {key}")


def _validate_memory_budget(
    retained_bytes: object, memory_budget: MemoryBudget
) -> None:
    if (
        not isinstance(memory_budget.available_bytes, int)
        or memory_budget.available_bytes < 0
    ):
        raise ValueError("MemoryBudget.available_bytes must be a non-negative integer")
    copy_count = memory_budget.required_copy_count
    if not isinstance(copy_count, int) or copy_count < 1:
        raise ValueError("MemoryBudget.required_copy_count must be a positive integer")
    if not isinstance(retained_bytes, int):
        raise ValueError("Validation artifact retained_bytes must be an integer")
    required_bytes = retained_bytes * copy_count
    if memory_budget.available_bytes < required_bytes:
        label = "three" if copy_count == 3 else str(copy_count)
        raise MemoryError(
            f"Validation artifact requires {label}-copy headroom "
            f"({required_bytes} bytes), but only {memory_budget.available_bytes} bytes are available"
        )


def _validated_loaded_data(
    loaded_tensors: Mapping[str, torch.Tensor], tensor_records: object
) -> BatchedDataDict[Mapping[str, torch.Tensor]]:
    if not isinstance(tensor_records, Mapping) or set(loaded_tensors) != set(
        tensor_records
    ):
        raise ValueError("Validation artifact tensor names do not match manifest")
    data = BatchedDataDict[Mapping[str, torch.Tensor]]()
    for key in sorted(loaded_tensors):
        tensor = loaded_tensors[key]
        _require_cpu_tensor(tensor)
        record = tensor_records[key]
        if not isinstance(record, Mapping):
            raise ValueError(f"Validation artifact tensor record {key!r} is invalid")
        if (
            record["dtype"] != str(tensor.dtype)
            or record["shape"] != list(tensor.shape)
            or record["nbytes"] != tensor.nbytes
        ):
            raise ValueError(
                f"Validation artifact tensor metadata mismatch for {key!r}"
            )
        if record["sha256"] != tensor_content_sha256(tensor):
            raise ValueError(f"Validation artifact tensor SHA-256 mismatch for {key!r}")
        data[key] = _clone_tensor(tensor)
    return data
