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
"""Storage for deterministic precomputed SFT validation events."""

import fcntl
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from typing_extensions import Self

import psutil
import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_ARTIFACT_VERSION = 3
_MANIFEST_FILE_NAME = "validation.manifest.json"
_WRITER_LOCK_FILE_NAME = ".validation-artifact.lock"
_TENSOR_FILE_PATTERN = re.compile(r"validation-([0-9a-f]{64})\.safetensors")
_LOWERCASE_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_ID_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_MAX_SAFETENSORS_HEADER_BYTES = 1024 * 1024
_MANIFEST_KEYS = frozenset(
    {
        "artifact_version",
        "eligibility",
        "fingerprint",
        "metadata",
        "num_valid_tokens",
        "payload_digest",
        "retained_bytes",
        "tensor_file",
        "tensor_file_sha256",
        "tensors",
    }
)
_ELIGIBILITY_KEYS = frozenset(
    {
        "dynamic_batching",
        "multimodal_data",
        "prepacked_input",
        "raw_online_packing",
        "stochastic_preprocessing",
    }
)
_REQUIRED_SFT_TENSOR_KEYS = frozenset(
    {"input_ids", "input_lengths", "sample_mask", "token_mask"}
)
_OPTIONAL_SFT_TENSOR_KEYS = frozenset({"position_ids", "target_ids"})
_OPTIONAL_BATCH_SFT_TENSOR_KEYS = frozenset({"processed_token_counts"})
_PACKED_SFT_TENSOR_KEYS = frozenset(
    {"packed_cu_seqlens", "packed_cu_seqlens_lengths", "packed_max_seqlens"}
)
_ALLOWED_SFT_TENSOR_KEYS = (
    _REQUIRED_SFT_TENSOR_KEYS
    | _OPTIONAL_SFT_TENSOR_KEYS
    | _OPTIONAL_BATCH_SFT_TENSOR_KEYS
    | _PACKED_SFT_TENSOR_KEYS
)
_ALLOWED_SFT_METADATA_KEYS = frozenset({"idx", "task_name"})
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
_TORCH_TO_SAFETENSORS_DTYPE = {
    "torch.bool": "BOOL",
    "torch.uint8": "U8",
    "torch.int8": "I8",
    "torch.int16": "I16",
    "torch.uint16": "U16",
    "torch.int32": "I32",
    "torch.uint32": "U32",
    "torch.int64": "I64",
    "torch.uint64": "U64",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float32": "F32",
    "torch.float64": "F64",
    "torch.complex64": "C64",
    "torch.complex128": "C128",
}
_SAFETENSORS_DTYPE_NBYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}


@dataclass(frozen=True)
class ValidationArtifactEligibility:
    prepacked_input: bool
    raw_online_packing: bool
    stochastic_preprocessing: bool
    dynamic_batching: bool
    multimodal_data: bool

    def __post_init__(self) -> None:
        for field_name in _ELIGIBILITY_KEYS:
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(
                    f"Validation artifact producer fact {field_name} must be a boolean"
                )

    @classmethod
    def from_producer_facts(
        cls,
        *,
        prepacked_input: bool,
        raw_online_packing: bool,
        stochastic_preprocessing: bool,
        dynamic_batching: bool,
        multimodal_data: bool,
    ) -> Self:
        """Create eligibility evidence from explicit producer facts."""
        return cls(
            prepacked_input=prepacked_input,
            raw_online_packing=raw_online_packing,
            stochastic_preprocessing=stochastic_preprocessing,
            dynamic_batching=dynamic_batching,
            multimodal_data=multimodal_data,
        )


_SUPPORTED_ELIGIBILITY = ValidationArtifactEligibility.from_producer_facts(
    prepacked_input=True,
    raw_online_packing=False,
    stochastic_preprocessing=False,
    dynamic_batching=False,
    multimodal_data=False,
)


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
    data: BatchedDataDict[Any]
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
        tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def save_validation_event(
    artifact_directory: Path,
    event: PrecomputedValidationEvent,
    fingerprint: ValidationArtifactFingerprint,
    eligibility: ValidationArtifactEligibility,
) -> Path:
    """Atomically persist a validation event and its manifest."""
    _validate_producer_eligibility(eligibility)
    _validate_fingerprint_semantics(fingerprint)
    tensors = _event_tensors(event.data)
    metadata = _event_metadata(event.data, tensors["input_ids"].shape[0])
    retained_bytes = sum(tensor.nbytes for tensor in tensors.values())
    _validate_event_metadata(event, retained_bytes)

    artifact_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_directory / _MANIFEST_FILE_NAME
    with _serialized_writer(artifact_directory):
        tensor_file, tensor_file_sha256 = _publish_safetensors(
            artifact_directory, tensors
        )
        manifest: dict[str, object] = {
            "artifact_version": _ARTIFACT_VERSION,
            "eligibility": _eligibility_as_manifest(eligibility),
            "fingerprint": _fingerprint_as_manifest(fingerprint),
            "metadata": metadata,
            "num_valid_tokens": list(event.num_valid_tokens),
            "payload_digest": event.payload_digest,
            "retained_bytes": retained_bytes,
            "tensor_file": tensor_file,
            "tensor_file_sha256": tensor_file_sha256,
            "tensors": {key: _tensor_record(tensor) for key, tensor in tensors.items()},
        }
        _atomic_write(manifest_path, _canonical_json_bytes(manifest))
        _fsync_directory(artifact_directory)
    return manifest_path


def load_validation_event(
    manifest_path: Path,
    fingerprint: ValidationArtifactFingerprint,
    memory_budget: MemoryBudget | None = None,
) -> PrecomputedValidationEvent:
    """Load a verified validation event with owning CPU tensor copies."""
    manifest = _load_manifest(manifest_path)
    _validate_fingerprint(manifest["fingerprint"], fingerprint)

    tensor_file = manifest["tensor_file"]
    if not isinstance(tensor_file, str):
        raise ValueError("Validation artifact tensor_file must be a string")
    tensor_path = manifest_path.parent / tensor_file
    if not tensor_path.is_file():
        raise ValueError(f"Validation artifact tensor file is missing: {tensor_path}")
    if _file_sha256(tensor_path) != manifest["tensor_file_sha256"]:
        raise ValueError(
            "Validation artifact tensor file SHA-256 does not match manifest"
        )

    header_records, header_payload_bytes, tensor_file_bytes = _read_safetensors_header(
        tensor_path
    )
    manifest_payload_bytes = _manifest_tensor_payload_bytes(manifest["tensors"])
    retained_bytes = manifest["retained_bytes"]
    if (
        not isinstance(retained_bytes, int)
        or isinstance(retained_bytes, bool)
        or retained_bytes < 0
    ):
        raise ValueError("Validation artifact retained_bytes must be non-negative")
    conservative_payload_bytes = max(
        retained_bytes,
        manifest_payload_bytes,
        header_payload_bytes,
        tensor_file_bytes,
    )
    _validate_memory_budget(
        conservative_payload_bytes,
        memory_budget or MemoryBudget(psutil.virtual_memory().available),
    )
    _validate_header_against_manifest(header_records, manifest["tensors"])
    if retained_bytes != header_payload_bytes:
        raise ValueError(
            "Validation artifact retained_bytes does not match safetensors payload"
        )

    try:
        tensors = load_safetensors_file(str(tensor_path), device="cpu")
    except Exception as error:
        raise ValueError(
            "Validation artifact tensor file could not be loaded"
        ) from error

    data = _validated_loaded_data(tensors, manifest["tensors"], manifest["metadata"])
    loaded_retained_bytes = sum(
        value.nbytes for value in data.values() if isinstance(value, torch.Tensor)
    )
    if loaded_retained_bytes != retained_bytes:
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
        retained_bytes=loaded_retained_bytes,
    )


def clone_validation_event_data(
    data: BatchedDataDict[Any],
) -> BatchedDataDict[Any]:
    """Clone event data so validation submission cannot mutate its canonical cache."""
    cloned = BatchedDataDict[Any]()
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            _require_named_cpu_tensor(key, value)
            cloned[key] = _clone_tensor(value)
        elif key in _ALLOWED_SFT_METADATA_KEYS and isinstance(value, list):
            cloned[key] = list(value)
        else:
            _require_named_cpu_tensor(key, value)
    tensors = {key: value for key, value in cloned.items() if torch.is_tensor(value)}
    _validate_sft_tensor_schema(tensors)
    _validate_sft_metadata(
        {key: value for key, value in cloned.items() if isinstance(value, list)},
        tensors["input_ids"].shape[0],
    )
    return cloned


def _event_tensors(
    data: BatchedDataDict[Any],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in sorted(data.items()):
        if isinstance(value, torch.Tensor):
            _require_named_cpu_tensor(key, value)
            tensors[key] = _clone_tensor(value)
        elif key not in _ALLOWED_SFT_METADATA_KEYS:
            _require_named_cpu_tensor(key, value)
    _validate_sft_tensor_schema(tensors)
    return tensors


def _event_metadata(
    data: BatchedDataDict[Any], batch_size: int
) -> dict[str, list[Any]]:
    metadata = {
        key: list(value)
        for key, value in sorted(data.items())
        if key in _ALLOWED_SFT_METADATA_KEYS and isinstance(value, list)
    }
    missing_or_invalid = [
        key
        for key in _ALLOWED_SFT_METADATA_KEYS & data.keys()
        if not isinstance(data[key], list)
    ]
    if missing_or_invalid:
        raise TypeError(
            "Validation artifact list metadata must use lists: "
            f"{sorted(missing_or_invalid)}"
        )
    _validate_sft_metadata(metadata, batch_size)
    return metadata


def _validate_producer_eligibility(eligibility: object) -> None:
    if not isinstance(eligibility, ValidationArtifactEligibility):
        raise TypeError(
            "Validation artifact eligibility must be a "
            "ValidationArtifactEligibility value"
        )
    for field_name in _ELIGIBILITY_KEYS:
        if type(getattr(eligibility, field_name)) is not bool:
            raise TypeError(
                f"Validation artifact producer fact {field_name} must be a boolean"
            )
    if eligibility != _SUPPORTED_ELIGIBILITY:
        raise ValueError(
            "Validation artifact producer eligibility is not supported; only "
            "deterministic prepacked text SFT data may be published"
        )


def _eligibility_as_manifest(
    eligibility: ValidationArtifactEligibility,
) -> dict[str, bool]:
    return {
        "dynamic_batching": eligibility.dynamic_batching,
        "multimodal_data": eligibility.multimodal_data,
        "prepacked_input": eligibility.prepacked_input,
        "raw_online_packing": eligibility.raw_online_packing,
        "stochastic_preprocessing": eligibility.stochastic_preprocessing,
    }


def _eligibility_from_manifest(value: object) -> ValidationArtifactEligibility:
    if not isinstance(value, Mapping):
        raise ValueError("Validation artifact eligibility must be an object")
    _require_exact_keys(value, _ELIGIBILITY_KEYS, "eligibility")
    try:
        eligibility = ValidationArtifactEligibility.from_producer_facts(
            prepacked_input=value["prepacked_input"],
            raw_online_packing=value["raw_online_packing"],
            stochastic_preprocessing=value["stochastic_preprocessing"],
            dynamic_batching=value["dynamic_batching"],
            multimodal_data=value["multimodal_data"],
        )
    except TypeError as error:
        raise ValueError(
            "Validation artifact eligibility facts must be booleans"
        ) from error
    try:
        _validate_producer_eligibility(eligibility)
    except (TypeError, ValueError) as error:
        raise ValueError("Validation artifact eligibility is not supported") from error
    return eligibility


def _validate_event_metadata(
    event: PrecomputedValidationEvent, retained_bytes: int
) -> None:
    if (
        not isinstance(event.num_valid_tokens, tuple)
        or len(event.num_valid_tokens) != 4
    ):
        raise ValueError(
            "num_valid_tokens must contain exactly four non-negative integers"
        )
    if any(not _is_nonnegative_int(value) for value in event.num_valid_tokens):
        raise ValueError(
            "num_valid_tokens must contain exactly four non-negative integers"
        )
    if not isinstance(event.payload_digest, str):
        raise TypeError("payload_digest must be a string")
    if not _is_nonnegative_int(event.retained_bytes):
        raise ValueError("retained_bytes must be a non-negative integer")
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


def _validate_sft_tensor_schema(tensors: Mapping[str, torch.Tensor]) -> None:
    keys = set(tensors)
    _validate_sft_tensor_keys(keys)
    _validate_sft_tensor_shapes(
        {key: tuple(tensor.shape) for key, tensor in tensors.items()}
    )


def _validate_sft_tensor_shapes(shapes: Mapping[str, tuple[int, ...]]) -> None:
    input_ids_shape = shapes["input_ids"]
    if len(input_ids_shape) != 2 or input_ids_shape[0] == 0:
        raise ValueError("Validation artifact input_ids must be a nonempty 2D tensor")
    batch_size, sequence_length = input_ids_shape
    expected_shapes = {
        "input_lengths": (batch_size,),
        "sample_mask": (batch_size,),
        "token_mask": (batch_size, sequence_length),
    }
    for key in _OPTIONAL_SFT_TENSOR_KEYS & shapes.keys():
        expected_shapes[key] = (batch_size, sequence_length)
    for key in _OPTIONAL_BATCH_SFT_TENSOR_KEYS & shapes.keys():
        expected_shapes[key] = (batch_size,)
    packed_keys = shapes.keys() & _PACKED_SFT_TENSOR_KEYS
    if packed_keys:
        expected_shapes["packed_cu_seqlens_lengths"] = (batch_size,)
        expected_shapes["packed_max_seqlens"] = (batch_size,)
        packed_cu_seqlens_shape = shapes["packed_cu_seqlens"]
        if (
            len(packed_cu_seqlens_shape) != 2
            or packed_cu_seqlens_shape[0] != batch_size
        ):
            raise ValueError(
                "Validation artifact packed_cu_seqlens must be a batch-aligned 2D tensor"
            )
    for key, expected_shape in expected_shapes.items():
        if shapes[key] != expected_shape:
            raise ValueError(
                f"Validation artifact tensor {key!r} has shape "
                f"{shapes[key]}; expected {expected_shape}"
            )


def _validate_sft_metadata(metadata: object, batch_size: int) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("Validation artifact metadata must be an object")
    unknown_keys = sorted(set(metadata) - _ALLOWED_SFT_METADATA_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Validation artifact has unknown SFT metadata keys: {unknown_keys}"
        )
    for key, values in metadata.items():
        if not isinstance(key, str) or not isinstance(values, list):
            raise ValueError("Validation artifact metadata entries must be named lists")
        if len(values) != batch_size:
            raise ValueError(
                f"Validation artifact metadata {key!r} has length {len(values)}; "
                f"expected {batch_size}"
            )
        if key == "idx" and any(type(value) is not int for value in values):
            raise ValueError("Validation artifact idx metadata must contain integers")
        if key == "task_name" and any(
            value is not None and not isinstance(value, str) for value in values
        ):
            raise ValueError(
                "Validation artifact task_name metadata must contain strings or nulls"
            )


def _validate_sft_tensor_keys(keys: set[str]) -> None:
    unknown_keys = sorted(keys - _ALLOWED_SFT_TENSOR_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Validation artifact has unknown SFT tensor keys: {unknown_keys}"
        )
    missing_keys = sorted(_REQUIRED_SFT_TENSOR_KEYS - keys)
    if missing_keys:
        raise ValueError(
            f"Validation artifact is missing required SFT tensor keys: {missing_keys}"
        )
    packed_keys = keys & _PACKED_SFT_TENSOR_KEYS
    if packed_keys and packed_keys != _PACKED_SFT_TENSOR_KEYS:
        raise ValueError(
            "Validation artifact packed metadata must include exactly "
            f"{sorted(_PACKED_SFT_TENSOR_KEYS)}"
        )


def _validate_fingerprint_semantics(
    fingerprint: ValidationArtifactFingerprint,
) -> None:
    for field_name in (
        "dataset_sha256",
        "tokenizer_sha256",
        "preprocessing_sha256",
        "container_sha256",
    ):
        value = getattr(fingerprint, field_name)
        if (
            not isinstance(value, str)
            or _LOWERCASE_SHA256_PATTERN.fullmatch(value) is None
        ):
            raise ValueError(
                f"Validation artifact fingerprint {field_name} must be a "
                "64-character lowercase SHA-256"
            )
    if (
        not isinstance(fingerprint.nemo_rl_commit, str)
        or _COMMIT_ID_PATTERN.fullmatch(fingerprint.nemo_rl_commit) is None
    ):
        raise ValueError(
            "Validation artifact fingerprint nemo_rl_commit must be a full "
            "lowercase hexadecimal commit ID"
        )
    submodule_commits = fingerprint.submodule_commits
    if not isinstance(submodule_commits, tuple) or not submodule_commits:
        raise ValueError(
            "Validation artifact fingerprint submodule_commits must be nonempty"
        )
    paths: set[str] = set()
    for entry in submodule_commits:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError(
                "Validation artifact fingerprint submodule_commits entries must be pairs"
            )
        path, commit = entry
        if not _is_valid_submodule_path(path):
            raise ValueError(
                "Validation artifact fingerprint submodule_commits contains an "
                f"invalid path: {path!r}"
            )
        if path in paths:
            raise ValueError(
                "Validation artifact fingerprint submodule_commits paths must be unique"
            )
        paths.add(path)
        if not isinstance(commit, str) or _COMMIT_ID_PATTERN.fullmatch(commit) is None:
            raise ValueError(
                "Validation artifact fingerprint submodule_commits contains an "
                f"invalid commit for {path!r}"
            )
    if tuple(sorted(submodule_commits)) != submodule_commits:
        raise ValueError(
            "Validation artifact fingerprint submodule_commits must be sorted"
        )


def _is_valid_submodule_path(path: object) -> bool:
    if (
        not isinstance(path, str)
        or path in {"", ".", ".."}
        or "\\" in path
        or "\x00" in path
    ):
        return False
    parsed = PurePosixPath(path)
    return (
        not parsed.is_absolute()
        and str(parsed) == path
        and all(part not in {"", ".", ".."} for part in parsed.parts)
    )


def _fingerprint_as_manifest(
    fingerprint: ValidationArtifactFingerprint,
) -> dict[str, object]:
    return {
        "container_sha256": fingerprint.container_sha256,
        "dataset_sha256": fingerprint.dataset_sha256,
        "nemo_rl_commit": fingerprint.nemo_rl_commit,
        "preprocessing_sha256": fingerprint.preprocessing_sha256,
        "submodule_commits": [list(commit) for commit in fingerprint.submodule_commits],
        "tokenizer_sha256": fingerprint.tokenizer_sha256,
    }


@contextmanager
def _serialized_writer(artifact_directory: Path) -> Generator[None, None, None]:
    lock_path = artifact_directory / _WRITER_LOCK_FILE_NAME
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _publish_safetensors(
    artifact_directory: Path, tensors: Mapping[str, torch.Tensor]
) -> tuple[str, str]:
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=artifact_directory,
        prefix=".validation-safetensors.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    try:
        save_safetensors_file(dict(tensors), temporary_path)
        with open(temporary_path, "rb") as file_handle:
            os.fsync(file_handle.fileno())
        tensor_file_sha256 = _file_sha256(Path(temporary_path))
        tensor_file = f"validation-{tensor_file_sha256}.safetensors"
        tensor_path = artifact_directory / tensor_file
        if tensor_path.exists():
            if _file_sha256(tensor_path) != tensor_file_sha256:
                raise ValueError(
                    "Content-addressed validation tensor file has invalid content"
                )
        else:
            os.replace(temporary_path, tensor_path)
            _fsync_directory(artifact_directory)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return tensor_file, tensor_file_sha256


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


def _fsync_directory(directory: Path) -> None:
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_safetensors_header(
    tensor_path: Path,
) -> tuple[dict[str, dict[str, object]], int, int]:
    tensor_file_bytes = tensor_path.stat().st_size
    if tensor_file_bytes < 8:
        raise ValueError("Validation artifact safetensors file is too small")
    with tensor_path.open("rb") as file_handle:
        header_length_bytes = file_handle.read(8)
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        if (
            header_length == 0
            or header_length > _MAX_SAFETENSORS_HEADER_BYTES
            or header_length > tensor_file_bytes - 8
        ):
            raise ValueError("Validation artifact safetensors header length is invalid")
        header_bytes = file_handle.read(header_length)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Validation artifact safetensors header is invalid") from error
    if not isinstance(header, dict) or "__metadata__" in header:
        raise ValueError(
            "Validation artifact safetensors header must contain tensors only"
        )

    records: dict[str, dict[str, object]] = {}
    intervals: list[tuple[int, int]] = []
    for key, value in header.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("Validation artifact safetensors header record is invalid")
        _require_exact_keys(
            value,
            frozenset({"data_offsets", "dtype", "shape"}),
            f"safetensors header record {key!r}",
        )
        dtype = value["dtype"]
        shape = value["shape"]
        offsets = value["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _SAFETENSORS_DTYPE_NBYTES:
            raise ValueError(
                f"Validation artifact safetensors header dtype for {key!r} is invalid"
            )
        if not isinstance(shape, list) or any(
            not _is_nonnegative_int(dimension) for dimension in shape
        ):
            raise ValueError(
                f"Validation artifact safetensors header shape for {key!r} is invalid"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not _is_nonnegative_int(offset) for offset in offsets)
        ):
            raise ValueError(
                f"Validation artifact safetensors header offsets for {key!r} are invalid"
            )
        start, end = offsets
        if end < start:
            raise ValueError(
                f"Validation artifact safetensors header offsets for {key!r} are invalid"
            )
        nbytes = end - start
        expected_nbytes = math.prod(shape) * _SAFETENSORS_DTYPE_NBYTES[dtype]
        if nbytes != expected_nbytes:
            raise ValueError(
                f"Validation artifact safetensors header nbytes for {key!r} is invalid"
            )
        records[key] = {"dtype": dtype, "nbytes": nbytes, "shape": shape}
        intervals.append((start, end))

    payload_bytes = tensor_file_bytes - 8 - header_length
    previous_end = 0
    for start, end in sorted(intervals):
        if start != previous_end:
            raise ValueError(
                "Validation artifact safetensors data offsets are not contiguous"
            )
        previous_end = end
    if previous_end != payload_bytes:
        raise ValueError(
            "Validation artifact safetensors payload size does not match its header"
        )
    return records, payload_bytes, tensor_file_bytes


def _validate_header_against_manifest(
    header_records: Mapping[str, Mapping[str, object]], tensor_records: object
) -> None:
    if not isinstance(tensor_records, Mapping) or set(header_records) != set(
        tensor_records
    ):
        raise ValueError(
            "Validation artifact tensor names do not match safetensors header"
        )
    for key, header_record in header_records.items():
        manifest_record = tensor_records[key]
        if not isinstance(manifest_record, Mapping):
            raise ValueError(f"Validation artifact tensor record {key!r} is invalid")
        manifest_dtype = manifest_record["dtype"]
        expected_safetensors_dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(manifest_dtype)
        if expected_safetensors_dtype != header_record["dtype"]:
            raise ValueError(
                f"Validation artifact tensor dtype for {key!r} does not match header"
            )
        if manifest_record["shape"] != header_record["shape"]:
            raise ValueError(
                f"Validation artifact tensor shape for {key!r} does not match header"
            )
        if manifest_record["nbytes"] != header_record["nbytes"]:
            raise ValueError(
                f"Validation artifact tensor nbytes for {key!r} does not match header"
            )
        if not _is_nonnegative_int(header_record["nbytes"]):
            raise ValueError(
                f"Validation artifact tensor nbytes for {key!r} must be non-negative"
            )


def _manifest_tensor_payload_bytes(tensor_records: object) -> int:
    if not isinstance(tensor_records, Mapping):
        raise ValueError("Validation artifact tensors must be an object")
    payload_bytes = 0
    for key, record in tensor_records.items():
        if not isinstance(record, Mapping) or not _is_nonnegative_int(record["nbytes"]):
            raise ValueError(
                f"Validation artifact tensor record {key!r} nbytes must be non-negative"
            )
        payload_bytes += record["nbytes"]
    return payload_bytes


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Validation artifact manifest could not be read") from error
    if not isinstance(manifest, dict):
        raise ValueError("Validation artifact manifest must be a JSON object")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    if type(manifest["artifact_version"]) is not int:
        raise ValueError("Validation artifact artifact_version must be an integer")
    if manifest["artifact_version"] != _ARTIFACT_VERSION:
        raise ValueError("Unsupported validation artifact version")
    tensor_file = manifest["tensor_file"]
    tensor_file_sha256 = manifest["tensor_file_sha256"]
    if not isinstance(tensor_file, str) or not isinstance(tensor_file_sha256, str):
        raise ValueError("Validation artifact tensor file name is invalid")
    tensor_file_match = _TENSOR_FILE_PATTERN.fullmatch(tensor_file)
    if tensor_file_match is None or tensor_file_match.group(1) != tensor_file_sha256:
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
    _fingerprint_from_manifest(manifest["fingerprint"])
    _eligibility_from_manifest(manifest["eligibility"])
    token_counts = manifest["num_valid_tokens"]
    if not isinstance(token_counts, list) or len(token_counts) != 4:
        raise ValueError(
            "Validation artifact num_valid_tokens must contain four values"
        )
    if any(not _is_nonnegative_int(value) for value in token_counts):
        raise ValueError(
            "Validation artifact num_valid_tokens must contain non-negative integers"
        )
    if not isinstance(manifest["payload_digest"], str):
        raise ValueError("Validation artifact payload_digest must be a string")
    retained_bytes = manifest["retained_bytes"]
    if not _is_nonnegative_int(retained_bytes):
        raise ValueError("Validation artifact retained_bytes must be non-negative")
    _require_sha256(manifest["tensor_file_sha256"], "tensor_file_sha256")
    tensor_records = manifest["tensors"]
    if not isinstance(tensor_records, Mapping):
        raise ValueError("Validation artifact tensors must be an object")
    _validate_sft_tensor_keys(set(tensor_records))
    for key, record in tensor_records.items():
        if not isinstance(key, str) or not isinstance(record, Mapping):
            raise ValueError("Validation artifact tensor records must be named objects")
        _require_exact_keys(record, _TENSOR_RECORD_KEYS, f"tensor record {key!r}")
        if not isinstance(record["dtype"], str) or not record["dtype"]:
            raise ValueError(
                f"Validation artifact tensor record {key!r} dtype must be a string"
            )
        if not _is_nonnegative_int(record["nbytes"]):
            raise ValueError(
                f"Validation artifact tensor record {key!r} nbytes must be non-negative"
            )
        _require_sha256(record["sha256"], f"tensor record {key!r} sha256")
        shape = record["shape"]
        if not isinstance(shape, list) or any(
            not _is_nonnegative_int(dimension) for dimension in shape
        ):
            raise ValueError(
                f"Validation artifact tensor record {key!r} shape must contain "
                "non-negative integers"
            )
    _validate_sft_tensor_shapes(
        {
            key: tuple(record["shape"])
            for key, record in tensor_records.items()
            if isinstance(key, str) and isinstance(record, Mapping)
        }
    )
    input_ids_record = tensor_records["input_ids"]
    if not isinstance(input_ids_record, Mapping):
        raise ValueError("Validation artifact input_ids tensor record is invalid")
    input_ids_shape = input_ids_record["shape"]
    if not isinstance(input_ids_shape, list) or not input_ids_shape:
        raise ValueError("Validation artifact input_ids tensor shape is invalid")
    _validate_sft_metadata(manifest["metadata"], input_ids_shape[0])


def _validate_fingerprint(
    saved_fingerprint: object, expected_fingerprint: ValidationArtifactFingerprint
) -> None:
    saved = _fingerprint_from_manifest(saved_fingerprint)
    _validate_fingerprint_semantics(expected_fingerprint)
    for key in _FINGERPRINT_KEYS:
        if getattr(saved, key) != getattr(expected_fingerprint, key):
            raise ValueError(f"Validation artifact fingerprint mismatch for {key}")


def _fingerprint_from_manifest(value: object) -> ValidationArtifactFingerprint:
    if not isinstance(value, Mapping):
        raise ValueError("Validation artifact fingerprint must be an object")
    _require_exact_keys(value, _FINGERPRINT_KEYS, "fingerprint")
    submodule_value = value["submodule_commits"]
    if not isinstance(submodule_value, list):
        raise ValueError(
            "Validation artifact fingerprint submodule_commits must be a list"
        )
    submodule_commits: list[tuple[str, str]] = []
    for entry in submodule_value:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not all(isinstance(item, str) for item in entry)
        ):
            raise ValueError(
                "Validation artifact fingerprint submodule_commits entries must be pairs"
            )
        submodule_commits.append((entry[0], entry[1]))
    fingerprint = ValidationArtifactFingerprint(
        dataset_sha256=_require_string(value["dataset_sha256"], "dataset_sha256"),
        tokenizer_sha256=_require_string(value["tokenizer_sha256"], "tokenizer_sha256"),
        preprocessing_sha256=_require_string(
            value["preprocessing_sha256"], "preprocessing_sha256"
        ),
        nemo_rl_commit=_require_string(value["nemo_rl_commit"], "nemo_rl_commit"),
        submodule_commits=tuple(submodule_commits),
        container_sha256=_require_string(value["container_sha256"], "container_sha256"),
    )
    _validate_fingerprint_semantics(fingerprint)
    return fingerprint


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _LOWERCASE_SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"Validation artifact {field_name} must be a 64-character lowercase SHA-256"
        )


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Validation artifact {field_name} must be a string")
    return value


def _validate_memory_budget(
    retained_bytes: object, memory_budget: MemoryBudget
) -> None:
    if not _is_nonnegative_int(memory_budget.available_bytes):
        raise ValueError("MemoryBudget.available_bytes must be a non-negative integer")
    copy_count = memory_budget.required_copy_count
    if type(copy_count) is not int or copy_count < 1:
        raise ValueError("MemoryBudget.required_copy_count must be a positive integer")
    if (
        not isinstance(retained_bytes, int)
        or isinstance(retained_bytes, bool)
        or retained_bytes < 0
    ):
        raise ValueError(
            "Validation artifact retained_bytes must be a non-negative integer"
        )
    required_bytes = retained_bytes * copy_count
    if memory_budget.available_bytes < required_bytes:
        label = "three" if copy_count == 3 else str(copy_count)
        raise MemoryError(
            f"Validation artifact requires {label}-copy headroom "
            f"({required_bytes} bytes), but only {memory_budget.available_bytes} bytes are available"
        )


def _validated_loaded_data(
    loaded_tensors: Mapping[str, torch.Tensor],
    tensor_records: object,
    metadata: object,
) -> BatchedDataDict[Any]:
    if not isinstance(tensor_records, Mapping) or set(loaded_tensors) != set(
        tensor_records
    ):
        raise ValueError("Validation artifact tensor names do not match manifest")
    data = BatchedDataDict[Any]()
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
    batch_size = data["input_ids"].shape[0]
    _validate_sft_metadata(metadata, batch_size)
    if not isinstance(metadata, Mapping):
        raise ValueError("Validation artifact metadata must be an object")
    for key, values in sorted(metadata.items()):
        if not isinstance(key, str) or not isinstance(values, list):
            raise ValueError("Validation artifact metadata entries must be named lists")
        data[key] = list(values)
    return data
