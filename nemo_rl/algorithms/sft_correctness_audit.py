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
import dataclasses
import base64
import binascii
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import math
from os import PathLike
from pathlib import Path
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader


_TORCHDATA_DISTRIBUTION = "torchdata"
_TORCHDATA_PACKAGE = "torchdata"
_TORCHDATA_RUNTIME_MODULE = "torchdata.stateful_dataloader.stateful_dataloader"
_TORCHDATA_RUNTIME_PACKAGE_PATH = "torchdata/stateful_dataloader/stateful_dataloader.py"
_SUPPORTED_TORCHDATA_VERSION = "0.11.0"


class _CorrectnessFingerprintPolicy(Protocol):
    def get_correctness_state_fingerprint(
        self,
        *,
        content_sample_count: int = 8,
        reduction_chunk_numel: int = 1 << 20,
    ) -> list[dict[str, Any]]: ...


class _StatefulLoader(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class CorrectnessSnapshot:
    """Exact-state digests and lossy worker fingerprints at one boundary."""

    python_rng_digest: str
    numpy_rng_digest: str
    torch_cpu_rng_digest: str
    torch_cuda_rng_digests: tuple[str, ...]
    explicit_generator_digest: str | None
    train_loader_digest: str
    next_train_batch_digest: str | None
    validation_payload_digest: str
    validation_sample_ids_digest: str
    validation_token_counts_digest: str
    worker_states: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class CorrectnessGateResult:
    ready: bool
    differences: tuple[str, ...]


@dataclass(frozen=True)
class CorrectnessValidationEvidence:
    """Exact digests of one validation execution's canonical runtime inputs."""

    payload_digest: str
    sample_ids_digest: str
    token_counts_digest: str


@dataclass(frozen=True)
class CorrectnessValidationEvidencePair:
    before: CorrectnessValidationEvidence
    after: CorrectnessValidationEvidence


@dataclass(frozen=True)
class CorrectnessNextTrainBatchEvidence:
    batch_digest: str
    sample_ids_digest: str
    token_counts_digest: str


@dataclass(frozen=True)
class CorrectnessAuditRecord:
    validation_step: int
    validation_succeeded: bool
    before: CorrectnessSnapshot
    after: CorrectnessSnapshot
    before_digest: str
    after_digest: str
    gate: CorrectnessGateResult
    validation_evidence: CorrectnessValidationEvidencePair | None
    next_train_batch: CorrectnessNextTrainBatchEvidence | None
    audit_time_s: float
    status: str


class CorrectnessAuditError(RuntimeError):
    """Raised when validation changes state protected by the audit gate."""


_ResultT = TypeVar("_ResultT")
_AuditRecord = CorrectnessAuditRecord


def _tensor_digest_record(tensor: torch.Tensor) -> dict[str, object]:
    detached = tensor.detach()
    if detached.device.type == "meta":
        raise ValueError("Correctness auditing cannot hash meta tensors")
    cpu_tensor = detached.to(device="cpu").contiguous()
    byte_view = cpu_tensor.reshape(-1).view(torch.uint8)
    payload = byte_view.numpy().tobytes()
    return {
        "type": "torch.Tensor",
        "dtype": str(cpu_tensor.dtype),
        "shape": list(cpu_tensor.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonicalize(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "float", "value": "nan"}
        if math.isinf(value):
            return {"type": "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, torch.Tensor):
        return _tensor_digest_record(value)
    if isinstance(value, torch.Generator):
        generator = cast(Any, value)
        return {
            "type": "torch.Generator",
            "state": _tensor_digest_record(generator.get_state()),
        }
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "type": "numpy.ndarray",
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return {
            "type": f"numpy.{value.dtype}",
            "value": _canonicalize(value.item()),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__qualname__,
            "fields": [
                [field.name, _canonicalize(getattr(value, field.name))]
                for field in dataclasses.fields(value)
            ],
        }
    if isinstance(value, Mapping):
        entries = [
            [_canonicalize(key), _canonicalize(item)] for key, item in value.items()
        ]
        entries.sort(
            key=lambda entry: json.dumps(
                entry[0], sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        )
        return {"type": "mapping", "entries": entries}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_canonicalize(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_canonicalize(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        )
        return {"type": type(value).__name__, "items": items}
    if isinstance(value, (torch.device, torch.dtype)):
        return {"type": type(value).__qualname__, "value": str(value)}
    raise TypeError(
        "Correctness audit state contains unsupported value type "
        f"{type(value).__qualname__}"
    )


def _state_digest(value: object) -> str:
    canonical = json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def capture_validation_evidence(
    *,
    validation_payload: object,
    validation_sample_ids: object,
    validation_token_counts: object,
) -> CorrectnessValidationEvidence:
    """Capture exact validation payload, identity, and token-count digests."""
    return CorrectnessValidationEvidence(
        payload_digest=_state_digest(validation_payload),
        sample_ids_digest=_state_digest(validation_sample_ids),
        token_counts_digest=_state_digest(validation_token_counts),
    )


def combine_validation_evidence(
    evidence: Sequence[CorrectnessValidationEvidence],
) -> CorrectnessValidationEvidence:
    """Combine ordered per-submission evidence into one comparable record."""
    if not evidence:
        raise ValueError("Validation correctness evidence cannot be empty")
    return CorrectnessValidationEvidence(
        payload_digest=_state_digest(tuple(item.payload_digest for item in evidence)),
        sample_ids_digest=_state_digest(
            tuple(item.sample_ids_digest for item in evidence)
        ),
        token_counts_digest=_state_digest(
            tuple(item.token_counts_digest for item in evidence)
        ),
    )


def snapshot_digest(snapshot: CorrectnessSnapshot) -> str:
    """Hash a snapshot independently of mapping insertion order."""
    return _state_digest(snapshot)


def _compare_values(
    before: object,
    after: object,
    *,
    path: str,
    differences: list[str],
) -> None:
    if dataclasses.is_dataclass(before) and dataclasses.is_dataclass(after):
        if type(before) is not type(after):
            differences.append(path)
            return
        for field in dataclasses.fields(before):
            field_path = f"{path}.{field.name}" if path else field.name
            _compare_values(
                getattr(before, field.name),
                getattr(after, field.name),
                path=field_path,
                differences=differences,
            )
        return
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        all_keys = sorted(set(before) | set(after), key=lambda key: str(key))
        for key in all_keys:
            key_path = f"{path}.{key}" if path else str(key)
            if key not in before or key not in after:
                differences.append(key_path)
                continue
            _compare_values(
                before[key], after[key], path=key_path, differences=differences
            )
        return
    if (
        isinstance(before, Sequence)
        and isinstance(after, Sequence)
        and not isinstance(before, (str, bytes))
        and not isinstance(after, (str, bytes))
    ):
        if len(before) != len(after):
            differences.append(path)
            return
        for index, (before_item, after_item) in enumerate(zip(before, after)):
            _compare_values(
                before_item,
                after_item,
                path=f"{path}.{index}",
                differences=differences,
            )
        return
    if _canonicalize(before) != _canonicalize(after):
        differences.append(path)


def compare_correctness_snapshots(
    before: CorrectnessSnapshot, after: CorrectnessSnapshot
) -> list[str]:
    """Return stable field paths whose exact digests or fingerprints changed."""
    differences: list[str] = []
    _compare_values(before, after, path="", differences=differences)
    return differences


def evaluate_correctness_gate(
    before: CorrectnessSnapshot, after: CorrectnessSnapshot
) -> CorrectnessGateResult:
    differences = tuple(compare_correctness_snapshots(before, after))
    return CorrectnessGateResult(ready=not differences, differences=differences)


def _normalize_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _decode_record_sha256(value: object) -> bytes:
    if type(value) is not str or not value or "=" in value:
        raise CorrectnessAuditError(
            "TorchData source RECORD SHA-256 is missing or non-canonical"
        )
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise CorrectnessAuditError(
            "TorchData source RECORD SHA-256 is not ASCII"
        ) from error
    urlsafe_alphabet = (
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    )
    if any(byte not in urlsafe_alphabet for byte in encoded):
        raise CorrectnessAuditError(
            "TorchData source RECORD SHA-256 is not URL-safe base64"
        )
    padded = encoded + b"=" * (-len(encoded) % 4)
    try:
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise CorrectnessAuditError(
            "TorchData source RECORD SHA-256 is malformed"
        ) from error
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if len(decoded) != hashlib.sha256().digest_size or canonical != value:
        raise CorrectnessAuditError("TorchData source RECORD SHA-256 is malformed")
    return decoded


def _canonical_source_path(value: object, *, label: str) -> Path:
    if not isinstance(value, (str, PathLike)):
        raise CorrectnessAuditError(
            f"Correctness audit could not resolve TorchData {label}"
        )
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise CorrectnessAuditError(
            f"Correctness audit could not resolve TorchData {label}"
        ) from error
    if not path.is_file():
        raise CorrectnessAuditError(f"TorchData {label} is not a regular file")
    return path


def _torchdata_source_identity(
    distribution: Any,
    runtime_module: Any,
) -> tuple[str, str]:
    distribution_files = distribution.files
    if type(distribution_files) is not list:
        raise CorrectnessAuditError(
            "TorchData distribution file records are unavailable"
        )
    source_records = []
    for package_path in distribution_files:
        as_posix = getattr(package_path, "as_posix", None)
        if not callable(as_posix):
            raise CorrectnessAuditError(
                "TorchData distribution contains an invalid PackagePath"
            )
        try:
            relative_path = as_posix()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise CorrectnessAuditError(
                "TorchData distribution contains an invalid PackagePath"
            ) from error
        if type(relative_path) is not str:
            raise CorrectnessAuditError(
                "TorchData distribution contains an invalid PackagePath"
            )
        if relative_path == _TORCHDATA_RUNTIME_PACKAGE_PATH:
            source_records.append(package_path)
    if len(source_records) != 1:
        raise CorrectnessAuditError(
            "TorchData distribution must contain exactly one runtime source record"
        )

    source_record = source_records[0]
    record_hash = getattr(source_record, "hash", None)
    if record_hash is None:
        raise CorrectnessAuditError(
            "TorchData runtime source RECORD hash is unavailable"
        )
    hash_mode = getattr(record_hash, "mode", None)
    if hash_mode != "sha256":
        raise CorrectnessAuditError(
            f"Unsupported TorchData runtime source RECORD hash {hash_mode!r}"
        )
    expected_digest = _decode_record_sha256(getattr(record_hash, "value", None))

    locate = getattr(source_record, "locate", None)
    if not callable(locate):
        raise CorrectnessAuditError(
            "TorchData runtime source PackagePath cannot be located"
        )
    try:
        located_value = locate()
    except (OSError, RuntimeError) as error:
        raise CorrectnessAuditError(
            "TorchData runtime source PackagePath cannot be located"
        ) from error
    located_path = _canonical_source_path(located_value, label="PackagePath")

    runtime_spec = getattr(runtime_module, "__spec__", None)
    runtime_origin = getattr(runtime_spec, "origin", None)
    runtime_file = getattr(runtime_module, "__file__", None)
    if type(runtime_origin) is not str or type(runtime_file) is not str:
        raise CorrectnessAuditError(
            "TorchData runtime source origin or __file__ is unavailable"
        )
    origin_path = _canonical_source_path(runtime_origin, label="module origin")
    file_path = _canonical_source_path(runtime_file, label="module __file__")
    if located_path != origin_path or located_path != file_path:
        raise CorrectnessAuditError(
            "TorchData PackagePath, module origin, and __file__ do not match"
        )

    try:
        source_bytes = located_path.read_bytes()
    except OSError as error:
        raise CorrectnessAuditError(
            "Correctness audit could not read the TorchData runtime source"
        ) from error
    actual_digest = hashlib.sha256(source_bytes).digest()
    if actual_digest != expected_digest:
        raise CorrectnessAuditError(
            "TorchData runtime source does not match its RECORD SHA-256"
        )
    return str(located_path), actual_digest.hex()


def _torchdata_runtime_identity() -> tuple[str, type[Any], str, str]:
    try:
        distribution = importlib_metadata.distribution(_TORCHDATA_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as error:
        raise CorrectnessAuditError(
            "Correctness audit requires the torchdata 0.11.0 distribution"
        ) from error

    distribution_name = distribution.metadata.get("Name")
    package_owners = importlib_metadata.packages_distributions().get(_TORCHDATA_PACKAGE)
    if type(distribution_name) is not str or type(package_owners) is not list:
        raise CorrectnessAuditError(
            "Correctness audit could not verify the torchdata package identity"
        )
    untyped_package_owners = cast(list[object], package_owners)
    if not all(type(owner) is str for owner in untyped_package_owners):
        raise CorrectnessAuditError(
            "Correctness audit could not verify the torchdata package identity"
        )
    verified_distribution_name = cast(str, distribution_name)
    verified_package_owners = cast(list[str], untyped_package_owners)
    normalized_distribution = _normalize_distribution_name(verified_distribution_name)
    normalized_owners = tuple(
        _normalize_distribution_name(owner) for owner in verified_package_owners
    )
    if normalized_distribution != _TORCHDATA_DISTRIBUTION or normalized_owners != (
        _TORCHDATA_DISTRIBUTION,
    ):
        raise CorrectnessAuditError(
            "Correctness audit found an unexpected torchdata package identity"
        )
    runtime_version = distribution.version
    if (
        type(runtime_version) is not str
        or runtime_version != _SUPPORTED_TORCHDATA_VERSION
    ):
        raise CorrectnessAuditError(
            "Correctness audit supports only torchdata "
            f"{_SUPPORTED_TORCHDATA_VERSION}, found {runtime_version!r}"
        )
    try:
        runtime_module = importlib.import_module(_TORCHDATA_RUNTIME_MODULE)
    except ImportError as error:
        raise CorrectnessAuditError(
            "Correctness audit could not import the locked torchdata runtime module"
        ) from error
    source_origin, source_sha256 = _torchdata_source_identity(
        distribution, runtime_module
    )
    runtime_loader_class = getattr(runtime_module, "StatefulDataLoader", None)
    iterator_class = getattr(runtime_module, "_StatefulBaseDataLoaderIter", None)
    if (
        runtime_loader_class is not StatefulDataLoader
        or not isinstance(iterator_class, type)
        or StatefulDataLoader.__module__ != _TORCHDATA_RUNTIME_MODULE
        or iterator_class.__module__ != _TORCHDATA_RUNTIME_MODULE
    ):
        raise CorrectnessAuditError(
            "Correctness audit found an unexpected torchdata runtime class layout"
        )
    return runtime_version, iterator_class, source_origin, source_sha256


def _capture_train_loader_state(train_loader: _StatefulLoader) -> object:
    """Read loader state without lazily creating a restored loader iterator."""
    if not isinstance(train_loader, StatefulDataLoader):
        return train_loader.state_dict()
    if type(train_loader) is not StatefulDataLoader:
        raise CorrectnessAuditError(
            "Correctness audit does not support StatefulDataLoader subclasses"
        )

    (
        runtime_version,
        iterator_class,
        source_origin,
        source_sha256,
    ) = _torchdata_runtime_identity()
    try:
        loader_attributes = vars(train_loader)
    except TypeError as error:
        raise CorrectnessAuditError(
            "TorchData StatefulDataLoader has no inspectable instance layout"
        ) from error
    required_fields = {
        "_iterator",
        "next_iter_state",
        "_initial_iter_for_state_dict",
    }
    missing_fields = sorted(required_fields - loader_attributes.keys())
    if missing_fields:
        raise CorrectnessAuditError(
            "TorchData StatefulDataLoader is missing required private fields: "
            + ", ".join(missing_fields)
        )

    iterator = loader_attributes["_iterator"]
    pending_state = loader_attributes["next_iter_state"]
    initial_iter_for_state_dict = loader_attributes["_initial_iter_for_state_dict"]
    if type(initial_iter_for_state_dict) is not bool:
        raise CorrectnessAuditError(
            "TorchData StatefulDataLoader has an unexpected initial-iterator flag type"
        )
    if pending_state is not None and type(pending_state) is not dict:
        raise CorrectnessAuditError(
            "TorchData StatefulDataLoader has an unexpected pending-state type"
        )
    if iterator is not None and not isinstance(iterator, iterator_class):
        raise CorrectnessAuditError(
            "TorchData StatefulDataLoader has an unexpected iterator type"
        )

    evidence = {
        "initial_iter_for_state_dict": initial_iter_for_state_dict,
        "loader_class": f"{type(train_loader).__module__}.{type(train_loader).__qualname__}",
        "package": _TORCHDATA_PACKAGE,
        "package_version": runtime_version,
        "source_origin": source_origin,
        "source_sha256": source_sha256,
    }
    if iterator is None:
        if initial_iter_for_state_dict:
            raise CorrectnessAuditError(
                "TorchData StatefulDataLoader with no iterator cannot have "
                "initial-iterator flag set"
            )
        if pending_state == {}:
            raise CorrectnessAuditError(
                "TorchData StatefulDataLoader cannot have an empty pending state"
            )
        return evidence | {
            "boundary": "pending" if pending_state is not None else "not_started",
            "state": pending_state,
        }
    if pending_state is not None:
        raise CorrectnessAuditError(
            "TorchData StatefulDataLoader active iterator cannot have pending state"
        )
    return {
        **evidence,
        "boundary": "active",
        "state": train_loader.state_dict(),
    }


def capture_correctness_snapshot(
    *,
    policy: _CorrectnessFingerprintPolicy,
    train_loader: _StatefulLoader,
    explicit_generator: torch.Generator | None,
    validation_payload: object,
    validation_sample_ids: object,
    validation_token_counts: object,
    next_train_batch: object | None = None,
) -> CorrectnessSnapshot:
    """Read driver and worker state without advancing any RNG or loader."""
    worker_records = policy.get_correctness_state_fingerprint()
    worker_states: dict[int, dict[str, Any]] = {}
    for record in worker_records:
        rank = record.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("Worker correctness fingerprint rank must be an integer")
        if rank in worker_states:
            raise ValueError(f"Duplicate worker correctness fingerprint rank {rank}")
        worker_states[rank] = dict(record)
    worker_states = dict(sorted(worker_states.items()))

    torch_cuda_rng_digests = (
        tuple(_state_digest(state) for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_initialized()
        else ()
    )
    return CorrectnessSnapshot(
        python_rng_digest=_state_digest(random.getstate()),
        numpy_rng_digest=_state_digest(np.random.get_state()),
        torch_cpu_rng_digest=_state_digest(torch.get_rng_state()),
        torch_cuda_rng_digests=torch_cuda_rng_digests,
        explicit_generator_digest=(
            _state_digest(explicit_generator.get_state())
            if explicit_generator is not None
            else None
        ),
        train_loader_digest=_state_digest(_capture_train_loader_state(train_loader)),
        next_train_batch_digest=(
            _state_digest(next_train_batch) if next_train_batch is not None else None
        ),
        validation_payload_digest=_state_digest(validation_payload),
        validation_sample_ids_digest=_state_digest(validation_sample_ids),
        validation_token_counts_digest=_state_digest(validation_token_counts),
        worker_states=worker_states,
    )


def _default_record_sink(record: _AuditRecord) -> None:
    def summarize_snapshot(snapshot: CorrectnessSnapshot) -> dict[str, object]:
        worker_states: dict[str, object] = {}
        for rank, state in snapshot.worker_states.items():
            model_state = state.get("model")
            model_parameter_source = (
                model_state.get("parameter_source")
                if isinstance(model_state, Mapping)
                else None
            )
            materialized_model_parameters_included = (
                model_state.get("materialized_model_parameters_included")
                if isinstance(model_state, Mapping)
                else None
            )
            frozen_model_parameters_included = (
                model_state.get("frozen_model_parameters_included")
                if isinstance(model_state, Mapping)
                else None
            )
            category_digests = {
                key: _state_digest(value)
                for key, value in sorted(state.items())
                if key not in {"rank", "device", "coordinates"}
            }
            worker_states[str(rank)] = {
                "state_digest": _state_digest(state),
                "category_digests": category_digests,
                "model_parameter_source": model_parameter_source,
                "materialized_model_parameters_included": (
                    materialized_model_parameters_included
                ),
                "frozen_model_parameters_included": frozen_model_parameters_included,
            }
        return {
            "python_rng_digest": snapshot.python_rng_digest,
            "numpy_rng_digest": snapshot.numpy_rng_digest,
            "torch_cpu_rng_digest": snapshot.torch_cpu_rng_digest,
            "torch_cuda_rng_digests": snapshot.torch_cuda_rng_digests,
            "explicit_generator_digest": snapshot.explicit_generator_digest,
            "train_loader_digest": snapshot.train_loader_digest,
            "next_train_batch_digest": snapshot.next_train_batch_digest,
            "validation_payload_digest": snapshot.validation_payload_digest,
            "validation_sample_ids_digest": snapshot.validation_sample_ids_digest,
            "validation_token_counts_digest": snapshot.validation_token_counts_digest,
            "worker_states": worker_states,
        }

    payload = {
        "schema_version": 1,
        "validation_step": record.validation_step,
        "validation_succeeded": record.validation_succeeded,
        "before_digest": record.before_digest,
        "after_digest": record.after_digest,
        "before": summarize_snapshot(record.before),
        "after": summarize_snapshot(record.after),
        "gate": {
            "ready": record.gate.ready,
            "difference_count": len(record.gate.differences),
            "differences_sha256": _state_digest(record.gate.differences),
            "difference_examples": record.gate.differences[:32],
        },
        "validation_evidence": (
            dataclasses.asdict(record.validation_evidence)
            if record.validation_evidence is not None
            else None
        ),
        "next_train_batch": (
            dataclasses.asdict(record.next_train_batch)
            if record.next_train_batch is not None
            else None
        ),
        "audit_time_s": record.audit_time_s,
        "status": record.status,
    }
    print(
        "SFT_CORRECTNESS_AUDIT_SUMMARY "
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _batch_metadata(batch: object) -> tuple[object, object]:
    if not isinstance(batch, Mapping):
        return None, None
    sample_ids = batch.get("idx")
    if "processed_token_counts" in batch:
        token_counts = batch["processed_token_counts"]
    elif "input_lengths" in batch:
        token_counts = batch["input_lengths"]
    else:
        token_counts = None
    return sample_ids, token_counts


def capture_next_train_batch_evidence(
    batch: object,
) -> CorrectnessNextTrainBatchEvidence:
    """Digest a train batch only after the normal training iterator yields it."""
    sample_ids, token_counts = _batch_metadata(batch)
    return CorrectnessNextTrainBatchEvidence(
        batch_digest=_state_digest(batch),
        sample_ids_digest=_state_digest(sample_ids),
        token_counts_digest=_state_digest(token_counts),
    )


def compare_next_train_batch_to_control(
    control: CorrectnessNextTrainBatchEvidence,
    audited: CorrectnessAuditRecord,
) -> CorrectnessGateResult:
    """Compare finalized audit evidence with an explicit no-validation control."""
    if audited.next_train_batch is None:
        return CorrectnessGateResult(
            ready=False,
            differences=("next_train_batch",),
        )
    differences: list[str] = []
    _compare_values(
        control,
        audited.next_train_batch,
        path="next_train_batch",
        differences=differences,
    )
    return CorrectnessGateResult(
        ready=not differences,
        differences=tuple(differences),
    )


class SFTCorrectnessAuditor:
    """Own validation-boundary gates and deferred natural-batch evidence."""

    def __init__(
        self,
        *,
        policy: _CorrectnessFingerprintPolicy,
        train_loader: _StatefulLoader,
        explicit_generator: torch.Generator | None,
        enforce_unchanged: bool = True,
        record_sink: Callable[[_AuditRecord], None] = _default_record_sink,
    ) -> None:
        self._policy = policy
        self._train_loader = train_loader
        self._explicit_generator = explicit_generator
        self._enforce_unchanged = enforce_unchanged
        self._record_sink = record_sink
        self._pending_records: list[CorrectnessAuditRecord] = []
        self._elapsed_seconds = 0.0

    def _capture(self) -> CorrectnessSnapshot:
        return capture_correctness_snapshot(
            policy=self._policy,
            train_loader=self._train_loader,
            explicit_generator=self._explicit_generator,
            validation_payload=None,
            validation_sample_ids=None,
            validation_token_counts=None,
        )

    def audit_validation(
        self,
        *,
        step: int,
        validation: Callable[[], _ResultT],
        validation_evidence: Callable[[], CorrectnessValidationEvidencePair],
    ) -> _ResultT:
        audit_start = time.perf_counter()
        before = self._capture()
        self._elapsed_seconds += time.perf_counter() - audit_start
        validation_succeeded = False
        captured_validation_evidence: CorrectnessValidationEvidencePair | None = None
        try:
            result = validation()
            validation_succeeded = True
        finally:
            previous_audit_time_s = self._elapsed_seconds
            audit_start = time.perf_counter()
            captured_validation_evidence = validation_evidence()
            after = self._capture()
            before = dataclasses.replace(
                before,
                validation_payload_digest=(
                    captured_validation_evidence.before.payload_digest
                ),
                validation_sample_ids_digest=(
                    captured_validation_evidence.before.sample_ids_digest
                ),
                validation_token_counts_digest=(
                    captured_validation_evidence.before.token_counts_digest
                ),
            )
            after = dataclasses.replace(
                after,
                validation_payload_digest=(
                    captured_validation_evidence.after.payload_digest
                ),
                validation_sample_ids_digest=(
                    captured_validation_evidence.after.sample_ids_digest
                ),
                validation_token_counts_digest=(
                    captured_validation_evidence.after.token_counts_digest
                ),
            )
            differences = compare_correctness_snapshots(before, after)
            gate = CorrectnessGateResult(
                ready=not differences,
                differences=tuple(differences),
            )
            before_digest = snapshot_digest(before)
            after_digest = snapshot_digest(after)
            audit_time_s = previous_audit_time_s + time.perf_counter() - audit_start
            record = CorrectnessAuditRecord(
                validation_step=step,
                validation_succeeded=validation_succeeded,
                before=before,
                after=after,
                before_digest=before_digest,
                after_digest=after_digest,
                gate=gate,
                validation_evidence=captured_validation_evidence,
                next_train_batch=None,
                audit_time_s=audit_time_s,
                status=(
                    "pending_next_train_batch"
                    if validation_succeeded
                    and (gate.ready or not self._enforce_unchanged)
                    else "validation_failed"
                    if not validation_succeeded
                    else "rejected"
                ),
            )
            self._elapsed_seconds = (
                previous_audit_time_s + time.perf_counter() - audit_start
            )
            if validation_succeeded and (
                gate.ready or not self._enforce_unchanged
            ):
                self._pending_records.append(record)
            else:
                self._record_sink(record)
            if self._enforce_unchanged and not gate.ready:
                changed = ", ".join(gate.differences)
                raise CorrectnessAuditError(
                    f"SFT correctness audit rejected validation step {step}: {changed}"
                )
        return result

    def record_next_train_batch(self, batch: object) -> None:
        if not self._pending_records:
            return
        audit_start = time.perf_counter()
        pending_record = self._pending_records.pop(0)
        next_train_batch = capture_next_train_batch_evidence(batch)
        digest_elapsed = time.perf_counter() - audit_start
        before = dataclasses.replace(
            pending_record.before,
            next_train_batch_digest=next_train_batch.batch_digest,
        )
        after = dataclasses.replace(
            pending_record.after,
            next_train_batch_digest=next_train_batch.batch_digest,
        )
        completed_record = dataclasses.replace(
            pending_record,
            before=before,
            after=after,
            before_digest=snapshot_digest(before),
            after_digest=snapshot_digest(after),
            next_train_batch=next_train_batch,
            audit_time_s=pending_record.audit_time_s + digest_elapsed,
            status=(
                "finalized"
                if pending_record.gate.ready
                else "finalized_with_state_changes"
            ),
        )
        self._record_sink(completed_record)
        self._elapsed_seconds += time.perf_counter() - audit_start

    def flush_pending(self) -> None:
        missing_steps: list[int] = []
        while self._pending_records:
            pending_record = self._pending_records.pop(0)
            missing_steps.append(pending_record.validation_step)
            missing_gate = CorrectnessGateResult(
                ready=False,
                differences=(*pending_record.gate.differences, "next_train_batch"),
            )
            self._record_sink(
                dataclasses.replace(
                    pending_record,
                    gate=missing_gate,
                    status="no_naturally_consumed_batch",
                )
            )
        if missing_steps:
            steps = ", ".join(str(step) for step in missing_steps)
            raise CorrectnessAuditError(
                "SFT correctness audit could not finalize naturally consumed "
                f"train-batch evidence for validation step(s): {steps}"
            )

    def consume_elapsed_seconds(self) -> float:
        elapsed = self._elapsed_seconds
        self._elapsed_seconds = 0.0
        return elapsed
