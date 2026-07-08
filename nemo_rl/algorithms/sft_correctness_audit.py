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
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast

import numpy as np
import torch


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
        train_loader_digest=_state_digest(train_loader.state_dict()),
        next_train_batch_digest=(
            _state_digest(next_train_batch) if next_train_batch is not None else None
        ),
        validation_payload_digest=_state_digest(validation_payload),
        validation_sample_ids_digest=_state_digest(validation_sample_ids),
        validation_token_counts_digest=_state_digest(validation_token_counts),
        worker_states=worker_states,
    )


def _default_record_sink(record: _AuditRecord) -> None:
    payload = dataclasses.asdict(record)
    print(
        "SFT_CORRECTNESS_AUDIT "
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
        record_sink: Callable[[_AuditRecord], None] = _default_record_sink,
    ) -> None:
        self._policy = policy
        self._train_loader = train_loader
        self._explicit_generator = explicit_generator
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
            return result
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
                    if validation_succeeded and gate.ready
                    else "validation_failed"
                    if not validation_succeeded
                    else "rejected"
                ),
            )
            self._elapsed_seconds = (
                previous_audit_time_s + time.perf_counter() - audit_start
            )
            if validation_succeeded and gate.ready:
                self._pending_records.append(record)
            else:
                self._record_sink(record)
            if not gate.ready:
                changed = ", ".join(gate.differences)
                raise CorrectnessAuditError(
                    f"SFT correctness audit rejected validation step {step}: {changed}"
                )

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
            status="finalized",
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
