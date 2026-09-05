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

"""Fail-fast supervision for the participant futures of one refit operation."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Generic, Literal, Protocol, TypeVar, cast

__all__ = [
    "RefitCompletion",
    "RefitParticipant",
    "RefitParticipantFailure",
    "RefitResultNormalizer",
    "RefitSupervisionProtocolError",
    "RefitSupervisionSummary",
    "RefitSupervisionTimeout",
    "normalize_current_refit_result",
    "normalize_exact_true_refit_result",
    "supervise_refit_futures",
]


@dataclass(frozen=True, slots=True)
class RefitParticipant:
    """Stable identity of one refit participant.

    Attributes:
        role: Whether the participant produces or consumes refit weights.
        index: Zero-based position within that role's submitted future sequence.
    """

    role: Literal["producer", "consumer"]
    index: int

    def __post_init__(self) -> None:
        try:
            role = self.role
            index = self.index
        except AttributeError as error:
            raise ValueError("participant must be fully initialized") from error
        if type(role) is not str or role not in ("producer", "consumer"):
            raise ValueError("role must be exactly 'producer' or 'consumer'")
        if type(index) is not int or index < 0:
            raise ValueError("index must be an exact nonnegative int")


_NormalizedResultT = TypeVar("_NormalizedResultT")
_NormalizedResultT_co = TypeVar("_NormalizedResultT_co", covariant=True)
_MAX_RAY_TIMEOUT_S = float(((1 << 63) - 1) // 1000 - 1)


class RefitResultNormalizer(Protocol[_NormalizedResultT_co]):
    """Validate a raw terminal result and return its canonical representation.

    A normalizer raises on an invalid result. Its normalized return value is retained
    in the supervision summary, so typed transaction acknowledgements need no mutable
    closure or other side channel. Implementations must be deterministic, bounded,
    nonblocking local validators: they must not perform I/O, synchronization, or call
    distributed runtimes. The shared deadline measures their execution but cannot
    safely preempt arbitrary Python code.
    """

    def __call__(
        self, participant: RefitParticipant, result: object, /
    ) -> _NormalizedResultT_co: ...


class _RayRuntime(Protocol):
    def wait(
        self,
        object_refs: list[object],
        *,
        num_returns: int,
        timeout: float,
        fetch_local: bool,
    ) -> tuple[list[object], list[object]]: ...

    def get(self, object_ref: object, /, *, timeout: float) -> object: ...


@dataclass(frozen=True, slots=True)
class RefitCompletion(Generic[_NormalizedResultT]):
    """Validated result from one completed refit participant.

    Attributes:
        participant: Role and role-local index of the completed future.
        result: Canonical value returned by the required result normalizer.
    """

    participant: RefitParticipant
    result: _NormalizedResultT

    def __post_init__(self) -> None:
        try:
            participant = self.participant
            self.result
        except AttributeError as error:
            raise ValueError("completion must be fully initialized") from error
        if type(participant) is not RefitParticipant:
            raise ValueError("participant must be an exact RefitParticipant")
        participant.__post_init__()


@dataclass(frozen=True, slots=True)
class RefitSupervisionSummary(Generic[_NormalizedResultT]):
    """Successful terminal state of all participants in one refit.

    Attributes:
        operation: Caller-provided operation identity.
        producer_count: Number of supervised producer futures.
        consumer_count: Number of supervised consumer futures.
        completions: Canonical results in fail-fast processing order.
    """

    operation: str
    producer_count: int
    consumer_count: int
    completions: tuple[RefitCompletion[_NormalizedResultT], ...]

    def __post_init__(self) -> None:
        try:
            operation = self.operation
            producer_count = self.producer_count
            consumer_count = self.consumer_count
            completions = self.completions
        except AttributeError as error:
            raise ValueError("summary must be fully initialized") from error
        _validate_operation(operation)
        for name, count in (
            ("producer_count", producer_count),
            ("consumer_count", consumer_count),
        ):
            if type(count) is not int or count <= 0:
                raise ValueError(f"{name} must be an exact positive int")
        if type(completions) is not tuple or not completions:
            raise ValueError("completions must be an exact non-empty tuple")
        for completion in completions:
            if type(completion) is not RefitCompletion:
                raise ValueError(
                    "completions must contain exact RefitCompletion values"
                )
            completion.__post_init__()

        participants = tuple(completion.participant for completion in completions)
        if len(set(participants)) != len(participants):
            raise ValueError("completions must contain unique participants")
        if len(participants) != producer_count + consumer_count:
            raise ValueError("completion count does not match participant counts")
        for role, expected_count in (
            ("producer", producer_count),
            ("consumer", consumer_count),
        ):
            seen_indices = bytearray(expected_count)
            for participant in participants:
                if participant.role != role:
                    continue
                if participant.index >= expected_count:
                    raise ValueError(
                        f"{role} completion indices must be contiguous from zero"
                    )
                seen_indices[participant.index] = 1
            if not all(seen_indices):
                raise ValueError(
                    f"{role} completion indices must be contiguous from zero"
                )

    @property
    def completion_order(self) -> tuple[RefitParticipant, ...]:
        """Participants in fail-fast processing order."""
        return tuple(completion.participant for completion in self.completions)


class RefitParticipantFailure(RuntimeError):
    """One producer or consumer did not complete its refit successfully."""

    def __init__(
        self,
        *,
        operation: str,
        participant: RefitParticipant,
        detail: str,
    ) -> None:
        self.operation = operation
        self.participant = participant
        super().__init__(
            f"Refit operation {operation!r} failed at "
            f"{participant.role}[{participant.index}]: {detail}"
        )


class RefitSupervisionTimeout(TimeoutError):
    """The shared deadline elapsed before every participant result was accepted."""

    def __init__(
        self,
        *,
        operation: str,
        timeout_s: float,
        pending: tuple[RefitParticipant, ...],
    ) -> None:
        self.operation = operation
        self.timeout_s = timeout_s
        self.pending = pending
        pending_labels = ", ".join(
            f"{participant.role}[{participant.index}]" for participant in pending
        )
        super().__init__(
            f"Refit operation {operation!r} exceeded its {timeout_s}s deadline; "
            f"pending participants: {pending_labels}"
        )


class RefitSupervisionProtocolError(RuntimeError):
    """Ray returned a wait state that violates the supervision contract."""

    def __init__(self, *, operation: str, detail: str) -> None:
        self.operation = operation
        super().__init__(f"Refit operation {operation!r}: {detail}")


def _load_ray() -> _RayRuntime:
    # Ray is optional for import-only users of the weight-sync package.
    import ray

    return cast(_RayRuntime, ray)


def _validate_operation(operation: object) -> str:
    if type(operation) is not str or not operation or operation != operation.strip():
        raise ValueError("operation must be an exact non-empty stripped string")
    return operation


def _normalize_timeout_s(timeout_s: object) -> float:
    if type(timeout_s) not in (int, float):
        raise ValueError("timeout_s must be a positive finite int or float")
    try:
        normalized = float(cast(int | float, timeout_s))
    except OverflowError as error:
        raise ValueError("timeout_s must be a positive finite int or float") from error
    if not isfinite(normalized) or normalized <= 0 or normalized > _MAX_RAY_TIMEOUT_S:
        raise ValueError(
            "timeout_s must be a positive finite int or float within Ray's "
            "signed 64-bit millisecond range"
        )
    return normalized


def normalize_exact_true_refit_result(
    participant: RefitParticipant, result: object, /
) -> Literal[True]:
    """Require an exact ``True`` acknowledgement from either participant role.

    Args:
        participant: Participant that returned the result.
        result: Raw value returned by its Ray future.

    Returns:
        The canonical ``True`` acknowledgement.

    Raises:
        ValueError: If the result is not the exact ``True`` singleton.
    """
    if result is not True:
        rendered_result = "False" if result is False else type(result).__name__
        raise ValueError(
            f"{participant.role} returned {rendered_result}; expected exactly True"
        )
    return True


def normalize_current_refit_result(
    participant: RefitParticipant, result: object, /
) -> Literal[True]:
    """Normalize the terminal values returned by current refit actor methods.

    Current producer RPCs complete normally with ``None`` and some return ``True``.
    Current consumer RPCs must return exact ``True``. False and all other values are
    rejected for both roles.

    Args:
        participant: Participant that returned the result.
        result: Raw value returned by its Ray future.

    Returns:
        Canonical exact ``True`` for every accepted result.

    Raises:
        ValueError: If the raw result does not satisfy the role-specific contract.
    """
    if participant.role == "producer" and result is None:
        return True
    if result is True:
        return True
    rendered_result = "False" if result is False else type(result).__name__
    expectation = (
        "exactly None or True" if participant.role == "producer" else "exactly True"
    )
    raise ValueError(
        f"{participant.role} returned {rendered_result}; expected {expectation}"
    )


def _participant_for_future(
    *,
    operation: str,
    participant_by_future: dict[object, RefitParticipant],
    future: object,
) -> RefitParticipant:
    try:
        return participant_by_future[future]
    except Exception as error:
        raise RefitSupervisionProtocolError(
            operation=operation,
            detail="a returned reference could not be mapped to its participant",
        ) from error


def _pending_participants(
    *,
    operation: str,
    participant_by_future: dict[object, RefitParticipant],
    futures: list[object],
) -> tuple[RefitParticipant, ...]:
    return tuple(
        _participant_for_future(
            operation=operation,
            participant_by_future=participant_by_future,
            future=future,
        )
        for future in futures
    )


def _validate_wait_partition(
    *,
    operation: str,
    phase: str,
    prior_pending: list[object],
    requested_ready_count: int,
    wait_result: object,
) -> tuple[list[object], list[object]]:
    if type(wait_result) is not tuple or len(wait_result) != 2:
        raise RefitSupervisionProtocolError(
            operation=operation,
            detail=f"{phase} ray.wait result must be an exact two-item tuple",
        )
    raw_ready, raw_remaining = cast(tuple[object, object], wait_result)
    if type(raw_ready) is not list or type(raw_remaining) is not list:
        raise RefitSupervisionProtocolError(
            operation=operation,
            detail=f"{phase} ray.wait partitions must be exact lists",
        )
    ready = cast(list[object], raw_ready)
    remaining = cast(list[object], raw_remaining)
    if len(ready) > requested_ready_count:
        raise RefitSupervisionProtocolError(
            operation=operation,
            detail=(
                f"{phase} ray.wait returned {len(ready)} ready references after "
                f"requesting at most {requested_ready_count}"
            ),
        )

    try:
        prior_lookup: dict[object, None] = {}
        for future in prior_pending:
            if future in prior_lookup:
                raise RefitSupervisionProtocolError(
                    operation=operation,
                    detail=f"{phase} pending references were not logically unique",
                )
            prior_lookup[future] = None
    except RefitSupervisionProtocolError:
        raise
    except Exception as error:
        raise RefitSupervisionProtocolError(
            operation=operation,
            detail=f"{phase} pending references could not be compared",
        ) from error

    seen: dict[object, str] = {}
    for partition_name, partition in (("ready", ready), ("remaining", remaining)):
        for position, future in enumerate(partition):
            try:
                if future not in prior_lookup:
                    raise RefitSupervisionProtocolError(
                        operation=operation,
                        detail=(
                            f"{phase} ray.wait returned an unknown {partition_name} "
                            f"reference at index {position}"
                        ),
                    )
                previous_partition = seen.get(future)
                if previous_partition is not None:
                    raise RefitSupervisionProtocolError(
                        operation=operation,
                        detail=(
                            f"{phase} ray.wait duplicated a logical reference across "
                            f"{previous_partition} and {partition_name} partitions"
                        ),
                    )
                seen[future] = partition_name
            except RefitSupervisionProtocolError:
                raise
            except Exception as error:
                raise RefitSupervisionProtocolError(
                    operation=operation,
                    detail=(
                        f"{phase} {partition_name} reference at index {position} "
                        "could not be compared"
                    ),
                ) from error

    if len(seen) != len(prior_lookup):
        raise RefitSupervisionProtocolError(
            operation=operation,
            detail=f"{phase} ray.wait partitions omitted pending references",
        )
    try:
        ready_by_future = {future: future for future in ready}
        remaining_by_future = {future: future for future in remaining}
        return (
            [
                ready_by_future[future]
                for future in prior_pending
                if future in ready_by_future
            ],
            [
                remaining_by_future[future]
                for future in prior_pending
                if future in remaining_by_future
            ],
        )
    except Exception as error:  # pragma: no cover - guarded by comparison checks above
        raise RefitSupervisionProtocolError(
            operation=operation,
            detail=f"{phase} ray.wait partitions could not be canonically ordered",
        ) from error


def _order_ready_wave(
    *,
    operation: str,
    prior_pending: list[object],
    ready_wave: list[object],
) -> list[object]:
    try:
        ready_by_future = {future: future for future in ready_wave}
        return [
            ready_by_future[future]
            for future in prior_pending
            if future in ready_by_future
        ]
    except Exception as error:
        raise RefitSupervisionProtocolError(
            operation=operation,
            detail="ready wave references could not be canonically ordered",
        ) from error


def _wait_for_partition(
    *,
    operation: str,
    phase: str,
    ray_runtime: _RayRuntime,
    pending: list[object],
    num_returns: int,
    timeout_s: float,
) -> tuple[list[object], list[object]]:
    try:
        wait_result: object = ray_runtime.wait(
            pending,
            num_returns=num_returns,
            timeout=timeout_s,
            fetch_local=True,
        )
    except Exception as error:
        raise RefitSupervisionProtocolError(
            operation=operation,
            detail=f"{phase} ray.wait raised {type(error).__name__}: {error}",
        ) from error
    return _validate_wait_partition(
        operation=operation,
        phase=phase,
        prior_pending=pending,
        requested_ready_count=num_returns,
        wait_result=wait_result,
    )


def _resolve_completion(
    *,
    operation: str,
    ray_runtime: _RayRuntime,
    participant: RefitParticipant,
    future: object,
    result_normalizer: RefitResultNormalizer[_NormalizedResultT],
    get_timeout_s: float,
) -> RefitCompletion[_NormalizedResultT]:
    try:
        result = ray_runtime.get(future, timeout=get_timeout_s)
    except Exception as error:
        raise RefitParticipantFailure(
            operation=operation,
            participant=participant,
            detail=f"raised {type(error).__name__}: {error}",
        ) from error
    try:
        normalized_result = result_normalizer(participant, result)
    except Exception as error:
        raise RefitParticipantFailure(
            operation=operation,
            participant=participant,
            detail=f"result normalization raised {type(error).__name__}: {error}",
        ) from error
    return RefitCompletion(participant=participant, result=normalized_result)


def supervise_refit_futures(
    *,
    operation: str,
    producer_futures: Sequence[object],
    consumer_futures: Sequence[object],
    result_normalizer: RefitResultNormalizer[_NormalizedResultT],
    timeout_s: float,
) -> RefitSupervisionSummary[_NormalizedResultT]:
    """Supervise one refit's producer and consumer futures as a single operation.

    All futures enter one ``ray.wait(..., num_returns=1)`` loop, so a consumer failure
    is observed even while a producer is blocked, and vice versa. A zero-timeout drain
    processes futures that are already ready without adding wait latency. ``timeout_s``
    is one monotonic deadline for the whole operation, not a fresh budget per future.

    Args:
        operation: Exact, non-empty, stripped label included in every failure.
        producer_futures: Non-empty sequence of producer Ray ObjectRefs.
        consumer_futures: Non-empty sequence of consumer Ray ObjectRefs.
        result_normalizer: Required validator and canonicalizer for terminal values.
        timeout_s: Required positive finite deadline in seconds for the whole operation.

    Returns:
        Immutable participant results in fail-fast processing order.

    Raises:
        ValueError: If operation metadata, timeout, or future groups are invalid.
        RefitParticipantFailure: If a remote call or result normalization fails.
        RefitSupervisionTimeout: If the shared deadline expires.
        RefitSupervisionProtocolError: If Ray violates its wait-result contract.
    """
    operation = _validate_operation(operation)
    producer_refs = tuple(producer_futures)
    consumer_refs = tuple(consumer_futures)
    if not producer_refs:
        raise ValueError("producer participant group must not be empty")
    if not consumer_refs:
        raise ValueError("consumer participant group must not be empty")
    normalized_timeout_s = _normalize_timeout_s(timeout_s)
    if not callable(result_normalizer):
        raise ValueError("result_normalizer must be callable")
    for role, refs in (("producer", producer_refs), ("consumer", consumer_refs)):
        for index, ref in enumerate(refs):
            if ref is None:
                raise ValueError(
                    f"{role} participant reference at index {index} must not be None"
                )

    participants: list[tuple[object, RefitParticipant]] = []
    participant_by_future: dict[object, RefitParticipant] = {}
    for role, refs in (("consumer", consumer_refs), ("producer", producer_refs)):
        for index, future in enumerate(refs):
            participant = RefitParticipant(role, index)
            try:
                previous = participant_by_future.get(future)
            except TypeError as error:
                raise ValueError(
                    f"{participant.role}[{participant.index}] reference must be hashable"
                ) from error
            if previous is not None:
                raise ValueError(
                    "duplicate participant reference at "
                    f"{participant.role}[{participant.index}]; first seen at "
                    f"{previous.role}[{previous.index}]"
                )
            participant_by_future[future] = participant
            participants.append((future, participant))
    ray_runtime = _load_ray()
    remaining = [future for future, _ in participants]
    completions: list[RefitCompletion[_NormalizedResultT]] = []
    deadline = monotonic() + normalized_timeout_s
    is_first_blocking_wait = True
    while remaining:
        if is_first_blocking_wait:
            wait_timeout = normalized_timeout_s
            is_first_blocking_wait = False
        else:
            wait_timeout = deadline - monotonic()
            if wait_timeout <= 0:
                raise RefitSupervisionTimeout(
                    operation=operation,
                    timeout_s=normalized_timeout_s,
                    pending=_pending_participants(
                        operation=operation,
                        participant_by_future=participant_by_future,
                        futures=remaining,
                    ),
                )
        wave_pending = remaining
        ready, next_remaining = _wait_for_partition(
            operation=operation,
            phase="blocking",
            ray_runtime=ray_runtime,
            pending=wave_pending,
            num_returns=1,
            timeout_s=wait_timeout,
        )
        if not ready:
            raise RefitSupervisionTimeout(
                operation=operation,
                timeout_s=normalized_timeout_s,
                pending=_pending_participants(
                    operation=operation,
                    participant_by_future=participant_by_future,
                    futures=remaining,
                ),
            )
        if deadline - monotonic() <= 0:
            raise RefitSupervisionTimeout(
                operation=operation,
                timeout_s=normalized_timeout_s,
                pending=_pending_participants(
                    operation=operation,
                    participant_by_future=participant_by_future,
                    futures=wave_pending,
                ),
            )

        ready_wave = ready
        wave_remaining = next_remaining
        drain_performed = bool(wave_remaining)
        if drain_performed:
            drained, wave_remaining = _wait_for_partition(
                operation=operation,
                phase="nonblocking drain",
                ray_runtime=ray_runtime,
                pending=wave_remaining,
                num_returns=len(wave_remaining),
                timeout_s=0.0,
            )
            ready_wave.extend(drained)

        ordered_ready_wave = _order_ready_wave(
            operation=operation,
            prior_pending=wave_pending,
            ready_wave=ready_wave,
        )
        ready_participants = [
            (
                future,
                _participant_for_future(
                    operation=operation,
                    participant_by_future=participant_by_future,
                    future=future,
                ),
            )
            for future in ordered_ready_wave
        ]
        if drain_performed and deadline - monotonic() <= 0:
            not_accepted = [future for future, _ in ready_participants]
            not_accepted.extend(wave_remaining)
            raise RefitSupervisionTimeout(
                operation=operation,
                timeout_s=normalized_timeout_s,
                pending=_pending_participants(
                    operation=operation,
                    participant_by_future=participant_by_future,
                    futures=not_accepted,
                ),
            )

        for position, (future, participant) in enumerate(ready_participants):
            get_timeout_s = deadline - monotonic()
            if get_timeout_s <= 0:
                not_accepted = [
                    pending_future
                    for pending_future, _ in ready_participants[position:]
                ]
                not_accepted.extend(wave_remaining)
                raise RefitSupervisionTimeout(
                    operation=operation,
                    timeout_s=normalized_timeout_s,
                    pending=_pending_participants(
                        operation=operation,
                        participant_by_future=participant_by_future,
                        futures=not_accepted,
                    ),
                )
            completion = _resolve_completion(
                operation=operation,
                participant=participant,
                ray_runtime=ray_runtime,
                future=future,
                result_normalizer=result_normalizer,
                get_timeout_s=get_timeout_s,
            )
            if deadline - monotonic() <= 0:
                not_accepted = [future]
                not_accepted.extend(
                    pending_future
                    for pending_future, _ in ready_participants[position + 1 :]
                )
                not_accepted.extend(wave_remaining)
                raise RefitSupervisionTimeout(
                    operation=operation,
                    timeout_s=normalized_timeout_s,
                    pending=_pending_participants(
                        operation=operation,
                        participant_by_future=participant_by_future,
                        futures=not_accepted,
                    ),
                )
            completions.append(completion)
        remaining = wave_remaining

    return RefitSupervisionSummary(
        operation=operation,
        producer_count=len(producer_refs),
        consumer_count=len(consumer_refs),
        completions=tuple(completions),
    )
