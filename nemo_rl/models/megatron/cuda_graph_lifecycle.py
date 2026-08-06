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

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol


def _validate_integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _validate_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


class TECudaGraphBankProtocol(Protocol):
    """Operations required from a Transformer Engine CUDA Graph bank."""

    def activate(self) -> None:
        """Install this bank as the active replay bank."""

    def reset(self) -> None:
        """Release this bank's captured graph resources."""


@dataclass(frozen=True)
class TECudaGraphScheduleKey:
    """Cache key for a pipeline schedule's runtime microbatch count."""

    num_microbatches: int

    def __post_init__(self) -> None:
        _validate_integer("num_microbatches", self.num_microbatches, minimum=1)

    @classmethod
    def from_runtime(
        cls,
        *,
        pipeline_parallel_size: int,
        num_microbatches: int,
        overlap_moe_expert_parallel_comm: bool = False,
    ) -> TECudaGraphScheduleKey:
        """Normalize the runtime schedule into a graph-bank cache key."""
        validated_pipeline_size = _validate_integer(
            "pipeline_parallel_size",
            pipeline_parallel_size,
            minimum=1,
        )
        validated_num_microbatches = _validate_integer(
            "num_microbatches",
            num_microbatches,
            minimum=1,
        )
        validated_overlap = _validate_bool(
            "overlap_moe_expert_parallel_comm",
            overlap_moe_expert_parallel_comm,
        )
        if validated_pipeline_size == 1 and not validated_overlap:
            return cls(1)
        return cls(validated_num_microbatches)


@dataclass(frozen=True)
class TECudaGraphEnsureResult:
    """Outcome of ensuring that a schedule's graph bank is active."""

    key: TECudaGraphScheduleKey
    status: Literal["warming", "hit", "captured"]
    evicted_key: TECudaGraphScheduleKey | None


@dataclass(frozen=True)
class CudaGraphStepMetrics:
    """Validated raw CUDA Graph counters for one policy-training step."""

    capture_count: int
    replay_count: int
    cache_hit_count: int
    cache_miss_count: int
    eviction_count: int
    fallback_count: int
    graph_calls: int
    eligible_calls: int
    logical_tokens: int
    padded_tokens: int
    capacity_tokens: int

    def __post_init__(self) -> None:
        for name in (
            "capture_count",
            "replay_count",
            "cache_hit_count",
            "cache_miss_count",
            "eviction_count",
            "fallback_count",
            "graph_calls",
            "eligible_calls",
            "logical_tokens",
            "padded_tokens",
            "capacity_tokens",
        ):
            _validate_integer(name, getattr(self, name), minimum=0)

        if self.fallback_count != 0:
            raise ValueError(f"fallback_count must be zero, got {self.fallback_count}")
        if self.capture_count > self.cache_miss_count:
            raise ValueError(
                "capture_count must not exceed cache_miss_count, got "
                f"{self.capture_count} > {self.cache_miss_count}"
            )
        if self.eviction_count > self.capture_count:
            raise ValueError(
                "eviction_count must not exceed capture_count, got "
                f"{self.eviction_count} > {self.capture_count}"
            )
        if self.graph_calls > self.eligible_calls:
            raise ValueError(
                "graph_calls must not exceed eligible_calls, got "
                f"{self.graph_calls} > {self.eligible_calls}"
            )
        if self.logical_tokens > self.padded_tokens:
            raise ValueError(
                "logical_tokens must not exceed padded_tokens, got "
                f"{self.logical_tokens} > {self.padded_tokens}"
            )
        if self.padded_tokens > self.capacity_tokens:
            raise ValueError(
                "padded_tokens must not exceed capacity_tokens, got "
                f"{self.padded_tokens} > {self.capacity_tokens}"
            )


class TECudaGraphLifecycle:
    """Own a bounded LRU of schedule-specific TE CUDA Graph banks."""

    def __init__(self, *, capacity: int, warmup_steps: int = 3) -> None:
        self._capacity = _validate_integer("capacity", capacity, minimum=1)
        self._warmup_steps = _validate_integer(
            "warmup_steps",
            warmup_steps,
            minimum=0,
        )
        self._successful_optimizer_steps = 0
        self._banks: OrderedDict[
            TECudaGraphScheduleKey,
            TECudaGraphBankProtocol,
        ] = OrderedDict()
        self._active_key: TECudaGraphScheduleKey | None = None
        self._closed = False

    @property
    def active_key(self) -> TECudaGraphScheduleKey | None:
        """Return the schedule key tracked as active."""
        return self._active_key

    @property
    def cached_keys(self) -> tuple[TECudaGraphScheduleKey, ...]:
        """Return cached keys from least to most recently used."""
        return tuple(self._banks)

    @property
    def successful_optimizer_steps(self) -> int:
        """Return the successful optimizer steps counted toward warmup."""
        return self._successful_optimizer_steps

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("TE CUDA Graph lifecycle is closed")

    def record_optimizer_step(self, *, successful: bool) -> None:
        """Record one optimizer result without changing cached graph banks."""
        self._raise_if_closed()
        validated_success = _validate_bool("successful", successful)
        if validated_success and self._successful_optimizer_steps < self._warmup_steps:
            self._successful_optimizer_steps += 1

    def ensure_active(
        self,
        key: TECudaGraphScheduleKey,
        capture_bank: Callable[[], TECudaGraphBankProtocol],
    ) -> TECudaGraphEnsureResult:
        """Activate a cached bank or capture and commit a missing schedule."""
        self._raise_if_closed()
        cached_bank = self._banks.get(key)
        if cached_bank is not None:
            cached_bank.activate()
            self._banks.move_to_end(key)
            self._active_key = key
            return TECudaGraphEnsureResult(
                key=key,
                status="hit",
                evicted_key=None,
            )

        if self._successful_optimizer_steps < self._warmup_steps:
            return TECudaGraphEnsureResult(
                key=key,
                status="warming",
                evicted_key=None,
            )

        new_bank = capture_bank()
        try:
            new_bank.activate()
        except Exception:
            if all(bank is not new_bank for bank in self._banks.values()):
                try:
                    new_bank.reset()
                except Exception:
                    pass
            raise

        evicted_key: TECudaGraphScheduleKey | None = None
        evicted_bank: TECudaGraphBankProtocol | None = None
        if len(self._banks) >= self._capacity:
            evicted_key, evicted_bank = self._banks.popitem(last=False)

        self._banks[key] = new_bank
        self._active_key = key

        if evicted_bank is not None and all(
            bank is not evicted_bank for bank in self._banks.values()
        ):
            evicted_bank.reset()

        return TECudaGraphEnsureResult(
            key=key,
            status="captured",
            evicted_key=evicted_key,
        )

    def _detach_distinct_banks(self) -> list[TECudaGraphBankProtocol]:
        distinct_banks: list[TECudaGraphBankProtocol] = []
        seen_bank_ids: set[int] = set()
        for bank in self._banks.values():
            if id(bank) in seen_bank_ids:
                continue
            seen_bank_ids.add(id(bank))
            distinct_banks.append(bank)

        self._banks.clear()
        self._active_key = None
        return distinct_banks

    @staticmethod
    def _reset_distinct_banks(banks: list[TECudaGraphBankProtocol]) -> None:
        first_error: Exception | None = None
        for bank in banks:
            try:
                bank.reset()
            except Exception as error:
                if first_error is None:
                    first_error = error

        if first_error is not None:
            raise first_error

    def reset_banks(self) -> None:
        """Clear graph banks while preserving successful warmup progress."""
        self._raise_if_closed()
        distinct_banks = self._detach_distinct_banks()
        self._reset_distinct_banks(distinct_banks)

    def close(self) -> None:
        """Terminally close the lifecycle and reset every distinct bank once."""
        if self._closed:
            return

        distinct_banks = self._detach_distinct_banks()
        self._successful_optimizer_steps = 0
        self._closed = True
        self._reset_distinct_banks(distinct_banks)
