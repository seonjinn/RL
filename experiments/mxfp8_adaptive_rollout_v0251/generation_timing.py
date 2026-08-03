from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class AsyncCallTimer:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self.calls = 0
        self.elapsed_seconds = 0.0

    def wrap(self, function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(function)
        async def timed(*args: P.args, **kwargs: P.kwargs) -> R:
            start = self._clock()
            try:
                return await function(*args, **kwargs)
            finally:
                self.elapsed_seconds += self._clock() - start
                self.calls += 1

        return timed


class GenerationLengthAudit:
    def __init__(self) -> None:
        self._lengths: list[int] = []

    def record(self, lengths: Iterable[int]) -> None:
        parsed = [int(length) for length in lengths]
        if any(length < 0 for length in parsed):
            raise ValueError("generated token lengths must be nonnegative")
        self._lengths.extend(parsed)

    @property
    def request_count(self) -> int:
        return len(self._lengths)

    @property
    def total_tokens(self) -> int:
        return sum(self._lengths)

    @property
    def min_tokens(self) -> int | None:
        return min(self._lengths) if self._lengths else None

    @property
    def max_tokens(self) -> int | None:
        return max(self._lengths) if self._lengths else None

    def validate(
        self, *, expected_requests: int, expected_tokens_per_response: int
    ) -> None:
        if expected_requests <= 0 or expected_tokens_per_response <= 0:
            raise ValueError("expected request and token counts must be positive")
        if self.request_count != expected_requests or any(
            length != expected_tokens_per_response for length in self._lengths
        ):
            raise RuntimeError(
                "forced output length mismatch: "
                f"requests={self.request_count}/{expected_requests}, "
                f"min_tokens={self.min_tokens}, max_tokens={self.max_tokens}, "
                f"expected_tokens_per_response={expected_tokens_per_response}"
            )
