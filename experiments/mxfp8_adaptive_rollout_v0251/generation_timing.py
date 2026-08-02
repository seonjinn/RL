from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
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
