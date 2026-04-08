"""Single-threaded background writer for overlapping CPU work with GPU processing."""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class BackgroundWriter:
    """Runs one callable at a time in a background thread.

    Useful for overlapping expensive CPU-bound serialization (e.g., .tolist() + JSON)
    with GPU work in the next training step. Each ``submit`` waits for the previous
    call to finish before starting the new one.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Wait for previous work, then submit *fn* to the background thread."""
        self.drain()
        self._future = self._executor.submit(fn, *args, **kwargs)

    def drain(self) -> None:
        """Block until the pending background work completes (re-raises exceptions)."""
        if self._future is not None:
            self._future.result()
            self._future = None
