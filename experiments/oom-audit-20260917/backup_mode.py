"""Experimental per-call CuMem pinning override, used only by probe workers."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol


class PinningModule(Protocol):
    PIN_MEMORY: bool


@contextmanager
def pageable_backups(module: PinningModule, enabled: bool) -> Iterator[None]:
    previous = module.PIN_MEMORY
    try:
        if enabled:
            module.PIN_MEMORY = False
        yield
    finally:
        module.PIN_MEMORY = previous
