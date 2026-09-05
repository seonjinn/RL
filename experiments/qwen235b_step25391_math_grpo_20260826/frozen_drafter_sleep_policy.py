#!/usr/bin/env python3
"""Memory policy for colocated frozen-drafter speculative decoding."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol


ENABLE_ENV = "NRL_FROZEN_DRAFTER_DISCARD_REFIT_TARGET"
DRAFT_WEIGHT_TAG = "nemo_rl_frozen_draft_weights"


class TaggedAllocator(Protocol):
    current_tag: str


def frozen_drafter_sleep_enabled() -> bool:
    """Return whether the job opted into the refit-aware sleep policy."""
    return os.environ.get(ENABLE_ENV) == "1"


def sleep_offload_tags(*, level: int, enabled: bool) -> tuple[str, ...]:
    """Select which allocations survive a vLLM sleep in host memory."""
    if enabled and level == 1:
        return (DRAFT_WEIGHT_TAG,)
    return ("weights",) if level == 1 else ()


def wake_tags(tags: list[str] | None, *, enabled: bool) -> list[str] | None:
    """Restore a frozen drafter together with the refittable target allocation."""
    if not enabled or tags is None or "weights" not in tags:
        return None if tags is None else list(tags)
    expanded = list(tags)
    if DRAFT_WEIGHT_TAG not in expanded:
        expanded.append(DRAFT_WEIGHT_TAG)
    return expanded


@contextmanager
def draft_weight_tag(
    allocator: TaggedAllocator,
    *,
    enabled: bool,
) -> Iterator[None]:
    """Tag only frozen-drafter allocations for host backup during sleep."""
    if not enabled:
        yield
        return

    original_tag = allocator.current_tag
    allocator.current_tag = DRAFT_WEIGHT_TAG
    try:
        yield
    finally:
        allocator.current_tag = original_tag
