"""Experimental reference snapshot; baseline implementation for a failing test."""

from collections.abc import Mapping, Sequence

import torch


def snapshot_policy_state(
    state: Mapping[str, object],
    buffers: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, object], int]:
    return {
        key: value.detach().to("cpu", non_blocking=True, copy=True)
        if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }, 0
