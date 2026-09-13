"""Borrow DDP backups for reference swaps without interpreting weight layouts."""

from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

import torch


@contextmanager
def borrowed_cpu_parameter_views(
    model: torch.nn.Module, buffers: Iterable[Any]
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield local parameter names backed by fresh, borrowed CPU snapshots.

    Pass the unwrapped local model chunk. Callers save unmatched state entries
    independently and restore weights before exiting this context.
    """
    with ExitStack() as stack:
        owners: dict[torch.nn.Parameter, torch.Tensor] = {}
        for buffer in buffers:
            borrow = getattr(buffer, "borrow_cpu_param_snapshot", None)
            if borrow is None:
                continue
            try:
                views = stack.enter_context(borrow())
            except NotImplementedError:
                continue
            for owner, view in views.items():
                if owner in owners:
                    raise RuntimeError("Parameter belongs to multiple snapshot buffers")
                if view.device.type != "cpu" or view.dtype != owner.dtype:
                    raise RuntimeError("Borrowed snapshot has incompatible device or dtype")
                if view.numel() != owner.numel():
                    raise RuntimeError("Borrowed snapshot has incompatible element count")
                owners[owner] = view
        yield {
            name: owners[owner]
            for name, owner in model.named_parameters(remove_duplicate=False)
            if owner in owners
        }
