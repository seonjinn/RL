# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import threading
from typing import Any

import torch


DistributedOptimizer: type[Any] | None = None
OptimizerStateOffloader: type[Any] | None = None
_TRANSFER_LOCK_CREATION_LOCK = threading.Lock()


def _load_mcore_offloader_types() -> tuple[type[Any], type[Any]]:
    global DistributedOptimizer, OptimizerStateOffloader

    if DistributedOptimizer is None:
        from megatron.core.optimizer.distrib_optimizer import (
            DistributedOptimizer as MCoreDistributedOptimizer,
        )

        DistributedOptimizer = MCoreDistributedOptimizer
    if OptimizerStateOffloader is None:
        from megatron.core.optimizer.cpu_offloading.optimizer_state_offloader import (
            OptimizerStateOffloader as MCoreOptimizerStateOffloader,
        )

        OptimizerStateOffloader = MCoreOptimizerStateOffloader
    assert DistributedOptimizer is not None
    assert OptimizerStateOffloader is not None
    return DistributedOptimizer, OptimizerStateOffloader


def _get_distributed_optimizers(optimizer: Any) -> list[Any]:
    distributed_optimizer_type, _ = _load_mcore_offloader_types()
    if isinstance(optimizer, distributed_optimizer_type):
        return [optimizer]

    chained_optimizers = getattr(optimizer, "chained_optimizers", None)
    if not chained_optimizers or not all(
        isinstance(item, distributed_optimizer_type) for item in chained_optimizers
    ):
        return []
    return list(chained_optimizers)


def _get_offloaders(distributed_optimizers: list[Any]) -> list[Any]:
    _, offloader_type = _load_mcore_offloader_types()
    offloaders = []
    for distributed_optimizer in distributed_optimizers:
        offloader = getattr(
            distributed_optimizer,
            "_nemo_rl_optimizer_state_offloader",
            None,
        )
        if offloader is None:
            offloader = offloader_type(distributed_optimizer)
            distributed_optimizer._nemo_rl_optimizer_state_offloader = offloader
        offloaders.append(offloader)
    return offloaders


def _get_transfer_lock(optimizer: Any) -> threading.Lock:
    with _TRANSFER_LOCK_CREATION_LOCK:
        lock = getattr(optimizer, "_nemo_rl_optimizer_state_transfer_lock", None)
        if lock is None:
            lock = threading.Lock()
            optimizer._nemo_rl_optimizer_state_transfer_lock = lock
        return lock


def is_distributed_optimizer_state_offloaded(optimizer: Any) -> bool:
    """Return whether any managed distributed optimizer child is offloaded."""
    distributed_optimizers = _get_distributed_optimizers(optimizer)
    return any(
        getattr(
            distributed_optimizer,
            "_nemo_rl_optimizer_state_offloader",
            None,
        )
        is not None
        and distributed_optimizer._nemo_rl_optimizer_state_offloader.is_offloaded
        for distributed_optimizer in distributed_optimizers
    )


def move_distributed_optimizer_state(optimizer: Any, device: str) -> bool:
    """Move MCore distributed optimizer state without rebinding optimizer tensors.

    Returns ``False`` when the optimizer is not an MCore ``DistributedOptimizer``
    (or a chain composed entirely of them), allowing callers to use their existing
    fallback path.
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError(
            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
        )

    distributed_optimizers = _get_distributed_optimizers(optimizer)
    if not distributed_optimizers:
        return False

    lock = _get_transfer_lock(optimizer)
    if not lock.acquire(blocking=False):
        raise RuntimeError("Optimizer state transfer is already in progress")

    try:
        try:
            offloaders = _get_offloaders(distributed_optimizers)
        except (AssertionError, ImportError):
            return False
        if device == "cpu":
            pending_offloaders = [
                offloader for offloader in offloaders if not offloader.is_offloaded
            ]
            try:
                for offloader in pending_offloaders:
                    if offloader.adam_optimizer.state:
                        offloader.mark_optimizer_states_initialized()
                    offloader.offload()
            except Exception:
                completed_offloaders = [
                    offloader
                    for offloader in pending_offloaders
                    if offloader.is_offloaded
                ]
                if completed_offloaders:
                    torch.cuda.synchronize()
                    for offloader in completed_offloaders:
                        offloader.release_gpu_memory()
                raise

            if pending_offloaders:
                # Preserve the old blocking move_optimizer contract. Synchronizing
                # once after all pinned copies avoids per-state pageable D2H waits.
                torch.cuda.synchronize()
                for offloader in pending_offloaders:
                    offloader.release_gpu_memory()
        else:
            pending_offloaders = [
                offloader for offloader in offloaders if offloader.is_offloaded
            ]
            completed_offloaders = []
            try:
                for offloader in pending_offloaders:
                    offloader.reload()
                    completed_offloaders.append(offloader)
            finally:
                for offloader in completed_offloaders:
                    offloader.sync_before_step()
    finally:
        lock.release()

    return True
