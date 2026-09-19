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

"""IPC (ZMQ) weight synchronizer for colocated vLLM generation.

Handles weight transfer between a colocated policy and vLLM generation
backend using ZMQ IPC sockets and CUDA IPC handles. This is the primary
transport for colocated vLLM deployments.

Lifecycle per sync:
  1. policy.offload_before_refit()       -- free GPU for weight staging
  2. generation.prepare_for_generation(tags=["weights"])  -- allocate buffers
  3. policy.stream_weights_via_ipc_zmq() -- send weights via ZMQ
     generation.update_weights_via_ipc_zmq() -- receive weights
  4. policy.offload_after_refit()        -- restore optimizer state
  5. generation.prepare_for_generation(tags=["kv_cache"]) -- rebuild KV cache
"""

import os
import time
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


def _cancel_refs(refs: Sequence[Any]) -> None:
    for ref in refs:
        try:
            ray.cancel(ref, force=False)
        except Exception:
            pass


class _ActiveRefitDeadline:
    """Cumulative budget for active refit work, excluding policy training."""

    def __init__(self, timeout_s: Optional[float]) -> None:
        self._remaining_s = None if timeout_s is None else max(0.0, timeout_s)
        self._started_at = time.monotonic()

    def remaining(self, operation: str) -> Optional[float]:
        if self._remaining_s is None:
            return None
        remaining_s = self._remaining_s - (time.monotonic() - self._started_at)
        if remaining_s <= 0:
            raise TimeoutError(f"IPC/ZMQ refit deadline expired before {operation}")
        return remaining_s

    def pause(self) -> Optional[float]:
        if self._remaining_s is None:
            return None
        return max(0.0, self._remaining_s - (time.monotonic() - self._started_at))

    def remaining_for_cleanup(self) -> Optional[float]:
        """Return a nonnegative budget so cleanup is dispatched even at expiry."""
        return self.pause()

    def get(
        self,
        refs: Sequence[Any],
        *,
        operation: str,
        cancel_refs: Optional[Sequence[Any]] = None,
    ) -> Any:
        try:
            timeout_s = self.remaining(operation)
            if timeout_s is None:
                return ray.get(refs)
            return ray.get(refs, timeout=timeout_s)
        except BaseException as error:
            _cancel_refs(cancel_refs if cancel_refs is not None else refs)
            error.add_note(f"IPC/ZMQ refit operation failed: {operation}")
            raise


class IPCWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using ZMQ IPC for colocated vLLM deployments.

    Both the policy and generation workers run on the same GPUs. Weights
    are transferred via CUDA IPC handles over ZMQ sockets, avoiding
    any network overhead.

    Args:
        policy: Policy object implementing ColocatablePolicyInterface.
        generation: Generation object implementing GenerationInterface
            (concretely a VllmGeneration instance).
        refit_buffer_size_gb: Fixed buffer size in GB for weight staging.
            If None, buffer size is computed dynamically from free GPU memory.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        refit_buffer_size_gb: Optional[float | int] = None,
        refit_timeout_s: Optional[float] = None,
    ):
        self._policy = policy
        self._generation = generation
        self._refit_buffer_size_gb = refit_buffer_size_gb
        self._refit_timeout_s = refit_timeout_s
        self._stale = True
        self._can_discard_generation_weights = False
        self._generation_weights_discarded = False
        self._remaining_refit_timeout_s: Optional[float] = None

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        self._stale = True
        self._can_discard_generation_weights = False
        generation_weights_discarded = self._generation_weights_discarded
        deadline = _ActiveRefitDeadline(
            self._remaining_refit_timeout_s
            if generation_weights_discarded
            else self._refit_timeout_s
        )
        coverage_complete = False
        try:
            self._call_policy_phase("offload_before_refit", deadline)
            if (
                self._prepare_generation(
                    tags=["weights"],
                    deadline=deadline,
                    operation="vLLM weight wake",
                )
                is not True
            ):
                raise RuntimeError("Failed to wake vLLM weights before IPC/ZMQ sync")

            timer_context = (
                timer.time("prepare_for_generation/transfer_and_update_weights")
                if timer is not None
                else nullcontext()
            )
            with timer_context:
                buffer_size_bytes = self._compute_buffer_size(deadline=deadline)

                futures_train = self._policy.stream_weights_via_ipc_zmq(
                    buffer_size_bytes=buffer_size_bytes,
                    kv_scales=kv_scales,
                )
                futures_inference = self._generation.update_weights_via_ipc_zmq()

                all_transfer_refs = [*futures_train, *futures_inference]
                deadline.get(
                    futures_train,
                    operation="policy IPC weight transfer",
                    cancel_refs=all_transfer_refs,
                )
                results = deadline.get(
                    futures_inference,
                    operation="vLLM IPC weight update",
                    cancel_refs=all_transfer_refs,
                )
                expected_worker_count = self._generation.dp_size
                update_success = (
                    isinstance(expected_worker_count, int)
                    and expected_worker_count > 0
                    and isinstance(futures_inference, list)
                    and bool(futures_inference)
                    and len(futures_inference) == expected_worker_count
                    and isinstance(results, list)
                    and len(results) == expected_worker_count
                    and all(result is True for result in results)
                )

                if not update_success:
                    raise RuntimeError(
                        "Weight transfer failed during IPC/ZMQ sync. "
                        "This often indicates an issue with cuda-ipc or the vLLM worker."
                    )

            coverage_complete = self._attest_generation(deadline) is True
            if generation_weights_discarded and not coverage_complete:
                raise RuntimeError(
                    "IPC/ZMQ refit did not prove complete runtime weight coverage "
                    "after vLLM generation weights were discarded"
                )
        except BaseException as primary_error:
            try:
                self._call_policy_phase(
                    "offload_after_refit", deadline, dispatch_if_expired=True
                )
            except BaseException as cleanup_error:
                primary_error.add_note(
                    f"Policy cleanup also failed after IPC/ZMQ sync: {cleanup_error!r}"
                )
            raise
        else:
            self._call_policy_phase(
                "offload_after_refit", deadline, dispatch_if_expired=True
            )

        if (
            self._prepare_generation(
                tags=["kv_cache"],
                deadline=deadline,
                operation="vLLM KV-cache wake",
            )
            is not True
        ):
            raise RuntimeError("Failed to wake vLLM KV cache after IPC/ZMQ sync")

        self._can_discard_generation_weights = coverage_complete
        self._generation_weights_discarded = False
        self._remaining_refit_timeout_s = None
        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    @property
    def can_discard_generation_weights(self) -> bool:
        return self._can_discard_generation_weights

    @property
    def generation_weights_discarded(self) -> bool:
        return self._generation_weights_discarded

    def mark_generation_weights_discarded(self) -> None:
        if not self._can_discard_generation_weights:
            raise RuntimeError("This synchronizer cannot reconstruct discarded weights")
        self._generation_weights_discarded = True
        self._remaining_refit_timeout_s = self._refit_timeout_s

    def wait_for_generation_sleep(
        self, futures: Sequence[Any], *, expected_owner_count: int
    ) -> bool:
        if (
            not isinstance(expected_owner_count, int)
            or isinstance(expected_owner_count, bool)
            or expected_owner_count <= 0
        ):
            raise ValueError("destructive sleep dp_size must be a positive integer")
        if not isinstance(futures, list) or len(futures) != expected_owner_count:
            raise RuntimeError(
                "Destructive sleep dispatched an unexpected owner count: "
                f"expected {expected_owner_count}, got "
                f"{len(futures) if isinstance(futures, list) else 'non-list'}"
            )

        deadline = _ActiveRefitDeadline(self._remaining_refit_timeout_s)
        try:
            results = deadline.get(futures, operation="destructive vLLM sleep")
        finally:
            self._remaining_refit_timeout_s = deadline.pause()
        if (
            not isinstance(results, list)
            or len(results) != expected_owner_count
            or not all(result is True for result in results)
        ):
            raise RuntimeError(
                "Destructive sleep requires literal True from every generation owner"
            )
        return True

    def invalidate_generation_weight_capability(self) -> None:
        self._can_discard_generation_weights = False

    def init_communicator(self) -> None:
        self.invalidate_generation_weight_capability()
        state_dict_info = self._policy.prepare_refit_info(
            refit_payload_mode=self._generation.get_refit_payload_mode()
        )
        self._generation.prepare_refit_info(state_dict_info)

    def shutdown(self) -> None:
        pass

    def _compute_buffer_size(
        self, *, deadline: Optional[_ActiveRefitDeadline] = None
    ) -> int:
        if self._refit_buffer_size_gb is not None:
            if self._refit_buffer_size_gb <= 0:
                raise ValueError("refit_buffer_size_gb must be > 0")
            return int(self._refit_buffer_size_gb * (1024**3))

        memory_ratio_raw = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.3")
        try:
            memory_ratio = float(memory_ratio_raw)
        except ValueError as exc:
            raise ValueError(
                f"NRL_REFIT_BUFFER_MEMORY_RATIO must be a valid float, got {memory_ratio_raw!r}"
            ) from exc
        if memory_ratio <= 0:
            raise ValueError(
                f"NRL_REFIT_BUFFER_MEMORY_RATIO must be > 0, got {memory_ratio}"
            )
        if deadline is None or self._refit_timeout_s is None:
            free_memory_bytes = self._policy.get_free_memory_bytes()
        else:
            free_memory_bytes = self._policy.get_free_memory_bytes(
                timeout_s=deadline.remaining("policy free-memory query")
            )
        return int(free_memory_bytes * memory_ratio)

    def _call_policy_phase(
        self,
        method_name: str,
        deadline: _ActiveRefitDeadline,
        *,
        dispatch_if_expired: bool = False,
    ) -> None:
        method = getattr(self._policy, method_name)
        if self._refit_timeout_s is None:
            method()
            return
        timeout_s = (
            deadline.remaining_for_cleanup()
            if dispatch_if_expired
            else deadline.remaining(f"policy {method_name}")
        )
        method(timeout_s=timeout_s)

    def _prepare_generation(
        self,
        *,
        tags: list[str],
        deadline: _ActiveRefitDeadline,
        operation: str,
    ) -> bool:
        if self._refit_timeout_s is None:
            return self._generation.prepare_for_generation(tags=tags)
        return self._generation.prepare_for_generation(
            tags=tags,
            timeout_s=deadline.remaining(operation),
        )

    def _attest_generation(self, deadline: _ActiveRefitDeadline) -> bool:
        if self._refit_timeout_s is None:
            return self._generation.refit_reconstructs_all_runtime_weights()
        return self._generation.refit_reconstructs_all_runtime_weights(
            timeout_s=deadline.remaining("vLLM runtime coverage attestation")
        )
