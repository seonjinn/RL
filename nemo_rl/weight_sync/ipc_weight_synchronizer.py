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
from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


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
    ):
        self._policy = policy
        self._generation = generation
        self._refit_buffer_size_gb = refit_buffer_size_gb
        self._stale = True
        self._can_discard_generation_weights = False
        self._generation_weights_discarded = False

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        self._stale = True
        self._can_discard_generation_weights = False
        generation_weights_discarded = self._generation_weights_discarded
        coverage_complete = False
        try:
            self._policy.offload_before_refit()
            if self._generation.prepare_for_generation(tags=["weights"]) is not True:
                raise RuntimeError("Failed to wake vLLM weights before IPC/ZMQ sync")

            timer_context = (
                timer.time("prepare_for_generation/transfer_and_update_weights")
                if timer is not None
                else nullcontext()
            )
            with timer_context:
                buffer_size_bytes = self._compute_buffer_size()

                futures_train = self._policy.stream_weights_via_ipc_zmq(
                    buffer_size_bytes=buffer_size_bytes,
                    kv_scales=kv_scales,
                )
                futures_inference = self._generation.update_weights_via_ipc_zmq()

                ray.get(futures_train)
                results = ray.get(futures_inference)
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

            coverage_complete = (
                self._generation.refit_reconstructs_all_runtime_weights() is True
            )
            if generation_weights_discarded and not coverage_complete:
                raise RuntimeError(
                    "IPC/ZMQ refit did not prove complete runtime weight coverage "
                    "after vLLM generation weights were discarded"
                )
        except BaseException as primary_error:
            try:
                self._policy.offload_after_refit()
            except BaseException as cleanup_error:
                primary_error.add_note(
                    f"Policy cleanup also failed after IPC/ZMQ sync: {cleanup_error!r}"
                )
            raise
        else:
            self._policy.offload_after_refit()

        if self._generation.prepare_for_generation(tags=["kv_cache"]) is not True:
            raise RuntimeError("Failed to wake vLLM KV cache after IPC/ZMQ sync")

        self._can_discard_generation_weights = coverage_complete
        self._generation_weights_discarded = False
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

    def init_communicator(self) -> None:
        state_dict_info = self._policy.prepare_refit_info(
            refit_payload_mode=self._generation.get_refit_payload_mode()
        )
        self._generation.prepare_refit_info(state_dict_info)

    def shutdown(self) -> None:
        pass

    def _compute_buffer_size(self) -> int:
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
        return int(self._policy.get_free_memory_bytes() * memory_ratio)
