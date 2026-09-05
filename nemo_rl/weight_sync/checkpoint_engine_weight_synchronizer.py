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

from contextlib import nullcontext
from dataclasses import InitVar, dataclass, field
from time import monotonic
from typing import Any, Optional

import ray

from nemo_rl.models.generation.interfaces import CheckpointEngineConfig
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from nemo_rl.weight_sync.refit_supervisor import (
    normalize_current_refit_result,
    normalize_refit_timeout_s,
    supervise_refit_futures,
)

_MEBIBYTE = 1024 * 1024
_CHECKPOINT_ENGINE_REFIT_TIMEOUT_S = 300.0
_CHECKPOINT_ENGINE_FINALIZE_TIMEOUT_S = 10.0


def _flatten_metadata(results: list[Any]) -> list[Any]:
    return [
        item
        for result in results
        for item in (result if isinstance(result, (list, tuple)) else [result])
    ]


def _sort_ranked_metadata(metadata: list[Any]) -> list[Any]:
    if all(isinstance(item, dict) and "rank" in item for item in metadata):
        return sorted(metadata, key=lambda item: item["rank"])
    return metadata


def _ordered_generation_metadata(generation_results: list[Any]) -> list[Any]:
    """Order vLLM generation metadata by global rollout rank.

    Each result belongs to one vLLM data-parallel group. Engine-local ranks
    are unique only within a group, so sort each group before concatenating
    them in worker-group order.
    """
    metadata: list[Any] = []
    for group_result in generation_results:
        group_metadata = (
            list(group_result)
            if isinstance(group_result, (list, tuple))
            else [group_result]
        )
        metadata.extend(_sort_ranked_metadata(group_metadata))
    return metadata


@dataclass
class CheckpointEngineWeightSynchronizer(WeightSynchronizer):
    """Coordinate checkpoint-engine setup and policy-to-rollout transfers.

    ``refit_timeout_s`` is one finite controller deadline for the producer and
    consumer futures. Passing ``None`` selects the 300-second compatibility default;
    it never disables supervision.
    """

    _policy: Any
    _generation: Any
    _checkpoint_engine_config: CheckpointEngineConfig
    refit_timeout_s: InitVar[float | int | None] = None
    _stale: bool = True
    _checkpoint_engine_ready: bool = False
    _bucket_size_bytes: int | None = None
    _refit_timeout_s: float = field(init=False)

    def __post_init__(self, refit_timeout_s: float | int | None) -> None:
        self._refit_timeout_s = normalize_refit_timeout_s(
            _CHECKPOINT_ENGINE_REFIT_TIMEOUT_S
            if refit_timeout_s is None
            else refit_timeout_s
        )

    def init_communicator(self) -> None:
        deadline_s = monotonic() + self._refit_timeout_s
        state_dict_info = self._policy.prepare_refit_info(
            refit_timeout_s=self._remaining_deadline_s(
                deadline_s=deadline_s,
                phase="policy refit-metadata preparation",
            )
        )
        self._generation.prepare_refit_info(
            state_dict_info,
            refit_timeout_s=self._remaining_deadline_s(
                deadline_s=deadline_s,
                phase="generation refit-metadata preparation",
            ),
        )
        self._ensure_checkpoint_engine_ready(deadline_s=deadline_s)

    @property
    def is_stale(self) -> bool:
        return self._stale

    def _release_after_refit(self) -> bool:
        cfg = self._checkpoint_engine_config
        return bool(cfg["engine_kwargs"][cfg["backend"]]["release_after_refit"])

    def _run_policy(
        self, checkpoint_method: str, **method_kwargs: Any
    ) -> list[ray.ObjectRef]:
        return self._policy.worker_group.run_all_workers_single_data(
            "checkpoint_engine_rpc",
            checkpoint_method=checkpoint_method,
            method_kwargs=method_kwargs,
        )

    def _generation_rpc(self) -> str:
        return (
            "checkpoint_engine_rpc_async"
            if self._generation.cfg["vllm_cfg"]["async_engine"]
            else "checkpoint_engine_rpc"
        )

    def _run_generation(
        self, checkpoint_method: str, method_args: tuple[Any, ...] = ()
    ) -> list[ray.ObjectRef]:
        return self._generation.worker_group.run_all_workers_single_data(
            self._generation_rpc(),
            checkpoint_method=checkpoint_method,
            method_args=method_args,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )

    def _participant_shape(self) -> tuple[int, int, int]:
        policy_worker_count = len(self._policy.worker_group.workers)
        generation_worker_count = len(self._generation.worker_group.workers)
        generation_group_count = self._generation.dp_size
        if policy_worker_count <= 0:
            raise RuntimeError("Checkpoint-engine policy worker group is empty.")
        if type(generation_group_count) is not int or generation_group_count <= 0:
            raise RuntimeError(
                "Checkpoint-engine generation dp_size must be an exact positive int."
            )
        if generation_worker_count <= 0 or (
            generation_worker_count % generation_group_count != 0
        ):
            raise RuntimeError(
                "Checkpoint-engine generation workers must divide evenly across "
                f"dp_size={generation_group_count}, got {generation_worker_count}."
            )
        return (
            policy_worker_count,
            generation_group_count,
            generation_worker_count // generation_group_count,
        )

    def _remaining_deadline_s(self, *, deadline_s: float, phase: str) -> float:
        remaining_s = deadline_s - monotonic()
        if remaining_s <= 0:
            raise TimeoutError(
                f"Checkpoint-engine {phase} exceeded its shared operation deadline."
            )
        return remaining_s

    def _get_phase_results(
        self,
        *,
        phase: str,
        policy_refs: list[ray.ObjectRef],
        generation_refs: list[ray.ObjectRef],
        deadline_s: float,
    ) -> tuple[list[Any], list[Any]]:
        policy_count, _ = self._require_phase_future_counts(
            phase=phase,
            policy_refs=policy_refs,
            generation_refs=generation_refs,
        )

        futures = policy_refs + generation_refs
        results = ray.get(
            futures,
            timeout=self._remaining_deadline_s(
                deadline_s=deadline_s,
                phase=phase,
            ),
        )
        if type(results) is not list or len(results) != len(futures):
            actual_count = len(results) if isinstance(results, list) else "non-list"
            raise RuntimeError(
                f"Checkpoint-engine {phase} expected exactly {len(futures)} "
                f"participant results, got {actual_count}."
            )
        return results[:policy_count], results[policy_count:]

    def _require_phase_future_counts(
        self,
        *,
        phase: str,
        policy_refs: list[ray.ObjectRef],
        generation_refs: list[ray.ObjectRef],
    ) -> tuple[int, int]:
        policy_count, generation_count, _ = self._participant_shape()
        for role, refs, expected_count in (
            ("policy", policy_refs, policy_count),
            ("generation", generation_refs, generation_count),
        ):
            if type(refs) is not list or len(refs) != expected_count:
                actual_count = len(refs) if isinstance(refs, list) else "non-list"
                raise RuntimeError(
                    f"Checkpoint-engine {phase} expected exactly {expected_count} "
                    f"{role} participant futures, got {actual_count}."
                )
        return policy_count, generation_count

    def _generation_group_results(
        self, generation_results: list[Any], *, phase: str
    ) -> list[list[Any]]:
        _, generation_group_count, workers_per_group = self._participant_shape()
        if len(generation_results) != generation_group_count:
            raise RuntimeError(
                f"Checkpoint-engine {phase} expected exactly "
                f"{generation_group_count} generation group results, got "
                f"{len(generation_results)}."
            )

        groups: list[list[Any]] = []
        for group_index, group_result in enumerate(generation_results):
            if not isinstance(group_result, (list, tuple)) or (
                len(group_result) != workers_per_group
            ):
                actual_count = (
                    len(group_result)
                    if isinstance(group_result, (list, tuple))
                    else "non-sequence"
                )
                raise RuntimeError(
                    f"Checkpoint-engine {phase} expected generation group "
                    f"{group_index} to return exactly {workers_per_group} results, "
                    f"got {actual_count}."
                )
            groups.append(list(group_result))
        return groups

    def _require_void_phase_results(
        self,
        *,
        phase: str,
        policy_results: list[Any],
        generation_results: list[Any],
    ) -> None:
        for rank, result in enumerate(policy_results):
            if result is not None:
                raise RuntimeError(
                    f"Checkpoint-engine {phase} policy rank {rank} must return "
                    "exact None."
                )
        for group_index, group_results in enumerate(
            self._generation_group_results(generation_results, phase=phase)
        ):
            for rank, result in enumerate(group_results):
                if result is not None:
                    raise RuntimeError(
                        f"Checkpoint-engine {phase} generation group {group_index} "
                        f"rank {rank} must return exact None."
                    )

    def _resolve_bucket_size_bytes(self, *, deadline_s: float | None = None) -> int:
        if self._bucket_size_bytes is not None:
            return self._bucket_size_bytes

        memory_ratio_raw = self._checkpoint_engine_config[
            "update_weights_bucket_memory_ratio"
        ]
        try:
            memory_ratio = float(memory_ratio_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "update_weights_bucket_memory_ratio must be a valid float, got "
                f"{memory_ratio_raw!r}."
            ) from exc
        if not 0 < memory_ratio < 1:
            raise ValueError(
                "update_weights_bucket_memory_ratio must be between 0 and 1, got "
                f"{memory_ratio_raw!r}."
            )

        if deadline_s is None:
            deadline_s = monotonic() + self._refit_timeout_s
        policy_results, generation_results = self._get_phase_results(
            phase="bucket-memory-discovery",
            policy_refs=self._run_policy("checkpoint_engine_total_memory_bytes"),
            generation_refs=self._run_generation(
                "checkpoint_engine_total_memory_bytes"
            ),
            deadline_s=deadline_s,
        )
        total_memory = policy_results + [
            value
            for group_results in self._generation_group_results(
                generation_results,
                phase="bucket-memory-discovery",
            )
            for value in group_results
        ]
        for participant_index, value in enumerate(total_memory):
            if type(value) is not int or value <= 0:
                raise RuntimeError(
                    "Checkpoint-engine bucket-memory-discovery participant "
                    f"{participant_index} must return an exact positive int."
                )
        self._remaining_deadline_s(
            deadline_s=deadline_s,
            phase="bucket-memory-discovery",
        )
        minimum_total_bytes = min(total_memory)
        bucket_size_bytes = int(minimum_total_bytes * memory_ratio)
        bucket_size_bytes = bucket_size_bytes // _MEBIBYTE * _MEBIBYTE
        if bucket_size_bytes < _MEBIBYTE:
            raise ValueError(
                "Checkpoint-engine bucket sizing produced less than 1 MiB per buffer."
            )

        self._bucket_size_bytes = bucket_size_bytes
        print(
            "[checkpoint engine] Bucket size: "
            f"{bucket_size_bytes // _MEBIBYTE} MiB per buffer "
            f"({memory_ratio:.1%} of {minimum_total_bytes / 1024**3:.2f} GiB "
            "minimum total GPU memory)."
        )
        return bucket_size_bytes

    def _ensure_checkpoint_engine_ready(
        self, *, deadline_s: float | None = None
    ) -> None:
        if self._checkpoint_engine_ready:
            return

        if deadline_s is None:
            deadline_s = monotonic() + self._refit_timeout_s
        cfg = self._checkpoint_engine_config
        backend = cfg["backend"]
        bucket_size_bytes = self._resolve_bucket_size_bytes(deadline_s=deadline_s)
        engine_kwargs = cfg["engine_kwargs"][backend]

        try:
            policy_init_refs = self._run_policy(
                "init_checkpoint_engine",
                backend=backend,
                bucket_size_bytes=bucket_size_bytes,
                engine_kwargs=engine_kwargs,
            )
            generation_init_refs = self._run_generation(
                "init_checkpoint_engine",
                (backend, bucket_size_bytes, engine_kwargs),
            )
            policy_init_results, generation_init_results = self._get_phase_results(
                phase="engine-initialization",
                policy_refs=policy_init_refs,
                generation_refs=generation_init_refs,
                deadline_s=deadline_s,
            )
            self._require_void_phase_results(
                phase="engine-initialization",
                policy_results=policy_init_results,
                generation_results=generation_init_results,
            )
            self._remaining_deadline_s(
                deadline_s=deadline_s,
                phase="engine-initialization",
            )

            policy_prepare_refs = self._run_policy("prepare_checkpoint_engine")
            generation_prepare_refs = self._run_generation("prepare_checkpoint_engine")
            policy_prepare_results, generation_prepare_results = (
                self._get_phase_results(
                    phase="metadata-preparation",
                    policy_refs=policy_prepare_refs,
                    generation_refs=generation_prepare_refs,
                    deadline_s=deadline_s,
                )
            )
            policy_metadata = _sort_ranked_metadata(
                _flatten_metadata(policy_prepare_results)
            )
            generation_groups = self._generation_group_results(
                generation_prepare_results,
                phase="metadata-preparation",
            )
            generation_metadata = _ordered_generation_metadata(generation_groups)
            topology = {
                "metadata": policy_metadata + generation_metadata,
                "train_world_size": len(policy_metadata),
                "rollout_world_size": len(generation_metadata),
            }
            self._remaining_deadline_s(
                deadline_s=deadline_s,
                phase="metadata-preparation",
            )
            worker_count = len(self._generation.worker_group.workers)
            workers_per_group = worker_count // self._generation.dp_size
            policy_group_refs = self._run_policy(
                "init_checkpoint_engine_process_group", **topology
            )
            generation_group_refs = (
                self._generation.worker_group.run_all_workers_multiple_data(
                    self._generation_rpc(),
                    method_args=[
                        (
                            rank_prefix,
                            topology["train_world_size"],
                            topology["rollout_world_size"],
                            topology["metadata"],
                        )
                        for rank_prefix in range(0, worker_count, workers_per_group)
                    ],
                    run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
                    common_kwargs={
                        "checkpoint_method": "init_checkpoint_engine_process_group"
                    },
                )
            )
            policy_group_results, generation_group_results = self._get_phase_results(
                phase="process-group-initialization",
                policy_refs=policy_group_refs,
                generation_refs=generation_group_refs,
                deadline_s=deadline_s,
            )
            self._require_void_phase_results(
                phase="process-group-initialization",
                policy_results=policy_group_results,
                generation_results=generation_group_results,
            )
            self._remaining_deadline_s(
                deadline_s=deadline_s,
                phase="process-group-initialization",
            )
        except BaseException:
            self._best_effort_finalize_after_startup_failure()
            self._checkpoint_engine_ready = False
            raise

        self._checkpoint_engine_ready = True

    def _finalize_checkpoint_engine(self) -> None:
        deadline_s = monotonic() + _CHECKPOINT_ENGINE_FINALIZE_TIMEOUT_S
        policy_results, generation_results = self._get_phase_results(
            phase="finalization",
            policy_refs=self._run_policy("finalize_checkpoint_engine"),
            generation_refs=self._run_generation("finalize_checkpoint_engine"),
            deadline_s=deadline_s,
        )
        self._require_void_phase_results(
            phase="finalization",
            policy_results=policy_results,
            generation_results=generation_results,
        )
        self._remaining_deadline_s(
            deadline_s=deadline_s,
            phase="finalization",
        )
        self._checkpoint_engine_ready = False

    def _best_effort_finalize_after_startup_failure(self) -> None:
        try:
            self._finalize_checkpoint_engine()
        except BaseException as cleanup_failure:
            try:
                print(
                    "[checkpoint engine] Bounded cleanup after startup failure also "
                    f"failed: {cleanup_failure!r}",
                    flush=True,
                )
            except BaseException:
                # Cleanup diagnostics must not replace the startup failure.
                pass

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        self._stale = True
        self._ensure_checkpoint_engine_ready()
        context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )

        with context:
            policy_refs = self._run_policy(
                "send_weights_via_checkpoint_engine", kv_scales=kv_scales
            )
            generation_refs = self._run_generation(
                "update_weights_from_checkpoint_engine"
            )
            self._require_phase_future_counts(
                phase="weight-transfer",
                policy_refs=policy_refs,
                generation_refs=generation_refs,
            )
            backend = self._checkpoint_engine_config["backend"]
            supervise_refit_futures(
                operation=f"{backend}-checkpoint-engine-weight-sync",
                producer_futures=policy_refs,
                consumer_futures=generation_refs,
                result_normalizer=normalize_current_refit_result,
                timeout_s=self._refit_timeout_s,
            )

        if self._release_after_refit():
            self.shutdown()
        self._stale = False

    def shutdown(self) -> None:
        if not self._checkpoint_engine_ready:
            return
        self._finalize_checkpoint_engine()
