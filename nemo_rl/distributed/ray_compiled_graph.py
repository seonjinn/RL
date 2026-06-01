# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Optional Ray Compiled Graph execution for NeMo-RL worker groups."""

import logging
from typing import Any, Optional

import ray
from ray.dag import InputNode, MultiOutputNode

from nemo_rl.distributed.named_sharding import NamedSharding

logger = logging.getLogger(__name__)


def should_use_compiled_graph(config: Optional[dict[str, Any]]) -> bool:
    return bool(config and config.get("enabled", False))


def get_compiled_graph_config(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    return {
        "enabled": bool(config and config.get("enabled", False)),
        "overlap_communication": bool(
            config and config.get("overlap_communication", False)
        ),
    }


class CompiledGraphExecutor:
    """One compiled training DAG for one data-parallel shard."""

    def __init__(
        self,
        workers: list[ray.actor.ActorHandle],
        sharding_annotations: NamedSharding,
        method_name: str,
        dp_rank: int,
        overlap_communication: bool = False,
    ):
        self.workers = workers
        self.sharding_annotations = sharding_annotations
        self.method_name = method_name
        self.dp_rank = dp_rank
        self.overlap_communication = overlap_communication
        self.pp_size = sharding_annotations.get_axis_size("pipeline_parallel")
        self.tp_size = sharding_annotations.get_axis_size("tensor_parallel")
        self.cp_size = sharding_annotations.get_axis_size("context_parallel")
        self.worker_ranks = self._worker_ranks_for_dp()
        self.compiled_dag = self._build_and_compile_dag()

    def _worker_ranks_for_dp(self) -> list[int]:
        ranks: list[int] = []
        for pp_rank in range(self.pp_size):
            for cp_rank in range(self.cp_size):
                for tp_rank in range(self.tp_size):
                    worker_rank = self.sharding_annotations.get_ranks(
                        pipeline_parallel=pp_rank,
                        data_parallel=self.dp_rank,
                        context_parallel=cp_rank,
                        tensor_parallel=tp_rank,
                    )
                    if not isinstance(worker_rank, int):
                        raise ValueError(
                            "Expected a single worker rank for "
                            f"PP={pp_rank}, DP={self.dp_rank}, CP={cp_rank}, TP={tp_rank}; "
                            f"got {worker_rank}"
                        )
                    ranks.append(worker_rank)
        return ranks

    def _build_and_compile_dag(self) -> Any:
        with InputNode() as input_dict:
            outputs = []
            for worker_rank in self.worker_ranks:
                worker = self.workers[worker_rank]
                if self.method_name == "train":
                    outputs.append(worker.train_compiled.bind(input_dict))
                else:
                    compiled_method = getattr(
                        worker, f"{self.method_name}_compiled", None
                    )
                    if compiled_method is None:
                        raise NotImplementedError(
                            f"No compiled method wrapper for {self.method_name}"
                        )
                    outputs.append(compiled_method.bind(input_dict))
            forward_dag = MultiOutputNode(outputs)

        return forward_dag.experimental_compile(
            _overlap_gpu_communication=self.overlap_communication,
        )

    def execute(self, input_dict: dict[str, Any]) -> list[Any]:
        refs = self.compiled_dag.execute(input_dict)
        return refs if isinstance(refs, list) else [refs]

    def teardown(self) -> None:
        if hasattr(self.compiled_dag, "teardown"):
            try:
                self.compiled_dag.teardown()
            except Exception as exc:
                logger.debug("Compiled graph teardown warning: %s", exc)


class MultiDPCompiledGraphExecutor:
    """Creates one compiled DAG per data-parallel shard."""

    def __init__(
        self,
        workers: list[ray.actor.ActorHandle],
        sharding_annotations: NamedSharding,
        method_name: str,
        overlap_communication: bool = False,
    ):
        self.dp_size = sharding_annotations.get_axis_size("data_parallel")
        self.executors = [
            CompiledGraphExecutor(
                workers=workers,
                sharding_annotations=sharding_annotations,
                method_name=method_name,
                dp_rank=dp_rank,
                overlap_communication=overlap_communication,
            )
            for dp_rank in range(self.dp_size)
        ]

    def execute(self, input_dicts_by_dp: dict[int, dict[str, Any]]) -> list[Any]:
        refs: list[Any] = []
        for dp_rank, executor in enumerate(self.executors):
            refs.extend(executor.execute(input_dicts_by_dp[dp_rank]))
        return refs

    @property
    def called_workers(self) -> list[int]:
        ranks: list[int] = []
        for executor in self.executors:
            ranks.extend(executor.worker_ranks)
        return ranks

    def teardown(self) -> None:
        for executor in self.executors:
            executor.teardown()
        self.executors.clear()


class GlobalCompiledGraphExecutor:
    """Creates one compiled DAG containing every worker.

    Megatron training has collectives that cross the data-parallel axis. A
    per-DP compiled DAG can let one DP shard enter those collectives before its
    peers have been scheduled, which is unsafe for DDP/optimizer communication.
    The global DAG preserves the standard Ray call semantics: all policy workers
    receive one train call for the step from a single graph execution.
    """

    def __init__(
        self,
        workers: list[ray.actor.ActorHandle],
        sharding_annotations: NamedSharding,
        method_name: str,
        overlap_communication: bool = False,
    ):
        self.workers = workers
        self.sharding_annotations = sharding_annotations
        self.method_name = method_name
        self.overlap_communication = overlap_communication
        self.dp_size = sharding_annotations.get_axis_size("data_parallel")
        self.worker_ranks = list(range(len(workers)))
        self.compiled_dag = self._build_and_compile_dag()

    def _build_and_compile_dag(self) -> Any:
        with InputNode() as input_by_dp:
            outputs = []
            for worker_rank, worker in enumerate(self.workers):
                worker_coords = self.sharding_annotations.get_worker_coords(
                    worker_rank
                )
                dp_rank = worker_coords["data_parallel"]
                worker_input = input_by_dp[dp_rank]
                if self.method_name == "train":
                    outputs.append(worker.train_compiled.bind(worker_input))
                else:
                    compiled_method = getattr(
                        worker, f"{self.method_name}_compiled", None
                    )
                    if compiled_method is None:
                        raise NotImplementedError(
                            f"No compiled method wrapper for {self.method_name}"
                        )
                    outputs.append(compiled_method.bind(worker_input))
            forward_dag = MultiOutputNode(outputs)

        return forward_dag.experimental_compile(
            _overlap_gpu_communication=self.overlap_communication,
        )

    def execute(self, input_dicts_by_dp: dict[int, dict[str, Any]]) -> list[Any]:
        input_list = [input_dicts_by_dp[dp_rank] for dp_rank in range(self.dp_size)]
        refs = self.compiled_dag.execute(input_list)
        return refs if isinstance(refs, list) else [refs]

    @property
    def called_workers(self) -> list[int]:
        return self.worker_ranks

    def teardown(self) -> None:
        if hasattr(self.compiled_dag, "teardown"):
            try:
                self.compiled_dag.teardown()
            except Exception as exc:
                logger.debug("Compiled graph teardown warning: %s", exc)


class CompiledGraphWorkerGroup:
    """Drop-in wrapper around RayWorkerGroup for compiled training calls."""

    def __init__(
        self,
        worker_group: "RayWorkerGroup",  # type: ignore[name-defined]
        compiled_graph_config: Optional[dict[str, Any]] = None,
    ):
        self.worker_group = worker_group
        self.config = get_compiled_graph_config(compiled_graph_config)
        self.compiled_executors: dict[str, Any] = {}

    def run_all_workers_sharded_data(
        self,
        method_name: str,
        *args,
        in_sharded_axes: list[str] | None = None,
        replicate_on_axes: list[str] | None = None,
        output_is_replicated: list[str] | None = None,
        make_dummy_calls_to_free_axes: bool = False,
        common_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        if (
            not self.config["enabled"]
            or method_name != "train"
            or self.worker_group.sharding_annotations is None
        ):
            return self.worker_group.run_all_workers_sharded_data(
                method_name,
                *args,
                in_sharded_axes=in_sharded_axes,
                replicate_on_axes=replicate_on_axes,
                output_is_replicated=output_is_replicated,
                make_dummy_calls_to_free_axes=make_dummy_calls_to_free_axes,
                common_kwargs=common_kwargs,
                **kwargs,
            )

        if in_sharded_axes != ["data_parallel"] or "data" not in kwargs:
            logger.warning(
                "Compiled graph path only supports train(data=...) sharded on data_parallel; "
                "falling back to standard Ray calls."
            )
            return self.worker_group.run_all_workers_sharded_data(
                method_name,
                *args,
                in_sharded_axes=in_sharded_axes,
                replicate_on_axes=replicate_on_axes,
                output_is_replicated=output_is_replicated,
                make_dummy_calls_to_free_axes=make_dummy_calls_to_free_axes,
                common_kwargs=common_kwargs,
                **kwargs,
            )

        executor_key = f"{method_name}_compiled"
        if executor_key not in self.compiled_executors:
            logger.info("Building Ray Compiled Graph for '%s'", method_name)
            self.compiled_executors[executor_key] = GlobalCompiledGraphExecutor(
                workers=self.worker_group._workers,
                sharding_annotations=self.worker_group.sharding_annotations,
                method_name=method_name,
                overlap_communication=self.config["overlap_communication"],
            )

        executor = self.compiled_executors[executor_key]
        sharded_data = kwargs["data"]
        dp_size = self.worker_group.sharding_annotations.get_axis_size("data_parallel")
        if isinstance(sharded_data, list):
            if len(sharded_data) != dp_size:
                raise ValueError(
                    f"Expected {dp_size} data shards for compiled graph, got {len(sharded_data)}"
                )
            sharded_data = {dp_rank: sharded_data[dp_rank] for dp_rank in range(dp_size)}

        input_dicts_by_dp = {}
        for dp_rank in range(dp_size):
            input_dict = {"data": sharded_data[dp_rank]}
            if common_kwargs:
                input_dict.update(common_kwargs)
            input_dicts_by_dp[dp_rank] = input_dict

        refs = executor.execute(input_dicts_by_dp)

        if output_is_replicated is None:
            output_is_replicated = []

        called_workers = executor.called_workers
        return_from_workers = []
        for worker_rank in called_workers:
            worker_coords = self.worker_group.sharding_annotations.get_worker_coords(
                worker_rank
            )
            should_return = True
            for axis in output_is_replicated:
                if axis in worker_coords and worker_coords[axis] != 0:
                    should_return = False
                    break
            if should_return:
                return_from_workers.append(worker_rank)

        from nemo_rl.distributed.worker_groups import MultiWorkerFuture

        return MultiWorkerFuture(
            futures=refs,
            return_from_workers=return_from_workers,
            called_workers=called_workers,
        )

    def teardown(self) -> None:
        for executor in self.compiled_executors.values():
            executor.teardown()
        self.compiled_executors.clear()

    def shutdown(self, *args, **kwargs) -> Any:
        self.teardown()
        return self.worker_group.shutdown(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.worker_group, name)
