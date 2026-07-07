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
import ast
import builtins
import copy
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class _Tensor:
    def __init__(self, value: float | list[float]) -> None:
        self.value = list(value) if isinstance(value, list) else float(value)
        self.device = "cpu"

    def add_(self, value: float) -> "_Tensor":
        assert isinstance(self.value, float)
        self.value += value
        return self

    def clamp(self, *, min: float) -> "_Tensor":
        assert isinstance(self.value, float)
        return _Tensor(max(self.value, min))

    def clone(self) -> "_Tensor":
        return _Tensor(self.value)

    def copy_(self, other: "_Tensor") -> "_Tensor":
        self.value = list(other.value) if isinstance(other.value, list) else other.value
        return self

    def cpu(self) -> "_Tensor":
        return self

    def detach(self) -> "_Tensor":
        return self

    def fill_(self, value: float) -> "_Tensor":
        self.value = value
        return self

    def float(self) -> "_Tensor":
        return self

    def item(self) -> builtins.float:
        if isinstance(self.value, list):
            assert len(self.value) == 1
            return self.value[0]
        return self.value

    def sum(self) -> "_Tensor":
        return _Tensor(sum(self.value) if isinstance(self.value, list) else self.value)

    def zero_(self) -> "_Tensor":
        self.value = 0.0
        return self

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Tensor) and self.value == other.value

    def __rtruediv__(self, numerator: builtins.float) -> "_Tensor":
        return _Tensor(numerator / self.item())

    def __truediv__(self, denominator: builtins.float) -> "_Tensor":
        return _Tensor(self.item() / denominator)


class _FakeTorch:
    Tensor = _Tensor

    class cuda:
        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def get_device_name() -> str:
            return "fake-gpu"

    class distributed:
        class ReduceOp:
            SUM = object()

        runtime: "_Runtime"

        @classmethod
        def all_reduce(cls, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            cls.runtime.events.append(("dataset_all_reduce",))

        @staticmethod
        def get_rank() -> int:
            return 0

    @staticmethod
    def no_grad():
        return nullcontext()

    @staticmethod
    def stack(values: list[_Tensor]) -> _Tensor:
        return _Tensor([value.item() for value in values])

    @staticmethod
    def tensor(value: float | list[float], **kwargs: Any) -> _Tensor:
        del kwargs
        return _Tensor(value)


class _ChainedOptimizer:
    pass


def _param_sync_func() -> None:
    pass


class _FaithfulDDP:
    def __init__(self, *, shared_param_grad_buffer: bool, runtime: "_Runtime") -> None:
        self.config = SimpleNamespace(
            moe_grad_scale_func=None,
            mtp_grad_scale_func=None,
            mtp_num_layers=0,
            num_moe_experts=0,
            param_sync_func=_param_sync_func,
        )
        self.extra_state = _Tensor(3.0)
        self.grad_buffer = _Tensor(0.0)
        self.param_buffer = (
            self.grad_buffer if shared_param_grad_buffer else _Tensor(2.0)
        )
        self.remove_forward_pre_hook_handles = {"hook": object()}
        self.runtime = runtime
        self.training = True
        self.weight = _Tensor(2.0)

    def disable_forward_pre_hook(self, param_sync: bool = True) -> None:
        assert self.remove_forward_pre_hook_handles
        self.remove_forward_pre_hook_handles.clear()
        self.runtime.events.append(("ddp_disable_forward_pre_hook", param_sync))
        if param_sync:
            self.start_param_sync(force_sync=True)

    def enable_forward_pre_hook(self) -> None:
        assert not self.remove_forward_pre_hook_handles
        self.remove_forward_pre_hook_handles["hook"] = object()
        self.runtime.events.append(("ddp_enable_forward_pre_hook",))

    def eval(self) -> None:
        self.training = False

    def finish_forward_pre_hook(self) -> None:
        if self.remove_forward_pre_hook_handles:
            self.start_param_sync(force_sync=False)

    def load_state_dict(
        self, state_dict: dict[str, _Tensor], strict: bool = False
    ) -> None:
        del strict
        self.extra_state.copy_(state_dict["layer._extra_state"])

    def modules(self) -> list[object]:
        return []

    def start_param_sync(self, *, force_sync: bool) -> None:
        self.runtime.events.append(("param_sync", force_sync))
        self.weight.copy_(self.param_buffer)

    def state_dict(self) -> dict[str, _Tensor]:
        return {
            "layer._extra_state": self.extra_state,
            "weight": self.weight,
        }

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def zero_grad_buffer(self) -> None:
        self.runtime.events.append(("model_zero_grad_buffer",))
        self.grad_buffer.zero_()


class _FaithfulOptimizer:
    def __init__(
        self,
        model: _FaithfulDDP,
        *,
        shared_param_grad_buffer: bool,
        runtime: "_Runtime",
    ) -> None:
        self.main_weight = model.weight.clone()
        self.model = model
        self.param_groups = [{}]
        self.runtime = runtime
        self.shared_param_grad_buffer = shared_param_grad_buffer
        self.state = {"last_grad": 0.0, "step": 0}

    def _copy_main_params_to_param_buffer(self) -> None:
        self.runtime.events.append(("copy_main_params_to_param_buffer",))
        self.model.param_buffer.copy_(self.main_weight)

    def step(self) -> tuple[bool, float, float]:
        grad = self.model.grad_buffer.item()
        self.main_weight.add_(grad)
        self.state["last_grad"] = grad
        self.state["step"] += 1
        if not self.shared_param_grad_buffer:
            self.model.weight.copy_(self.main_weight)
            self.model.param_buffer.copy_(self.main_weight)
        self.runtime.events.append(("optimizer_step", grad))
        return True, abs(grad), 0.0

    def zero_grad(self) -> None:
        self.runtime.events.append(("optimizer_zero_grad",))


class _FaithfulScheduler:
    def __init__(self) -> None:
        self.param_sync_samples = 0
        self.state = {"steps": 0}

    def get_lr(self, param_group: dict[str, object]) -> float:
        del param_group
        return 1.0e-5

    def get_wd(self) -> float:
        return 0.1

    def step(self, increment: int) -> None:
        self.param_sync_samples += increment
        self.state["steps"] += 1


class _SingleForwardRerunState:
    def __init__(self) -> None:
        self.calls = 0

    def should_run_forward_backward(self, data_iterator: object) -> bool:
        del data_iterator
        self.calls += 1
        return self.calls == 1


class _Runtime:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def forward_backward(self, *args: Any, **kwargs: Any) -> list[dict[str, _Tensor]]:
        del args
        model = kwargs["model"]
        forward_only = kwargs["forward_only"]
        hook_enabled = bool(model.remove_forward_pre_hook_handles)
        self.events.append(
            (
                "forward",
                forward_only,
                hook_enabled,
                model.config.param_sync_func,
            )
        )
        model.finish_forward_pre_hook()
        if forward_only:
            model.extra_state.add_(5.0)
        else:
            model.grad_buffer.fill_(model.weight.item() + 0.25)
        return [{"loss": model.weight.clone()}]


_EXTRACTED_METHODS = {
    "_collect_mtp_metrics",
    "_compute_moe_grad_scale",
    "_copy_main_params_to_param_buffer",
    "_disable_forward_pre_hook_until_next_train_step",
    "_forward_pre_hook_enabled",
    "_get_model_config",
    "_get_model_extra_state_dict",
    "_restore_model_extra_state_dict",
    "_set_moe_grad_scale_func",
    "_set_mtp_grad_scale_func",
    "_uses_mxfp8_overlap_shared_param_buffer",
    "disable_forward_pre_hook",
    "enable_forward_pre_hook",
    "train",
}


def _strip_function_metadata(method: ast.FunctionDef) -> ast.FunctionDef:
    method.decorator_list = []
    for function in (
        node for node in ast.walk(method) if isinstance(node, ast.FunctionDef)
    ):
        function.returns = None
        for argument in (
            function.args.posonlyargs + function.args.args + function.args.kwonlyargs
        ):
            argument.annotation = None
    return method


def _extract_worker_class(runtime: _Runtime):
    source_path = (
        Path(__file__).parents[4]
        / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
    )
    tree = ast.parse(source_path.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = [
        _strip_function_metadata(copy.deepcopy(node))
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in _EXTRACTED_METHODS
    ]
    assert {method.name for method in methods} == _EXTRACTED_METHODS

    class_kwargs: dict[str, Any] = {
        "name": "_ExtractedWorker",
        "bases": [],
        "keywords": [],
        "body": methods,
        "decorator_list": [],
    }
    if "type_params" in ast.ClassDef._fields:
        class_kwargs["type_params"] = []
    test_module = ast.Module(
        body=[ast.ClassDef(**class_kwargs)],
        type_ignores=[],
    )
    ast.fix_missing_locations(test_module)

    _FakeTorch.distributed.runtime = runtime
    namespace = {
        "ChainedOptimizer": _ChainedOptimizer,
        "DistributedDataParallel": _FaithfulDDP,
        "LossPostProcessor": lambda **kwargs: object(),
        "aggregate_training_statistics": (
            lambda *, all_mb_metrics, losses, data_parallel_group: (
                {"loss": [_Tensor(1.0)]},
                _Tensor(sum(loss.item() for loss in losses)),
            )
        ),
        "broadcast_loss_metrics_from_last_stage": lambda metrics: metrics,
        "copy": copy,
        "get_microbatch_iterator": (
            lambda *args, **kwargs: (iter([object()]), 1, 1, 1, 1)
        ),
        "get_model_config": lambda model: model.config,
        "get_moe_metrics": lambda **kwargs: {},
        "get_pg_collection": lambda model: SimpleNamespace(mp=object()),
        "get_rerun_state_machine": _SingleForwardRerunState,
        "logical_and_across_model_parallel_group": lambda value, **kwargs: value,
        "maybe_r3_trace_stage": lambda *args, **kwargs: nullcontext(),
        "megatron_forward_backward": runtime.forward_backward,
        "nullcontext": nullcontext,
        "parallel_state": SimpleNamespace(
            get_data_parallel_group=lambda **kwargs: None,
            is_pipeline_last_stage=lambda **kwargs: True,
        ),
        "process_global_batch": lambda *args, **kwargs: {
            "batch": {
                "sample_mask": _Tensor(1.0),
                "token_mask": _Tensor(1.0),
            },
            "global_valid_seqs": _Tensor(1.0),
            "global_valid_toks": _Tensor(1.0),
        },
        "reduce_max_stat_across_model_parallel_group": (lambda value, **kwargs: value),
        "should_reduce_loss_across_context_parallel": lambda cfg, batch: False,
        "strip_context_parallel_local_loss_metric": (lambda metrics, **kwargs: metrics),
        "time": time,
        "torch": _FakeTorch,
        "_should_use_router_replay": lambda **kwargs: False,
    }
    exec(compile(test_module, str(source_path), "exec"), namespace)
    return namespace["_ExtractedWorker"]


def _build_worker(*, shared_param_grad_buffer: bool):
    runtime = _Runtime()
    worker_class = _extract_worker_class(runtime)
    model = _FaithfulDDP(
        shared_param_grad_buffer=shared_param_grad_buffer,
        runtime=runtime,
    )
    optimizer = _FaithfulOptimizer(
        model,
        shared_param_grad_buffer=shared_param_grad_buffer,
        runtime=runtime,
    )
    scheduler = _FaithfulScheduler()
    worker = worker_class()
    worker.cfg = {
        "train_global_batch_size": 1,
        "train_micro_batch_size": 1,
        "megatron_cfg": {
            "empty_unused_memory_level": 0,
            "eval_mode_fast_path": True,
            "moe_per_layer_logging": False,
            "use_fused_linear_logprobs": False,
        },
    }
    worker.defer_fp32_logits = False
    worker.delegate_pack_to_model = False
    worker.dp_size = 1
    worker.draft_model = None
    worker.dtype = "float32"
    worker.fp8_cfg = {"enabled": True}
    worker.mcore_state = SimpleNamespace(straggler_timer=None)
    worker.megatron_cfg = SimpleNamespace(
        ddp=SimpleNamespace(overlap_param_gather=True),
        optimizer=SimpleNamespace(
            reuse_grad_buf_for_mxfp8_param_ag=shared_param_grad_buffer
        ),
    )
    worker.model = model
    worker.optimizer = optimizer
    worker.sampling_params = None
    worker.scheduler = scheduler
    worker.should_disable_forward_pre_hook = True
    worker._first_train_step_forward_pre_hook_disabled = False
    worker._first_train_step_param_sync_func = None
    worker._router_replay_enabled = False
    return worker, model, optimizer, scheduler, runtime


def _snapshot(worker, model, optimizer, scheduler) -> dict[str, Any]:
    return {
        "first_train_step_forward_pre_hook_disabled": (
            worker._first_train_step_forward_pre_hook_disabled
        ),
        "first_train_step_param_sync_func": worker._first_train_step_param_sync_func,
        "forward_pre_hook_enabled": worker._forward_pre_hook_enabled(),
        "grad_buffer": model.grad_buffer.clone(),
        "model_extra_state": model.extra_state.clone(),
        "model_training": model.training,
        "model_weight": model.weight.clone(),
        "optimizer_main_weight": optimizer.main_weight.clone(),
        "optimizer_state": dict(optimizer.state),
        "param_buffer": model.param_buffer.clone(),
        "param_sync_func": model.config.param_sync_func,
        "scheduler_samples": scheduler.param_sync_samples,
        "scheduler_state": dict(scheduler.state),
    }


def _train(worker) -> None:
    worker.train(
        SimpleNamespace(size=1),
        object(),
        eval_mode=False,
        gbs=1,
        mbs=1,
    )


def _fast_eval(worker) -> None:
    worker.train(
        SimpleNamespace(size=1),
        object(),
        eval_mode=True,
        gbs=1,
        mbs=1,
    )


class EvalFastPathStateEquivalenceTest(unittest.TestCase):
    def _assert_state_equivalence(self, *, shared_param_grad_buffer: bool) -> None:
        control = _build_worker(shared_param_grad_buffer=shared_param_grad_buffer)
        control_worker, control_model, control_optimizer, control_scheduler, _ = control
        _train(control_worker)
        _train(control_worker)

        experiment = _build_worker(shared_param_grad_buffer=shared_param_grad_buffer)
        worker, model, optimizer, scheduler, runtime = experiment
        _train(worker)
        before_eval = _snapshot(worker, model, optimizer, scheduler)
        event_start = len(runtime.events)
        _fast_eval(worker)
        eval_events = runtime.events[event_start:]

        if shared_param_grad_buffer:
            self.assertIn(("copy_main_params_to_param_buffer",), eval_events)
            self.assertIn(("model_zero_grad_buffer",), eval_events)
            self.assertIn(("ddp_disable_forward_pre_hook", True), eval_events)
            self.assertIn(("param_sync", True), eval_events)
            self.assertFalse(worker._forward_pre_hook_enabled())
            self.assertTrue(worker._first_train_step_forward_pre_hook_disabled)
            self.assertIs(worker._first_train_step_param_sync_func, _param_sync_func)
            self.assertIsNone(model.config.param_sync_func)
            self.assertEqual(model.weight, optimizer.main_weight)
            self.assertIs(model.param_buffer, model.grad_buffer)
        else:
            self.assertNotIn(("copy_main_params_to_param_buffer",), eval_events)
            self.assertFalse(
                any(event[0] == "ddp_disable_forward_pre_hook" for event in eval_events)
            )
            self.assertNotIn(("param_sync", True), eval_events)
            self.assertEqual(
                _snapshot(worker, model, optimizer, scheduler),
                before_eval,
            )

        next_train_event_start = len(runtime.events)
        _train(worker)
        next_train_events = runtime.events[next_train_event_start:]
        next_train_forward = next(
            event
            for event in next_train_events
            if event[0] == "forward" and event[1] is False
        )
        if shared_param_grad_buffer:
            self.assertFalse(next_train_forward[2])
            self.assertIsNone(next_train_forward[3])
            self.assertNotIn(("copy_main_params_to_param_buffer",), next_train_events)
            self.assertIn(("ddp_enable_forward_pre_hook",), next_train_events)
        else:
            self.assertTrue(next_train_forward[2])
            self.assertIs(next_train_forward[3], _param_sync_func)

        self.assertTrue(worker._forward_pre_hook_enabled())
        self.assertFalse(worker._first_train_step_forward_pre_hook_disabled)
        self.assertIsNone(worker._first_train_step_param_sync_func)
        self.assertIs(model.config.param_sync_func, _param_sync_func)
        self.assertEqual(
            _snapshot(worker, model, optimizer, scheduler),
            _snapshot(
                control_worker,
                control_model,
                control_optimizer,
                control_scheduler,
            ),
        )

    def test_normal_buffers_match_train_train_control(self) -> None:
        self._assert_state_equivalence(shared_param_grad_buffer=False)

    def test_mxfp8_shared_buffer_matches_train_train_control(self) -> None:
        self._assert_state_equivalence(shared_param_grad_buffer=True)


if __name__ == "__main__":
    unittest.main()
