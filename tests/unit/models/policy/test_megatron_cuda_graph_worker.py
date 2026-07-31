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

from __future__ import annotations

import ast
import copy
import sys
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_rl.models.megatron.cuda_graph_lifecycle import (
    TECudaGraphLifecycle,
    TECudaGraphScheduleKey,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKER_PATH = _REPO_ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py"


class _Phase(Enum):
    IDLE = auto()
    SPLIT_OPEN_BEFORE_FIRST = auto()
    GRAPH_SCHEDULE_LIVE = auto()
    SPLIT_OPEN_AFTER_FIRST = auto()


def _extract_worker_methods(
    method_names: set[str], namespace: dict[str, Any] | None = None
) -> type:
    """Compile selected production methods without importing GPU dependencies."""
    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = {
        node.name: copy.deepcopy(node)
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    }
    missing = method_names - methods.keys()
    assert not missing, f"worker is missing required methods: {sorted(missing)}"
    for method in methods.values():
        method.decorator_list = []

    class_kwargs: dict[str, Any] = {
        "name": "_Worker",
        "bases": [],
        "keywords": [],
        "body": list(methods.values()),
        "decorator_list": [],
    }
    if "type_params" in ast.ClassDef._fields:
        class_kwargs["type_params"] = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            ast.ClassDef(**class_kwargs),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    globals_dict = {
        "Any": Any,
        "Iterator": Any,
        "ProcessedMicrobatch": Any,
        "TECudaGraphScheduleKey": TECudaGraphScheduleKey,
        "_TECudaGraphWorkerPhase": _Phase,
    }
    if namespace is not None:
        globals_dict.update(namespace)
    exec(compile(module, str(_WORKER_PATH), "exec"), globals_dict)
    return globals_dict["_Worker"]


def test_runtime_schedule_provider_and_explicit_drained_phases() -> None:
    worker_type = _extract_worker_methods(
        {
            "_te_cuda_graph_runtime_num_microbatches",
            "_assert_te_cuda_graph_model_drained",
        }
    )
    worker = worker_type()
    worker._te_cuda_graph_runtime_schedule_count = 7

    worker._te_cuda_graph_phase = _Phase.IDLE
    assert worker._te_cuda_graph_runtime_num_microbatches() == 7
    assert worker._assert_te_cuda_graph_model_drained()

    worker._te_cuda_graph_phase = _Phase.SPLIT_OPEN_BEFORE_FIRST
    worker._train_step_state = {
        "te_cuda_graph_key": None,
        "total_num_microbatches": 0,
    }
    assert worker._assert_te_cuda_graph_model_drained()

    worker._te_cuda_graph_phase = _Phase.GRAPH_SCHEDULE_LIVE
    assert not worker._assert_te_cuda_graph_model_drained()

    worker._te_cuda_graph_phase = _Phase.SPLIT_OPEN_AFTER_FIRST
    assert not worker._assert_te_cuda_graph_model_drained()


def test_initialization_uses_topology_only_helper_and_direct_fixed_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    lifecycle_calls: list[dict[str, Any]] = []

    class FakeManager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            manager_calls.append((args, kwargs))

        @classmethod
        def from_helper(cls, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("from_helper must not use the global calculator")

    class FakeLifecycle:
        def __init__(self, **kwargs: Any) -> None:
            lifecycle_calls.append(kwargs)

    fake_module = SimpleNamespace(TECudaGraphBankManager=FakeManager)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.te_cuda_graph_bank",
        fake_module,
    )
    worker_type = _extract_worker_methods(
        {
            "_initialize_te_cuda_graph_lifecycle",
            "_te_cuda_graph_runtime_num_microbatches",
        },
        {
            "TECudaGraphLifecycle": FakeLifecycle,
            "_TE_CUDA_GRAPH_CACHE_CAPACITY": 2,
            "_TE_CUDA_GRAPH_WARMUP_STEPS": 3,
            "log": SimpleNamespace(info=lambda *_args, **_kwargs: None),
        },
    )
    worker = worker_type()
    topology_layers = (object(), object())
    topology_helper = SimpleNamespace(flattened_callables=topology_layers)
    sample_args: list[Any] = []
    worker._build_te_cuda_graph_helper = lambda sample: (
        sample_args.append(sample) or topology_helper
    )
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(cuda_graph_modules=("attention",))
    )
    worker._te_cuda_graph_runtime_schedule_count = 1
    worker._assert_te_cuda_graph_model_drained = lambda: True

    worker._initialize_te_cuda_graph_lifecycle()

    assert sample_args == [None]
    assert lifecycle_calls == [{"capacity": 2, "warmup_steps": 3}]
    assert len(manager_calls) == 1
    args, kwargs = manager_calls[0]
    assert args == (topology_layers,)
    assert kwargs["cuda_graph_modules"] == ("attention",)
    assert kwargs["assert_model_drained"]() is True
    assert kwargs["runtime_num_microbatches"]() == 1
    assert worker._te_cuda_graph_capture_helper is None
    assert worker._te_cuda_graph_capture_sample_packed_seq_params is None


def test_effective_graph_config_rpc_uses_validated_model_config() -> None:
    worker_type = _extract_worker_methods({"get_effective_te_cuda_graph_config"})
    worker = worker_type()
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(
            cuda_graph_impl="transformer_engine",
            thd_max_packed_sequences=65,
        )
    )
    worker._te_cuda_graph_lifecycle = object()

    assert worker.get_effective_te_cuda_graph_config() == {
        "cuda_graph_impl": "transformer_engine",
        "thd_max_packed_sequences": 65,
        "training_enabled": True,
    }


def test_peek_preserves_actual_microbatches_and_counts_geometry_on_yield() -> None:
    worker_type = _extract_worker_methods(
        {
            "_peek_te_cuda_graph_training_iterator",
            "_record_te_cuda_graph_geometry",
        }
    )
    worker = worker_type()
    packed_params = object()
    first = SimpleNamespace(
        packed_seq_params=packed_params,
        packed_geometry=SimpleNamespace(
            logical_tokens=11,
            padded_tokens=16,
            capacity_tokens=32,
            real_sequence_count=3,
            cu_seqlens_capacity_entries=9,
        ),
    )
    second = SimpleNamespace(
        packed_seq_params=object(),
        packed_geometry=SimpleNamespace(
            logical_tokens=7,
            padded_tokens=8,
            capacity_tokens=32,
            real_sequence_count=2,
            cu_seqlens_capacity_entries=9,
        ),
    )
    call_state = SimpleNamespace(
        logical_tokens=0,
        padded_tokens=0,
        capacity_tokens=0,
    )

    actual_first, preserved = worker._peek_te_cuda_graph_training_iterator(
        iter((first, second)), call_state
    )

    assert actual_first is first
    assert actual_first.packed_seq_params is packed_params
    assert call_state.logical_tokens == 0
    assert list(preserved) == [first, second]
    assert (
        call_state.logical_tokens,
        call_state.padded_tokens,
        call_state.capacity_tokens,
    ) == (18, 24, 64)


def test_first_microbatch_preflight_failure_is_raised_collectively() -> None:
    worker_type = _extract_worker_methods(
        {"_collectively_peek_te_cuda_graph_training_iterator"}
    )
    worker = worker_type()
    local_error = ValueError("bad local THD geometry")
    worker._peek_te_cuda_graph_training_iterator = lambda *_args: (_ for _ in ()).throw(
        local_error
    )
    collective_calls: list[tuple[Any, str]] = []

    def raise_collectively(error: Any, *, operation: str) -> None:
        collective_calls.append((error, operation))
        raise RuntimeError("collective preflight failure") from error

    worker._collectively_raise_te_cuda_graph_failure = raise_collectively

    with pytest.raises(RuntimeError, match="collective preflight failure"):
        worker._collectively_peek_te_cuda_graph_training_iterator(iter(()), object())

    assert collective_calls == [(local_error, "microbatch preflight")]


def test_schedule_uses_exact_sample_and_two_entry_lru() -> None:
    worker_type = _extract_worker_methods({"_ensure_te_cuda_graph_schedule"})
    worker = worker_type()
    worker._te_cuda_graph_lifecycle = TECudaGraphLifecycle(
        capacity=2,
        warmup_steps=0,
    )
    worker._te_cuda_graph_phase = _Phase.IDLE
    worker._te_cuda_graph_runtime_schedule_count = 1
    worker._te_cuda_graph_installed_key = None
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(overlap_moe_expert_parallel_comm=False)
    )
    worker._te_cuda_graph_pipeline_parallel_size = lambda: 2
    worker._assert_te_cuda_graph_model_drained = lambda: True
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name, group=None: value
    )
    worker._collectively_raise_te_cuda_graph_failure = lambda error, *, operation: None
    manager = SimpleNamespace(active_bank=None, uninstall=lambda: None)
    worker._te_cuda_graph_bank_manager = manager
    captures: list[tuple[TECudaGraphScheduleKey, Any]] = []

    class Bank:
        def __init__(self, key: TECudaGraphScheduleKey) -> None:
            self.key = key
            self.reset_count = 0

        def activate(self) -> None:
            manager.active_bank = self

        def reset(self) -> None:
            self.reset_count += 1
            if manager.active_bank is self:
                manager.active_bank = None

    def capture(key: TECudaGraphScheduleKey, sample: Any) -> Bank:
        captures.append((key, sample))
        return Bank(key)

    worker._capture_te_cuda_graph_bank = capture
    worker._install_te_cuda_graph_manual_hooks = lambda: None
    samples = [SimpleNamespace(packed_seq_params=object()) for _ in range(4)]
    call_states = []

    for count, first in zip((5, 3, 5, 7), samples):
        call_state = SimpleNamespace(
            capture_count=0,
            replay_count=0,
            cache_hit_count=0,
            eviction_count=0,
            normalized_schedule_key=None,
        )
        call_states.append(call_state)
        worker._ensure_te_cuda_graph_schedule(
            num_microbatches=count,
            first_microbatch=first,
            call_state=call_state,
            ensure_active=True,
        )

    assert [(key.num_microbatches, sample) for key, sample in captures] == [
        (5, samples[0].packed_seq_params),
        (3, samples[1].packed_seq_params),
        (7, samples[3].packed_seq_params),
    ]
    assert sum(state.capture_count for state in call_states) == 3
    assert sum(state.cache_hit_count for state in call_states) == 1
    assert sum(state.eviction_count for state in call_states) == 1
    assert [state.normalized_schedule_key for state in call_states] == [5, 3, 5, 7]
    assert worker._te_cuda_graph_runtime_schedule_count == 7
    assert worker._te_cuda_graph_installed_key == TECudaGraphScheduleKey(7)


def test_exactly_three_successful_updates_precede_first_capture() -> None:
    worker_type = _extract_worker_methods(
        {
            "_ensure_te_cuda_graph_schedule",
            "_record_te_cuda_graph_optimizer_step",
        }
    )
    worker = worker_type()
    worker._te_cuda_graph_lifecycle = TECudaGraphLifecycle(
        capacity=2,
        warmup_steps=3,
    )
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(overlap_moe_expert_parallel_comm=False)
    )
    worker._te_cuda_graph_pipeline_parallel_size = lambda: 1
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name, group=None: value
    )
    worker._collectively_raise_te_cuda_graph_failure = lambda error, *, operation: None
    worker._assert_te_cuda_graph_model_drained = lambda: True
    worker._global_te_cuda_graph_optimizer_success = lambda successful: successful
    manager = SimpleNamespace(active_bank=None, uninstall=lambda: None)
    worker._te_cuda_graph_bank_manager = manager
    worker._te_cuda_graph_installed_key = None
    worker._te_cuda_graph_runtime_schedule_count = 1
    worker._install_te_cuda_graph_manual_hooks = lambda: None
    captures: list[Any] = []

    class Bank:
        def activate(self) -> None:
            manager.active_bank = self

        def reset(self) -> None:
            manager.active_bank = None

    worker._capture_te_cuda_graph_bank = lambda key, sample: (
        captures.append(sample) or Bank()
    )
    first = SimpleNamespace(packed_seq_params=object())

    def ensure_once() -> SimpleNamespace:
        state = SimpleNamespace(
            capture_count=0,
            replay_count=0,
            cache_hit_count=0,
            eviction_count=0,
            normalized_schedule_key=None,
        )
        worker._ensure_te_cuda_graph_schedule(
            num_microbatches=9,
            first_microbatch=first,
            call_state=state,
            ensure_active=True,
        )
        return state

    assert ensure_once().capture_count == 0
    for _ in range(2):
        worker._record_te_cuda_graph_optimizer_step(True)
        assert ensure_once().capture_count == 0
    worker._record_te_cuda_graph_optimizer_step(False)
    assert ensure_once().capture_count == 0
    worker._record_te_cuda_graph_optimizer_step(True)
    assert ensure_once().capture_count == 1
    assert captures == [first.packed_seq_params]


def test_split_schedule_pins_first_key_without_second_transition() -> None:
    worker_type = _extract_worker_methods({"_ensure_te_cuda_graph_schedule"})
    worker = worker_type()
    transitions: list[int] = []

    class Lifecycle:
        def ensure_active(self, key: TECudaGraphScheduleKey, capture: Any) -> Any:
            transitions.append(key.num_microbatches)
            bank = capture()
            bank.activate()
            return SimpleNamespace(status="captured", evicted_key=None)

    manager = SimpleNamespace(active_bank=None)

    class Bank:
        def activate(self) -> None:
            manager.active_bank = self

        def reset(self) -> None:
            manager.active_bank = None

    worker._te_cuda_graph_lifecycle = Lifecycle()
    worker._te_cuda_graph_bank_manager = manager
    worker._te_cuda_graph_installed_key = None
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(overlap_moe_expert_parallel_comm=False)
    )
    worker._te_cuda_graph_pipeline_parallel_size = lambda: 2
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name, group=None: value
    )
    worker._collectively_raise_te_cuda_graph_failure = lambda error, *, operation: None
    worker._assert_te_cuda_graph_model_drained = lambda: True
    worker._capture_te_cuda_graph_bank = lambda key, sample: Bank()
    worker._install_te_cuda_graph_manual_hooks = lambda: None
    first = SimpleNamespace(packed_seq_params=object())
    call_state = SimpleNamespace(
        capture_count=0,
        replay_count=0,
        cache_hit_count=0,
        eviction_count=0,
        normalized_schedule_key=None,
    )

    key = worker._ensure_te_cuda_graph_schedule(
        num_microbatches=5,
        first_microbatch=first,
        call_state=call_state,
        ensure_active=True,
    )
    assert key == TECudaGraphScheduleKey(5)
    assert (
        worker._ensure_te_cuda_graph_schedule(
            num_microbatches=5,
            first_microbatch=first,
            call_state=call_state,
            ensure_active=False,
        )
        == key
    )
    assert transitions == [5]
    with pytest.raises(RuntimeError, match="pinned"):
        worker._ensure_te_cuda_graph_schedule(
            num_microbatches=3,
            first_microbatch=first,
            call_state=call_state,
            ensure_active=False,
        )


def test_capture_helper_receives_exact_first_metadata_identity() -> None:
    worker_type = _extract_worker_methods({"_capture_te_cuda_graph_bank"})
    worker = worker_type()
    sample = object()
    manager = object()
    bank = SimpleNamespace(reset=lambda: None)
    helper = SimpleNamespace(
        create_cuda_graph_bank=lambda actual_manager, *, num_microbatches: (
            bank
            if actual_manager is manager and num_microbatches == 4
            else (_ for _ in ()).throw(AssertionError("capture inputs changed"))
        )
    )
    samples: list[Any] = []
    worker._build_te_cuda_graph_helper = lambda actual_sample: (
        samples.append(actual_sample) or helper
    )
    worker._te_cuda_graph_bank_manager = manager
    worker.should_disable_forward_pre_hook = False
    worker._model_parallel_te_cuda_graphs_created = lambda actual_helper: (
        actual_helper is helper
    )

    assert worker._capture_te_cuda_graph_bank(TECudaGraphScheduleKey(4), sample) is bank
    assert samples == [sample]
    assert worker._te_cuda_graph_capture_sample_packed_seq_params is sample


def test_capture_releases_registered_bank_when_hook_restore_fails() -> None:
    worker_type = _extract_worker_methods({"_capture_te_cuda_graph_bank"})
    worker = worker_type()
    events: list[str] = []
    bank = SimpleNamespace(reset=lambda: events.append("bank.reset"))
    helper = SimpleNamespace(
        create_cuda_graph_bank=lambda manager, *, num_microbatches: (
            events.append("capture") or bank
        )
    )
    worker._build_te_cuda_graph_helper = lambda sample: helper
    worker._te_cuda_graph_bank_manager = object()
    worker.should_disable_forward_pre_hook = True
    worker._forward_pre_hook_enabled = lambda: True
    worker.disable_forward_pre_hook = lambda *, param_sync: events.append("disable")

    def fail_enable() -> None:
        events.append("enable")
        raise RuntimeError("hook restore failed")

    worker.enable_forward_pre_hook = fail_enable

    with pytest.raises(RuntimeError, match="hook restore failed"):
        worker._capture_te_cuda_graph_bank(TECudaGraphScheduleKey(2), object())

    assert events == ["disable", "capture", "enable", "bank.reset"]


def test_iterator_modes_eager_isolation_and_relocation_order_are_explicit() -> None:
    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = {
        node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)
    }

    def iterator_flags(method_name: str) -> list[str]:
        return [
            ast.unparse(keyword.value)
            for call in ast.walk(methods[method_name])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "get_microbatch_iterator"
            for keyword in call.keywords
            if keyword.arg == "for_cuda_graph_training"
        ]

    assert iterator_flags("train") == ["te_cuda_graph_call_state is not None"]
    assert iterator_flags("_train_microbatch_body") == [
        "state['te_cuda_graph_call_state'] is not None"
    ]
    assert iterator_flags("get_logprobs") == ["False"]
    assert iterator_flags("get_topk_logits") == ["False"]

    for method_name in ("get_logprobs", "use_reference_model", "get_topk_logits"):
        assert any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_deactivate_te_cuda_graphs_for_eager_path"
            for call in ast.walk(methods[method_name])
        )

    reference_calls = [
        statement.value.func.attr
        for statement in methods["use_reference_model"].body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
    ]
    assert reference_calls[:2] == [
        "_deactivate_te_cuda_graphs_for_eager_path",
        "_reset_te_cuda_graph_banks_for_storage_relocation",
    ]

    for method_name in ("move_model", "move_optimizer", "load_checkpoint"):
        first_statement = next(
            statement
            for statement in methods[method_name].body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        )
        assert isinstance(first_statement, ast.Expr)
        assert isinstance(first_statement.value, ast.Call)
        assert isinstance(first_statement.value.func, ast.Attribute)
        assert (
            first_statement.value.func.attr
            == "_reset_te_cuda_graph_banks_for_storage_relocation"
        )


def test_metrics_validate_and_reduce_across_tp_cp_then_pp_for_cp_replicas() -> None:
    class FakeTensor:
        def __init__(self, value: Any) -> None:
            self.values = list(value) if isinstance(value, list) else [value]

        def clone(self) -> "FakeTensor":
            return FakeTensor(list(self.values))

        def item(self) -> Any:
            assert len(self.values) == 1
            return self.values[0]

        def zero_(self) -> "FakeTensor":
            self.values = [0 for _ in self.values]
            return self

        def __getitem__(self, index: int) -> "FakeTensor":
            return FakeTensor(self.values[index])

    reduce_op = SimpleNamespace(MIN="min", MAX="max", SUM="sum")

    collective_calls: list[tuple[Any, Any]] = []

    def all_reduce(tensor: FakeTensor, *, op: Any, group: Any = None) -> None:
        collective_calls.append((op, group))
        assert group in {"tp_cp", "pp"}
        if op != reduce_op.SUM:
            return
        if len(tensor.values) == 2:
            tensor.values = [5, 8] if group == "tp_cp" else [13, 21]
        elif len(tensor.values) == 3 and group == "pp":
            tensor.values = [23, 27, 32]

    fake_torch = SimpleNamespace(
        int64="int64",
        tensor=lambda value, *, dtype, device: FakeTensor(value),
        distributed=SimpleNamespace(ReduceOp=reduce_op, all_reduce=all_reduce),
    )
    fake_parallel_state = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        get_context_parallel_rank=lambda: 1,
        get_pipeline_model_parallel_rank=lambda: 1,
    )
    worker_type = _extract_worker_methods(
        {
            "_collectively_validate_te_cuda_graph_integer",
            "_finalize_te_cuda_graph_call",
            "_te_cuda_graph_token_capacity_per_microbatch",
        },
        {
            "torch": fake_torch,
            "parallel_state": fake_parallel_state,
            "get_pg_collection": lambda model: SimpleNamespace(
                tp_cp="tp_cp", pp="pp", mp="mp"
            ),
            "CudaGraphStepMetrics": __import__(
                "nemo_rl.models.megatron.cuda_graph_lifecycle",
                fromlist=["CudaGraphStepMetrics"],
            ).CudaGraphStepMetrics,
            "asdict": __import__("dataclasses").asdict,
            "cast": lambda _type, value: value,
        },
    )
    worker = worker_type()
    worker.model = object()
    worker.cfg = {"sequence_packing": {"train_mb_tokens": 32}}
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(thd_max_packed_sequences=9)
    )
    worker._te_cuda_graph_device = lambda: "cpu"
    worker._te_cuda_graph_bank_manager = SimpleNamespace(
        execution_counter_delta=lambda start: SimpleNamespace(
            graph_calls=5,
            eligible_calls=8,
        )
    )
    call_state = SimpleNamespace(
        execution_snapshot=object(),
        capture_count=1,
        replay_count=4,
        cache_hit_count=2,
        eviction_count=1,
        logical_tokens=23,
        padded_tokens=27,
        capacity_tokens=32,
        normalized_schedule_key=5,
    )

    metrics, contract = worker._finalize_te_cuda_graph_call(call_state)

    assert metrics == {
        "capture_count": 1,
        "replay_count": 4,
        "cache_hit_count": 2,
        "eviction_count": 1,
        "fallback_count": 0,
        "graph_calls": 13,
        "eligible_calls": 21,
        "logical_tokens": 23,
        "padded_tokens": 27,
        "capacity_tokens": 32,
    }
    assert contract == {
        "normalized_schedule_key": 5,
        "token_capacity_per_microbatch": 32,
        "thd_max_packed_sequences": 9,
    }
    validation_groups = [
        group for op, group in collective_calls if op in {reduce_op.MIN, reduce_op.MAX}
    ]
    sum_groups = [group for op, group in collective_calls if op == reduce_op.SUM]
    assert validation_groups == ["tp_cp", "tp_cp", "pp", "pp"] * 11
    assert sum_groups == ["tp_cp", "pp", "tp_cp", "pp"]
    assert all(group != "mp" for _, group in collective_calls)


def test_metric_mismatch_reaches_pp_collectives_before_raise() -> None:
    class Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def clone(self) -> "Scalar":
            return Scalar(self.value)

        def item(self) -> int:
            return self.value

    reduce_op = SimpleNamespace(MIN="min", MAX="max", SUM="sum")
    collective_calls: list[tuple[Any, Any]] = []

    def all_reduce(tensor: Scalar, *, op: Any, group: Any = None) -> None:
        collective_calls.append((op, group))
        if group == "tp_cp" and op == reduce_op.MIN:
            tensor.value = 1
        elif group == "tp_cp" and op == reduce_op.MAX:
            tensor.value = 2

    fake_torch = SimpleNamespace(
        int64="int64",
        tensor=lambda value, *, dtype, device: Scalar(value),
        distributed=SimpleNamespace(ReduceOp=reduce_op, all_reduce=all_reduce),
    )
    worker_type = _extract_worker_methods(
        {
            "_collectively_validate_te_cuda_graph_integer",
            "_finalize_te_cuda_graph_call",
        },
        {
            "torch": fake_torch,
            "get_pg_collection": lambda model: SimpleNamespace(tp_cp="tp_cp", pp="pp"),
        },
    )
    worker = worker_type()
    worker.model = object()
    worker._te_cuda_graph_device = lambda: "cpu"
    worker._te_cuda_graph_bank_manager = object()
    call_state = SimpleNamespace(
        capture_count=1,
        replay_count=0,
        cache_hit_count=0,
        eviction_count=0,
        normalized_schedule_key=1,
    )

    with pytest.raises(RuntimeError, match="capture_count differs across ranks"):
        worker._finalize_te_cuda_graph_call(call_state)

    assert collective_calls == [
        (reduce_op.MIN, "tp_cp"),
        (reduce_op.MAX, "tp_cp"),
        (reduce_op.MIN, "pp"),
        (reduce_op.MAX, "pp"),
    ]


def test_global_optimizer_consensus_uses_world_min() -> None:
    class Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    calls: list[tuple[Any, Any]] = []
    reduce_op = SimpleNamespace(MIN=object())

    def all_reduce(tensor: Scalar, *, op: Any) -> None:
        calls.append((op, None))
        tensor.value = 0

    fake_torch = SimpleNamespace(
        int32="int32",
        tensor=lambda value, *, dtype, device: Scalar(value),
        distributed=SimpleNamespace(ReduceOp=reduce_op, all_reduce=all_reduce),
    )
    worker_type = _extract_worker_methods(
        {"_global_te_cuda_graph_optimizer_success"},
        {"torch": fake_torch},
    )
    worker = worker_type()
    worker._te_cuda_graph_device = lambda: "cuda"

    assert not worker._global_te_cuda_graph_optimizer_success(True)
    assert calls == [(reduce_op.MIN, None)]


def test_remote_graph_failure_raises_and_never_falls_back() -> None:
    class Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    reduce_op = SimpleNamespace(MAX=object())

    def all_reduce(tensor: Scalar, *, op: Any) -> None:
        assert op is reduce_op.MAX
        tensor.value = 1

    fake_torch = SimpleNamespace(
        int32="int32",
        tensor=lambda value, *, dtype, device: Scalar(value),
        distributed=SimpleNamespace(ReduceOp=reduce_op, all_reduce=all_reduce),
    )
    worker_type = _extract_worker_methods(
        {"_collectively_raise_te_cuda_graph_failure"},
        {"torch": fake_torch},
    )
    worker = worker_type()
    worker._te_cuda_graph_device = lambda: "cuda"
    cleanup: list[str] = []
    worker._reset_te_cuda_graph_banks_after_failure = lambda: cleanup.append("reset")

    with pytest.raises(RuntimeError, match="another rank"):
        worker._collectively_raise_te_cuda_graph_failure(
            None,
            operation="replay",
        )
    assert cleanup == ["reset"]


def test_split_abort_restores_idle_and_runs_deferred_failure_cleanup() -> None:
    worker_type = _extract_worker_methods({"abort_train_step"})
    worker = worker_type()
    events: list[str] = []
    state = {
        "te_cuda_graph_key": object(),
        "te_cuda_graph_call_state": object(),
    }
    worker._train_step_state = state
    worker._te_cuda_graph_phase = _Phase.SPLIT_OPEN_AFTER_FIRST
    worker._te_cuda_graph_reset_required = True
    worker._restore_saved_grad_sync_func = lambda actual: events.append("restore")
    worker.model = SimpleNamespace(zero_grad_buffer=lambda: events.append("model.zero"))
    worker.optimizer = SimpleNamespace(zero_grad=lambda: events.append("opt.zero"))
    worker._reset_te_cuda_graph_banks_after_failure = lambda: events.append("reset")

    worker.abort_train_step()

    assert events == ["restore", "model.zero", "opt.zero", "reset"]
    assert worker._train_step_state is None
    assert worker._te_cuda_graph_phase is _Phase.IDLE
    assert state["te_cuda_graph_key"] is None
    assert state["te_cuda_graph_call_state"] is None


def test_empty_graph_split_finish_fails_before_optimizer_or_scheduler_mutation() -> (
    None
):
    worker_type = _extract_worker_methods({"_validate_te_cuda_graph_split_finish"})
    worker = worker_type()
    call_state = SimpleNamespace(
        normalized_schedule_key=None,
        capacity_tokens=0,
    )
    state = {
        "te_cuda_graph_key": None,
        "te_cuda_graph_call_state": call_state,
        "total_num_microbatches": 0,
    }
    collective_calls: list[tuple[Any, str]] = []

    def raise_collectively(error: Any, *, operation: str) -> None:
        collective_calls.append((error, operation))
        if error is not None:
            raise error

    worker._collectively_raise_te_cuda_graph_failure = raise_collectively

    with pytest.raises(RuntimeError, match="no processed microbatch"):
        worker._validate_te_cuda_graph_split_finish(state)
    assert len(collective_calls) == 1
    assert collective_calls[0][1] == "split finish preflight"

    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    finish = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_finish_train_step_body"
    )
    calls = [node for node in ast.walk(finish) if isinstance(node, ast.Call)]
    guard_line = min(
        call.lineno
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "_validate_te_cuda_graph_split_finish"
    )
    mutation_lines = [
        call.lineno
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr in {"scale_gradients", "step"}
    ]
    assert mutation_lines
    assert guard_line < min(mutation_lines)


def test_empty_graph_split_can_abort_without_finish_validation() -> None:
    worker_type = _extract_worker_methods({"abort_train_step"})
    worker = worker_type()
    events: list[str] = []
    state = {
        "te_cuda_graph_key": None,
        "te_cuda_graph_call_state": object(),
    }
    worker._train_step_state = state
    worker._te_cuda_graph_phase = _Phase.SPLIT_OPEN_BEFORE_FIRST
    worker._te_cuda_graph_reset_required = False
    worker._restore_saved_grad_sync_func = lambda actual: events.append("restore")
    worker.model = SimpleNamespace(zero_grad_buffer=lambda: events.append("model.zero"))
    worker.optimizer = SimpleNamespace(zero_grad=lambda: events.append("opt.zero"))

    worker.abort_train_step()

    assert events == ["restore", "model.zero", "opt.zero"]
    assert worker._train_step_state is None
    assert worker._te_cuda_graph_phase is _Phase.IDLE


@pytest.mark.parametrize(
    ("successful", "global_success", "expected"),
    [(True, True, [True]), (True, False, [False]), (False, False, [False])],
)
def test_warmup_records_only_global_optimizer_consensus(
    successful: bool, global_success: bool, expected: list[bool]
) -> None:
    worker_type = _extract_worker_methods({"_record_te_cuda_graph_optimizer_step"})
    worker = worker_type()
    recorded: list[bool] = []
    worker._te_cuda_graph_lifecycle = SimpleNamespace(
        record_optimizer_step=lambda *, successful: recorded.append(successful)
    )
    worker._global_te_cuda_graph_optimizer_success = lambda _: global_success

    worker._record_te_cuda_graph_optimizer_step(successful)

    assert recorded == expected


def test_storage_reset_preserves_lifecycle_and_manager_but_discards_capture_refs() -> (
    None
):
    worker_type = _extract_worker_methods(
        {"_reset_te_cuda_graph_banks_for_storage_relocation"}
    )
    worker = worker_type()
    events: list[str] = []
    lifecycle = SimpleNamespace(reset_banks=lambda: events.append("reset"))
    manager = object()
    worker._te_cuda_graph_phase = _Phase.IDLE
    worker._te_cuda_graph_lifecycle = lifecycle
    worker._te_cuda_graph_bank_manager = manager
    worker._te_cuda_graph_installed_key = object()
    worker._te_cuda_graph_capture_helper = object()
    worker._te_cuda_graph_capture_sample_packed_seq_params = object()

    worker._reset_te_cuda_graph_banks_for_storage_relocation()

    assert events == ["reset"]
    assert worker._te_cuda_graph_lifecycle is lifecycle
    assert worker._te_cuda_graph_bank_manager is manager
    assert worker._te_cuda_graph_installed_key is None
    assert worker._te_cuda_graph_capture_helper is None
    assert worker._te_cuda_graph_capture_sample_packed_seq_params is None


def test_eval_extra_state_restore_resets_graph_banks_before_storage_load() -> None:
    worker_type = _extract_worker_methods({"_restore_model_extra_state_dict"})
    worker = worker_type()
    events: list[str] = []
    worker._reset_te_cuda_graph_banks_for_storage_relocation = lambda: events.append(
        "reset"
    )
    worker.model = SimpleNamespace(
        load_state_dict=lambda state, *, strict: events.append(("load", state, strict))
    )
    extra_state = {"layer._extra_state": object()}

    worker._restore_model_extra_state_dict({})
    assert events == []

    worker._restore_model_extra_state_dict(extra_state)
    assert events == ["reset", ("load", extra_state, False)]


def test_eager_uninstall_requires_idle_and_clears_only_installed_bank() -> None:
    worker_type = _extract_worker_methods({"_deactivate_te_cuda_graphs_for_eager_path"})
    worker = worker_type()
    events: list[str] = []
    worker._te_cuda_graph_phase = _Phase.IDLE
    worker._te_cuda_graph_bank_manager = SimpleNamespace(
        active_bank=object(), uninstall=lambda: events.append("uninstall")
    )
    worker._te_cuda_graph_installed_key = object()

    worker._deactivate_te_cuda_graphs_for_eager_path()

    assert events == ["uninstall"]
    assert worker._te_cuda_graph_installed_key is None

    worker._te_cuda_graph_phase = _Phase.GRAPH_SCHEDULE_LIVE
    with pytest.raises(RuntimeError, match="drained"):
        worker._deactivate_te_cuda_graphs_for_eager_path()


def test_shutdown_aborts_closes_lifecycle_and_detaches_manager() -> None:
    events: list[str] = []

    class FakeBase:
        @staticmethod
        def shutdown(_worker: Any) -> bool:
            events.append("base")
            return True

    worker_type = _extract_worker_methods(
        {"shutdown"},
        {"AbstractPolicyWorker": FakeBase},
    )
    worker = worker_type()
    worker._train_step_state = {"open": True}
    worker.abort_train_step = lambda: (
        events.append("abort") or setattr(worker, "_train_step_state", None)
    )
    worker._te_cuda_graph_lifecycle = SimpleNamespace(
        close=lambda: events.append("lifecycle.close")
    )
    worker._te_cuda_graph_bank_manager = SimpleNamespace(
        close=lambda: events.append("manager.close")
    )
    worker._te_cuda_graph_installed_key = object()
    worker._te_cuda_graph_capture_helper = object()
    worker._te_cuda_graph_capture_sample_packed_seq_params = object()

    assert worker.shutdown()
    assert events == ["abort", "lifecycle.close", "manager.close", "base"]
    assert worker._te_cuda_graph_lifecycle is None
    assert worker._te_cuda_graph_bank_manager is None
    assert worker._te_cuda_graph_installed_key is None
    assert worker._te_cuda_graph_capture_helper is None
    assert worker._te_cuda_graph_capture_sample_packed_seq_params is None
