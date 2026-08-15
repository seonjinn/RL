from __future__ import annotations

import ast
import copy
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKER_PATH = _REPO_ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py"


def _extract_worker_methods(
    method_names: set[str], namespace: dict[str, Any] | None = None
) -> type:
    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = {
        node.name: copy.deepcopy(node)
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    }
    missing = method_names - methods.keys()
    assert not missing, f"worker is missing required methods: {sorted(missing)}"
    for method_name, method in methods.items():
        if method_name != "_hybridep_uneven_padding_for_eager_path":
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
    globals_dict = {"Any": Any, "contextmanager": contextmanager}
    if namespace is not None:
        globals_dict.update(namespace)
    exec(compile(module, str(_WORKER_PATH), "exec"), globals_dict)
    return globals_dict["_Worker"]


def _phase_padding_worker(*, modules: tuple[str, ...]) -> tuple[Any, Any, Any]:
    worker_type = _extract_worker_methods(
        {
            "_te_cuda_graph_uses_hybridep_preprocess_capture",
            "_hybridep_dispatcher_configs",
            "_hybridep_uneven_padding_for_eager_path",
            "_assert_hybridep_preprocess_capture_padding_disabled",
        }
    )
    provider_config = SimpleNamespace(
        cuda_graph_modules=modules,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_pad_uneven_dispatch_inputs=False,
    )
    runtime_config = SimpleNamespace(
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_pad_uneven_dispatch_inputs=False,
    )
    worker = worker_type()
    worker.megatron_cfg = SimpleNamespace(model=provider_config)
    worker._te_cuda_graph_lifecycle = object()
    worker.model = SimpleNamespace(
        modules=lambda: (SimpleNamespace(config=runtime_config),)
    )
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name, group=None: value
    )
    return worker, provider_config, runtime_config


def test_eager_phase_enables_uneven_padding_and_restores_capture_config() -> None:
    worker, provider_config, runtime_config = _phase_padding_worker(
        modules=("moe_router", "moe_preprocess")
    )

    with worker._hybridep_uneven_padding_for_eager_path():
        assert provider_config.moe_hybridep_pad_uneven_dispatch_inputs is True
        assert runtime_config.moe_hybridep_pad_uneven_dispatch_inputs is True

    assert provider_config.moe_hybridep_pad_uneven_dispatch_inputs is False
    assert runtime_config.moe_hybridep_pad_uneven_dispatch_inputs is False


def test_eager_phase_restores_capture_config_after_forward_failure() -> None:
    worker, provider_config, runtime_config = _phase_padding_worker(
        modules=("attn", "mamba", "moe_router", "moe_preprocess")
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        with worker._hybridep_uneven_padding_for_eager_path():
            raise RuntimeError("forward failed")

    assert provider_config.moe_hybridep_pad_uneven_dispatch_inputs is False
    assert runtime_config.moe_hybridep_pad_uneven_dispatch_inputs is False


def test_non_preprocess_scope_preserves_static_uneven_padding() -> None:
    worker, provider_config, runtime_config = _phase_padding_worker(
        modules=("attn", "mamba", "moe_router")
    )
    provider_config.moe_hybridep_pad_uneven_dispatch_inputs = True
    runtime_config.moe_hybridep_pad_uneven_dispatch_inputs = True

    with worker._hybridep_uneven_padding_for_eager_path():
        assert provider_config.moe_hybridep_pad_uneven_dispatch_inputs is True
        assert runtime_config.moe_hybridep_pad_uneven_dispatch_inputs is True

    assert provider_config.moe_hybridep_pad_uneven_dispatch_inputs is True
    assert runtime_config.moe_hybridep_pad_uneven_dispatch_inputs is True


def test_preprocess_capture_fails_closed_while_eager_padding_is_enabled() -> None:
    worker, provider_config, runtime_config = _phase_padding_worker(
        modules=("moe_router", "moe_preprocess")
    )
    runtime_config.moe_hybridep_pad_uneven_dispatch_inputs = True

    with pytest.raises(RuntimeError, match="must be disabled during capture"):
        worker._assert_hybridep_preprocess_capture_padding_disabled()

    runtime_config.moe_hybridep_pad_uneven_dispatch_inputs = False
    worker._assert_hybridep_preprocess_capture_padding_disabled()


def test_eager_phase_validates_initial_padding_state_collectively() -> None:
    worker, provider_config, runtime_config = _phase_padding_worker(
        modules=("moe_router", "moe_preprocess")
    )
    collective_calls: list[tuple[int, str]] = []

    def reject_divergence(value: int, *, name: str, group: Any = None) -> int:
        assert group is None
        collective_calls.append((value, name))
        raise RuntimeError("padding state differs across ranks")

    worker._collectively_validate_te_cuda_graph_integer = reject_divergence

    with pytest.raises(RuntimeError, match="differs across ranks"):
        with worker._hybridep_uneven_padding_for_eager_path():
            raise AssertionError("eager body must not run")

    assert collective_calls == [(0, "HybridEP eager padding precondition")]
    assert provider_config.moe_hybridep_pad_uneven_dispatch_inputs is False
    assert runtime_config.moe_hybridep_pad_uneven_dispatch_inputs is False


def test_capture_guard_validates_padding_state_collectively() -> None:
    worker, _provider_config, _runtime_config = _phase_padding_worker(
        modules=("moe_router", "moe_preprocess")
    )
    collective_calls: list[tuple[int, str]] = []

    def validate(value: int, *, name: str, group: Any = None) -> int:
        assert group is None
        collective_calls.append((value, name))
        return value

    worker._collectively_validate_te_cuda_graph_integer = validate

    worker._assert_hybridep_preprocess_capture_padding_disabled()

    assert collective_calls == [(0, "HybridEP capture padding state")]


def test_get_logprobs_enables_padding_during_the_eager_forward() -> None:
    events: list[tuple[str, bool, bool]] = []
    provider_config = SimpleNamespace(
        cuda_graph_modules=("moe_router", "moe_preprocess"),
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_pad_uneven_dispatch_inputs=False,
    )
    runtime_config = SimpleNamespace(
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_pad_uneven_dispatch_inputs=False,
    )

    class NoGrad:
        def __enter__(self) -> None:
            events.append(("no_grad_enter", False, False))

        def __exit__(self, *_args: Any) -> None:
            events.append(("no_grad_exit", False, False))

    class Batch:
        def __init__(self, **values: Any) -> None:
            self.values = values

        @classmethod
        def __class_getitem__(cls, _item: Any) -> type[Batch]:
            return cls

        def to(self, device: str) -> Batch:
            assert device == "cpu"
            return self

    def forward(**_kwargs: Any) -> list[Any]:
        events.append(
            (
                "forward",
                provider_config.moe_hybridep_pad_uneven_dispatch_inputs,
                runtime_config.moe_hybridep_pad_uneven_dispatch_inputs,
            )
        )
        return []

    worker_type = _extract_worker_methods(
        {
            "_te_cuda_graph_uses_hybridep_preprocess_capture",
            "_hybridep_dispatcher_configs",
            "_hybridep_uneven_padding_for_eager_path",
            "get_logprobs",
        },
        {
            "BatchedDataDict": Batch,
            "LogprobOutputSpec": Any,
            "LogprobsPostProcessor": lambda **_kwargs: object(),
            "_should_use_router_replay": lambda **_kwargs: False,
            "broadcast_tensors_from_last_stage": lambda _tensors: {
                "logprobs": object()
            },
            "get_microbatch_iterator": lambda *_args, **_kwargs: (
                iter(()),
                1,
                1,
                1,
                1,
            ),
            "megatron_forward_backward": forward,
            "maybe_r3_trace_stage": lambda *_args, **_kwargs: nullcontext(),
            "parallel_state": SimpleNamespace(
                is_pipeline_last_stage=lambda **_kwargs: False
            ),
            "torch": SimpleNamespace(no_grad=lambda: NoGrad()),
        },
    )
    worker = worker_type()
    worker.megatron_cfg = SimpleNamespace(model=provider_config)
    worker._te_cuda_graph_lifecycle = object()
    worker.model = SimpleNamespace(
        modules=lambda: (SimpleNamespace(config=runtime_config),),
        eval=lambda: None,
    )
    worker.cfg = {"logprob_batch_size": 1, "megatron_cfg": {}}
    worker.timer = SimpleNamespace(start=lambda _name: None, stop=lambda _name: None)
    worker.mcore_state = SimpleNamespace(straggler_timer=None)
    worker.sampling_params = None
    worker.delegate_pack_to_model = False
    worker.delegate_mtp_loss_mask_to_model = False
    worker.model_slices_context_parallel_inputs = False
    worker.defer_fp32_logits = False
    worker.draft_model = None
    worker._router_replay_enabled = False
    worker._deactivate_te_cuda_graphs_for_eager_path = lambda: None
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name, group=None: value
    )

    output = worker.get_logprobs(data=object())

    assert isinstance(output, Batch)
    assert ("forward", True, True) in events
    assert provider_config.moe_hybridep_pad_uneven_dispatch_inputs is False
    assert runtime_config.moe_hybridep_pad_uneven_dispatch_inputs is False
