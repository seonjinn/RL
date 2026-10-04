"""Read-only storage accounting checks, run in the GB200 policy environment."""

import ast
import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

import torch


def load_inventory() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "nemo_rl/utils/storage_inventory.py"
    assert path.exists(), "Read-only storage inventory is not implemented"
    spec = importlib.util.spec_from_file_location("storage_inventory_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_deduplicates_views_by_storage_not_tensor_pointer() -> None:
    probe = load_inventory()
    weight = torch.arange(32, device="cuda", dtype=torch.float32)
    before = weight.clone()
    inventory = probe.collect_tensor_storages(
        [("ddp.param_data", weight), ("master", weight[8:16]), ("grad", weight[:0])]
    )
    assert len(inventory.storages) == 1
    record = inventory.storages[0]
    assert record.nbytes == weight.untyped_storage().nbytes() == 128
    assert record.aliases == ["ddp.param_data", "master", "grad"]
    assert record.pointer == weight.untyped_storage().data_ptr()
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


def test_separates_independent_gradients_and_zero_storage() -> None:
    probe = load_inventory()
    weight = torch.ones(16, device="cuda", dtype=torch.bfloat16)
    grad = weight.float()
    empty = torch.empty(0, device="cuda")
    inventory = probe.collect_tensor_storages(
        [("weight", weight), ("master.grad", grad), ("empty", empty)]
    )
    assert sorted(record.nbytes for record in inventory.storages) == [32, 64]
    assert inventory.zero_storage_aliases == ["empty"]
    assert not inventory.errors


def test_te_metadata_counts_payloads_without_saving_or_dequantizing() -> None:
    probe = load_inventory()
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
    from transformer_engine_torch import DType

    row = torch.ones((32, 32), device="cuda", dtype=torch.uint8)
    col = row.clone()
    row_scale = torch.ones((128, 4), device="cuda", dtype=torch.uint8)
    col_scale = row_scale.clone()
    weight = MXFP8Tensor(
        shape=(32, 32),
        dtype=torch.bfloat16,
        requires_grad=False,
        rowwise_data=row,
        columnwise_data=col,
        rowwise_scale_inv=row_scale,
        columnwise_scale_inv=col_scale,
        fp8_dtype=DType.kFloat8E4M3,
        quantizer=None,
        with_gemm_swizzled_scales=False,
    )
    metadata = weight.get_metadata()
    pointers = {
        key: value.untyped_storage().data_ptr()
        for key, value in metadata.items()
        if isinstance(value, torch.Tensor)
    }
    with (
        patch.object(
            MXFP8Tensor,
            "prepare_for_saving",
            side_effect=AssertionError("mutating API"),
        ),
        patch.object(
            MXFP8Tensor, "dequantize", side_effect=AssertionError("value API")
        ),
        patch.object(
            torch.cuda, "reset_peak_memory_stats", side_effect=AssertionError("reset")
        ),
        patch.object(torch.cuda, "empty_cache", side_effect=AssertionError("clear")),
        patch.object(torch.cuda, "synchronize", side_effect=AssertionError("sync")),
    ):
        inventory = probe.collect_tensor_storages(
            [("expert", weight), ("row_alias", row)]
        )
    assert len(inventory.storages) == 4
    assert sum(record.nbytes for record in inventory.storages) == 3072
    assert not inventory.errors
    for key, pointer in pointers.items():
        assert weight.get_metadata()[key].untyped_storage().data_ptr() == pointer
    assert "row_alias" in next(
        record.aliases
        for record in inventory.storages
        if record.pointer == row.untyped_storage().data_ptr()
    )


def test_policy_inventory_exposes_shared_master_and_independent_grad_owners() -> None:
    probe = load_inventory()
    model = torch.nn.Linear(4, 4, bias=False, device="cuda", dtype=torch.bfloat16)
    master = torch.nn.Parameter(model.weight.float(), requires_grad=True)
    master.grad = torch.ones_like(master)
    model.weight.main_param = master
    model.weight.main_grad = torch.ones_like(model.weight)
    model.expert_parallel_buffers = []
    model.buffers = [
        SimpleNamespace(param_data=model.weight, grad_data=model.weight.main_grad)
    ]
    inner = SimpleNamespace(
        param_groups=[{"params": [master]}],
        state={master: {"exp_avg": master.detach()}},
    )
    optimizer = SimpleNamespace(
        optimizer=inner, shard_fp32_from_float16_groups=[[master]]
    )
    inventory = probe.collect_policy_storages(model, optimizer)
    assert not inventory.errors
    assert len(inventory.storages) == 4
    record = next(
        record
        for record in inventory.storages
        if record.pointer == master.untyped_storage().data_ptr()
    )
    assert any("main_param" in alias for alias in record.aliases)
    assert any("shard_fp32_from_float16_groups" in alias for alias in record.aliases)
    assert any("param_groups" in alias for alias in record.aliases)
    assert any("exp_avg" in alias for alias in record.aliases)
    assert master.grad is not None and master.device.type == "cuda"


def test_disabled_probe_does_not_inspect_or_initialize_cuda() -> None:
    probe = load_inventory()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(
            probe, "collect_policy_storages", side_effect=AssertionError("traversal")
        ),
        patch.object(
            torch.cuda, "current_device", side_effect=AssertionError("CUDA init")
        ),
    ):
        probe.log_policy_storage_inventory(object(), object(), rank=0, phase="disabled")


def test_enabled_probe_preserves_values_storage_and_allocator_counters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe = load_inventory()
    model = torch.nn.Linear(4, 4, bias=False, device="cuda", dtype=torch.bfloat16)
    before = model.weight.detach().clone()
    pointer = model.weight.untyped_storage().data_ptr()
    counters_before = torch.cuda.memory_stats()
    with (
        patch.dict(os.environ, {"NRL_STORAGE_INVENTORY": "1"}),
        patch.object(
            torch.cuda, "reset_peak_memory_stats", side_effect=AssertionError("reset")
        ),
        patch.object(torch.cuda, "empty_cache", side_effect=AssertionError("clear")),
        patch.object(torch.cuda, "synchronize", side_effect=AssertionError("sync")),
    ):
        probe.log_policy_storage_inventory(model, None, rank=0, phase="test")
    assert "[NRL_STORAGE_INVENTORY]" in capsys.readouterr().out
    assert model.weight.untyped_storage().data_ptr() == pointer
    assert torch.cuda.memory_stats() == counters_before
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_wake_preserves_tags_and_original_failure(
    asynchronous: bool, enabled: bool
) -> None:
    probe = load_inventory()
    filename = "vllm_worker_async.py" if asynchronous else "vllm_worker.py"
    method = "wake_up_async" if asynchronous else "wake_up"
    path = (
        Path(__file__).resolve().parents[2]
        / "nemo_rl/models/generation/vllm"
        / filename
    )
    tree = ast.parse(path.read_text())
    definition = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method
    )
    failure = RuntimeError("distinctive native wake failure")
    wake = AsyncMock(side_effect=failure) if asynchronous else Mock(side_effect=failure)
    rpc = Mock(return_value=None)
    worker = SimpleNamespace(
        llm=SimpleNamespace(wake_up=wake, collective_rpc=rpc),
        cfg={"vllm_cfg": {"async_engine": asynchronous}},
    )
    log = Mock()
    namespace = {
        "asyncio": asyncio,
        "log_wake_event": log,
        "storage_inventory_enabled": probe.storage_inventory_enabled,
        "resolve_collective_rpc_result": AsyncMock(return_value=None),
    }
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    with patch.dict(os.environ, {"NRL_STORAGE_INVENTORY": "1" if enabled else "0"}):
        with pytest.raises(RuntimeError) as caught:
            if asynchronous:
                asyncio.run(namespace[method](worker, tags=["weights"]))
            else:
                namespace[method](worker, tags=["weights"])
    assert caught.value is failure
    wake.assert_called_once_with(tags=["weights"])
    assert rpc.call_count == int(enabled)
    assert [call.kwargs["phase"] for call in log.call_args_list] == ["enter", "failed"]


def test_existing_grouped_te_storage_includes_scales_and_offsets() -> None:
    probe = load_inventory()
    from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import (
        GroupedTensorStorage,
    )

    data = torch.ones(2048, device="cuda", dtype=torch.uint8)
    scale = torch.ones(512, device="cuda", dtype=torch.uint8)
    offsets = torch.ones(3, device="cuda", dtype=torch.int64)
    grouped = GroupedTensorStorage(
        shape=(64, 32),
        dtype=torch.bfloat16,
        num_tensors=2,
        data=data,
        columnwise_data=data,
        scale_inv=scale,
        columnwise_scale_inv=scale,
        tensor_offsets=offsets,
    )
    with patch.object(
        GroupedTensorStorage,
        "prepare_for_saving",
        side_effect=AssertionError("mutating API"),
    ):
        inventory = probe.collect_tensor_storages([("grouped", grouped)])
    assert not inventory.errors
    assert sum(record.nbytes for record in inventory.storages) == 2584
    assert grouped.quantized_tensors is None
    assert grouped.scale_inv is scale and grouped.tensor_offsets is offsets


def test_tied_weights_keep_both_owner_names() -> None:
    probe = load_inventory()
    model = torch.nn.Module()
    model.first = torch.nn.Linear(4, 4, bias=False, device="cuda")
    model.second = torch.nn.Linear(4, 4, bias=False, device="cuda")
    model.second.weight = model.first.weight
    model.token_dispatcher = object()
    inventory = probe.collect_policy_storages(model, None)
    assert len(inventory.storages) == 1
    assert "model.parameters.first.weight" in inventory.storages[0].aliases
    assert "model.parameters.second.weight" in inventory.storages[0].aliases
    assert not inventory.errors


def test_storage_error_does_not_drop_next_sibling() -> None:
    probe = load_inventory()
    bad = torch.ones(1, device="cuda")
    good = torch.ones(2, device="cuda")
    with patch.object(
        bad, "untyped_storage", side_effect=RuntimeError("distinctive storage error")
    ):
        inventory = probe.collect_tensor_storages([("siblings", [bad, good])])
    assert len(inventory.errors) == 1
    assert "distinctive storage error" in inventory.errors[0]
    assert len(inventory.storages) == 1
    assert inventory.storages[0].pointer == good.untyped_storage().data_ptr()


def test_metadata_logging_is_best_effort_even_when_stdout_is_closed() -> None:
    probe = load_inventory()
    with (
        patch.dict(os.environ, {"NRL_STORAGE_INVENTORY": "1"}),
        patch("builtins.print", side_effect=BrokenPipeError("closed output")),
    ):
        probe.log_wake_event(
            phase="failed", tags=["weights"], error=RuntimeError("wake")
        )
        probe.log_policy_storage_inventory(None, None, rank=0, phase="test")
        probe.log_rollout_storage_boundary(rank=0, phase="test")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_probe_rpc_failure_does_not_prevent_native_wake(asynchronous: bool) -> None:
    probe = load_inventory()
    filename = "vllm_worker_async.py" if asynchronous else "vllm_worker.py"
    method = "wake_up_async" if asynchronous else "wake_up"
    path = (
        Path(__file__).resolve().parents[2]
        / "nemo_rl/models/generation/vllm"
        / filename
    )
    definition = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method
    )
    wake = AsyncMock() if asynchronous else Mock()
    worker = SimpleNamespace(
        llm=SimpleNamespace(
            wake_up=wake,
            collective_rpc=Mock(side_effect=RuntimeError("probe unavailable")),
        ),
        cfg={"vllm_cfg": {"async_engine": asynchronous}},
    )
    namespace = {
        "asyncio": asyncio,
        "log_wake_event": probe.log_wake_event,
        "storage_inventory_enabled": probe.storage_inventory_enabled,
        "resolve_collective_rpc_result": AsyncMock(return_value=None),
    }
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    with patch.dict(os.environ, {"NRL_STORAGE_INVENTORY": "1"}):
        if asynchronous:
            asyncio.run(namespace[method](worker, tags=["kv_cache"]))
        else:
            namespace[method](worker, tags=["kv_cache"])
    wake.assert_called_once_with(tags=["kv_cache"])
