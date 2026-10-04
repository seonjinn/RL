"""Read-only storage accounting checks, run in the GB200 policy environment."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def load_inventory():
    path = Path(__file__).resolve().parents[2] / "nemo_rl/utils/storage_inventory.py"
    assert path.exists(), "Read-only storage inventory is not implemented"
    spec = importlib.util.spec_from_file_location("storage_inventory_probe", path)
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
        shape=(32, 32), dtype=torch.bfloat16, requires_grad=False,
        rowwise_data=row, columnwise_data=col,
        rowwise_scale_inv=row_scale, columnwise_scale_inv=col_scale,
        fp8_dtype=DType.kFloat8E4M3, quantizer=None,
        with_gemm_swizzled_scales=False,
    )
    metadata = weight.get_metadata()
    pointers = {key: value.untyped_storage().data_ptr() for key, value in metadata.items()
                if isinstance(value, torch.Tensor)}
    with patch.object(MXFP8Tensor, "prepare_for_saving", side_effect=AssertionError("mutating API")), \
         patch.object(MXFP8Tensor, "dequantize", side_effect=AssertionError("value API")), \
         patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=AssertionError("reset")), \
         patch.object(torch.cuda, "empty_cache", side_effect=AssertionError("clear")), \
         patch.object(torch.cuda, "synchronize", side_effect=AssertionError("sync")):
        inventory = probe.collect_tensor_storages([("expert", weight), ("row_alias", row)])
    assert len(inventory.storages) == 4
    assert sum(record.nbytes for record in inventory.storages) == 3072
    assert not inventory.errors
    for key, pointer in pointers.items():
        assert weight.get_metadata()[key].untyped_storage().data_ptr() == pointer
    assert "row_alias" in next(record.aliases for record in inventory.storages
                              if record.pointer == row.untyped_storage().data_ptr())


def test_policy_inventory_exposes_shared_master_and_independent_grad_owners() -> None:
    probe = load_inventory()
    model = torch.nn.Linear(4, 4, bias=False, device="cuda", dtype=torch.bfloat16)
    master = torch.nn.Parameter(model.weight.float(), requires_grad=True)
    master.grad = torch.ones_like(master)
    model.weight.main_param = master
    model.weight.main_grad = torch.ones_like(model.weight)
    model.expert_parallel_buffers = []
    model.buffers = [SimpleNamespace(param_data=model.weight, grad_data=model.weight.main_grad)]
    inner = SimpleNamespace(param_groups=[{"params": [master]}], state={master: {"exp_avg": master.detach()}})
    optimizer = SimpleNamespace(optimizer=inner, shard_fp32_from_float16_groups=[[master]])
    inventory = probe.collect_policy_storages(model, optimizer)
    assert not inventory.errors
    assert len(inventory.storages) == 4
    record = next(record for record in inventory.storages if record.pointer == master.untyped_storage().data_ptr())
    assert any("main_param" in alias for alias in record.aliases)
    assert any("shard_fp32_from_float16_groups" in alias for alias in record.aliases)
    assert any("param_groups" in alias for alias in record.aliases)
    assert any("exp_avg" in alias for alias in record.aliases)
    assert master.grad is not None and master.device.type == "cuda"


def test_disabled_probe_does_not_inspect_or_initialize_cuda() -> None:
    probe = load_inventory()
    with patch.dict(os.environ, {}, clear=True), \
         patch.object(probe, "collect_policy_storages", side_effect=AssertionError("traversal")), \
         patch.object(torch.cuda, "current_device", side_effect=AssertionError("CUDA init")):
        probe.log_policy_storage_inventory(object(), object(), rank=0, phase="disabled")

