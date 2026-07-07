# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _function_node(
    path: Path, name: str, class_name: str | None = None
) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        class_node = next(
            node
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = class_node.body
    return next(
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_functions(
    path: Path,
    names: list[str],
    *,
    class_name: str | None = None,
    namespace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    future_annotations = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )
    module = ast.Module(
        body=[
            future_annotations,
            *[_function_node(path, name, class_name) for name in names],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    loaded: dict[str, Any] = {} if namespace is None else dict(namespace)
    exec(compile(module, str(path), "exec"), loaded)
    return {name: loaded[name] for name in names}


class _FakeTensorType:
    pass


class _FakeTorch:
    Tensor = _FakeTensorType

    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, _FakeTensorType)


class _FakePackedTensor:
    def __init__(self, tensors: list[object] | None = None) -> None:
        self.tensors = tensors or []


class _SlicedDataDict(dict):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.micro_batch_indices = None
        self.micro_batch_lengths = None
        self.elem_counts_per_gb = None


def test_real_policy_sharding_preserves_four_global_batch_chunks() -> None:
    batch_functions = _load_functions(
        REPO_ROOT / "nemo_rl/distributed/batched_data_dict.py",
        ["shard_by_batch_size"],
        class_name="BatchedDataDict",
        namespace={
            "torch": _FakeTorch,
            "PackedTensor": _FakePackedTensor,
            "SlicedDataDict": _SlicedDataDict,
            "get_packer": None,
            "os": os,
        },
    )
    policy_functions = _load_functions(
        REPO_ROOT / "nemo_rl/models/policy/lm_policy.py",
        ["_shard_for_train"],
        class_name="Policy",
    )

    batch = SimpleNamespace(data={"row_id": list(range(256))})
    batch.shard_by_batch_size = batch_functions["shard_by_batch_size"].__get__(batch)
    policy = SimpleNamespace(
        data_parallel_size=4,
        use_dynamic_batches=False,
        use_sequence_packing=False,
    )

    shards = policy_functions["_shard_for_train"](policy, batch, 64)

    assert len(shards) == 4
    assert [len(shard["row_id"]) for shard in shards] == [64, 64, 64, 64]
    assert shards[0]["row_id"] == [
        *range(0, 16),
        *range(64, 80),
        *range(128, 144),
        *range(192, 208),
    ]


def test_megatron_worker_processes_and_normalizes_four_losses() -> None:
    worker_path = REPO_ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
    worker_functions = _load_functions(
        worker_path,
        ["_global_batch_indices", "_normalize_global_batch_loss_metrics"],
    )
    global_batch_indices = worker_functions["_global_batch_indices"]
    normalize_metrics = worker_functions["_normalize_global_batch_loss_metrics"]

    processed_indices = []
    processed_losses = []
    raw_losses = [8.0, 12.0, 16.0, 20.0]
    indices = global_batch_indices(total_dataset_size=256, global_batch_size=64)
    for global_batch_idx in indices:
        processed_indices.append(global_batch_idx)
        normalized = normalize_metrics(
            {"loss": raw_losses[global_batch_idx], "loss_min": 1.0},
            num_global_batches=len(indices),
        )
        processed_losses.append(normalized["loss"])
        assert normalized["loss_min"] == 1.0

    assert processed_indices == [0, 1, 2, 3]
    assert processed_losses == [2.0, 3.0, 4.0, 5.0]

    train_node = _function_node(
        worker_path, "train", class_name="MegatronPolicyWorkerImpl"
    )
    called_names = {
        node.func.id
        for node in ast.walk(train_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_global_batch_indices" in called_names
    assert "_normalize_global_batch_loss_metrics" in called_names


class _PayloadTensor:
    def __init__(self, numel: int, element_size: int) -> None:
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class _PayloadTorch:
    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, _PayloadTensor)


def test_recursive_payload_bytes_and_capacity_guard_execute_real_helpers() -> None:
    functions = _load_functions(
        REPO_ROOT / "nemo_rl/algorithms/sft.py",
        ["_recursive_tensor_payload_bytes", "_validate_event_memory_capacity"],
        namespace={
            "torch": _PayloadTorch,
            "PackedTensor": _FakePackedTensor,
            "Mapping": Mapping,
            "math": math,
        },
    )
    payload_bytes = functions["_recursive_tensor_payload_bytes"](
        {
            "input": _PayloadTensor(8, 4),
            "nested": [_FakePackedTensor([_PayloadTensor(3, 8)])],
        }
    )

    assert payload_bytes == 56
    required_bytes = functions["_validate_event_memory_capacity"](
        payload_bytes,
        max_payload_bytes=64,
        host_available_bytes=256,
        verified_ray_object_store_available_bytes=256,
        safety_multiplier=2.0,
    )
    assert required_bytes == 112

    with pytest.raises(MemoryError, match="payload budget"):
        functions["_validate_event_memory_capacity"](
            payload_bytes,
            max_payload_bytes=55,
            host_available_bytes=256,
            verified_ray_object_store_available_bytes=256,
            safety_multiplier=2.0,
        )


def test_event_contract_validation_and_submission_order() -> None:
    sft_path = REPO_ROOT / "nemo_rl/algorithms/sft.py"
    functions = _load_functions(
        sft_path,
        ["_validate_event_execution_config"],
        namespace={"math": math},
    )
    config = SimpleNamespace(
        validation_execution_mode="event_batch",
        val_batches=4,
        val_global_batch_size=64,
        val_micro_batch_size=1,
        validation_event_max_payload_bytes=1_000,
        validation_event_verified_ray_object_store_available_bytes=10_000,
        validation_event_memory_safety_multiplier=2.0,
    )

    with pytest.raises(ValueError, match="val_batches=4"):
        functions["_validate_event_execution_config"](
            config,
            val_batches=3,
            val_batch_size=64,
            val_mbs=1,
        )

    validate_node = _function_node(sft_path, "_validate_with_loss_availability")
    calls = [node for node in ast.walk(validate_node) if isinstance(node, ast.Call)]
    config_call = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "_validate_event_execution_config"
    )
    enumerate_call = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "enumerate"
    )
    prepare_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "prepare_for_training"
    ]
    capacity_call = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "_validate_event_memory_capacity"
    )
    policy_train_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "train"
    ]

    assert config_call.lineno < enumerate_call.lineno
    assert all(config_call.lineno < call.lineno for call in prepare_calls)
    assert capacity_call.lineno < max(call.lineno for call in policy_train_calls)

    combine_node = _function_node(sft_path, "_combine_validation_event_batches")
    combine_calls = [
        node for node in ast.walk(combine_node) if isinstance(node, ast.Call)
    ]
    from_batches_call = next(
        node
        for node in combine_calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "from_batches"
    )
    clear_call = next(
        node
        for node in combine_calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "clear"
    )
    assert from_batches_call.lineno < clear_call.lineno
