# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import ast
import base64
import binascii
import hashlib
import math
import os
import sys
from collections import UserDict
from copy import deepcopy
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
    matches = [
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert matches, f"Missing required function {name} in {path}"
    return matches[0]


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


def test_deep_payload_counts_userdict_backing_state_and_nested_objects() -> None:
    functions = _load_functions(
        REPO_ROOT / "nemo_rl/algorithms/sft.py",
        ["_iter_payload_children", "_recursive_deep_payload_bytes"],
        namespace={
            "torch": _FakeTorch,
            "PackedTensor": _FakePackedTensor,
            "Mapping": Mapping,
            "sys": sys,
        },
    )

    class NestedPayload:
        def __init__(self) -> None:
            self.blob = bytearray(4096)

    class PayloadDict(UserDict[str, list[NestedPayload]]):
        def __init__(self) -> None:
            super().__init__({"nested": [NestedPayload()]})
            self.metadata = bytearray(2048)

    payload = PayloadDict()
    measured = functions["_recursive_deep_payload_bytes"](payload)
    lower_bound = sum(
        sys.getsizeof(value)
        for value in (
            payload,
            vars(payload),
            payload.data,
            payload.metadata,
            payload["nested"][0],
            vars(payload["nested"][0]),
            payload["nested"][0].blob,
        )
    )

    assert measured >= lower_bound


def test_deep_payload_counts_frozenset_tensor_state_and_private_slots() -> None:
    functions = _load_functions(
        REPO_ROOT / "nemo_rl/algorithms/sft.py",
        ["_iter_payload_children", "_recursive_deep_payload_bytes"],
        namespace={
            "torch": _FakeTorch,
            "PackedTensor": _FakePackedTensor,
            "Mapping": Mapping,
            "sys": sys,
        },
    )

    class Storage:
        def nbytes(self) -> int:
            return 64

        def data_ptr(self) -> int:
            return 1234

    class StatefulTensor(_FakeTensorType):
        def __init__(self) -> None:
            self.device = "cpu"
            self.metadata = bytearray(1024)

        def untyped_storage(self) -> Storage:
            return Storage()

    class PrivateSlot:
        __slots__ = ("__hidden",)

        def __init__(self) -> None:
            self.__hidden = bytearray(2048)

    tensor = StatefulTensor()
    private_slot = PrivateSlot()
    payload = {"frozen": frozenset({private_slot}), "tensor": tensor}
    measured = functions["_recursive_deep_payload_bytes"](payload)
    lower_bound = (
        sum(
            sys.getsizeof(value)
            for value in (
                tensor,
                vars(tensor),
                tensor.metadata,
                private_slot,
                object.__getattribute__(private_slot, "_PrivateSlot__hidden"),
            )
        )
        + 64
    )

    assert measured >= lower_bound


def test_cache_budget_and_loss_clone_helpers_preserve_off_path() -> None:
    functions = _load_functions(
        REPO_ROOT / "nemo_rl/algorithms/sft.py",
        ["_validation_event_budget_payload_bytes", "_clone_validation_loss_fn"],
        namespace={"deepcopy": deepcopy},
    )
    original_loss = SimpleNamespace(mutable_state=[])

    assert (
        functions["_validation_event_budget_payload_bytes"](
            cache_mode="off",
            logical_payload_bytes=100,
            deep_payload_bytes=150,
        )
        == 100
    )
    assert (
        functions["_validation_event_budget_payload_bytes"](
            cache_mode="cpu",
            logical_payload_bytes=100,
            deep_payload_bytes=150,
        )
        == 150
    )
    assert functions["_clone_validation_loss_fn"](original_loss, "off") is original_loss
    cloned_loss = functions["_clone_validation_loss_fn"](original_loss, "cpu")
    assert cloned_loss is not original_loss
    cloned_loss.mutable_state.append("changed")
    assert original_loss.mutable_state == []


def test_cpu_cache_payload_rejects_reachable_non_cpu_tensor() -> None:
    functions = _load_functions(
        REPO_ROOT / "nemo_rl/algorithms/sft.py",
        ["_iter_payload_children", "_validate_cpu_cache_payload"],
        namespace={
            "torch": _FakeTorch,
            "PackedTensor": _FakePackedTensor,
            "Mapping": Mapping,
        },
    )

    class DeviceTensor(_FakeTensorType):
        def __init__(self, device_type: str) -> None:
            self.device = SimpleNamespace(type=device_type)

    functions["_validate_cpu_cache_payload"](
        {"nested": [_FakePackedTensor([DeviceTensor("cpu")])]}
    )
    with pytest.raises(ValueError, match="CPU cache.*meta"):
        functions["_validate_cpu_cache_payload"](
            {"nested": [_FakePackedTensor([DeviceTensor("meta")])]}
        )

    class TensorWithState(DeviceTensor):
        def __init__(self) -> None:
            super().__init__("cpu")
            self.attached = DeviceTensor("meta")

    class PrivateSlot:
        __slots__ = ("__hidden",)

        def __init__(self) -> None:
            self.__hidden = DeviceTensor("meta")

    for payload in (
        TensorWithState(),
        frozenset({DeviceTensor("meta")}),
        PrivateSlot(),
    ):
        with pytest.raises(ValueError, match="CPU cache.*meta"):
            functions["_validate_cpu_cache_payload"](payload)


def test_payload_child_iterator_bypasses_custom_slot_accessor() -> None:
    functions = _load_functions(
        REPO_ROOT / "nemo_rl/algorithms/sft.py",
        ["_iter_payload_children"],
        namespace={"PackedTensor": _FakePackedTensor, "Mapping": Mapping},
    )

    class CountingSlot:
        __slots__ = ("value", "reads")

        def __init__(self) -> None:
            object.__setattr__(self, "value", bytearray(32))
            object.__setattr__(self, "reads", 0)

        def __getattribute__(self, name: str) -> object:
            if name == "value":
                reads = object.__getattribute__(self, "reads")
                object.__setattr__(self, "reads", reads + 1)
            return object.__getattribute__(self, name)

    payload = CountingSlot()
    children = list(functions["_iter_payload_children"](payload))

    assert payload.reads == 0
    assert any(isinstance(child, bytearray) for child in children)


def test_event_contract_validation_and_submission_order() -> None:
    sft_path = REPO_ROOT / "nemo_rl/algorithms/sft.py"
    functions = _load_functions(
        sft_path,
        ["_validate_event_execution_config"],
        namespace={"math": math},
    )
    config = SimpleNamespace(
        validation_input_mode="dataloader",
        validation_execution_mode="event_batch",
        validation_event_cache_mode="off",
        validation_event_cache_dataset_sha256=None,
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

    validate_node = _function_node(sft_path, "_validate_with_loss_availability_impl")
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


def test_precomputed_event_contract_fails_closed_without_runtime_dependencies() -> None:
    functions = _load_functions(
        REPO_ROOT / "nemo_rl/algorithms/sft.py",
        ["_validate_event_execution_config"],
        namespace={"math": math, "re": __import__("re")},
    )
    base = {
        "validation_input_mode": "precomputed_event",
        "validation_execution_mode": "event_batch",
        "validation_event_cache_mode": "off",
        "validation_event_cache_dataset_sha256": None,
        "validation_precomputed_manifest": "/tmp/validation.manifest.json",
        "validation_precomputed_dataset_sha256": "a" * 64,
        "validation_precomputed_tokenizer_sha256": "b" * 64,
        "validation_precomputed_container_sha256": "c" * 64,
        "val_batches": 4,
        "val_global_batch_size": 64,
        "val_micro_batch_size": 1,
        "validation_event_max_payload_bytes": None,
        "validation_event_verified_ray_object_store_available_bytes": None,
        "validation_event_memory_safety_multiplier": 2.0,
    }

    assert (
        functions["_validate_event_execution_config"](
            SimpleNamespace(**base),
            val_batches=4,
            val_batch_size=64,
            val_mbs=1,
        )
        is None
    )

    for field_name, message in (
        ("validation_precomputed_manifest", "event_batch and a manifest"),
        ("validation_precomputed_dataset_sha256", "dataset SHA-256"),
        ("validation_precomputed_tokenizer_sha256", "tokenizer SHA-256"),
        ("validation_precomputed_container_sha256", "container SHA-256"),
    ):
        invalid = dict(base)
        invalid[field_name] = None
        with pytest.raises(ValueError, match=message):
            functions["_validate_event_execution_config"](
                SimpleNamespace(**invalid),
                val_batches=4,
                val_batch_size=64,
                val_mbs=1,
            )

    cached = dict(base)
    cached["validation_event_cache_mode"] = "cpu"
    with pytest.raises(ValueError, match="runtime CPU cache"):
        functions["_validate_event_execution_config"](
            SimpleNamespace(**cached),
            val_batches=4,
            val_batch_size=64,
            val_mbs=1,
        )


def test_validation_provenance_is_shared_with_the_producer() -> None:
    producer_path = REPO_ROOT / "examples/prepare_sft_validation_event.py"
    producer_tree = ast.parse(producer_path.read_text())
    shared_import = next(
        node
        for node in producer_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "nemo_rl.algorithms.sft_validation_provenance"
    )
    imported_names = {alias.name for alias in shared_import.names}
    assert {
        "build_validation_artifact_fingerprint",
        "derive_preprocessing_sha256",
        "validate_validation_source_config",
    } <= imported_names

    producer_functions = {
        node.name for node in producer_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert (
        not {
            "build_validation_artifact_fingerprint",
            "derive_preprocessing_sha256",
            "validate_validation_source_config",
        }
        & producer_functions
    )

    shared_path = REPO_ROOT / "nemo_rl/algorithms/sft_validation_provenance.py"
    shared_functions = {
        node.name
        for node in ast.parse(shared_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
    }
    assert {
        "build_validation_artifact_fingerprint",
        "derive_preprocessing_sha256",
        "validate_validation_source_config",
    } <= shared_functions


def test_precomputed_event_is_explicit_cloned_and_restored_in_finally() -> None:
    sft_path = REPO_ROOT / "nemo_rl/algorithms/sft.py"
    validation_node = _function_node(sft_path, "_validate_with_loss_availability")
    validation_impl_node = _function_node(
        sft_path, "_validate_with_loss_availability_impl"
    )
    validate_node = _function_node(sft_path, "validate")
    train_node = _function_node(sft_path, "sft_train")

    for function_node in (validation_node, validate_node, train_node):
        assert "precomputed_validation_event" in {
            argument.arg for argument in function_node.args.kwonlyargs
        }

    clone_calls = [
        node
        for node in ast.walk(validation_impl_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "clone_validation_event_data"
    ]
    assert len(clone_calls) == 1
    successful_restore_calls = [
        node
        for node in ast.walk(validation_impl_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "restore_training_mode"
    ]
    assert len(successful_restore_calls) == 1

    restoration_tries = []
    for try_node in (
        node for node in ast.walk(validation_node) if isinstance(node, ast.Try)
    ):
        final_calls = [
            node
            for statement in try_node.finalbody
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "prepare_for_training"
        ]
        if final_calls:
            restoration_tries.append(try_node)
    assert restoration_tries

    cache_publications = [
        node
        for node in ast.walk(validation_node)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "entry"
            for target in node.targets
        )
    ]
    assert len(cache_publications) == 1
    assert cache_publications[0].lineno > restoration_tries[0].end_lineno

    for caller_node, callee_name in (
        (validate_node, "_validate_with_loss_availability"),
        (train_node, "_validate_with_loss_availability"),
    ):
        forwarded_calls = [
            node
            for node in ast.walk(caller_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == callee_name
        ]
        assert forwarded_calls
        assert all(
            any(
                keyword.arg == "precomputed_validation_event"
                for keyword in call.keywords
            )
            for call in forwarded_calls
        )


def test_runtime_loads_verified_event_before_distributed_or_data_setup() -> None:
    runner_path = REPO_ROOT / "examples/run_sft.py"
    loader_node = _function_node(runner_path, "_load_precomputed_validation_event")
    main_node = _function_node(runner_path, "main")

    loader_calls = [
        node for node in ast.walk(loader_node) if isinstance(node, ast.Call)
    ]
    called_names = {
        node.func.id for node in loader_calls if isinstance(node.func, ast.Name)
    }
    assert {
        "_validate_event_execution_config",
        "validate_validation_source_config",
        "derive_preprocessing_sha256",
        "build_validation_artifact_fingerprint",
        "load_validation_event",
    } <= called_names

    loader_source = ast.unparse(loader_node)
    for field_name in (
        "validation_precomputed_dataset_sha256",
        "validation_precomputed_tokenizer_sha256",
        "validation_precomputed_container_sha256",
    ):
        assert field_name in loader_source

    load_call = next(
        node
        for node in loader_calls
        if isinstance(node.func, ast.Name) and node.func.id == "load_validation_event"
    )
    assert isinstance(load_call.args[1], ast.Name)
    assert load_call.args[1].id == "expected_fingerprint"

    main_calls = [node for node in ast.walk(main_node) if isinstance(node, ast.Call)]
    precomputed_load = next(
        node
        for node in main_calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "_load_precomputed_validation_event"
    )
    init_ray_call = next(
        node
        for node in main_calls
        if isinstance(node.func, ast.Name) and node.func.id == "init_ray"
    )
    tokenizer_call = next(
        node
        for node in main_calls
        if isinstance(node.func, ast.Name) and node.func.id == "get_tokenizer"
    )
    setup_data_calls = [
        node
        for node in main_calls
        if isinstance(node.func, ast.Name) and node.func.id == "setup_data"
    ]
    setup_call = next(
        node
        for node in main_calls
        if isinstance(node.func, ast.Name) and node.func.id == "setup"
    )
    assert precomputed_load.lineno < init_ray_call.lineno
    assert precomputed_load.lineno < tokenizer_call.lineno
    assert all(precomputed_load.lineno < call.lineno for call in setup_data_calls)
    assert precomputed_load.lineno < setup_call.lineno
    assert any(
        any(
            keyword.arg == "load_validation"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
        for call in setup_data_calls
    )

    sft_train_call = next(
        node
        for node in main_calls
        if isinstance(node.func, ast.Name) and node.func.id == "sft_train"
    )
    assert any(
        keyword.arg == "precomputed_validation_event"
        for keyword in sft_train_call.keywords
    )


@pytest.mark.parametrize(
    "config_path",
    [
        Path("examples/configs/sft.yaml"),
        Path("examples/configs/sft_superv3_prepacked.yaml"),
        Path("tests/unit/reference_configs/sft.yaml"),
    ],
)
def test_sft_exemplar_configs_document_precomputed_inputs(config_path: Path) -> None:
    config_text = (REPO_ROOT / config_path).read_text()

    assert "validation_input_mode: dataloader" in config_text
    for field_name in (
        "validation_precomputed_manifest",
        "validation_precomputed_dataset_sha256",
        "validation_precomputed_tokenizer_sha256",
        "validation_precomputed_container_sha256",
    ):
        assert f"{field_name}: null" in config_text


def test_correctness_worker_covers_optimizer_parameters_and_rejects_bad_tracker_maps() -> (
    None
):
    worker_path = REPO_ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
    method = _function_node(
        worker_path,
        "get_correctness_state_fingerprint",
        class_name="MegatronPolicyWorkerImpl",
    )
    source = ast.unparse(method)

    assert "optimizer_parameter_records" in source
    assert "'parameters': optimizer_parameter_records" in source
    assert "isinstance(tracker_states, Mapping)" in source
    assert "not tracker_states" in source
    assert "get_cuda_rng_tracker" not in source
    call_attributes = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert call_attributes.isdisjoint(
        {
            "state_dict",
            "load_state_dict",
            "eval",
            "train",
            "set_rng_state",
            "manual_seed",
            "synchronize",
        }
    )


def test_correctness_audit_finalizes_next_batch_and_captures_runtime_validation_data() -> (
    None
):
    audit_path = REPO_ROOT / "nemo_rl/algorithms/sft_correctness_audit.py"
    audit_tree = ast.parse(audit_path.read_text())
    record = next(
        node
        for node in audit_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CorrectnessAuditRecord"
    )
    record_fields = {
        statement.target.id
        for statement in record.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    }
    functions = {
        node.name for node in audit_tree.body if isinstance(node, ast.FunctionDef)
    }

    assert "next_train_batch" in record_fields
    assert "validation_evidence" in record_fields
    assert "compare_next_train_batch_to_control" in functions

    sft_path = REPO_ROOT / "nemo_rl/algorithms/sft.py"
    validation_impl = _function_node(sft_path, "_validate_with_loss_availability_impl")
    evidence_capture = _function_node(sft_path, "_capture_runtime_validation_evidence")
    train = _function_node(sft_path, "sft_train")
    validation_source = ast.unparse(validation_impl)
    evidence_capture_source = ast.unparse(evidence_capture)
    train_source = ast.unparse(train)

    assert "capture_before" in validation_source
    assert "capture_after" in validation_source
    assert "capture_validation_evidence" in evidence_capture_source
    assert "correctness_evidence_collector" in validation_source
    assert "_ValidationCorrectnessEvidenceCollector" in train_source
    assert '"input_mode"' not in train_source


class _FakeCorrectnessAuditError(RuntimeError):
    pass


class _FakeTorchDataIterator:
    def state_dict(self) -> dict[str, object]:
        return {"position": 5}


class _FakeTorchDataLoader:
    def __init__(
        self,
        *,
        iterator: object | None = None,
        pending_state: object = None,
        initial_iter_for_state_dict: object = False,
    ) -> None:
        self._iterator = iterator
        self.next_iter_state = pending_state
        self._initial_iter_for_state_dict = initial_iter_for_state_dict
        self.state_dict_calls = 0

    def state_dict(self) -> dict[str, object]:
        self.state_dict_calls += 1
        return {"position": 5}


_TORCHDATA_SOURCE_RECORD = "torchdata/stateful_dataloader/stateful_dataloader.py"
_FAKE_TORCHDATA_SOURCE = (
    REPO_ROOT / "nemo_rl/algorithms/sft_correctness_audit.py"
).resolve()
_DEFAULT_DISTRIBUTION_FILES = object()
_DEFAULT_RUNTIME_PATH = object()


def _record_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class _FakePackagePath:
    def __init__(
        self,
        relative_path: str,
        located_path: Path,
        *,
        hash_mode: str | None = "sha256",
        hash_value: str | None = None,
    ) -> None:
        self.relative_path = relative_path
        self.located_path = located_path
        self.hash = (
            None
            if hash_mode is None
            else SimpleNamespace(
                mode=hash_mode,
                value=(
                    _record_sha256(located_path) if hash_value is None else hash_value
                ),
            )
        )

    def as_posix(self) -> str:
        return self.relative_path

    def locate(self) -> Path:
        return self.located_path


class _FakeImportlibMetadata:
    class PackageNotFoundError(Exception):
        pass

    def __init__(
        self,
        *,
        version: str = "0.11.0",
        distribution_name: str = "torchdata",
        package_owners: tuple[str, ...] = ("torchdata",),
        files: object = _DEFAULT_DISTRIBUTION_FILES,
    ) -> None:
        self.version = version
        self.distribution_name = distribution_name
        self.package_owners = package_owners
        self.source_path = _FAKE_TORCHDATA_SOURCE
        self.files = (
            [
                _FakePackagePath(
                    _TORCHDATA_SOURCE_RECORD,
                    self.source_path,
                )
            ]
            if files is _DEFAULT_DISTRIBUTION_FILES
            else files
        )
        self.distribution_calls = 0
        self.package_owner_calls = 0

    def distribution(self, name: str) -> SimpleNamespace:
        assert name == "torchdata"
        self.distribution_calls += 1
        return SimpleNamespace(
            version=self.version,
            metadata={"Name": self.distribution_name},
            files=self.files,
        )

    def packages_distributions(self) -> dict[str, list[str]]:
        self.package_owner_calls += 1
        return {"torchdata": list(self.package_owners)}


class _MissingTorchDataMetadata(_FakeImportlibMetadata):
    def distribution(self, name: str) -> SimpleNamespace:
        raise self.PackageNotFoundError(name)


def _load_train_loader_capture(
    metadata: _FakeImportlibMetadata,
    *,
    runtime_loader_class: object = _FakeTorchDataLoader,
    runtime_iterator_class: object = _FakeTorchDataIterator,
    runtime_origin: object = _DEFAULT_RUNTIME_PATH,
    runtime_file: object = _DEFAULT_RUNTIME_PATH,
    expected_runtime_module: str = _FakeTorchDataLoader.__module__,
) -> tuple[Any, Any]:
    audit_path = REPO_ROOT / "nemo_rl/algorithms/sft_correctness_audit.py"
    resolved_runtime_origin = (
        str(metadata.source_path)
        if runtime_origin is _DEFAULT_RUNTIME_PATH
        else runtime_origin
    )
    resolved_runtime_file = (
        str(metadata.source_path)
        if runtime_file is _DEFAULT_RUNTIME_PATH
        else runtime_file
    )
    runtime_module = SimpleNamespace(
        StatefulDataLoader=runtime_loader_class,
        _StatefulBaseDataLoaderIter=runtime_iterator_class,
        __spec__=SimpleNamespace(origin=resolved_runtime_origin),
        __file__=resolved_runtime_file,
    )
    functions = _load_functions(
        audit_path,
        [
            "_normalize_distribution_name",
            "_decode_record_sha256",
            "_canonical_source_path",
            "_torchdata_source_identity",
            "_torchdata_runtime_identity",
            "_capture_train_loader_state",
        ],
        namespace={
            "CorrectnessAuditError": _FakeCorrectnessAuditError,
            "StatefulDataLoader": _FakeTorchDataLoader,
            "_SUPPORTED_TORCHDATA_VERSION": "0.11.0",
            "_TORCHDATA_DISTRIBUTION": "torchdata",
            "_TORCHDATA_PACKAGE": "torchdata",
            "_TORCHDATA_RUNTIME_MODULE": expected_runtime_module,
            "_TORCHDATA_RUNTIME_PACKAGE_PATH": _TORCHDATA_SOURCE_RECORD,
            "Path": Path,
            "PathLike": os.PathLike,
            "base64": base64,
            "binascii": binascii,
            "hashlib": hashlib,
            "importlib": SimpleNamespace(
                import_module=lambda name: runtime_module,
            ),
            "importlib_metadata": metadata,
        },
    )
    return functions["_capture_train_loader_state"], functions


def test_correctness_audit_requires_exact_torchdata_identity_and_version() -> None:
    capture, _ = _load_train_loader_capture(_FakeImportlibMetadata())
    loader = _FakeTorchDataLoader(pending_state={"position": 3})
    pending_state = loader.next_iter_state
    source_sha256 = hashlib.sha256(_FAKE_TORCHDATA_SOURCE.read_bytes()).hexdigest()

    captured = capture(loader)

    assert captured == {
        "boundary": "pending",
        "initial_iter_for_state_dict": False,
        "loader_class": (
            f"{_FakeTorchDataLoader.__module__}.{_FakeTorchDataLoader.__qualname__}"
        ),
        "package": "torchdata",
        "package_version": "0.11.0",
        "source_origin": str(_FAKE_TORCHDATA_SOURCE),
        "source_sha256": source_sha256,
        "state": {"position": 3},
    }
    assert loader.state_dict_calls == 0
    assert loader._iterator is None
    assert loader.next_iter_state is pending_state

    not_started_loader = _FakeTorchDataLoader()
    assert capture(not_started_loader) == {
        "boundary": "not_started",
        "initial_iter_for_state_dict": False,
        "loader_class": (
            f"{_FakeTorchDataLoader.__module__}.{_FakeTorchDataLoader.__qualname__}"
        ),
        "package": "torchdata",
        "package_version": "0.11.0",
        "source_origin": str(_FAKE_TORCHDATA_SOURCE),
        "source_sha256": source_sha256,
        "state": None,
    }
    assert not_started_loader.state_dict_calls == 0

    active_loader = _FakeTorchDataLoader(iterator=_FakeTorchDataIterator())
    assert capture(active_loader) == {
        "boundary": "active",
        "initial_iter_for_state_dict": False,
        "loader_class": (
            f"{_FakeTorchDataLoader.__module__}.{_FakeTorchDataLoader.__qualname__}"
        ),
        "package": "torchdata",
        "package_version": "0.11.0",
        "source_origin": str(_FAKE_TORCHDATA_SOURCE),
        "source_sha256": source_sha256,
        "state": {"position": 5},
    }
    assert active_loader.state_dict_calls == 1

    for metadata in (
        _MissingTorchDataMetadata(),
        _FakeImportlibMetadata(version="0.12.0"),
        _FakeImportlibMetadata(distribution_name="not-torchdata"),
        _FakeImportlibMetadata(package_owners=("not-torchdata",)),
    ):
        capture, _ = _load_train_loader_capture(metadata)
        ambiguous_loader = _FakeTorchDataLoader(pending_state={"position": 3})
        with pytest.raises(_FakeCorrectnessAuditError):
            capture(ambiguous_loader)
        assert ambiguous_loader.state_dict_calls == 0

    for runtime_loader_class, runtime_iterator_class in (
        (object, _FakeTorchDataIterator),
        (_FakeTorchDataLoader, None),
    ):
        capture, _ = _load_train_loader_capture(
            _FakeImportlibMetadata(),
            runtime_loader_class=runtime_loader_class,
            runtime_iterator_class=runtime_iterator_class,
        )
        ambiguous_loader = _FakeTorchDataLoader(pending_state={"position": 3})
        with pytest.raises(_FakeCorrectnessAuditError):
            capture(ambiguous_loader)
        assert ambiguous_loader.state_dict_calls == 0

    class TorchDataSubclass(_FakeTorchDataLoader):
        pass

    capture, _ = _load_train_loader_capture(_FakeImportlibMetadata())
    subclass_loader = TorchDataSubclass(pending_state={"position": 3})
    with pytest.raises(_FakeCorrectnessAuditError):
        capture(subclass_loader)
    assert subclass_loader.state_dict_calls == 0


@pytest.mark.parametrize(
    "case",
    [
        "missing-files",
        "missing-source-record",
        "duplicate-source-record",
        "missing-record-hash",
        "unsupported-record-hash",
        "padded-record-hash",
        "unresolvable-located-path",
        "locate-origin-mismatch",
        "origin-file-mismatch",
        "missing-origin",
        "missing-file",
        "content-hash-mismatch",
        "public-loader-module-mismatch",
        "private-iterator-module-mismatch",
    ],
)
def test_correctness_audit_rejects_unproven_torchdata_source_before_state_dict(
    case: str,
    tmp_path: Path,
) -> None:
    metadata = _FakeImportlibMetadata()
    runtime_origin: object = _DEFAULT_RUNTIME_PATH
    runtime_file: object = _DEFAULT_RUNTIME_PATH
    expected_runtime_module = _FakeTorchDataLoader.__module__
    runtime_iterator_class: object = _FakeTorchDataIterator
    alternate_source = tmp_path / "shadow_stateful_dataloader.py"
    alternate_source.write_text("shadow source")

    if case == "missing-files":
        metadata.files = None
    elif case == "missing-source-record":
        metadata.files = [
            _FakePackagePath(
                "torchdata/stateful_dataloader/other.py", metadata.source_path
            )
        ]
    elif case == "duplicate-source-record":
        metadata.files = [
            _FakePackagePath(_TORCHDATA_SOURCE_RECORD, metadata.source_path),
            _FakePackagePath(_TORCHDATA_SOURCE_RECORD, metadata.source_path),
        ]
    elif case == "missing-record-hash":
        package_path = _FakePackagePath(_TORCHDATA_SOURCE_RECORD, metadata.source_path)
        package_path.hash = None
        metadata.files = [package_path]
    elif case == "unsupported-record-hash":
        metadata.files = [
            _FakePackagePath(
                _TORCHDATA_SOURCE_RECORD,
                metadata.source_path,
                hash_mode="sha512",
            )
        ]
    elif case == "padded-record-hash":
        metadata.files = [
            _FakePackagePath(
                _TORCHDATA_SOURCE_RECORD,
                metadata.source_path,
                hash_value=_record_sha256(metadata.source_path) + "=",
            )
        ]
    elif case == "unresolvable-located-path":
        metadata.files = [
            _FakePackagePath(
                _TORCHDATA_SOURCE_RECORD,
                tmp_path / "missing.py",
                hash_value=_record_sha256(metadata.source_path),
            )
        ]
    elif case == "locate-origin-mismatch":
        metadata.files = [_FakePackagePath(_TORCHDATA_SOURCE_RECORD, alternate_source)]
    elif case == "origin-file-mismatch":
        runtime_origin = str(alternate_source)
    elif case == "missing-origin":
        runtime_origin = None
    elif case == "missing-file":
        runtime_file = None
    elif case == "content-hash-mismatch":
        wrong_hash = base64.urlsafe_b64encode(b"\x00" * 32).rstrip(b"=").decode()
        metadata.files = [
            _FakePackagePath(
                _TORCHDATA_SOURCE_RECORD,
                metadata.source_path,
                hash_value=wrong_hash,
            )
        ]
    elif case == "public-loader-module-mismatch":
        expected_runtime_module = "shadow.runtime"
    elif case == "private-iterator-module-mismatch":
        wrong_iterator_class = type(
            "WrongIterator",
            (),
            {"__module__": "shadow.runtime"},
        )
        runtime_iterator_class = wrong_iterator_class
    else:
        raise AssertionError(f"Unhandled provenance case {case}")

    capture, _ = _load_train_loader_capture(
        metadata,
        runtime_iterator_class=runtime_iterator_class,
        runtime_origin=runtime_origin,
        runtime_file=runtime_file,
        expected_runtime_module=expected_runtime_module,
    )
    loader = _FakeTorchDataLoader(pending_state={"position": 3})

    with pytest.raises(_FakeCorrectnessAuditError):
        capture(loader)

    assert loader.state_dict_calls == 0


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        pytest.param(
            lambda loader: vars(loader).pop("_iterator"),
            "missing required private fields",
            id="missing-iterator",
        ),
        pytest.param(
            lambda loader: vars(loader).pop("next_iter_state"),
            "missing required private fields",
            id="renamed-pending-state",
        ),
        pytest.param(
            lambda loader: vars(loader).pop("_initial_iter_for_state_dict"),
            "missing required private fields",
            id="missing-initial-flag",
        ),
        pytest.param(
            lambda loader: setattr(loader, "_iterator", _FakeTorchDataIterator()),
            "active iterator cannot have pending state",
            id="active-and-pending",
        ),
        pytest.param(
            lambda loader: (
                setattr(loader, "next_iter_state", None),
                setattr(loader, "_initial_iter_for_state_dict", True),
            ),
            "no iterator cannot have initial-iterator flag",
            id="none-and-initial-true",
        ),
        pytest.param(
            lambda loader: setattr(loader, "_iterator", object()),
            "unexpected iterator type",
            id="bad-iterator-type",
        ),
        pytest.param(
            lambda loader: setattr(loader, "next_iter_state", []),
            "unexpected pending-state type",
            id="bad-pending-type",
        ),
        pytest.param(
            lambda loader: setattr(loader, "next_iter_state", {}),
            "empty pending state",
            id="empty-pending-state",
        ),
        pytest.param(
            lambda loader: setattr(loader, "_initial_iter_for_state_dict", 0),
            "unexpected initial-iterator flag type",
            id="bad-initial-flag-type",
        ),
    ],
)
def test_correctness_audit_rejects_ambiguous_torchdata_layout_before_state_dict(
    mutation: Any,
    expected_error: str,
) -> None:
    capture, _ = _load_train_loader_capture(_FakeImportlibMetadata())
    loader = _FakeTorchDataLoader(pending_state={"position": 3})
    mutation(loader)

    with pytest.raises(_FakeCorrectnessAuditError, match=expected_error):
        capture(loader)

    assert loader.state_dict_calls == 0


def test_correctness_audit_custom_colliding_loader_uses_protocol_state_dict() -> None:
    metadata = _FakeImportlibMetadata()
    capture, _ = _load_train_loader_capture(metadata)

    class CollidingProtocolLoader:
        def __init__(self) -> None:
            self._iterator = None
            self.next_iter_state = {"position": 99}
            self._initial_iter_for_state_dict = True
            self.state_dict_calls = 0

        def state_dict(self) -> dict[str, object]:
            self.state_dict_calls += 1
            return {"position": 4}

    loader = CollidingProtocolLoader()

    assert capture(loader) == {"position": 4}
    assert loader.state_dict_calls == 1
    assert metadata.distribution_calls == 0
    assert metadata.package_owner_calls == 0

    audit_path = REPO_ROOT / "nemo_rl/algorithms/sft_correctness_audit.py"
    capture_node = _function_node(audit_path, "capture_correctness_snapshot")
    assert "_capture_train_loader_state" in ast.unparse(capture_node)


def test_correctness_evidence_survives_validation_exceptions() -> None:
    sft_path = REPO_ROOT / "nemo_rl/algorithms/sft.py"
    sft_tree = ast.parse(sft_path.read_text())
    class_names = {
        node.name for node in sft_tree.body if isinstance(node, ast.ClassDef)
    }
    validation_impl = _function_node(sft_path, "_validate_with_loss_availability_impl")
    validation_wrapper = _function_node(sft_path, "_validate_with_loss_availability")
    impl_source = ast.unparse(validation_impl)
    wrapper_source = ast.unparse(validation_wrapper)

    assert "_ValidationCorrectnessEvidenceCollector" in class_names
    assert "correctness_evidence_collector" in {
        argument.arg for argument in validation_impl.args.kwonlyargs
    }
    assert "capture_after" in impl_source
    assert "finally" in impl_source
    assert "finalize_restoration_boundary" in wrapper_source

    audit_path = REPO_ROOT / "nemo_rl/algorithms/sft_correctness_audit.py"
    audit_method = _function_node(
        audit_path,
        "audit_validation",
        class_name="SFTCorrectnessAuditor",
    )
    audit_source = ast.unparse(audit_method)
    assert "validation_evidence" in {
        argument.arg for argument in audit_method.args.kwonlyargs
    }
    assert "validation_evidence()" in audit_source
