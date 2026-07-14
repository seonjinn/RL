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

import re
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.mcore


class _RecordingUpstreamFullCudaGraphWrapper:
    def __init__(
        self,
        forward_backward_func: Callable[..., Any],
        cuda_graph_warmup_steps: int = 1,
        use_single_mempool: bool = False,
    ) -> None:
        self.forward_backward_func = forward_backward_func
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool
        self.upstream_reset_calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward_backward_func(*args, **kwargs)

    def reset_cuda_graph(self, stage: Optional[str] = None) -> None:
        del stage
        self.upstream_reset_calls += 1


def _wrapper_call_signature() -> Any:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphCallSignature

    return FullCudaGraphCallSignature(
        num_microbatches=1,
        seq_length=4,
        micro_batch_size=1,
        loss_signature="NLLLossFn:v1",
    )


def _call_wrapper(wrapper: Any) -> Any:
    return wrapper(
        model=object(),
        data_iterator=iter(()),
        num_microbatches=1,
        seq_length=4,
        micro_batch_size=1,
        forward_only=False,
        nemo_rl_signature=_wrapper_call_signature(),
    )


def _new_wrapper(
    raw_schedule: Optional[Callable[..., Any]] = None,
    *,
    warmup_steps: int = 1,
) -> Any:
    from nemo_rl.models.megatron.full_cuda_graph import NemoRLFullCudaGraphWrapper

    return NemoRLFullCudaGraphWrapper(
        raw_schedule or (lambda **_kwargs: "ok"),
        cuda_graph_warmup_steps=warmup_steps,
        use_single_mempool=False,
        upstream_wrapper_cls=_RecordingUpstreamFullCudaGraphWrapper,
    )


class _StorageModel(torch.nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value, value + 1]))


class _LeafOptimizer:
    def __init__(self, parameter: torch.Tensor, *, state_value: float = 1.0) -> None:
        self.param_groups = [{"params": [parameter], "lr": 0.1}]
        self.state = {
            parameter: {
                "step": torch.tensor(state_value),
                "nested": {"exp_avg": [torch.full_like(parameter, state_value)]},
            }
        }

    def state_dict(self) -> dict[str, Any]:
        raise AssertionError("storage signatures must inspect live state only")


class _ChainedOptimizer:
    def __init__(self, *children: _LeafOptimizer) -> None:
        self.chained_optimizers = list(children)

    @property
    def optimizer(self) -> Any:
        raise AssertionError("multi-child ChainedOptimizer.optimizer is invalid")


class _TraversalFailureIterator:
    def __init__(self, pointer: int, *, fail_on_iter: bool) -> None:
        self.pointer = pointer
        self.fail_on_iter = fail_on_iter

    def __iter__(self) -> "_TraversalFailureIterator":
        if self.fail_on_iter:
            raise RuntimeError(f"private pointer {self.pointer} / {hex(self.pointer)}")
        return self

    def __next__(self) -> Any:
        raise RuntimeError(f"private pointer {self.pointer} / {hex(self.pointer)}")


class _TraversalFailureList(list[Any]):
    def __init__(self, values: list[Any], pointer: int, *, fail_on_iter: bool) -> None:
        super().__init__(values)
        self.pointer = pointer
        self.fail_on_iter = fail_on_iter

    def __iter__(self) -> Any:
        return iter(
            _TraversalFailureIterator(
                self.pointer,
                fail_on_iter=self.fail_on_iter,
            )
        )


def _assert_sanitized_traversal_failure(
    error: pytest.ExceptionInfo[TypeError],
    *,
    pointer: int,
    field: str,
) -> None:
    message = str(error.value)
    assert re.fullmatch(
        "full-iteration CUDA graph traversal unavailable "
        rf"component_id_sha256=[0-9a-f]{{64}} field={re.escape(field)} "
        "reason=unsupported_traversal",
        message,
    )
    assert str(pointer) not in message
    assert hex(pointer) not in message
    assert "private pointer" not in message
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


def _assert_sanitized_tensor_class_metadata_failure(
    error: pytest.ExceptionInfo[TypeError],
    *,
    pointer: int,
) -> None:
    message = str(error.value)
    assert re.fullmatch(
        "full-iteration CUDA graph tensor attribute unavailable "
        r"tensor_id_sha256=[0-9a-f]{64} field=class_metadata "
        "reason=unsupported_tensor_attribute",
        message,
    )
    assert str(pointer) not in message
    assert hex(pointer) not in message
    assert "private pointer" not in message
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


def test_full_cuda_graph_storage_guard_rejects_custom_fsdp_config_before_setup() -> (
    None
):
    from nemo_rl.models.megatron.full_cuda_graph import (
        validate_full_cuda_graph_policy_config,
    )

    config = {
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "generation": {"colocated": {"enabled": False}},
        "megatron_cfg": {
            "cuda_graph_impl": "full_iteration",
            "context_parallel_size": 1,
            "distributed_data_parallel_config": {"use_custom_fsdp": True},
        },
    }

    with pytest.raises(ValueError, match="custom FSDP/DTensor"):
        validate_full_cuda_graph_policy_config(config, init_optimizer=True)


def test_full_cuda_graph_execution_stats_count_capture_as_replay() -> None:
    wrapper = _new_wrapper(warmup_steps=1)

    for _ in range(3):
        assert _call_wrapper(wrapper) == "ok"

    stats = wrapper.execution_stats()
    assert stats.warmup_calls == 1
    assert stats.capture_calls == 1
    assert stats.replay_calls == 2
    assert stats.reset_calls == 0


def test_full_cuda_graph_execution_stats_are_instance_local() -> None:
    first = _new_wrapper(warmup_steps=1)
    second = _new_wrapper(warmup_steps=1)

    _call_wrapper(first)
    _call_wrapper(first)
    _call_wrapper(second)

    assert first.execution_stats().warmup_calls == 1
    assert first.execution_stats().capture_calls == 1
    assert first.execution_stats().replay_calls == 1
    assert second.execution_stats().warmup_calls == 1
    assert second.execution_stats().capture_calls == 0
    assert second.execution_stats().replay_calls == 0


@pytest.mark.parametrize("successful_calls", [0, 1, 2])
def test_full_cuda_graph_execution_stats_ignore_upstream_exceptions(
    successful_calls: int,
) -> None:
    calls = 0

    def schedule(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == successful_calls + 1:
            raise RuntimeError("upstream failed")
        return "ok"

    wrapper = _new_wrapper(schedule, warmup_steps=1)
    for _ in range(successful_calls):
        _call_wrapper(wrapper)
    before = wrapper.execution_stats()

    with pytest.raises(RuntimeError, match="upstream failed"):
        _call_wrapper(wrapper)

    assert wrapper.execution_stats() == before


def test_full_cuda_graph_execution_stats_count_only_explicit_reset() -> None:
    wrapper = _new_wrapper(warmup_steps=1)
    assert wrapper.upstream_reset_calls == 1
    assert wrapper.execution_stats().reset_calls == 0

    _call_wrapper(wrapper)
    wrapper.reset_cuda_graph(stage="training")

    assert wrapper.upstream_reset_calls == 2
    assert wrapper.execution_stats().reset_calls == 1
    assert wrapper.execution_stats().warmup_calls == 1
    assert wrapper.will_capture_next_call() is False


@pytest.mark.parametrize("warmup_steps", [0, -1])
def test_full_cuda_graph_execution_stats_reject_invalid_warmup_steps(
    warmup_steps: int,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import NemoRLFullCudaGraphWrapper

    with pytest.raises(ValueError, match="warmup steps must be at least 1"):
        NemoRLFullCudaGraphWrapper(
            lambda **_kwargs: None,
            cuda_graph_warmup_steps=warmup_steps,
            use_single_mempool=False,
            upstream_wrapper_cls=_RecordingUpstreamFullCudaGraphWrapper,
        )


def test_full_cuda_graph_storage_guard_rejects_parameter_reallocation() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    signature = FullCudaGraphStorageSignature.capture(model, optimizer)
    model.weight = torch.nn.Parameter(model.weight.detach().clone())

    with pytest.raises(
        RuntimeError,
        match=r"tensor_id_sha256=[0-9a-f]{64} field=storage_data_ptr",
    ):
        signature.require_match(model, optimizer)


@pytest.mark.parametrize("gradient_name", ["grad", "main_grad"])
def test_full_cuda_graph_storage_guard_rejects_gradient_reallocation(
    gradient_name: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    setattr(model.weight, gradient_name, torch.ones_like(model.weight))
    optimizer = _LeafOptimizer(model.weight)
    signature = FullCudaGraphStorageSignature.capture(model, optimizer)
    setattr(model.weight, gradient_name, torch.zeros_like(model.weight))

    with pytest.raises(
        RuntimeError,
        match=r"tensor_id_sha256=[0-9a-f]{64} field=storage_data_ptr",
    ):
        signature.require_match(model, optimizer)


def test_full_cuda_graph_storage_guard_rejects_optimizer_state_reallocation() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    signature = FullCudaGraphStorageSignature.capture(model, optimizer)
    optimizer.state[model.weight]["nested"]["exp_avg"][0] = torch.zeros_like(
        model.weight
    )

    with pytest.raises(
        RuntimeError,
        match=r"tensor_id_sha256=[0-9a-f]{64} field=storage_data_ptr",
    ):
        signature.require_match(model, optimizer)


def test_full_cuda_graph_storage_guard_detects_flat_and_nested_mapping_drift() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    nested_state = torch.ones_like(model.weight)
    flat_state = torch.full_like(model.weight, 2.0)
    optimizer.state = {"a": {"b": nested_state}, "a.b": flat_state}
    signature = FullCudaGraphStorageSignature.capture(model, optimizer)
    optimizer.state["a"]["b"] = torch.zeros_like(nested_state)

    with pytest.raises(RuntimeError, match="storage signature mismatch"):
        signature.require_match(model, optimizer)


def test_full_cuda_graph_storage_guard_rejects_duplicate_logical_tensor_names() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    class DuplicateNameModel:
        def named_parameters(self) -> Any:
            duplicated_parameter = torch.ones(1)
            return iter(
                (
                    ("weight", duplicated_parameter),
                    ("weight", duplicated_parameter),
                )
            )

    with pytest.raises(TypeError, match="duplicate logical tensor name"):
        FullCudaGraphStorageSignature.capture(
            DuplicateNameModel(), SimpleNamespace(param_groups=[], state={})
        )


def test_full_cuda_graph_storage_guard_traverses_chained_optimizer_nested_state() -> (
    None
):
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    second_parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    first = _LeafOptimizer(model.weight, state_value=1.0)
    second = _LeafOptimizer(second_parameter, state_value=2.0)
    optimizer = _ChainedOptimizer(first, second)

    signature = FullCudaGraphStorageSignature.capture(model, optimizer)
    names = tuple(entry.name for entry in signature._entries)

    assert any(name.startswith("optimizer_leaf[0]") for name in names)
    assert any(name.startswith("optimizer_leaf[1]") for name in names)
    assert all("key_sha256=" in name for name in names if ".state." in name)
    assert first.state_dict  # prove the forbidden method exists without calling it
    second.state[second_parameter]["nested"]["exp_avg"][0] = torch.zeros_like(
        second_parameter
    )
    with pytest.raises(
        RuntimeError,
        match=r"tensor_id_sha256=[0-9a-f]{64} field=storage_data_ptr",
    ):
        signature.require_match(model, optimizer)


def test_full_cuda_graph_storage_guard_handles_single_list_and_tuple_model_chunks() -> (
    None
):
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    first = _StorageModel(1.0)
    second = _StorageModel(3.0)
    optimizer = _LeafOptimizer(first.weight)

    single = FullCudaGraphStorageSignature.capture(first, optimizer)
    listed = FullCudaGraphStorageSignature.capture([first, second], optimizer)
    tupled = FullCudaGraphStorageSignature.capture((first, second), optimizer)

    assert any(entry.name.startswith("model_chunk[0]") for entry in single._entries)
    assert listed.digest() == tupled.digest()
    assert any(entry.name.startswith("model_chunk[1]") for entry in listed._entries)
    second.weight = torch.nn.Parameter(second.weight.detach().clone())
    with pytest.raises(
        RuntimeError,
        match=r"tensor_id_sha256=[0-9a-f]{64} field=storage_data_ptr",
    ):
        listed.require_match([first, second], optimizer)


@pytest.mark.parametrize("drift", ["offset", "stride"])
def test_full_cuda_graph_storage_guard_rejects_view_address_drift(drift: str) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    base = torch.arange(8.0)
    model = _StorageModel()
    if drift == "offset":
        original = base[:4].reshape(2, 2)
        changed = base[1:5].reshape(2, 2)
    else:
        original = base[:4].as_strided((2, 2), (2, 1))
        changed = base[:4].as_strided((2, 2), (1, 2))
    model.weight = torch.nn.Parameter(original)
    optimizer = _LeafOptimizer(model.weight)
    signature = FullCudaGraphStorageSignature.capture(model, optimizer)
    model.weight = torch.nn.Parameter(changed)

    with pytest.raises(RuntimeError, match=rf"field={drift}|field=effective_data_ptr"):
        signature.require_match(model, optimizer)


def test_full_cuda_graph_storage_signature_digest_changes_after_address_or_view_drift() -> (
    None
):
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    base = torch.arange(5.0)
    model = _StorageModel()
    model.weight = torch.nn.Parameter(base[:4])
    optimizer = _LeafOptimizer(model.weight)
    first = FullCudaGraphStorageSignature.capture(model, optimizer)
    model.weight = torch.nn.Parameter(base[1:5])
    second = FullCudaGraphStorageSignature.capture(model, optimizer)

    assert re.fullmatch(r"[0-9a-f]{64}", first.digest())
    assert first.digest() != second.digest()

    with pytest.raises(FrozenInstanceError):
        first._digest = "0" * 64


def test_full_cuda_graph_storage_guard_sanitizes_pointer_values() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    original_pointer = model.weight.data_ptr()
    signature = FullCudaGraphStorageSignature.capture(model, optimizer)
    model.weight = torch.nn.Parameter(model.weight.detach().clone())
    changed_pointer = model.weight.data_ptr()

    with pytest.raises(RuntimeError) as error:
        signature.require_match(model, optimizer)

    message = str(error.value)
    assert str(original_pointer) not in message
    assert str(changed_pointer) not in message
    assert hex(original_pointer) not in message
    assert hex(changed_pointer) not in message
    assert "tensor=" not in message
    assert len(re.findall(r"[0-9a-f]{64}", message)) == 3


@pytest.mark.parametrize("key_kind", ["integer", "string", "tuple"])
def test_full_cuda_graph_storage_guard_sanitizes_pointer_like_mapping_keys(
    key_kind: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    pointer = model.weight.data_ptr()
    pointer_keys = {
        "integer": pointer,
        "string": f"{pointer}:{hex(pointer)}",
        "tuple": (pointer, hex(pointer)),
    }
    pointer_key = pointer_keys[key_kind]
    optimizer.state = {pointer_key: torch.ones_like(model.weight)}
    signature = FullCudaGraphStorageSignature.capture(model, optimizer)
    optimizer.state[pointer_key] = torch.zeros_like(model.weight)

    with pytest.raises(RuntimeError) as error:
        signature.require_match(model, optimizer)

    message = str(error.value)
    assert str(pointer) not in message
    assert hex(pointer) not in message
    assert "tensor=" not in message
    assert re.search(r"tensor_id_sha256=[0-9a-f]{64}", message)


def test_full_cuda_graph_storage_guard_rejects_dtensor_and_custom_fsdp() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    class DTensor(torch.Tensor):
        pass

    distributed_parameter = torch.Tensor._make_subclass(
        DTensor, torch.ones(2), require_grad=False
    )

    class TensorModel:
        def named_parameters(self) -> Any:
            return iter((("weight", distributed_parameter),))

    with pytest.raises(TypeError, match="DTensor"):
        FullCudaGraphStorageSignature.capture(
            TensorModel(), _LeafOptimizer(distributed_parameter)
        )

    class FullyShardedDataParallel:
        def named_parameters(self) -> Any:
            raise AssertionError("custom FSDP must be rejected before traversal")

    with pytest.raises(TypeError, match="custom FSDP"):
        FullCudaGraphStorageSignature.capture(
            FullyShardedDataParallel(), SimpleNamespace(param_groups=[], state={})
        )


def test_full_cuda_graph_storage_guard_rejects_unsupported_tensor_storage() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    class UnsupportedStorageTensor(torch.Tensor):
        def untyped_storage(self) -> Any:
            raise RuntimeError(f"private pointer {super().data_ptr()}")

    unsupported = torch.Tensor._make_subclass(
        UnsupportedStorageTensor, torch.ones(2), require_grad=False
    )

    class TensorModel:
        def named_parameters(self) -> Any:
            return iter((("weight", unsupported),))

    with pytest.raises(TypeError, match="field=storage_data_ptr") as error:
        FullCudaGraphStorageSignature.capture(
            TensorModel(), SimpleNamespace(param_groups=[], state={})
        )

    assert str(unsupported.data_ptr()) not in str(error.value)
    assert hex(unsupported.data_ptr()) not in str(error.value)


@pytest.mark.parametrize("attribute", ["__fsdp_param__", "grad", "main_grad"])
def test_full_cuda_graph_storage_guard_sanitizes_tensor_attribute_failures(
    attribute: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    class AttributeFailureTensor(torch.Tensor):
        def __getattribute__(self, name: str) -> Any:
            if name == attribute:
                pointer = super().__getattribute__("data_ptr")()
                raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")
            return super().__getattribute__(name)

    unsupported = torch.Tensor._make_subclass(
        AttributeFailureTensor, torch.ones(2), require_grad=False
    )

    class TensorModel:
        def named_parameters(self) -> Any:
            return iter((("weight", unsupported),))

    with pytest.raises(
        TypeError,
        match=rf"field={re.escape(attribute)} reason=unsupported_tensor_attribute",
    ) as error:
        FullCudaGraphStorageSignature.capture(
            TensorModel(), SimpleNamespace(param_groups=[], state={})
        )

    message = str(error.value)
    assert str(unsupported.data_ptr()) not in message
    assert hex(unsupported.data_ptr()) not in message
    assert re.search(r"tensor_id_sha256=[0-9a-f]{64}", message)


@pytest.mark.parametrize("class_field", ["__name__", "__module__"])
def test_full_cuda_graph_storage_guard_sanitizes_model_class_metadata_failures(
    class_field: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    pointer = model.weight.data_ptr()
    blocked_field: Optional[str] = None

    class MetadataFailureMeta(type):
        def __getattribute__(cls, name: str) -> Any:
            if name == blocked_field:
                raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")
            return super().__getattribute__(name)

    class MetadataFailureModel(metaclass=MetadataFailureMeta):
        def named_parameters(self) -> Any:
            return iter((("weight", model.weight),))

    blocked_field = class_field

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(
            MetadataFailureModel(),
            SimpleNamespace(param_groups=[], state={}),
        )

    _assert_sanitized_traversal_failure(
        error,
        pointer=pointer,
        field="class_metadata",
    )


@pytest.mark.parametrize("class_field", ["__name__", "__module__"])
def test_full_cuda_graph_storage_guard_sanitizes_tensor_class_metadata_failures(
    class_field: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    blocked_field: Optional[str] = None
    pointer = 0

    class MetadataFailureTensorMeta(type(torch.Tensor)):
        def __getattribute__(cls, name: str) -> Any:
            if name == blocked_field:
                raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")
            return super().__getattribute__(name)

    class MetadataFailureTensor(torch.Tensor, metaclass=MetadataFailureTensorMeta):
        pass

    tensor = torch.Tensor._make_subclass(
        MetadataFailureTensor,
        torch.ones(2),
        require_grad=False,
    )
    pointer = tensor.data_ptr()
    blocked_field = class_field

    class TensorModel:
        def named_parameters(self) -> Any:
            return iter((("weight", tensor),))

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(
            TensorModel(),
            SimpleNamespace(param_groups=[], state={}),
        )

    _assert_sanitized_tensor_class_metadata_failure(error, pointer=pointer)


def test_full_cuda_graph_storage_guard_rejects_chained_optimizer_cycle() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()

    class CyclicChainedOptimizer:
        def __init__(self) -> None:
            self.chained_optimizers = [self]

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(model, CyclicChainedOptimizer())

    _assert_sanitized_traversal_failure(
        error,
        pointer=model.weight.data_ptr(),
        field="chained_optimizers",
    )


def test_full_cuda_graph_storage_guard_rejects_optimizer_mapping_cycle() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    cyclic_state: dict[str, Any] = {}
    cyclic_state["self"] = cyclic_state
    optimizer.state = cyclic_state

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(model, optimizer)

    _assert_sanitized_traversal_failure(
        error,
        pointer=model.weight.data_ptr(),
        field="optimizer_container",
    )


@pytest.mark.parametrize("container_kind", ["list", "tuple"])
def test_full_cuda_graph_storage_guard_rejects_optimizer_sequence_cycle(
    container_kind: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    cyclic_list: list[Any] = []
    if container_kind == "list":
        cyclic_value: Any = cyclic_list
    else:
        cyclic_value = (cyclic_list,)
    cyclic_list.append(cyclic_value)
    optimizer.state = {"cycle": cyclic_value}

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(model, optimizer)

    _assert_sanitized_traversal_failure(
        error,
        pointer=model.weight.data_ptr(),
        field="optimizer_container",
    )


def test_full_cuda_graph_storage_guard_allows_shared_acyclic_optimizer_paths() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    leaf = _LeafOptimizer(model.weight)
    shared_tensor = torch.ones_like(model.weight)
    shared_sequence = [shared_tensor]
    leaf.state = {"first": shared_sequence, "second": shared_sequence}
    optimizer = _ChainedOptimizer(leaf, leaf)

    signature = FullCudaGraphStorageSignature.capture(model, optimizer)
    state_entries = [entry for entry in signature._entries if ".state." in entry.name]

    assert len(state_entries) == 1
    signature.require_match(model, optimizer)


@pytest.mark.parametrize("phase", ["access", "iter", "next"])
def test_full_cuda_graph_storage_guard_sanitizes_chained_optimizer_traversal(
    phase: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    pointer = model.weight.data_ptr()

    class FailingChainedOptimizer:
        def __getattribute__(self, name: str) -> Any:
            if name == "chained_optimizers" and phase == "access":
                raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")
            return super().__getattribute__(name)

        @property
        def chained_optimizers(self) -> Any:
            return _TraversalFailureList(
                [_LeafOptimizer(model.weight)],
                pointer,
                fail_on_iter=phase == "iter",
            )

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(model, FailingChainedOptimizer())

    _assert_sanitized_traversal_failure(
        error,
        pointer=pointer,
        field="chained_optimizers",
    )


@pytest.mark.parametrize("phase", ["access", "call", "iter", "next"])
def test_full_cuda_graph_storage_guard_sanitizes_named_parameter_traversal(
    phase: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    pointer = model.weight.data_ptr()

    class FailingNamedParametersModel:
        def __getattribute__(self, name: str) -> Any:
            if name == "named_parameters" and phase == "access":
                raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")
            return super().__getattribute__(name)

        def named_parameters(self) -> Any:
            if phase == "call":
                raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")
            return _TraversalFailureIterator(pointer, fail_on_iter=phase == "iter")

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(
            FailingNamedParametersModel(),
            SimpleNamespace(param_groups=[], state={}),
        )

    _assert_sanitized_traversal_failure(
        error,
        pointer=pointer,
        field="named_parameters",
    )


@pytest.mark.parametrize("phase", ["access", "call", "iter", "next"])
def test_full_cuda_graph_storage_guard_sanitizes_mapping_items_traversal(
    phase: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    pointer = model.weight.data_ptr()

    class FailingItemsMapping(dict[Any, Any]):
        def __getattribute__(self, name: str) -> Any:
            if name == "items" and phase == "access":
                raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")
            return super().__getattribute__(name)

        def items(self) -> Any:
            if phase == "call":
                raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")
            return _TraversalFailureIterator(pointer, fail_on_iter=phase == "iter")

    optimizer.state = FailingItemsMapping()

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(model, optimizer)

    _assert_sanitized_traversal_failure(
        error,
        pointer=pointer,
        field="mapping_items",
    )


@pytest.mark.parametrize("phase", ["access", "call"])
def test_full_cuda_graph_storage_guard_sanitizes_parameter_group_get(
    phase: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    pointer = model.weight.data_ptr()

    class FailingGetMapping(dict[Any, Any]):
        def __getattribute__(self, name: str) -> Any:
            if name == "get" and phase == "access":
                raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")
            return super().__getattribute__(name)

        def get(self, key: Any, default: Any = None) -> Any:
            del key, default
            raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")

    optimizer.param_groups = [FailingGetMapping()]

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(model, optimizer)

    _assert_sanitized_traversal_failure(
        error,
        pointer=pointer,
        field="params",
    )


def test_full_cuda_graph_storage_guard_rejects_parameter_name_str_subclass() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    pointer = model.weight.data_ptr()

    class FailingParameterName(str):
        def __format__(self, format_spec: str) -> str:
            del format_spec
            raise RuntimeError(f"private pointer {pointer} / {hex(pointer)}")

    class FailingParameterNameModel:
        def named_parameters(self) -> Any:
            return iter(((FailingParameterName("weight"), model.weight),))

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(
            FailingParameterNameModel(),
            SimpleNamespace(param_groups=[], state={}),
        )

    _assert_sanitized_traversal_failure(
        error,
        pointer=pointer,
        field="parameter_name",
    )


@pytest.mark.parametrize(
    ("surface", "field"),
    [
        ("model_chunks", "model_chunks"),
        ("param_groups", "param_groups"),
        ("params", "params"),
        ("optimizer_sequence", "optimizer_sequence"),
    ],
)
def test_full_cuda_graph_storage_guard_sanitizes_sequence_traversal(
    surface: str,
    field: str,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    pointer = model.weight.data_ptr()

    if surface == "model_chunks":
        captured_model: Any = _TraversalFailureList([model], pointer, fail_on_iter=True)
    else:
        captured_model = model
        if surface == "param_groups":
            optimizer.param_groups = _TraversalFailureList(
                optimizer.param_groups, pointer, fail_on_iter=True
            )
        elif surface == "params":
            optimizer.param_groups[0]["params"] = _TraversalFailureList(
                [model.weight], pointer, fail_on_iter=True
            )
        else:
            optimizer.state = {
                "nested": _TraversalFailureList(
                    [torch.ones_like(model.weight)], pointer, fail_on_iter=True
                )
            }

    with pytest.raises(TypeError) as error:
        FullCudaGraphStorageSignature.capture(captured_model, optimizer)

    _assert_sanitized_traversal_failure(
        error,
        pointer=pointer,
        field=field,
    )


def test_full_cuda_graph_storage_guard_rejects_nondeterministic_state_container() -> (
    None
):
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    optimizer.state[model.weight]["unsupported"] = {"unordered"}

    with pytest.raises(TypeError, match="nondeterministic optimizer container"):
        FullCudaGraphStorageSignature.capture(model, optimizer)


def test_full_cuda_graph_storage_guard_distinguishes_structured_mapping_keys() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import FullCudaGraphStorageSignature

    model = _StorageModel()
    optimizer = _LeafOptimizer(model.weight)
    first_key = "tuple=a"
    second_key = ("a",)
    optimizer.state = {
        first_key: torch.tensor(1.0),
        second_key: torch.tensor(2.0),
    }
    signature = FullCudaGraphStorageSignature.capture(model, optimizer)
    state_names = [
        entry.name for entry in signature._entries if ".state." in entry.name
    ]

    assert len(state_names) == 2
    assert len(set(state_names)) == 2
    assert all("key_sha256=" in name for name in state_names)
    optimizer.state[second_key] = torch.tensor(3.0)

    with pytest.raises(RuntimeError, match="storage signature mismatch"):
        signature.require_match(model, optimizer)


def test_full_cuda_graph_storage_guard_failed_warmup_cannot_enter_capture() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    model = _StorageModel()
    wrapper = _new_wrapper(warmup_steps=1)
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker._full_cuda_graph_enabled = True
    worker._full_cuda_graph_wrapper = wrapper
    worker._full_cuda_graph_storage_signature = None
    worker.model = model
    worker.optimizer = _LeafOptimizer(model.weight)

    _call_wrapper(wrapper)
    worker._capture_full_cuda_graph_storage_after_update(update_successful=False)

    with pytest.raises(RuntimeError, match="successful optimizer step"):
        worker._validate_full_cuda_graph_storage_before_schedule()


def test_full_cuda_graph_storage_guard_reset_does_not_authorize_recapture() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    model = _StorageModel()
    wrapper = _new_wrapper(warmup_steps=1)
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker._full_cuda_graph_enabled = True
    worker._full_cuda_graph_wrapper = wrapper
    worker._full_cuda_graph_storage_signature = None
    worker.model = model
    worker.optimizer = _LeafOptimizer(model.weight)

    worker._capture_full_cuda_graph_storage_after_update(update_successful=True)
    captured = worker._full_cuda_graph_storage_signature
    wrapper.reset_cuda_graph(stage="training")
    model.weight = torch.nn.Parameter(model.weight.detach().clone())
    worker._capture_full_cuda_graph_storage_after_update(update_successful=True)

    assert worker._full_cuda_graph_storage_signature is captured
    with pytest.raises(RuntimeError, match="storage signature mismatch"):
        worker._validate_full_cuda_graph_storage_before_schedule()


def test_full_cuda_graph_worker_stats_emit_digest_without_pointer_values() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    model = _StorageModel()
    wrapper = _new_wrapper(warmup_steps=1)
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker._full_cuda_graph_enabled = True
    worker._full_cuda_graph_wrapper = wrapper
    worker._full_cuda_graph_storage_signature = None
    worker.model = model
    worker.optimizer = _LeafOptimizer(model.weight)
    worker._capture_full_cuda_graph_storage_after_update(update_successful=True)
    for _ in range(3):
        _call_wrapper(wrapper)

    metrics: dict[str, Any] = {}
    worker._add_full_cuda_graph_execution_metrics(metrics)

    assert metrics["full_cuda_graph_warmup_calls"] == 1
    assert metrics["full_cuda_graph_capture_calls"] == 1
    assert metrics["full_cuda_graph_replay_calls"] == 2
    assert metrics["full_cuda_graph_reset_calls"] == 0
    assert re.fullmatch(
        r"[0-9a-f]{64}", metrics["full_cuda_graph_storage_signature_sha256"]
    )
    rendered = repr(metrics)
    assert str(model.weight.data_ptr()) not in rendered
    assert hex(model.weight.data_ptr()) not in rendered


def test_full_graph_forward_step_preserves_a2a_schedule_plan() -> None:
    from megatron.core.packed_seq_params import PackedSeqParams

    from nemo_rl.algorithms.loss import NLLLossFn
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron.data import ProcessedMicrobatch
    from nemo_rl.models.megatron.full_cuda_graph import (
        FULL_CUDA_GRAPH_GLOBAL_VALID_SEQS,
        FULL_CUDA_GRAPH_GLOBAL_VALID_TOKS,
        ProcessedMicrobatchStaticBufferLoader,
    )
    from nemo_rl.models.megatron.train import (
        LossPostProcessor,
        megatron_forward_backward,
    )

    observed_valid_seqs: Optional[torch.Tensor] = None
    observed_valid_toks: Optional[torch.Tensor] = None
    static_microbatch: Optional[ProcessedMicrobatch] = None

    class RecordingLossPostProcessor(LossPostProcessor):
        def __call__(
            self,
            data_dict: BatchedDataDict[Any],
            packed_seq_params: Optional[PackedSeqParams] = None,
            global_valid_seqs: Optional[torch.Tensor] = None,
            global_valid_toks: Optional[torch.Tensor] = None,
        ) -> Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, Any]]]:
            nonlocal observed_valid_seqs, observed_valid_toks
            observed_valid_seqs = global_valid_seqs
            observed_valid_toks = global_valid_toks
            return lambda output_tensor: (output_tensor.new_zeros(()), {})

    input_ids = torch.tensor([[1, 2, 3]])
    microbatch = ProcessedMicrobatch(
        data_dict=BatchedDataDict(
            {
                "input_ids": input_ids,
                "token_mask": torch.ones_like(input_ids),
                "sample_mask": torch.ones(1),
            }
        ),
        input_ids=input_ids,
        input_ids_cp_sharded=input_ids,
        attention_mask=torch.ones(1, 3),
        position_ids=torch.tensor([[0, 1, 2]]),
        packed_seq_params=None,
        cu_seqlens_padded=None,
    )
    model = MagicMock()
    schedule_plan = MagicMock()
    model.build_schedule_plan.return_value = schedule_plan
    static_loader = ProcessedMicrobatchStaticBufferLoader()

    def fake_raw_schedule(
        *,
        forward_step_func: Callable[..., tuple[Any, Callable[..., Any]]],
        data_iterator: Any,
        model: Any,
        **_: Any,
    ) -> Any:
        nonlocal static_microbatch
        attached_microbatch = next(data_iterator)
        static_microbatch = static_loader(attached_microbatch, "training", 0)
        output, _ = forward_step_func(
            iter([static_microbatch]),
            model,
            return_schedule_plan=True,
        )
        return output

    output = megatron_forward_backward(
        model=model,
        data_iterator=iter([microbatch]),
        num_microbatches=1,
        seq_length=3,
        mbs=1,
        post_processing_fn=RecordingLossPostProcessor(
            loss_fn=NLLLossFn(),
            cfg={"sequence_packing": {"enabled": False}},
        ),
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(2.0),
        forward_backward_func=fake_raw_schedule,
    )

    assert static_microbatch is not None
    assert output is schedule_plan
    model.build_schedule_plan.assert_called_once_with(
        input_ids=static_microbatch.input_ids_cp_sharded,
        position_ids=static_microbatch.position_ids,
        attention_mask=static_microbatch.attention_mask,
    )
    assert (
        observed_valid_seqs
        is static_microbatch.data_dict[FULL_CUDA_GRAPH_GLOBAL_VALID_SEQS]
    )
    assert (
        observed_valid_toks
        is static_microbatch.data_dict[FULL_CUDA_GRAPH_GLOBAL_VALID_TOKS]
    )
