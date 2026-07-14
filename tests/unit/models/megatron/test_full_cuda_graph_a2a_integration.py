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

import pickle
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
            "expert_model_parallel_size": 1,
            "distributed_data_parallel_config": {"use_custom_fsdp": True},
        },
    }

    with pytest.raises(ValueError, match="custom FSDP/DTensor"):
        validate_full_cuda_graph_policy_config(config, init_optimizer=True)


def _full_cuda_graph_moe_policy_config() -> dict[str, Any]:
    return {
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "generation": {"backend": "vllm", "colocated": {"enabled": False}},
        "megatron_cfg": {
            "cuda_graph_impl": "full_iteration",
            "cuda_graph_modules": [],
            "context_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 4,
            "distributed_data_parallel_config": {"use_custom_fsdp": False},
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "hybridep",
            "moe_grouped_gemm": True,
            "moe_expert_rank_capacity_factor": 1.5,
            "moe_paged_stash": True,
            "use_transformer_engine_op_fuser": True,
            "moe_mlp_glu_interleave_size": 32,
            "moe_hybridep_num_sms_preprocessing": 32,
            "offload_modules": [],
            "env_vars": {"NVTE_CUTEDSL_FUSED_GROUPED_MLP": "1"},
            "fp8_cfg": {"enabled": True, "fp8_recipe": "mxfp8"},
        },
    }


def test_full_cuda_graph_moe_rejects_host_synchronized_alltoall_dispatcher() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        validate_full_cuda_graph_policy_config,
    )

    config = _full_cuda_graph_moe_policy_config()
    config["megatron_cfg"]["moe_token_dispatcher_type"] = "alltoall"
    config["megatron_cfg"]["moe_flex_dispatcher_backend"] = None

    with pytest.raises(ValueError, match="flex/HybridEP"):
        validate_full_cuda_graph_policy_config(config, init_optimizer=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("moe_expert_rank_capacity_factor", None, "static routed-token budget"),
        ("moe_grouped_gemm", False, "grouped GEMM"),
        ("expert_tensor_parallel_size", 2, "expert tensor parallel size 1"),
        ("cuda_graph_modules", ["moe"], "cuda_graph_modules must be empty"),
        ("moe_paged_stash", False, "paged stash"),
        ("use_transformer_engine_op_fuser", False, "TE operation fuser"),
        ("moe_mlp_glu_interleave_size", None, "GLU interleave size 32"),
        ("moe_hybridep_num_sms_preprocessing", None, "preprocessing SM count"),
        ("offload_modules", None, "offload_modules must be a list"),
        ("offload_modules", ["expert_fc1"], "expert_fc1/moe_act offloading"),
        ("env_vars", {}, "NVTE_CUTEDSL_FUSED_GROUPED_MLP=1"),
        ("fp8_cfg", {"enabled": False}, "MXFP8"),
    ],
)
def test_full_cuda_graph_moe_rejects_incomplete_host_free_bundle(
    field: str, value: Any, message: str
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        validate_full_cuda_graph_policy_config,
    )

    config = _full_cuda_graph_moe_policy_config()
    config["megatron_cfg"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_full_cuda_graph_policy_config(config, init_optimizer=True)


def test_full_cuda_graph_moe_accepts_host_free_hybridep_bundle() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        validate_full_cuda_graph_policy_config,
    )

    validate_full_cuda_graph_policy_config(
        _full_cuda_graph_moe_policy_config(), init_optimizer=True
    )


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


def _full_cuda_graph_rank_evidence(
    rank: int,
    digest: str,
    *,
    counters: tuple[int, int, int, int] = (1, 1, 2, 0),
) -> tuple[int, tuple[int, int, int, int], str]:
    return rank, counters, digest


def test_full_cuda_graph_rank_consensus_is_order_independent() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        build_full_cuda_graph_evidence_consensus,
    )

    rank_zero_digest = "a" * 64
    rank_one_digest = "b" * 64
    ascending = [
        _full_cuda_graph_rank_evidence(0, rank_zero_digest),
        _full_cuda_graph_rank_evidence(1, rank_one_digest),
    ]
    descending = list(reversed(ascending))

    first = build_full_cuda_graph_evidence_consensus(ascending, expected_world_size=2)
    second = build_full_cuda_graph_evidence_consensus(descending, expected_world_size=2)

    assert first == second
    assert first == {
        "full_cuda_graph_warmup_calls": 1,
        "full_cuda_graph_capture_calls": 1,
        "full_cuda_graph_replay_calls": 2,
        "full_cuda_graph_reset_calls": 0,
        "full_cuda_graph_storage_signature_sha256": first[
            "full_cuda_graph_storage_signature_sha256"
        ],
    }
    cohort_digest = first["full_cuda_graph_storage_signature_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", cohort_digest)
    assert cohort_digest not in {rank_zero_digest, rank_one_digest}


def test_full_cuda_graph_evidence_envelopes_preserve_all_disabled_baseline() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        build_full_cuda_graph_evidence_envelope_consensus,
    )

    assert (
        build_full_cuda_graph_evidence_envelope_consensus(
            [(False, None), (False, None)],
            expected_world_size=2,
        )
        == {}
    )


def test_full_cuda_graph_evidence_envelopes_reject_mixed_rank_state() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        build_full_cuda_graph_evidence_envelope_consensus,
    )

    with pytest.raises(ValueError, match="enabled state mismatch"):
        build_full_cuda_graph_evidence_envelope_consensus(
            [
                (True, _full_cuda_graph_rank_evidence(0, "a" * 64)),
                (False, None),
            ],
            expected_world_size=2,
        )


@pytest.mark.parametrize(
    "envelopes",
    [
        [None],
        [(False,)],
        [(0, None)],
        [(False, _full_cuda_graph_rank_evidence(0, "a" * 64))],
    ],
)
def test_full_cuda_graph_evidence_envelopes_reject_malformed_shapes(
    envelopes: list[Any],
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        build_full_cuda_graph_evidence_envelope_consensus,
    )

    with pytest.raises(ValueError, match="malformed evidence envelope"):
        build_full_cuda_graph_evidence_envelope_consensus(
            envelopes,
            expected_world_size=1,
        )


@pytest.mark.parametrize("expected_world_size", [0, True])
def test_full_cuda_graph_evidence_envelopes_reject_invalid_world_size(
    expected_world_size: Any,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        build_full_cuda_graph_evidence_envelope_consensus,
    )

    envelopes = [] if expected_world_size == 0 else [(False, None)]
    with pytest.raises(ValueError, match="invalid policy world size"):
        build_full_cuda_graph_evidence_envelope_consensus(
            envelopes,
            expected_world_size=expected_world_size,
        )


@pytest.mark.parametrize(
    ("rank_evidence", "expected_world_size", "match"),
    [
        (
            [
                _full_cuda_graph_rank_evidence(0, "a" * 64),
                _full_cuda_graph_rank_evidence(1, "b" * 64, counters=(1, 1, 3, 0)),
            ],
            2,
            "counter mismatch",
        ),
        (
            [(0, (1, 1, 2, 0))],
            1,
            "partial rank evidence",
        ),
        (
            [_full_cuda_graph_rank_evidence(0, "a" * 64)],
            2,
            "missing rank",
        ),
        (
            [
                _full_cuda_graph_rank_evidence(0, "a" * 64),
                _full_cuda_graph_rank_evidence(0, "b" * 64),
            ],
            2,
            "duplicate rank",
        ),
        (
            [
                _full_cuda_graph_rank_evidence(0, "a" * 64),
                _full_cuda_graph_rank_evidence(2, "b" * 64),
            ],
            2,
            "invalid rank",
        ),
        (
            [
                _full_cuda_graph_rank_evidence(0, "A" * 64),
                _full_cuda_graph_rank_evidence(1, "b" * 64),
            ],
            2,
            "malformed storage digest",
        ),
        (
            [
                _full_cuda_graph_rank_evidence(0, "a" * 64, counters=(-1, 1, 2, 0)),
                _full_cuda_graph_rank_evidence(1, "b" * 64),
            ],
            2,
            "malformed counters",
        ),
    ],
)
def test_full_cuda_graph_rank_consensus_rejects_invalid_evidence(
    rank_evidence: list[Any], expected_world_size: int, match: str
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        build_full_cuda_graph_evidence_consensus,
    )

    with pytest.raises(ValueError, match=match):
        build_full_cuda_graph_evidence_consensus(
            rank_evidence, expected_world_size=expected_world_size
        )


def test_full_cuda_graph_result_aggregation_copies_consensus_exactly() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        aggregate_full_cuda_graph_evidence,
    )

    expected = {
        "full_cuda_graph_warmup_calls": 3,
        "full_cuda_graph_capture_calls": 1,
        "full_cuda_graph_replay_calls": 4,
        "full_cuda_graph_reset_calls": 0,
        "full_cuda_graph_storage_signature_sha256": "c" * 64,
    }

    assert aggregate_full_cuda_graph_evidence([expected.copy(), expected.copy()]) == (
        expected
    )


@pytest.mark.parametrize("failure", ["partial", "mismatch", "malformed"])
def test_full_cuda_graph_result_aggregation_fails_closed(failure: str) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        aggregate_full_cuda_graph_evidence,
    )

    first = {
        "full_cuda_graph_warmup_calls": 3,
        "full_cuda_graph_capture_calls": 1,
        "full_cuda_graph_replay_calls": 4,
        "full_cuda_graph_reset_calls": 0,
        "full_cuda_graph_storage_signature_sha256": "c" * 64,
    }
    second = first.copy()
    if failure == "partial":
        del second["full_cuda_graph_replay_calls"]
    elif failure == "mismatch":
        second["full_cuda_graph_capture_calls"] = 2
    else:
        second["full_cuda_graph_storage_signature_sha256"] = "C" * 64

    with pytest.raises(ValueError):
        aggregate_full_cuda_graph_evidence([first, second])


def test_full_cuda_graph_result_aggregation_is_empty_when_disabled() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        aggregate_full_cuda_graph_evidence,
    )

    assert aggregate_full_cuda_graph_evidence([{"loss": 1.0}, {"loss": 2.0}]) == {}


def test_policy_aggregators_share_full_cuda_graph_consensus_helper() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    legacy_source = (root / "nemo_rl/models/policy/lm_policy.py").read_text()
    tq_source = (root / "nemo_rl/models/policy/tq_policy.py").read_text()

    assert legacy_source.count("aggregate_full_cuda_graph_evidence(results)") == 1
    assert tq_source.count("aggregate_full_cuda_graph_evidence(results)") == 1


def test_full_cuda_graph_worker_stats_emit_cohort_digest_without_pointer_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    gathered_payloads: list[Any] = []

    def all_gather_object(gathered: list[Any], local_payload: Any) -> None:
        gathered_payloads.append(local_payload)
        gathered[:] = [
            local_payload,
            (
                True,
                _full_cuda_graph_rank_evidence(1, "b" * 64),
                _full_cuda_graph_rank_evidence(1, "b" * 64, counters=(0, 0, 0, 0)),
            ),
        ]

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    metrics: dict[str, Any] = {}
    worker._add_full_cuda_graph_execution_metrics(metrics)

    assert metrics["full_cuda_graph_warmup_calls"] == 1
    assert metrics["full_cuda_graph_capture_calls"] == 1
    assert metrics["full_cuda_graph_replay_calls"] == 2
    assert metrics["full_cuda_graph_reset_calls"] == 0
    assert metrics["full_cuda_graph_validation_warmup_calls"] == 0
    assert metrics["full_cuda_graph_validation_capture_calls"] == 0
    assert metrics["full_cuda_graph_validation_replay_calls"] == 0
    assert metrics["full_cuda_graph_validation_reset_calls"] == 0
    assert re.fullmatch(
        r"[0-9a-f]{64}", metrics["full_cuda_graph_storage_signature_sha256"]
    )
    assert len(gathered_payloads) == 1
    assert gathered_payloads[0][0] is True
    local_digest = gathered_payloads[0][1][2]
    assert metrics["full_cuda_graph_storage_signature_sha256"] != local_digest
    rendered = repr(metrics)
    assert str(model.weight.data_ptr()) not in rendered
    assert hex(model.weight.data_ptr()) not in rendered
    assert local_digest not in rendered
    assert "entries" not in rendered


def test_full_cuda_graph_worker_gathers_before_rejecting_malformed_local_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    class BrokenWrapper:
        def execution_stats(self, stage: str = "training") -> Any:
            del stage
            raise RuntimeError("rank-local failure")

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker._full_cuda_graph_enabled = True
    worker._full_cuda_graph_wrapper = BrokenWrapper()
    worker._full_cuda_graph_storage_signature = None
    collective_calls: list[Any] = []

    def all_gather_object(gathered: list[Any], local_payload: Any) -> None:
        collective_calls.append(local_payload)
        gathered[:] = [local_payload]

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(ValueError, match="malformed rank evidence"):
        worker._add_full_cuda_graph_execution_metrics({})

    assert collective_calls == [(True, None, None)]


def test_full_cuda_graph_worker_mixed_rank_state_rejects_after_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    enabled_worker = object.__new__(MegatronPolicyWorkerImpl)
    enabled_worker._full_cuda_graph_enabled = True
    enabled_worker._full_cuda_graph_wrapper = SimpleNamespace(
        execution_stats=lambda stage="training": SimpleNamespace(
            warmup_calls=1,
            capture_calls=1,
            replay_calls=2,
            reset_calls=0,
        )
    )
    enabled_worker._full_cuda_graph_storage_signature = SimpleNamespace(
        digest=lambda: "a" * 64
    )

    disabled_worker = object.__new__(MegatronPolicyWorkerImpl)
    disabled_worker._full_cuda_graph_enabled = False
    collective_payloads: list[Any] = []

    def all_gather_object(gathered: list[Any], local_payload: Any) -> None:
        collective_payloads.append(local_payload)
        gathered[:] = [
            (
                True,
                _full_cuda_graph_rank_evidence(0, "a" * 64),
                _full_cuda_graph_rank_evidence(0, "a" * 64),
            ),
            (False, None, None),
        ]

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    for worker in (enabled_worker, disabled_worker):
        with pytest.raises(ValueError, match="enabled state mismatch"):
            worker._add_full_cuda_graph_execution_metrics({})

    assert collective_payloads == [
        (
            True,
            _full_cuda_graph_rank_evidence(0, "a" * 64),
            _full_cuda_graph_rank_evidence(0, "a" * 64),
        ),
        (False, None, None),
    ]


def test_full_cuda_graph_worker_all_disabled_preserves_metrics_after_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker._full_cuda_graph_enabled = False
    collective_payloads: list[Any] = []

    def all_gather_object(gathered: list[Any], local_payload: Any) -> None:
        collective_payloads.append(local_payload)
        gathered[:] = [(False, None, None), (False, None, None)]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    metrics: dict[str, Any] = {"loss": 1.25, "all_mb_metrics": {"loss": [1.25]}}
    before = pickle.dumps(metrics)
    worker._add_full_cuda_graph_execution_metrics(metrics)

    assert pickle.dumps(metrics) == before
    assert collective_payloads == [(False, None, None)]


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
