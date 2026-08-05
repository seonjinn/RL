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

import importlib
import importlib.util
from types import ModuleType

import pytest

_MODULE_NAME = "nemo_rl.models.megatron.cuda_graph_storage"


class _FakeStorage:
    def __init__(self, pointer: int) -> None:
        self._pointer = pointer

    def data_ptr(self) -> int:
        return self._pointer


class _FakeTensor:
    dtype = "bfloat16"
    device = "cuda:0"
    layout = "strided"

    def __init__(self, *, data_pointer: int, storage_pointer: int) -> None:
        self._data_pointer = data_pointer
        self._storage = _FakeStorage(storage_pointer)
        self.value = 0

    def data_ptr(self) -> int:
        return self._data_pointer

    def untyped_storage(self) -> _FakeStorage:
        return self._storage

    def storage_offset(self) -> int:
        return 0

    def size(self) -> tuple[int, int]:
        return (2, 4)

    def stride(self) -> tuple[int, int]:
        return (4, 1)

    def copy_value_(self, value: int) -> None:
        self.value = value


def _get_storage_module() -> ModuleType:
    module_spec = importlib.util.find_spec(_MODULE_NAME)
    assert module_spec is not None, f"{_MODULE_NAME} is not implemented"
    return importlib.import_module(_MODULE_NAME)


def test_in_place_value_copy_preserves_storage_fingerprint() -> None:
    storage_module = _get_storage_module()
    tensor = _FakeTensor(data_pointer=101, storage_pointer=100)
    before = storage_module.fingerprint_named_tensors((("weight", tensor),))

    tensor.copy_value_(7)

    after = storage_module.fingerprint_named_tensors((("weight", tensor),))
    assert after == before


def test_recreated_tensor_view_with_same_address_preserves_storage_fingerprint() -> (
    None
):
    storage_module = _get_storage_module()
    original_view = _FakeTensor(data_pointer=101, storage_pointer=100)
    recreated_view = _FakeTensor(data_pointer=101, storage_pointer=100)
    before = storage_module.fingerprint_named_tensors(
        (("weight.main_grad", original_view),)
    )

    after = storage_module.fingerprint_named_tensors(
        (("weight.main_grad", recreated_view),)
    )

    assert after == before


def test_model_and_gradient_replacement_are_classified_separately() -> None:
    storage_module = _get_storage_module()
    model_tensor = _FakeTensor(data_pointer=101, storage_pointer=100)
    grad_tensor = _FakeTensor(data_pointer=201, storage_pointer=200)
    baseline = storage_module.GraphStorageFingerprint(
        model=storage_module.fingerprint_named_tensors((("weight", model_tensor),)),
        grads=storage_module.fingerprint_named_tensors(
            (("weight.main_grad", grad_tensor),)
        ),
    )

    replaced_model = storage_module.GraphStorageFingerprint(
        model=storage_module.fingerprint_named_tensors(
            (("weight", _FakeTensor(data_pointer=301, storage_pointer=300)),)
        ),
        grads=baseline.grads,
    )
    replaced_grad = storage_module.GraphStorageFingerprint(
        model=baseline.model,
        grads=storage_module.fingerprint_named_tensors(
            (("weight.main_grad", _FakeTensor(data_pointer=401, storage_pointer=400)),)
        ),
    )

    assert storage_module.classify_storage_change(baseline, baseline) == (
        storage_module.StorageChange.NONE
    )
    assert storage_module.classify_storage_change(baseline, replaced_model) == (
        storage_module.StorageChange.MODEL
    )
    assert storage_module.classify_storage_change(baseline, replaced_grad) == (
        storage_module.StorageChange.GRAD
    )


@pytest.mark.parametrize(
    ("generation_colocated", "offload_optimizer_for_logprob", "message"),
    (
        (True, False, "non-colocated generation"),
        (False, True, "offload_optimizer_for_logprob=false"),
    ),
)
def test_training_graph_storage_lifecycle_rejects_relocation_requests(
    generation_colocated: bool,
    offload_optimizer_for_logprob: bool,
    message: str,
) -> None:
    storage_module = _get_storage_module()

    with pytest.raises(ValueError, match=message):
        storage_module.validate_training_graph_storage_lifecycle(
            cuda_graph_impl="transformer_engine",
            generation_colocated=generation_colocated,
            generation_backend="vllm",
            fp8_enabled=False,
            use_custom_fsdp=False,
            offload_optimizer_for_logprob=offload_optimizer_for_logprob,
        )


def test_training_graph_storage_lifecycle_rejects_megatron_generation_refit() -> None:
    storage_module = _get_storage_module()

    with pytest.raises(ValueError, match="Megatron generation backend"):
        storage_module.validate_training_graph_storage_lifecycle(
            cuda_graph_impl="transformer_engine",
            generation_colocated=False,
            generation_backend="megatron",
            fp8_enabled=False,
            use_custom_fsdp=False,
            offload_optimizer_for_logprob=False,
        )


@pytest.mark.parametrize(
    ("fp8_enabled", "use_custom_fsdp", "message"),
    (
        (True, False, "FP8"),
        (False, True, "custom FSDP"),
    ),
)
def test_training_graph_storage_lifecycle_rejects_untracked_storage_owners(
    fp8_enabled: bool,
    use_custom_fsdp: bool,
    message: str,
) -> None:
    storage_module = _get_storage_module()

    with pytest.raises(ValueError, match=message):
        storage_module.validate_training_graph_storage_lifecycle(
            cuda_graph_impl="transformer_engine",
            generation_colocated=False,
            generation_backend="vllm",
            fp8_enabled=fp8_enabled,
            use_custom_fsdp=use_custom_fsdp,
            offload_optimizer_for_logprob=False,
        )


@pytest.mark.parametrize("cuda_graph_impl", ("none", "local", "full_iteration"))
def test_non_te_graph_modes_preserve_existing_storage_lifecycle_options(
    cuda_graph_impl: str,
) -> None:
    storage_module = _get_storage_module()

    storage_module.validate_training_graph_storage_lifecycle(
        cuda_graph_impl=cuda_graph_impl,
        generation_colocated=True,
        generation_backend="megatron",
        fp8_enabled=True,
        use_custom_fsdp=True,
        offload_optimizer_for_logprob=True,
    )
