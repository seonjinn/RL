# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.vllm


@pytest.fixture()
def fp8_module():
    pytest.importorskip("vllm")

    from nemo_rl.models.generation.vllm.quantization import fp8

    old_config = fp8.global_fp8_config
    old_state = fp8.fp8_state
    old_patches_applied = fp8.fp8_patches_applied
    fp8.global_fp8_config = None
    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False

    try:
        yield fp8
    finally:
        fp8.global_fp8_config = old_config
        fp8.fp8_state = old_state
        fp8.fp8_patches_applied = old_patches_applied


@pytest.fixture()
def mxfp8_linear_module(monkeypatch):
    """Load the MXFP8 helper with only its vLLM import boundary stubbed."""

    class Tensor:
        def __init__(self, shape, value):
            self.shape = tuple(shape)
            self.value = value
            self.ndim = len(shape)

        @property
        def data(self):
            return self

        def __getitem__(self, _index):
            return self

        def __add__(self, value):
            return Tensor(self.shape, self.value + value)

        def clone(self):
            return Tensor(self.shape, self.value)

        def contiguous(self):
            return self

        def copy_(self, source):
            self.value = source.value
            return self

    class Parameter:
        def __init__(self, data, **_kwargs):
            self.data = data

        @property
        def ndim(self):
            return self.data.ndim

        def copy_(self, source):
            self.data.copy_(source.data)
            return self

    torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Parameter=Parameter),
        squeeze=lambda tensor, **_kwargs: tensor,
        zeros=lambda *shape: Tensor(shape, 0.0),
        ones=lambda *shape: Tensor(shape, 1.0),
    )

    def add_module(name, **attributes):
        module = types.ModuleType(name)
        module.__path__ = []
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class ModelWeightParameter(Parameter):
        pass

    add_module("ray")
    add_module("torch", nn=torch.nn, squeeze=torch.squeeze)
    add_module("accelerate", init_empty_weights=lambda: None)
    add_module("transformers", AutoConfig=object, AutoModel=object)
    add_module("nemo_rl")
    add_module("nemo_rl.models")
    add_module("nemo_rl.models.generation")
    add_module("nemo_rl.models.generation.vllm")
    add_module("nemo_rl.models.generation.vllm.quantization")
    add_module(
        "nemo_rl.models.generation.vllm.quantization.mxfp8_utils",
        pad_flashinfer_scale_k=lambda input_tensor: input_tensor,
    )
    add_module("vllm")
    add_module("vllm.logger", init_logger=lambda _name: object())
    add_module("vllm.model_executor")
    add_module("vllm.model_executor.layers")
    add_module("vllm.model_executor.layers.fused_moe")
    add_module(
        "vllm.model_executor.layers.fused_moe.routed_experts", RoutedExperts=object
    )
    add_module("vllm.model_executor.layers.fused_moe.runner")
    add_module(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner", MoERunner=object
    )
    add_module("vllm.model_executor.layers.linear", LinearBase=object)
    add_module("vllm.model_executor.layers.quantization")
    mxfp8_linear_backend = types.SimpleNamespace(
        FLASHINFER_CUTLASS="FLASHINFER_CUTLASS",
        FLASHINFER_CUTEDSL="FLASHINFER_CUTEDSL",
    )
    mxfp8_utils = add_module(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        Mxfp8LinearBackend=mxfp8_linear_backend,
        mxfp8_e4m3_quantize=lambda tensor: (
            Tensor(tensor.shape, 2.0),
            Tensor((tensor.shape[0], tensor.shape[1] // 32), 3.0),
        ),
        swizzle_mxfp8_scale=lambda weight_scale, **_kwargs: weight_scale,
    )
    add_module("vllm.model_executor.layers.quantization.utils")
    add_module(
        "vllm.model_executor.parameter", ModelWeightParameter=ModelWeightParameter
    )
    add_module(
        "vllm.triton_utils",
        tl=types.SimpleNamespace(constexpr=object),
        triton=types.SimpleNamespace(jit=lambda function: function),
    )
    add_module("vllm.v1")
    add_module(
        "vllm.v1.engine.core",
        EngineCoreProc=type("EngineCoreProc", (), {"run_engine_core": None}),
    )
    add_module(
        "vllm.v1.engine.utils",
        CoreEngineProcManager=type("CoreEngineProcManager", (), {"__init__": None}),
    )

    module_name = "fp8_under_test"
    source_path = (
        Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/quantization/fp8.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    fp8 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, fp8)
    spec.loader.exec_module(fp8)
    return fp8, mxfp8_utils, torch, Tensor


def test_load_weights_targets_native_mxfp8_checkpoint_scale(
    mxfp8_linear_module, monkeypatch
):
    fp8, _, _, Tensor = mxfp8_linear_module
    loaded_weights = []
    kernel = types.SimpleNamespace(preserves_checkpoint_weight_scale_for_refit=True)
    layer = types.SimpleNamespace(quant_method=types.SimpleNamespace(kernel=kernel))
    model = types.SimpleNamespace(
        load_weights=lambda weights: loaded_weights.extend(weights)
    )
    model_runner = types.SimpleNamespace(model=model)

    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda _name, _model: True)
    monkeypatch.setattr(fp8, "_get_module_from_param_name", lambda _model, _name: layer)
    fp8.global_fp8_config = types.SimpleNamespace(is_mx=True)

    fp8.load_weights(
        [("layers.0.self_attn.o_proj.weight", Tensor((64, 32), 1.0))],
        model_runner,
    )

    assert [name for name, _ in loaded_weights] == [
        "layers.0.self_attn.o_proj.weight",
        "layers.0.self_attn.o_proj.weight_scale",
    ]


def test_load_weights_preserves_legacy_mxfp8_checkpoint_scale_name(
    mxfp8_linear_module, monkeypatch
):
    fp8, _, _, Tensor = mxfp8_linear_module
    loaded_weights = []
    layer = types.SimpleNamespace(quant_method=types.SimpleNamespace(kernel=object()))
    model = types.SimpleNamespace(
        load_weights=lambda weights: loaded_weights.extend(weights)
    )

    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda _name, _model: True)
    monkeypatch.setattr(fp8, "_get_module_from_param_name", lambda _model, _name: layer)
    fp8.global_fp8_config = types.SimpleNamespace(is_mx=True)

    fp8.load_weights(
        [("layers.0.self_attn.o_proj.weight", Tensor((64, 32), 1.0))],
        types.SimpleNamespace(model=model),
    )

    assert [name for name, _ in loaded_weights] == [
        "layers.0.self_attn.o_proj.weight",
        "layers.0.self_attn.o_proj.weight_scale_from_checkpoint",
    ]


def test_init_fp8_uses_mxfp8_quantization_config(fp8_module, monkeypatch):
    fp8 = fp8_module
    applied_configs = []

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(
        fp8,
        "monkey_patch_vllm_ray_executor",
        lambda fp8_config: applied_configs.append(fp8_config),
    )
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
            "use_deep_gemm": True,
        },
        "dummy-model",
        model_parallel_size=1,
    )

    assert vllm_kwargs == {
        "quantization": "fp8",
        "kv_cache_dtype": "auto",
        "hf_overrides": {"quantization_config": fp8.MXFP8_BLOCK_QUANT_KWARGS},
    }
    assert applied_configs == [fp8.global_fp8_config]
    assert fp8.global_fp8_config.is_mx is True
    assert "VLLM_USE_DEEP_GEMM" not in fp8.os.environ
    assert "VLLM_USE_DEEP_GEMM_E8M0" not in fp8.os.environ


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("pow2_weight_scaling_factors", "only pow2 weight scaling factors"),
        ("pow2_activation_scaling_factors", "only pow2 activation scaling factors"),
    ],
)
def test_init_fp8_rejects_non_pow2_mxfp8_scales(fp8_module, monkeypatch, field, error):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _fp8_config: None)

    with pytest.raises(ValueError, match=error):
        fp8.init_fp8(
            {
                "precision": "fp8",
                "kv_cache_dtype": "auto",
                "async_engine": False,
                "is_mx": True,
                field: False,
            },
            "dummy-model",
            model_parallel_size=1,
        )


def test_apply_fp8_patches_registers_modelopt_patches_only_for_mxfp8(
    fp8_module, monkeypatch
):
    fp8 = fp8_module
    patched_paths = []

    class FakePatch:
        def __init__(self, path):
            self.path = path
            self.started = False

        def start(self):
            self.started = True

    def fake_patch(path, _replacement):
        patched_paths.append(path)
        return FakePatch(path)

    monkeypatch.setattr(fp8, "patch", fake_patch)

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(use_fp8_weights=True, model_parallel_size=1, is_mx=False),
    )
    assert not any("ModelOptMxFp8" in path for path in patched_paths)
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)

    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False
    patched_paths.clear()

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(
            use_fp8_weights=True,
            model_parallel_size=1,
            use_activation_pow2_scale=True,
        ),
    )
    assert any("per_token_group_quant_fp8" in path for path in patched_paths)
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)

    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False
    patched_paths.clear()

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(use_fp8_weights=True, model_parallel_size=1, is_mx=True),
    )

    assert any("ModelOptMxFp8LinearMethod" in path for path in patched_paths)
    assert any("ModelOptMxFp8FusedMoE.create_weights" in path for path in patched_paths)
    assert any(
        "ModelOptMxFp8FusedMoE.process_weights_after_loading" in path
        for path in patched_paths
    )
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)


def test_process_weights_after_loading_copies_in_place_on_refit(monkeypatch):
    """Refit runs this every step; rebinding .data each time fragments memory.

    Regression guard for the CuMemAllocator wake-up OOM (~75 steps into the
    fp8-rollouts nightlies): the 0.25 port rebound weight/weight_scale_inv to
    fresh allocations on every call, where 0.20 copied in place. Nothing in the
    suite pinned that, so a refactor back to .data rebinding would have
    produced no test failure -- just a slow OOM in a nightly days later.
    """
    import torch
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    from nemo_rl.models.generation.vllm.quantization import fp8

    layer = types.SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False),
        weight_scale_inv=torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False),
    )
    # Same shape/dtype back, but a *fresh* tensor each call -- exactly what the
    # real helper returns once the processed layout is stable.
    monkeypatch.setattr(
        fp8_utils,
        "process_fp8_weight_block_strategy",
        lambda w, s: (torch.ones_like(w), torch.ones_like(s)),
    )
    monkeypatch.setattr(fp8, "maybe_post_process_fp8_weight_block", lambda _layer: None)

    method = types.SimpleNamespace(
        block_quant=True,
        quant_config=types.SimpleNamespace(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic"
        ),
    )

    weight_ptr = layer.weight.data.data_ptr()
    scale_ptr = layer.weight_scale_inv.data.data_ptr()
    weight_param, scale_param = layer.weight, layer.weight_scale_inv

    for _ in range(3):  # initial load + two refits
        fp8.process_weights_after_loading(method, layer)

    assert layer.weight.data.data_ptr() == weight_ptr, (
        "weight storage was rebound instead of copied in place; on a real refit "
        "this leaks a fresh allocation every step until wake_up OOMs"
    )
    assert layer.weight_scale_inv.data.data_ptr() == scale_ptr, (
        "weight_scale_inv storage was rebound instead of copied in place"
    )
    # Parameter identity (and therefore weight_loader) must also survive.
    assert layer.weight is weight_param
    assert layer.weight_scale_inv is scale_param
    # The processed values must actually land.
    assert torch.equal(layer.weight.data, torch.ones(4, 4))


@pytest.mark.parametrize(
    "kernel_name",
    [
        "FlashInferCutedslMxfp8LinearKernel",
        "FlashInferTrtllmMxfp8LinearKernel",
    ],
)
def test_mxfp8_linear_delegates_to_refit_safe_native_kernel(fp8_module, kernel_name):
    calls = []

    def process_weights_after_loading(self, layer):
        calls.append(layer)

    kernel_type = type(
        kernel_name,
        (),
        {
            "preserves_checkpoint_weight_scale_for_refit": True,
            "process_weights_after_loading": process_weights_after_loading,
        },
    )
    layer = types.SimpleNamespace(weight=types.SimpleNamespace(ndim=2))
    method = types.SimpleNamespace(kernel=kernel_type())

    fp8_module.process_weights_after_loading_mxfp8_linear(method, layer)

    assert calls == [layer]
    assert method.kernel.__class__.__name__ == kernel_name
    assert not hasattr(layer, "weight_scale_from_checkpoint")


def test_mxfp8_linear_rejects_refit_unsafe_cutedsl_kernel(fp8_module):
    kernel_type = type("FlashInferCutedslMxfp8LinearKernel", (), {})
    layer = types.SimpleNamespace(weight=types.SimpleNamespace(ndim=2))
    method = types.SimpleNamespace(kernel=kernel_type())

    with pytest.raises(
        RuntimeError,
        match=(
            "FlashInferCutedslMxfp8LinearKernel.*"
            "preserves_checkpoint_weight_scale_for_refit"
        ),
    ):
        fp8_module.process_weights_after_loading_mxfp8_linear(method, layer)

    assert method.kernel.__class__.__name__ == "FlashInferCutedslMxfp8LinearKernel"


def test_mxfp8_linear_legacy_cutlass_prepares_scales_on_refit(
    mxfp8_linear_module,
):
    fp8, mxfp8_utils, torch, _ = mxfp8_linear_module
    swizzle_inputs = []

    def swizzle_mxfp8_scale(weight_scale, *, M, K):
        swizzle_inputs.append((weight_scale.clone(), M, K))
        return weight_scale + len(swizzle_inputs)

    mxfp8_utils.swizzle_mxfp8_scale = swizzle_mxfp8_scale

    class Layer:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(2, 32), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(
                torch.ones(2, 1), requires_grad=False
            )
            self.weight_scale.weight_loader = object()

        def register_parameter(self, name, parameter):
            setattr(self, name, parameter)

    layer = Layer()
    method = types.SimpleNamespace(
        backend=mxfp8_utils.Mxfp8LinearBackend.FLASHINFER_CUTLASS
    )

    fp8.process_weights_after_loading_mxfp8_linear(method, layer)
    assert layer.weight_scale.data.value == 2.0

    fp8.process_weights_after_loading_mxfp8_linear(method, layer)

    assert layer.weight_scale.data.value == 3.0
    assert layer.weight_scale_from_checkpoint.data.value == 1.0
    assert [(M, K) for _, M, K in swizzle_inputs] == [(2, 32), (2, 32)]
    assert all(
        weight_scale.shape == (2, 1) and weight_scale.value == 1.0
        for weight_scale, _, _ in swizzle_inputs
    )


def test_mxfp8_linear_rejects_legacy_non_cutlass_backend(mxfp8_linear_module):
    fp8, mxfp8_utils, torch, _ = mxfp8_linear_module
    layer = types.SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(2, 32), requires_grad=False)
    )
    method = types.SimpleNamespace(
        backend=mxfp8_utils.Mxfp8LinearBackend.FLASHINFER_CUTEDSL
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "FLASHINFER_CUTEDSL.*None.*"
            "preserves_checkpoint_weight_scale_for_refit=True.*"
            "process_weights_after_loading\\(layer\\)"
        ),
    ):
        fp8.process_weights_after_loading_mxfp8_linear(method, layer)


def test_mxfp8_linear_rejects_non_2d_weight_before_backend_dispatch(
    mxfp8_linear_module,
):
    fp8, _, torch, _ = mxfp8_linear_module

    class Method:
        @property
        def backend(self):
            raise AssertionError("backend dispatch must not run for non-2D weights")

    layer = types.SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(2, 2, 2), requires_grad=False)
    )

    with pytest.raises(ValueError, match="must be 2D, but got 3D"):
        fp8.process_weights_after_loading_mxfp8_linear(Method(), layer)


def test_initialize_mxfp8_moe_kernel_is_idempotent(fp8_module, monkeypatch):
    fp8 = fp8_module
    created_kernel = object()
    calls = []

    def make_fp8_moe_kernel(**kwargs):
        calls.append(kwargs)
        return created_kernel

    from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle

    monkeypatch.setattr(fp8_oracle, "make_fp8_moe_kernel", make_fp8_moe_kernel)

    quant_config = object()
    experts_cls = object()
    routing_tables = object()
    layer = types.SimpleNamespace(
        _expert_routing_tables=lambda: routing_tables,
    )
    method = types.SimpleNamespace(
        moe_kernel=None,
        moe_quant_config=None,
        moe=object(),
        mxfp8_backend=object(),
        experts_cls=experts_cls,
        get_fused_moe_quant_config=lambda _layer: quant_config,
    )

    fp8._initialize_mxfp8_moe_kernel(method, layer)
    fp8._initialize_mxfp8_moe_kernel(method, layer)

    assert method.moe_kernel is created_kernel
    assert method.moe_quant_config is quant_config
    assert len(calls) == 1
    assert calls[0]["moe_quant_config"] is quant_config
    assert calls[0]["experts_cls"] is experts_cls
    assert calls[0]["routing_tables"] is routing_tables
    assert calls[0]["layer"] is layer
