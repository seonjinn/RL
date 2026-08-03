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

import types

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


def test_init_fp8_can_exclude_causal_lm_head(fp8_module, monkeypatch):
    fp8 = fp8_module

    class FakeCausalLM:
        def named_parameters(self):
            return iter(
                [
                    ("model.layers.0.mlp.gate.weight", object()),
                    ("lm_head.weight", object()),
                ]
            )

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=1),
    )
    monkeypatch.setattr(
        fp8.AutoModelForCausalLM,
        "from_config",
        lambda *_args, **_kwargs: FakeCausalLM(),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
            "quantization_ignored_layer_kws": [".mlp.gate", "lm_head"],
        },
        "dummy-model",
        model_parallel_size=1,
    )

    quantization_config = vllm_kwargs["hf_overrides"]["quantization_config"]
    assert quantization_config["ignored_layers"] == [
        "model.layers.0.mlp.gate",
        "lm_head",
    ]
    assert quantization_config["ignore"] == quantization_config["ignored_layers"]


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


def test_mxfp8_trtllm_refit_preserves_checkpoint_and_prepared_buffers(
    fp8_module,
):
    import torch

    fp8 = fp8_module

    class FlashInferTrtllmMxfp8LinearKernel:
        def process_weights_after_loading(self, layer):
            layer.weight = torch.nn.Parameter(
                (layer.weight.data.float() + 10).to(torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data.reshape(-1) + 20,
                requires_grad=False,
            )
            layer._mxfp8_trtllm_output_features = 2

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.ones((2, 4), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight.weight_loader = object()
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((2, 1), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale.weight_loader = object()
    method = types.SimpleNamespace(kernel=FlashInferTrtllmMxfp8LinearKernel())

    fp8.process_weights_after_loading_mxfp8_linear(method, layer)

    assert hasattr(layer, "weight_from_checkpoint")
    assert hasattr(layer, "weight_scale_from_checkpoint")
    assert torch.equal(layer.weight_from_checkpoint.float(), torch.ones((2, 4)))
    assert torch.equal(layer.weight.float(), torch.full((2, 4), 11.0))
    prepared_weight_ptr = layer.weight.data_ptr()
    prepared_scale_ptr = layer.weight_scale.data_ptr()

    with torch.no_grad():
        layer.weight_from_checkpoint.fill_(2)
        layer.weight_scale_from_checkpoint.fill_(3)
    fp8.process_weights_after_loading_mxfp8_linear(method, layer)

    assert layer.weight.data_ptr() == prepared_weight_ptr
    assert layer.weight_scale.data_ptr() == prepared_scale_ptr
    assert torch.equal(layer.weight.float(), torch.full((2, 4), 12.0))
    assert torch.equal(layer.weight_scale, torch.full((2,), 23, dtype=torch.uint8))


def test_mxfp8_refit_loads_trtllm_weight_into_checkpoint_parameter(
    fp8_module, monkeypatch
):
    import torch
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    fp8 = fp8_module
    loaded = []
    layer = types.SimpleNamespace(weight_from_checkpoint=object())
    model = types.SimpleNamespace(load_weights=lambda weights: loaded.extend(weights))
    runner = types.SimpleNamespace(model=model)

    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda _name, _model: True)
    monkeypatch.setattr(fp8, "_get_module_from_param_name", lambda *_args: layer)
    monkeypatch.setattr(
        mxfp8_utils,
        "mxfp8_e4m3_quantize",
        lambda tensor: (
            tensor.to(torch.float8_e4m3fn),
            torch.ones((tensor.shape[0], tensor.shape[1] // 32, 1)),
        ),
    )
    fp8.global_fp8_config = fp8.FP8Config(is_mx=True)

    fp8.load_weights(
        [("model.proj.weight", torch.ones((2, 32), dtype=torch.bfloat16))],
        runner,
    )

    assert [name for name, _value in loaded] == [
        "model.proj.weight_from_checkpoint",
        "model.proj.weight_scale_from_checkpoint",
    ]


def test_mxfp8_moe_initializes_kernel_once(fp8_module, monkeypatch):
    from vllm.model_executor.layers.quantization import fp8 as vllm_fp8

    fp8 = fp8_module
    calls = []
    kernel = object()
    routing_tables = object()
    layer = types.SimpleNamespace(
        _expert_routing_tables=lambda: routing_tables,
    )
    method = types.SimpleNamespace(
        experts_cls=object(),
        get_fused_moe_quant_config=lambda _layer: "mx-config",
        moe="moe-config",
        moe_kernel=None,
        mxfp8_backend="flashinfer-trtllm",
    )

    def make_kernel(**kwargs):
        calls.append(kwargs)
        return kernel

    monkeypatch.setattr(vllm_fp8, "make_fp8_moe_kernel", make_kernel)

    fp8._initialize_mxfp8_moe_kernel(method, layer)
    fp8._initialize_mxfp8_moe_kernel(method, layer)

    assert method.moe_quant_config == "mx-config"
    assert method.moe_kernel is kernel
    assert calls == [
        {
            "moe_quant_config": "mx-config",
            "moe_config": "moe-config",
            "fp8_backend": "flashinfer-trtllm",
            "experts_cls": method.experts_cls,
            "routing_tables": routing_tables,
            "layer": layer,
        }
    ]


def test_mxfp8_moe_refit_uses_vllm_backend_conversion_and_preserves_raw_scales(
    fp8_module, monkeypatch
):
    import torch
    from vllm.model_executor.layers.quantization import fp8 as vllm_fp8
    from vllm.model_executor import parameter as vllm_parameter

    fp8 = fp8_module
    calls = []

    def model_weight_parameter(*, data, **kwargs):
        parameter = torch.nn.Parameter(data, requires_grad=False)
        parameter.weight_loader = kwargs.get("weight_loader")
        return parameter

    def convert(**kwargs):
        calls.append(kwargs)
        return (
            kwargs["w13"] + 10,
            kwargs["w2"] + 11,
            kwargs["w13_scale"] + 20,
            kwargs["w2_scale"] + 21,
        )

    monkeypatch.setattr(vllm_parameter, "ModelWeightParameter", model_weight_parameter)
    monkeypatch.setattr(vllm_fp8, "convert_to_fp8_moe_kernel_format", convert)
    monkeypatch.setattr(fp8, "_initialize_mxfp8_moe_kernel", lambda *_args: None)

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones((2, 2)), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.full((2, 2), 2.0), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones((2, 1), dtype=torch.uint8), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.full((2, 1), 2, dtype=torch.uint8), requires_grad=False
    )
    layer.w13_weight_scale.weight_loader = object()
    layer.w2_weight_scale.weight_loader = object()
    method = types.SimpleNamespace(
        mxfp8_backend="flashinfer-trtllm",
        weight_block_size=[1, 32],
    )

    w13_parameter = layer.w13_weight
    w2_parameter = layer.w2_weight
    fp8.process_weights_after_loading_mxfp8_moe(method, layer)

    assert layer.weight_block_size == [1, 32]
    assert layer.w13_weight is w13_parameter
    assert layer.w2_weight is w2_parameter
    assert torch.equal(layer.w13_weight, torch.full((2, 2), 11.0))
    assert torch.equal(layer.w2_weight, torch.full((2, 2), 13.0))
    assert torch.equal(
        layer.w13_weight_scale_from_checkpoint,
        torch.ones((2, 1), dtype=torch.uint8),
    )
    assert torch.equal(
        layer.w2_weight_scale_from_checkpoint,
        torch.full((2, 1), 2, dtype=torch.uint8),
    )
    prepared_w13_scale = layer.w13_weight_scale
    prepared_w2_scale = layer.w2_weight_scale

    with torch.no_grad():
        layer.w13_weight.fill_(3)
        layer.w2_weight.fill_(4)
        layer.w13_weight_scale_from_checkpoint.fill_(5)
        layer.w2_weight_scale_from_checkpoint.fill_(6)
    fp8.process_weights_after_loading_mxfp8_moe(method, layer)

    assert layer.w13_weight is w13_parameter
    assert layer.w2_weight is w2_parameter
    assert layer.w13_weight_scale is prepared_w13_scale
    assert layer.w2_weight_scale is prepared_w2_scale
    assert torch.equal(layer.w13_weight, torch.full((2, 2), 13.0))
    assert torch.equal(layer.w2_weight, torch.full((2, 2), 15.0))
    assert torch.equal(
        layer.w13_weight_scale_from_checkpoint,
        torch.full((2, 1), 5, dtype=torch.uint8),
    )
    assert torch.equal(
        layer.w2_weight_scale_from_checkpoint,
        torch.full((2, 1), 6, dtype=torch.uint8),
    )
    assert len(calls) == 2
    assert all(call["fp8_backend"] == "flashinfer-trtllm" for call in calls)
    assert all(call["w13_input_scale"] is None for call in calls)
    assert all(call["w2_input_scale"] is None for call in calls)
