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
import os
import sys
import types
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest
import torch

from examples.modelopt import export_nvfp4_calibration
import nemo_rl.modelopt.models.generation.vllm_modelopt as vllm_modelopt
import nemo_rl.modelopt.utils as modelopt_utils
from nemo_rl.modelopt.calibration_artifact import save_nvfp4_calibration
from nemo_rl.modelopt.models.generation.nvfp4_refit import NVFP4Calibration
from nemo_rl.modelopt.models.generation.vllm_modelopt import (
    NEMO_MODELOPT_W4A4,
    NEMO_MODELOPT_W4A16,
    quantization_method_for_mode,
    register_nemo_modelopt_nvfp4,
)
from nemo_rl.modelopt.utils import (
    build_vllm_modelopt_nvfp4_config,
    iter_quant_ignore_name_candidates,
    matches_quant_ignore_pattern,
    resolve_nvfp4_real_quant_mode,
    resolve_quant_cfg,
)


@pytest.fixture(autouse=True)
def _disable_nvtx_without_cuda(monkeypatch):
    if torch.cuda.is_available():
        return
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)


@pytest.fixture(autouse=True)
def _install_optional_modelopt_config_api(monkeypatch):
    """Provide ModelOpt's config APIs when the optional dependency is absent."""
    try:
        import modelopt.torch.export.convert_hf_config  # noqa: F401
        import modelopt.torch.quantization.config  # noqa: F401

        return
    except ImportError:
        pass

    module_names = (
        "modelopt",
        "modelopt.recipe",
        "modelopt.torch",
        "modelopt.torch.export",
        "modelopt.torch.quantization",
    )
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)

    def missing_recipe(config_name):
        raise FileNotFoundError(config_name)

    sys.modules["modelopt.recipe"].load_config = missing_recipe

    convert_module = types.ModuleType("modelopt.torch.export.convert_hf_config")

    def convert_hf_quant_config_format(config):
        quantization = config["quantization"]
        algo = quantization["quant_algo"]
        group = {
            "weights": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": quantization["group_size"],
            },
            "targets": ["Linear"],
        }
        if algo == "NVFP4":
            group["input_activations"] = dict(group["weights"])
        return {
            "config_groups": {"group_0": group},
            "ignore": quantization["exclude_modules"],
            "quant_algo": algo,
            "producer": config["producer"],
            "quant_method": "modelopt",
        }

    convert_module.convert_hf_quant_config_format = convert_hf_quant_config_format
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch.export.convert_hf_config",
        convert_module,
    )

    config_module = types.ModuleType("modelopt.torch.quantization.config")

    class QuantizerCfgEntry:
        def __init__(self, entry):
            self.entry = {"enable": True, **entry}

        def model_dump(self, **kwargs):
            del kwargs
            return {
                key: value for key, value in self.entry.items() if value is not None
            }

    class QuantizeConfig:
        def __init__(self, quant_cfg, **kwargs):
            del kwargs
            self.quant_cfg = [QuantizerCfgEntry(entry) for entry in quant_cfg]

    config_module.QuantizeConfig = QuantizeConfig
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch.quantization.config",
        config_module,
    )


def _install_fake_vllm_worker(monkeypatch):
    """Install the minimal vLLM worker hierarchy needed by the backend import."""
    module_names = ["vllm", "vllm.distributed", "vllm.v1", "vllm.v1.worker"]
    modules = {}
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        modules[module_name] = module
        monkeypatch.setitem(sys.modules, module_name, module)

    gpu_worker_module = types.ModuleType("vllm.v1.worker.gpu_worker")

    class FakeVllmWorker:
        pass

    gpu_worker_module.Worker = FakeVllmWorker
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker_module)
    parallel_state_module = types.ModuleType("vllm.distributed.parallel_state")

    def get_pp_group() -> types.SimpleNamespace:
        return types.SimpleNamespace(is_last_rank=True)

    parallel_state_module.get_pp_group = get_pp_group
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.parallel_state", parallel_state_module
    )
    modules["vllm"].distributed = modules["vllm.distributed"]
    modules["vllm.distributed"].parallel_state = parallel_state_module
    modules["vllm"].v1 = modules["vllm.v1"]
    modules["vllm.v1"].worker = modules["vllm.v1.worker"]
    modules["vllm.v1.worker"].gpu_worker = gpu_worker_module


def _clear_vllm_backend_modules(monkeypatch):
    for module_name in (
        "nemo_rl.modelopt.models.generation.vllm_quant_backend",
        "nemo_rl.models.generation.vllm.vllm_backend",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)


def _import_vllm_quant_backend(monkeypatch):
    """Import the NeMo-RL backend without requiring the vLLM C extension."""
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    _install_fake_vllm_worker(monkeypatch)
    _install_fake_vllm_reload(monkeypatch)
    _install_fake_modelopt_tensor_quantizer(monkeypatch)
    _clear_vllm_backend_modules(monkeypatch)
    try:
        return importlib.import_module(
            "nemo_rl.modelopt.models.generation.vllm_quant_backend"
        )
    except ImportError as exc:
        pytest.skip(f"could not import vLLM quant backend: {exc}")


def _base_vllm_backend():
    return sys.modules["nemo_rl.models.generation.vllm.vllm_backend"]


def _install_fake_vllm_reload(monkeypatch):
    """Install the public vLLM layerwise-reload API used by real-quant refits."""
    module_names = (
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.models",
    )
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)

    config_module = types.ModuleType("vllm.config")
    config_module.current = None

    @contextmanager
    def set_current_vllm_config(config):
        previous = config_module.current
        config_module.current = config
        try:
            yield
        finally:
            config_module.current = previous

    def get_current_vllm_config():
        if config_module.current is None:
            raise AssertionError("Current vLLM config is not set")
        return config_module.current

    config_module.set_current_vllm_config = set_current_vllm_config
    config_module.get_current_vllm_config = get_current_vllm_config
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)

    reload_module = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.__path__ = []
    reload_module.initialize_layerwise_reload = lambda model: None
    reload_module.finalize_layerwise_reload = lambda model, model_config: None
    layerwise_module = types.ModuleType(
        "vllm.model_executor.model_loader.reload.layerwise"
    )
    layerwise_module.get_layerwise_info = lambda module: types.SimpleNamespace(
        loaded_weights=[],
        load_numel=0,
        load_numel_total=None,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.reload",
        reload_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.reload.layerwise",
        layerwise_module,
    )
    model_utils_module = types.ModuleType("vllm.model_executor.models.utils")

    class FakeWeightsMapper:
        def __init__(
            self,
            *,
            orig_to_new_substr=None,
            orig_to_new_stacked=None,
            orig_to_new_prefix=None,
        ):
            self.orig_to_new_substr = orig_to_new_substr or {}
            self.orig_to_new_stacked = orig_to_new_stacked or {}
            self.orig_to_new_prefix = orig_to_new_prefix or {}
            self.apply_list_calls = []

        def _map_name_with_shard(self, key):
            for substring, replacement in self.orig_to_new_substr.items():
                if substring in key:
                    if replacement is None:
                        return None
                    key = key.replace(substring, replacement, 1)
            shard_id = None
            for substring, (
                replacement,
                mapped_shard_id,
            ) in self.orig_to_new_stacked.items():
                if substring in key:
                    key = key.replace(substring, replacement, 1)
                    shard_id = mapped_shard_id
            for prefix, replacement in self.orig_to_new_prefix.items():
                if key.startswith(prefix):
                    if replacement is None:
                        return None
                    key = key.replace(prefix, replacement, 1)
            return key, shard_id

        def _map_name(self, key):
            result = self._map_name_with_shard(key)
            return result[0] if result is not None else None

        def apply_list(self, values):
            self.apply_list_calls.append(list(values))
            return [
                mapped
                for value in values
                if (mapped := self._map_name(value)) is not None
            ]

    model_utils_module.WeightsMapper = FakeWeightsMapper
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.utils",
        model_utils_module,
    )
    linear_module = types.ModuleType("vllm.model_executor.layers.linear")
    linear_module.LinearBase = torch.nn.Linear
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.linear",
        linear_module,
    )
    vocab_module = types.ModuleType(
        "vllm.model_executor.layers.vocab_parallel_embedding"
    )
    vocab_module.ParallelLMHead = torch.nn.Linear
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.vocab_parallel_embedding",
        vocab_module,
    )
    routed_experts_module = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.routed_experts"
    )

    class FakeRoutedExperts(torch.nn.Module):
        pass

    routed_experts_module.RoutedExperts = FakeRoutedExperts
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.routed_experts",
        routed_experts_module,
    )
    modelopt_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.modelopt"
    )
    modelopt_module.ModelOptNvFp4Config = type("ModelOptNvFp4Config", (), {})

    class FakeModelOptNvFp4FusedMoE:
        def __init__(self):
            self.quant_config = modelopt_module.ModelOptNvFp4Config()

    class FakeModelOptNvFp4LinearMethod:
        def __init__(self):
            self.quant_config = modelopt_module.ModelOptNvFp4Config()

    modelopt_module.ModelOptNvFp4FusedMoE = FakeModelOptNvFp4FusedMoE
    modelopt_module.ModelOptNvFp4LinearMethod = FakeModelOptNvFp4LinearMethod
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        modelopt_module,
    )
    attention_module = types.ModuleType("vllm.model_executor.layers.attention")
    attention_module.Attention = type("Attention", (torch.nn.Module,), {})
    attention_module.MLAAttention = type("MLAAttention", (torch.nn.Module,), {})
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.attention",
        attention_module,
    )
    kv_cache_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.kv_cache"
    )
    kv_cache_module.BaseKVCacheMethod = type("BaseKVCacheMethod", (), {})
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.kv_cache",
        kv_cache_module,
    )
    return reload_module


def _install_fake_registered_vllm_modelopt(monkeypatch):
    """Install the public vLLM registration surface used by vllm_modelopt."""
    module_names = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.kernels",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.oracle",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.utils",
    )
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)

    registry = {}
    events = []

    quantization_module = sys.modules["vllm.model_executor.layers.quantization"]

    def register_quantization_config(name):
        def register(config_cls):
            registry[name] = config_cls
            return config_cls

        return register

    quantization_module.register_quantization_config = register_quantization_config

    weight_loader_v2_supported = []
    linear_module = types.ModuleType("vllm.model_executor.layers.linear")

    def register_weight_loader_v2_supported_method(method_cls: type) -> type:
        weight_loader_v2_supported.append(method_cls.__name__)
        return method_cls

    linear_module.register_weight_loader_v2_supported_method = (
        register_weight_loader_v2_supported_method
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.linear",
        linear_module,
    )

    class FakeModelOptNvFp4LinearMethod:
        def __init__(self, quant_config):
            self.quant_config = quant_config

        def create_weights(self, layer, *args, **kwargs):
            del args, kwargs
            if not hasattr(layer, "input_scale"):
                layer.input_scale = torch.nn.Parameter(torch.ones(1))

    class FakeModelOptNvFp4FusedMoE:
        def __init__(self, quant_config, moe_config):
            self.quant_config = quant_config
            self.moe = moe_config
            self.moe_kernel = None
            self.moe_quant_config = None
            # Mirror vLLM 0.25: weight-only mode and backend selection are
            # keyed off the config's quant_method in the base __init__.
            self.use_a16 = (
                getattr(quant_config, "quant_method", "NVFP4") == "W4A16_NVFP4"
            )
            self.nvfp4_backend = "marlin"
            self.use_global_sf = False

        def create_weights(self, layer, *args, **kwargs):
            events.append(("native_create_weights", layer, args, kwargs))
            num_experts = args[0] if args else 1
            if not hasattr(layer, "w13_input_scale"):
                layer.w13_input_scale = torch.nn.Parameter(torch.zeros(num_experts, 2))
            if not hasattr(layer, "w2_input_scale"):
                layer.w2_input_scale = torch.nn.Parameter(torch.zeros(num_experts))

        def get_fused_moe_quant_config(self, layer):
            del layer
            return object()

        def process_weights_after_loading(self, layer):
            events.append(
                (
                    "native_process_moe",
                    getattr(
                        getattr(layer, "moe_config", None),
                        "intermediate_size_per_partition",
                        None,
                    ),
                )
            )
            self.moe_kernel = types.SimpleNamespace(source="native", layer=layer)
            w13_input_scale = getattr(layer, "w13_input_scale", None)
            w2_input_scale = getattr(layer, "w2_input_scale", None)
            if isinstance(w13_input_scale, torch.Tensor) and isinstance(
                w2_input_scale, torch.Tensor
            ):
                self.moe_quant_config = types.SimpleNamespace(
                    source="native",
                    a1_gscale=1.0 / w13_input_scale,
                    a2_gscale=1.0 / w2_input_scale,
                )
            else:
                self.moe_quant_config = types.SimpleNamespace(source="native")

    class FakeModelOptNvFp4W4A16LinearMethod(FakeModelOptNvFp4LinearMethod):
        pass

    class FakeModelOptNvFp4Config:
        LinearMethodCls = FakeModelOptNvFp4LinearMethod
        FusedMoEMethodCls = FakeModelOptNvFp4FusedMoE

        def __init__(self, quant_method="NVFP4", group_size=16):
            self.quant_method = quant_method
            self.group_size = group_size
            # Mirror vLLM 0.25: __init__ installs LinearMethodCls as an
            # *instance* attribute keyed off quant_method, shadowing any
            # subclass class attribute.
            if quant_method == "NVFP4":
                self.LinearMethodCls = FakeModelOptNvFp4LinearMethod
            elif quant_method == "W4A16_NVFP4":
                self.LinearMethodCls = FakeModelOptNvFp4W4A16LinearMethod
            else:
                raise ValueError(
                    f"Unsupported ModelOpt NVFP4 quant_algo: {quant_method}"
                )

        @classmethod
        def from_config(cls, config):
            target = config.get("quantization", config)
            instance = cls(
                quant_method=str(target.get("quant_algo", "NVFP4")).upper(),
                group_size=target.get("group_size", 16),
            )
            instance.parsed_config = config
            return instance

        @classmethod
        def _extract_modelopt_quant_algo(cls, config):
            del cls
            target = config.get("quantization", config)
            return str(target.get("quant_algo", "")).upper()

    modelopt_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.modelopt"
    )
    modelopt_module.ModelOptNvFp4Config = FakeModelOptNvFp4Config
    modelopt_module.ModelOptNvFp4LinearMethod = FakeModelOptNvFp4LinearMethod
    modelopt_module.ModelOptNvFp4FusedMoE = FakeModelOptNvFp4FusedMoE
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        modelopt_module,
    )

    class FakeFusedMoEMethodBase:
        def __init__(self, moe_config):
            self.moe = moe_config
            self.moe_kernel = None
            self.moe_quant_config = None

    fused_method_module = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base"
    )
    fused_method_module.FusedMoEMethodBase = FakeFusedMoEMethodBase
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base",
        fused_method_module,
    )

    class FakeNvFp4LinearLayerConfig:
        pass

    class FakeMarlinNvFp4LinearKernel:
        def __init__(self, config):
            self.config = config

        def process_weights_after_loading(self, layer):
            events.append(("process_marlin_kernel", layer))

        def apply_weights(self, **kwargs):
            events.append(("apply_marlin_kernel", kwargs))
            return "output"

    linear_kernel_module = types.ModuleType("vllm.model_executor.kernels.linear")
    linear_kernel_module.MarlinNvFp4LinearKernel = FakeMarlinNvFp4LinearKernel
    linear_kernel_module.NvFp4LinearLayerConfig = FakeNvFp4LinearLayerConfig
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.kernels.linear",
        linear_kernel_module,
    )

    class FakeMarlinExperts:
        pass

    oracle_module = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4"
    )
    oracle_module.NvFp4MoeBackend = types.SimpleNamespace(MARLIN="marlin")

    def convert_to_nvfp4_moe_kernel_format(**kwargs):
        events.append(
            (
                "convert_moe",
                kwargs["layer"].moe_config.intermediate_size_per_partition,
            )
        )
        return tuple(
            kwargs[name]
            for name in (
                "w13",
                "w13_scale",
                "w13_scale_2",
                "a13_scale",
                "w2",
                "w2_scale",
                "w2_scale_2",
                "a2_scale",
            )
        )

    oracle_module.convert_to_nvfp4_moe_kernel_format = (
        convert_to_nvfp4_moe_kernel_format
    )
    oracle_module.is_global_sf_supported_for_nvfp4_backend = lambda backend: False
    oracle_module.select_nvfp4_moe_backend = lambda **kwargs: (
        oracle_module.NvFp4MoeBackend.MARLIN,
        FakeMarlinExperts,
    )
    oracle_module.make_nvfp4_moe_kernel = lambda **kwargs: (
        events.append(("make_moe_kernel", kwargs))
        or types.SimpleNamespace(
            fused_experts=types.SimpleNamespace(
                process_weights_after_loading=lambda layer: events.append(
                    ("process_moe", layer)
                )
            )
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
        oracle_module,
    )

    quant_utils_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.utils.quant_utils"
    )
    quant_utils_module.kNvfp4Static = object()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        quant_utils_module,
    )

    utils_module = types.ModuleType("vllm.model_executor.utils")

    def replace_parameter(layer, name, value):
        if name in layer._parameters:
            del layer._parameters[name]
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))

    utils_module.replace_parameter = replace_parameter
    monkeypatch.setitem(sys.modules, "vllm.model_executor.utils", utils_module)
    return types.SimpleNamespace(
        registry=registry,
        events=events,
        weight_loader_v2_supported=weight_loader_v2_supported,
    )


def _install_fake_modelopt_tensor_quantizer(monkeypatch):
    """Install the minimal ModelOpt module hierarchy needed by vLLM backend import."""
    module_names = [
        "modelopt",
        "modelopt.torch",
        "modelopt.torch.quantization",
        "modelopt.torch.quantization.nn",
        "modelopt.torch.quantization.nn.modules",
    ]
    modules = {}
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        modules[module_name] = module
        monkeypatch.setitem(sys.modules, module_name, module)

    tensor_quantizer_module = types.ModuleType(
        "modelopt.torch.quantization.nn.modules.tensor_quantizer"
    )

    class FakeTensorQuantizer(torch.nn.Module):
        pass

    tensor_quantizer_module.TensorQuantizer = FakeTensorQuantizer
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch.quantization.nn.modules.tensor_quantizer",
        tensor_quantizer_module,
    )
    modules["modelopt"].torch = modules["modelopt.torch"]
    modules["modelopt.torch"].quantization = modules["modelopt.torch.quantization"]
    modules["modelopt.torch.quantization"].nn = modules[
        "modelopt.torch.quantization.nn"
    ]
    modules["modelopt.torch.quantization.nn"].modules = modules[
        "modelopt.torch.quantization.nn.modules"
    ]
    modules[
        "modelopt.torch.quantization.nn.modules"
    ].tensor_quantizer = tensor_quantizer_module


def _make_real_quant_extension(
    backend,
    model,
    ignore,
    *,
    quant_algo="W4A16_NVFP4",
    model_id="org/model",
    model_revision="0123456789abcdef",
    resolved_revision="0123456789abcdef",
):
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.device = torch.device("cpu")
    extension._nrl_w13_num_shards_by_prefix = {}
    hf_config = types.SimpleNamespace(
        quantization_config={"ignore": ignore, "quant_algo": quant_algo},
        _commit_hash=resolved_revision,
    )
    model_config = types.SimpleNamespace(
        model=model_id,
        revision=model_revision,
        hf_config=hf_config,
    )
    extension.model_config = model_config
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(enable_expert_parallel=False),
            model_config=model_config,
        ),
    )
    return extension


def _patch_real_quant_load(monkeypatch, backend, forwarded=None):
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    if forwarded is not None:
        monkeypatch.setattr(
            backend.VllmInternalWorkerExtension,
            "_load_weights",
            lambda self, weights: forwarded.extend(weights) or "loaded",
        )


def _bf16_weight_info(*names: str) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {name: ((32, 16), torch.bfloat16) for name in names}


def _write_calibration_artifact(
    path: Path,
    projection_amax: dict[str, float],
    *,
    model_id: str = "org/model",
    model_revision: str = "0123456789abcdef",
    quant_cfg: str = "NVFP4_DEFAULT_CFG",
) -> None:
    save_nvfp4_calibration(
        path,
        {name: torch.tensor(amax) for name, amax in projection_amax.items()},
        model_id=model_id,
        model_revision=model_revision,
        quant_cfg=quant_cfg,
        dataset="cnn_dailymail",
        sample_count=16,
        sequence_length=1024,
        seed=1234,
    )


def _packed_weight_info(
    prefix: str,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        f"{prefix}.weight": ((32, 8), torch.uint8),
        f"{prefix}.weight_scale": ((32, 1), torch.float8_e4m3fn),
        f"{prefix}.weight_scale_2": ((), torch.float32),
    }


def _packed_moe_info(
    prefix: str,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        f"{prefix}.experts.w13_weight": ((2, 4, 3), torch.uint8),
        f"{prefix}.experts.w13_weight_scale": ((2, 4, 1), torch.uint8),
        f"{prefix}.experts.w13_weight_scale_2": ((2, 2), torch.float32),
        f"{prefix}.experts.w2_weight": ((2, 3, 2), torch.uint8),
        f"{prefix}.experts.w2_weight_scale": ((2, 3, 1), torch.uint8),
        f"{prefix}.experts.w2_weight_scale_2": ((2,), torch.float32),
    }


def _mark_as_modelopt_layer(model):
    modelopt_module = sys.modules["vllm.model_executor.layers.quantization.modelopt"]
    model.quant_method = modelopt_module.ModelOptNvFp4LinearMethod()
    return model


def _mark_as_modelopt_moe(model):
    modelopt_module = sys.modules["vllm.model_executor.layers.quantization.modelopt"]
    model.quant_method = modelopt_module.ModelOptNvFp4FusedMoE()
    return model


def _new_modelopt_moe():
    routed_experts = sys.modules[
        "vllm.model_executor.layers.fused_moe.routed_experts"
    ].RoutedExperts()
    return _mark_as_modelopt_moe(routed_experts)


def test_real_quant_discovers_custom_w4a16_linear_reload_root(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    modelopt_module = sys.modules["vllm.model_executor.layers.quantization.modelopt"]
    linear_base = sys.modules["vllm.model_executor.layers.linear"].LinearBase

    class NemoModelOptW4A16LinearMethod:
        def __init__(self):
            self.quant_config = modelopt_module.ModelOptNvFp4Config()

    model = torch.nn.Module()
    model.q_proj = linear_base(16, 32, bias=False)
    model.q_proj.quant_method = NemoModelOptW4A16LinearMethod()

    assert backend._iter_modelopt_quant_modules(model) == [("q_proj", model.q_proj)]


def test_real_quant_target_resolver_handles_fused_linear_mapper_variants(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    modelopt_module = sys.modules["vllm.model_executor.layers.quantization.modelopt"]
    linear_base = sys.modules["vllm.model_executor.layers.linear"].LinearBase
    weights_mapper = sys.modules["vllm.model_executor.models.utils"].WeightsMapper

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].self_attn = torch.nn.Module()
    model.model.layers[0].self_attn.qkv_proj = linear_base(16, 32, bias=False)
    model.model.layers[0].self_attn.qkv_proj.quant_method = types.SimpleNamespace(
        quant_config=modelopt_module.ModelOptNvFp4Config()
    )
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.gate_up_proj = linear_base(16, 32, bias=False)
    model.model.layers[0].mlp.gate_up_proj.quant_method = types.SimpleNamespace(
        quant_config=modelopt_module.ModelOptNvFp4Config()
    )
    model.model.layers[0].mlp.shared_expert = torch.nn.Module()
    model.model.layers[0].mlp.shared_expert.gate_up_proj = linear_base(
        16, 32, bias=False
    )
    model.model.layers[
        0
    ].mlp.shared_expert.gate_up_proj.quant_method = types.SimpleNamespace(
        quant_config=modelopt_module.ModelOptNvFp4Config()
    )
    model.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }
    model.hf_to_vllm_mapper = weights_mapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        },
        orig_to_new_prefix={"decoder.": "model."},
    )

    assert backend._is_bf16_quantization_candidate(
        "decoder.layers.0.self_attn.q_proj.weight",
        (32, 16),
        model=model,
    )
    assert backend._is_bf16_quantization_candidate(
        "model.layers.0.mlp.gate_proj.weight",
        (32, 16),
        model=model,
    )
    assert backend._is_bf16_quantization_candidate(
        "model.layers.0.mlp.shared_expert.up_proj.weight",
        (32, 16),
        model=model,
    )
    mapped_inputs = {
        name for call in model.hf_to_vllm_mapper.apply_list_calls for name in call
    }
    assert "decoder.layers.0.self_attn.q_proj.weight" in mapped_inputs
    assert "model.layers.0.mlp.gate_proj.weight" in mapped_inputs
    assert "model.layers.0.mlp.shared_expert.up_proj.weight" in mapped_inputs
    assert backend._iter_modelopt_quant_modules(model) == [
        (
            "model.layers.0.self_attn.qkv_proj",
            model.model.layers[0].self_attn.qkv_proj,
        ),
        ("model.layers.0.mlp.gate_up_proj", model.model.layers[0].mlp.gate_up_proj),
        (
            "model.layers.0.mlp.shared_expert.gate_up_proj",
            model.model.layers[0].mlp.shared_expert.gate_up_proj,
        ),
    ]


def test_real_quant_mapper_drop_on_original_name_is_authoritative(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    modelopt_module = sys.modules["vllm.model_executor.layers.quantization.modelopt"]
    linear_base = sys.modules["vllm.model_executor.layers.linear"].LinearBase
    weights_mapper = sys.modules["vllm.model_executor.models.utils"].WeightsMapper

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].self_attn = torch.nn.Module()
    model.model.layers[0].self_attn.qkv_proj = linear_base(16, 32, bias=False)
    model.model.layers[0].self_attn.qkv_proj.quant_method = types.SimpleNamespace(
        quant_config=modelopt_module.ModelOptNvFp4Config()
    )
    model.hf_to_vllm_mapper = weights_mapper(
        orig_to_new_stacked={".q_proj": (".qkv_proj", "q")},
        orig_to_new_prefix={"layers.": None},
    )
    original_name = "layers.0.self_attn.q_proj.weight"
    prefixed_name = f"model.{original_name}"

    assert model.hf_to_vllm_mapper.apply_list([original_name]) == []
    assert model.hf_to_vllm_mapper.apply_list([prefixed_name]) == [
        "model.layers.0.self_attn.qkv_proj.weight"
    ]
    assert backend._mapped_weight_name_variants(model, original_name) == set()
    assert not backend._is_bf16_quantization_candidate(
        original_name,
        (32, 16),
        model=model,
    )


def test_real_quant_target_resolver_handles_routed_experts_and_passthrough_embedding(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    modelopt_module = sys.modules["vllm.model_executor.layers.quantization.modelopt"]
    routed_experts = sys.modules[
        "vllm.model_executor.layers.fused_moe.routed_experts"
    ].RoutedExperts

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = routed_experts()
    model.model.layers[0].mlp.experts.quant_method = types.SimpleNamespace(
        quant_config=modelopt_module.ModelOptNvFp4Config()
    )
    model.embed_tokens = torch.nn.Embedding(16, 32)
    model.lm_head = torch.nn.Linear(32, 16, bias=False)
    model.lm_head.quant_method = types.SimpleNamespace(quant_config=None)

    assert backend._is_bf16_quantization_candidate(
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        (32, 16),
        model=model,
    )
    assert not backend._is_bf16_quantization_candidate(
        "embed_tokens.weight",
        (16, 32),
        model=model,
    )
    assert not backend._is_bf16_quantization_candidate(
        "lm_head.weight",
        (16, 32),
        model=model,
    )


def test_base_ipc_data_ack_fence_synchronizes_current_stream_once(monkeypatch):
    _import_vllm_quant_backend(monkeypatch)
    backend = _base_vllm_backend()
    extension = object.__new__(backend.VllmInternalWorkerExtension)
    calls = []
    stream = types.SimpleNamespace(synchronize=lambda: calls.append("sync"))
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_stream",
        lambda: calls.append("current_stream") or stream,
    )

    extension._synchronize_before_ipc_data_ack()

    assert calls == ["current_stream", "sync"]


def test_w4a16_real_quant_config_is_weight_only():
    cfg = build_vllm_modelopt_nvfp4_config(mode="w4a16")

    group = cfg["config_groups"]["group_0"]
    assert cfg["quant_method"] == "modelopt"
    assert cfg["quant_algo"] == "W4A16_NVFP4"
    assert "input_activations" not in group
    assert group["weights"] == {
        "dynamic": False,
        "num_bits": 4,
        "type": "float",
        "group_size": 16,
    }
    assert cfg["ignore"] == [
        "lm_head",
        "*output_layer*",
        "*mlp.gate",
        "*router*",
        "*block_sparse_moe.gate*",
        "*self_attention*",
        "*self_attn*",
    ]


def test_w4a4_real_quant_config_has_static_input_activations():
    cfg = build_vllm_modelopt_nvfp4_config(mode="w4a4")

    group = cfg["config_groups"]["group_0"]
    assert cfg["quant_method"] == "modelopt"
    assert cfg["quant_algo"] == "NVFP4"
    assert group["input_activations"] == {
        "dynamic": False,
        "num_bits": 4,
        "type": "float",
        "group_size": 16,
    }


def test_real_quant_config_rejects_unsupported_mode():
    with pytest.raises(ValueError, match="expected 'w4a4' or 'w4a16'"):
        build_vllm_modelopt_nvfp4_config(mode="w4a8")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "method"),
    [("w4a4", NEMO_MODELOPT_W4A4), ("w4a16", NEMO_MODELOPT_W4A16)],
)
def test_quantization_method_for_mode_uses_registered_names(mode, method):
    assert quantization_method_for_mode(mode) == method


def test_quantization_method_for_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported ModelOpt NVFP4 rollout mode"):
        quantization_method_for_mode("w4a8")


def test_real_quant_config_allows_explicit_ignore_override():
    ignore = ["lm_head", "*.mixer.in_proj*"]
    cfg = build_vllm_modelopt_nvfp4_config(mode="w4a16", ignore=ignore)

    assert cfg["ignore"] == ignore
    assert matches_quant_ignore_pattern(
        "model.layers.0.mixer.in_proj.weight",
        cfg["ignore"],
    )


def test_default_ignore_patterns_match_expected_layers():
    ignore_patterns = build_vllm_modelopt_nvfp4_config(mode="w4a16")["ignore"]

    assert matches_quant_ignore_pattern(
        "model.layers.0.self_attn.o_proj.weight", ignore_patterns
    )
    assert matches_quant_ignore_pattern(
        "layers.0.self_attn.o_proj.weight", ignore_patterns
    )
    assert matches_quant_ignore_pattern(
        "model.layers.0.mlp.gate.weight", ignore_patterns
    )
    assert matches_quant_ignore_pattern("model.layers.0.router.weight", ignore_patterns)
    assert matches_quant_ignore_pattern("lm_head.weight", ignore_patterns)
    assert matches_quant_ignore_pattern(
        "model.layers.0.mlp.gate.weight_scale", ignore_patterns
    )
    assert matches_quant_ignore_pattern(
        "model.layers.0.mlp.gate.input_scale", ignore_patterns
    )
    assert not matches_quant_ignore_pattern(
        "model.layers.0.mlp.experts.0.w1.weight", ignore_patterns
    )


def test_quant_ignore_name_candidates_include_model_prefix_and_base_names():
    assert list(
        iter_quant_ignore_name_candidates("layers.0.self_attn.q_proj.weight")
    ) == [
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj",
    ]
    assert list(iter_quant_ignore_name_candidates("model.lm_head.weight_scale")) == [
        "model.lm_head.weight_scale",
        "model.lm_head",
        "lm_head.weight_scale",
        "lm_head",
    ]
    assert list(iter_quant_ignore_name_candidates("model.lm_head.input_scale")) == [
        "model.lm_head.input_scale",
        "model.lm_head",
        "lm_head.input_scale",
        "lm_head",
    ]


def test_configure_quant_engine_kwargs_for_fake_quant(monkeypatch, tmp_path):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)

    quant_cfg = "quant.yaml"
    (tmp_path / quant_cfg).touch()
    monkeypatch.chdir(tmp_path)

    llm_kwargs = {}
    worker_mod._configure_quant_engine_kwargs(
        {"quant_cfg": quant_cfg},
        llm_kwargs,
    )

    assert llm_kwargs["worker_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
    )
    assert llm_kwargs["worker_extension_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_backend.VllmQuantInternalWorkerExtension"
    )
    assert os.environ["VLLM_QUANT_CFG"] == os.path.abspath(quant_cfg)
    assert "quantization" not in llm_kwargs


def test_quant_worker_forwards_snapshot_pythonpath_to_inner_vllm_workers():
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )

    assert "PYTHONPATH" in worker_mod._EXTRA_ENV_VARS


def test_configure_quant_engine_kwargs_preserves_checkpoint_extension(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    cfg = {
        "quant_cfg": "examples/modelopt/quant_configs/nvfp4_w4a8_fp8.yaml",
        "refit_transport": "nixl",
        "refit_cfg": {"nixl": {}},
    }
    llm_kwargs = {}

    worker_mod._configure_quant_engine_kwargs(cfg, llm_kwargs)

    assert llm_kwargs["worker_extension_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_backend."
        "VllmQuantInternalWorkerExtensionWithCheckpointEngine"
    )


def test_fake_quant_worker_inherits_nixl_worker():
    patch_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_patch"
    )
    from nemo_rl.models.generation.vllm.vllm_backend import NixlVllmWorker

    assert issubclass(patch_mod.FakeQuantWorker, NixlVllmWorker)


def test_configure_quant_engine_kwargs_for_real_quant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    registration_calls = []
    monkeypatch.setattr(
        vllm_modelopt,
        "register_nemo_modelopt_nvfp4",
        lambda: registration_calls.append(True),
    )
    resolved_quant_cfg: list[str] = []

    def resolve_mode(quant_cfg: str) -> str:
        resolved_quant_cfg.append(quant_cfg)
        return "w4a16"

    monkeypatch.setattr(modelopt_utils, "resolve_nvfp4_real_quant_mode", resolve_mode)
    monkeypatch.chdir(tmp_path)

    llm_kwargs = {}
    worker_mod._configure_quant_engine_kwargs(
        {
            "quant_cfg": "examples/modelopt/quant_configs/nvfp4_a16_mlp_only.yaml",
            "real_quant": True,
            "real_quant_ignore": ["lm_head"],
        },
        llm_kwargs,
    )

    assert registration_calls == [True]
    assert resolved_quant_cfg == [
        str(
            (
                Path(worker_mod.__file__).resolve().parents[4]
                / "examples/modelopt/quant_configs/nvfp4_a16_mlp_only.yaml"
            ).resolve()
        )
    ]
    assert os.environ["VLLM_MODELOPT_REAL_QUANT"] == "1"
    assert "VLLM_QUANT_CFG" not in os.environ
    assert "worker_cls" not in llm_kwargs
    assert llm_kwargs["quantization"] == NEMO_MODELOPT_W4A16
    assert llm_kwargs["hf_overrides"]["quantization_config"] == (
        build_vllm_modelopt_nvfp4_config(mode="w4a16", ignore=["lm_head"])
    )


@pytest.mark.parametrize("mode", ["w4a4", "w4a16"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_configure_real_quant_preserves_kv_cache_dtype(
    monkeypatch,
    mode,
    kv_cache_dtype,
):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.setattr(vllm_modelopt, "register_nemo_modelopt_nvfp4", lambda: None)
    monkeypatch.setattr(
        modelopt_utils,
        "resolve_nvfp4_real_quant_mode",
        lambda _: mode,
    )

    llm_kwargs = {"kv_cache_dtype": kv_cache_dtype}
    worker_mod._configure_quant_engine_kwargs(
        {"quant_cfg": "NVFP4_EXPERTS_ONLY_CFG", "real_quant": True},
        llm_kwargs,
    )

    assert llm_kwargs["kv_cache_dtype"] == kv_cache_dtype
    assert llm_kwargs["quantization"] == quantization_method_for_mode(mode)
    assert "kv_cache" not in llm_kwargs["hf_overrides"]["quantization_config"]


def test_configure_quant_engine_kwargs_preserves_hf_overrides(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    monkeypatch.setattr(vllm_modelopt, "register_nemo_modelopt_nvfp4", lambda: None)
    monkeypatch.setattr(
        modelopt_utils, "resolve_nvfp4_real_quant_mode", lambda _: "w4a16"
    )

    llm_kwargs = {"hf_overrides": {"trust_remote_code": True}}
    worker_mod._configure_quant_engine_kwargs(
        {"quant_cfg": "NVFP4_DEFAULT_CFG", "real_quant": True},
        llm_kwargs,
    )

    assert llm_kwargs["hf_overrides"]["trust_remote_code"] is True
    assert (
        llm_kwargs["hf_overrides"]["quantization_config"]["quant_method"] == "modelopt"
    )


def test_configure_quant_engine_kwargs_for_fake_quant_without_quant_cfg(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    monkeypatch.delenv("VLLM_QUANT_CFG", raising=False)
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)

    llm_kwargs = {}
    worker_mod._configure_quant_engine_kwargs({"quant_cfg": None}, llm_kwargs)

    assert "VLLM_QUANT_CFG" not in os.environ
    assert llm_kwargs["worker_cls"] == (
        "nemo_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
    )


def test_quant_generation_worker_create_engine_configures_quant(monkeypatch):
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    worker_cls = worker_mod.VllmQuantGenerationWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.cfg = {"quant_cfg": None}
    calls = []

    def fake_configure(cfg, llm_kwargs):
        calls.append(("configure", cfg, llm_kwargs))
        llm_kwargs["configured"] = True

    def fake_base_create_engine(self, llm_kwargs):
        calls.append(("base", dict(llm_kwargs)))

    monkeypatch.setattr(worker_mod, "_configure_quant_engine_kwargs", fake_configure)
    monkeypatch.setattr(
        worker_mod.VllmGenerationWorkerImpl,
        "_create_engine",
        fake_base_create_engine,
    )

    llm_kwargs = {}
    worker._create_engine(llm_kwargs)

    assert calls == [
        ("configure", worker.cfg, {"configured": True}),
        ("base", {"configured": True}),
    ]


def test_quant_generation_worker_collective_rpc_accessors():
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    worker_cls = worker_mod.VllmQuantGenerationWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    calls = []

    class FakeLLM:
        def collective_rpc(self, name, args):
            calls.append((name, args))
            return [{"name": name, "args": args}]

    worker.llm = FakeLLM()

    assert worker.get_quantizer_stats() == {
        "name": "get_quantizer_stats",
        "args": tuple(),
    }
    assert worker.get_weight_snapshot("weight") == {
        "name": "get_weight_snapshot",
        "args": ("weight",),
    }
    assert calls == [
        ("get_quantizer_stats", tuple()),
        ("get_weight_snapshot", ("weight",)),
    ]


@pytest.mark.asyncio
async def test_async_quant_generation_worker_collective_rpc_accessors():
    worker_mod = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker"
    )
    worker_cls = (
        worker_mod.VllmQuantAsyncGenerationWorker.__ray_metadata__.modified_class
    )
    worker = object.__new__(worker_cls)
    calls = []

    class FakeLLM:
        async def collective_rpc(self, name, args):
            calls.append((name, args))
            return [{"name": name, "args": args}]

    worker.llm = FakeLLM()

    assert await worker.get_quantizer_stats() == {
        "name": "get_quantizer_stats",
        "args": tuple(),
    }
    assert await worker.get_weight_snapshot("weight") == {
        "name": "get_weight_snapshot",
        "args": ("weight",),
    }
    assert calls == [
        ("get_quantizer_stats", tuple()),
        ("get_weight_snapshot", ("weight",)),
    ]


def test_vllm_modelopt_backend_imports_without_gpt_oss_helper(monkeypatch):
    _import_vllm_quant_backend(monkeypatch)


def test_real_quant_backend_uses_modelopt_refit_timeout(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    events = []

    class FakeSocket:
        def setsockopt(self, option, value):
            events.append(("setsockopt", option, value))

        def connect(self, address):
            events.append(("connect", address))

    class FakeContext:
        def socket(self, socket_type):
            events.append(("socket", socket_type))
            return FakeSocket()

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.get_zmq_address = lambda: "ipc:///tmp/modelopt-test.sock"
    monkeypatch.setattr(backend.zmq, "Context", FakeContext)
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )

    extension.maybe_init_zmq()

    assert events[0] == ("socket", backend.zmq.REP)
    assert ("setsockopt", backend.zmq.LINGER, 0) in events
    assert ("connect", "ipc:///tmp/modelopt-test.sock") in events
    assert events[-2:] == [
        (
            "setsockopt",
            backend.zmq.SNDTIMEO,
            modelopt_utils.MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS,
        ),
        (
            "setsockopt",
            backend.zmq.RCVTIMEO,
            modelopt_utils.MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS,
        ),
    ]


def test_vllm_modelopt_backend_registers_real_quant_configs_on_import(monkeypatch):
    calls = []

    monkeypatch.setenv("VLLM_MODELOPT_REAL_QUANT", "1")
    _install_fake_vllm_worker(monkeypatch)
    _install_fake_modelopt_tensor_quantizer(monkeypatch)
    monkeypatch.setattr(
        vllm_modelopt,
        "register_nemo_modelopt_nvfp4",
        lambda: calls.append("registered"),
    )
    _clear_vllm_backend_modules(monkeypatch)

    importlib.import_module("nemo_rl.modelopt.models.generation.vllm_quant_backend")

    assert calls == ["registered"]


def test_modelopt_moe_manifest_requires_complete_w4a4_family(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    prefix = "model.layers.0.mixer"
    state_dict_info = {
        f"{prefix}.experts.w13_weight": ((2, 4, 3), torch.uint8),
        f"{prefix}.experts.w13_weight_scale": ((2, 4, 1), torch.uint8),
        f"{prefix}.experts.w13_weight_scale_2": ((2, 2), torch.float32),
        f"{prefix}.experts.w13_input_scale": ((2, 2), torch.float32),
        f"{prefix}.experts.w2_weight": ((2, 3, 4), torch.uint8),
        f"{prefix}.experts.w2_weight_scale": ((2, 1, 4), torch.uint8),
        f"{prefix}.experts.w2_weight_scale_2": ((2,), torch.float32),
        f"{prefix}.experts.w2_input_scale": ((2,), torch.float32),
    }

    assert backend._w13_num_shards_from_state_dict_info(
        state_dict_info, require_input_scales=True
    ) == {prefix: 2}

    legacy_state_dict_info = dict(state_dict_info)
    legacy_state_dict_info[f"{prefix}.experts.w13_weight_scale_2"] = (
        (2,),
        torch.float32,
    )
    legacy_state_dict_info[f"{prefix}.experts.w13_input_scale"] = (
        (2,),
        torch.float32,
    )
    assert backend._w13_num_shards_from_state_dict_info(
        legacy_state_dict_info, require_input_scales=True
    ) == {prefix: 1}

    mismatched_state_dict_info = dict(state_dict_info)
    mismatched_state_dict_info[f"{prefix}.experts.w13_input_scale"] = (
        (2, 1),
        torch.float32,
    )
    with pytest.raises(RuntimeError, match="input/global scale layouts disagree"):
        backend._w13_num_shards_from_state_dict_info(
            mismatched_state_dict_info, require_input_scales=True
        )

    del state_dict_info[f"{prefix}.experts.w2_input_scale"]
    with pytest.raises(RuntimeError, match="missing.*w2_input_scale"):
        backend._w13_num_shards_from_state_dict_info(
            state_dict_info, require_input_scales=True
        )


def test_real_quant_prepare_refit_classifies_bf16_manifest(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
    )
    _patch_real_quant_load(monkeypatch, backend)

    requests = extension.prepare_refit_info(
        _bf16_weight_info(
            "q_proj.weight",
        )
    )

    assert extension._nrl_real_quant_source == "bf16"
    assert extension._nrl_bf16_staging == {}
    assert len(requests) == 1
    assert requests[0].parameter_names == ("q_proj.weight",)
    assert requests[0].source_format == "bf16"
    assert requests[0].target_format == "nvfp4_w4a16"


def test_real_quant_bf16_w4a4_prepare_requires_calibration_path(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    monkeypatch.delenv("VLLM_MODELOPT_CALIBRATION_PATH", raising=False)
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "NVFP4_DEFAULT_CFG",
    )

    with pytest.raises(ValueError, match="VLLM_MODELOPT_CALIBRATION_PATH"):
        extension.prepare_refit_info(_bf16_weight_info("q_proj.weight"))


def test_real_quant_bf16_w4a4_prepare_requires_calibration_quant_cfg(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_PATH",
        "/worker/calibration.safetensors",
    )
    monkeypatch.delenv("VLLM_MODELOPT_CALIBRATION_QUANT_CFG", raising=False)

    with pytest.raises(ValueError, match="VLLM_MODELOPT_CALIBRATION_QUANT_CFG"):
        extension.prepare_refit_info(_bf16_weight_info("q_proj.weight"))


def test_real_quant_bf16_w4a4_prepare_rejects_missing_calibration_file(
    monkeypatch,
    tmp_path,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    missing_path = tmp_path / "missing.safetensors"
    monkeypatch.setenv("VLLM_MODELOPT_CALIBRATION_PATH", str(missing_path))
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "NVFP4_DEFAULT_CFG",
    )

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        extension.prepare_refit_info(_bf16_weight_info("q_proj.weight"))


@pytest.mark.parametrize(
    ("projection_amax", "expected_error"),
    [
        ({}, "missing.*q_proj.weight"),
        (
            {"q_proj.weight": 12.0, "k_proj.weight": 24.0},
            "unexpected.*k_proj.weight",
        ),
    ],
    ids=("missing-projection", "unexpected-projection"),
)
def test_real_quant_bf16_w4a4_prepare_requires_exact_calibration_targets(
    monkeypatch,
    tmp_path,
    projection_amax,
    expected_error,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    artifact_path = tmp_path / "calibration.safetensors"
    artifact_values = projection_amax or {"k_proj.weight": 24.0}
    _write_calibration_artifact(artifact_path, artifact_values)
    monkeypatch.setenv("VLLM_MODELOPT_CALIBRATION_PATH", str(artifact_path))
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "NVFP4_DEFAULT_CFG",
    )

    with pytest.raises(ValueError, match=expected_error):
        extension.prepare_refit_info(_bf16_weight_info("q_proj.weight"))


@pytest.mark.parametrize(
    ("artifact_identity", "expected_error"),
    [
        ({"model_id": "other/model"}, "model_id"),
        ({"model_revision": "other-revision"}, "model_revision"),
        ({"quant_cfg": "OTHER_CFG"}, "quant_cfg"),
    ],
    ids=("model", "revision", "quant-config"),
)
def test_real_quant_bf16_w4a4_prepare_rejects_calibration_identity_mismatch(
    monkeypatch,
    tmp_path,
    artifact_identity,
    expected_error,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    artifact_path = tmp_path / "calibration.safetensors"
    _write_calibration_artifact(
        artifact_path,
        {"q_proj.weight": 12.0},
        **artifact_identity,
    )
    monkeypatch.setenv("VLLM_MODELOPT_CALIBRATION_PATH", str(artifact_path))
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "NVFP4_DEFAULT_CFG",
    )

    with pytest.raises(ValueError, match=expected_error):
        extension.prepare_refit_info(_bf16_weight_info("q_proj.weight"))


def test_real_quant_bf16_w4a4_prepare_requires_explicit_model_revision(
    monkeypatch,
    tmp_path,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
        model_revision=None,
        resolved_revision="0123456789abcdef",
    )
    _patch_real_quant_load(monkeypatch, backend)
    artifact_path = tmp_path / "calibration.safetensors"
    _write_calibration_artifact(artifact_path, {"q_proj.weight": 12.0})
    monkeypatch.setenv("VLLM_MODELOPT_CALIBRATION_PATH", str(artifact_path))
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "NVFP4_DEFAULT_CFG",
    )

    with pytest.raises(ValueError, match="explicit model revision"):
        extension.prepare_refit_info(_bf16_weight_info("q_proj.weight"))


def test_real_quant_bf16_w4a4_prepare_uses_resolved_vllm_provenance_once(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
        model_revision="release-tag",
        resolved_revision="0123456789abcdef",
    )
    _patch_real_quant_load(monkeypatch, backend)
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_PATH",
        "/worker/calibration.safetensors",
    )
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "NVFP4_DEFAULT_CFG",
    )
    calibration = NVFP4Calibration(input_amax={"q_proj.weight": torch.tensor(12.0)})
    calls = []

    def load_calibration(path, **kwargs):
        calls.append((path, kwargs))
        return calibration

    monkeypatch.setattr(
        backend,
        "load_nvfp4_calibration",
        load_calibration,
        raising=False,
    )

    extension.prepare_refit_info(_bf16_weight_info("q_proj.weight"))

    assert calls == [
        (
            "/worker/calibration.safetensors",
            {
                "model_id": "org/model",
                "model_revision": "0123456789abcdef",
                "quant_cfg": "NVFP4_DEFAULT_CFG",
                "expected_projection_names": {"q_proj.weight"},
            },
        )
    ]
    assert extension._nrl_bf16_calibration is calibration


def test_exported_tag_revision_round_trips_with_resolved_vllm_commit(
    monkeypatch,
    tmp_path,
):
    resolved_revision = "0123456789abcdef0123456789abcdef01234567"
    real_transformers = importlib.import_module("transformers")

    class FakeProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((32, 16)))
            self.input_quantizer = types.SimpleNamespace(
                is_enabled=True,
                _amax=torch.tensor(12.0),
            )

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(_commit_hash=resolved_revision)
            self.q_proj = FakeProjection()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeModel()

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    algorithms_utils = types.ModuleType("nemo_rl.algorithms.utils")
    algorithms_utils.set_seed = lambda _seed: None
    worker_utils = types.ModuleType("nemo_rl.modelopt.models.policy.workers.utils")
    worker_utils.get_tokenizer = lambda *_args, **_kwargs: types.SimpleNamespace(
        init_kwargs={"_commit_hash": resolved_revision}
    )
    worker_utils.quantize_model = lambda **_kwargs: None
    exporter_modelopt_utils = types.ModuleType("nemo_rl.modelopt.utils")
    exporter_modelopt_utils.resolve_nvfp4_real_quant_mode = lambda _cfg: "w4a4"
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "nemo_rl.algorithms.utils", algorithms_utils)
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.modelopt.models.policy.workers.utils",
        worker_utils,
    )
    monkeypatch.setitem(sys.modules, "nemo_rl.modelopt.utils", exporter_modelopt_utils)
    quant_cfg_path = tmp_path / "nvfp4.yaml"
    quant_cfg_path.write_text("quant_cfg: nvfp4\n")
    artifact_path = tmp_path / "calibration.safetensors"

    export_nvfp4_calibration.main(
        [
            "--model",
            "org/model",
            "--model-revision",
            "release-tag",
            "--quant-cfg",
            str(quant_cfg_path),
            "--dataset",
            "cnn_dailymail",
            "--sample-count",
            "1",
            "--sequence-length",
            "16",
            "--seed",
            "1234",
            "--output",
            str(artifact_path),
        ]
    )

    monkeypatch.setitem(sys.modules, "transformers", real_transformers)
    monkeypatch.setitem(sys.modules, "nemo_rl.modelopt.utils", modelopt_utils)
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
        model_revision="release-tag",
        resolved_revision=resolved_revision,
    )
    _patch_real_quant_load(monkeypatch, backend)
    monkeypatch.setenv("VLLM_MODELOPT_CALIBRATION_PATH", str(artifact_path))
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        str(quant_cfg_path.resolve()),
    )

    extension.prepare_refit_info(_bf16_weight_info("q_proj.weight"))

    assert extension._nrl_bf16_calibration is not None
    assert set(extension._nrl_bf16_calibration.input_amax) == {"q_proj.weight"}


def test_real_quant_prepacked_w4a4_keeps_actor_scales_without_artifact(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    monkeypatch.delenv("VLLM_MODELOPT_CALIBRATION_PATH", raising=False)
    monkeypatch.delenv("VLLM_MODELOPT_CALIBRATION_QUANT_CFG", raising=False)
    monkeypatch.setattr(
        backend,
        "load_nvfp4_calibration",
        lambda *_args, **_kwargs: pytest.fail(
            "prepacked W4A4 must not load BF16 calibration"
        ),
        raising=False,
    )
    state_dict_info = _packed_weight_info("q_proj")
    state_dict_info["q_proj.input_scale"] = ((), torch.float32)

    extension.prepare_refit_info(state_dict_info)

    forwarded = []
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, weights: forwarded.extend(weights) or "loaded",
    )
    for value in (0.25, 0.5):
        extension._load_weights(
            [("q_proj.input_scale", torch.tensor(value, dtype=torch.float32))]
        )

    assert extension._nrl_real_quant_source == "modelopt"
    assert [name for name, _ in forwarded] == [
        "q_proj.input_scale",
        "q_proj.input_scale",
    ]
    torch.testing.assert_close(forwarded[0][1], torch.tensor(0.25))
    torch.testing.assert_close(forwarded[1][1], torch.tensor(0.5))


def test_real_quant_prepare_refit_rejects_manifest_without_receiver_targets(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    extension = _make_real_quant_extension(backend, torch.nn.Module(), [])
    _patch_real_quant_load(monkeypatch, backend)

    with pytest.raises(
        ValueError,
        match="no receiver ModelOpt quantization targets found",
    ):
        extension.prepare_refit_info(
            {"embed_tokens.weight": ((128, 32), torch.bfloat16)}
        )


@pytest.mark.parametrize(
    ("state_dict_info", "ignore_patterns"),
    [
        ({}, []),
        (_bf16_weight_info("q_proj.weight"), ["q_proj"]),
    ],
)
def test_real_quant_prepare_refit_rejects_empty_or_all_ignored_manifest(
    monkeypatch,
    state_dict_info,
    ignore_patterns,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(backend, model, ignore_patterns)
    _patch_real_quant_load(monkeypatch, backend)

    with pytest.raises(
        ValueError,
        match="no receiver ModelOpt quantization targets found",
    ):
        extension.prepare_refit_info(state_dict_info)


def test_real_quant_bf16_manifest_passes_through_layernorm_and_bias(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    kept = "q_proj.weight"
    layernorm = "input_layernorm.weight"
    bias = "q_proj.bias"
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
    )
    _patch_real_quant_load(monkeypatch, backend)
    state_dict_info = _bf16_weight_info(kept)
    state_dict_info[layernorm] = ((32,), torch.bfloat16)
    state_dict_info[bias] = ((32,), torch.bfloat16)
    extension.prepare_refit_info(state_dict_info)

    forwarded = []
    monkeypatch.setattr(
        backend,
        "serialize_bf16_nvfp4_group",
        lambda tensors, *, mode, calibration: [(kept, tensors[kept].clone())],
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, weights: forwarded.extend(weights) or "loaded",
    )

    kept_weight = torch.ones((32, 16), dtype=torch.bfloat16)
    layernorm_weight = torch.full((32,), 2, dtype=torch.bfloat16)
    bias_weight = torch.full((32,), 3, dtype=torch.bfloat16)
    assert (
        extension._load_weights(
            [(kept, kept_weight), (layernorm, layernorm_weight), (bias, bias_weight)]
        )
        == "loaded"
    )

    assert extension._nrl_real_quant_source == "bf16"
    assert [name for name, _ in forwarded] == [layernorm, bias, kept]
    assert forwarded[0][1] is layernorm_weight
    assert forwarded[1][1] is bias_weight


def test_real_quant_prepare_refit_classifies_complete_modelopt_manifest(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
    )
    _patch_real_quant_load(monkeypatch, backend)

    extension.prepare_refit_info(_packed_weight_info("q_proj"))

    assert extension._nrl_real_quant_source == "modelopt"


def test_real_quant_prepare_refit_rejects_incomplete_packed_weight_family(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(backend, model, [])
    _patch_real_quant_load(monkeypatch, backend)

    with pytest.raises(
        RuntimeError,
        match=r"Incomplete ModelOpt weight family for q_proj: missing.*weight_scale",
    ):
        extension.prepare_refit_info({"q_proj.weight": ((32, 8), torch.uint8)})


def test_real_quant_prepare_refit_derives_w13_metadata_once_from_receiver_targets(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    prefix = "model.layers.0.mlp"
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = _new_modelopt_moe()
    extension = _make_real_quant_extension(backend, model, [])
    _patch_real_quant_load(monkeypatch, backend)
    state_dict_info = _packed_moe_info(prefix)
    unrelated = "unrelated.layers.0.mlp.experts.w13_weight_scale_2"
    state_dict_info[unrelated] = ((2, 3), torch.float32)
    original = backend._w13_num_shards_from_state_dict_info
    calls = []

    def track_w13_info(filtered_info, *, require_input_scales=False):
        calls.append(set(filtered_info))
        return original(
            filtered_info,
            require_input_scales=require_input_scales,
        )

    monkeypatch.setattr(
        backend,
        "_w13_num_shards_from_state_dict_info",
        track_w13_info,
    )

    extension.prepare_refit_info(state_dict_info)

    assert calls == [set(_packed_moe_info(prefix))]
    assert extension._nrl_w13_num_shards_by_prefix == {prefix: 2}


def test_real_quant_modelopt_manifest_allows_bf16_layernorm_passthrough(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
    )
    _patch_real_quant_load(monkeypatch, backend)
    state_dict_info = _packed_weight_info("q_proj")
    state_dict_info["input_layernorm.weight"] = (
        (32,),
        torch.bfloat16,
    )

    extension.prepare_refit_info(state_dict_info)

    assert extension._nrl_real_quant_source == "modelopt"


def test_real_quant_prepare_refit_rejects_mixed_source_manifest(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
    )
    _patch_real_quant_load(monkeypatch, backend)
    mixed_info = _bf16_weight_info("q_proj.weight")
    mixed_info.update(
        {
            "q_proj.weight_scale": ((32, 1), torch.float8_e4m3fn),
            "q_proj.weight_scale_2": ((), torch.float32),
        }
    )

    with pytest.raises(ValueError, match="mixed BF16 and ModelOpt"):
        extension.prepare_refit_info(mixed_info)


def test_real_quant_bf16_split_group_stages_owned_weights_and_forwards_directly(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    gate = "model.layers.0.mlp.experts.0.gate_proj.weight"
    up = "model.layers.0.mlp.experts.0.up_proj.weight"
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = _new_modelopt_moe()
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
    )
    _patch_real_quant_load(monkeypatch, backend)
    extension.prepare_refit_info(_bf16_weight_info(gate, up))
    assert extension._nrl_bf16_quantizable_names == {gate, up}

    serialized_calls = []
    forwarded = []

    def serialize(tensors, *, mode, calibration):
        serialized_calls.append((dict(tensors), mode, calibration))
        return [(gate, torch.tensor(1)), (up, torch.tensor(2))]

    monkeypatch.setattr(backend, "serialize_bf16_nvfp4_group", serialize)
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, weights: forwarded.extend(weights) or "loaded",
    )
    monkeypatch.setattr(
        backend,
        "_batch_fused_modelopt_moe_weights",
        lambda *_args, **_kwargs: pytest.fail("BF16 groups use direct forwarding"),
    )
    source_gate = torch.ones((32, 16), dtype=torch.bfloat16)
    assert extension._load_weights([(gate, source_gate)]) is None
    staged_gate = extension._nrl_bf16_staging["model.layers.0.mlp.experts.0.w13"][gate]
    assert staged_gate is not source_gate
    assert (
        staged_gate.untyped_storage().data_ptr()
        != source_gate.untyped_storage().data_ptr()
    )

    source_gate.zero_()
    source_up = torch.full((32, 16), 2, dtype=torch.bfloat16)
    assert extension._load_weights([(up, source_up)]) == "loaded"

    assert serialized_calls[0][1:] == ("w4a16", None)
    assert serialized_calls[0][0][gate].sum().item() == 32 * 16
    assert serialized_calls[0][0][up].sum().item() == 2 * 32 * 16
    assert [name for name, _ in forwarded] == [gate, up]


@pytest.mark.parametrize("mode", ["w4a16", "w4a4"])
def test_real_quant_bf16_non_gated_up_uses_manifest_group_membership(
    monkeypatch,
    mode,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    up = "model.layers.0.mlp.experts.0.up_proj.weight"
    group_name = "model.layers.0.mlp.experts.0.w13"
    input_scale_name = up.removesuffix(".weight") + ".input_scale"
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = _new_modelopt_moe()
    extension = _make_real_quant_extension(backend, model, [])
    _patch_real_quant_load(monkeypatch, backend)
    extension.prepare_refit_info(_bf16_weight_info(up))
    extension._nrl_bf16_mode = mode
    extension._nrl_bf16_calibration = (
        NVFP4Calibration({up: torch.tensor(12.0)}) if mode == "w4a4" else None
    )
    if mode == "w4a4":
        extension._nrl_bf16_expected_input_scale_names = {input_scale_name}

    serialized_calls = []
    forwarded = []

    def serialize(tensors, *, mode, calibration, expected_names=None):
        serialized_calls.append((dict(tensors), mode, calibration, expected_names))
        outputs = [(up, tensors[up].clone())]
        if mode == "w4a4":
            outputs.append((input_scale_name, torch.tensor(0.25)))
        return outputs

    monkeypatch.setattr(backend, "serialize_bf16_nvfp4_group", serialize)
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, weights: forwarded.extend(weights) or "loaded",
    )

    assert extension._nrl_bf16_group_members == {group_name: (up,)}
    assert (
        extension._load_weights([(up, torch.ones((32, 16), dtype=torch.bfloat16))])
        == "loaded"
    )
    assert serialized_calls[0][3] == (up,)
    assert [name for name, _ in forwarded] == [
        up,
        *([input_scale_name] if mode == "w4a4" else []),
    ]


def test_real_quant_bf16_manifest_rejects_gate_only_expert_group(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    gate = "model.layers.0.mlp.experts.0.gate_proj.weight"

    with pytest.raises(ValueError, match="gate projection without its up projection"):
        backend._nvfp4_manifest_group_members({gate})


def test_real_quant_bf16_ignored_weight_passes_through_and_scales_stay_filtered(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    ignored = "lm_head.weight"
    kept = "q_proj.weight"
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        ["lm_head"],
    )
    _patch_real_quant_load(monkeypatch, backend)
    extension.prepare_refit_info(_bf16_weight_info(ignored, kept))

    forwarded = []
    monkeypatch.setattr(
        backend,
        "serialize_bf16_nvfp4_group",
        lambda tensors, *, mode, calibration: [(kept, tensors[kept].clone())],
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, weights: forwarded.extend(weights) or "loaded",
    )
    ignored_weight = torch.full((32, 16), 7, dtype=torch.bfloat16)
    kept_weight = torch.full((32, 16), 3, dtype=torch.bfloat16)
    assert (
        extension._load_weights(
            [
                (ignored, ignored_weight),
                ("lm_head.weight_scale", torch.ones(1)),
                (kept, kept_weight),
            ]
        )
        == "loaded"
    )

    assert [name for name, _ in forwarded] == [ignored, kept]
    assert forwarded[0][1] is ignored_weight
    torch.testing.assert_close(forwarded[1][1], kept_weight)


def test_real_quant_bf16_incomplete_group_fails_before_lifecycle_finalization(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    gate = "model.layers.0.mlp.experts.0.gate_proj.weight"
    up = "model.layers.0.mlp.experts.0.up_proj.weight"
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = _new_modelopt_moe()
    extension = _make_real_quant_extension(backend, model, [])
    _patch_real_quant_load(monkeypatch, backend)
    extension.prepare_refit_info(_bf16_weight_info(gate, up))
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, _weights: "loaded",
    )
    with extension._weight_update_lifecycle("ipc") as finish:
        extension._load_weights([(gate, torch.ones((32, 16), dtype=torch.bfloat16))])
        with pytest.raises(RuntimeError, match="missing.*up_proj.weight"):
            finish()


def test_real_quant_bf16_w4a4_two_refits_open_artifact_once_and_replay_fixed_scale(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    model.q_proj.input_scale = torch.nn.Parameter(torch.tensor(1.0))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_PATH",
        "/worker/calibration.safetensors",
    )
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "NVFP4_DEFAULT_CFG",
    )
    name = "q_proj.weight"
    input_scale_name = "q_proj.input_scale"
    calibration = NVFP4Calibration(input_amax={name: torch.tensor(12.0)})
    artifact_loads = []

    def load_calibration(path, **kwargs):
        artifact_loads.append((path, kwargs))
        return calibration

    monkeypatch.setattr(
        backend,
        "load_nvfp4_calibration",
        load_calibration,
        raising=False,
    )
    requests = extension.prepare_refit_info(_bf16_weight_info(name))
    assert len(requests) == 1
    assert requests[0].parameter_names == (name,)
    assert requests[0].source_format == "bf16"
    assert requests[0].target_format == "nvfp4_w4a4"

    serializer_scales = iter((0.25, 0.75))
    serializer_calls = []

    def serialize(tensors, *, mode, calibration):
        serializer_calls.append((mode, calibration))
        return [
            (name, tensors[name].clone()),
            (input_scale_name, torch.tensor(next(serializer_scales))),
        ]

    monkeypatch.setattr(backend, "serialize_bf16_nvfp4_group", serialize)
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    active_runtime = {}
    forwarded_names = []
    replayed_scales = []
    finalizations = []

    def initialize(root):
        active_runtime[root] = (
            root.weight,
            root.input_scale,
            root.quant_method,
        )

    def load_weights(_self, weights):
        loaded_weights = dict(weights)
        forwarded_names.append(list(loaded_weights))
        replayed_scales.append(loaded_weights[input_scale_name].clone())
        model.q_proj.weight = torch.nn.Parameter(loaded_weights[name].clone())
        model.q_proj.input_scale = torch.nn.Parameter(
            loaded_weights[input_scale_name].clone()
        )
        model.q_proj.quant_method = object()
        return "loaded"

    def finalize(root, config):
        original_weight, original_input_scale, original_kernel = active_runtime.pop(
            root
        )
        original_weight.data.copy_(root.weight.data)
        original_input_scale.data.copy_(root.input_scale.data)
        root.weight = original_weight
        root.input_scale = original_input_scale
        root.quant_method = original_kernel
        finalizations.append((root, config))

    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        load_weights,
    )
    monkeypatch.setattr(reload_mod, "initialize_layerwise_reload", initialize)
    monkeypatch.setattr(reload_mod, "finalize_layerwise_reload", finalize)
    monkeypatch.setattr(backend.torch.accelerator, "synchronize", lambda: None)
    weight_identity = model.q_proj.weight
    input_scale_identity = model.q_proj.input_scale
    kernel_identity = model.q_proj.quant_method

    for value in (1.0, 2.0):
        transport_weights = [(name, torch.full((32, 16), value, dtype=torch.bfloat16))]
        assert [transport_name for transport_name, _ in transport_weights] == [name]
        with extension._weight_update_lifecycle("ipc") as finish:
            assert extension._load_weights(transport_weights) == "loaded"
            finish()
        assert model.q_proj.weight is weight_identity
        assert model.q_proj.input_scale is input_scale_identity
        assert model.q_proj.quant_method is kernel_identity
        torch.testing.assert_close(model.q_proj.input_scale, torch.tensor(0.25))

    assert len(artifact_loads) == 1
    assert serializer_calls == [
        ("w4a4", calibration),
        ("w4a4", calibration),
    ]
    assert forwarded_names == [
        [name, input_scale_name],
        [name, input_scale_name],
    ]
    torch.testing.assert_close(replayed_scales[0], torch.tensor(0.25))
    torch.testing.assert_close(replayed_scales[1], torch.tensor(0.25))
    assert len(finalizations) == 2


def test_real_quant_bf16_w4a4_does_not_cache_scale_when_base_load_fails(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_PATH",
        "/worker/calibration.safetensors",
    )
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "NVFP4_DEFAULT_CFG",
    )
    name = "q_proj.weight"
    calibration = NVFP4Calibration(input_amax={name: torch.tensor(12.0)})
    monkeypatch.setattr(
        backend,
        "load_nvfp4_calibration",
        lambda *_args, **_kwargs: calibration,
        raising=False,
    )
    extension.prepare_refit_info(_bf16_weight_info(name))
    monkeypatch.setattr(
        backend,
        "serialize_bf16_nvfp4_group",
        lambda tensors, *, mode, calibration: [
            (name, tensors[name].clone()),
            ("q_proj.input_scale", torch.tensor(0.25)),
        ],
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, _weights: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        extension._load_weights([(name, torch.ones((32, 16), dtype=torch.bfloat16))])

    assert extension._nrl_bf16_input_scale_cache == {}


def test_real_quant_bf16_w4a4_incomplete_gate_up_fails_before_finalizer(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    gate = "model.layers.0.mlp.experts.0.gate_proj.weight"
    up = "model.layers.0.mlp.experts.0.up_proj.weight"
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = _new_modelopt_moe()
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_PATH",
        "/worker/calibration.safetensors",
    )
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "NVFP4_DEFAULT_CFG",
    )
    calibration = NVFP4Calibration(
        input_amax={gate: torch.tensor(12.0), up: torch.tensor(12.0)}
    )
    monkeypatch.setattr(
        backend,
        "load_nvfp4_calibration",
        lambda *_args, **_kwargs: calibration,
        raising=False,
    )
    extension.prepare_refit_info(_bf16_weight_info(gate, up))
    native_finalizations = []
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda *_args: native_finalizations.append("finalized"),
    )
    monkeypatch.setattr(backend.torch.accelerator, "synchronize", lambda: None)

    with extension._weight_update_lifecycle("ipc") as finish:
        assert (
            extension._load_weights(
                [(gate, torch.ones((32, 16), dtype=torch.bfloat16))]
            )
            is None
        )
        with pytest.raises(RuntimeError, match="missing.*up_proj.weight"):
            finish()

    assert native_finalizations == []


def test_real_quant_bf16_w4a4_experts_emit_only_canonical_projection_families(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    prefix = "model.layers.0.mlp.experts.0"
    gate = f"{prefix}.gate_proj.weight"
    up = f"{prefix}.up_proj.weight"
    down = f"{prefix}.down_proj.weight"
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = _new_modelopt_moe()
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_PATH",
        "/worker/calibration.safetensors",
    )
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "NVFP4_DEFAULT_CFG",
    )
    calibration = NVFP4Calibration(
        input_amax={
            gate: torch.tensor(12.0),
            up: torch.tensor(12.0),
            down: torch.tensor(24.0),
        }
    )
    monkeypatch.setattr(
        backend,
        "load_nvfp4_calibration",
        lambda *_args, **_kwargs: calibration,
        raising=False,
    )
    extension.prepare_refit_info(_bf16_weight_info(gate, up, down))

    def serialize(tensors, *, mode, calibration):
        assert mode == "w4a4"
        serialized = []
        for weight_name, weight in tensors.items():
            projection = weight_name.removesuffix(".weight")
            serialized.extend(
                [
                    (weight_name, weight.clone()),
                    (f"{projection}.weight_scale", torch.ones(1)),
                    (f"{projection}.weight_scale_2", torch.ones(())),
                    (f"{projection}.input_scale", torch.full((), 0.25)),
                ]
            )
        return serialized

    forwarded = []
    monkeypatch.setattr(backend, "serialize_bf16_nvfp4_group", serialize)
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, weights: forwarded.extend(weights) or "loaded",
    )

    assert (
        extension._load_weights(
            [
                (gate, torch.ones((32, 16), dtype=torch.bfloat16)),
                (up, torch.full((32, 16), 2, dtype=torch.bfloat16)),
                (down, torch.full((32, 16), 3, dtype=torch.bfloat16)),
            ]
        )
        == "loaded"
    )

    forwarded_names = [name for name, _ in forwarded]
    assert [name for name in forwarded_names if name.endswith(".input_scale")] == [
        f"{prefix}.gate_proj.input_scale",
        f"{prefix}.up_proj.input_scale",
        f"{prefix}.down_proj.input_scale",
    ]
    assert not any(".w13" in name or ".w2" in name for name in forwarded_names)


def test_real_quant_bf16_complete_group_finalizes_once_and_preserves_identity(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.q_proj = _mark_as_modelopt_layer(torch.nn.Linear(16, 32, bias=False))
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
    )
    _patch_real_quant_load(monkeypatch, backend)
    weight_identity = model.q_proj.weight
    kernel_identity = model.q_proj.quant_method
    name = "q_proj.weight"
    extension.prepare_refit_info(_bf16_weight_info(name))

    monkeypatch.setattr(
        backend,
        "serialize_bf16_nvfp4_group",
        lambda tensors, *, mode, calibration: [(name, tensors[name].clone())],
    )
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    active_runtime = {}
    replacement_identities = []
    finalizations = []

    def initialize(root):
        active_runtime[root] = (root.weight, root.quant_method)

    def load_weights(_self, weights):
        loaded_weights = dict(weights)
        replacement_weight = torch.nn.Parameter(loaded_weights[name].clone())
        replacement_kernel = object()
        model.q_proj.weight = replacement_weight
        model.q_proj.quant_method = replacement_kernel
        replacement_identities.append((replacement_weight, replacement_kernel))
        return "loaded"

    def finalize(root, config):
        original_weight, original_kernel = active_runtime.pop(root)
        original_weight.data.copy_(root.weight.data)
        root.weight = original_weight
        root.quant_method = original_kernel
        finalizations.append((root, config))

    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        load_weights,
    )
    monkeypatch.setattr(reload_mod, "initialize_layerwise_reload", initialize)
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        finalize,
    )
    monkeypatch.setattr(backend.torch.accelerator, "synchronize", lambda: None)

    for value in (1.0, 2.0):
        with extension._weight_update_lifecycle("ipc") as finish:
            extension._load_weights(
                [(name, torch.full((32, 16), value, dtype=torch.bfloat16))]
            )
            assert model.q_proj.weight is replacement_identities[-1][0]
            assert model.q_proj.quant_method is replacement_identities[-1][1]
            finish()
        assert model.q_proj.weight is weight_identity
        assert model.q_proj.quant_method is kernel_identity
        torch.testing.assert_close(
            model.q_proj.weight,
            torch.full_like(model.q_proj.weight, value),
        )

    assert len(finalizations) == 2
    assert len(replacement_identities) == 2
    assert replacement_identities[0][0] is not replacement_identities[1][0]
    assert replacement_identities[0][1] is not replacement_identities[1][1]


def test_real_quant_load_weights_batches_full_experts_and_expands_global_scales(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    modelopt_config = sys.modules[
        "vllm.model_executor.layers.quantization.modelopt"
    ].ModelOptNvFp4Config

    class ModelOptNvFp4FusedMoE:
        def __init__(self):
            self.quant_config = modelopt_config()
            self.quant_config.get_name = lambda: NEMO_MODELOPT_W4A16

    def make_model(expert_map):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.model.layers[0].mlp = torch.nn.Module()
        model.model.layers[0].mlp.experts = _new_modelopt_moe()
        model.model.layers[0].mlp.experts.quant_method = ModelOptNvFp4FusedMoE()
        model.model.layers[0].mlp.experts._expert_map = expert_map
        model.model.layers[0].mlp.experts.local_num_experts = (
            2 if expert_map is None else 1
        )
        model.model.layers[0].mlp.experts.global_num_experts = 2
        # ModelOpt assigns the same quant config to attention's FP8 KV method;
        # this must not be mistaken for expert parallelism.
        model.attention = torch.nn.Module()
        model.attention.quant_method = ModelOptNvFp4FusedMoE()
        return model

    prefix = "model.layers.0.mlp"
    w13_weight = torch.arange(24).reshape(2, 4, 3)
    w13_scale_2 = torch.tensor([[1.0], [2.0]])
    state_dict_info = {
        f"{prefix}.experts.w13_weight": ((2, 4, 3), torch.uint8),
        f"{prefix}.experts.w13_weight_scale": ((2, 4, 1), torch.uint8),
        f"{prefix}.experts.w13_weight_scale_2": ((2, 1), torch.float32),
        f"{prefix}.experts.w2_weight": ((2, 3, 2), torch.uint8),
        f"{prefix}.experts.w2_weight_scale": ((2, 3, 1), torch.uint8),
        f"{prefix}.experts.w2_weight_scale_2": ((2,), torch.float32),
    }

    batched_forwarded = []
    extension = _make_real_quant_extension(backend, make_model(None), [])
    _patch_real_quant_load(monkeypatch, backend, batched_forwarded)
    assert extension.prepare_refit_info(state_dict_info) is None
    extension._nrl_w13_num_shards_by_prefix = {prefix: 1}
    _patch_real_quant_load(monkeypatch, backend, batched_forwarded)
    assert (
        extension._load_weights(
            [
                (f"{prefix}.experts.w13_weight", w13_weight),
                (f"{prefix}.experts.w13_weight_scale_2", w13_scale_2),
            ]
        )
        == "loaded"
    )
    assert [name for name, _ in batched_forwarded] == [
        f"{prefix}.experts.0.up_proj.weight",
        f"{prefix}.experts.0.up_proj.weight_scale_2",
        f"{prefix}.experts.1.up_proj.weight_scale_2",
    ]
    assert batched_forwarded[0][1] is w13_weight
    torch.testing.assert_close(batched_forwarded[1][1], w13_scale_2[0, 0])
    torch.testing.assert_close(batched_forwarded[2][1], w13_scale_2[1, 0])

    extension = _make_real_quant_extension(
        backend,
        make_model(torch.tensor([0, -1])),
        [],
    )
    extension.model_runner.vllm_config.parallel_config.enable_expert_parallel = True
    with pytest.raises(RuntimeError, match="all experts local"):
        extension.prepare_refit_info(state_dict_info)


def test_real_quant_load_weights_expands_gated_experts_per_expert(monkeypatch):
    """Gated fused W13 tensors must be split into per-expert 2-D shards.

    Batched 3-D tensors would route through vLLM 0.25's
    RoutedExperts.load_weights fused branch, whose orientation heuristic
    mis-transposes packed NVFP4 weights and block scales.
    """
    backend = _import_vllm_quant_backend(monkeypatch)
    modelopt_config = sys.modules[
        "vllm.model_executor.layers.quantization.modelopt"
    ].ModelOptNvFp4Config

    class ModelOptNvFp4FusedMoE:
        def __init__(self):
            self.quant_config = modelopt_config()
            self.quant_config.get_name = lambda: NEMO_MODELOPT_W4A16

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = _new_modelopt_moe()
    model.model.layers[0].mlp.experts.quant_method = ModelOptNvFp4FusedMoE()
    model.model.layers[0].mlp.experts._expert_map = None
    model.model.layers[0].mlp.experts.local_num_experts = 2
    model.model.layers[0].mlp.experts.global_num_experts = 2

    prefix = "model.layers.0.mlp"
    w13_weight = torch.arange(24, dtype=torch.uint8).reshape(2, 4, 3)
    w13_scale = torch.arange(8, dtype=torch.uint8).reshape(2, 4, 1)
    state_dict_info = {
        f"{prefix}.experts.w13_weight": ((2, 4, 3), torch.uint8),
        f"{prefix}.experts.w13_weight_scale": ((2, 4, 1), torch.uint8),
        f"{prefix}.experts.w13_weight_scale_2": ((2, 2), torch.float32),
        f"{prefix}.experts.w2_weight": ((2, 3, 2), torch.uint8),
        f"{prefix}.experts.w2_weight_scale": ((2, 3, 1), torch.uint8),
        f"{prefix}.experts.w2_weight_scale_2": ((2,), torch.float32),
    }

    assert backend._w13_num_shards_from_state_dict_info(state_dict_info) == {prefix: 2}

    forwarded = []
    extension = _make_real_quant_extension(backend, model, [])
    extension.prepare_refit_info(state_dict_info)
    extension._nrl_w13_num_shards_by_prefix = {prefix: 2}
    _patch_real_quant_load(monkeypatch, backend, forwarded)
    assert (
        extension._load_weights(
            [
                (f"{prefix}.experts.w13_weight", w13_weight),
                (f"{prefix}.experts.w13_weight_scale", w13_scale),
            ]
        )
        == "loaded"
    )

    assert [name for name, _ in forwarded] == [
        f"{prefix}.experts.0.gate_proj.weight",
        f"{prefix}.experts.1.gate_proj.weight",
        f"{prefix}.experts.0.up_proj.weight",
        f"{prefix}.experts.1.up_proj.weight",
        f"{prefix}.experts.0.gate_proj.weight_scale",
        f"{prefix}.experts.1.gate_proj.weight_scale",
        f"{prefix}.experts.0.up_proj.weight_scale",
        f"{prefix}.experts.1.up_proj.weight_scale",
    ]
    for _, tensor in forwarded:
        assert tensor.ndim == 2
    torch.testing.assert_close(forwarded[0][1], w13_weight[0, :2])
    torch.testing.assert_close(forwarded[1][1], w13_weight[1, :2])
    torch.testing.assert_close(forwarded[2][1], w13_weight[0, 2:])
    torch.testing.assert_close(forwarded[3][1], w13_weight[1, 2:])
    torch.testing.assert_close(forwarded[4][1], w13_scale[0, :2])
    torch.testing.assert_close(forwarded[7][1], w13_scale[1, 2:])


def test_real_quant_load_weights_forwards_ignored_float_weights(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)
            self.keep = torch.nn.Linear(2, 2, bias=False)

    model = TinyModel()
    forwarded = []
    extension = _make_real_quant_extension(backend, model, ["lm_head"])
    _patch_real_quant_load(monkeypatch, backend, forwarded)

    ignored_weight = torch.full_like(model.lm_head.weight, 7.0)
    kept_weight = torch.full_like(model.keep.weight, 3.0)

    assert (
        extension._load_weights(
            [
                ("lm_head.weight", ignored_weight),
                ("lm_head.weight_scale", torch.ones(1)),
                ("keep.weight", kept_weight),
            ]
        )
        == "loaded"
    )

    assert [name for name, _ in forwarded] == ["lm_head.weight", "keep.weight"]
    torch.testing.assert_close(forwarded[0][1], ignored_weight)
    torch.testing.assert_close(forwarded[1][1], kept_weight)


def test_real_quant_load_weights_returns_when_only_ignored_scales(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    extension = _make_real_quant_extension(backend, model, ["lm_head"])
    _patch_real_quant_load(monkeypatch, backend)

    assert (
        extension._load_weights(
            [
                ("lm_head.weight_scale", torch.ones(1)),
                ("lm_head.weight_scale_2", torch.ones(1)),
            ]
        )
        is None
    )


def test_real_quant_load_weights_forwards_ignored_weights_to_vllm_loader(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    forwarded = []
    extension = _make_real_quant_extension(backend, model, ["lm_head"])
    _patch_real_quant_load(monkeypatch, backend, forwarded)

    mismatched = torch.ones(1, dtype=model.lm_head.weight.dtype)

    assert extension._load_weights([("lm_head.weight", mismatched)]) == "loaded"
    assert forwarded == [("lm_head.weight", mismatched)]


def test_real_quant_load_weights_detaches_pending_layerwise_views(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    layerwise_mod = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    model = torch.nn.Module()
    model.reload_root = torch.nn.Linear(2, 2, bias=False)
    model.unrelated = torch.nn.Linear(2, 2, bias=False)
    extension = _make_real_quant_extension(backend, model, [])
    extension._nrl_modelopt_reload_roots = (model.reload_root,)
    _patch_real_quant_load(monkeypatch, backend, [])

    source = torch.arange(4, dtype=torch.float32)
    incoming = source.view(2, 2)
    bound_arguments = types.SimpleNamespace(arguments={"loaded_weight": incoming})
    layerwise_info = types.SimpleNamespace(loaded_weights=[("weight", bound_arguments)])
    inspected = []

    def get_layerwise_info(module):
        inspected.append(module)
        if module is model.reload_root:
            return layerwise_info
        return types.SimpleNamespace(loaded_weights=[])

    monkeypatch.setattr(
        layerwise_mod,
        "get_layerwise_info",
        get_layerwise_info,
    )

    assert extension._load_weights([("reload_root.weight", incoming)]) == "loaded"

    detached = bound_arguments.arguments["loaded_weight"]
    assert detached.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    torch.testing.assert_close(detached, incoming)
    source.zero_()
    torch.testing.assert_close(detached, torch.arange(4).view(2, 2).float())
    assert inspected == [model.reload_root]


def test_real_quant_pre_ack_fence_is_device_wide_and_load_does_not_fence(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Linear(1, 1)
    extension = _make_real_quant_extension(backend, model, [])
    extension._nrl_modelopt_reload_roots = (model,)
    extension.device = types.SimpleNamespace(type="cuda")
    events = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, _weights: events.append("load") or "loaded",
    )
    monkeypatch.setattr(
        backend,
        "_detach_pending_layerwise_weights",
        lambda _roots, _storage_ptrs: events.append("detach"),
    )
    monkeypatch.setattr(backend.torch, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        backend.torch.accelerator,
        "synchronize",
        lambda: events.append("sync"),
    )
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_stream",
        lambda: pytest.fail("real quant must use one device-wide IPC ACK fence"),
    )

    assert extension._load_weights([("weight", torch.ones(1))]) == "loaded"
    assert events == ["load", "detach"]

    extension._synchronize_before_ipc_data_ack()
    assert events == ["load", "detach", "sync"]


@pytest.mark.parametrize("load_numel", [0, 10])
def test_real_quant_rejects_incomplete_modelopt_layerwise_reload(
    monkeypatch, load_numel
):
    backend = _import_vllm_quant_backend(monkeypatch)
    layerwise_mod = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    modelopt_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.modelopt"
    )
    modelopt_module.ModelOptNvFp4Config = type("ModelOptNvFp4Config", (), {})
    modelopt_base = type("ModelOptNvFp4FusedMoE", (), {})
    modelopt_module.ModelOptNvFp4FusedMoE = modelopt_base
    modelopt_module.ModelOptNvFp4LinearMethod = type(
        "ModelOptNvFp4LinearMethod", (), {}
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        modelopt_module,
    )
    experts = sys.modules[
        "vllm.model_executor.layers.fused_moe.routed_experts"
    ].RoutedExperts()
    experts.quant_method = type(
        "NemoModelOptNvFp4FusedMoE",
        (modelopt_base,),
        {"quant_config": modelopt_module.ModelOptNvFp4Config()},
    )()
    model = torch.nn.Module()
    model.experts = experts
    info = types.SimpleNamespace(
        load_numel=load_numel,
        load_numel_total=12,
        loaded_weights=[("w13_weight", object())] if load_numel else [],
    )
    monkeypatch.setattr(layerwise_mod, "get_layerwise_info", lambda _module: info)

    with pytest.raises(
        RuntimeError,
        match=rf"experts: {load_numel}/12 elements",
    ):
        backend._require_complete_modelopt_layerwise_reload(model)


def test_real_quant_accepts_processed_modelopt_layerwise_reload(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    layerwise_mod = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    modelopt_module = types.ModuleType(
        "vllm.model_executor.layers.quantization.modelopt"
    )
    modelopt_module.ModelOptNvFp4Config = type("ModelOptNvFp4Config", (), {})
    modelopt_module.ModelOptNvFp4FusedMoE = type("ModelOptNvFp4FusedMoE", (), {})
    modelopt_base = type("ModelOptNvFp4LinearMethod", (), {})
    modelopt_module.ModelOptNvFp4LinearMethod = modelopt_base
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        modelopt_module,
    )
    linear = sys.modules["vllm.model_executor.layers.linear"].LinearBase(
        1, 1, bias=False
    )
    linear.quant_method = type(
        "NemoModelOptW4A16LinearMethod",
        (modelopt_base,),
        {"quant_config": modelopt_module.ModelOptNvFp4Config()},
    )()
    model = torch.nn.Module()
    model.linear = linear
    info = types.SimpleNamespace(
        load_numel=0,
        load_numel_total=None,
        loaded_weights=[],
    )
    monkeypatch.setattr(layerwise_mod, "get_layerwise_info", lambda _module: info)

    backend._require_complete_modelopt_layerwise_reload(model)


def test_real_quant_scopes_native_reload_away_from_mamba_alias_buffers(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    modelopt_module = sys.modules["vllm.model_executor.layers.quantization.modelopt"]
    attention_module = sys.modules["vllm.model_executor.layers.attention"]
    kv_cache_module = sys.modules["vllm.model_executor.layers.quantization.kv_cache"]

    class MambaMixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = torch.nn.Linear(3, 2, bias=False)
            self.register_buffer(
                "conv_weights",
                self.conv1d.weight.detach().view(-1),
                persistent=False,
            )

    class KVAttention(attention_module.Attention):
        def __init__(self):
            super().__init__()
            self.quant_method = kv_cache_module.BaseKVCacheMethod()
            self.kv_cache_dtype = "fp8"
            self.projection = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))

    model = torch.nn.Module()
    model.mamba = MambaMixer()
    model.experts = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    model.attention = KVAttention()

    assert backend._modelopt_layerwise_reload_roots(
        model,
        include_fp8_kv_cache=False,
    ) == [model.experts, model.attention.projection]
    assert backend._modelopt_layerwise_reload_roots(
        model,
        include_fp8_kv_cache=True,
    ) == [model.experts, model.attention]

    model.attention.kv_cache_dtype = "auto"
    assert backend._modelopt_layerwise_reload_roots(
        model,
        include_fp8_kv_cache=True,
    ) == [model.experts, model.attention.projection]
    model.attention.kv_cache_dtype = "fp8"

    model.shared = torch.nn.Module()
    model.shared.experts = model.experts
    assert backend._modelopt_layerwise_reload_roots(
        model,
        include_fp8_kv_cache=True,
    ) == [model.experts, model.attention]

    for roots in (
        backend._modelopt_layerwise_reload_roots(model, include_fp8_kv_cache=False),
        backend._modelopt_layerwise_reload_roots(model, include_fp8_kv_cache=True),
    ):
        assert model.mamba not in roots
        assert model.mamba.conv1d not in roots


def test_real_quant_caches_scoped_reload_roots(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    model = torch.nn.Linear(1, 1)
    extension = _make_real_quant_extension(backend, model, [])
    extension._nrl_modelopt_reload_roots = None
    selected_roots = [model]
    calls = []

    def select_modelopt_roots(model_arg, *, include_fp8_kv_cache):
        calls.append((model_arg, include_fp8_kv_cache))
        return selected_roots

    monkeypatch.setattr(
        backend,
        "_modelopt_layerwise_reload_roots",
        select_modelopt_roots,
    )

    first = extension._get_modelopt_reload_roots()
    second = extension._get_modelopt_reload_roots()

    assert first is second
    assert first == (model,)
    assert calls == [(model, False)]


def test_fake_quant_load_weights_exposes_input_quantizer_buffers(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    child = torch.nn.Module()
    child.weight = torch.nn.Parameter(torch.ones(1))
    child.register_buffer("input_quantizer_amax", torch.tensor([1.0]))
    child.register_buffer("weight_quantizer_amax", torch.tensor([2.0]))
    model = torch.nn.Module()
    model.child = child
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    seen_names = []

    def fake_base_load_weights(self, weights):
        params = dict(child.named_parameters())
        seen_names.extend(params)
        params["input_quantizer_amax"].weight_loader(
            params["input_quantizer_amax"],
            torch.tensor([3.0]),
        )
        return "loaded"

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: False,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        fake_base_load_weights,
    )

    assert extension._load_weights([("unused", torch.ones(1))]) == "loaded"

    assert "weight" in seen_names
    assert "input_quantizer_amax" in seen_names
    assert "weight_quantizer_amax" not in seen_names
    assert not hasattr(child.input_quantizer_amax, "weight_loader")
    torch.testing.assert_close(child.input_quantizer_amax, torch.tensor([3.0]))


def test_real_quant_reload_keeps_vllm_config_active_during_layerwise_processing(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    config_mod = sys.modules["vllm.config"]
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]

    model = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    vllm_config = object()
    model_config = object()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=vllm_config,
    )
    extension.model_config = model_config
    extension.device = torch.device("cpu")
    extension._nrl_modelopt_reload_roots = (model,)
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "initialize_layerwise_reload",
        lambda root: calls.append(("initialize", root)),
    )

    def finalize(root, config):
        assert config_mod.get_current_vllm_config() is vllm_config
        calls.append(("finalize", root, config))

    monkeypatch.setattr(reload_mod, "finalize_layerwise_reload", finalize)
    monkeypatch.setattr(
        backend.torch.accelerator,
        "synchronize",
        lambda: calls.append("sync"),
    )

    with extension._weight_update_lifecycle("collective") as finish:
        # FlashInferExperts performs this lookup when online layer processing
        # reconstructs its kernel during the yielded weight-load phase.
        assert config_mod.get_current_vllm_config() is vllm_config
        calls.append("load")
        finish()

    assert config_mod.current is None
    assert calls == [
        ("initialize", model),
        "load",
        ("finalize", model, model_config),
        "sync",
    ]


def test_real_quant_nccl_reshard_leaves_completion_fence_to_transport(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]

    model = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=object(),
    )
    extension.model_config = object()
    extension.device = torch.device("cpu")
    extension._nrl_modelopt_reload_roots = (model,)
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "initialize_layerwise_reload",
        lambda root: calls.append(("initialize", root)),
    )
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda root, config: calls.append(("finalize", root, config)),
    )
    monkeypatch.setattr(
        backend.torch.accelerator,
        "synchronize",
        lambda: calls.append("lifecycle-sync"),
    )

    with extension._weight_update_lifecycle("nccl_reshard") as finish:
        calls.append("load")
        finish()

    assert calls == [
        ("initialize", model),
        "load",
        ("finalize", model, extension.model_config),
    ]


def test_real_quant_collective_reload_uses_vllm_layerwise_lifecycle(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]

    model = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    model_config = object()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=object(),
    )
    extension.model_config = model_config
    extension.device = torch.device("cpu")
    extension.state_dict_info = {}
    extension.model_update_group = object()
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "initialize_layerwise_reload",
        lambda model_arg: calls.append(("initialize", model_arg)),
    )
    monkeypatch.setattr(
        base_backend,
        "packed_broadcast_consumer",
        lambda **kwargs: calls.append(("consume", kwargs["post_unpack_func"].__name__)),
    )
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda model_arg, config_arg: calls.append(("finalize", model_arg, config_arg)),
    )
    monkeypatch.setattr(
        backend.torch.accelerator,
        "synchronize",
        lambda: calls.append("sync"),
    )

    assert extension.update_weights_from_collective() is True
    assert calls == [
        ("initialize", model),
        ("consume", "_load_weights"),
        ("finalize", model, model_config),
        "sync",
    ]


@pytest.mark.parametrize("mode", ["w4a16", "w4a4"])
@pytest.mark.parametrize("gated", [True, False], ids=["gated", "non_gated"])
def test_real_quant_nccl_receiver_uses_owned_grouped_bf16_scratch(
    monkeypatch,
    mode,
    gated,
):
    from torch.distributed.tensor.placement_types import Replicate

    backend = _import_vllm_quant_backend(monkeypatch)
    prefix = "model.layers.0.mlp.experts"
    gate_name = f"{prefix}.gate_proj.weight"
    up_name = f"{prefix}.up_proj.weight"
    down_name = f"{prefix}.down_proj.weight"
    expert_projections = ("gate", "up", "down") if gated else ("up", "down")
    per_expert_names = {
        f"{prefix}.{expert_id}.{projection}_proj.weight"
        for expert_id in range(2)
        for projection in expert_projections
    }
    w13_runtime = torch.full((2, 128 if gated else 64, 16), 255, dtype=torch.uint8)
    w2_runtime = torch.full((2, 64, 16), 127, dtype=torch.uint8)
    model = torch.nn.Module()
    extension = _make_real_quant_extension(
        backend,
        model,
        [],
        quant_algo="NVFP4" if mode == "w4a4" else "W4A16_NVFP4",
    )
    _patch_real_quant_load(monkeypatch, backend)
    extension._nrl_real_quant_source = "bf16"
    extension._nrl_bf16_mode = mode
    extension._nrl_bf16_calibration = (
        NVFP4Calibration(
            input_amax={name: torch.tensor(12.0) for name in per_expert_names}
        )
        if mode == "w4a4"
        else None
    )
    extension._nrl_bf16_quantizable_names = set(per_expert_names)
    extension._nrl_bf16_group_members = backend._nvfp4_manifest_group_members(
        set(per_expert_names)
    )
    extension._nrl_bf16_staging = {}
    extension._nrl_bf16_input_scale_cache = {}
    extension._nrl_bf16_expected_input_scale_names = (
        {name.removesuffix(".weight") + ".input_scale" for name in per_expert_names}
        if mode == "w4a4"
        else set()
    )
    extension._nrl_modelopt_reload_roots = ()

    wire_component = {
        "role": "weight",
        "global_shape": (2, 64, 32),
        "dtype": "torch.bfloat16",
        "src_placements": [Replicate()],
        "dst_placements": [Replicate()],
    }
    destination_components = [
        {
            "role": "weight",
            "global_shape": (2, 64, 16),
            "dtype": "torch.uint8",
            "source": "codec",
            "dst_placements": [Replicate()],
        },
        {
            "role": "weight_scale",
            "global_shape": (2, 64, 2),
            "dtype": "torch.float8_e4m3fn",
            "source": "codec",
            "dst_placements": [Replicate()],
        },
        {
            "role": "weight_scale_2",
            "global_shape": (2,),
            "dtype": "torch.float32",
            "source": "codec",
            "dst_placements": [Replicate()],
        },
    ]
    if mode == "w4a4":
        destination_components.append(
            {
                "role": "input_scale",
                "global_shape": (2,),
                "dtype": "torch.float32",
                "source": "calibration",
                "dst_placements": [Replicate()],
            }
        )
    mesh = types.SimpleNamespace(mesh=torch.arange(1))
    w13_params = (
        ((gate_name, "gate_proj"), (up_name, "up_proj"))
        if gated
        else ((up_name, "up_proj"),)
    )
    params = [
        {
            "name": name,
            "global_shape": (2, 64, 32),
            "grouped_expert_proj": projection,
            "transform_id": f"bf16_to_nvfp4_{mode}",
            "wire_components": [wire_component],
            "components": [wire_component],
            "destination_components": destination_components,
            "completion_key": f"{prefix}.w13",
            "finalize_scope": "model",
            "dst_mesh_info": mesh,
        }
        for name, projection in w13_params
    ]
    params.append(
        {
            "name": down_name,
            "global_shape": (2, 64, 32),
            "grouped_expert_proj": "down_proj",
            "transform_id": f"bf16_to_nvfp4_{mode}",
            "wire_components": [wire_component],
            "components": [wire_component],
            "destination_components": destination_components,
            "completion_key": f"{prefix}.w2",
            "finalize_scope": "model",
            "dst_mesh_info": mesh,
        }
    )
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {"model.layers.0": params},
    }
    extension._build_hf_to_gen_backend_mapping = lambda _info: {
        **(
            {
                gate_name: (
                    w13_runtime,
                    (slice(None), slice(0, 64), slice(None)),
                ),
                up_name: (
                    w13_runtime,
                    (slice(None), slice(64, 128), slice(None)),
                ),
            }
            if gated
            else {up_name: (w13_runtime, None)}
        ),
        down_name: (w2_runtime, None),
    }

    serializer_calls = []
    serializer_refit = 0

    def serialize(tensors, *, mode, calibration, expected_names=None):
        nonlocal serializer_refit
        serializer_calls.append((dict(tensors), mode, calibration, expected_names))
        if len(serializer_calls) % 4 == 1:
            serializer_refit += 1
        serialized = []
        for name, tensor in tensors.items():
            serialized.append((name, tensor.to(torch.uint8)))
            if mode == "w4a4":
                serialized.append(
                    (
                        name.removesuffix(".weight") + ".input_scale",
                        torch.tensor(0.25 if serializer_refit == 1 else 0.75),
                    )
                )
        return serialized

    loaded_batches = []
    monkeypatch.setattr(backend, "serialize_bf16_nvfp4_group", serialize)
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "_load_weights",
        lambda _self, weights: loaded_batches.append(list(weights)) or "loaded",
    )
    monkeypatch.setattr(
        backend,
        "_detach_pending_layerwise_weights",
        lambda *_args: None,
    )

    param_map = extension.build_hf_to_local_param_map(refit_info)
    gate_spec = param_map.get(gate_name)
    up_spec = param_map.get(up_name)
    down_spec = param_map.get(down_name)
    if gated:
        assert gate_spec is not None and gate_spec.pre is not None and gate_spec.post
    else:
        assert gate_spec is None
    assert up_spec is not None and up_spec.pre is not None and up_spec.post
    assert down_spec is not None and down_spec.pre is not None and down_spec.post

    for refit_value in (1.0, 2.0):
        up_ctx = up_spec.pre(up_spec.base)
        assert up_ctx.buf.shape == (2, 64, 32)
        assert up_ctx.buf.dtype == torch.bfloat16
        assert (
            up_ctx.buf.untyped_storage().data_ptr()
            != w13_runtime.untyped_storage().data_ptr()
        )
        up_ctx.buf.fill_(refit_value + 1)
        up_spec.post(up_ctx)
        if gated:
            with pytest.raises(
                RuntimeError, match="collective group.*missing.*gate_proj"
            ):
                extension._require_complete_bf16_refit_groups()

            assert gate_spec is not None and gate_spec.pre is not None
            assert gate_spec.post is not None
            gate_ctx = gate_spec.pre(gate_spec.base)
            gate_ctx.buf.fill_(refit_value)
            gate_spec.post(gate_ctx)
        else:
            extension._require_complete_bf16_refit_groups()

        down_ctx = down_spec.pre(down_spec.base)
        down_ctx.buf.fill_(refit_value + 2)
        down_spec.post(down_ctx)
        extension._require_complete_bf16_refit_groups()

    assert len(serializer_calls) == 8
    for tensors, actual_mode, calibration, expected_names in serializer_calls:
        assert actual_mode == mode
        assert calibration is extension._nrl_bf16_calibration
        assert len(tensors) in {1, 2}
        assert {tensor.ndim for tensor in tensors.values()} == {2}
        assert {name.rsplit(".", 3)[-3] for name in tensors} == {"0"} or {
            name.rsplit(".", 3)[-3] for name in tensors
        } == {"1"}
        projections = {name.rsplit(".", 2)[-2] for name in tensors}
        allowed_projections = (
            ({"gate_proj", "up_proj"}, {"down_proj"})
            if gated
            else ({"up_proj"}, {"down_proj"})
        )
        assert projections in allowed_projections
        if not gated and projections == {"up_proj"}:
            assert expected_names == tuple(tensors)
        else:
            assert expected_names is None
    assert torch.equal(w13_runtime, torch.full_like(w13_runtime, 255))
    assert torch.equal(w2_runtime, torch.full_like(w2_runtime, 127))
    if mode == "w4a4":
        loaded_scales = [
            tensor
            for batch in loaded_batches
            for name, tensor in batch
            if name.endswith(".input_scale")
        ]
        assert len(loaded_scales) == len(per_expert_names) * 2
        assert all(scale.item() == pytest.approx(0.25) for scale in loaded_scales)


def test_nccl_reshard_wraps_bulk_and_misc_in_one_collective_lifecycle(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    from nemo_rl.weight_sync import xferdtensor as xferdtensor_module
    from nemo_rl.weight_sync.nccl_reshard_utils import (
        HFToLocalParamMap,
        LocalParamSpec,
    )

    name = "model.layers.0.mlp.down_proj.weight"
    buffer = torch.zeros((2, 2))
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.nccl_reshard_refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "src_mesh_info": object(),
                    "dst_mesh_info": object(),
                    "components": [
                        {
                            "role": "weight",
                            "global_shape": (2, 2),
                            "dtype": "torch.float32",
                            "src_placements": [object()],
                            "dst_placements": [object()],
                        }
                    ],
                }
            ]
        },
    }
    extension.hf_to_local_param_map = HFToLocalParamMap(
        specs={name: LocalParamSpec(base=buffer)}
    )
    extension.pp_comm_groups = {0: object()}
    calls = []

    @contextmanager
    def lifecycle(transport):
        calls.append(("enter", transport))

        def finalize():
            calls.append("finalize")

        yield finalize
        calls.append("exit")

    extension._weight_update_lifecycle = lifecycle
    extension._receive_and_load_misc_params = lambda: calls.append("misc")
    monkeypatch.setattr(
        xferdtensor_module,
        "xferdtensor",
        lambda *_args, **_kwargs: calls.append("bulk"),
    )
    monkeypatch.setattr(backend.torch.cuda, "Stream", lambda: object())
    monkeypatch.setattr(backend.torch.cuda, "stream", lambda _stream: nullcontext())

    class FakeEvent:
        def record(self):
            return None

    monkeypatch.setattr(backend.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(backend.torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(backend.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(backend.torch.distributed, "get_rank", lambda: 1)

    assert extension.nccl_reshard_refit() is True
    assert calls == [
        ("enter", "nccl_reshard"),
        "bulk",
        "sync",
        "misc",
        "sync",
        "finalize",
        "sync",
        "exit",
    ]


def test_real_quant_collective_reload_raises_on_failure(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]

    model = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=object(),
    )
    extension.model_config = object()
    extension.device = torch.device("cpu")
    extension.state_dict_info = {}
    extension.model_update_group = object()
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "initialize_layerwise_reload",
        lambda model_arg: calls.append(("initialize", model_arg)),
    )

    def _raise_consume(**kwargs):
        raise ValueError("broadcast boom")

    monkeypatch.setattr(base_backend, "packed_broadcast_consumer", _raise_consume)
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda _model, _model_config: pytest.fail(
            "a failed transfer must not be finalized"
        ),
    )

    with pytest.raises(RuntimeError, match="collective refit failed"):
        extension.update_weights_from_collective()
    assert calls == [("initialize", model)]


def test_non_real_quant_collective_reload_delegates(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: False,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "update_weights_from_collective",
        lambda self: "delegated",
    )

    assert extension.update_weights_from_collective() == "delegated"


def test_real_quant_ipc_complete_finalizes_vllm_layerwise_reload_and_acks(
    monkeypatch,
):
    backend = _import_vllm_quant_backend(monkeypatch)
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    from nemo_rl.models.policy.utils import IPCProtocol

    class FakeSocket:
        def __init__(self):
            self.sent = []

        def recv_pyobj(self):
            return IPCProtocol.COMPLETE

        def send(self, payload):
            self.sent.append(payload)

    model = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    model_config = object()
    socket = FakeSocket()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=object(),
    )
    extension.model_config = model_config
    extension.device = torch.device("cpu")
    extension.zmq_socket = socket
    extension.state_dict_info = {}
    extension.maybe_init_zmq = lambda: None
    calls = []

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "initialize_layerwise_reload",
        lambda model_arg: calls.append(("initialize", model_arg)),
    )
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda model_arg, config_arg: calls.append(("finalize", model_arg, config_arg)),
    )
    monkeypatch.setattr(
        backend.torch.accelerator,
        "synchronize",
        lambda: calls.append("sync"),
    )
    monkeypatch.setattr(
        backend.torch.cuda, "empty_cache", lambda: calls.append("empty")
    )

    assert extension.update_weights_via_ipc_zmq() is True
    assert calls == [
        ("initialize", model),
        ("finalize", model, model_config),
        "sync",
        "empty",
    ]
    assert socket.sent == [IPCProtocol.ACK.value.encode()]


def test_real_quant_ipc_finalize_failure_acks_complete(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    from nemo_rl.models.policy.utils import IPCProtocol

    socket = types.SimpleNamespace(
        recv_pyobj=lambda: IPCProtocol.COMPLETE,
        sent=[],
    )
    socket.send = socket.sent.append
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=_mark_as_modelopt_layer(torch.nn.Linear(1, 1)),
        vllm_config=object(),
    )
    extension.model_config = object()
    extension.device = torch.device("cpu")
    extension.zmq_socket = socket
    extension.state_dict_info = {}
    extension.maybe_init_zmq = lambda: None
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )

    def fail_finalize(_model, _model_config):
        raise RuntimeError("bad scales")

    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        fail_finalize,
    )

    with pytest.raises(
        RuntimeError, match="ModelOpt real-quant refit post-processing failed"
    ):
        extension.update_weights_via_ipc_zmq()
    assert socket.sent == [IPCProtocol.ACK.value.encode()]


@pytest.mark.parametrize(
    ("payload_groups", "state_dict_info", "error"),
    [
        (
            [["decoder.weight"]],
            {
                "decoder.weight": ([1], torch.float32),
                "decoder.bias": ([1], torch.float32),
            },
            "missing keys",
        ),
        (
            [["decoder.weight"], ["decoder.weight"]],
            {"decoder.weight": ([1], torch.float32)},
            "duplicate keys",
        ),
        (
            [["decoder.weight", "decoder.weight"]],
            {"decoder.weight": ([1], torch.float32)},
            "duplicate keys",
        ),
        (
            [["unexpected.weight"]],
            {"decoder.weight": ([1], torch.float32)},
            "unexpected keys",
        ),
    ],
)
def test_real_quant_ipc_rejects_invalid_key_manifest(
    monkeypatch, payload_groups, state_dict_info, error
):
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    from nemo_rl.models.policy.utils import IPCProtocol

    payload_buffer = torch.tensor([1.0], dtype=torch.float32).view(torch.uint8)
    used_bytes = base_backend.calculate_aligned_size(payload_buffer.numel())
    payloads = [
        ("ipc-handle", keys, used_bytes * len(keys)) for keys in payload_groups
    ] + [IPCProtocol.COMPLETE]

    class FakeSocket:
        def __init__(self):
            self.payloads = iter(payloads)
            self.sent = []

        def recv_pyobj(self):
            return next(self.payloads)

        def send(self, payload):
            self.sent.append(payload)

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        vllm_config=object(),
    )
    extension.model_config = object()
    extension.device = torch.device("cuda:0")
    extension.zmq_socket = FakeSocket()
    extension.state_dict_info = state_dict_info
    extension.maybe_init_zmq = lambda: None
    extension._load_weights = lambda _weights: None
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda _model, _model_config: pytest.fail(
            "an invalid refit must not be finalized"
        ),
    )
    monkeypatch.setattr(
        base_backend,
        "rebuild_cuda_tensor_from_ipc",
        lambda _ipc_handle, _device_index: payload_buffer,
    )
    monkeypatch.setattr(
        base_backend.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(backend.torch.accelerator, "synchronize", lambda: None)

    with pytest.raises(RuntimeError, match=error):
        extension.update_weights_via_ipc_zmq()
    assert extension.zmq_socket.sent == [IPCProtocol.ACK.value.encode()] * len(payloads)


def test_real_quant_ipc_payload_loads_weights_and_handles_gpt_oss(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)
    base_backend = _base_vllm_backend()
    reload_mod = sys.modules["vllm.model_executor.model_loader.reload"]
    from nemo_rl.models.policy.utils import IPCProtocol

    payload_weight = torch.tensor([1.0, 2.0], dtype=torch.float32)
    payload_buffer = payload_weight.view(torch.uint8)
    used_bytes = base_backend.calculate_aligned_size(payload_weight.nbytes)
    loaded = []
    calls = []
    view_refs = []

    class FakeSocket:
        def __init__(self):
            self.payloads = iter(
                [
                    ("ipc-handle", ["decoder.weight"], used_bytes),
                    ("ipc-handle", ["decoder.bias"], used_bytes),
                    IPCProtocol.COMPLETE,
                ]
            )
            self.sent = []

        def recv_pyobj(self):
            return next(self.payloads)

        def send(self, payload):
            if len(self.sent) < 2:
                assert view_refs
                assert all(view_ref() is None for view_ref in view_refs)
                calls.append("views_released")
            self.sent.append(payload)

    model = _mark_as_modelopt_layer(torch.nn.Linear(1, 1))
    model_config = object()
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(architectures=["GptOssForCausalLM"])
        ),
    )
    extension.model_config = model_config
    extension.device = torch.device("cuda:0")
    extension.zmq_socket = FakeSocket()
    extension.state_dict_info = {
        "decoder.weight": ([2], torch.float32),
        "decoder.bias": ([2], torch.float32),
    }
    extension.maybe_init_zmq = lambda: None

    def load_weights(weights):
        for name, weight in weights:
            view_refs.append(weakref.ref(weight))
            loaded.append((name, weight.clone()))

    extension._load_weights = load_weights

    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )
    monkeypatch.setattr(
        reload_mod,
        "initialize_layerwise_reload",
        lambda model_arg: calls.append(("initialize", model_arg)),
    )
    monkeypatch.setattr(
        reload_mod,
        "finalize_layerwise_reload",
        lambda model_arg, config_arg: calls.append(("finalize", model_arg, config_arg)),
    )
    monkeypatch.setattr(
        base_backend,
        "rebuild_cuda_tensor_from_ipc",
        lambda ipc_handle, device_index: payload_buffer,
    )
    monkeypatch.setattr(
        base_backend.torch.cuda,
        "current_stream",
        lambda: pytest.fail("real quant must not use a current-stream IPC ACK fence"),
    )
    monkeypatch.setattr(
        backend.torch.accelerator,
        "synchronize",
        lambda: calls.append("sync"),
    )
    monkeypatch.setattr(
        backend.torch.cuda, "empty_cache", lambda: calls.append("empty")
    )
    monkeypatch.setattr(base_backend.gc, "collect", lambda: calls.append("gc"))

    assert extension.update_weights_via_ipc_zmq() is True

    assert extension.zmq_socket.sent == [
        IPCProtocol.ACK.value.encode(),
        IPCProtocol.ACK.value.encode(),
        IPCProtocol.ACK.value.encode(),
    ]
    assert [name for name, _ in loaded] == ["decoder.weight", "decoder.bias"]
    for _, loaded_weight in loaded:
        torch.testing.assert_close(loaded_weight, payload_weight)
    assert calls == [
        ("initialize", model),
        "sync",
        "views_released",
        "sync",
        "views_released",
        ("finalize", model, model_config),
        "sync",
        "gc",
        "empty",
    ]


def test_non_real_quant_ipc_delegates(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: False,
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "update_weights_via_ipc_zmq",
        lambda self: "delegated",
    )

    assert extension.update_weights_via_ipc_zmq() == "delegated"


def test_weight_snapshot_returns_cpu_clone_and_missing_name_raises(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    model = torch.nn.Linear(2, 1, bias=False)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)

    snapshot = extension.get_weight_snapshot("weight")
    model.weight.data.add_(1.0)

    assert snapshot.device.type == "cpu"
    assert not torch.equal(snapshot, model.weight.detach().cpu())
    with pytest.raises(KeyError, match="missing"):
        extension.get_weight_snapshot("missing")


def test_get_quantizer_stats_counts_enabled_positive_amax(monkeypatch):
    backend = _import_vllm_quant_backend(monkeypatch)

    class FakeQuantizer(torch.nn.Module):
        def __init__(self, enabled, amax):
            super().__init__()
            self.is_enabled = enabled
            self.amax = amax

    model = torch.nn.Module()
    model.q_enabled_positive = FakeQuantizer(True, torch.tensor([1.0]))
    model.q_enabled_missing = FakeQuantizer(True, None)
    model.q_disabled_positive = FakeQuantizer(False, torch.tensor([2.0]))
    model.q_enabled_zero = FakeQuantizer(True, torch.tensor([0.0]))
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(model=model)
    monkeypatch.setattr(backend, "TensorQuantizer", FakeQuantizer)

    assert extension.get_quantizer_stats() == {
        "total": 4,
        "enabled": 3,
        "with_amax": 2,
        "positive_amax": 1,
    }


def _nvfp4_source_format() -> dict:
    return {
        "num_bits": "e2m1",
        "block_sizes": {
            -1: 16,
            "type": "dynamic",
            "scale_bits": "e4m3",
        },
    }


def test_resolve_nvfp4_real_quant_mode_detects_model_specific_w4a16(monkeypatch):
    resolved = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*mixer.experts.*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
        ],
        "algorithm": "max",
    }
    monkeypatch.setattr(modelopt_utils, "resolve_quant_cfg", lambda _: resolved)

    assert resolve_nvfp4_real_quant_mode("custom-nvfp4-config") == "w4a16"


def test_resolve_nvfp4_real_quant_mode_detects_w4a4(monkeypatch):
    resolved = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*mlp.experts*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*input_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*input_quantizer",
                "parent_class": "nn.LeakyReLU",
                "enable": False,
            },
        ],
        "algorithm": "max",
    }
    monkeypatch.setattr(modelopt_utils, "resolve_quant_cfg", lambda _: resolved)

    assert resolve_nvfp4_real_quant_mode("not-named-after-the-format") == "w4a4"


def test_resolve_nvfp4_real_quant_mode_honors_late_generic_disable(monkeypatch):
    resolved = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*input_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {"quantizer_name": "*input_quantizer", "enable": False},
        ],
        "algorithm": "max",
    }
    monkeypatch.setattr(modelopt_utils, "resolve_quant_cfg", lambda _: resolved)

    assert resolve_nvfp4_real_quant_mode("disabled-input") == "w4a16"


@pytest.mark.parametrize(
    ("weight_format", "input_format", "error_match"),
    [
        (
            {"num_bits": "e4m3", "axis": None},
            {"num_bits": "e4m3", "axis": None},
            "only block-16 NVFP4.*weights",
        ),
        (
            _nvfp4_source_format(),
            {"num_bits": "e4m3", "axis": None},
            "only block-16 NVFP4.*input activations",
        ),
        (
            _nvfp4_source_format(),
            [_nvfp4_source_format()],
            "single NVFP4 input activations format",
        ),
        (
            _nvfp4_source_format(),
            {
                "num_bits": "e2m1",
                "block_sizes": {
                    -1: 32,
                    "type": "dynamic",
                    "scale_bits": "e4m3",
                },
            },
            "only block-16 NVFP4.*input activations",
        ),
    ],
    ids=["fp8", "w4a8", "sequential-activation", "unsupported-nvfp4-block"],
)
def test_resolve_nvfp4_real_quant_mode_rejects_unsupported_formats(
    monkeypatch,
    weight_format,
    input_format,
    error_match,
):
    resolved = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": weight_format},
            {"quantizer_name": "*input_quantizer", "cfg": input_format},
        ],
        "algorithm": "max",
    }
    monkeypatch.setattr(modelopt_utils, "resolve_quant_cfg", lambda _: resolved)

    with pytest.raises(ValueError, match=error_match):
        resolve_nvfp4_real_quant_mode("unsupported-real-quant-config")


def test_resolve_nvfp4_real_quant_mode_rejects_mixed_activation_formats(
    monkeypatch,
):
    resolved = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*mixer.experts.*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*weight_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mixer.experts.*input_quantizer",
                "cfg": _nvfp4_source_format(),
            },
            {
                "quantizer_name": "*mlp.experts*input_quantizer",
                "cfg": {"num_bits": "e4m3", "axis": None},
            },
        ],
        "algorithm": "max",
    }
    monkeypatch.setattr(modelopt_utils, "resolve_quant_cfg", lambda _: resolved)

    with pytest.raises(ValueError, match="only block-16 NVFP4.*input activations"):
        resolve_nvfp4_real_quant_mode("mixed-input-formats")


def test_resolve_quant_cfg_passes_relative_names_to_modelopt(monkeypatch):
    modelopt_recipe = pytest.importorskip("modelopt.recipe")
    captured = {}

    def fake_load_config(config_file):
        captured["config_file"] = config_file
        return {"quant_cfg": [{"name": "mock"}], "algorithm": "max"}

    monkeypatch.setattr(modelopt_recipe, "load_config", fake_load_config)

    assert resolve_quant_cfg("examples/modelopt/quant_configs/nvfp4_a16.yaml") == {
        "quant_cfg": [{"name": "mock"}],
        "algorithm": "max",
    }

    assert captured["config_file"] == "examples/modelopt/quant_configs/nvfp4_a16.yaml"


def test_resolve_quant_cfg_accepts_builtin_modelopt_constant(monkeypatch):
    mtq = pytest.importorskip("modelopt.torch.quantization")
    sentinel = {"quant_cfg": [{"name": "builtin"}], "algorithm": "max"}
    monkeypatch.setattr(mtq, "UNIT_TEST_CFG", sentinel, raising=False)

    assert resolve_quant_cfg("UNIT_TEST_CFG") is sentinel


def test_resolve_quant_cfg_defaults_missing_algorithm_to_max(monkeypatch):
    modelopt_recipe = pytest.importorskip("modelopt.recipe")

    monkeypatch.setattr(
        modelopt_recipe,
        "load_config",
        lambda config_name: {"quant_cfg": [{"name": config_name}]},
    )

    assert resolve_quant_cfg("unit-test-recipe") == {
        "quant_cfg": [{"name": "unit-test-recipe"}],
        "algorithm": "max",
    }


def test_resolve_quant_cfg_extracts_nested_quantize_section(monkeypatch):
    modelopt_recipe = pytest.importorskip("modelopt.recipe")

    monkeypatch.setattr(
        modelopt_recipe,
        "load_config",
        lambda config_name: {
            "quantize": {
                "quant_cfg": [{"name": config_name}],
                "algorithm": "max",
            }
        },
    )

    assert resolve_quant_cfg("unit-test-recipe") == {
        "quant_cfg": [{"name": "unit-test-recipe"}],
        "algorithm": "max",
    }


def test_resolve_quant_cfg_rejects_unknown_config(monkeypatch):
    modelopt_recipe = pytest.importorskip("modelopt.recipe")

    def fake_load_config(config_name):
        raise FileNotFoundError(config_name)

    monkeypatch.setattr(modelopt_recipe, "load_config", fake_load_config)

    with pytest.raises(ValueError, match="Unknown quant_cfg"):
        resolve_quant_cfg("does-not-exist")


def test_resolve_quant_cfg_rejects_recipe_without_quant_cfg(monkeypatch):
    modelopt_recipe = pytest.importorskip("modelopt.recipe")
    monkeypatch.setattr(modelopt_recipe, "load_config", lambda config_name: {})

    with pytest.raises(ValueError, match="must contain a 'quant_cfg'"):
        resolve_quant_cfg("missing-quant-cfg")


def test_register_nemo_modelopt_nvfp4_uses_public_vllm_registry(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)

    register_nemo_modelopt_nvfp4()

    assert set(fake_vllm.registry) == {
        NEMO_MODELOPT_W4A4,
        NEMO_MODELOPT_W4A16,
    }
    w4a4_config = fake_vllm.registry[NEMO_MODELOPT_W4A4]()
    assert w4a4_config.get_name() == NEMO_MODELOPT_W4A4

    source_config = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    w4a16_config = fake_vllm.registry[NEMO_MODELOPT_W4A16].from_config(source_config)
    assert source_config["quant_algo"] == "W4A16_NVFP4"
    # vLLM 0.25 understands W4A16_NVFP4 natively; the algo passes through.
    assert w4a16_config.parsed_config["quant_algo"] == "W4A16_NVFP4"
    assert w4a16_config.get_name() == NEMO_MODELOPT_W4A16
    # The base __init__ installs its own LinearMethodCls instance attribute;
    # the NeMo config must rebind it to the refit-friendly Marlin method.
    assert w4a16_config.LinearMethodCls.__name__ == "NemoModelOptW4A16LinearMethod"

    with pytest.raises(ValueError, match="requires quant_algo='W4A16_NVFP4'"):
        fake_vllm.registry[NEMO_MODELOPT_W4A16].from_config({"quant_algo": "NVFP4"})


def test_registered_configs_select_only_the_exact_custom_override(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    w4a4_config_cls = fake_vllm.registry[NEMO_MODELOPT_W4A4]
    w4a16_config_cls = fake_vllm.registry[NEMO_MODELOPT_W4A16]

    assert (
        w4a4_config_cls.override_quantization_method(
            {"quant_algo": "NVFP4"}, NEMO_MODELOPT_W4A4
        )
        == NEMO_MODELOPT_W4A4
    )
    assert (
        w4a16_config_cls.override_quantization_method(
            {"quantization": {"quant_algo": "W4A16_NVFP4"}},
            NEMO_MODELOPT_W4A16,
        )
        == NEMO_MODELOPT_W4A16
    )
    assert (
        w4a4_config_cls.override_quantization_method(
            {"quant_algo": "NVFP4"}, NEMO_MODELOPT_W4A16
        )
        is None
    )
    assert (
        w4a4_config_cls.override_quantization_method(
            {"quant_algo": "W4A16_NVFP4"}, NEMO_MODELOPT_W4A4
        )
        is None
    )
    assert (
        w4a16_config_cls.override_quantization_method(
            {"quant_algo": "W4A16_NVFP4"}, "modelopt"
        )
        is None
    )


def test_registered_w4a16_dense_method_supports_weight_loader_v2(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)

    register_nemo_modelopt_nvfp4()

    w4a16_config_cls = fake_vllm.registry[NEMO_MODELOPT_W4A16]
    assert fake_vllm.weight_loader_v2_supported == [
        w4a16_config_cls.LinearMethodCls.__name__
    ]


def test_registered_w4a4_moe_loader_is_sanitizer_compatible(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()

    config = fake_vllm.registry[NEMO_MODELOPT_W4A4]()
    quant_method = config.FusedMoEMethodCls(config, object())

    class FakeMoeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = quant_method
            self.w13_input_scale = torch.nn.Parameter(torch.zeros(2, 2))
            self.w2_input_scale = torch.nn.Parameter(torch.zeros(2))

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return expert_id

    layer = FakeMoeLayer()
    quant_method.create_weights(layer)

    w13_loader = layer.w13_input_scale.weight_loader
    assert isinstance(w13_loader, types.MethodType)
    assert w13_loader.__self__ is layer

    layer_ref_sentinel = object()
    layer.w13_input_scale.weight_loader = w13_loader.__func__.__get__(
        layer_ref_sentinel
    )
    assert layer.w13_input_scale.weight_loader.__self__ is layer_ref_sentinel
    layer.w13_input_scale.weight_loader = (
        layer.w13_input_scale.weight_loader.__func__.__get__(layer)
    )
    w13_loader = layer.w13_input_scale.weight_loader
    assert w13_loader.__self__ is layer

    assert w13_loader(
        layer.w13_input_scale,
        torch.tensor(1.0),
        "gate.input_scale",
        "w1",
        0,
        True,
    )
    assert w13_loader(
        layer.w13_input_scale,
        torch.tensor(2.0),
        "up.input_scale",
        "w3",
        0,
        True,
    )
    assert layer.w2_input_scale.weight_loader(
        layer.w2_input_scale,
        torch.tensor(3.0),
        "down.input_scale",
        "w2",
        1,
        True,
    )

    torch.testing.assert_close(layer.w13_input_scale[0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(layer.w2_input_scale, torch.tensor([0.0, 3.0]))


def test_registered_w4a4_moe_materializes_initial_input_scales(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    config = fake_vllm.registry[NEMO_MODELOPT_W4A4]()
    quant_method = config.FusedMoEMethodCls(config, object())

    layer = torch.nn.Module()
    w13_input_scale = torch.nn.Parameter(
        torch.tensor([2.0]).expand(4), requires_grad=False
    )
    w2_input_scale = torch.nn.Parameter(
        torch.tensor([3.0]).expand(4), requires_grad=False
    )
    layer.register_parameter("w13_input_scale", w13_input_scale)
    layer.register_parameter("w2_input_scale", w2_input_scale)

    quant_method.process_weights_after_loading(layer)

    assert layer.w13_input_scale is w13_input_scale
    assert layer.w2_input_scale is w2_input_scale
    assert layer.w13_input_scale.is_contiguous()
    assert layer.w2_input_scale.is_contiguous()
    torch.testing.assert_close(layer.w13_input_scale, torch.full((4,), 2.0))
    torch.testing.assert_close(layer.w2_input_scale, torch.full((4,), 3.0))
    with torch.no_grad():
        layer.w13_input_scale.copy_(torch.arange(4, dtype=torch.float32))
        layer.w2_input_scale.copy_(torch.arange(4, dtype=torch.float32))


def test_registered_w4a4_moe_refreshes_stable_activation_scales(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    config = fake_vllm.registry[NEMO_MODELOPT_W4A4]()
    quant_method = config.FusedMoEMethodCls(config, object())

    original_kernel = object()
    original_a1_gscale = torch.full((4,), 1.0)
    original_a2_gscale = torch.full((4,), 0.5)
    original_quant_config = types.SimpleNamespace(
        a1_gscale=original_a1_gscale,
        a2_gscale=original_a2_gscale,
    )
    quant_method.moe_kernel = original_kernel
    quant_method.moe_quant_config = original_quant_config
    a1_data_ptr = original_a1_gscale.data_ptr()
    a2_data_ptr = original_a2_gscale.data_ptr()

    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_input_scale",
        torch.nn.Parameter(torch.full((4,), 4.0), requires_grad=False),
    )
    layer.register_parameter(
        "w2_input_scale",
        torch.nn.Parameter(torch.full((4,), 5.0), requires_grad=False),
    )

    quant_method.process_weights_after_loading(layer)

    assert quant_method.moe_kernel is original_kernel
    assert quant_method.moe_quant_config is original_quant_config
    assert original_quant_config.a1_gscale.data_ptr() == a1_data_ptr
    assert original_quant_config.a2_gscale.data_ptr() == a2_data_ptr
    torch.testing.assert_close(original_quant_config.a1_gscale, torch.full((4,), 0.25))
    torch.testing.assert_close(original_quant_config.a2_gscale, torch.full((4,), 0.2))


def test_registered_w4a16_dense_method_uses_marlin_weight_only(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    config = fake_vllm.registry[NEMO_MODELOPT_W4A16].from_config(
        {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    )
    quant_method = config.LinearMethodCls(config)

    created_layer = torch.nn.Module()
    quant_method.create_weights(created_layer)
    assert not hasattr(created_layer, "input_scale")

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([[-1.0, 2.0], [0.5, -4.0]]),
        requires_grad=False,
    )
    layer.weight_scale_2 = torch.nn.Parameter(
        torch.tensor([2.0, 3.0]),
        requires_grad=False,
    )
    layer.output_size_per_partition = 2
    layer.input_size_per_partition = 2

    quant_method.process_weights_after_loading(layer)
    output = quant_method.apply(layer, torch.ones(1, 2))

    assert output == "output"
    assert not hasattr(layer, "weight_scale_2")
    torch.testing.assert_close(
        layer.weight_scale,
        torch.tensor([[1.0, 2.0], [0.5, 4.0]]),
    )
    torch.testing.assert_close(layer.weight_global_scale, torch.tensor(3.0))
    assert fake_vllm.events[0] == ("process_marlin_kernel", layer)
    event_name, kernel_args = fake_vllm.events[1]
    assert event_name == "apply_marlin_kernel"
    assert kernel_args["layer"] is layer
    torch.testing.assert_close(kernel_args["x"], torch.ones(1, 2))
    assert kernel_args["bias"] is None


def test_registered_w4a16_moe_create_weights_keeps_checkpoint_layout(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    config = fake_vllm.registry[NEMO_MODELOPT_W4A16].from_config(
        {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    )
    quant_method = config.FusedMoEMethodCls(
        config,
        types.SimpleNamespace(is_act_and_mul=False),
    )
    layer = torch.nn.Module()

    quant_method.create_weights(
        layer,
        num_experts=2,
        hidden_size=4096,
        intermediate_size_per_partition=672,
        params_dtype=torch.bfloat16,
    )

    assert not hasattr(layer, "w13_input_scale")
    assert not hasattr(layer, "w2_input_scale")
    assert fake_vllm.events == [
        (
            "native_create_weights",
            layer,
            (2, 4096, 672, torch.bfloat16),
            {},
        )
    ]


def test_registered_w4a16_moe_preserves_kernel_during_reload(monkeypatch):
    fake_vllm = _install_fake_registered_vllm_modelopt(monkeypatch)
    monkeypatch.setattr(vllm_modelopt, "_registered", False)
    register_nemo_modelopt_nvfp4()
    config = fake_vllm.registry[NEMO_MODELOPT_W4A16].from_config(
        {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    )
    quant_method = config.FusedMoEMethodCls(
        config,
        types.SimpleNamespace(is_act_and_mul=False),
    )
    original_kernel = object()
    original_quant_config = object()
    quant_method.moe_kernel = original_kernel
    quant_method.moe_quant_config = original_quant_config

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones(1, 80, 32))
    layer.w13_weight_scale = torch.nn.Parameter(-torch.ones(1, 80, 2))
    layer.w13_weight_scale_2 = torch.nn.Parameter(torch.ones(1, 1))
    layer.w2_weight = torch.nn.Parameter(torch.ones(1, 2, 40))
    layer.w2_weight_scale = torch.nn.Parameter(-torch.ones(1, 2, 5))
    layer.w2_weight_scale_2 = torch.nn.Parameter(torch.ones(1))
    layer.moe_config = types.SimpleNamespace(intermediate_size_per_partition=80)
    layer.shared_experts = None
    layer._maybe_init_expert_routing_tables = lambda: None

    quant_method.process_weights_after_loading(layer)

    assert quant_method.moe_kernel is original_kernel
    assert quant_method.moe_quant_config is original_quant_config
    # vLLM 0.25's native Marlin converter owns tile padding, so the NeMo
    # override leaves shapes and moe_config untouched and only canonicalizes
    # the ModelOpt sign-carrying scales in place.
    assert layer.moe_config.intermediate_size_per_partition == 80
    assert layer.w13_weight.shape == (1, 80, 32)
    assert layer.w13_weight_scale.shape == (1, 80, 2)
    assert layer.w2_weight.shape == (1, 2, 40)
    assert layer.w2_weight_scale.shape == (1, 2, 5)
    assert torch.all(layer.w13_weight_scale >= 0)
    assert torch.all(layer.w2_weight_scale >= 0)
    assert fake_vllm.events == [("native_process_moe", 80)]
