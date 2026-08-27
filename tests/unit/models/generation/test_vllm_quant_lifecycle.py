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

import ast
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest


def _load_real_quant_lifecycle(*, torch_module, require_complete):
    source_path = (
        Path(__file__).parents[4]
        / "nemo_rl/modelopt/models/generation/vllm_quant_backend.py"
    )
    module = ast.parse(source_path.read_text())
    lifecycle = next(
        item
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VllmQuantInternalWorkerExtension"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "_weight_update_lifecycle"
    )
    extracted = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            lifecycle,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(extracted)

    class IPCWeightManifestError(Exception):
        pass

    namespace = {
        "IPCWeightManifestError": IPCWeightManifestError,
        "_require_complete_modelopt_layerwise_reload": require_complete,
        "contextmanager": contextmanager,
        "torch": torch_module,
    }
    exec(compile(extracted, source_path, "exec"), namespace)
    return namespace["_weight_update_lifecycle"]


def _load_base_lifecycle(*, torch_module, refresh_hpc):
    source_path = (
        Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/vllm_backend.py"
    )
    module = ast.parse(source_path.read_text())
    lifecycle = next(
        item
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "VllmInternalWorkerExtension"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "_weight_update_lifecycle"
    )
    extracted = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            lifecycle,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(extracted)
    namespace = {
        "_refresh_hpc_modules_after_layerwise_reload": refresh_hpc,
        "contextmanager": contextmanager,
        "torch": torch_module,
    }
    exec(compile(extracted, source_path, "exec"), namespace)
    return namespace["_weight_update_lifecycle"]


def test_native_base_lifecycle_finalizes_external_draft_after_target(monkeypatch):
    calls = []
    config_module = types.ModuleType("vllm.config")
    config_module.set_current_vllm_config = lambda _config: nullcontext()
    reload_module = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.initialize_layerwise_reload = lambda root: calls.append(
        ("initialize", root)
    )
    reload_module.finalize_layerwise_reload = lambda root, config: calls.append(
        ("finalize_target", root, config)
    )
    loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils.process_weights_after_loading = lambda *_args: None
    for module_name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
    ):
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.reload",
        reload_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        loader_utils,
    )

    extension = types.SimpleNamespace(
        device="cpu",
        model_config="target_config",
        model_runner=types.SimpleNamespace(
            model="target_model",
            vllm_config="vllm_config",
        ),
        _nrl_layerwise_reload_active=False,
        _nrl_layerwise_reload_failure=None,
        _uses_native_layerwise_refit=lambda _transport: True,
        _validate_native_layerwise_refit=lambda: None,
        _maybe_process_draft_after_loading=lambda process: calls.append(
            "finalize_draft"
        ),
        _maybe_process_mtp_drafter_after_loading=lambda: None,
        _require_refit_usable=lambda: None,
        _mark_refit_unusable=lambda _error: None,
    )
    lifecycle = _load_base_lifecycle(
        refresh_hpc=lambda root: calls.append(("hpc", root)),
        torch_module=types.SimpleNamespace(
            device=lambda _device: nullcontext(),
            cuda=types.SimpleNamespace(synchronize=lambda: calls.append("sync")),
        ),
    )

    with lifecycle(extension, "collective") as finish:
        calls.append("load")
        finish(True)

    assert calls == [
        ("initialize", "target_model"),
        "load",
        ("finalize_target", "target_model", "target_config"),
        ("hpc", "target_model"),
        "finalize_draft",
        "sync",
    ]


def test_native_base_lifecycle_marks_common_refit_state_unusable(monkeypatch):
    config_module = types.ModuleType("vllm.config")
    config_module.set_current_vllm_config = lambda _config: nullcontext()
    reload_module = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.initialize_layerwise_reload = lambda _root: None
    reload_module.finalize_layerwise_reload = lambda _root, _config: None
    loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils.process_weights_after_loading = lambda *_args: None
    for module_name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
    ):
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.reload",
        reload_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        loader_utils,
    )

    marked_errors = []
    extension = types.SimpleNamespace(
        device="cpu",
        model_config="target_config",
        model_runner=types.SimpleNamespace(
            model="target_model",
            vllm_config="vllm_config",
        ),
        _nrl_layerwise_reload_active=False,
        _nrl_layerwise_reload_failure=None,
        _uses_native_layerwise_refit=lambda _transport: True,
        _validate_native_layerwise_refit=lambda: None,
        _maybe_process_draft_after_loading=lambda _process: None,
        _maybe_process_mtp_drafter_after_loading=lambda: None,
        _require_refit_usable=lambda: None,
        _mark_refit_unusable=marked_errors.append,
    )
    lifecycle = _load_base_lifecycle(
        refresh_hpc=lambda _root: None,
        torch_module=types.SimpleNamespace(
            device=lambda _device: nullcontext(),
            cuda=types.SimpleNamespace(synchronize=lambda: None),
        ),
    )
    failure = RuntimeError("transport failed after native initialization")

    with pytest.raises(RuntimeError, match="transport failed"):
        with lifecycle(extension, "collective"):
            raise failure

    assert extension._nrl_layerwise_reload_failure is failure
    assert marked_errors == [failure]


def test_real_quant_lifecycle_finalizes_cotrained_draft_after_target(monkeypatch):
    calls = []
    config_module = types.ModuleType("vllm.config")
    config_module.set_current_vllm_config = lambda _config: nullcontext()
    reload_module = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.initialize_layerwise_reload = lambda root: calls.append(
        ("initialize", root)
    )
    reload_module.finalize_layerwise_reload = lambda root, config: calls.append(
        ("finalize_target", root, config)
    )
    loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils.process_weights_after_loading = lambda model, config, device: (
        calls.append(("finalize_draft", model, config, device))
    )
    for module_name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
    ):
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.reload",
        reload_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        loader_utils,
    )

    extension = types.SimpleNamespace(
        device="cpu",
        model_config="target_config",
        model_runner=types.SimpleNamespace(
            model="target_model",
            vllm_config="vllm_config",
        ),
        _is_real_quant_model=lambda: True,
        _get_modelopt_reload_roots=lambda: ("target_model",),
        _maybe_process_draft_after_loading=lambda process: process(
            "draft_model", "draft_config", "cpu"
        ),
        _maybe_process_mtp_drafter_after_loading=lambda: None,
    )
    lifecycle = _load_real_quant_lifecycle(
        require_complete=lambda model: calls.append(("require_complete", model)),
        torch_module=types.SimpleNamespace(
            device=lambda _device: nullcontext(),
            accelerator=types.SimpleNamespace(synchronize=lambda: calls.append("sync")),
        ),
    )

    with lifecycle(extension, "collective") as finish:
        calls.append("load")
        finish(True)

    assert calls == [
        ("initialize", "target_model"),
        "load",
        ("require_complete", "target_model"),
        ("finalize_target", "target_model", "target_config"),
        ("finalize_draft", "draft_model", "draft_config", "cpu"),
        "sync",
    ]
