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

# NOTE: vllm_backend imports `vllm` eagerly at module top, so it is only imported
# inside the test bodies (which are marked @pytest.mark.vllm). This keeps the
# module collectable in the non-vllm unit lane, where these tests are deselected.

import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture(autouse=True)
def _stub_top_level_vllm_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("vllm") is None:
        monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))


def _make_collective_update_extension(backend):
    ext = backend.VllmInternalWorkerExtension.__new__(
        backend.VllmInternalWorkerExtension
    )
    state_info = object()
    ext.state_dict_info = {"model.weight": state_info}
    ext.model_update_group = object()
    ext.model_runner = SimpleNamespace(model=object(), vllm_config=object())
    ext.model_config = object()
    ext.device = object()
    return ext, state_info


@pytest.mark.vllm
def test_get_cudagraph_dispatch_metrics_separates_target_and_draft() -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        cudagraph_dispatcher=SimpleNamespace(
            _nrl_cudagraph_dispatch_metrics={
                "calls_none": 2,
                "calls_piecewise": 8,
            }
        ),
        drafter=SimpleNamespace(
            cudagraph_dispatcher=SimpleNamespace(
                _nrl_cudagraph_dispatch_metrics={
                    "calls_none": 1,
                    "calls_piecewise": 9,
                }
            )
        ),
    )

    assert ext.get_cudagraph_dispatch_metrics() == {
        "vllm:spec_decode_cudagraph_target_calls_none": 2.0,
        "vllm:spec_decode_cudagraph_target_calls_piecewise": 8.0,
        "vllm:spec_decode_cudagraph_draft_calls_none": 1.0,
        "vllm:spec_decode_cudagraph_draft_calls_piecewise": 9.0,
    }


@pytest.mark.vllm
def test_get_cudagraph_dispatch_metrics_omits_non_neural_suffix_drafter() -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        cudagraph_dispatcher=SimpleNamespace(
            _nrl_cudagraph_dispatch_metrics={"calls_piecewise": 3}
        ),
        drafter=SimpleNamespace(),
    )

    assert ext.get_cudagraph_dispatch_metrics() == {
        "vllm:spec_decode_cudagraph_target_calls_piecewise": 3.0,
    }


@pytest.mark.vllm
def test_get_cudagraph_dispatch_metrics_supports_v2_speculator_managers() -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(
            _nrl_cudagraph_dispatch_metrics={"calls_piecewise": 8}
        ),
        speculator=SimpleNamespace(
            num_speculative_steps=5,
            prefill_cudagraph_manager=SimpleNamespace(
                _nrl_cudagraph_dispatch_metrics={"calls_piecewise": 5}
            ),
            decode_cudagraph_manager=SimpleNamespace(
                _nrl_cudagraph_dispatch_metrics={"calls_none": 20}
            ),
        ),
    )

    assert ext.get_cudagraph_dispatch_metrics() == {
        "vllm:spec_decode_cudagraph_target_calls_piecewise": 8.0,
        "vllm:spec_decode_cudagraph_draft_prefill_calls_piecewise": 5.0,
        "vllm:spec_decode_cudagraph_draft_decode_calls_none": 80.0,
    }


@pytest.mark.vllm
def test_prepare_refit_info_rejects_empty_weight_manifest() -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)

    with pytest.raises(ValueError, match="refit weight manifest is empty"):
        ext.prepare_refit_info({})


@pytest.mark.vllm
def test_begin_weight_update_rejects_empty_weight_manifest() -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.state_dict_info = {}

    with pytest.raises(RuntimeError, match="refit weight manifest is empty"):
        ext._begin_weight_update()


def _write_sharded_checkpoint(model_dir, shards):
    """Write safetensors shards plus a model.safetensors.index.json.

    Args:
        model_dir: Directory (pathlib.Path) to write the checkpoint into.
        shards: Mapping of shard filename -> {weight_name: tensor}.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for shard_name, tensors in shards.items():
        save_file(tensors, str(model_dir / shard_name))
        for name in tensors:
            weight_map[name] = shard_name
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


def _make_extension_with_drafter(mtp_start_layer_idx, num_mtp_layers):
    """Build a VllmInternalWorkerExtension with a mocked drafter and stubbed refit."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    predictor = SimpleNamespace(
        mtp_start_layer_idx=mtp_start_layer_idx, num_mtp_layers=num_mtp_layers
    )
    ext.model_runner = MagicMock()
    ext.model_runner.drafter.model = SimpleNamespace(model=predictor)
    # Isolate this test from _load_draft_weights internals.
    ext._load_draft_weights = MagicMock()
    return ext


def _patch_vllm_postload(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub the vLLM post-load helpers imported inside load_mtp_weights_from_disk."""

    def set_current_vllm_config(
        _config: object,
    ) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    config_module = ModuleType("vllm.config")
    setattr(config_module, "set_current_vllm_config", set_current_vllm_config)
    process_weights = MagicMock()
    model_executor_module = ModuleType("vllm.model_executor")
    model_loader_module = ModuleType("vllm.model_executor.model_loader")
    model_loader_utils_module = ModuleType("vllm.model_executor.model_loader.utils")
    setattr(model_loader_utils_module, "process_weights_after_loading", process_weights)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor_module)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.model_loader", model_loader_module
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        model_loader_utils_module,
    )
    return process_weights


def _patch_pp_rank(monkeypatch: pytest.MonkeyPatch, *, is_last_rank: bool) -> None:
    parallel_state_module = ModuleType("vllm.distributed.parallel_state")
    setattr(
        parallel_state_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=is_last_rank),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        parallel_state_module,
    )


def _make_extension_for_draft_load(draft_model: object | None) -> Any:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext: Any = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        drafter=None if draft_model is None else SimpleNamespace(model=draft_model)
    )
    return ext


@pytest.mark.vllm
def test_get_draft_model_supports_v2_speculator_owner() -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    draft_model = object()
    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        drafter=None,
        speculator=SimpleNamespace(model=draft_model),
    )

    assert ext._get_draft_model() is draft_model


@pytest.mark.vllm
def test_split_policy_and_draft_weights_routes_mtp_layers_and_buffers() -> None:
    predictor = SimpleNamespace(mtp_start_layer_idx=2, num_mtp_layers=1)
    draft_model = SimpleNamespace(model=predictor)
    ext = _make_extension_for_draft_load(draft_model)
    ext.model_runner.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp")
    )
    policy_weight = torch.randn(1)
    mtp_weight = torch.randn(1)
    mtp_buffer = torch.randn(1)

    policy_weights, draft_weights = ext._split_policy_and_draft_weights(
        [
            ("model.layers.0.self_attn.q_proj.weight", policy_weight),
            ("model.layers.2.self_attn.q_proj.weight", mtp_weight),
            (
                "model.layers.2.mlp.gate.e_score_correction_bias",
                mtp_buffer,
            ),
        ]
    )

    assert policy_weights == [("model.layers.0.self_attn.q_proj.weight", policy_weight)]
    assert draft_weights == [
        ("model.layers.2.self_attn.q_proj.weight", mtp_weight),
        ("model.layers.2.mlp.gate.e_score_correction_bias", mtp_buffer),
    ]


@pytest.mark.vllm
def test_split_policy_and_draft_weights_keeps_eagle_prefix_contract() -> None:
    ext = _make_extension_for_draft_load(draft_model=SimpleNamespace())
    weight = torch.randn(1)

    policy_weights, draft_weights = ext._split_policy_and_draft_weights(
        [("draft.fc.weight", weight)]
    )

    assert policy_weights == []
    assert draft_weights == [("fc.weight", weight)]


@pytest.mark.vllm
def test_split_policy_and_draft_weights_routes_qwen_mtp_namespace() -> None:
    ext = _make_extension_for_draft_load(draft_model=SimpleNamespace())
    ext.model_runner.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp")
    )
    weight = torch.randn(1)

    policy_weights, draft_weights = ext._split_policy_and_draft_weights(
        [("mtp.layers.0.self_attn.q_proj.weight", weight)]
    )

    assert policy_weights == []
    assert draft_weights == [("mtp.layers.0.self_attn.q_proj.weight", weight)]


@pytest.mark.vllm
def test_split_policy_and_draft_weights_routes_mimo_mtp_namespace() -> None:
    ext = _make_extension_for_draft_load(draft_model=SimpleNamespace())
    ext.model_runner.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mimo_mtp")
    )
    weight = torch.randn(1)

    policy_weights, draft_weights = ext._split_policy_and_draft_weights(
        [("model.mtp_layers.0.self_attn.q_proj.weight", weight)]
    )

    assert policy_weights == []
    assert draft_weights == [("model.mtp_layers.0.self_attn.q_proj.weight", weight)]


@pytest.mark.vllm
def test_split_policy_and_draft_weights_keeps_mtp_namespace_without_mtp_specdec() -> (
    None
):
    ext = _make_extension_for_draft_load(draft_model=None)
    ext.model_runner.vllm_config = SimpleNamespace(speculative_config=None)
    weight = torch.randn(1)

    policy_weights, draft_weights = ext._split_policy_and_draft_weights(
        [("mtp.layers.0.self_attn.q_proj.weight", weight)]
    )

    assert policy_weights == [("mtp.layers.0.self_attn.q_proj.weight", weight)]
    assert draft_weights == []


@pytest.mark.vllm
def test_mtp_layer_indices_fall_back_to_draft_model_config() -> None:
    draft_model = SimpleNamespace(model=SimpleNamespace())
    ext = _make_extension_for_draft_load(draft_model)
    ext.model_runner.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_hidden_layers=32, n_predict=2)
            )
        )
    )

    assert ext._get_mtp_layer_indices() == {32, 33}


@pytest.mark.vllm
def test_load_draft_weights_is_noop_for_empty_input_without_drafter() -> None:
    ext = _make_extension_for_draft_load(draft_model=None)

    ext._load_draft_weights([])


@pytest.mark.vllm
def test_load_draft_weights_raises_for_nonempty_input_without_drafter() -> None:
    ext = _make_extension_for_draft_load(draft_model=None)

    with pytest.raises(RuntimeError, match="draft weights.*drafter is unavailable"):
        ext._load_draft_weights([("model.layers.0.weight", torch.randn(1))])


@pytest.mark.vllm
def test_load_draft_weights_calls_loader_once_with_trimmed_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_model = SimpleNamespace(
        load_weights=MagicMock(return_value={"model.layers.0.weight"})
    )
    ext = _make_extension_for_draft_load(draft_model)
    trimmed_weights = [("model.layers.0.weight", torch.randn(1))]
    trim_vocab_padding = MagicMock(return_value=trimmed_weights)
    monkeypatch.setattr(ext, "_trim_vocab_padding", trim_vocab_padding)
    weights = [("model.layers.0.weight", torch.randn(2))]

    ext._load_draft_weights(weights)

    trim_vocab_padding.assert_called_once_with(draft_model, weights)
    draft_model.load_weights.assert_called_once_with(weights=trimmed_weights)


@pytest.mark.vllm
def test_load_draft_weights_rejects_missing_loader_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_model = SimpleNamespace(load_weights=MagicMock(return_value=None))
    ext = _make_extension_for_draft_load(draft_model)
    monkeypatch.setattr(ext, "_trim_vocab_padding", lambda _model, weights: weights)

    with pytest.raises(RuntimeError, match="returned no load receipt"):
        ext._load_draft_weights([("model.layers.0.weight", torch.randn(1))])


@pytest.mark.vllm
def test_load_draft_weights_accepts_nonempty_loaded_name_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_model = SimpleNamespace(
        load_weights=MagicMock(return_value={"model.layers.0.weight"})
    )
    ext = _make_extension_for_draft_load(draft_model)
    monkeypatch.setattr(ext, "_trim_vocab_padding", lambda _model, weights: weights)

    ext._load_draft_weights([("model.layers.0.weight", torch.randn(1))])


@pytest.mark.vllm
def test_load_draft_weights_rejects_empty_loaded_name_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_model = SimpleNamespace(load_weights=MagicMock(return_value=set()))
    ext = _make_extension_for_draft_load(draft_model)
    monkeypatch.setattr(ext, "_trim_vocab_padding", lambda _model, weights: weights)

    with pytest.raises(RuntimeError, match="reported no loaded weights"):
        ext._load_draft_weights([("model.layers.0.weight", torch.randn(1))])


@pytest.mark.vllm
@pytest.mark.parametrize("load_result", [False, object()])
def test_load_draft_weights_rejects_unknown_or_failed_loader_result(
    monkeypatch: pytest.MonkeyPatch,
    load_result: object,
) -> None:
    draft_model = SimpleNamespace(load_weights=MagicMock(return_value=load_result))
    ext = _make_extension_for_draft_load(draft_model)
    monkeypatch.setattr(ext, "_trim_vocab_padding", lambda _model, weights: weights)

    with pytest.raises(RuntimeError, match="drafter loader returned"):
        ext._load_draft_weights([("model.layers.0.weight", torch.randn(1))])


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("load_result", "message"),
    [
        (
            SimpleNamespace(missing_keys=["model.layers.0.weight"], unexpected_keys=[]),
            "missing.*model.layers.0.weight",
        ),
        (
            SimpleNamespace(missing_keys=[], unexpected_keys=["unknown.weight"]),
            "unexpected.*unknown.weight",
        ),
    ],
)
def test_load_draft_weights_rejects_incompatible_loader_result(
    monkeypatch: pytest.MonkeyPatch, load_result: object, message: str
) -> None:
    draft_model = SimpleNamespace(load_weights=MagicMock(return_value=load_result))
    ext = _make_extension_for_draft_load(draft_model)
    monkeypatch.setattr(ext, "_trim_vocab_padding", lambda _model, weights: weights)

    with pytest.raises(RuntimeError, match=message):
        ext._load_draft_weights([("model.layers.0.weight", torch.randn(1))])


@pytest.mark.vllm
def test_weight_update_accumulates_draft_chunks_and_postprocesses_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_model = SimpleNamespace(
        load_weights=MagicMock(return_value={"layer0.weight", "layer1.weight"})
    )
    target_model = SimpleNamespace(load_weights=MagicMock())
    draft_model_config = object()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[]),
        speculative_config=SimpleNamespace(
            method="mtp", draft_model_config=draft_model_config
        ),
    )
    ext = _make_extension_for_draft_load(draft_model)
    ext.model_runner.model = target_model
    ext.model_runner.vllm_config = vllm_config
    ext.model_config = object()
    ext.device = torch.device("cpu")
    ext.state_dict_info = {
        "draft.layer0.weight": object(),
        "draft.layer1.weight": object(),
    }
    monkeypatch.setattr(ext, "_trim_vocab_padding", lambda _model, weights: weights)
    fp8_module = ModuleType("nemo_rl.models.generation.vllm.quantization.fp8")
    setattr(fp8_module, "is_fp8_model", lambda _config: False)
    monkeypatch.setitem(
        sys.modules, "nemo_rl.models.generation.vllm.quantization.fp8", fp8_module
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    ext._begin_weight_update()
    ext._load_weights([("draft.layer0.weight", torch.ones(1))])
    ext._load_weights([("draft.layer1.weight", torch.ones(1))])

    draft_model.load_weights.assert_not_called()
    ext._finish_weight_update()

    loaded_names = [
        name for name, _ in draft_model.load_weights.call_args.kwargs["weights"]
    ]
    assert loaded_names == ["layer0.weight", "layer1.weight"]
    assert all(
        tensor.device.type == "cpu"
        for _, tensor in draft_model.load_weights.call_args.kwargs["weights"]
    )
    assert process_weights.call_args_list == [
        ((target_model, ext.model_config, ext.device),),
        ((draft_model, draft_model_config, ext.device),),
    ]


@pytest.mark.vllm
def test_weight_update_rejects_missing_transport_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_model = SimpleNamespace(load_weights=MagicMock(return_value=None))
    target_model = SimpleNamespace(load_weights=MagicMock())
    ext = _make_extension_for_draft_load(draft_model)
    ext.model_runner.model = target_model
    ext.model_runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[]),
        speculative_config=SimpleNamespace(
            method="mtp",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_hidden_layers=32, n_predict=1)
            ),
        ),
    )
    ext.model_config = object()
    ext.device = torch.device("cpu")
    ext.state_dict_info = {
        "draft.layer0.weight": object(),
        "draft.layer1.weight": object(),
    }
    fp8_module = ModuleType("nemo_rl.models.generation.vllm.quantization.fp8")
    setattr(fp8_module, "is_fp8_model", lambda _config: False)
    monkeypatch.setitem(
        sys.modules, "nemo_rl.models.generation.vllm.quantization.fp8", fp8_module
    )

    ext._begin_weight_update()
    ext._load_weights([("draft.layer0.weight", torch.ones(1))])

    with pytest.raises(RuntimeError, match="missing weights.*draft.layer1.weight"):
        ext._finish_weight_update()
    draft_model.load_weights.assert_not_called()


@pytest.mark.vllm
def test_mtp_refit_owner_rejects_update_without_draft_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_model = SimpleNamespace(load_weights=MagicMock(return_value=None))
    target_model = SimpleNamespace(load_weights=MagicMock())
    ext = _make_extension_for_draft_load(draft_model)
    ext.model_runner.model = target_model
    ext.model_runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[]),
        speculative_config=SimpleNamespace(method="mtp", draft_model_config=object()),
    )
    ext.model_config = object()
    ext.device = torch.device("cpu")
    ext.state_dict_info = {"model.layers.0.self_attn.q_proj.weight": object()}
    ext.require_mtp_draft_weights = True
    fp8_module = ModuleType("nemo_rl.models.generation.vllm.quantization.fp8")
    setattr(fp8_module, "is_fp8_model", lambda _config: False)
    monkeypatch.setitem(
        sys.modules, "nemo_rl.models.generation.vllm.quantization.fp8", fp8_module
    )
    _patch_pp_rank(monkeypatch, is_last_rank=True)

    ext._begin_weight_update()
    ext._load_weights([("model.layers.0.self_attn.q_proj.weight", torch.ones(1))])

    with pytest.raises(RuntimeError, match="MTP refit completed without draft weights"):
        ext._finish_weight_update()
    draft_model.load_weights.assert_not_called()


@pytest.mark.vllm
def test_eagle_weight_update_loads_each_chunk_without_full_draft_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_model = SimpleNamespace(load_weights=MagicMock(return_value=None))
    ext = _make_extension_for_draft_load(draft_model)
    ext.model_runner.model = SimpleNamespace(load_weights=MagicMock())
    ext.model_runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[]),
        speculative_config=SimpleNamespace(
            method="eagle3", draft_model_config=object()
        ),
    )
    ext.model_config = object()
    ext.device = torch.device("cpu")
    ext.state_dict_info = {
        "draft.layer0.weight": object(),
        "draft.layer1.weight": object(),
    }
    monkeypatch.setattr(ext, "_trim_vocab_padding", lambda _model, weights: weights)
    fp8_module = ModuleType("nemo_rl.models.generation.vllm.quantization.fp8")
    setattr(fp8_module, "is_fp8_model", lambda _config: False)
    monkeypatch.setitem(
        sys.modules, "nemo_rl.models.generation.vllm.quantization.fp8", fp8_module
    )
    _patch_vllm_postload(monkeypatch)

    ext._begin_weight_update()
    ext._load_weights([("draft.layer0.weight", torch.ones(1))])
    ext._load_weights([("draft.layer1.weight", torch.ones(1))])

    assert draft_model.load_weights.call_count == 2
    assert ext._pending_draft_weights == []
    ext._finish_weight_update()


def test_static_eagle_refit_diagnostics_report_zero_draft_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ext = _make_extension_for_draft_load(draft_model=SimpleNamespace())
    ext.model_runner.model = SimpleNamespace(load_weights=MagicMock())
    ext.model_runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[]),
        speculative_config=SimpleNamespace(
            method="eagle3", draft_model_config=object()
        ),
    )
    ext.model_config = object()
    ext.device = torch.device("cpu")
    ext.state_dict_info = {"model.layers.0.weight": object()}
    fp8_module = ModuleType("nemo_rl.models.generation.vllm.quantization.fp8")
    setattr(fp8_module, "is_fp8_model", lambda _config: False)
    monkeypatch.setitem(
        sys.modules, "nemo_rl.models.generation.vllm.quantization.fp8", fp8_module
    )
    _patch_vllm_postload(monkeypatch)
    monkeypatch.setenv("NRL_VLLM_REFIT_DIAGNOSTICS", "true")

    ext._begin_weight_update()
    ext._load_weights([("model.layers.0.weight", torch.ones(4))])
    ext._finish_weight_update()

    output = capsys.readouterr().out
    assert (
        "[refit] trainer_weight_tensors=1 draft_weight_tensors=0 "
        "draft_weight_bytes=0 draft_weights_updated=false"
    ) in output


@pytest.mark.vllm
def test_weight_update_skips_draft_load_on_non_owner_pipeline_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_model = SimpleNamespace(load_weights=MagicMock())
    ext = _make_extension_for_draft_load(draft_model=None)
    ext.model_runner.model = target_model
    ext.model_runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[]),
        speculative_config=SimpleNamespace(
            method="mtp",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_hidden_layers=32, n_predict=1)
            ),
        ),
    )
    ext.model_config = object()
    ext.device = torch.device("cpu")
    ext.state_dict_info = {
        "draft.layer0.weight": object(),
        "model.layers.32.self_attn.q_proj.weight": object(),
    }
    fp8_module = ModuleType("nemo_rl.models.generation.vllm.quantization.fp8")
    setattr(fp8_module, "is_fp8_model", lambda _config: False)
    monkeypatch.setitem(
        sys.modules, "nemo_rl.models.generation.vllm.quantization.fp8", fp8_module
    )
    _patch_pp_rank(monkeypatch, is_last_rank=False)
    process_weights = _patch_vllm_postload(monkeypatch)

    ext._begin_weight_update()
    ext._load_weights(
        [
            ("draft.layer0.weight", torch.ones(1)),
            ("model.layers.32.self_attn.q_proj.weight", torch.ones(1)),
        ]
    )
    ext._finish_weight_update()

    process_weights.assert_called_once_with(target_model, ext.model_config, ext.device)
    target_model.load_weights.assert_not_called()


@pytest.mark.vllm
def test_weight_update_rejects_missing_drafter_on_owner_pipeline_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext = _make_extension_for_draft_load(draft_model=None)
    ext.model_runner.model = SimpleNamespace(load_weights=MagicMock())
    ext.model_runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[]),
        speculative_config=SimpleNamespace(draft_model_config=object()),
    )
    ext.model_config = object()
    ext.device = torch.device("cpu")
    ext.state_dict_info = {"draft.layer0.weight": object()}
    fp8_module = ModuleType("nemo_rl.models.generation.vllm.quantization.fp8")
    setattr(fp8_module, "is_fp8_model", lambda _config: False)
    monkeypatch.setitem(
        sys.modules, "nemo_rl.models.generation.vllm.quantization.fp8", fp8_module
    )
    _patch_pp_rank(monkeypatch, is_last_rank=True)

    ext._begin_weight_update()
    with pytest.raises(RuntimeError, match="drafter is unavailable.*last pipeline"):
        ext._load_weights([("draft.layer0.weight", torch.ones(1))])


@pytest.mark.vllm
def test_update_weights_from_collective_processes_weights_after_loading(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    call_order = []
    process_calls = []

    def process_weights_after_loading(model, model_config, device):
        call_order.append("process")
        process_calls.append((model, model_config, device))

    process_weights = _patch_vllm_postload(monkeypatch)
    process_weights.side_effect = process_weights_after_loading
    ext, expected_state_info = _make_collective_update_extension(vllm_backend)

    def load_weights(weights):
        call_order.append("load")
        assert weights == [("model.weight", "weight-value")]
        ext._observed_update_weight_names.add("model.weight")

    def packed_broadcast_consumer(iterator, group, src, post_unpack_func):
        call_order.append("broadcast")
        assert list(iterator) == [("model.weight", expected_state_info)]
        assert group is ext.model_update_group
        assert src == 0
        post_unpack_func([("model.weight", "weight-value")])

    ext._load_weights = load_weights
    ext._maybe_process_fp8_kv_cache = lambda: call_order.append("kv")
    monkeypatch.setattr(
        vllm_backend, "packed_broadcast_consumer", packed_broadcast_consumer
    )
    monkeypatch.setattr(vllm_backend.gc, "collect", lambda: call_order.append("gc"))
    monkeypatch.setattr(
        vllm_backend.torch.cuda,
        "synchronize",
        lambda _device=None: call_order.append("sync"),
    )
    monkeypatch.setattr(
        vllm_backend.torch.cuda,
        "empty_cache",
        lambda: call_order.append("empty_cache"),
    )

    assert ext.update_weights_from_collective() is True

    assert process_calls == [(ext.model_runner.model, ext.model_config, ext.device)]
    assert call_order == [
        "broadcast",
        "load",
        "sync",
        "process",
        "gc",
        "empty_cache",
    ]


@pytest.mark.vllm
def test_ipc_weight_update_replies_with_error_when_complete_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.models.generation.vllm import vllm_backend

    class FakeSocket:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        def recv_pyobj(self):
            return vllm_backend.IPCProtocol.COMPLETE

        def send(self, payload: bytes) -> None:
            self.sent.append(payload)

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.state_dict_info = {"missing.weight": object()}
    ext.zmq_socket = FakeSocket()
    ext.maybe_init_zmq = lambda: None
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    assert ext.update_weights_via_ipc_zmq() is False
    assert len(ext.zmq_socket.sent) == 1
    assert ext.zmq_socket.sent[0].startswith(
        vllm_backend.IPCProtocol.ERROR.value.encode()
    )


@pytest.mark.vllm
def test_read_mtp_layer_weights_from_checkpoint_filters_and_reads(tmp_path):
    """Only the requested MTP layer tensors are read, across the shards holding them."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        _read_mtp_layer_weights_from_checkpoint,
    )

    model_dir = tmp_path / "ckpt"
    mtp_block = torch.randn(4, 4)
    mtp_head = torch.randn(2, 4)
    other_layer = torch.randn(4, 4)
    embed = torch.randn(8, 4)
    # MTP layer index is 2; layer 0 and the top-level embed must be ignored.
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00002.safetensors": {
                "model.layers.2.mlp.up_proj.weight": mtp_block,  # MTP, shard 1
                "model.layers.0.mlp.up_proj.weight": other_layer,  # non-MTP, same shard
            },
            "model-00002-of-00002.safetensors": {
                "model.layers.2.shared_head.head.weight": mtp_head,  # MTP, shard 2
                "model.embed_tokens.weight": embed,  # non-MTP, no layer index
            },
        },
    )

    weights = _read_mtp_layer_weights_from_checkpoint(str(model_dir), {2})

    by_name = dict(weights)
    assert set(by_name) == {
        "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.shared_head.head.weight",
    }
    assert torch.equal(by_name["model.layers.2.mlp.up_proj.weight"], mtp_block)
    assert torch.equal(by_name["model.layers.2.shared_head.head.weight"], mtp_head)


@pytest.mark.vllm
def test_read_mtp_layer_weights_from_checkpoint_includes_mtp_namespace(tmp_path):
    from nemo_rl.models.generation.vllm.vllm_backend import (
        _read_mtp_layer_weights_from_checkpoint,
    )

    model_dir = tmp_path / "ckpt"
    mtp_weight = torch.randn(4, 4)
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "mtp.layers.0.self_attn.q_proj.weight": mtp_weight,
                "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
            }
        },
    )

    weights = _read_mtp_layer_weights_from_checkpoint(str(model_dir), {32})

    assert [name for name, _ in weights] == ["mtp.layers.0.self_attn.q_proj.weight"]
    assert torch.equal(weights[0][1], mtp_weight)


@pytest.mark.vllm
def test_read_mtp_layer_weights_from_checkpoint_includes_mimo_namespace(tmp_path):
    from nemo_rl.models.generation.vllm.vllm_backend import (
        _read_mtp_layer_weights_from_checkpoint,
    )

    model_dir = tmp_path / "ckpt"
    mtp_weight = torch.randn(4, 4)
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "model.mtp_layers.0.self_attn.q_proj.weight": mtp_weight,
                "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
            }
        },
    )

    weights = _read_mtp_layer_weights_from_checkpoint(str(model_dir), {32})

    assert [name for name, _ in weights] == [
        "model.mtp_layers.0.self_attn.q_proj.weight"
    ]
    assert torch.equal(weights[0][1], mtp_weight)


@pytest.mark.vllm
def test_read_mtp_layer_weights_from_single_safetensors(tmp_path):
    from nemo_rl.models.generation.vllm.vllm_backend import (
        _read_mtp_layer_weights_from_checkpoint,
    )

    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    mtp_weight = torch.randn(4, 4)
    save_file(
        {
            "model.mtp_layers.0.self_attn.q_proj.weight": mtp_weight,
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
        },
        str(model_dir / "model.safetensors"),
    )

    weights = _read_mtp_layer_weights_from_checkpoint(str(model_dir), {32})

    assert [name for name, _ in weights] == [
        "model.mtp_layers.0.self_attn.q_proj.weight"
    ]
    assert torch.equal(weights[0][1], mtp_weight)


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_loads_only_mtp_layer(tmp_path, monkeypatch):
    """Success path: only MTP-layer weights are handed to the drafter, then post-loaded."""
    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "model.layers.2.mlp.up_proj.weight": torch.randn(4, 4),  # MTP
                "model.layers.2.embed_tokens.weight": torch.randn(8, 4),  # MTP
                "model.layers.0.mlp.up_proj.weight": torch.randn(4, 4),  # non-MTP
                "model.embed_tokens.weight": torch.randn(8, 4),  # non-MTP
            }
        },
    )
    ext = _make_extension_with_drafter(mtp_start_layer_idx=2, num_mtp_layers=1)
    process_weights = _patch_vllm_postload(monkeypatch)

    result = ext.load_mtp_weights_from_disk(str(model_dir))

    assert result is True
    ext._load_draft_weights.assert_called_once()
    loaded_names = {name for name, _ in ext._load_draft_weights.call_args[0][0]}
    assert loaded_names == {
        "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.embed_tokens.weight",
    }
    process_weights.assert_called_once()


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_raises_without_drafter_on_owner_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing vLLM drafter fails before reading MTP checkpoint weights."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    ext.model_runner = MagicMock()
    ext.model_runner.drafter = None
    ext.model_runner.speculator = None
    ext._load_draft_weights = MagicMock()
    _patch_pp_rank(monkeypatch, is_last_rank=True)

    with pytest.raises(RuntimeError, match="MTP weights.*last pipeline rank"):
        ext.load_mtp_weights_from_disk(str(tmp_path))
    ext._load_draft_weights.assert_not_called()


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_skips_non_owner_pipeline_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    ext.model_runner = MagicMock()
    ext.model_runner.drafter = None
    ext.model_runner.speculator = None
    ext._load_draft_weights = MagicMock()
    _patch_pp_rank(monkeypatch, is_last_rank=False)

    assert ext.load_mtp_weights_from_disk(str(tmp_path)) is None
    ext._load_draft_weights.assert_not_called()


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_raises_when_mtp_weights_missing(
    tmp_path, monkeypatch
):
    """A checkpoint without the MTP layer(s) fails loudly instead of silently."""
    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "model.layers.0.mlp.up_proj.weight": torch.randn(4, 4),
                "model.embed_tokens.weight": torch.randn(8, 4),
            }
        },
    )
    ext = _make_extension_with_drafter(mtp_start_layer_idx=2, num_mtp_layers=1)
    _patch_vllm_postload(monkeypatch)

    with pytest.raises(ValueError, match="No MTP draft weights"):
        ext.load_mtp_weights_from_disk(str(model_dir))
    ext._load_draft_weights.assert_not_called()
