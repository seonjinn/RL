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
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file


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
def test_split_policy_and_draft_weights_routes_mtp_layers_and_buffers() -> None:
    predictor = SimpleNamespace(mtp_start_layer_idx=2, num_mtp_layers=1)
    draft_model = SimpleNamespace(model=predictor)
    ext = _make_extension_for_draft_load(draft_model)
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
    draft_model = SimpleNamespace(load_weights=MagicMock(return_value=None))
    ext = _make_extension_for_draft_load(draft_model)
    trimmed_weights = [("model.layers.0.weight", torch.randn(1))]
    trim_vocab_padding = MagicMock(return_value=trimmed_weights)
    monkeypatch.setattr(ext, "_trim_vocab_padding", trim_vocab_padding)
    weights = [("model.layers.0.weight", torch.randn(2))]

    ext._load_draft_weights(weights)

    trim_vocab_padding.assert_called_once_with(draft_model, weights)
    draft_model.load_weights.assert_called_once_with(weights=trimmed_weights)


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

    with pytest.raises(ValueError, match="No MTP layer weights"):
        ext.load_mtp_weights_from_disk(str(model_dir))
    ext._load_draft_weights.assert_not_called()
