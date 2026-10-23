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

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_rl.models.megatron.draft import utils as draft_utils
from nemo_rl.models.megatron.draft.dflash import DFlashBodyConfig
from nemo_rl.models.megatron.draft.training import resolve_draft_speculator
from nemo_rl.models.megatron.draft.utils import (
    resolve_dflash_sliding_window,
    validate_dflash_checkpoint_sliding_window,
)
from nemo_rl.models.policy.draft_config import MegatronDFlashDraftConfig

pytestmark = pytest.mark.mcore


def _hf_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "num_hidden_layers": 5,
        "layer_types": ["full_attention"] * 5,
        "sliding_window": 8192,
        "dflash_config": {
            "use_swa": True,
            "swa_window_size": 8192,
            "causal": False,
        },
    }
    config.update(overrides)
    return config


def _write_config(directory: Path, config: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def test_use_swa_turns_full_attention_layers_into_one_window() -> None:
    assert resolve_dflash_sliding_window(_hf_config()) == 8192


def test_swa_window_size_takes_priority_over_sliding_window() -> None:
    config = _hf_config(sliding_window=4096)

    assert resolve_dflash_sliding_window(config) == 8192


def test_sliding_window_is_the_fallback_window() -> None:
    config = _hf_config(dflash_config={"use_swa": True, "causal": False})

    assert resolve_dflash_sliding_window(config) == 8192


def test_a_null_swa_window_size_does_not_fall_back_to_sliding_window() -> None:
    config = _hf_config(
        sliding_window=4096,
        dflash_config={"use_swa": True, "swa_window_size": None, "causal": False},
    )

    with pytest.raises(ValueError, match="no positive window size"):
        resolve_dflash_sliding_window(config)


def test_use_swa_without_layer_types_or_layer_count_uses_the_window() -> None:
    config = {"dflash_config": {"use_swa": True, "swa_window_size": 8192}}

    assert resolve_dflash_sliding_window(config) == 8192


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"num_hidden_layers": 5},
        _hf_config(dflash_config={"causal": False}),
        _hf_config(dflash_config={"use_swa": False, "swa_window_size": 8192}),
    ],
    ids=["empty", "no-layer-types", "no-use-swa", "use-swa-false"],
)
def test_full_attention_configs_have_no_window(config: dict[str, Any]) -> None:
    assert resolve_dflash_sliding_window(config) is None


def test_bidirectional_sliding_layer_types_use_the_window() -> None:
    config = _hf_config(
        layer_types=["sliding_attention"] * 5,
        dflash_config={},
        is_causal=False,
    )

    assert resolve_dflash_sliding_window(config) == 8192


@pytest.mark.parametrize(
    "config",
    [
        _hf_config(dflash_config={"use_swa": True, "causal": True}),
        _hf_config(is_causal=True),
        # vLLM makes sliding layers causal when no causal flag is set.
        _hf_config(layer_types=["sliding_attention"] * 5, dflash_config={}),
    ],
    ids=["dflash-causal", "is-causal", "implicit-causal-sliding"],
)
def test_causal_layers_raise(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="only bidirectional"):
        resolve_dflash_sliding_window(config)


def test_mixed_sliding_and_full_layers_raise() -> None:
    config = _hf_config(
        layer_types=["sliding_attention", "full_attention"] * 2 + ["sliding_attention"],
        is_causal=False,
    )

    with pytest.raises(ValueError, match="one attention window"):
        resolve_dflash_sliding_window(config)


@pytest.mark.parametrize("window", [None, 0, -1])
def test_sliding_layers_without_a_positive_window_raise(window: int | None) -> None:
    config = _hf_config(
        sliding_window=window,
        dflash_config={"use_swa": True, "causal": False},
    )

    with pytest.raises(ValueError, match="no positive window size"):
        resolve_dflash_sliding_window(config)


def test_a_matching_local_checkpoint_window_passes(tmp_path: Path) -> None:
    checkpoint = _write_config(tmp_path / "drafter", _hf_config())

    validate_dflash_checkpoint_sliding_window(str(checkpoint), 8192)


def test_a_matching_hub_checkpoint_window_shares_the_weight_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _write_config(tmp_path / "snapshot", _hf_config())
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    draft_utils._download_hub_checkpoint.cache_clear()
    try:
        validate_dflash_checkpoint_sliding_window("org/swa-drafter", 8192)
        draft_utils._download_hub_checkpoint("org/swa-drafter", None)
    finally:
        draft_utils._download_hub_checkpoint.cache_clear()

    assert len(calls) == 1
    assert "config.json" in calls[0]["allow_patterns"]


def test_an_unresolvable_hub_checkpoint_raises_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_snapshot_download(**kwargs: Any) -> str:
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.snapshot_download", failing_snapshot_download)
    draft_utils._download_hub_checkpoint.cache_clear()

    with pytest.raises(FileNotFoundError, match="Could not resolve"):
        validate_dflash_checkpoint_sliding_window("org/missing-drafter", 8192)


@pytest.mark.parametrize("weights_only", ["file", "directory"])
def test_a_checkpoint_without_config_json_warns_and_skips_the_check(
    tmp_path: Path,
    weights_only: str,
) -> None:
    checkpoint = _write_config(tmp_path / "drafter", _hf_config())
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"")
    if weights_only == "directory":
        (checkpoint / "config.json").unlink()
        source = checkpoint
    else:
        # A config.json next to a weight file can belong to another model.
        source = weights

    with pytest.warns(UserWarning, match="has no config.json"):
        validate_dflash_checkpoint_sliding_window(str(source), None)


@pytest.mark.parametrize("model_name", ["", "  "])
def test_an_empty_checkpoint_name_raises(model_name: str) -> None:
    with pytest.raises(ValueError, match="non-empty model name"):
        validate_dflash_checkpoint_sliding_window(model_name, None)


@pytest.mark.parametrize("window", [0, -8])
def test_body_config_rejects_a_non_positive_window(window: int) -> None:
    with pytest.raises(ValueError, match="sliding_window must be positive"):
        DFlashBodyConfig(sliding_window=window)


def _speculator(*, model_name: str, sliding_window: int | None):
    speculator = resolve_draft_speculator(
        MegatronDFlashDraftConfig(
            enabled=True,
            gamma=2,
            anchors_per_sample=2,
            mask_token_id=7,
            target_hidden_state_layer_ids=[0, 2],
            num_layers=1,
            model_name=model_name,
            sliding_window=sliding_window,
        )
    )
    assert speculator is not None
    return speculator


@pytest.mark.parametrize(
    ("checkpoint_config", "config_window", "message"),
    [
        (_hf_config(), None, "uses sliding_window=8192"),
        (_hf_config(), 4096, "uses sliding_window=8192"),
        ({"num_hidden_layers": 5}, 8192, "uses sliding_window=None"),
    ],
    ids=["swa-checkpoint-full-config", "window-mismatch", "full-checkpoint-swa-config"],
)
def test_build_model_rejects_a_window_that_differs_from_the_checkpoint(
    tmp_path: Path,
    checkpoint_config: dict[str, Any],
    config_window: int | None,
    message: str,
) -> None:
    checkpoint = _write_config(tmp_path / "drafter", checkpoint_config)
    speculator = _speculator(model_name=str(checkpoint), sliding_window=config_window)

    with pytest.raises(ValueError, match=message):
        speculator.build_model(
            model_provider=SimpleNamespace(num_layers=4),
            pg_collection=None,
            policy_model_chunk=None,
        )
