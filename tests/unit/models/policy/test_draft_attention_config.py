# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint attention metadata must not silently change during co-training."""

import json
from pathlib import Path

import pytest

from nemo_rl.models.policy.draft_config import DFlashDraftConfig, DSparkDraftConfig


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_new_drafter_sliding_window_is_a_supported_config(method: str) -> None:
    config_type = DFlashDraftConfig if method == "dflash" else DSparkDraftConfig
    shape = {"gamma": 7} if method == "dflash" else {"block_size": 8}
    config = config_type.model_validate(
        dict(
            shape,
            anchors_per_sample=1,
            mask_token_id=3,
            target_hidden_state_layer_ids=[1],
            sliding_window=2048,
        )
    )
    assert config.sliding_window == 2048


@pytest.mark.parametrize("method", ["dflash", "dspark"])
@pytest.mark.parametrize("window", [0, -1, 7, True, 8.5, "2048"])
def test_window_must_cover_training_block(method: str, window: object) -> None:
    config_type = DFlashDraftConfig if method == "dflash" else DSparkDraftConfig
    shape = {"gamma": 7} if method == "dflash" else {"block_size": 8}
    with pytest.raises(ValueError):
        config_type.model_validate(
            dict(
                shape,
                anchors_per_sample=1,
                mask_token_id=3,
                target_hidden_state_layer_ids=[1],
                sliding_window=window,
            )
        )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, None),
        ({"sliding_window": None}, None),
        ({"sliding_window": 2048}, 2048),
        ({"dflash_config": {"use_swa": True, "swa_window_size": 2048}}, 2048),
        (
            {
                "sliding_window": 2048,
                "dflash_config": {
                    "use_swa": True,
                    "swa_window_size": 2048,
                    "causal": False,
                },
            },
            2048,
        ),
        ({"sliding_window": 4096, "use_sliding_window": False}, None),
    ],
)
def test_checkpoint_window_is_inherited(
    tmp_path: Path, metadata: dict, expected: int | None
) -> None:
    from nemo_rl.models.policy.draft_attention_config import (
        resolve_draft_sliding_window,
    )

    (tmp_path / "config.json").write_text(json.dumps(metadata))
    assert (
        resolve_draft_sliding_window(
            model_name=str(tmp_path),
            model_revision=None,
            sliding_window=None,
            block_size=8,
        )
        == expected
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"sliding_window": 7},
        {"sliding_window": True},
        {"sliding_window": 2048, "dflash_config": {"causal": True}},
        {"dflash_config": {"use_swa": False, "causal": True}},
        {"dflash_config": {"use_swa": True}},
        {"dflash_config": {"use_swa": True, "swa_window_size": 2048, "causal": True}},
        {
            "sliding_window": 1024,
            "dflash_config": {"use_swa": True, "swa_window_size": 2048},
        },
    ],
)
def test_invalid_checkpoint_attention_fails_before_weights(
    tmp_path: Path, metadata: dict
) -> None:
    from nemo_rl.models.policy.draft_attention_config import (
        resolve_draft_sliding_window,
    )

    (tmp_path / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        resolve_draft_sliding_window(
            model_name=str(tmp_path),
            model_revision=None,
            sliding_window=None,
            block_size=8,
        )


@pytest.mark.parametrize("metadata", [{}, {"sliding_window": 1024}])
def test_explicit_window_cannot_override_checkpoint_attention(
    tmp_path: Path, metadata: dict
) -> None:
    from nemo_rl.models.policy.draft_attention_config import (
        resolve_draft_sliding_window,
    )

    (tmp_path / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_draft_sliding_window(
            model_name=str(tmp_path),
            model_revision=None,
            sliding_window=2048,
            block_size=8,
        )


def test_random_initialization_can_set_window() -> None:
    from nemo_rl.models.policy.draft_attention_config import (
        resolve_draft_sliding_window,
    )

    assert (
        resolve_draft_sliding_window(
            model_name=None, model_revision=None, sliding_window=2048, block_size=8
        )
        == 2048
    )


def test_legacy_weight_only_checkpoint_preserves_full_context(tmp_path: Path) -> None:
    from nemo_rl.models.policy.draft_attention_config import (
        resolve_draft_sliding_window,
    )

    assert (
        resolve_draft_sliding_window(
            model_name=str(tmp_path),
            model_revision=None,
            sliding_window=None,
            block_size=8,
        )
        is None
    )
