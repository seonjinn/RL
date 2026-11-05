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

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.models.policy.draft_config import (
    DraftOptimizerConfig,
    Eagle3DraftConfig,
    MegatronDFlashDraftConfig,
    MegatronDSparkDraftConfig,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).resolve().parents[4]

register_omegaconf_resolvers()


def test_draft_optimizer_config_is_typed() -> None:
    draft = Eagle3DraftConfig(
        enabled=True,
        optimizer={
            "lr": 1.0e-5,
            "min_lr": 1.0e-6,
            "weight_decay": 0.02,
        },
    )

    assert draft.optimizer == DraftOptimizerConfig(
        lr=1.0e-5,
        min_lr=1.0e-6,
        weight_decay=0.02,
    )


def test_eagle3_draft_config_accepts_legacy_mapping_without_speculator_type() -> None:
    from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

    config = Eagle3DraftConfig.model_validate(
        {"enabled": True, "model_name": "draft", "loss_weight": 0.25}
    )

    assert config.speculator_type == "eagle3"


def test_eagle3_draft_config_preserves_extra_legacy_keys() -> None:
    from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

    config = Eagle3DraftConfig.model_validate({"legacy_extension": 7})

    assert config.model_dump()["legacy_extension"] == 7


def test_eagle3_draft_config_rejects_near_miss_typo_keys() -> None:
    from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

    with pytest.raises(ValidationError, match="did you mean 'enabled'"):
        Eagle3DraftConfig.model_validate({"enalbed": True})
    with pytest.raises(ValidationError, match="did you mean 'loss_weight'"):
        Eagle3DraftConfig.model_validate({"loss_wieght": 0.5})


def test_grpo_master_config_parses_nested_draft_model() -> None:
    path = REPO_ROOT / "examples/configs/grpo_math_1B.yaml"
    raw = OmegaConf.to_container(load_config(path), resolve=True)

    config = MasterConfig(**raw)

    assert isinstance(config.policy["draft"], Eagle3DraftConfig)
    assert config.policy["draft"].speculator_type == "eagle3"


def test_grpo_master_config_preserves_legacy_missing_method_discriminator() -> None:
    path = REPO_ROOT / "examples/configs/grpo_math_1B.yaml"
    raw = OmegaConf.to_container(load_config(path), resolve=True)
    del raw["policy"]["draft"]["speculator_type"]

    config = MasterConfig(**raw)

    assert isinstance(config.policy["draft"], Eagle3DraftConfig)
    assert config.policy["draft"].speculator_type == "eagle3"


def test_omitted_draft_config_does_not_request_refit() -> None:
    from nemo_rl.models.policy.draft_config import draft_refit_enabled

    assert draft_refit_enabled(None) is False


def test_dflash_config_window_must_hold_one_block() -> None:
    """Catches windows that cannot hold the anchor block of gamma + 1 slots."""
    values = {
        "enabled": True,
        "gamma": 5,
        "anchors_per_sample": 4,
        "mask_token_id": 151665,
        "target_hidden_state_layer_ids": [1, 17, 33],
    }

    with pytest.raises(ValidationError, match="sliding_window"):
        MegatronDFlashDraftConfig(**values, sliding_window=5)
    MegatronDFlashDraftConfig(**values, sliding_window=6)


def test_dspark_config_window_must_hold_one_block() -> None:
    """Catches windows that cannot hold one DSpark block of block_size slots."""
    values = {
        "enabled": True,
        "block_size": 8,
        "anchors_per_sample": 4,
        "mask_token_id": 151669,
        "target_hidden_state_layer_ids": [1, 12, 23],
    }

    with pytest.raises(ValidationError, match="sliding_window"):
        MegatronDSparkDraftConfig(**values, sliding_window=7)
    MegatronDSparkDraftConfig(**values, sliding_window=8)


def test_dspark_config_rejects_vocab_different_from_live_target() -> None:
    config = MegatronDSparkDraftConfig(
        enabled=True,
        block_size=7,
        anchors_per_sample=2,
        mask_token_id=151669,
        target_hidden_state_layer_ids=[1, 9, 17, 25, 33],
        draft_vocab_size=32_000,
    )

    with pytest.raises(ValueError, match="must match the live target vocabulary"):
        config.resolve_draft_vocab_size(target_vocab_size=151_936)


def test_dspark_config_resolves_target_owned_vocab() -> None:
    values = {
        "enabled": True,
        "block_size": 7,
        "anchors_per_sample": 2,
        "mask_token_id": 151669,
        "target_hidden_state_layer_ids": [1, 9, 17, 25, 33],
    }

    assert (
        MegatronDSparkDraftConfig(**values).resolve_draft_vocab_size(
            target_vocab_size=151_936
        )
        == 151_936
    )
    assert (
        MegatronDSparkDraftConfig(
            **values,
            draft_vocab_size=151_936,
        ).resolve_draft_vocab_size(target_vocab_size=151_936)
        == 151_936
    )


def test_block_draft_config_drops_only_default_inherited_eagle3_keys() -> None:
    values = {
        "enabled": True,
        "gamma": 5,
        "anchors_per_sample": 4,
        "mask_token_id": 151665,
        "target_hidden_state_layer_ids": [1, 17, 33],
        "ttt_steps": 3,
    }

    MegatronDFlashDraftConfig.model_validate(values)
    with pytest.raises(ValidationError, match="ttt_steps"):
        MegatronDFlashDraftConfig.model_validate({**values, "ttt_steps": 4})


def test_dflash_config_rejects_duplicate_target_taps() -> None:
    with pytest.raises(ValidationError, match="unique"):
        MegatronDFlashDraftConfig.model_validate(
            {
                "enabled": True,
                "gamma": 5,
                "anchors_per_sample": 4,
                "mask_token_id": 151665,
                "target_hidden_state_layer_ids": [1, 1],
            }
        )


def test_coerce_draft_config_accepts_a_raw_mapping() -> None:
    """PolicyConfig is a TypedDict, so hand-built configs arrive as dicts.

    Every downstream reader accesses the draft config by attribute, so the
    mapping input has to survive as a validated model rather than a dict.
    """
    from nemo_rl.models.policy.draft_config import coerce_draft_config

    coerced = coerce_draft_config({"enabled": True, "model_name": "draft"})

    assert isinstance(coerced, Eagle3DraftConfig)
    assert coerced.enabled is True
    assert coerced.model_name == "draft"


def test_coerce_draft_config_validates_a_nested_optimizer_mapping() -> None:
    from nemo_rl.models.policy.draft_config import coerce_draft_config

    coerced = coerce_draft_config({"enabled": True, "optimizer": {"lr": 1.0e-5}})

    assert coerced is not None
    assert coerced.optimizer == DraftOptimizerConfig(lr=1.0e-5)


def test_coerce_draft_config_passes_through_models_and_none() -> None:
    from nemo_rl.models.policy.draft_config import coerce_draft_config

    already = Eagle3DraftConfig(enabled=True)

    assert coerce_draft_config(already) is already
    assert coerce_draft_config(None) is None


def test_raw_mapping_draft_config_requests_refit_when_enabled() -> None:
    from nemo_rl.models.policy.draft_config import draft_refit_enabled

    assert draft_refit_enabled({"enabled": True}) is True
    assert draft_refit_enabled({"enabled": False}) is False


def _megatron_dflash_values() -> dict:
    return {
        "enabled": True,
        "gamma": 5,
        "anchors_per_sample": 4,
        "mask_token_id": 151665,
        "target_hidden_state_layer_ids": [1, 17, 33],
    }


@pytest.mark.parametrize(
    ("schedule", "enabled", "can_pause"),
    [
        (None, True, False),
        ({"type": "interval", "interval": 5}, True, True),
        ({"type": "acceptance"}, True, True),
        ({"type": "acceptance"}, False, False),
    ],
)
def test_draft_update_can_pause(
    schedule: dict | None, enabled: bool, can_pause: bool
) -> None:
    from nemo_rl.models.policy.draft_config import draft_update_can_pause

    values = {**_megatron_dflash_values(), "enabled": enabled}
    if schedule is not None:
        values.update(optimizer={"lr": 1.0e-4}, update_schedule=schedule)

    assert draft_update_can_pause(MegatronDFlashDraftConfig(**values)) is can_pause
    assert draft_update_can_pause(None) is False


@pytest.mark.parametrize(
    "update_schedule",
    [{"type": "interval", "interval": 5}, {"type": "acceptance"}],
)
def test_paused_draft_update_schedule_requires_draft_optimizer(
    update_schedule: dict,
) -> None:
    """Catches a schedule that would freeze a param group shared with the policy."""
    with pytest.raises(ValidationError, match="requires policy.draft.optimizer"):
        MegatronDFlashDraftConfig(
            **_megatron_dflash_values(), update_schedule=update_schedule
        )

    MegatronDFlashDraftConfig(
        **_megatron_dflash_values(),
        optimizer={"lr": 1.0e-4},
        update_schedule=update_schedule,
    )
