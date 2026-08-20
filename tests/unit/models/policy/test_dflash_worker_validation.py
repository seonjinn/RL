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

from types import SimpleNamespace

import pytest

from nemo_rl.models.megatron.draft.training import resolve_draft_speculator
from nemo_rl.models.policy.draft_config import DFlashDraftConfig
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    _validate_dflash_training_setup,
)

pytestmark = pytest.mark.mcore


def _provider():
    provider = resolve_draft_speculator(
        DFlashDraftConfig(
            enabled=True,
            gamma=3,
            anchors_per_sample=1,
            mask_token_id=7,
            target_hidden_state_layer_ids=[1, 3],
        )
    )
    assert provider is not None
    return provider


def test_dflash_setup_rejects_layout_and_target_mismatches_together() -> None:
    config = {
        "sequence_packing": {"enabled": True},
        "generation": {"vllm_kwargs": {"speculative_config": {"method": "eagle3"}}},
    }
    model_cfg = SimpleNamespace(
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        sequence_parallel=True,
        num_layers=3,
    )

    with pytest.raises(ValueError) as error:
        _validate_dflash_training_setup(
            draft_provider=_provider(),
            config=config,
            model_cfg=model_cfg,
        )

    message = str(error.value)
    assert "pipeline_model_parallel_size must be 1" in message
    assert "context_parallel_size must be 1" in message
    assert "sequence_parallel must be disabled" in message
    assert "sequence_packing must be disabled" in message
    assert "target_hidden_state_layer_ids exceed the target model: 3" in message
    assert "generation speculative method must be dflash" in message


def test_dflash_setup_allows_training_without_generation() -> None:
    _validate_dflash_training_setup(
        draft_provider=_provider(),
        config={"sequence_packing": {"enabled": False}, "generation": None},
        model_cfg=SimpleNamespace(
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=False,
            num_layers=4,
        ),
    )
