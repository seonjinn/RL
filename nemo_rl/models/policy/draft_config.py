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

from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, Field, model_validator


class DraftOptimizerConfig(BaseModel, extra="forbid"):
    """Optional optimizer schedule for draft-model parameters."""

    lr: Annotated[float, Field(gt=0)]
    min_lr: Annotated[float, Field(ge=0)] | None = None
    weight_decay: Annotated[float, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def validate_lr_range(self) -> Self:
        """Require the draft minimum learning rate to fit its schedule."""
        if self.min_lr is not None and self.min_lr > self.lr:
            raise ValueError("draft optimizer min_lr must not exceed lr")
        return self


class Eagle3DraftConfig(BaseModel, extra="allow"):
    """Configuration for EAGLE-3 draft-model co-training with the policy."""

    speculator_type: Literal["eagle3"] = "eagle3"
    enabled: bool = False
    model_name: str | None = None
    loss_weight: float = 0.1
    num_layers: int | None = None
    aux_layer_indices: list[int] | None = None
    optimizer: DraftOptimizerConfig | None = None


class DFlashDraftConfig(BaseModel, extra="forbid"):
    """Configuration for body-only DFlash co-training with a live target."""

    speculator_type: Literal["dflash"] = "dflash"
    enabled: bool = False
    model_name: str | None = None
    loss_weight: Annotated[float, Field(gt=0)] = 0.1
    gamma: Annotated[int, Field(gt=0)]
    anchors_per_sample: Annotated[int, Field(gt=0)]
    mask_token_id: Annotated[int, Field(ge=0)]
    target_hidden_state_layer_ids: Annotated[list[int], Field(min_length=1)]
    num_layers: Annotated[int, Field(gt=0)] = 5
    seed: int = 0
    vocab_tile_size: Annotated[int, Field(gt=0)] = 256
    position_decay: Annotated[float, Field(gt=0, le=1)] = 1.0
    optimizer: DraftOptimizerConfig | None = None

    @model_validator(mode="after")
    def validate_target_taps(self) -> Self:
        """Reject ambiguous and out-of-range layer taps before model creation."""
        if any(layer_id < 0 for layer_id in self.target_hidden_state_layer_ids):
            raise ValueError("target hidden-state layer IDs must be non-negative")
        if len(set(self.target_hidden_state_layer_ids)) != len(
            self.target_hidden_state_layer_ids
        ):
            raise ValueError("target hidden-state layer IDs must be unique")
        return self


DraftConfig: TypeAlias = Eagle3DraftConfig | DFlashDraftConfig


def draft_refit_enabled(config: DraftConfig | None) -> bool:
    """Return whether generation must accept refitted draft weights."""
    return config is not None and config.enabled
