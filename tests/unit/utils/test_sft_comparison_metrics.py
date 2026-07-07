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

import math

import pytest

from nemo_rl.utils.sft_comparison_metrics import (
    SFTComparisonObservation,
    build_sft_comparison_metrics,
)


def test_builds_validation_comparison_metrics() -> None:
    observation = SFTComparisonObservation(
        step=20,
        train_step_time_s=55.28,
        e2e_step_time_s=182.39,
        validation_time_s=126.99,
        main_lm_loss=2.5176,
        validation_loss=2.5803,
        grad_norm=42.0,
        learning_rate=4.2e-7,
    )

    assert build_sft_comparison_metrics(observation) == {
        "comparison/step": 20,
        "performance/train_step_time_s": 55.28,
        "performance/e2e_step_time_s": 182.39,
        "performance/validation_time_s": 126.99,
        "accuracy/main_lm_loss": 2.5176,
        "accuracy/validation_loss": 2.5803,
        "accuracy/grad_norm": 42.0,
        "accuracy/learning_rate": 4.2e-7,
        "context/is_validation_step": 1,
    }


def test_omits_unavailable_metrics_for_training_step() -> None:
    observation = SFTComparisonObservation(
        step=19,
        train_step_time_s=54.98,
        main_lm_loss=2.61,
    )

    assert build_sft_comparison_metrics(observation) == {
        "comparison/step": 19,
        "performance/train_step_time_s": 54.98,
        "accuracy/main_lm_loss": 2.61,
        "context/is_validation_step": 0,
    }


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_metric_values(value: float) -> None:
    observation = SFTComparisonObservation(step=1, main_lm_loss=value)

    with pytest.raises(ValueError, match="main_lm_loss"):
        build_sft_comparison_metrics(observation)
