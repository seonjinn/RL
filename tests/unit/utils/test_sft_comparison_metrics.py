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
from decimal import Decimal
from typing import Any, cast

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
        processed_tokens=16_631_382,
        num_gpus=512,
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
        "throughput/processed_tokens_per_second": pytest.approx(16_631_382 / 55.28),
        "throughput/processed_tokens_per_second_per_gpu": pytest.approx(
            16_631_382 / 55.28 / 512
        ),
        "context/processed_tokens": 16_631_382,
        "context/num_gpus": 512,
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


def test_rejects_non_python_numeric_values() -> None:
    observation = SFTComparisonObservation(
        step=1,
        main_lm_loss=cast(Any, Decimal("2.5")),
    )

    with pytest.raises(TypeError, match="main_lm_loss"):
        build_sft_comparison_metrics(observation)


@pytest.mark.parametrize(
    ("train_step_time_s", "processed_tokens", "num_gpus", "field_name"),
    [
        pytest.param(0.0, 16_631_382, 512, "train_step_time_s", id="zero-duration"),
        pytest.param(55.28, -1, 512, "processed_tokens", id="negative-tokens"),
        pytest.param(55.28, 16_631_382, 0, "num_gpus", id="zero-gpus"),
    ],
)
def test_rejects_invalid_throughput_inputs(
    train_step_time_s: float,
    processed_tokens: int,
    num_gpus: int,
    field_name: str,
) -> None:
    observation = SFTComparisonObservation(
        step=1,
        train_step_time_s=train_step_time_s,
        processed_tokens=processed_tokens,
        num_gpus=num_gpus,
    )

    with pytest.raises(ValueError, match=field_name):
        build_sft_comparison_metrics(observation)


@pytest.mark.parametrize(
    "throughput_inputs",
    [
        pytest.param({"processed_tokens": 16_631_382}, id="missing-num-gpus"),
        pytest.param({"num_gpus": 512}, id="missing-processed-tokens"),
    ],
)
def test_requires_both_throughput_inputs(throughput_inputs: dict[str, int]) -> None:
    observation = SFTComparisonObservation(
        step=1,
        train_step_time_s=55.28,
        **throughput_inputs,
    )

    with pytest.raises(ValueError, match="processed_tokens and num_gpus"):
        build_sft_comparison_metrics(observation)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        pytest.param("processed_tokens", 1.0, id="float-processed-tokens"),
        pytest.param("num_gpus", 1.0, id="float-num-gpus"),
        pytest.param("processed_tokens", True, id="bool-processed-tokens"),
        pytest.param("num_gpus", True, id="bool-num-gpus"),
    ],
)
def test_requires_exact_python_integers_for_throughput_inputs(
    field_name: str, value: int | float
) -> None:
    throughput_inputs: dict[str, int | float] = {
        "processed_tokens": 16_631_382,
        "num_gpus": 512,
    }
    throughput_inputs[field_name] = value
    observation = SFTComparisonObservation(
        step=1,
        train_step_time_s=55.28,
        **throughput_inputs,
    )

    with pytest.raises(TypeError, match=field_name):
        build_sft_comparison_metrics(observation)
