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

"""Common comparison metrics emitted by the SFT training loop."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SFTComparisonObservation:
    """Scalars available after one SFT training step and optional validation."""

    step: int
    train_step_time_s: float | None = None
    e2e_step_time_s: float | None = None
    validation_time_s: float | None = None
    main_lm_loss: float | None = None
    validation_loss: float | None = None
    grad_norm: float | None = None
    learning_rate: float | None = None


def build_sft_comparison_metrics(
    observation: SFTComparisonObservation,
) -> dict[str, float | int]:
    """Build a normalized W&B payload from an SFT step observation.

    Args:
        observation: Training and optional validation scalars for one SFT step.

    Returns:
        The normalized comparison payload. Unavailable optional values are omitted.

    Raises:
        TypeError: If a value is not a Python int or float.
        ValueError: If an emitted scalar is NaN or infinite.
    """
    if not isinstance(observation.step, int) or isinstance(observation.step, bool):
        raise TypeError(
            f"step must be a Python int, got {type(observation.step).__name__}"
        )

    metrics: dict[str, float | int] = {"comparison/step": observation.step}
    field_names = {
        "performance/train_step_time_s": "train_step_time_s",
        "performance/e2e_step_time_s": "e2e_step_time_s",
        "performance/validation_time_s": "validation_time_s",
        "accuracy/main_lm_loss": "main_lm_loss",
        "accuracy/validation_loss": "validation_loss",
        "accuracy/grad_norm": "grad_norm",
        "accuracy/learning_rate": "learning_rate",
    }

    for metric_name, field_name in field_names.items():
        value = getattr(observation, field_name)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"{field_name} must be a Python int or float, "
                f"got {type(value).__name__}"
            )
        float_value = float(value)
        if not math.isfinite(float_value):
            raise ValueError(f"{field_name} must be finite, got {float_value!r}")
        metrics[metric_name] = float_value

    metrics["context/is_validation_step"] = int(
        observation.validation_time_s is not None
        or observation.validation_loss is not None
    )
    return metrics
