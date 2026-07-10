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

"""Safe speculative-decoding toggle decisions."""

import math
from typing import TypedDict

from .config import SDToggleConfig
from .roofline import compute_r, compute_v, predict_speedup


class PredictionDecision(TypedDict):
    """Intermediate roofline values used to make a tail-gate decision."""

    sd_on: bool
    speedup: float
    r: float
    v: float
    L_accept: float
    gamma: int
    B: int
    S: int


def should_enable_sd(
    config: SDToggleConfig,
    B: int,
    S: int,
    gamma: int,
    L_accept: float | None = None,
    margin: float = 0.0,
) -> bool:
    """Returns whether the roofline predicts speedup above the safety margin.

    Invalid, non-finite, and non-positive denominator results fail closed by
    raising ``ValueError`` instead of being interpreted as profitable.
    """
    _validate_query(B=B, S=S, gamma=gamma, L_accept=L_accept, margin=margin)
    speedup = predict_speedup(B, S, gamma, L_accept, config)
    if not math.isfinite(speedup):
        raise ValueError("roofline prediction must be finite")
    return speedup >= 1.0 + margin


def predict_decision(
    config: SDToggleConfig,
    B: int,
    S: int,
    gamma: int,
    L_accept: float | None = None,
    margin: float = 0.0,
) -> PredictionDecision:
    """Returns a validated roofline decision and its intermediate ratios."""
    _validate_query(B=B, S=S, gamma=gamma, L_accept=L_accept, margin=margin)
    expected_acceptance = float(gamma) if L_accept is None else L_accept
    r = compute_r(B, S, gamma, config)
    v = compute_v(B, S, gamma, config)
    denominator = gamma * r + v
    speedup = predict_speedup(B, S, gamma, expected_acceptance, config)
    if not all(math.isfinite(value) for value in (r, v, denominator, speedup)):
        raise ValueError("roofline prediction must be finite")
    if denominator <= 0.0:
        raise ValueError("roofline prediction denominator must be positive")
    return {
        "sd_on": speedup >= 1.0 + margin,
        "speedup": speedup,
        "r": r,
        "v": v,
        "L_accept": expected_acceptance,
        "gamma": gamma,
        "B": B,
        "S": S,
    }


def _validate_query(
    *, B: int, S: int, gamma: int, L_accept: float | None, margin: float
) -> None:
    if isinstance(B, bool) or not isinstance(B, int) or B <= 0:
        raise ValueError("B must be a positive integer")
    if isinstance(S, bool) or not isinstance(S, int) or S < 0:
        raise ValueError("S must be a non-negative integer")
    if isinstance(gamma, bool) or not isinstance(gamma, int) or gamma <= 0:
        raise ValueError("gamma must be a positive integer")
    if L_accept is not None and (not math.isfinite(L_accept) or L_accept <= 0.0):
        raise ValueError("L_accept must be finite and positive")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and non-negative")
