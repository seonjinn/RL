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

"""Dependency-light scalar roofline model ported from EfficientRollout."""

import math

from .config import SDToggleConfig


def predict_T_T(
    B: int,
    S: int,
    W_t: float,
    kappa_eff: float,
    BW_eff: float,
    C_dense: float,
    C_attn: float,
    F_eff: float,
    c_T: float = 0.0,
    c_comm: float = 0.0,
) -> float:
    """Predicts target decode latency in seconds."""
    memory_time = _divide(W_t + kappa_eff * B * S, BW_eff)
    compute_time = _divide(B * C_dense + B * S * C_attn, F_eff)
    return _combine_times(
        memory_time=memory_time,
        compute_time=compute_time,
        overhead_time=c_T * B * 1e-6 + c_comm,
    )


def predict_T_D(
    B: int,
    S: int,
    W_d: float,
    eta_d: float,
    kappa_eff: float,
    BW_eff: float,
    C_dense: float,
    C_attn: float,
    F_eff: float,
    c_D: float = 0.0,
    c_comm: float = 0.0,
) -> float:
    """Predicts one drafter decode latency in seconds."""
    memory_time = _divide(eta_d * W_d + kappa_eff * B * S, BW_eff)
    compute_time = _divide(B * C_dense + B * S * C_attn, F_eff)
    return _combine_times(
        memory_time=memory_time,
        compute_time=compute_time,
        overhead_time=c_D * B * 1e-6 + c_comm,
    )


def predict_T_V(
    B: int,
    S: int,
    gamma: int,
    W_t: float,
    kappa_eff: float,
    BW_eff: float,
    C_dense: float,
    C_attn: float,
    F_eff: float,
    c_V: float = 0.0,
    c_comm: float = 0.0,
    beta: float = 0.0,
) -> float:
    """Predicts target verification latency in seconds."""
    memory_time = _divide(W_t + kappa_eff * B * S * (1.0 + beta * gamma), BW_eff)
    compute_time = _divide(
        B * (gamma + 1) * C_dense + B * S * (gamma + 1) * C_attn,
        F_eff,
    )
    return _combine_times(
        memory_time=memory_time,
        compute_time=compute_time,
        overhead_time=c_V * B * 1e-6 + c_comm,
    )


def compute_r(B: int, S: int, gamma: int, config: SDToggleConfig) -> float:
    """Computes the drafter-to-target latency ratio."""
    target_time = _target_time(B, S, gamma, config)
    drafter_time = _drafter_time(B, S, gamma, config)
    return _divide(drafter_time, target_time)


def compute_v(B: int, S: int, gamma: int, config: SDToggleConfig) -> float:
    """Computes the verification-to-target latency ratio."""
    target_time = _target_time(B, S, gamma, config)
    verification_time = _verification_time(B, S, gamma, config)
    return _divide(verification_time, target_time)


def predict_speedup(
    B: int,
    S: int,
    gamma: int,
    L_accept: float | None,
    config: SDToggleConfig,
) -> float:
    """Predicts speculative-decoding speedup over autoregressive decoding."""
    expected_acceptance = float(gamma) if L_accept is None else L_accept
    denominator = gamma * compute_r(B, S, gamma, config) + compute_v(
        B, S, gamma, config
    )
    return _divide(expected_acceptance, denominator)


def _target_time(B: int, S: int, gamma: int, config: SDToggleConfig) -> float:
    calibration = config.calibration
    hardware = config.hardware
    model = config.model
    overhead = config.get_gamma_params(gamma)
    return predict_T_T(
        B,
        S,
        model.W_t,
        calibration.kappa_eff,
        hardware.BW_eff,
        model.C_dense,
        model.C_attn,
        calibration.F_eff,
        overhead.c_T,
        hardware.c_comm,
    )


def _drafter_time(B: int, S: int, gamma: int, config: SDToggleConfig) -> float:
    calibration = config.calibration
    hardware = config.hardware
    model = config.model
    overhead = config.get_gamma_params(gamma)
    return predict_T_D(
        B,
        S,
        model.W_d,
        calibration.eta_d,
        calibration.kappa_eff,
        hardware.BW_eff,
        model.C_dense,
        model.C_attn,
        calibration.F_eff,
        overhead.c_D,
        hardware.c_comm,
    )


def _verification_time(B: int, S: int, gamma: int, config: SDToggleConfig) -> float:
    calibration = config.calibration
    hardware = config.hardware
    model = config.model
    overhead = config.get_gamma_params(gamma)
    return predict_T_V(
        B,
        S,
        gamma,
        model.W_t,
        calibration.kappa_eff,
        hardware.BW_eff,
        model.C_dense,
        model.C_attn,
        calibration.F_eff,
        overhead.c_V,
        hardware.c_comm,
        calibration.beta,
    )


def _divide(numerator: float, denominator: float) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        return math.nan
    return numerator / denominator


def _combine_times(
    *, memory_time: float, compute_time: float, overhead_time: float
) -> float:
    if not all(math.isfinite(value) for value in (memory_time, compute_time)):
        raise ValueError("roofline component times must be finite")
    combined_time = max(memory_time, compute_time) + overhead_time
    if not math.isfinite(combined_time):
        raise ValueError("roofline combined time must be finite")
    return combined_time
