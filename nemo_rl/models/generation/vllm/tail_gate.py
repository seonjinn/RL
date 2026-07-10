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

"""Pure controller for monotone tail-gated speculative decoding."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from .sd_toggle import SDToggleConfig, load_config, predict_decision

TailGateMode = Literal["off", "threshold", "roofline"]
TailGateState = Literal["RAMPING_OFF", "ARMED_OFF", "ON_LATCHED"]


@dataclass(frozen=True)
class TailGateConfig:
    """Validated, rollout-independent tail-gate policy configuration."""

    mode: TailGateMode
    threshold: int = 32
    consecutive_checks: int = 10
    gamma: int = 5
    margin: float = 0.05
    expected_accept_length: float = 3.0
    roofline_config: SDToggleConfig | None = None
    ramp_threshold: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("off", "threshold", "roofline"):
            raise ValueError(f"unsupported tail-gate mode: {self.mode}")
        _require_positive_int(self.threshold, "threshold")
        _require_positive_int(self.consecutive_checks, "consecutive_checks")
        _require_positive_int(self.gamma, "gamma")
        _require_finite_nonnegative(self.margin, "margin")
        _require_finite_positive(self.expected_accept_length, "expected_accept_length")
        if self.ramp_threshold == 0:
            object.__setattr__(self, "ramp_threshold", self.threshold)
        _require_positive_int(self.ramp_threshold, "ramp_threshold")
        if self.mode == "roofline" and self.roofline_config is None:
            raise ValueError("roofline mode requires roofline_config")
        if self.mode == "roofline":
            assert self.roofline_config is not None
            _validate_roofline_config(
                self.roofline_config,
                gamma=self.gamma,
                expected_accept_length=self.expected_accept_length,
                margin=self.margin,
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TailGateConfig":
        """Builds a controller config from JSON-compatible gate settings.

        A roofline gate must provide ``roofline_config_path`` pointing to an
        EfficientRollout-compatible calibration JSON file.
        """
        mode = _get_mode(data, "mode")
        roofline_config_path = _get_optional_string(data, "roofline_config_path")
        roofline_config = (
            load_config(Path(roofline_config_path))
            if roofline_config_path is not None
            else None
        )
        return cls(
            mode=mode,
            threshold=_get_int(data, "threshold", 32),
            consecutive_checks=_get_int(data, "consecutive_checks", 10),
            gamma=_get_int(data, "gamma", 5),
            margin=_get_float(data, "margin", 0.05),
            expected_accept_length=_get_float(data, "expected_accept_length", 3.0),
            roofline_config=roofline_config,
            ramp_threshold=_get_int(data, "ramp_threshold", 0),
        )


@dataclass(frozen=True)
class TailGateObservation:
    """One scheduler observation supplied to the rollout-local controller."""

    active_requests: int
    mean_sequence_length: int
    is_decode: bool

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.active_requests, "active_requests")
        _require_nonnegative_int(self.mean_sequence_length, "mean_sequence_length")
        if not isinstance(self.is_decode, bool):
            raise ValueError("is_decode must be a boolean")


@dataclass(frozen=True)
class TailGateTelemetry:
    """Snapshot of controller state and the most recent roofline prediction."""

    state: TailGateState
    tick: int
    expected_accept_length: float
    qualifying_checks: int
    activation_tick: int | None
    activation_active_requests: int | None
    activation_sequence_length: int | None
    predicted_speedup: float | None
    drafter_target_ratio: float | None
    verify_target_ratio: float | None


@dataclass(frozen=True)
class TailGateDecision:
    """Per-observation enablement decision with diagnostic telemetry."""

    enabled: bool
    just_activated: bool
    reason: str
    telemetry: TailGateTelemetry


class TailGateController:
    """Controls one monotone speculation transition during a rollout."""

    def __init__(self, config: TailGateConfig) -> None:
        self.config: TailGateConfig = config
        self._expected_accept_length: float = config.expected_accept_length
        self._enabled: bool = False
        self._seen_ramp: bool = False
        self._qualifying_checks: int = 0
        self._tick: int = 0
        self._activation_tick: int | None = None
        self._activation_active_requests: int | None = None
        self._activation_sequence_length: int | None = None
        self._predicted_speedup: float | None = None
        self._drafter_target_ratio: float | None = None
        self._verify_target_ratio: float | None = None
        self._reset_rollout_state()

    @property
    def enabled(self) -> bool:
        """Whether speculation is currently enabled for this rollout."""
        return self._enabled

    @property
    def expected_accept_length(self) -> float:
        """Expected accepted length carried from the previous training rollout."""
        return self._expected_accept_length

    @property
    def telemetry(self) -> TailGateTelemetry:
        """Returns the current controller telemetry snapshot."""
        return self._telemetry()

    def observe(self, observation: TailGateObservation) -> TailGateDecision:
        """Evaluates one scheduler observation without mutating external runtime state."""
        self._tick += 1
        if self.config.mode == "off":
            return self._decision(enabled=True, reason="controller_off")
        if self._enabled:
            return self._decision(enabled=True, reason="latched")
        if not observation.is_decode or observation.active_requests == 0:
            return self._decision(enabled=False, reason="not_decode")
        if not self._seen_ramp:
            if observation.active_requests > self.config.ramp_threshold:
                self._seen_ramp = True
            return self._decision(enabled=False, reason="ramp_guard")
        if self.config.mode == "threshold":
            predicate = observation.active_requests <= self.config.threshold
        else:
            predicate = self._roofline_predicate(observation)
        if predicate:
            self._qualifying_checks += 1
        else:
            self._qualifying_checks = 0
        if self._qualifying_checks >= self.config.consecutive_checks:
            self._enabled = True
            self._activation_tick = self._tick
            self._activation_active_requests = observation.active_requests
            self._activation_sequence_length = observation.mean_sequence_length
            return self._decision(enabled=True, just_activated=True, reason="activated")
        return self._decision(enabled=False, reason="waiting")

    def finish_rollout(
        self, *, accepted_tokens: int, num_drafts: int, validation: bool
    ) -> TailGateTelemetry:
        """Records training acceptance feedback and resets rollout-local state."""
        _require_nonnegative_int(accepted_tokens, "accepted_tokens")
        _require_nonnegative_int(num_drafts, "num_drafts")
        if not isinstance(validation, bool):
            raise ValueError("validation must be a boolean")
        if not validation and num_drafts > 0:
            self._expected_accept_length = 1.0 + accepted_tokens / num_drafts
        previous_telemetry = self._telemetry()
        self._reset_rollout_state()
        return previous_telemetry

    def _roofline_predicate(self, observation: TailGateObservation) -> bool:
        roofline_config = self.config.roofline_config
        if self.config.mode != "roofline" or roofline_config is None:
            raise RuntimeError("roofline predicate requires roofline mode")
        prediction = predict_decision(
            roofline_config,
            observation.active_requests,
            observation.mean_sequence_length,
            self.config.gamma,
            self._expected_accept_length,
            self.config.margin,
        )
        self._predicted_speedup = prediction["speedup"]
        self._drafter_target_ratio = prediction["r"]
        self._verify_target_ratio = prediction["v"]
        return prediction["sd_on"]

    def _decision(
        self, *, enabled: bool, reason: str, just_activated: bool = False
    ) -> TailGateDecision:
        return TailGateDecision(
            enabled=enabled,
            just_activated=just_activated,
            reason=reason,
            telemetry=self._telemetry(),
        )

    def _telemetry(self) -> TailGateTelemetry:
        return TailGateTelemetry(
            state=self._state(),
            tick=self._tick,
            expected_accept_length=self._expected_accept_length,
            qualifying_checks=self._qualifying_checks,
            activation_tick=self._activation_tick,
            activation_active_requests=self._activation_active_requests,
            activation_sequence_length=self._activation_sequence_length,
            predicted_speedup=self._predicted_speedup,
            drafter_target_ratio=self._drafter_target_ratio,
            verify_target_ratio=self._verify_target_ratio,
        )

    def _state(self) -> TailGateState:
        if self._enabled:
            return "ON_LATCHED"
        if self._seen_ramp:
            return "ARMED_OFF"
        return "RAMPING_OFF"

    def _reset_rollout_state(self) -> None:
        self._enabled = self.config.mode == "off"
        self._seen_ramp = False
        self._qualifying_checks = 0
        self._tick = 0
        self._activation_tick = None
        self._activation_active_requests = None
        self._activation_sequence_length = None
        self._predicted_speedup = None
        self._drafter_target_ratio = None
        self._verify_target_ratio = None


def _validate_roofline_config(
    config: SDToggleConfig,
    *,
    gamma: int,
    expected_accept_length: float,
    margin: float,
) -> None:
    if gamma not in config.calibration.per_gamma:
        raise ValueError(f"roofline calibration requires exact K={gamma} fit")

    _require_nonempty_string(config.hardware.gpu, "hardware.gpu")
    _require_positive_int(config.hardware.tp, "hardware.tp")
    _require_finite_positive(config.hardware.BW_eff, "hardware.BW_eff")
    _require_nonempty_string(config.model.name, "model.name")
    for name, value in (
        ("model.W_t", config.model.W_t),
        ("model.W_d", config.model.W_d),
        ("model.C_dense", config.model.C_dense),
        ("model.C_attn", config.model.C_attn),
        ("calibration.eta_d", config.calibration.eta_d),
        ("calibration.kappa_eff", config.calibration.kappa_eff),
        ("calibration.F_eff", config.calibration.F_eff),
    ):
        _require_finite_positive(value, name)
    _require_positive_int(config.model.kappa_theoretical, "model.kappa_theoretical")

    gamma_fit = config.calibration.per_gamma[gamma]
    for name, value in (
        (f"calibration.per_gamma[{gamma}].c_T", gamma_fit.c_T),
        (f"calibration.per_gamma[{gamma}].c_D", gamma_fit.c_D),
        (f"calibration.per_gamma[{gamma}].c_V", gamma_fit.c_V),
    ):
        _require_finite_positive(value, name)

    predict_decision(
        config,
        1,
        0,
        gamma,
        expected_accept_length,
        margin,
    )


def _get_mode(data: Mapping[str, object], name: str) -> TailGateMode:
    value = data.get(name, "off")
    if value == "off":
        return "off"
    if value == "threshold":
        return "threshold"
    if value == "roofline":
        return "roofline"
    raise ValueError(f"{name} must be one of off, threshold, roofline")


def _get_int(data: Mapping[str, object], name: str, default: int) -> int:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _get_float(data: Mapping[str, object], name: str, default: float) -> float:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _get_optional_string(data: Mapping[str, object], name: str) -> str | None:
    value = data.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_nonempty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
