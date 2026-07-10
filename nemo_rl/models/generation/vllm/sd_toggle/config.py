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

"""Typed configuration for the EfficientRollout roofline model."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware parameters used by the roofline model."""

    gpu: str
    tp: int
    BW_eff: float
    BW_peak: float = 0.0
    F_peak: float = 0.0
    c_comm: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    """Target and drafter model constants used by the roofline model."""

    name: str
    W_t: float
    W_d: float
    C_dense: float
    C_attn: float
    kappa_theoretical: int
    rho: float = 0.0
    gqa: int = 1


@dataclass(frozen=True)
class PerGammaCalibration:
    """Per-draft-length latency overhead calibration."""

    c_D: float
    c_V: float
    c_T: float = 0.0
    R2: float = 0.0


@dataclass(frozen=True)
class CalibrationConfig:
    """Roofline parameters calibrated for one model and hardware topology."""

    eta_d: float
    kappa_eff: float
    F_eff: float
    c_T: float = 0.0
    c_D: float = 0.0
    c_V: float = 0.0
    beta: float = 0.0
    per_gamma: dict[int, PerGammaCalibration] = field(default_factory=dict)
    per_gamma_full: dict[int, dict[str, float]] | None = None


@dataclass(frozen=True)
class SDToggleConfig:
    """Complete EfficientRollout-compatible speculative-decoding calibration."""

    hardware: HardwareConfig
    model: ModelConfig
    calibration: CalibrationConfig
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def get_gamma_params(self, gamma: int) -> PerGammaCalibration:
        """Returns calibrated overheads for ``gamma`` or its nearest legacy match."""
        if gamma in self.calibration.per_gamma:
            return self.calibration.per_gamma[gamma]
        if self.calibration.c_D > 0.0 or self.calibration.c_V > 0.0:
            return PerGammaCalibration(
                c_D=self.calibration.c_D,
                c_V=self.calibration.c_V,
                c_T=self.calibration.c_T,
            )
        if not self.calibration.per_gamma:
            raise ValueError("No calibration data available")
        nearest_gamma = min(
            self.calibration.per_gamma,
            key=lambda calibrated_gamma: abs(calibrated_gamma - gamma),
        )
        return self.calibration.per_gamma[nearest_gamma]


def load_config(path: str | Path) -> SDToggleConfig:
    """Loads an EfficientRollout-compatible calibration JSON file.

    All parsed numeric values must be finite. Values that make a roofline
    denominator non-positive are rejected by the prediction entry points.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    root = _as_dict(data, "root")
    hardware_data = _as_dict(_required(root, "hardware"), "hardware")
    model_data = _as_dict(_required(root, "model"), "model")
    calibration_data = _as_dict(_required(root, "calibration"), "calibration")

    hardware = HardwareConfig(
        gpu=_as_string(_required(hardware_data, "gpu"), "hardware.gpu"),
        tp=_as_positive_int(_required(hardware_data, "tp"), "hardware.tp"),
        BW_eff=_as_float(_required(hardware_data, "BW_eff"), "hardware.BW_eff"),
        BW_peak=_as_float(hardware_data.get("BW_peak", 0.0), "hardware.BW_peak"),
        F_peak=_as_float(hardware_data.get("F_peak", 0.0), "hardware.F_peak"),
        c_comm=_as_float(hardware_data.get("c_comm", 0.0), "hardware.c_comm"),
    )
    model = ModelConfig(
        name=_as_string(_required(model_data, "name"), "model.name"),
        W_t=_as_nonnegative_float(_required(model_data, "W_t"), "model.W_t"),
        W_d=_as_nonnegative_float(_required(model_data, "W_d"), "model.W_d"),
        C_dense=_as_nonnegative_float(
            _required(model_data, "C_dense"), "model.C_dense"
        ),
        C_attn=_as_nonnegative_float(_required(model_data, "C_attn"), "model.C_attn"),
        kappa_theoretical=_as_nonnegative_int(
            _required(model_data, "kappa_theoretical"), "model.kappa_theoretical"
        ),
        rho=_as_float(model_data.get("rho", 0.0), "model.rho"),
        gqa=_as_positive_int(model_data.get("gqa", 1), "model.gqa"),
    )
    calibration = CalibrationConfig(
        eta_d=_as_at_least_one(
            _required(calibration_data, "eta_d"), "calibration.eta_d"
        ),
        kappa_eff=_as_nonnegative_float(
            _required(calibration_data, "kappa_eff"), "calibration.kappa_eff"
        ),
        F_eff=_as_nonnegative_float(
            _required(calibration_data, "F_eff"), "calibration.F_eff"
        ),
        c_T=_as_float(calibration_data.get("c_T", 0.0), "calibration.c_T"),
        c_D=_as_float(calibration_data.get("c_D", 0.0), "calibration.c_D"),
        c_V=_as_float(calibration_data.get("c_V", 0.0), "calibration.c_V"),
        beta=_as_nonnegative_float(
            calibration_data.get("beta", 0.0), "calibration.beta"
        ),
        per_gamma=_parse_per_gamma(calibration_data.get("per_gamma", {})),
        per_gamma_full=_parse_per_gamma_full(calibration_data.get("per_gamma_full")),
    )
    metadata = _as_json_dict(root.get("metadata", {}), "metadata")
    return SDToggleConfig(
        hardware=hardware,
        model=model,
        calibration=calibration,
        metadata=metadata,
    )


def _parse_per_gamma(value: object) -> dict[int, PerGammaCalibration]:
    data = _as_dict(value, "calibration.per_gamma")
    result: dict[int, PerGammaCalibration] = {}
    for gamma_text, gamma_data in data.items():
        gamma = _parse_gamma_key(gamma_text, "calibration.per_gamma key")
        overhead = _as_dict(gamma_data, f"calibration.per_gamma.{gamma_text}")
        result[gamma] = PerGammaCalibration(
            c_D=_as_float(
                _required(overhead, "c_D"), f"calibration.per_gamma.{gamma_text}.c_D"
            ),
            c_V=_as_float(
                _required(overhead, "c_V"), f"calibration.per_gamma.{gamma_text}.c_V"
            ),
            c_T=_as_float(
                overhead.get("c_T", 0.0), f"calibration.per_gamma.{gamma_text}.c_T"
            ),
            R2=_as_float(
                overhead.get("R2", 0.0), f"calibration.per_gamma.{gamma_text}.R2"
            ),
        )
    return result


def _parse_per_gamma_full(value: object) -> dict[int, dict[str, float]] | None:
    if value is None:
        return None
    data = _as_dict(value, "calibration.per_gamma_full")
    result: dict[int, dict[str, float]] = {}
    for gamma_text, gamma_data in data.items():
        gamma = _parse_gamma_key(gamma_text, "calibration.per_gamma_full key")
        parameters = _as_dict(gamma_data, f"calibration.per_gamma_full.{gamma_text}")
        result[gamma] = {
            parameter_name: _as_float(
                parameter_value,
                f"calibration.per_gamma_full.{gamma_text}.{parameter_name}",
            )
            for parameter_name, parameter_value in parameters.items()
        }
    return result


def _required(data: dict[str, object], name: str) -> object:
    if name not in data:
        raise ValueError(f"Missing required field: {name}")
    return data[name]


def _as_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _as_json_dict(value: object, name: str) -> dict[str, JsonValue]:
    data = _as_dict(value, name)
    return {key: _as_json_value(item, f"{name}.{key}") for key, item in data.items()}


def _as_json_value(value: object, name: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _as_float(value, name)
    if isinstance(value, list):
        return [_as_json_value(item, name) for item in value]
    if isinstance(value, dict):
        return _as_json_dict(value, name)
    raise ValueError(f"{name} must be JSON-compatible")


def _as_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _as_positive_int(value: object, name: str) -> int:
    result = _as_int(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _parse_gamma_key(value: str, name: str) -> int:
    if not value.isdecimal():
        raise ValueError(f"{name} must be a positive integer")
    return _as_positive_int(int(value), name)


def _as_nonnegative_int(value: object, name: str) -> int:
    result = _as_int(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _as_at_least_one(value: object, name: str) -> float:
    result = _as_float(value, name)
    if result < 1.0:
        raise ValueError(f"{name} must be at least 1.0")
    return result


def _as_nonnegative_float(value: object, name: str) -> float:
    result = _as_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
