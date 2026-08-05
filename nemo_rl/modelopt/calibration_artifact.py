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

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nemo_rl.modelopt.models.generation.nvfp4_refit import NVFP4Calibration

_REQUIRED_METADATA_KEYS = frozenset(
    {
        "model_id",
        "model_revision",
        "quant_cfg",
        "dataset",
        "sample_count",
        "sequence_length",
        "seed",
    }
)
_INPUT_AMAX_SUFFIX = ".input_quantizer._amax"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def normalize_quant_cfg_identity(quant_cfg: str) -> str:
    """Resolve an existing config path while preserving symbolic config names."""
    config_path = Path(quant_cfg).expanduser()
    if config_path.is_file():
        return str(config_path.resolve())
    project_config_path = _PROJECT_ROOT / config_path
    if project_config_path.is_file():
        return str(project_config_path.resolve())
    return quant_cfg


def save_nvfp4_calibration(
    path: str | Path,
    input_amax: Mapping[str, torch.Tensor],
    *,
    model_id: str,
    model_revision: str,
    quant_cfg: str,
    dataset: str,
    sample_count: int,
    sequence_length: int,
    seed: int,
) -> None:
    """Write validated HF projection input amax values to safetensors."""
    metadata: dict[str, str | int] = {
        "model_id": model_id,
        "model_revision": model_revision,
        "quant_cfg": quant_cfg,
        "dataset": dataset,
        "sample_count": sample_count,
        "sequence_length": sequence_length,
        "seed": seed,
    }
    _validate_metadata(metadata)
    tensors = _normalize_input_amax(input_amax)
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output_path),
        metadata={key: json.dumps(value) for key, value in metadata.items()},
    )


def load_nvfp4_calibration(
    path: str | Path,
    *,
    model_id: str,
    model_revision: str,
    quant_cfg: str,
    expected_projection_names: Collection[str] | None = None,
) -> NVFP4Calibration:
    """Load a calibration artifact after identity and projection validation."""
    artifact_path = Path(path).expanduser()
    with safe_open(artifact_path, framework="pt", device="cpu") as artifact:
        metadata = _decode_metadata(artifact.metadata())
        _validate_identity(
            metadata,
            model_id=model_id,
            model_revision=model_revision,
            quant_cfg=quant_cfg,
        )
        raw_names = list(artifact.keys())
        normalized_names = _validate_artifact_names(raw_names)
        if expected_projection_names is not None:
            _validate_expected_projection_names(
                normalized_names,
                expected_projection_names,
            )
        input_amax = {
            normalized_name: _validated_input_amax(
                normalized_name,
                artifact.get_tensor(raw_name),
            ).clone()
            for raw_name, normalized_name in zip(
                raw_names, normalized_names, strict=True
            )
        }
    return NVFP4Calibration(input_amax=input_amax)


def _normalize_projection_name(name: str) -> str:
    if name.endswith(".weight"):
        return name
    if name.endswith(_INPUT_AMAX_SUFFIX):
        return name.removesuffix(_INPUT_AMAX_SUFFIX) + ".weight"
    raise ValueError(
        "NVFP4 calibration names must be exact HF projection '.weight' names "
        f"or ModelOpt input amax names; got {name!r}"
    )


def _normalize_input_amax(
    input_amax: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not input_amax:
        raise ValueError("NVFP4 calibration requires at least one input amax tensor")

    normalized: dict[str, torch.Tensor] = {}
    for name, tensor in input_amax.items():
        normalized_name = _normalize_projection_name(name)
        if normalized_name in normalized:
            raise ValueError(
                "NVFP4 calibration has duplicate normalized projection name "
                f"{normalized_name!r}"
            )
        normalized[normalized_name] = (
            _validated_input_amax(normalized_name, tensor).detach().cpu().contiguous()
        )
    return normalized


def _validated_input_amax(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 0:
        raise ValueError(f"NVFP4 calibration requires a scalar input amax for {name!r}")
    if not bool(torch.isfinite(tensor).item()) or not bool((tensor > 0).item()):
        raise ValueError(
            f"NVFP4 calibration input amax must be finite and positive for {name!r}"
        )
    return tensor


def _decode_metadata(raw_metadata: dict[str, str] | None) -> dict[str, object]:
    metadata = raw_metadata or {}
    missing = sorted(_REQUIRED_METADATA_KEYS.difference(metadata))
    if missing:
        raise ValueError(
            "NVFP4 calibration artifact is missing required metadata: "
            + ", ".join(missing)
        )

    decoded: dict[str, object] = {}
    for key in _REQUIRED_METADATA_KEYS:
        try:
            decoded[key] = json.loads(metadata[key])
        except json.JSONDecodeError as error:
            raise ValueError(
                f"NVFP4 calibration metadata {key!r} is not valid JSON"
            ) from error
    _validate_metadata(decoded)
    return decoded


def _validate_metadata(metadata: Mapping[str, object]) -> None:
    for key in ("model_id", "model_revision", "quant_cfg", "dataset"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"NVFP4 calibration metadata {key!r} must be a non-empty string"
            )
    for key in ("sample_count", "sequence_length"):
        value = metadata.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"NVFP4 calibration metadata {key!r} must be a positive integer"
            )
    seed = metadata.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("NVFP4 calibration metadata 'seed' must be an integer")


def _validate_identity(
    metadata: Mapping[str, object],
    *,
    model_id: str,
    model_revision: str,
    quant_cfg: str,
) -> None:
    expected = {
        "model_id": model_id,
        "model_revision": model_revision,
        "quant_cfg": quant_cfg,
    }
    for key, expected_value in expected.items():
        actual_value = metadata[key]
        if actual_value != expected_value:
            raise ValueError(
                f"NVFP4 calibration {key} {actual_value!r} does not match "
                f"expected {expected_value!r}"
            )


def _validate_artifact_names(raw_names: list[str]) -> list[str]:
    if not raw_names:
        raise ValueError("NVFP4 calibration requires at least one input amax tensor")

    normalized_names: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_names:
        normalized_name = _normalize_projection_name(raw_name)
        if normalized_name in seen:
            raise ValueError(
                "NVFP4 calibration has duplicate normalized projection name "
                f"{normalized_name!r}"
            )
        seen.add(normalized_name)
        normalized_names.append(normalized_name)

    noncanonical = [
        raw_name
        for raw_name, normalized_name in zip(raw_names, normalized_names, strict=True)
        if raw_name != normalized_name
    ]
    if noncanonical:
        raise ValueError(
            "NVFP4 calibration artifact keys must be exact HF projection "
            f"'.weight' names: {noncanonical}"
        )
    return normalized_names


def _validate_expected_projection_names(
    artifact_names: Collection[str],
    expected_projection_names: Collection[str],
) -> None:
    expected: set[str] = set()
    for name in expected_projection_names:
        normalized_name = _normalize_projection_name(name)
        if name != normalized_name:
            raise ValueError(
                "Expected NVFP4 calibration projection names must be exact HF "
                f"'.weight' names; got {name!r}"
            )
        if name in expected:
            raise ValueError(f"Duplicate expected NVFP4 projection name {name!r}")
        expected.add(name)

    actual = set(artifact_names)
    missing = sorted(expected.difference(actual))
    unexpected = sorted(actual.difference(expected))
    if missing or unexpected:
        raise ValueError(
            "NVFP4 calibration projection names do not match: "
            f"missing {missing}; unexpected {unexpected}"
        )
