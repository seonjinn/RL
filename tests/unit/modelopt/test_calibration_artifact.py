from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from examples.modelopt.export_nvfp4_calibration import collect_nvfp4_input_amax
from nemo_rl.modelopt.calibration_artifact import (
    load_nvfp4_calibration,
    save_nvfp4_calibration,
)
from nemo_rl.modelopt.models.generation.nvfp4_refit import NVFP4Calibration


@pytest.fixture
def metadata() -> dict[str, str | int]:
    return {
        "model_id": "nvidia/Nemotron-3-Super-120B-A12B",
        "model_revision": "0123456789abcdef",
        "quant_cfg": "examples/modelopt/quant_configs/nvfp4_experts.yaml",
        "dataset": "cnn_dailymail",
        "sample_count": 16,
        "sequence_length": 1024,
        "seed": 1234,
    }


def _json_metadata(metadata: dict[str, str | int]) -> dict[str, str]:
    return {key: json.dumps(value) for key, value in metadata.items()}


def _load(
    path: Path,
    metadata: dict[str, str | int],
    *,
    expected_projection_names: set[str] | None = None,
) -> NVFP4Calibration:
    return load_nvfp4_calibration(
        path,
        model_id=str(metadata["model_id"]),
        model_revision=str(metadata["model_revision"]),
        quant_cfg=str(metadata["quant_cfg"]),
        expected_projection_names=expected_projection_names,
    )


def test_collect_input_amax_uses_enabled_hf_projection_names() -> None:
    class FakeQuantizer:
        def __init__(self, value: float, *, enabled: bool) -> None:
            self.is_enabled = enabled
            self._amax = torch.tensor(value, requires_grad=False)

    class FakeProjection(torch.nn.Module):
        def __init__(self, value: float, *, enabled: bool = True) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2, 2)))
            self.input_quantizer = FakeQuantizer(value, enabled=enabled)

    model = torch.nn.Module()
    model.add_module("gate_proj", FakeProjection(12.0))
    model.add_module("up_proj", FakeProjection(24.0))
    model.add_module("down_proj", FakeProjection(48.0, enabled=False))

    input_amax = collect_nvfp4_input_amax(model)

    assert list(input_amax) == ["gate_proj.weight", "up_proj.weight"]
    assert torch.equal(input_amax["gate_proj.weight"], torch.tensor(12.0))
    assert torch.equal(input_amax["up_proj.weight"], torch.tensor(24.0))
    assert all(not value.requires_grad for value in input_amax.values())


def test_round_trip_preserves_exact_hf_projection_names_and_metadata(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    path = tmp_path / "calibration.safetensors"
    input_amax = {
        "model.layers.0.mlp.gate_proj.weight": torch.tensor(12.0),
        "model.layers.0.mlp.up_proj.input_quantizer._amax": torch.tensor(24.0),
    }

    save_nvfp4_calibration(path, input_amax, **metadata)

    with safe_open(path, framework="pt", device="cpu") as artifact:
        assert artifact.keys() == [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ]
        assert artifact.metadata() == _json_metadata(metadata)

    calibration = _load(path, metadata)
    assert list(calibration.input_amax) == [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    ]
    assert torch.equal(
        calibration.input_amax["model.layers.0.mlp.gate_proj.weight"],
        torch.tensor(12.0),
    )
    assert torch.equal(
        calibration.input_amax["model.layers.0.mlp.up_proj.weight"],
        torch.tensor(24.0),
    )


def test_load_rejects_missing_required_metadata(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    path = tmp_path / "missing-metadata.safetensors"
    incomplete = _json_metadata(metadata)
    del incomplete["seed"]
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
        path,
        metadata=incomplete,
    )

    with pytest.raises(ValueError, match="missing required metadata.*seed"):
        _load(path, metadata)


def test_load_rejects_duplicate_normalized_projection_names(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    path = tmp_path / "duplicate-names.safetensors"
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor(1.0),
            "model.layers.0.mlp.up_proj.input_quantizer._amax": torch.tensor(2.0),
        },
        path,
        metadata=_json_metadata(metadata),
    )

    with pytest.raises(ValueError, match="duplicate normalized.*up_proj.weight"):
        _load(path, metadata)


def test_load_rejects_missing_and_unexpected_exact_projection_names(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    path = tmp_path / "wrong-projections.safetensors"
    save_file(
        {
            "layers.0.mlp.gate_proj.weight": torch.tensor(1.0),
            "model.layers.0.mlp.up_proj.weight": torch.tensor(2.0),
        },
        path,
        metadata=_json_metadata(metadata),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"projection names do not match.*missing .*model\.layers\.0\.mlp\."
            r"gate_proj\.weight.*unexpected .*layers\.0\.mlp\.gate_proj\.weight"
        ),
    ):
        _load(
            path,
            metadata,
            expected_projection_names={
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
            },
        )


@pytest.mark.parametrize(
    ("identity_key", "unexpected", "match"),
    [
        ("model_id", "other/model", "model_id"),
        ("model_revision", "other-revision", "model_revision"),
        ("quant_cfg", "NVFP4_DEFAULT_CFG", "quant_cfg"),
    ],
)
def test_load_rejects_unexpected_artifact_identity(
    tmp_path: Path,
    metadata: dict[str, str | int],
    identity_key: str,
    unexpected: str,
    match: str,
) -> None:
    path = tmp_path / f"wrong-{identity_key}.safetensors"
    artifact_metadata = dict(metadata)
    artifact_metadata[identity_key] = unexpected
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.tensor(1.0)},
        path,
        metadata=_json_metadata(artifact_metadata),
    )

    with pytest.raises(ValueError, match=f"{match}.*does not match"):
        _load(path, metadata)


@pytest.mark.parametrize(
    "invalid_amax",
    [
        torch.empty(0),
        torch.tensor([1.0]),
        torch.tensor(0.0),
        torch.tensor(-1.0),
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
    ],
    ids=("empty", "non-scalar", "zero", "negative", "nan", "infinite"),
)
def test_load_rejects_invalid_input_amax(
    tmp_path: Path,
    metadata: dict[str, str | int],
    invalid_amax: torch.Tensor,
) -> None:
    path = tmp_path / "invalid-amax.safetensors"
    save_file(
        {"model.layers.0.mlp.up_proj.weight": invalid_amax},
        path,
        metadata=_json_metadata(metadata),
    )

    with pytest.raises(ValueError, match="scalar input amax|finite and positive"):
        _load(path, metadata)


def test_save_rejects_empty_artifact_and_alias_collisions(
    tmp_path: Path,
    metadata: dict[str, str | int],
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        save_nvfp4_calibration(tmp_path / "empty.safetensors", {}, **metadata)

    duplicate = {
        "model.layers.0.mlp.up_proj.weight": torch.tensor(1.0),
        "model.layers.0.mlp.up_proj.input_quantizer._amax": torch.tensor(2.0),
    }
    with pytest.raises(ValueError, match="duplicate normalized.*up_proj.weight"):
        save_nvfp4_calibration(
            tmp_path / "duplicate.safetensors", duplicate, **metadata
        )
