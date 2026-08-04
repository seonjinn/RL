from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from nemo_rl.modelopt.models.generation import nvfp4_refit


@dataclass(frozen=True)
class _FakeQuantMeta:
    qformat: str
    block_size: int
    weight_amax: torch.Tensor
    input_amax: torch.Tensor | None


@pytest.fixture(autouse=True)
def _fake_pinned_quant_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvfp4_refit, "_load_quant_meta", lambda: _FakeQuantMeta)


def _fake_exporter(
    calls: list[tuple[str, torch.Tensor, object]],
    *,
    with_input_scale: bool,
) -> Callable[[str, torch.Tensor, Any], Iterator[tuple[str, torch.Tensor]]]:
    def export(
        name: str, weight: torch.Tensor, meta: Any
    ) -> Iterator[tuple[str, torch.Tensor]]:
        calls.append((name, weight, meta))
        rows, columns = weight.shape
        output = [
            (name, torch.zeros((rows, columns // 2), dtype=torch.uint8)),
            (
                name.removesuffix(".weight") + ".weight_scale",
                torch.zeros((rows, columns // 16), dtype=torch.float8_e4m3fn),
            ),
            (
                name.removesuffix(".weight") + ".weight_scale_2",
                torch.tensor(0.5, dtype=torch.float32),
            ),
        ]
        if with_input_scale:
            output.append(
                (
                    name.removesuffix(".weight") + ".input_scale",
                    torch.tensor(0.25, dtype=torch.float32),
                )
            )
        return iter(output)

    return export


def test_nvfp4_refit_group_names() -> None:
    gate = "model.layers.0.mlp.experts.3.gate_proj.weight"
    up = "model.layers.0.mlp.experts.3.up_proj.weight"
    down = "model.layers.0.mlp.experts.3.down_proj.weight"

    assert nvfp4_refit.nvfp4_refit_group(gate) == (
        "model.layers.0.mlp.experts.3.w13",
        (gate, up),
    )
    assert nvfp4_refit.nvfp4_refit_group(up) == (
        "model.layers.0.mlp.experts.3.w13",
        (gate, up),
    )
    assert nvfp4_refit.nvfp4_refit_group(down) == (
        "model.layers.0.mlp.experts.3.w2",
        (down,),
    )
    assert nvfp4_refit.nvfp4_refit_group("model.layers.0.mlp.down_proj.weight") == (
        "model.layers.0.mlp.down_proj.weight",
        ("model.layers.0.mlp.down_proj.weight",),
    )


def test_serialize_singleton_down_projection_uses_exact_canonical_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.Tensor, object]] = []
    requested_modes: list[str] = []

    def get_exporter(mode: str) -> tuple[str, Callable[..., Any]]:
        requested_modes.append(mode)
        return (
            "modelopt_w4a16_nvfp4",
            _fake_exporter(calls, with_input_scale=False),
        )

    monkeypatch.setattr(
        nvfp4_refit,
        "get_modelopt_quant_exporter",
        get_exporter,
    )

    name = "model.layers.0.mlp.experts.3.down_proj.weight"
    result = nvfp4_refit.serialize_bf16_nvfp4_group(
        {name: torch.ones((32, 16), dtype=torch.bfloat16)},
        mode="w4a16",
        calibration=None,
    )

    assert [output_name for output_name, _ in result] == [
        name,
        "model.layers.0.mlp.experts.3.down_proj.weight_scale",
        "model.layers.0.mlp.experts.3.down_proj.weight_scale_2",
    ]
    assert [tensor.dtype for _, tensor in result] == [
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float32,
    ]
    assert [tuple(tensor.shape) for _, tensor in result] == [(32, 8), (32, 1), ()]
    assert calls[0][0] == name
    assert calls[0][2].qformat == "modelopt_w4a16_nvfp4"
    assert calls[0][2].block_size == 16
    assert calls[0][2].input_amax is None
    assert requested_modes == ["w4a16_nvfp4"]


def test_serialize_gate_up_is_one_group_with_shared_weight_amax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.Tensor, object]] = []
    monkeypatch.setattr(
        nvfp4_refit,
        "get_modelopt_quant_exporter",
        lambda mode: (
            "modelopt_w4a16_nvfp4",
            _fake_exporter(calls, with_input_scale=False),
        ),
    )
    gate = "model.layers.0.mlp.experts.3.gate_proj.weight"
    up = "model.layers.0.mlp.experts.3.up_proj.weight"

    result = nvfp4_refit.serialize_bf16_nvfp4_group(
        {
            gate: torch.full((32, 16), 2.0, dtype=torch.bfloat16),
            up: torch.full((32, 16), 3.0, dtype=torch.bfloat16),
        },
        mode="w4a16",
        calibration=None,
    )

    assert [output_name for output_name, _ in result] == [
        gate,
        "model.layers.0.mlp.experts.3.gate_proj.weight_scale",
        "model.layers.0.mlp.experts.3.gate_proj.weight_scale_2",
        up,
        "model.layers.0.mlp.experts.3.up_proj.weight_scale",
        "model.layers.0.mlp.experts.3.up_proj.weight_scale_2",
    ]
    assert len(calls) == 2
    assert {call[0] for call in calls} == {gate, up}
    assert calls[0][2].weight_amax.item() == 3.0
    assert calls[1][2].weight_amax.item() == 3.0


def test_serialize_w4a4_requires_named_calibration_and_emits_input_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.Tensor, object]] = []
    input_scale_calls: list[torch.Tensor] = []
    monkeypatch.setattr(
        nvfp4_refit,
        "get_modelopt_quant_exporter",
        lambda mode: ("modelopt_nvfp4", _fake_exporter(calls, with_input_scale=True)),
    )
    monkeypatch.setattr(
        nvfp4_refit,
        "compute_nvfp4_input_scale",
        lambda value: input_scale_calls.append(value) or torch.tensor(0.25),
    )
    name = "model.layers.0.mlp.up_proj.weight"
    amax = torch.tensor(12.0)

    result = nvfp4_refit.serialize_bf16_nvfp4_group(
        {name: torch.ones((32, 16), dtype=torch.bfloat16)},
        mode="w4a4",
        calibration=nvfp4_refit.NVFP4Calibration(input_amax={name: amax}),
    )

    assert [output_name for output_name, _ in result][
        -1
    ] == "model.layers.0.mlp.up_proj.input_scale"
    assert input_scale_calls == [amax]
    assert calls[0][2].qformat == "modelopt_nvfp4"
    assert calls[0][2].input_amax is amax


def test_serialize_w4a4_preserves_distinct_gate_up_calibration_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.Tensor, object]] = []
    input_scale_calls: list[torch.Tensor] = []
    monkeypatch.setattr(
        nvfp4_refit,
        "get_modelopt_quant_exporter",
        lambda mode: ("modelopt_nvfp4", _fake_exporter(calls, with_input_scale=True)),
    )
    monkeypatch.setattr(
        nvfp4_refit,
        "compute_nvfp4_input_scale",
        lambda value: input_scale_calls.append(value) or torch.tensor(0.25),
    )
    gate = "model.layers.0.mlp.experts.3.gate_proj.weight"
    up = "model.layers.0.mlp.experts.3.up_proj.weight"
    gate_amax = torch.tensor(10.0)
    up_amax = torch.tensor(20.0)

    result = nvfp4_refit.serialize_bf16_nvfp4_group(
        {
            gate: torch.ones((32, 16)),
            up: torch.ones((32, 16)),
        },
        mode="w4a4",
        calibration=nvfp4_refit.NVFP4Calibration(
            input_amax={gate: gate_amax, up: up_amax}
        ),
    )

    result_names = [name for name, _ in result]
    assert result_names == [
        gate,
        "model.layers.0.mlp.experts.3.gate_proj.weight_scale",
        "model.layers.0.mlp.experts.3.gate_proj.weight_scale_2",
        "model.layers.0.mlp.experts.3.gate_proj.input_scale",
        up,
        "model.layers.0.mlp.experts.3.up_proj.weight_scale",
        "model.layers.0.mlp.experts.3.up_proj.weight_scale_2",
        "model.layers.0.mlp.experts.3.up_proj.input_scale",
    ]
    assert len(result_names) == len(set(result_names))
    assert input_scale_calls == [gate_amax, up_amax]
    assert [call[2].input_amax for call in calls] == [gate_amax, up_amax]


@pytest.mark.parametrize(
    ("tensors", "match"),
    [
        (
            {"model.layers.0.mlp.experts.0.gate_proj.weight": torch.ones((32, 16))},
            "not complete",
        ),
        (
            {
                "model.layers.0.mlp.experts.0.gate_proj.weight": torch.ones((32, 16)),
                "model.layers.0.mlp.experts.0.up_proj.weight": torch.ones((32, 32)),
            },
            "same K",
        ),
        (
            {"model.layers.0.mlp.down_proj.weight": torch.ones((32, 15))},
            "divisible by 16",
        ),
        ({"model.layers.0.mlp.down_proj.weight": torch.ones(16)}, "2-D"),
        ({"model.layers.0.mlp.down_proj.weight": torch.zeros((32, 16))}, "positive"),
        (
            {"model.layers.0.mlp.down_proj.weight": torch.full((32, 16), float("nan"))},
            "finite",
        ),
    ],
)
def test_serialize_rejects_invalid_or_incomplete_groups(
    tensors: Mapping[str, torch.Tensor], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        nvfp4_refit.serialize_bf16_nvfp4_group(
            tensors,
            mode="w4a16",
            calibration=None,
        )


@pytest.mark.parametrize(
    "input_amax",
    [None, torch.tensor(0.0), torch.tensor(float("nan")), torch.tensor(float("inf"))],
)
def test_serialize_w4a4_rejects_missing_or_invalid_named_calibration(
    input_amax: torch.Tensor | None,
) -> None:
    name = "model.layers.0.mlp.up_proj.weight"
    calibration = (
        None if input_amax is None else nvfp4_refit.NVFP4Calibration({name: input_amax})
    )
    with pytest.raises((KeyError, ValueError, RuntimeError), match="input amax"):
        nvfp4_refit.serialize_bf16_nvfp4_group(
            {name: torch.ones((32, 16))},
            mode="w4a4",
            calibration=calibration,
        )


def test_ignored_names_are_not_passed_through_to_the_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.Tensor, object]] = []
    monkeypatch.setattr(
        nvfp4_refit,
        "get_modelopt_quant_exporter",
        lambda mode: (
            "modelopt_w4a16_nvfp4",
            _fake_exporter(calls, with_input_scale=False),
        ),
    )

    for name in ("model.lm_head.weight", "model.layers.0.mlp.gate_proj.weight"):
        result = nvfp4_refit.serialize_bf16_nvfp4_group(
            {name: torch.ones((32, 16))},
            mode="w4a16",
            calibration=None,
        )
        assert result == []
    assert calls == []
