from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from nemo_rl.modelopt.models.generation import nvfp4_refit
from nemo_rl.models.generation.vllm.config import VllmConfig


@dataclass(frozen=True)
class _FakeQuantMeta:
    qformat: str
    block_size: int
    weight_amax: torch.Tensor
    input_amax: torch.Tensor | None


@pytest.fixture
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
    _fake_pinned_quant_meta: None,
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
    _fake_pinned_quant_meta: None,
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
    _fake_pinned_quant_meta: None,
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
    _fake_pinned_quant_meta: None,
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


@pytest.mark.parametrize(
    "input_amax",
    [torch.tensor([1.0, 2.0]), torch.ones((2, 2))],
    ids=("vector", "matrix"),
)
def test_serialize_w4a4_rejects_non_scalar_input_amax(
    input_amax: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
    _fake_pinned_quant_meta: None,
) -> None:
    calls: list[tuple[str, torch.Tensor, object]] = []
    monkeypatch.setattr(
        nvfp4_refit,
        "get_modelopt_quant_exporter",
        lambda mode: (
            "modelopt_nvfp4",
            _fake_exporter(calls, with_input_scale=True),
        ),
    )
    monkeypatch.setattr(
        nvfp4_refit,
        "compute_nvfp4_input_scale",
        lambda value: torch.tensor(0.25),
    )
    name = "model.layers.0.mlp.up_proj.weight"

    with pytest.raises(ValueError, match="scalar input amax"):
        nvfp4_refit.serialize_bf16_nvfp4_group(
            {name: torch.ones((32, 16))},
            mode="w4a4",
            calibration=nvfp4_refit.NVFP4Calibration({name: input_amax}),
        )

    assert calls == []


def test_serializer_does_not_apply_caller_ignore_policy(
    monkeypatch: pytest.MonkeyPatch,
    _fake_pinned_quant_meta: None,
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
        assert result[0][0] == name
    assert [name for name, _, _ in calls] == [
        "model.lm_head.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ]


def test_non_weight_only_input_emits_nothing() -> None:
    assert (
        nvfp4_refit.serialize_bf16_nvfp4_group(
            {"model.layers.0.mlp.up_proj.bias": torch.ones(32)},
            mode="w4a16",
            calibration=None,
        )
        == []
    )


def test_vllm_config_exposes_optional_real_quant_calibration_path() -> None:
    assert "real_quant_calibration_path" in VllmConfig.__optional_keys__


def test_w4a4_calibration_path_is_absolute_and_forwarded_to_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nemo_rl.modelopt import utils as modelopt_utils
    from nemo_rl.modelopt.models.generation import vllm_modelopt
    from nemo_rl.modelopt.models.generation import vllm_quant_worker

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VLLM_MODELOPT_CALIBRATION_PATH", raising=False)
    monkeypatch.delenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        raising=False,
    )
    monkeypatch.setattr(vllm_modelopt, "register_nemo_modelopt_nvfp4", lambda: None)
    monkeypatch.setattr(
        vllm_modelopt, "quantization_method_for_mode", lambda mode: f"quant-{mode}"
    )
    monkeypatch.setattr(
        modelopt_utils, "resolve_nvfp4_real_quant_mode", lambda _: "w4a4"
    )
    monkeypatch.setattr(
        modelopt_utils,
        "build_vllm_modelopt_nvfp4_config",
        lambda **kwargs: kwargs,
    )
    calibration_path = Path("artifacts/calibration.safetensors")
    quant_cfg_path = Path("configs/nvfp4.yaml")
    quant_cfg_path.parent.mkdir()
    quant_cfg_path.write_text("quant_cfg: nvfp4\n")

    vllm_quant_worker._configure_quant_engine_kwargs(
        {
            "quant_cfg": str(quant_cfg_path),
            "real_quant": True,
            "real_quant_calibration_path": str(calibration_path),
        },
        {},
    )

    assert os.environ["VLLM_MODELOPT_CALIBRATION_PATH"] == str(
        calibration_path.resolve()
    )
    assert os.environ["VLLM_MODELOPT_CALIBRATION_QUANT_CFG"] == str(
        quant_cfg_path.resolve()
    )
    assert "VLLM_MODELOPT_CALIBRATION_PATH" in vllm_quant_worker._EXTRA_ENV_VARS
    assert "VLLM_MODELOPT_CALIBRATION_QUANT_CFG" in vllm_quant_worker._EXTRA_ENV_VARS


def test_w4a16_ignores_and_clears_calibration_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.modelopt import utils as modelopt_utils
    from nemo_rl.modelopt.models.generation import vllm_modelopt
    from nemo_rl.modelopt.models.generation import vllm_quant_worker

    monkeypatch.setenv("VLLM_MODELOPT_CALIBRATION_PATH", "/stale/calibration")
    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "/stale/quant-cfg.yaml",
    )
    monkeypatch.setattr(vllm_modelopt, "register_nemo_modelopt_nvfp4", lambda: None)
    monkeypatch.setattr(
        vllm_modelopt, "quantization_method_for_mode", lambda mode: f"quant-{mode}"
    )
    monkeypatch.setattr(
        modelopt_utils, "resolve_nvfp4_real_quant_mode", lambda _: "w4a16"
    )
    monkeypatch.setattr(
        modelopt_utils,
        "build_vllm_modelopt_nvfp4_config",
        lambda **kwargs: kwargs,
    )

    vllm_quant_worker._configure_quant_engine_kwargs(
        {
            "quant_cfg": "NVFP4_WEIGHT_ONLY_CFG",
            "real_quant": True,
            "real_quant_calibration_path": "does/not/need/to/exist.safetensors",
        },
        {},
    )

    assert "VLLM_MODELOPT_CALIBRATION_PATH" not in os.environ
    assert "VLLM_MODELOPT_CALIBRATION_QUANT_CFG" not in os.environ


def test_fake_quant_clears_calibration_quant_cfg_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.modelopt.models.generation import vllm_quant_worker

    monkeypatch.setenv(
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
        "/stale/quant-cfg.yaml",
    )

    vllm_quant_worker._configure_quant_engine_kwargs(
        {
            "quant_cfg": "NVFP4_DEFAULT_CFG",
            "real_quant": False,
        },
        {},
    )

    assert "VLLM_MODELOPT_CALIBRATION_QUANT_CFG" not in os.environ


def test_serializer_uses_modelopt_without_megatron_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quantized: list[tuple[torch.Tensor, torch.Tensor, str, torch.Tensor, int]] = []

    class FakeNVFP4QTensor:
        @staticmethod
        def get_weights_scaling_factor(
            weight: torch.Tensor,
            block_size: int,
            *,
            weights_scaling_factor_2: torch.Tensor,
            keep_high_precision: bool,
        ) -> tuple[torch.Tensor]:
            assert block_size == 16
            assert weights_scaling_factor_2.numel() == 1
            assert keep_high_precision is False
            return (torch.ones((weight.shape[0], weight.shape[1] // 16)),)

        @staticmethod
        def get_activation_scaling_factor(view: object) -> torch.Tensor:
            assert torch.equal(getattr(view, "input_amax"), torch.tensor(12.0))
            return torch.tensor(0.25)

    def to_quantized_weight(
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        qformat: str,
        weight_scale_2: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        quantized.append((weight, weight_scale, qformat, weight_scale_2, block_size))
        return torch.zeros((weight.shape[0], weight.shape[1] // 2), dtype=torch.uint8)

    packages = {
        name: ModuleType(name)
        for name in (
            "modelopt",
            "modelopt.torch",
            "modelopt.torch.export",
            "modelopt.torch.quantization",
            "modelopt.torch.quantization.qtensor",
        )
    }
    for name, package in packages.items():
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    quant_utils = ModuleType("modelopt.torch.export.quant_utils")
    quant_utils.QUANTIZATION_NVFP4 = "modelopt_nvfp4"
    quant_utils.QUANTIZATION_W4A16_NVFP4 = "modelopt_w4a16_nvfp4"
    quant_utils.to_quantized_weight = to_quantized_weight
    monkeypatch.setitem(sys.modules, quant_utils.__name__, quant_utils)

    nvfp4_tensor = ModuleType("modelopt.torch.quantization.qtensor.nvfp4_tensor")
    nvfp4_tensor.NVFP4QTensor = FakeNVFP4QTensor
    monkeypatch.setitem(sys.modules, nvfp4_tensor.__name__, nvfp4_tensor)
    monkeypatch.setitem(sys.modules, "megatron", None)

    name = "model.layers.0.mlp.up_proj.weight"
    input_amax = torch.tensor(12.0)
    result = nvfp4_refit.serialize_bf16_nvfp4_group(
        {name: torch.ones((32, 16), dtype=torch.bfloat16)},
        mode="w4a4",
        calibration=nvfp4_refit.NVFP4Calibration({name: input_amax}),
    )

    assert [output_name for output_name, _ in result] == [
        name,
        "model.layers.0.mlp.up_proj.weight_scale",
        "model.layers.0.mlp.up_proj.weight_scale_2",
        "model.layers.0.mlp.up_proj.input_scale",
    ]
    assert len(quantized) == 1
    _, _, qformat, weight_scale_2, block_size = quantized[0]
    assert qformat == "modelopt_nvfp4"
    assert weight_scale_2.numel() == 1
    assert block_size == 16
    assert result[0][0] == name
    assert result[0][1].dtype == torch.uint8


@pytest.mark.parametrize(
    ("quant_mode", "input_amax"),
    [
        ("w4a16_nvfp4", None),
        ("nvfp4", torch.tensor(12.0)),
    ],
)
def test_dependency_light_exporter_matches_megatron_bridge(
    quant_mode: str,
    input_amax: torch.Tensor | None,
) -> None:
    modelopt_utils = pytest.importorskip(
        "megatron.bridge.models.conversion.modelopt_utils"
    )
    torch.manual_seed(42)
    name = "model.layers.0.mlp.up_proj.weight"
    weight = torch.randn((32, 32), dtype=torch.bfloat16)
    weight_amax = weight.float().abs().amax().reshape(())

    local_qformat, local_exporter = nvfp4_refit.get_modelopt_quant_exporter(quant_mode)
    local_meta = nvfp4_refit._QuantMeta(
        qformat=local_qformat,
        block_size=16,
        weight_amax=weight_amax,
        input_amax=input_amax,
    )
    local_outputs = list(local_exporter(name, weight, local_meta))

    bridge_qformat, bridge_exporter = modelopt_utils.get_modelopt_quant_exporter(
        quant_mode
    )
    bridge_meta = modelopt_utils.QuantMeta(
        qformat=bridge_qformat,
        block_size=16,
        weight_amax=weight_amax,
        input_amax=input_amax,
    )
    bridge_outputs = list(bridge_exporter(name, weight, bridge_meta))

    assert [name for name, _ in local_outputs] == [name for name, _ in bridge_outputs]
    for (_, local_tensor), (_, bridge_tensor) in zip(
        local_outputs, bridge_outputs, strict=True
    ):
        assert local_tensor.dtype == bridge_tensor.dtype
        assert local_tensor.shape == bridge_tensor.shape
        assert torch.equal(local_tensor, bridge_tensor)
