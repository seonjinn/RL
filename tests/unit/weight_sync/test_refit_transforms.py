"""Unit tests for cross-precision NCCL-Reshard transform contracts."""

import pytest

from nemo_rl.weight_sync import refit_transforms
from nemo_rl.weight_sync.refit_transforms import (
    RefitTransformPlan,
    RefitTransformRequest,
    TransformComponentSpec,
    plan_signature,
    resolve_transform,
)


def test_bf16_to_mxfp8_codec_describes_weight_and_scale_outputs() -> None:
    """BF16 storage expands into ordered MXFP8 value and scale components."""
    request = RefitTransformRequest(
        parameter_names=("model.layers.0.mlp.down_proj.weight",),
        source_format="bf16",
        target_format="mxfp8_e4m3_e8m0",
    )

    codec = resolve_transform(request.source_format, request.target_format)
    components = codec.describe_outputs((64, 128), "torch.bfloat16")

    assert [(item.role, item.global_shape, item.dtype_name) for item in components] == [
        ("weight", (64, 128), "torch.float8_e4m3fn"),
        ("weight_scale", (64, 4), "torch.uint8"),
    ]


def test_bf16_to_mxfp8_codec_rejects_invalid_input_contracts() -> None:
    """A wrong storage dtype or unaligned K dimension cannot enter transfer."""
    codec = resolve_transform("bf16", "mxfp8_e4m3_e8m0")

    with pytest.raises(ValueError, match="torch.bfloat16"):
        codec.describe_outputs((64, 128), "torch.float16")

    with pytest.raises(ValueError, match="divisible by 32"):
        codec.describe_outputs((64, 127), "torch.bfloat16")


def test_resolve_transform_reports_both_unknown_formats() -> None:
    """Unsupported storage pairs identify both endpoints for configuration repair."""
    with pytest.raises(ValueError) as exc_info:
        resolve_transform("fp8_e4m3", "nvfp4")

    assert "fp8_e4m3" in str(exc_info.value)
    assert "nvfp4" in str(exc_info.value)


def test_plan_signature_ignores_parameter_mapping_insertion_order() -> None:
    """Equivalent parameter plans get the same pre-transfer signature."""
    weight = TransformComponentSpec("weight", (64, 128), "torch.float8_e4m3fn")
    scale = TransformComponentSpec("weight_scale", (64, 4), "torch.uint8")
    mxfp8_plan = RefitTransformPlan(
        transform_id="bf16_to_mxfp8_e4m3_e8m0",
        components=(weight, scale),
        finalize_scope="parameter",
    )
    identity_plan = RefitTransformPlan(
        transform_id="identity",
        components=(TransformComponentSpec("weight", (64, 64), "torch.bfloat16"),),
        finalize_scope="parameter",
    )

    first = {
        "model.layers.0.mlp.down_proj.weight": mxfp8_plan,
        "model.layers.0.input_layernorm.weight": identity_plan,
    }
    second = {
        "model.layers.0.input_layernorm.weight": identity_plan,
        "model.layers.0.mlp.down_proj.weight": mxfp8_plan,
    }

    assert plan_signature(first) == plan_signature(second)


def test_registry_preserves_test_codec_component_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generic plan retains every component supplied by a future codec."""

    class FourComponentCodec:
        transform_id = "test_four_component"

        def describe_outputs(
            self,
            global_shape: tuple[int, ...],
            input_dtype_name: str,
        ) -> tuple[TransformComponentSpec, ...]:
            del input_dtype_name
            return (
                TransformComponentSpec("weight", global_shape, "torch.uint8"),
                TransformComponentSpec("weight_scale", (64, 4), "torch.uint8"),
                TransformComponentSpec("weight_scale_2", (64,), "torch.float32"),
                TransformComponentSpec("input_scale", (1,), "torch.float32"),
            )

    monkeypatch.setitem(
        refit_transforms._TRANSFORM_CODECS,
        ("test_source", "test_target"),
        FourComponentCodec(),
    )
    codec = resolve_transform("test_source", "test_target")
    plan = RefitTransformPlan(
        transform_id=codec.transform_id,
        components=codec.describe_outputs((64, 128), "torch.bfloat16"),
        finalize_scope="layer",
    )

    assert [component.role for component in plan.components] == [
        "weight",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
    ]
