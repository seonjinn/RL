"""Unit tests for cross-precision NCCL-Reshard transform contracts."""

import pytest

from nemo_rl.weight_sync import refit_transforms
from nemo_rl.weight_sync.refit_transforms import (
    DestinationComponentSpec,
    REFIT_PLAN_PROTOCOL_VERSION,
    RefitTransformPlan,
    RefitTransformRequest,
    TransformComponentSpec,
    build_plan_agreement,
    plan_signature,
    plans_from_serialized_metadata,
    resolve_transform,
    validate_serialized_plan_agreement,
)


def test_bf16_to_nvfp4_w4a4_distinguishes_wire_from_destination_components() -> None:
    """Receiver conversion transfers BF16 while calibration stays at the destination."""
    codec = resolve_transform("bf16", "nvfp4_w4a4")

    assert codec.describe_outputs((64, 128), "torch.bfloat16") == (
        TransformComponentSpec("weight", (64, 128), "torch.bfloat16"),
    )
    assert codec.describe_destination((64, 128), "torch.bfloat16") == (
        DestinationComponentSpec("weight", (64, 64), "torch.uint8", "codec"),
        DestinationComponentSpec(
            "weight_scale", (64, 8), "torch.float8_e4m3fn", "codec"
        ),
        DestinationComponentSpec("weight_scale_2", (), "torch.float32", "codec"),
        DestinationComponentSpec("input_scale", (), "torch.float32", "calibration"),
    )


def test_plan_signature_includes_destination_source_and_completion_scope() -> None:
    """Agreement rejects plans that differ after the identical BF16 transfer."""
    wire = (TransformComponentSpec("weight", (64, 128), "torch.bfloat16"),)
    destination = (
        DestinationComponentSpec("weight", (64, 64), "torch.uint8", "codec"),
        DestinationComponentSpec(
            "weight_scale", (64, 8), "torch.float8_e4m3fn", "codec"
        ),
        DestinationComponentSpec("weight_scale_2", (), "torch.float32", "codec"),
        DestinationComponentSpec("input_scale", (), "torch.float32", "calibration"),
    )
    first = RefitTransformPlan(
        transform_id="bf16_to_nvfp4_w4a4",
        wire_components=wire,
        destination_components=destination,
        completion_key="model.layers.0.mlp.experts.w13",
        finalize_scope="model",
    )
    changed_completion = RefitTransformPlan(
        transform_id="bf16_to_nvfp4_w4a4",
        wire_components=wire,
        destination_components=destination,
        completion_key="model.layers.0.mlp.experts.w2",
        finalize_scope="model",
    )
    changed_source = RefitTransformPlan(
        transform_id="bf16_to_nvfp4_w4a4",
        wire_components=wire,
        destination_components=(
            *destination[:-1],
            DestinationComponentSpec("input_scale", (), "torch.float32", "codec"),
        ),
        completion_key="model.layers.0.mlp.experts.w13",
        finalize_scope="model",
    )
    metadata = {
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "transform_id": first.transform_id,
                    "wire_components": [
                        {
                            "role": component.role,
                            "global_shape": component.global_shape,
                            "dtype": component.dtype_name,
                        }
                        for component in first.wire_components
                    ],
                    "destination_components": [
                        {
                            "role": component.role,
                            "global_shape": component.global_shape,
                            "dtype": component.dtype_name,
                            "source": component.source,
                        }
                        for component in first.destination_components
                    ],
                    "completion_key": first.completion_key,
                    "finalize_scope": first.finalize_scope,
                }
            ]
        }
    }

    assert plans_from_serialized_metadata(metadata) == {
        "model.layers.0.mlp.experts.gate_proj.weight": first
    }
    assert plan_signature(
        {"model.layers.0.mlp.experts.gate_proj.weight": first}
    ) != plan_signature(
        {"model.layers.0.mlp.experts.gate_proj.weight": changed_completion}
    )
    assert plan_signature(
        {"model.layers.0.mlp.experts.gate_proj.weight": first}
    ) != plan_signature({"model.layers.0.mlp.experts.gate_proj.weight": changed_source})


def _mixed_serialized_metadata() -> tuple[dict, dict]:
    metadata = {
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.input_layernorm.weight",
                    "transform_id": "identity",
                    "finalize_scope": "parameter",
                    "components": [
                        {
                            "role": "weight",
                            "global_shape": [64],
                            "dtype": "torch.bfloat16",
                        }
                    ],
                },
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "transform_id": "bf16_to_mxfp8_e4m3_e8m0",
                    "finalize_scope": "parameter",
                    "components": [
                        {
                            "role": "weight",
                            "global_shape": [64, 64],
                            "dtype": "torch.float8_e4m3fn",
                        },
                        {
                            "role": "weight_scale",
                            "global_shape": [64, 2],
                            "dtype": "torch.uint8",
                        },
                    ],
                },
            ]
        }
    }
    agreement = build_plan_agreement(plans_from_serialized_metadata(metadata))
    metadata.update(
        {
            "refit_protocol_version": agreement["protocol_version"],
            "refit_component_count": agreement["component_count"],
            "plan_signature": agreement["plan_signature"],
        }
    )
    return metadata, agreement


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


def test_serialized_metadata_reproduces_plan_agreement() -> None:
    metadata = {
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "transform_id": "bf16_to_mxfp8_e4m3_e8m0",
                    "finalize_scope": "parameter",
                    "components": [
                        {
                            "role": "weight",
                            "global_shape": [64, 64],
                            "dtype": "torch.float8_e4m3fn",
                        },
                        {
                            "role": "weight_scale",
                            "global_shape": [64, 2],
                            "dtype": "torch.uint8",
                        },
                    ],
                }
            ]
        }
    }

    plans = plans_from_serialized_metadata(metadata)
    agreement = build_plan_agreement(plans)

    assert agreement == {
        "protocol_version": REFIT_PLAN_PROTOCOL_VERSION,
        "component_count": 2,
        "plan_signature": plan_signature(plans),
    }


def test_serialized_component_order_changes_plan_agreement() -> None:
    parameter = {
        "name": "model.layers.0.mlp.down_proj.weight",
        "transform_id": "bf16_to_mxfp8_e4m3_e8m0",
        "finalize_scope": "parameter",
        "components": [
            {
                "role": "weight",
                "global_shape": [64, 64],
                "dtype": "torch.float8_e4m3fn",
            },
            {
                "role": "weight_scale",
                "global_shape": [64, 2],
                "dtype": "torch.uint8",
            },
        ],
    }
    reordered = {**parameter, "components": list(reversed(parameter["components"]))}

    first = build_plan_agreement(
        plans_from_serialized_metadata(
            {"per_layer_params": {"model.layers.0": [parameter]}}
        )
    )
    second = build_plan_agreement(
        plans_from_serialized_metadata(
            {"per_layer_params": {"model.layers.0": [reordered]}}
        )
    )

    assert first["plan_signature"] != second["plan_signature"]


@pytest.mark.parametrize(
    ("field", "corrupted_value"),
    [
        ("refit_protocol_version", REFIT_PLAN_PROTOCOL_VERSION + 1),
        ("refit_component_count", 4),
        ("plan_signature", "corrupted"),
    ],
)
def test_validate_serialized_plan_agreement_rejects_corrupted_advertisement(
    field: str, corrupted_value: int | str
) -> None:
    metadata, _ = _mixed_serialized_metadata()
    metadata[field] = corrupted_value

    with pytest.raises(ValueError, match="does not match parameter metadata"):
        validate_serialized_plan_agreement(metadata)


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
        finalize_scope="parameter",
    )

    assert [component.role for component in plan.components] == [
        "weight",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
    ]
