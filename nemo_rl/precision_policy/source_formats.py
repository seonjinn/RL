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

"""Evidence-reviewed canonical source formats used during topology discovery."""

from __future__ import annotations

from collections.abc import Sequence

from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    BLOCK_SCALES,
    MXFP8_FORMAT,
    VALUES,
    AxisExtentRounding,
    ComponentDescriptor,
    ComponentRole,
    FormatDescriptor,
    LiteralComponentAxisSpec,
    LogicalComponentAxisSpec,
)


INVERSE_SCALES = ComponentRole("inverse_scales")
PACKED_VALUES = ComponentRole("packed_values")
GROUP_SCALES = ComponentRole("group_scales")
LOGICAL_SHAPE = ComponentRole("logical_shape")
GLOBAL_SCALE = ComponentRole("global_scale")


def _logical_axis(
    name: str,
    divisor: int,
    rounding: AxisExtentRounding = AxisExtentRounding.EXACT,
) -> LogicalComponentAxisSpec:
    return LogicalComponentAxisSpec(name, divisor=divisor, rounding=rounding)


BLOCK_FP8_F32_SCALE_INV_FORMAT = FormatDescriptor(
    format_id="block-fp8.e4m3-f32-scale-inv-block128x128.v1",
    family="block_fp8",
    components=(
        ComponentDescriptor(
            role=VALUES,
            dtype="e4m3",
            encoding="float8_e4m3_values",
        ),
        ComponentDescriptor(
            role=INVERSE_SCALES,
            dtype="float32",
            encoding="inverse_scale_float32",
            component_axes=(
                _logical_axis("output_features", 128),
                _logical_axis("input_features", 128),
            ),
        ),
    ),
)

BLOCK_FP8_BF16_SCALE_INV_FORMAT = FormatDescriptor(
    format_id="block-fp8.e4m3-bf16-scale-inv-block128x128.v1",
    family="block_fp8",
    components=(
        ComponentDescriptor(
            role=VALUES,
            dtype="e4m3",
            encoding="float8_e4m3_values",
        ),
        ComponentDescriptor(
            role=INVERSE_SCALES,
            dtype="bfloat16",
            encoding="inverse_scale_bfloat16",
            component_axes=(
                _logical_axis("output_features", 128),
                _logical_axis("input_features", 128),
            ),
        ),
    ),
)

PACKED_INT4_I32_BF16_FORMAT = FormatDescriptor(
    format_id="packed-int4.i32-bf16-group32-shape-i32.v1",
    family="packed_int4",
    components=(
        ComponentDescriptor(
            role=PACKED_VALUES,
            dtype="int32",
            encoding="int4_offset_binary_pack8",
            component_axes=(
                _logical_axis("output_features", 1),
                _logical_axis("input_features", 8),
            ),
        ),
        ComponentDescriptor(
            role=GROUP_SCALES,
            dtype="bfloat16",
            encoding="symmetric_group_scale",
            component_axes=(
                _logical_axis("output_features", 1),
                _logical_axis("input_features", 32),
            ),
        ),
        ComponentDescriptor(
            role=LOGICAL_SHAPE,
            dtype="int32",
            encoding="logical_shape_vector",
            component_axes=(LiteralComponentAxisSpec("literal", 2),),
        ),
    ),
)

PACKED_INT4_I32_F16_FORMAT = FormatDescriptor(
    format_id="packed-int4.i32-f16-group32-shape-i64.v1",
    family="packed_int4",
    components=(
        ComponentDescriptor(
            role=PACKED_VALUES,
            dtype="int32",
            encoding="int4_offset_binary_pack8",
            component_axes=(
                _logical_axis("output_features", 1),
                _logical_axis("input_features", 8),
            ),
        ),
        ComponentDescriptor(
            role=GROUP_SCALES,
            dtype="float16",
            encoding="symmetric_group_scale",
            component_axes=(
                _logical_axis("output_features", 1),
                _logical_axis("input_features", 32),
            ),
        ),
        ComponentDescriptor(
            role=LOGICAL_SHAPE,
            dtype="int64",
            encoding="logical_shape_vector",
            component_axes=(LiteralComponentAxisSpec("literal", 2),),
        ),
    ),
)

MXFP4_U8_FORMAT = FormatDescriptor(
    format_id="mxfp4.u8-u8-block32-input-features.v1",
    family="mxfp4",
    components=(
        ComponentDescriptor(
            role=PACKED_VALUES,
            dtype="uint8",
            encoding="mxfp4_pack2",
            component_axes=(
                _logical_axis("output_features", 1),
                _logical_axis("input_features", 2),
            ),
        ),
        ComponentDescriptor(
            role=BLOCK_SCALES,
            dtype="uint8",
            encoding="mxfp4_block_scale",
            component_axes=(
                _logical_axis("output_features", 1),
                _logical_axis("input_features", 32),
            ),
        ),
    ),
)

NVFP4_U8_FORMAT = FormatDescriptor(
    format_id="nvfp4.u8-e4m3-f32-block16-input-features.v1",
    family="nvfp4",
    components=(
        ComponentDescriptor(
            role=PACKED_VALUES,
            dtype="uint8",
            encoding="nvfp4_pack2",
            component_axes=(
                _logical_axis("output_features", 1),
                _logical_axis("input_features", 2),
            ),
        ),
        ComponentDescriptor(
            role=BLOCK_SCALES,
            dtype="e4m3",
            encoding="nvfp4_block_scale",
            component_axes=(
                _logical_axis("output_features", 1),
                _logical_axis("input_features", 16),
            ),
        ),
        ComponentDescriptor(
            role=GLOBAL_SCALE,
            dtype="float32",
            encoding="nvfp4_global_scale",
            component_axes=(),
        ),
    ),
)


def build_source_format_catalog(
    formats: Sequence[FormatDescriptor],
) -> tuple[FormatDescriptor, ...]:
    """Build a literal catalog while rejecting every repeated stable ID."""
    catalog = tuple(formats)
    if any(not isinstance(item, FormatDescriptor) for item in catalog):
        raise TypeError("source format catalog must contain FormatDescriptor values")
    seen: set[str] = set()
    for descriptor in catalog:
        if descriptor.format_id in seen:
            raise ValueError(
                f"duplicate source format_id in literal catalog: {descriptor.format_id}"
            )
        for builtin in (BF16_FORMAT, MXFP8_FORMAT):
            if descriptor.format_id == builtin.format_id and descriptor is not builtin:
                raise ValueError(
                    "reserved source format_id must reuse its canonical object: "
                    f"{descriptor.format_id}"
                )
        seen.add(descriptor.format_id)
    return catalog


SOURCE_FORMAT_CATALOG = build_source_format_catalog(
    (
        BF16_FORMAT,
        MXFP8_FORMAT,
        BLOCK_FP8_F32_SCALE_INV_FORMAT,
        BLOCK_FP8_BF16_SCALE_INV_FORMAT,
        PACKED_INT4_I32_BF16_FORMAT,
        PACKED_INT4_I32_F16_FORMAT,
        MXFP4_U8_FORMAT,
        NVFP4_U8_FORMAT,
    )
)


__all__ = [
    "BLOCK_FP8_BF16_SCALE_INV_FORMAT",
    "BLOCK_FP8_F32_SCALE_INV_FORMAT",
    "GLOBAL_SCALE",
    "GROUP_SCALES",
    "INVERSE_SCALES",
    "LOGICAL_SHAPE",
    "MXFP4_U8_FORMAT",
    "NVFP4_U8_FORMAT",
    "PACKED_INT4_I32_BF16_FORMAT",
    "PACKED_INT4_I32_F16_FORMAT",
    "PACKED_VALUES",
    "SOURCE_FORMAT_CATALOG",
    "build_source_format_catalog",
]
