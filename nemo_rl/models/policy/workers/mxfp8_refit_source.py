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

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class NativeMXFP8Components:
    """Native MXFP8 weight and compact rowwise scale components.

    Attributes:
        weight: E4M3 values obtained by byte-preserving reinterpretation of the
            native uint8 weight storage.
        weight_scale: Compact E8M0 uint8 scales with shape ``[..., K / 32]``.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor


def extract_native_mxfp8_components(tensor: Any) -> NativeMXFP8Components:
    """Extract native MXFP8 components from a Transformer Engine tensor.

    Args:
        tensor: An object with ``shape`` and ``get_metadata()`` exposing
            ``rowwise_data``, ``rowwise_scale_inv``, and
            ``with_gemm_swizzled_scales``.

    Returns:
        Components whose weight is an E4M3 byte reinterpretation and whose
        scale is compact E8M0 storage with shape ``[..., K / 32]``.

    Raises:
        ValueError: If a native component is absent, malformed, has the wrong
            dtype, has insufficient storage, or uses swizzled scales.
    """
    metadata = tensor.get_metadata()
    if metadata.get("with_gemm_swizzled_scales"):
        raise ValueError("Native MXFP8 refit requires compact rowwise scales")
    shape = tuple(int(size) for size in tensor.shape)
    if not shape or shape[-1] % 32:
        raise ValueError(f"Native MXFP8 refit requires K divisible by 32; got {shape}")

    rows = math.prod(shape[:-1])
    columns = shape[-1] // 32
    data = metadata.get("rowwise_data")
    scale = metadata.get("rowwise_scale_inv")
    if not isinstance(data, torch.Tensor):
        raise ValueError(
            "Native MXFP8 refit requires rowwise_data to be a torch.Tensor; "
            f"got {type(data).__name__}"
        )
    if not isinstance(scale, torch.Tensor):
        raise ValueError(
            "Native MXFP8 refit requires rowwise_scale_inv to be a torch.Tensor; "
            f"got {type(scale).__name__}"
        )
    if data.dtype != torch.uint8:
        raise ValueError(
            "Native MXFP8 refit requires rowwise_data dtype torch.uint8; "
            f"got {data.dtype}"
        )
    if scale.dtype != torch.uint8:
        raise ValueError(
            "Native MXFP8 refit requires rowwise_scale_inv dtype torch.uint8; "
            f"got {scale.dtype}"
        )

    expected_data_size = rows * shape[-1]
    if data.numel() != expected_data_size:
        raise ValueError(
            "Native MXFP8 refit requires rowwise_data to contain exactly "
            f"{expected_data_size} bytes; got {data.numel()}"
        )
    if scale.ndim != 2 or scale.shape[0] < rows or scale.shape[1] < columns:
        raise ValueError(
            "Native MXFP8 refit requires rowwise_scale_inv to be a 2-D tensor "
            f"with at least logical shape ({rows}, {columns}); got {tuple(scale.shape)}"
        )

    weight = data.reshape(shape).view(torch.float8_e4m3fn)
    compact_scale = scale[:rows, :columns].reshape(*shape[:-1], columns)
    return NativeMXFP8Components(weight=weight, weight_scale=compact_scale)
