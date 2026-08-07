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
    weight: torch.Tensor
    weight_scale: torch.Tensor


def extract_native_mxfp8_components(tensor: Any) -> NativeMXFP8Components:
    metadata = tensor.get_metadata()
    if metadata.get("with_gemm_swizzled_scales"):
        raise ValueError("Native MXFP8 refit requires compact rowwise scales")
    shape = tuple(int(size) for size in tensor.shape)
    rows = math.prod(shape[:-1])
    if not shape or shape[-1] % 32:
        raise ValueError(f"Native MXFP8 refit requires K divisible by 32; got {shape}")
    data = metadata["rowwise_data"]
    scale = metadata["rowwise_scale_inv"]
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
    weight = data.reshape(shape).view(torch.float8_e4m3fn)
    compact_scale = scale[:rows, : shape[-1] // 32].reshape(
        *shape[:-1], shape[-1] // 32
    )
    return NativeMXFP8Components(weight=weight, weight_scale=compact_scale)
