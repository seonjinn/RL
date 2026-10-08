# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Write CuTeDSL refit outputs into existing CUDA Graph storage."""

import torch


def swizzle_into(source: torch.Tensor, target: torch.Tensor, *, n: int, k: int) -> None:
    """Write F8_128x4 scales, including padding, without intermediate tensors."""
    from nemo_rl.utils.mxfp8_scale_kernel import swizzle_kernel

    elements = ((n + 127) // 128) * ((k + 127) // 128) * 512
    swizzle_kernel[((elements + 255) // 256,)](
        source,
        target,
        n,
        k // 32,
        source.stride(0),
        source.stride(1),
        K_TILES=(k + 127) // 128,
        ELEMENTS=elements,
        BLOCK=256,
    )


def copy_into_runtime(
    weight: torch.Tensor,
    scale: torch.Tensor,
    runtime_weight: torch.Tensor,
    runtime_scale: torch.Tensor,
) -> None:
    """Preserve runtime storage while finalizing checkpoint-layout tensors.

    The caller must have completed all shard loading and stopped generation.
    Validate all layouts before either destination is modified. Any execution
    failure must propagate to the refit lifecycle's fatal-error handler.
    """
    if weight.ndim != 2:
        raise ValueError("Expected a two-dimensional checkpoint weight")
    n, k = weight.shape
    scale_elements = ((n + 127) // 128) * ((k + 127) // 128) * 512
    if (
        n <= 0
        or k <= 0
        or k % 32
        or weight.dtype != torch.float8_e4m3fn
        or runtime_weight.dtype != weight.dtype
        or runtime_weight.shape != (k, n)
        or runtime_weight.stride() != (1, k)
        or scale.ndim != 2
        or scale.shape[0] < n
        or scale.shape[1] < k // 32
        or scale.dtype != torch.uint8
        or runtime_scale.dtype != torch.uint8
        or runtime_scale.shape != (scale_elements,)
        or not runtime_scale.is_contiguous()
        or any(
            t.device != weight.device for t in (scale, runtime_weight, runtime_scale)
        )
        or scale.untyped_storage().data_ptr()
        == runtime_scale.untyped_storage().data_ptr()
    ):
        raise ValueError(
            "Unsupported checkpoint/runtime layout for direct CuTeDSL refit"
        )
    runtime_weight.copy_(weight.t())
    swizzle_into(scale, runtime_scale, n=n, k=k)
