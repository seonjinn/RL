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

"""Fused checkpoint-scale padding and swizzle for MXFP8 refit."""

from vllm.triton_utils import tl, triton


@triton.jit
def swizzle_kernel(
    source,
    target,
    n,
    scale_cols,
    stride_n,
    stride_k,
    K_TILES: tl.constexpr,
    ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Output axes are [N//128, K_tiles, 32, 4, 4].
    k_inner = offset % 4
    n_group = (offset // 4) % 4
    n_inner = (offset // 16) % 32
    k_tile = (offset // 512) % K_TILES
    n_tile = offset // (512 * K_TILES)
    row = n_tile * 128 + n_group * 32 + n_inner
    col = k_tile * 4 + k_inner
    value = tl.load(
        source + row * stride_n + col * stride_k,
        mask=(offset < ELEMENTS) & (row < n) & (col < scale_cols),
        other=0,
    )
    tl.store(target + offset, value, mask=offset < ELEMENTS)
