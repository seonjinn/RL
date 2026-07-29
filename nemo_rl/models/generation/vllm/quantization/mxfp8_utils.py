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

from typing import Any


def mxfp8_refit_scale_name(weight_name: str, module: Any) -> str:
    quant_methods = (
        getattr(module, "quant_method", None),
        getattr(module, "base_quant_method", None),
    )
    preserves_checkpoint_scale = any(
        getattr(method, "preserves_checkpoint_weight_scale_for_refit", False)
        or getattr(
            getattr(method, "kernel", None),
            "preserves_checkpoint_weight_scale_for_refit",
            False,
        )
        for method in quant_methods
        if method is not None
    )
    suffix = "_scale" if preserves_checkpoint_scale else "_scale_from_checkpoint"
    return weight_name + suffix


def flashinfer_scale_k_pad_width(k: int) -> int:
    if k < 0:
        raise ValueError("MXFP8 scale K dimension must be non-negative")
    return (-k) % 4


def pad_flashinfer_scale_k(input_tensor: Any) -> Any:
    pad_width = flashinfer_scale_k_pad_width(input_tensor.shape[-1])
    if pad_width == 0:
        return input_tensor

    padded_shape = (*input_tensor.shape[:-1], input_tensor.shape[-1] + pad_width)
    padded = input_tensor.new_zeros(padded_shape)
    padded[..., : input_tensor.shape[-1]] = input_tensor
    return padded
