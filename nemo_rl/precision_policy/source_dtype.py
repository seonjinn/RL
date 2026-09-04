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

"""Closed source-dtype normalization at discovery boundaries."""

from enum import StrEnum


class CanonicalSourceDType(StrEnum):
    """The supported scalar dtypes of discovered source tensors."""

    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    E4M3 = "e4m3"
    E8M0 = "e8m0"
    UINT8 = "uint8"
    INT32 = "int32"
    INT64 = "int64"


def normalize_safetensors_dtype(dtype: object) -> CanonicalSourceDType:
    """Normalize one exact safetensors metadata dtype token."""
    if type(dtype) is not str:
        raise TypeError("safetensors dtype must be a string")
    match dtype:
        case "BF16":
            return CanonicalSourceDType.BFLOAT16
        case "F16":
            return CanonicalSourceDType.FLOAT16
        case "F32":
            return CanonicalSourceDType.FLOAT32
        case "F8_E4M3":
            return CanonicalSourceDType.E4M3
        case "F8_E8M0":
            return CanonicalSourceDType.E8M0
        case "U8":
            return CanonicalSourceDType.UINT8
        case "I32":
            return CanonicalSourceDType.INT32
        case "I64":
            return CanonicalSourceDType.INT64
        case _:
            raise ValueError(f"unsupported safetensors dtype: {dtype!r}")


def normalize_torch_dtype(dtype: object) -> CanonicalSourceDType:
    """Normalize one supported Torch dtype singleton without eager Torch imports."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Torch is required to normalize a Torch dtype") from error

    candidates: list[tuple[object, CanonicalSourceDType]] = [
        (torch.bfloat16, CanonicalSourceDType.BFLOAT16),
        (torch.float16, CanonicalSourceDType.FLOAT16),
        (torch.float32, CanonicalSourceDType.FLOAT32),
        (torch.uint8, CanonicalSourceDType.UINT8),
        (torch.int32, CanonicalSourceDType.INT32),
        (torch.int64, CanonicalSourceDType.INT64),
    ]
    float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    if float8_e4m3fn is not None:
        candidates.append((float8_e4m3fn, CanonicalSourceDType.E4M3))
    float8_e8m0fnu = getattr(torch, "float8_e8m0fnu", None)
    if float8_e8m0fnu is not None:
        candidates.append((float8_e8m0fnu, CanonicalSourceDType.E8M0))
    for candidate, canonical_dtype in candidates:
        if dtype is candidate:
            return canonical_dtype
    if isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    raise TypeError("torch dtype must be a supported Torch dtype singleton")
