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

"""Topology-independent component contracts for transformed refit payloads."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class RefitTransformRequest:
    """Requested storage-format conversion for logical parameters."""

    parameter_names: tuple[str, ...]
    source_format: str
    target_format: str


@dataclass(frozen=True)
class TransformComponentSpec:
    """One ordered tensor component produced by a refit transform."""

    role: str
    global_shape: tuple[int, ...]
    dtype_name: str


@dataclass(frozen=True)
class RefitTransformPlan:
    """Ordered components and completion scope for one logical parameter."""

    transform_id: str
    components: tuple[TransformComponentSpec, ...]
    finalize_scope: Literal["parameter", "layer", "model"]


class RefitTransformCodec(Protocol):
    """Describes the tensors a source adapter must produce for a target format."""

    transform_id: str

    def describe_outputs(
        self,
        global_shape: tuple[int, ...],
        input_dtype_name: str,
    ) -> tuple[TransformComponentSpec, ...]:
        """Return output components in transfer order."""
        ...


class _BF16ToMXFP8Codec:
    """Describe MXFP8 E4M3 values and E8M0 block scales from BF16 storage."""

    transform_id = "bf16_to_mxfp8_e4m3_e8m0"

    def describe_outputs(
        self,
        global_shape: tuple[int, ...],
        input_dtype_name: str,
    ) -> tuple[TransformComponentSpec, ...]:
        """Return MXFP8 value and block-scale specifications for one tensor."""
        if input_dtype_name != "torch.bfloat16":
            raise ValueError(
                "BF16-to-MXFP8 refit requires input dtype torch.bfloat16; "
                f"got {input_dtype_name}."
            )
        if not global_shape or global_shape[-1] % 32:
            raise ValueError(
                "BF16-to-MXFP8 refit requires K to be divisible by 32; "
                f"got global shape {global_shape}."
            )

        return (
            TransformComponentSpec("weight", global_shape, "torch.float8_e4m3fn"),
            TransformComponentSpec(
                "weight_scale",
                (*global_shape[:-1], global_shape[-1] // 32),
                "torch.uint8",
            ),
        )


_TRANSFORM_CODECS: dict[tuple[str, str], RefitTransformCodec] = {
    ("bf16", "mxfp8_e4m3_e8m0"): _BF16ToMXFP8Codec(),
}


def resolve_transform(source_format: str, target_format: str) -> RefitTransformCodec:
    """Return the codec for a supported source and target storage-format pair."""
    try:
        return _TRANSFORM_CODECS[(source_format, target_format)]
    except KeyError as error:
        raise ValueError(
            "No refit transform registered for source format "
            f"{source_format!r} and target format {target_format!r}."
        ) from error


def plan_signature(plans: Mapping[str, RefitTransformPlan]) -> str:
    """Return the canonical SHA-256 signature for parameter transform plans."""
    payload = [
        {
            "parameter_name": parameter_name,
            "transform_id": plan.transform_id,
            "components": [
                {
                    "role": component.role,
                    "global_shape": component.global_shape,
                    "dtype_name": component.dtype_name,
                }
                for component in plan.components
            ],
            "finalize_scope": plan.finalize_scope,
        }
        for parameter_name, plan in sorted(plans.items())
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
