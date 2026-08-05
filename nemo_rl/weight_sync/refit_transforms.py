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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict, cast


REFIT_PLAN_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class RefitTransformRequest:
    """Requested storage-format conversion for logical parameters."""

    parameter_names: tuple[str, ...]
    source_format: str
    target_format: str


RefitTransformResponse = list[str] | list[RefitTransformRequest] | None


def merge_refit_transform_requests(
    responses: Iterable[RefitTransformResponse],
) -> list[RefitTransformRequest]:
    """Merge worker transform responses into deterministic typed requests.

    A list of parameter names is the legacy MXFP8 response. New workers return
    ``RefitTransformRequest`` instances so the requested target format survives
    both internal vLLM and outer Ray RPC boundaries.
    """
    names_by_format: dict[tuple[str, str], set[str]] = {}
    format_by_name: dict[str, tuple[str, str]] = {}
    for response in responses:
        if not response:
            continue
        for item in response:
            if isinstance(item, str):
                source_format = "bf16"
                target_format = "mxfp8_e4m3_e8m0"
                parameter_names = (item,)
            elif isinstance(item, RefitTransformRequest):
                source_format = item.source_format
                target_format = item.target_format
                parameter_names = item.parameter_names
            else:
                raise TypeError(
                    "Refit transform response entries must be parameter names or "
                    f"RefitTransformRequest instances, got {type(item).__name__}."
                )

            format_key = (source_format, target_format)
            requested_names = names_by_format.setdefault(format_key, set())
            for name in parameter_names:
                previous_format = format_by_name.setdefault(name, format_key)
                if previous_format != format_key:
                    raise ValueError(
                        f"Refit parameter {name!r} was requested with conflicting "
                        f"formats {previous_format!r} and {format_key!r}."
                    )
                requested_names.add(name)

    return [
        RefitTransformRequest(
            parameter_names=tuple(sorted(parameter_names)),
            source_format=source_format,
            target_format=target_format,
        )
        for (source_format, target_format), parameter_names in sorted(
            names_by_format.items()
        )
    ]


@dataclass(frozen=True)
class TransformComponentSpec:
    """One ordered tensor component carried by the refit transport."""

    role: str
    global_shape: tuple[int, ...]
    dtype_name: str


@dataclass(frozen=True)
class DestinationComponentSpec:
    """One checkpoint-layout component materialized after the transfer."""

    role: str
    global_shape: tuple[int, ...]
    dtype_name: str
    source: Literal["codec", "calibration"]


@dataclass(frozen=True, init=False)
class RefitTransformPlan:
    """Transport and destination contracts for one logical parameter.

    ``components`` remains a compatibility view of ``wire_components`` for
    identity and existing MXFP8 callers.
    """

    transform_id: str
    wire_components: tuple[TransformComponentSpec, ...]
    destination_components: tuple[DestinationComponentSpec, ...]
    completion_key: str
    finalize_scope: Literal["parameter", "model"]

    def __init__(
        self,
        *,
        transform_id: str,
        wire_components: tuple[TransformComponentSpec, ...] | None = None,
        destination_components: tuple[DestinationComponentSpec, ...] | None = None,
        completion_key: str = "",
        finalize_scope: Literal["parameter", "model"] = "parameter",
        components: tuple[TransformComponentSpec, ...] | None = None,
    ) -> None:
        """Create a plan, accepting legacy wire-only component construction."""
        if wire_components is None:
            if components is None:
                raise ValueError("RefitTransformPlan requires wire_components.")
            wire_components = components
        elif components is not None:
            raise ValueError(
                "RefitTransformPlan cannot receive both wire_components and components."
            )
        if not wire_components:
            raise ValueError("RefitTransformPlan requires at least one wire component.")
        if destination_components is None:
            destination_components = tuple(
                DestinationComponentSpec(
                    role=component.role,
                    global_shape=component.global_shape,
                    dtype_name=component.dtype_name,
                    source="codec",
                )
                for component in wire_components
            )
        if not destination_components:
            raise ValueError(
                "RefitTransformPlan requires at least one destination component."
            )
        object.__setattr__(self, "transform_id", transform_id)
        object.__setattr__(self, "wire_components", wire_components)
        object.__setattr__(self, "destination_components", destination_components)
        object.__setattr__(self, "completion_key", completion_key)
        object.__setattr__(self, "finalize_scope", finalize_scope)

    @property
    def components(self) -> tuple[TransformComponentSpec, ...]:
        """Return wire components for compatibility with legacy callers."""
        return self.wire_components


class RefitPlanAgreement(TypedDict):
    """Serializable agreement checked before a refit transfer starts."""

    protocol_version: int
    component_count: int
    plan_signature: str


class RefitTransformCodec(Protocol):
    """Describes transport and destination tensors for a target format."""

    transform_id: str

    def describe_outputs(
        self,
        global_shape: tuple[int, ...],
        input_dtype_name: str,
    ) -> tuple[TransformComponentSpec, ...]:
        """Return wire components in transfer order."""
        ...

    def describe_destination(
        self,
        global_shape: tuple[int, ...],
        input_dtype_name: str,
    ) -> tuple[DestinationComponentSpec, ...]:
        """Return destination checkpoint components in materialization order."""
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

    def describe_destination(
        self,
        global_shape: tuple[int, ...],
        input_dtype_name: str,
    ) -> tuple[DestinationComponentSpec, ...]:
        """Return destination components matching the legacy transfer layout."""
        return tuple(
            DestinationComponentSpec(
                role=component.role,
                global_shape=component.global_shape,
                dtype_name=component.dtype_name,
                source="codec",
            )
            for component in self.describe_outputs(global_shape, input_dtype_name)
        )


class _BF16ToNVFP4Codec:
    """Describe BF16 transport and receiver-owned NVFP4 checkpoint state."""

    def __init__(self, *, mode: Literal["w4a16", "w4a4"]) -> None:
        self._mode = mode
        self.transform_id = f"bf16_to_nvfp4_{mode}"

    def _validate_input(
        self, global_shape: tuple[int, ...], input_dtype_name: str
    ) -> None:
        if input_dtype_name != "torch.bfloat16":
            raise ValueError(
                "BF16-to-NVFP4 refit requires input dtype torch.bfloat16; "
                f"got {input_dtype_name}."
            )
        if len(global_shape) < 2 or global_shape[-1] % 16:
            raise ValueError(
                "BF16-to-NVFP4 refit requires K to be divisible by 16; "
                f"got global shape {global_shape}."
            )

    def describe_outputs(
        self,
        global_shape: tuple[int, ...],
        input_dtype_name: str,
    ) -> tuple[TransformComponentSpec, ...]:
        """Return the unmodified BF16 payload transferred to the receiver."""
        self._validate_input(global_shape, input_dtype_name)
        return (TransformComponentSpec("weight", global_shape, input_dtype_name),)

    def describe_destination(
        self,
        global_shape: tuple[int, ...],
        input_dtype_name: str,
    ) -> tuple[DestinationComponentSpec, ...]:
        """Return the NVFP4 family generated from the received BF16 tensor."""
        self._validate_input(global_shape, input_dtype_name)
        components = (
            DestinationComponentSpec(
                "weight",
                (*global_shape[:-1], global_shape[-1] // 2),
                "torch.uint8",
                "codec",
            ),
            DestinationComponentSpec(
                "weight_scale",
                (*global_shape[:-1], global_shape[-1] // 16),
                "torch.float8_e4m3fn",
                "codec",
            ),
            DestinationComponentSpec(
                "weight_scale_2", global_shape[:-2], "torch.float32", "codec"
            ),
        )
        if self._mode == "w4a4":
            return (
                *components,
                DestinationComponentSpec(
                    "input_scale",
                    global_shape[:-2],
                    "torch.float32",
                    "calibration",
                ),
            )
        return components


_TRANSFORM_CODECS: dict[tuple[str, str], RefitTransformCodec] = {
    ("bf16", "mxfp8_e4m3_e8m0"): _BF16ToMXFP8Codec(),
    ("bf16", "nvfp4_w4a16"): _BF16ToNVFP4Codec(mode="w4a16"),
    ("bf16", "nvfp4_w4a4"): _BF16ToNVFP4Codec(mode="w4a4"),
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
            "wire_components": [
                {
                    "role": component.role,
                    "global_shape": component.global_shape,
                    "dtype_name": component.dtype_name,
                }
                for component in plan.wire_components
            ],
            "destination_components": [
                {
                    "role": component.role,
                    "global_shape": component.global_shape,
                    "dtype_name": component.dtype_name,
                    "source": component.source,
                }
                for component in plan.destination_components
            ],
            "completion_key": plan.completion_key or parameter_name,
            "finalize_scope": plan.finalize_scope,
        }
        for parameter_name, plan in sorted(plans.items())
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_plan_agreement(
    plans: Mapping[str, RefitTransformPlan],
) -> RefitPlanAgreement:
    """Build the compact protocol agreement for a set of parameter plans."""
    return {
        "protocol_version": REFIT_PLAN_PROTOCOL_VERSION,
        "component_count": sum(len(plan.wire_components) for plan in plans.values()),
        "plan_signature": plan_signature(plans),
    }


def plans_from_serialized_metadata(
    refit_info: Mapping[str, Any],
) -> dict[str, RefitTransformPlan]:
    """Rebuild topology-independent plans from primitive refit metadata."""
    plans: dict[str, RefitTransformPlan] = {}
    for params in refit_info.get("per_layer_params", {}).values():
        for param in params:
            name = str(param["name"])
            if name in plans:
                raise ValueError(f"Duplicate refit parameter metadata for {name!r}.")
            serialized_wire_components = param.get("wire_components")
            if serialized_wire_components is None:
                serialized_wire_components = param["components"]
            wire_components = tuple(
                TransformComponentSpec(
                    role=str(component["role"]),
                    global_shape=tuple(int(size) for size in component["global_shape"]),
                    dtype_name=str(component["dtype"]),
                )
                for component in serialized_wire_components
            )
            if not wire_components:
                raise ValueError(f"Refit parameter {name!r} has no components.")
            destination_components = tuple(
                DestinationComponentSpec(
                    role=str(component["role"]),
                    global_shape=tuple(int(size) for size in component["global_shape"]),
                    dtype_name=str(component["dtype"]),
                    source=cast(Literal["codec", "calibration"], component["source"]),
                )
                for component in param.get("destination_components", [])
            )
            if any(
                component.source not in {"codec", "calibration"}
                for component in destination_components
            ):
                raise ValueError(
                    f"Refit parameter {name!r} has an invalid destination source."
                )
            if not destination_components:
                destination_components = tuple(
                    DestinationComponentSpec(
                        role=component.role,
                        global_shape=component.global_shape,
                        dtype_name=component.dtype_name,
                        source="codec",
                    )
                    for component in wire_components
                )
            finalize_scope = str(param["finalize_scope"])
            if finalize_scope not in {"parameter", "model"}:
                raise ValueError(
                    f"Refit parameter {name!r} has invalid finalize scope "
                    f"{finalize_scope!r}."
                )
            plans[name] = RefitTransformPlan(
                transform_id=str(param["transform_id"]),
                wire_components=wire_components,
                destination_components=destination_components,
                completion_key=str(param.get("completion_key", name)),
                finalize_scope=cast(Literal["parameter", "model"], finalize_scope),
            )
    return plans


def agreement_from_serialized_metadata(
    refit_info: Mapping[str, Any],
) -> RefitPlanAgreement:
    """Independently derive the canonical agreement from serialized metadata."""
    return build_plan_agreement(plans_from_serialized_metadata(refit_info))


def advertised_agreement_from_metadata(
    refit_info: Mapping[str, Any],
) -> RefitPlanAgreement:
    """Read the primitive agreement fields carried on the wire."""
    return {
        "protocol_version": int(refit_info["refit_protocol_version"]),
        "component_count": int(refit_info["refit_component_count"]),
        "plan_signature": str(refit_info["plan_signature"]),
    }


def validate_serialized_plan_agreement(
    refit_info: Mapping[str, Any],
) -> RefitPlanAgreement:
    """Require advertised fields to match the plan rebuilt from parameters."""
    advertised = advertised_agreement_from_metadata(refit_info)
    derived = agreement_from_serialized_metadata(refit_info)
    if advertised != derived:
        raise ValueError(
            "Serialized refit plan agreement does not match parameter metadata: "
            f"advertised={advertised}, derived={derived}."
        )
    return derived


def require_matching_agreements(
    agreements: list[RefitPlanAgreement], *, participants: str
) -> RefitPlanAgreement:
    """Collapse actor agreements, rejecting missing or divergent results."""
    if not agreements:
        raise ValueError(f"No refit plan agreement returned by {participants}.")
    first = agreements[0]
    if any(agreement != first for agreement in agreements[1:]):
        raise ValueError(f"Refit plan agreement mismatch among {participants}.")
    return first
