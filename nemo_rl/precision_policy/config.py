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

"""Typed, backend-independent semantic precision policy schema."""

from typing import Literal, Self

from pydantic import BaseModel, Field, NonNegativeInt, model_validator

PrecisionName = Literal["bf16", "mxfp8"]
LayerIndexSpace = Literal["global_decoder", "moe_ordinal"]
AtomicConflictMode = Literal["error", "expand"]
SemanticAttributeScalar = str | int | float | bool
SemanticAttributePredicate = SemanticAttributeScalar | list[SemanticAttributeScalar]
SemanticStringPredicate = str | list[str]


def _reject_undocumented_model_extra(model: BaseModel) -> None:
    if model.model_extra:
        extras = ", ".join(sorted(model.model_extra))
        raise ValueError(f"Undocumented precision policy field(s): {extras}")


def _validate_non_empty_predicate(
    predicate: SemanticStringPredicate | SemanticAttributePredicate,
    *,
    field_name: str,
) -> None:
    if isinstance(predicate, list) and not predicate:
        raise ValueError(f"{field_name} must not be an empty predicate")
    if isinstance(predicate, str) and not predicate.strip():
        raise ValueError(f"{field_name} must not be empty")


class LayerSelectorConfig(BaseModel, extra="allow"):
    """Restrict a scope using a canonical semantic layer coordinate."""

    index_space: LayerIndexSpace = "global_decoder"
    exclude_first: NonNegativeInt = 0
    exclude_last: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        _reject_undocumented_model_extra(self)
        return self


class AdvancedMatchConfig(BaseModel, extra="allow"):
    """Structured semantic matcher for specialized precision experiments."""

    graph_instance_id: SemanticStringPredicate | None = None
    semantic_graph_path: SemanticStringPredicate | None = None
    model_part: SemanticStringPredicate | None = None
    module_kind: SemanticStringPredicate | None = None
    parameter_role: SemanticStringPredicate | None = None
    attributes: dict[str, SemanticAttributePredicate] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        _reject_undocumented_model_extra(self)
        for field_name, predicate in (
            ("graph_instance_id", self.graph_instance_id),
            ("semantic_graph_path", self.semantic_graph_path),
            ("model_part", self.model_part),
            ("module_kind", self.module_kind),
            ("parameter_role", self.parameter_role),
        ):
            if predicate is not None:
                _validate_non_empty_predicate(predicate, field_name=field_name)
        for attribute_name, predicate in self.attributes.items():
            if not attribute_name.strip():
                raise ValueError("attribute names must not be empty")
            _validate_non_empty_predicate(predicate, field_name=attribute_name)
        if (
            self.graph_instance_id is None
            and self.semantic_graph_path is None
            and self.model_part is None
            and self.module_kind is None
            and self.parameter_role is None
            and not self.attributes
        ):
            raise ValueError(
                "advanced_match must include at least one semantic predicate"
            )
        return self


class SemanticAddressSelectorConfig(BaseModel, extra="allow"):
    """One qualified, canonical semantic tensor address."""

    graph_instance_id: str
    semantic_graph_path: str
    semantic_id: str

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        _reject_undocumented_model_extra(self)
        for field_name, value in (
            ("graph_instance_id", self.graph_instance_id),
            ("semantic_graph_path", self.semantic_graph_path),
            ("semantic_id", self.semantic_id),
        ):
            if not value or value != value.strip():
                raise ValueError(f"{field_name} must be non-empty without whitespace")
        if self.graph_instance_id != "main" and not (
            self.graph_instance_id.startswith(("mtp.", "draft."))
            and self.graph_instance_id not in {"mtp.", "draft."}
        ):
            raise ValueError("graph_instance_id must be main, mtp.*, or draft.*")
        if not self.semantic_id.startswith(f"{self.semantic_graph_path}."):
            raise ValueError("semantic_id must be a descendant of semantic_graph_path")
        return self


class PrecisionScopeConfig(BaseModel, extra="allow"):
    """One positive semantic selection and its endpoint precision requests."""

    id: str
    role: str | None = None
    advanced_match: AdvancedMatchConfig | None = None
    addresses: list[SemanticAddressSelectorConfig] | None = None
    layers: LayerSelectorConfig = Field(default_factory=LayerSelectorConfig)
    training: PrecisionName | None = None
    rollout: PrecisionName | None = None
    atomic_conflict: AtomicConflictMode = "error"

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        _reject_undocumented_model_extra(self)
        if not self.id.strip():
            raise ValueError("scope id must be non-empty")
        selectors = (self.role, self.advanced_match, self.addresses)
        if sum(selector is not None for selector in selectors) != 1:
            raise ValueError(
                "exactly one of role, advanced_match, or addresses is required"
            )
        if self.role is not None and not self.role.strip():
            raise ValueError("scope role must be non-empty")
        if self.addresses is not None:
            if not self.addresses:
                raise ValueError("addresses must not be empty")
            identities = [
                (address.graph_instance_id, address.semantic_id)
                for address in self.addresses
            ]
            if len(identities) != len(set(identities)):
                raise ValueError("semantic addresses must be unique per graph instance")
        if self.training != "mxfp8" and self.rollout != "mxfp8":
            raise ValueError("scope must request mxfp8 for training or rollout")
        return self


class PrecisionPolicyConfig(BaseModel, extra="allow"):
    """Versioned positive selection policy for training and rollout precision."""

    schema_version: Literal[1] = 1
    default: Literal["bf16"] = "bf16"
    require_match: bool = True
    atomic_conflict: AtomicConflictMode = "error"
    scopes: list[PrecisionScopeConfig]

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        _reject_undocumented_model_extra(self)
        scope_ids = [scope.id for scope in self.scopes]
        if len(scope_ids) != len(set(scope_ids)):
            raise ValueError("precision policy scope IDs must be unique")
        return self


def parse_precision_policy(value: object) -> PrecisionPolicyConfig | None:
    """Parse an optional precision policy YAML value."""
    if value is None:
        return None
    return PrecisionPolicyConfig.model_validate(value)
