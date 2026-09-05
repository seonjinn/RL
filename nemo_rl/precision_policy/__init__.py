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

"""Public schema for semantic precision policy configuration."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_rl.precision_policy.config import (
        AdvancedMatchConfig,
        LayerSelectorConfig,
        PrecisionName,
        PrecisionPolicyConfig,
        PrecisionScopeConfig,
        SemanticAddressSelectorConfig,
        parse_precision_policy,
    )
    from nemo_rl.precision_policy.source_discovery import (
        HF_SAFETENSORS_HEADER_V1,
        MEGATRON_BRIDGE_STATE_DICT_V1,
        NEMO_AUTOMODEL_STATE_DICT_V1,
        TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1,
        DiscoveryCompletenessReceipt,
        DiscoveryContribution,
        ExpectedContributorAuthority,
        ExpectedContributorSet,
        GraphDiscoveryPartition,
        GraphTopologyInput,
        SourceDiscoveryInventory,
        SourceDiscoveryRecord,
        SourceProducerFingerprint,
        SourceRecordProvenance,
        SourceSchemaId,
        assemble_graph_discovery_partition,
        graph_input_identity_digest,
        validate_discovery_inventory,
    )
    from nemo_rl.precision_policy.source_dtype import (
        CanonicalSourceDType,
        normalize_safetensors_dtype,
        normalize_torch_dtype,
    )
    from nemo_rl.precision_policy.source_storage import (
        IDENTITY_PERMUTATION_ID,
        IDENTITY_SWIZZLE_ID,
        SourceAxisExtent,
        SourceDerivedRealization,
        SourceExtentRounding,
        SourceLiteralAxisExtent,
        SourceNormalizationContract,
        SourceNormalizationKind,
        SourceNormalizedAxisExtent,
        SourceNormalizerManifest,
        SourcePaddingSemantics,
        SourcePhysicalAxisSpec,
        SourceRealization,
        SourceStorageComponent,
        SourceStorageRealization,
        SourceStorageRealizationInventory,
        source_normalizer_manifest_digest,
        source_realization_is_wire_eligible,
        source_realizations_have_exact_physical_representation,
        source_storage_inventory_digest,
        validate_source_storage_realization_inventory,
    )


_LAZY_EXPORTS = {
    "AdvancedMatchConfig": ("nemo_rl.precision_policy.config", "AdvancedMatchConfig"),
    "CanonicalSourceDType": (
        "nemo_rl.precision_policy.source_dtype",
        "CanonicalSourceDType",
    ),
    "DiscoveryCompletenessReceipt": (
        "nemo_rl.precision_policy.source_discovery",
        "DiscoveryCompletenessReceipt",
    ),
    "DiscoveryContribution": (
        "nemo_rl.precision_policy.source_discovery",
        "DiscoveryContribution",
    ),
    "ExpectedContributorAuthority": (
        "nemo_rl.precision_policy.source_discovery",
        "ExpectedContributorAuthority",
    ),
    "ExpectedContributorSet": (
        "nemo_rl.precision_policy.source_discovery",
        "ExpectedContributorSet",
    ),
    "GraphDiscoveryPartition": (
        "nemo_rl.precision_policy.source_discovery",
        "GraphDiscoveryPartition",
    ),
    "GraphTopologyInput": (
        "nemo_rl.precision_policy.source_discovery",
        "GraphTopologyInput",
    ),
    "HF_SAFETENSORS_HEADER_V1": (
        "nemo_rl.precision_policy.source_discovery",
        "HF_SAFETENSORS_HEADER_V1",
    ),
    "IDENTITY_PERMUTATION_ID": (
        "nemo_rl.precision_policy.source_storage",
        "IDENTITY_PERMUTATION_ID",
    ),
    "IDENTITY_SWIZZLE_ID": (
        "nemo_rl.precision_policy.source_storage",
        "IDENTITY_SWIZZLE_ID",
    ),
    "LayerSelectorConfig": (
        "nemo_rl.precision_policy.config",
        "LayerSelectorConfig",
    ),
    "MEGATRON_BRIDGE_STATE_DICT_V1": (
        "nemo_rl.precision_policy.source_discovery",
        "MEGATRON_BRIDGE_STATE_DICT_V1",
    ),
    "NEMO_AUTOMODEL_STATE_DICT_V1": (
        "nemo_rl.precision_policy.source_discovery",
        "NEMO_AUTOMODEL_STATE_DICT_V1",
    ),
    "PrecisionName": ("nemo_rl.precision_policy.config", "PrecisionName"),
    "PrecisionPolicyConfig": (
        "nemo_rl.precision_policy.config",
        "PrecisionPolicyConfig",
    ),
    "PrecisionScopeConfig": (
        "nemo_rl.precision_policy.config",
        "PrecisionScopeConfig",
    ),
    "SemanticAddressSelectorConfig": (
        "nemo_rl.precision_policy.config",
        "SemanticAddressSelectorConfig",
    ),
    "SourceDiscoveryInventory": (
        "nemo_rl.precision_policy.source_discovery",
        "SourceDiscoveryInventory",
    ),
    "SourceDiscoveryRecord": (
        "nemo_rl.precision_policy.source_discovery",
        "SourceDiscoveryRecord",
    ),
    "SourceAxisExtent": (
        "nemo_rl.precision_policy.source_storage",
        "SourceAxisExtent",
    ),
    "SourceDerivedRealization": (
        "nemo_rl.precision_policy.source_storage",
        "SourceDerivedRealization",
    ),
    "SourceExtentRounding": (
        "nemo_rl.precision_policy.source_storage",
        "SourceExtentRounding",
    ),
    "SourceLiteralAxisExtent": (
        "nemo_rl.precision_policy.source_storage",
        "SourceLiteralAxisExtent",
    ),
    "SourceNormalizationContract": (
        "nemo_rl.precision_policy.source_storage",
        "SourceNormalizationContract",
    ),
    "SourceNormalizationKind": (
        "nemo_rl.precision_policy.source_storage",
        "SourceNormalizationKind",
    ),
    "SourceNormalizedAxisExtent": (
        "nemo_rl.precision_policy.source_storage",
        "SourceNormalizedAxisExtent",
    ),
    "SourceNormalizerManifest": (
        "nemo_rl.precision_policy.source_storage",
        "SourceNormalizerManifest",
    ),
    "SourcePaddingSemantics": (
        "nemo_rl.precision_policy.source_storage",
        "SourcePaddingSemantics",
    ),
    "SourcePhysicalAxisSpec": (
        "nemo_rl.precision_policy.source_storage",
        "SourcePhysicalAxisSpec",
    ),
    "SourceProducerFingerprint": (
        "nemo_rl.precision_policy.source_discovery",
        "SourceProducerFingerprint",
    ),
    "SourceRealization": (
        "nemo_rl.precision_policy.source_storage",
        "SourceRealization",
    ),
    "SourceRecordProvenance": (
        "nemo_rl.precision_policy.source_discovery",
        "SourceRecordProvenance",
    ),
    "SourceSchemaId": (
        "nemo_rl.precision_policy.source_discovery",
        "SourceSchemaId",
    ),
    "SourceStorageComponent": (
        "nemo_rl.precision_policy.source_storage",
        "SourceStorageComponent",
    ),
    "SourceStorageRealization": (
        "nemo_rl.precision_policy.source_storage",
        "SourceStorageRealization",
    ),
    "SourceStorageRealizationInventory": (
        "nemo_rl.precision_policy.source_storage",
        "SourceStorageRealizationInventory",
    ),
    "TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1": (
        "nemo_rl.precision_policy.source_discovery",
        "TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1",
    ),
    "assemble_graph_discovery_partition": (
        "nemo_rl.precision_policy.source_discovery",
        "assemble_graph_discovery_partition",
    ),
    "graph_input_identity_digest": (
        "nemo_rl.precision_policy.source_discovery",
        "graph_input_identity_digest",
    ),
    "parse_precision_policy": (
        "nemo_rl.precision_policy.config",
        "parse_precision_policy",
    ),
    "normalize_safetensors_dtype": (
        "nemo_rl.precision_policy.source_dtype",
        "normalize_safetensors_dtype",
    ),
    "normalize_torch_dtype": (
        "nemo_rl.precision_policy.source_dtype",
        "normalize_torch_dtype",
    ),
    "source_normalizer_manifest_digest": (
        "nemo_rl.precision_policy.source_storage",
        "source_normalizer_manifest_digest",
    ),
    "source_realization_is_wire_eligible": (
        "nemo_rl.precision_policy.source_storage",
        "source_realization_is_wire_eligible",
    ),
    "source_realizations_have_exact_physical_representation": (
        "nemo_rl.precision_policy.source_storage",
        "source_realizations_have_exact_physical_representation",
    ),
    "source_storage_inventory_digest": (
        "nemo_rl.precision_policy.source_storage",
        "source_storage_inventory_digest",
    ),
    "validate_discovery_inventory": (
        "nemo_rl.precision_policy.source_discovery",
        "validate_discovery_inventory",
    ),
    "validate_source_storage_realization_inventory": (
        "nemo_rl.precision_policy.source_storage",
        "validate_source_storage_realization_inventory",
    ),
}

__all__ = [
    "AdvancedMatchConfig",
    "CanonicalSourceDType",
    "DiscoveryCompletenessReceipt",
    "DiscoveryContribution",
    "ExpectedContributorAuthority",
    "ExpectedContributorSet",
    "GraphDiscoveryPartition",
    "GraphTopologyInput",
    "HF_SAFETENSORS_HEADER_V1",
    "IDENTITY_PERMUTATION_ID",
    "IDENTITY_SWIZZLE_ID",
    "LayerSelectorConfig",
    "MEGATRON_BRIDGE_STATE_DICT_V1",
    "NEMO_AUTOMODEL_STATE_DICT_V1",
    "PrecisionName",
    "PrecisionPolicyConfig",
    "PrecisionScopeConfig",
    "SemanticAddressSelectorConfig",
    "SourceDiscoveryInventory",
    "SourceDiscoveryRecord",
    "SourceAxisExtent",
    "SourceDerivedRealization",
    "SourceExtentRounding",
    "SourceLiteralAxisExtent",
    "SourceNormalizationContract",
    "SourceNormalizationKind",
    "SourceNormalizedAxisExtent",
    "SourceNormalizerManifest",
    "SourcePaddingSemantics",
    "SourcePhysicalAxisSpec",
    "SourceProducerFingerprint",
    "SourceRealization",
    "SourceRecordProvenance",
    "SourceSchemaId",
    "SourceStorageComponent",
    "SourceStorageRealization",
    "SourceStorageRealizationInventory",
    "TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1",
    "assemble_graph_discovery_partition",
    "graph_input_identity_digest",
    "parse_precision_policy",
    "normalize_safetensors_dtype",
    "normalize_torch_dtype",
    "source_normalizer_manifest_digest",
    "source_realization_is_wire_eligible",
    "source_realizations_have_exact_physical_representation",
    "source_storage_inventory_digest",
    "validate_discovery_inventory",
    "validate_source_storage_realization_inventory",
]

if len(__all__) != len(set(__all__)) or set(__all__) != set(_LAZY_EXPORTS):
    raise RuntimeError("precision-policy lazy exports and __all__ must agree exactly")


def __getattr__(name: str) -> object:
    """Load public compatibility exports without coupling package import phases."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to introspection tools."""
    return sorted((*globals(), *__all__))
