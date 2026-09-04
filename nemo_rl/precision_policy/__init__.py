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

from nemo_rl.precision_policy.config import (
    AdvancedMatchConfig,
    LayerSelectorConfig,
    PrecisionName,
    PrecisionPolicyConfig,
    PrecisionScopeConfig,
    SemanticAddressSelectorConfig,
    parse_precision_policy,
)
from nemo_rl.precision_policy.source_dtype import (
    CanonicalSourceDType,
    normalize_safetensors_dtype,
    normalize_torch_dtype,
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
    "LayerSelectorConfig",
    "MEGATRON_BRIDGE_STATE_DICT_V1",
    "NEMO_AUTOMODEL_STATE_DICT_V1",
    "PrecisionName",
    "PrecisionPolicyConfig",
    "PrecisionScopeConfig",
    "SemanticAddressSelectorConfig",
    "SourceDiscoveryInventory",
    "SourceDiscoveryRecord",
    "SourceProducerFingerprint",
    "SourceRecordProvenance",
    "SourceSchemaId",
    "TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1",
    "assemble_graph_discovery_partition",
    "graph_input_identity_digest",
    "parse_precision_policy",
    "normalize_safetensors_dtype",
    "normalize_torch_dtype",
    "validate_discovery_inventory",
]
