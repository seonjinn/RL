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

"""Framework-independent runtime source metadata producer contract."""

from collections.abc import Sequence
from typing import Protocol

from nemo_rl.precision_policy.source_discovery import (
    DiscoveryContribution,
    ExpectedContributorSet,
    RuntimeGraphSourceRequest,
    SourceProducerFingerprint,
    SourceSchemaId,
)


class SourceMetadataProducer(Protocol):
    """Produce complete metadata contributions for an explicitly bound graph."""

    @property
    def producer_id(self) -> str:
        """Return the implementation identity committed by ``fingerprint``."""
        ...

    @property
    def schema_id(self) -> SourceSchemaId:
        """Return the source metadata schema committed by ``fingerprint``."""
        ...

    def fingerprint(self) -> SourceProducerFingerprint:
        """Return the immutable implementation and normalization identity."""
        ...

    def discover_contributions(
        self,
        request: RuntimeGraphSourceRequest,
        trusted_expected_contributors: ExpectedContributorSet,
    ) -> Sequence[DiscoveryContribution]:
        """Discover one graph's contributions under resolver-owned authority."""
        ...


__all__ = ["SourceMetadataProducer"]
