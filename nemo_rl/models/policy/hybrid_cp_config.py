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

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class HybridCPConfig:
    """Configuration for hybrid or dynamic context parallel scheduling."""

    enabled: bool = False
    max_seqlen_per_dp_cp_rank: Optional[int] = None
    scheduling_strategy: Literal["dp", "pp"] = "dp"
    balance_slack: float = 0.05
    eps_bucket: float = 0.10
    force_full_cp: bool = False

    def __post_init__(self) -> None:
        if not self.enabled:
            return

        if self.scheduling_strategy not in {"dp", "pp"}:
            raise ValueError(
                f"scheduling_strategy must be 'dp' or 'pp', got {self.scheduling_strategy}"
            )
        if self.scheduling_strategy == "pp":
            raise NotImplementedError(
                "Pipeline parallel strategy is not yet supported for hybrid CP"
            )
        if not 0 <= self.balance_slack <= 1:
            raise ValueError(
                f"balance_slack must be between 0 and 1, got {self.balance_slack}"
            )
        if not 0 <= self.eps_bucket <= 1:
            raise ValueError(
                f"eps_bucket must be between 0 and 1, got {self.eps_bucket}"
            )
