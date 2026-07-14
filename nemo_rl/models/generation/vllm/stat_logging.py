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

from collections.abc import Mapping
from typing import Any


def flush_cudagraph_metrics(llm: Any, llm_kwargs: Mapping[str, Any]) -> bool:
    """Flush vLLM's accumulated CUDA graph dispatch statistics when enabled."""
    if not llm_kwargs.get("cudagraph_metrics", False):
        return False

    logger_manager = getattr(llm, "logger_manager", None)
    if logger_manager is None:
        return False

    logger_manager.log()
    return True
