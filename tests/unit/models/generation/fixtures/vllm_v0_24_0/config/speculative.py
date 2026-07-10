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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Exact excerpt from vllm/config/speculative.py at ee0da84ab9e04ac7610e28580af62c365e898389.
class SpeculativeConfig:
    # dynamic speculative decoding control
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None
    """Batch-size schedule used to dynamically choose speculative-token count.

    Each entry is ``(range_start, range_end, num_speculative_tokens)`` with an
    inclusive batch-size range.
    """

    # params generated in the post-init stage
