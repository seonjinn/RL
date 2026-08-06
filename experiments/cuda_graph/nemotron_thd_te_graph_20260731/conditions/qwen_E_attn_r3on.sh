#!/bin/bash
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

set -euo pipefail

case "${MODEL:-}" in
  qwen3_30ba3b|qwen3_235b) ;;
  *)
    echo "MODEL must be qwen3_30ba3b or qwen3_235b" >&2
    exit 2
    ;;
esac

export ROUTER_REPLAY=on
export QWEN_CAMPAIGN_ARM=E
bash "$(dirname "${BASH_SOURCE[0]}")/../scopes/17_attn.sh"
