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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# tools/launch parses this block statically; the sourced runner owns runtime config.
: <<'LAUNCH_CONFIG'
# =+ BEGIN CONFIG =+
# ===== BEGIN CONFIG =====
NUM_NODES=16
GPUS_PER_NODE=4
SEGMENT_SIZE=16
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=100
# ===== END CONFIG =====
# =+ END CONFIG =+
LAUNCH_CONFIG

source "${SCRIPT_DIR}/grpo-qwen3-235b-16n4g.sh"
