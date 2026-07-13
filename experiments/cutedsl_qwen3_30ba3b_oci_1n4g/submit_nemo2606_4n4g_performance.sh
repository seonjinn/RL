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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly SCRIPT_DIR

export NEMO2606_FACTORIAL_CONTEXTS="${NEMO2606_FACTORIAL_CONTEXTS:-g0a0}"
export NEMO2606_FACTORIAL_RECIPE="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-megatron-mxfp8-cutedsl.yaml"
export NEMO2606_FACTORIAL_NUM_NODES=4
export NEMO2606_FACTORIAL_GPUS_PER_NODE=4
export NEMO2606_FACTORIAL_SEGMENT_SIZE=4
export NEMO2606_FACTORIAL_TRAIN_GLOBAL_BATCH_SIZE=2048
export NEMO2606_FACTORIAL_EXPERT_MODEL_PARALLEL_SIZE=16

exec bash "${SCRIPT_DIR}/submit_nemo2606_2n4g_factorial.sh" "$@"
