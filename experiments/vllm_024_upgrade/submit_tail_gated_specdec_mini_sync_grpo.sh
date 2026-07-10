#!/usr/bin/env bash
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

if (( $# > 1 )); then
  echo "ERROR: mini launcher accepts only an optional mode" >&2
  exit 2
fi

MODE="${1:-test-only}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MAX_STEPS="${MAX_STEPS:-2}"
export NUM_PROMPTS="${NUM_PROMPTS:-16}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export TRAIN_GBS="${TRAIN_GBS:-64}"
export MAX_OSL="${MAX_OSL:-1024}"
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-1024}"
export SPECDEC_CONTEXT_HEADROOM_TOKENS="${SPECDEC_CONTEXT_HEADROOM_TOKENS:-32}"
export CLUSTER_GPUS_PER_NODE="${CLUSTER_GPUS_PER_NODE:-4}"
export CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES:-4}"
export ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
export PARTITION="${PARTITION:-batch}"
export CLUSTER_NAME="${CLUSTER_NAME:-pre-tyche}"
export LYRIS_ROOT="${LYRIS_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
export CONTAINER="${CONTAINER:-${LYRIS_ROOT}/containers/nemo_rl_nightly_20260705.sqsh}"
export RUNTIME_VERSION="${RUNTIME_VERSION:-nightly-20260705}"
export WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
export WANDB_PROJECT="${WANDB_PROJECT:-nemorl-vllm024-tail-gated-mini-sync-grpo-pre-tyche}"
export RUN_TAG="${RUN_TAG:-vllm024-mini-sync-grpo-20260710}"
export ATTEMPT_ID="${ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export TAIL_GATE_THRESHOLD="${TAIL_GATE_THRESHOLD:-32}"
export TAIL_GATE_CONSECUTIVE_CHECKS="${TAIL_GATE_CONSECUTIVE_CHECKS:-10}"
export DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"

for variant in baseline_v2 always_on_v2_k5 fastrl_threshold_v2_k5; do
  bash "${SCRIPT_DIR}/submit_tail_gated_specdec_step20.sh" \
    "${MODE}" qwen32b "${variant}"
done
