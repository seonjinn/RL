#!/usr/bin/env bash
set -euo pipefail

CUDA_GRAPH_IMPL=none \
SCOPE='[no_cg]' \
SCOPE_NAME=drop-pad-baseline-no-cg \
WARMUP_STEPS=3 \
CACHE_CAPACITY=2 \
MAX_PACKED_SEQS=16 \
CHECKPOINTING_ENABLED=false \
WANDB_PROJECT=sna-cg-study \
RUN_NAME=mamba-moe-te-drop-pad-baseline-no-cg \
DROP_PAD_MOE_PAIR=true \
MOE_EXPERT_CAPACITY_FACTOR=1.0 \
MOE_PAD_EXPERT_INPUT_TO_CAPACITY=true \
bash "$(dirname "${BASH_SOURCE[0]}")/../run_scope.sh"
