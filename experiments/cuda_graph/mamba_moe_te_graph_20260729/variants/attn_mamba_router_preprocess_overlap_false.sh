#!/usr/bin/env bash
set -euo pipefail

CUDA_GRAPH_IMPL=transformer_engine \
SCOPE='[attn,mamba,moe_router,moe_preprocess]' \
SCOPE_NAME=attn-mamba-moe-router-preprocess \
MOE_SHARED_EXPERT_OVERLAP=false \
MOE_ACT_RECOMPUTE=false \
WARMUP_STEPS=3 \
CACHE_CAPACITY=2 \
MAX_PACKED_SEQS=16 \
CHECKPOINTING_ENABLED=false \
WANDB_PROJECT=sna-cg-study \
RUN_NAME=mamba-moe-te-attn-mamba-router-preprocess-overlap-false \
bash "$(dirname "${BASH_SOURCE[0]}")/../run_scope.sh"
