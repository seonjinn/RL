#!/usr/bin/env bash
set -euo pipefail
CUDA_GRAPH_IMPL=transformer_engine SCOPE='[moe_router,moe_preprocess]' SCOPE_NAME=moe-router-preprocess MOE_SHARED_EXPERT_OVERLAP=true MOE_ACT_RECOMPUTE=true WARMUP_STEPS=3 CACHE_CAPACITY=2 MAX_PACKED_SEQS=16 CHECKPOINTING_ENABLED=false WANDB_PROJECT=sna-cg-study RUN_NAME=mamba-moe-te-router-preprocess-overlap-true-moe-act-true \
bash "$(dirname "${BASH_SOURCE[0]}")/../run_scope.sh"
