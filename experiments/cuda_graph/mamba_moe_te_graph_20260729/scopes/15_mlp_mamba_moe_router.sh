#!/usr/bin/env bash
set -euo pipefail
CUDA_GRAPH_IMPL=transformer_engine SCOPE='[mlp,mamba,moe_router]' SCOPE_NAME=mlp-mamba-moe-router WARMUP_STEPS=3 CACHE_CAPACITY=2 MAX_PACKED_SEQS=16 CHECKPOINTING_ENABLED=false WANDB_PROJECT=sna-cg-study RUN_NAME=mamba-moe-te-mlp-mamba-moe-router \
bash "$(dirname "${BASH_SOURCE[0]}")/../run_scope.sh"
