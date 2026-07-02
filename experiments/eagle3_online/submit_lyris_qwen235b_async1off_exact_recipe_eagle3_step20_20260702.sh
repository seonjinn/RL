#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export REMOTE_HOST="${REMOTE_HOST:-login-lyris}"
export REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701}"
export CONFIG="${CONFIG:-examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off.yaml}"
export RUN_ID="${RUN_ID:-20260702_lyris_q235_async1off_exact_recipe_no_sharp_eagle3_step20_r2}"
export RUN_ROOT="${RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${RUN_ID}}"
export RUN_TAG="${RUN_TAG:-20260702}"
export MODE_LABEL="${MODE_LABEL:-async1off-exact}"
export WANDB_PROJECT="${WANDB_PROJECT:-sna-nemorl-specdec-lyris}"
export MEGATRON_CHECKPOINT_DIR="${MEGATRON_CHECKPOINT_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260702_lyris_qwen235b_sync_baseline_recipe_step20_cudagraph_ncclretry_r5/megatron_checkpoints/qwen235b_sync_baseline}"
export ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
export PARTITION="${PARTITION:-gb200}"
export WALLTIME="${WALLTIME:-05:00:00}"
export VARIANTS="${VARIANTS:-baseline,eagle3_k7}"
export OUT="${OUT:-${ROOT_DIR}/docs/latest_lyris_qwen235b_async1off_exact_recipe_eagle3_step20_20260702_jobs.csv}"

exec "${SCRIPT_DIR}/submit_lyris_qwen235b_sync_eagle3_k7_k9_k11_recipe_step20_20260701.sh"
