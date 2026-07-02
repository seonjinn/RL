#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export REMOTE_HOST="${REMOTE_HOST:-login-ptyche}"
export REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-pr2879-20260702}"
export RUN_ID="${RUN_ID:-20260702_pretyche_q235_sync_exact_recipe_asyncengine_eagle3_step20_r2}"
export RUN_ROOT="${RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${RUN_ID}}"
export RUN_TAG="${RUN_TAG:-20260702}"
export MODE_LABEL="${MODE_LABEL:-sync-exact-asyncengine}"
export WANDB_PROJECT="${WANDB_PROJECT:-sna-nemorl-specdec-pretyche}"
export MEGATRON_CHECKPOINT_DIR="${MEGATRON_CHECKPOINT_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/megatron_checkpoints/1271b1530/pretyche_q235_pr2879_sync_smoke1}"
export ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
export PARTITION="${PARTITION:-batch}"
export WALLTIME="${WALLTIME:-05:00:00}"
export VARIANTS="${VARIANTS:-baseline,eagle3_k5,eagle3_k7,eagle3_k9,eagle3_k11}"
export OUT="${OUT:-${ROOT_DIR}/docs/latest_pretyche_qwen235b_sync_exact_recipe_eagle3_step20_20260702_jobs.csv}"

exec "${SCRIPT_DIR}/submit_lyris_qwen235b_sync_eagle3_k7_k9_k11_recipe_step20_20260701.sh"
