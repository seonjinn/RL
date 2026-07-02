#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export REMOTE_HOST="${REMOTE_HOST:-login-lyris}"
export REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701}"
export CONFIG="${CONFIG:-examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g.yaml}"
export RUN_ID="${RUN_ID:-20260702_lyris_q235_sync_exact_recipe_pard_noarrms_smoke2_r2}"
export RUN_ROOT="${RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${RUN_ID}}"
export RUN_TAG="${RUN_TAG:-20260702}"
export MODE_LABEL="${MODE_LABEL:-sync-exact-pard-noarrms}"
export STEP_LABEL="${STEP_LABEL:-smoke2}"
export WANDB_PROJECT="${WANDB_PROJECT:-sna-nemorl-specdec-lyris}"
export MEGATRON_CHECKPOINT_DIR="${MEGATRON_CHECKPOINT_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260702_lyris_qwen235b_sync_baseline_recipe_step20_cudagraph_ncclretry_r5/megatron_checkpoints/qwen235b_sync_baseline}"
export PARD_DRAFT_MODEL="${PARD_DRAFT_MODEL:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/f9f650fbab180c26498817718f0db5cae8f25136}"
export PARD_DRAFT_TP="${PARD_DRAFT_TP:-8}"
export ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
export PARTITION="${PARTITION:-gb200}"
export WALLTIME="${WALLTIME:-02:00:00}"
export MAX_STEPS="${MAX_STEPS:-2}"
export VARIANTS="${VARIANTS:-baseline_noarrms,pard_k1}"
export OUT="${OUT:-${ROOT_DIR}/docs/latest_lyris_qwen235b_sync_exact_recipe_pard_noarrms_smoke2_20260702_jobs.csv}"

exec "${SCRIPT_DIR}/submit_lyris_qwen235b_sync_eagle3_k7_k9_k11_recipe_step20_20260701.sh"
