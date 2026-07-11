#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

MODE="${1:-test-only}"
case "${MODE}" in
  dry-run|test-only|submit) ;;
  *)
    echo "ERROR: mode must be dry-run, test-only, or submit" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/submit_eagle3_dynamicsd_step20.sh"
REPO_DIR="${REPO_DIR:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
LYRIS_ROOT="${LYRIS_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER="${CONTAINER:-${LYRIS_ROOT}/containers/nemo_rl_nightly_20260707.sqsh}"
HF_HOME="${HF_HOME:-${LYRIS_ROOT}/hf_home}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${LYRIS_ROOT}/.secrets/wandb_api_key}"
RUN_TAG="${RUN_TAG:-vllm024-cg-top-p-refit-profile-20260711}"
ATTEMPT_ID="${ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
BASE_EXPERIMENT_ROOT="${BASE_EXPERIMENT_ROOT:-${REPO_DIR}/experiments/vllm_024_upgrade/runs/${RUN_TAG}}"
WANDB_PROJECT="${WANDB_PROJECT:-nemorl-vllm024-cg-top-p-refit-profile-lyris}"
MAX_STEPS="${MAX_STEPS:-5}"

models=(qwen30ba3b qwen32b qwen235b)
top_p_values=(1.0 0.7)
variants=(baseline eagle3_k1)

for model in "${models[@]}"; do
  case "${model}" in
    qwen30ba3b)
      max_num_seqs=128
      baseline_capture_sizes='[1,2,4,8,16,32,48,64,96,128]'
      baseline_capture_max=128
      eagle_capture_sizes='[2,4,8,16,32,64,96,128,192,256]'
      eagle_capture_max=256
      ;;
    qwen32b)
      max_num_seqs=256
      baseline_capture_sizes='[1,2,4,8,16,32,48,64,96,128,192,256]'
      baseline_capture_max=256
      eagle_capture_sizes='[2,4,8,16,32,64,96,128,192,256,384,512]'
      eagle_capture_max=512
      ;;
    qwen235b)
      max_num_seqs=64
      baseline_capture_sizes='[1,2,4,8,16,32,64]'
      baseline_capture_max=64
      eagle_capture_sizes='[2,4,8,16,32,64,128]'
      eagle_capture_max=128
      ;;
  esac

  for top_p in "${top_p_values[@]}"; do
    case "${top_p}" in
      1.0) top_p_label=top_p10 ;;
      0.7) top_p_label=top_p07 ;;
    esac

    for variant in "${variants[@]}"; do
      if [[ "${variant}" == "baseline" ]]; then
        capture_sizes="${baseline_capture_sizes}"
        capture_max="${baseline_capture_max}"
      else
        capture_sizes="${eagle_capture_sizes}"
        capture_max="${eagle_capture_max}"
      fi

      printf '[PROFILE-MATRIX] model=%s top_p=%s variant=%s max_cudagraph_capture_size=%s\n' \
        "${model}" "${top_p_label}" "${variant}" "${capture_max}"

      env \
        REPO_DIR="${REPO_DIR}" \
        ACCOUNT=coreai_dlalgo_llm \
        PARTITION=gb200 \
        USE_GRES=false \
        GPUS_PER_NODE=4 \
        WALLTIME=04:00:00 \
        CONTAINER="${CONTAINER}" \
        HF_HOME="${HF_HOME}" \
        WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE}" \
        WANDB_PROJECT="${WANDB_PROJECT}" \
        RUN_TAG="${RUN_TAG}-${top_p_label}" \
        ATTEMPT_ID="${ATTEMPT_ID}" \
        EXPERIMENT_ROOT="${BASE_EXPERIMENT_ROOT}/${top_p_label}" \
        MAX_STEPS="${MAX_STEPS}" \
        TEMPERATURE=1.0 \
        TOP_P="${top_p}" \
        STATIC_K=1 \
        REJECTION_SAMPLE_METHOD=standard \
        DRAFT_SAMPLE_METHOD=probabilistic \
        REFIT_DIAGNOSTICS=true \
        MAX_NUM_SEQS="${max_num_seqs}" \
        MAX_CUDAGRAPH_CAPTURE_SIZE="${capture_max}" \
        CUDAGRAPH_CAPTURE_SIZES="${capture_sizes}" \
        NCCL_NVLS_ENABLE=0 \
        bash "${LAUNCHER}" "${MODE}" "${model}" "${variant}"
    done
  done
done
