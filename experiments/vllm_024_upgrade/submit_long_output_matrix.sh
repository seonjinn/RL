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
RUN_TAG="${RUN_TAG:-vllm024-long-output-20260711}"
ATTEMPT_ID="${ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
BASE_EXPERIMENT_ROOT="${BASE_EXPERIMENT_ROOT:-${REPO_DIR}/experiments/vllm_024_upgrade/runs/${RUN_TAG}}"
WANDB_PROJECT="${WANDB_PROJECT:-nemorl-vllm024-long-output-lyris}"
MAX_STEPS="${MAX_STEPS:-5}"
OUTPUT_LENGTH_SELECTION="${OUTPUT_LENGTH_SELECTION:-all}"

qwen30_draft="${QWEN30_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf}"
qwen32_draft="${QWEN32_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-Thinking-speculator.eagle3/snapshots/a1403e07b73a66fc9ef561463631c31864616933}"

models=(qwen30ba3b qwen32b)
case "${OUTPUT_LENGTH_SELECTION}" in
  all) output_lengths=(16k 32k) ;;
  16k|32k) output_lengths=("${OUTPUT_LENGTH_SELECTION}") ;;
  *)
    echo "ERROR: OUTPUT_LENGTH_SELECTION must be all, 16k, or 32k" >&2
    exit 2
    ;;
esac
variants=(baseline eagle3_k3)

for output_length in "${output_lengths[@]}"; do
  case "${output_length}" in
    16k)
      max_new_tokens=16384
      max_total_sequence_length=20480
      specdec_context_headroom_tokens=8
      logprob_batch_size=""
      ;;
    32k)
      max_new_tokens=32768
      max_total_sequence_length=40960
      specdec_context_headroom_tokens=0
      logprob_batch_size=1
      ;;
  esac

  for model in "${models[@]}"; do
    for variant in "${variants[@]}"; do
      if [[ "${variant}" == "baseline" ]]; then
        capture_sizes='[1,2,4,8,16,32,64]'
        capture_max=64
      else
        capture_sizes='[4,8,16,32,64,128,256]'
        capture_max=256
      fi

      printf '[LONG-OUTPUT] model=%s osl=%s variant=%s max_total_sequence_length=%s\n' \
        "${model}" "${output_length}" "${variant}" \
        "${max_total_sequence_length}"

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
        RUN_TAG="${RUN_TAG}-${output_length}" \
        ATTEMPT_ID="${ATTEMPT_ID}" \
        EXPERIMENT_ROOT="${BASE_EXPERIMENT_ROOT}/${output_length}" \
        MAX_STEPS="${MAX_STEPS}" \
        NUM_PROMPTS_PER_STEP=16 \
        NUM_GENERATIONS_PER_PROMPT=16 \
        TRAIN_GLOBAL_BATCH_SIZE=256 \
        LOGPROB_BATCH_SIZE="${logprob_batch_size}" \
        MAX_TOTAL_SEQUENCE_LENGTH="${max_total_sequence_length}" \
        MAX_NEW_TOKENS="${max_new_tokens}" \
        ACTIVATION_CHECKPOINTING=true \
        OUTPUT_MAX_MODEL_LEN="${max_total_sequence_length}" \
        SPECDEC_CONTEXT_HEADROOM_TOKENS="${specdec_context_headroom_tokens}" \
        TEMPERATURE=1.0 \
        TOP_P=1.0 \
        REJECTION_SAMPLE_METHOD=standard \
        DRAFT_SAMPLE_METHOD=probabilistic \
        REFIT_DIAGNOSTICS=true \
        CUDAGRAPH_DISPATCH_METRICS=true \
        MAX_NUM_SEQS=64 \
        MAX_CUDAGRAPH_CAPTURE_SIZE="${capture_max}" \
        CUDAGRAPH_CAPTURE_SIZES="${capture_sizes}" \
        QWEN30_DRAFT_MODEL="${qwen30_draft}" \
        QWEN32_DRAFT_MODEL="${qwen32_draft}" \
        NCCL_NVLS_ENABLE=0 \
        bash "${LAUNCHER}" "${MODE}" "${model}" "${variant}"
    done
  done
done
