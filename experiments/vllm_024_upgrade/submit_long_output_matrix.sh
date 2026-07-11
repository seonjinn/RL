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
MATRIX_SELECTION="${MATRIX_SELECTION:-standard}"
MODEL_SELECTION="${MODEL_SELECTION:-core}"
OUTPUT_LENGTH_SELECTION="${OUTPUT_LENGTH_SELECTION:-all}"
VARIANT_SELECTION="${VARIANT_SELECTION:-core}"

qwen30_draft="${QWEN30_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf}"
qwen32_draft="${QWEN32_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-Thinking-speculator.eagle3/snapshots/a1403e07b73a66fc9ef561463631c31864616933}"
qwen235_draft="${QWEN235_DRAFT_MODEL:-${HF_HOME}/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba}"

qwen30_base_target="${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
qwen30_instruct_target="${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
qwen30_thinking_target="${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
qwen30_base_draft="${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-speculator.eagle3/snapshots/6afc5aa2477b923467fb9a8d906782b984a9a6ba"
qwen30_instruct_draft="${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Instruct-2507-speculator.eagle3/snapshots/a7600ef6ca94c4e06cc1022879944be15949aee4"
qwen30_thinking_draft="${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf"

case "${MODEL_SELECTION}" in
  core) models=(qwen30ba3b qwen32b) ;;
  all) models=(qwen30ba3b qwen32b qwen235b) ;;
  qwen30ba3b|qwen32b|qwen235b) models=("${MODEL_SELECTION}") ;;
  *)
    echo "ERROR: MODEL_SELECTION must be core, all, qwen30ba3b, qwen32b, or qwen235b" >&2
    exit 2
    ;;
esac
case "${OUTPUT_LENGTH_SELECTION}" in
  all) output_lengths=(16k 32k) ;;
  16k|32k) output_lengths=("${OUTPUT_LENGTH_SELECTION}") ;;
  *)
    echo "ERROR: OUTPUT_LENGTH_SELECTION must be all, 16k, or 32k" >&2
    exit 2
    ;;
esac
case "${VARIANT_SELECTION}" in
  core) variants=(baseline eagle3_k3) ;;
  compare) variants=(baseline eagle3_k3 eagle3_k5) ;;
  k5-control) variants=(baseline eagle3_k5) ;;
  *)
    echo "ERROR: VARIANT_SELECTION must be core, compare, or k5-control" >&2
    exit 2
    ;;
esac

case "${MATRIX_SELECTION}" in
  standard)
    identities=(default)
    ;;
  qwen30-drafter)
    models=(qwen30ba3b)
    output_lengths=(16k)
    identities=(
      base__base
      base__instruct2507
      instruct2507__instruct2507
      thinking2507__thinking2507
    )
    printf '[DRAFTER-ALIAS] base=thinking2507 weight_sha256=%s config_blob=%s\n' \
      'd2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa' \
      '4e11c4dbb9b0bd911748a6f567d41f57c3dcdbe3'
    ;;
  *)
    echo "ERROR: MATRIX_SELECTION must be standard or qwen30-drafter" >&2
    exit 2
    ;;
esac

if [[ "${MATRIX_SELECTION}" == "qwen30-drafter" \
  && "${MODE}" == "submit" \
  && "${MAX_STEPS}" =~ ^[1-9][0-9]*$ \
  && -z "${DYNAMIC_SCHEDULE:-}" ]] \
  && ((MAX_STEPS >= 20)); then
  echo "ERROR: DYNAMIC_SCHEDULE is required for a 20-step qwen30-drafter submit" >&2
  exit 2
fi

for identity in "${identities[@]}"; do
  identity_target="${POLICY_MODEL_NAME:-}"
  identity_draft="${qwen30_draft}"
  identity_recipe="${QWEN30_RECIPE:-}"
  identity_nodes="${QWEN30_NODES:-}"
  identity_cudagraph_mode="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
  identity_uv_cache_dir="${UV_CACHE_DIR:-}"
  identity_uv_cache_seed_dir="${UV_CACHE_SEED_DIR:-}"
  identity_variants=("${variants[@]}")
  case "${identity}" in
    default)
      ;;
    base__base)
      identity_target="${qwen30_base_target}"
      identity_draft="${qwen30_base_draft}"
      identity_variants=(baseline eagle3_k5 dynamic)
      ;;
    base__instruct2507)
      identity_target="${qwen30_base_target}"
      identity_draft="${qwen30_instruct_draft}"
      identity_variants=(eagle3_k5 dynamic)
      ;;
    instruct2507__instruct2507)
      identity_target="${qwen30_instruct_target}"
      identity_draft="${qwen30_instruct_draft}"
      identity_variants=(baseline eagle3_k5 dynamic)
      ;;
    thinking2507__thinking2507)
      identity_target="${qwen30_thinking_target}"
      identity_draft="${qwen30_thinking_draft}"
      identity_variants=(baseline eagle3_k5 dynamic)
      ;;
  esac
  if [[ "${MATRIX_SELECTION}" == "qwen30-drafter" ]]; then
    identity_recipe="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g-40K.yaml"
    identity_nodes=8
    identity_cudagraph_mode=PIECEWISE
    identity_uv_cache_dir="${UV_CACHE_DIR:-${LYRIS_ROOT}/uv_cache/vllm024}"
    identity_uv_cache_seed_dir="${UV_CACHE_SEED_DIR:-}"
    if [[ "${MODE}" != "dry-run" && ! -d "${identity_target}" ]]; then
      echo "ERROR: target model directory not found: ${identity_target}" >&2
      exit 2
    fi
    if [[ "${MODE}" != "dry-run" && ! -d "${identity_draft}" ]]; then
      echo "ERROR: draft model directory not found: ${identity_draft}" >&2
      exit 2
    fi
  fi

  for output_length in "${output_lengths[@]}"; do
  case "${output_length}" in
    16k)
      max_new_tokens=16384
      max_total_sequence_length=20480
      specdec_context_headroom_tokens=8
      logprob_batch_size=""
      if [[ "${MATRIX_SELECTION}" == "qwen30-drafter" ]]; then
        logprob_batch_size=1
      fi
      ;;
    32k)
      max_new_tokens=32768
      max_total_sequence_length=40960
      specdec_context_headroom_tokens=0
      logprob_batch_size=1
      ;;
  esac

  for model in "${models[@]}"; do
    for variant in "${identity_variants[@]}"; do
      if [[ "${MATRIX_SELECTION}" == "qwen30-drafter" ]]; then
        capture_sizes='[1,2,4,8,16,32,64,128,256]'
        capture_max=256
      elif [[ "${variant}" == "baseline" ]]; then
        capture_sizes='[1,2,4,8,16,32,64]'
        capture_max=64
      else
        capture_sizes='[4,8,16,32,64,128,256]'
        capture_max=256
      fi

      printf '[LONG-OUTPUT] identity=%s model=%s osl=%s variant=%s max_total_sequence_length=%s\n' \
        "${identity}" "${model}" "${output_length}" "${variant}" \
        "${max_total_sequence_length}"

      if [[ "${identity}" == "default" ]]; then
        identity_run_tag="${RUN_TAG}-${output_length}"
        identity_root="${BASE_EXPERIMENT_ROOT}/${output_length}"
      else
        identity_run_tag="${RUN_TAG}-${identity}-${output_length}"
        identity_root="${BASE_EXPERIMENT_ROOT}/${output_length}/${identity}"
      fi

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
        RUN_TAG="${identity_run_tag}" \
        ATTEMPT_ID="${ATTEMPT_ID}" \
        EXPERIMENT_ROOT="${identity_root}" \
        MAX_STEPS="${MAX_STEPS}" \
        NUM_PROMPTS_PER_STEP=16 \
        NUM_GENERATIONS_PER_PROMPT=16 \
        TRAIN_GLOBAL_BATCH_SIZE=256 \
        LOGPROB_BATCH_SIZE="${logprob_batch_size}" \
        MAX_TOTAL_SEQUENCE_LENGTH="${max_total_sequence_length}" \
        MAX_NEW_TOKENS="${max_new_tokens}" \
        ACTIVATION_CHECKPOINTING=true \
        MAX_NUM_BATCHED_TOKENS=32768 \
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
        CUDAGRAPH_MODE="${identity_cudagraph_mode}" \
        UV_CACHE_DIR="${identity_uv_cache_dir}" \
        UV_CACHE_SEED_DIR="${identity_uv_cache_seed_dir}" \
        POLICY_MODEL_NAME="${identity_target}" \
        QWEN30_RECIPE="${identity_recipe}" \
        QWEN30_NODES="${identity_nodes}" \
        QWEN30_DRAFT_MODEL="${identity_draft}" \
        QWEN32_DRAFT_MODEL="${qwen32_draft}" \
        QWEN235_DRAFT_MODEL="${qwen235_draft}" \
        NCCL_NVLS_ENABLE=0 \
        bash "${LAUNCHER}" "${MODE}" "${model}" "${variant}"
    done
  done
done
done
