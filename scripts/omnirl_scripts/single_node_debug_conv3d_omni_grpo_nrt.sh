#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ -f "${NEMORL}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${NEMORL}/.env"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-examples/configs/vlmConv3d_grpo_mix_omnirlSDG-videorlSDG-videor1Comm-2minVidFilter-imageCommRB5-aud_nomni_32f_dedup_draco_super.yaml}"
JOB_NAME="${JOB_NAME:-grpo_omni_videor1_test_1node_super}"
JOB_HASH="${JOB_HASH:-$(printf '%s' "${JOB_NAME}" | openssl dgst -sha1 -binary | od -An -tx1 | tr -d ' \n' | cut -c1-12)}"
MODEL_NAME="${OMNI_GRPO_DEBUG_MODEL_NAME:-${OMNI_GRPO_MODEL_NAME:-${MODEL_NAME:-}}}"
TRAIN_DATA_PATH="${OMNI_GRPO_DEBUG_TRAIN_DATA_PATH:-${OMNI_GRPO_TRAIN_DATA_PATH:-${TRAIN_DATA_PATH:-}}}"
: "${MODEL_NAME:?Set OMNI_GRPO_DEBUG_MODEL_NAME, OMNI_GRPO_MODEL_NAME, or MODEL_NAME}"
: "${TRAIN_DATA_PATH:?Set OMNI_GRPO_DEBUG_TRAIN_DATA_PATH, OMNI_GRPO_TRAIN_DATA_PATH, or TRAIN_DATA_PATH}"

RESULTS_ROOT="${RESULTS_ROOT:-${NEMORL}/../debug_results}"
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/${JOB_NAME}}"
mkdir -p "${RESULTS_DIR}"

export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-true}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NRL_DEBUG="${NRL_DEBUG:-1}"
export CACHE_ROOT="${CACHE_ROOT:-${NEMORL}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${JOB_HASH}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
mkdir -p "${HF_HOME}" "${HF_MODULES_CACHE}" "${NRL_MEGATRON_CHECKPOINT_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

PYTHONPATH_ROOTS="${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM"
if [[ "${USE_REPO_VLLM:-0}" == "1" ]]; then
  PYTHONPATH_ROOTS="${NEMORL}/3rdparty/vllm:${PYTHONPATH_ROOTS}"
fi
export PYTHONPATH="${PYTHONPATH_ROOTS}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${NEMORL}"

uv run --no-sync examples/run_vlm_grpo.py \
  --config "${CONFIG_PATH}" \
  cluster.num_nodes=1 \
  cluster.gpus_per_node="${GPUS_PER_NODE:-8}" \
  policy.model_name="${MODEL_NAME}" \
  policy.megatron_cfg.expert_model_parallel_size="${DEBUG_EP_SIZE:-2}" \
  policy.megatron_cfg.tensor_model_parallel_size="${DEBUG_TP_SIZE:-4}" \
  policy.megatron_cfg.context_parallel_size=1 \
  grpo.num_prompts_per_step="${DEBUG_NUM_PROMPTS_PER_STEP:-2}" \
  grpo.num_generations_per_prompt="${DEBUG_NUM_GENERATIONS_PER_PROMPT:-8}" \
  policy.train_global_batch_size="${DEBUG_TRAIN_GLOBAL_BATCH_SIZE:-16}" \
  policy.generation.vllm_cfg.tensor_parallel_size="${DEBUG_VLLM_TP_SIZE:-2}" \
  data.default.num_frames="${DEBUG_NUM_FRAMES:-4}" \
  checkpointing.checkpoint_dir="${RESULTS_DIR}/checkpoints" \
  logger.log_dir="${RESULTS_DIR}/logs" \
  logger.wandb_enabled="${WANDB_ENABLED:-true}" \
  logger.wandb.name="${JOB_NAME}" \
  logger.wandb.project="${WANDB_PROJECT:-Nemotron-omni-RL}" \
  policy.max_total_sequence_length="${DEBUG_MAX_TOTAL_SEQUENCE_LENGTH:-10000}" \
  policy.sequence_packing.enabled=false \
  loss_fn.reference_policy_kl_penalty="${DEBUG_REFERENCE_KL:-0.1}" \
  data.train.train_data_path="${TRAIN_DATA_PATH}"
