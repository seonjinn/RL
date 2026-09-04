#!/bin/bash
set -euo pipefail

# Run from the NeMo-RL repository root. CONFIG is the only required switch
# between the Qwen3.5 and Nemotron 3.5 performance recipes.
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "${REPO_ROOT}"

account=${SLURM_ACCOUNT:-nemotron_n4_post}
CONFIG=${CONFIG:-examples/configs/recipes/llm/performance/grpo-qwen3.5-35ba3b-8n4g-async-1off-mxfp8-moe-rollout.yaml}
NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
RUN_NAME=${RUN_NAME:-qwen35-mixed-mxfp8-moe-20step}
MODEL_PATH=${MODEL_PATH:-}

test -f "${CONFIG}"
test "$(git status --porcelain)" = ""

MODEL_OVERRIDE=""
if [[ -n "${MODEL_PATH}" ]]; then
  test -f "${MODEL_PATH}/config.json"
  MODEL_OVERRIDE="policy.model_name=${MODEL_PATH} policy.tokenizer.name=${MODEL_PATH}"
fi

COMMAND="cd ${REPO_ROOT} && \
NRL_FORCE_REBUILD_VENVS=true \
HF_HOME=/raid/scratch/\${USER}/pr3659-mixed/\${SLURM_JOB_ID}/hf \
HF_DATASETS_CACHE=/raid/scratch/\${USER}/pr3659-mixed/\${SLURM_JOB_ID}/datasets \
UV_CACHE_DIR=/raid/scratch/\${USER}/pr3659-mixed/cache/uv \
TORCHINDUCTOR_CACHE_DIR=/raid/scratch/\${USER}/pr3659-mixed/cache/inductor \
TRITON_CACHE_DIR=/raid/scratch/\${USER}/pr3659-mixed/cache/triton \
VLLM_CACHE_ROOT=/raid/scratch/\${USER}/pr3659-mixed/cache/vllm \
NRL_VLLM_USE_V1=1 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
uv run examples/run_grpo.py \
  --config ${CONFIG} \
  ${MODEL_OVERRIDE} \
  grpo.max_num_steps=20 \
  checkpointing.enabled=false \
  logger.log_dir=/raid/scratch/\${USER}/pr3659-mixed/\${SLURM_JOB_ID}/results \
  logger.wandb_enabled=True \
  logger.wandb.project=nemo-rl-pr3659-mixed-refit \
  logger.wandb.name=${RUN_NAME}"

CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/images/high_stripe/rl.55639700.sqsh}
BASE_LOG_DIR=${BASE_LOG_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/results/pr3659-mixed-recipes/${RUN_NAME}}
MOUNTS="${REPO_ROOT}:${REPO_ROOT},/lustre:/lustre,/raid/scratch:/raid/scratch"

mkdir -p "${BASE_LOG_DIR}"
export COMMAND CONTAINER BASE_LOG_DIR MOUNTS GPUS_PER_NODE

sbatch_args=()
if [[ "${TEST_ONLY:-0}" == "1" ]]; then
  sbatch_args+=(--test-only)
fi

sbatch "${sbatch_args[@]}" \
  --nodes="${NUM_ACTOR_NODES}" \
  --account="${account}" \
  --job-name="${account}.${RUN_NAME}" \
  --partition=batch \
  --time=04:00:00 \
  --gres="gpu:${GPUS_PER_NODE}" \
  --exclusive \
  --mem=0 \
  --segment=2 \
  ray.sub
