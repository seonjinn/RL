#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 RECIPE RUN_NAME [MODEL_PATH]" >&2
  exit 2
fi

RECIPE=$1
RUN_NAME=$2
MODEL_PATH=${3:-}
REPO_ROOT=$(git rev-parse --show-toplevel)

test -f "${REPO_ROOT}/${RECIPE}"
test "$(git -C "${REPO_ROOT}" status --porcelain)" = ""

ACCOUNT=${SLURM_ACCOUNT:-nemotron_n4_post}
PARTITION=${PARTITION:-batch}
NODES=${NODES:-6}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
WALLTIME=${WALLTIME:-4:00:00}
SEGMENT_SIZE=${SEGMENT_SIZE:-2}
CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/images/high_stripe/rl.55639700.sqsh}
BASE_LOG_DIR=${BASE_LOG_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/results/pr3659-mixed-recipes/slurm}

mkdir -p "${BASE_LOG_DIR}"

MODEL_OVERRIDE=""
if [[ -n "${MODEL_PATH}" ]]; then
  test -f "${MODEL_PATH}/config.json"
  MODEL_OVERRIDE="policy.model_name=${MODEL_PATH} policy.tokenizer.name=${MODEL_PATH}"
fi

read -r -d '' COMMAND <<EOF || true
cd ${REPO_ROOT} &&
export PYTHONPATH=${REPO_ROOT}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src\${PYTHONPATH:+:\${PYTHONPATH}} &&
export HF_HOME=/raid/scratch/\${USER}/pr3659-mixed/\${SLURM_JOB_ID}/hf &&
export HF_HUB_CACHE=\${HF_HOME}/hub &&
export HF_DATASETS_CACHE=\${HF_HOME}/datasets &&
export TMPDIR=/raid/scratch/\${USER}/pr3659-mixed/\${SLURM_JOB_ID}/tmp &&
export TORCHINDUCTOR_CACHE_DIR=/raid/scratch/\${USER}/pr3659-mixed/cache/inductor &&
export TRITON_CACHE_DIR=/raid/scratch/\${USER}/pr3659-mixed/cache/triton &&
export VLLM_CACHE_ROOT=/raid/scratch/\${USER}/pr3659-mixed/cache/vllm &&
export UV_CACHE_DIR=/raid/scratch/\${USER}/pr3659-mixed/cache/uv &&
export NRL_VLLM_USE_V1=1 &&
export VLLM_USE_FLASHINFER_MOE_FP8=1 &&
export VLLM_FLASHINFER_MOE_BACKEND=latency &&
export NRL_IGNORE_VERSION_MISMATCH=1 &&
mkdir -p \${HF_HUB_CACHE} \${HF_DATASETS_CACHE} \${TMPDIR} \${TORCHINDUCTOR_CACHE_DIR} \${TRITON_CACHE_DIR} \${VLLM_CACHE_ROOT} \${UV_CACHE_DIR} &&
/opt/nemo_rl_venv/bin/python examples/run_grpo.py --config ${RECIPE} ${MODEL_OVERRIDE}
EOF

export COMMAND CONTAINER GPUS_PER_NODE BASE_LOG_DIR
export MOUNTS="${REPO_ROOT}:${REPO_ROOT},/lustre:/lustre,/raid/scratch:/raid/scratch"
export SETUP_COMMAND='rm -rf "/raid/scratch/${USER}/pr3659-mixed/${SLURM_JOB_ID}" && mkdir -p "/raid/scratch/${USER}/pr3659-mixed/${SLURM_JOB_ID}"'

sbatch_args=(
  --nodes="${NODES}"
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --job-name="${ACCOUNT}.${RUN_NAME}"
  --gres="gpu:${GPUS_PER_NODE}"
  --exclusive
  --mem=0
  --segment="${SEGMENT_SIZE}"
)

if [[ "${TEST_ONLY:-0}" == "1" ]]; then
  sbatch_args+=(--test-only)
fi

sbatch "${sbatch_args[@]}" "${REPO_ROOT}/ray.sub"
