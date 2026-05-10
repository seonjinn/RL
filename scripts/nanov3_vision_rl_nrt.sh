#!/usr/bin/env bash
#
# Parity launcher: runs the SAME workload as
# nemo-rl-recipes/scripts/nanov3_vision_rl.sh, but on the
# nemo-rl-super-vllm0.18 codebase (vllm 0.18 + super container).
#
# Both launchers point at examples/omni/nanov3_vision_rl.yaml with the
# identical Hydra override surface, identical scale (NUM_NODES=4,
# GPUS_PER_NODE=8), and identical model / dataset (sourced from
# IMAGE_GRPO_MODEL_NAME / IMAGE_GRPO_CACHE_DIR in NEMORL/.env).
#
# Wandb is enabled and forced to the same project as the recipes
# baseline (nemo-rl-omni). JOB_NAME_BASE defaults to image-grpo-vllm018
# so the two runs are easy to tell apart in the dashboard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -f "${NEMORL}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${NEMORL}/.env"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-examples/omni/nanov3_vision_rl.yaml}"
NUM_NODES="${NUM_NODES:-4}"
JOB_NAME_BASE="${JOB_NAME_BASE:-image-grpo-vllm018}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%3N)}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_BASE}-${RUN_ID}}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${CP_SIZE:-}}"
MODEL_NAME="${IMAGE_GRPO_MODEL_NAME:-${MODEL_NAME:-}}"
CACHE_DIR="${IMAGE_GRPO_CACHE_DIR:-${CACHE_DIR:-}}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-omni}"
: "${MODEL_NAME:?Set IMAGE_GRPO_MODEL_NAME or MODEL_NAME, or define it in ${NEMORL}/.env}"
: "${CACHE_DIR:?Set IMAGE_GRPO_CACHE_DIR or CACHE_DIR, or define it in ${NEMORL}/.env}"
RESULTS_ROOT="${RESULTS_ROOT:-${NEMORL}/results}"
RESULTS_DIR="${RESULTS_ROOT}/${JOB_NAME}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT or define it in ${NEMORL}/.env}"
SBATCH_PARTITION="${SBATCH_PARTITION:-${PARTITION:-batch}}"
SBATCH_TIME="${SBATCH_TIME:-4:00:00}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# ray.sub only sees exported vars; without this it falls back to its own
# default and trips the "GPUS_PER_NODE doesn't match cluster GRES" check.
export NUM_NODES

# Container + mounts. Default to the super-omni-rl image that ships
# pre-built /opt/ray_venvs and the vllm-0.18 wheel. Overridable via .env.
CONTAINER_ROOT="${CONTAINER_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/hanrongy/project/nemotron_omni/rl/images}"
export CONTAINER="${CONTAINER:-${CONTAINER_ROOT}/super-omni-rl-20260501-vllm0.18.sqsh}"
export MOUNTS="${MOUNTS:-/lustre:/lustre}"

# Trust the baked /opt/ray_venvs/<actor>/ in the container so
# create_local_venv() short-circuits and we don't re-resolve nemo-rl
# extras through the private flashinfer-cubin index at runtime.
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
# flashinfer-jit-cache=0.6.5+cu129 vs flashinfer=0.6.9 ships in the
# image; the strict version assert is harmless for this workload.
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

export CACHE_ROOT="${CACHE_ROOT:-${NEMORL}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${RUN_ID}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
export NEMO_RL_TRAIN_STEP_MEM_DIAG="${NEMO_RL_TRAIN_STEP_MEM_DIAG:-1}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-true}"
# Provide auth credentials for the private flashinfer-cubin gitlab pypi
# index if NRL_VENVS_TRUST_EXISTING is ever flipped off. Sourced from
# the user's glab CLI config (no token literal in the script).
if [[ -z "${GITLAB_FLASHINFER_TOKEN:-}" ]] && [[ -f "${HOME}/.config/glab-cli/config.yml" ]]; then
  GITLAB_FLASHINFER_TOKEN=$(grep -A 1 "gitlab-master.nvidia.com:" "${HOME}/.config/glab-cli/config.yml" | grep -oE 'glpat-[A-Za-z0-9_-]+' | head -1 || true)
fi
if [[ -n "${GITLAB_FLASHINFER_TOKEN:-}" ]]; then
  export UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME="${UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME:-oauth2}"
  export UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD="${GITLAB_FLASHINFER_TOKEN}"
fi

if [[ ! -f "${NEMORL}/ray.sub" ]]; then
  echo "ray.sub not found under NEMORL=${NEMORL}" >&2
  exit 1
fi

if [[ "${CONFIG_PATH}" = /* ]]; then
  CONFIG_ABS_PATH="${CONFIG_PATH}"
else
  CONFIG_ABS_PATH="${NEMORL}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_ABS_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

EXTRA_OVERRIDES=""
if [[ -n "${CONTEXT_PARALLEL_SIZE}" ]]; then
  EXTRA_OVERRIDES+=" policy.megatron_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE}"
fi
# super-vllm0.18's grpo.py requires grpo.val_at_end (recipes' grpo.py doesn't
# read this key). The recipes-derived omni YAML doesn't define it, so inject
# the super-side default (false) here so the run reaches Step 1.
EXTRA_OVERRIDES+=" +grpo.val_at_end=${GRPO_VAL_AT_END:-false}"
# Resume the same wandb run instead of starting a new one when WANDB_RUN_ID
# is set. Use Hydra's `+` so the keys are added (they aren't in the YAML).
# Pair WANDB_RESUME=allow with a pre-chosen id to chain a fresh run + N
# continuations under one wandb run (first to start creates, rest attach).
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  EXTRA_OVERRIDES+=" +logger.wandb.id=${WANDB_RUN_ID} +logger.wandb.resume=${WANDB_RESUME:-must}"
fi

PYTHONPATH_ROOTS="${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM"
if [[ "${USE_REPO_VLLM:-0}" == "1" ]]; then
  PYTHONPATH_ROOTS="${NEMORL}/3rdparty/vllm:${PYTHONPATH_ROOTS}"
fi

# Match recipes' Hydra override surface 1:1 and explicitly enable wandb
# against the same project so the two runs land side-by-side.
export COMMAND="\
mkdir -p '${HF_HOME}' '${HF_MODULES_CACHE}' '${NRL_MEGATRON_CHECKPOINT_DIR}' '${TRITON_CACHE_DIR}' '${TMPDIR}' '${RESULTS_DIR}' && \
export PYTHONPATH=${PYTHONPATH_ROOTS}\${PYTHONPATH:+:\$PYTHONPATH} && \
uv run --no-sync examples/run_vlm_grpo.py --config '${CONFIG_PATH}' \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.model_name='${MODEL_NAME}' \
checkpointing.checkpoint_dir='${RESULTS_DIR}' \
logger.log_dir='${RESULTS_DIR}' \
logger.wandb_enabled=true \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${JOB_NAME}' \
data.train.cache_dir='${CACHE_DIR}'\
${EXTRA_OVERRIDES}"

cd "${NEMORL}"

sbatch \
    --nodes=${NUM_NODES} \
    --account=${SBATCH_ACCOUNT} \
    --job-name=${JOB_NAME} \
    --partition=${SBATCH_PARTITION} \
    --time=${SBATCH_TIME} \
    --gres=gpu:${GPUS_PER_NODE} \
    ray.sub
