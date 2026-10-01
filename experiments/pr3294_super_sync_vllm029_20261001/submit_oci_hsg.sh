#!/usr/bin/env bash

set -euo pipefail

: "${CODE_ROOT:?Set CODE_ROOT to an immutable NeMo-RL worktree}"
: "${SOURCE_SHA:?Set SOURCE_SHA to the exact commit under test}"
: "${ARM:?Set ARM to baseline or optimized}"

if [[ "${ARM}" != "baseline" && "${ARM}" != "optimized" ]]; then
  echo "ARM must be baseline or optimized, got ${ARM}" >&2
  exit 2
fi

actual_sha=$(git -C "${CODE_ROOT}" rev-parse HEAD)
if [[ "${actual_sha}" != "${SOURCE_SHA}" ]]; then
  echo "CODE_ROOT is at ${actual_sha}, expected ${SOURCE_SHA}" >&2
  exit 2
fi
if [[ -n "$(git -C "${CODE_ROOT}" status --short)" ]]; then
  echo "CODE_ROOT must be clean: ${CODE_ROOT}" >&2
  exit 2
fi

ACCOUNT=${ACCOUNT:-nemotron_n4_post}
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/images/vllm029-20260930/nemo_rl_nightly_vllm029_20260930_7563566.sqsh}
RAY_SUB=${RAY_SUB:-/home/sna/worktrees/mxfp8-vllm029-audit-20260930/ray.sub}
NUM_NODES=${NUM_NODES:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_WORKER=${CPUS_PER_WORKER:-144}
EP_SIZE=${EP_SIZE:-32}
MAX_STEPS=${MAX_STEPS:-20}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.69}
WALLTIME=${WALLTIME:-04:00:00}
SEGMENT_SIZE=${SEGMENT_SIZE:-8}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)}
GPU_MEMORY_TAG=${GPU_MEMORY_UTILIZATION/./}
RUN_NAME="pr3294-v029-super-sync-ep${EP_SIZE}-gpu${GPU_MEMORY_TAG}-${ARM}-${RUN_TAG}"
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/pr3294-super-sync-vllm029-20261001}
RUN_DIR="${RESULT_ROOT}/${RUN_NAME}"
MODEL_PATH=${MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e}
if [[ ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/tokenizer_config.json" ]]; then
  echo "MODEL_PATH is not a complete local snapshot: ${MODEL_PATH}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

CONFIG=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-mxfp8-rollout.yaml
OPTIMIZATION_OVERRIDES=""
if [[ "${ARM}" == "optimized" ]]; then
  OPTIMIZATION_OVERRIDES="policy.generation.vllm_cfg.refit_prequantize=true \
+policy.generation.vllm_cfg.refit_cache_loader_routes=true \
policy.refit_persistent_ipc_buffers=true"
fi

export COMMAND="exec >${RUN_DIR}/driver.log 2>&1; \
set -euxo pipefail; \
cd /opt/nemo-rl; \
NRL_IGNORE_VERSION_MISMATCH=1 \
uv run --no-sync examples/run_grpo.py \
--config ${CONFIG} \
policy.model_name=${MODEL_PATH} \
policy.tokenizer.name=${MODEL_PATH} \
policy.megatron_cfg.expert_model_parallel_size=${EP_SIZE} \
policy.megatron_cfg.env_vars.NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=\\\"${EP_SIZE}\\\" \
policy.generation.vllm_cfg.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
grpo.max_num_steps=${MAX_STEPS} \
checkpointing.enabled=false \
logger.log_dir=${RUN_DIR}/logs \
logger.wandb_enabled=true \
logger.wandb.project=sna-pr3294-super-sync-vllm029-ab \
logger.wandb.name=${RUN_NAME} \
logger.monitor_gpus=true \
${OPTIMIZATION_OVERRIDES}"

export CONTAINER
export GPUS_PER_NODE
export CPUS_PER_WORKER
export PATH="/cm/local/apps/slurm/current/bin:${PATH}"
export BASE_LOG_DIR="${RUN_DIR}/ray"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/cache}
export MOUNTS="/lustre:/lustre,${CODE_ROOT}/nemo_rl:/opt/nemo-rl/nemo_rl,${CODE_ROOT}/examples:/opt/nemo-rl/examples"

SBATCH_ARGS=(
  --nodes="${NUM_NODES}"
  --account="${ACCOUNT}"
  --job-name="${ACCOUNT}.${RUN_NAME}"
  --partition=batch
  --time="${WALLTIME}"
  --gres="gpu:${GPUS_PER_NODE}"
  --segment="${SEGMENT_SIZE}"
  --chdir="${RUN_DIR}"
  --exclusive
  --mem=0
  --output="${RUN_DIR}/slurm-%j.out"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  sbatch --test-only "${SBATCH_ARGS[@]}" "${RAY_SUB}"
else
  sbatch --parsable "${SBATCH_ARGS[@]}" "${RAY_SUB}"
fi
