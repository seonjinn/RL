#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

SCOPE_SLUG=${1:?Pass the scope slug.}
CUDA_GRAPH_SCOPE=${2:?Pass the Hydra CUDA Graph scope list.}
CUDA_GRAPH_IMPL=${3:?Pass the CUDA Graph implementation.}

case "${CUDA_GRAPH_IMPL}" in
  none|transformer_engine) ;;
  *)
    echo "CUDA Graph implementation must be none or transformer_engine" >&2
    exit 2
    ;;
esac

case "${SCOPE_SLUG}:${CUDA_GRAPH_SCOPE}:${CUDA_GRAPH_IMPL}" in
  "nocg:[]:none" | \
  "attn:[attn]:transformer_engine" | \
  "moe-router-preprocess:[moe_router,moe_preprocess]:transformer_engine" | \
  "attn-moe-router-preprocess:[attn,moe_router,moe_preprocess]:transformer_engine")
    ;;
  *)
    echo "Unsupported Qwen CUDA Graph scope request" >&2
    exit 2
    ;;
esac

CLUSTER=${CLUSTER:-ptyche}
PROFILE="${SCRIPT_DIR}/profiles/${CLUSTER}.env"
if [[ ! -f "${PROFILE}" ]]; then
  echo "Missing cluster profile: ${PROFILE}" >&2
  exit 2
fi
source "${PROFILE}"
unset UV_CACHE_DIR_OVERRIDE

PHASE=${PHASE:-performance}
case "${PHASE}" in
  smoke|performance|accuracy) ;;
  *)
    echo "PHASE must be smoke, performance, or accuracy" >&2
    exit 2
    ;;
esac

PARTITION=${PARTITION_OVERRIDE:-${PARTITION}}
TIME_LIMIT=${TIME_LIMIT_OVERRIDE:-${TIME_LIMIT}}
STEPS=${STEPS:-20}
if [[ ! "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS must be a positive integer" >&2
  exit 2
fi

CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
RUN_NAME="qwen3-30ba3b-${SCOPE_SLUG}-${PHASE}"
LOG_ROOT=${LOG_ROOT_OVERRIDE:-exp_logs}
RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_LOG_DIR}"

if [[ "${CUDA_GRAPH_IMPL}" == "transformer_engine" ]]; then
  CUDA_GRAPH_PACKED_SEQ=true
else
  CUDA_GRAPH_PACKED_SEQ=false
fi

COMMAND_ARGS=(
  env
  NRL_FORCE_REBUILD_VENVS=true
  uv
  run
  --extra
  mcore
  examples/run_grpo.py
  --config
  "${CONFIG}"
  "policy.model_name=${QWEN3_30BA3B_SNAPSHOT:?Set QWEN3_30BA3B_SNAPSHOT}"
  "policy.tokenizer.name=${QWEN3_30BA3B_SNAPSHOT}"
  "cluster.num_nodes=${NUM_ACTOR_NODES}"
  "cluster.gpus_per_node=${GPUS_PER_NODE}"
  "cluster.segment_size=${SEGMENT_SIZE}"
  policy.generation.colocated.enabled=false
  "policy.generation.colocated.resources.num_nodes=${INFERENCE_NODES}"
  "policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE}"
  "grpo.max_num_steps=${STEPS}"
  checkpointing.enabled=false
  "logger.log_dir=${RUN_LOG_DIR}"
  logger.wandb_enabled=true
  logger.tensorboard_enabled=true
  logger.wandb.project=sna-async-grpo-gb200
  "logger.wandb.name=${RUN_NAME}"
  "policy.megatron_cfg.cuda_graph_impl=${CUDA_GRAPH_IMPL}"
  "policy.megatron_cfg.cuda_graph_scope=${CUDA_GRAPH_SCOPE}"
  "policy.megatron_cfg.cuda_graph_packed_seq=${CUDA_GRAPH_PACKED_SEQ}"
  policy.megatron_cfg.cuda_graph_max_packed_seqs=16
  policy.megatron_cfg.cuda_graph_warmup_steps=3
)
printf -v COMMAND '%q ' "${COMMAND_ARGS[@]}"
COMMAND=${COMMAND% }

if [[ -z "${CONTAINER:-}" ]]; then
  echo "CONTAINER must not be blank" >&2
  exit 2
fi

SBATCH_CMD=(sbatch)
if [[ "${TEST_ONLY:-0}" == "1" ]]; then
  SBATCH_CMD+=(--test-only)
fi
SBATCH_CMD+=(
  "--nodes=${NUM_ACTOR_NODES}"
  "${SBATCH_GPU_ARGS[@]+${SBATCH_GPU_ARGS[@]}}"
  "--account=${ACCOUNT}"
  "--job-name=${ACCOUNT}-sna.${RUN_NAME}"
  "--partition=${PARTITION}"
  "--time=${TIME_LIMIT}"
  "--segment=${SEGMENT_SIZE}"
  "--output=${RUN_LOG_DIR}/slurm-%j.out"
  "--error=${RUN_LOG_DIR}/slurm-%j.out"
  ray.sub
)

printf 'COMMAND:\n%s\n' "${COMMAND}"
printf 'SBATCH:'
printf ' %q' "${SBATCH_CMD[@]}"
printf '\n'

COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
WANDB_MODE="${WANDB_MODE_OVERRIDE:-offline}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
MOUNTS="${MOUNTS}" \
BASE_LOG_DIR="${RUN_LOG_DIR}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
"${SBATCH_CMD[@]}"
