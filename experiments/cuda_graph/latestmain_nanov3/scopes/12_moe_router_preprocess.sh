#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/../profiles/${CLUSTER:?Set CLUSTER to ptyche or oci-hsg}.env"
PARTITION="${PARTITION_OVERRIDE:-${PARTITION}}"
TIME_LIMIT="${TIME_LIMIT_OVERRIDE:-${TIME_LIMIT}}"

PHASE="${PHASE:-smoke}"
case "${PHASE}" in
  smoke|performance|accuracy) ;;
  *)
    echo "PHASE must be smoke, performance, or accuracy" >&2
    exit 2
    ;;
esac

CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp-cudagraph-matrix.yaml
MOE_OVERRIDES=""
RUN_SUFFIX=""
SHARED_EXPERT_OVERLAP="${SHARED_EXPERT_OVERLAP:-0}"
MOE_ACT_RECOMPUTE="${MOE_ACT_RECOMPUTE:-0}"
if [[ "${SHARED_EXPERT_OVERLAP}" != "0" && "${SHARED_EXPERT_OVERLAP}" != "1" ]]; then
  echo "SHARED_EXPERT_OVERLAP must be 0 or 1" >&2
  exit 2
fi
if [[ "${MOE_ACT_RECOMPUTE}" != "0" && "${MOE_ACT_RECOMPUTE}" != "1" ]]; then
  echo "MOE_ACT_RECOMPUTE must be 0 or 1" >&2
  exit 2
fi
if [[ "${SHARED_EXPERT_OVERLAP}" == "1" ]]; then
  MOE_OVERRIDES+=" policy.megatron_cfg.moe_shared_expert_overlap=true"
  RUN_SUFFIX+="-shared-expert-overlap"
fi
if [[ "${MOE_ACT_RECOMPUTE}" == "1" ]]; then
  MOE_OVERRIDES+=" policy.megatron_cfg.activation_checkpointing=true"
  MOE_OVERRIDES+=" policy.megatron_cfg.recompute_granularity=selective"
  MOE_OVERRIDES+=" 'policy.megatron_cfg.recompute_modules=[moe_act]'"
  RUN_SUFFIX+="-moe-act-recompute"
fi
RUN_NAME="latestmain-nanov3-moe-router-preprocess-${PHASE}${RUN_SUFFIX}"
COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run --extra mcore examples/run_grpo.py \
  --config ${CONFIG} \
  policy.model_name=${NANOV3_MODEL_SNAPSHOT:?Set NANOV3_MODEL_SNAPSHOT} \
  policy.tokenizer.name=${NANOV3_TOKENIZER_SNAPSHOT:?Set NANOV3_TOKENIZER_SNAPSHOT} \
  cluster.num_nodes=${NUM_ACTOR_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  grpo.max_num_steps=${STEPS:-5} \
  checkpointing.enabled=false \
  logger.log_dir=exp_logs/${RUN_NAME} \
  logger.wandb_enabled=true \
  logger.wandb.project=sna-async-grpo-gb200 \
  logger.wandb.name=${RUN_NAME} \
  '+policy.megatron_cfg.cuda_graph_scope=[moe_router,moe_preprocess]' \
  '+policy.megatron_cfg.cuda_graph_impl=transformer_engine' \
  '+policy.megatron_cfg.cuda_graph_packed_seq=true' \
  '+policy.megatron_cfg.cuda_graph_warmup_steps=3'"
COMMAND+=" ${MOE_OVERRIDES}"

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
  "--job-name=${ACCOUNT}.${RUN_NAME}"
  "--partition=${PARTITION}"
  "--time=${TIME_LIMIT}"
  "--segment=${SEGMENT_SIZE}"
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
WANDB_MODE=offline \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
MOUNTS="${MOUNTS}" \
BASE_LOG_DIR="exp_logs/${RUN_NAME}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
"${SBATCH_CMD[@]}"
