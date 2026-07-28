#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/../profiles/${CLUSTER:?Set CLUSTER to ptyche or oci-hsg}.env"

PHASE="${PHASE:-smoke}"
case "${PHASE}" in
  smoke|performance|accuracy) ;;
  *)
    echo "PHASE must be smoke, performance, or accuracy" >&2
    exit 2
    ;;
esac

CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp-cudagraph-matrix.yaml
RUN_NAME="latestmain-nanov3-nocg-${PHASE}"
COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run --extra mcore examples/run_grpo.py \
  --config ${CONFIG} \
  policy.model_name=${NANOV3_MODEL_SNAPSHOT:?Set NANOV3_MODEL_SNAPSHOT} \
  policy.tokenizer.name=${NANOV3_TOKENIZER_SNAPSHOT:?Set NANOV3_TOKENIZER_SNAPSHOT} \
  cluster.num_nodes=${NUM_ACTOR_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=${INFERENCE_NODES:?Set INFERENCE_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} \
  grpo.max_num_steps=${STEPS:-5} \
  checkpointing.enabled=false \
  logger.log_dir=exp_logs/${RUN_NAME} \
  logger.wandb_enabled=true \
  logger.wandb.project=sna-async-grpo-gb200 \
  logger.wandb.name=${RUN_NAME}"

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
