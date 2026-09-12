#!/bin/bash

set -euo pipefail

: "${ACCOUNT:?Set ACCOUNT}"
: "${CONTAINER:?Set CONTAINER to an immutable nightly .sqsh}"
: "${HF_HOME:?Set HF_HOME to the shared model cache}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a shared result directory}"

RUN_NAME=${RUN_NAME:-super-32n4g-hybridep-main-20step}
NUM_NODES=${NUM_NODES:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MAX_STEPS=${MAX_STEPS:-20}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-04:00:00}
SEGMENT_SIZE=${SEGMENT_SIZE:-8}
WANDB_PROJECT=${WANDB_PROJECT:-nemo-rl-hybridep-validation}

repo_root=$(git rev-parse --show-toplevel)
source_sha=$(git rev-parse HEAD)
run_root="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${run_root}"

command="UV_NO_SYNC=1 uv run examples/run_grpo.py \
  --config experiments/hybridep-super-gb200-20260912/config.yaml \
  grpo.max_num_steps=${MAX_STEPS} \
  checkpointing.enabled=false \
  logger.log_dir=${run_root}/logs \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.name=${RUN_NAME}"

cat >"${run_root}/manifest.txt" <<EOF
source_sha=${source_sha}
container=${CONTAINER}
config=experiments/hybridep-super-gb200-20260912/config.yaml
nodes=${NUM_NODES}
gpus_per_node=${GPUS_PER_NODE}
max_steps=${MAX_STEPS}
partition=${PARTITION}
segment_size=${SEGMENT_SIZE}
EOF

cd "${repo_root}"
CONTAINER="${CONTAINER}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_HOME}/datasets" \
MOUNTS="/home:/home,/lustre:/lustre" \
COMMAND="${command}" \
sbatch --parsable \
  --nodes="${NUM_NODES}" \
  --account="${ACCOUNT}" \
  --job-name="${ACCOUNT}.${RUN_NAME}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --segment="${SEGMENT_SIZE}" \
  --constraint="36x2" \
  --comment=metrics \
  --output="${run_root}/slurm-%j.out" \
  ray.sub

