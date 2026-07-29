#!/usr/bin/env bash

set -euo pipefail

: "${MODE:?MODE is required}"
: "${REPO:?REPO is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${TOTAL_NODES:?TOTAL_NODES is required}"
: "${GPUS_PER_NODE:?GPUS_PER_NODE is required}"
: "${GEN_NODES:?GEN_NODES is required}"
: "${SEGMENT_SIZE:?SEGMENT_SIZE is required}"
: "${MAX_STEPS:?MAX_STEPS is required}"
: "${RUN_PREFIX:?RUN_PREFIX is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"

ARM_RUNNER=${ARM_RUNNER:-${REPO}/experiments/nccl_reshard_pr3294/run_arm.sbatch}
test -x "${ARM_RUNNER}"
mkdir -p "${RESULT_ROOT}/results"

for ARM in baseline optimized; do
  RUN_NAME=${RUN_PREFIX}-${ARM}
  EXPERIMENT_ROOT=${RESULT_ROOT}/results/${RUN_NAME}
  echo "Starting ${ARM}: ${RUN_NAME}"
  ARM="${ARM}" \
  RUN_NAME="${RUN_NAME}" \
  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
    bash "${ARM_RUNNER}"
done
