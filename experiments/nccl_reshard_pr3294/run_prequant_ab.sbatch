#!/usr/bin/env bash

set -euo pipefail

: "${REPO:?REPO is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${MAX_STEPS:?MAX_STEPS is required}"
: "${RUN_SUFFIX:?RUN_SUFFIX is required}"

for MODE in bf16 mxfp8-nccl-prequant mxfp8-rollout; do
  RUN_NAME="pr3294-nccl-gcp-b200-${MODE}-${MAX_STEPS}step-${RUN_SUFFIX}"
  ARM=optimized \
  MODE="${MODE}" \
  RUN_NAME="${RUN_NAME}" \
  EXPERIMENT_ROOT="${RESULT_ROOT}/results/${RUN_NAME}" \
  bash "${REPO}/experiments/nccl_reshard_pr3294/run_arm.sbatch"
done
