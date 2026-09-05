#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
MAX_STEPS=${MAX_STEPS:-6}
PROFILE_STEP_RANGE=${PROFILE_STEP_RANGE:-3:5}

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
LAUNCHER=${ROOT}/research/mxfp8_training_rl/submit_oci_hsg.sh

submit_arm() {
  local training_precision=$1
  TRAINING_PRECISION=${training_precision} \
  ROLLOUT_PRECISION=mxfp8 \
  MODEL=nano \
  ACTION=${ACTION} \
  RUN_GROUP=${RUN_GROUP} \
  MAX_STEPS=${MAX_STEPS} \
  PROFILE_POLICY=true \
  PROFILE_STEP_RANGE=${PROFILE_STEP_RANGE} \
  EXTRA_OVERRIDES="grpo.seed=42 data.shuffle=false" \
  "${LAUNCHER}"
}

submit_arm bf16
submit_arm mxfp8
