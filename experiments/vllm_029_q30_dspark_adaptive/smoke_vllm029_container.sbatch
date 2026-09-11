#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to the immutable sqsh path}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the checked-out experiment repository}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT to the exact vLLM commit}"
: "${RECEIPT_PATH:?Set RECEIPT_PATH to a scoped Lustre result path}"

readonly image_sha256=$(sha256sum "${CONTAINER_IMAGE}" | awk '{print $1}')
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/home:/home,/lustre:/lustre \
  bash -lc "cd '${SOURCE_ROOT}' && CUDA_VISIBLE_DEVICES=0 python -m experiments.vllm_029_q30_dspark_adaptive.smoke --output '${RECEIPT_PATH}' --image '${CONTAINER_IMAGE}' --image-sha256 '${image_sha256}' --source-commit '${SOURCE_COMMIT}'"

