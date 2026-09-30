#!/usr/bin/env bash

set -euo pipefail

: "${REPO:?Set REPO to the committed NeMo-RL worktree}"
: "${CONTAINER:?Set CONTAINER to the pinned nightly image}"
: "${SRUN_BIN:?Set SRUN_BIN to the resolved srun path from the login node}"

SCRATCH_ROOT="/raid/scratch/${USER}/mxfp8-vllm029-preflight-${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_ROOT}"

"${SRUN_BIN}" --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid/scratch:/raid/scratch,/lustre:/lustre" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export HOME='${SCRATCH_ROOT}/home'
    export TMPDIR='${SCRATCH_ROOT}/tmp'
    export PREFLIGHT_OUTPUT='${SCRATCH_ROOT}/rendered'
    export UV_CACHE_DIR='${SCRATCH_ROOT}/uv'
    export PYTHONPATH='${REPO}:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM'
    mkdir -p \"\${HOME}\" \"\${TMPDIR}\" \"\${PREFLIGHT_OUTPUT}\" \"\${UV_CACHE_DIR}\"
    /opt/nemo_rl_venv/bin/python \
      experiments/mxfp8_vllm029_fullperf_20260930/preflight.py
  "

rm -rf "${SCRATCH_ROOT}"
