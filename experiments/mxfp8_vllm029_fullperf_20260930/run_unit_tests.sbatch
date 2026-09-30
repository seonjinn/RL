#!/usr/bin/env bash

# Scheduler flags are supplied by the submitter so this payload can be reused
# with any MARS account.

set -euo pipefail

: "${REPO:?Set REPO to the committed NeMo-RL worktree}"
: "${CONTAINER:?Set CONTAINER to the pinned nightly image}"
: "${SRUN_BIN:?Set SRUN_BIN to the resolved srun path from the login node}"

SCRATCH_ROOT="/raid/scratch/${USER}/u/${SLURM_JOB_ID}"
RAY_ROOT="/tmp/nrlu-${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_ROOT}"

"${SRUN_BIN}" --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/raid/scratch:/raid/scratch,/lustre:/lustre" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export HOME='${SCRATCH_ROOT}/home'
    export TMPDIR='${RAY_ROOT}'
    export RAY_TMPDIR='${RAY_ROOT}'
    export UV_CACHE_DIR='${SCRATCH_ROOT}/uv'
    export TORCHINDUCTOR_CACHE_DIR='${SCRATCH_ROOT}/inductor'
    export TRITON_CACHE_DIR='${SCRATCH_ROOT}/triton'
    export PYTHONPATH='${REPO}:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM'
    mkdir -p \"\${HOME}\" \"\${TMPDIR}\" \"\${UV_CACHE_DIR}\"
    /opt/nemo_rl_venv/bin/python -m pytest -q \
      tests/unit/models/generation/test_vllm_fp8_quantization.py \
      tests/unit/models/generation/test_vllm_mxfp8_utils.py \
      tests/unit/models/generation/test_vllm_refit_adapter.py \
      tests/unit/models/generation/test_vllm_generation.py \
      tests/unit/models/policy/test_mxfp8_refit_source.py \
      tests/unit/weight_sync/test_nccl_reshard_utils.py \
      tests/unit/weight_sync/test_refit_components.py \
      tests/unit/weight_sync/test_weight_synchronizer.py \
      tests/unit/weight_sync/test_reshard_rebuild.py
  "

rm -rf "${SCRATCH_ROOT}"
rm -rf "${RAY_ROOT}"
