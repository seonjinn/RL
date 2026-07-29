#!/usr/bin/env bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

: "${REPO:?REPO is required}"
: "${CONTAINER:?CONTAINER is required}"

TEST_LOG=${TEST_LOG:-${REPO}/experiments/nccl_reshard_pr3294/unit-tests-${SLURM_JOB_ID}.log}
mkdir -p "$(dirname "${TEST_LOG}")"

srun \
  --nodes=1 \
  --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  /bin/bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export PYTHONPATH='${REPO}'
    /opt/nemo_rl_venv/bin/python -m pytest -q \
      tests/unit/weight_sync/test_nccl_reshard_utils.py \
      tests/unit/models/generation/test_vllm_backend.py::test_nccl_reshard_refit_processes_weights_once
  " 2>&1 | tee "${TEST_LOG}"

