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
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  /bin/bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export PYTHONPATH='${REPO}'
    PYTHON_BIN=/opt/nemo_rl_venv/bin/python
    if [[ ! -x \"\${PYTHON_BIN}\" ]]; then
      PYTHON_BIN=\$(find /root/.local/share/uv/python \
        -type f -path '*/bin/python3.13' -perm -u+x -print -quit)
    fi
    test -n \"\${PYTHON_BIN}\" && test -x \"\${PYTHON_BIN}\"
    \"\${PYTHON_BIN}\" -m pytest -q \
      tests/unit/weight_sync/test_nccl_reshard_utils.py
    VLLM_SITE_PACKAGES=\$(find /opt/ray_venvs \
      -type d -path '*Vllm*GenerationWorker/lib/python*/site-packages' \
      -print -quit)
    test -n \"\${VLLM_SITE_PACKAGES}\"
    export PYTHONPATH=\"\${VLLM_SITE_PACKAGES}:${REPO}\"
    \"\${PYTHON_BIN}\" -m pytest -q --vllm-only \
      tests/unit/models/generation/test_vllm_backend.py::test_nccl_reshard_refit_processes_weights_once
  " 2>&1 | tee "${TEST_LOG}"
