#!/usr/bin/env bash
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

: "${REPO:?REPO is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${OUTPUT:?OUTPUT is required}"

srun \
  --nodes=1 \
  --ntasks=1 \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  /bin/bash -lc "
    set -euo pipefail
    cd '${REPO}'
    PYTHON_BIN=/opt/nemo_rl_venv/bin/python
    test -x \"\${PYTHON_BIN}\"
    VLLM_SITE_PACKAGES=\$(find /opt/ray_venvs \
      -type d -path '*Vllm*GenerationWorker/lib/python*/site-packages' \
      -print -quit)
    test -n \"\${VLLM_SITE_PACKAGES}\"
    export PYTHONPATH=\"\${VLLM_SITE_PACKAGES}:${REPO}\"
    \"\${PYTHON_BIN}\" \
      experiments/nccl_reshard_pr3294/benchmark_mxfp8_shuffle.py \
      --output '${OUTPUT}'
  "
