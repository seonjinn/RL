#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?}"
: "${SOURCE_SHA:?}"
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
export PATH=${SLURM_BIN:-/cm/local/apps/slurm/current/bin}:${PATH}
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
srun --ntasks-per-node=1 --kill-on-bad-exit=1 --wait=30 \
  --container-image="${CONTAINER}" \
  --container-mounts="/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch" \
  --container-workdir="${REPO}" \
  --output="${RESULT_DIR}/node-%N.log" \
  timeout --signal=TERM --kill-after=30s 45m \
  bash -o pipefail -c 'git -C "$REPO" show "$SOURCE_SHA:experiments/native_mxfp8_m2n/run_node.sh" | bash'
