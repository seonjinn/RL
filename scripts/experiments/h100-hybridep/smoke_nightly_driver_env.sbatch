#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --job-name=hybridep-nightly-driver-smoke

set -euo pipefail

: "${CONTAINER:?CONTAINER is required}"
: "${CODE_DIR:?CODE_DIR is required}"

readonly driver_venv="/raid/scratch/nemo-rl-driver-venv-${SLURM_JOB_ID}"
readonly uv_cache="/raid/scratch/nemo-rl-uv-cache-${USER}-${SLURM_JOB_ID}"

srun --ntasks-per-node=1 \
  --container-image="$CONTAINER" \
  --container-mounts=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch \
  bash -lc "
    set -euo pipefail
    cd '$CODE_DIR'
    export UV_PROJECT_ENVIRONMENT='$driver_venv'
    export UV_CACHE_DIR='$uv_cache'
    uv sync --locked --extra mcore --no-install-project
    PYTHONPATH='$CODE_DIR' uv run --no-sync python -c '
import torch
import transformer_engine.pytorch
import megatron.core
import nemo_rl

print(f\"cuda_available={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}\")
'
  "
