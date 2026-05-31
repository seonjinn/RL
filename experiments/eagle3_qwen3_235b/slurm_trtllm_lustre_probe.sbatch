#!/bin/bash
#SBATCH --job-name=q235b-trtllm-lustre-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT}"
CONTAINER="${CONTAINER:?set CONTAINER}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/tmp}"

mkdir -p logs

srun \
  --container-image="$CONTAINER" \
  --container-mounts="$MOUNTS" \
  --container-workdir="$CONTAINER_WORKDIR" \
  bash -lc '
    set -eux
    hostname
    ls -ld /lustre || true
    ls -ld /lustre/fsw || true
    ls -ld /lustre/fs1 || true
    ls -ld "'"$REPO_ROOT"'"
    cd "'"$REPO_ROOT"'"
    pwd
    python3 -c "import sys; print(sys.executable); import tensorrt_llm; print(tensorrt_llm.__version__)"
  '
