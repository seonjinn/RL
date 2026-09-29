#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:20:00

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to the immutable nightly squashfs path}"
[[ -f "$CONTAINER" && ! -L "$CONTAINER" ]]

archive=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr4037-nightly-validation-20260929/pr4037-head17752c9e-source.tar.gz
archive_sha256=15a5305ab615e07ad45d1be4d2bf0b4d0feba6cd0c3c8963d0587b4853604e6a
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum -c -

mkdir -p /raid/scratch/sna
scratch_dir=$(mktemp -d /raid/scratch/sna/pr4037-smoke-XXXXXXXX)
trap 'rm -rf "$scratch_dir"' EXIT
tar -xzf "$archive" -C "$scratch_dir"
export PYTHONPATH="$scratch_dir"

srun --nodes=1 --ntasks=1 --gres=gpu:1 \
  --no-container-mount-home \
  --container-image="$CONTAINER" \
  --container-mounts=/lustre:/lustre,/raid/scratch:/raid/scratch \
  --container-workdir="$scratch_dir" \
  /opt/nemo_rl_venv/bin/python -c '
import torch
from nemo.lens.groups import SpanRegistry
from nemo_rl.telemetry.span_groups import RLSpanGroup

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
print("span_registry_ok", SpanRegistry, RLSpanGroup)
print("torch", torch.__version__, "gpus", torch.cuda.device_count())
'
