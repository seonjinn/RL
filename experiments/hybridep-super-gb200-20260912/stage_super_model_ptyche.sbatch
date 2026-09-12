#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --job-name=stage_nemotron3_super
#SBATCH --output=stage_nemotron3_super_%j.log

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to an immutable nightly .sqsh}"
: "${HF_HOME:?Set HF_HOME to the shared model cache}"

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=0

srun \
  --container-image="${CONTAINER}" \
  --container-mounts="/lustre:/lustre" \
  --no-container-mount-home \
  bash -lc 'python - <<"PY"
import os
from pathlib import Path

import deep_ep
import torch
from huggingface_hub import snapshot_download

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
print(f"deep_ep={deep_ep.__file__}")

snapshot = snapshot_download(
    repo_id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    max_workers=8,
)
weights = list(Path(snapshot).glob("*.safetensors"))
if not weights:
    raise RuntimeError(f"No safetensors weights found in {snapshot}")

print(f"snapshot={snapshot}")
print(f"weight_files={len(weights)}")
deep_ep_multinode = os.environ.get("HYBRID_EP_MULTINODE")
print(f"deep_ep_multinode={deep_ep_multinode}")
PY'
