#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --job-name=hybridep-nightly-smoke

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to an immutable sqsh path}"
: "${REPO_DIR:?Set REPO_DIR to the validation checkout}"

srun \
  --container-image="$CONTAINER" \
  --container-mounts="$REPO_DIR:/workspace,/lustre:/lustre" \
  --no-container-mount-home \
  bash -lc '
    export PYTHONPATH=/workspace/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:/workspace/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
    python -c "import megatron.core; import torch; import transformer_engine.pytorch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.device_count())"
  '
