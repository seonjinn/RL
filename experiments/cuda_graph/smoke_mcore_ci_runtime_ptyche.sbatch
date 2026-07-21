#!/bin/bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --job-name=coreai_dlalgo_llm-pr5672.mcore-smoke
#SBATCH --output=mcore_ci_smoke_%j.log

set -euo pipefail

CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/megatron_lm_mcore_ci_dev_main.sqsh}

srun --nodes=1 --ntasks=1 --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  python -c '
from importlib.metadata import version

import mamba_ssm
import megatron.core
import torch
import transformer_engine.pytorch

assert torch.cuda.is_available()
print(f"cuda_devices={torch.cuda.device_count()}")
print(f"torch={torch.__version__}")
print(f"transformer_engine={version('transformer_engine')}")
print(f"mamba_ssm={version('mamba_ssm')}")
print("megatron_core=imported")
'
