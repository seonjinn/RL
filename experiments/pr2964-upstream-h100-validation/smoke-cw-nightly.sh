#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --job-name=pr2964-nightly-smoke

set -euo pipefail

python3 - <<'PY'
import importlib.util
import platform

import torch
import transformer_engine.pytorch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert importlib.util.find_spec("megatron.core") is not None

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"visible_gpus={torch.cuda.device_count()}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"transformer_engine={transformer_engine.pytorch.__file__}")
print(f"megatron_core={importlib.util.find_spec('megatron.core').origin}")
PY
