#!/usr/bin/env bash

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to the staged nightly image}"
: "${SRUN_BIN:?Set SRUN_BIN to the compute-node srun path}"

"${SRUN_BIN}" --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="/raid/scratch:/raid/scratch,/lustre:/lustre" \
  /opt/nemo_rl_venv/bin/python - <<'PY'
import importlib
import importlib.metadata

import torch
import transformer_engine.pytorch
import vllm
from megatron import core

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
assert importlib.metadata.version("vllm") == "0.29.0"
for package in ("causal_conv1d", "cupy", "mamba_ssm"):
    importlib.import_module(package)

print(f"torch={torch.__version__}")
print(f"vllm={vllm.__version__}")
print(f"gpus={torch.cuda.device_count()}")
print(f"megatron_core={core.__file__}")
PY
