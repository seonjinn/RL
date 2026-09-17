#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1
readonly scratch=/raid/scratch/sna/nightly-smoke-${SLURM_JOB_ID}
mkdir -p "${scratch}"
export HF_HOME=${scratch}/hf UV_CACHE_DIR=${scratch}/uv
export TRITON_CACHE_DIR=${scratch}/triton CUDA_CACHE_PATH=${scratch}/cuda
export TORCH_EXTENSIONS_DIR=${scratch}/torch-extensions

/opt/nemo_rl_venv/bin/python - <<'PY'
import sys
import ray
import torch
import nemo_rl
print("DRIVER_IMPORT_OK", sys.executable, torch.__version__, ray.__version__, nemo_rl.__file__)
PY

/usr/local/bin/python-MegatronPolicyWorker - <<'PY'
import sys
import importlib.metadata
import torch
import transformer_engine.pytorch
import megatron.core
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
for device in range(4):
    assert torch.ones(16, device=f"cuda:{device}").sum().item() == 16
print("TRAINING_GPU_SMOKE_OK", sys.executable, torch.__version__, importlib.metadata.version("transformer-engine"), megatron.core.__file__)
PY

for worker in VllmGenerationWorker VllmAsyncGenerationWorker; do
  /usr/local/bin/python-${worker} - <<'PY'
import sys
import importlib.metadata
import torch
import vllm
from vllm.platforms import current_platform
assert current_platform.is_cuda(), current_platform
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
for device in range(4):
    assert torch.ones(16, device=f"cuda:{device}").sum().item() == 16
print("GENERATION_GPU_SMOKE_OK", sys.executable, torch.__version__, vllm.__version__, importlib.metadata.version("flashinfer-python"), type(current_platform).__name__)
PY
done
echo 'SMOKE_COMPLETE: native imports and CUDA only; no model execution or old-overlay compatibility claim'
