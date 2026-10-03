#!/usr/bin/env bash

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to the staged nightly image}"
ALIGNED_RUNTIME_REQUIRED=${ALIGNED_RUNTIME_REQUIRED:-0}
export ALIGNED_RUNTIME_REQUIRED
SRUN_BIN=${SRUN_BIN:-/cm/local/apps/slurm/current/bin/srun}
[[ -x "${SRUN_BIN}" ]] || { echo "Missing srun: ${SRUN_BIN}" >&2; exit 127; }

"${SRUN_BIN}" --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-env=ALIGNED_RUNTIME_REQUIRED \
  --container-mounts="/raid/scratch:/raid/scratch,/lustre:/lustre" \
  bash -s <<'BASH'
set -euo pipefail

scratch="/raid/scratch/${USER}/aligned-image-smoke-${SLURM_JOB_ID}"
mkdir -p "${scratch}"
export HOME="${scratch}/home" TMPDIR="${scratch}/tmp"
export UV_CACHE_DIR="${scratch}/uv" TRITON_CACHE_DIR="${scratch}/triton"
export TORCHINDUCTOR_CACHE_DIR="${scratch}/inductor" VLLM_CACHE_ROOT="${scratch}/vllm"
mkdir -p "${HOME}" "${TMPDIR}"
trap 'rm -rf "${scratch}"' EXIT
if [[ "${ALIGNED_RUNTIME_REQUIRED}" == 1 ]]; then
  [[ -s /opt/nemo_rl_aligned_source.json ]]
  /opt/nemo_rl_venv/bin/python \
    /opt/nemo-rl/experiments/mxfp8_vllm029_fullperf_20260930/check_aligned_runtime.py \
    --output "${scratch}/runtime.json"
  /opt/nemo_rl_venv/bin/ray --version
fi

VLLM_PY=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
MCORE_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
[[ -x "${VLLM_PY}" ]]
[[ -x "${MCORE_PY}" ]]

"${VLLM_PY}" - <<'PY'
import importlib.metadata

import torch
import vllm

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
assert importlib.metadata.version("vllm") == "0.29.0"

print(f"torch={torch.__version__}")
print(f"vllm={vllm.__version__}")
print(f"gpus={torch.cuda.device_count()}")
PY

"${MCORE_PY}" - <<'PY'
import importlib

import torch
import transformer_engine.pytorch
from megatron import core

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
for package in ("causal_conv1d", "cupy", "mamba_ssm"):
    importlib.import_module(package)

print(f"transformer_engine={transformer_engine.pytorch.__file__}")
print(f"megatron_core={core.__file__}")
PY
BASH
