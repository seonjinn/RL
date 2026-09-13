#!/bin/bash
set -euo pipefail

# Run inside the nightly container in a dedicated GB200 allocation.
: "${SOURCE_REPO:?Set the frozen source repository}"
: "${SOURCE_SHA:?Set the tested source revision}"
: "${BRIDGE_SHA:?Set the tested Bridge revision}"
ROOT="/raid/scratch/${USER}/native-mixed-views/${SLURM_JOB_ID}"
BRIDGE="${SOURCE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
MCORE="${BRIDGE}/3rdparty/Megatron-LM"
PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
mkdir -p "${ROOT}/source" "${ROOT}/bridge" "${ROOT}/mcore"
git -C "${SOURCE_REPO}" archive "${SOURCE_SHA}" nemo_rl tests/unit/models/megatron/test_group_experts.py pyproject.toml | tar -xf - -C "${ROOT}/source"
git -C "${BRIDGE}" archive "${BRIDGE_SHA}" src | tar -xf - -C "${ROOT}/bridge"
MCORE_SHA=$(git -C "${BRIDGE}" ls-tree "${BRIDGE_SHA}" 3rdparty/Megatron-LM | awk '{print $3}')
git -C "${MCORE}" archive "${MCORE_SHA}" megatron | tar -xf - -C "${ROOT}/mcore"
export UV_CACHE_DIR="/raid/scratch/${USER}/uv-cache"
export PYTHONPYCACHEPREFIX="${ROOT}/pycache"
export TRITON_CACHE_DIR="${ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${ROOT}/extensions"
export PYTHONPATH="${ROOT}/deps:${ROOT}/source:${ROOT}/bridge/src:${ROOT}/mcore"
uv pip install --python "${PYTHON}" --target "${ROOT}/deps" pytest
cd "${ROOT}/source"
printf 'SOURCE_SHA=%s\nBRIDGE_SHA=%s\nMCORE_SHA=%s\n' "${SOURCE_SHA}" "${BRIDGE_SHA}" "${MCORE_SHA}"
"${PYTHON}" -P -c 'import torch, megatron.bridge; print(torch.cuda.get_device_name()); print(megatron.bridge.__file__)'
"${PYTHON}" -P -m pytest -q --tb=short -o addopts= tests/unit/models/megatron/test_group_experts.py
