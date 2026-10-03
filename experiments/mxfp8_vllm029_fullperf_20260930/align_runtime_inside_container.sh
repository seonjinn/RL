#!/usr/bin/env bash

set -euo pipefail

: "${ALIGN_INPUT:?Set ALIGN_INPUT to the read-only source payload}"
: "${ALIGN_SCRATCH:?Set ALIGN_SCRATCH to node-local build storage}"

unset WANDB_API_KEY HF_TOKEN GH_TOKEN GITHUB_TOKEN SSH_AUTH_SOCK
export HOME="${ALIGN_SCRATCH}/home"
export TMPDIR="${ALIGN_SCRATCH}/tmp"
export UV_CACHE_DIR="${ALIGN_SCRATCH}/uv"
export UV_PYTHON_INSTALL_DIR=/opt/uv/python
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=copy
export MAX_JOBS=8
export CMAKE_BUILD_PARALLEL_LEVEL=8
export TORCH_CUDA_ARCH_LIST=10.0
export PATH="/opt/nemo_rl_venv/bin:${PATH}"
mkdir -p "${HOME}" "${TMPDIR}" "${UV_CACHE_DIR}"

tar -xf "${ALIGN_INPUT}/source.tar" -C "${ALIGN_SCRATCH}/source"
rsync -a --delete --exclude=.venv --exclude=3rdparty/vllm \
  "${ALIGN_SCRATCH}/source/" /opt/nemo-rl/
cd /opt/nemo-rl
checker=experiments/mxfp8_vllm029_fullperf_20260930/check_aligned_runtime.py
if /opt/nemo_rl_venv/bin/python "${checker}" \
  --output "${ALIGN_SCRATCH}/results/before.json"; then
  echo "The base image already satisfies the matrix runtime checks."
else
  echo "The base image fails runtime alignment; synchronizing the pinned lock."
fi

/opt/nemo_rl_venv/bin/python - <<'PY' >"${ALIGN_SCRATCH}/environments.tsv"
import importlib.util
from pathlib import Path

path = Path("experiments/mxfp8_vllm029_fullperf_20260930/check_aligned_runtime.py")
spec = importlib.util.spec_from_file_location("runtime_check", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for role, environment, extras in module.environments(Path.cwd()):
    print(role, environment, " ".join(f"--extra {extra}" for extra in extras), sep="\t")
PY

while IFS=$'\t' read -r role environment flags; do
  [[ -x "${environment}/bin/python" ]]
  read -r -a extra_flags <<<"${flags}"
  echo "Synchronizing ${role}: ${flags}"
  UV_PROJECT_ENVIRONMENT="${environment}" \
    timeout --kill-after=30s 3600s \
    uv sync --frozen --inexact --no-install-project "${extra_flags[@]}"
done <"${ALIGN_SCRATCH}/environments.tsv"

/opt/nemo_rl_venv/bin/python "${checker}" \
  --output "${ALIGN_SCRATCH}/results/after.json"
cp "${ALIGN_SCRATCH}/results/after.json" /opt/nemo_rl_aligned_runtime.json
cp "${ALIGN_INPUT}/fingerprint.json" /opt/nemo_rl_container_fingerprint
cp "${ALIGN_INPUT}/source-metadata.json" /opt/nemo_rl_aligned_source.json
echo "Runtime alignment passed. Other backend environments are not certified."
