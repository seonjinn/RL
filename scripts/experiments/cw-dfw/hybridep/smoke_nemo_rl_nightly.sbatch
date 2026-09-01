#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --job-name=smoke_nemo_rl_nightly
#SBATCH --output=smoke_nemo_rl_nightly_%j.log

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to an immutable NeMo-RL sqsh image.}"
: "${DEEPEP_WHEEL:?Set DEEPEP_WHEEL to an x86_64 H100 wheel.}"
: "${EXPECTED_DEEPEP_VERSION:?Set EXPECTED_DEEPEP_VERSION.}"

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)
BRIDGE_DIR=${PROJECT_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MCORE_DIR=${BRIDGE_DIR}/3rdparty/Megatron-LM

for path in "${CONTAINER}" "${DEEPEP_WHEEL}"; do
  if [[ "${path}" != /lustre/* || ! -f "${path}" ]]; then
    printf 'Expected a readable Lustre file: %q\n' "${path}" >&2
    exit 2
  fi
done

export CONTAINER DEEPEP_WHEEL EXPECTED_DEEPEP_VERSION PROJECT_ROOT BRIDGE_DIR MCORE_DIR

# shellcheck disable=SC2016
srun \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT},/lustre:/lustre,/raid/scratch:/raid/scratch" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -c '
    set -euo pipefail
    gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    gpu_count=$(sed "/^$/d" <<< "${gpu_names}" | wc -l)
    if [[ "${gpu_count}" != 1 ]] || ! grep -q H100 <<< "${gpu_names}"; then
      printf "Expected one visible H100; observed count=%s names=%q\n" \
        "${gpu_count}" "${gpu_names}" >&2
      exit 2
    fi

    overlay_dir=$(mktemp -d /raid/scratch/deepep-smoke.XXXXXX)
    cleanup() {
      rm -rf -- "${overlay_dir}"
    }
    trap cleanup EXIT

    python_bin=$(command -v python)
    uv pip install \
      --no-config \
      --python "${python_bin}" \
      --target "${overlay_dir}" \
      --no-deps \
      --reinstall \
      "${DEEPEP_WHEEL}"

    PYTHONPATH="${overlay_dir}:${PROJECT_ROOT}:${BRIDGE_DIR}/src:${MCORE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" -c "
import importlib.metadata as metadata
import os

import torch
import hybrid_ep_cpp
import megatron.core
import nemo.lens
import transformer_engine.pytorch

version = metadata.version(\"deep_ep\")
assert version == os.environ[\"EXPECTED_DEEPEP_VERSION\"], version
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print(\"NIGHTLY_GPU_SMOKE_OK\", torch.cuda.get_device_name(0), version)
"
  '
