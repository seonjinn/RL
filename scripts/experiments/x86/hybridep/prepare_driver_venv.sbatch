#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

set -euo pipefail

: "${CONTAINER:?CONTAINER is required}"
: "${DRIVER_VENV:?DRIVER_VENV is required}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR is required}"
: "${HYBRID_EP_PROJECT_ROOT:?HYBRID_EP_PROJECT_ROOT is required}"

case "${DRIVER_VENV}" in
  /lustre/* | /home/*) ;;
  *)
    printf 'DRIVER_VENV must be on shared /lustre or /home storage: %s\n' \
      "${DRIVER_VENV}" >&2
    exit 2
    ;;
esac

case "${UV_CACHE_DIR}" in
  /lustre/* | /home/*) ;;
  *)
    printf 'UV_CACHE_DIR must be on shared /lustre or /home storage: %s\n' \
      "${UV_CACHE_DIR}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT=$(readlink -f -- "${HYBRID_EP_PROJECT_ROOT}")
if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" || ! -f "${PROJECT_ROOT}/uv.lock" ]]; then
  printf 'NeMo-RL project files are missing from %s.\n' "${PROJECT_ROOT}" >&2
  exit 2
fi

mkdir -p "$(dirname -- "${DRIVER_VENV}")" "${UV_CACHE_DIR}"

MOUNTS="${PROJECT_ROOT}:${PROJECT_ROOT},/lustre:/lustre,/home:/home"
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
fi

srun \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc '
    set -euo pipefail
    export UV_PROJECT_ENVIRONMENT="${DRIVER_VENV}"
    export UV_CACHE_DIR="${UV_CACHE_DIR}"
    uv sync --frozen
    sed -i \
      "s/context\\.py_executable = \" \"\\.join(self\\.nsight_cmd) + \" python\"/context.py_executable = \" \".join(self.nsight_cmd) + f\" {context.py_executable}\"/g" \
      "${DRIVER_VENV}"/lib/python*/site-packages/ray/_private/runtime_env/nsight.py
    export PATH="${DRIVER_VENV}/bin:${PATH}"
    ray --version
    python -c "import ray; print(ray.__version__)"
  '
