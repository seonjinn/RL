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
#SBATCH --time=02:00:00

set -euo pipefail

: "${CONTAINER:?CONTAINER is required}"
: "${DRIVER_VENV:?DRIVER_VENV is required}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR is required}"
: "${NEMO_RL_VENV_DIR:?NEMO_RL_VENV_DIR is required}"
: "${HYBRID_EP_PROJECT_ROOT:?HYBRID_EP_PROJECT_ROOT is required}"

for shared_path_name in DRIVER_VENV UV_CACHE_DIR NEMO_RL_VENV_DIR; do
  shared_path=${!shared_path_name}
  shared_path=$(python3 -c \
    'import os, sys; print(os.path.realpath(sys.argv[1]))' "${shared_path}")
  printf -v "${shared_path_name}" '%s' "${shared_path}"
  case "${shared_path}" in
    /lustre/*) ;;
    *)
      printf '%s must be on shared /lustre storage: %s\n' \
        "${shared_path_name}" "${shared_path}" >&2
      exit 2
      ;;
  esac
done

UV_PYTHON_INSTALL_DIR="${NEMO_RL_VENV_DIR}/.uv-python"
UV_PYTHON_INSTALL_DIR=$(python3 -c \
  'import os, sys; print(os.path.realpath(sys.argv[1]))' \
  "${UV_PYTHON_INSTALL_DIR}")
case "${UV_PYTHON_INSTALL_DIR}" in
  "${NEMO_RL_VENV_DIR}"/*) ;;
  *)
    printf 'UV_PYTHON_INSTALL_DIR must resolve under NEMO_RL_VENV_DIR: %s\n' \
      "${UV_PYTHON_INSTALL_DIR}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT=$(readlink -f -- "${HYBRID_EP_PROJECT_ROOT}")
if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" || ! -f "${PROJECT_ROOT}/uv.lock" ]]; then
  printf 'NeMo-RL project files are missing from %s.\n' "${PROJECT_ROOT}" >&2
  exit 2
fi

mkdir -p \
  "$(dirname -- "${DRIVER_VENV}")" \
  "${UV_CACHE_DIR}" \
  "${NEMO_RL_VENV_DIR}" \
  "${UV_PYTHON_INSTALL_DIR}"

MOUNTS="${PROJECT_ROOT}:${PROJECT_ROOT},/lustre:/lustre"
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
fi

srun \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS}" \
  --container-workdir="${PROJECT_ROOT}" \
  env \
    DRIVER_VENV="${DRIVER_VENV}" \
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
    NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR}" \
    UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR}" \
    bash -lc '
    set -euo pipefail
    printf "Effective DRIVER_VENV=%s\\n" "${DRIVER_VENV}"
    printf "Effective UV_CACHE_DIR=%s\\n" "${UV_CACHE_DIR}"
    printf "Effective NEMO_RL_VENV_DIR=%s\\n" "${NEMO_RL_VENV_DIR}"
    printf "Effective UV_PYTHON_INSTALL_DIR=%s\\n" "${UV_PYTHON_INSTALL_DIR}"
    export UV_PROJECT_ENVIRONMENT="${DRIVER_VENV}"
    export UV_CACHE_DIR
    export NEMO_RL_VENV_DIR
    export UV_PYTHON_INSTALL_DIR
    uv sync --frozen
    sed -i \
      "s/context\\.py_executable = \" \"\\.join(self\\.nsight_cmd) + \" python\"/context.py_executable = \" \".join(self.nsight_cmd) + f\" {context.py_executable}\"/g" \
      "${DRIVER_VENV}"/lib/python*/site-packages/ray/_private/runtime_env/nsight.py
    export PATH="${DRIVER_VENV}/bin:${PATH}"
    ray --version
    python -c "import ray; print(ray.__version__)"
    for actor_fqn in \
      nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker \
      nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker; do
      actor_python="${NEMO_RL_VENV_DIR}/${actor_fqn}/bin/python"
      if [[ -e "${actor_python}" || -L "${actor_python}" ]]; then
        resolved_actor_python=$(readlink -f -- "${actor_python}" 2>/dev/null || true)
        case "${resolved_actor_python}" in
          "${UV_PYTHON_INSTALL_DIR}"/*) ;;
          *)
            printf "Stale actor environment uses unmanaged Python: %s -> %s\\n" \
              "${actor_python}" "${resolved_actor_python:-unresolved}" >&2
            printf "Quarantine or remove the actor environment at %s and rerun preparation.\\n" \
              "${NEMO_RL_VENV_DIR}/${actor_fqn}" >&2
            exit 2
            ;;
        esac
      fi
      if ! prefetch_output=$(python -m nemo_rl.utils.prefetch_venvs "${actor_fqn}" 2>&1); then
        printf "%s\\n" "${prefetch_output}"
        printf "Actor environment prefetch failed: %s\\n" "${actor_fqn}" >&2
        exit 2
      fi
      printf "%s\\n" "${prefetch_output}"
      if grep -Eq "^[[:space:]]*Failed: [1-9][0-9]*$" <<< "${prefetch_output}"; then
        printf "Actor environment prefetch failed: %s\\n" "${actor_fqn}" >&2
        exit 2
      fi
      if [[ ! -x "${actor_python}" ]]; then
        printf "Missing prefetched actor interpreter: %s\\n" "${actor_python}" >&2
        exit 2
      fi
      case "${actor_fqn}" in
        *VllmGenerationWorker) required_import=vllm ;;
        *MegatronPolicyWorker) required_import=megatron.core ;;
      esac
      if ! "${actor_python}" -c "import ${required_import}"; then
        printf "Actor environment validation failed: %s cannot import %s\\n" \
          "${actor_python}" "${required_import}" >&2
        exit 2
      fi
      resolved_actor_python=$(readlink -f -- "${actor_python}" 2>/dev/null || true)
      case "${resolved_actor_python}" in
        "${UV_PYTHON_INSTALL_DIR}"/*) ;;
        *)
          printf "Actor environment uses unmanaged Python: %s -> %s; expected a target under %s.\\n" \
            "${actor_python}" "${resolved_actor_python:-unresolved}" \
            "${UV_PYTHON_INSTALL_DIR}" >&2
          exit 2
          ;;
      esac
    done
  '
