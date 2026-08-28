#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=hybridep-h100-wheel

set -euo pipefail

: "${SOURCE_DIR:?SOURCE_DIR is required.}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required.}"
: "${DEEPEP_COMMIT:?DEEPEP_COMMIT is required.}"
: "${DEEPEP_EXPECTED_VERSION:?DEEPEP_EXPECTED_VERSION is required.}"
: "${GPU_ARCH:?GPU_ARCH is required.}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required.}"

PYTHON_BIN=${PYTHON_BIN:-/opt/nemo_rl_venv/bin/python}
SOURCE_ROOT_PREFIX=${SOURCE_ROOT_PREFIX:-/home}
OUTPUT_ROOT_PREFIX=${OUTPUT_ROOT_PREFIX:-/lustre}
DEEPEP_BUILD_STAGE=${DEEPEP_BUILD_STAGE:-host}
if [[ "${DEEPEP_BUILD_STAGE}" == container && -n "${BUILD_SOURCE_DIR:-}" ]]; then
  : "${SLURM_TMPDIR:?SLURM_TMPDIR is required in the container stage.}"
  SOURCE_DIR=${BUILD_SOURCE_DIR}
  SOURCE_ROOT_PREFIX=${SLURM_TMPDIR}
fi

SOURCE_DIR=$(realpath "${SOURCE_DIR}")
OUTPUT_DIR=$(realpath "${OUTPUT_DIR}")
SOURCE_ROOT_PREFIX=$(realpath "${SOURCE_ROOT_PREFIX}")
OUTPUT_ROOT_PREFIX=$(realpath "${OUTPUT_ROOT_PREFIX}")

case "${SOURCE_DIR}" in
  "${SOURCE_ROOT_PREFIX}"/*) ;;
  *)
    printf 'SOURCE_DIR must be under %s: %s\n' \
      "${SOURCE_ROOT_PREFIX}" "${SOURCE_DIR}" >&2
    exit 2
    ;;
esac
case "${OUTPUT_DIR}" in
  "${OUTPUT_ROOT_PREFIX}"/*) ;;
  *)
    printf 'OUTPUT_DIR must be under %s: %s\n' \
      "${OUTPUT_ROOT_PREFIX}" "${OUTPUT_DIR}" >&2
    exit 2
    ;;
esac
[[ "${DEEPEP_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${GPU_ARCH}" == 9.0 ]]

actual_commit=$(git -C "${SOURCE_DIR}" rev-parse HEAD)
if [[ "${actual_commit}" != "${DEEPEP_COMMIT}" ]]; then
  printf 'DeepEP checkout mismatch: expected %s, got %s\n' \
    "${DEEPEP_COMMIT}" "${actual_commit}" >&2
  exit 2
fi
if [[ -n "$(git -C "${SOURCE_DIR}" status --porcelain)" ]]; then
  printf 'DeepEP checkout must be clean: %s\n' "${SOURCE_DIR}" >&2
  exit 2
fi
submodule_state=$(git -C "${SOURCE_DIR}" submodule status --recursive)
if grep -Eq '^[-+U]' <<< "${submodule_state}"; then
  printf 'DeepEP submodules must be initialized at their pinned commits.\n' >&2
  exit 2
fi

if [[ "${DEEPEP_BUILD_STAGE}" == host ]]; then
  : "${CONTAINER:?CONTAINER is required.}"
  : "${SLURM_TMPDIR:=/raid/scratch/${USER}}"
  CONTAINER=$(realpath "${CONTAINER}")
  [[ "${CONTAINER}" == /lustre/* && -f "${CONTAINER}" ]]
  mkdir -p "${SLURM_TMPDIR}"
  BUILD_ROOT=${BUILD_ROOT:-"${SLURM_TMPDIR}/nemo-rl-deepep-${DEEPEP_COMMIT:0:12}-${SLURM_JOB_ID}"}
  case "${BUILD_ROOT}" in
    "${SLURM_TMPDIR}"/*) ;;
    *)
      printf 'BUILD_ROOT must be under node-local SLURM_TMPDIR: %s\n' \
        "${BUILD_ROOT}" >&2
      exit 2
      ;;
  esac
  if [[ -e "${BUILD_ROOT}" ]]; then
    printf 'Refusing to reuse build root: %s\n' "${BUILD_ROOT}" >&2
    exit 2
  fi
  mkdir -p "${BUILD_ROOT}/source"
  cp -a "${SOURCE_DIR}/." "${BUILD_ROOT}/source/"
  cp "$0" "${BUILD_ROOT}/builder.sbatch"

  export BUILD_ROOT
  export BUILD_SOURCE_DIR="${BUILD_ROOT}/source"
  export DEEPEP_BUILD_STAGE=container
  export SOURCE_DIR OUTPUT_DIR DEEPEP_COMMIT DEEPEP_EXPECTED_VERSION GPU_ARCH
  export PYTHON_BIN SOURCE_ROOT_PREFIX OUTPUT_ROOT_PREFIX SLURM_TMPDIR
  container_env_vars=BUILD_ROOT,BUILD_SOURCE_DIR,DEEPEP_BUILD_STAGE,SOURCE_DIR,OUTPUT_DIR,DEEPEP_COMMIT,DEEPEP_EXPECTED_VERSION,GPU_ARCH,PYTHON_BIN,SOURCE_ROOT_PREFIX,OUTPUT_ROOT_PREFIX,SLURM_TMPDIR
  srun \
    --container-image="${CONTAINER}" \
    --no-container-mount-home \
    --container-mounts="/lustre:/lustre,${BUILD_ROOT}:${BUILD_ROOT}" \
    --container-env="${container_env_vars}" \
    bash "${BUILD_ROOT}/builder.sbatch"
  rm -rf -- "${BUILD_ROOT}"
  exit 0
fi

if [[ "${DEEPEP_BUILD_STAGE}" != container ]]; then
  printf 'DEEPEP_BUILD_STAGE must be host or container.\n' >&2
  exit 2
fi

BUILD_SOURCE_DIR=${BUILD_SOURCE_DIR:-${SOURCE_DIR}}
BUILD_ROOT=${BUILD_ROOT:?BUILD_ROOT is required in the container stage.}
: "${SLURM_TMPDIR:?SLURM_TMPDIR is required in the container stage.}"
case "${BUILD_ROOT}" in
  "${SLURM_TMPDIR}"/*) ;;
  *)
    printf 'BUILD_ROOT must be under node-local SLURM_TMPDIR: %s\n' \
      "${BUILD_ROOT}" >&2
    exit 2
    ;;
esac

actual_commit=$(git -C "${BUILD_SOURCE_DIR}" rev-parse HEAD)
if [[ "${actual_commit}" != "${DEEPEP_COMMIT}" ]]; then
  printf 'Staged DeepEP checkout mismatch: expected %s, got %s\n' \
    "${DEEPEP_COMMIT}" "${actual_commit}" >&2
  exit 2
fi

gpu_capability=$("${PYTHON_BIN}" -c \
  'import torch; assert torch.cuda.is_available(); print(".".join(map(str, torch.cuda.get_device_capability(0))))')
if [[ "${gpu_capability}" != "${GPU_ARCH}" ]]; then
  printf 'Allocated GPU capability %s does not match GPU_ARCH %s.\n' \
    "${gpu_capability}" "${GPU_ARCH}" >&2
  exit 2
fi

wheel_stage="${BUILD_ROOT}/wheel"
overlay="${BUILD_ROOT}/overlay"
mkdir -p "${wheel_stage}" "${overlay}"
export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
export HYBRID_EP_MULTINODE=1
export RDMA_CORE_HOME=${RDMA_CORE_HOME:-/usr}
unset UV_CONFIG_FILE CONDA_PREFIX VIRTUAL_ENV

uv build --no-config --wheel --no-build-isolation \
  --python "${PYTHON_BIN}" \
  --out-dir "${wheel_stage}" \
  "${BUILD_SOURCE_DIR}"

wheels=()
while IFS= read -r wheel_path; do
  wheels+=("${wheel_path}")
done < <(find "${wheel_stage}" -maxdepth 1 -type f -name '*.whl' -print)
if [[ "${#wheels[@]}" != 1 ]]; then
  printf 'Expected exactly one wheel, found %s.\n' "${#wheels[@]}" >&2
  exit 2
fi
wheel=${wheels[0]}
wheel_entries=$(unzip -Z1 "${wheel}")
if ! grep -Eq '^deep_ep_cpp.*\.so$' <<< "${wheel_entries}"; then
  printf 'Built wheel is missing deep_ep_cpp extension: %s\n' "${wheel}" >&2
  exit 2
fi
if ! grep -Eq '^hybrid_ep_cpp.*\.so$' <<< "${wheel_entries}"; then
  printf 'Built wheel is missing hybrid_ep_cpp extension: %s\n' "${wheel}" >&2
  exit 2
fi

uv pip install --no-config --python "${PYTHON_BIN}" --target "${overlay}" \
  --no-deps --reinstall "${wheel}"
PYTHONPATH="${overlay}" "${PYTHON_BIN}" -c \
  'import importlib.metadata as md, os; import deep_ep_cpp, hybrid_ep_cpp; from deep_ep import Buffer, HybridEPBuffer; version = md.version("deep_ep"); print("DEEPEP_BUILD_VERSION", version); print("DEEPEP_BUILD_EXTENSIONS", deep_ep_cpp.__file__, hybrid_ep_cpp.__file__); assert version == os.environ["DEEPEP_EXPECTED_VERSION"], version'

artifact_name="hybridep-${DEEPEP_COMMIT}-sm${GPU_ARCH//./}-${SLURM_JOB_ID}"
artifact_dir="${OUTPUT_DIR}/${artifact_name}"
artifact_stage="${OUTPUT_DIR}/.${artifact_name}.staging"
if [[ -e "${artifact_dir}" || -e "${artifact_stage}" ]]; then
  printf 'Refusing to overwrite immutable artifact: %s\n' "${artifact_dir}" >&2
  exit 2
fi
mkdir "${artifact_stage}"
staged_wheel="${artifact_stage}/$(basename "${wheel}")"
cp "${wheel}" "${staged_wheel}"
if command -v sha256sum >/dev/null; then
  wheel_sha256=$(sha256sum "${staged_wheel}" | cut -d' ' -f1)
else
  wheel_sha256=$(shasum -a 256 "${staged_wheel}" | cut -d' ' -f1)
fi
printf '%s  %s\n' "${wheel_sha256}" "$(basename "${staged_wheel}")" \
  > "${staged_wheel}.sha256"
{
  printf 'deepep_commit=%q\n' "${DEEPEP_COMMIT}"
  printf 'deepep_version=%q\n' "${DEEPEP_EXPECTED_VERSION}"
  printf 'gpu_arch=%q\n' "${GPU_ARCH}"
  printf 'hybrid_ep_multinode=1\n'
  printf 'source_commit=%q\n' "${actual_commit}"
  printf 'wheel_sha256=%q\n' "${wheel_sha256}"
  printf 'slurm_job_id=%q\n' "${SLURM_JOB_ID}"
  printf 'built_at=%q\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "${artifact_stage}/metadata.env"
mv "${artifact_stage}" "${artifact_dir}"
printf 'DEEPEP_WHEEL=%s\n' "${artifact_dir}/$(basename "${wheel}")"
