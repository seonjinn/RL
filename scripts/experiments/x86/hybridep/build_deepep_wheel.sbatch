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
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=hybridep-x86-wheel

set -euo pipefail

: "${CONTAINER:?CONTAINER is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${GPU_ARCH:?GPU_ARCH is required}"
: "${DEEPEP_COMMIT:?DEEPEP_COMMIT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
if [[ -z "${HYBRID_EP_SCRIPT_PATH:-}" ]]; then
  : "${SLURM_SUBMIT_DIR:?HYBRID_EP_SCRIPT_PATH or SLURM_SUBMIT_DIR is required}"
  HYBRID_EP_SCRIPT_PATH="${SLURM_SUBMIT_DIR}/scripts/experiments/x86/hybridep/build_deepep_wheel.sbatch"
fi
: "${HYBRID_EP_SCRIPT_PATH:?HYBRID_EP_SCRIPT_PATH is required}"

[[ "${DEEPEP_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
case "${GPU_ARCH}" in
  9.0 | 10.0) ;;
  *)
    printf 'GPU_ARCH must be 9.0 for H100 or 10.0 for B200: %s\n' \
      "${GPU_ARCH}" >&2
    exit 2
    ;;
esac

HYBRID_EP_TRANSPORT=${HYBRID_EP_TRANSPORT:-doca}
case "${HYBRID_EP_TRANSPORT}" in
  doca | nixl) ;;
  *)
    printf 'HYBRID_EP_TRANSPORT must be doca or nixl: %s\n' \
      "${HYBRID_EP_TRANSPORT}" >&2
    exit 2
    ;;
esac

PYTHON_BIN=${PYTHON_BIN:-/opt/nemo_rl_venv/bin/python}
DEEPEP_REPO_URL=${DEEPEP_REPO_URL:-https://github.com/deepseek-ai/DeepEP.git}
SCRIPT_PATH=$(readlink -f -- "${HYBRID_EP_SCRIPT_PATH}")
CONTAINER=$(readlink -f -- "${CONTAINER}")

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  printf 'Original build script does not exist: %s\n' \
    "${HYBRID_EP_SCRIPT_PATH}" >&2
  exit 2
fi

for shared_path in "${SCRIPT_PATH}" "${CONTAINER}" "${OUTPUT_DIR}"; do
  case "${shared_path}" in
    /lustre/*) ;;
    *)
      printf 'Shared paths must be under /lustre: %s\n' "${shared_path}" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CONTAINER}" ]]; then
  printf 'Container does not exist: %s\n' "${CONTAINER}" >&2
  exit 2
fi

LOCAL_SCRATCH_ROOT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
LOCAL_SCRATCH_ROOT=$(readlink -f -- "${LOCAL_SCRATCH_ROOT}")
if [[ ! -d "${LOCAL_SCRATCH_ROOT}" || ! -w "${LOCAL_SCRATCH_ROOT}" ]]; then
  printf 'Local scratch root must be a writable directory: %s\n' \
    "${LOCAL_SCRATCH_ROOT}" >&2
  exit 2
fi

BUILD_ROOT_EPHEMERAL=${BUILD_ROOT_EPHEMERAL:-0}
if [[ -z "${BUILD_ROOT:-}" ]]; then
  BUILD_ROOT=$(mktemp -d "${LOCAL_SCRATCH_ROOT}/nemo-rl-hybridep-${SLURM_JOB_ID}.XXXXXX")
  BUILD_ROOT_EPHEMERAL=1
else
  BUILD_ROOT=$(readlink -m -- "${BUILD_ROOT}")
fi

case "${BUILD_ROOT}" in
  /lustre/*) ;;
  *)
    case "${BUILD_ROOT_EPHEMERAL}:${BUILD_ROOT}" in
      "1:${LOCAL_SCRATCH_ROOT}/nemo-rl-hybridep-${SLURM_JOB_ID}."*) ;;
      *)
        printf 'Non-Lustre BUILD_ROOT must be a job-local mktemp path: %s\n' \
          "${BUILD_ROOT}" >&2
        exit 2
        ;;
    esac
    ;;
esac

mkdir -p "${OUTPUT_DIR}" "${BUILD_ROOT}"
container_sha256=$(sha256sum "${CONTAINER}" | cut -d' ' -f1)
export CONTAINER
export OUTPUT_DIR
export BUILD_ROOT
export GPU_ARCH
export DEEPEP_COMMIT
export HYBRID_EP_TRANSPORT
export PYTHON_BIN
export DEEPEP_REPO_URL
export SCRIPT_PATH
export HYBRID_EP_SCRIPT_PATH="${SCRIPT_PATH}"
export CONTAINER_SHA256="${container_sha256}"
export LOCAL_SCRATCH_ROOT
export BUILD_ROOT_EPHEMERAL

if [[ "${HYBRID_EP_BUILD_STAGE:-host}" != "container" ]]; then
  container_mounts="/lustre:/lustre"
  if [[ "${BUILD_ROOT}" != /lustre/* ]]; then
    container_mounts="${container_mounts},${BUILD_ROOT}:${BUILD_ROOT}"
  fi
  srun_status=0
  srun --container-image="${CONTAINER}" \
    --no-container-mount-home \
    --container-mounts="${container_mounts}" \
    --export=ALL,HYBRID_EP_BUILD_STAGE=container \
    bash "${SCRIPT_PATH}" || srun_status=$?
  if [[ "${BUILD_ROOT_EPHEMERAL}" == "1" ]]; then
    case "${BUILD_ROOT}" in
      "${LOCAL_SCRATCH_ROOT}"/nemo-rl-hybridep-"${SLURM_JOB_ID}".*)
        rmdir -- "${BUILD_ROOT}" || {
          printf 'Failed to remove empty job-local build root: %s\n' \
            "${BUILD_ROOT}" >&2
          if [[ "${srun_status}" == "0" ]]; then
            srun_status=1
          fi
        }
        ;;
      *)
        printf 'Refusing to clean unexpected host build root: %s\n' \
          "${BUILD_ROOT}" >&2
        if [[ "${srun_status}" == "0" ]]; then
          srun_status=1
        fi
        ;;
    esac
  fi
  exit "${srun_status}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python interpreter does not exist in the container: %s\n' \
    "${PYTHON_BIN}" >&2
  exit 2
fi
if ! command -v uv >/dev/null; then
  printf 'uv is required in the build container.\n' >&2
  exit 2
fi

export HYBRID_EP_MULTINODE=1
export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
if [[ "${HYBRID_EP_TRANSPORT}" == "nixl" ]]; then
  : "${NIXL_HOME:?NIXL_HOME is required for NIXL builds}"
  : "${UCX_HOME:?UCX_HOME is required for NIXL builds}"
  export USE_NIXL=1
else
  unset USE_NIXL
  export RDMA_CORE_HOME=${RDMA_CORE_HOME:-/usr}
fi

gpu_capability=$(
  "${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(); print(".".join(map(str, torch.cuda.get_device_capability(0))))'
)
if [[ "${gpu_capability}" != "${GPU_ARCH}" ]]; then
  printf 'Allocated GPU capability %s does not match GPU_ARCH %s.\n' \
    "${gpu_capability}" "${GPU_ARCH}" >&2
  exit 2
fi

short_commit=${DEEPEP_COMMIT:0:12}
arch_tag=${GPU_ARCH//./}
artifact_dir="${OUTPUT_DIR}/${DEEPEP_COMMIT}-sm${arch_tag}-${SLURM_JOB_ID}"
if [[ -e "${artifact_dir}" ]]; then
  printf 'Refusing to overwrite immutable artifact directory: %s\n' \
    "${artifact_dir}" >&2
  exit 2
fi
artifact_stage="${OUTPUT_DIR}/.${DEEPEP_COMMIT}-sm${arch_tag}-${SLURM_JOB_ID}.staging"
if [[ -e "${artifact_stage}" ]]; then
  printf 'Refusing to overwrite artifact staging directory: %s\n' \
    "${artifact_stage}" >&2
  exit 2
fi

build_dir=$(mktemp -d \
  "${BUILD_ROOT}/deepep-${short_commit}-sm${arch_tag}-${SLURM_JOB_ID}.XXXXXX")
source_dir="${build_dir}/DeepEP"
wheel_stage="${build_dir}/wheel"
import_overlay="${build_dir}/import-overlay"
build_succeeded=0

cleanup() {
  if [[ "${BUILD_ROOT_EPHEMERAL}" == "1" ]]; then
    case "${build_dir}" in
      "${BUILD_ROOT}"/deepep-"${short_commit}"-sm"${arch_tag}"-"${SLURM_JOB_ID}".*)
        rm -rf -- "${build_dir}"
        ;;
      *)
        printf 'Refusing to clean unexpected ephemeral build directory: %s\n' \
          "${build_dir}" >&2
        ;;
    esac
  elif [[ "${build_succeeded}" == "1" ]]; then
    case "${build_dir}" in
      "${BUILD_ROOT}"/deepep-"${short_commit}"-sm"${arch_tag}"-"${SLURM_JOB_ID}".*)
        rm -rf -- "${build_dir}"
        ;;
      *)
        printf 'Refusing to clean unexpected build directory: %s\n' \
          "${build_dir}" >&2
        ;;
    esac
  else
    printf 'Preserving failed build directory for diagnosis: %s\n' \
      "${build_dir}" >&2
  fi
}
trap cleanup EXIT

git clone --filter=blob:none --recurse-submodules --branch hybrid-ep \
  "${DEEPEP_REPO_URL}" "${source_dir}"
git -C "${source_dir}" checkout --detach "${DEEPEP_COMMIT}"
git -C "${source_dir}" submodule update --init --recursive
actual_commit=$(git -C "${source_dir}" rev-parse HEAD)
if [[ "${actual_commit}" != "${DEEPEP_COMMIT}" ]]; then
  printf 'DeepEP checkout mismatch: expected %s, got %s\n' \
    "${DEEPEP_COMMIT}" "${actual_commit}" >&2
  exit 2
fi

mkdir -p "${wheel_stage}" "${import_overlay}"
unset UV_CONFIG_FILE
UV_NO_CONFIG=1 uv build --wheel --no-build-isolation \
  --python "${PYTHON_BIN}" \
  --out-dir "${wheel_stage}" \
  "${source_dir}"

mapfile -t built_wheels < <(find "${wheel_stage}" -maxdepth 1 -type f -name '*.whl')
if [[ "${#built_wheels[@]}" != "1" ]]; then
  printf 'Expected exactly one wheel, found %s.\n' "${#built_wheels[@]}" >&2
  exit 2
fi
built_wheel=${built_wheels[0]}

UV_NO_CONFIG=1 uv pip install \
  --python "${PYTHON_BIN}" \
  --target "${import_overlay}" \
  --reinstall \
  --no-deps \
  --no-index \
  "${built_wheel}"
PYTHONPATH="${import_overlay}" "${PYTHON_BIN}" -c \
  'import deep_ep, deep_ep_cpp, hybrid_ep_cpp; print(deep_ep.__file__); print(deep_ep_cpp.__file__); print(hybrid_ep_cpp.__file__)'

mkdir "${artifact_stage}"
final_wheel="${artifact_dir}/$(basename "${built_wheel}")"
staged_wheel="${artifact_stage}/$(basename "${built_wheel}")"
if [[ -e "${final_wheel}" ]]; then
  printf 'Refusing to overwrite immutable wheel: %s\n' "${final_wheel}" >&2
  exit 2
fi
mv "${built_wheel}" "${staged_wheel}"
wheel_sha256=$(sha256sum "${staged_wheel}" | cut -d' ' -f1)
printf '%s  %s\n' "${wheel_sha256}" "$(basename "${staged_wheel}")" \
  > "${staged_wheel}.sha256"

metadata_path="${artifact_stage}/metadata.env"
{
  printf 'deepep_commit=%q\n' "${DEEPEP_COMMIT}"
  printf 'deepep_repo=%q\n' "${DEEPEP_REPO_URL}"
  printf 'gpu_arch=%q\n' "${GPU_ARCH}"
  printf 'hybrid_ep_multinode=1\n'
  printf 'hybrid_ep_transport=%q\n' "${HYBRID_EP_TRANSPORT}"
  printf 'container=%q\n' "${CONTAINER}"
  printf 'container_sha256=%q\n' "${CONTAINER_SHA256}"
  printf 'python_bin=%q\n' "${PYTHON_BIN}"
  printf 'slurm_job_id=%q\n' "${SLURM_JOB_ID}"
  printf 'wheel=%q\n' "${final_wheel}"
  printf 'wheel_sha256=%q\n' "${wheel_sha256}"
  printf 'built_at=%q\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "${metadata_path}"

mv "${artifact_stage}" "${artifact_dir}"
build_succeeded=1
printf 'Built immutable HybridEP wheel: %s\nMetadata: %s\n' \
  "${final_wheel}" "${artifact_dir}/metadata.env"
