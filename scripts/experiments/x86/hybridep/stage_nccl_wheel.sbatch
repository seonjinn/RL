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
#SBATCH --time=00:30:00
#SBATCH --job-name=hybridep-x86-nccl

set -euo pipefail

: "${CONTAINER:?CONTAINER is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

NCCL_PACKAGE=nvidia-nccl-cu13
NCCL_VERSION=2.30.4

if [[ -z "${HYBRID_EP_SCRIPT_PATH:-}" ]]; then
  : "${SLURM_SUBMIT_DIR:?HYBRID_EP_SCRIPT_PATH or SLURM_SUBMIT_DIR is required}"
  HYBRID_EP_SCRIPT_PATH="${SLURM_SUBMIT_DIR}/scripts/experiments/x86/hybridep/stage_nccl_wheel.sbatch"
fi
: "${HYBRID_EP_SCRIPT_PATH:?HYBRID_EP_SCRIPT_PATH is required}"

SCRIPT_PATH=$(readlink -f -- "${HYBRID_EP_SCRIPT_PATH}")
CONTAINER=$(readlink -f -- "${CONTAINER}")
OUTPUT_DIR=$(readlink -m -- "${OUTPUT_DIR}")

for shared_path in "${SCRIPT_PATH}" "${CONTAINER}" "${OUTPUT_DIR}"; do
  case "${shared_path}" in
    /lustre/*) ;;
    *)
      printf 'Shared paths must be under /lustre: %s\n' "${shared_path}" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  printf 'Original NCCL staging script does not exist: %s\n' \
    "${HYBRID_EP_SCRIPT_PATH}" >&2
  exit 2
fi
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

if [[ "${HYBRID_EP_NCCL_STAGE:-host}" != "container" ]]; then
  stage_root=$(mktemp -d \
    "${LOCAL_SCRATCH_ROOT}/nemo-rl-nccl-${SLURM_JOB_ID}.XXXXXX")
  container_sha256=$(sha256sum "${CONTAINER}" | cut -d' ' -f1)
  container_mounts="/lustre:/lustre,${stage_root}:${stage_root}"
  srun_status=0
  srun --container-image="${CONTAINER}" \
    --no-container-mount-home \
    --container-mounts="${container_mounts}" \
    --export=ALL,HYBRID_EP_NCCL_STAGE=container,CONTAINER_SHA256="${container_sha256}",NCCL_STAGE_ROOT="${stage_root}" \
    bash "${SCRIPT_PATH}" || srun_status=$?
  case "${stage_root}" in
    "${LOCAL_SCRATCH_ROOT}"/nemo-rl-nccl-"${SLURM_JOB_ID}".*)
      rm -rf -- "${stage_root}"
      ;;
    *)
      printf 'Refusing to clean unexpected job-local staging root: %s\n' \
        "${stage_root}" >&2
      if [[ "${srun_status}" == "0" ]]; then
        srun_status=1
      fi
      ;;
  esac
  exit "${srun_status}"
fi

: "${CONTAINER_SHA256:?CONTAINER_SHA256 is required in the container stage}"
: "${NCCL_STAGE_ROOT:?NCCL_STAGE_ROOT is required in the container stage}"

case "${NCCL_STAGE_ROOT}" in
  "${LOCAL_SCRATCH_ROOT}"/nemo-rl-nccl-"${SLURM_JOB_ID}".*) ;;
  *)
    printf 'NCCL_STAGE_ROOT must be a job-local staging directory: %s\n' \
      "${NCCL_STAGE_ROOT}" >&2
    exit 2
    ;;
esac

artifact_dir="${OUTPUT_DIR}/${NCCL_PACKAGE}-${NCCL_VERSION}-${SLURM_JOB_ID}"
artifact_stage="${OUTPUT_DIR}/.${NCCL_PACKAGE}-${NCCL_VERSION}-${SLURM_JOB_ID}.staging"
if [[ -e "${artifact_dir}" ]]; then
  printf 'Refusing to overwrite immutable artifact directory: %s\n' \
    "${artifact_dir}" >&2
  exit 2
fi
if [[ -e "${artifact_stage}" ]]; then
  printf 'Refusing to overwrite artifact staging directory: %s\n' \
    "${artifact_stage}" >&2
  exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "${PYTHON_BIN}" >/dev/null; then
  printf 'Python interpreter is unavailable in the container: %s\n' \
    "${PYTHON_BIN}" >&2
  exit 2
fi

wheel_stage="${NCCL_STAGE_ROOT}/wheel"
mkdir -p "${wheel_stage}" "${OUTPUT_DIR}"
trap 'rm -rf -- "${wheel_stage}"' EXIT

"${PYTHON_BIN}" -m pip download --no-deps --only-binary=:all: \
  --dest "${wheel_stage}" "${NCCL_PACKAGE}==${NCCL_VERSION}"

mapfile -t downloaded_wheels < <(
  find "${wheel_stage}" -maxdepth 1 -type f -name '*.whl' -print
)
if [[ "${#downloaded_wheels[@]}" != "1" ]]; then
  printf 'Expected exactly one NCCL wheel, found %s.\n' \
    "${#downloaded_wheels[@]}" >&2
  exit 2
fi
downloaded_wheel=${downloaded_wheels[0]}
wheel_basename=$(basename "${downloaded_wheel}")
wheel_pattern='^nvidia_nccl_cu13-2\.30\.4-py3-none-manylinux_[0-9_]+_x86_64\.whl$'
if [[ ! "${wheel_basename}" =~ ${wheel_pattern} ]]; then
  printf 'Unexpected NCCL wheel filename: %s\n' "${downloaded_wheel}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${downloaded_wheel}" "${NCCL_PACKAGE}" "${NCCL_VERSION}" <<'PY'
import sys
import zipfile

wheel_path, expected_name, expected_version = sys.argv[1:]
with zipfile.ZipFile(wheel_path) as wheel:
    metadata_paths = [
        path for path in wheel.namelist() if path.endswith(".dist-info/METADATA")
    ]
    if len(metadata_paths) != 1:
        raise SystemExit(f"expected one wheel metadata file, found {len(metadata_paths)}")
    metadata = wheel.read(metadata_paths[0]).decode("utf-8")

fields = {
    line.split(": ", maxsplit=1)[0]: line.split(": ", maxsplit=1)[1]
    for line in metadata.splitlines()
    if ": " in line
}
if fields.get("Name") != expected_name or fields.get("Version") != expected_version:
    raise SystemExit(
        f"wheel metadata mismatch: Name={fields.get('Name')!r}, "
        f"Version={fields.get('Version')!r}"
    )
PY

mkdir "${artifact_stage}"
staged_wheel="${artifact_stage}/${wheel_basename}"
final_wheel="${artifact_dir}/${wheel_basename}"
mv "${downloaded_wheel}" "${staged_wheel}"
wheel_sha256=$(sha256sum "${staged_wheel}" | cut -d' ' -f1)
printf '%s  %s\n' "${wheel_sha256}" "${wheel_basename}" \
  > "${staged_wheel}.sha256"
staged_at=$(date '+%Y-%m-%dT%H:%M:%S%z')

{
  printf 'package=%q\n' "${NCCL_PACKAGE}"
  printf 'version=%q\n' "${NCCL_VERSION}"
  printf 'wheel=%q\n' "${final_wheel}"
  printf 'wheel_sha256=%q\n' "${wheel_sha256}"
  printf 'container=%q\n' "${CONTAINER}"
  printf 'container_sha256=%q\n' "${CONTAINER_SHA256}"
  printf 'slurm_job_id=%q\n' "${SLURM_JOB_ID}"
  printf 'staged_at=%q\n' "${staged_at}"
} > "${artifact_stage}/metadata.env"

mv "${artifact_stage}" "${artifact_dir}"
trap - EXIT
rm -rf -- "${wheel_stage}"
printf 'Staged immutable NCCL wheel: %s\nMetadata: %s\n' \
  "${final_wheel}" "${artifact_dir}/metadata.env"
