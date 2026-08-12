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
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --job-name=stage_training_container

set -euo pipefail

: "${SOURCE_IMAGE:?Set SOURCE_IMAGE to registry/repository:tag or @sha256 digest}"
if [[ -z "${SOURCE_DIGEST:-}" ]]; then
  echo "Set SOURCE_DIGEST to the resolved sha256 digest" >&2
  exit 2
fi
if [[ ! "${SOURCE_COMMIT:-}" =~ ^[[:xdigit:]]{40}$ ]]; then
  echo "SOURCE_COMMIT must be a full 40-character commit" >&2
  exit 2
fi
: "${OUTPUT_PREFIX:?Set OUTPUT_PREFIX to a filesystem-safe name}"
: "${CONTAINER_DIR:?Set CONTAINER_DIR to an absolute shared-storage path}"

ACCOUNT=${ACCOUNT:-coreai_dlalgo_nemorl}
PARTITION=${PARTITION:-batch}
TEST_ONLY=${TEST_ONLY:-0}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}
STAGE_CPUS_PER_TASK=${STAGE_CPUS_PER_TASK:-32}
STAGE_TIME_LIMIT=${STAGE_TIME_LIMIT:-04:00:00}

valid_source_image() {
  local reference=$1
  local registry
  local repository_reference
  local repository
  local final_component
  local tag

  if [[ "${reference}" == *://* || "${reference}" == *\?* || \
        "${reference}" == *\#* || "${reference}" =~ [[:space:]] ]]; then
    return 1
  fi
  [[ "${reference}" == */* ]] || return 1
  registry=${reference%%/*}
  repository_reference=${reference#*/}
  [[ "${registry}" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*(:[0-9]+)?$ ]] || return 1

  if [[ "${repository_reference}" == *@* ]]; then
    [[ "${repository_reference}" == *@sha256:* ]] || return 1
    [[ "${repository_reference##*@}" =~ ^sha256:[[:xdigit:]]{64}$ ]] || return 1
    repository=${repository_reference%@sha256:*}
  else
    repository=${repository_reference}
    final_component=${repository##*/}
    if [[ "${final_component}" == *:* ]]; then
      tag=${final_component#*:}
      [[ "${tag}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || return 1
      repository=${repository%:*}
    fi
  fi
  [[ "${repository}" =~ ^[A-Za-z0-9]+([._-][A-Za-z0-9]+)*(/[A-Za-z0-9]+([._-][A-Za-z0-9]+)*)*$ ]]
}

if ! valid_source_image "${SOURCE_IMAGE}"; then
  echo "SOURCE_IMAGE must be a credential-free registry/repository reference" >&2
  exit 2
fi
if [[ ! "${SOURCE_DIGEST}" =~ ^sha256:[[:xdigit:]]{64}$ ]]; then
  echo "SOURCE_DIGEST must be a sha256 digest" >&2
  exit 2
fi
if [[ "${SOURCE_IMAGE}" == *@sha256:* && "${SOURCE_IMAGE##*@}" != "${SOURCE_DIGEST}" ]]; then
  echo "SOURCE_IMAGE digest and SOURCE_DIGEST disagree" >&2
  exit 2
fi
if [[ ! "${OUTPUT_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "OUTPUT_PREFIX is not filesystem-safe" >&2
  exit 2
fi
if [[ "${CONTAINER_DIR}" != /* ]]; then
  echo "CONTAINER_DIR must be absolute" >&2
  exit 2
fi
case "${PARTITION}" in
  batch | cpu | cpu_datamover) ;;
  *)
    echo "PARTITION must be one of: batch, cpu, cpu_datamover" >&2
    exit 2
    ;;
esac
if [[ "${PARTITION}" == "cpu" || "${PARTITION}" == "cpu_datamover" ]]; then
  if [[ ! "${STAGE_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] ||
     ((STAGE_CPUS_PER_TASK < 1 || STAGE_CPUS_PER_TASK > 96)); then
    echo "STAGE_CPUS_PER_TASK must be an integer from 1 through 96" >&2
    exit 2
  fi
  if [[ ! "${STAGE_TIME_LIMIT}" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]]; then
    echo "STAGE_TIME_LIMIT must use HH:MM:SS with two-digit fields" >&2
    exit 2
  fi
fi
if [[ ! "${TEST_ONLY}" =~ ^[01]$ || ! "${SBATCH_TEST_ONLY}" =~ ^[01]$ ]]; then
  echo "TEST_ONLY and SBATCH_TEST_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${TEST_ONLY}" == "1" && "${SBATCH_TEST_ONLY}" == "1" ]]; then
  echo "TEST_ONLY and SBATCH_TEST_ONLY are mutually exclusive" >&2
  exit 2
fi

script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export ACCOUNT CONTAINER_DIR OUTPUT_PREFIX PARTITION SOURCE_COMMIT SOURCE_DIGEST
  export SOURCE_IMAGE STAGE_CPUS_PER_TASK STAGE_TIME_LIMIT
  export_spec=SOURCE_IMAGE,SOURCE_DIGEST,SOURCE_COMMIT,OUTPUT_PREFIX,CONTAINER_DIR,ACCOUNT,PARTITION,STAGE_CPUS_PER_TASK,STAGE_TIME_LIMIT,HOME,USER,LOGNAME,SHELL,PATH,LD_LIBRARY_PATH,HTTP_PROXY,HTTPS_PROXY,NO_PROXY,http_proxy,https_proxy,no_proxy,ENROOT_CACHE_PATH,ENROOT_DATA_PATH
  submission_flag=--parsable
  if [[ "${SBATCH_TEST_ONLY}" == "1" ]]; then
    submission_flag=--test-only
  fi
  sbatch_command=(
    sbatch
    "${submission_flag}"
    "--partition=${PARTITION}"
    "--account=${ACCOUNT}"
    "--output=${CONTAINER_DIR}/${OUTPUT_PREFIX}_stage-%j.log"
    "--export=${export_spec}"
  )
  if [[ "${PARTITION}" == "cpu" || "${PARTITION}" == "cpu_datamover" ]]; then
    sbatch_command+=(
      "--cpus-per-task=${STAGE_CPUS_PER_TASK}"
      "--time=${STAGE_TIME_LIMIT}"
    )
  fi
  sbatch_command+=("${script_path}")
  printf 'SOURCE_IMAGE=%s\n' "${SOURCE_IMAGE}"
  printf 'SOURCE_DIGEST=%s\n' "${SOURCE_DIGEST}"
  printf 'SBATCH:'
  printf ' %q' "${sbatch_command[@]}"
  printf '\n'
  if [[ "${TEST_ONLY}" == "1" ]]; then
    echo "TEST_ONLY: no submission performed (local render only; scheduler not contacted)"
    exit 0
  fi
  if [[ "${SBATCH_TEST_ONLY}" != "1" ]]; then
    mkdir -p "${CONTAINER_DIR}"
  fi
  "${sbatch_command[@]}"
  exit 0
fi

unset PYTHONHOME PYTHONPATH VIRTUAL_ENV UV_PROJECT_ENVIRONMENT

if [[ "${PARTITION}" == "cpu" || "${PARTITION}" == "cpu_datamover" ]]; then
  if [[ "${SLURM_CPUS_PER_TASK:-}" != "${STAGE_CPUS_PER_TASK}" ]]; then
    echo "SLURM_CPUS_PER_TASK does not match STAGE_CPUS_PER_TASK" >&2
    exit 1
  fi
fi

registry=${SOURCE_IMAGE%%/*}
repository_reference=${SOURCE_IMAGE#*/}
registry_host=${registry%%:*}
if [[ "${repository_reference}" == *@sha256:* ]]; then
  repository=${repository_reference%@sha256:*}
else
  final_component=${repository_reference##*/}
  if [[ "${final_component}" == *:* ]]; then
    repository=${repository_reference%:*}
  else
    repository=${repository_reference}
  fi
fi
enroot_image_uri="docker://${registry}#${repository}@${SOURCE_DIGEST}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${registry_host},${registry}"
export no_proxy="${NO_PROXY}"

date_stamp=$(date +%Y%m%d)
output_file=${CONTAINER_DIR}/${OUTPUT_PREFIX}_${date_stamp}_${SLURM_JOB_ID}.sqsh
metadata_file=${output_file}.metadata.txt
staging_file=${output_file}.staging.txt
current_link=${CONTAINER_DIR}/${OUTPUT_PREFIX}.sqsh
partial_file=${output_file}.partial.${SLURM_JOB_ID}
partial_metadata_file=${metadata_file}.partial.${SLURM_JOB_ID}
partial_staging_file=${staging_file}.partial.${SLURM_JOB_ID}
partial_current_link=${current_link}.partial.${SLURM_JOB_ID}

mkdir -p "${CONTAINER_DIR}"
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-${CONTAINER_DIR}/.enroot_cache}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-${CONTAINER_DIR}/.enroot_data}
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}"

cleanup() {
  rm -f \
    "${partial_file}" \
    "${partial_metadata_file}" \
    "${partial_staging_file}" \
    "${partial_current_link}"
}
trap cleanup EXIT
cleanup

for immutable_artifact in "${output_file}" "${metadata_file}" "${staging_file}"; do
  if [[ -L "${immutable_artifact}" ]]; then
    echo "Immutable artifact must not be a symlink: ${immutable_artifact}" >&2
    exit 1
  fi
done

command -v enroot >/dev/null
command -v sha256sum >/dev/null

echo "source_image=${SOURCE_IMAGE}"
echo "source_digest=${SOURCE_DIGEST}"
echo "source_commit=${SOURCE_COMMIT}"
echo "output_file=${output_file}"

metadata_value() {
  local key=$1
  local path=$2
  local line

  line=$(grep -m1 -E "^${key}=" "${path}" || true)
  [[ -n "${line}" ]] || return 1
  printf '%s\n' "${line#*=}"
}

provenance_matches() {
  local path=$1

  [[ "$(metadata_value source_image "${path}" || true)" == "${SOURCE_IMAGE}" ]] &&
    [[ "$(metadata_value source_digest "${path}" || true)" == "${SOURCE_DIGEST}" ]] &&
    [[ "$(metadata_value source_commit "${path}" || true)" == "${SOURCE_COMMIT}" ]] &&
    [[ "$(metadata_value slurm_job_id "${path}" || true)" == "${SLURM_JOB_ID}" ]]
}

require_matching_provenance() {
  local path=$1
  if ! provenance_matches "${path}"; then
    echo "Immutable provenance mismatch: ${path}" >&2
    exit 1
  fi
}

publish_no_clobber() {
  local partial_path=$1
  local immutable_path=$2

  if ! ln "${partial_path}" "${immutable_path}"; then
    echo "Refusing to overwrite immutable artifact: ${immutable_path}" >&2
    exit 1
  fi
  rm -f "${partial_path}"
}

write_staging_record() {
  {
    echo "source_image=${SOURCE_IMAGE}"
    echo "source_digest=${SOURCE_DIGEST}"
    echo "source_commit=${SOURCE_COMMIT}"
    echo "slurm_job_id=${SLURM_JOB_ID}"
  } >"${partial_staging_file}"
  publish_no_clobber "${partial_staging_file}" "${staging_file}"
}

write_metadata() {
  local image_sha256=$1

  {
    echo "metadata_format_version=1"
    echo "source_image=${SOURCE_IMAGE}"
    echo "source_digest=${SOURCE_DIGEST}"
    echo "imported_enroot_uri=${enroot_image_uri}"
    echo "source_commit=${SOURCE_COMMIT}"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "download_host=$(hostname)"
    echo "downloaded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "sha256=${image_sha256}"
  } >"${partial_metadata_file}"
  publish_no_clobber "${partial_metadata_file}" "${metadata_file}"
}

publish_current_link() {
  if [[ -e "${current_link}" && ! -L "${current_link}" ]]; then
    echo "Refusing to overwrite non-symlink stable path: ${current_link}" >&2
    exit 1
  fi
  ln -s "$(basename "${output_file}")" "${partial_current_link}"
  mv -f "${partial_current_link}" "${current_link}"
}

if [[ -e "${output_file}" && -e "${metadata_file}" ]]; then
  require_matching_provenance "${metadata_file}"
  recorded_sha256=$(metadata_value sha256 "${metadata_file}" || true)
  actual_sha256=$(sha256sum "${output_file}" | awk '{print $1}')
  if [[ ! "${recorded_sha256}" =~ ^[0-9a-f]{64}$ || "${recorded_sha256}" != "${actual_sha256}" ]]; then
    echo "Immutable image SHA256 does not match metadata: ${output_file}" >&2
    exit 1
  fi
  if [[ -e "${staging_file}" ]]; then
    require_matching_provenance "${staging_file}"
    rm -f "${staging_file}"
  fi
  publish_current_link
  cleanup
  trap - EXIT
  echo "container=${output_file}"
  echo "metadata=${metadata_file}"
  ls -lh "${output_file}" "${metadata_file}" "${current_link}"
  exit 0
fi

if [[ -e "${output_file}" ]]; then
  if [[ ! -f "${staging_file}" ]]; then
    echo "Cannot recover image without matching staging provenance: ${output_file}" >&2
    exit 1
  fi
  require_matching_provenance "${staging_file}"
  image_sha256=$(sha256sum "${output_file}" | awk '{print $1}')
  write_metadata "${image_sha256}"
  rm -f "${staging_file}"
  publish_current_link
  cleanup
  trap - EXIT
  echo "container=${output_file}"
  echo "metadata=${metadata_file}"
  ls -lh "${output_file}" "${metadata_file}" "${current_link}"
  exit 0
fi

if [[ -e "${metadata_file}" ]]; then
  require_matching_provenance "${metadata_file}"
  if [[ -e "${staging_file}" ]]; then
    require_matching_provenance "${staging_file}"
    rm -f "${staging_file}"
  fi
  rm -f "${metadata_file}"
fi

if [[ -e "${staging_file}" ]]; then
  require_matching_provenance "${staging_file}"
else
  write_staging_record
fi

enroot import -o "${partial_file}" "${enroot_image_uri}"
image_sha256=$(sha256sum "${partial_file}" | awk '{print $1}')
publish_no_clobber "${partial_file}" "${output_file}"
write_metadata "${image_sha256}"
rm -f "${staging_file}"
publish_current_link

cleanup
trap - EXIT

echo "container=${output_file}"
echo "metadata=${metadata_file}"
ls -lh "${output_file}" "${metadata_file}" "${current_link}"
