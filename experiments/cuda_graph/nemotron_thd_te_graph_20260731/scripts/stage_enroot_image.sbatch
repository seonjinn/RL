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
#SBATCH --gres=gpu:4
#SBATCH --mem=0
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
if [[ "${PARTITION}" != "batch" ]]; then
  echo "Container staging requires PARTITION=batch" >&2
  exit 2
fi

script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  sbatch_command=(
    sbatch
    --parsable
    "--partition=${PARTITION}"
    "--account=${ACCOUNT}"
    --gres=gpu:4
    "--output=${CONTAINER_DIR}/${OUTPUT_PREFIX}_stage-%j.log"
    --export=ALL
    "${script_path}"
  )
  printf 'SOURCE_IMAGE=%s\n' "${SOURCE_IMAGE}"
  printf 'SOURCE_DIGEST=%s\n' "${SOURCE_DIGEST}"
  printf 'SBATCH:'
  printf ' %q' "${sbatch_command[@]}"
  printf '\n'
  if [[ "${TEST_ONLY}" == "1" ]]; then
    echo "TEST_ONLY: no submission performed"
    exit 0
  fi
  mkdir -p "${CONTAINER_DIR}"
  "${sbatch_command[@]}"
  exit 0
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
current_link=${CONTAINER_DIR}/${OUTPUT_PREFIX}.sqsh
partial_file=${output_file}.partial.${SLURM_JOB_ID}
partial_metadata_file=${metadata_file}.partial.${SLURM_JOB_ID}
partial_current_link=${current_link}.partial.${SLURM_JOB_ID}

mkdir -p "${CONTAINER_DIR}"
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-${CONTAINER_DIR}/.enroot_cache}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-${CONTAINER_DIR}/.enroot_data}
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}"

cleanup() {
  rm -f "${partial_file}" "${partial_metadata_file}" "${partial_current_link}"
}
trap cleanup EXIT

if [[ -e "${output_file}" || -e "${metadata_file}" ]]; then
  echo "Immutable output already exists: ${output_file}" >&2
  exit 1
fi

command -v enroot >/dev/null
command -v sha256sum >/dev/null

echo "source_image=${SOURCE_IMAGE}"
echo "source_digest=${SOURCE_DIGEST}"
echo "source_commit=${SOURCE_COMMIT}"
echo "output_file=${output_file}"

enroot import -o "${partial_file}" "${enroot_image_uri}"
image_sha256=$(sha256sum "${partial_file}" | awk '{print $1}')

{
  echo "source_image=${SOURCE_IMAGE}"
  echo "source_digest=${SOURCE_DIGEST}"
  echo "imported_enroot_uri=${enroot_image_uri}"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "download_host=$(hostname)"
  echo "downloaded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "sha256=${image_sha256}"
} >"${partial_metadata_file}"

mv "${partial_metadata_file}" "${metadata_file}"
mv "${partial_file}" "${output_file}"
ln -s "${output_file}" "${partial_current_link}"
mv -Tf "${partial_current_link}" "${current_link}"
cleanup
trap - EXIT

echo "container=${output_file}"
echo "metadata=${metadata_file}"
ls -lh "${output_file}" "${metadata_file}" "${current_link}"
