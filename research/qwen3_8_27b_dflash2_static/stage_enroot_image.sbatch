#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=stage_dflash2_vllm

set -euo pipefail

: "${CONTAINER_DIR:?Set CONTAINER_DIR to an absolute shared-storage path}"
: "${REPO_ROOT:?Set REPO_ROOT to the /home checkout path}"
: "${SLURM_JOB_ID:?stage_enroot_image.sbatch must run inside an sbatch allocation}"

IMAGE_CONTRACT_HELPER="${REPO_ROOT}/research/qwen3_8_27b_dflash2_static/image_contract.py"

pin_contract_value() {
  local variable_name=$1
  local contract_field=$2
  local expected_value
  expected_value=$(python3 "${IMAGE_CONTRACT_HELPER}" get "${contract_field}")
  if [[ -n "${!variable_name:-}" && "${!variable_name}" != "${expected_value}" ]]; then
    echo "${variable_name} does not match the pinned image contract" >&2
    exit 2
  fi
  printf -v "${variable_name}" '%s' "${expected_value}"
}

pin_contract_value SOURCE_IMAGE source_image
pin_contract_value SOURCE_COMMIT source_commit
pin_contract_value SOURCE_INDEX_DIGEST source_index_digest
pin_contract_value SOURCE_ARM64_DIGEST source_arm64_digest
pin_contract_value DFLASH2_MERGE_ANCESTOR dflash2_merge_ancestor
OUTPUT_PREFIX=${OUTPUT_PREFIX:-vllm-openai-nightly-f94666b}

if [[ "${SOURCE_IMAGE}" != */* ]]; then
  echo "SOURCE_IMAGE must include a registry and repository" >&2
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
if [[ "${REPO_ROOT}" != /* ]]; then
  echo "REPO_ROOT must be absolute" >&2
  exit 2
fi

registry=${SOURCE_IMAGE%%/*}
registry_host=${registry%%:*}
enroot_image_uri=$(python3 "${IMAGE_CONTRACT_HELPER}" enroot-uri)
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${registry_host},${registry},auth.docker.io,registry-1.docker.io"
export no_proxy="${NO_PROXY}"

date_stamp=$(date +%Y%m%d)
output_file="${CONTAINER_DIR}/${OUTPUT_PREFIX}_${date_stamp}_${SLURM_JOB_ID}.sqsh"
metadata_file="${output_file}.metadata.txt"
current_link="${CONTAINER_DIR}/${OUTPUT_PREFIX}.sqsh"
partial_file="${output_file}.partial.${SLURM_JOB_ID}"
partial_metadata_file="${metadata_file}.partial.${SLURM_JOB_ID}"
partial_current_link="${current_link}.partial.${SLURM_JOB_ID}"

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
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "DFlash2 image contract requires an aarch64 staging host" >&2
  exit 1
fi

echo "source_image=${SOURCE_IMAGE}"
echo "source_commit=${SOURCE_COMMIT}"
echo "output_file=${output_file}"

python3 "${IMAGE_CONTRACT_HELPER}" verify-registry
enroot import -o "${partial_file}" "${enroot_image_uri}"
image_sha256=$(sha256sum "${partial_file}" | awk '{print $1}')

{
  echo "source_image=${SOURCE_IMAGE}"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "source_index_digest=${SOURCE_INDEX_DIGEST}"
  echo "source_arm64_digest=${SOURCE_ARM64_DIGEST}"
  echo "dflash2_merge_ancestor=${DFLASH2_MERGE_ANCESTOR}"
  echo "platform=linux/arm64"
  echo "sha256=${image_sha256}"
} >"${partial_metadata_file}"

python3 "${IMAGE_CONTRACT_HELPER}" validate-metadata \
  "${partial_metadata_file}" >/dev/null

mv "${partial_metadata_file}" "${metadata_file}"
mv "${partial_file}" "${output_file}"
ln -s "${output_file}" "${partial_current_link}"
mv -Tf "${partial_current_link}" "${current_link}"
cleanup
trap - EXIT

echo "container=${output_file}"
echo "metadata=${metadata_file}"
ls -lh "${output_file}" "${metadata_file}" "${current_link}"
