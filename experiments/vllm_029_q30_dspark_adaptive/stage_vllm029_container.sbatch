#!/usr/bin/env bash
# Imported from the stage-training-containers skill and adapted for OCI-HSG.
# Scheduler/account/output options are supplied by the submission command.
set -euo pipefail

: "${SOURCE_IMAGE:?Set SOURCE_IMAGE to registry/repository:tag}"
: "${OUTPUT_PREFIX:?Set OUTPUT_PREFIX to a filesystem-safe name}"
: "${CONTAINER_DIR:?Set CONTAINER_DIR to an absolute shared-storage path}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT to the exact vLLM commit}"

if [[ "${SOURCE_IMAGE}" != */* ]]; then
  echo "SOURCE_IMAGE must include a registry and repository" >&2
  exit 2
fi
if [[ ! "${OUTPUT_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "OUTPUT_PREFIX is not filesystem-safe" >&2
  exit 2
fi
if [[ "${CONTAINER_DIR}" != /lustre/* ]]; then
  echo "CONTAINER_DIR must be a scoped Lustre directory" >&2
  exit 2
fi

readonly registry=${SOURCE_IMAGE%%/*}
readonly repository_and_tag=${SOURCE_IMAGE#*/}
readonly registry_host=${registry%%:*}
readonly enroot_image_uri="docker://${registry}#${repository_and_tag}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${registry_host},${registry}"
export no_proxy="${NO_PROXY}"

readonly date_stamp=$(date +%Y%m%d)
readonly output_file="${CONTAINER_DIR}/${OUTPUT_PREFIX}_${date_stamp}_${SLURM_JOB_ID}.sqsh"
readonly metadata_file="${output_file}.metadata.txt"
readonly current_link="${CONTAINER_DIR}/${OUTPUT_PREFIX}.sqsh"
readonly partial_file="${output_file}.partial.${SLURM_JOB_ID}"
readonly partial_metadata="${metadata_file}.partial.${SLURM_JOB_ID}"
readonly partial_link="${current_link}.partial.${SLURM_JOB_ID}"
readonly local_cache="/raid/scratch/${USER}/vllm029-container-stage-${SLURM_JOB_ID}"

mkdir -p "${CONTAINER_DIR}" "${local_cache}/cache" "${local_cache}/data"
export ENROOT_CACHE_PATH="${local_cache}/cache"
export ENROOT_DATA_PATH="${local_cache}/data"

cleanup() {
  rm -f "${partial_file}" "${partial_metadata}" "${partial_link}"
  rm -rf "${local_cache}"
}
trap cleanup EXIT

if [[ -e "${output_file}" || -e "${metadata_file}" ]]; then
  echo "immutable output already exists: ${output_file}" >&2
  exit 1
fi

command -v enroot >/dev/null
command -v sha256sum >/dev/null
enroot import -o "${partial_file}" "${enroot_image_uri}"
readonly image_sha256=$(sha256sum "${partial_file}" | awk '{print $1}')

{
  echo "source_image=${SOURCE_IMAGE}"
  echo "imported_enroot_uri=${enroot_image_uri}"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "download_host=$(hostname)"
  echo "downloaded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "sha256=${image_sha256}"
} >"${partial_metadata}"

mv "${partial_metadata}" "${metadata_file}"
mv "${partial_file}" "${output_file}"
ln -s "${output_file}" "${partial_link}"
mv -Tf "${partial_link}" "${current_link}"
trap - EXIT
rm -rf "${local_cache}"

echo "container=${output_file}"
echo "metadata=${metadata_file}"
echo "sha256=${image_sha256}"
ls -lh "${output_file}" "${metadata_file}" "${current_link}"

