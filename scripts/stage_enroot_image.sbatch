#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=stage_training_container
#SBATCH --output=stage_training_container_%j.log

# Import one registry image into an immutable Enroot squashfs artifact.  The
# import is CPU/I/O-only, so deliberately do not request a GPU; this keeps the
# script portable to Ptyche, whose scheduler rejects the legacy GRES syntax.
#
# SOURCE_DIGEST is deliberately explicit: a mutable registry tag alone is not
# sufficient provenance for a performance experiment. Resolve the tag before
# submission, pass the resulting sha256 digest here, and use that digest for
# every subsequent cluster import.
set -euo pipefail

: "${SOURCE_IMAGE:?Set SOURCE_IMAGE to registry/repository:tag or @sha256 digest}"
: "${SOURCE_DIGEST:?Set SOURCE_DIGEST to the resolved sha256 digest}"
: "${OUTPUT_PREFIX:?Set OUTPUT_PREFIX to a filesystem-safe name}"
: "${CONTAINER_DIR:?Set CONTAINER_DIR to an absolute shared-storage path}"
SOURCE_COMMIT=${SOURCE_COMMIT:-unknown}

if [[ "${SOURCE_IMAGE}" != */* ]]; then
  echo "SOURCE_IMAGE must include a registry and repository" >&2
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

registry=${SOURCE_IMAGE%%/*}
repository_and_tag=${SOURCE_IMAGE#*/}
registry_host=${registry%%:*}
enroot_image_uri="docker://${registry}#${repository_and_tag}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${registry_host},${registry}"
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
