#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=stage_training_container
#SBATCH --output=stage_training_container_%j.log

set -euo pipefail

: "${SOURCE_IMAGE:?Set SOURCE_IMAGE to registry/repository:tag}"
: "${OUTPUT_PREFIX:?Set OUTPUT_PREFIX to a filesystem-safe name}"
: "${CONTAINER_DIR:?Set CONTAINER_DIR to an absolute shared-storage path}"
SOURCE_COMMIT=${SOURCE_COMMIT:-unknown}

if [[ "$SOURCE_IMAGE" != */* ]]; then
  printf 'SOURCE_IMAGE must include a registry and repository\n' >&2
  exit 2
fi
if [[ ! "$OUTPUT_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'OUTPUT_PREFIX is not filesystem-safe\n' >&2
  exit 2
fi
if [[ "$CONTAINER_DIR" != /* ]]; then
  printf 'CONTAINER_DIR must be absolute\n' >&2
  exit 2
fi

registry=${SOURCE_IMAGE%%/*}
repository_and_tag=${SOURCE_IMAGE#*/}
registry_host=${registry%%:*}
enroot_image_uri="docker://${registry}#${repository_and_tag}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${registry_host},${registry}"
export no_proxy="$NO_PROXY"

date_stamp=$(date +%Y%m%d)
output_file="${CONTAINER_DIR}/${OUTPUT_PREFIX}_${date_stamp}_${SLURM_JOB_ID}.sqsh"
metadata_file="${output_file}.metadata.txt"
current_link="${CONTAINER_DIR}/${OUTPUT_PREFIX}.sqsh"
partial_file="${output_file}.partial.${SLURM_JOB_ID}"
partial_metadata_file="${metadata_file}.partial.${SLURM_JOB_ID}"
partial_current_link="${current_link}.partial.${SLURM_JOB_ID}"

mkdir -p "$CONTAINER_DIR"
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-${CONTAINER_DIR}/.enroot_cache}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-${CONTAINER_DIR}/.enroot_data}
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"
if [[ -n "${ENROOT_TEMP_PATH:-}" ]]; then
  mkdir -p "$ENROOT_TEMP_PATH"
fi

cleanup() {
  rm -f "$partial_file" "$partial_metadata_file" "$partial_current_link"
}
trap cleanup EXIT

if [[ -e "$output_file" || -e "$metadata_file" ]]; then
  printf 'Immutable output already exists: %s\n' "$output_file" >&2
  exit 1
fi

command -v enroot >/dev/null
command -v sha256sum >/dev/null

enroot import -o "$partial_file" "$enroot_image_uri"
image_sha256=$(sha256sum "$partial_file" | awk '{print $1}')

{
  printf 'source_image=%s\n' "$SOURCE_IMAGE"
  printf 'imported_enroot_uri=%s\n' "$enroot_image_uri"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'download_host=%s\n' "$(hostname)"
  printf 'downloaded_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'sha256=%s\n' "$image_sha256"
} >"$partial_metadata_file"

mv "$partial_metadata_file" "$metadata_file"
mv "$partial_file" "$output_file"
ln -s "$output_file" "$partial_current_link"
mv -Tf "$partial_current_link" "$current_link"
cleanup
trap - EXIT

printf 'container=%s\n' "$output_file"
printf 'metadata=%s\n' "$metadata_file"
ls -lh "$output_file" "$metadata_file" "$current_link"
