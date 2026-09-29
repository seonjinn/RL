#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=02:00:00

set -euo pipefail

: "${SOURCE_IMAGE:?Set SOURCE_IMAGE to registry/repository:tag}"
: "${OUTPUT_PREFIX:?Set OUTPUT_PREFIX to a filesystem-safe name}"
: "${CONTAINER_DIR:?Set CONTAINER_DIR to an absolute shared-storage path}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT to the tested source SHA}"

[[ "$SOURCE_IMAGE" == */* ]]
[[ "$OUTPUT_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
[[ "$CONTAINER_DIR" == /* ]]

registry=${SOURCE_IMAGE%%/*}
repository_and_tag=${SOURCE_IMAGE#*/}
registry_host=${registry%%:*}
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${registry_host},${registry}"
export no_proxy="$NO_PROXY"

output_file="$CONTAINER_DIR/${OUTPUT_PREFIX}_$(date +%Y%m%d)_${SLURM_JOB_ID}.sqsh"
metadata_file="$output_file.metadata.txt"
partial_file="$output_file.partial.${SLURM_JOB_ID}"
partial_metadata_file="$metadata_file.partial.${SLURM_JOB_ID}"
current_link="$CONTAINER_DIR/${OUTPUT_PREFIX}.sqsh"
partial_link="$current_link.partial.${SLURM_JOB_ID}"

mkdir -p "$CONTAINER_DIR" "/raid/scratch/sna/enroot-cache-${SLURM_JOB_ID}" "/raid/scratch/sna/enroot-data-${SLURM_JOB_ID}"
export ENROOT_CACHE_PATH="/raid/scratch/sna/enroot-cache-${SLURM_JOB_ID}"
export ENROOT_DATA_PATH="/raid/scratch/sna/enroot-data-${SLURM_JOB_ID}"

cleanup() {
  rm -f "$partial_file" "$partial_metadata_file" "$partial_link"
}
trap cleanup EXIT

[[ ! -e "$output_file" && ! -e "$metadata_file" ]]
command -v enroot >/dev/null
command -v sha256sum >/dev/null

echo "source_image=$SOURCE_IMAGE"
echo "source_commit=$SOURCE_COMMIT"
echo "output_file=$output_file"
enroot import -o "$partial_file" "docker://${registry}#${repository_and_tag}"
image_sha256=$(sha256sum "$partial_file" | awk '{print $1}')

{
  echo "source_image=$SOURCE_IMAGE"
  echo "source_commit=$SOURCE_COMMIT"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "download_host=$(hostname)"
  echo "downloaded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "sha256=$image_sha256"
} > "$partial_metadata_file"

mv "$partial_metadata_file" "$metadata_file"
mv "$partial_file" "$output_file"
ln -s "$output_file" "$partial_link"
mv -Tf "$partial_link" "$current_link"
trap - EXIT

echo "container=$output_file"
echo "metadata=$metadata_file"
ls -lh "$output_file" "$metadata_file" "$current_link"
