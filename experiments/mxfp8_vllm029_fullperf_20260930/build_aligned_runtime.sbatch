#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

: "${REPO:?Set REPO to the clean, recursively initialized checkout}"
: "${SOURCE_SHA:?Pin SOURCE_SHA to the full commit SHA}"
: "${CONTAINER:?Set CONTAINER to the immutable base nightly}"
: "${BASE_IMAGE_SHA256:?Set BASE_IMAGE_SHA256 from its recorded provenance}"
: "${CONTAINER_DIR:?Set CONTAINER_DIR to the durable image directory}"
: "${RESULT_DIR:?Set RESULT_DIR to durable build logs}"
SRUN_BIN=${SRUN_BIN:-/cm/local/apps/slurm/25.11/bin/srun}
PREFIX=${PREFIX:-nemo_rl_runtime_aligned_20261003}

[[ "${SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]
[[ "${BASE_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${PREFIX}" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${SOURCE_SHA}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]
submodule_state=$(git -C "${REPO}" submodule status --recursive)
while IFS= read -r line; do
  if [[ "${line:0:1}" != ' ' ]]; then
    echo "Submodule pins are not initialized/matched: ${line}" >&2
    exit 2
  fi
done <<<"${submodule_state}"
[[ -x "${SRUN_BIN}" && -f "${CONTAINER}" ]]
[[ "$(sha256sum "${CONTAINER}" | cut -d ' ' -f 1)" == "${BASE_IMAGE_SHA256}" ]]

scratch="/raid/scratch/${USER}/runtime-align-${SLURM_JOB_ID}"
output="${CONTAINER_DIR}/${PREFIX}_${SLURM_JOB_ID}.sqsh"
partial="${output}.partial"
[[ ! -e "${output}" && ! -e "${partial}" ]]
mkdir -p "${scratch}/input" "${scratch}/source" "${scratch}/results" \
  "${CONTAINER_DIR}" "${RESULT_DIR}"
export ENROOT_CACHE_PATH="${scratch}/enroot-cache"
export ENROOT_DATA_PATH="${scratch}/enroot-data"
export ENROOT_TEMP_PATH="${scratch}/enroot-tmp"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_TEMP_PATH}"
df -h "${scratch}" "${CONTAINER_DIR}"
cleanup() {
  cp -a "${scratch}/results/." "${RESULT_DIR}/"
  rm -f "${partial}"
  rm -rf "${scratch}"
}
trap cleanup EXIT

git -C "${REPO}" ls-files -z --recurse-submodules --cached --full-name >"${scratch}/manifest"
tar --null -cf "${scratch}/input/source.tar" -C "${REPO}" -T "${scratch}/manifest"
cp "${REPO}/experiments/mxfp8_vllm029_fullperf_20260930/align_runtime_inside_container.sh" \
  "${scratch}/input/align.sh"
python3 "${REPO}/tools/generate_fingerprint.py" >"${scratch}/input/fingerprint.json"
git -C "${REPO}" submodule status --recursive >"${scratch}/input/submodule-pins.txt"
export SOURCE_SHA BASE_IMAGE_SHA256
python3 - "${REPO}" "${scratch}/input/source-metadata.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root, output = map(Path, sys.argv[1:])
metadata = {
    "source_sha": os.environ["SOURCE_SHA"],
    "base_image_sha256": os.environ["BASE_IMAGE_SHA256"],
    "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
    "project_sha256": hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest(),
    "submodule_pins": (output.parent / "submodule-pins.txt").read_text().splitlines(),
}
output.write_text(json.dumps(metadata, indent=2) + "\n")
PY

export ALIGN_INPUT="${scratch}/input" ALIGN_SCRATCH="${scratch}"
"${SRUN_BIN}" --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER}" --container-writable --container-remap-root \
  --container-save="${partial}" \
  --container-env=ALIGN_INPUT,ALIGN_SCRATCH \
  --container-mounts="/raid/scratch:/raid/scratch" \
  bash "${ALIGN_INPUT}/align.sh"

[[ -s "${partial}" && -s "${scratch}/results/after.json" ]]
mv "${partial}" "${output}"
sha256sum "${output}" >"${output}.sha256"
cp "${scratch}/input/source-metadata.json" "${output}.source.json"
cp "${scratch}/results/after.json" "${output}.runtime.json"
printf '%s\n' "${output}" >"${RESULT_DIR}/image-path.txt"
printf 'image=%s\nsource=%s\n' "${output}" "${SOURCE_SHA}"
