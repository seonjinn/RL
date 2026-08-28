#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=stage_vllm028_container
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/stage-vllm028-%j.out

set -euo pipefail

readonly SOURCE_DIGEST="sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4"
readonly SOURCE_IMAGE="registry-1.docker.io/vllm/vllm-openai@${SOURCE_DIGEST}"
readonly SOURCE_COMMIT="2cf0a69"
readonly RAY_VERSION="2.48.0"
readonly RAY_REQUIREMENT="ray[cgraph,default]==2.48.0"
readonly RAY_LOCK="${RAY_LOCK_PATH:-${SLURM_SUBMIT_DIR}/ray248-aarch64.lock}"
readonly CONTAINER_DIR="/lustre/fsw/coreai_dlalgo_llm/users/sna/containers"
readonly OUTPUT_PREFIX="vllm-openai-v0.28.0-aarch64-ubuntu2404"
readonly STABLE_IMAGE="${CONTAINER_DIR}/${OUTPUT_PREFIX}.sqsh"
readonly NODE_STAGE="/raid/scratch/${USER}/vllm028-container-${SLURM_JOB_ID}"
readonly IMMUTABLE_IMAGE="${CONTAINER_DIR}/${OUTPUT_PREFIX}_${SLURM_JOB_ID}.sqsh"
readonly METADATA="${IMMUTABLE_IMAGE}.metadata.json"
readonly PARTIAL_IMAGE="${NODE_STAGE}/${OUTPUT_PREFIX}.sqsh.partial"
readonly PARTIAL_LINK="${CONTAINER_DIR}/.${OUTPUT_PREFIX}.${SLURM_JOB_ID}.link"
readonly RAY_OUTPUT_PREFIX="vllm-ray-2.48.0-aarch64"
readonly RAY_SITE_PACKAGES="${NODE_STAGE}/ray-sidecar"
readonly LOCAL_RAY_BUNDLE="${NODE_STAGE}/${RAY_OUTPUT_PREFIX}.tar.gz"
readonly IMMUTABLE_RAY_BUNDLE="${CONTAINER_DIR}/${RAY_OUTPUT_PREFIX}_${SLURM_JOB_ID}.tar.gz"
readonly RAY_METADATA="${IMMUTABLE_RAY_BUNDLE}.metadata.json"
readonly STABLE_RAY_BUNDLE="${CONTAINER_DIR}/${RAY_OUTPUT_PREFIX}.tar.gz"
readonly PARTIAL_RAY_LINK="${CONTAINER_DIR}/.${RAY_OUTPUT_PREFIX}.${SLURM_JOB_ID}.link"

cleanup_node_stage() {
  if [[ "${NODE_STAGE}" == "/raid/scratch/${USER}/vllm028-container-${SLURM_JOB_ID}" ]]; then
    rm -rf -- "${NODE_STAGE}"
  fi
}
trap cleanup_node_stage EXIT

mkdir -p "${CONTAINER_DIR}" "${NODE_STAGE}/cache" "${NODE_STAGE}/data"
if [[ ! -r "${RAY_LOCK}" ]]; then
  echo "Missing committed Ray lock: ${RAY_LOCK}" >&2
  exit 1
fi
RAY_LOCK_SHA256=$(sha256sum "${RAY_LOCK}" | awk '{print $1}')
readonly RAY_LOCK_SHA256
export ENROOT_CACHE_PATH="${NODE_STAGE}/cache"
export ENROOT_DATA_PATH="${NODE_STAGE}/data"

enroot import \
  --output "${PARTIAL_IMAGE}" \
  "docker://registry-1.docker.io#vllm/vllm-openai@${SOURCE_DIGEST}"
enroot start "${PARTIAL_IMAGE}" python3 -c \
  "from importlib.metadata import version; assert version('vllm') == '0.28.0'"
mkdir -p "${RAY_SITE_PACKAGES}"
enroot start \
  --root \
  --mount "${RAY_SITE_PACKAGES}:/opt/ray-sidecar" \
  --mount "${RAY_LOCK}:/opt/ray248-aarch64.lock" \
  "${PARTIAL_IMAGE}" \
  python3 -m pip install \
    --no-cache-dir \
    --only-binary=:all: \
    --require-hashes \
    --target /opt/ray-sidecar \
    --requirement /opt/ray248-aarch64.lock
enroot start \
  --mount "${RAY_SITE_PACKAGES}:/opt/ray-sidecar" \
  --env PYTHONPATH=/opt/ray-sidecar \
  "${PARTIAL_IMAGE}" \
  python3 -c "import ray; assert ray.__version__ == '2.48.0'"
mkdir -p "${RAY_SITE_PACKAGES}/_provenance"
install -m 0644 "${RAY_LOCK}" "${RAY_SITE_PACKAGES}/_provenance/ray248-aarch64.lock"
enroot start \
  --mount "${RAY_SITE_PACKAGES}:/opt/ray-sidecar" \
  --env PYTHONPATH=/opt/ray-sidecar \
  "${PARTIAL_IMAGE}" \
  python3 -m pip freeze --path /opt/ray-sidecar \
  > "${RAY_SITE_PACKAGES}/_provenance/installed-packages.txt"
INSTALLED_PACKAGES_SHA256=$(sha256sum "${RAY_SITE_PACKAGES}/_provenance/installed-packages.txt" | awk '{print $1}')
readonly INSTALLED_PACKAGES_SHA256
tar \
  --sort=name \
  --mtime='UTC 1970-01-01' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -C "${RAY_SITE_PACKAGES}" \
  -cf - . | gzip -n > "${LOCAL_RAY_BUNDLE}"

mv "${PARTIAL_IMAGE}" "${IMMUTABLE_IMAGE}"
mv "${LOCAL_RAY_BUNDLE}" "${IMMUTABLE_RAY_BUNDLE}"
LOCAL_SHA256=$(sha256sum "${IMMUTABLE_IMAGE}" | awk '{print $1}')
readonly LOCAL_SHA256
RAY_BUNDLE_SHA256=$(sha256sum "${IMMUTABLE_RAY_BUNDLE}" | awk '{print $1}')
readonly RAY_BUNDLE_SHA256
export SOURCE_IMAGE SOURCE_COMMIT SOURCE_DIGEST IMMUTABLE_IMAGE LOCAL_SHA256
export RAY_VERSION RAY_REQUIREMENT RAY_LOCK_SHA256 INSTALLED_PACKAGES_SHA256
export IMMUTABLE_RAY_BUNDLE RAY_BUNDLE_SHA256
python3 - "${METADATA}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "source_image": os.environ["SOURCE_IMAGE"],
    "source_digest": os.environ["SOURCE_DIGEST"],
    "source_commit": os.environ["SOURCE_COMMIT"],
    "artifact": os.environ["IMMUTABLE_IMAGE"],
    "artifact_sha256": os.environ["LOCAL_SHA256"],
    "ray_version": os.environ["RAY_VERSION"],
    "ray_requirement": os.environ["RAY_REQUIREMENT"],
    "ray_lock_sha256": os.environ["RAY_LOCK_SHA256"],
    "ray_installed_packages_sha256": os.environ["INSTALLED_PACKAGES_SHA256"],
    "ray_bundle": os.environ["IMMUTABLE_RAY_BUNDLE"],
    "ray_bundle_sha256": os.environ["RAY_BUNDLE_SHA256"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
install -m 0644 "${METADATA}" "${RAY_METADATA}"

ln -sfn "${IMMUTABLE_IMAGE}" "${PARTIAL_LINK}"
mv -Tf "${PARTIAL_LINK}" "${STABLE_IMAGE}"
ln -sfn "${IMMUTABLE_RAY_BUNDLE}" "${PARTIAL_RAY_LINK}"
mv -Tf "${PARTIAL_RAY_LINK}" "${STABLE_RAY_BUNDLE}"
echo "Staged ${STABLE_IMAGE} -> ${IMMUTABLE_IMAGE}"
echo "Staged ${STABLE_RAY_BUNDLE} -> ${IMMUTABLE_RAY_BUNDLE}"
