#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=stage_vllm028_mrv2_dynamick_core
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/stage-vllm028-mrv2-dynamick-core-%j.out

set -euo pipefail

readonly VLLM_BASE_COMMIT="2cf0a6915ce544dc493a0990f2ea38d81601128a"
readonly EXPERIMENT_DIR="${EXPERIMENT_DIR:?EXPERIMENT_DIR must point to the committed experiment package}"
readonly PATCH_MANIFEST="${EXPERIMENT_DIR}/mrv2_patch_manifest.json"
readonly PATCH_DIR="${EXPERIMENT_DIR}/patches"
readonly PATCHER="${EXPERIMENT_DIR}/mrv2_patchset.py"
readonly CONTAINER_DIR="/lustre/fsw/coreai_dlalgo_llm/users/sna/containers"
readonly BASE_IMAGE="${CONTAINER_DIR}/vllm-openai-v0.28.0-aarch64-ubuntu2404.sqsh"
readonly OUTPUT_PREFIX="vllm-openai-v0.28.0-mrv2-dynamick-core-aarch64-ubuntu2404"
readonly STABLE_IMAGE="${CONTAINER_DIR}/${OUTPUT_PREFIX}.sqsh"
readonly NODE_STAGE="/raid/scratch/${USER}/vllm028-mrv2-patch-${SLURM_JOB_ID}"
readonly VLLM_SOURCE="${NODE_STAGE}/vllm"
readonly CHANGED_FILES="${NODE_STAGE}/changed-vllm-files.txt"
readonly PARTIAL_IMAGE="${NODE_STAGE}/${OUTPUT_PREFIX}.sqsh.partial"
readonly IMMUTABLE_IMAGE="${CONTAINER_DIR}/${OUTPUT_PREFIX}_${SLURM_JOB_ID}.sqsh"
readonly METADATA="${IMMUTABLE_IMAGE}.metadata.json"
readonly PARTIAL_LINK="${CONTAINER_DIR}/.${OUTPUT_PREFIX}.${SLURM_JOB_ID}.link"
readonly ENROOT_NAME="vllm028-mrv2-dynamick-${SLURM_JOB_ID}"

cleanup() {
  enroot remove -f "${ENROOT_NAME}" >/dev/null 2>&1 || true
  if [[ "${NODE_STAGE}" == "/raid/scratch/${USER}/vllm028-mrv2-patch-${SLURM_JOB_ID}" ]]; then
    rm -rf -- "${NODE_STAGE}"
  fi
}
trap cleanup EXIT

for required in "${PATCH_MANIFEST}" "${PATCHER}" "${BASE_IMAGE}"; do
  if [[ ! -r "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done
mkdir -p "${NODE_STAGE}/enroot-cache" "${NODE_STAGE}/enroot-data"
export ENROOT_CACHE_PATH="${NODE_STAGE}/enroot-cache"
export ENROOT_DATA_PATH="${NODE_STAGE}/enroot-data"

git init -q "${VLLM_SOURCE}"
git -C "${VLLM_SOURCE}" remote add origin https://github.com/vllm-project/vllm.git
git -C "${VLLM_SOURCE}" fetch --quiet --depth=1 origin tag v0.28.0
git -C "${VLLM_SOURCE}" checkout --quiet --detach "${VLLM_BASE_COMMIT}"
git -C "${VLLM_SOURCE}" config user.name "vLLM MRV2 Patch Stager"
git -C "${VLLM_SOURCE}" config user.email "noreply@openai.com"
python3 "${PATCHER}" \
  --source-root "${VLLM_SOURCE}" \
  --manifest "${PATCH_MANIFEST}" \
  --patch-dir "${PATCH_DIR}"
PATCHED_VLLM_HEAD=$(git -C "${VLLM_SOURCE}" rev-parse HEAD)
readonly PATCHED_VLLM_HEAD
git -C "${VLLM_SOURCE}" diff --name-only \
  "${VLLM_BASE_COMMIT}..${PATCHED_VLLM_HEAD}" -- vllm > "${CHANGED_FILES}"
readonly EXPECTED_CHANGED_COUNT=7
if [[ "$(wc -l < "${CHANGED_FILES}")" -ne "${EXPECTED_CHANGED_COUNT}" ]]; then
  echo "Unexpected number of patched vLLM runtime files" >&2
  sed -n '1,40p' "${CHANGED_FILES}" >&2
  exit 1
fi
if grep -Ev '^vllm/[A-Za-z0-9_./-]+\.py$' "${CHANGED_FILES}"; then
  echo "Patch stack changed a non-Python runtime file" >&2
  exit 1
fi
PATCHSET_MANIFEST_SHA256=$(sha256sum "${PATCH_MANIFEST}" | awk '{print $1}')
readonly PATCHSET_MANIFEST_SHA256
SOURCE_FILES_SHA256=$(while IFS= read -r path; do sha256sum "${VLLM_SOURCE}/${path}"; done < "${CHANGED_FILES}" | sha256sum | awk '{print $1}')
readonly SOURCE_FILES_SHA256

BASE_IMAGE_REAL=$(readlink -f "${BASE_IMAGE}")
readonly BASE_IMAGE_REAL
enroot create --name "${ENROOT_NAME}" "${BASE_IMAGE_REAL}"
enroot start \
  --root \
  --rw \
  --mount "${VLLM_SOURCE}:/opt/vllm-source" \
  --mount "${CHANGED_FILES}:/opt/changed-vllm-files.txt" \
  "${ENROOT_NAME}" \
  bash -lc '
set -euo pipefail
vllm_package=$(python3 -c "import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)")
while IFS= read -r source_path; do
  relative_path=${source_path#vllm/}
  install -D -m 0644 "/opt/vllm-source/${source_path}" "${vllm_package}/${relative_path}"
done < /opt/changed-vllm-files.txt
python3 -m compileall -q "${vllm_package}"
python3 - <<PY
from importlib.metadata import version
from pathlib import Path
import vllm

assert version("vllm") == "0.28.0"
root = Path(vllm.__file__).parent
cg = (root / "v1/worker/gpu/cudagraph_utils.py").read_text()
runner = (root / "v1/worker/gpu/model_runner.py").read_text()
speculator = (root / "v1/worker/gpu/spec_decode/autoregressive/speculator.py").read_text()
mamba = (root / "v1/attention/backends/mamba_attn.py").read_text()
assert "use_dynamic_decode_query_len" in cg
assert "num_spec_tokens_to_schedule" in runner
assert "decode_cudagraph_managers[active_num_speculative_tokens]" in speculator
assert "m.max_query_len == 1 + self.num_spec_tokens" not in mamba
assert "m.max_query_len <= 1 + self.num_spec_tokens" in mamba
PY
'
enroot export --output "${PARTIAL_IMAGE}" "${ENROOT_NAME}"
mv "${PARTIAL_IMAGE}" "${IMMUTABLE_IMAGE}"
ARTIFACT_SHA256=$(sha256sum "${IMMUTABLE_IMAGE}" | awk '{print $1}')
readonly ARTIFACT_SHA256

export ARTIFACT_SHA256 BASE_IMAGE_REAL IMMUTABLE_IMAGE PATCHED_VLLM_HEAD
export PATCHSET_MANIFEST_SHA256 SOURCE_FILES_SHA256 VLLM_BASE_COMMIT
python3 - "${METADATA}" "${PATCH_MANIFEST}" "${CHANGED_FILES}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
payload = {
    "vllm_version": "0.28.0",
    "vllm_base_commit": os.environ["VLLM_BASE_COMMIT"],
    "patched_vllm_head": os.environ["PATCHED_VLLM_HEAD"],
    "patchset_manifest_sha256": os.environ["PATCHSET_MANIFEST_SHA256"],
    "patches": manifest["patches"],
    "changed_runtime_files": Path(sys.argv[3]).read_text(encoding="utf-8").splitlines(),
    "changed_runtime_files_sha256": os.environ["SOURCE_FILES_SHA256"],
    "base_image": os.environ["BASE_IMAGE_REAL"],
    "artifact": os.environ["IMMUTABLE_IMAGE"],
    "artifact_sha256": os.environ["ARTIFACT_SHA256"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

ln -sfn "${IMMUTABLE_IMAGE}" "${PARTIAL_LINK}"
mv -Tf "${PARTIAL_LINK}" "${STABLE_IMAGE}"
echo "Staged ${STABLE_IMAGE} -> ${IMMUTABLE_IMAGE}"
