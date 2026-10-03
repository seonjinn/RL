#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

: "${REPO:?Set REPO to the clean, recursively initialized checkout}"
: "${SOURCE_SHA:?Pin SOURCE_SHA to a full commit SHA}"
: "${CONTAINER:?Set CONTAINER to the immutable nightly squashfs}"
: "${RESULT_DIR:?Set RESULT_DIR to the durable audit directory}"
SRUN_BIN=${SRUN_BIN:-/cm/local/apps/slurm/25.11/bin/srun}

[[ "${SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${SOURCE_SHA}" ]] || exit 2
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]] || exit 2
[[ -x "${SRUN_BIN}" && -f "${CONTAINER}" ]] || exit 2

scratch="/raid/scratch/${USER}/runtime-lock-audit-${SLURM_JOB_ID}"
mkdir -p "${scratch}/source" "${scratch}/results" "${RESULT_DIR}"
cleanup() {
  cp -a "${scratch}/results/." "${RESULT_DIR}/"
  rm -rf "${scratch}"
}
trap cleanup EXIT

# Source and dependency workspaces are read-only; installs are dry runs only.
git -C "${REPO}" archive "${SOURCE_SHA}" | tar -xf - -C "${scratch}/source"
rm -rf "${scratch}/source/3rdparty"
ln -s "${REPO}/3rdparty" "${scratch}/source/3rdparty"
common_dir=$(git -C "${REPO}" rev-parse --path-format=absolute --git-common-dir)
mounts="${REPO}:${REPO}:ro,/raid/scratch:/raid/scratch"
if [[ "${common_dir}" != "${REPO}/"* ]]; then
  mounts+=",${common_dir}:${common_dir}:ro"
fi

export AUDIT_SOURCE="${scratch}/source"
export AUDIT_RESULTS="${scratch}/results"
export AUDIT_SHA="${SOURCE_SHA}"

"${SRUN_BIN}" --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER}" --container-mounts="${mounts}" \
  bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail
unset WANDB_API_KEY HF_TOKEN GH_TOKEN GITHUB_TOKEN SSH_AUTH_SOCK
export UV_CACHE_DIR="${AUDIT_RESULTS}/../uv"
export UV_PYTHON_INSTALL_DIR="${AUDIT_RESULTS}/../python"
export TMPDIR="${AUDIT_RESULTS}/../tmp"
mkdir -p "${UV_CACHE_DIR}" "${TMPDIR}"
cd "${AUDIT_SOURCE}"

driver=/opt/nemo_rl_venv
vllm=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
mcore=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
failures=0
for role in driver vllm mcore; do
  case "${role}" in
    driver) env_path=${driver}; extras=() ;;
    vllm) env_path=${vllm}; extras=(--extra vllm --extra nemo_gym) ;;
    mcore) env_path=${mcore}; extras=(--extra mcore --extra nemo_gym) ;;
  esac
  "${env_path}/bin/python" - "${role}" >"${AUDIT_RESULTS}/${role}-inventory.json" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
import pathlib
import sys
import tomllib

lock_path = pathlib.Path("uv.lock")
lock = tomllib.loads(lock_path.read_text())
packages = ("ray", "torch", "vllm", "flashinfer-python", "transformer-engine", "wandb")
installed = {}
for package in packages:
    try:
        installed[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        installed[package] = None
locked = {
    package: sorted({entry["version"] for entry in lock["package"] if entry["name"] == package})
    for package in packages
}
print(json.dumps({
    "role": sys.argv[1],
    "source_sha": os.environ["AUDIT_SHA"],
    "python": sys.version,
    "executable": sys.executable,
    "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    "installed": installed,
    "locked_candidates": locked,
}, indent=2))
PY
  cat "${AUDIT_RESULTS}/${role}-inventory.json"
  if UV_PROJECT_ENVIRONMENT="${env_path}" timeout --kill-after=30s 900s \
    uv sync --frozen --dry-run --inexact --no-install-project "${extras[@]}" \
    >"${AUDIT_RESULTS}/${role}-sync-plan.log" 2>&1; then
    printf '%s\n' PASS >"${AUDIT_RESULTS}/${role}-sync-status.txt"
  else
    status=$?
    printf 'FAIL %s\n' "${status}" >"${AUDIT_RESULTS}/${role}-sync-status.txt"
    failures=$((failures + 1))
  fi
  cat "${AUDIT_RESULTS}/${role}-sync-plan.log"
done
[[ "${failures}" == 0 ]]
CONTAINER_SCRIPT
