#!/bin/bash
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=nemotron_n3_post-smoke_validated_nightly
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/oci-smoke/slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/oci-smoke/slurm-%j.err

set -euo pipefail

: "${SCRIPT_ROOT:?Set SCRIPT_ROOT to the absolute clean staging worktree root}"
: "${EXPECTED_TOOLING_SHA:?Set EXPECTED_TOOLING_SHA to the staging worktree HEAD}"
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly SEMANTIC_WORKTREE=/home/sna/nemorl-semantic-precision-test-597c93b28
EXPECTED_REPO_SHA=${EXPECTED_REPO_SHA:-6d69f234aed4e0dfb2219308dd3160f55edf5480}

if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  echo 'SLURM_JOB_ID must be a decimal job ID' >&2
  exit 2
fi

readonly SCRATCH_DIRECTORY=/raid/scratch/nemo-rl-validated-nightly/oci-smoke-${SLURM_JOB_ID}
readonly VALIDATOR=${SCRIPT_ROOT}/experiments/pr3652_validation_container/scripts/validate_transferred_nightly.sh
readonly SMOKE_BODY=${SCRIPT_ROOT}/experiments/pr3652_validation_container/scripts/oci_hsg_smoke_validated_nightly.sh

validate_tooling_root() {
  [[ "${SCRIPT_ROOT}" = /* ]]
  test -d "${SCRIPT_ROOT}"
  test "$(git -C "${SCRIPT_ROOT}" rev-parse --show-toplevel)" = "${SCRIPT_ROOT}"
  test "$(git -C "${SCRIPT_ROOT}" rev-parse HEAD)" = "${EXPECTED_TOOLING_SHA}"
  test -z "$(git -C "${SCRIPT_ROOT}" status --porcelain)"
  test -x "${VALIDATOR}"
  test -x "${SMOKE_BODY}"
}

cleanup() {
  local exit_status=$?

  trap - EXIT
  rm -rf -- "${SCRATCH_DIRECTORY}"
  exit "${exit_status}"
}

test -f "${CONTAINER}"
test -d "${SEMANTIC_WORKTREE}"
test "$(git -C "${SEMANTIC_WORKTREE}" rev-parse HEAD)" = "${EXPECTED_REPO_SHA}"
test -z "$(git -C "${SEMANTIC_WORKTREE}" status --porcelain)"
validate_tooling_root
test ! -e "${SCRATCH_DIRECTORY}"
mkdir -p "${SCRATCH_DIRECTORY}"
trap cleanup EXIT
export TMPDIR=${SCRATCH_DIRECTORY}
export PYTHONPYCACHEPREFIX=${SCRATCH_DIRECTORY}/pycache
export XDG_CACHE_HOME=${SCRATCH_DIRECTORY}/xdg-cache
export UV_CACHE_DIR=${SCRATCH_DIRECTORY}/uv-cache
export TORCHINDUCTOR_CACHE_DIR=${SCRATCH_DIRECTORY}/torchinductor-cache
export TRITON_CACHE_DIR=${SCRATCH_DIRECTORY}/triton-cache
mkdir -p \
  "${PYTHONPYCACHEPREFIX}" \
  "${XDG_CACHE_HOME}" \
  "${UV_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}"

"${VALIDATOR}" "${CONTAINER}" "${CONTAINER}.metadata.txt" "${CONTAINER}.complete"

/cm/local/apps/slurm/current/bin/srun \
  --nodes=1 \
  --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch \
  --container-remap-root \
  "${SMOKE_BODY}" \
  "${SEMANTIC_WORKTREE}" \
  "${EXPECTED_REPO_SHA}" \
  "${SCRATCH_DIRECTORY}"
