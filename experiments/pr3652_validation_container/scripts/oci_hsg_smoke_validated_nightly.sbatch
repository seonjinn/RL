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

set -euo pipefail

readonly CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh}
readonly SEMANTIC_WORKTREE=/home/sna/nemorl-semantic-precision-test-597c93b28
EXPECTED_REPO_SHA=${EXPECTED_REPO_SHA:-6d69f234aed4e0dfb2219308dd3160f55edf5480}

if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  echo 'SLURM_JOB_ID must be a decimal job ID' >&2
  exit 2
fi

readonly SCRATCH_DIRECTORY=/raid/scratch/nemo-rl-validated-nightly/oci-smoke-${SLURM_JOB_ID}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

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
test ! -e "${SCRATCH_DIRECTORY}"
mkdir -p "${SCRATCH_DIRECTORY}"
trap cleanup EXIT

"${script_dir}/validate_transferred_nightly.sh"

/cm/local/apps/slurm/current/bin/srun \
  --nodes=1 \
  --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch \
  --container-remap-root \
  "${script_dir}/oci_hsg_smoke_validated_nightly.sh" \
  "${SEMANTIC_WORKTREE}" \
  "${EXPECTED_REPO_SHA}"
