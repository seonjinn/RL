#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=experiments/specdec_perf_20261020/logs/unit_tests_%j.out

# Run unit tests from this checkout in the nightly container on one node.
# Submit from the repo root: sbatch --account=<account> experiments/specdec_perf_20261020/unit_tests.sbatch <pytest args>
set -euo pipefail

U=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna
CONTAINER=${CONTAINER:-$U/containers/nemo_rl_nightly.sqsh}
REPO=${SLURM_SUBMIT_DIR}
echo "commit=$(git -C "${REPO}" rev-parse HEAD) container=$(readlink -f "${CONTAINER}")"

# Caches go to node-local /tmp so test runs do not grow the lustre quota.
srun --ntasks=1 --container-image="${CONTAINER}" --container-mounts=/lustre:/lustre \
  --no-container-mount-home --container-workdir="${REPO}" \
  env PYTHONPATH="${REPO}" PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache \
  TRITON_CACHE_DIR=/tmp/triton HF_HUB_OFFLINE=1 \
  /opt/nemo_rl_venv/bin/python -m pytest -p no:cacheprovider -q -rs "$@"