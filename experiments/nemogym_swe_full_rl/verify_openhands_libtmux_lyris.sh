#!/bin/bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --segment=1
#SBATCH --time=00:10:00
#SBATCH --job-name=coreai_dlalgo_llm-nemorl.openhands-tmux

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR must point to the committed NeMo-RL worktree}"
SIF="${SIF:-/lustre/fsw/coreai_dlalgo_llm/users/sna/sweb_sifs/astropy__astropy-12907.sif}"
NEMO_CONTAINER="${NEMO_CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh}"
OPENHANDS_SETUP="${OPENHANDS_SETUP:?OPENHANDS_SETUP must contain OpenHands and miniforge3}"

srun \
    --no-container-mount-home \
    --mpi=pmix \
    --container-mounts="/lustre:/lustre,/dev/fuse:/dev/fuse" \
    --container-image="${NEMO_CONTAINER}" \
    --container-workdir="${REPO_DIR}" \
    --nodes=1 \
    --ntasks=1 \
    --partition="${SLURM_JOB_PARTITION}" \
    --account="${SLURM_JOB_ACCOUNT}" \
    apptainer exec \
        --cleanenv \
        --bind "${REPO_DIR}:${REPO_DIR}:ro" \
        --bind "${OPENHANDS_SETUP}:/openhands_setup:ro" \
        --bind "${OPENHANDS_SETUP}:${OPENHANDS_SETUP}:ro" \
        "${SIF}" \
        env \
        PATH="/openhands_setup/miniforge3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        /openhands_setup/OpenHands/.venv/bin/python \
        "${REPO_DIR}/experiments/nemogym_swe_full_rl/verify_openhands_libtmux.py"
