#!/bin/bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --segment=1
#SBATCH --time=00:10:00
#SBATCH --job-name=coreai_dlalgo_llm-nemorl.swe-entry

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR must point to the committed NeMo-RL worktree}"
SIF="${SIF:-/lustre/fsw/coreai_dlalgo_llm/users/sna/sweb_sifs/astropy__astropy-12907.sif}"
NEMO_CONTAINER="${NEMO_CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh}"
OPENHANDS_SETUP="${OPENHANDS_SETUP:?OPENHANDS_SETUP must contain OpenHands and miniforge3}"
DATASET_JSONL="${DATASET_JSONL:?DATASET_JSONL must contain an Astropy SWE instance}"
ENTRY_SCRIPT="${OPENHANDS_SETUP}/OpenHands/evaluation/benchmarks/swe_bench/scripts/setup/instance_swe_entry.sh"

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
        --writable-tmpfs \
        --cleanenv \
        --pid \
        --no-mount home,tmp,bind-paths \
        --bind "${REPO_DIR}:${REPO_DIR}:ro" \
        --bind "${OPENHANDS_SETUP}:/openhands_setup:ro" \
        --bind "${OPENHANDS_SETUP}:${OPENHANDS_SETUP}:ro" \
        --mount "type=bind,src=${DATASET_JSONL},dst=/mnt/nemorl_swe_entry_data.jsonl,ro" \
        "${SIF}" \
        env \
        PATH="/openhands_setup/OpenHands/.venv/bin:/openhands_setup/miniforge3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        PYTHONPATH="/openhands_setup/OpenHands" \
        TMUX_MEMORY_LIMIT=32768 \
        /openhands_setup/OpenHands/.venv/bin/python \
        "${REPO_DIR}/experiments/nemogym_swe_full_rl/verify_openhands_swe_entry.py" \
        --dataset /mnt/nemorl_swe_entry_data.jsonl \
        --entry-script "${ENTRY_SCRIPT}" \
        --timeout-seconds 20
