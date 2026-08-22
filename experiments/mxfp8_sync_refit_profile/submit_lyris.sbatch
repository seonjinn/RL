#!/usr/bin/env bash

#SBATCH --partition=gb200
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH --segment=4

set -euo pipefail

export ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna
export REPO=/home/sna/RL-sync-refit-phase-profile
export CONTAINER=${ROOT}/containers/nemo_rl_nightly_main_f25d_20260813_2683633.sqsh
export WANDB_HOME=/home/sna
export LOCAL_SCRATCH=/tmp/sna
export MOUNTS=/lustre:/lustre,/home:/home,/tmp:/tmp,${WANDB_HOME}/.netrc:/root/.netrc

exec bash "${REPO}/experiments/mxfp8_sync_refit_profile/submit_oci_hsg.sbatch"
