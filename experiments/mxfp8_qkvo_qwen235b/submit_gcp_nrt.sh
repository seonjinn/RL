#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export BASE=${BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
export REPO=${REPO:-$BASE/experiments/refit-opt-qwen30b/nemo-rl-mxfp8-qkv}
export WORK=${WORK:-$BASE/experiments/mxfp8-qkvo-qwen235b-gcp-nrt}
export CONTAINER=${CONTAINER:-$BASE/mopd_nano_fast/images/nemo-rl-nightly-main-20260705.sqsh}
export CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/lustre:/lustre}
export NRL_HF_HOME=${NRL_HF_HOME:-$BASE/.cache/huggingface}
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
export PARTITION=${PARTITION:-batch}
export NUM_NODES=8
export GPUS_PER_NODE=8
export GPU_REQUEST_MODE=gpus-per-node
export SLURM_NETWORK=
# GCP-NRT Slurm does not expose --segment. Keep the application segment at the
# full 8-node allocation, matching completed Qwen3-235B jobs on this cluster.
export SLURM_SEGMENT=
export WALLTIME=${WALLTIME:-4:00:00}
export JOB_PREFIX=${JOB_PREFIX:-coreai_chef_posttrain-mxfp8.qkvo-235b}
export WANDB_PROJECT=${WANDB_PROJECT:-sna-mxfp8-qkvo-qwen235b-gcp-nrt}
export WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
export EXPERIMENT_CLUSTER=gcp-nrt-b200
export INIT_SUBMODULES=0
export SBATCH_COMMENT=${SBATCH_COMMENT:-'{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"qwen235b_qkvo","description":"Model load and MXFP8 autotuning may leave GPUs idle during startup"}}'}

exec "$SCRIPT_DIR/submit_suite.sh"
