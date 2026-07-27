#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export BASE=${BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
export REPO=${REPO:-$BASE/experiments/refit-opt-qwen30b/nemo-rl-refit-opt-r2}
export WORK=${WORK:-$BASE/experiments/mxfp8-qkvo-nano-gcp-nrt}
export CONTAINER=${CONTAINER:-$BASE/mopd_nano_fast/images/nemo-rl-nightly-main-20260705.sqsh}
export CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/lustre:/lustre}
export NANO_MODEL_PATH=${NANO_MODEL_PATH:-$BASE/nemo-evaluator-rundirs/nano_v35/conversions/Ultra-SFTb2-512K-hermes20k-lr2e-5-iter_0005000/hf}
export NRL_HF_HOME=${NRL_HF_HOME:-$BASE/.cache/huggingface}
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
export PARTITION=${PARTITION:-batch}
export NUM_NODES=2
export GPUS_PER_NODE=8
export GPU_REQUEST_MODE=gpus-per-node
export SLURM_NETWORK=
export SLURM_SEGMENT=
export WALLTIME=${WALLTIME:-4:00:00}
export MAX_STEPS=20
export JOB_PREFIX=${JOB_PREFIX:-coreai_chef_posttrain-mxfp8.qkvo-nano}
export WANDB_PROJECT=${WANDB_PROJECT:-sna-mxfp8-qkvo-nano-gcp-nrt}
export WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
export EXPERIMENT_CLUSTER=gcp-nrt-b200
export TENSORBOARD_ENABLED=False
export INIT_SUBMODULES=0
export SBATCH_COMMENT=${SBATCH_COMMENT:-'{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"nano_qkvo","description":"Model load and MXFP8 autotuning may leave GPUs idle during startup"}}'}

exec "$SCRIPT_DIR/submit_suite.sh"
