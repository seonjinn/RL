#!/usr/bin/env bash

set -euo pipefail

export ACTION=${ACTION:-submit}
export MODEL=${MODEL:-qwen235smoke}
export PRECISION_MODE=${PRECISION_MODE:-mxfp8}
export FP8_PARAM=${FP8_PARAM:-false}
export MAX_STEPS=${MAX_STEPS:-20}
export RUN_GROUP=${RUN_GROUP:-20260909-qwen235-mxfp8-ab}
export REPO=${REPO:-/home/${USER}/RL-qwen235-mxfp8-ab-20260909}
export CONTAINER=${CONTAINER:-$(readlink -f "/lustre/fsw/coreai_dlalgo_llm/users/${USER}/containers/nemo_rl_nightly.sqsh")}
export HF_HOME=${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/hf_home}
export WANDB_HOME=${WANDB_HOME:-/home/${USER}/.config/nemo-rl-wandb}
export RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/experiments/qwen235-mxfp8-ab-20260909}
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
export PARTITION=${PARTITION:-gb200}
export WALLTIME=${WALLTIME:-04:00:00}
export USE_GRES=0

exec "${REPO}/experiments/native_mxfp8_source_refit/submit_oci_hsg.sh"
