#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

BASE=${BASE:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
REPO=${REPO:-$BASE/RL-mxfp8-qkvo-pr3294-ab}
WORK=${WORK:-$BASE/experiments/mxfp8-qkvo-nano}
CONTAINER=${CONTAINER:-$BASE/containers/nemo_rl_nightly.sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/lustre:/lustre,/project:/project}
NANO_MODEL_PATH=${NANO_MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/models/nemotron-nano3/Ultra-SFTb2-512K-hermes20k-lr2e-5-iter_0005000/hf}
NRL_HF_HOME=${NRL_HF_HOME:-$BASE/hf_home}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
GPU_REQUEST_MODE=${GPU_REQUEST_MODE:-none}
SLURM_NETWORK=${SLURM_NETWORK-sharp}
SLURM_SEGMENT=${SLURM_SEGMENT-$NUM_NODES}
SBATCH_COMMENT=${SBATCH_COMMENT-metrics}
WALLTIME=${WALLTIME:-4:00:00}
DEPENDENCY=${DEPENDENCY:-}
JOB_PREFIX=${JOB_PREFIX:-coreai_dlalgo_llm-mxfp8.qkvo-nano}
ACTION=${ACTION:-test-only}
MAX_STEPS=${MAX_STEPS:-20}
ARM_FILTER=${ARM_FILTER:-}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
WANDB_PROJECT=${WANDB_PROJECT:-sna-mxfp8-qkvo-nano}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
EXPERIMENT_CLUSTER=${EXPERIMENT_CLUSTER:-lyris}
TENSORBOARD_ENABLED=${TENSORBOARD_ENABLED:-True}
INIT_SUBMODULES=${INIT_SUBMODULES:-1}
BATCH_SCRIPT=$REPO/experiments/mxfp8_qkvo_nano/run_arm.sbatch

case "$ACTION" in
  test-only)
    ACTION_ARG=--test-only
    ;;
  submit)
    ACTION_ARG=
    ;;
  *)
    echo "ACTION must be test-only or submit" >&2
    exit 2
    ;;
esac

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
if [[ "$TOTAL_GPUS" -ne 16 ]]; then
  echo "Nano suite requires 16 GPUs total, got $TOTAL_GPUS" >&2
  exit 2
fi
if [[ ! ("$NUM_NODES" == "4" && "$GPUS_PER_NODE" == "4") \
  && ! ("$NUM_NODES" == "2" && "$GPUS_PER_NODE" == "8") ]]; then
  echo "Nano suite supports only 4x4 or 2x8 topology" >&2
  exit 2
fi

test -x "$BATCH_SCRIPT"
test -f "$NANO_MODEL_PATH/config.json"
before_pull_sha=$(git -C "$REPO" rev-parse HEAD)
git -C "$REPO" -c fetch.recurseSubmodules=false \
  pull --ff-only --recurse-submodules=no
after_pull_sha=$(git -C "$REPO" rev-parse HEAD)
if [[ "$before_pull_sha" != "$after_pull_sha" && "${SUBMIT_SUITE_REEXEC:-0}" != "1" ]]; then
  export SUBMIT_SUITE_REEXEC=1
  exec "${BASH_SOURCE[0]}"
fi
if [[ "$INIT_SUBMODULES" == "1" ]]; then
  git -C "$REPO" submodule update --init --recursive
  SUBMODULE_STATUS_MODE=dirty
elif [[ "$INIT_SUBMODULES" != "0" ]]; then
  echo "INIT_SUBMODULES must be 0 or 1" >&2
  exit 2
else
  SUBMODULE_STATUS_MODE=all
fi
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=all --ignore-submodules="$SUBMODULE_STATUS_MODE")" ]]; then
  echo "Repository has uncommitted superproject changes: $REPO" >&2
  git -C "$REPO" status --short --ignore-submodules="$SUBMODULE_STATUS_MODE" >&2
  exit 2
fi
upstream_sha=$(git -C "$REPO" rev-parse '@{upstream}')
if [[ "$after_pull_sha" != "$upstream_sha" ]]; then
  echo "Repository HEAD does not match its upstream" >&2
  exit 2
fi
CONTAINER=$(readlink -f "$CONTAINER")
test -f "$CONTAINER"
mkdir -p "$WORK/slurm" "$WORK/manifests"
export CONTAINER_MOUNTS

ARMS=(
  "bf16:grpo-nanov3-30BA3B-2n8g-megatron-pack-cp:0"
  "moe-baseline:grpo-nanov3-30BA3B-4n4g-megatron-pack-cp-mxfp8-rollout:0"
  "moe-optimized:grpo-nanov3-30BA3B-4n4g-megatron-pack-cp-mxfp8-rollout:1"
  "qkvo-baseline:grpo-nanov3-30BA3B-4n4g-megatron-pack-cp-mxfp8-qkvo-rollout:0"
  "qkvo-optimized:grpo-nanov3-30BA3B-4n4g-megatron-pack-cp-mxfp8-qkvo-rollout:1"
)

arm_is_selected() {
  local arm=$1
  local selected_arm
  local -a selected_arms

  [[ -z "$ARM_FILTER" ]] && return 0
  IFS=, read -r -a selected_arms <<<"$ARM_FILTER"
  for selected_arm in "${selected_arms[@]}"; do
    [[ "$selected_arm" == "$arm" ]] && return 0
  done
  return 1
}

REPO_SHA=$(git -C "$REPO" rev-parse HEAD)
MANIFEST=$WORK/manifests/submission-$RUN_SUFFIX.tsv
printf 'arm\taction\tjob_id\trepo_sha\tcontainer\trun_name\n' >"$MANIFEST"
selected_count=0

for arm_spec in "${ARMS[@]}"; do
  IFS=: read -r ARM CONFIG_NAME REFIT_OPT <<<"$arm_spec"
  if ! arm_is_selected "$ARM"; then
    continue
  fi
  selected_count=$((selected_count + 1))
  RUN_NAME="${JOB_PREFIX}-${ARM}-${MAX_STEPS}step-${RUN_SUFFIX}"

  args=(
    --account="$SLURM_ACCOUNT"
    --partition="$PARTITION"
    --nodes="$NUM_NODES"
    --ntasks-per-node=1
    --time="$WALLTIME"
    --job-name="$RUN_NAME"
    --output="$WORK/slurm/%x-%j.out"
    --export="ALL,ARM=$ARM,CONFIG_NAME=$CONFIG_NAME,REFIT_OPT=$REFIT_OPT,RUN_NAME=$RUN_NAME,MAX_STEPS=$MAX_STEPS,BASE=$BASE,REPO=$REPO,WORK=$WORK,CONTAINER=$CONTAINER,NANO_MODEL_PATH=$NANO_MODEL_PATH,NRL_HF_HOME=$NRL_HF_HOME,NUM_NODES=$NUM_NODES,GPUS_PER_NODE=$GPUS_PER_NODE,EXPECTED_REPO_SHA=$REPO_SHA,WANDB_PROJECT=$WANDB_PROJECT,WANDB_ENTITY=$WANDB_ENTITY,EXPERIMENT_CLUSTER=$EXPERIMENT_CLUSTER,TENSORBOARD_ENABLED=$TENSORBOARD_ENABLED"
  )
  case "$GPU_REQUEST_MODE" in
    none)
      ;;
    gres)
      args+=(--gres="gpu:$GPUS_PER_NODE")
      ;;
    gpus-per-node)
      args+=(--gpus-per-node="$GPUS_PER_NODE")
      ;;
    *)
      echo "GPU_REQUEST_MODE must be none, gres, or gpus-per-node" >&2
      exit 2
      ;;
  esac
  if [[ -n "$SLURM_NETWORK" ]]; then
    args+=(--network="$SLURM_NETWORK")
  fi
  if [[ -n "$SLURM_SEGMENT" ]]; then
    args+=(--segment="$SLURM_SEGMENT")
  fi
  if [[ -n "$SBATCH_COMMENT" ]]; then
    args+=(--comment="$SBATCH_COMMENT")
  fi
  if [[ -n "$DEPENDENCY" ]]; then
    args+=(--dependency="$DEPENDENCY")
  fi
  if [[ -n "$ACTION_ARG" ]]; then
    args+=("$ACTION_ARG")
  fi

  output=$(sbatch "${args[@]}" "$BATCH_SCRIPT")
  job_id=$(sed -n 's/^Submitted batch job //p' <<<"$output")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ARM" "$ACTION" "${job_id:-n/a}" "$REPO_SHA" "$CONTAINER" "$RUN_NAME" \
    | tee -a "$MANIFEST"
done

if [[ "$selected_count" -eq 0 ]]; then
  echo "ARM_FILTER did not match any arm: $ARM_FILTER" >&2
  exit 2
fi

echo "manifest=$MANIFEST"
