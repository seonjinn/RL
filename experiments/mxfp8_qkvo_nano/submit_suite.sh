#!/bin/bash

set -euo pipefail

BASE=${BASE:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
REPO=${REPO:-$BASE/RL-mxfp8-qkvo-pr3294-ab}
WORK=${WORK:-$BASE/experiments/mxfp8-qkvo-nano}
CONTAINER=${CONTAINER:-$BASE/containers/nemo_rl_nightly.sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/lustre:/lustre,/project:/project}
NANO_MODEL_PATH=${NANO_MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/models/nemotron-nano3/Ultra-SFTb2-512K-hermes20k-lr2e-5-iter_0005000/hf}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
USE_GRES=${USE_GRES:-0}
SLURM_NETWORK=${SLURM_NETWORK:-sharp}
DEPENDENCY=${DEPENDENCY:-}
JOB_PREFIX=${JOB_PREFIX:-coreai_dlalgo_llm-mxfp8.qkvo-nano}
ACTION=${ACTION:-test-only}
MAX_STEPS=${MAX_STEPS:-20}
ARM_FILTER=${ARM_FILTER:-}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
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

if [[ "$NUM_NODES" != "4" || "$GPUS_PER_NODE" != "4" ]]; then
  echo "Nano suite requires NUM_NODES=4 and GPUS_PER_NODE=4" >&2
  exit 2
fi

test -x "$BATCH_SCRIPT"
test -f "$NANO_MODEL_PATH/config.json"
test -f "$CONTAINER"
before_pull_sha=$(git -C "$REPO" rev-parse HEAD)
git -C "$REPO" -c fetch.recurseSubmodules=false \
  pull --ff-only --recurse-submodules=no
after_pull_sha=$(git -C "$REPO" rev-parse HEAD)
if [[ "$before_pull_sha" != "$after_pull_sha" && "${SUBMIT_SUITE_REEXEC:-0}" != "1" ]]; then
  export SUBMIT_SUITE_REEXEC=1
  exec "${BASH_SOURCE[0]}"
fi
git -C "$REPO" submodule update --init --recursive
mkdir -p "$WORK/slurm" "$WORK/manifests"
export CONTAINER_MOUNTS
export NANO_MODEL_PATH

ARMS=(
  "bf16:grpo-nanov3-30BA3B-2n8g-megatron-pack-cp:0"
  "moe-baseline:grpo-nanov3-30BA3B-2n8g-megatron-pack-cp-mxfp8-rollout:0"
  "moe-optimized:grpo-nanov3-30BA3B-2n8g-megatron-pack-cp-mxfp8-rollout:1"
  "qkvo-baseline:grpo-nanov3-30BA3B-2n8g-megatron-pack-cp-mxfp8-qkvo-rollout:0"
  "qkvo-optimized:grpo-nanov3-30BA3B-2n8g-megatron-pack-cp-mxfp8-qkvo-rollout:1"
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
printf 'arm\taction\tjob_id\trepo_sha\trun_name\n' >"$MANIFEST"
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
    --segment="$NUM_NODES"
    --job-name="$RUN_NAME"
    --output="$WORK/slurm/%x-%j.out"
    --export="ALL,ARM=$ARM,CONFIG_NAME=$CONFIG_NAME,REFIT_OPT=$REFIT_OPT,RUN_NAME=$RUN_NAME,MAX_STEPS=$MAX_STEPS,BASE=$BASE,REPO=$REPO,WORK=$WORK,CONTAINER=$CONTAINER,GPUS_PER_NODE=$GPUS_PER_NODE,NANO_MODEL_PATH=$NANO_MODEL_PATH"
  )
  if [[ "$USE_GRES" == "1" ]]; then
    args+=(--gres="gpu:$GPUS_PER_NODE")
  fi
  if [[ -n "$SLURM_NETWORK" ]]; then
    args+=(--network="$SLURM_NETWORK")
  fi
  if [[ -n "$DEPENDENCY" ]]; then
    args+=(--dependency="$DEPENDENCY")
  fi
  if [[ -n "$ACTION_ARG" ]]; then
    args+=("$ACTION_ARG")
  fi

  output=$(sbatch "${args[@]}" "$BATCH_SCRIPT")
  job_id=$(sed -n 's/^Submitted batch job //p' <<<"$output")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$ARM" "$ACTION" "${job_id:-n/a}" "$REPO_SHA" "$RUN_NAME" | tee -a "$MANIFEST"
done

if [[ "$selected_count" -eq 0 ]]; then
  echo "ARM_FILTER did not match any arm: $ARM_FILTER" >&2
  exit 2
fi

echo "manifest=$MANIFEST"
