#!/bin/bash

set -euo pipefail

BASE=${BASE:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna}
REPO=${REPO:-$BASE/RL-mxfp8-qkvo-pr3294-ab}
WORK=${WORK:-$BASE/experiments/mxfp8-qkvo-pr3294-ab}
CONTAINER=${CONTAINER:-$BASE/containers/nemo_rl_nightly.sqsh}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-nemotron_sw_post}
PARTITION=${PARTITION:-batch}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
ACTION=${ACTION:-test-only}
MAX_STEPS=${MAX_STEPS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
BATCH_SCRIPT=$REPO/experiments/mxfp8_qkvo_pr3294/run_arm.sbatch

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

test -x "$BATCH_SCRIPT"
git -C "$REPO" pull --ff-only
git -C "$REPO" submodule update --init --recursive
mkdir -p "$WORK/slurm" "$WORK/manifests"

ARMS=(
  "moe-baseline:grpo-qwen3-30ba3b-4n4g-mxfp8-rollout:0"
  "moe-optimized:grpo-qwen3-30ba3b-4n4g-mxfp8-rollout:1"
  "qkvo-baseline:grpo-qwen3-30ba3b-4n4g-mxfp8-qkvo-rollout:0"
  "qkvo-optimized:grpo-qwen3-30ba3b-4n4g-mxfp8-qkvo-rollout:1"
)

REPO_SHA=$(git -C "$REPO" rev-parse HEAD)
MANIFEST=$WORK/manifests/submission-$RUN_SUFFIX.tsv
printf 'arm\taction\tjob_id\trepo_sha\trun_name\n' >"$MANIFEST"

for arm_spec in "${ARMS[@]}"; do
  IFS=: read -r ARM CONFIG_NAME REFIT_OPT <<<"$arm_spec"
  RUN_NAME="aws-dfw-pr3294-${ARM}-${MAX_STEPS}step-${RUN_SUFFIX}"

  args=(
    --account="$SLURM_ACCOUNT"
    --partition="$PARTITION"
    --nodes="$NUM_NODES"
    --ntasks-per-node=1
    --gres="gpu:$GPUS_PER_NODE"
    --segment="$NUM_NODES"
    --job-name="$RUN_NAME"
    --output="$WORK/slurm/%x-%j.out"
    --export="ALL,ARM=$ARM,CONFIG_NAME=$CONFIG_NAME,REFIT_OPT=$REFIT_OPT,RUN_NAME=$RUN_NAME,MAX_STEPS=$MAX_STEPS,BASE=$BASE,REPO=$REPO,WORK=$WORK,CONTAINER=$CONTAINER,GPUS_PER_NODE=$GPUS_PER_NODE"
  )
  if [[ -n "$ACTION_ARG" ]]; then
    args+=("$ACTION_ARG")
  fi

  output=$(sbatch "${args[@]}" "$BATCH_SCRIPT")
  job_id=$(sed -n 's/^Submitted batch job //p' <<<"$output")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$ARM" "$ACTION" "${job_id:-n/a}" "$REPO_SHA" "$RUN_NAME" | tee -a "$MANIFEST"
done

echo "manifest=$MANIFEST"
