#!/bin/bash

set -euo pipefail

case_name=${1:?Usage: $0 clevr|mmpr|geo3k [--test-only|--submit]}
mode=${2:---test-only}
case "$mode" in
  --test-only|--submit) ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

case "$case_name" in
  clevr)
    script=vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v2
    nodes=1
    max_steps=10
    time_limit=02:00:00
    ;;
  mmpr)
    script=vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1
    nodes=4
    max_steps=10
    time_limit=02:00:00
    ;;
  geo3k)
    script=vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16
    nodes=2
    max_steps=20
    time_limit=04:00:00
    ;;
  *) echo "Unknown case: $case_name" >&2; exit 2 ;;
esac

experiment_dir=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr4037-nightly-validation-20260929
archive="$experiment_dir/pr4037-head17752c9e-source.tar.gz"
archive_sha256=15a5305ab615e07ad45d1be4d2bf0b4d0feba6cd0c3c8963d0587b4853604e6a
ray_sub=/home/sna/job-scripts/hybridep/pr4037_ray.sub
container=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo-rl-nightly-pr4037-20260929/nemo_rl_nightly_pr4037_20260929_19520749.sqsh}
run_id="$(date -u +%Y%m%dT%H%M%S)-$$"
result_dir="$experiment_dir/$case_name/$run_id"

[[ -f "$archive" && -f "$ray_sub" && -f "$container" && ! -L "$container" ]]
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum -c -
mkdir -p "$result_dir"

export CONTAINER="$container"
export MOUNTS=/lustre:/lustre,/raid/scratch:/raid/scratch
export BASE_LOG_DIR="$result_dir"
export GPUS_PER_NODE=8
export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export NRL_FORCE_REBUILD_VENVS=true
export UV_HTTP_TIMEOUT=600
export PR4037_ARCHIVE="$archive"
export PR4037_ARCHIVE_SHA256="$archive_sha256"
export PR4037_RESULT_DIR="$result_dir"
export PR4037_SCRIPT="$script"
export PR4037_MAX_STEPS="$max_steps"
export PR4037_RUN_ID="$run_id"

# shellcheck disable=SC2016
export SETUP_COMMAND='#!/bin/bash
set -euo pipefail
source_dir="/raid/scratch/sna/pr4037-17752c9e-${PR4037_RUN_ID}"
mkdir -p /raid/scratch/sna
(
  flock -x 9
  if [[ ! -f "$source_dir/.extract_complete" ]]; then
    mkdir -p "$source_dir"
    printf "%s  %s\n" "$PR4037_ARCHIVE_SHA256" "$PR4037_ARCHIVE" | sha256sum -c -
    tar -xzf "$PR4037_ARCHIVE" -C "$source_dir"
    test -f "$source_dir/3rdparty/Automodel-workspace/Automodel/pyproject.toml"
    test -f "$source_dir/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/pyproject.toml"
    touch "$source_dir/.extract_complete"
  fi
) 9>"$source_dir.lock"'

# shellcheck disable=SC2016
export COMMAND='#!/bin/bash
set -euo pipefail
source_dir="/raid/scratch/sna/pr4037-17752c9e-${PR4037_RUN_ID}"
test -f "$source_dir/.extract_complete"
ln -s "$PR4037_RESULT_DIR" "$source_dir/tests/test_suites/vlm/$PR4037_SCRIPT"
cd "$source_dir"
export HF_DATASETS_CACHE="/raid/scratch/sna/pr4037-hf-datasets-${PR4037_RUN_ID}"
mkdir -p "$HF_DATASETS_CACHE"
bash "tests/test_suites/vlm/$PR4037_SCRIPT.sh" \
  logger.wandb_enabled=false \
  checkpointing.enabled=false \
  logger.monitor_gpus=false
jq -e --argjson expected "$PR4037_MAX_STEPS" \
  '\''[.["train/loss"] | keys[] | tonumber] | max >= $expected'\'' \
  "$PR4037_RESULT_DIR/metrics.json"'

cd "$experiment_dir"
sbatch_args=()
if [[ "$mode" == --test-only ]]; then
  sbatch_args+=(--test-only)
fi
sbatch "${sbatch_args[@]}" \
  --account=coreai_dlalgo_nemorl \
  --partition=batch \
  --nodes="$nodes" \
  --gres=gpu:8 \
  --time="$time_limit" \
  --job-name="coreai_dlalgo_nemorl-pr4037-$case_name" \
  --output="$result_dir/slurm-%j.log" \
  "$ray_sub"
