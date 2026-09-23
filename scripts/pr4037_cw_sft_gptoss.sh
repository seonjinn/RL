#!/bin/bash

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to an immutable squashfs image}"
[[ -f "$CONTAINER" && ! -L "$CONTAINER" ]]

experiment_dir=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr4037-validation-20260923
archive="$experiment_dir/pr4037-source-d72a905d-20260923.tar.gz"
archive_sha256=b019d10689de13f0d4de9e25994cf904845e7939215ea2a36431e0726a878eb8
ray_sub=/home/sna/job-scripts/hybridep/pr4037_ray.sub

[[ -f "$archive" && -f "$ray_sub" ]]
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum -c -

export CONTAINER
export MOUNTS=/lustre:/lustre,/raid/scratch:/raid/scratch
export BASE_LOG_DIR="$experiment_dir"
export GPUS_PER_NODE=8
export UV_CACHE_DIR_OVERRIDE=/raid/scratch/sna/pr4037-uv-cache
export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home
export NRL_FORCE_REBUILD_VENVS=true

# shellcheck disable=SC2016
export COMMAND='set -euo pipefail
mkdir -p /raid/scratch/sna
scratch_dir="$(mktemp -d /raid/scratch/sna/pr4037-d72a905d-XXXXXX)"
archive=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr4037-validation-20260923/pr4037-source-d72a905d-20260923.tar.gz
tar -xzf "$archive" -C "$scratch_dir"
cd "$scratch_dir"
test -f 3rdparty/Automodel-workspace/Automodel/pyproject.toml
test -f 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/pyproject.toml
export HF_DATASETS_CACHE="$scratch_dir/hf_datasets"
export UV_HTTP_TIMEOUT=600
uv run examples/run_sft.py \
  --config examples/configs/recipes/llm/sft-gpt-oss-20b-1n8g-fsdp8ep8-automodel.yaml \
  sft.max_num_steps=50 \
  sft.val_period=10 \
  checkpointing.enabled=false \
  logger.log_dir=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr4037-validation-20260923/gptoss-sft-50 \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=false \
  cluster.num_nodes=1'

cd "$experiment_dir"
sbatch "${1:---test-only}" \
  --account=coreai_dlalgo_nemorl \
  --partition=batch \
  --nodes=1 \
  --gres=gpu:8 \
  --time=02:00:00 \
  --job-name=coreai_dlalgo_nemorl-pr4037-sft-gptoss \
  --output="$experiment_dir/sft-gptoss-%j.log" \
  "$ray_sub"
