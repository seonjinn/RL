#!/bin/bash
#SBATCH --job-name=q235b-speculators-eagle3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
SCRIPT="$REPO_ROOT/experiments/eagle3_qwen3_235b/speculators_qwen3_235b_offline_pipeline.sh"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$REPO_ROOT:$REPO_ROOT}"

mkdir -p "$REPO_ROOT/logs"

SRUN_CONTAINER_ARGS=()
if [[ -n "$CONTAINER" ]]; then
  SRUN_CONTAINER_ARGS+=(--container-image="$CONTAINER")
  [[ -n "$MOUNTS" ]] && SRUN_CONTAINER_ARGS+=(--container-mounts="$MOUNTS")
  SRUN_CONTAINER_ARGS+=(--container-workdir="$REPO_ROOT")
fi

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks=1 bash -lc "
  set -euo pipefail
  cd '$REPO_ROOT'
  bash '$SCRIPT'
"
