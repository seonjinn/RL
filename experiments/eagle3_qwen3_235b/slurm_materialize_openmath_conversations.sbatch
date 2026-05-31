#!/bin/bash
#SBATCH --job-name=q235b-openmath-50k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
SCRIPT="$REPO_ROOT/experiments/eagle3_qwen3_235b/materialize_openmath_conversations.py"

OUTPUT_DATA="${OUTPUT_DATA:?set OUTPUT_DATA}"
DATASET="${DATASET:-nvidia/OpenMathInstruct-2}"
SPLIT="${SPLIT:-train_1M}"
LIMIT="${LIMIT:-50000}"
OFFSET="${OFFSET:-4096}"
MAX_SCAN="${MAX_SCAN:-200000}"
MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
EXCLUDE_PROMPTS_FROM="${EXCLUDE_PROMPTS_FROM:-}"
ALLOW_DUPLICATE_PROMPTS="${ALLOW_DUPLICATE_PROMPTS:-false}"
CONTAINER="${CONTAINER:-}"
MOUNTS="${MOUNTS:-}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$REPO_ROOT/logs" "$(dirname "$OUTPUT_DATA")"

SRUN_CONTAINER_ARGS=()
if [[ -n "$CONTAINER" ]]; then
  SRUN_CONTAINER_ARGS+=(--container-image="$CONTAINER")
  [[ -n "$MOUNTS" ]] && SRUN_CONTAINER_ARGS+=(--container-mounts="$MOUNTS")
  SRUN_CONTAINER_ARGS+=(--container-workdir="$REPO_ROOT")
fi

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks=1 bash -lc '
  set -euo pipefail
  export HF_HOME="'"$HF_HOME"'"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-'"$HF_HOME"'/datasets}"
  cd "'"$REPO_ROOT"'"
  materialize_cmd=("'"$PYTHON_BIN"'" "'"$SCRIPT"'" \
    --output "'"$OUTPUT_DATA"'" \
    --dataset "'"$DATASET"'" \
    --split "'"$SPLIT"'" \
    --limit "'"$LIMIT"'" \
    --offset "'"$OFFSET"'" \
    --max-scan "'"$MAX_SCAN"'" \
    --model "'"$MODEL"'")
  if [[ -n "'"$EXCLUDE_PROMPTS_FROM"'" ]]; then
    for exclude_file in '"$EXCLUDE_PROMPTS_FROM"'; do
      materialize_cmd+=(--exclude-prompts-from "$exclude_file")
    done
  fi
  if [[ "'"$ALLOW_DUPLICATE_PROMPTS"'" == "true" || "'"$ALLOW_DUPLICATE_PROMPTS"'" == "True" ]]; then
    materialize_cmd+=(--allow-duplicate-prompts)
  fi
  printf "%q " "${materialize_cmd[@]}"; printf "\n"
  "${materialize_cmd[@]}"
'
