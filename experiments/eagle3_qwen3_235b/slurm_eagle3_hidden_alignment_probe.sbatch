#!/bin/bash
#SBATCH --job-name=q235b-eagle3-align
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
MODELOPT_DIR="${MODELOPT_DIR:-$REPO_ROOT/Model-Optimizer}"
SCRIPT="$REPO_ROOT/experiments/eagle3_qwen3_235b/diagnose_eagle3_hidden_alignment.py"

BASE_MODEL="${BASE_MODEL:?set BASE_MODEL to the verifier HF checkpoint or local snapshot}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:?set HIDDEN_STATES_DIR}"
JSON_OUT="${JSON_OUT:?set JSON_OUT}"
SAMPLE_FILES="${SAMPLE_FILES:-8}"
MAX_POSITIONS_PER_FILE="${MAX_POSITIONS_PER_FILE:-256}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
CONTAINER="${CONTAINER:-}"
MOUNTS="${MOUNTS:-}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"

mkdir -p "$REPO_ROOT/logs" "$(dirname "$JSON_OUT")"

SRUN_CONTAINER_ARGS=()
if [[ -n "$CONTAINER" ]]; then
  SRUN_CONTAINER_ARGS+=(--container-image="$CONTAINER")
  [[ -n "$MOUNTS" ]] && SRUN_CONTAINER_ARGS+=(--container-mounts="$MOUNTS")
  SRUN_CONTAINER_ARGS+=(--container-workdir="$REPO_ROOT")
fi

TRUST_ARGS=()
if [[ "$TRUST_REMOTE_CODE" == "true" || "$TRUST_REMOTE_CODE" == "True" ]]; then
  TRUST_ARGS+=(--trust-remote-code)
fi

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks=1 bash -lc '
  set -euo pipefail
  export PYTHONPATH="'"$MODELOPT_DIR"':'"$REPO_ROOT"':${PYTHONPATH:-}"
  export HF_HOME="'"$HF_HOME"'"
  cd "'"$REPO_ROOT"'"
  python3 "'"$SCRIPT"'" \
    --hidden-states-dir "'"$HIDDEN_STATES_DIR"'" \
    --base-model "'"$BASE_MODEL"'" \
    --modelopt-dir "'"$MODELOPT_DIR"'" \
    --json-out "'"$JSON_OUT"'" \
    --sample-files "'"$SAMPLE_FILES"'" \
    --max-positions-per-file "'"$MAX_POSITIONS_PER_FILE"'" \
    --chunk-size "'"$CHUNK_SIZE"'" \
    '"${TRUST_ARGS[*]}"'
'
