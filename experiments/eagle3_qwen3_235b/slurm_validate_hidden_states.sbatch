#!/bin/bash
#SBATCH --job-name=q235b-eagle3-validate-hiddens
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Cheap gate between hidden-state dump and ModelOpt training.
# Required at submission time:
#   sbatch --account=<account> \
#     --export=ALL,HIDDEN_STATES_DIR=/path/hiddens \
#     experiments/eagle3_qwen3_235b/slurm_validate_hidden_states.sbatch

REPO_ROOT="${REPO_ROOT:-$PWD}"
SCRIPT="$REPO_ROOT/experiments/eagle3_qwen3_235b/validate_hidden_state_dump.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ARCH_ENV_FILE="${ARCH_ENV_FILE:-}"
if [[ -n "$ARCH_ENV_FILE" ]]; then
  if [[ ! -f "$ARCH_ENV_FILE" ]]; then
    echo "ARCH_ENV_FILE does not exist: $ARCH_ENV_FILE" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$ARCH_ENV_FILE"
fi
HIDDEN_STATES_VALIDATE_LIMIT="${HIDDEN_STATES_VALIDATE_LIMIT:-64}"
EXPECTED_HIDDEN_SIZE="${EXPECTED_HIDDEN_SIZE:-4096}"
EXPECTED_AUX_COUNT="${EXPECTED_AUX_COUNT:-3}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
REQUIRE_LOSS_MASK="${REQUIRE_LOSS_MASK:-true}"
REQUIRE_POSITIVE_LOSS_MASK="${REQUIRE_POSITIVE_LOSS_MASK:-true}"
VALIDATE_MODELOPT_LOADER="${VALIDATE_MODELOPT_LOADER:-true}"
MODELOPT_DIR="${MODELOPT_DIR:-$REPO_ROOT/Model-Optimizer}"
HIDDEN_STATES_VALIDATION_JSON="${HIDDEN_STATES_VALIDATION_JSON:-}"
CONTAINER="${CONTAINER:-${EAGLE_CONTAINER:-${EAGLE_IMAGE:-}}}"
MOUNTS="${MOUNTS:-${EAGLE_MOUNTS:-${EAGLE_BIND:-}}}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$REPO_ROOT}"

SRUN_CONTAINER_ARGS=()
if [[ -n "$CONTAINER" ]]; then
  echo "Using container image: $CONTAINER"
  SRUN_CONTAINER_ARGS+=(--container-image="$CONTAINER")
  [[ -n "$MOUNTS" ]] && SRUN_CONTAINER_ARGS+=(--container-mounts="$MOUNTS")
  SRUN_CONTAINER_ARGS+=(--container-workdir="$CONTAINER_WORKDIR")
else
  echo "No container image configured; running on host Python." >&2
fi

mkdir -p logs

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks=1 bash -lc '
  set -euo pipefail
  cd "'"$REPO_ROOT"'"

  args=(
    "'"$SCRIPT"'"
    "${HIDDEN_STATES_DIR:?set HIDDEN_STATES_DIR}"
    --limit "'"$HIDDEN_STATES_VALIDATE_LIMIT"'"
    --expected-hidden-size "'"$EXPECTED_HIDDEN_SIZE"'"
    --expected-aux-count "'"$EXPECTED_AUX_COUNT"'"
    --max-seq-len "'"$MAX_SEQ_LEN"'"
    --modelopt-dir "'"$MODELOPT_DIR"'"
  )

  if [[ "'"$REQUIRE_LOSS_MASK"'" == "true" || "'"$REQUIRE_LOSS_MASK"'" == "True" ]]; then
    args+=(--require-loss-mask)
  fi

  if [[ "'"$REQUIRE_POSITIVE_LOSS_MASK"'" == "true" || "'"$REQUIRE_POSITIVE_LOSS_MASK"'" == "True" ]]; then
    args+=(--require-positive-loss-mask)
  fi

  if [[ "'"$VALIDATE_MODELOPT_LOADER"'" == "true" || "'"$VALIDATE_MODELOPT_LOADER"'" == "True" ]]; then
    args+=(--validate-modelopt-loader)
  fi

  if [[ -n "'"$HIDDEN_STATES_VALIDATION_JSON"'" ]]; then
    args+=(--json-out "'"$HIDDEN_STATES_VALIDATION_JSON"'")
  fi

  "'"$PYTHON_BIN"'" "${args[@]}"
'
