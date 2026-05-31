#!/bin/bash
#SBATCH --job-name=q235b-eagle3-preflight
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Fast gate before expensive Qwen3-235B hidden-state dump/training jobs.
# Run this inside the same container/module environment intended for ModelOpt.

REPO_ROOT="${REPO_ROOT:-$PWD}"
SCRIPT="$REPO_ROOT/experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELOPT_DIR="${MODELOPT_DIR:-$REPO_ROOT/Model-Optimizer}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
REFERENCE_ARCH="${REFERENCE_ARCH:-$REPO_ROOT/experiments/eagle3_qwen3_235b/qwen3_235b_thinking_eagle3_architecture.json}"
PREFLIGHT_REQUIRE_MODELOPT_IMPORT="${PREFLIGHT_REQUIRE_MODELOPT_IMPORT:-true}"
PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK="${PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK:-true}"
PREFLIGHT_SKIP_EXISTING_PATH_CHECKS="${PREFLIGHT_SKIP_EXISTING_PATH_CHECKS:-false}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/outputs/qwen3_235b_eagle3}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-$ARTIFACT_ROOT/reports/container_preflight_pipeline.json}"
PREFLIGHT_MARKDOWN="${PREFLIGHT_MARKDOWN:-$ARTIFACT_ROOT/reports/container_preflight_pipeline.md}"
CONTAINER="${CONTAINER:-}"
MOUNTS="${MOUNTS:-}"

SRUN_CONTAINER_ARGS=()
if [[ -n "$CONTAINER" ]]; then
  SRUN_CONTAINER_ARGS+=(--container-image="$CONTAINER")
  [[ -n "$MOUNTS" ]] && SRUN_CONTAINER_ARGS+=(--container-mounts="$MOUNTS")
  SRUN_CONTAINER_ARGS+=(--container-workdir="$REPO_ROOT")
fi

mkdir -p logs
mkdir -p "$(dirname "$PREFLIGHT_JSON")" "$(dirname "$PREFLIGHT_MARKDOWN")"

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks=1 bash -lc '
  set -euo pipefail
  cd "'"$REPO_ROOT"'"

  args=(
    "'"$SCRIPT"'"
    --modelopt-dir "'"$MODELOPT_DIR"'"
    --base-model "'"$BASE_MODEL"'"
    --reference-arch "'"$REFERENCE_ARCH"'"
    --json-out "'"$PREFLIGHT_JSON"'"
    --markdown-out "'"$PREFLIGHT_MARKDOWN"'"
  )

  if [[ "'"$PREFLIGHT_REQUIRE_MODELOPT_IMPORT"'" == "true" || "'"$PREFLIGHT_REQUIRE_MODELOPT_IMPORT"'" == "True" ]]; then
    args+=(--require-modelopt-import)
  fi

  if [[ "'"$PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK"'" == "true" || "'"$PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK"'" == "True" ]]; then
    args+=(--require-chat-template-mask)
  fi

  if [[ "'"$PREFLIGHT_SKIP_EXISTING_PATH_CHECKS"'" == "true" || "'"$PREFLIGHT_SKIP_EXISTING_PATH_CHECKS"'" == "True" ]]; then
    args+=(--skip-existing-path-checks)
  fi

  "'"$PYTHON_BIN"'" "${args[@]}"
'
