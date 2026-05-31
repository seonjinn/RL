#!/usr/bin/env bash
set -euo pipefail

# Submit a short NeMo-RL SWE baseline job without speculative decoding.
# Use this as the apples-to-apples comparison for run_static_specdec_smoke.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MAX_NUM_STEPS="${MAX_NUM_STEPS:-1}"
WANDB_NAME="${WANDB_NAME:-qwen3-235b-swe-baseline-smoke}"
EXP_SUFFIX_OVERRIDE="${EXP_SUFFIX_OVERRIDE:-$WANDB_NAME}"
DRY_RUN="${DRY_RUN:-false}"

export MAX_NUM_STEPS
export WANDB_NAME
export EXP_SUFFIX_OVERRIDE
export SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-singleton}"
BASELINE_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-}"
if [[ -n "${BASELINE_EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  BASELINE_OVERRIDES="${BASELINE_OVERRIDES} ${BASELINE_EXTRA_HYDRA_OVERRIDES}"
fi
export EXTRA_HYDRA_OVERRIDES="${BASELINE_OVERRIDES}"

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  echo "MAX_NUM_STEPS=$MAX_NUM_STEPS"
  echo "WANDB_NAME=$WANDB_NAME"
  echo "EXP_SUFFIX_OVERRIDE=$EXP_SUFFIX_OVERRIDE"
  echo "SBATCH_DEPENDENCY=$SBATCH_DEPENDENCY"
  echo "EXTRA_HYDRA_OVERRIDES=$EXTRA_HYDRA_OVERRIDES"
  printf '%q\n' "$ROOT_DIR/run_grpo_qwen3_235b_swe.sh"
  exit 0
fi

exec "$ROOT_DIR/run_grpo_qwen3_235b_swe.sh"
