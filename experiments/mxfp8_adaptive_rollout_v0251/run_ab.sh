#!/usr/bin/env bash
set -euo pipefail

ROOT=${NEMO_RL_REPO_ROOT:?set NEMO_RL_REPO_ROOT}
RESULT_ROOT=${CANARY_RESULT_ROOT:?set CANARY_RESULT_ROOT}
export NEMO_RL_REPO_ROOT="$ROOT"
export CANARY_RESULT_ROOT="$RESULT_ROOT"

bash "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh" baseline
bash "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh" adaptive
python3 -m experiments.mxfp8_adaptive_rollout_v0251.summarize \
  "$RESULT_ROOT/baseline/run.log" \
  "$RESULT_ROOT/adaptive/run.log" \
  --output "$RESULT_ROOT/summary.json"
cat "$RESULT_ROOT/summary.json"

