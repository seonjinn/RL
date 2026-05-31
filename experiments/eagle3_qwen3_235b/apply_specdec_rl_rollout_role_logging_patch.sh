#!/usr/bin/env bash
set -euo pipefail

# Check or apply the SpecDec-RL train_data JSONL role logging patch.
#
# Defaults to check-only mode. Set APPLY=true to mutate the SpecDec-RL checkout.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

SPECDEC_RL_DIR="${SPECDEC_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}"
PATCH_FILE="${PATCH_FILE:-$EXP_DIR/specdec_rl_rollout_role_logging.patch}"
APPLY="${APPLY:-false}"

GRPO_PY="$SPECDEC_RL_DIR/nemo_rl/algorithms/grpo.py"

is_true() {
  [[ "$1" == "true" || "$1" == "True" || "$1" == "1" || "$1" == "yes" || "$1" == "YES" ]]
}

if [[ ! -d "$SPECDEC_RL_DIR" ]]; then
  echo "SpecDec-RL checkout not visible: $SPECDEC_RL_DIR" >&2
  exit 1
fi

if [[ ! -f "$GRPO_PY" ]]; then
  echo "SpecDec-RL grpo.py not visible: $GRPO_PY" >&2
  exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Patch file not visible: $PATCH_FILE" >&2
  exit 1
fi

if grep -q 'metrics_logging_data\["role"\]\|flat_messages_role\|log_data = {"content": flat_messages_content, "role": flat_messages_role}' "$GRPO_PY"; then
  echo "Role logging patch already appears to be applied: $GRPO_PY"
  exit 0
fi

if git -C "$SPECDEC_RL_DIR" apply --check "$PATCH_FILE"; then
  if is_true "$APPLY"; then
    git -C "$SPECDEC_RL_DIR" apply "$PATCH_FILE"
    echo "Applied role logging patch to: $SPECDEC_RL_DIR"
  else
    echo "Patch applies cleanly. Re-run with APPLY=true to apply it:"
    printf 'APPLY=true SPECDEC_RL_DIR=%q PATCH_FILE=%q bash %q\n' "$SPECDEC_RL_DIR" "$PATCH_FILE" "$0"
  fi
  exit 0
fi

echo "Patch does not apply cleanly. Inspect current SpecDec-RL diffs before editing." >&2
exit 1
