#!/usr/bin/env bash
set -euo pipefail

# Convert NeMo-RL rollout capture train_data_step*.jsonl files into ModelOpt
# Eagle3 conversation JSONL and validate the result.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke}"
OUTPUT_DATA="${OUTPUT_DATA:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}"
VALIDATION_JSON="${VALIDATION_JSON:-${OUTPUT_DATA%.jsonl}.validation.json}"
DISCOVERY_JSON="${DISCOVERY_JSON:-${OUTPUT_DATA%.jsonl}.discovery.json}"
DISCOVERY_MARKDOWN="${DISCOVERY_MARKDOWN:-${OUTPUT_DATA%.jsonl}.discovery.md}"
INFER_FLAT_CONTENT_ROLES="${INFER_FLAT_CONTENT_ROLES:-false}"
COMPACT_CURRENT_TURN="${COMPACT_CURRENT_TURN:-false}"
INCLUDE_REASONING_CONTENT="${INCLUDE_REASONING_CONTENT:-false}"
MIN_ASSISTANT_CHARS="${MIN_ASSISTANT_CHARS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
OUTPUT_SCHEMA="${OUTPUT_SCHEMA:-modelopt}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-false}"

shopt -s nullglob
files=(
  "$ROLLOUT_LOG_DIR"/train_data_step*.jsonl
  "$ROLLOUT_LOG_DIR"/exp_*/train_data_step*.jsonl
)
shopt -u nullglob

if (( ${#files[@]} == 0 )); then
  echo "No train_data_step*.jsonl files found under: $ROLLOUT_LOG_DIR" >&2
  echo "Expected a completed rollout capture run before materializing corpus." >&2
  exit 1
fi

printf 'Rollout capture files (%d):\n' "${#files[@]}"
printf '  %s\n' "${files[@]}"

cmd=(
  env
  MODE=rollout
  INPUT_PATHS="${files[*]}"
  OUTPUT_DATA="$OUTPUT_DATA"
  VALIDATION_JSON="$VALIDATION_JSON"
  DISCOVERY_JSON="$DISCOVERY_JSON"
  DISCOVERY_MARKDOWN="$DISCOVERY_MARKDOWN"
  INCLUDE_METADATA=true
  INFER_FLAT_CONTENT_ROLES="$INFER_FLAT_CONTENT_ROLES"
  COMPACT_CURRENT_TURN="$COMPACT_CURRENT_TURN"
  INCLUDE_REASONING_CONTENT="$INCLUDE_REASONING_CONTENT"
  MIN_ASSISTANT_CHARS="$MIN_ASSISTANT_CHARS"
  MAX_SEQ_LEN="$MAX_SEQ_LEN"
  OUTPUT_SCHEMA="$OUTPUT_SCHEMA"
  DRY_RUN="$DRY_RUN"
)
if [[ -n "$LIMIT" ]]; then
  cmd+=(LIMIT="$LIMIT")
fi
cmd+=(bash "$EXP_DIR/prepare_training_conversations.sh")

printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
