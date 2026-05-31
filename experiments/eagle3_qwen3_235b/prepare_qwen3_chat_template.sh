#!/usr/bin/env bash
set -euo pipefail

# Prepare and validate a Qwen3 chat template with Hugging Face generation tags.
# ModelOpt answer-only loss needs these tags to build assistant-token labels.
#
# Source priority:
#   1. TEMPLATE=/path/template.jinja2
#   2. TOKENIZER_CONFIG=/path/tokenizer_config.json
#   3. BASE_MODEL or MODEL_OR_TOKENIZER HF id, fetched via raw tokenizer_config.json
#
# Example:
#   TOKENIZER_CONFIG=/path/to/Qwen3-235B-A22B-Thinking-2507/tokenizer_config.json \
#   OUTPUT_TEMPLATE=/path/to/qwen3_generation_template.jinja2 \
#   bash experiments/eagle3_qwen3_235b/prepare_qwen3_chat_template.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"
DRY_RUN="${DRY_RUN:-false}"

BASE_MODEL="${BASE_MODEL:-${MODEL_OR_TOKENIZER:-Qwen/Qwen3-235B-A22B-Thinking-2507}}"
MODEL_OR_TOKENIZER="${MODEL_OR_TOKENIZER:-$BASE_MODEL}"
REVISION="${REVISION:-main}"
TEMPLATE="${TEMPLATE:-}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-}"
OUTPUT_TEMPLATE="${OUTPUT_TEMPLATE:-$ROOT_DIR/outputs/qwen3_generation_template.jinja2}"
VALIDATION_JSON="${VALIDATION_JSON:-${OUTPUT_TEMPLATE%.jinja2}.mask_validation.json}"

FORCE="${FORCE:-true}"
NO_PATCH="${NO_PATCH:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
ALLOW_MISSING_TRANSFORMERS="${ALLOW_MISSING_TRANSFORMERS:-false}"

run_cmd() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
    return 0
  fi
  "$@"
}

prepare_cmd=(
  python3 "$EXP_DIR/prepare_qwen3_generation_template.py"
  --output "$OUTPUT_TEMPLATE"
)
if [[ -n "$TEMPLATE" ]]; then
  prepare_cmd+=(--template "$TEMPLATE")
elif [[ -n "$TOKENIZER_CONFIG" ]]; then
  prepare_cmd+=(--tokenizer-config "$TOKENIZER_CONFIG")
else
  prepare_cmd+=(--model "$BASE_MODEL" --revision "$REVISION")
fi
if [[ "$FORCE" == "true" || "$FORCE" == "True" ]]; then
  prepare_cmd+=(--force)
fi
if [[ "$NO_PATCH" == "true" || "$NO_PATCH" == "True" ]]; then
  prepare_cmd+=(--no-patch)
fi

run_cmd "${prepare_cmd[@]}"

validate_cmd=(
  python3 "$EXP_DIR/validate_chat_template_loss_mask.py"
  --model-or-tokenizer "$MODEL_OR_TOKENIZER"
  --chat-template "$OUTPUT_TEMPLATE"
  --json-out "$VALIDATION_JSON"
)
if [[ "$TRUST_REMOTE_CODE" == "true" || "$TRUST_REMOTE_CODE" == "True" ]]; then
  validate_cmd+=(--trust-remote-code)
fi
if [[ "$ALLOW_MISSING_TRANSFORMERS" == "true" || "$ALLOW_MISSING_TRANSFORMERS" == "True" ]]; then
  validate_cmd+=(--allow-missing-transformers)
fi

run_cmd "${validate_cmd[@]}"

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  echo "# output would be: $OUTPUT_TEMPLATE"
  echo "# validation summary would be: $VALIDATION_JSON"
fi
