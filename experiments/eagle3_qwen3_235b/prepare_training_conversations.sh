#!/usr/bin/env bash
set -euo pipefail

# Prepare ModelOpt Eagle3 conversation JSONL for Qwen3-235B.
#
# Modes:
#   discover: scan rollout roots, pick extractable JSONL files, normalize them
#   rollout:  normalize known rollout JSONL files or directories
#   existing: convert prompt/conversation rows that already include assistant text
#   generate: call an OpenAI-compatible endpoint to generate assistant text
#
# OUTPUT_SCHEMA=modelopt writes conversation_id/messages for the ModelOpt path.
# OUTPUT_SCHEMA=specforge writes id/conversations for SGLang SpecForge
# comparison runs. The expensive Qwen3-235B pipeline should keep modelopt.
#
# Examples:
#   MODE=discover ROLLOUT_ROOTS="/path/to/run1 /path/to/run2" OUTPUT_DATA=/path/conversations.jsonl \
#     bash experiments/eagle3_qwen3_235b/prepare_training_conversations.sh
#
#   MODE=generate PROMPT_DATA=/path/prompts.jsonl OPENAI_BASE_URL=http://localhost:8000/v1 \
#     MODEL_PATH=Qwen/Qwen3-235B-A22B-Thinking-2507 OUTPUT_DATA=/path/conversations.jsonl \
#     bash experiments/eagle3_qwen3_235b/prepare_training_conversations.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

MODE="${MODE:-discover}"
OUTPUT_DATA="${OUTPUT_DATA:-$ROOT_DIR/outputs/qwen3_235b_swe_rollout_conversations.jsonl}"
VALIDATION_JSON="${VALIDATION_JSON:-${OUTPUT_DATA%.jsonl}.validation.json}"
DISCOVERY_JSON="${DISCOVERY_JSON:-${OUTPUT_DATA%.jsonl}.discovery.json}"
DISCOVERY_MARKDOWN="${DISCOVERY_MARKDOWN:-${OUTPUT_DATA%.jsonl}.discovery.md}"
DRY_RUN="${DRY_RUN:-false}"

BASE_MODEL="${BASE_MODEL:-${MODEL_PATH:-Qwen/Qwen3-235B-A22B-Thinking-2507}}"
OUTPUT_SCHEMA="${OUTPUT_SCHEMA:-modelopt}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
LIMIT="${LIMIT:-}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
MIN_ASSISTANT_CHARS="${MIN_ASSISTANT_CHARS:-1}"
INCLUDE_METADATA="${INCLUDE_METADATA:-false}"
INCLUDE_REASONING_CONTENT="${INCLUDE_REASONING_CONTENT:-false}"
INFER_FLAT_CONTENT_ROLES="${INFER_FLAT_CONTENT_ROLES:-false}"
COMPACT_CURRENT_TURN="${COMPACT_CURRENT_TURN:-false}"
REASONING_OPEN_TAG="${REASONING_OPEN_TAG:-<think>
}"
REASONING_CLOSE_TAG="${REASONING_CLOSE_TAG:-
</think>

}"
ID_KEY="${ID_KEY:-}"
TOKENIZER="${TOKENIZER:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
REQUIRE_TOKENIZER="${REQUIRE_TOKENIZER:-false}"
FAIL_ON_OVERLENGTH="${FAIL_ON_OVERLENGTH:-false}"

ROLLOUT_ROOTS="${ROLLOUT_ROOTS:-}"
INPUT_PATHS="${INPUT_PATHS:-}"
PROMPT_DATA="${PROMPT_DATA:-}"
INPUT_DATA_SOURCE="${INPUT_DATA_SOURCE:-}"

NUM_RESPONSES="${NUM_RESPONSES:-1}"
GENERATION_CONCURRENCY="${GENERATION_CONCURRENCY:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-$MAX_SEQ_LEN}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
APPEND="${APPEND:-false}"
GENERATION_SKIP_FAILED="${GENERATION_SKIP_FAILED:-false}"

run_cmd() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
    return 0
  fi
  "$@"
}

mkdir -p "$(dirname "$OUTPUT_DATA")"

case "$MODE" in
  discover)
    if [[ -z "$ROLLOUT_ROOTS" ]]; then
      echo "MODE=discover requires ROLLOUT_ROOTS with one or more files/directories." >&2
      exit 1
    fi
    roots=()
    # shellcheck disable=SC2206
    roots=($ROLLOUT_ROOTS)
    discover_cmd=(
      python3 "$EXP_DIR/discover_rollout_conversation_sources.py"
      "${roots[@]}"
      --prepare-output "$OUTPUT_DATA"
      --model "$BASE_MODEL"
      --output-schema "$OUTPUT_SCHEMA"
      --json-out "$DISCOVERY_JSON"
      --markdown-out "$DISCOVERY_MARKDOWN"
      --min-assistant-chars "$MIN_ASSISTANT_CHARS"
    )
    if [[ "$INCLUDE_METADATA" == "true" || "$INCLUDE_METADATA" == "True" ]]; then
      discover_cmd+=(--include-metadata)
    fi
    if [[ "$INCLUDE_REASONING_CONTENT" == "true" || "$INCLUDE_REASONING_CONTENT" == "True" ]]; then
      discover_cmd+=(--include-reasoning-content --reasoning-open-tag "$REASONING_OPEN_TAG" --reasoning-close-tag "$REASONING_CLOSE_TAG")
    fi
    if [[ "$INFER_FLAT_CONTENT_ROLES" == "true" || "$INFER_FLAT_CONTENT_ROLES" == "True" ]]; then
      discover_cmd+=(--infer-flat-content-roles)
    fi
    if [[ "$COMPACT_CURRENT_TURN" == "true" || "$COMPACT_CURRENT_TURN" == "True" ]]; then
      discover_cmd+=(--compact-current-turn)
    fi
    if [[ -n "$ID_KEY" ]]; then
      discover_cmd+=(--id-key "$ID_KEY")
    fi
    if [[ -n "$LIMIT" ]]; then
      discover_cmd+=(--limit "$LIMIT")
    fi
    run_cmd "${discover_cmd[@]}"
    ;;
  rollout)
    if [[ -z "$INPUT_PATHS" ]]; then
      echo "MODE=rollout requires INPUT_PATHS with one or more rollout JSONL files/directories." >&2
      exit 1
    fi
    inputs=()
    # shellcheck disable=SC2206
    inputs=($INPUT_PATHS)
    rollout_cmd=(
      python3 "$EXP_DIR/normalize_rl_rollouts_to_conversations.py"
      --input "${inputs[@]}"
      --output "$OUTPUT_DATA"
      --model "$BASE_MODEL"
      --output-schema "$OUTPUT_SCHEMA"
      --min-assistant-chars "$MIN_ASSISTANT_CHARS"
    )
    if [[ "$INCLUDE_METADATA" == "true" || "$INCLUDE_METADATA" == "True" ]]; then
      rollout_cmd+=(--include-metadata)
    fi
    if [[ "$INCLUDE_REASONING_CONTENT" == "true" || "$INCLUDE_REASONING_CONTENT" == "True" ]]; then
      rollout_cmd+=(--include-reasoning-content --reasoning-open-tag "$REASONING_OPEN_TAG" --reasoning-close-tag "$REASONING_CLOSE_TAG")
    fi
    if [[ "$INFER_FLAT_CONTENT_ROLES" == "true" || "$INFER_FLAT_CONTENT_ROLES" == "True" ]]; then
      rollout_cmd+=(--infer-flat-content-roles)
    fi
    if [[ "$COMPACT_CURRENT_TURN" == "true" || "$COMPACT_CURRENT_TURN" == "True" ]]; then
      rollout_cmd+=(--compact-current-turn)
    fi
    if [[ "$APPEND" == "true" || "$APPEND" == "True" ]]; then
      rollout_cmd+=(--append)
    fi
    if [[ -n "$ID_KEY" ]]; then
      rollout_cmd+=(--id-key "$ID_KEY")
    fi
    if [[ -n "$LIMIT" ]]; then
      rollout_cmd+=(--limit "$LIMIT")
    fi
    run_cmd "${rollout_cmd[@]}"
    ;;
  existing)
    if [[ -z "$INPUT_DATA_SOURCE" ]]; then
      echo "MODE=existing requires INPUT_DATA_SOURCE JSONL with assistant responses." >&2
      exit 1
    fi
    existing_cmd=(
      python3 "$EXP_DIR/generate_training_conversations_openai.py"
      --input "$INPUT_DATA_SOURCE"
      --output "$OUTPUT_DATA"
      --model "$BASE_MODEL"
      --output-schema "$OUTPUT_SCHEMA"
      --use-existing-assistant
    )
    if [[ "$APPEND" == "true" || "$APPEND" == "True" ]]; then
      existing_cmd+=(--append)
    fi
    if [[ -n "$ID_KEY" ]]; then
      existing_cmd+=(--id-key "$ID_KEY")
    fi
    if [[ -n "$LIMIT" ]]; then
      existing_cmd+=(--limit "$LIMIT")
    fi
    if [[ -n "$SAMPLE_OFFSET" && "$SAMPLE_OFFSET" != "0" ]]; then
      existing_cmd+=(--offset "$SAMPLE_OFFSET")
    fi
    run_cmd "${existing_cmd[@]}"
    ;;
  generate)
    if [[ -z "$PROMPT_DATA" ]]; then
      echo "MODE=generate requires PROMPT_DATA JSONL." >&2
      exit 1
    fi
    generate_cmd=(
      python3 "$EXP_DIR/generate_training_conversations_openai.py"
      --input "$PROMPT_DATA"
      --output "$OUTPUT_DATA"
      --api-base "$OPENAI_BASE_URL"
      --api-key "$OPENAI_API_KEY"
      --model "$BASE_MODEL"
      --output-schema "$OUTPUT_SCHEMA"
      --num-responses "$NUM_RESPONSES"
      --concurrency "$GENERATION_CONCURRENCY"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --max-tokens "$MAX_TOKENS"
    )
    if [[ "$APPEND" == "true" || "$APPEND" == "True" ]]; then
      generate_cmd+=(--append)
    fi
    if [[ -n "$ID_KEY" ]]; then
      generate_cmd+=(--id-key "$ID_KEY")
    fi
    if [[ -n "$LIMIT" ]]; then
      generate_cmd+=(--limit "$LIMIT")
    fi
    if [[ -n "$SAMPLE_OFFSET" && "$SAMPLE_OFFSET" != "0" ]]; then
      generate_cmd+=(--offset "$SAMPLE_OFFSET")
    fi
    if [[ "$GENERATION_SKIP_FAILED" == "true" || "$GENERATION_SKIP_FAILED" == "True" ]]; then
      generate_cmd+=(--skip-failed)
    fi
    run_cmd "${generate_cmd[@]}"
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use discover, rollout, existing, or generate." >&2
    exit 1
    ;;
esac

validate_cmd=(
  python3 "$EXP_DIR/validate_training_conversations.py"
  "$OUTPUT_DATA"
  --max-seq-len "$MAX_SEQ_LEN"
  --min-assistant-chars "$MIN_ASSISTANT_CHARS"
  --json-out "$VALIDATION_JSON"
)
if [[ -n "$TOKENIZER" ]]; then
  validate_cmd+=(--tokenizer "$TOKENIZER")
fi
if [[ -n "$CHAT_TEMPLATE" ]]; then
  validate_cmd+=(--chat-template "$CHAT_TEMPLATE")
fi
if [[ "$TRUST_REMOTE_CODE" == "true" || "$TRUST_REMOTE_CODE" == "True" ]]; then
  validate_cmd+=(--trust-remote-code)
fi
if [[ "$REQUIRE_TOKENIZER" == "true" || "$REQUIRE_TOKENIZER" == "True" ]]; then
  validate_cmd+=(--require-tokenizer)
fi
if [[ "$FAIL_ON_OVERLENGTH" == "true" || "$FAIL_ON_OVERLENGTH" == "True" ]]; then
  validate_cmd+=(--fail-on-overlength)
fi

run_cmd "${validate_cmd[@]}"

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  echo "# output would be: $OUTPUT_DATA"
  echo "# validation summary would be: $VALIDATION_JSON"
fi
