#!/usr/bin/env bash
set -euo pipefail

# Dump hidden states for offline Eagle3 training.
#
# Default backend is TRT-LLM because Qwen3-235B is too large for the simple HF
# path in most useful settings. Use BACKEND=hf only for small dry-runs.
#
# Required:
#   INPUT_DATA=/path/to/conversations.jsonl
#   HIDDEN_STATES_DIR=/path/to/hidden_states
#
# Example:
#   INPUT_DATA=/lustre/.../qwen3_235b_swe_rollout_conversations.jsonl \
#   HIDDEN_STATES_DIR=/lustre/.../qwen3_235b_eagle3_hidden_states \
#   DP_WORLD_SIZE=8 DP_RANK=${SLURM_PROCID:-0} TP=8 \
#   ANSWER_ONLY_LOSS=true CHAT_TEMPLATE=/path/to/qwen3_fixed_template.jinja2 \
#   bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_dump_hidden_states.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELOPT_DIR="${MODELOPT_DIR:-$ROOT_DIR/Model-Optimizer}"
DUMP_DIR="$MODELOPT_DIR/examples/speculative_decoding/collect_hidden_states"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-false}"
ARCH_ENV_FILE="${ARCH_ENV_FILE:-}"
LAUNCHER_PREFIX="${LAUNCHER_PREFIX:-}"

if [[ -n "$ARCH_ENV_FILE" ]]; then
  if [[ ! -f "$ARCH_ENV_FILE" ]]; then
    echo "ARCH_ENV_FILE does not exist: $ARCH_ENV_FILE" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$ARCH_ENV_FILE"
fi

BACKEND="${BACKEND:-trtllm}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
INPUT_DATA="${INPUT_DATA:?set INPUT_DATA to a .jsonl file or directory of .jsonl files}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:?set HIDDEN_STATES_DIR to the output hidden-state directory}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
DP_RANK="${DP_RANK:-0}"
DP_WORLD_SIZE="${DP_WORLD_SIZE:-1}"
AUX_LAYERS="${EAGLE_DUMP_AUX_LAYERS:-${AUX_LAYERS:-1,46,90}}"
AUX_LAYERS="${AUX_LAYERS#[}"
AUX_LAYERS="${AUX_LAYERS%]}"
AUX_LAYERS="${AUX_LAYERS// /}"
AUX_LAYERS="${AUX_LAYERS//:/,}"
AUX_LAYERS="${AUX_LAYERS//;/,}"
ANSWER_ONLY_LOSS="${ANSWER_ONLY_LOSS:-true}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
DEBUG_MAX_NUM_CONVERSATIONS="${DEBUG_MAX_NUM_CONVERSATIONS:-}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"

TP="${TP:-8}"
MOE_EP="${MOE_EP:-}"
MOE_TP="${MOE_TP:-}"
MOE_CP="${MOE_CP:-}"

case "$BACKEND" in
  trtllm)
    script="$DUMP_DIR/compute_hidden_states_trtllm.py"
    ;;
  hf)
    script="$DUMP_DIR/compute_hidden_states_hf.py"
    ;;
  *)
    echo "BACKEND must be 'trtllm' or 'hf', got: $BACKEND" >&2
    exit 1
    ;;
esac

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" && ! -f "$script" ]]; then
  echo "Missing hidden-state dumper: $script" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" ]]; then
  if [[ ! -e "$INPUT_DATA" ]]; then
    echo "INPUT_DATA is not visible: $INPUT_DATA" >&2
    exit 1
  fi
  if [[ "$ANSWER_ONLY_LOSS" == "true" || "$ANSWER_ONLY_LOSS" == "True" ]]; then
    if [[ -z "$CHAT_TEMPLATE" ]]; then
      echo "CHAT_TEMPLATE is required when ANSWER_ONLY_LOSS=true." >&2
      exit 1
    fi
    if [[ ! -f "$CHAT_TEMPLATE" ]]; then
      echo "CHAT_TEMPLATE is not visible: $CHAT_TEMPLATE" >&2
      exit 1
    fi
    if ! grep -q "generation" "$CHAT_TEMPLATE" || ! grep -q "endgeneration" "$CHAT_TEMPLATE"; then
      echo "CHAT_TEMPLATE lacks generation/endgeneration tags: $CHAT_TEMPLATE" >&2
      exit 1
    fi
  fi
  mkdir -p "$HIDDEN_STATES_DIR"
  cd "$DUMP_DIR"
fi

cmd=(
  "$PYTHON_BIN" "$(basename "$script")"
  --model "$BASE_MODEL"
  --input-data "$INPUT_DATA"
  --output-dir "$HIDDEN_STATES_DIR"
  --max-seq-len "$MAX_SEQ_LEN"
  --dp-rank "$DP_RANK"
  --dp-world-size "$DP_WORLD_SIZE"
  --aux-layers "$AUX_LAYERS"
)

if [[ "$ANSWER_ONLY_LOSS" == "true" || "$ANSWER_ONLY_LOSS" == "True" ]]; then
  cmd+=(--answer-only-loss)
fi

if [[ -n "$CHAT_TEMPLATE" ]]; then
  cmd+=(--chat-template "$CHAT_TEMPLATE")
fi

if [[ -n "$DEBUG_MAX_NUM_CONVERSATIONS" ]]; then
  cmd+=(--debug-max-num-conversations "$DEBUG_MAX_NUM_CONVERSATIONS")
fi

if [[ "$BACKEND" == "trtllm" ]]; then
  cmd+=(--tp "$TP")
  cmd+=(--max-inflight "$MAX_INFLIGHT")
  [[ -n "$MOE_EP" ]] && cmd+=(--moe-ep "$MOE_EP")
  [[ -n "$MOE_TP" ]] && cmd+=(--moe-tp "$MOE_TP")
  [[ -n "$MOE_CP" ]] && cmd+=(--moe-cp "$MOE_CP")
else
  if [[ "$TRUST_REMOTE_CODE" == "true" || "$TRUST_REMOTE_CODE" == "True" ]]; then
    cmd+=(--trust_remote_code)
  fi
fi

printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  echo "# run from: $DUMP_DIR"
  exit 0
fi
if [[ -n "$LAUNCHER_PREFIX" ]]; then
  # Intentionally split a simple launcher prefix such as:
  #   mpirun -n 1 --oversubscribe --allow-run-as-root
  # shellcheck disable=SC2206
  launcher=($LAUNCHER_PREFIX)
  exec "${launcher[@]}" "${cmd[@]}"
fi
exec "${cmd[@]}"
