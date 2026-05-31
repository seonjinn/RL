#!/usr/bin/env bash
set -euo pipefail

# Run inside an already-started Ray allocation. It starts a vLLM OpenAI server
# for Qwen3-235B, generates a small OpenMathInstruct conversation corpus, and
# validates the ModelOpt input JSONL.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
DATA_DIR="${DATA_DIR:-$ARTIFACT_ROOT/data}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
SOURCE_VLLM_SITE="${SOURCE_VLLM_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
HF_HOME_PATH="${HF_HOME_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"

PROMPT_DATA="${PROMPT_DATA:-$DATA_DIR/openmath_direct_vllm_prompts_smoke.jsonl}"
OUTPUT_CONVERSATIONS="${OUTPUT_CONVERSATIONS:-$DATA_DIR/qwen3_235b_math_direct_vllm_conversations_smoke.jsonl}"
VALIDATION_JSON="${VALIDATION_JSON:-${OUTPUT_CONVERSATIONS%.jsonl}.validation.json}"
SERVER_LOG="${SERVER_LOG:-$REPORT_DIR/direct_vllm_math_rollout_server.log}"
GENERATION_LOG="${GENERATION_LOG:-$REPORT_DIR/direct_vllm_math_rollout_generation.log}"
SUMMARY_JSON="${SUMMARY_JSON:-$REPORT_DIR/direct_vllm_math_rollout_summary.json}"

OPENMATH_SPLIT="${OPENMATH_SPLIT:-train_1M}"
PROMPT_LIMIT="${PROMPT_LIMIT:-8}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
NUM_RESPONSES="${NUM_RESPONSES:-1}"
GENERATION_CONCURRENCY="${GENERATION_CONCURRENCY:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
OUTPUT_SCHEMA="${OUTPUT_SCHEMA:-modelopt}"
LIMIT="${LIMIT:-}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
APPEND="${APPEND:-false}"
GENERATION_SKIP_FAILED="${GENERATION_SKIP_FAILED:-false}"
ID_KEY="${ID_KEY:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_TP="${VLLM_TP:-8}"
VLLM_PP="${VLLM_PP:-1}"
VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-ray}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.82}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS:-true}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_SPECULATIVE_CONFIG="${VLLM_SPECULATIVE_CONFIG:-}"
VLLM_SPECULATIVE_CONFIG_FILE="${VLLM_SPECULATIVE_CONFIG_FILE:-}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-1800}"
SKIP_PROMPT_MATERIALIZE="${SKIP_PROMPT_MATERIALIZE:-false}"

mkdir -p "$REPORT_DIR" "$DATA_DIR" "$(dirname "$OUTPUT_CONVERSATIONS")"

export PYTHONPATH="$SOURCE_VLLM_SITE:$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME="$HF_HOME_PATH"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME_PATH/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME_PATH/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ARTIFACT_ROOT/vllm_cache}"
export VLLM_CONFIGURE_LOGGING="${VLLM_CONFIGURE_LOGGING:-1}"
export VLLM_DISABLE_USAGE_STATS=1
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_USE_RAY_COMPILED_DAG="${VLLM_USE_RAY_COMPILED_DAG:-0}"
export VLLM_USE_RAY_SPMD_WORKER="${VLLM_USE_RAY_SPMD_WORKER:-0}"
export VLLM_USE_RAY_WRAPPED_PP_COMM="${VLLM_USE_RAY_WRAPPED_PP_COMM:-0}"

if [[ -n "$VLLM_SPECULATIVE_CONFIG_FILE" ]]; then
  if [[ ! -s "$VLLM_SPECULATIVE_CONFIG_FILE" ]]; then
    echo "VLLM_SPECULATIVE_CONFIG_FILE is missing or empty: $VLLM_SPECULATIVE_CONFIG_FILE" >&2
    exit 1
  fi
  VLLM_SPECULATIVE_CONFIG="$(tr -d '\n' < "$VLLM_SPECULATIVE_CONFIG_FILE")"
fi

if [[ "$SKIP_PROMPT_MATERIALIZE" == "true" || "$SKIP_PROMPT_MATERIALIZE" == "True" ]]; then
  if [[ ! -s "$PROMPT_DATA" ]]; then
    echo "SKIP_PROMPT_MATERIALIZE=true but PROMPT_DATA is missing or empty: $PROMPT_DATA" >&2
    exit 1
  fi
  echo "Using existing prompt data: $PROMPT_DATA"
else
  "$PYTHON_BIN" "$EXP_DIR/materialize_openmath_prompts.py" \
    --output "$PROMPT_DATA" \
    --split "$OPENMATH_SPLIT" \
    --limit "$PROMPT_LIMIT" \
    --offset "$PROMPT_OFFSET"
fi

server_args=(
  -m vllm.entrypoints.openai.api_server
  --model "$MODEL_PATH"
  --served-model-name "$MODEL_PATH"
  --tokenizer "$MODEL_PATH"
  --host 0.0.0.0
  --port "$VLLM_PORT"
  --dtype bfloat16
  --tensor-parallel-size "$VLLM_TP"
  --pipeline-parallel-size "$VLLM_PP"
  --distributed-executor-backend "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$VLLM_GPU_UTIL"
  --trust-remote-code
)

case "$VLLM_ENFORCE_EAGER" in
  true|True|TRUE|1|yes|Yes|YES) server_args+=(--enforce-eager) ;;
esac
case "$VLLM_DISABLE_LOG_STATS" in
  true|True|TRUE|1|yes|Yes|YES) server_args+=(--disable-log-stats) ;;
esac
if [[ -n "$VLLM_SPECULATIVE_CONFIG" ]]; then
  server_args+=(--speculative-config "$VLLM_SPECULATIVE_CONFIG")
fi
if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=($VLLM_EXTRA_ARGS)
  server_args+=("${extra_args[@]}")
fi

echo "Starting direct vLLM server on port $VLLM_PORT" | tee "$SERVER_LOG"
"$PYTHON_BIN" "${server_args[@]}" >>"$SERVER_LOG" 2>&1 &
server_pid=$!

cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" || true
    wait "$server_pid" || true
  fi
}
trap cleanup EXIT

deadline=$((SECONDS + SERVER_READY_TIMEOUT_SEC))
while true; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM server exited early; tail follows" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${VLLM_PORT}/v1/models", timeout=5).read()
PY
  then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM server; tail follows" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done

echo "vLLM server is ready; generating conversations" | tee "$GENERATION_LOG"
MODE=generate \
PROMPT_DATA="$PROMPT_DATA" \
OUTPUT_DATA="$OUTPUT_CONVERSATIONS" \
VALIDATION_JSON="$VALIDATION_JSON" \
OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1" \
OPENAI_API_KEY=EMPTY \
MODEL_PATH="$MODEL_PATH" \
OUTPUT_SCHEMA="$OUTPUT_SCHEMA" \
NUM_RESPONSES="$NUM_RESPONSES" \
GENERATION_CONCURRENCY="$GENERATION_CONCURRENCY" \
GENERATION_SKIP_FAILED="$GENERATION_SKIP_FAILED" \
TEMPERATURE="$TEMPERATURE" \
TOP_P="$TOP_P" \
MAX_TOKENS="$MAX_TOKENS" \
LIMIT="$LIMIT" \
SAMPLE_OFFSET="$SAMPLE_OFFSET" \
APPEND="$APPEND" \
ID_KEY="$ID_KEY" \
MAX_SEQ_LEN="$MAX_MODEL_LEN" \
TOKENIZER="$MODEL_PATH" \
TRUST_REMOTE_CODE=true \
DRY_RUN=false \
  bash "$EXP_DIR/prepare_training_conversations.sh" 2>&1 | tee -a "$GENERATION_LOG"

"$PYTHON_BIN" - <<PY
import json
import time
from pathlib import Path

output = Path("$OUTPUT_CONVERSATIONS")
validation = Path("$VALIDATION_JSON")
summary = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "overall_status": "pass" if output.exists() and output.stat().st_size > 0 and validation.exists() else "fail",
    "prompt_data": "$PROMPT_DATA",
    "output_conversations": str(output),
    "validation_json": str(validation),
    "server_log": "$SERVER_LOG",
    "generation_log": "$GENERATION_LOG",
    "records": sum(1 for _ in output.open(encoding="utf-8")) if output.exists() else 0,
}
Path("$SUMMARY_JSON").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary["overall_status"] == "pass" else 1)
PY
