#!/usr/bin/env bash
set -euo pipefail

# Resume one direct-vLLM target-generation chunk. This is meant to run inside a
# Ray allocation, after a previous chunk job has ended. It computes the current
# JSONL row count at runtime, then appends the next non-overlapping prompt slice.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
OUTPUT_CONVERSATIONS="${OUTPUT_CONVERSATIONS:?set OUTPUT_CONVERSATIONS}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_CONVERSATIONS%.jsonl}.resume_summary.json}"
MODEL_LABEL="${MODEL_LABEL:-target}"
CHUNK_INDEX="${CHUNK_INDEX:?set CHUNK_INDEX}"
CHUNK_SIZE="${CHUNK_SIZE:-5000}"
WAVE_LIMIT="${WAVE_LIMIT:-1000}"
BASE_OFFSET="${BASE_OFFSET:-$((CHUNK_INDEX * CHUNK_SIZE))}"

mkdir -p "$(dirname "$OUTPUT_CONVERSATIONS")" "$(dirname "$SUMMARY_JSON")"

existing_count=0
if [[ -s "$OUTPUT_CONVERSATIONS" ]]; then
  existing_count="$(wc -l < "$OUTPUT_CONVERSATIONS" | tr -d ' ')"
fi

remaining=$((CHUNK_SIZE - existing_count))
if (( remaining <= 0 )); then
  "$PYTHON_BIN" - <<PY
import json, time
from pathlib import Path
summary = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "overall_status": "pass",
    "mode": "resume_chunk",
    "model_label": "$MODEL_LABEL",
    "chunk_index": int("$CHUNK_INDEX"),
    "chunk_size": int("$CHUNK_SIZE"),
    "existing_count": int("$existing_count"),
    "remaining": 0,
    "output_conversations": "$OUTPUT_CONVERSATIONS",
    "note": "chunk already complete; no vLLM server started",
}
Path("$SUMMARY_JSON").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
  exit 0
fi

run_limit="$WAVE_LIMIT"
if (( run_limit > remaining )); then
  run_limit="$remaining"
fi
sample_offset=$((BASE_OFFSET + existing_count))

echo "# resume direct vLLM chunk"
echo "MODEL_LABEL=$MODEL_LABEL"
echo "CHUNK_INDEX=$CHUNK_INDEX"
echo "CHUNK_SIZE=$CHUNK_SIZE"
echo "BASE_OFFSET=$BASE_OFFSET"
echo "EXISTING_COUNT=$existing_count"
echo "SAMPLE_OFFSET=$sample_offset"
echo "LIMIT=$run_limit"
echo "OUTPUT_CONVERSATIONS=$OUTPUT_CONVERSATIONS"

LIMIT="$run_limit" \
SAMPLE_OFFSET="$sample_offset" \
APPEND=true \
SKIP_PROMPT_MATERIALIZE=true \
  bash "$EXP_DIR/run_direct_vllm_math_rollout.sh"

