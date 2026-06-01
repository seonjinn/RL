#!/usr/bin/env bash
set -euo pipefail

# Generate and append exact missing ids for one or more direct-vLLM target chunks.
# This is for chunks where older out-of-order concurrent writes made count-based
# resume invalid.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
DATA_DIR="${DATA_DIR:-$ARTIFACT_ROOT/data}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports/mixed_target_chunks}"
MODEL_LABEL="${MODEL_LABEL:?set MODEL_LABEL}"
PROMPT_DATA="${PROMPT_DATA:-$DATA_DIR/mixed_math_nonopenmath_500k_prompts.jsonl}"
CHUNK_DIR="${CHUNK_DIR:-$DATA_DIR/mixed_target_chunks}"
CHUNK_SIZE="${CHUNK_SIZE:-5000}"
REPAIR_CHUNKS="${REPAIR_CHUNKS:?set REPAIR_CHUNKS, e.g. 5,6,7,8}"
REPAIR_TAG="${REPAIR_TAG:-${MODEL_LABEL}_missing_id_repair_${REPAIR_CHUNKS//,/_}_${SLURM_JOB_ID:-manual}}"

SUMMARY_JSON="${SUMMARY_JSON:-$REPORT_DIR/${REPAIR_TAG}_summary.json}"
PREPARE_JSON="${PREPARE_JSON:-$REPORT_DIR/${REPAIR_TAG}_prepare.json}"
APPLY_JSON="${APPLY_JSON:-$REPORT_DIR/${REPAIR_TAG}_apply.json}"
MISSING_PROMPTS="${MISSING_PROMPTS:-$REPORT_DIR/${REPAIR_TAG}_missing_prompts.jsonl}"
GENERATED_REPAIR_OUTPUT="${GENERATED_REPAIR_OUTPUT:-$REPORT_DIR/${REPAIR_TAG}_generated.jsonl}"
ROLLOUT_SUMMARY_JSON="${ROLLOUT_SUMMARY_JSON:-$REPORT_DIR/${REPAIR_TAG}_rollout_summary.json}"

mkdir -p "$REPORT_DIR" "$CHUNK_DIR"

echo "# direct vLLM missing-id repair"
echo "MODEL_LABEL=$MODEL_LABEL"
echo "REPAIR_CHUNKS=$REPAIR_CHUNKS"
echo "PROMPT_DATA=$PROMPT_DATA"
echo "CHUNK_DIR=$CHUNK_DIR"
echo "MISSING_PROMPTS=$MISSING_PROMPTS"
echo "GENERATED_REPAIR_OUTPUT=$GENERATED_REPAIR_OUTPUT"

if [[ "$(basename "$PROMPT_DATA")" == "openmath_direct_vllm_prompts_smoke.jsonl" && "${ALLOW_SMOKE_PROMPT_DATA_FOR_REPAIR:-false}" != "true" ]]; then
  echo "Refusing to run missing-id repair with smoke PROMPT_DATA: $PROMPT_DATA" >&2
  echo "Set PROMPT_DATA to the full mixed prompt file or ALLOW_SMOKE_PROMPT_DATA_FOR_REPAIR=true for an intentional smoke test." >&2
  exit 2
fi
if [[ ! -s "$PROMPT_DATA" ]]; then
  echo "PROMPT_DATA is missing or empty: $PROMPT_DATA" >&2
  exit 2
fi

python3 "$EXP_DIR/repair_direct_vllm_missing_chunk_ids.py" prepare \
  --prompt-data "$PROMPT_DATA" \
  --chunk-dir "$CHUNK_DIR" \
  --model-label "$MODEL_LABEL" \
  --chunk-size "$CHUNK_SIZE" \
  --chunks "$REPAIR_CHUNKS" \
  --missing-prompts "$MISSING_PROMPTS" \
  --json-out "$PREPARE_JSON"

missing_total="$(
  python3 - "$PREPARE_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["missing_total"])
PY
)"

if [[ "$missing_total" == "0" ]]; then
  python3 - "$PREPARE_JSON" "$SUMMARY_JSON" <<'PY'
import json, sys, time
prepare = json.load(open(sys.argv[1]))
summary = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "status": "pass",
    "mode": "missing_id_repair",
    "note": "no missing ids to repair",
    "prepare_json": sys.argv[1],
    "missing_total": 0,
}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  exit 0
fi

PROMPT_DATA="$MISSING_PROMPTS" \
OUTPUT_CONVERSATIONS="$GENERATED_REPAIR_OUTPUT" \
SUMMARY_JSON="$ROLLOUT_SUMMARY_JSON" \
VALIDATION_JSON="${GENERATED_REPAIR_OUTPUT%.jsonl}.validation.json" \
APPEND="${REPAIR_APPEND:-true}" \
LIMIT="" \
SAMPLE_OFFSET=0 \
SKIP_PROMPT_MATERIALIZE=true \
GENERATION_SKIP_FAILED=false \
  bash "$EXP_DIR/run_direct_vllm_math_rollout.sh"

python3 "$EXP_DIR/repair_direct_vllm_missing_chunk_ids.py" apply \
  --prepare-json "$PREPARE_JSON" \
  --generated-output "$GENERATED_REPAIR_OUTPUT" \
  --chunk-dir "$CHUNK_DIR" \
  --model-label "$MODEL_LABEL" \
  --chunk-size "$CHUNK_SIZE" \
  --chunks "$REPAIR_CHUNKS" \
  --json-out "$APPLY_JSON"

python3 - "$PREPARE_JSON" "$ROLLOUT_SUMMARY_JSON" "$APPLY_JSON" "$SUMMARY_JSON" <<'PY'
import json, sys, time
prepare = json.load(open(sys.argv[1]))
rollout = json.load(open(sys.argv[2]))
apply = json.load(open(sys.argv[3]))
summary = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "status": apply["status"],
    "mode": "missing_id_repair",
    "prepare_json": sys.argv[1],
    "rollout_summary_json": sys.argv[2],
    "apply_json": sys.argv[3],
    "missing_total": prepare["missing_total"],
    "generated_records": rollout.get("records"),
    "appended_counts": apply.get("appended_counts"),
    "rows_after": apply.get("rows_after"),
    "short_chunks": apply.get("short_chunks"),
    "reasons": apply.get("reasons"),
}
open(sys.argv[4], "w", encoding="utf-8").write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if summary["status"] == "pass" else 2)
PY
