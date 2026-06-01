#!/usr/bin/env bash
set -euo pipefail

# Finalize the Qwen3-235B mixed non-OpenMath target-response corpus after all
# restartable target-generation chunks have reached 5K rows each.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
CHUNK_DIR="${CHUNK_DIR:-$ARTIFACT_ROOT/data/mixed_target_chunks}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
PREFIX="${PREFIX:-qwen3_235b}"
CHUNKS="${CHUNKS:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-5000}"
EXPECTED_COUNT="${EXPECTED_COUNT:-500000}"

MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
DENYLIST="${DENYLIST:-$ARTIFACT_ROOT/data/openmath_reasoning_cot_conversations_50k.jsonl}"
REPLACEMENT_CONVERSATIONS="${REPLACEMENT_CONVERSATIONS:-$ARTIFACT_ROOT/data/mixed_math_nonopenmath_qwen3_235b_replacement_conversations_dapo100.jsonl}"
FINAL_CONVERSATIONS="${FINAL_CONVERSATIONS:-$ARTIFACT_ROOT/data/mixed_math_nonopenmath_qwen3_235b_conversations_500k_unique.jsonl}"
SPECULATORS_JSONL="${SPECULATORS_JSONL:-$ARTIFACT_ROOT/data/mixed_math_nonopenmath_qwen3_235b_conversations_500k_unique_speculators.jsonl}"
SPECULATORS_OUTPUT_DIR="${SPECULATORS_OUTPUT_DIR:-$ARTIFACT_ROOT/speculators/eagle3_qwen3_235b_mixed_math_nonopenmath_500k_parallel}"

READINESS_JSON="${READINESS_JSON:-$REPORT_DIR/mixed_math_nonopenmath_qwen3_235b_500k_chunk_readiness.json}"
MERGE_SUMMARY_JSON="${MERGE_SUMMARY_JSON:-$REPORT_DIR/mixed_math_nonopenmath_qwen3_235b_500k_unique_merge_summary.json}"
CONVERSION_JSON="${CONVERSION_JSON:-$REPORT_DIR/mixed_math_nonopenmath_qwen3_235b_speculators_conversion.json}"
CONVERSION_MD="${CONVERSION_MD:-$REPORT_DIR/mixed_math_nonopenmath_qwen3_235b_speculators_conversion.md}"
FINALIZE_JSON="${FINALIZE_JSON:-$REPORT_DIR/mixed_math_nonopenmath_qwen3_235b_500k_finalize_summary.json}"

mkdir -p "$REPORT_DIR" "$(dirname "$FINAL_CONVERSATIONS")" "$(dirname "$SPECULATORS_JSONL")" "$SPECULATORS_OUTPUT_DIR"

export CHUNK_DIR PREFIX CHUNKS CHUNK_SIZE EXPECTED_COUNT READINESS_JSON FINALIZE_JSON
python3 - <<'PY'
import json
import os
import time
from pathlib import Path

chunk_dir = Path(os.environ["CHUNK_DIR"])
prefix = os.environ["PREFIX"]
chunks = int(os.environ["CHUNKS"])
chunk_size = int(os.environ["CHUNK_SIZE"])
expected = int(os.environ["EXPECTED_COUNT"])
readiness_json = Path(os.environ["READINESS_JSON"])
finalize_json = Path(os.environ["FINALIZE_JSON"])

counts = []
missing = []
for idx in range(chunks):
    path = chunk_dir / f"{prefix}_{idx:03d}.jsonl"
    if not path.exists():
        counts.append(0)
        missing.append(idx)
        continue
    with path.open("rb") as fh:
        counts.append(sum(1 for _ in fh))

short = [{"chunk": idx, "rows": rows, "missing_rows": chunk_size - rows} for idx, rows in enumerate(counts) if rows < chunk_size]
overfull = [{"chunk": idx, "rows": rows} for idx, rows in enumerate(counts) if rows > chunk_size]
summary = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "status": "ready" if sum(counts) == expected and not missing and not short and not overfull else "incomplete",
    "chunk_dir": str(chunk_dir),
    "prefix": prefix,
    "chunks": chunks,
    "chunk_size": chunk_size,
    "expected_count": expected,
    "total_rows": sum(counts),
    "complete_chunks": sum(1 for rows in counts if rows == chunk_size),
    "nonzero_chunks": sum(1 for rows in counts if rows > 0),
    "missing_chunks": missing,
    "short_chunks": short[:100],
    "short_chunk_count": len(short),
    "overfull_chunks": overfull[:100],
    "overfull_chunk_count": len(overfull),
    "min_nonzero_rows": min([rows for rows in counts if rows] or [0]),
    "max_rows": max(counts or [0]),
}
readiness_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if summary["status"] != "ready":
    finalize_json.write_text(json.dumps({
        "generated_at": summary["generated_at"],
        "status": "incomplete",
        "reason": "target chunk files are not all exactly CHUNK_SIZE rows",
        "readiness_json": str(readiness_json),
        "total_rows": summary["total_rows"],
        "complete_chunks": summary["complete_chunks"],
        "short_chunk_count": summary["short_chunk_count"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(2)
PY

inputs=()
for idx in $(seq 0 $((CHUNKS - 1))); do
  inputs+=("$CHUNK_DIR/${PREFIX}_$(printf '%03d' "$idx").jsonl")
done

tmp_final="${FINAL_CONVERSATIONS}.tmp.${SLURM_JOB_ID:-manual}"
rm -f "$tmp_final"
replacement_args=()
if [[ -s "$REPLACEMENT_CONVERSATIONS" ]]; then
  replacement_args=(--replacement "$REPLACEMENT_CONVERSATIONS")
else
  echo "WARNING: replacement conversations file is missing or empty: $REPLACEMENT_CONVERSATIONS" >&2
  echo "WARNING: final merge will fail if primary chunks contain duplicate or denylisted prompts." >&2
fi
python3 "$EXP_DIR/build_unique_training_conversations.py" \
  --output "$tmp_final" \
  --summary-json "$MERGE_SUMMARY_JSON" \
  --expected-count "$EXPECTED_COUNT" \
  --denylist-prompts-from "$DENYLIST" \
  --primary "${inputs[@]}" \
  "${replacement_args[@]}"
mv -f "$tmp_final" "$FINAL_CONVERSATIONS"

rows="$(wc -l < "$FINAL_CONVERSATIONS" | tr -d ' ')"
test "$rows" = "$EXPECTED_COUNT"

python3 "$EXP_DIR/convert_conversations_to_speculators_jsonl.py" \
  --input "$FINAL_CONVERSATIONS" \
  --output "$SPECULATORS_JSONL" \
  --model "$MODEL" \
  --seq-length "$SEQ_LENGTH" \
  --prepared-output-dir "$SPECULATORS_OUTPUT_DIR" \
  --minimum-valid-tokens 1 \
  --json-out "$CONVERSION_JSON" \
  --markdown-out "$CONVERSION_MD"

spec_rows="$(wc -l < "$SPECULATORS_JSONL" | tr -d ' ')"
test "$spec_rows" = "$EXPECTED_COUNT"

export FINAL_CONVERSATIONS SPECULATORS_JSONL MERGE_SUMMARY_JSON CONVERSION_JSON READINESS_JSON FINALIZE_JSON rows spec_rows
python3 - <<'PY'
import json
import os
import time
from pathlib import Path

summary = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "status": "pass",
    "final_conversations": os.environ["FINAL_CONVERSATIONS"],
    "final_conversation_rows": int(os.environ["rows"]),
    "speculators_jsonl": os.environ["SPECULATORS_JSONL"],
    "speculators_rows": int(os.environ["spec_rows"]),
    "chunk_readiness_json": os.environ["READINESS_JSON"],
    "merge_summary_json": os.environ["MERGE_SUMMARY_JSON"],
    "conversion_json": os.environ["CONVERSION_JSON"],
}
Path(os.environ["FINALIZE_JSON"]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
