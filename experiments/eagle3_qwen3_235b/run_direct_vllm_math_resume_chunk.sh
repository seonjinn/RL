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

if [[ "${GENERATION_SKIP_FAILED:-false}" == "true" || "${GENERATION_SKIP_FAILED:-false}" == "True" ]]; then
  echo "GENERATION_SKIP_FAILED=true is incompatible with count-based chunk resume." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CONVERSATIONS")" "$(dirname "$SUMMARY_JSON")"

if [[ -s "$OUTPUT_CONVERSATIONS" ]]; then
  "$PYTHON_BIN" - "$OUTPUT_CONVERSATIONS" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb+") as handle:
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    if size == 0:
        raise SystemExit(0)

    handle.seek(size - 1)
    if handle.read(1) != b"\n":
        handle.seek(0)
        data = handle.read()
        last_newline = data.rfind(b"\n")
        tail = data[last_newline + 1 :]
        try:
            json.loads(tail.decode("utf-8"))
        except Exception:
            quarantine = path.with_name(
                f"{path.name}.partial.{time.strftime('%Y%m%d%H%M%S')}.{os.environ.get('SLURM_JOB_ID', 'manual')}"
            )
            quarantine.write_bytes(tail)
            handle.seek(last_newline + 1 if last_newline >= 0 else 0)
            handle.truncate()
            print(f"truncated incomplete JSONL tail to {quarantine}", flush=True)
        else:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            print(f"added missing trailing newline to {path}", flush=True)

    handle.seek(0)
    for line_no, raw_line in enumerate(handle, 1):
        if not raw_line.endswith(b"\n"):
            raise SystemExit(
                f"{path}: line {line_no} is not newline-terminated after tail repair"
            )
        try:
            json.loads(raw_line)
        except Exception as exc:
            raise SystemExit(
                f"{path}: invalid JSONL at line {line_no}; refusing to append: {exc}"
            )
PY
fi

existing_count=0
if [[ -s "$OUTPUT_CONVERSATIONS" ]]; then
  existing_count="$(wc -l < "$OUTPUT_CONVERSATIONS" | tr -d ' ')"
fi

if (( existing_count > 0 )); then
  PROMPT_DATA="${PROMPT_DATA:?set PROMPT_DATA for count-based resume prefix validation}"
  "$PYTHON_BIN" - "$OUTPUT_CONVERSATIONS" "$PROMPT_DATA" "$BASE_OFFSET" "$CHUNK_SIZE" "$existing_count" "${ID_KEY:-}" <<'PY'
import json
import sys
from pathlib import Path
from typing import Any

output_path = Path(sys.argv[1])
prompt_path = Path(sys.argv[2])
base_offset = int(sys.argv[3])
chunk_size = int(sys.argv[4])
existing_count = int(sys.argv[5])
id_key = sys.argv[6] or None


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                yield line_num, json.loads(line)


def record_id(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    for key in ("conversation_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def source_id(record: dict[str, Any], line_num: int) -> str:
    keys = [id_key] if id_key else []
    keys.extend(["conversation_id", "uuid", "id", "task_id", "instance_id"])
    for key in keys:
        if key and record.get(key) not in (None, ""):
            return str(record[key])
    return f"row-{line_num:08d}"


if not prompt_path.exists():
    raise SystemExit(f"PROMPT_DATA is missing: {prompt_path}")

got: list[str] = []
seen: set[str] = set()
for _, record in iter_jsonl(output_path):
    cid = record_id(record)
    if not cid:
        raise SystemExit(f"{output_path} contains a record without conversation_id/id")
    if cid in seen:
        raise SystemExit(
            f"{output_path} contains duplicate id {cid!r}; use missing-id repair, not count-based resume"
        )
    seen.add(cid)
    got.append(cid)

expected: list[str] = []
start_line = base_offset + 1
end_line = base_offset + chunk_size
for line_num, record in iter_jsonl(prompt_path):
    if line_num < start_line:
        continue
    if line_num > end_line:
        break
    expected.append(f"{source_id(record, line_num)}-r00")

if len(expected) != chunk_size:
    raise SystemExit(
        f"{prompt_path} did not contain the expected {chunk_size} rows for "
        f"base_offset={base_offset}; found {len(expected)}"
    )

expected_prefix = set(expected[:existing_count])
got_set = set(got)
missing_prefix = [idx for idx, cid in enumerate(expected[:existing_count]) if cid not in got_set]
extra_existing = sorted(got_set - set(expected))
future_existing = [idx for idx, cid in enumerate(expected[existing_count:], existing_count) if cid in got_set]

if missing_prefix or extra_existing or future_existing:
    raise SystemExit(
        "count-based resume is unsafe for this chunk because existing rows are not "
        "a contiguous leading prefix of PROMPT_DATA. Use "
        "run_direct_vllm_math_missing_id_repair.sh instead. "
        f"missing_prefix_examples={missing_prefix[:10]} "
        f"future_existing_examples={future_existing[:10]} "
        f"extra_existing_examples={extra_existing[:3]}"
    )

print(
    "count-based resume prefix validation passed: "
    f"{existing_count}/{chunk_size} existing ids form a contiguous prefix",
    flush=True,
)
PY
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
