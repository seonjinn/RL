#!/usr/bin/env bash
set -euo pipefail

ROOT=${NEMO_RL_REPO_ROOT:?set NEMO_RL_REPO_ROOT}
RESULT_ROOT=${CANARY_RESULT_ROOT:?set CANARY_RESULT_ROOT}

bash "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_trace.sh"

python3 - "$RESULT_ROOT/trace/shape_summary.json" <<'PY'
import json
from pathlib import Path
import sys

summary_path = Path(sys.argv[1])
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(
        f"Qwen235 trace gate failed: could not read {summary_path}: {error}"
    ) from error

counts = ("record_count", "unique_signature_count")
if summary.get("eligible") is not True or any(
    type(summary.get(name)) is not int or summary[name] <= 0 for name in counts
):
    raise SystemExit(
        "Qwen235 trace gate failed: expected eligible=true, record_count>0, "
        "and unique_signature_count>0"
    )
PY
