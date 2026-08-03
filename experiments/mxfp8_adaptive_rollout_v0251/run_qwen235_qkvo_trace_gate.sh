#!/usr/bin/env bash
set -euo pipefail

ROOT=${NEMO_RL_REPO_ROOT:?set NEMO_RL_REPO_ROOT}
RESULT_ROOT=${CANARY_RESULT_ROOT:?set CANARY_RESULT_ROOT}

bash "$ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_trace.sh"

python3 - "$RESULT_ROOT/trace/shape_summary.json" "$SHAPE_TRACE_DIR" \
  "$RESULT_ROOT/trace/qkvo_coverage.json" "${SHAPE_TRACE_MAX:?set SHAPE_TRACE_MAX}" \
  "${CANARY_EXPECTED_TRACE_WORKERS:-8}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

summary_path = Path(sys.argv[1])
trace_dir = Path(sys.argv[2])
coverage_path = Path(sys.argv[3])
trace_cap = int(sys.argv[4])
expected_workers = int(sys.argv[5])

try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"Qwen235 QKVO trace gate failed: {error}") from error

if not (
    summary.get("eligible") is True
    and int(summary.get("record_count", 0)) > 0
    and int(summary.get("unique_signature_count", 0)) > 0
):
    raise SystemExit("Qwen235 QKVO trace gate failed: no eligible MXFP8 shapes")

prefixes: set[str] = set()
hostnames: set[str] = set()
workers: set[tuple[str, int]] = set()
record_count = 0
for path in sorted(trace_dir.glob("*.jsonl")):
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        record_count += 1
        if record.get("event") != "mxfp8_dense_shape":
            raise SystemExit(
                f"Qwen235 QKVO trace gate failed: invalid event in {path}:{line_number}"
            )
        prefix = record.get("prefix")
        if not isinstance(prefix, str) or not prefix:
            raise SystemExit(
                f"Qwen235 QKVO trace gate failed: invalid prefix in {path}:{line_number}"
            )
        prefixes.add(prefix)
        hostname = record.get("hostname")
        pid = record.get("pid")
        if not isinstance(hostname, str) or not hostname or not isinstance(pid, int):
            raise SystemExit(
                f"Qwen235 QKVO trace gate failed: invalid worker in {path}:{line_number}"
            )
        hostnames.add(hostname)
        workers.add((hostname, pid))
        k = int(record["k"])
        n_logical = int(record["n_logical"])
        n_physical = int(record["n_physical"])
        if k % 256 != 0 or n_physical < n_logical or n_physical % 128 != 0:
            raise SystemExit(
                "Qwen235 QKVO trace gate failed: invalid MXFP8 physical signature "
                f"in {path}:{line_number}"
            )

if record_count >= trace_cap:
    raise SystemExit(
        f"Qwen235 QKVO trace gate failed: trace cap reached ({record_count}/{trace_cap})"
    )
if record_count != int(summary["record_count"]):
    raise SystemExit(
        "Qwen235 QKVO trace gate failed: raw/summary record mismatch: "
        f"raw={record_count}, summary={summary['record_count']}"
    )
if len(hostnames) < 2 or len(workers) < expected_workers:
    raise SystemExit(
        "Qwen235 QKVO trace gate failed: incomplete distributed provenance: "
        f"hosts={len(hostnames)}, workers={len(workers)}, expected_workers={expected_workers}"
    )

qkv_prefixes = sorted(prefix for prefix in prefixes if ".qkv_proj" in prefix)
o_prefixes = sorted(prefix for prefix in prefixes if ".o_proj" in prefix)
missing = [
    name
    for name, values in (("qkv_proj", qkv_prefixes), ("o_proj", o_prefixes))
    if not values
]
if missing:
    raise SystemExit(
        "Qwen235 QKVO trace gate failed: missing MXFP8 trace families: "
        + ", ".join(missing)
    )
qkv_layers = {prefix.rsplit(".", 1)[0] for prefix in qkv_prefixes}
o_layers = {prefix.rsplit(".", 1)[0] for prefix in o_prefixes}
if qkv_layers != o_layers:
    raise SystemExit(
        "Qwen235 QKVO trace gate failed: fused QKV/O layer coverage mismatch"
    )

coverage = {
    "hostname_count": len(hostnames),
    "attention_layer_count": len(qkv_layers),
    "qkv_prefix_count": len(qkv_prefixes),
    "o_prefix_count": len(o_prefixes),
    "qkv_prefixes": qkv_prefixes,
    "o_prefixes": o_prefixes,
    "record_count": record_count,
    "worker_count": len(workers),
}
coverage_path.write_text(
    json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(coverage, indent=2, sort_keys=True))
PY

trace_files=("$SHAPE_TRACE_DIR"/*.jsonl)
if [[ ! -e "${trace_files[0]}" ]]; then
  echo "Qwen235 QKVO trace gate failed: no raw trace files" >&2
  exit 2
fi
PYTHONPATH="$ROOT" python3 \
  -m experiments.mxfp8_adaptive_rollout_v0251.build_shape_manifest \
  "${trace_files[@]}" \
  --output "$RESULT_ROOT/trace/qkvo_manifest.json" \
  --shmoo-dir "$RESULT_ROOT/trace/shmoo" \
  --family QKV \
  --family O
