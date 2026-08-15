#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT NTRACE_SOURCE NTRACE_RUNTIME" >&2
  exit 2
fi

RUN_ROOT=$(realpath "$1")
NTRACE_SOURCE=$(realpath "$2")
export NTRACE_RUNTIME
NTRACE_RUNTIME=$(realpath "$3")
DATA_DIR=${RUN_ROOT}/rollout/vllm
ANALYSIS_ROOT=${RUN_ROOT}/analysis
PYTHON_BIN=${PYTHON_BIN:-python}

export PYTHONPATH="${NTRACE_RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ANALYSIS_ROOT}/per_rank"

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import ntrace

runtime = Path(os.environ["NTRACE_RUNTIME"]).resolve()
module = Path(ntrace.__file__).resolve()
assert runtime == module or runtime in module.parents, (runtime, module)
assert hasattr(ntrace.NemoRLRolloutTraceController, "close")
print(f"ntrace={ntrace.__version__} module={module}")
PY

printf "rank\trecords_bytes\tstacks_bytes\tgraph_nodes_bytes\n" \
  > "${ANALYSIS_ROOT}/capture_manifest.tsv"
for rank in {0..7}; do
  records=${DATA_DIR}/ntrace_records_rank${rank}.parquet
  stacks=${DATA_DIR}/ntrace_stacks_rank${rank}.parquet
  graph_nodes=${DATA_DIR}/ntrace_graph_nodes_rank${rank}.parquet
  sources=${DATA_DIR}/ntrace_sources_rank${rank}/manifest.json
  test -s "${records}"
  test -s "${stacks}"
  test -s "${graph_nodes}"
  test -s "${sources}"
  printf "%d\t%d\t%d\t%d\n" \
    "${rank}" \
    "$(stat -c %s "${records}")" \
    "$(stat -c %s "${stacks}")" \
    "$(stat -c %s "${graph_nodes}")" \
    >> "${ANALYSIS_ROOT}/capture_manifest.tsv"
done

"${PYTHON_BIN}" "${NTRACE_SOURCE}/scripts/audit_graph_replay_coverage.py" \
  "${DATA_DIR}" \
  --rank 0 --rank 1 --rank 2 --rank 3 \
  --rank 4 --rank 5 --rank 6 --rank 7 \
  --output-json "${ANALYSIS_ROOT}/graph_replay_coverage.json" \
  2>&1 | tee "${ANALYSIS_ROOT}/graph_replay_coverage.log"

for rank in {0..7}; do
  rank_dir=${ANALYSIS_ROOT}/per_rank/rank${rank}
  mkdir -p "${rank_dir}"
  "${PYTHON_BIN}" -m ntrace breakdown "${DATA_DIR}" \
    --input-format ntrace \
    --rank "${rank}" \
    --output-dir "${rank_dir}" \
    2>&1 | tee "${rank_dir}/breakdown.log"
done

"${PYTHON_BIN}" -m ntrace multirank "${DATA_DIR}" \
  --input-format ntrace \
  --ranks 0,1,2,3,4,5,6,7 \
  --clock-align utc \
  --output-dir "${ANALYSIS_ROOT}/native_multirank" \
  --output-report-json \
    "${ANALYSIS_ROOT}/native_multirank/ntrace_multirank_report.json" \
  2>&1 | tee "${ANALYSIS_ROOT}/native_multirank.log"

test -s "${ANALYSIS_ROOT}/native_multirank/ntrace_multirank.html"
test -s "${ANALYSIS_ROOT}/native_multirank/ntrace_multirank.json.gz"

echo "analysis complete: ${ANALYSIS_ROOT}"
