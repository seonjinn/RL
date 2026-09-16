#!/usr/bin/env bash
set -euo pipefail
driver_python=${1:?validated driver interpreter}
arm=${2:?study arm}
result_root=${3:?existing result directory}
mode=${4:---canary}
output_root=${5:-${result_root}}
case "${mode}" in
    --canary|--resume-check|--production|--long-context) renderer=(research.qwen3_8b_rp25_swa.study "${mode}") ;;
    --graph-fap-default|--graph-fap-8|--graph-fap-32|--graph-fap-64) renderer=(research.qwen3_8b_rp25_swa.graph_study --seqs "${mode#--graph-fap-}") ;;
    *) exit 64 ;;
esac
if [[ ! -x "${driver_python}" ]]; then
    printf 'DRIVER_PYTHON_UNUSABLE: %s\n' "${driver_python}" >&2
    exit 69
fi
driver=(uv run --no-project --no-sync --python "${driver_python}" python)
"${driver[@]}" -m "${renderer[@]}" \
    --arm "${arm}" --result-dir "${result_root}" > "${output_root}/overrides.txt"
"${driver[@]}" -m "${renderer[@]}" \
    --arm "${arm}" --result-dir "${result_root}" --recipe > "${output_root}/recipe.txt"
