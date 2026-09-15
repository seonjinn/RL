#!/usr/bin/env bash
set -euo pipefail
driver_python=${1:?validated driver interpreter}
arm=${2:?study arm}
result_root=${3:?existing result directory}
if [[ ! -x "${driver_python}" ]]; then
    printf 'DRIVER_PYTHON_UNUSABLE: %s\n' "${driver_python}" >&2
    exit 69
fi
driver=(uv run --no-project --no-sync --python "${driver_python}" python)
"${driver[@]}" -m research.qwen3_8b_rp25_swa.study \
    --arm "${arm}" --result-dir "${result_root}" --canary > "${result_root}/overrides.txt"
"${driver[@]}" -m research.qwen3_8b_rp25_swa.study \
    --arm "${arm}" --result-dir "${result_root}" --canary --recipe > "${result_root}/recipe.txt"
