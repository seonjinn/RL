#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly LAUNCHER="${SCRIPT_DIR}/submit_smoke.sh"
readonly ARMS=(
  dflash_k5_fap
  dflash_k5_no_graph
  dspark_k5_fap
)

export Q30_LATEST_MAIN_CONTEXT_LENGTH=32768
export Q30_LATEST_MAIN_MAX_STEPS=3
export Q30_LATEST_MAIN_DIAGNOSTIC=true
export Q30_LATEST_MAIN_NSYS=true

run_arm() {
  local action="$1"
  local diagnostic_arm="$2"
  local method_arm=""
  case "${diagnostic_arm}" in
    dflash_k5_fap)
      method_arm=dflash_k5
      export Q30_LATEST_MAIN_GRAPH_MODE=FAP
      ;;
    dflash_k5_no_graph)
      method_arm=dflash_k5
      export Q30_LATEST_MAIN_GRAPH_MODE=NONE
      ;;
    dspark_k5_fap)
      method_arm=dspark_k5
      export Q30_LATEST_MAIN_GRAPH_MODE=FAP
      ;;
    *)
      echo "unknown diagnostic arm: ${diagnostic_arm}" >&2
      exit 2
      ;;
  esac
  bash "${LAUNCHER}" "${action}" "${method_arm}"
}

case "${1:-}" in
  --list)
    printf '%s\n' "${ARMS[@]}"
    ;;
  --test-only|--submit)
    for arm in "${ARMS[@]}"; do
      run_arm "$1" "${arm}"
    done
    ;;
  *)
    echo "usage: $0 --list|--test-only|--submit" >&2
    exit 2
    ;;
esac
