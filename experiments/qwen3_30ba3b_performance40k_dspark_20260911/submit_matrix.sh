#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly LAUNCHER="${SCRIPT_DIR}/submit_performance40k.sh"
readonly ARMS=(baseline dspark_k3 dspark_k5)

case "${1:-}" in
  --list)
    printf '%s\n' "${ARMS[@]}"
    ;;
  --test-only)
    export Q30_PERF40K_MAX_STEPS="${Q30_PERF40K_MAX_STEPS:-1}"
    for arm in "${ARMS[@]}"; do
      bash "${LAUNCHER}" --test-only "${arm}"
    done
    ;;
  --submit-gates)
    export Q30_PERF40K_MAX_STEPS=1
    for arm in "${ARMS[@]}"; do
      bash "${LAUNCHER}" --submit "${arm}"
    done
    ;;
  --submit-20)
    export Q30_PERF40K_MAX_STEPS=20
    for arm in "${ARMS[@]}"; do
      bash "${LAUNCHER}" --submit "${arm}"
    done
    ;;
  *)
    echo "usage: $0 --list|--test-only|--submit-gates|--submit-20" >&2
    exit 2
    ;;
esac
