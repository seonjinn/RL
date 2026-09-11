#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly LAUNCHER="${SCRIPT_DIR}/submit_gbs2048.sh"
readonly ARMS=(baseline dspark_k3 dspark_k5 dspark_k7)

case "${1:-}" in
  --list)
    printf '%s\n' "${ARMS[@]}"
    ;;
  --test-only)
    for arm in "${ARMS[@]}"; do
      bash "${LAUNCHER}" --test-only "${arm}"
    done
    ;;
  --submit)
    for arm in "${ARMS[@]}"; do
      bash "${LAUNCHER}" --submit "${arm}"
    done
    ;;
  *)
    echo "usage: $0 --list|--test-only|--submit" >&2
    exit 2
    ;;
esac
