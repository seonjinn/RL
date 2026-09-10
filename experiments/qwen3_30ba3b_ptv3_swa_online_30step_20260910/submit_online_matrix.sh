#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly LAUNCHER="${SCRIPT_DIR}/submit_math_gate.sh"
arms=(
  dflash_k5_fixed5
  dflash_k5_fixed10
  dspark_k7_fixed5
  dspark_k7_fixed10
)

usage() {
  echo "usage: $0 --list|--test-only|--submit" >&2
  exit 2
}

case "${1:-}" in
  --list)
    printf '%s\n' "${arms[@]}"
    ;;
  --test-only)
    for arm in "${arms[@]}"; do
      bash "${LAUNCHER}" --test-only "${arm}"
    done
    ;;
  --submit)
    for arm in "${arms[@]}"; do
      bash "${LAUNCHER}" --submit "${arm}"
    done
    ;;
  *) usage ;;
esac
