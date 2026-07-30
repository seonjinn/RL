#!/usr/bin/env bash
set -euo pipefail

: "${CLUSTER:?Set CLUSTER to ptyche or oci-hsg}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if (($#)); then
  launchers=("$@")
elif [[ -n "${PERFORMANCE_SCRIPTS:-}" ]]; then
  read -r -a launchers <<<"${PERFORMANCE_SCRIPTS}"
else
  echo "Pass launcher paths or set PERFORMANCE_SCRIPTS" >&2
  exit 2
fi

for relative_launcher in "${launchers[@]}"; do
  launcher="${SCRIPT_DIR}/${relative_launcher}"
  [[ -f "${launcher}" ]] || {
    echo "Missing performance launcher: ${relative_launcher}" >&2
    exit 2
  }
  case "${relative_launcher}" in
    scopes/*.sh|variants/*.sh) ;;
    *)
      echo "Performance launcher must be under scopes/ or variants/: ${relative_launcher}" >&2
      exit 2
      ;;
  esac
  printf 'Submitting performance launcher: %s\n' "${relative_launcher}"
  CLUSTER="${CLUSTER}" \
  MODEL="${MODEL:-nano-hybrid}" \
  PHASE=performance \
  STEPS="${STEPS:-20}" \
  TEST_ONLY="${TEST_ONLY:-0}" \
  RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}" \
  bash "${launcher}"
done
