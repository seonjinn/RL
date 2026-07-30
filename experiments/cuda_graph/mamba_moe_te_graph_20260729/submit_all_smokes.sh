#!/usr/bin/env bash
set -euo pipefail

: "${CLUSTER:?Set CLUSTER to ptyche or oci-hsg}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

for launcher in "${SCRIPT_DIR}"/scopes/*.sh "${SCRIPT_DIR}"/variants/*.sh; do
  printf 'Submitting smoke launcher: %s\n' "${launcher#${SCRIPT_DIR}/}"
  CLUSTER="${CLUSTER}" \
  MODEL="${MODEL:-nano-hybrid}" \
  PHASE=smoke \
  STEPS="${STEPS:-5}" \
  TEST_ONLY="${TEST_ONLY:-0}" \
  RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}" \
  bash "${launcher}"
done
