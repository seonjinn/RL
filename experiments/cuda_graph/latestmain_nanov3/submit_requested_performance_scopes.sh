#!/usr/bin/env bash
set -euo pipefail

: "${CLUSTER:?Set CLUSTER to ptyche or oci-hsg.}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

SCOPE_SCRIPTS=(
  01_attn.sh
  02_mamba.sh
  08_moe_router.sh
  12_moe_router_preprocess.sh
  15_attn_mamba_moe_router_preprocess.sh
)

for script_name in "${SCOPE_SCRIPTS[@]}"; do
  printf 'Submitting independent performance scope: %s\n' "${script_name}"
  CLUSTER="${CLUSTER}" \
  PHASE=performance \
  STEPS=20 \
  PARTITION_OVERRIDE=backfill \
  TIME_LIMIT_OVERRIDE=01:00:00 \
  TEST_ONLY="${TEST_ONLY:-0}" \
  bash "${SCRIPT_DIR}/scopes/${script_name}"
done
