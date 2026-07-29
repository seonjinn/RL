#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLUSTER=${CLUSTER:-ptyche}

SCOPE_SCRIPTS=(
  00_nocg.sh
  01_attn.sh
  02_moe_router_preprocess.sh
  03_attn_moe_router_preprocess.sh
)

for script_name in "${SCOPE_SCRIPTS[@]}"; do
  printf 'Submitting independent Qwen3-30B-A3B scope: %s\n' "${script_name}"
  CLUSTER="${CLUSTER}" \
  PHASE=performance \
  STEPS=20 \
  PARTITION_OVERRIDE="${PARTITION_OVERRIDE:-batch}" \
  TIME_LIMIT_OVERRIDE="${TIME_LIMIT_OVERRIDE:-04:00:00}" \
  TEST_ONLY="${TEST_ONLY:-0}" \
  bash "${SCRIPT_DIR}/scopes/${script_name}"
done
