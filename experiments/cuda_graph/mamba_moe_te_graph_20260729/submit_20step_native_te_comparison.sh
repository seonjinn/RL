#!/usr/bin/env bash
set -euo pipefail

: "${CLUSTER:?Set CLUSTER to ptyche or oci-hsg}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}

CLUSTER="${CLUSTER}" \
MODEL=nano-hybrid \
STEPS=20 \
RUN_TAG="${RUN_TAG}" \
bash "${SCRIPT_DIR}/submit_performance.sh" \
  scopes/00_baseline_no_cg.sh \
  scopes/17_attn.sh \
  scopes/05_mamba.sh \
  scopes/03_moe_router.sh \
  variants/router_preprocess_overlap_false_moe_act_false.sh \
  variants/attn_mamba_router_preprocess_overlap_false.sh
