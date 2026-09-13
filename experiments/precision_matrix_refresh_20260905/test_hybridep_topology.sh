#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:?Set REPO to the integration checkout}
for model in qwen30 super; do
  for mode in sync async; do
    output=$(CLUSTER=lyris MODEL="$model" MODE="$mode" ARM=bf16-mxfp8 \
      PERFORMANCE_RECIPE=1 PERFORMANCE_HYBRIDEP=1 ACTION=render \
      bash "$REPO/experiments/precision_matrix_refresh_20260905/submit.sh")
    if [[ "$output" != *moe_flex_dispatcher_backend=hybridep* ]]; then
      echo "HybridEP missing: $model/$mode" >&2
      exit 1
    fi
    for forced in NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN NVLINK_DOMAIN_SIZE USE_MNNVL; do
      if [[ "$output" == *"env_vars.$forced="* ]]; then
        echo "Launcher overrides native topology detection: $model/$mode/$forced" >&2
        exit 1
      fi
    done
    printf 'PASS %s/%s: native topology detection\n' "$model" "$mode"
  done
done
