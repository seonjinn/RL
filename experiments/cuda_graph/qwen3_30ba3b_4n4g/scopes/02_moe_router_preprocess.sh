#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/../run_scope.sh" \
  moe-router-preprocess \
  '[moe_router,moe_preprocess]' \
  transformer_engine
