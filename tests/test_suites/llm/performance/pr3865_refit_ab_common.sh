#!/bin/bash

set -euo pipefail

: "${CONFIG_REL:?CONFIG_REL must be set by the model launcher}"
: "${MAX_STEPS:?MAX_STEPS must be set by the model launcher}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../../..")
REFIT_ARM=${REFIT_ARM:-reshard}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
BASE_NAME=$(basename "${BASH_SOURCE[1]}" .sh)

case "${REFIT_ARM}" in
  legacy)
    REFIT_TRANSPORT=null
    ;;
  reshard)
    REFIT_TRANSPORT=nccl_reshard
    ;;
  *)
    echo "REFIT_ARM must be legacy or reshard" >&2
    exit 2
    ;;
esac

CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_REL}"
EXP_NAME="${BASE_NAME}-${REFIT_ARM}-${RUN_GROUP}"
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
LOG_DIR="${EXP_DIR}/logs"
JSON_METRICS="${EXP_DIR}/metrics.json"
RUN_LOG="${EXP_DIR}/run.log"

test -f "${CONFIG_PATH}"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -n "${TEST_DRYRUN:-}" ]]; then
  printf 'model=%s arm=%s transport=%s config=%s steps=%s\n' \
    "${BASE_NAME}" "${REFIT_ARM}" "${REFIT_TRANSPORT}" "${CONFIG_PATH}" "${MAX_STEPS}"
  exit 0
fi

cd "${PROJECT_ROOT}"
uv run examples/run_grpo.py \
  --config "${CONFIG_PATH}" \
  "$@" \
  "${MODEL_OVERRIDES[@]}" \
  "policy.generation.refit_transport=${REFIT_TRANSPORT}" \
  "grpo.max_num_steps=${MAX_STEPS}" \
  "grpo.val_at_start=false" \
  "++grpo.val_at_end=false" \
  "checkpointing.enabled=false" \
  "logger.log_dir=${LOG_DIR}" \
  "logger.wandb_enabled=true" \
  "logger.wandb.project=sna-pr3865-nccl-reshard-ab" \
  "logger.wandb.name=${EXP_NAME}" \
  "logger.monitor_gpus=true" \
  "logger.tensorboard_enabled=true" \
  2>&1 | tee "${RUN_LOG}"

uv run tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${JSON_METRICS}"
