#!/usr/bin/env bash
set -euo pipefail

readonly RUN_DIR=${1:?run directory is required}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
readonly REPO
readonly CONFIG=$REPO/examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n8g-fp8-async-1off.yaml

export PYTHONPATH=$REPO
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export EXPECTED_REPO=$REPO
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_DIR/logs"
uv run --no-sync python -c 'import nemo_rl, os; actual = os.path.realpath(nemo_rl.__file__); expected = os.path.realpath(os.environ["EXPECTED_REPO"]) + os.sep; assert actual.startswith(expected), (actual, expected); print("nemo_rl_source=" + actual)'

uv run --no-sync examples/run_grpo.py \
  --config "$CONFIG" \
  grpo.max_num_steps=10 \
  checkpointing.enabled=false \
  logger.log_dir="$RUN_DIR/logs" \
  logger.wandb_enabled=false \
  2>&1 | tee "$RUN_DIR/run.log"
