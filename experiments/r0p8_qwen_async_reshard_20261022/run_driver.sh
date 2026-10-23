#!/usr/bin/env bash
set -euo pipefail

readonly MODEL=${1:?model is required}
readonly RUN_DIR=${2:?run directory is required}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
readonly REPO

case "$MODEL" in
  32b)
    readonly CONFIG=$REPO/examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n8g-async-1off.yaml
    REFIT_BUFFER_OVERRIDES=()
    ;;
  235b)
    readonly CONFIG=$REPO/examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n8g-async-1off.yaml
    REFIT_BUFFER_OVERRIDES=(
      '++policy.megatron_cfg.env_vars.NRL_REFIT_BUFFER_MEMORY_RATIO="0.005"'
      '++policy.generation.vllm_cfg.env_vars.NRL_REFIT_BUFFER_MEMORY_RATIO="0.005"'
    )
    ;;
  *)
    printf 'unknown model: %s\n' "$MODEL" >&2
    exit 2
    ;;
esac

export PYTHONPATH=$REPO:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export EXPECTED_REPO=$REPO
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_DIR/logs"
uv run --no-sync python -c 'import megatron, nemo_rl, os; actual = os.path.realpath(nemo_rl.__file__); expected = os.path.realpath(os.environ["EXPECTED_REPO"]) + os.sep; assert actual.startswith(expected), (actual, expected); print("nemo_rl_source=" + actual); print("megatron_path=" + str(megatron.__path__))'

uv run --no-sync examples/run_grpo.py \
  --config "$CONFIG" \
  ++policy.generation.vllm_kwargs.moe_backend=triton \
  policy.generation.refit_transport=nccl_reshard \
  "${REFIT_BUFFER_OVERRIDES[@]}" \
  grpo.max_num_steps=10 \
  checkpointing.enabled=false \
  logger.log_dir="$RUN_DIR/logs" \
  logger.wandb_enabled=false \
  2>&1 | tee "$RUN_DIR/run.log"
