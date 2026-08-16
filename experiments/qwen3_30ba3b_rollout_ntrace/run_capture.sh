#!/usr/bin/env bash

set -euo pipefail

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=4
SEGMENT_SIZE=2
STEPS_PER_RUN=4
MAX_STEPS=4
NUM_RUNS=1
NUM_MINUTES=180
# ===== END CONFIG =====

: "${NTRACE_ARM:?set NTRACE_ARM to bf16 or mxfp8}"
: "${NTRACE_SOURCE:?set NTRACE_SOURCE to the pinned ntrace source tree}"
: "${NTRACE_RUNTIME:?set NTRACE_RUNTIME to the shared ntrace install target}"
: "${NTRACE_SOURCE_COMMIT:?set NTRACE_SOURCE_COMMIT to the ntrace git commit}"
: "${NEMO_SOURCE_COMMIT:?set NEMO_SOURCE_COMMIT to the NeMo-RL git commit}"
: "${NTRACE_RESULTS_ROOT:?set NTRACE_RESULTS_ROOT to a shared result directory}"

case "${NTRACE_ARM}" in
  bf16)
    QUANTIZATION_SCOPE=bf16
    ;;
  mxfp8)
    QUANTIZATION_SCOPE=expert_fc1_fc2
    ;;
  *)
    echo "NTRACE_ARM must be bf16 or mxfp8, got ${NTRACE_ARM}" >&2
    exit 2
    ;;
esac

RUN_ID=${SLURM_JOB_ID:-local}-$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT=${NTRACE_RESULTS_ROOT}/${NTRACE_ARM}/${RUN_ID}
mkdir -p "${RUN_ROOT}"

if [[ ! -d "${NTRACE_RUNTIME}/ntrace" ]]; then
  NTRACE_INSTALL_SOURCE="${NTRACE_SOURCE}" \
  NTRACE_INSTALL_TARGET="${NTRACE_RUNTIME}" \
  NTRACE_INSTALL_PYTHON="$(command -v python)" \
    bash "${NTRACE_SOURCE}/scripts/ntrace_nemo_rl_install_target.sh"
fi

export PYTHONPATH="${NTRACE_RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}"
export NRL_ROLLOUT_PROFILER_CLASS=ntrace.NemoRLRolloutTraceController
export NTRACE_ROLLOUT_RANKS=0-7
export NTRACE_ROLLOUT_CAPTURE_ITER=1
export NTRACE_ROLLOUT_NUM_ITERS=3
export NTRACE_ROLLOUT_GRAPH_CAPTURE=early
export NTRACE_INCLUDE_STACK_TRACES=1
export NTRACE_INCLUDE_NVTX_RANGES=1
export NTRACE_SAVE_CPU_NVTX=1
export NTRACE_INCLUDE_MEMOPS=0
export NTRACE_GPU_METRICS_INTERVAL_US=0
export NTRACE_OUTPUT_DIR=${RUN_ROOT}

python - <<'PY'
import pyarrow

import ntrace

assert hasattr(ntrace.NemoRLRolloutTraceController, "close")
print(
    f"ntrace={ntrace.__version__} pyarrow={pyarrow.__version__} "
    "rollout_close_hook=present",
    flush=True,
)
PY

cat > "${RUN_ROOT}/provenance.txt" <<EOF
arm=${NTRACE_ARM}
nemo_rl_commit=${NEMO_SOURCE_COMMIT}
ntrace_commit=${NTRACE_SOURCE_COMMIT}
container=${CONTAINER:-unknown}
slurm_job_id=${SLURM_JOB_ID:-unknown}
slurm_nodes=${SLURM_JOB_NODELIST:-unknown}
capture_iter=${NTRACE_ROLLOUT_CAPTURE_ITER}
num_iters=${NTRACE_ROLLOUT_NUM_ITERS}
cuda_graph=enabled
moe_backend=flashinfer_trtllm
refit_transport=nccl_reshard
quantization_scope=${QUANTIZATION_SCOPE}
EOF

CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout.yaml

common_args=(
  --config "${CONFIG}"
  grpo.max_num_steps=4
  grpo.async_grpo.enabled=false
  grpo.async_grpo.in_flight_weight_updates=false
  policy.megatron_cfg.optimizer.lr=0.0
  policy.megatron_cfg.optimizer.min_lr=0.0
  policy.megatron_cfg.scheduler.lr_warmup_init=0.0
  checkpointing.enabled=false
  logger.log_dir="${RUN_ROOT}/logs"
  logger.wandb_enabled=false
  logger.tensorboard_enabled=true
  logger.monitor_gpus=true
)

precision_args=()
if [[ "${NTRACE_ARM}" == bf16 ]]; then
  precision_args=(
    policy.generation.vllm_cfg.precision=bfloat16
    '~policy.generation.vllm_cfg.is_mx'
    '~policy.generation.vllm_cfg.quantization_ignored_layer_kws'
    ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm
    policy.generation.refit_transport=nccl_reshard
  )
else
  precision_args=(
    '~policy.generation.vllm_cfg.quantization_ignored_layer_kws'
    '++policy.generation.vllm_cfg.quantization_ignore_patterns=[model.layers.*.self_attn.*,model.layers.*.mlp.gate,lm_head]'
    ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm
    policy.generation.refit_transport=nccl_reshard
  )
fi

uv run examples/run_grpo.py \
  "${common_args[@]}" \
  "${precision_args[@]}" \
  "$@" \
  2>&1 | tee "${RUN_ROOT}/run.log"
