#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-${SCRIPT_DIR}}"
cd "${NEMORL}"

DEBUG_MODE=0
DEBUG_PORT="${DEBUG_PORT:-5678}"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [options]

Options:
  -d, --debug              Launch the entrypoint under debugpy and wait for an
                           attach from Cursor / VS Code before running.
      --debug-port PORT    Port for debugpy to listen on (default: ${DEBUG_PORT}).
  -h, --help               Show this help message.

Environment variable overrides (DEBUG_PORT, NUM_NODES, GPUS_PER_NODE, ...) still
apply. Pass --debug when running on an interactive compute node so you can
attach Cursor's "Python: Attach to SLURM Node" configuration to this host.

------------------------------------------------------------------------------
Gate 9 dedup-end-to-end-validation recipe
------------------------------------------------------------------------------

This launcher emits driver-side multimodal payload metrics at all sync-rollout
boundaries (driver_to_vllm_generation, driver_to_policy_get_logprobs,
driver_to_policy_train, plus reference / FP8-calibration boundaries when
those features are enabled). The metrics are lines of the form

    ▶ [PAYLOAD] payload_bytes/<boundary>/tensor_mm=<N>, ...

To verify that grpo.deduplicate_multimodal_data actually compacts the
multimodal payload Ray ships between the driver, vLLM workers, and the
Megatron policy worker, run the launcher twice with everything held fixed
except the dedup flag, then diff the [PAYLOAD] lines:

    # Dedup OFF (baseline)
    DEDUPLICATE_MULTIMODAL_DATA=false NUM_GENERATIONS_PER_PROMPT=16 \\
      bash run_interactive_step_3_nanov3_vision_rl.sh

    # Dedup ON
    DEDUPLICATE_MULTIMODAL_DATA=true  NUM_GENERATIONS_PER_PROMPT=16 \\
      bash run_interactive_step_3_nanov3_vision_rl.sh

    # Compare
    rg '\[PAYLOAD\]' results/<dedup-false-job>/run.log
    rg '\[PAYLOAD\]' results/<dedup-true-job>/run.log

Acceptance (per the runbook's Gate 9 dedup verification methodology):
  - logical_rows must MATCH between the two runs at every boundary
  - with dedup ON, unique_prompts < logical_rows and tensor_mm bytes drop by
    roughly 1/NUM_GENERATIONS_PER_PROMPT for the multimodal payload
  - finite metrics (loss, rewards, logprobs, KL, token_mult_prob_error) agree
    between the two runs to within deterministic-generation tolerance
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--debug)
      DEBUG_MODE=1
      shift
      ;;
    --debug-port)
      DEBUG_PORT="$2"
      shift 2
      ;;
    --debug-port=*)
      DEBUG_PORT="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

OMNI_SHARED_ROOT="${OMNI_SHARED_ROOT:-/lustre/fs1/portfolios/coreai/users/aroshanghias}"
USER_ROOT="${USER_ROOT:-/lustre/fs1/portfolios/coreai/users/aroshanghias}"
DATASET_ROOT="${DATASET_ROOT:-${OMNI_SHARED_ROOT}/data/mmpr_miniscule/processed}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OMNI_SHARED_ROOT}/checkpoints}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/nanov3_vision_rl_truncated.yaml}"

SEED="${SEED:-42}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${CP_SIZE:-1}}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-1}"
# Default = 16 rollouts/prompt so the Gate 9 dedup pair shows a clean 16x
# tensor_mm savings between dedup-on and dedup-off. Bump up/down as needed.
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-16}"
# GRPO requires train_global_batch_size == num_prompts_per_step * num_generations_per_prompt.
# Auto-compute when the caller doesn't override it so changing rollout shape doesn't
# silently trip the consistency assertion.
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
OVERLONG_FILTERING="${OVERLONG_FILTERING:-false}"
# Auto-enable sequence packing when CP>1 because nemo-rl asserts the
# conjunction at nemo_rl/models/megatron/setup.py:
#     assert config["sequence_packing"]["enabled"], (
#         "Sequence Packing must be enabled to use Context Parallelism with MCore"
#     )
# Mirror that here so callers don't waste a job to discover the constraint.
# The caller can still force packing off explicitly by passing
# SEQUENCE_PACKING_ENABLED=false; that will fail at startup when CP>1, which
# is the intended behaviour.
if [[ -z "${SEQUENCE_PACKING_ENABLED+x}" ]] && [[ "${CONTEXT_PARALLEL_SIZE}" -gt 1 ]]; then
  SEQUENCE_PACKING_ENABLED=true
fi
SEQUENCE_PACKING_ENABLED="${SEQUENCE_PACKING_ENABLED:-false}"
BAD_WORDS="${BAD_WORDS:-[]}"
DEDUPLICATE_MULTIMODAL_DATA="${DEDUPLICATE_MULTIMODAL_DATA:-false}"
# Driver-side multimodal payload metrics are gated by grpo.debug_payload_metrics.
# The truncated YAML already sets it to true; expose it here as well so the
# launcher is self-documenting and so it can be flipped without editing the YAML.
DEBUG_PAYLOAD_METRICS="${DEBUG_PAYLOAD_METRICS:-true}"
MODEL_NAME="${MODEL_NAME:-${CHECKPOINT_ROOT}/mpo-nanov3omni-mmpr-nanov2-filtered-conv3d-truncated}"
WANDB_PROJECT="${WANDB_PROJECT:-grpo-nanov3vl}"

# Job name embeds the dedup state and the rollouts-per-prompt so back-to-back
# runs of the Gate 9 dedup pair land in distinguishable result directories.
JOB_NAME_BASE="${JOB_NAME_BASE:-interactive-vllm018-smoke-gens${NUM_GENERATIONS_PER_PROMPT}-dedup${DEDUPLICATE_MULTIMODAL_DATA}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%3N)}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_BASE}-${RUN_ID}}"
RESULTS_ROOT="${RESULTS_ROOT:-${NEMORL}/results}"
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/${JOB_NAME}}"

export CACHE_ROOT="${CACHE_ROOT:-${USER_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${USER:-u}}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-true}"

export PYTHONPATH="${HF_MODULES_CACHE}:${NEMORL}/3rdparty/vllm:${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${HF_HOME}" "${HF_MODULES_CACHE}" "${NRL_MEGATRON_CHECKPOINT_DIR}" \
  "${TRITON_CACHE_DIR}" "${TMPDIR}" "${RESULTS_DIR}"

echo "Running interactive Nano v3 VL smoke"
echo "  repo=${NEMORL}"
echo "  model_name=${MODEL_NAME}"
echo "  dataset_root=${DATASET_ROOT}"
echo "  results_dir=${RESULTS_DIR}"
echo "  seed=${SEED}"
echo "  num_prompts_per_step=${NUM_PROMPTS_PER_STEP}"
echo "  num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT}"
echo "  context_parallel_size=${CONTEXT_PARALLEL_SIZE}"
echo "  sequence_packing_enabled=${SEQUENCE_PACKING_ENABLED}"
echo "  deduplicate_multimodal_data=${DEDUPLICATE_MULTIMODAL_DATA}"
echo "  debug_payload_metrics=${DEBUG_PAYLOAD_METRICS}"
if [[ "${NUM_GENERATIONS_PER_PROMPT}" -lt 2 && "${DEDUPLICATE_MULTIMODAL_DATA}" == "true" ]]; then
  echo "  [WARN] dedup is on but NUM_GENERATIONS_PER_PROMPT=${NUM_GENERATIONS_PER_PROMPT}; with no repeated prompts dedup will report"
  echo "         logical_to_unique=1.0 and tensor_mm savings of zero. Set NUM_GENERATIONS_PER_PROMPT>=2 to actually exercise the path." >&2
fi
if [[ "${CONTEXT_PARALLEL_SIZE}" -gt 1 && "${SEQUENCE_PACKING_ENABLED}" != "true" ]]; then
  echo "  [WARN] CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE} requires SEQUENCE_PACKING_ENABLED=true; nemo-rl will assert" >&2
  echo "         at startup. Either drop CP back to 1 or unset SEQUENCE_PACKING_ENABLED so the launcher auto-enables packing." >&2
fi

if [[ "${DEBUG_MODE}" -eq 1 ]]; then
  ATTACH_HOST="$(hostname -s 2>/dev/null || hostname)"
  echo "============================================================"
  echo "DEBUG MODE: launching driver under debugpy"
  echo "  listen      : 0.0.0.0:${DEBUG_PORT}"
  echo "  attach host : ${ATTACH_HOST}"
  echo ""
  echo "In Cursor on the login node:"
  echo "  1. Open 'Run and Debug'"
  echo "  2. Pick 'Python: Attach to SLURM Node'"
  echo "  3. Enter host: ${ATTACH_HOST}"
  echo "  (Driver-only attach; for Ray worker code, inject debugpy in-process.)"
  echo "============================================================"
  RUN_CMD=(uv run --no-sync --with debugpy python -m debugpy
           --listen "0.0.0.0:${DEBUG_PORT}" --wait-for-client
           examples/run_vlm_grpo.py)
else
  RUN_CMD=(uv run --no-sync examples/run_vlm_grpo.py)
fi

"${RUN_CMD[@]}" \
  --config "${CONFIG_PATH}" \
  cluster.num_nodes="${NUM_NODES}" \
  cluster.gpus_per_node="${GPUS_PER_NODE}" \
  policy.megatron_cfg.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
  policy.megatron_cfg.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
  policy.megatron_cfg.expert_model_parallel_size="${EXPERT_MODEL_PARALLEL_SIZE}" \
  policy.generation.vllm_cfg.tensor_parallel_size="${VLLM_TENSOR_PARALLEL_SIZE}" \
  policy.generation.vllm_cfg.async_engine=false \
  policy.model_name="${MODEL_NAME}" \
  policy.train_global_batch_size="${TRAIN_GLOBAL_BATCH_SIZE}" \
  policy.sequence_packing.enabled="${SEQUENCE_PACKING_ENABLED}" \
  grpo.num_prompts_per_step="${NUM_PROMPTS_PER_STEP}" \
  grpo.num_generations_per_prompt="${NUM_GENERATIONS_PER_PROMPT}" \
  grpo.seed="${SEED}" \
  grpo.max_num_steps="${MAX_NUM_STEPS:-1}" \
  grpo.overlong_filtering="${OVERLONG_FILTERING}" \
  grpo.deduplicate_multimodal_data="${DEDUPLICATE_MULTIMODAL_DATA}" \
  grpo.debug_payload_metrics="${DEBUG_PAYLOAD_METRICS}" \
  data.train.cache_dir="${DATASET_ROOT}" \
  policy.generation.max_new_tokens="${MAX_NEW_TOKENS}" \
  policy.generation.temperature="${TEMPERATURE}" \
  policy.generation.top_p="${TOP_P}" \
  policy.generation.bad_words="${BAD_WORDS}" \
  +policy.generation.debug_payload_metrics="${DEBUG_PAYLOAD_METRICS}" \
  policy.megatron_cfg.scheduler.lr_warmup_iters="${LR_WARMUP_ITERS:-0}" \
  checkpointing.checkpoint_dir="${RESULTS_DIR}" \
  logger.log_dir="${RESULTS_DIR}" \
  logger.wandb_enabled=false \
  logger.wandb.project="${WANDB_PROJECT}" \
  logger.wandb.name="${JOB_NAME}" \
  2>&1 | tee "${RESULTS_DIR}/run.log"
