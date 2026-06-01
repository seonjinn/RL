#!/bin/bash
# ============================================================================
# NeMo-RL Async GRPO Math Rollout: Qwen3-235B-A22B
#
# Purpose: run a short math GRPO job that emits train_data_step*.jsonl rows for
# Eagle3 hidden-state training. This is a launcher around the official NeMo-RL
# Qwen3-235B math recipe, adjusted for the oci-hsg 4-GPU/node profile when
# NUM_GPU=4.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SPECDEC_RL_DIR="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
if [ -d "${DEFAULT_SPECDEC_RL_DIR}" ]; then
  DEFAULT_REPO_ROOT="${DEFAULT_SPECDEC_RL_DIR}"
else
  DEFAULT_REPO_ROOT="${SCRIPT_DIR}"
fi

REPO_ROOT="${REPO_ROOT:-${MATH_REPO_ROOT:-${DEFAULT_REPO_ROOT}}}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n8g-async-1off.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/results}"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env.sh}"
if [ ! -f "${ENV_FILE}" ] && [ -f "${REPO_ROOT}/env.sh" ]; then
  ENV_FILE="${REPO_ROOT}/env.sh"
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_math_capture_smoke}"

NUM_GPU="${NUM_GPU:-${GPUS_PER_NODE:-8}}"
export GPUS_PER_NODE="${NUM_GPU}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-$((NUM_GPU * 16))}"
NUM_NODES="${NUM_NODES:-32}"
NUM_GEN_NODES="${NUM_GEN_NODES:-16}"

PPS="${PPS:-16}"
GPP="${GPP:-32}"
GBS="${GBS:-512}"
LR="${LR:-5.0e-07}"
SEQLEN="${SEQLEN:-8192}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-1}"

TP="${TP:-4}"
ETP="${ETP:-1}"
EP="${EP:-16}"
CP="${CP:-1}"
if [[ -z "${PP+x}" ]]; then
  if [[ "$NUM_GPU" == "4" ]]; then
    PP=4
  else
    PP=8
  fi
fi
if [[ -z "${PP_FIRST_STAGE+x}" ]]; then
  if [[ "$PP" == "4" ]]; then
    PP_FIRST_STAGE=23
  else
    PP_FIRST_STAGE=11
  fi
fi
if [[ -z "${PP_LAST_STAGE+x}" ]]; then
  if [[ "$PP" == "4" ]]; then
    PP_LAST_STAGE=23
  else
    PP_LAST_STAGE=11
  fi
fi

VLLM_TP="${VLLM_TP:-8}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.8}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-}"
VLLM_COMPILATION_LEVEL="${VLLM_COMPILATION_LEVEL:-}"
VLLM_USE_INDUCTOR="${VLLM_USE_INDUCTOR:-}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
WANDB_PROJ="${WANDB_PROJ:-sna-nemo-rl}"
WANDB_NAME="${WANDB_NAME:-qwen3-235b-math-rollout-capture-smoke}"
EXP_SUFFIX="${EXP_SUFFIX_OVERRIDE:-$WANDB_NAME}"
CHECKPOINT_SUBDIR="${CHECKPOINT_SUBDIR:-$WANDB_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${CHECKPOINT_SUBDIR}"

SAVE_PERIOD="${SAVE_PERIOD:-1000000}"
VAL_PERIOD="${VAL_PERIOD:-1000000}"
KEEP_TOP_K="${KEEP_TOP_K:-1}"
NUM_VAL_SAMPLES_TO_PRINT="${NUM_VAL_SAMPLES_TO_PRINT:-0}"

mkdir -p "${CHECKPOINT_DIR}" "${ROLLOUT_LOG_DIR}"

if [ ! -d "${REPO_ROOT}" ]; then
  echo "ERROR: REPO_ROOT is not visible: ${REPO_ROOT}" >&2
  exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: CONFIG_FILE is not visible: ${CONFIG_FILE}" >&2
  exit 1
fi
if [ ! -f "${REPO_ROOT}/ray.sub" ]; then
  echo "ERROR: ray.sub is not visible under REPO_ROOT: ${REPO_ROOT}/ray.sub" >&2
  exit 1
fi
if [ ! -f "${REPO_ROOT}/examples/run_grpo.py" ]; then
  echo "ERROR: math GRPO entrypoint is not visible under REPO_ROOT: ${REPO_ROOT}/examples/run_grpo.py" >&2
  exit 1
fi
if [ -f "${ENV_FILE}" ]; then
  source "${ENV_FILE}"
else
  echo "WARN: ENV_FILE is not visible; relying on caller-provided tokens/env: ${ENV_FILE}" >&2
fi

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/fsw/portfolios/coreai/users/sna/uv_cache}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/users/sna/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export RAY_DEDUP_LOGS=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=16

PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/coreai/users/sna/.cache/qwen3_235b_math}"
export LUSTRE_VLLM_CACHE="${LUSTRE_VLLM_CACHE:-${PERSISTENT_CACHE}/vllm_compile_cache}"
export LUSTRE_INDUCTOR_CACHE="${LUSTRE_INDUCTOR_CACHE:-${PERSISTENT_CACHE}/inductor_cache}"
export LUSTRE_TRITON_CACHE="${LUSTRE_TRITON_CACHE:-${PERSISTENT_CACHE}/triton_cache}"
export INDUCTOR_CACHE_DIR="${INDUCTOR_CACHE_DIR:-/tmp/nemo_rl_inductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/nemo_rl_triton_cache}"
mkdir -p "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-4:0:0}"
SBATCH_EXCLUDE="${SBATCH_EXCLUDE:-}"
DEFAULT_CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh"
export CONTAINER="${CONTAINER:-$DEFAULT_CONTAINER}"
RUN_UV_SYNC="${RUN_UV_SYNC:-false}"
DRIVER_LAUNCHER="${DRIVER_LAUNCHER:-/opt/venv/bin/python}"
NEMO_RL_PY_EXECUTABLES_SYSTEM="${NEMO_RL_PY_EXECUTABLES_SYSTEM:-0}"
NEMO_RL_VLLM_EXECUTABLE_SYSTEM="${NEMO_RL_VLLM_EXECUTABLE_SYSTEM:-1}"
NEMO_RL_MCORE_EXECUTABLE_SYSTEM="${NEMO_RL_MCORE_EXECUTABLE_SYSTEM:-1}"
INSTALL_VLLM_IN_SYSTEM="${INSTALL_VLLM_IN_SYSTEM:-true}"
SHARED_VLLM_SITE="${SHARED_VLLM_SITE:-${ARTIFACT_ROOT}/python_site/vllm_0_10_2_nodeps_py312}"
SYSTEM_VLLM_BOOTSTRAP="${SYSTEM_VLLM_BOOTSTRAP:-${SCRIPT_DIR}/experiments/eagle3_qwen3_235b/bootstrap_system_vllm_site.sh}"
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-vllm==0.10.2}"
INSTALL_MATH_VERIFY="${INSTALL_MATH_VERIFY:-true}"
MATH_VERIFY_PIP_SPECS="${MATH_VERIFY_PIP_SPECS:-math-verify latex2sympy2-extended sympy mpmath}"
MATH_VERIFY_SITE="${MATH_VERIFY_SITE:-${ARTIFACT_ROOT}/python_site/math_verify_py312}"

DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR="${SCRIPT_DIR}/experiments/eagle3_qwen3_235b/megatron_bridge_qwen3moe"
MEGATRON_BRIDGE_PLUGIN_DIR="${MEGATRON_BRIDGE_PLUGIN_DIR:-}"
if [ -z "${MEGATRON_BRIDGE_PLUGIN_DIR}" ] && [ -d "${DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR}" ]; then
  MEGATRON_BRIDGE_PLUGIN_DIR="${DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR}"
fi
MEGATRON_BRIDGE_QWEN3MOE_PLUGIN="${MEGATRON_BRIDGE_QWEN3MOE_PLUGIN:-1}"
MEGATRON_BRIDGE_SRC="${MEGATRON_BRIDGE_SRC:-}"
MEGATRON_LM_SRC="${MEGATRON_LM_SRC:-}"
if [ -n "${MEGATRON_BRIDGE_PLUGIN_DIR}" ] && [ ! -d "${MEGATRON_BRIDGE_PLUGIN_DIR}" ]; then
  echo "ERROR: MEGATRON_BRIDGE_PLUGIN_DIR is set but not visible: ${MEGATRON_BRIDGE_PLUGIN_DIR}" >&2
  exit 1
fi
if [ ! -f "${CONTAINER}" ] && [[ "${DRY_RUN:-false}" != "true" && "${DRY_RUN:-false}" != "True" ]]; then
  echo "ERROR: CONTAINER is not visible: ${CONTAINER}" >&2
  exit 1
fi
if [ ! -f "${SYSTEM_VLLM_BOOTSTRAP}" ]; then
  echo "ERROR: SYSTEM_VLLM_BOOTSTRAP is not visible: ${SYSTEM_VLLM_BOOTSTRAP}" >&2
  exit 1
fi

MEGATRON_EXTRA_PYTHONPATH=""
for path in "${MEGATRON_BRIDGE_PLUGIN_DIR}" "${MEGATRON_BRIDGE_SRC}" "${MEGATRON_LM_SRC}"; do
  if [ -n "${path}" ]; then
    MEGATRON_EXTRA_PYTHONPATH="${MEGATRON_EXTRA_PYTHONPATH}${path}:"
  fi
done
export INSTALL_MATH_VERIFY MATH_VERIFY_PIP_SPECS MATH_VERIFY_SITE
export PYTHONPATH="${MATH_VERIFY_SITE}:${MEGATRON_EXTRA_PYTHONPATH}${REPO_ROOT}:${PYTHONPATH:-}"

echo "=========================================="
echo "Experiment: ${EXP_SUFFIX}"
echo "Repo root: ${REPO_ROOT}"
echo "Config: ${CONFIG_FILE}"
echo "Container: ${CONTAINER}"
echo "Nodes: ${NUM_NODES}, GPUs/node: ${NUM_GPU}, generation nodes: ${NUM_GEN_NODES}"
echo "Parallelism: TP=${TP}, ETP=${ETP}, EP=${EP}, CP=${CP}, PP=${PP}, vLLM_TP=${VLLM_TP}"
echo "PP stages: first=${PP_FIRST_STAGE}, last=${PP_LAST_STAGE}"
echo "GRPO: PPS=${PPS}, GPP=${GPP}, GBS=${GBS}, max_steps=${MAX_NUM_STEPS}"
echo "SeqLen: ${SEQLEN}"
echo "Model: ${MODEL_PATH}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Rollout log dir: ${ROLLOUT_LOG_DIR}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Shared vLLM site: ${SHARED_VLLM_SITE}"
echo "vLLM pip spec: ${VLLM_PIP_SPEC}"
echo "Math verify site: ${MATH_VERIFY_SITE}"
echo "Math verify pip specs: ${MATH_VERIFY_PIP_SPECS}"
echo "Megatron bridge plugin: ${MEGATRON_BRIDGE_PLUGIN_DIR:-<disabled>}"
echo "Dry run: ${DRY_RUN:-false}"
echo "=========================================="

cd "${REPO_ROOT}"

read -r -d '' SETUP_COMMAND <<SETUPEOF || true
if [[ "${INSTALL_MATH_VERIFY}" == "true" || "${INSTALL_MATH_VERIFY}" == "True" ]]; then
  mkdir -p "${MATH_VERIFY_SITE}"
  if PYTHONPATH="${MATH_VERIFY_SITE}:\${PYTHONPATH:-}" "${DRIVER_LAUNCHER}" -c "import omegaconf; import math_verify; from math_verify import grader" >/dev/null 2>&1; then
    echo "[SETUP] math_verify already available at ${MATH_VERIFY_SITE}"
  else
    echo "[SETUP] Installing math_verify dependencies without antlr4 into ${MATH_VERIFY_SITE}"
    rm -rf "${MATH_VERIFY_SITE}"
    mkdir -p "${MATH_VERIFY_SITE}"
    "${DRIVER_LAUNCHER}" -m pip install --no-cache-dir --target "${MATH_VERIFY_SITE}" --no-deps ${MATH_VERIFY_PIP_SPECS}
  fi
fi
if [[ "${RUN_UV_SYNC}" == "true" || "${RUN_UV_SYNC}" == "True" ]]; then
  UV_HTTP_TIMEOUT=3600 uv sync --frozen
else
  echo "[SETUP] Skipping uv sync; using container preinstalled /opt/venv."
fi
SETUPEOF
export SETUP_COMMAND

export COMMAND="export ARTIFACT_ROOT=${ARTIFACT_ROOT} INSTALL_VLLM_IN_SYSTEM=${INSTALL_VLLM_IN_SYSTEM} VLLM_PIP_SPEC=${VLLM_PIP_SPEC} SHARED_VLLM_SITE=${SHARED_VLLM_SITE}; \
  if [[ \"${INSTALL_MATH_VERIFY}\" == \"true\" || \"${INSTALL_MATH_VERIFY}\" == \"True\" ]]; then \
    mkdir -p ${MATH_VERIFY_SITE} && \
    if PYTHONPATH=${MATH_VERIFY_SITE}:\${PYTHONPATH:-} ${DRIVER_LAUNCHER} -c 'import omegaconf; import math_verify; from math_verify import grader' >/dev/null 2>&1; then \
      echo '[SETUP] math_verify already available'; \
    else \
      echo '[SETUP] Installing math_verify dependencies without antlr4 into ${MATH_VERIFY_SITE}' && \
      rm -rf ${MATH_VERIFY_SITE} && mkdir -p ${MATH_VERIFY_SITE} && \
      ${DRIVER_LAUNCHER} -m pip install --no-cache-dir --target ${MATH_VERIFY_SITE} --no-deps ${MATH_VERIFY_PIP_SPECS}; \
    fi; \
  fi; \
  . ${SYSTEM_VLLM_BOOTSTRAP} && \
  NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  MEGATRON_BRIDGE_QWEN3MOE_PLUGIN=${MEGATRON_BRIDGE_QWEN3MOE_PLUGIN} \
  PYTHONPATH=${MATH_VERIFY_SITE}:${MEGATRON_EXTRA_PYTHONPATH}${REPO_ROOT}:\${PYTHONPATH:-} \
  NEMO_RL_PY_EXECUTABLES_SYSTEM=${NEMO_RL_PY_EXECUTABLES_SYSTEM} \
  NEMO_RL_VLLM_EXECUTABLE_SYSTEM=${NEMO_RL_VLLM_EXECUTABLE_SYSTEM} \
  NEMO_RL_MCORE_EXECUTABLE_SYSTEM=${NEMO_RL_MCORE_EXECUTABLE_SYSTEM} \
  WANDB_API_KEY=\${WANDB_API_KEY:-} \
  HUGGINGFACE_TOKEN=\${HUGGINGFACE_TOKEN:-} \
  HF_TOKEN=\${HF_TOKEN:-\${HUGGINGFACE_TOKEN:-}} \
  HF_HOME=${HF_HOME} \
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
  UV_CACHE_DIR=${UV_CACHE_DIR} \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  VLLM_CACHE_ROOT=${LUSTRE_VLLM_CACHE} \
  DG_JIT_CACHE_DIR=${LUSTRE_VLLM_CACHE}/deep_gemm \
  VLLM_DEEP_GEMM_WARMUP=skip \
  NRL_FORCE_REBUILD_VENVS=true \
  NRL_IGNORE_VERSION_MISMATCH=1 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
  UV_HTTP_TIMEOUT=3600 \
  TORCH_CUDA_ARCH_LIST='9.0 10.0' \
  ${DRIVER_LAUNCHER} ./examples/run_grpo.py \
  --config=${CONFIG_FILE} \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${NUM_GPU} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  grpo.val_period=${VAL_PERIOD} \
  grpo.val_at_start=False \
  grpo.val_at_end=False \
  grpo.max_num_steps=${MAX_NUM_STEPS} \
  policy.model_name=${MODEL_PATH} \
  policy.tokenizer.name=${TOKENIZER_PATH} \
  policy.train_global_batch_size=${GBS} \
  policy.max_total_sequence_length=${SEQLEN} \
  policy.generation.max_new_tokens=${SEQLEN} \
  policy.generation.vllm_cfg.max_model_len=${SEQLEN} \
  policy.megatron_cfg.tensor_model_parallel_size=${TP} \
  policy.megatron_cfg.expert_tensor_parallel_size=${ETP} \
  policy.megatron_cfg.expert_model_parallel_size=${EP} \
  policy.megatron_cfg.context_parallel_size=${CP} \
  policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
  policy.megatron_cfg.num_layers_in_first_pipeline_stage=${PP_FIRST_STAGE} \
  policy.megatron_cfg.num_layers_in_last_pipeline_stage=${PP_LAST_STAGE} \
  policy.megatron_cfg.optimizer.lr=${LR} \
  policy.megatron_cfg.optimizer.min_lr=${LR} \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
  policy.generation.colocated.enabled=False \
  policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${NUM_GPU} \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  checkpointing.save_period=${SAVE_PERIOD} \
  checkpointing.keep_top_k=${KEEP_TOP_K} \
  logger.log_dir=${ROLLOUT_LOG_DIR} \
  logger.wandb_enabled=${WANDB_ENABLED:-False} \
  logger.tensorboard_enabled=False \
  logger.mlflow_enabled=False \
  logger.swanlab_enabled=False \
  logger.num_val_samples_to_print=${NUM_VAL_SAMPLES_TO_PRINT} \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ}"

if [ -n "${VLLM_ENFORCE_EAGER}" ]; then
  export COMMAND="${COMMAND} policy.generation.vllm_cfg.enforce_eager=${VLLM_ENFORCE_EAGER}"
fi
if [ -n "${VLLM_COMPILATION_LEVEL}" ]; then
  export COMMAND="${COMMAND} +policy.generation.vllm_kwargs.compilation_config.level=${VLLM_COMPILATION_LEVEL}"
fi
if [ -n "${VLLM_USE_INDUCTOR}" ]; then
  export COMMAND="${COMMAND} +policy.generation.vllm_kwargs.compilation_config.use_inductor=${VLLM_USE_INDUCTOR}"
fi
if [ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]; then
  export COMMAND="${COMMAND} ${EXTRA_HYDRA_OVERRIDES}"
fi

export MOUNTS="${MOUNTS:-/lustre:/lustre,${REPO_ROOT}:${REPO_ROOT},${SCRIPT_DIR}:${SCRIPT_DIR},${ARTIFACT_ROOT}:${ARTIFACT_ROOT}}"

SBATCH_DEPENDENCY_VALUE="${SBATCH_DEPENDENCY-singleton}"
SBATCH_CMD=(
  sbatch
  --nodes="${NUM_NODES}"
  --account="${SBATCH_ACCOUNT}"
  --job-name="${WANDB_NAME}"
  --partition="${SBATCH_PARTITION}"
  --time="${SBATCH_TIME}"
  --gres=gpu:${NUM_GPU}
  --exclusive
  --mem="${SBATCH_MEM:-0}"
)
if [[ -n "${SBATCH_DEPENDENCY_VALUE}" && "${SBATCH_DEPENDENCY_VALUE}" != "none" && "${SBATCH_DEPENDENCY_VALUE}" != "NONE" ]]; then
  SBATCH_CMD+=(--dependency="${SBATCH_DEPENDENCY_VALUE}")
fi
if [[ -n "$SBATCH_EXCLUDE" ]]; then
  SBATCH_CMD+=(--exclude="$SBATCH_EXCLUDE")
fi
SBATCH_CMD+=(ray.sub)

redact_command_for_log() {
  sed -E \
    -e 's/(WANDB_API_KEY=)[^[:space:]]+/\1<redacted>/g' \
    -e 's/(HUGGINGFACE_TOKEN=)[^[:space:]]+/\1<redacted>/g' \
    -e 's/(HF_TOKEN=)[^[:space:]]+/\1<redacted>/g'
}

if [[ "${DRY_RUN:-false}" == "true" || "${DRY_RUN:-false}" == "True" ]]; then
  echo "[DRY-RUN] COMMAND:"
  printf '%s\n' "$COMMAND" | redact_command_for_log
  echo "[DRY-RUN] sbatch:"
  printf '%q ' "${SBATCH_CMD[@]}"
  printf '\n'
  exit 0
fi

"${SBATCH_CMD[@]}" | tee /dev/stderr | grep -o '[0-9]\+' > latest_235b_math_job_id.txt

JOB_ID="$(cat latest_235b_math_job_id.txt)"
echo "=========================================="
echo "Job submitted: ${EXP_SUFFIX}"
echo "Job ID: ${JOB_ID}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Logs: ${REPO_ROOT}/${JOB_ID}-logs"
echo "=========================================="
