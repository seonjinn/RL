#!/usr/bin/env bash
set -euo pipefail

# Submit a generation-only NeMo-RL vLLM backend acceptance-rate check.
# This uses VllmGeneration directly, so it does not initialize Megatron policy.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REMOTE_ROADMAP_ROOT="${REMOTE_ROADMAP_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap}"
SPECDEC_RL_DIR="${SPECDEC_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
PROMPT_DATA="${PROMPT_DATA:-$ARTIFACT_ROOT/data/openmath_direct_vllm_prompts_bench16_offset2000_eagle3k1_r0.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-$REPORT_DIR/nemo_vllm_acceptance_eagle3_math_pilot1k_k1.json}"
DRAFT_MODEL="${DRAFT_MODEL:-$ARTIFACT_ROOT/vllm_drafts/eagle3_math_pilot1k_offset144_train200}"
SOURCE_VLLM_SITE="${SOURCE_VLLM_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
INSTALL_VLLM_IN_SYSTEM="${INSTALL_VLLM_IN_SYSTEM:-false}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"

CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
NUM_NODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
JOB_NAME="${JOB_NAME:-q235b-nemorl-vllm-eagle3-accept-k1}"
DRY_RUN="${DRY_RUN:-false}"
DEPENDENCY="${DEPENDENCY:-}"
SBATCH_DEPENDENCY_ARG=()
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_DEPENDENCY_ARG=(--dependency="$DEPENDENCY")
fi
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
SBATCH_EXTRA_ARGV=()
if [[ -n "$SBATCH_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  SBATCH_EXTRA_ARGV=($SBATCH_EXTRA_ARGS)
fi

mkdir -p "$REPORT_DIR"
rm -f "$OUTPUT_JSON"

REMOTE_SCRIPT="$REMOTE_ROADMAP_ROOT/experiments/eagle3_qwen3_235b/run_nemo_vllm_generation_acceptance.py"
MOUNTS="${MOUNTS:-/lustre:/lustre,$REMOTE_ROADMAP_ROOT:$REMOTE_ROADMAP_ROOT,$SPECDEC_RL_DIR:$SPECDEC_RL_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
PYTHONPATH_VALUE="${SOURCE_VLLM_SITE:+$SOURCE_VLLM_SITE:}$REMOTE_ROADMAP_ROOT:$SPECDEC_RL_DIR:${PYTHONPATH:-}"
HF_HOME_PATH="${HF_HOME_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
BOOTSTRAP="$REMOTE_ROADMAP_ROOT/experiments/eagle3_qwen3_235b/bootstrap_system_vllm_site.sh"
DRIVER_LAUNCHER="${DRIVER_LAUNCHER:-/opt/venv/bin/python}"
RAY_INCLUDE_DASHBOARD_VALUE="${RAY_INCLUDE_DASHBOARD:-False}"
RAY_PYTHON_VERSION_VALUE="${RAY_PYTHON_VERSION:-3.12.13}"
RAY_PYTHON_SPEC_VALUE="${RAY_PYTHON_SPEC:-/opt/venv/bin/python}"
RAY_VERSION_VALUE="${RAY_VERSION:-2.54.0}"
RAY_USE_EXISTING_ENV_VALUE="${RAY_USE_EXISTING_ENV:-true}"
ENFORCE_EAGER_ARG="--no-enforce-eager"
if [[ "$VLLM_ENFORCE_EAGER" == "true" || "$VLLM_ENFORCE_EAGER" == "True" ]]; then
  ENFORCE_EAGER_ARG="--enforce-eager"
fi
ASYNC_ENGINE_ARG="--async-engine"
if [[ "${ASYNC_ENGINE:-true}" == "false" || "${ASYNC_ENGINE:-true}" == "False" ]]; then
  ASYNC_ENGINE_ARG="--no-async-engine"
fi

COMMAND=$(cat <<EOF
set -euo pipefail
cd "$SPECDEC_RL_DIR"
export ARTIFACT_ROOT="$ARTIFACT_ROOT"
export DRIVER_LAUNCHER="$DRIVER_LAUNCHER"
export INSTALL_VLLM_IN_SYSTEM="$INSTALL_VLLM_IN_SYSTEM"
export VLLM_PIP_SPEC=vllm==0.10.2
export SHARED_VLLM_SITE="$SOURCE_VLLM_SITE"
. "$BOOTSTRAP"
export PYTHONPATH="$PYTHONPATH_VALUE"
export HF_HOME="$HF_HOME_PATH"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME_PATH/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME_PATH/hub}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ARTIFACT_ROOT/vllm_cache_nemorl_acceptance}"
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_DISABLE_USAGE_STATS=1
export VLLM_COMPILATION_LEVEL="${VLLM_COMPILATION_LEVEL:-}"
export VLLM_CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE:-}"
export VLLM_CUDAGRAPH_CAPTURE_SIZES="${VLLM_CUDAGRAPH_CAPTURE_SIZES:-}"
export NRL_VLLM_USE_V1="${NRL_VLLM_USE_V1:-1}"
export NRL_WG_USE_RAY_REF=1
export NEMO_RL_VLLM_EXECUTABLE_SYSTEM=1
export NEMO_RL_PY_EXECUTABLES_SYSTEM="${NEMO_RL_PY_EXECUTABLES_SYSTEM:-1}"
export NRL_FORCE_REBUILD_VENVS=false
export NRL_IGNORE_VERSION_MISMATCH=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_USE_RAY_SPMD_WORKER=0
export VLLM_USE_RAY_WRAPPED_PP_COMM=0
export RAY_DEDUP_LOGS=0
"$DRIVER_LAUNCHER" "$REMOTE_SCRIPT" \
  --model "$MODEL_PATH" \
  --draft-model "$DRAFT_MODEL" \
  --prompt-data "$PROMPT_DATA" \
  --output-json "$OUTPUT_JSON" \
  --prompt-limit "${PROMPT_LIMIT:-8}" \
  --prompt-offset "${PROMPT_OFFSET:-0}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS:-1}" \
  --draft-tp "${DRAFT_TP:-1}" \
  --vllm-tp "${VLLM_TP:-8}" \
  --vllm-pp "${VLLM_PP:-1}" \
  --num-nodes "$NUM_NODES" \
  --gpus-per-node "$GPUS_PER_NODE" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.8}" \
  --attention-backend "${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-4}" \
  --max-cudagraph-capture-size "${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE:-0}" \
  "$ENFORCE_EAGER_ARG" \
  "$ASYNC_ENGINE_ARG"
EOF
)

echo "# NeMo-RL VllmGeneration Eagle3 acceptance check"
echo "SPECDEC_RL_DIR=$SPECDEC_RL_DIR"
echo "REMOTE_ROADMAP_ROOT=$REMOTE_ROADMAP_ROOT"
echo "MODEL_PATH=$MODEL_PATH"
echo "DRAFT_MODEL=$DRAFT_MODEL"
echo "PROMPT_DATA=$PROMPT_DATA"
echo "OUTPUT_JSON=$OUTPUT_JSON"
echo "NUM_NODES=$NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE"
echo "VLLM_ENFORCE_EAGER=$VLLM_ENFORCE_EAGER"
echo "ASYNC_ENGINE=${ASYNC_ENGINE:-true}"
echo "DRIVER_LAUNCHER=$DRIVER_LAUNCHER"
echo "RAY_INCLUDE_DASHBOARD=$RAY_INCLUDE_DASHBOARD_VALUE"
echo "RAY_PYTHON_VERSION=$RAY_PYTHON_VERSION_VALUE"
echo "RAY_PYTHON_SPEC=$RAY_PYTHON_SPEC_VALUE"
echo "RAY_VERSION=$RAY_VERSION_VALUE"
echo "RAY_USE_EXISTING_ENV=$RAY_USE_EXISTING_ENV_VALUE"
echo "DEPENDENCY=$DEPENDENCY"
echo "SBATCH_EXTRA_ARGS=$SBATCH_EXTRA_ARGS"
echo "DRY_RUN=$DRY_RUN"

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  printf '%q ' env \
    COMMAND="$COMMAND" \
    CONTAINER="$CONTAINER" \
    MOUNTS="$MOUNTS" \
    BASE_LOG_DIR="$SPECDEC_RL_DIR" \
    GPUS_PER_NODE="$GPUS_PER_NODE" \
    RAY_INCLUDE_DASHBOARD="$RAY_INCLUDE_DASHBOARD_VALUE" \
    RAY_PYTHON_VERSION="$RAY_PYTHON_VERSION_VALUE" \
    RAY_PYTHON_SPEC="$RAY_PYTHON_SPEC_VALUE" \
    RAY_VERSION="$RAY_VERSION_VALUE" \
    RAY_PYTHON_VENV_TAG="${RAY_PYTHON_VENV_TAG:-}" \
    RAY_USE_EXISTING_ENV="$RAY_USE_EXISTING_ENV_VALUE" \
    RAY_ENV_DIR="${RAY_ENV_DIR:-}" \
    PYTHONPATH="$PYTHONPATH_VALUE" \
    HF_HOME="$HF_HOME_PATH" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME_PATH/datasets}" \
    TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME_PATH/hub}" \
    sbatch \
      "${SBATCH_DEPENDENCY_ARG[@]}" \
      "${SBATCH_EXTRA_ARGV[@]}" \
      --nodes="$NUM_NODES" \
      --account="$ACCOUNT" \
      --partition="$PARTITION" \
      --gres="gpu:$GPUS_PER_NODE" \
      --time="$TIME_LIMIT" \
      --chdir="$SPECDEC_RL_DIR" \
      --job-name="$JOB_NAME" \
      "$SPECDEC_RL_DIR/ray.sub"
  printf '\n'
  exit 0
fi

submit_out=$(
  COMMAND="$COMMAND" \
  CONTAINER="$CONTAINER" \
  MOUNTS="$MOUNTS" \
  BASE_LOG_DIR="$SPECDEC_RL_DIR" \
  GPUS_PER_NODE="$GPUS_PER_NODE" \
  RAY_INCLUDE_DASHBOARD="$RAY_INCLUDE_DASHBOARD_VALUE" \
  RAY_PYTHON_VERSION="$RAY_PYTHON_VERSION_VALUE" \
  RAY_PYTHON_SPEC="$RAY_PYTHON_SPEC_VALUE" \
  RAY_VERSION="$RAY_VERSION_VALUE" \
  RAY_PYTHON_VENV_TAG="${RAY_PYTHON_VENV_TAG:-}" \
  RAY_USE_EXISTING_ENV="$RAY_USE_EXISTING_ENV_VALUE" \
  RAY_ENV_DIR="${RAY_ENV_DIR:-}" \
  PYTHONPATH="$PYTHONPATH_VALUE" \
  HF_HOME="$HF_HOME_PATH" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME_PATH/datasets}" \
  TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME_PATH/hub}" \
  sbatch \
    "${SBATCH_DEPENDENCY_ARG[@]}" \
    "${SBATCH_EXTRA_ARGV[@]}" \
    --nodes="$NUM_NODES" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --gres="gpu:$GPUS_PER_NODE" \
    --time="$TIME_LIMIT" \
    --chdir="$SPECDEC_RL_DIR" \
    --job-name="$JOB_NAME" \
    "$SPECDEC_RL_DIR/ray.sub"
)
echo "$submit_out"
job_id="$(awk '/Submitted batch job/{print $4}' <<<"$submit_out" | tail -1)"
if [[ -n "$job_id" ]]; then
  echo "$job_id" > "$REPORT_DIR/latest_nemo_vllm_acceptance_job.txt"
  echo "job_id=$job_id"
fi
