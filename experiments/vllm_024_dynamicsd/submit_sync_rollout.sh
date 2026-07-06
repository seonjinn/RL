#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${CLUSTER:-auto}"
if [[ "${CLUSTER}" == "auto" ]]; then
  case "$(hostname)" in
    *lyris*) CLUSTER="lyris" ;;
    *ptyche*) CLUSTER="ptyche" ;;
    *)
      echo "Set CLUSTER=lyris or CLUSTER=ptyche" >&2
      exit 2
      ;;
  esac
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
case "${CLUSTER}" in
  lyris) PARTITION="${PARTITION:-gb200}" ;;
  ptyche) PARTITION="${PARTITION:-batch}" ;;
  *)
    echo "Unsupported CLUSTER=${CLUSTER}" >&2
    exit 2
    ;;
esac

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
MODEL="${MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
DRAFT_MODEL="${DRAFT_MODEL-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/sync-rollout}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
VARIANTS="${VARIANTS:-baseline static dynamic}"
JOB_LABEL="${JOB_LABEL:-sync}"
STATIC_K="${STATIC_K:-5}"
DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-1:16:5,17:32:4,33:64:3,65:128:1,129:512:0}"
TP="${TP:-2}"
PP="${PP:-1}"
NODES="${NODES:-1}"
SEGMENT="${SEGMENT:-${NODES}}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-1234}"
SMOKE="${SMOKE:-true}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
ENGINE_MAX_NUM_SEQS="${ENGINE_MAX_NUM_SEQS:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
MOE_BACKEND="${MOE_BACKEND:-}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"
DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-false}"
MODEL_LOADER_NUM_THREADS="${MODEL_LOADER_NUM_THREADS:-0}"
DISABLE_FUSE_ALLREDUCE_RMS="${DISABLE_FUSE_ALLREDUCE_RMS:-false}"
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-}"
MAMBA_BACKEND="${MAMBA_BACKEND:-}"
ENABLE_MAMBA_CACHE_STOCHASTIC_ROUNDING="${ENABLE_MAMBA_CACHE_STOCHASTIC_ROUNDING:-false}"
MAMBA_CACHE_PHILOX_ROUNDS="${MAMBA_CACHE_PHILOX_ROUNDS:-}"
RAY_SITE="${RAY_SITE:-${LUSTRE_ROOT}/vllm024-dynamicsd/python-sites/ray-2.55.1-py312}"
PROMPT_JSONL="${PROMPT_JSONL:-}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
REQUEST_PLAN="${REQUEST_PLAN:-}"
REQUEST_PLAN_IN_CONTAINER="${REQUEST_PLAN_IN_CONTAINER:-}"
RESOLVED_REQUEST_PLAN_OUTPUT="${RESOLVED_REQUEST_PLAN_OUTPUT:-}"
RESPONSE_OUTPUT="${RESPONSE_OUTPUT:-}"
RUNTIME_IMAGE_SHA256="${RUNTIME_IMAGE_SHA256:-}"
SOURCE_RECIPE="${SOURCE_RECIPE:-}"
GLOBAL_NUM_PROMPTS="${GLOBAL_NUM_PROMPTS:-}"
GLOBAL_GENERATION_REPLICAS="${GLOBAL_GENERATION_REPLICAS:-}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

if [[ "${SMOKE}" == "true" ]]; then
  NUM_PROMPTS="${NUM_PROMPTS:-4}"
  SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-2}"
  ROLLOUT_BATCHES="${ROLLOUT_BATCHES:-2}"
  MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
  TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
else
  NUM_PROMPTS="${NUM_PROMPTS:-16}"
  SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-16}"
  ROLLOUT_BATCHES="${ROLLOUT_BATCHES:-3}"
  MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
  TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_TOKENS + MAX_NEW_TOKENS + 256))}"
if [[ "${SMOKE}" != "true" && -z "${PROMPT_JSONL}" ]]; then
  echo "SMOKE=false requires PROMPT_JSONL from a pinned prompt set" >&2
  exit 2
fi
if [[ -n "${REQUEST_PLAN}" && -z "${REQUEST_PLAN_IN_CONTAINER}" && "${REQUEST_PLAN}" == "${SCRIPT_DIR}/"* ]]; then
  REQUEST_PLAN_IN_CONTAINER="/workspace/experiment/${REQUEST_PLAN#${SCRIPT_DIR}/}"
fi
MATRIX_ROOT="${RESULT_ROOT}/${RUN_ID}"
MANIFEST="${MATRIX_ROOT}/jobs.tsv"

render_sbatch() {
  local variant="$1"
  local run_dir="$2"
  local prompt_arg=""
  local attention_arg=""
  local moe_arg=""
  local distributed_arg=""
  local timeout_arg=""
  local expert_parallel_arg=""
  local model_loader_arg=""
  local compilation_arg=""
  local mamba_ssm_arg=""
  local mamba_backend_arg=""
  local mamba_rounding_arg=""
  local mamba_philox_arg=""
  local batched_tokens_arg=""
  local request_plan_arg=""
  local resolved_request_plan_arg=""
  local response_output_arg=""
  local recipe_arg=""
  local global_prompts_arg=""
  local global_replicas_arg=""
  local runner_prefix=""
  local container_pythonpath=""
  if [[ -n "${PROMPT_JSONL}" ]]; then
    prompt_arg="--prompt-jsonl '${PROMPT_JSONL}' --prompt-offset '${PROMPT_OFFSET}'"
  fi
  if [[ -n "${ATTENTION_BACKEND}" ]]; then
    attention_arg="--attention-backend '${ATTENTION_BACKEND}'"
  fi
  if [[ -n "${MOE_BACKEND}" ]]; then
    moe_arg="--moe-backend '${MOE_BACKEND}'"
  fi
  if [[ -n "${DISTRIBUTED_EXECUTOR_BACKEND}" ]]; then
    distributed_arg="--distributed-executor-backend '${DISTRIBUTED_EXECUTOR_BACKEND}'"
  fi
  if [[ -n "${DIST_TIMEOUT_SECONDS}" ]]; then
    timeout_arg="--distributed-timeout-seconds '${DIST_TIMEOUT_SECONDS}'"
  fi
  if [[ "${ENABLE_EXPERT_PARALLEL}" == "true" ]]; then
    expert_parallel_arg="--enable-expert-parallel"
  fi
  if (( MODEL_LOADER_NUM_THREADS > 0 )); then
    model_loader_arg="--model-loader-num-threads '${MODEL_LOADER_NUM_THREADS}'"
  fi
  if [[ "${DISABLE_FUSE_ALLREDUCE_RMS}" == "true" ]]; then
    compilation_arg="--disable-fuse-allreduce-rms"
  fi
  if [[ -n "${MAMBA_SSM_CACHE_DTYPE}" ]]; then
    mamba_ssm_arg="--mamba-ssm-cache-dtype '${MAMBA_SSM_CACHE_DTYPE}'"
  fi
  if [[ -n "${MAMBA_BACKEND}" ]]; then
    mamba_backend_arg="--mamba-backend '${MAMBA_BACKEND}'"
  fi
  if [[ "${ENABLE_MAMBA_CACHE_STOCHASTIC_ROUNDING}" == "true" ]]; then
    mamba_rounding_arg="--enable-mamba-cache-stochastic-rounding"
  fi
  if [[ -n "${MAMBA_CACHE_PHILOX_ROUNDS}" ]]; then
    mamba_philox_arg="--mamba-cache-philox-rounds '${MAMBA_CACHE_PHILOX_ROUNDS}'"
  fi
  if [[ "${MAX_NUM_BATCHED_TOKENS}" != "recipe" && "${MAX_NUM_BATCHED_TOKENS}" != "default" ]]; then
    batched_tokens_arg="--max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}'"
  fi
  if [[ -n "${REQUEST_PLAN}" ]]; then
    request_plan_arg="--request-plan '${REQUEST_PLAN_IN_CONTAINER:-${REQUEST_PLAN}}'"
  fi
  if [[ -n "${RESOLVED_REQUEST_PLAN_OUTPUT}" ]]; then
    if [[ "${RESOLVED_REQUEST_PLAN_OUTPUT}" == "auto" ]]; then
      resolved_request_plan_arg="--resolved-request-plan-output '${run_dir}/resolved_request_plan.json'"
    else
      resolved_request_plan_arg="--resolved-request-plan-output '${RESOLVED_REQUEST_PLAN_OUTPUT}'"
    fi
  fi
  if [[ -n "${RESPONSE_OUTPUT}" ]]; then
    if [[ "${RESPONSE_OUTPUT}" == "auto" ]]; then
      response_output_arg="--response-output '${run_dir}/responses.jsonl'"
    else
      response_output_arg="--response-output '${RESPONSE_OUTPUT}'"
    fi
  fi
  if [[ -n "${SOURCE_RECIPE}" ]]; then
    recipe_arg="--source-recipe '${SOURCE_RECIPE}'"
  fi
  if [[ -n "${GLOBAL_NUM_PROMPTS}" ]]; then
    global_prompts_arg="--global-num-prompts '${GLOBAL_NUM_PROMPTS}'"
  fi
  if [[ -n "${GLOBAL_GENERATION_REPLICAS}" ]]; then
    global_replicas_arg="--global-generation-replicas '${GLOBAL_GENERATION_REPLICAS}'"
  fi
  if (( NODES > 1 )); then
    runner_prefix="/workspace/experiment/run_multinode_ray.sh"
    container_pythonpath="${RAY_SITE}"
  fi
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --segment=${SEGMENT}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --job-name=coreai_dlalgo_llm-vllm024.${JOB_LABEL}-${variant}
#SBATCH --output=${run_dir}/slurm-%j.out

set -euo pipefail

test -s '${CONTAINER_IMAGE}'
test -d '${MODEL}'
if [[ '${variant}' == 'static' || '${variant}' == 'dynamic' ]]; then
  test -d '${DRAFT_MODEL}'
fi
if [[ -n '${PROMPT_JSONL}' ]]; then
  test -s '${PROMPT_JSONL}'
fi
if [[ -n '${REQUEST_PLAN}' ]]; then
  test -s '${REQUEST_PLAN}'
fi
if (( ${NODES} > 1 )); then
  test -d '${RAY_SITE}/ray'
fi

runtime_image_sha256="\$(if [[ -n '${RUNTIME_IMAGE_SHA256}' ]]; then printf '%s\n' '${RUNTIME_IMAGE_SHA256}'; elif [[ -s '${CONTAINER_IMAGE}.sha256' ]]; then awk '{print \$1; exit}' '${CONTAINER_IMAGE}.sha256'; else sha256sum '${CONTAINER_IMAGE}' | awk '{print \$1; exit}'; fi)"

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
export CUDA_MODULE_LOADING=LAZY
export PYTHONUNBUFFERED=1
export HF_HOME='${HF_HOME}'
export HUGGINGFACE_HUB_CACHE='${HF_HOME}/hub'
export HF_DATASETS_CACHE='${HF_HOME}/datasets'
export PYTHONPATH='${container_pythonpath}'
export NODE_LOCAL_CACHE_ROOT="/tmp/sna/vllm024_sync_\${SLURM_JOB_ID}_${variant}"
export XDG_CACHE_HOME="\${NODE_LOCAL_CACHE_ROOT}/xdg"
export VLLM_CACHE_ROOT="\${NODE_LOCAL_CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="\${NODE_LOCAL_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="\${NODE_LOCAL_CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="\${NODE_LOCAL_CACHE_ROOT}/cuda"
mkdir -p "\${XDG_CACHE_HOME}" "\${VLLM_CACHE_ROOT}" "\${TORCHINDUCTOR_CACHE_DIR}" "\${TRITON_CACHE_DIR}" "\${CUDA_CACHE_PATH}"

echo 'vllm_version_expected=0.24.0'
echo 'scenario=synchronous_rl_rollout'
echo 'sync_barrier=LLM.generate_return'
echo 'cudagraph_mode=${CUDAGRAPH_MODE}'
echo 'variant=${variant}'
echo 'temperature=${TEMPERATURE}'
echo 'top_p=${TOP_P}'
echo 'num_prompts=${NUM_PROMPTS}'
echo 'samples_per_prompt=${SAMPLES_PER_PROMPT}'
echo 'requests_per_rollout_batch=$((NUM_PROMPTS * SAMPLES_PER_PROMPT))'
echo 'engine_max_num_seqs=${ENGINE_MAX_NUM_SEQS}'
echo 'request_plan=${REQUEST_PLAN}'
echo 'source_recipe=${SOURCE_RECIPE}'
echo 'moe_backend=${MOE_BACKEND:-auto}'
echo 'nodes=${NODES}'
echo 'target_tp=${TP}'
if [[ '${variant}' == 'mtp_static' || '${variant}' == 'mtp_dynamic' ]]; then
  echo 'method=mtp'
fi
if [[ '${variant}' == 'mtp_dynamic' ]]; then
  echo 'num_speculative_tokens_per_batch_size=${DYNAMIC_SCHEDULE}'
fi

if (( ${NODES} > 1 )); then
  export HEAD_NODE="\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)"
  export HEAD_IP="\$(srun --nodes=1 --ntasks=1 --nodelist="\${HEAD_NODE}" hostname -I | awk '{print \$1}')"
  export RAY_PORT="\$((20000 + SLURM_JOB_ID % 10000))"
  export RAY_SYNC_DIR='${run_dir}/ray-sync'
  export GPUS_PER_NODE=4
  rm -rf "\${RAY_SYNC_DIR}"
fi

srun --nodes=${NODES} --ntasks=${NODES} --ntasks-per-node=1 \\
  --container-image='${CONTAINER_IMAGE}' \\
  --container-mounts='/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment' \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
  bash -lc "set -euo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
python3 -c 'import vllm; assert vllm.__version__ == \"0.24.0\", vllm.__version__'
${runner_prefix} python3 /workspace/experiment/benchmark_sync_rollout.py \\
  --model '${MODEL}' \\
  --draft-model '${DRAFT_MODEL}' \\
  --mode '${variant}' \\
  --static-k '${STATIC_K}' \\
  --dynamic-schedule '${DYNAMIC_SCHEDULE}' \\
  --tensor-parallel-size '${TP}' \\
  --pipeline-parallel-size '${PP}' \\
  --dtype bfloat16 \\
  --kv-cache-dtype '${KV_CACHE_DTYPE}' \\
  --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \\
  --max-model-len '${MAX_MODEL_LEN}' \\
  ${batched_tokens_arg} \\
  --engine-max-num-seqs ${ENGINE_MAX_NUM_SEQS} \\
  --cudagraph-mode '${CUDAGRAPH_MODE}' \\
  --num-prompts ${NUM_PROMPTS} \\
  --samples-per-prompt ${SAMPLES_PER_PROMPT} \\
  --rollout-batches ${ROLLOUT_BATCHES} \\
  --max-prompt-tokens ${MAX_PROMPT_TOKENS} \\
  --max-new-tokens ${MAX_NEW_TOKENS} \\
  --temperature ${TEMPERATURE} \\
  --top-p ${TOP_P} \\
  --seed ${SEED} \\
  --runtime-image-sha256 '\${runtime_image_sha256}' \\
  ${prompt_arg} \\
  ${request_plan_arg} \\
  ${resolved_request_plan_arg} \\
  ${response_output_arg} \\
  ${attention_arg} \\
  ${moe_arg} \\
  ${distributed_arg} \\
  ${timeout_arg} \\
  ${expert_parallel_arg} \\
  ${model_loader_arg} \\
  ${compilation_arg} \\
  ${mamba_ssm_arg} \\
  ${mamba_backend_arg} \\
  ${mamba_rounding_arg} \\
  ${mamba_philox_arg} \\
  ${recipe_arg} \\
  ${global_prompts_arg} \\
  ${global_replicas_arg} \\
  --output '${run_dir}/result.json' \\
  --tag '${RUN_ID}_${variant}'" \\
  2>&1 | tee '${run_dir}/benchmark.log'
EOF
}

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  if [[ "${TEST_ONLY}" != "true" && ! -s "${CONTAINER_IMAGE}" && -z "${DEPENDENCY}" ]]; then
    echo "Missing image and no dependency supplied: ${CONTAINER_IMAGE}" >&2
    exit 3
  fi
  mkdir -p "${MATRIX_ROOT}"
  printf 'job_id\tvariant\trun_dir\n' >"${MANIFEST}"
fi

for variant in ${VARIANTS}; do
  case "${variant}" in
    baseline|static|dynamic|mtp_static|mtp_dynamic) ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 2
      ;;
  esac
  run_dir="${MATRIX_ROOT}/${variant}"
  sbatch_file="${run_dir}/submit.sbatch"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] sync_variant=${variant}"
    render_sbatch "${variant}" "${run_dir}"
    continue
  fi

  mkdir -p "${run_dir}"
  render_sbatch "${variant}" "${run_dir}" >"${sbatch_file}"
  sbatch_args=()
  if [[ -n "${DEPENDENCY}" ]]; then
    sbatch_args+=("--dependency=${DEPENDENCY}")
  fi
  if [[ "${TEST_ONLY}" == "true" ]]; then
    sbatch --test-only "${sbatch_args[@]}" "${sbatch_file}"
    printf 'test-only\t%s\t%s\n' "${variant}" "${run_dir}" >>"${MANIFEST}"
    continue
  fi
  job_id="$(sbatch --parsable "${sbatch_args[@]}" "${sbatch_file}")"
  printf '%s\t%s\t%s\n' "${job_id}" "${variant}" "${run_dir}" | tee -a "${MANIFEST}"
done

if [[ "${DRY_RUN}" != "true" ]]; then
  echo "manifest=${MANIFEST}"
  echo "matrix_root=${MATRIX_ROOT}"
fi
