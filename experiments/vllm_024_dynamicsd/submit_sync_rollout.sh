#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() {
  echo "$1" >&2
  exit "${2:-2}"
}

require_safe_identifier() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[A-Za-z0-9._:-]+$ ]]; then
    die "invalid scheduler identifier ${name}=${value}"
  fi
}

require_safe_time_limit() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    die "invalid scheduler identifier ${name}=${value}"
  fi
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    die "invalid scheduler integer ${name}=${value}"
  fi
}

require_safe_dependency() {
  local value="$1"
  if [[ -n "${value}" && ! "${value}" =~ ^[A-Za-z0-9._,:+-]+$ ]]; then
    die "invalid scheduler identifier DEPENDENCY=${value}"
  fi
}

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
CONTEXT_PROFILE="${CONTEXT_PROFILE:-sync_rollout}"
GLOBAL_NUM_PROMPTS="${GLOBAL_NUM_PROMPTS:-}"
GLOBAL_GENERATION_REPLICAS="${GLOBAL_GENERATION_REPLICAS:-}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

require_safe_identifier "ACCOUNT" "${ACCOUNT}"
require_safe_identifier "PARTITION" "${PARTITION}"
require_safe_identifier "JOB_LABEL" "${JOB_LABEL}"
require_positive_integer "TP" "${TP}"
require_positive_integer "PP" "${PP}"
require_positive_integer "NODES" "${NODES}"
require_positive_integer "SEGMENT" "${SEGMENT}"
require_safe_dependency "${DEPENDENCY}"

if [[ -z "${DISTRIBUTED_EXECUTOR_BACKEND}" ]]; then
  if (( NODES == 1 )); then
    if (( TP * PP == 1 )); then
      DISTRIBUTED_EXECUTOR_BACKEND="uni"
    else
      DISTRIBUTED_EXECUTOR_BACKEND="mp"
    fi
  else
    echo "DISTRIBUTED_EXECUTOR_BACKEND is required for multi-node runs" >&2
    exit 2
  fi
fi

shell_quote() {
  printf "%q" "$1"
}

variant_path() {
  local value="$1"
  local variant="$2"
  printf "%s" "${value//\{variant\}/${variant}}"
}

variant_count=0
for _variant_for_count in ${VARIANTS}; do
  variant_count=$((variant_count + 1))
done

require_variant_specific_path() {
  local name="$1"
  local value="$2"
  if (( variant_count <= 1 )); then
    return
  fi
  if [[ -z "${value}" || "${value}" == "auto" || "${value}" == *"{variant}"* ]]; then
    return
  fi
  echo "${name} must be auto or contain {variant} when multiple variants are requested" >&2
  exit 2
}

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
require_safe_time_limit "TIME_LIMIT" "${TIME_LIMIT}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_TOKENS + MAX_NEW_TOKENS + 256))}"
if [[ "${SMOKE}" != "true" && -z "${PROMPT_JSONL}" ]]; then
  echo "SMOKE=false requires PROMPT_JSONL from a pinned prompt set" >&2
  exit 2
fi
require_variant_specific_path "RESPONSE_OUTPUT" "${RESPONSE_OUTPUT}"
require_variant_specific_path "RESOLVED_REQUEST_PLAN_OUTPUT" "${RESOLVED_REQUEST_PLAN_OUTPUT}"
if [[ -n "${REQUEST_PLAN}" && -z "${REQUEST_PLAN_IN_CONTAINER}" && "${REQUEST_PLAN}" == "${SCRIPT_DIR}/"* ]]; then
  REQUEST_PLAN_IN_CONTAINER="/workspace/experiment/${REQUEST_PLAN#${SCRIPT_DIR}/}"
fi
MATRIX_ROOT="${RESULT_ROOT}/${RUN_ID}"
MANIFEST="${MATRIX_ROOT}/jobs.tsv"

emit_arg_pair() {
  local flag="$1"
  local value="$2"
  printf 'args+=(%s %s)\n' "$(shell_quote "${flag}")" "$(shell_quote "${value}")"
}

emit_arg_flag() {
  local flag="$1"
  printf 'args+=(%s)\n' "$(shell_quote "${flag}")"
}

render_run_benchmark() {
  local variant="$1"
  local run_dir="$2"
  local request_plan_value=""
  local resolved_request_plan_output_value=""
  local response_output_value=""
  if [[ -n "${REQUEST_PLAN}" ]]; then
    request_plan_value="${REQUEST_PLAN_IN_CONTAINER:-${REQUEST_PLAN}}"
  fi
  if [[ -n "${RESOLVED_REQUEST_PLAN_OUTPUT}" ]]; then
    if [[ "${RESOLVED_REQUEST_PLAN_OUTPUT}" == "auto" ]]; then
      resolved_request_plan_output_value="${run_dir}/resolved_request_plan.json"
    else
      resolved_request_plan_output_value="$(variant_path "${RESOLVED_REQUEST_PLAN_OUTPUT}" "${variant}")"
    fi
  fi
  if [[ -n "${RESPONSE_OUTPUT}" ]]; then
    if [[ "${RESPONSE_OUTPUT}" == "auto" ]]; then
      response_output_value="${run_dir}/responses.jsonl"
    else
      response_output_value="$(variant_path "${RESPONSE_OUTPUT}" "${variant}")"
    fi
  fi

  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

benchmark_python="\${BENCHMARK_PYTHON:-python3}"
benchmark_script="\${BENCHMARK_SCRIPT:-/workspace/experiment/benchmark_sync_rollout.py}"
runtime_image_sha256=$(shell_quote "${RUNTIME_IMAGE_SHA256}")
if [[ -z "\${runtime_image_sha256}" ]]; then
  runtime_image_sha256="\${BENCH_RUNTIME_IMAGE_SHA256:?BENCH_RUNTIME_IMAGE_SHA256 is required}"
fi

runner_prefix=()
EOF
  if (( NODES > 1 )); then
    printf 'runner_prefix+=(%s)\n' "$(shell_quote "/workspace/experiment/run_multinode_ray.sh")"
  fi
  cat <<'EOF'

if [[ "${CHECK_VLLM_VERSION:-true}" == "true" ]]; then
  "${benchmark_python}" -c 'import vllm; assert vllm.__version__ == "0.24.0", vllm.__version__'
fi

args=()
EOF
  emit_arg_pair "--model" "${MODEL}"
  emit_arg_pair "--draft-model" "${DRAFT_MODEL}"
  if [[ -n "${request_plan_value}" ]]; then
    emit_arg_pair "--request-plan" "${request_plan_value}"
  fi
  if [[ -n "${resolved_request_plan_output_value}" ]]; then
    emit_arg_pair "--resolved-request-plan-output" "${resolved_request_plan_output_value}"
  fi
  if [[ -n "${response_output_value}" ]]; then
    emit_arg_pair "--response-output" "${response_output_value}"
  fi
  emit_arg_pair "--mode" "${variant}"
  emit_arg_pair "--static-k" "${STATIC_K}"
  emit_arg_pair "--dynamic-schedule" "${DYNAMIC_SCHEDULE}"
  emit_arg_pair "--tensor-parallel-size" "${TP}"
  emit_arg_pair "--pipeline-parallel-size" "${PP}"
  emit_arg_pair "--node-count" "${NODES}"
  emit_arg_pair "--context-profile" "${CONTEXT_PROFILE}"
  emit_arg_pair "--dtype" "bfloat16"
  emit_arg_pair "--kv-cache-dtype" "${KV_CACHE_DTYPE}"
  emit_arg_pair "--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}"
  emit_arg_pair "--max-model-len" "${MAX_MODEL_LEN}"
  if [[ "${MAX_NUM_BATCHED_TOKENS}" != "recipe" && "${MAX_NUM_BATCHED_TOKENS}" != "default" ]]; then
    emit_arg_pair "--max-num-batched-tokens" "${MAX_NUM_BATCHED_TOKENS}"
  fi
  emit_arg_pair "--engine-max-num-seqs" "${ENGINE_MAX_NUM_SEQS}"
  emit_arg_pair "--cudagraph-mode" "${CUDAGRAPH_MODE}"
  emit_arg_pair "--num-prompts" "${NUM_PROMPTS}"
  emit_arg_pair "--samples-per-prompt" "${SAMPLES_PER_PROMPT}"
  emit_arg_pair "--rollout-batches" "${ROLLOUT_BATCHES}"
  emit_arg_pair "--max-prompt-tokens" "${MAX_PROMPT_TOKENS}"
  emit_arg_pair "--max-new-tokens" "${MAX_NEW_TOKENS}"
  emit_arg_pair "--temperature" "${TEMPERATURE}"
  emit_arg_pair "--top-p" "${TOP_P}"
  emit_arg_pair "--seed" "${SEED}"
  printf 'args+=(%s "${runtime_image_sha256}")\n' "$(shell_quote "--runtime-image-sha256")"
  if [[ -n "${PROMPT_JSONL}" ]]; then
    emit_arg_pair "--prompt-jsonl" "${PROMPT_JSONL}"
    emit_arg_pair "--prompt-offset" "${PROMPT_OFFSET}"
  fi
  if [[ -n "${ATTENTION_BACKEND}" ]]; then
    emit_arg_pair "--attention-backend" "${ATTENTION_BACKEND}"
  fi
  if [[ -n "${MOE_BACKEND}" ]]; then
    emit_arg_pair "--moe-backend" "${MOE_BACKEND}"
  fi
  emit_arg_pair "--distributed-executor-backend" "${DISTRIBUTED_EXECUTOR_BACKEND}"
  if [[ -n "${DIST_TIMEOUT_SECONDS}" ]]; then
    emit_arg_pair "--distributed-timeout-seconds" "${DIST_TIMEOUT_SECONDS}"
  fi
  if [[ "${ENABLE_EXPERT_PARALLEL}" == "true" ]]; then
    emit_arg_flag "--enable-expert-parallel"
  fi
  if (( MODEL_LOADER_NUM_THREADS > 0 )); then
    emit_arg_pair "--model-loader-num-threads" "${MODEL_LOADER_NUM_THREADS}"
  fi
  if [[ "${DISABLE_FUSE_ALLREDUCE_RMS}" == "true" ]]; then
    emit_arg_flag "--disable-fuse-allreduce-rms"
  fi
  if [[ -n "${MAMBA_SSM_CACHE_DTYPE}" ]]; then
    emit_arg_pair "--mamba-ssm-cache-dtype" "${MAMBA_SSM_CACHE_DTYPE}"
  fi
  if [[ -n "${MAMBA_BACKEND}" ]]; then
    emit_arg_pair "--mamba-backend" "${MAMBA_BACKEND}"
  fi
  if [[ "${ENABLE_MAMBA_CACHE_STOCHASTIC_ROUNDING}" == "true" ]]; then
    emit_arg_flag "--enable-mamba-cache-stochastic-rounding"
  fi
  if [[ -n "${MAMBA_CACHE_PHILOX_ROUNDS}" ]]; then
    emit_arg_pair "--mamba-cache-philox-rounds" "${MAMBA_CACHE_PHILOX_ROUNDS}"
  fi
  if [[ -n "${SOURCE_RECIPE}" ]]; then
    emit_arg_pair "--source-recipe" "${SOURCE_RECIPE}"
  fi
  if [[ -n "${GLOBAL_NUM_PROMPTS}" ]]; then
    emit_arg_pair "--global-num-prompts" "${GLOBAL_NUM_PROMPTS}"
  fi
  if [[ -n "${GLOBAL_GENERATION_REPLICAS}" ]]; then
    emit_arg_pair "--global-generation-replicas" "${GLOBAL_GENERATION_REPLICAS}"
  fi
  emit_arg_pair "--output" "${run_dir}/result.json"
  emit_arg_pair "--tag" "${RUN_ID}_${variant}"
  cat <<'EOF'

if ((${#runner_prefix[@]})); then
  "${runner_prefix[@]}" "${benchmark_python}" "${benchmark_script}" "${args[@]}"
else
  "${benchmark_python}" "${benchmark_script}" "${args[@]}"
fi
EOF
}

render_sbatch() {
  local variant="$1"
  local run_dir="$2"
  local container_pythonpath=""
  local container_image_q=""
  local container_image_sha_q=""
  local model_q=""
  local draft_model_q=""
  local prompt_jsonl_q=""
  local request_plan_host_q=""
  local ray_site_q=""
  local ray_sync_dir_q=""
  local runtime_image_q=""
  local hf_home_q=""
  local hub_cache_q=""
  local datasets_cache_q=""
  local container_pythonpath_q=""
  local container_mounts_q=""
  local run_script_q=""
  local benchmark_log_q=""
  local request_plan_line_q=""
  local source_recipe_line_q=""
  if (( NODES > 1 )); then
    container_pythonpath="${RAY_SITE}"
  fi
  container_image_q="$(shell_quote "${CONTAINER_IMAGE}")"
  container_image_sha_q="$(shell_quote "${CONTAINER_IMAGE}.sha256")"
  model_q="$(shell_quote "${MODEL}")"
  draft_model_q="$(shell_quote "${DRAFT_MODEL}")"
  prompt_jsonl_q="$(shell_quote "${PROMPT_JSONL}")"
  request_plan_host_q="$(shell_quote "${REQUEST_PLAN}")"
  ray_site_q="$(shell_quote "${RAY_SITE}/ray")"
  ray_sync_dir_q="$(shell_quote "${run_dir}/ray-sync")"
  runtime_image_q="$(shell_quote "${RUNTIME_IMAGE_SHA256}")"
  hf_home_q="$(shell_quote "${HF_HOME}")"
  hub_cache_q="$(shell_quote "${HF_HOME}/hub")"
  datasets_cache_q="$(shell_quote "${HF_HOME}/datasets")"
  container_pythonpath_q="$(shell_quote "${container_pythonpath}")"
  container_mounts_q="$(shell_quote "/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment")"
  run_script_q="$(shell_quote "${run_dir}/run_benchmark.sh")"
  benchmark_log_q="$(shell_quote "${run_dir}/benchmark.log")"
  request_plan_line_q="$(shell_quote "request_plan=${REQUEST_PLAN}")"
  source_recipe_line_q="$(shell_quote "source_recipe=${SOURCE_RECIPE}")"
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

set -euo pipefail

test -s ${container_image_q}
test -d ${model_q}
if [[ '${variant}' == 'static' || '${variant}' == 'dynamic' ]]; then
  test -d ${draft_model_q}
fi
if [[ -n ${prompt_jsonl_q} ]]; then
  test -s ${prompt_jsonl_q}
fi
if [[ -n ${request_plan_host_q} ]]; then
  test -s ${request_plan_host_q}
fi
if (( ${NODES} > 1 )); then
  test -d ${ray_site_q}
fi

runtime_image_sha256="\$(if [[ -n ${runtime_image_q} ]]; then printf '%s\n' ${runtime_image_q}; elif [[ -s ${container_image_sha_q} ]]; then awk '{print \$1; exit}' ${container_image_sha_q}; else sha256sum ${container_image_q} | awk '{print \$1; exit}'; fi)"
export BENCH_RUNTIME_IMAGE_SHA256="\${runtime_image_sha256}"

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
export CUDA_MODULE_LOADING=LAZY
export PYTHONUNBUFFERED=1
export HF_HOME=${hf_home_q}
export HUGGINGFACE_HUB_CACHE=${hub_cache_q}
export HF_DATASETS_CACHE=${datasets_cache_q}
export PYTHONPATH=${container_pythonpath_q}
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
printf '%s\n' ${request_plan_line_q}
printf '%s\n' ${source_recipe_line_q}
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
  export RAY_SYNC_DIR=${ray_sync_dir_q}
  export GPUS_PER_NODE=4
  rm -rf "\${RAY_SYNC_DIR}"
fi

srun --nodes=${NODES} --ntasks=${NODES} --ntasks-per-node=1 \\
  --container-image=${container_image_q} \\
  --container-mounts=${container_mounts_q} \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
${run_script_q} \\
  2>&1 | tee ${benchmark_log_q}
EOF
}

render_planned_variant() {
  local marker="$1"
  local variant="$2"
  local run_dir="$3"
  echo "${marker} sync_variant=${variant}"
  echo "${marker} planned_run_script=${run_dir}/run_benchmark.sh"
  echo "# BEGIN run_benchmark.sh ${variant}"
  render_run_benchmark "${variant}" "${run_dir}"
  echo "# END run_benchmark.sh ${variant}"
  echo "# BEGIN submit.sbatch ${variant}"
  render_sbatch "${variant}" "${run_dir}"
  echo "# END submit.sbatch ${variant}"
}

TEST_ONLY_ROOT=""
cleanup_test_only_root() {
  if [[ -n "${TEST_ONLY_ROOT}" ]]; then
    rm -rf "${TEST_ONLY_ROOT}"
  fi
}

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" == "true" ]]; then
  TEST_ONLY_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vllm024-sync-test-only.XXXXXX")"
  trap cleanup_test_only_root EXIT
fi

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" ]]; then
  if [[ ! -s "${CONTAINER_IMAGE}" && -z "${DEPENDENCY}" ]]; then
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
  run_script="${run_dir}/run_benchmark.sh"
  sbatch_args=(
    "--job-name=coreai_dlalgo_llm-vllm024.${JOB_LABEL}-${variant}"
    "--output=${run_dir}/slurm-%j.out"
  )
  if [[ -n "${DEPENDENCY}" ]]; then
    sbatch_args+=("--dependency=${DEPENDENCY}")
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    render_planned_variant "[DRY-RUN]" "${variant}" "${run_dir}"
    continue
  fi
  if [[ "${TEST_ONLY}" == "true" ]]; then
    echo "[TEST-ONLY] sync_variant=${variant}"
    test_run_dir="${TEST_ONLY_ROOT}/${variant}"
    test_sbatch_file="${test_run_dir}/submit.sbatch"
    test_run_script="${test_run_dir}/run_benchmark.sh"
    mkdir -p "${test_run_dir}"
    render_run_benchmark "${variant}" "${test_run_dir}" >"${test_run_script}"
    chmod 755 "${test_run_script}"
    render_sbatch "${variant}" "${test_run_dir}" >"${test_sbatch_file}"
    sbatch --test-only "${sbatch_args[@]}" "${test_sbatch_file}"
    continue
  fi

  mkdir -p "${run_dir}"
  render_run_benchmark "${variant}" "${run_dir}" >"${run_script}"
  chmod 755 "${run_script}"
  render_sbatch "${variant}" "${run_dir}" >"${sbatch_file}"
  job_id="$(sbatch --parsable "${sbatch_args[@]}" "${sbatch_file}")"
  printf '%s\t%s\t%s\n' "${job_id}" "${variant}" "${run_dir}" | tee -a "${MANIFEST}"
done

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" ]]; then
  echo "manifest=${MANIFEST}"
  echo "matrix_root=${MATRIX_ROOT}"
fi
