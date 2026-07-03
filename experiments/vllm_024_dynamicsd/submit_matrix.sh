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
DRAFT_MODEL="${DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
VARIANTS="${VARIANTS:-baseline static dynamic}"
JOB_LABEL="${JOB_LABEL:-q32}"
TEMPERATURES="${TEMPERATURES:-0 1}"
STATIC_K="${STATIC_K:-5}"
DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-1:16:5,17:32:4,33:64:3,65:128:1,129:512:0}"
TP="${TP:-1}"
PP="${PP:-1}"
DRAFT_TP="${DRAFT_TP:-1}"
DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-}"
ISL="${ISL:-1024}"
TOP_P="${TOP_P:-1.0}"
SMOKE="${SMOKE:-true}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
MOE_BACKEND="${MOE_BACKEND:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-false}"
THROUGHPUT_GPU_COUNT="${THROUGHPUT_GPU_COUNT:-}"
PROMPT_JSONL="${PROMPT_JSONL:-}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
SUFFIX_MAX_CACHED_REQUESTS="${SUFFIX_MAX_CACHED_REQUESTS:-10000}"
SUFFIX_MAX_SPEC_FACTOR="${SUFFIX_MAX_SPEC_FACTOR:-1.0}"
SUFFIX_MIN_TOKEN_PROB="${SUFFIX_MIN_TOKEN_PROB:-0.1}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

if [[ "${SMOKE}" == "true" ]]; then
  OSL="${OSL:-128}"
  BATCH_SIZES="${BATCH_SIZES:-1 2}"
  WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
  MEASURE_REPEATS="${MEASURE_REPEATS:-1}"
  TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
else
  OSL="${OSL:-512}"
  BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64}"
  WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
  MEASURE_REPEATS="${MEASURE_REPEATS:-3}"
  TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((ISL + OSL + 256))}"
MATRIX_ROOT="${RESULT_ROOT}/${RUN_ID}"
MANIFEST="${MATRIX_ROOT}/jobs.tsv"

normalize_temperature() {
  printf '%s' "$1" | tr '.' 'p'
}

render_sbatch() {
  local variant="$1"
  local temperature="$2"
  local run_dir="$3"
  local output_json="${run_dir}/result.json"
  local attention_arg=""
  local moe_arg=""
  local eager_arg=""
  local custom_all_reduce_arg=""
  local throughput_gpu_arg=""
  local prompt_arg=""
  if [[ -n "${ATTENTION_BACKEND}" ]]; then
    attention_arg="--attention-backend '${ATTENTION_BACKEND}'"
  fi
  if [[ -n "${MOE_BACKEND}" ]]; then
    moe_arg="--moe-backend '${MOE_BACKEND}'"
  fi
  if [[ "${ENFORCE_EAGER}" == "true" ]]; then
    eager_arg="--enforce-eager"
  fi
  if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "true" ]]; then
    custom_all_reduce_arg="--disable-custom-all-reduce"
  fi
  if [[ -n "${THROUGHPUT_GPU_COUNT}" ]]; then
    throughput_gpu_arg="--throughput-gpu-count '${THROUGHPUT_GPU_COUNT}'"
  fi
  if [[ -n "${PROMPT_JSONL}" ]]; then
    prompt_arg="--prompt-jsonl '${PROMPT_JSONL}' --prompt-offset '${PROMPT_OFFSET}'"
  fi
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --segment=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --job-name=coreai_dlalgo_llm-vllm024.${JOB_LABEL}-${variant}-t$(normalize_temperature "${temperature}")
#SBATCH --output=${run_dir}/slurm-%j.out

set -euo pipefail

test -s '${CONTAINER_IMAGE}'
test -d '${MODEL}'
if [[ '${variant}' == 'static' || '${variant}' == 'dynamic' || '${variant}' == 'pard' || '${variant}' == 'pard2' || '${variant}' == 'dflash' ]]; then
  test -d '${DRAFT_MODEL}'
fi
if [[ -n '${PROMPT_JSONL}' ]]; then
  test -s '${PROMPT_JSONL}'
fi

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
export CUDA_MODULE_LOADING=LAZY
export PYTHONUNBUFFERED=1
export HF_HOME='${HF_HOME}'
export HUGGINGFACE_HUB_CACHE='${HF_HOME}/hub'
export HF_DATASETS_CACHE='${HF_HOME}/datasets'
export PYTHONPATH='${EXTRA_PYTHONPATH}'
export NODE_LOCAL_CACHE_ROOT="/tmp/sna/vllm024_\${SLURM_JOB_ID}_${variant}_t$(normalize_temperature "${temperature}")"
export XDG_CACHE_HOME="\${NODE_LOCAL_CACHE_ROOT}/xdg"
export VLLM_CACHE_ROOT="\${NODE_LOCAL_CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="\${NODE_LOCAL_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="\${NODE_LOCAL_CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="\${NODE_LOCAL_CACHE_ROOT}/cuda"
mkdir -p "\${XDG_CACHE_HOME}" "\${VLLM_CACHE_ROOT}" "\${TORCHINDUCTOR_CACHE_DIR}" "\${TRITON_CACHE_DIR}" "\${CUDA_CACHE_PATH}"

echo 'vllm_version_expected=0.24.0'
echo 'model_runner=V1'
echo 'cudagraph_mode=${CUDAGRAPH_MODE}'
echo 'variant=${variant}'
echo 'temperature=${temperature}'
echo 'top_p=${TOP_P}'
if [[ '${variant}' == 'dynamic' ]]; then
  echo 'num_speculative_tokens_per_batch_size=${DYNAMIC_SCHEDULE}'
fi

srun --ntasks=1 \\
  --container-image='${CONTAINER_IMAGE}' \\
  --container-mounts='/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment' \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
  bash -lc "set -euo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
export CUDA_MODULE_LOADING=LAZY
export PYTHONPATH='${EXTRA_PYTHONPATH}'
python3 -c 'import vllm; assert vllm.__version__ == \"0.24.0\", vllm.__version__'
python3 /workspace/experiment/benchmark.py \\
  --model '${MODEL}' \\
  --draft-model '${DRAFT_MODEL}' \\
  --mode '${variant}' \\
  --static-k '${STATIC_K}' \\
  --dynamic-schedule '${DYNAMIC_SCHEDULE}' \\
  --draft-tensor-parallel-size '${DRAFT_TP}' \\
  --draft-attention-backend '${DRAFT_ATTENTION_BACKEND}' \\
  --suffix-max-cached-requests '${SUFFIX_MAX_CACHED_REQUESTS}' \\
  --suffix-max-spec-factor '${SUFFIX_MAX_SPEC_FACTOR}' \\
  --suffix-min-token-prob '${SUFFIX_MIN_TOKEN_PROB}' \\
  --tensor-parallel-size '${TP}' \\
  --pipeline-parallel-size '${PP}' \\
  --dtype bfloat16 \\
  --kv-cache-dtype '${KV_CACHE_DTYPE}' \\
  --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \\
  --max-model-len '${MAX_MODEL_LEN}' \\
  --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}' \\
  --cudagraph-mode '${CUDAGRAPH_MODE}' \\
  --enable-prefix-caching \\
  --enable-chunked-prefill \\
  --isl '${ISL}' \\
  --osl '${OSL}' \\
  --batch-sizes ${BATCH_SIZES} \\
  --temperature ${temperature} \\
  --top-p ${TOP_P} \\
  --warmup-repeats '${WARMUP_REPEATS}' \\
  --measure-repeats '${MEASURE_REPEATS}' \\
  ${attention_arg} \\
  ${moe_arg} \\
  ${eager_arg} \\
  ${custom_all_reduce_arg} \\
  ${throughput_gpu_arg} \\
  ${prompt_arg} \\
  --output '${output_json}' \\
  --tag '${RUN_ID}_${variant}_t$(normalize_temperature "${temperature}")'" \\
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
  printf 'job_id\tvariant\ttemperature\trun_dir\n' >"${MANIFEST}"
fi

for temperature in ${TEMPERATURES}; do
  for variant in ${VARIANTS}; do
    case "${variant}" in
      baseline|static|dynamic|suffix|pard|pard2|dflash) ;;
      *)
        echo "Unsupported variant: ${variant}" >&2
        exit 2
        ;;
    esac
    temp_label="$(normalize_temperature "${temperature}")"
    run_dir="${MATRIX_ROOT}/${variant}_t${temp_label}"
    sbatch_file="${run_dir}/submit.sbatch"
    if [[ "${DRY_RUN}" == "true" ]]; then
      echo "[DRY-RUN] variant=${variant} temperature=${temperature}"
      render_sbatch "${variant}" "${temperature}" "${run_dir}"
      continue
    fi

    mkdir -p "${run_dir}"
    render_sbatch "${variant}" "${temperature}" "${run_dir}" >"${sbatch_file}"
    sbatch_args=()
    if [[ -n "${DEPENDENCY}" ]]; then
      sbatch_args+=("--dependency=${DEPENDENCY}")
    fi
    if [[ "${TEST_ONLY}" == "true" ]]; then
      sbatch --test-only "${sbatch_args[@]}" "${sbatch_file}"
      printf 'test-only\t%s\t%s\t%s\n' "${variant}" "${temperature}" "${run_dir}" >>"${MANIFEST}"
      continue
    fi
    job_id="$(sbatch --parsable "${sbatch_args[@]}" "${sbatch_file}")"
    printf '%s\t%s\t%s\t%s\n' "${job_id}" "${variant}" "${temperature}" "${run_dir}" | tee -a "${MANIFEST}"
  done
done

if [[ "${DRY_RUN}" != "true" ]]; then
  echo "manifest=${MANIFEST}"
  echo "matrix_root=${MATRIX_ROOT}"
fi
