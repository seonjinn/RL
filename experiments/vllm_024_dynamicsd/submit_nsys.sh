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
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/profiles}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
VARIANTS="${VARIANTS:-baseline static dynamic}"
STATIC_K="${STATIC_K:-5}"
DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-1:16:5,17:32:4,33:64:3,65:128:1,129:512:0}"
PROFILE_BATCH_SIZE="${PROFILE_BATCH_SIZE:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TP="${TP:-1}"
PP="${PP:-1}"
ISL="${ISL:-1024}"
OSL="${OSL:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((ISL + OSL + 256))}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"
PROFILE_ROOT="${RESULT_ROOT}/${RUN_ID}"
MANIFEST="${PROFILE_ROOT}/jobs.tsv"

render_sbatch() {
  local variant="$1"
  local run_dir="$2"
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
#SBATCH --job-name=v024-nsys-${variant}-bs${PROFILE_BATCH_SIZE}
#SBATCH --output=${run_dir}/slurm-%j.out

set -euo pipefail

test -s '${CONTAINER_IMAGE}'
test -d '${MODEL}'
if [[ '${variant}' != 'baseline' ]]; then
  test -d '${DRAFT_MODEL}'
fi

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
export CUDA_MODULE_LOADING=LAZY
export PYTHONUNBUFFERED=1
export HF_HOME='${HF_HOME}'
export HUGGINGFACE_HUB_CACHE='${HF_HOME}/hub'
export NODE_LOCAL_CACHE_ROOT="/tmp/sna/vllm024_nsys_\${SLURM_JOB_ID}_${variant}_bs${PROFILE_BATCH_SIZE}"
export XDG_CACHE_HOME="\${NODE_LOCAL_CACHE_ROOT}/xdg"
export VLLM_CACHE_ROOT="\${NODE_LOCAL_CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="\${NODE_LOCAL_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="\${NODE_LOCAL_CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="\${NODE_LOCAL_CACHE_ROOT}/cuda"
mkdir -p "\${XDG_CACHE_HOME}" "\${VLLM_CACHE_ROOT}" "\${TORCHINDUCTOR_CACHE_DIR}" "\${TRITON_CACHE_DIR}" "\${CUDA_CACHE_PATH}"

echo 'vllm_version_expected=0.24.0'
echo 'model_runner=V1'
echo 'cudagraph_mode=${CUDAGRAPH_MODE}'
echo 'profile_variant=${variant}'
echo 'profile_batch_size=${PROFILE_BATCH_SIZE}'

srun --ntasks=1 \\
  --container-image='${CONTAINER_IMAGE}' \\
  --container-mounts='/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment' \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
  bash -lc "set -euo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
nsys_bin=\"\$(command -v nsys || true)\"
if [[ -z \"\${nsys_bin}\" ]]; then
  for candidate in /opt/nvidia/nsight-systems/*/bin/nsys /opt/nvidia/nsight-systems-cli/*/bin/nsys /usr/local/cuda/bin/nsys; do
    if [[ -x \"\${candidate}\" ]]; then
      nsys_bin=\"\${candidate}\"
      break
    fi
  done
fi
if [[ -z \"\${nsys_bin}\" ]]; then
  echo 'nsys is not installed in the vLLM 0.24 image' | tee '${run_dir}/nsys_unavailable.txt'
  exit 3
fi
\"\${nsys_bin}\" --version | tee '${run_dir}/nsys_version.txt'
\"\${nsys_bin}\" profile \\
  --trace=cuda,nvtx,osrt,cublas,cudnn,nccl \\
  --cuda-graph-trace=node \\
  --trace-fork-before-exec=true \\
  --capture-range=cudaProfilerApi \\
  --capture-range-end=stop \\
  --sample=none \\
  --cpuctxsw=none \\
  --force-overwrite=true \\
  --output='${run_dir}/profile' \\
  python3 /workspace/experiment/benchmark.py \\
    --model '${MODEL}' \\
    --draft-model '${DRAFT_MODEL}' \\
    --mode '${variant}' \\
    --static-k '${STATIC_K}' \\
    --dynamic-schedule '${DYNAMIC_SCHEDULE}' \\
    --tensor-parallel-size '${TP}' \\
    --pipeline-parallel-size '${PP}' \\
    --dtype bfloat16 \\
    --kv-cache-dtype auto \\
    --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \\
    --max-model-len '${MAX_MODEL_LEN}' \\
    --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}' \\
    --cudagraph-mode '${CUDAGRAPH_MODE}' \\
    --enable-prefix-caching \\
    --enable-chunked-prefill \\
    --isl '${ISL}' \\
    --osl '${OSL}' \\
    --batch-sizes ${PROFILE_BATCH_SIZE} \\
    --temperature '${TEMPERATURE}' \\
    --top-p '${TOP_P}' \\
    --warmup-repeats 2 \\
    --measure-repeats 1 \\
    --cuda-profiler-range \\
    --output '${run_dir}/result.json' \\
    --tag '${RUN_ID}_${variant}_bs${PROFILE_BATCH_SIZE}'
\"\${nsys_bin}\" stats \\
  --report cuda_gpu_kern_sum,cuda_api_sum \\
  --format csv \\
  --output '${run_dir}/nsys_stats' \\
  '${run_dir}/profile.nsys-rep' || true
" 2>&1 | tee '${run_dir}/profile.log'
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
  mkdir -p "${PROFILE_ROOT}"
  printf 'job_id\tvariant\tbatch_size\trun_dir\n' >"${MANIFEST}"
fi

for variant in ${VARIANTS}; do
  case "${variant}" in
    baseline|static|dynamic) ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 2
      ;;
  esac
  run_dir="${PROFILE_ROOT}/${variant}_bs${PROFILE_BATCH_SIZE}"
  sbatch_file="${run_dir}/submit.sbatch"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] nsys_variant=${variant} batch_size=${PROFILE_BATCH_SIZE}"
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
    printf 'test-only\t%s\t%s\t%s\n' "${variant}" "${PROFILE_BATCH_SIZE}" "${run_dir}" >>"${MANIFEST}"
    continue
  fi
  job_id="$(sbatch --parsable "${sbatch_args[@]}" "${sbatch_file}")"
  printf '%s\t%s\t%s\t%s\n' "${job_id}" "${variant}" "${PROFILE_BATCH_SIZE}" "${run_dir}" | tee -a "${MANIFEST}"
done

if [[ "${DRY_RUN}" != "true" ]]; then
  echo "manifest=${MANIFEST}"
  echo "profile_root=${PROFILE_ROOT}"
fi
