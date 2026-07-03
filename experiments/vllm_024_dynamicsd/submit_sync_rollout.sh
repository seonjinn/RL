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
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/sync-rollout}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
VARIANTS="${VARIANTS:-baseline static dynamic}"
STATIC_K="${STATIC_K:-5}"
DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-1:16:5,17:32:4,33:64:3,65:128:1,129:512:0}"
TP="${TP:-2}"
PP="${PP:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-1234}"
SMOKE="${SMOKE:-true}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
ENGINE_MAX_NUM_SEQS="${ENGINE_MAX_NUM_SEQS:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
PROMPT_JSONL="${PROMPT_JSONL:-}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
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
  echo "SMOKE=false requires PROMPT_JSONL from a pinned RL math dataset" >&2
  exit 2
fi
MATRIX_ROOT="${RESULT_ROOT}/${RUN_ID}"
MANIFEST="${MATRIX_ROOT}/jobs.tsv"

render_sbatch() {
  local variant="$1"
  local run_dir="$2"
  local prompt_arg=""
  local attention_arg=""
  if [[ -n "${PROMPT_JSONL}" ]]; then
    prompt_arg="--prompt-jsonl '${PROMPT_JSONL}' --prompt-offset '${PROMPT_OFFSET}'"
  fi
  if [[ -n "${ATTENTION_BACKEND}" ]]; then
    attention_arg="--attention-backend '${ATTENTION_BACKEND}'"
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
#SBATCH --job-name=coreai_dlalgo_llm-vllm024.sync-${variant}
#SBATCH --output=${run_dir}/slurm-%j.out

set -euo pipefail

test -s '${CONTAINER_IMAGE}'
test -d '${MODEL}'
if [[ '${variant}' != 'baseline' ]]; then
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

srun --ntasks=1 \\
  --container-image='${CONTAINER_IMAGE}' \\
  --container-mounts='/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment' \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
  bash -lc "set -euo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
python3 -c 'import vllm; assert vllm.__version__ == \"0.24.0\", vllm.__version__'
python3 /workspace/experiment/benchmark_sync_rollout.py \\
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
  ${prompt_arg} \\
  ${attention_arg} \\
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
    baseline|static|dynamic) ;;
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
