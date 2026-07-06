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

shell_quote() {
  printf "%q" "$1"
}

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
MODEL="${MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
DRAFT_MODEL="${DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/speedbench-k-calibration}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_speedbench_k_calibration}"
PREPARED_JSONL="${PREPARED_JSONL:-${LUSTRE_ROOT}/vllm024-dynamicsd/speedbench/overlay_prompts.jsonl}"
REQUEST_PLAN="${REQUEST_PLAN:-${SCRIPT_DIR}/profiles/swe_sync_32k.json}"
REQUEST_PLAN_IN_CONTAINER="${REQUEST_PLAN_IN_CONTAINER:-/workspace/experiment/profiles/swe_sync_32k.json}"
CONCURRENCIES="${CONCURRENCIES:-1 8 32 64}"
K_VALUES="${K_VALUES:-1 2 3 4 5}"
DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-1:16:5,17:32:4,33:64:3,65:128:1,129:512:0}"
TP="${TP:-2}"
PP="${PP:-1}"
NODES="${NODES:-1}"
SEGMENT="${SEGMENT:-${NODES}}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-1234}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
ROLLOUT_BATCHES="${ROLLOUT_BATCHES:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
RUNTIME_IMAGE_SHA256="${RUNTIME_IMAGE_SHA256:-}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

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
  local static_k="$2"
  local concurrency="$3"
  local run_dir="$4"
  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

benchmark_python="\${BENCHMARK_PYTHON:-python3}"
benchmark_script="\${BENCHMARK_SCRIPT:-/workspace/experiment/benchmark_speedbench_sync_rollout.py}"
runtime_image_sha256=$(shell_quote "${RUNTIME_IMAGE_SHA256}")
if [[ -z "\${runtime_image_sha256}" ]]; then
  runtime_image_sha256="\${BENCH_RUNTIME_IMAGE_SHA256:-unknown}"
fi

args=()
EOF
  emit_arg_pair "--cohort" "overlay"
  emit_arg_pair "--model" "${MODEL}"
  emit_arg_pair "--draft-model" "${DRAFT_MODEL}"
  emit_arg_pair "--mode" "${variant}"
  emit_arg_pair "--static-k" "${static_k}"
  emit_arg_pair "--dynamic-schedule" "${DYNAMIC_SCHEDULE}"
  emit_arg_pair "--tensor-parallel-size" "${TP}"
  emit_arg_pair "--pipeline-parallel-size" "${PP}"
  emit_arg_pair "--kv-cache-dtype" "${KV_CACHE_DTYPE}"
  emit_arg_pair "--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}"
  emit_arg_pair "--max-model-len" "${MAX_MODEL_LEN}"
  emit_arg_pair "--active-concurrency" "${concurrency}"
  emit_arg_pair "--samples-per-prompt" "${SAMPLES_PER_PROMPT}"
  emit_arg_pair "--rollout-batches" "${ROLLOUT_BATCHES}"
  emit_arg_pair "--temperature" "${TEMPERATURE}"
  emit_arg_pair "--top-p" "${TOP_P}"
  emit_arg_pair "--seed" "${SEED}"
  emit_arg_pair "--cudagraph-mode" "${CUDAGRAPH_MODE}"
  emit_arg_pair "--prepared-jsonl" "${PREPARED_JSONL}"
  emit_arg_pair "--request-plan" "${REQUEST_PLAN_IN_CONTAINER}"
  emit_arg_flag "--request-plan-exact-work"
  emit_arg_pair "--runtime-image-sha256" "\${runtime_image_sha256}"
  emit_arg_pair "--output" "${run_dir}/result.json"
  cat <<'EOF'

"${benchmark_python}" "${benchmark_script}" "${args[@]}"
EOF
}

render_sbatch() {
  local variant="$1"
  local method="$2"
  local concurrency="$3"
  local run_dir="$4"
  local container_mounts="/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment"
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
#SBATCH --job-name=coreai_dlalgo_llm-speedbench-c${concurrency}-${variant}
#SBATCH --output=${run_dir}/slurm-%j.out

set -euo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
export HF_HOME=$(shell_quote "${HF_HOME}")
echo 'cohort=overlay'
echo 'method=${method}'
echo 'variant=${variant}'
echo 'active_concurrency=${concurrency}'
echo 'cudagraph_mode=${CUDAGRAPH_MODE}'
srun --nodes=${NODES} --ntasks=${NODES} --ntasks-per-node=1 \\
  --container-image=$(shell_quote "${CONTAINER_IMAGE}") \\
  --container-mounts=$(shell_quote "${container_mounts}") \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
  $(shell_quote "${run_dir}/run_benchmark.sh") \\
  2>&1 | tee $(shell_quote "${run_dir}/benchmark.log")
EOF
}

render_planned_job() {
  local marker="$1"
  local variant="$2"
  local method="$3"
  local static_k="$4"
  local concurrency="$5"
  local run_dir="$6"
  echo "${marker} speedbench_overlay=${variant}"
  echo "${marker} cohort=overlay method=${method} active_concurrency=${concurrency} static_k=${static_k}"
  echo "${marker} planned_run_script=${run_dir}/run_benchmark.sh"
  echo "# BEGIN run_benchmark.sh ${variant}-c${concurrency}-k${static_k}"
  render_run_benchmark "${variant}" "${static_k}" "${concurrency}" "${run_dir}"
  echo "# END run_benchmark.sh ${variant}-c${concurrency}-k${static_k}"
  echo "# BEGIN submit.sbatch ${variant}-c${concurrency}-k${static_k}"
  render_sbatch "${variant}" "${method}" "${concurrency}" "${run_dir}"
  echo "# END submit.sbatch ${variant}-c${concurrency}-k${static_k}"
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
  TEST_ONLY_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/speedbench-k-test-only.XXXXXX")"
  trap cleanup_test_only_root EXIT
fi

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" ]]; then
  mkdir -p "${MATRIX_ROOT}"
  printf 'job_id\tcohort\tmethod\tvariant\tactive_concurrency\tstatic_k\trun_dir\n' >"${MANIFEST}"
fi

for concurrency in ${CONCURRENCIES}; do
  jobs=("baseline:baseline:0")
  for k_value in ${K_VALUES}; do
    jobs+=("static:eagle3:${k_value}")
  done
  for job in "${jobs[@]}"; do
    IFS=: read -r variant method static_k <<<"${job}"
    run_dir="${MATRIX_ROOT}/c${concurrency}/${variant}"
    if [[ "${variant}" == "static" ]]; then
      run_dir="${MATRIX_ROOT}/c${concurrency}/static_k${static_k}"
    fi
    if [[ "${DRY_RUN}" == "true" ]]; then
      render_planned_job "[DRY-RUN]" "${variant}" "${method}" "${static_k}" "${concurrency}" "${run_dir}"
      continue
    fi
    if [[ "${TEST_ONLY}" == "true" ]]; then
      echo "[TEST-ONLY] speedbench_overlay=${variant}"
      test_run_dir="${TEST_ONLY_ROOT}/c${concurrency}/${variant}_${static_k}"
      mkdir -p "${test_run_dir}"
      render_run_benchmark "${variant}" "${static_k}" "${concurrency}" "${test_run_dir}" >"${test_run_dir}/run_benchmark.sh"
      chmod 755 "${test_run_dir}/run_benchmark.sh"
      render_sbatch "${variant}" "${method}" "${concurrency}" "${test_run_dir}" >"${test_run_dir}/submit.sbatch"
      sbatch --test-only "${test_run_dir}/submit.sbatch"
      continue
    fi
    mkdir -p "${run_dir}"
    render_run_benchmark "${variant}" "${static_k}" "${concurrency}" "${run_dir}" >"${run_dir}/run_benchmark.sh"
    chmod 755 "${run_dir}/run_benchmark.sh"
    render_sbatch "${variant}" "${method}" "${concurrency}" "${run_dir}" >"${run_dir}/submit.sbatch"
    sbatch_args=()
    if [[ -n "${DEPENDENCY}" ]]; then
      sbatch_args+=("--dependency=${DEPENDENCY}")
    fi
    if ((${#sbatch_args[@]})); then
      job_id="$(sbatch --parsable "${sbatch_args[@]}" "${run_dir}/submit.sbatch")"
    else
      job_id="$(sbatch --parsable "${run_dir}/submit.sbatch")"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${job_id}" "overlay" "${method}" "${variant}" "${concurrency}" "${static_k}" "${run_dir}" | tee -a "${MANIFEST}"
  done
done

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" ]]; then
  echo "manifest=${MANIFEST}"
  echo "matrix_root=${MATRIX_ROOT}"
fi
