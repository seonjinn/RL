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

shell_quote() {
  printf "%q" "$1"
}

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
MODEL="${MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
DRAFT_MODEL="${DRAFT_MODEL-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/speedbench-k-calibration}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_speedbench_k_calibration}"
PREPARED_RUN_ROOT="${PREPARED_RUN_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/speedbench/speedbench-487aa718-43fee0cd}"
PREPARED_ROOT="${PREPARED_ROOT:-${PREPARED_RUN_ROOT}/prepared/speed}"
PREPARED_MANIFEST="${PREPARED_MANIFEST:-${PREPARED_RUN_ROOT}/prepared_manifest.json}"
PREPARED_CHECKSUMS="${PREPARED_CHECKSUMS:-${PREPARED_RUN_ROOT}/resolved_parquet.sha256}"
DATASET_CONFIG="${DATASET_CONFIG:-throughput_1k}"
REQUEST_PLAN="${REQUEST_PLAN:-${SCRIPT_DIR}/profiles/swe_sync_32k.json}"
REQUEST_PLAN_IN_CONTAINER="${REQUEST_PLAN_IN_CONTAINER:-/workspace/experiment/profiles/swe_sync_32k.json}"
CONTEXT_PROFILE="${CONTEXT_PROFILE:-speedbench_32k}"
CONCURRENCIES="${CONCURRENCIES:-1 8 32 64}"
K_VALUES="${K_VALUES:-1 2 3 4 5}"
REPEATS="${REPEATS:-3}"
STATIC_VARIANT="${STATIC_VARIANT:-static}"
METHOD="${METHOD:-eagle3}"
DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-1:16:5,17:32:4,33:64:3,65:128:1,129:512:0}"
TP="${TP:-2}"
PP="${PP:-1}"
NODES="${NODES:-1}"
SEGMENT="${SEGMENT:-${NODES}}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
TEMPERATURE="${TEMPERATURE:-1.0}"
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

require_safe_identifier "ACCOUNT" "${ACCOUNT}"
require_safe_identifier "PARTITION" "${PARTITION}"
require_safe_time_limit "TIME_LIMIT" "${TIME_LIMIT}"
require_positive_integer "NODES" "${NODES}"
require_positive_integer "SEGMENT" "${SEGMENT}"
require_safe_dependency "${DEPENDENCY}"
for concurrency in ${CONCURRENCIES}; do
  require_positive_integer "ACTIVE_CONCURRENCY" "${concurrency}"
done
for k_value in ${K_VALUES}; do
  require_positive_integer "STATIC_K" "${k_value}"
done

if [[ "${REPEATS}" != "3" ]]; then
  echo "REPEATS must be exactly 3 for calibration artifacts" >&2
  exit 2
fi
case "${STATIC_VARIANT}:${METHOD}" in
  static:eagle3|mtp_static:mtp) ;;
  *)
    echo "STATIC_VARIANT/METHOD must be static/eagle3 or mtp_static/mtp" >&2
    exit 2
    ;;
esac

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
  local repeat="$4"
  local run_dir="$5"
  local container_image_q
  local container_image_sha_q
  local runtime_image_q
  container_image_q="$(shell_quote "${CONTAINER_IMAGE}")"
  container_image_sha_q="$(shell_quote "${CONTAINER_IMAGE}.sha256")"
  runtime_image_q="$(shell_quote "${RUNTIME_IMAGE_SHA256}")"
  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

benchmark_python="\${BENCHMARK_PYTHON:-python3}"
benchmark_script="\${BENCHMARK_SCRIPT:-/workspace/experiment/benchmark_speedbench_sync_rollout.py}"
runtime_image_sha256="\$(if [[ -n ${runtime_image_q} ]]; then printf '%s\n' ${runtime_image_q}; elif [[ -n "\${BENCH_RUNTIME_IMAGE_SHA256:-}" ]]; then printf '%s\n' "\${BENCH_RUNTIME_IMAGE_SHA256}"; elif [[ -s ${container_image_sha_q} ]]; then awk '{print \$1; exit}' ${container_image_sha_q}; else sha256sum ${container_image_q} | awk '{print \$1; exit}'; fi)"
if [[ -z "\${runtime_image_sha256}" || "\${runtime_image_sha256}" == "unknown" ]]; then
  echo "runtime_image_sha256 must resolve to a real digest" >&2
  exit 2
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
  emit_arg_pair "--context-profile" "${CONTEXT_PROFILE}"
  emit_arg_pair "--calibration-repeat" "${repeat}"
  emit_arg_pair "--cudagraph-mode" "${CUDAGRAPH_MODE}"
  emit_arg_pair "--prepared-root" "${PREPARED_ROOT}"
  emit_arg_pair "--prepared-manifest" "${PREPARED_MANIFEST}"
  emit_arg_pair "--prepared-checksums" "${PREPARED_CHECKSUMS}"
  emit_arg_pair "--dataset-config" "${DATASET_CONFIG}"
  emit_arg_pair "--request-plan" "${REQUEST_PLAN_IN_CONTAINER}"
  emit_arg_flag "--request-plan-exact-work"
  cat <<'EOF'
args+=(--runtime-image-sha256 "${runtime_image_sha256}")
EOF
  emit_arg_pair "--output" "${run_dir}/result.json"
  cat <<'EOF'

"${benchmark_python}" "${benchmark_script}" "${args[@]}"
EOF
}

render_sbatch() {
  local variant="$1"
  local method="$2"
  local concurrency="$3"
  local repeat="$4"
  local run_dir="$5"
  local container_image_q
  local container_image_sha_q
  local runtime_image_q
  container_image_q="$(shell_quote "${CONTAINER_IMAGE}")"
  container_image_sha_q="$(shell_quote "${CONTAINER_IMAGE}.sha256")"
  runtime_image_q="$(shell_quote "${RUNTIME_IMAGE_SHA256}")"
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

set -euo pipefail
runtime_image_sha256="\$(if [[ -n ${runtime_image_q} ]]; then printf '%s\n' ${runtime_image_q}; elif [[ -s ${container_image_sha_q} ]]; then awk '{print \$1; exit}' ${container_image_sha_q}; else sha256sum ${container_image_q} | awk '{print \$1; exit}'; fi)"
export BENCH_RUNTIME_IMAGE_SHA256="\${runtime_image_sha256}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
export HF_HOME=$(shell_quote "${HF_HOME}")
echo 'cohort=overlay'
echo 'method=${method}'
echo 'variant=${variant}'
echo 'active_concurrency=${concurrency}'
echo 'calibration_repeat=${repeat}'
echo 'cudagraph_mode=${CUDAGRAPH_MODE}'
srun --nodes=${NODES} --ntasks=${NODES} --ntasks-per-node=1 \\
  --container-image=${container_image_q} \\
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
  local repeat="$6"
  local run_dir="$7"
  echo "${marker} speedbench_overlay=${variant}"
  echo "${marker} cohort=overlay method=${method} active_concurrency=${concurrency} static_k=${static_k} repeat=${repeat}"
  echo "${marker} planned_run_script=${run_dir}/run_benchmark.sh"
  echo "# BEGIN run_benchmark.sh ${variant}-c${concurrency}-k${static_k}-r${repeat}"
  render_run_benchmark "${variant}" "${static_k}" "${concurrency}" "${repeat}" "${run_dir}"
  echo "# END run_benchmark.sh ${variant}-c${concurrency}-k${static_k}-r${repeat}"
  echo "# BEGIN submit.sbatch ${variant}-c${concurrency}-k${static_k}-r${repeat}"
  render_sbatch "${variant}" "${method}" "${concurrency}" "${repeat}" "${run_dir}"
  echo "# END submit.sbatch ${variant}-c${concurrency}-k${static_k}-r${repeat}"
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
  printf 'job_id\tcohort\tmethod\tvariant\tactive_concurrency\tstatic_k\trepeat\trun_dir\n' >"${MANIFEST}"
fi

for concurrency in ${CONCURRENCIES}; do
  jobs=("baseline:baseline:0")
  for k_value in ${K_VALUES}; do
    jobs+=("${STATIC_VARIANT}:${METHOD}:${k_value}")
  done
  for job in "${jobs[@]}"; do
    IFS=: read -r variant method static_k <<<"${job}"
    for repeat in $(seq 1 "${REPEATS}"); do
      run_dir="${MATRIX_ROOT}/c${concurrency}/${variant}/repeat${repeat}"
      if [[ "${variant}" != "baseline" ]]; then
        run_dir="${MATRIX_ROOT}/c${concurrency}/${variant}_k${static_k}/repeat${repeat}"
      fi
      if [[ "${DRY_RUN}" == "true" ]]; then
        render_planned_job "[DRY-RUN]" "${variant}" "${method}" "${static_k}" "${concurrency}" "${repeat}" "${run_dir}"
        continue
      fi
      sbatch_args=(
        "--job-name=coreai_dlalgo_llm-speedbench-c${concurrency}-${variant}"
        "--output=${run_dir}/slurm-%j.out"
      )
      if [[ -n "${DEPENDENCY}" ]]; then
        sbatch_args+=("--dependency=${DEPENDENCY}")
      fi
      if [[ "${TEST_ONLY}" == "true" ]]; then
        echo "[TEST-ONLY] speedbench_overlay=${variant} repeat=${repeat}"
        test_run_dir="${TEST_ONLY_ROOT}/c${concurrency}/${variant}_${static_k}/repeat${repeat}"
        mkdir -p "${test_run_dir}"
        render_run_benchmark "${variant}" "${static_k}" "${concurrency}" "${repeat}" "${test_run_dir}" >"${test_run_dir}/run_benchmark.sh"
        chmod 755 "${test_run_dir}/run_benchmark.sh"
        render_sbatch "${variant}" "${method}" "${concurrency}" "${repeat}" "${test_run_dir}" >"${test_run_dir}/submit.sbatch"
        sbatch --test-only "${sbatch_args[@]}" "${test_run_dir}/submit.sbatch"
        continue
      fi
      mkdir -p "${run_dir}"
      render_run_benchmark "${variant}" "${static_k}" "${concurrency}" "${repeat}" "${run_dir}" >"${run_dir}/run_benchmark.sh"
      chmod 755 "${run_dir}/run_benchmark.sh"
      render_sbatch "${variant}" "${method}" "${concurrency}" "${repeat}" "${run_dir}" >"${run_dir}/submit.sbatch"
      job_id="$(sbatch --parsable "${sbatch_args[@]}" "${run_dir}/submit.sbatch")"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${job_id}" "overlay" "${method}" "${variant}" "${concurrency}" "${static_k}" "${repeat}" "${run_dir}" | tee -a "${MANIFEST}"
    done
  done
done

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" ]]; then
  echo "manifest=${MANIFEST}"
  echo "matrix_root=${MATRIX_ROOT}"
fi
