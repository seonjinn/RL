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
  [[ "${value}" =~ ^[A-Za-z0-9._:-]+$ ]] || die "invalid scheduler identifier ${name}=${value}"
}

require_safe_time_limit() {
  [[ "$1" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || die "invalid scheduler identifier TIME_LIMIT=$1"
}

shell_quote() {
  printf "%q" "$1"
}

emit_arg_pair() {
  printf 'args+=(%s %s)\n' "$(shell_quote "$1")" "$(shell_quote "$2")"
}

CLUSTER="${CLUSTER:-auto}"
if [[ "${CLUSTER}" == "auto" ]]; then
  case "$(hostname)" in
    *lyris*) CLUSTER="lyris" ;;
    *ptyche*) CLUSTER="ptyche" ;;
    *) die "Set CLUSTER=lyris or CLUSTER=ptyche" ;;
  esac
fi
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
case "${CLUSTER}" in
  lyris) PARTITION="${PARTITION:-gb200}" ;;
  ptyche) PARTITION="${PARTITION:-batch}" ;;
  *) die "Unsupported CLUSTER=${CLUSTER}" ;;
esac

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
MODEL="${MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
DRAFT_MODEL="${DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
PREPARED_RUN_ROOT="${PREPARED_RUN_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/speedbench/speedbench-487aa718-43fee0cd}"
PREPARED_ROOT="${PREPARED_ROOT:-${PREPARED_RUN_ROOT}/prepared/speed}"
PREPARED_MANIFEST="${PREPARED_MANIFEST:-${PREPARED_RUN_ROOT}/prepared_manifest.json}"
PREPARED_CHECKSUMS="${PREPARED_CHECKSUMS:-${PREPARED_RUN_ROOT}/resolved_parquet.sha256}"
MODELOPT_ROOT="${MODELOPT_ROOT:-${PREPARED_RUN_ROOT}/sources/modelopt}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/speedbench-official}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_speedbench_official}"
VARIANTS="${VARIANTS:-baseline static}"
DATASET_CONFIG="${DATASET_CONFIG:-throughput_1k}"
TP="${TP:-2}"
STATIC_K="${STATIC_K:-5}"
ACTIVE_CONCURRENCY="${ACTIVE_CONCURRENCY:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
RUNTIME_IMAGE_SHA256="${RUNTIME_IMAGE_SHA256:-}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

require_safe_identifier "ACCOUNT" "${ACCOUNT}"
require_safe_identifier "PARTITION" "${PARTITION}"
require_safe_identifier "RUN_ID" "${RUN_ID}"
require_safe_time_limit "${TIME_LIMIT}"

MATRIX_ROOT="${RESULT_ROOT}/${RUN_ID}"
MANIFEST="${MATRIX_ROOT}/jobs.tsv"

render_run_benchmark() {
  local variant="$1"
  local run_dir="$2"
  local image_q image_sha_q runtime_q
  image_q="$(shell_quote "${CONTAINER_IMAGE}")"
  image_sha_q="$(shell_quote "${CONTAINER_IMAGE}.sha256")"
  runtime_q="$(shell_quote "${RUNTIME_IMAGE_SHA256}")"
  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

benchmark_python="\${BENCHMARK_PYTHON:-python3}"
benchmark_script="\${BENCHMARK_SCRIPT:-/workspace/experiment/benchmark_speedbench_sync_rollout.py}"
runtime_image_sha256="\$(if [[ -n ${runtime_q} ]]; then printf '%s\n' ${runtime_q}; elif [[ -s ${image_sha_q} ]]; then awk '{print \$1; exit}' ${image_sha_q}; else sha256sum ${image_q} | awk '{print \$1; exit}'; fi)"
if [[ -z "\${runtime_image_sha256}" || "\${runtime_image_sha256}" == "unknown" ]]; then
  echo "runtime_image_sha256 must resolve to a real digest" >&2
  exit 2
fi

args=()
EOF
  emit_arg_pair "--cohort" "official"
  emit_arg_pair "--model" "${MODEL}"
  emit_arg_pair "--draft-model" "${DRAFT_MODEL}"
  emit_arg_pair "--mode" "${variant}"
  emit_arg_pair "--static-k" "${STATIC_K}"
  emit_arg_pair "--tensor-parallel-size" "${TP}"
  emit_arg_pair "--pipeline-parallel-size" "1"
  emit_arg_pair "--active-concurrency" "${ACTIVE_CONCURRENCY}"
  emit_arg_pair "--max-model-len" "${MAX_MODEL_LEN}"
  emit_arg_pair "--max-new-tokens" "${MAX_NEW_TOKENS}"
  emit_arg_pair "--dataset-config" "${DATASET_CONFIG}"
  emit_arg_pair "--prepared-root" "${PREPARED_ROOT}"
  emit_arg_pair "--prepared-manifest" "${PREPARED_MANIFEST}"
  emit_arg_pair "--prepared-checksums" "${PREPARED_CHECKSUMS}"
  emit_arg_pair "--modelopt-root" "${MODELOPT_ROOT}"
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
  local run_dir="$2"
  local image_q mounts_q run_script_q log_q
  image_q="$(shell_quote "${CONTAINER_IMAGE}")"
  mounts_q="$(shell_quote "/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment")"
  run_script_q="$(shell_quote "${run_dir}/run_benchmark.sh")"
  log_q="$(shell_quote "${run_dir}/benchmark.log")"
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

set -euo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
export HF_HOME=$(shell_quote "${HF_HOME}")
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --container-image=${image_q} \
  --container-mounts=${mounts_q} \
  --no-container-mount-home \
  --container-remap-root \
  --mpi=pmix \
  ${run_script_q} 2>&1 | tee ${log_q}
EOF
}

render_planned_job() {
  local marker="$1"
  local variant="$2"
  local run_dir="$3"
  echo "${marker} speedbench_official=${variant}"
  echo "# BEGIN run_benchmark.sh official-${variant}"
  render_run_benchmark "${variant}" "${run_dir}"
  echo "# END run_benchmark.sh official-${variant}"
  echo "# BEGIN submit.sbatch official-${variant}"
  render_sbatch "${variant}" "${run_dir}"
  echo "# END submit.sbatch official-${variant}"
}

TEST_ONLY_ROOT=""
cleanup() {
  [[ -z "${TEST_ONLY_ROOT}" ]] || rm -rf "${TEST_ONLY_ROOT}"
}
trap cleanup EXIT

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi
if [[ "${TEST_ONLY}" == "true" ]]; then
  TEST_ONLY_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/speedbench-official-test.XXXXXX")"
fi
if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" ]]; then
  mkdir -p "${MATRIX_ROOT}"
  printf 'job_id\tcohort\tvariant\trun_dir\n' >"${MANIFEST}"
fi

for variant in ${VARIANTS}; do
  case "${variant}" in
    baseline|static) ;;
    *) die "official SPEED-Bench supports only baseline and static: ${variant}" ;;
  esac
  run_dir="${MATRIX_ROOT}/${variant}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    render_planned_job "[DRY-RUN]" "${variant}" "${run_dir}"
    continue
  fi
  render_dir="${run_dir}"
  if [[ "${TEST_ONLY}" == "true" ]]; then
    render_dir="${TEST_ONLY_ROOT}/${variant}"
  fi
  mkdir -p "${render_dir}"
  render_run_benchmark "${variant}" "${render_dir}" >"${render_dir}/run_benchmark.sh"
  chmod 755 "${render_dir}/run_benchmark.sh"
  render_sbatch "${variant}" "${render_dir}" >"${render_dir}/submit.sbatch"
  sbatch_args=("--job-name=coreai_dlalgo_llm-speedbench-official-${variant}" "--output=${render_dir}/slurm-%j.out")
  [[ -z "${DEPENDENCY}" ]] || sbatch_args+=("--dependency=${DEPENDENCY}")
  if [[ "${TEST_ONLY}" == "true" ]]; then
    sbatch --test-only "${sbatch_args[@]}" "${render_dir}/submit.sbatch"
    continue
  fi
  job_id="$(sbatch --parsable "${sbatch_args[@]}" "${render_dir}/submit.sbatch")"
  printf '%s\tofficial\t%s\t%s\n' "${job_id}" "${variant}" "${run_dir}" | tee -a "${MANIFEST}"
done

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" ]]; then
  echo "manifest=${MANIFEST}"
fi
