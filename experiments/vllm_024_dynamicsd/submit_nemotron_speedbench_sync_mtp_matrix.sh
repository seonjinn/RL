#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX_FILE="${MATRIX_FILE:-${SCRIPT_DIR}/model_method_matrix.json}"

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
RAY_SITE="${RAY_SITE:-${LUSTRE_ROOT}/vllm024-dynamicsd/python-sites/ray-2.55.1-py312}"
MODELS="${MODELS:-ultra super}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_nemotron_speedbench_sync_mtp}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/nemotron-speedbench-sync/${RUN_ID}}"
PREPARED_RUN_ROOT="${PREPARED_RUN_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/speedbench/speedbench-487aa718-43fee0cd}"
PREPARED_ROOT="${PREPARED_ROOT:-${PREPARED_RUN_ROOT}/prepared/speed}"
PREPARED_MANIFEST="${PREPARED_MANIFEST:-${PREPARED_RUN_ROOT}/prepared_manifest.json}"
PREPARED_CHECKSUMS="${PREPARED_CHECKSUMS:-${PREPARED_RUN_ROOT}/resolved_parquet.sha256}"
DATASET_CONFIG="${DATASET_CONFIG:-throughput_1k}"
REQUEST_PROFILES="${REQUEST_PROFILES:-32k 64k}"
VARIANTS="${VARIANTS:-baseline mtp_static}"
SMOKE="${SMOKE:-true}"
CALIBRATION_ARTIFACT_ROOT="${CALIBRATION_ARTIFACT_ROOT:-}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-1234}"
if [[ "${SMOKE}" == "true" ]]; then
  SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
  ROLLOUT_BATCHES="${ROLLOUT_BATCHES:-1}"
  ACTIVE_CONCURRENCY="${ACTIVE_CONCURRENCY:-1}"
else
  SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
  ROLLOUT_BATCHES="${ROLLOUT_BATCHES:-3}"
  ACTIVE_CONCURRENCY="${ACTIVE_CONCURRENCY:-48}"
fi
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
MAMBA_BACKEND="${MAMBA_BACKEND:-flashinfer}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_trtllm}"
RUNTIME_IMAGE_SHA256="${RUNTIME_IMAGE_SHA256:-}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

require_safe_identifier "ACCOUNT" "${ACCOUNT}"
require_safe_identifier "PARTITION" "${PARTITION}"
require_safe_time_limit "TIME_LIMIT" "${TIME_LIMIT}"
require_safe_dependency "${DEPENDENCY}"

MANIFEST="${RESULT_ROOT}/jobs.tsv"
TEMP_MANIFEST_ROOT=""

variant_requested() {
  [[ " ${VARIANTS} " == *" $1 "* ]]
}

record_manifest_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >>"${MANIFEST}"
}

cleanup_temp_manifest() {
  if [[ -n "${TEMP_MANIFEST_ROOT}" ]]; then
    if [[ -f "${MANIFEST}" ]]; then
      cat "${MANIFEST}"
    fi
    rm -rf "${TEMP_MANIFEST_ROOT}"
  fi
}

clear_model_state() {
  unset \
    model_path \
    target_tp \
    nodes \
    segment \
    distributed_executor_backend \
    enable_expert_parallel \
    mamba_ssm_cache_dtype \
    enable_stochastic_rounding \
    mamba_philox_rounds \
    model_loader_threads \
    disable_fuse_allreduce_rms \
    gpu_memory_utilization \
    static_k \
    dynamic_schedule \
    method_eagle3_reason_code \
    method_eagle3_reason \
    method_mtp_static_status \
    method_mtp_dynamic_status
}

load_model_entry() {
  local model_key="$1"
  python3 - "$MATRIX_FILE" "$model_key" <<'PY'
import json
import shlex
import sys
from pathlib import Path

matrix = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_key = sys.argv[2]
model = next(
    item
    for item in matrix["models"]
    if item["launcher"] == "nemotron_sync_rl_mtp" and item["key"] == model_key
)


def emit(name: str, value: object) -> None:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = ""
    else:
        text = str(value)
    print(f"{name}={shlex.quote(text)}")


emit("model_path", model["target_checkpoint"])
for key, value in model["topology"].items():
    emit(key, value)
for method in matrix["method_order"]:
    entry = model["methods"][method]
    emit(f"method_{method}_status", entry["status"])
    emit(f"method_{method}_reason_code", entry.get("reason_code", ""))
    emit(f"method_{method}_reason", entry.get("reason", ""))
PY
}

load_model_state() {
  local model_key="$1"
  local loader_output=""
  clear_model_state
  if ! loader_output="$(load_model_entry "${model_key}")"; then
    clear_model_state
    echo "Failed to load Nemotron SPEED-Bench matrix entry for model=${model_key}" >&2
    return 1
  fi
  if ! eval "${loader_output}"; then
    clear_model_state
    echo "Failed to evaluate Nemotron SPEED-Bench matrix entry for model=${model_key}" >&2
    return 1
  fi
  require_positive_integer "NODES" "${nodes}"
  require_positive_integer "SEGMENT" "${segment}"
}

load_profile_state() {
  local profile_key="$1"
  context_profile="speedbench_${profile_key}"
  case "${profile_key}" in
    32k)
      profile_max_model_len=36864
      if [[ "${SMOKE}" == "true" ]]; then
        profile_request_plan_host="${SCRIPT_DIR}/profiles/speedbench_tail_32k.json"
        profile_request_plan_container="/workspace/experiment/profiles/speedbench_tail_32k.json"
      else
        profile_request_plan_host="${SCRIPT_DIR}/profiles/swe_sync_32k.json"
        profile_request_plan_container="/workspace/experiment/profiles/swe_sync_32k.json"
      fi
      ;;
    64k)
      profile_max_model_len=69632
      if [[ "${SMOKE}" == "true" ]]; then
        profile_request_plan_host="${SCRIPT_DIR}/profiles/speedbench_tail_64k.json"
        profile_request_plan_container="/workspace/experiment/profiles/speedbench_tail_64k.json"
      else
        profile_request_plan_host="${SCRIPT_DIR}/profiles/swe_sync_64k.json"
        profile_request_plan_container="/workspace/experiment/profiles/swe_sync_64k.json"
      fi
      ;;
    *)
      echo "Unsupported REQUEST_PROFILES entry: ${profile_key}" >&2
      return 1
      ;;
  esac
}

request_plan_hash() {
  python3 - "$1" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
buckets = [
    {
        "ignore_eos": bucket.get("ignore_eos", False),
        "max_tokens": bucket["max_tokens"],
        "min_tokens": bucket.get("min_tokens", bucket["max_tokens"]),
        "weight": bucket["weight"],
    }
    for bucket in payload["buckets"]
]
buckets.sort(
    key=lambda bucket: (
        bucket["max_tokens"],
        bucket["min_tokens"],
        bucket["weight"],
        bucket["ignore_eos"],
    )
)
canonical = {
    "buckets": buckets,
    "max_model_len": payload["max_model_len"],
    "name": payload["name"],
}
encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
PY
}

model_config_hash() {
  python3 - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "config.json"
if not path.is_file():
    raise SystemExit(f"missing model config for calibration gate: {path}")
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
}

runtime_digest_for_gate() {
  if [[ -n "${RUNTIME_IMAGE_SHA256}" ]]; then
    printf '%s\n' "${RUNTIME_IMAGE_SHA256}"
  elif [[ -s "${CONTAINER_IMAGE}.sha256" ]]; then
    awk '{print $1; exit}' "${CONTAINER_IMAGE}.sha256"
  elif [[ -s "${CONTAINER_IMAGE}" ]]; then
    sha256sum "${CONTAINER_IMAGE}" | awk '{print $1; exit}'
  else
    echo "runtime image SHA is required to validate a dynamic calibration artifact" >&2
    return 1
  fi
}

validated_dynamic_schedule() {
  local model_key="$1"
  local profile_key="$2"
  if [[ -z "${CALIBRATION_ARTIFACT_ROOT}" ]]; then
    echo "CALIBRATION_ARTIFACT_ROOT is required for mtp_dynamic calibration artifact validation" >&2
    return 1
  fi
  local artifact="${CALIBRATION_ARTIFACT_ROOT}/${model_key}/${profile_key}/schedule.json"
  if [[ ! -s "${artifact}" ]]; then
    echo "calibration artifact is required for ${model_key}/${profile_key}: ${artifact}" >&2
    return 1
  fi
  local model_hash plan_hash runtime_hash
  model_hash="$(model_config_hash "${model_path}")"
  plan_hash="$(request_plan_hash "${profile_request_plan_host}")"
  runtime_hash="$(runtime_digest_for_gate)"
  python3 "${SCRIPT_DIR}/summarize_speedbench_k_calibration.py" validate \
    --artifact "${artifact}" \
    --model "${model_path}" \
    --model-config-hash "${model_hash}" \
    --context-profile "${context_profile}" \
    --request-plan-hash "${plan_hash}" \
    --runtime-image-sha256 "${runtime_hash}" \
    --method mtp \
    --dataset-config "${DATASET_CONFIG}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --seed "${SEED}"
}

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
  local model_key="$2"
  local run_dir="$3"
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

runner_prefix=()
EOF
  if (( nodes > 1 )); then
    printf 'runner_prefix+=(%s)\n' "$(shell_quote "/workspace/experiment/run_multinode_ray.sh")"
  fi
  cat <<'EOF'

args=()
EOF
  emit_arg_pair "--cohort" "overlay"
  emit_arg_pair "--model" "${model_path}"
  emit_arg_pair "--draft-model" ""
  emit_arg_pair "--mode" "${variant}"
  emit_arg_pair "--static-k" "${static_k}"
  emit_arg_pair "--dynamic-schedule" "${effective_dynamic_schedule}"
  emit_arg_pair "--tensor-parallel-size" "${target_tp}"
  emit_arg_pair "--pipeline-parallel-size" "1"
  emit_arg_pair "--kv-cache-dtype" "${KV_CACHE_DTYPE}"
  emit_arg_pair "--gpu-memory-utilization" "${gpu_memory_utilization}"
  emit_arg_pair "--max-model-len" "${profile_max_model_len}"
  emit_arg_pair "--active-concurrency" "${ACTIVE_CONCURRENCY}"
  emit_arg_pair "--samples-per-prompt" "${SAMPLES_PER_PROMPT}"
  emit_arg_pair "--rollout-batches" "${ROLLOUT_BATCHES}"
  emit_arg_pair "--temperature" "${TEMPERATURE}"
  emit_arg_pair "--top-p" "${TOP_P}"
  emit_arg_pair "--seed" "${SEED}"
  emit_arg_pair "--context-profile" "${context_profile}"
  emit_arg_pair "--cudagraph-mode" "${CUDAGRAPH_MODE}"
  emit_arg_pair "--prepared-root" "${PREPARED_ROOT}"
  emit_arg_pair "--prepared-manifest" "${PREPARED_MANIFEST}"
  emit_arg_pair "--prepared-checksums" "${PREPARED_CHECKSUMS}"
  emit_arg_pair "--dataset-config" "${DATASET_CONFIG}"
  emit_arg_pair "--request-plan" "${profile_request_plan_container}"
  emit_arg_flag "--request-plan-exact-work"
  if [[ -n "${distributed_executor_backend}" ]]; then
    emit_arg_pair "--distributed-executor-backend" "${distributed_executor_backend}"
  fi
  if [[ "${enable_expert_parallel}" == "true" ]]; then
    emit_arg_flag "--enable-expert-parallel"
  fi
  if [[ -n "${mamba_ssm_cache_dtype}" ]]; then
    emit_arg_pair "--mamba-ssm-cache-dtype" "${mamba_ssm_cache_dtype}"
    emit_arg_pair "--mamba-backend" "${MAMBA_BACKEND}"
  fi
  if [[ "${enable_stochastic_rounding}" == "true" ]]; then
    emit_arg_flag "--enable-mamba-cache-stochastic-rounding"
  fi
  if [[ -n "${mamba_philox_rounds}" ]]; then
    emit_arg_pair "--mamba-cache-philox-rounds" "${mamba_philox_rounds}"
  fi
  if [[ -n "${model_loader_threads}" ]] && (( model_loader_threads > 0 )); then
    emit_arg_pair "--model-loader-num-threads" "${model_loader_threads}"
  fi
  if [[ "${disable_fuse_allreduce_rms}" == "true" ]]; then
    emit_arg_flag "--disable-fuse-allreduce-rms"
  fi
  if [[ -n "${MOE_BACKEND}" ]]; then
    emit_arg_pair "--moe-backend" "${MOE_BACKEND}"
  fi
  cat <<'EOF'
args+=(--runtime-image-sha256 "${runtime_image_sha256}")
EOF
  emit_arg_pair "--output" "${run_dir}/result.json"
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
  local model_key="$2"
  local run_dir="$3"
  local method="baseline"
  local container_pythonpath=""
  local container_image_q
  local container_image_sha_q
  local runtime_image_q
  local ray_sync_dir_q
  if [[ "${variant}" == "mtp_static" || "${variant}" == "mtp_dynamic" ]]; then
    method="mtp"
  fi
  if (( nodes > 1 )); then
    container_pythonpath="${RAY_SITE}"
  fi
  container_image_q="$(shell_quote "${CONTAINER_IMAGE}")"
  container_image_sha_q="$(shell_quote "${CONTAINER_IMAGE}.sha256")"
  runtime_image_q="$(shell_quote "${RUNTIME_IMAGE_SHA256}")"
  ray_sync_dir_q="$(shell_quote "${run_dir}/ray-sync")"
  local container_mounts="/lustre:/lustre,${SCRIPT_DIR}:/workspace/experiment"
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --segment=${segment}
#SBATCH --time=${TIME_LIMIT}

set -euo pipefail
runtime_image_sha256="\$(if [[ -n ${runtime_image_q} ]]; then printf '%s\n' ${runtime_image_q}; elif [[ -s ${container_image_sha_q} ]]; then awk '{print \$1; exit}' ${container_image_sha_q}; else sha256sum ${container_image_q} | awk '{print \$1; exit}'; fi)"
export BENCH_RUNTIME_IMAGE_SHA256="\${runtime_image_sha256}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_USAGE_STATS=1
export HF_HOME=$(shell_quote "${HF_HOME}")
export PYTHONPATH=$(shell_quote "${container_pythonpath}")
echo 'cohort=overlay'
echo 'model_key=${model_key}'
echo 'variant=${variant}'
echo 'method=${method}'
echo 'active_concurrency=${ACTIVE_CONCURRENCY}'
echo 'target_tp=${target_tp}'
if [[ '${variant}' == 'mtp_dynamic' ]]; then
  echo 'num_speculative_tokens_per_batch_size=${effective_dynamic_schedule}'
fi
if (( ${nodes} > 1 )); then
  export HEAD_NODE="\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)"
  export HEAD_IP="\$(srun --nodes=1 --ntasks=1 --nodelist="\${HEAD_NODE}" hostname -I | awk '{print \$1}')"
  export RAY_PORT="\$((20000 + SLURM_JOB_ID % 10000))"
  export RAY_SYNC_DIR=${ray_sync_dir_q}
  export GPUS_PER_NODE=4
  rm -rf "\${RAY_SYNC_DIR}"
fi
srun --nodes=${nodes} --ntasks=${nodes} --ntasks-per-node=1 \\
  --container-image=${container_image_q} \\
  --container-mounts=$(shell_quote "${container_mounts}") \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
  $(shell_quote "${run_dir}/run_benchmark.sh") \\
  2>&1 | tee $(shell_quote "${run_dir}/benchmark.log")
EOF
}

render_planned_variant() {
  local marker="$1"
  local variant="$2"
  local model_key="$3"
  local run_dir="$4"
  echo "${marker} speedbench_overlay=${variant}"
  echo "${marker} model_key=${model_key} cohort=overlay active_concurrency=${ACTIVE_CONCURRENCY}"
  echo "# BEGIN run_benchmark.sh ${model_key}-${variant}"
  render_run_benchmark "${variant}" "${model_key}" "${run_dir}"
  echo "# END run_benchmark.sh ${model_key}-${variant}"
  echo "# BEGIN submit.sbatch ${model_key}-${variant}"
  render_sbatch "${variant}" "${model_key}" "${run_dir}"
  echo "# END submit.sbatch ${model_key}-${variant}"
}

TEST_ONLY_ROOT=""
cleanup_test_only_root() {
  if [[ -n "${TEST_ONLY_ROOT}" ]]; then
    rm -rf "${TEST_ONLY_ROOT}"
  fi
}

cleanup_all() {
  cleanup_test_only_root
  cleanup_temp_manifest
}

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" == "true" || "${TEST_ONLY}" == "true" ]]; then
  TEMP_MANIFEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/speedbench-nemotron-matrix.XXXXXX")"
  MANIFEST="${TEMP_MANIFEST_ROOT}/jobs.tsv"
  trap cleanup_all EXIT
else
  mkdir -p "${RESULT_ROOT}"
fi
printf 'status\tmodel_key\tprofile_key\tmethod\tvariant\trun_dir\treason_code\treason\n' >"${MANIFEST}"

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" == "true" ]]; then
  TEST_ONLY_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/speedbench-nemotron-test-only.XXXXXX")"
fi

for model_key in ${MODELS}; do
  load_model_state "${model_key}"
  for profile_key in ${REQUEST_PROFILES}; do
    load_profile_state "${profile_key}"
    record_manifest_row \
      "UNSUPPORTED" \
      "${model_key}" \
      "${profile_key}" \
      "eagle3" \
      "-" \
      "-" \
      "${method_eagle3_reason_code}" \
      "${method_eagle3_reason}"
    supported_variants=()
    if variant_requested baseline; then
      supported_variants+=(baseline)
    fi
    if [[ "${method_mtp_static_status}" == "supported" ]] && variant_requested mtp_static; then
      supported_variants+=(mtp_static)
    fi
    effective_dynamic_schedule="${dynamic_schedule}"
    if [[ "${SMOKE}" != "true" && "${method_mtp_dynamic_status}" == "supported" ]] && variant_requested mtp_dynamic; then
      effective_dynamic_schedule="$(validated_dynamic_schedule "${model_key}" "${profile_key}")"
      supported_variants+=(mtp_dynamic)
    fi
    for variant in "${supported_variants[@]}"; do
      method_key="${variant}"
      record_manifest_row \
        "SUPPORTED" \
        "${model_key}" \
        "${profile_key}" \
        "${method_key}" \
        "${variant}" \
        "${RESULT_ROOT}/${model_key}/${profile_key}/${variant}" \
        "-" \
        "-"
    done
    for variant in "${supported_variants[@]}"; do
      run_dir="${RESULT_ROOT}/${model_key}/${profile_key}/${variant}"
      sbatch_args=(
        "--job-name=coreai_dlalgo_llm-speedbench-${model_key}-${variant}"
        "--output=${run_dir}/slurm-%j.out"
      )
      if [[ -n "${DEPENDENCY}" ]]; then
        sbatch_args+=("--dependency=${DEPENDENCY}")
      fi
      if [[ "${DRY_RUN}" == "true" ]]; then
        render_planned_variant "[DRY-RUN]" "${variant}" "${model_key}" "${run_dir}"
        continue
      fi
      if [[ "${TEST_ONLY}" == "true" ]]; then
        echo "[TEST-ONLY] speedbench_overlay=${variant}"
        test_run_dir="${TEST_ONLY_ROOT}/${model_key}/${profile_key}/${variant}"
        mkdir -p "${test_run_dir}"
        render_run_benchmark "${variant}" "${model_key}" "${test_run_dir}" >"${test_run_dir}/run_benchmark.sh"
        chmod 755 "${test_run_dir}/run_benchmark.sh"
        render_sbatch "${variant}" "${model_key}" "${test_run_dir}" >"${test_run_dir}/submit.sbatch"
        sbatch --test-only "${sbatch_args[@]}" "${test_run_dir}/submit.sbatch"
        continue
      fi
      mkdir -p "${run_dir}"
      render_run_benchmark "${variant}" "${model_key}" "${run_dir}" >"${run_dir}/run_benchmark.sh"
      chmod 755 "${run_dir}/run_benchmark.sh"
      render_sbatch "${variant}" "${model_key}" "${run_dir}" >"${run_dir}/submit.sbatch"
      job_id="$(sbatch --parsable "${sbatch_args[@]}" "${run_dir}/submit.sbatch")"
      printf '%s\t%s\t%s\t%s\t%s\n' "${job_id}" "${model_key}" "${profile_key}" "${variant}" "${run_dir}"
    done
  done
done

if [[ "${DRY_RUN}" != "true" && "${TEST_ONLY}" != "true" ]]; then
  echo "manifest=${MANIFEST}"
fi
