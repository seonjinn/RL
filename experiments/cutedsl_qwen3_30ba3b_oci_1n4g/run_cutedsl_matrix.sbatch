#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --job-name=cutedsl-ab-qwen3-30ba3b-1n4g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

RECIPE="${CUTEDSL_BENCHMARK_RECIPE:-examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl.yaml}"
EXISTING_RAY="${CUTEDSL_BENCHMARK_EXISTING_RAY:-0}"
BENCHMARK_NUM_NODES="${CUTEDSL_BENCHMARK_NUM_NODES:-1}"
BENCHMARK_GPUS_PER_NODE="${CUTEDSL_BENCHMARK_GPUS_PER_NODE:-4}"
NEMO2606_FULL_CG_ENABLED="${NEMO2606_FULL_CG_ENABLED:-0}"
NEMO2606_A2A_ENABLED="${NEMO2606_A2A_ENABLED:-0}"
FEATURE_CONTEXT="${NEMO2606_FACTORIAL_CONTEXT:-g0a0}"
readonly RECIPE EXISTING_RAY BENCHMARK_NUM_NODES BENCHMARK_GPUS_PER_NODE
readonly NEMO2606_FULL_CG_ENABLED NEMO2606_A2A_ENABLED FEATURE_CONTEXT
readonly CUTEDSL_SELECTOR_PATH="policy.megatron_cfg.env_vars.NVTE_CUTEDSL_FUSED_GROUPED_MLP"
readonly EXPECTED_ON_SELECTOR="1"
readonly OFF_OVERRIDE="${CUTEDSL_SELECTOR_PATH}=\"0\""
export RECIPE CUTEDSL_SELECTOR_PATH EXPECTED_ON_SELECTOR OFF_OVERRIDE

readonly SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:?submit this benchmark with sbatch from the repository checkout}"
REPO_ROOT=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)
readonly REPO_ROOT
readonly EXPERIMENT_DIR="${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
source "${EXPERIMENT_DIR}/lib/cluster_profile.sh"
source "${EXPERIMENT_DIR}/lib/events.sh"
load_cutedsl_cluster_profile
readonly IMAGE="${CUTEDSL_IMAGE}"
BENCHMARK_SEGMENT_SIZE="${CUTEDSL_BENCHMARK_SEGMENT_SIZE:-${CUTEDSL_SEGMENT:-1}}"
readonly RUN_ID="${SLURM_JOB_ID:?submit this benchmark with sbatch}${SLURM_RESTART_COUNT:+-r${SLURM_RESTART_COUNT}}"
RESULT_ROOT="${CUTEDSL_BENCHMARK_RESULT_ROOT:-${EXPERIMENT_DIR}/results}"
readonly RESULT_ROOT
readonly RESULT_DIR="${RESULT_ROOT}/${RUN_ID}"
if [[ "${EXISTING_RAY}" == "1" ]]; then
    CONTAINER_REPO_ROOT="${REPO_ROOT}"
    CONTAINER_RESULT_DIR="${RESULT_DIR}"
    runtime_root="${CUTEDSL_BENCHMARK_RUNTIME_ROOT:?existing-Ray mode requires CUTEDSL_BENCHMARK_RUNTIME_ROOT}"
    HOST_RUNTIME_DIR="${runtime_root}/${RUN_ID}"
    CONTAINER_RUNTIME_DIR="${HOST_RUNTIME_DIR}"
else
    CONTAINER_REPO_ROOT="/workspace/nemo-rl"
    CONTAINER_RESULT_DIR="/results"
    HOST_RUNTIME_DIR="/tmp/${USER}/nemo-2606-cutedsl-benchmark/${RUN_ID}"
    CONTAINER_RUNTIME_DIR="/runtime"
fi
readonly CONTAINER_REPO_ROOT CONTAINER_RESULT_DIR HOST_RUNTIME_DIR
readonly CONTAINER_RUNTIME_DIR
export CONTAINER_REPO_ROOT CONTAINER_RESULT_DIR CONTAINER_RUNTIME_DIR

WARMUP_UPDATES="${CUTEDSL_BENCHMARK_WARMUP_UPDATES:-5}"
MEASURED_UPDATES="${CUTEDSL_BENCHMARK_MEASURED_UPDATES:-20}"
TIMING_ORDER="${CUTEDSL_BENCHMARK_ORDER:-on,off}"
PROFILE_ENABLED="${CUTEDSL_BENCHMARK_PROFILE:-1}"
FUNCTIONAL_GATE="${NEMO2606_FUNCTIONAL_GATE:-0}"
FUNCTIONAL_UPDATES="${NEMO2606_FUNCTIONAL_UPDATES:-3}"
REPLICATE_INDEX="${CUTEDSL_BENCHMARK_REPLICATE:-0}"
SUBMISSION_GROUP="${CUTEDSL_BENCHMARK_SUBMISSION_GROUP:-direct}"
readonly REPLICATE_INDEX SUBMISSION_GROUP

# CUTEDSL_RUNTIME_TMPDIR_START
mkdir -p "${RESULT_DIR}" "${HOST_RUNTIME_DIR}/tmp"
export TMPDIR="${HOST_RUNTIME_DIR}/tmp"
# CUTEDSL_RUNTIME_TMPDIR_END
export CUTEDSL_EVENT_CLUSTER="${CUTEDSL_PROFILE_NAME}"
export CUTEDSL_EVENT_JOB_ID="${SLURM_JOB_ID}"
cutedsl_events_init "${RESULT_DIR}"
exec > >(tee "${RESULT_DIR}/slurm.out") 2>&1

on_exit() {
    local exit_code=$?
    if ! cutedsl_begin_finalization; then
        return "${exit_code}"
    fi
    set +e
    cutedsl_finalize_run "${exit_code}" "Benchmark payload finished" \
        "${EXPERIMENT_DIR}/render_cutedsl_report.py"
    local final_exit_code=$?
    rm -rf --one-file-system -- "${HOST_RUNTIME_DIR}"
    exit "${final_exit_code}"
}
trap on_exit EXIT
cutedsl_install_signal_traps

cutedsl_write_event preflight start '' "Validate benchmark controls, source, and image" slurm.out

if ! [[ "${FUNCTIONAL_GATE}" == "0" || "${FUNCTIONAL_GATE}" == "1" ]]; then
    echo "[ERROR] NEMO2606_FUNCTIONAL_GATE must be 0 or 1." >&2
    exit 1
fi
if [[ "${PROFILE_ENABLED}" != "0" && "${PROFILE_ENABLED}" != "1" ]]; then
    echo "[ERROR] CUTEDSL_BENCHMARK_PROFILE must be 0 or 1." >&2
    exit 1
fi
if [[ ! "${REPLICATE_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] CUTEDSL_BENCHMARK_REPLICATE must be a non-negative integer." >&2
    exit 1
fi
for boolean_value in "${EXISTING_RAY}" "${NEMO2606_FULL_CG_ENABLED}" \
    "${NEMO2606_A2A_ENABLED}"; do
    if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
        echo "[ERROR] Existing-Ray and feature selectors must be 0 or 1." >&2
        exit 1
    fi
done
case "${FEATURE_CONTEXT}:${NEMO2606_FULL_CG_ENABLED}:${NEMO2606_A2A_ENABLED}" in
    g0a0:0:0|g1a0:1:0|g0a1:0:1|g1a1:1:1) ;;
    *)
        echo "[ERROR] Feature context does not match full-CG/A2A selectors." >&2
        exit 1
        ;;
esac
if [[ ! "${BENCHMARK_NUM_NODES}" =~ ^[0-9]+$ ]] || \
    [[ ! "${BENCHMARK_GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || \
    [[ ! "${BENCHMARK_SEGMENT_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Benchmark node, GPU, and segment counts must be positive integers." >&2
    exit 1
fi
if ((BENCHMARK_NUM_NODES < 1 || BENCHMARK_GPUS_PER_NODE < 1 || BENCHMARK_SEGMENT_SIZE < 1)); then
    echo "[ERROR] Benchmark node, GPU, and segment counts must be positive integers." >&2
    exit 1
fi
require_source_feature() {
    local source_path=$1
    local pattern=$2
    local feature_name=$3
    if [[ ! -r "${source_path}" ]] || ! grep -q "${pattern}" "${source_path}"; then
        echo "[ERROR] ${feature_name} was requested but its NeMo-RL implementation is absent: ${source_path} / ${pattern}" >&2
        exit 1
    fi
}

if [[ "${NEMO2606_FULL_CG_ENABLED}" == "1" ]]; then
    full_cg_source="${REPO_ROOT}/nemo_rl/models/megatron/full_cuda_graph.py"
    require_source_feature "${full_cg_source}" "build_full_cuda_graph_schedule" \
        "full-iteration CUDA Graph"
    if grep -q "colocated generation/refit is not supported" "${full_cg_source}"; then
        echo "[ERROR] full-iteration CUDA Graph currently rejects colocated generation/refit. This policy occupies all GPUs, so this context requires either colocated support or additional generation GPUs." >&2
        exit 1
    fi
fi
if [[ "${NEMO2606_A2A_ENABLED}" == "1" ]]; then
    require_source_feature "${REPO_ROOT}/nemo_rl/models/megatron/train.py" \
        "return_schedule_plan" "EP A2A overlap"
    require_source_feature "${REPO_ROOT}/nemo_rl/models/megatron/setup.py" \
        "overlap_moe_expert_parallel_comm" "EP A2A overlap"
fi

if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then
    if ! [[ "${FUNCTIONAL_UPDATES}" == "3" ]]; then
        echo "[ERROR] Functional gate requires NEMO2606_FUNCTIONAL_UPDATES=3." >&2
        exit 1
    fi
    if ! [[ "${TIMING_ORDER}" == "on" ]]; then
        echo "[ERROR] Functional gate requires CUTEDSL_BENCHMARK_ORDER=on." >&2
        exit 1
    fi
    if ! [[ "${PROFILE_ENABLED}" == "0" ]]; then
        echo "[ERROR] Functional gate requires CUTEDSL_BENCHMARK_PROFILE=0." >&2
        exit 1
    fi
    if ! [[ "${FEATURE_CONTEXT}" == "g0a0" ]]; then
        echo "[ERROR] Functional gate requires NEMO2606_FACTORIAL_CONTEXT=g0a0." >&2
        exit 1
    fi
    if ! [[ "${NEMO2606_FULL_CG_ENABLED}" == "0" ]]; then
        echo "[ERROR] Functional gate requires NEMO2606_FULL_CG_ENABLED=0." >&2
        exit 1
    fi
    if ! [[ "${NEMO2606_A2A_ENABLED}" == "0" ]]; then
        echo "[ERROR] Functional gate requires NEMO2606_A2A_ENABLED=0." >&2
        exit 1
    fi
    if [[ "${EXISTING_RAY}" != "1" ]]; then
        echo "[ERROR] Multinode functional gate requires an existing Ray cluster." >&2
        exit 1
    fi
    timing_arms=(on)
    WARMUP_UPDATES=0
    MEASURED_UPDATES=0
    TOTAL_UPDATES="${FUNCTIONAL_UPDATES}"
else
    if [[ ! "${WARMUP_UPDATES}" =~ ^[0-9]+$ ]] || ((WARMUP_UPDATES < 5)); then
        echo "[ERROR] CUTEDSL_BENCHMARK_WARMUP_UPDATES must be an integer >= 5." >&2
        exit 1
    fi
    if [[ ! "${MEASURED_UPDATES}" =~ ^[0-9]+$ ]] || ((MEASURED_UPDATES < 10)); then
        echo "[ERROR] CUTEDSL_BENCHMARK_MEASURED_UPDATES must be an integer >= 10." >&2
        exit 1
    fi
    if [[ "${NEMO2606_FULL_CG_ENABLED}" == "1" ]]; then
        if ! [[ "${TIMING_ORDER}" == "on" ]]; then
            echo "[ERROR] Full-iteration CUDA Graph requires CUTEDSL_BENCHMARK_ORDER=on because CuTeDSL-OFF is not applicable." >&2
            exit 1
        fi
        timing_arms=(on)
    else
        IFS=',' read -r -a timing_arms <<< "${TIMING_ORDER}"
        if [[ ${#timing_arms[@]} -ne 2 ]]; then
            echo "[ERROR] CUTEDSL_BENCHMARK_ORDER must contain exactly two comma-separated arms." >&2
            exit 1
        fi
        on_count=0
        off_count=0
        for arm in "${timing_arms[@]}"; do
            case "${arm}" in
                on) on_count=$((on_count + 1)) ;;
                off) off_count=$((off_count + 1)) ;;
                *)
                    echo "[ERROR] Unknown benchmark arm in order: ${arm}" >&2
                    exit 1
                    ;;
            esac
        done
        if [[ ${on_count} -ne 1 || ${off_count} -ne 1 ]]; then
            echo "[ERROR] CUTEDSL_BENCHMARK_ORDER must contain on and off exactly once." >&2
            exit 1
        fi
    fi
    TOTAL_UPDATES=$((WARMUP_UPDATES + MEASURED_UPDATES))
fi

TRAINING_GPU_COUNT=$((BENCHMARK_NUM_NODES * BENCHMARK_GPUS_PER_NODE))
TRAIN_GLOBAL_BATCH_SIZE="${CUTEDSL_BENCHMARK_TRAIN_GLOBAL_BATCH_SIZE:-$((TRAINING_GPU_COUNT * 2))}"
EXPERT_MODEL_PARALLEL_SIZE="${CUTEDSL_BENCHMARK_EXPERT_MODEL_PARALLEL_SIZE:-${TRAINING_GPU_COUNT}}"
if [[ ! "${TRAIN_GLOBAL_BATCH_SIZE}" =~ ^[0-9]+$ ]] || \
    [[ ! "${EXPERT_MODEL_PARALLEL_SIZE}" =~ ^[0-9]+$ ]] || \
    ((TRAIN_GLOBAL_BATCH_SIZE < 1 || EXPERT_MODEL_PARALLEL_SIZE < 1)) || \
    ((TRAINING_GPU_COUNT % EXPERT_MODEL_PARALLEL_SIZE != 0)); then
    echo "[ERROR] Invalid train global batch or expert model parallel size." >&2
    exit 1
fi
CUDA_GRAPH_WARMUP_STEPS=3
readonly BENCHMARK_SEGMENT_SIZE FUNCTIONAL_GATE FUNCTIONAL_UPDATES
readonly WARMUP_UPDATES MEASURED_UPDATES TIMING_ORDER PROFILE_ENABLED
readonly TOTAL_UPDATES TRAINING_GPU_COUNT TRAIN_GLOBAL_BATCH_SIZE
readonly EXPERT_MODEL_PARALLEL_SIZE CUDA_GRAPH_WARMUP_STEPS
export RUN_ID WARMUP_UPDATES MEASURED_UPDATES TIMING_ORDER TOTAL_UPDATES PROFILE_ENABLED
export REPLICATE_INDEX SUBMISSION_GROUP
export TRAINING_GPU_COUNT FEATURE_CONTEXT NEMO2606_FULL_CG_ENABLED
export NEMO2606_A2A_ENABLED
export TRAIN_GLOBAL_BATCH_SIZE EXPERT_MODEL_PARALLEL_SIZE
export BENCHMARK_NUM_NODES BENCHMARK_GPUS_PER_NODE BENCHMARK_SEGMENT_SIZE
export EXISTING_RAY CUDA_GRAPH_WARMUP_STEPS
export FUNCTIONAL_GATE FUNCTIONAL_UPDATES

if [[ "${EXISTING_RAY}" == "1" ]]; then
    RAY_LOG_ATTEMPT_ID="${SLURM_JOB_ID}"
    if [[ -n "${SLURM_RESTART_COUNT:-}" ]]; then
        RAY_LOG_ATTEMPT_ID="${SLURM_JOB_ID}-${SLURM_RESTART_COUNT}"
    fi
    RAY_CLUSTER_LOG_DIR="${BASE_LOG_DIR:?existing-Ray mode requires BASE_LOG_DIR}/${RAY_LOG_ATTEMPT_ID}-logs/ray"
    export RAY_LOG_ATTEMPT_ID RAY_CLUSTER_LOG_DIR
    ray status > "${RESULT_DIR}/ray_status.txt"
    worker_units=$(grep "worker_units" "${RESULT_DIR}/ray_status.txt" | \
        awk -F'[/. ]' '{print $4}' || echo 0)
    if [[ "${worker_units}" != "${TRAINING_GPU_COUNT}" ]]; then
        echo "[ERROR] Existing Ray cluster has ${worker_units} worker units; expected ${TRAINING_GPU_COUNT}." >&2
        exit 1
    fi
fi

cat > "${HOST_RUNTIME_DIR}/log_runtime_environment.sh" <<'BASH'
printf '[INFO] Runtime environment: VIRTUAL_ENV=%s UV_PROJECT_ENVIRONMENT=%s RUNTIME_PYTHON=%s\n' \
    "${VIRTUAL_ENV-<unset>}" "${UV_PROJECT_ENVIRONMENT-<unset>}" "${RUNTIME_PYTHON-<unset>}"
runtime_uv_version='<unavailable>'
runtime_python_version='<unavailable>'
runtime_python_realpath='<unavailable>'
runtime_python_prefix='<unavailable>'
if [[ -x "${UV_BIN-}" ]]; then
    runtime_uv_version=$("${UV_BIN}" --version 2>&1)
fi
if [[ -x "${RUNTIME_PYTHON-}" ]]; then
    runtime_python_version=$("${RUNTIME_PYTHON}" --version 2>&1)
    runtime_python_realpath=$("${RUNTIME_PYTHON}" -c "import os, sys; print(os.path.realpath(sys.executable))" 2>&1)
    runtime_python_prefix=$("${RUNTIME_PYTHON}" -c "import sys; print(sys.prefix)" 2>&1)
fi
printf '[INFO] Runtime versions: uv=%s python=%s\n' "${runtime_uv_version}" "${runtime_python_version}"
printf '[INFO] Runtime Python: realpath=%s prefix=%s\n' "${runtime_python_realpath}" "${runtime_python_prefix}"
printf '[INFO] Runtime build environment: NVTE_CUDA_ARCHS=%s\n' \
    "${NVTE_CUDA_ARCHS-<unset>}"
printf '[INFO] Runtime temporary directory: TMPDIR=%s\n' "${TMPDIR-<unset>}"
unset runtime_uv_version runtime_python_version runtime_python_realpath runtime_python_prefix
BASH

cutedsl_write_event image_hash start '' "Verify immutable container SHA256" image.sha256
if [[ ! -r "${IMAGE}" ]]; then
    echo "[ERROR] Required immutable image is not readable: ${IMAGE}" >&2
    exit 1
fi

SOURCE_STATUS=$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal --ignore-submodules=none)
readonly SOURCE_STATUS
if [[ -n "${SOURCE_STATUS}" ]]; then
    echo "[ERROR] Refusing to benchmark a dirty parent checkout:" >&2
    printf '%s\n' "${SOURCE_STATUS}" >&2
    exit 1
fi
if ! git -C "${REPO_ROOT}" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain --untracked-files=normal --ignore-submodules=none)"'; then
    echo "[ERROR] Refusing to benchmark dirty recursive submodules." >&2
    exit 1
fi

SUBMODULE_STATUS=$(git -C "${REPO_ROOT}" submodule status --recursive)
readonly SUBMODULE_STATUS
while IFS= read -r line; do
    if [[ -n "${line}" && "${line:0:1}" != " " ]]; then
        echo "[ERROR] Recursive submodule is not clean and initialized: ${line}" >&2
        exit 1
    fi
done <<< "${SUBMODULE_STATUS}"

GIT_SHA=$(git -C "${REPO_ROOT}" rev-parse HEAD)
GIT_BRANCH=$(git -C "${REPO_ROOT}" branch --show-current)
UPSTREAM_REF=$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')
UPSTREAM_SHA=$(git -C "${REPO_ROOT}" rev-parse '@{upstream}')
IMAGE_SHA256=$(sha256sum "${IMAGE}" | awk '{print $1}')
readonly GIT_SHA GIT_BRANCH UPSTREAM_REF UPSTREAM_SHA IMAGE_SHA256
validate_cutedsl_runtime_source "${GIT_BRANCH}" "${GIT_SHA}"
if [[ "${IMAGE_SHA256}" != "${CUTEDSL_IMAGE_SHA256}" ]]; then
    echo "[ERROR] Image SHA256 ${IMAGE_SHA256} does not match profile SHA256 ${CUTEDSL_IMAGE_SHA256}." >&2
    exit 1
fi
cutedsl_write_event image_hash pass 0 "Container SHA256 matches cluster profile" image.sha256
if [[ "${GIT_SHA}" != "${UPSTREAM_SHA}" ]]; then
    echo "[ERROR] HEAD ${GIT_SHA} does not match ${UPSTREAM_REF} at ${UPSTREAM_SHA}." >&2
    exit 1
fi
export GIT_SHA GIT_BRANCH UPSTREAM_REF UPSTREAM_SHA IMAGE IMAGE_SHA256

printf '%s\n' "${SUBMODULE_STATUS}" > "${RESULT_DIR}/submodule_status.txt"
printf '%s  %s\n' "${IMAGE_SHA256}" "${IMAGE}" > "${RESULT_DIR}/image.sha256"
cutedsl_write_event preflight pass 0 \
    "Benchmark controls, source, submodules, upstream, and image are immutable" \
    benchmark_manifest.json

A2A_OVERLAP_OVERRIDE="policy.megatron_cfg.overlap_moe_expert_parallel_comm=false"
A2A_HIGH_PRIORITY_OVERRIDE="policy.megatron_cfg.high_priority_a2a_comm_stream=false"
A2A_DELAY_WGRAD_OVERRIDE="policy.megatron_cfg.delay_wgrad_compute=false"
if [[ "${NEMO2606_A2A_ENABLED}" == "1" ]]; then
    A2A_OVERLAP_OVERRIDE="policy.megatron_cfg.overlap_moe_expert_parallel_comm=true"
    A2A_HIGH_PRIORITY_OVERRIDE="policy.megatron_cfg.high_priority_a2a_comm_stream=true"
    A2A_DELAY_WGRAD_OVERRIDE="policy.megatron_cfg.delay_wgrad_compute=true"
fi
readonly A2A_OVERLAP_OVERRIDE A2A_HIGH_PRIORITY_OVERRIDE
readonly A2A_DELAY_WGRAD_OVERRIDE

COMMON_OVERRIDES=(
    "grpo.max_num_steps=${TOTAL_UPDATES}"
    "cluster.num_nodes=${BENCHMARK_NUM_NODES}"
    "cluster.gpus_per_node=${BENCHMARK_GPUS_PER_NODE}"
    "cluster.segment_size=${BENCHMARK_SEGMENT_SIZE}"
    "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}"
    "policy.train_micro_batch_size=1"
    "policy.megatron_cfg.expert_model_parallel_size=${EXPERT_MODEL_PARALLEL_SIZE}"
    "${A2A_OVERLAP_OVERRIDE}"
    "${A2A_HIGH_PRIORITY_OVERRIDE}"
    "${A2A_DELAY_WGRAD_OVERRIDE}"
    "logger.log_dir=${CONTAINER_RUNTIME_DIR}/benchmark_logs"
    "logger.wandb_enabled=false"
    "logger.tensorboard_enabled=true"
    "logger.monitor_gpus=true"
    "checkpointing.enabled=false"
    "checkpointing.checkpoint_dir=${CONTAINER_RUNTIME_DIR}/benchmark_checkpoints"
)
# NEMO2606_PERFORMANCE_VALIDATION_OVERRIDES_START
if [[ "${FUNCTIONAL_GATE}" == "0" ]]; then
    COMMON_OVERRIDES+=(
        "grpo.val_period=0"
        "grpo.val_at_start=false"
        "grpo.val_at_end=false"
    )
fi
# NEMO2606_PERFORMANCE_VALIDATION_OVERRIDES_END
if [[ "${NEMO2606_FULL_CG_ENABLED}" == "1" ]]; then
    COMMON_OVERRIDES+=(
        "policy.megatron_cfg.cuda_graph_impl=full_iteration"
        "policy.megatron_cfg.cuda_graph_warmup_steps=${CUDA_GRAPH_WARMUP_STEPS}"
        "policy.megatron_cfg.cuda_graph_use_single_mempool=true"
    )
fi
readonly -a COMMON_OVERRIDES
printf '%s\n' "${COMMON_OVERRIDES[@]}" > "${RESULT_DIR}/common_overrides.txt"
printf '%s\n' "${timing_arms[@]}" > "${RESULT_DIR}/timing_order.txt"

export CUDA_HOME="/usr/local/cuda"
export NVTE_CUDA_ARCHS="100"
if [[ "${EXISTING_RAY}" == "1" ]]; then
    NODE_LOCAL_RUNTIME_ROOT="/tmp/${USER}/nemo2606-factorial/${RUN_ID}"
    NODE_LOCAL_WORKER_VENV_ROOT="${NODE_LOCAL_RUNTIME_ROOT}/worker_venvs"
    TRITON_CACHE_DIR="${NODE_LOCAL_RUNTIME_ROOT}/triton_cache"
    NEMO2606_TRITON_CACHE_SCOPE="job_node_local"
else
    NODE_LOCAL_RUNTIME_ROOT="${CONTAINER_RUNTIME_DIR}"
    NODE_LOCAL_WORKER_VENV_ROOT="${CONTAINER_RUNTIME_DIR}/worker_venvs"
    TRITON_CACHE_DIR="${CONTAINER_RUNTIME_DIR}/triton_cache"
    NEMO2606_TRITON_CACHE_SCOPE="run_local_container"
fi
readonly NODE_LOCAL_RUNTIME_ROOT NODE_LOCAL_WORKER_VENV_ROOT
export NEMO_RL_VENV_DIR="${NODE_LOCAL_WORKER_VENV_ROOT}"
export TRITON_CACHE_DIR NEMO2606_TRITON_CACHE_SCOPE
mkdir -p "${TRITON_CACHE_DIR}"
MEGATRON_CHECKPOINT_ROOT="${CONTAINER_RUNTIME_DIR}/megatron_checkpoints"
readonly MEGATRON_CHECKPOINT_ROOT
mkdir -p "${MEGATRON_CHECKPOINT_ROOT}"
export NRL_MEGATRON_CHECKPOINT_DIR="${MEGATRON_CHECKPOINT_ROOT}"
export NRL_FORCE_REBUILD_VENVS="true"
export NRL_IGNORE_VERSION_MISMATCH="1"
export PYTHONPATH="${CONTAINER_REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${CONTAINER_RUNTIME_DIR}/torch_extensions"
export UV_PROJECT_ENVIRONMENT="${CONTAINER_RUNTIME_DIR}/venv"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export RUNTIME_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
export UV_CACHE_DIR="${CONTAINER_RUNTIME_DIR}/uv-cache"
export UV_VERSION="0.11.6"
export UV_PYTHON_VERSION="3.13.13"
export UV_NO_MODIFY_PATH=1
export UV_INSTALL_DIR="${CONTAINER_RUNTIME_DIR}/uv-bin"
export UV_PYTHON_INSTALL_DIR="${CONTAINER_RUNTIME_DIR}/uv-python"
export UV_BIN="${UV_INSTALL_DIR}/uv"
# CUTEDSL_SHARED_HF_PREFLIGHT_START
if [[ "${CUTEDSL_SHARED_HF_HOME-}" != /* ]]; then
    echo "[ERROR] shared Hugging Face cache root must be an absolute path." >&2
    exit 1
fi
SHARED_HF_HOME="${CUTEDSL_SHARED_HF_HOME}"
SHARED_HF_MANIFEST="${SHARED_HF_HOME}/nemo2606_qwen3_30ba3b_manifest.json"
if [[ ! -d "${SHARED_HF_HOME}" || ! -r "${SHARED_HF_HOME}" || \
    ! -x "${SHARED_HF_HOME}" ]]; then
    echo "[ERROR] shared Hugging Face cache root must already exist and be readable." >&2
    exit 1
fi
if [[ ! -f "${SHARED_HF_MANIFEST}" || ! -r "${SHARED_HF_MANIFEST}" || \
    ! -s "${SHARED_HF_MANIFEST}" ]]; then
    echo "[ERROR] shared Hugging Face cache requires a completed manifest." >&2
    exit 1
fi
readonly SHARED_HF_HOME SHARED_HF_MANIFEST
export HF_HOME="${SHARED_HF_HOME}"
export SHARED_HF_MANIFEST
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# CUTEDSL_SHARED_HF_PREFLIGHT_END

SRUN=()
if [[ "${EXISTING_RAY}" == "0" ]]; then
    SRUN=(
        srun
        --nodes=1
        --ntasks=1
        --cpus-per-task=72
        --mpi=pmix
        --container-image="${IMAGE}"
        --container-env=VIRTUAL_ENV,UV_PROJECT_ENVIRONMENT,NVTE_CUDA_ARCHS
        --container-mounts="${REPO_ROOT}:${CONTAINER_REPO_ROOT},${RESULT_DIR}:${CONTAINER_RESULT_DIR},${HOST_RUNTIME_DIR}:${CONTAINER_RUNTIME_DIR},${SHARED_HF_HOME}:${SHARED_HF_HOME}"
        --container-workdir="${CONTAINER_REPO_ROOT}"
    )
    if [[ -n "${CUTEDSL_GRES}" ]]; then
        SRUN+=("--gres=${CUTEDSL_GRES}")
    fi
fi
readonly -a SRUN

echo "[INFO] Benchmark run ID: ${RUN_ID}"
echo "[INFO] Timing order: ${TIMING_ORDER}"
echo "[INFO] Updates per timing arm: ${WARMUP_UPDATES} warmup + ${MEASURED_UPDATES} measured"
echo "[INFO] Source: ${GIT_SHA}; image SHA256: ${IMAGE_SHA256}"

echo "[INFO] Bootstrapping the locked node-local environment and validating matched configs."
cutedsl_write_event runtime_bootstrap start '' \
    "Create locked run-local Python environment" uv_lock_check.log
"${SRUN[@]}" bash -s <<'BASH'
export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"
set -euo pipefail
printf '[INFO] Container temporary directory: TMPDIR=%s uid=%s gid=%s\n' \
    "${TMPDIR}" "$(id -u)" "$(id -g)"
[[ "${TMPDIR}" == "${CONTAINER_RUNTIME_DIR}/tmp" ]]
[[ -d "${TMPDIR}" && -w "${TMPDIR}" ]]
cd "${CONTAINER_REPO_ROOT}"
[[ "${NVTE_CUDA_ARCHS}" == "100" ]]
curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
[[ "$("${UV_BIN}" --version)" == "uv ${UV_VERSION} "* ]]
"${UV_BIN}" python install "${UV_PYTHON_VERSION}"
"${UV_BIN}" python find "${UV_PYTHON_VERSION}"
"${UV_BIN}" sync --locked --extra mcore --group test --group dev --python "${UV_PYTHON_VERSION}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c 'import sys; print(sys.prefix)')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
"${UV_BIN}" lock --check 2>&1 | tee "${CONTAINER_RESULT_DIR}/uv_lock_check.log"
"${UV_BIN}" run --active --no-sync python \
    "${CONTAINER_REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g/prepare_hf_cache.py" \
    --hf-home "${HF_HOME}" \
    --shared-manifest "${SHARED_HF_MANIFEST}" \
    --job-manifest "${CONTAINER_RESULT_DIR}/hf_cache_manifest.json"

"${UV_BIN}" run --active --no-sync python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from omegaconf import OmegaConf


def collect_differences(on_value: Any, off_value: Any, path: str = "") -> dict[str, Any]:
    if isinstance(on_value, dict) and isinstance(off_value, dict):
        differences: dict[str, Any] = {}
        for key in sorted(set(on_value) | set(off_value)):
            child_path = f"{path}.{key}" if path else key
            if key not in on_value or key not in off_value:
                differences[child_path] = {
                    "on": on_value.get(key),
                    "off": off_value.get(key),
                }
            else:
                differences.update(
                    collect_differences(on_value[key], off_value[key], child_path)
                )
        return differences
    if on_value != off_value:
        return {path: {"on": on_value, "off": off_value}}
    return {}


def value_at(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for key in dotted_path.split("."):
        value = value[key]
    return value


# NEMO2606_BASE_CONFIG_IDENTITY_START
NORMALIZED_FEATURE_PATHS = (
    "policy.megatron_cfg.env_vars.NVTE_CUTEDSL_FUSED_GROUPED_MLP",
    "policy.megatron_cfg.cuda_graph_impl",
    "policy.megatron_cfg.cuda_graph_warmup_steps",
    "policy.megatron_cfg.cuda_graph_use_single_mempool",
    "policy.megatron_cfg.overlap_moe_expert_parallel_comm",
    "policy.megatron_cfg.high_priority_a2a_comm_stream",
    "policy.megatron_cfg.delay_wgrad_compute",
)
NORMALIZED_OPERATIONAL_PATHS = (
    "logger.log_dir",
    "checkpointing.checkpoint_dir",
)


def _set_normalized_value(config: dict[str, Any], dotted_path: str) -> None:
    parent: dict[str, Any] = config
    keys = dotted_path.split(".")
    for key in keys[:-1]:
        child = parent.get(key)
        if child is None:
            child = {}
            parent[key] = child
        assert isinstance(child, dict), dotted_path
        parent = child
    parent[keys[-1]] = "<normalized-feature-or-operational-value>"


def canonical_base_config_sha256(config: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(config))
    for path in (*NORMALIZED_FEATURE_PATHS, *NORMALIZED_OPERATIONAL_PATHS):
        _set_normalized_value(normalized, path)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
# NEMO2606_BASE_CONFIG_IDENTITY_END


result_dir = Path(os.environ["CONTAINER_RESULT_DIR"])
recipe = os.environ["RECIPE"]
common_overrides = (result_dir / "common_overrides.txt").read_text().splitlines()
selector_path = os.environ["CUTEDSL_SELECTOR_PATH"]
off_override = os.environ["OFF_OVERRIDE"]
expected_arms = os.environ["TIMING_ORDER"].split(",")

register_omegaconf_resolvers()


def resolve(overrides: list[str]) -> dict[str, Any]:
    config = parse_hydra_overrides(load_config(recipe), overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    return MasterConfig(**resolved).model_dump(mode="json")


on_config = resolve(common_overrides)
on_selector = value_at(on_config, selector_path)
assert on_selector == "1", on_selector
OmegaConf.save(OmegaConf.create(on_config), result_dir / "effective_config_on.yaml")
configs_by_arm = {"on": on_config}
if "off" in expected_arms:
    off_config = resolve([*common_overrides, off_override])
    off_selector = value_at(off_config, selector_path)
    assert off_selector == "0", off_selector
    differences = collect_differences(on_config, off_config)
    assert differences == {selector_path: {"on": "1", "off": "0"}}, differences
    configs_by_arm["off"] = off_config
    OmegaConf.save(OmegaConf.create(off_config), result_dir / "effective_config_off.yaml")
    (result_dir / "matched_config_diff.json").write_text(
        json.dumps(differences, indent=2) + "\n"
    )

timing_order = os.environ["TIMING_ORDER"].split(",")
cluster_config = on_config["cluster"]
megatron_config = on_config["policy"]["megatron_cfg"]
topology = {
    "num_nodes": cluster_config["num_nodes"],
    "gpus_per_node": cluster_config["gpus_per_node"],
    "segment_size": cluster_config["segment_size"],
    "tensor_model_parallel_size": megatron_config["tensor_model_parallel_size"],
    "pipeline_model_parallel_size": megatron_config["pipeline_model_parallel_size"],
    "context_parallel_size": megatron_config["context_parallel_size"],
    "expert_tensor_parallel_size": megatron_config["expert_tensor_parallel_size"],
    "expert_model_parallel_size": megatron_config["expert_model_parallel_size"],
}
assert topology == {
    "num_nodes": int(os.environ["BENCHMARK_NUM_NODES"]),
    "gpus_per_node": int(os.environ["BENCHMARK_GPUS_PER_NODE"]),
    "segment_size": int(os.environ["BENCHMARK_SEGMENT_SIZE"]),
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 1,
    "expert_tensor_parallel_size": 1,
    "expert_model_parallel_size": int(os.environ["EXPERT_MODEL_PARALLEL_SIZE"]),
}, topology
fixed_config_paths = (
    "policy.megatron_cfg.moe_grouped_gemm",
    "policy.megatron_cfg.moe_router_dtype",
    "policy.megatron_cfg.use_transformer_engine_op_fuser",
    "policy.megatron_cfg.moe_mlp_glu_interleave_size",
    "policy.megatron_cfg.expert_tensor_parallel_size",
    "policy.megatron_cfg.fp8_cfg.enabled",
    "policy.megatron_cfg.fp8_cfg.fp8_recipe",
    "policy.megatron_cfg.fp8_cfg.fp8_param",
    "policy.megatron_cfg.overlap_moe_expert_parallel_comm",
    "policy.megatron_cfg.high_priority_a2a_comm_stream",
    "policy.megatron_cfg.delay_wgrad_compute",
    "policy.train_global_batch_size",
    "policy.train_micro_batch_size",
    "policy.logprob_batch_size",
    "policy.max_total_sequence_length",
    "policy.sequence_packing.enabled",
    "policy.generation.vllm_cfg.precision",
    "grpo.num_prompts_per_step",
    "grpo.num_generations_per_prompt",
    "grpo.val_period",
    "grpo.val_at_start",
    "grpo.val_at_end",
)
fixed_config_by_arm = {
    arm: {path: value_at(config, path) for path in fixed_config_paths}
    for arm, config in configs_by_arm.items()
}
fixed_config_evidence = {
    arm: {
        **fixed_config_by_arm[arm],
        "NEMO2606_TRITON_CACHE_SCOPE": os.environ["NEMO2606_TRITON_CACHE_SCOPE"],
    }
    for arm in expected_arms
}
for arm in timing_order:
    assert fixed_config_evidence[arm] == fixed_config_evidence["on"]
    assert (
        fixed_config_evidence[arm]["NEMO2606_TRITON_CACHE_SCOPE"]
        == os.environ["NEMO2606_TRITON_CACHE_SCOPE"]
    )
    assert fixed_config_evidence[arm]["policy.megatron_cfg.moe_grouped_gemm"] is True
    assert fixed_config_evidence[arm]["policy.megatron_cfg.moe_router_dtype"] == "fp32"
    assert (
        fixed_config_evidence[arm][
            "policy.megatron_cfg.use_transformer_engine_op_fuser"
        ]
        is True
    )
    assert (
        fixed_config_evidence[arm][
            "policy.megatron_cfg.moe_mlp_glu_interleave_size"
        ]
        == 32
    )
    assert (
        fixed_config_evidence[arm][
            "policy.megatron_cfg.expert_tensor_parallel_size"
        ]
        == 1
    )
    assert fixed_config_evidence[arm]["policy.megatron_cfg.fp8_cfg.enabled"] is True
    assert (
        fixed_config_evidence[arm]["policy.megatron_cfg.fp8_cfg.fp8_recipe"]
        == "mxfp8"
    )
    assert (
        fixed_config_evidence[arm]["policy.megatron_cfg.fp8_cfg.fp8_param"]
        is False
    )
    assert fixed_config_evidence[arm]["policy.train_global_batch_size"] == int(
        os.environ["TRAIN_GLOBAL_BATCH_SIZE"]
    )
    rollout_batch_size = (
        fixed_config_evidence[arm]["grpo.num_prompts_per_step"]
        * fixed_config_evidence[arm]["grpo.num_generations_per_prompt"]
    )
    assert rollout_batch_size == int(os.environ["TRAIN_GLOBAL_BATCH_SIZE"])
    expected_val_period = 10 if os.environ["FUNCTIONAL_GATE"] == "1" else 0
    assert fixed_config_evidence[arm]["grpo.val_period"] == expected_val_period
    assert fixed_config_evidence[arm]["grpo.val_at_start"] is False
    assert fixed_config_evidence[arm]["grpo.val_at_end"] is False
(result_dir / "config_equivalence.json").write_text(
    json.dumps(fixed_config_evidence, indent=2) + "\n"
)
expected_full_cg = os.environ["NEMO2606_FULL_CG_ENABLED"] == "1"
expected_a2a = os.environ["NEMO2606_A2A_ENABLED"] == "1"
full_cg_config_evidence = {
    arm: {
        "cuda_graph_impl": configs_by_arm[arm]["policy"]["megatron_cfg"].get(
            "cuda_graph_impl"
        ),
        "cuda_graph_warmup_steps": configs_by_arm[arm]["policy"][
            "megatron_cfg"
        ].get("cuda_graph_warmup_steps"),
        "cuda_graph_use_single_mempool": configs_by_arm[arm]["policy"][
            "megatron_cfg"
        ].get("cuda_graph_use_single_mempool"),
    }
    for arm in timing_order
}
for arm in timing_order:
    graph_impl = full_cg_config_evidence[arm]["cuda_graph_impl"]
    assert graph_impl == ("full_iteration" if expected_full_cg else None) or (
        not expected_full_cg and graph_impl == "none"
    )
    if expected_full_cg:
        assert full_cg_config_evidence[arm]["cuda_graph_warmup_steps"] == int(
            os.environ["CUDA_GRAPH_WARMUP_STEPS"]
        )
        assert (
            full_cg_config_evidence[arm]["cuda_graph_use_single_mempool"] is True
        )
    for path in (
        "policy.megatron_cfg.overlap_moe_expert_parallel_comm",
        "policy.megatron_cfg.high_priority_a2a_comm_stream",
        "policy.megatron_cfg.delay_wgrad_compute",
    ):
        assert fixed_config_evidence[arm][path] is expected_a2a
assert selector_path == NORMALIZED_FEATURE_PATHS[0], selector_path
base_config_sha256 = canonical_base_config_sha256(on_config)
hf_cache_manifest = json.loads((result_dir / "hf_cache_manifest.json").read_text())
expected_artifacts = {
    "model": ("Qwen/Qwen3-30B-A3B", None),
    "dataset": ("nvidia/OpenMathInstruct-2", "dataset"),
}
artifact_revisions = {}
for label, (expected_repo_id, expected_repo_type) in expected_artifacts.items():
    repository = hf_cache_manifest["repositories"][label]
    assert repository["repo_id"] == expected_repo_id
    assert repository["repo_type"] == expected_repo_type
    revision = repository["revision"]
    assert len(revision) == 40
    int(revision, 16)
    artifact_revisions[label] = {
        "repo_id": expected_repo_id,
        "repo_type": expected_repo_type,
        "revision": revision,
    }
    if label == "dataset":
        assert repository["split"] == "train_1M"
        assert isinstance(repository["num_rows"], int)
        assert repository["num_rows"] > 0
        artifact_revisions[label].update(
            {"split": repository["split"], "num_rows": repository["num_rows"]}
        )
manifest = {
    "functional_gate": os.environ["FUNCTIONAL_GATE"] == "1",
    "performance_eligible": os.environ["FUNCTIONAL_GATE"] != "1",
    "run_id": os.environ["RUN_ID"],
    "cluster_profile": os.environ["CUTEDSL_PROFILE_NAME"],
    "replicate_index": int(os.environ["REPLICATE_INDEX"]),
    "submission_group": os.environ["SUBMISSION_GROUP"],
    "timing_order": timing_order,
    "warmup_updates": int(os.environ["WARMUP_UPDATES"]),
    "measured_updates": int(os.environ["MEASURED_UPDATES"]),
    "total_updates": int(os.environ["TOTAL_UPDATES"]),
    "profile_enabled": os.environ["PROFILE_ENABLED"] == "1",
    "feature_context": os.environ["FEATURE_CONTEXT"],
    "full_cg_enabled": os.environ["NEMO2606_FULL_CG_ENABLED"] == "1",
    "a2a_enabled": os.environ["NEMO2606_A2A_ENABLED"] == "1",
    "triton_cache_scope": os.environ["NEMO2606_TRITON_CACHE_SCOPE"],
    "base_config_sha256": base_config_sha256,
    "artifact_revisions": artifact_revisions,
    "hf_cache_manifest": "hf_cache_manifest.json",
    "available_arms": expected_arms,
    "not_applicable_arms": {
        "off": "full-iteration CUDA Graph requires device-initiated CuTeDSL"
    }
    if os.environ["NEMO2606_FULL_CG_ENABLED"] == "1"
    else {},
    "aggregation_scope": (
        "context_single_arm" if expected_full_cg else "context_local_cutedsl_pair"
    ),
    "full_cg_config_evidence": full_cg_config_evidence,
    "cross_context_factorial_aggregate_available": False,
    "build_environment": {
        "NVTE_CUDA_ARCHS": os.environ["NVTE_CUDA_ARCHS"],
        "TMPDIR": os.environ["TMPDIR"],
    },
    "source_sha": os.environ["GIT_SHA"],
    "upstream_ref": os.environ["UPSTREAM_REF"],
    "upstream_sha": os.environ["UPSTREAM_SHA"],
    "image": os.environ["IMAGE"],
    "image_sha256": os.environ["IMAGE_SHA256"],
    "recipe": recipe,
    "scheduler": {
        "account": os.environ["CUTEDSL_ACCOUNT"],
        "partition": os.environ["CUTEDSL_PARTITION"],
        "gres": os.environ["CUTEDSL_GRES"],
        "segment": int(os.environ["BENCHMARK_SEGMENT_SIZE"]),
    },
    "topology": topology,
    "workload": {
        "train_global_batch_size": on_config["policy"]["train_global_batch_size"],
        "train_micro_batch_size": on_config["policy"]["train_micro_batch_size"],
        "logprob_batch_size": on_config["policy"]["logprob_batch_size"],
        "max_total_sequence_length": on_config["policy"][
            "max_total_sequence_length"
        ],
        "sequence_packing_enabled": on_config["policy"]["sequence_packing"][
            "enabled"
        ],
        "rollout_precision": on_config["policy"]["generation"]["vllm_cfg"][
            "precision"
        ],
        "num_prompts_per_step": on_config["grpo"]["num_prompts_per_step"],
        "num_generations_per_prompt": on_config["grpo"][
            "num_generations_per_prompt"
        ],
        "validation": {
            "period": on_config["grpo"]["val_period"],
            "at_start": on_config["grpo"]["val_at_start"],
            "at_end": on_config["grpo"]["val_at_end"],
        },
    },
    "fixed_config_evidence": fixed_config_evidence,
    "arms": [
        {"arm": arm, "order_index": order_index}
        for order_index, arm in enumerate(timing_order)
    ],
}
(result_dir / "benchmark_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
PY
BASH
cutedsl_write_event runtime_bootstrap pass 0 \
    "Locked runtime environment is active" uv_lock_check.log
if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then
    cutedsl_write_event config_validation start '' \
        "Verify fixed CuTeDSL ON configuration and exact topology" benchmark_manifest.json
    cutedsl_write_event config_validation pass 0 \
        "Functional configuration and benchmark manifest recorded" benchmark_manifest.json
elif [[ "${NEMO2606_FULL_CG_ENABLED}" == "1" ]]; then
    cutedsl_write_event config_validation start '' \
        "Verify CuTeDSL ON config for the full-CG-dependent context" benchmark_manifest.json
    cutedsl_write_event config_validation pass 0 \
        "Single-arm config and benchmark manifest recorded" benchmark_manifest.json
else
    cutedsl_write_event config_validation start '' \
        "Verify matched ON/OFF configs differ only by the selector" matched_config_diff.json
    cutedsl_write_event config_validation pass 0 \
        "Matched config and benchmark manifest recorded" benchmark_manifest.json
fi
cutedsl_write_event focused_tests skip 0 \
    "Focused tests are enforced by run_cutedsl_functional.sbatch" ''
cutedsl_write_event gpu_smoke skip 0 \
    "GPU smoke is enforced by run_cutedsl_functional.sbatch" ''
if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then
    cutedsl_write_event functional_grpo start '' \
        "Run three-update EP${EXPERT_MODEL_PARALLEL_SIZE} CuTeDSL ON functional arm" \
        functional_gate_summary.json
    cutedsl_write_event timing skip 0 \
        "Functional gate is not performance eligible" ''
    echo "[INFO] Running one functional CuTeDSL ON arm without performance extraction."
else
    cutedsl_write_event functional_grpo skip 0 \
        "Functional GRPO is enforced by the staged gate" ''
    echo "[INFO] Running timing-only arms without profiling instrumentation."
    cutedsl_write_event timing start '' \
        "Run timing for the manifest-declared available arms" timing_summary.json
fi
"${SRUN[@]}" bash -s <<'BASH'
export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"
set -euo pipefail
printf '[INFO] Container temporary directory: TMPDIR=%s uid=%s gid=%s\n' \
    "${TMPDIR}" "$(id -u)" "$(id -g)"
[[ "${TMPDIR}" == "${CONTAINER_RUNTIME_DIR}/tmp" ]]
[[ -d "${TMPDIR}" && -w "${TMPDIR}" ]]
cd "${CONTAINER_REPO_ROOT}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${NVTE_CUDA_ARCHS}" == "100" ]]
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c 'import sys; print(sys.prefix)')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
"${UV_BIN}" run --active --no-sync python \
    "${CONTAINER_REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g/prepare_hf_cache.py" \
    --hf-home "${HF_HOME}" \
    --shared-manifest "${SHARED_HF_MANIFEST}" \
    --job-manifest "${CONTAINER_RESULT_DIR}/hf_cache_manifest.json"
mapfile -t COMMON_OVERRIDES < "${CONTAINER_RESULT_DIR}/common_overrides.txt"
IFS=',' read -r -a timing_arms <<< "${TIMING_ORDER}"

run_timing_arm() {
    local arm=$1
    local order_index=$2
    local arm_dir
    if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then
        arm_dir="${CONTAINER_RESULT_DIR}/functional/${order_index}-${arm}"
    else
        arm_dir="${CONTAINER_RESULT_DIR}/timing/${order_index}-${arm}"
    fi
    local -a arm_overrides=()
    if [[ "${arm}" == "off" ]]; then
        arm_overrides=("${OFF_OVERRIDE}")
    fi

    rm -rf -- "${CONTAINER_RUNTIME_DIR}/benchmark_logs" \
        "${CONTAINER_RUNTIME_DIR}/benchmark_checkpoints" \
        "${CONTAINER_RUNTIME_DIR}/ray_timing"
    mkdir -p "${arm_dir}" "${CONTAINER_RUNTIME_DIR}/ray_timing" \
        "${CONTAINER_RUNTIME_DIR}/tmp"
    printf '%s\n' "${COMMON_OVERRIDES[@]}" "${arm_overrides[@]}" \
        > "${arm_dir}/overrides.txt"

    export BENCHMARK_ARM="${arm}"
    export BENCHMARK_ORDER_INDEX="${order_index}"
    export RAY_TMPDIR="${CONTAINER_RUNTIME_DIR}/ray_timing"
    export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"
    unset NRL_NSYS_WORKER_PATTERNS
    unset NRL_NSYS_PROFILE_STEP_RANGE
    unset NRL_NSYS_EXTRA_OPTIONS

    set +e
    "${UV_BIN}" run --active --no-sync examples/run_grpo.py \
        --config "${RECIPE}" \
        "${COMMON_OVERRIDES[@]}" "${arm_overrides[@]}" \
        2>&1 | tee "${arm_dir}/grpo.log"
    local grpo_exit_code=${PIPESTATUS[0]}
    set -e
    if [[ ${grpo_exit_code} -ne 0 ]]; then
        find "${RAY_TMPDIR}" -type f -print > "${arm_dir}/ray_artifacts.txt" 2>/dev/null || true
        return "${grpo_exit_code}"
    fi

    "${UV_BIN}" run --active --no-sync tests/json_dump_tb_logs.py \
        "${CONTAINER_RUNTIME_DIR}/benchmark_logs" \
        --output_path "${arm_dir}/metrics.json"

    if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then
        return 0
    fi

    "${UV_BIN}" run --active --no-sync python - "${arm_dir}" <<'PY'
# CUTEDSL_METRIC_EXTRACTOR_START
import csv
import json
import os
import statistics
import sys
from pathlib import Path

arm_dir = Path(sys.argv[1])
metrics = json.loads((arm_dir / "metrics.json").read_text())
WARMUP_UPDATES = int(os.environ["WARMUP_UPDATES"])
MEASURED_UPDATES = int(os.environ["MEASURED_UPDATES"])
TOTAL_STEP_METRIC = "timing/train/total_step_time"
GENERATION_METRIC = "timing/train/generation"
GENERATION_FINALIZE_METRIC = "timing/train/generation_finalize"
LOGPROB_METRIC = "timing/train/get_logprobs"
POLICY_TIME_METRIC = "timing/train/policy_training"
REFIT_METRIC = "timing/train/prepare_for_generation/transfer_and_update_weights"
TOKEN_COUNT_METRIC = "train/total_num_tokens"
VALID_TOKEN_COUNT_METRIC = "train/global_valid_toks"
MEAN_PROMPT_LENGTH_METRIC = "train/mean_prompt_length"
NUM_VALID_SAMPLES_METRIC = "train/num_valid_samples"
TOTAL_TURNS_METRIC = "train/total_turns"
NORMALIZED_THROUGHPUT_METRIC = "performance/policy_training_tokens_per_sec_per_gpu"
E2E_THROUGHPUT_METRIC = "performance/tokens_per_sec_per_gpu"
GENERATION_THROUGHPUT_METRIC = "performance/generation_tokens_per_sec_per_gpu"
LOGPROB_THROUGHPUT_METRIC = (
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu"
)
TRAINING_GPU_COUNT = int(os.environ["TRAINING_GPU_COUNT"])
METRIC_ALIASES = {
    TOTAL_STEP_METRIC: (TOTAL_STEP_METRIC,),
    GENERATION_METRIC: (GENERATION_METRIC,),
    GENERATION_FINALIZE_METRIC: (GENERATION_FINALIZE_METRIC,),
    LOGPROB_METRIC: (
        LOGPROB_METRIC,
        "timing/train/policy_and_reference_logprobs",
    ),
    POLICY_TIME_METRIC: (POLICY_TIME_METRIC,),
    REFIT_METRIC: (REFIT_METRIC,),
    E2E_THROUGHPUT_METRIC: (E2E_THROUGHPUT_METRIC,),
    GENERATION_THROUGHPUT_METRIC: (GENERATION_THROUGHPUT_METRIC,),
    LOGPROB_THROUGHPUT_METRIC: (LOGPROB_THROUGHPUT_METRIC,),
    NORMALIZED_THROUGHPUT_METRIC: (NORMALIZED_THROUGHPUT_METRIC,),
    TOKEN_COUNT_METRIC: (TOKEN_COUNT_METRIC,),
    VALID_TOKEN_COUNT_METRIC: (VALID_TOKEN_COUNT_METRIC,),
    MEAN_PROMPT_LENGTH_METRIC: (MEAN_PROMPT_LENGTH_METRIC,),
    NUM_VALID_SAMPLES_METRIC: (NUM_VALID_SAMPLES_METRIC,),
    TOTAL_TURNS_METRIC: (TOTAL_TURNS_METRIC,),
}


def resolve_metric_name(canonical_name: str) -> str:
    candidates = METRIC_ALIASES[canonical_name]
    resolved = [name for name in candidates if name in metrics]
    if not resolved:
        raise KeyError(
            f"required metric {canonical_name!r} not found; "
            f"accepted source names: {candidates!r}; "
            f"available metrics: {sorted(metrics)!r}"
        )
    if len(resolved) != 1:
        raise ValueError(
            f"required metric {canonical_name!r} is ambiguous; "
            f"resolved source names: {resolved!r}"
        )
    return resolved[0]


def ordered_values(source_name: str) -> list[tuple[int, float]]:
    series = metrics[source_name]
    return sorted((int(step), float(value)) for step, value in series.items())


resolved_metric_names = {
    canonical_name: resolve_metric_name(canonical_name)
    for canonical_name in METRIC_ALIASES
}
total_step_points = ordered_values(resolved_metric_names[TOTAL_STEP_METRIC])
expected_updates = WARMUP_UPDATES + MEASURED_UPDATES
if len(total_step_points) != expected_updates:
    raise ValueError(
        f"required metric {TOTAL_STEP_METRIC!r} has {len(total_step_points)} "
        f"updates; expected {expected_updates} ({WARMUP_UPDATES} warmup + "
        f"{MEASURED_UPDATES} measured)"
    )
measured_steps = [step for step, _ in total_step_points[-MEASURED_UPDATES:]]
measured_step_set = set(measured_steps)


def measured_points(canonical_name: str) -> list[tuple[int, float]]:
    source_name = resolved_metric_names[canonical_name]
    values_by_step = dict(ordered_values(source_name))
    missing_steps = sorted(measured_step_set - values_by_step.keys())
    if missing_steps:
        raise ValueError(
            f"required metric {canonical_name!r}, resolved as {source_name!r}, "
            f"is missing measured steps: {missing_steps!r}"
        )
    return [(step, values_by_step[step]) for step in measured_steps]


measured_components = {
    canonical_name: measured_points(canonical_name)
    for canonical_name in METRIC_ALIASES
}
generation_points = measured_components[GENERATION_METRIC]
generation_finalize_points = measured_components[GENERATION_FINALIZE_METRIC]
logprob_points = measured_components[LOGPROB_METRIC]
policy_points = measured_components[POLICY_TIME_METRIC]
refit_points = measured_components[REFIT_METRIC]
token_points = measured_components[TOKEN_COUNT_METRIC]
valid_token_points = measured_components[VALID_TOKEN_COUNT_METRIC]
mean_prompt_length_points = measured_components[MEAN_PROMPT_LENGTH_METRIC]
num_valid_samples_points = measured_components[NUM_VALID_SAMPLES_METRIC]
total_turns_points = measured_components[TOTAL_TURNS_METRIC]
e2e_throughput_points = measured_components[E2E_THROUGHPUT_METRIC]
generation_throughput_points = measured_components[GENERATION_THROUGHPUT_METRIC]
logprob_throughput_points = measured_components[LOGPROB_THROUGHPUT_METRIC]
policy_throughput_points = measured_components[NORMALIZED_THROUGHPUT_METRIC]
measured_values = [value for _, value in policy_points]

nonpositive_refit_steps = [
    step for step, duration in refit_points if duration <= 0.0
]
if nonpositive_refit_steps:
    raise ValueError(
        f"required metric {REFIT_METRIC!r} must be positive at measured steps; "
        f"nonpositive steps: {nonpositive_refit_steps!r}"
    )

measured_step_workload = [
    {
        "step": step,
        "total_step_seconds": measured_components[TOTAL_STEP_METRIC][index][1],
        "generation_seconds": generation_points[index][1],
        "generation_finalize_seconds": generation_finalize_points[index][1],
        "logprob_seconds": logprob_points[index][1],
        "policy_training_seconds": policy_points[index][1],
        "refit_transfer_update_seconds": refit_points[index][1],
        "total_num_tokens": token_points[index][1],
        "global_valid_toks": valid_token_points[index][1],
        "mean_prompt_length": mean_prompt_length_points[index][1],
        "num_valid_samples": num_valid_samples_points[index][1],
        "total_turns": total_turns_points[index][1],
        "e2e_tokens_per_sec_per_gpu": e2e_throughput_points[index][1],
        "generation_tokens_per_sec_per_gpu": generation_throughput_points[index][1],
        "policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
            logprob_throughput_points[index][1]
        ),
        "policy_training_tokens_per_sec_per_gpu": policy_throughput_points[index][1],
        "refit_effective_tokens_per_sec_per_gpu": (
            token_points[index][1] / refit_points[index][1] / TRAINING_GPU_COUNT
        ),
    }
    for index, step in enumerate(measured_steps)
]

timing_series = {
    name: [
        {"step": step, "seconds": value}
        for step, value in ordered_values(name)
        if step in measured_step_set
    ]
    for name in sorted(metrics)
    if name.startswith("timing/")
}
raw = {
    "run_id": os.environ["RUN_ID"],
    "arm": os.environ["BENCHMARK_ARM"],
    "order_index": int(os.environ["BENCHMARK_ORDER_INDEX"]),
    "warmup_updates": WARMUP_UPDATES,
    "measured_updates": MEASURED_UPDATES,
    "policy_training_seconds": measured_values,
    "policy_training_median_seconds": statistics.median(measured_values),
    "resolved_metric_names": resolved_metric_names,
    "measured_component_series": {
        canonical_name: [
            {"step": step, "value": value} for step, value in points
        ]
        for canonical_name, points in measured_components.items()
    },
    "token_count_metric": TOKEN_COUNT_METRIC,
    "valid_token_count_metric": VALID_TOKEN_COUNT_METRIC,
    "normalized_throughput_metric": NORMALIZED_THROUGHPUT_METRIC,
    "refit_effective_throughput_metric": "refit_effective_tokens_per_sec_per_gpu",
    "training_gpu_count": TRAINING_GPU_COUNT,
    "measured_step_workload": measured_step_workload,
    "timing_series": timing_series,
}
(arm_dir / "raw_timing.json").write_text(json.dumps(raw, indent=2) + "\n")

with (arm_dir / "raw_timing.csv").open("w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "run_id",
            "arm",
            "order_index",
            "step",
            "total_step_seconds",
            "generation_seconds",
            "generation_finalize_seconds",
            "logprob_seconds",
            "policy_training_seconds",
            "refit_transfer_update_seconds",
            "total_num_tokens",
            "global_valid_toks",
            "mean_prompt_length",
            "num_valid_samples",
            "total_turns",
            "e2e_tokens_per_sec_per_gpu",
            "generation_tokens_per_sec_per_gpu",
            "policy_and_reference_logprobs_tokens_per_sec_per_gpu",
            "policy_training_tokens_per_sec_per_gpu",
            "refit_effective_tokens_per_sec_per_gpu",
        ]
    )
    for item in measured_step_workload:
        writer.writerow(
            [
                raw["run_id"],
                raw["arm"],
                raw["order_index"],
                item["step"],
                item["total_step_seconds"],
                item["generation_seconds"],
                item["generation_finalize_seconds"],
                item["logprob_seconds"],
                item["policy_training_seconds"],
                item["refit_transfer_update_seconds"],
                item["total_num_tokens"],
                item["global_valid_toks"],
                item["mean_prompt_length"],
                item["num_valid_samples"],
                item["total_turns"],
                item["e2e_tokens_per_sec_per_gpu"],
                item["generation_tokens_per_sec_per_gpu"],
                item["policy_and_reference_logprobs_tokens_per_sec_per_gpu"],
                item["policy_training_tokens_per_sec_per_gpu"],
                item["refit_effective_tokens_per_sec_per_gpu"],
            ]
        )
# CUTEDSL_METRIC_EXTRACTOR_END
PY
}

for order_index in "${!timing_arms[@]}"; do
    run_timing_arm "${timing_arms[order_index]}" "${order_index}"
done

if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then
    if [[ "${EXISTING_RAY}" == "1" ]]; then
        sleep 10
    fi
    "${UV_BIN}" run --active --no-sync python - <<'PY'
# CUTEDSL_FUNCTIONAL_SUMMARIZER_START
import json
import math
import os
import re
from pathlib import Path

result_dir = Path(os.environ["CONTAINER_RESULT_DIR"])
arm_dir = result_dir / "functional" / "0-on"
metrics = json.loads((arm_dir / "metrics.json").read_text())
TOTAL_STEP_METRIC = "timing/train/total_step_time"
GENERATION_METRIC = "timing/train/generation"
LOGPROB_METRIC = "timing/train/get_logprobs"
POLICY_TIME_METRIC = "timing/train/policy_training"
REFIT_METRIC = "timing/train/prepare_for_generation/transfer_and_update_weights"
COMPONENT_METRICS = {
    "generation": (GENERATION_METRIC,),
    "policy_and_reference_logprobs": (
        LOGPROB_METRIC,
        "timing/train/policy_and_reference_logprobs",
    ),
    "policy_training": (POLICY_TIME_METRIC,),
    "refit": (REFIT_METRIC,),
}
TRAINING_GPU_COUNT = int(os.environ["TRAINING_GPU_COUNT"])
MAX_FUNCTIONAL_EVIDENCE_FILES = max(512, 64 * TRAINING_GPU_COUNT)
MAX_FUNCTIONAL_EVIDENCE_BYTES = 256 * 1024 * 1024
MAX_FUNCTIONAL_EVIDENCE_BYTES_PER_FILE = 16 * 1024 * 1024
MAX_FUNCTIONAL_EVIDENCE_MATCHES = 4096
MAX_FUNCTIONAL_EVIDENCE_LINE_CHARS = 4096
MEMORY_LIMIT_FRACTION = 0.95
CUTEDSL_ACTIVATION_SIGNATURE = "GroupedGemmGluSm100"
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RAY_WORKER_EVIDENCE_BASENAME = re.compile(
    r"worker-[0-9a-f]+-[0-9a-f]+-[0-9]+\.(?:out|err)", re.IGNORECASE
)


def resolve_metric_name(candidates: tuple[str, ...]) -> str:
    resolved = [name for name in candidates if name in metrics]
    if len(resolved) != 1:
        raise ValueError(
            f"required functional metric candidates {candidates!r} resolved to "
            f"{resolved!r}; available metrics: {sorted(metrics)!r}"
        )
    return resolved[0]


def metric_steps(metric_name: str) -> list[int]:
    return sorted(int(step) for step in metrics[metric_name])


total_step_name = resolve_metric_name((TOTAL_STEP_METRIC,))
completed_steps = metric_steps(total_step_name)
completed_updates = len(completed_steps)
if completed_updates != int(os.environ["FUNCTIONAL_UPDATES"]):
    raise ValueError(
        f"functional total-step series has {completed_updates} updates; "
        f"expected {os.environ['FUNCTIONAL_UPDATES']}: {completed_steps!r}"
    )

component_series = {}
for component, candidates in COMPONENT_METRICS.items():
    source_name = resolve_metric_name(candidates)
    steps = metric_steps(source_name)
    if len(steps) != completed_updates or steps != completed_steps:
        raise ValueError(
            f"functional component {component!r} from {source_name!r} must contain "
            f"all updates {completed_steps!r}; observed {steps!r}"
        )
    component_series[component] = {"metric": source_name, "steps": steps}


def bounded_tail(path: Path, remaining_bytes: int) -> tuple[str, int, set[str]]:
    size = path.stat().st_size
    read_size = min(size, MAX_FUNCTIONAL_EVIDENCE_BYTES_PER_FILE, remaining_bytes)
    truncation_reasons: set[str] = set()
    if size > MAX_FUNCTIONAL_EVIDENCE_BYTES_PER_FILE:
        truncation_reasons.add("per_file_tail_limit")
    if min(size, MAX_FUNCTIONAL_EVIDENCE_BYTES_PER_FILE) > remaining_bytes:
        truncation_reasons.add("total_byte_limit")
    if read_size <= 0:
        return "", 0, truncation_reasons
    with path.open("rb") as stream:
        if size > read_size:
            stream.seek(-read_size, os.SEEK_END)
        data = stream.read(read_size)
    return data.decode(errors="replace"), len(data), truncation_reasons


def parse_evidence_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        key, separator, value = token.partition("=")
        if separator:
            fields[key] = value
    return fields


driver_log = arm_dir / "grpo.log"
ray_log_root = Path(os.environ["RAY_CLUSTER_LOG_DIR"])
if not driver_log.is_file() or not ray_log_root.is_dir():
    raise FileNotFoundError(
        f"functional evidence requires driver log {driver_log} and Ray log root "
        f"{ray_log_root}"
    )
ray_log_paths = sorted(
    path
    for path in ray_log_root.rglob("*")
    if path.is_file()
    and not path.is_symlink()
    and RAY_WORKER_EVIDENCE_BASENAME.fullmatch(path.name)
)
candidate_paths = [driver_log, *ray_log_paths[:MAX_FUNCTIONAL_EVIDENCE_FILES]]
scan_truncation_reasons: set[str] = set()
if len(ray_log_paths) > MAX_FUNCTIONAL_EVIDENCE_FILES:
    scan_truncation_reasons.add("file_count_limit")
scanned_bytes = 0
telemetry_matches: list[dict[str, str]] = []
cutedsl_matches: list[dict[str, str]] = []
for path_index, path in enumerate(candidate_paths):
    text, consumed_bytes, file_truncation_reasons = bounded_tail(
        path, MAX_FUNCTIONAL_EVIDENCE_BYTES - scanned_bytes
    )
    scanned_bytes += consumed_bytes
    scan_truncation_reasons.update(file_truncation_reasons)
    try:
        source = str(path.relative_to(result_dir))
    except ValueError:
        source = str(path)
    for line in text.splitlines():
        clean_line = ANSI_ESCAPE_PATTERN.sub("", line)
        if parse_evidence_fields(clean_line).get("event") == "megatron_policy_offload_memory":
            if len(telemetry_matches) < MAX_FUNCTIONAL_EVIDENCE_MATCHES:
                if len(clean_line) > MAX_FUNCTIONAL_EVIDENCE_LINE_CHARS:
                    scan_truncation_reasons.add("retained_line_limit")
                telemetry_matches.append(
                    {
                        "source": source,
                        "line": clean_line[-MAX_FUNCTIONAL_EVIDENCE_LINE_CHARS:],
                    }
                )
            else:
                scan_truncation_reasons.add("match_count_limit")
        if CUTEDSL_ACTIVATION_SIGNATURE in clean_line:
            if len(cutedsl_matches) < MAX_FUNCTIONAL_EVIDENCE_MATCHES:
                if len(clean_line) > MAX_FUNCTIONAL_EVIDENCE_LINE_CHARS:
                    scan_truncation_reasons.add("retained_line_limit")
                cutedsl_matches.append(
                    {
                        "source": source,
                        "line": clean_line[-MAX_FUNCTIONAL_EVIDENCE_LINE_CHARS:],
                    }
                )
            else:
                scan_truncation_reasons.add("match_count_limit")
    if scanned_bytes >= MAX_FUNCTIONAL_EVIDENCE_BYTES:
        if path_index + 1 < len(candidate_paths):
            scan_truncation_reasons.add("total_byte_limit")
        break

if scan_truncation_reasons:
    raise ValueError(
        "functional evidence scan was truncated; bounded evidence cannot prove "
        f"complete per-rank evidence: {sorted(scan_truncation_reasons)!r}"
    )

completed_ranks: set[int] = set()
finite_limit_ranks: set[int] = set()
unavailable_limit_ranks: set[int] = set()
cgroup_memory_observations: list[dict[str, object]] = []
for evidence in telemetry_matches:
    fields = parse_evidence_fields(evidence["line"])
    global_rank = fields.get("global_rank", "")
    if (
        fields.get("phase") == "after_completion"
        and fields.get("offload_sequence") == "3"
        and re.fullmatch(r"[0-9]+", global_rank)
    ):
        rank = int(global_rank)
        completed_ranks.add(rank)
        max_raw = fields.get("cgroup_memory_max_gib", "unavailable")
        peak_raw = fields.get("cgroup_memory_peak_gib", "unavailable")
        if max_raw in {"", "max", "unavailable", "unbounded"}:
            unavailable_limit_ranks.add(rank)
            cgroup_memory_observations.append(
                {
                    "global_rank": rank,
                    "status": "limit_unavailable_or_unbounded",
                    "peak_gib": peak_raw,
                    "max_gib": max_raw or "unavailable",
                }
            )
            continue
        try:
            max_gib = float(max_raw)
            peak_gib = float(peak_raw)
        except ValueError as error:
            raise ValueError(
                f"rank {rank} has invalid finite cgroup memory evidence: "
                f"peak={peak_raw!r}, max={max_raw!r}"
            ) from error
        if not math.isfinite(max_gib) or max_gib <= 0:
            raise ValueError(
                f"rank {rank} cgroup memory max must be finite and positive; "
                f"observed {max_raw!r}"
            )
        if not math.isfinite(peak_gib) or peak_gib < 0:
            raise ValueError(
                f"rank {rank} cgroup memory peak must be finite and non-negative "
                f"when max is finite; observed {peak_raw!r}"
            )
        fraction = peak_gib / max_gib
        if fraction >= MEMORY_LIMIT_FRACTION:
            raise ValueError(
                "cgroup memory peak/limit must be < 0.95 for every exact mature "
                f"completion; rank {rank} observed {peak_gib}/{max_gib}={fraction:.6f}"
            )
        finite_limit_ranks.add(rank)
        cgroup_memory_observations.append(
            {
                "global_rank": rank,
                "status": "finite_limit_below_threshold",
                "peak_gib": peak_gib,
                "max_gib": max_gib,
                "peak_to_limit_fraction": fraction,
            }
        )
expected_ranks = set(range(int(os.environ["TRAINING_GPU_COUNT"])))
if len(completed_ranks) != int(os.environ["TRAINING_GPU_COUNT"]):
    raise ValueError(
        "functional offload telemetry requires phase=after_completion "
        f"offload_sequence=3 for {sorted(expected_ranks)!r}; observed "
        f"{sorted(completed_ranks)!r}"
    )
if completed_ranks != expected_ranks:
    raise ValueError(
        f"functional offload telemetry ranks must be {sorted(expected_ranks)!r}; "
        f"observed {sorted(completed_ranks)!r}"
    )
if not cutedsl_matches:
    raise ValueError(
        f"functional CuTeDSL activation signature {CUTEDSL_ACTIVATION_SIGNATURE!r} "
        "was not found in driver or Ray logs"
    )

summary = {
    "functional_gate": True,
    "completed_updates": completed_updates,
    "performance_eligible": False,
    "arm": "on",
    "component_series": component_series,
    "offload_memory_evidence": {
        "required_phase": "after_completion",
        "required_offload_sequence": 3,
        "completed_global_ranks": sorted(completed_ranks),
        "cgroup_memory": {
            "limit_fraction_exclusive": MEMORY_LIMIT_FRACTION,
            "finite_limit_global_ranks": sorted(finite_limit_ranks),
            "unavailable_limit_global_ranks": sorted(unavailable_limit_ranks),
            "observations": cgroup_memory_observations,
        },
        "matches": telemetry_matches,
    },
    "cutedsl_activation_evidence": {
        "signature": CUTEDSL_ACTIVATION_SIGNATURE,
        "matches": cutedsl_matches,
    },
    "evidence_scan": {
        "driver_log": str(driver_log.relative_to(result_dir)),
        "ray_log_root": str(ray_log_root),
        "files_scanned": min(len(candidate_paths), MAX_FUNCTIONAL_EVIDENCE_FILES + 1),
        "bytes_scanned": scanned_bytes,
        "truncated": False,
        "truncation_reasons": [],
        "max_retained_line_chars": MAX_FUNCTIONAL_EVIDENCE_LINE_CHARS,
        "max_matches_per_evidence_type": MAX_FUNCTIONAL_EVIDENCE_MATCHES,
    },
    "post_job_slurm_accounting_required": True,
}
(result_dir / "functional_gate_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
manifest_path = result_dir / "benchmark_manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["functional_gate_summary"] = "functional_gate_summary.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
# CUTEDSL_FUNCTIONAL_SUMMARIZER_END
PY
else
"${UV_BIN}" run --active --no-sync python - <<'PY'
# CUTEDSL_TIMING_SUMMARIZER_START
import json
import os
import statistics
from pathlib import Path

result_dir = Path(os.environ["CONTAINER_RESULT_DIR"])
POLICY_TIME_METRIC = "timing/train/policy_training"
TOKEN_COUNT_METRIC = "train/total_num_tokens"
NORMALIZED_THROUGHPUT_METRIC = "performance/policy_training_tokens_per_sec_per_gpu"
WORKLOAD_TOKEN_FIELDS = ("total_num_tokens", "global_valid_toks")
WORKLOAD_EXACT_OBSERVED_FIELDS = (
    "mean_prompt_length",
    "num_valid_samples",
    "total_turns",
)
WORKLOAD_ARM_TOTAL_RELATIVE_DELTA_LIMIT = 0.01
WORKLOAD_PAIRED_STEP_RELATIVE_DELTA_LIMIT = 0.02
arm_results = [
    json.loads(path.read_text())
    for path in sorted((result_dir / "timing").glob("*-*/raw_timing.json"))
]
expected_arms = os.environ["TIMING_ORDER"].split(",")
assert {result["arm"] for result in arm_results} == set(expected_arms), arm_results
by_arm = {result["arm"]: result for result in arm_results}
policy_seconds = {
    arm: by_arm[arm]["policy_training_seconds"] for arm in expected_arms
}
workload = {
    arm: [
        item["total_num_tokens"]
        for item in by_arm[arm]["measured_step_workload"]
    ]
    for arm in expected_arms
}
throughput = {
    arm: [
        item["policy_training_tokens_per_sec_per_gpu"]
        for item in by_arm[arm]["measured_step_workload"]
    ]
    for arm in expected_arms
}
workload_equivalence_required = len(expected_arms) > 1
exact_invariants_observed = None
workload_equivalence_metrics = {}
workload_equivalence_observed = None
if workload_equivalence_required:
    exact_invariants_observed = all(
        [item[field] for item in by_arm["on"]["measured_step_workload"]]
        == [item[field] for item in by_arm["off"]["measured_step_workload"]]
        for field in WORKLOAD_EXACT_OBSERVED_FIELDS
    )
    workload_equivalence_observed = exact_invariants_observed
    for field in WORKLOAD_TOKEN_FIELDS:
        values = {
            arm: [item[field] for item in by_arm[arm]["measured_step_workload"]]
            for arm in ("on", "off")
        }
        on_total = sum(values["on"])
        off_total = sum(values["off"])
        arm_total_relative_delta = abs(on_total - off_total) / (
            (on_total + off_total) / 2.0
        )
        max_paired_step_relative_delta = max(
            abs(on_value - off_value) / ((on_value + off_value) / 2.0)
            for on_value, off_value in zip(values["on"], values["off"], strict=True)
        )
        workload_equivalence_metrics[field] = {
            "on_total": on_total,
            "off_total": off_total,
            "arm_total_relative_delta": arm_total_relative_delta,
            "max_paired_step_relative_delta": max_paired_step_relative_delta,
        }
        workload_equivalence_observed = workload_equivalence_observed and (
            arm_total_relative_delta <= WORKLOAD_ARM_TOTAL_RELATIVE_DELTA_LIMIT
            and max_paired_step_relative_delta
            <= WORKLOAD_PAIRED_STEP_RELATIVE_DELTA_LIMIT
        )
workload_equivalence = {
    "schema_version": 2,
    "relative_delta_formula": "abs(on-off)/mean(on,off)",
    "required": workload_equivalence_required,
    "observed": workload_equivalence_observed,
    "actual_token_normalization_required": True,
    "normalization_metric": TOKEN_COUNT_METRIC,
    "exact_observed_invariants": {
        "fields": list(WORKLOAD_EXACT_OBSERVED_FIELDS),
        "observed": exact_invariants_observed,
    },
    "prompt_sequence_identity_verified": False,
    "limits": {
        "arm_total_relative_delta": WORKLOAD_ARM_TOTAL_RELATIVE_DELTA_LIMIT,
        "paired_step_relative_delta": WORKLOAD_PAIRED_STEP_RELATIVE_DELTA_LIMIT,
    },
    "metrics": workload_equivalence_metrics,
}
if not workload_equivalence_required:
    workload_equivalence["not_applicable_reason"] = "single timing arm"
summary = {
    "run_id": arm_results[0]["run_id"],
    "timing_order": [
        result["arm"] for result in sorted(arm_results, key=lambda item: item["order_index"])
    ],
    "raw_timing_files": [
        str(path.relative_to(result_dir))
        for path in sorted((result_dir / "timing").glob("*-*/raw_timing.json"))
    ],
    "available_arms": expected_arms,
    "median_policy_training_seconds": {
        arm: statistics.median(policy_seconds[arm]) for arm in expected_arms
    },
    "workload_metric": TOKEN_COUNT_METRIC,
    "workload_equivalence": workload_equivalence,
    "measured_total_num_tokens": workload,
    "primary_metric": NORMALIZED_THROUGHPUT_METRIC,
    "primary_metric_normalized_by": "train/total_num_tokens and training GPU count",
    "median_normalized_throughput": {
        arm: statistics.median(throughput[arm]) for arm in expected_arms
    },
    "secondary_metric": POLICY_TIME_METRIC,
    "secondary_metric_confounded_by_live_workload": True,
    "cutedsl_contrast_available": "off" in expected_arms,
}
if "off" in expected_arms:
    summary["primary_on_over_off_speedup"] = statistics.median(throughput["on"]) / statistics.median(throughput["off"])
    summary["secondary_policy_time_on_over_off_speedup"] = statistics.median(policy_seconds["off"]) / statistics.median(policy_seconds["on"])
else:
    summary["not_applicable_contrasts"] = {
        "cutedsl_on_over_off": "full-iteration CUDA Graph requires device-initiated CuTeDSL"
    }
(result_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
if workload_equivalence_required and not workload_equivalence_observed:
    raise ValueError(
        "measured workload equivalence failed: "
        f"contract={workload_equivalence!r}"
    )
manifest_path = result_dir / "benchmark_manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["resolved_metric_names"] = {
    arm: by_arm[arm]["resolved_metric_names"] for arm in expected_arms
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
# CUTEDSL_TIMING_SUMMARIZER_END
PY
fi
BASH
if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then
    [[ ! -e "${CONTAINER_RESULT_DIR}/timing_summary.json" ]]
    [[ ! -e "${CONTAINER_RESULT_DIR}/profiles" ]]
    [[ ! -e "${CONTAINER_RESULT_DIR}/kernel_attribution.json" ]]
    [[ ! -e "${CONTAINER_RESULT_DIR}/feature_attribution.json" ]]
    cutedsl_write_event functional_grpo pass 0 \
        "Three-update functional arm and runtime evidence passed" \
        functional_gate_summary.json
else
    cutedsl_write_event timing pass 0 \
        "Available-arm timing and normalized throughput summary recorded" timing_summary.json
fi

if [[ "${FUNCTIONAL_GATE}" == "0" && "${PROFILE_ENABLED}" == "1" ]]; then
    cutedsl_write_event profile start '' \
        "Run separate profiling-only ON/OFF arms" profiles
    echo "[INFO] Timing is complete; running separate profiling-only arms."
    "${SRUN[@]}" bash -s <<'BASH'
export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"
set -euo pipefail
printf '[INFO] Container temporary directory: TMPDIR=%s uid=%s gid=%s\n' \
    "${TMPDIR}" "$(id -u)" "$(id -g)"
[[ "${TMPDIR}" == "${CONTAINER_RUNTIME_DIR}/tmp" ]]
[[ -d "${TMPDIR}" && -w "${TMPDIR}" ]]
cd "${CONTAINER_REPO_ROOT}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${NVTE_CUDA_ARCHS}" == "100" ]]
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c 'import sys; print(sys.prefix)')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
mapfile -t common_timing_overrides < "${CONTAINER_RESULT_DIR}/common_overrides.txt"
profile_overrides=()
for override in "${common_timing_overrides[@]}"; do
    if [[ "${override}" != grpo.max_num_steps=* ]]; then
        profile_overrides+=("${override}")
    fi
done
profile_max_steps=2
profile_step_range="1:2"
if [[ "${NEMO2606_FULL_CG_ENABLED}" == "1" ]]; then
    profile_max_steps=$((CUDA_GRAPH_WARMUP_STEPS + 2))
    profile_step_range="$((CUDA_GRAPH_WARMUP_STEPS + 1)):$((CUDA_GRAPH_WARMUP_STEPS + 2))"
fi
profile_overrides+=("grpo.max_num_steps=${profile_max_steps}")
IFS=',' read -r -a timing_arms <<< "${TIMING_ORDER}"

run_profile_arm() {
    local arm=$1
    local order_index=$2
    local arm_dir="${CONTAINER_RESULT_DIR}/profiles/${order_index}-${arm}"
    local -a arm_overrides=()
    if [[ "${arm}" == "off" ]]; then
        arm_overrides=("${OFF_OVERRIDE}")
    fi

    rm -rf -- "${CONTAINER_RUNTIME_DIR}/benchmark_logs" \
        "${CONTAINER_RUNTIME_DIR}/benchmark_checkpoints" \
        "${CONTAINER_RUNTIME_DIR}/ray_profile"
    mkdir -p "${arm_dir}/nsight" "${CONTAINER_RUNTIME_DIR}/ray_profile" \
        "${CONTAINER_RUNTIME_DIR}/tmp"
    local profile_marker="${arm_dir}/profile_start.marker"
    touch "${profile_marker}"
    export RAY_TMPDIR="${CONTAINER_RUNTIME_DIR}/ray_profile"
    export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"
    export NRL_NSYS_WORKER_PATTERNS="megatron_policy_worker"
    export NRL_NSYS_PROFILE_STEP_RANGE="${profile_step_range}"
    export NRL_NSYS_EXTRA_OPTIONS='{"cuda-memory-usage":"true","cpuctxsw":"none"}'

    set +e
    "${UV_BIN}" run --active --no-sync examples/run_grpo.py \
        --config "${RECIPE}" \
        "${profile_overrides[@]}" "${arm_overrides[@]}" \
        2>&1 | tee "${arm_dir}/grpo_profile.log"
    local grpo_exit_code=${PIPESTATUS[0]}

    local profile_search_root="${RAY_TMPDIR}"
    if [[ "${EXISTING_RAY}" == "1" ]]; then
        profile_search_root="${RAY_CLUSTER_LOG_DIR}"
        sleep 10
    fi
    mapfile -d '' -t nsight_reports < <(
        find "${profile_search_root}" -type f -path '*/logs/nsight/*.nsys-rep' \
            -newer "${profile_marker}" -print0 2>/dev/null | sort -z
    )
    if ((${#nsight_reports[@]} > 16)); then
        echo "[ERROR] Profiling arm ${arm} produced ${#nsight_reports[@]} reports; bounded limit is 16." >&2
        return 1
    fi
    : > "${arm_dir}/kernel_evidence.txt"
    for report in "${nsight_reports[@]}"; do
        cp -a "${report}" "${arm_dir}/nsight/"
        printf '\n===== %s =====\n' "${report}" >> "${arm_dir}/kernel_evidence.txt"
        nsys stats --report cuda_gpu_kern_sum --format csv "${report}" \
            >> "${arm_dir}/kernel_evidence.txt" 2>&1 || true
        nsys stats --report cuda_api_sum --format csv "${report}" \
            >> "${arm_dir}/kernel_evidence.txt" 2>&1 || true
    done
    find "${profile_search_root}" -type f -print | head -n 200 \
        > "${arm_dir}/ray_artifacts.txt" 2>/dev/null || true
    local report_count
    local profile_artifact_exit_code=0
    report_count=$(find "${arm_dir}/nsight" -type f -name "*.nsys-rep" | wc -l | tr -d " ")
    if ((report_count < 1)); then
        echo "[ERROR] Profiling arm ${arm} produced no .nsys-rep artifact." >&2
        profile_artifact_exit_code=1
    fi
    if [[ ! -s "${arm_dir}/kernel_evidence.txt" ]]; then
        echo "[ERROR] Profiling arm ${arm} produced empty kernel evidence." >&2
        profile_artifact_exit_code=1
    fi
    printf '{\n  "arm": "%s",\n  "order_index": %d,\n  "nsight_report_count": %d,\n  "kernel_evidence": "%s"\n}\n' \
        "${arm}" "${order_index}" "${report_count}" "kernel_evidence.txt" \
        > "${arm_dir}/profile_summary.json"
    set -e
    if [[ ${grpo_exit_code} -ne 0 ]]; then
        return "${grpo_exit_code}"
    fi
    return "${profile_artifact_exit_code}"
}

for order_index in "${!timing_arms[@]}"; do
    run_profile_arm "${timing_arms[order_index]}" "${order_index}"
done

"${UV_BIN}" run --active --no-sync python - "${CONTAINER_RESULT_DIR}" <<'PY'
# CUTEDSL_KERNEL_ATTRIBUTION_START
import json
import re
import sys
from pathlib import Path

FUSED_GLU_SIGNATURES = {
    "cudnn_frontend_fused_glu": (
        r"BlockScaledMoEGroupedGemmGluBiasKernel(?:_|\b)"
    ),
}
FUSED_DGLU_SIGNATURES = {
    "cudnn_frontend_fused_dglu": (
        r"BlockScaledMoEGroupedGemmDgluDbiasKernel(?:_|\b)"
    ),
}
FUSED_QUANT_SIGNATURES = {
    "cudnn_frontend_fused_quant": (
        r"BlockScaledMoEGroupedGemmQuantKernel(?:_|\b)"
    ),
}
FUSED_GROUPED_GEMM_SIGNATURES = {
    "cudnn_frontend_grouped_gemm": (
        r"BlockScaledMoEGroupedGemm\w*Kernel(?:_|\b)"
    ),
}
BASELINE_EXPERT_GEMM_SIGNATURES = {
    "nvjet_sm100": r"(?:^|[^A-Za-z0-9])nvjet_sm100_[A-Za-z0-9_]*",
}
SIGNATURE_REGEXES = {
    "fused_glu": FUSED_GLU_SIGNATURES,
    "fused_dglu": FUSED_DGLU_SIGNATURES,
    "fused_quant": FUSED_QUANT_SIGNATURES,
    "fused_grouped_gemm": FUSED_GROUPED_GEMM_SIGNATURES,
    "baseline_expert_gemm": BASELINE_EXPERT_GEMM_SIGNATURES,
}


def signature_counts(text: str, signatures: dict[str, str]) -> dict[str, int]:
    return {
        name: len(re.findall(pattern, text, flags=re.IGNORECASE))
        for name, pattern in signatures.items()
    }


def total_matches(counts: dict[str, int]) -> int:
    return sum(counts.values())


result_dir = Path(sys.argv[1])
manifest_path = result_dir / "benchmark_manifest.json"
manifest = json.loads(manifest_path.read_text())
config_evidence = manifest.get("fixed_config_evidence", {})
expected_arms = manifest["available_arms"]
failures: list[str] = []
arm_results: dict[str, dict[str, object]] = {}
evidence_by_arm: dict[str, str] = {}

for arm in expected_arms:
    evidence_paths = sorted((result_dir / "profiles").glob(f"*-{arm}/kernel_evidence.txt"))
    if len(evidence_paths) != 1:
        failures.append(
            f"{arm.upper()} expected exactly one kernel evidence file, found {len(evidence_paths)}"
        )
        evidence = ""
        evidence_file = None
    else:
        evidence_path = evidence_paths[0]
        evidence = evidence_path.read_text()
        evidence_file = str(evidence_path.relative_to(result_dir))
    evidence_by_arm[arm] = evidence

    category_counts = {
        category: signature_counts(evidence, signatures)
        for category, signatures in SIGNATURE_REGEXES.items()
    }
    arm_config = config_evidence.get(arm, {})
    baseline_config_eligible = (
        arm_config.get("policy.megatron_cfg.moe_grouped_gemm") is True
        and arm_config.get(
            "policy.megatron_cfg.use_transformer_engine_op_fuser"
        )
        is True
    )
    arm_results[arm] = {
        "kernel_evidence": evidence_file,
        "signature_match_counts": category_counts,
        "fused_glu_match_count": total_matches(category_counts["fused_glu"]),
        "fused_dglu_match_count": total_matches(category_counts["fused_dglu"]),
        "fused_quant_match_count": total_matches(category_counts["fused_quant"]),
        "fused_grouped_gemm_match_count": total_matches(
            category_counts["fused_grouped_gemm"]
        ),
        "baseline_expert_gemm_match_count": (
            total_matches(category_counts["baseline_expert_gemm"])
            if baseline_config_eligible
            else 0
        ),
    }

on_result = arm_results["on"]
if on_result["fused_glu_match_count"] == 0:
    failures.append("ON fused GLU kernel signature was not found")
if on_result["fused_dglu_match_count"] == 0:
    failures.append("ON fused dGLU kernel signature was not found")
if on_result["fused_quant_match_count"] == 0:
    failures.append("ON fused quant kernel signature was not found")
if on_result["fused_grouped_gemm_match_count"] == 0:
    failures.append("ON fused grouped-GEMM kernel signature was not found")
if "off" in expected_arms:
    off_result = arm_results["off"]
    if off_result["fused_glu_match_count"] != 0:
        failures.append("OFF fused GLU kernel signature must be absent")
    if off_result["fused_dglu_match_count"] != 0:
        failures.append("OFF fused dGLU kernel signature must be absent")
    if off_result["fused_quant_match_count"] != 0:
        failures.append("OFF fused quant kernel signature must be absent")
    if off_result["fused_grouped_gemm_match_count"] != 0:
        failures.append("OFF fused grouped-GEMM kernel signature must be absent")
    if off_result["baseline_expert_gemm_match_count"] == 0:
        failures.append(
            "OFF baseline expert-GEMM signature with fixed config predicates was not found"
        )

config_contracts = {
    "policy.megatron_cfg.moe_grouped_gemm": "grouped GEMM",
    "policy.megatron_cfg.use_transformer_engine_op_fuser": "op fuser",
}
for arm in expected_arms:
    for path, label in config_contracts.items():
        if config_evidence.get(arm, {}).get(path) is not True:
            failures.append(f"{arm.upper()} {label} config evidence is not true")

feature_counts = {
    arm: {
        "cuda_graph_launch_api": len(
            re.findall(r"\bcudaGraphLaunch(?:_ptsz)?\b", evidence, re.IGNORECASE)
        ),
        "nccl_a2a_kernel": len(
            re.findall(
                r"\bnccl\w*(?:AllToAll|SendRecv)\w*\b", evidence, re.IGNORECASE
            )
        ),
    }
    for arm, evidence in evidence_by_arm.items()
}
if manifest["full_cg_enabled"]:
    for arm in expected_arms:
        if feature_counts[arm]["cuda_graph_launch_api"] == 0:
            failures.append(f"{arm.upper()} full-CG cudaGraphLaunch evidence was not found")
if manifest["a2a_enabled"]:
    for arm in expected_arms:
        if feature_counts[arm]["nccl_a2a_kernel"] == 0:
            failures.append(f"{arm.upper()} NCCL A2A kernel evidence was not found")

attribution = {
    "passed": not failures,
    "signature_regexes": SIGNATURE_REGEXES,
    "config_evidence": config_evidence,
    "arms": arm_results,
    "failures": failures,
}
(result_dir / "kernel_attribution.json").write_text(
    json.dumps(attribution, indent=2, sort_keys=True) + "\n"
)
manifest["kernel_attribution"] = {
    **attribution,
    "artifact": "kernel_attribution.json",
}
feature_limitations = [
    "a single profile job cannot establish the required replicated performance aggregate"
]
if manifest["full_cg_enabled"]:
    feature_limitations.append(
        "cudaGraphLaunch presence does not prove full-iteration replay"
    )
if manifest["a2a_enabled"]:
    feature_limitations.append(
        "NCCL A2A kernel presence does not prove temporal overlap with GEMM"
    )
manifest["feature_attribution"] = {
    "feature_context": manifest["feature_context"],
    "full_cg_enabled": manifest["full_cg_enabled"],
    "a2a_enabled": manifest["a2a_enabled"],
    "counts": feature_counts,
    "kernel_presence_passed": not failures,
    "full_iteration_replay_verified": (
        False if manifest["full_cg_enabled"] else None
    ),
    "a2a_temporal_overlap_verified": False if manifest["a2a_enabled"] else None,
    "performance_claim_eligible": False,
    "limitations": feature_limitations,
}
(result_dir / "feature_attribution.json").write_text(
    json.dumps(manifest["feature_attribution"], indent=2, sort_keys=True) + "\n"
)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
if failures:
    raise RuntimeError("kernel attribution failed: " + "; ".join(failures))
# CUTEDSL_KERNEL_ATTRIBUTION_END
PY
BASH
    cutedsl_write_event profile pass 0 \
        "Nsight reports and kernel-presence attribution passed for available arms; temporal overlap remains a separate gate" \
        kernel_attribution.json
else
    cutedsl_write_event profile skip 0 \
        "Profiling disabled by CUTEDSL_BENCHMARK_PROFILE=0" ''
fi

if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then
    cutedsl_write_event metrics_export start '' \
        "Publish functional gate summary" functional_gate_summary.json
    cutedsl_write_event metrics_export pass 0 \
        "Functional evidence is ready for report rendering" functional_gate_summary.json
    echo "[PASS] Three-update EP${EXPERT_MODEL_PARALLEL_SIZE} CuTeDSL functional gate completed: ${RESULT_DIR}"
else
    cutedsl_write_event metrics_export start '' \
        "Publish benchmark timing and profile summaries" timing_summary.json
    cutedsl_write_event metrics_export pass 0 \
        "Benchmark evidence is ready for report rendering" timing_summary.json
    echo "[PASS] NeMo 26.06 feature benchmark completed: ${RESULT_DIR}"
fi
