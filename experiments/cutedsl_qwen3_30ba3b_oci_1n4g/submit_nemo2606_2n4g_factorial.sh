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

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
readonly REPO_ROOT
readonly EXPERIMENT_DIR="${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
MODEL_PROFILE_PATH="${NEMO2606_MODEL_PROFILE:-${EXPERIMENT_DIR}/model_profiles/qwen3_30ba3b_2n4g.json}"
TEST_ONLY=0
while (($# > 0)); do
    case "$1" in
        --model-profile)
            if (($# < 2)); then
                echo "[ERROR] --model-profile requires a path." >&2
                exit 2
            fi
            MODEL_PROFILE_PATH=$2
            shift 2
            ;;
        --test-only)
            TEST_ONLY=1
            shift
            ;;
        *)
            echo "Usage: $0 [--model-profile PATH] [--test-only]" >&2
            exit 2
            ;;
    esac
done
readonly MODEL_PROFILE_PATH TEST_ONLY
PROFILE_BOOTSTRAP_PYTHON="${NEMO2606_PROFILE_BOOTSTRAP_PYTHON:-python3}"
readonly PROFILE_BOOTSTRAP_PYTHON
profile_exports=$(
    "${PROFILE_BOOTSTRAP_PYTHON}" \
        "${EXPERIMENT_DIR}/lib/model_profile_bootstrap.py" shell \
        --profile "${MODEL_PROFILE_PATH}"
)
eval "${profile_exports}"
"${PROFILE_BOOTSTRAP_PYTHON}" \
    "${EXPERIMENT_DIR}/lib/model_profile_bootstrap.py" validate \
    --profile "${MODEL_PROFILE_PATH}" --repo-root "${REPO_ROOT}" >/dev/null

require_profile_value() {
    local label=$1
    local actual=$2
    local expected=$3
    if [[ "${actual}" != "${expected}" ]]; then
        echo "[ERROR] ${label}=${actual} does not match selected profile value ${expected}." >&2
        exit 1
    fi
}

CONTEXTS="${NEMO2606_FACTORIAL_CONTEXTS:-${CUTEDSL_PROFILE_DEFAULT_CONTEXTS}}"
REPLICATES="${NEMO2606_FACTORIAL_REPLICATES:-3}"
WARMUP_UPDATES="${NEMO2606_FACTORIAL_WARMUP_UPDATES:-5}"
MEASURED_UPDATES="${NEMO2606_FACTORIAL_MEASURED_UPDATES:-20}"
PROFILE_REPLICATE="${NEMO2606_FACTORIAL_PROFILE_REPLICATE:-0}"
FUNCTIONAL_GATE="${NEMO2606_FUNCTIONAL_GATE:-0}"
FUNCTIONAL_CONTEXT="${NEMO2606_FUNCTIONAL_CONTEXT:-g0a0}"
BENCHMARK_RECIPE="${NEMO2606_FACTORIAL_RECIPE:-${CUTEDSL_PROFILE_RECIPE}}"
BENCHMARK_NUM_NODES="${NEMO2606_FACTORIAL_NUM_NODES:-${CUTEDSL_PROFILE_NUM_NODES}}"
BENCHMARK_GPUS_PER_NODE="${NEMO2606_FACTORIAL_GPUS_PER_NODE:-${CUTEDSL_PROFILE_GPUS_PER_NODE}}"
BENCHMARK_SEGMENT_SIZE="${NEMO2606_FACTORIAL_SEGMENT_SIZE:-${CUTEDSL_PROFILE_SEGMENT_SIZE}}"
TRAIN_GLOBAL_BATCH_SIZE="${NEMO2606_FACTORIAL_TRAIN_GLOBAL_BATCH_SIZE:-${CUTEDSL_PROFILE_TRAIN_GLOBAL_BATCH_SIZE}}"
EXPERT_MODEL_PARALLEL_SIZE="${NEMO2606_FACTORIAL_EXPERT_MODEL_PARALLEL_SIZE:-${CUTEDSL_PROFILE_EP}}"
NUM_PROMPTS_PER_STEP="${NEMO2606_FACTORIAL_NUM_PROMPTS_PER_STEP:-${CUTEDSL_PROFILE_NUM_PROMPTS_PER_STEP}}"
REQUESTED_POLICY_TRAINING_GPU_COUNT="${NEMO2606_FACTORIAL_TRAINING_GPU_COUNT:-${CUTEDSL_PROFILE_POLICY_TRAINING_GPU_COUNT}}"
REQUESTED_CONFIG_SEGMENT_SIZE="${NEMO2606_FACTORIAL_CONFIG_SEGMENT_SIZE:-${CUTEDSL_PROFILE_CONFIG_SEGMENT_SIZE}}"
FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS=60
readonly CONTEXTS REPLICATES WARMUP_UPDATES MEASURED_UPDATES PROFILE_REPLICATE
readonly FUNCTIONAL_GATE FUNCTIONAL_CONTEXT
readonly BENCHMARK_RECIPE BENCHMARK_NUM_NODES BENCHMARK_GPUS_PER_NODE
readonly BENCHMARK_SEGMENT_SIZE TRAIN_GLOBAL_BATCH_SIZE
readonly EXPERT_MODEL_PARALLEL_SIZE FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS
readonly NUM_PROMPTS_PER_STEP
readonly REQUESTED_POLICY_TRAINING_GPU_COUNT REQUESTED_CONFIG_SEGMENT_SIZE

require_profile_value NEMO2606_FACTORIAL_RECIPE \
    "${BENCHMARK_RECIPE}" "${CUTEDSL_PROFILE_RECIPE}"
require_profile_value NEMO2606_FACTORIAL_NUM_NODES \
    "${BENCHMARK_NUM_NODES}" "${CUTEDSL_PROFILE_NUM_NODES}"
require_profile_value NEMO2606_FACTORIAL_GPUS_PER_NODE \
    "${BENCHMARK_GPUS_PER_NODE}" "${CUTEDSL_PROFILE_GPUS_PER_NODE}"
require_profile_value NEMO2606_FACTORIAL_SEGMENT_SIZE \
    "${BENCHMARK_SEGMENT_SIZE}" "${CUTEDSL_PROFILE_SEGMENT_SIZE}"
require_profile_value NEMO2606_FACTORIAL_TRAIN_GLOBAL_BATCH_SIZE \
    "${TRAIN_GLOBAL_BATCH_SIZE}" "${CUTEDSL_PROFILE_TRAIN_GLOBAL_BATCH_SIZE}"
require_profile_value NEMO2606_FACTORIAL_EXPERT_MODEL_PARALLEL_SIZE \
    "${EXPERT_MODEL_PARALLEL_SIZE}" "${CUTEDSL_PROFILE_EP}"
require_profile_value NEMO2606_FACTORIAL_NUM_PROMPTS_PER_STEP \
    "${NUM_PROMPTS_PER_STEP}" "${CUTEDSL_PROFILE_NUM_PROMPTS_PER_STEP}"

for positive_integer in \
    "${BENCHMARK_NUM_NODES}" \
    "${BENCHMARK_GPUS_PER_NODE}" \
    "${TRAIN_GLOBAL_BATCH_SIZE}" \
    "${EXPERT_MODEL_PARALLEL_SIZE}"; do
    if [[ ! "${positive_integer}" =~ ^[0-9]+$ ]] || ((positive_integer < 1)); then
        echo "[ERROR] Official workload topology and batch controls must be positive integers." >&2
        exit 1
    fi
done
if [[ ! "${BENCHMARK_SEGMENT_SIZE}" =~ ^[0-9]+$ ]] || \
    ((BENCHMARK_SEGMENT_SIZE < 1)); then
    echo "[ERROR] Scheduler segment size must be a positive integer; null is only valid for the NeMo config segment size." >&2
    exit 1
fi
if [[ -n "${NUM_PROMPTS_PER_STEP}" ]] && \
    { [[ ! "${NUM_PROMPTS_PER_STEP}" =~ ^[0-9]+$ ]] || \
        ((NUM_PROMPTS_PER_STEP < 1)); }; then
    echo "[ERROR] NEMO2606_FACTORIAL_NUM_PROMPTS_PER_STEP must be empty or a positive integer." >&2
    exit 1
fi
readonly WORLD_SIZE=$((BENCHMARK_NUM_NODES * BENCHMARK_GPUS_PER_NODE))
if ((WORLD_SIZE % EXPERT_MODEL_PARALLEL_SIZE != 0)); then
    echo "[ERROR] Expert model parallel size must divide the benchmark world size." >&2
    exit 1
fi

if [[ "${FUNCTIONAL_GATE}" != "1" ]]; then
    if [[ ! "${REPLICATES}" =~ ^[0-9]+$ ]] || ((REPLICATES < 3)); then
        echo "[ERROR] NEMO2606_FACTORIAL_REPLICATES must be an integer >= 3." >&2
        exit 1
    fi
    if [[ ! "${WARMUP_UPDATES}" =~ ^[0-9]+$ ]] || ((WARMUP_UPDATES < 5)); then
        echo "[ERROR] NEMO2606_FACTORIAL_WARMUP_UPDATES must be an integer >= 5." >&2
        exit 1
    fi
    if [[ ! "${MEASURED_UPDATES}" =~ ^[0-9]+$ ]] || ((MEASURED_UPDATES < 20)); then
        echo "[ERROR] NEMO2606_FACTORIAL_MEASURED_UPDATES must be an integer >= 20." >&2
        exit 1
    fi
    if [[ ! "${PROFILE_REPLICATE}" =~ ^[0-9]+$ ]] || \
        ((PROFILE_REPLICATE >= REPLICATES)); then
        echo "[ERROR] NEMO2606_FACTORIAL_PROFILE_REPLICATE must be in [0, REPLICATES)." >&2
        exit 1
    fi
fi

readonly MATRIX_PAYLOAD="${EXPERIMENT_DIR}/run_cutedsl_matrix.sbatch"
readonly RAY_SUB="${REPO_ROOT}/ray.sub"
readonly RECIPE="${BENCHMARK_RECIPE}"
source "${EXPERIMENT_DIR}/lib/cluster_profile.sh"
capture_cutedsl_submission_source "${REPO_ROOT}"
load_cutedsl_cluster_profile

if [[ ! -x "${MATRIX_PAYLOAD}" ]]; then
    echo "[ERROR] Matrix payload is not executable: ${MATRIX_PAYLOAD}" >&2
    exit 1
fi
if [[ ! -r "${RAY_SUB}" ]]; then
    echo "[ERROR] ray.sub is not readable: ${RAY_SUB}" >&2
    exit 1
fi

sbatch_args=()
while IFS= read -r argument; do
    sbatch_args+=("${argument}")
done <<< "${CUTEDSL_SBATCH_ARGS}"
if [[ -n "${CUTEDSL_SEGMENT}" ]]; then
    for index in "${!sbatch_args[@]}"; do
        if [[ "${sbatch_args[${index}]}" == --segment=* ]]; then
            sbatch_args[${index}]="--segment=${BENCHMARK_SEGMENT_SIZE}"
        fi
    done
fi
sbatch_args+=(
    "--nodes=${BENCHMARK_NUM_NODES}"
    "--exclusive"
    "--time=${CUTEDSL_BENCHMARK_TIME}"
)
if [[ "${TEST_ONLY}" == "1" ]]; then
    sbatch_args+=("--test-only")
fi

SUBMISSION_GROUP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly SUBMISSION_GROUP
readonly RESULT_ROOT="${EXPERIMENT_DIR}/results"
readonly RUNTIME_ROOT="${RESULT_ROOT}/multinode_runtime/${SUBMISSION_GROUP}"
readonly RAY_LOG_ROOT="${RESULT_ROOT}/ray_logs/${SUBMISSION_GROUP}"
readonly SUBMISSION_DIR="${RESULT_ROOT}/factorial/submissions"
readonly COHORT_SUBMISSION="${SUBMISSION_DIR}/${SUBMISSION_GROUP}.jsonl"
readonly COHORT_SUBMISSION_TEMP="${COHORT_SUBMISSION}.tmp.$$"
readonly CACHE_DIAGNOSTIC="${EXPERIMENT_DIR}/collect_triton_cache_diagnostics.py"
printf -v FAILURE_COMMAND 'export TRITON_CACHE_DIR="/tmp/${USER}/nemo2606-factorial/${SLURM_JOB_ID}${SLURM_RESTART_COUNT:+-r${SLURM_RESTART_COUNT}}/triton_cache"; exec python3 %q --from-slurm-env' "${CACHE_DIAGNOSTIC}"
readonly FAILURE_COMMAND
GIT_COMMON_DIR=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
readonly GIT_COMMON_DIR
RAY_MOUNTS="${REPO_ROOT}:${REPO_ROOT}"
case "${GIT_COMMON_DIR}" in
    "${REPO_ROOT}"|"${REPO_ROOT}"/*) ;;
    *) RAY_MOUNTS+=",${GIT_COMMON_DIR}:${GIT_COMMON_DIR}" ;;
esac
RAY_MOUNTS+=",${CUTEDSL_SHARED_HF_HOME}:${CUTEDSL_SHARED_HF_HOME}"
RAY_MOUNTS+=",${CUTEDSL_IMAGE}:${CUTEDSL_IMAGE}"
readonly RAY_MOUNTS
readonly RUNTIME_CANARY="${RUNTIME_ROOT}/.shared_fs_canary"
readonly RAY_SETUP_COMMAND="test -r ${RUNTIME_CANARY} && grep -Fx ${CUTEDSL_SUBMISSION_GIT_SHA} ${RUNTIME_CANARY} && test \"\$(git -C ${REPO_ROOT} rev-parse HEAD)\" = ${CUTEDSL_SUBMISSION_GIT_SHA}"
EXPORT_PAYLOAD=$(mktemp "${TMPDIR:-/tmp}/nemo2606-factorial-export.XXXXXX")
readonly EXPORT_PAYLOAD
cleanup_submission_files() {
    local status=$?
    trap - EXIT
    rm -f "${EXPORT_PAYLOAD}" "${COHORT_SUBMISSION_TEMP}"
    exit "${status}"
}
trap cleanup_submission_files EXIT
chmod 600 "${EXPORT_PAYLOAD}"
if [[ "${TEST_ONLY}" == "0" ]]; then
    mkdir -p "${SUBMISSION_DIR}" "${RUNTIME_ROOT}" "${RAY_LOG_ROOT}"
    printf '%s\n' "${CUTEDSL_SUBMISSION_GIT_SHA}" > "${RUNTIME_CANARY}"
    if [[ "${FUNCTIONAL_GATE}" != "1" ]]; then
        if [[ -e "${COHORT_SUBMISSION}" || -L "${COHORT_SUBMISSION}" || \
            -e "${COHORT_SUBMISSION_TEMP}" || -L "${COHORT_SUBMISSION_TEMP}" ]]; then
            echo "[ERROR] Refusing to overwrite factorial cohort submission record." >&2
            exit 1
        fi
        : > "${COHORT_SUBMISSION_TEMP}"
        chmod 600 "${COHORT_SUBMISSION_TEMP}"
    fi
fi

resolve_context() {
    case "$1" in
        g0a0) full_cg_enabled=0; a2a_enabled=0 ;;
        g1a0) full_cg_enabled=1; a2a_enabled=0 ;;
        g0a1) full_cg_enabled=0; a2a_enabled=1 ;;
        g1a1) full_cg_enabled=1; a2a_enabled=1 ;;
        *)
            echo "[ERROR] Unknown factorial context: $1" >&2
            exit 1
            ;;
    esac
}

require_profile_feature_support() {
    if [[ "${full_cg_enabled}" == "1" && \
        "${CUTEDSL_PROFILE_ALLOW_FULL_CG-}" != "true" ]]; then
        echo "[ERROR] selected profile does not allow full-iteration CUDA Graph." >&2
        exit 1
    fi
    if [[ "${a2a_enabled}" == "1" && \
        "${CUTEDSL_PROFILE_ALLOW_A2A-}" != "true" ]]; then
        echo "[ERROR] selected profile does not allow A2A." >&2
        exit 1
    fi
}

resolve_submit_topology() {
    local require_explicit=$1
    if [[ -z "${REQUESTED_POLICY_TRAINING_GPU_COUNT}" ]]; then
        if [[ "${require_explicit}" == "1" ]]; then
            echo "[ERROR] Full-CG/noncolocated submission requires explicit NEMO2606_FACTORIAL_TRAINING_GPU_COUNT." >&2
            exit 1
        fi
        POLICY_TRAINING_GPU_COUNT=${WORLD_SIZE}
    else
        POLICY_TRAINING_GPU_COUNT=${REQUESTED_POLICY_TRAINING_GPU_COUNT}
    fi
    if [[ ! "${POLICY_TRAINING_GPU_COUNT}" =~ ^[0-9]+$ ]] || \
        ((POLICY_TRAINING_GPU_COUNT < 1 || POLICY_TRAINING_GPU_COUNT > WORLD_SIZE)); then
        echo "[ERROR] Policy-training GPU count must be in [1, ${WORLD_SIZE}]." >&2
        exit 1
    fi
    if ((POLICY_TRAINING_GPU_COUNT % EXPERT_MODEL_PARALLEL_SIZE != 0)); then
        echo "[ERROR] Expert model parallel size must divide the policy-training GPU count." >&2
        exit 1
    fi

    if [[ -z "${REQUESTED_CONFIG_SEGMENT_SIZE}" ]]; then
        if [[ "${require_explicit}" == "1" ]]; then
            echo "[ERROR] Full-CG/noncolocated submission requires explicit NEMO2606_FACTORIAL_CONFIG_SEGMENT_SIZE (use null to preserve the recipe)." >&2
            exit 1
        fi
        CONFIG_SEGMENT_SIZE=${BENCHMARK_SEGMENT_SIZE}
    else
        CONFIG_SEGMENT_SIZE=${REQUESTED_CONFIG_SEGMENT_SIZE}
    fi
    if [[ "${CONFIG_SEGMENT_SIZE}" != "null" ]] && \
        { [[ ! "${CONFIG_SEGMENT_SIZE}" =~ ^[0-9]+$ ]] || \
            ((CONFIG_SEGMENT_SIZE < 1)); }; then
        echo "[ERROR] NeMo config segment size must be null or a positive integer." >&2
        exit 1
    fi
    readonly POLICY_TRAINING_GPU_COUNT CONFIG_SEGMENT_SIZE
}

if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then
    resolve_context "${FUNCTIONAL_CONTEXT}"
    require_profile_feature_support
    if [[ "${full_cg_enabled}" == "1" ]]; then
        functional_updates=6
        resolve_submit_topology 1
    else
        functional_updates=3
        resolve_submit_topology 0
    fi
    env -0 \
        -u COMMAND \
        -u CONTAINER \
        -u MOUNTS \
        -u SETUP_COMMAND \
        -u FAILURE_COMMAND \
        -u FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS \
        -u BASE_LOG_DIR \
        -u GPUS_PER_NODE \
        -u CUTEDSL_BENCHMARK_EXISTING_RAY \
        -u CUTEDSL_BENCHMARK_NUM_NODES \
        -u CUTEDSL_BENCHMARK_GPUS_PER_NODE \
        -u CUTEDSL_BENCHMARK_SEGMENT_SIZE \
        -u CUTEDSL_BENCHMARK_TRAIN_GLOBAL_BATCH_SIZE \
        -u CUTEDSL_BENCHMARK_EXPERT_MODEL_PARALLEL_SIZE \
        -u CUTEDSL_BENCHMARK_TRAINING_GPU_COUNT \
        -u CUTEDSL_BENCHMARK_CONFIG_SEGMENT_SIZE \
        -u CUTEDSL_BENCHMARK_NUM_PROMPTS_PER_STEP \
        -u CUTEDSL_BENCHMARK_ORDER \
        -u CUTEDSL_BENCHMARK_PROFILE \
        -u CUTEDSL_BENCHMARK_RESULT_ROOT \
        -u CUTEDSL_SHARED_HF_HOME \
        -u NEMO2606_FUNCTIONAL_GATE \
        -u NEMO2606_FUNCTIONAL_UPDATES \
        -u NEMO2606_FACTORIAL_CONTEXT \
        -u NEMO2606_FULL_CG_ENABLED \
        -u NEMO2606_A2A_ENABLED \
        -u UV_NO_EDITABLE \
        -u SLURM_EXPORT_ENV \
        "CONTAINER=${CUTEDSL_IMAGE}" \
        "MOUNTS=${RAY_MOUNTS}" \
        "SETUP_COMMAND=${RAY_SETUP_COMMAND}" \
        "FAILURE_COMMAND=${FAILURE_COMMAND}" \
        "FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS=${FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS}" \
        "COMMAND=exec bash ${MATRIX_PAYLOAD}" \
        "BASE_LOG_DIR=${RAY_LOG_ROOT}" \
        "GPUS_PER_NODE=${BENCHMARK_GPUS_PER_NODE}" \
        "RAY_LOG_SYNC_FREQUENCY=5" \
        "CUTEDSL_BENCHMARK_EXISTING_RAY=1" \
        "CUTEDSL_BENCHMARK_RECIPE=${RECIPE}" \
        "CUTEDSL_BENCHMARK_NUM_NODES=${BENCHMARK_NUM_NODES}" \
        "CUTEDSL_BENCHMARK_GPUS_PER_NODE=${BENCHMARK_GPUS_PER_NODE}" \
        "CUTEDSL_BENCHMARK_SEGMENT_SIZE=${BENCHMARK_SEGMENT_SIZE}" \
        "CUTEDSL_BENCHMARK_TRAIN_GLOBAL_BATCH_SIZE=${TRAIN_GLOBAL_BATCH_SIZE}" \
        "CUTEDSL_BENCHMARK_EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE}" \
        "CUTEDSL_BENCHMARK_TRAINING_GPU_COUNT=${POLICY_TRAINING_GPU_COUNT}" \
        "CUTEDSL_BENCHMARK_CONFIG_SEGMENT_SIZE=${CONFIG_SEGMENT_SIZE}" \
        "CUTEDSL_BENCHMARK_NUM_PROMPTS_PER_STEP=${NUM_PROMPTS_PER_STEP}" \
        "CUTEDSL_BENCHMARK_RUNTIME_ROOT=${RUNTIME_ROOT}" \
        "CUTEDSL_BENCHMARK_ORDER=on" \
        "CUTEDSL_BENCHMARK_PROFILE=0" \
        "CUTEDSL_BENCHMARK_RESULT_ROOT=${RESULT_ROOT}" \
        "CUTEDSL_SHARED_HF_HOME=${CUTEDSL_SHARED_HF_HOME}" \
        "CUTEDSL_BENCHMARK_SUBMISSION_GROUP=${SUBMISSION_GROUP}" \
        "NEMO2606_FUNCTIONAL_GATE=1" \
        "NEMO2606_FUNCTIONAL_UPDATES=${functional_updates}" \
        "NEMO2606_FACTORIAL_CONTEXT=${FUNCTIONAL_CONTEXT}" \
        "NEMO2606_FULL_CG_ENABLED=${full_cg_enabled}" \
        "NEMO2606_A2A_ENABLED=${a2a_enabled}" \
        "UV_NO_EDITABLE=1" \
        "SLURM_EXPORT_ENV=ALL" \
        > "${EXPORT_PAYLOAD}"

    functional_submission_id=$(sbatch --parsable "${sbatch_args[@]}" \
        "--job-name=${CUTEDSL_ACCOUNT}-n2606.functional.${FUNCTIONAL_CONTEXT}.${SUBMISSION_GROUP}" \
        "--export-file=${EXPORT_PAYLOAD}" \
        "${RAY_SUB}")
    functional_record=$(printf \
        '{"functional_gate":true,"functional_updates":%d,"factorial_context":"%s","full_cg_enabled":%s,"a2a_enabled":%s,"timing_order":"on","profile_enabled":false,"job_id":"%s","submission_group":"%s"}' \
        "${functional_updates}" "${FUNCTIONAL_CONTEXT}" \
        "${full_cg_enabled}" "${a2a_enabled}" \
        "${functional_submission_id}" "${SUBMISSION_GROUP}")
    printf '%s\n' "${functional_record}"
    if [[ "${TEST_ONLY}" == "0" ]]; then
        printf '%s\n' "${functional_record}" \
            >> "${SUBMISSION_DIR}/${SUBMISSION_GROUP}-functional.jsonl"
        echo "[INFO] Submitted EP${EXPERT_MODEL_PARALLEL_SIZE} ${FUNCTIONAL_CONTEXT} functional gate."
    else
        echo "[INFO] Validated EP${EXPERT_MODEL_PARALLEL_SIZE} ${FUNCTIONAL_CONTEXT} functional gate; no job submitted."
    fi
    exit 0
fi

IFS=',' read -r -a contexts <<< "${CONTEXTS}"
if [[ ${#contexts[@]} -eq 0 ]]; then
    echo "[ERROR] NEMO2606_FACTORIAL_CONTEXTS must not be empty." >&2
    exit 1
fi

needs_full_cg="0"
needs_a2a="0"
for context in "${contexts[@]}"; do
    resolve_context "${context}"
    require_profile_feature_support
    if [[ "${full_cg_enabled}" == "1" ]]; then
        needs_full_cg="1"
    fi
    if [[ "${a2a_enabled}" == "1" ]]; then
        needs_a2a="1"
    fi
done
if [[ "${TEST_ONLY}" == "0" && "${needs_a2a}" == "1" ]]; then
    if ! grep -q "return_schedule_plan" "${REPO_ROOT}/nemo_rl/models/megatron/train.py" || \
        ! grep -q "overlap_moe_expert_parallel_comm" \
            "${REPO_ROOT}/nemo_rl/models/megatron/setup.py"; then
        echo "[ERROR] Requested A2A contexts require the NeMo-RL schedule-plan and config-propagation implementation." >&2
        exit 1
    fi
fi
if [[ "${TEST_ONLY}" == "0" && "${needs_full_cg}" == "1" ]]; then
    full_cg_source="${REPO_ROOT}/nemo_rl/models/megatron/full_cuda_graph.py"
    if [[ ! -r "${full_cg_source}" ]] || \
        ! grep -q "build_full_cuda_graph_schedule" "${full_cg_source}"; then
        echo "[ERROR] Requested full-CG contexts require the NeMo-RL full-iteration implementation." >&2
        exit 1
    fi
fi
if [[ "${needs_full_cg}" == "1" ]]; then
    resolve_submit_topology 1
else
    resolve_submit_topology 0
fi

for ((replicate_index = 0; replicate_index < REPLICATES; replicate_index++)); do
    for ((context_offset = 0; context_offset < ${#contexts[@]}; context_offset++)); do
        context_index=$(((replicate_index + context_offset) % ${#contexts[@]}))
        context="${contexts[context_index]}"
        resolve_context "${context}"
        submission_record="${SUBMISSION_DIR}/${SUBMISSION_GROUP}-${context}.jsonl"
        if [[ "${full_cg_enabled}" == "1" ]]; then
            timing_order="on"
        elif ((replicate_index % 2 == 0)); then
            timing_order="on,off"
        else
            timing_order="off,on"
        fi
        profile_enabled=0
        if ((replicate_index == PROFILE_REPLICATE)); then
            profile_enabled=1
        fi

        env -0 \
            -u COMMAND \
            -u CONTAINER \
            -u MOUNTS \
            -u SETUP_COMMAND \
            -u FAILURE_COMMAND \
            -u FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS \
            -u BASE_LOG_DIR \
            -u GPUS_PER_NODE \
            -u CUTEDSL_BENCHMARK_EXISTING_RAY \
            -u CUTEDSL_BENCHMARK_SEGMENT_SIZE \
            -u CUTEDSL_BENCHMARK_TRAIN_GLOBAL_BATCH_SIZE \
            -u CUTEDSL_BENCHMARK_EXPERT_MODEL_PARALLEL_SIZE \
            -u CUTEDSL_BENCHMARK_TRAINING_GPU_COUNT \
            -u CUTEDSL_BENCHMARK_CONFIG_SEGMENT_SIZE \
            -u CUTEDSL_BENCHMARK_NUM_PROMPTS_PER_STEP \
            -u CUTEDSL_BENCHMARK_ORDER \
            -u CUTEDSL_BENCHMARK_REPLICATE \
            -u CUTEDSL_BENCHMARK_PROFILE \
            -u CUTEDSL_BENCHMARK_RESULT_ROOT \
            -u CUTEDSL_SHARED_HF_HOME \
            -u CUTEDSL_BENCHMARK_SUBMISSION_GROUP \
            -u CUTEDSL_BENCHMARK_WARMUP_UPDATES \
            -u CUTEDSL_BENCHMARK_MEASURED_UPDATES \
            -u NEMO2606_FACTORIAL_CONTEXT \
            -u NEMO2606_FULL_CG_ENABLED \
            -u NEMO2606_A2A_ENABLED \
            -u UV_NO_EDITABLE \
            -u SLURM_EXPORT_ENV \
            "CONTAINER=${CUTEDSL_IMAGE}" \
            "MOUNTS=${RAY_MOUNTS}" \
            "SETUP_COMMAND=${RAY_SETUP_COMMAND}" \
            "FAILURE_COMMAND=${FAILURE_COMMAND}" \
            "FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS=${FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS}" \
            "COMMAND=exec bash ${MATRIX_PAYLOAD}" \
            "BASE_LOG_DIR=${RAY_LOG_ROOT}" \
            "GPUS_PER_NODE=${BENCHMARK_GPUS_PER_NODE}" \
            "RAY_LOG_SYNC_FREQUENCY=5" \
            "CUTEDSL_BENCHMARK_EXISTING_RAY=1" \
            "CUTEDSL_BENCHMARK_RECIPE=${RECIPE}" \
            "CUTEDSL_BENCHMARK_NUM_NODES=${BENCHMARK_NUM_NODES}" \
            "CUTEDSL_BENCHMARK_GPUS_PER_NODE=${BENCHMARK_GPUS_PER_NODE}" \
            "CUTEDSL_BENCHMARK_SEGMENT_SIZE=${BENCHMARK_SEGMENT_SIZE}" \
            "CUTEDSL_BENCHMARK_TRAIN_GLOBAL_BATCH_SIZE=${TRAIN_GLOBAL_BATCH_SIZE}" \
            "CUTEDSL_BENCHMARK_EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE}" \
            "CUTEDSL_BENCHMARK_TRAINING_GPU_COUNT=${POLICY_TRAINING_GPU_COUNT}" \
            "CUTEDSL_BENCHMARK_CONFIG_SEGMENT_SIZE=${CONFIG_SEGMENT_SIZE}" \
            "CUTEDSL_BENCHMARK_NUM_PROMPTS_PER_STEP=${NUM_PROMPTS_PER_STEP}" \
            "CUTEDSL_BENCHMARK_RUNTIME_ROOT=${RUNTIME_ROOT}" \
            "CUTEDSL_BENCHMARK_ORDER=${timing_order}" \
            "CUTEDSL_BENCHMARK_REPLICATE=${replicate_index}" \
            "CUTEDSL_BENCHMARK_PROFILE=${profile_enabled}" \
            "CUTEDSL_BENCHMARK_RESULT_ROOT=${RESULT_ROOT}" \
            "CUTEDSL_SHARED_HF_HOME=${CUTEDSL_SHARED_HF_HOME}" \
            "CUTEDSL_BENCHMARK_SUBMISSION_GROUP=${SUBMISSION_GROUP}" \
            "CUTEDSL_BENCHMARK_WARMUP_UPDATES=${WARMUP_UPDATES}" \
            "CUTEDSL_BENCHMARK_MEASURED_UPDATES=${MEASURED_UPDATES}" \
            "NEMO2606_FACTORIAL_CONTEXT=${context}" \
            "NEMO2606_FULL_CG_ENABLED=${full_cg_enabled}" \
            "NEMO2606_A2A_ENABLED=${a2a_enabled}" \
            "UV_NO_EDITABLE=1" \
            "SLURM_EXPORT_ENV=ALL" \
            > "${EXPORT_PAYLOAD}"

        job_id=$(sbatch --parsable "${sbatch_args[@]}" \
            "--job-name=${CUTEDSL_ACCOUNT}-n2606.${context}.r${replicate_index}" \
            "--export-file=${EXPORT_PAYLOAD}" \
            "${RAY_SUB}")
        record=$(printf \
            '{"factorial_context":"%s","full_cg_enabled":%s,"a2a_enabled":%s,"replicate_index":%d,"timing_order":"%s","profile_enabled":%s,"job_id":"%s","submission_group":"%s"}' \
            "${context}" "${full_cg_enabled}" "${a2a_enabled}" \
            "${replicate_index}" "${timing_order}" "${profile_enabled}" \
            "${job_id}" "${SUBMISSION_GROUP}")
        printf '%s\n' "${record}"
        if [[ "${TEST_ONLY}" == "0" ]]; then
            printf '%s\n' "${record}" >> "${submission_record}"
            printf '%s\n' "${record}" >> "${COHORT_SUBMISSION_TEMP}"
        fi
    done
done

if [[ "${TEST_ONLY}" == "1" ]]; then
    echo "[INFO] Scheduler/export preflighted ${#contexts[@]} contexts x ${REPLICATES} replicas; feature-source checks were skipped and no jobs were submitted."
else
    expected_cohort_records=$((${#contexts[@]} * REPLICATES))
    actual_cohort_records=$(wc -l < "${COHORT_SUBMISSION_TEMP}" | tr -d " ")
    if ((actual_cohort_records != expected_cohort_records)); then
        echo "[ERROR] Factorial cohort has ${actual_cohort_records}/${expected_cohort_records} records." >&2
        exit 1
    fi
    if [[ ! -f "${COHORT_SUBMISSION_TEMP}" || -L "${COHORT_SUBMISSION_TEMP}" || \
        -e "${COHORT_SUBMISSION}" || -L "${COHORT_SUBMISSION}" ]]; then
        echo "[ERROR] Factorial cohort finalization preconditions failed." >&2
        exit 1
    fi
    mv -- "${COHORT_SUBMISSION_TEMP}" "${COHORT_SUBMISSION}"
    echo "[INFO] Submitted ${#contexts[@]} contexts x ${REPLICATES} replicas."
    echo "[INFO] Collector cohort JSONL: ${COHORT_SUBMISSION}"
    echo "[INFO] Collect paired g0 contexts with collect_cutedsl_ab_replicates.py; ON-only g1 contexts require the dependency-constrained collector."
fi
