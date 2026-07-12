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

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--test-only" ) ]]; then
    echo "Usage: $0 [--test-only]" >&2
    exit 2
fi

CONTEXTS="${NEMO2606_FACTORIAL_CONTEXTS:-g0a0,g1a0,g0a1,g1a1}"
REPLICATES="${NEMO2606_FACTORIAL_REPLICATES:-3}"
WARMUP_UPDATES="${NEMO2606_FACTORIAL_WARMUP_UPDATES:-5}"
MEASURED_UPDATES="${NEMO2606_FACTORIAL_MEASURED_UPDATES:-20}"
PROFILE_REPLICATE="${NEMO2606_FACTORIAL_PROFILE_REPLICATE:-0}"
readonly CONTEXTS REPLICATES WARMUP_UPDATES MEASURED_UPDATES PROFILE_REPLICATE

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

REPO_ROOT=$(git rev-parse --show-toplevel)
readonly REPO_ROOT
readonly EXPERIMENT_DIR="${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
readonly MATRIX_PAYLOAD="${EXPERIMENT_DIR}/run_cutedsl_matrix.sbatch"
readonly RAY_SUB="${REPO_ROOT}/ray.sub"
readonly RECIPE="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-2n4g-megatron-mxfp8-factorial.yaml"
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
sbatch_args+=(
    "--nodes=2"
    "--exclusive"
    "--time=${CUTEDSL_BENCHMARK_TIME}"
)
TEST_ONLY=0
if [[ ${1-} == "--test-only" ]]; then
    sbatch_args+=("--test-only")
    TEST_ONLY=1
fi
readonly TEST_ONLY

SUBMISSION_GROUP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly SUBMISSION_GROUP
readonly RESULT_ROOT="${EXPERIMENT_DIR}/results"
readonly RUNTIME_ROOT="${RESULT_ROOT}/multinode_runtime/${SUBMISSION_GROUP}"
readonly RAY_LOG_ROOT="${RESULT_ROOT}/ray_logs/${SUBMISSION_GROUP}"
readonly SUBMISSION_DIR="${RESULT_ROOT}/factorial/submissions"
GIT_COMMON_DIR=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
readonly GIT_COMMON_DIR
RAY_MOUNTS="${REPO_ROOT}:${REPO_ROOT}"
case "${GIT_COMMON_DIR}" in
    "${REPO_ROOT}"|"${REPO_ROOT}"/*) ;;
    *) RAY_MOUNTS+=",${GIT_COMMON_DIR}:${GIT_COMMON_DIR}" ;;
esac
RAY_MOUNTS+=",${CUTEDSL_IMAGE}:${CUTEDSL_IMAGE}"
readonly RAY_MOUNTS
readonly RUNTIME_CANARY="${RUNTIME_ROOT}/.shared_fs_canary"
readonly RAY_SETUP_COMMAND="test -r ${RUNTIME_CANARY} && grep -Fx ${CUTEDSL_SUBMISSION_GIT_SHA} ${RUNTIME_CANARY} && test \"\$(git -C ${REPO_ROOT} rev-parse HEAD)\" = ${CUTEDSL_SUBMISSION_GIT_SHA}"
EXPORT_PAYLOAD=$(mktemp "${TMPDIR:-/tmp}/nemo2606-factorial-export.XXXXXX")
readonly EXPORT_PAYLOAD
trap 'rm -f "${EXPORT_PAYLOAD}"' EXIT
chmod 600 "${EXPORT_PAYLOAD}"
if [[ "${TEST_ONLY}" == "0" ]]; then
    mkdir -p "${SUBMISSION_DIR}" "${RUNTIME_ROOT}" "${RAY_LOG_ROOT}"
    printf '%s\n' "${CUTEDSL_SUBMISSION_GIT_SHA}" > "${RUNTIME_CANARY}"
fi

IFS=',' read -r -a contexts <<< "${CONTEXTS}"
if [[ ${#contexts[@]} -eq 0 ]]; then
    echo "[ERROR] NEMO2606_FACTORIAL_CONTEXTS must not be empty." >&2
    exit 1
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

needs_full_cg="0"
needs_a2a="0"
for context in "${contexts[@]}"; do
    resolve_context "${context}"
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
    if grep -q "colocated generation/refit is not supported" "${full_cg_source}"; then
        echo "[ERROR] Requested full-CG contexts cannot use the 2x4 EP8 E2E allocation until colocated generation/refit is supported or generation GPUs are added." >&2
        exit 1
    fi
fi

for context in "${contexts[@]}"; do
    resolve_context "${context}"

    submission_record="${SUBMISSION_DIR}/${SUBMISSION_GROUP}-${context}.jsonl"
    for ((replicate_index = 0; replicate_index < REPLICATES; replicate_index++)); do
        if ((replicate_index % 2 == 0)); then
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
            -u BASE_LOG_DIR \
            -u GPUS_PER_NODE \
            -u CUTEDSL_BENCHMARK_EXISTING_RAY \
            -u CUTEDSL_BENCHMARK_ORDER \
            -u CUTEDSL_BENCHMARK_REPLICATE \
            -u CUTEDSL_BENCHMARK_PROFILE \
            -u CUTEDSL_BENCHMARK_SUBMISSION_GROUP \
            -u CUTEDSL_BENCHMARK_WARMUP_UPDATES \
            -u CUTEDSL_BENCHMARK_MEASURED_UPDATES \
            -u NEMO2606_FACTORIAL_CONTEXT \
            -u NEMO2606_FULL_CG_ENABLED \
            -u NEMO2606_A2A_ENABLED \
            -u SLURM_EXPORT_ENV \
            "CONTAINER=${CUTEDSL_IMAGE}" \
            "MOUNTS=${RAY_MOUNTS}" \
            "SETUP_COMMAND=${RAY_SETUP_COMMAND}" \
            "COMMAND=exec bash ${MATRIX_PAYLOAD}" \
            "BASE_LOG_DIR=${RAY_LOG_ROOT}" \
            "GPUS_PER_NODE=4" \
            "RAY_LOG_SYNC_FREQUENCY=5" \
            "CUTEDSL_BENCHMARK_EXISTING_RAY=1" \
            "CUTEDSL_BENCHMARK_RECIPE=${RECIPE}" \
            "CUTEDSL_BENCHMARK_NUM_NODES=2" \
            "CUTEDSL_BENCHMARK_GPUS_PER_NODE=4" \
            "CUTEDSL_BENCHMARK_RUNTIME_ROOT=${RUNTIME_ROOT}" \
            "CUTEDSL_BENCHMARK_ORDER=${timing_order}" \
            "CUTEDSL_BENCHMARK_REPLICATE=${replicate_index}" \
            "CUTEDSL_BENCHMARK_PROFILE=${profile_enabled}" \
            "CUTEDSL_BENCHMARK_SUBMISSION_GROUP=${SUBMISSION_GROUP}" \
            "CUTEDSL_BENCHMARK_WARMUP_UPDATES=${WARMUP_UPDATES}" \
            "CUTEDSL_BENCHMARK_MEASURED_UPDATES=${MEASURED_UPDATES}" \
            "NEMO2606_FACTORIAL_CONTEXT=${context}" \
            "NEMO2606_FULL_CG_ENABLED=${full_cg_enabled}" \
            "NEMO2606_A2A_ENABLED=${a2a_enabled}" \
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
        fi
    done
done

if [[ "${TEST_ONLY}" == "1" ]]; then
    echo "[INFO] Validated ${#contexts[@]} contexts x ${REPLICATES} replicas; no jobs submitted."
else
    echo "[INFO] Submitted ${#contexts[@]} contexts x ${REPLICATES} replicas."
    echo "[INFO] Collect each context with collect_cutedsl_ab_replicates.py and its context JSONL."
fi
