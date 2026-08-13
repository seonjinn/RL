#!/bin/bash

set -euo pipefail

readonly EXPECTED_RUNTIME_COMMIT=01398467224921c058a70702cb4a8285eb98fc71
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT to the immutable evidence commit}"
readonly EXPECTED_SOURCE_COMMIT
SOURCE_REMOTE=${SOURCE_REMOTE:-fork}
SOURCE_BRANCH=${SOURCE_BRANCH:-sna/pr2279-activation-offload-evidence-013984672-20260812}

ROOT=${ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-pr2279-013984672}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
CONTAINER=${CONTAINER:?Set CONTAINER to an immutable NeMo-RL nightly .sqsh}
HF_HOME=${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}
STEPS=${STEPS:-3}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
DRY_RUN=${DRY_RUN:-0}
ARM_FILTER=${ARM_FILTER-on}
RUN_LABEL=${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}
EXPERIMENT_ROOT="${ROOT}/experiments/pr2279_activation_offload_20260812"
VENV_ROOT="/lustre/fsw/coreai_dlalgo_llm/users/sna/venvs/pr2279-perf-${EXPECTED_RUNTIME_COMMIT:0:10}"

git -C "${ROOT}" pull --ff-only "${SOURCE_REMOTE}" "${SOURCE_BRANCH}"
git -C "${ROOT}" submodule update --init --recursive
actual_source_commit=$(git -C "${ROOT}" rev-parse HEAD)
if [[ "${actual_source_commit}" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
    echo "Expected source ${EXPECTED_SOURCE_COMMIT}, got ${actual_source_commit}" >&2
    exit 2
fi
if ! git -C "${ROOT}" merge-base --is-ancestor \
    "${EXPECTED_RUNTIME_COMMIT}" "${EXPECTED_SOURCE_COMMIT}"; then
    echo "Evidence commit is not based on PR runtime ${EXPECTED_RUNTIME_COMMIT}" >&2
    exit 2
fi
if ! git -C "${ROOT}" diff --quiet \
    "${EXPECTED_RUNTIME_COMMIT}" "${EXPECTED_SOURCE_COMMIT}" -- \
    . ':(exclude)experiments/pr2279_activation_offload_20260812/**'; then
    echo "Evidence branch changes NeMo-RL runtime files" >&2
    exit 2
fi
if ! git -C "${ROOT}" diff-index --quiet --ignore-submodules=untracked HEAD --; then
    echo "Tracked source changes are not allowed" >&2
    exit 2
fi

container_realpath=$(readlink -f "${CONTAINER}")
container_metadata="${container_realpath}.metadata.txt"
if [[ ! -f "${container_realpath}" || ! -f "${container_metadata}" ]]; then
    echo "Container or immutable metadata is missing: ${container_realpath}" >&2
    exit 2
fi

mkdir -p "${EXPERIMENT_ROOT}/logs"

submit_arm() {
    local arm=$1
    local config="${EXPERIMENT_ROOT}/configs/qwen30_${arm}.yaml"
    local log_dir="${EXPERIMENT_ROOT}/logs/${RUN_LABEL}-${arm}"
    local job_name="coreai_dlalgo_llm-sna.pr2279-q30-${arm}"
    local venv_root="${VENV_ROOT}-${arm}"
    local lifecycle_log="${log_dir}/lifecycle.log"
    local metrics_json="${log_dir}/metrics.json"
    local acceptance_json="${log_dir}/acceptance.json"
    local command

    mkdir -p "${log_dir}"
    printf -v command \
        'set -euo pipefail; cd %q; export NRL_IGNORE_VERSION_MISMATCH=1 NRL_FORCE_REBUILD_VENVS=false NVTE_CUDA_ARCHS=100 NEMO_RL_VENV_DIR=%q HF_HOME=%q HF_DATASETS_CACHE=%q/cache; uv run --frozen --extra mcore examples/run_grpo.py --config %q grpo.max_num_steps=%q logger.log_dir=%q logger.wandb_enabled=false logger.tensorboard_enabled=true 2>&1 | tee %q; uv run --no-sync python tests/json_dump_tb_logs.py %q --output_path %q; uv run --no-sync python %q --log %q --metrics %q --expected-steps %q --expected-world-size 16 --expect-offload %q --output %q' \
        "${ROOT}" \
        "${venv_root}" \
        "${HF_HOME}" \
        "${HF_HOME}" \
        "${config}" \
        "${STEPS}" \
        "${log_dir}/metrics" \
        "${lifecycle_log}" \
        "${log_dir}/metrics" \
        "${metrics_json}" \
        "${EXPERIMENT_ROOT}/scripts/check_lifecycle.py" \
        "${lifecycle_log}" \
        "${metrics_json}" \
        "${STEPS}" \
        "${arm}" \
        "${acceptance_json}"

    {
        echo "source_commit=${actual_source_commit}"
        echo "runtime_commit=${EXPECTED_RUNTIME_COMMIT}"
        echo "source_remote=${SOURCE_REMOTE}"
        echo "source_branch=${SOURCE_BRANCH}"
        echo "config=${config}"
        echo "config_sha256=$(sha256sum "${config}" | awk '{print $1}')"
        echo "container=${container_realpath}"
        cat "${container_metadata}"
        echo "container_metadata_sha256=$(sha256sum "${container_metadata}" | awk '{print $1}')"
        echo "uv_lock_sha256=$(sha256sum "${ROOT}/uv.lock" | awk '{print $1}')"
        git -C "${ROOT}" submodule status --recursive
        grep -m1 'TransformerEngine.git@' "${ROOT}/uv.lock" || true
        printf 'command=%q\n' "${command}"
    } >"${log_dir}/provenance.txt"

    CONTAINER="${CONTAINER}" \
        MOUNTS=/lustre:/lustre \
        GPUS_PER_NODE=4 \
        COMMAND="${command}" \
        BASE_LOG_DIR="${log_dir}" \
        sbatch --test-only \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --nodes=4 \
        --segment=4 \
        --time="${TIME_LIMIT}" \
        --comment=metrics \
        --job-name="${job_name}" \
        --output="${log_dir}/slurm-%j.out" \
        "${ROOT}/ray.sub"

    if [[ "${DRY_RUN}" == 1 ]]; then
        return
    fi

    CONTAINER="${CONTAINER}" \
        MOUNTS=/lustre:/lustre \
        GPUS_PER_NODE=4 \
        COMMAND="${command}" \
        BASE_LOG_DIR="${log_dir}" \
        sbatch --parsable \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --nodes=4 \
        --segment=4 \
        --time="${TIME_LIMIT}" \
        --comment=metrics \
        --job-name="${job_name}" \
        --output="${log_dir}/slurm-%j.out" \
        "${ROOT}/ray.sub"
}

for arm in off on; do
    if [[ -n "${ARM_FILTER}" && "${ARM_FILTER}" != "${arm}" ]]; then
        continue
    fi
    submit_arm "${arm}"
done
