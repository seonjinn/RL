#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR_OVERRIDE:-$(realpath "${SCRIPT_DIR}/../..")}
source "${SCRIPT_DIR}/provenance.sh"

ACTION=${ACTION:-dry-run}
case "${ACTION}" in
    test-only|dry-run|submit) ;;
    *) echo "Unsupported ACTION: ${ACTION}" >&2; exit 2 ;;
esac

EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-b9eea5bbbec24a2af6acd0d92c02a3640a748e9c}
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
if [[ -n "${RUN_ID:-}" ]]; then RUN_ID=${RUN_ID}; elif [[ "${ACTION}" == submit ]]; then RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$; else RUN_ID=dry-run; fi
RUN_ROOT=${RUN_ROOT:-${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/shmoo/${RUN_ID}}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
VLLM_ENVIRONMENT_ROOT=${VLLM_ENVIRONMENT_ROOT:-}
DRIVER_VENV=${VLLM_ENVIRONMENT_ROOT:+${VLLM_ENVIRONMENT_ROOT}/vllm-canonical}
SHMOO_PYTHON=${SHMOO_PYTHON:-${DRIVER_VENV:+${DRIVER_VENV}/bin/python}}
SHMOO_PYTHON=${SHMOO_PYTHON:-python}
HF_MODEL_CACHE_DIR=${HF_MODEL_CACHE_DIR:-${WORK_ROOT}/hf/hub/models--Qwen--Qwen3-30B-A3B}
SELECTED_PROFILES=${SELECTED_PROFILES:-${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/selected_profiles.json}
STOCK_INPUT_CACHE_ROOT=${STOCK_INPUT_CACHE_ROOT:-${WORK_ROOT}/.cache/mxfp8-moe-tactic-audit/shmoo/stock-input}
SHMOO_OUTPUT_ROOT=${SHMOO_OUTPUT_ROOT:-${RUN_ROOT}}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
NSYS_CAPTURE_TACTICS=${NSYS_CAPTURE_TACTICS:-stock,winners}
SHMOO_WEIGHT_MODE=${SHMOO_WEIGHT_MODE:-prepacked}
REPLAY_MODE=${REPLAY_MODE:-routed}
PROFILE_LIMIT=${PROFILE_LIMIT:-}
TACTIC_LIMIT=${TACTIC_LIMIT:-}
TACTIC_PAIRS=${TACTIC_PAIRS:-}
REPETITIONS=${REPETITIONS:-10}
PAIR_ONLY=${PAIR_ONLY:-1}
case "${SHMOO_WEIGHT_MODE}" in
    prepacked|synthetic) ;;
    *) echo "Unsupported SHMOO_WEIGHT_MODE: ${SHMOO_WEIGHT_MODE}" >&2; exit 2 ;;
esac
case "${REPLAY_MODE}" in
    routed) REPLAY_ARGUMENT= ;;
    monolithic) REPLAY_ARGUMENT=--monolithic-replay ;;
    *) echo "Unsupported REPLAY_MODE: ${REPLAY_MODE}" >&2; exit 2 ;;
esac
for value_name in PROFILE_LIMIT TACTIC_LIMIT; do
    value=${!value_name}
    if [[ -n "${value}" && ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer: ${value}" >&2
        exit 2
    fi
done
if [[ ! "${REPETITIONS}" =~ ^[1-9][0-9]*$ || "${REPETITIONS}" -lt 10 ]]; then
    echo "REPETITIONS must be an integer of at least 10: ${REPETITIONS}" >&2
    exit 2
fi
PROFILE_LIMIT_ARGUMENT=${PROFILE_LIMIT:+--profile-limit ${PROFILE_LIMIT}}
TACTIC_LIMIT_ARGUMENT=${TACTIC_LIMIT:+--tactic-limit ${TACTIC_LIMIT}}
TACTIC_PAIR_ARGUMENTS=
if [[ -n "${TACTIC_PAIRS}" ]]; then
    IFS=';' read -r -a tactic_pairs <<< "${TACTIC_PAIRS}"
    for tactic_pair in "${tactic_pairs[@]}"; do
        [[ "${tactic_pair}" =~ ^[0-9]+,[0-9]+$ ]] || {
            echo "TACTIC_PAIRS entries must use GEMM1,GEMM2: ${tactic_pair}" >&2
            exit 2
        }
        TACTIC_PAIR_ARGUMENTS+=" --tactic-pair ${tactic_pair}"
    done
fi
case "${PAIR_ONLY}" in
    0) PAIR_ONLY_ARGUMENT= ;;
    1) PAIR_ONLY_ARGUMENT=--pair-only ;;
    *) echo "PAIR_ONLY must be 0 or 1: ${PAIR_ONLY}" >&2; exit 2 ;;
esac
if [[ "${REPLAY_MODE}" == monolithic ]]; then
    [[ "${SHMOO_WEIGHT_MODE}" == prepacked ]] || {
        echo "monolithic replay requires SHMOO_WEIGHT_MODE=prepacked" >&2
        exit 2
    }
    [[ "${PAIR_ONLY}" == 1 ]] || {
        echo "monolithic replay requires PAIR_ONLY=1" >&2
        exit 2
    }
    [[ -n "${PROFILE_LIMIT}" && -n "${TACTIC_LIMIT}" ]] || {
        echo "monolithic replay requires explicit PROFILE_LIMIT and TACTIC_LIMIT" >&2
        exit 2
    }
fi
if [[ "${ACTION}" == submit ]]; then
    audit_prepare_submit "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" "${EXPECTED_VLLM_COMMIT}"
fi

MODEL_SNAPSHOT=dry-run-not-validated
MODEL_REVISION=dry-run-not-validated
if [[ "${ACTION}" != dry-run ]]; then
    IFS=$'\t' read -r MODEL_SNAPSHOT MODEL_REVISION < <(
        audit_resolve_model_snapshot "${HF_MODEL_CACHE_DIR}" 16
    )
    [[ -s "${SELECTED_PROFILES}" ]] || { echo "Missing selected profiles: ${SELECTED_PROFILES}" >&2; exit 1; }
    [[ -f "${CONTAINER}" ]] || { echo "Missing container: ${CONTAINER}" >&2; exit 1; }
    if [[ -n "${VLLM_ENVIRONMENT_ROOT}" ]]; then
        [[ -f "${VLLM_ENVIRONMENT_ROOT}/READY" ]] || {
            echo "Prepared vLLM environment marker is missing: ${VLLM_ENVIRONMENT_ROOT}/READY" >&2
            exit 1
        }
    fi
    [[ "${SHMOO_OUTPUT_ROOT}" == "${RUN_ROOT}" || "${SHMOO_OUTPUT_ROOT}" == "${RUN_ROOT}/"* ]] || {
        echo "SHMOO_OUTPUT_ROOT must be inside RUN_ROOT" >&2
        exit 1
    }
fi
PREPACKED_WEIGHT_ROOT=${PREPACKED_WEIGHT_ROOT:-${WORK_ROOT}/.cache/mxfp8-moe-tactic-audit/prepacked/stock-v0251-20260808}
MOE_WEIGHTS=${MOE_WEIGHTS:-${PREPACKED_WEIGHT_ROOT}/flashinfer_mxfp8_moe_prepacked_v1.pt}
if [[ "${SHMOO_WEIGHT_MODE}" == prepacked ]]; then
    WEIGHT_ARGUMENT="--weights ${MOE_WEIGHTS}"
    STOCK_CACHE_ARGUMENT="--stock-cache ${STOCK_INPUT_CACHE_ROOT}/autotune_configs.json"
    MANIFEST_CACHE_ROOT=${STOCK_INPUT_CACHE_ROOT}
    if [[ "${ACTION}" != dry-run ]]; then
        audit_require_nonempty_dir "${STOCK_INPUT_CACHE_ROOT}"
        [[ -s "${STOCK_INPUT_CACHE_ROOT}/autotune_configs.json" ]] || {
            echo "Missing stock tactic cache: ${STOCK_INPUT_CACHE_ROOT}/autotune_configs.json" >&2
            exit 1
        }
        [[ -s "${MOE_WEIGHTS}" ]] || {
            echo "Missing prepacked MoE weights: ${MOE_WEIGHTS}" >&2
            exit 1
        }
    fi
else
    WEIGHT_ARGUMENT=--synthetic-smoke
    STOCK_CACHE_ARGUMENT=
    MANIFEST_CACHE_ROOT=-
fi
if [[ "${ACTION}" == submit ]]; then
    [[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 1; }
fi
NEMO_RL_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD)

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO_DIR}
runtime_nemo_rl_commit=\$(git rev-parse HEAD)
runtime_vllm_commit=\$(git -C ${CUSTOM_VLLM_ROOT} rev-parse HEAD)
[[ "\${runtime_nemo_rl_commit}" == "${NEMO_RL_COMMIT}" ]]
[[ "\${runtime_vllm_commit}" == "${EXPECTED_VLLM_COMMIT}" ]]
if [[ "${SHMOO_WEIGHT_MODE}" == prepacked ]]; then
  export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${STOCK_INPUT_CACHE_ROOT}
else
  unset VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
fi
export VIRTUAL_ENV=${DRIVER_VENV}
export PATH=$(dirname "${SHMOO_PYTHON}"):\${PATH}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MXFP8_MOE_CUDA_GRAPH_REPLAY=required
export MXFP8_MOE_NSYS_CAPTURE_TACTICS=${NSYS_CAPTURE_TACTICS}
mkdir -p ${SHMOO_OUTPUT_ROOT}
printf 'cuda_graph_replay=required\\n'
printf 'crash_rows_preserved=true\\n'
printf 'nsys_capture_tactics=%s\\n' "\${MXFP8_MOE_NSYS_CAPTURE_TACTICS}"
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --force-overwrite=true --output ${RUN_ROOT}/nsys-selected \\
  ${SHMOO_PYTHON} experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py \\
  --profiles ${SELECTED_PROFILES} \\
  ${WEIGHT_ARGUMENT} \\
  ${STOCK_CACHE_ARGUMENT} \\
  ${PROFILE_LIMIT_ARGUMENT} \\
  ${TACTIC_LIMIT_ARGUMENT} \\
  ${TACTIC_PAIR_ARGUMENTS} \\
  ${PAIR_ONLY_ARGUMENT} \\
  ${REPLAY_ARGUMENT} \\
  --warmups 3 \\
  --repetitions ${REPETITIONS} \\
  --output ${SHMOO_OUTPUT_ROOT}/measurements.jsonl
nsys stats --quiet --report nvtx_gpu_proj_sum --format csv --output - \\
  ${RUN_ROOT}/nsys-selected.nsys-rep > ${RUN_ROOT}/nsys-nvtx.csv
${SHMOO_PYTHON} experiments/mxfp8_moe_tactic_audit/nsys_to_component_csv.py \\
  --nvtx-csv ${RUN_ROOT}/nsys-nvtx.csv \\
  --output ${SHMOO_OUTPUT_ROOT}/nsys_components.csv
EOF
)

SBATCH_ARGS=(
    --nodes=1
    --ntasks=1
    --exclusive
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --time=05:00:00
    --job-name="coreai_dlalgo_llm-mxmoe.shmoo-${RUN_ID}"
    --output="${RUN_ROOT}/slurm-%j.out"
)
if [[ -n "${QOS}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi

printf 'action=%s\n' "${ACTION}"
printf 'run_root=%s\n' "${RUN_ROOT}"
printf 'shmoo_weight_mode=%s\n' "${SHMOO_WEIGHT_MODE}"
printf 'replay_mode=%s\n' "${REPLAY_MODE}"
printf 'pair_only=%s\n' "${PAIR_ONLY}"
printf 'stock_input_cache_root=%s\n' "${STOCK_INPUT_CACHE_ROOT}"
printf 'job_script=%s\n' "${SCRIPT_DIR}/single_gpu.sub"
printf 'CUDA Graph replay required\n'
printf 'NSys captures selected winners plus stock\n'
printf 'sbatch_args='; printf ' %q' "${SBATCH_ARGS[@]}"; printf '\n'
printf '%s\n' "${COMMAND}"

case "${ACTION}" in
    dry-run) ;;
    test-only)
        CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=1 \
            BASE_LOG_DIR="${RUN_ROOT}" sbatch --test-only "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/single_gpu.sub"
        ;;
    submit)
        audit_write_manifest "${RUN_ROOT}" shmoo "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" \
            "${EXPECTED_VLLM_COMMIT}" "${CONTAINER}" \
            examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml \
            "${MODEL_SNAPSHOT}" "${MANIFEST_CACHE_ROOT}" "${SCRIPT_DIR}" \
            "${SCRIPT_DIR}/submit_shmoo_ptyche.sh" "${SCRIPT_DIR}/provenance.sh" \
            "${SCRIPT_DIR}/single_gpu.sub" \
            "${SCRIPT_DIR}/shmoo_moe_tactics.py" "${SCRIPT_DIR}/nsys_to_component_csv.py" \
            "${SELECTED_PROFILES}"
        CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=1 \
            BASE_LOG_DIR="${RUN_ROOT}" sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/single_gpu.sub"
        ;;
esac
