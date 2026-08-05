#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)

ACTION=${ACTION:-test-only}
QUANT_MODE=${QUANT_MODE:?QUANT_MODE is required: w4a16 or w4a4}
TRANSPORT=${TRANSPORT:?TRANSPORT is required: legacy or nccl}
ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-04:00:00}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
RUNTIME_ROOT=${RUNTIME_ROOT:-${WORK_ROOT}/containers/nemo-rl-nightly-refresh}
CONTAINER=${CONTAINER:-${RUNTIME_ROOT}/nemo_rl_nightly_20260730_483099.sqsh}
PYTHON_OVERLAY=${PYTHON_OVERLAY:-${RUNTIME_ROOT}/python-overlay-483099}
ROOT_CACHE_OVERLAY=${ROOT_CACHE_OVERLAY:-${RUNTIME_ROOT}/root-cache-overlay-483099}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-${WORK_ROOT}/experiments/bf16-nvfp4-rollout}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${CAMPAIGN_ROOT}/results/${QUANT_MODE}-${TRANSPORT}-${RUN_SUFFIX}}
CACHE_BASE=${CACHE_BASE:-${WORK_ROOT}/mopd_nano_fast/.cache/bf16-nvfp4-rollout}
WANDB_PROJECT=${WANDB_PROJECT:-sna-bf16-nvfp4-rollout}
WANDB_NAME=${WANDB_NAME:-${QUANT_MODE}-${TRANSPORT}-${RUN_SUFFIX}}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}

case "${ACTION}" in
  submit) SBATCH_ACTION=() ;;
  test-only) SBATCH_ACTION=(--test-only) ;;
  *) echo "ACTION must be submit or test-only" >&2; exit 2 ;;
esac

case "${QUANT_MODE}" in
  w4a16)
    TEST_SCRIPT=tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.sh
    ;;
  w4a4)
    TEST_SCRIPT=tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.sh
    : "${NVFP4_CALIBRATION_ARTIFACT:?NVFP4_CALIBRATION_ARTIFACT is required for w4a4}"
    test -f "${NVFP4_CALIBRATION_ARTIFACT}"
    ;;
  *) echo "QUANT_MODE must be w4a16 or w4a4" >&2; exit 2 ;;
esac

case "${TRANSPORT}" in
  legacy)
    REFIT_TRANSPORT=null
    SEGMENT_SIZE=2
    ;;
  nccl)
    REFIT_TRANSPORT=nccl_reshard
    SEGMENT_SIZE=1
    ;;
  *) echo "TRANSPORT must be legacy or nccl" >&2; exit 2 ;;
esac

git -C "${REPO}" pull --ff-only
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
if git -C "${REPO}" submodule status --recursive | grep -q '^-'; then
  echo "All pinned submodules must be initialized" >&2
  exit 2
fi

REPO_SHA=$(git -C "${REPO}" rev-parse HEAD)
CACHE_ROOT=${CACHE_ROOT:-${CACHE_BASE}/${REPO_SHA:0:9}/${QUANT_MODE}-${TRANSPORT}}
for path in \
  "${REPO}/${TEST_SCRIPT}" \
  "${REPO}/ray.sub" \
  "${CONTAINER}" \
  "${PYTHON_OVERLAY}" \
  "${ROOT_CACHE_OVERLAY}"; do
  test -e "${path}"
done

SNAPSHOT_GROUP=code_snapshots_nvfp4/${REPO_SHA:0:9}-${RUN_SUFFIX}-${QUANT_MODE}-${TRANSPORT}
SNAPSHOT_REPO=$(
  CODE_SNAPSHOT_DIRNAME="${SNAPSHOT_GROUP}" \
    bash "${REPO}/tools/code_snapshot.sh" "${QUANT_MODE}-${TRANSPORT}"
)

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}" "${WORK_ROOT}/.cache/huggingface"
WANDB_KEY_FILE=${CACHE_ROOT}/.wandb_key
if [[ -f "${HOME}/.netrc" ]]; then
  (umask 077; awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "machine" && $(i + 1) == "api.wandb.ai") found = 1
        if (found && $i == "password") { print $(i + 1); exit }
      }
    }
  ' "${HOME}/.netrc" >"${WANDB_KEY_FILE}")
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
  (umask 077; printf '%s\n' "${WANDB_API_KEY}" >"${WANDB_KEY_FILE}")
fi
test -s "${WANDB_KEY_FILE}"

CALIBRATION_EXPORT=""
if [[ "${QUANT_MODE}" == w4a4 ]]; then
  CALIBRATION_EXPORT="export NVFP4_CALIBRATION_ARTIFACT=${NVFP4_CALIBRATION_ARTIFACT}"
fi

cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
quant_mode=${QUANT_MODE}
transport=${TRANSPORT}
refit_transport=${REFIT_TRANSPORT}
repo_sha=${REPO_SHA}
snapshot_repo=${SNAPSHOT_REPO}
test_script=${TEST_SCRIPT}
container=${CONTAINER}
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
calibration_artifact=${NVFP4_CALIBRATION_ARTIFACT:-}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${SNAPSHOT_REPO}
export HF_HOME=${WORK_ROOT}/.cache/huggingface
export HF_DATASETS_CACHE=${WORK_ROOT}/.cache/huggingface/datasets
export NRL_FORCE_REBUILD_VENVS=false
export NEMO_RL_VENV_DIR=${CACHE_ROOT}/worker-venvs
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${SNAPSHOT_REPO}
export TORCH_CUDA_ARCH_LIST=10.0
export UV_CACHE_DIR=/root/.cache/uv
export UV_PROJECT_ENVIRONMENT=${CACHE_ROOT}/driver-venv
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python
export UV_LOCK_TIMEOUT=7200
export WANDB_API_KEY="\$(cat ${WANDB_KEY_FILE})"
export WANDB_PROJECT_OVERRIDE=${WANDB_PROJECT}
export WANDB_NAME_OVERRIDE=${WANDB_NAME}
export REFIT_TRANSPORT=${REFIT_TRANSPORT}
export SCHEDULER_SEGMENT_SIZE=${SEGMENT_SIZE}
${CALIBRATION_EXPORT}
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' '${REPO_SHA}'
uv run --frozen pytest -q \
  tests/test_nvfp4_rollout_recipes.py \
  tests/unit/modelopt/test_calibration_artifact.py::test_normalize_quant_cfg_identity_resolves_project_relative_path \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_configure_quant_engine_kwargs_for_real_quant
uv run --frozen bash ${TEST_SCRIPT} \
  ++git_meta=${REPO_SHA:0:9} \
  ++container=${CONTAINER} \
  ++logger.wandb.entity=${WANDB_ENTITY}
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre,${PYTHON_OVERLAY}:/root/.local/share/uv/python,${ROOT_CACHE_OVERLAY}:/root/.cache
export CONTAINER_REMAP_ROOT=1
export COMMAND
export GPUS_PER_NODE=8
export BASE_LOG_DIR=${EXPERIMENT_ROOT}

SBATCH_ARGS=(
  --nodes=2
  --gpus-per-node=8
  --exclusive
  --segment="${SEGMENT_SIZE}"
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --job-name="sna-nvfp4-${QUANT_MODE}-${TRANSPORT}"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"environment and Qwen3-30B NVFP4 initialization"}}'
)

printf 'mode=%s\ntransport=%s\nsha=%s\nsnapshot=%s\nresult=%s\n' \
  "${QUANT_MODE}" "${TRANSPORT}" "${REPO_SHA}" "${SNAPSHOT_REPO}" "${EXPERIMENT_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${SNAPSHOT_REPO}/ray.sub"
