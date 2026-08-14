#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)

ACTION=${ACTION:-test-only}
MODEL=${MODEL:-qwen30b}
LINEAR_BACKEND=${LINEAR_BACKEND:-flashinfer_cutlass}
MAX_STEPS=${MAX_STEPS:-2}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
TOTAL_NODES=${TOTAL_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
WALLTIME=${WALLTIME:-05:00:00}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_main_f25d.sqsh}
HF_HOME=${HF_HOME:-${WORK_ROOT}/nemo-rl-internal/hf_home}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${WORK_ROOT}/experiments/mxfp8-linear-backend-ab}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
WANDB_PROJECT=${WANDB_PROJECT:-sna-mxfp8-linear-backend-ab}
WANDB_ENABLED=${WANDB_ENABLED:-true}

VLLM_REPO_URL=${VLLM_REPO_URL:-https://github.com/seonjinn/vllm.git}
VLLM_REF=${VLLM_REF:-sna/mxfp8-refit-safe-linear-backends-v0251}
VLLM_COMMIT=${VLLM_COMMIT:-a76062edee3a3ac23d47a93c7ce466f06a19111f}
VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_aarch64.whl}

QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}
NANO_MODEL_PATH=${NANO_MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/guyueh/hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/snapshots/97ab8012882a655dc38df4fee47422aca9caca07}
NANO_TOKENIZER_PATH=${NANO_TOKENIZER_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/guyueh/hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848}

case "${ACTION}" in
  submit) SBATCH_ACTION=() ;;
  test-only) SBATCH_ACTION=(--test-only) ;;
  *) echo "ACTION must be submit or test-only" >&2; exit 2 ;;
esac

case "${LINEAR_BACKEND}" in
  flashinfer_cutlass|flashinfer_cutedsl|flashinfer_trtllm) ;;
  *) echo "Unsupported LINEAR_BACKEND=${LINEAR_BACKEND}" >&2; exit 2 ;;
esac

case "${MODEL}" in
  qwen30b)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-qkvo-rollout.yaml
    MODEL_OVERRIDES=(
      policy.model_name="${QWEN_MODEL_PATH}"
      policy.tokenizer.name="${QWEN_MODEL_PATH}"
    )
    ;;
  nano)
    if [[ "${LINEAR_BACKEND}" == flashinfer_trtllm ]]; then
      echo "Nano dense K=2688 is not divisible by TRTLLM's required 256; refusing a hidden CuTeDSL fallback." >&2
      exit 2
    fi
    CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-4n4g-mxfp8-qkvo-rollout.yaml
    MODEL_OVERRIDES=(
      policy.model_name="${NANO_MODEL_PATH}"
      policy.tokenizer.name="${NANO_TOKENIZER_PATH}"
    )
    ;;
  *) echo "MODEL must be qwen30b or nano" >&2; exit 2 ;;
esac

if [[ "${TOTAL_NODES}" != 4 || "${GPUS_PER_NODE}" != 4 ]]; then
  echo "This matched Lyris experiment requires TOTAL_NODES=4 and GPUS_PER_NODE=4." >&2
  exit 2
fi

git -C "${REPO}" pull --ff-only
if [[ -n $(git -C "${REPO}" status --porcelain --untracked-files=no) ]]; then
  echo "Tracked files in ${REPO} are dirty; commit or restore them before submission." >&2
  exit 2
fi

VLLM_DIR=${REPO}/3rdparty/vllm
if [[ ! -d "${VLLM_DIR}/.git" ]]; then
  git clone --filter=blob:none --branch "${VLLM_REF}" "${VLLM_REPO_URL}" "${VLLM_DIR}"
fi
if [[ $(git -C "${VLLM_DIR}" rev-parse HEAD) != "${VLLM_COMMIT}" ]]; then
  echo "${VLLM_DIR} must be pinned at ${VLLM_COMMIT}." >&2
  exit 2
fi

for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" "${HF_HOME}"; do
  test -e "${path}"
done
for path in "${MODEL_OVERRIDES[@]#*=}"; do
  test -e "${path}"
done

ARM=${MODEL}-${LINEAR_BACKEND}
RUN_NAME=${RUN_NAME:-${ARM}-${MAX_STEPS}step-${RUN_SUFFIX}}
RUN_ROOT=${EXPERIMENT_ROOT}/results/${RUN_NAME}
CACHE_ROOT=${EXPERIMENT_ROOT}/cache/${ARM}
mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}"

if [[ "${WANDB_ENABLED}" == true && -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY=$(bash -lc 'source ~/.bashrc >/dev/null 2>&1 || true; printf %s "${WANDB_API_KEY:-}"')
fi
if [[ "${WANDB_ENABLED}" == true && -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
  WANDB_API_KEY=$(awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "machine" && (i + 1) <= NF) {
          in_wandb = ($(i + 1) == "api.wandb.ai")
        } else if (in_wandb && $i == "password" && (i + 1) <= NF) {
          print $(i + 1)
          exit
        }
      }
    }
  ' "${HOME}/.netrc")
fi
if [[ "${WANDB_ENABLED}" == true && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required when WANDB_ENABLED=true." >&2
  exit 2
fi

REPO_SHA=$(git -C "${REPO}" rev-parse HEAD)
COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HF_HOME=${HF_HOME}
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=/tmp/nemo-rl-mxfp8-linear-${ARM}
export UV_CACHE_DIR=${CACHE_ROOT}/uv
export UV_PROJECT_ENVIRONMENT=${CACHE_ROOT}/driver-venv
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python
export UV_LOCK_TIMEOUT=7200
export VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION}
export WANDB_API_KEY='${WANDB_API_KEY:-}'
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' "\$(git rev-parse HEAD)"
printf 'VLLM_SOURCE_COMMIT=%s\n' "\$(git -C 3rdparty/vllm rev-parse HEAD)"
uv run --frozen examples/run_grpo.py \
  --config ${CONFIG} \
  ${MODEL_OVERRIDES[*]} \
  ++policy.generation.vllm_kwargs.linear_backend=${LINEAR_BACKEND} \
  grpo.max_num_steps=${MAX_STEPS} \
  grpo.seed=42 \
  grpo.val_at_start=false \
  grpo.val_at_end=false \
  loss_fn.force_on_policy_ratio=false \
  loss_fn.use_importance_sampling_correction=true \
  checkpointing.enabled=false \
  logger.log_dir=${RUN_ROOT}/logs \
  logger.wandb_enabled=${WANDB_ENABLED} \
  ++logger.wandb.entity=${WANDB_ENTITY} \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.name=${RUN_NAME}
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre
export CONTAINER_REMAP_ROOT=1
export COMMAND
export GPUS_PER_NODE
export BASE_LOG_DIR=${RUN_ROOT}

SBATCH_ARGS=(
  --nodes="${TOTAL_NODES}"
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --job-name="${ACCOUNT}-mxfp8-linear.${MODEL}-${LINEAR_BACKEND#flashinfer_}"
  --output="${RUN_ROOT}/slurm-%j.out"
)

printf 'repo=%s\nrepo_sha=%s\nvllm_sha=%s\nmodel=%s\nbackend=%s\nconfig=%s\nrun_root=%s\n' \
  "${REPO}" "${REPO_SHA}" "${VLLM_COMMIT}" "${MODEL}" "${LINEAR_BACKEND}" "${CONFIG}" "${RUN_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
