#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)

ACTION=${ACTION:-test-only}
MODEL=${MODEL:-qwen30b}
LINEAR_BACKEND=${LINEAR_BACKEND:-flashinfer_cutlass}
ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
PARTITION=${PARTITION:-batch}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MAX_STEPS=${MAX_STEPS:-5}
WALLTIME=${WALLTIME:-04:00:00}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
RUNTIME_ROOT=${RUNTIME_ROOT:-${WORK_ROOT}/containers/nemo-rl-nightly-refresh}
CONTAINER=${CONTAINER:-${RUNTIME_ROOT}/nemo_rl_nightly_20260730_483099.sqsh}
PYTHON_OVERLAY=${PYTHON_OVERLAY:-${RUNTIME_ROOT}/python-overlay-483099}
ROOT_CACHE_OVERLAY=${ROOT_CACHE_OVERLAY:-${RUNTIME_ROOT}/root-cache-overlay-483099}
WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3478-cutedsl-linear-ab}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
WANDB_ENABLED=${WANDB_ENABLED:-true}

case "${LINEAR_BACKEND}" in
  flashinfer_cutlass|flashinfer_cutedsl) ;;
  *)
    echo "LINEAR_BACKEND must be flashinfer_cutlass or flashinfer_cutedsl" >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  qwen30b)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-qkvo-rollout.yaml
    TOTAL_NODES=${TOTAL_NODES:-4}
    MODEL_OVERRIDES=(
      cluster.segment_size=1
      policy.generation.colocated.enabled=false
      policy.generation.colocated.resources.num_nodes=2
      policy.generation.colocated.resources.gpus_per_node="${GPUS_PER_NODE}"
      policy.generation.refit_transport=nccl_reshard
      policy.megatron_cfg.expert_tensor_parallel_size=1
      policy.generation.vllm_cfg.tensor_parallel_size=1
      policy.generation.vllm_cfg.pipeline_parallel_size=1
      policy.generation.vllm_cfg.expert_parallel_size=1
      policy.train_global_batch_size=2048
    )
    ;;
  qwen235b)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-mxfp8-qkvo-rollout.yaml
    TOTAL_NODES=${TOTAL_NODES:-8}
    MODEL_OVERRIDES=(
      cluster.segment_size="${TOTAL_NODES}"
      policy.generation.colocated.enabled=true
      policy.generation.refit_transport=nccl_reshard
      policy.megatron_cfg.moe_token_dispatcher_type=alltoall
      policy.megatron_cfg.moe_flex_dispatcher_backend=deepep
    )
    ;;
  *)
    echo "MODEL must be qwen30b or qwen235b" >&2
    exit 2
    ;;
esac

case "${ACTION}" in
  submit) SBATCH_ACTION=() ;;
  test-only) SBATCH_ACTION=(--test-only) ;;
  *)
    echo "ACTION must be submit or test-only" >&2
    exit 2
    ;;
esac

if [[ "${GPUS_PER_NODE}" != 8 ]]; then
  echo "GCP-NRT B200 experiments require GPUS_PER_NODE=8" >&2
  exit 2
fi

git -C "${REPO}" pull --ff-only
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"

REPO_SHA=$(git -C "${REPO}" rev-parse HEAD)
for path in \
  "${REPO}/${CONFIG}" \
  "${REPO}/ray.sub" \
  "${CONTAINER}" \
  "${PYTHON_OVERLAY}" \
  "${ROOT_CACHE_OVERLAY}"; do
  test -e "${path}"
done

ARM=${MODEL}-${LINEAR_BACKEND}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/mopd_nano_fast/.cache/pr3478-cutedsl-linear-ab/${ARM}}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${WORK_ROOT}/experiments/pr3478-cutedsl-linear-ab/results/${ARM}-${MAX_STEPS}step-${RUN_SUFFIX}}
WANDB_NAME=${WANDB_NAME:-${ARM}-${MAX_STEPS}step-${RUN_SUFFIX}}
WANDB_KEY_FILE=${CACHE_ROOT}/.wandb_key

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}" "${WORK_ROOT}/.cache/huggingface"
WANDB_EXPORT=
if [[ "${WANDB_ENABLED}" == true ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    (umask 077; printf '%s\n' "${WANDB_API_KEY}" >"${WANDB_KEY_FILE}")
  elif [[ -s "${WANDB_KEY_FILE}" && $(wc -c <"${WANDB_KEY_FILE}") -ge 20 ]]; then
    :
  else
    if [[ -f "${HOME}/.bashrc" ]]; then
      WANDB_API_KEY=$(bash -lc \
        'source ~/.bashrc >/dev/null 2>&1 || true; printf %s "${WANDB_API_KEY:-}"')
    fi
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      (umask 077; printf '%s\n' "${WANDB_API_KEY}" >"${WANDB_KEY_FILE}")
    elif [[ -f "${HOME}/.netrc" ]]; then
      (umask 077; awk '
        {
          for (i = 1; i <= NF; i++) {
            if ($i == "machine" && $(i + 1) == "api.wandb.ai") found = 1
            if (found && $i == "password") { print $(i + 1); exit }
          }
        }
      ' "${HOME}/.netrc" >"${WANDB_KEY_FILE}")
    fi
  fi
  if [[ ! -s "${WANDB_KEY_FILE}" || $(wc -c <"${WANDB_KEY_FILE}") -lt 20 ]]; then
    echo "A valid W&B API key was not found in the environment, .bashrc, or .netrc" >&2
    exit 2
  fi
  WANDB_EXPORT="export WANDB_API_KEY=\"\$(cat ${WANDB_KEY_FILE})\""
fi

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HF_HOME=${WORK_ROOT}/.cache/huggingface
export NRL_FORCE_REBUILD_VENVS=false
export NEMO_RL_VENV_DIR=/tmp/nemo-rl-pr3478-cutedsl-${ARM}-workers
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${REPO}
export TORCH_CUDA_ARCH_LIST=10.0
export UV_CACHE_DIR=/root/.cache/uv
export UV_PROJECT_ENVIRONMENT=${CACHE_ROOT}/driver-venv
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python
export UV_LOCK_TIMEOUT=7200
${WANDB_EXPORT}
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' "\$(git rev-parse HEAD)"
uv run --frozen examples/run_grpo.py \
  --config ${CONFIG} \
  cluster.num_nodes=${TOTAL_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  ${MODEL_OVERRIDES[*]} \
  ++policy.generation.vllm_kwargs.linear_backend=${LINEAR_BACKEND} \
  policy.generation.vllm_cfg.use_tqdm=false \
  grpo.max_num_steps=${MAX_STEPS} \
  grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  checkpointing.enabled=false \
  logger.log_dir=${EXPERIMENT_ROOT}/logs \
  logger.wandb_enabled=${WANDB_ENABLED} \
  ++logger.wandb.entity=${WANDB_ENTITY} \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.name=${WANDB_NAME}
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre,${PYTHON_OVERLAY}:/root/.local/share/uv/python,${ROOT_CACHE_OVERLAY}:/root/.cache
export CONTAINER_REMAP_ROOT=1
export COMMAND
export GPUS_PER_NODE
export BASE_LOG_DIR=${EXPERIMENT_ROOT}

SBATCH_ARGS=(
  --nodes="${TOTAL_NODES}"
  --gpus-per-node="${GPUS_PER_NODE}"
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --job-name="sna-cute-${MODEL}-${LINEAR_BACKEND#flashinfer_}"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"CuTeDSL versus CUTLASS MXFP8 linear A/B"}}'
)

printf 'repo=%s\nsha=%s\nconfig=%s\nbackend=%s\nresult=%s\n' \
  "${REPO}" "${REPO_SHA}" "${CONFIG}" "${LINEAR_BACKEND}" "${EXPERIMENT_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
