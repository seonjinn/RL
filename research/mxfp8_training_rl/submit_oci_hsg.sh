#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
MODEL=${MODEL:-qwen30}
TRAINING_PRECISION=${TRAINING_PRECISION:-mxfp8}
ROLLOUT_PRECISION=${ROLLOUT_PRECISION:-mxfp8}
MAX_STEPS=${MAX_STEPS:-2}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
WALLTIME=${WALLTIME:-04:00:00}
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-false}
NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-100}

case "${TRAINING_PRECISION}" in
  bf16|mxfp8) ;;
  *)
    echo "TRAINING_PRECISION must be bf16 or mxfp8" >&2
    exit 2
    ;;
esac

case "${ROLLOUT_PRECISION}" in
  bf16|mxfp8) ;;
  *)
    echo "ROLLOUT_PRECISION must be bf16 or mxfp8" >&2
    exit 2
    ;;
esac

case "${MODEL}:${TRAINING_PRECISION}:${ROLLOUT_PRECISION}" in
  qwen30:mxfp8:mxfp8)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-e2e-fp8param-false.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    ;;
  qwen30:bf16:mxfp8)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-bf16-train-mxfp8-rollout.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    ;;
  nano:mxfp8:mxfp8)
    CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-mxfp8-e2e-fp8param-false.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    ;;
  nano:bf16:mxfp8)
    CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-bf16-train-mxfp8-rollout.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    ;;
  qwen30:mxfp8:bf16)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-train-bf16-rollout.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    ;;
  qwen30:bf16:bf16)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-bf16-rollout.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    ;;
  nano:mxfp8:bf16)
    CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-mxfp8-train-bf16-rollout.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    ;;
  nano:bf16:bf16)
    CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-bf16-rollout.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    ;;
  *)
    echo "MODEL must be qwen30 or nano" >&2
    exit 2
    ;;
esac

if [[ "${ACTION}" == render ]]; then
  printf 'model=%s\ntraining_precision=%s\nrollout_precision=%s\nconfig=%s\nnodes=%s\nsteps=%s\n' \
    "${MODEL}" "${TRAINING_PRECISION}" "${ROLLOUT_PRECISION}" "${CONFIG}" "${NUM_NODES}" "${MAX_STEPS}"
  exit 0
fi

case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *)
    echo "ACTION must be render, test-only, or submit" >&2
    exit 2
    ;;
esac

: "${REPO:?Set REPO to the branch checkout under /home}"
: "${CONTAINER:?Set CONTAINER to an immutable NeMo-RL image under /lustre}"
: "${HF_HOME:?Set HF_HOME to the model cache under /lustre}"
: "${WANDB_HOME:?Set WANDB_HOME to a directory containing .netrc}"
: "${RESULT_ROOT:?Set RESULT_ROOT to a durable directory under /lustre}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}"

SLURM_BIN_DIR=$(dirname "$(readlink -f "$(command -v scontrol)")")
export PATH="${SLURM_BIN_DIR}:${PATH}"

LOCAL_SCRATCH=${LOCAL_SCRATCH:-/raid/scratch/${USER}}
PARTITION=${PARTITION:-batch}
RUN_NAME=${TRAINING_PRECISION}-training-${MODEL}-${ROLLOUT_PRECISION}-rollout-${RUN_GROUP}
RUN_ROOT=${RESULT_ROOT}/${RUN_NAME}

git -C "${REPO}" pull --ff-only
git -C "${REPO}" submodule update --init --recursive --checkout
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=all)"
if git -C "${REPO}" submodule status --recursive | grep -q '^[+-U]'; then
  echo "All submodules must be initialized at pinned revisions" >&2
  exit 2
fi

for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" \
  "${HF_HOME}" "${WANDB_HOME}/.netrc"; do
  test -e "${path}"
done

mkdir -p "${RUN_ROOT}/logs"

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HOME=/root
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=\${HF_HOME}/cache
export HUGGINGFACE_HUB_CACHE=\${HF_HOME}/hub
export NCCL_NVLS_ENABLE=0
export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS}
export NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}
export NEMO_RL_VENV_DIR=${LOCAL_SCRATCH}/nemo-rl-worker-cache/${LOCAL_HEAD}
export VLLM_CACHE_ROOT=${LOCAL_SCRATCH}/vllm-cache/${LOCAL_HEAD}
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_SCRATCH}/inductor-cache/${LOCAL_HEAD}
export TRITON_CACHE_DIR=${LOCAL_SCRATCH}/triton-cache/${LOCAL_HEAD}
export UV_CACHE_DIR=${LOCAL_SCRATCH}/uv-cache
export UV_PYTHON_INSTALL_DIR=${LOCAL_SCRATCH}/uv-python
export UV_LOCK_TIMEOUT=7200
export PYTHONPATH=${REPO}
unset UV_PROJECT_ENVIRONMENT WANDB_API_KEY
mkdir -p "\${NEMO_RL_VENV_DIR}" "\${VLLM_CACHE_ROOT}" \
  "\${TORCHINDUCTOR_CACHE_DIR}" "\${TRITON_CACHE_DIR}" \
  "\${UV_CACHE_DIR}" "\${UV_PYTHON_INSTALL_DIR}"
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config ${CONFIG} \
  grpo.max_num_steps=${MAX_STEPS} \
  grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  checkpointing.enabled=false \
  policy.generation.vllm_cfg.use_tqdm=false \
  logger.log_dir=${RUN_ROOT}/logs \
  logger.wandb_enabled=true \
  logger.wandb.project=nemo-rl-mxfp8-training \
  logger.wandb.name=${RUN_NAME} \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre,/home:/home,/raid/scratch:/raid/scratch,${WANDB_HOME}/.netrc:/root/.netrc
export CONTAINER_REMAP_ROOT=1
export COMMAND
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=${CPUS_PER_WORKER:-144}
export BASE_LOG_DIR=${RUN_ROOT}
export SETUP_COMMAND="mkdir -p ${LOCAL_SCRATCH}/nemo-rl-worker-cache ${LOCAL_SCRATCH}/vllm-cache ${LOCAL_SCRATCH}/inductor-cache ${LOCAL_SCRATCH}/triton-cache ${LOCAL_SCRATCH}/uv-cache ${LOCAL_SCRATCH}/uv-python"

SBATCH_ARGS=(
  --nodes="${NUM_NODES}"
  --gres=gpu:4
  --exclusive
  --account="${SLURM_ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --segment="${SEGMENT_SIZE}"
  --job-name="${SLURM_ACCOUNT}.${RUN_NAME}"
  --output="${RUN_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"MXFP8 training smoke"}}'
)

printf 'repo=%s\nsha=%s\nconfig=%s\nresult=%s\n' \
  "${REPO}" "${LOCAL_HEAD}" "${CONFIG}" "${RUN_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
