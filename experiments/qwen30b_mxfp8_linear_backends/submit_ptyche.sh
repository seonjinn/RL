#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR_OVERRIDE:-$(realpath "${SCRIPT_DIR}/../..")}

BACKEND=${BACKEND:-flashinfer_cutedsl}
ACTION=${ACTION:-dry-run}
case "${BACKEND}" in
    flashinfer_cutedsl|flashinfer_cutlass|flashinfer_trtllm) ;;
    *)
        echo "Unsupported BACKEND: ${BACKEND}" >&2
        exit 2
        ;;
esac
case "${ACTION}" in
    dry-run|test-only|submit) ;;
    *)
        echo "Unsupported ACTION: ${ACTION}" >&2
        exit 2
        ;;
esac

EXPECTED_NEMO_RL_BASE_COMMIT=${EXPECTED_NEMO_RL_BASE_COMMIT:-93e41795e2d6f340728ac238e7a426b4770473e3}
EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-cf856f2fb510f7de46e06fc6bdb6ca3a7fdfc5df}
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml

ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
WALLTIME=${WALLTIME:-05:00:00}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SEGMENT_SIZE=${SEGMENT_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-8}
TRAIN_GLOBAL_BATCH_SIZE=${TRAIN_GLOBAL_BATCH_SIZE:-2048}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${WORK_ROOT}/experiments/qwen30b-mxfp8-linear-backends/${RUN_ID}/${BACKEND}}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/.cache/qwen30b-mxfp8-linear-backends/${BACKEND}}
HF_HOME=${HF_HOME:-${WORK_ROOT}/.cache/huggingface}
DRIVER_VENV=${DRIVER_VENV:-${CACHE_ROOT}/driver-venv}
WORKER_VENV=${WORKER_VENV:-/tmp/nemo-rl-qwen30b-${BACKEND}-workers}
WANDB_MODE=${WANDB_MODE:-disabled}

if [[ "${ACTION}" != "dry-run" ]]; then
    [[ -f "${CONTAINER}" ]] || { echo "Missing container: ${CONTAINER}" >&2; exit 1; }
    [[ -d "${CUSTOM_VLLM_ROOT}/.git" ]] || {
        echo "Custom vLLM is not prepared at ${CUSTOM_VLLM_ROOT}" >&2
        exit 1
    }
    actual_vllm_commit=$(git -C "${CUSTOM_VLLM_ROOT}" rev-parse HEAD)
    git -C "${REPO_DIR}" merge-base --is-ancestor \
        "${EXPECTED_NEMO_RL_BASE_COMMIT}" HEAD || {
        echo "NeMo-RL HEAD does not contain ${EXPECTED_NEMO_RL_BASE_COMMIT}" >&2
        exit 1
    }
    [[ "${actual_vllm_commit}" == "${EXPECTED_VLLM_COMMIT}" ]] || {
        echo "Unexpected vLLM commit: ${actual_vllm_commit}" >&2
        exit 1
    }
fi

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}" "${HF_HOME}"

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO_DIR}
export HF_HOME=${HF_HOME}
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=${WORKER_VENV}
export NVTE_CUDA_ARCHS=100
export TORCH_CUDA_ARCH_LIST=10.0
export UV_PROJECT_ENVIRONMENT=${DRIVER_VENV}
export UV_LOCK_TIMEOUT=7200
export WANDB_MODE=${WANDB_MODE}
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_aarch64.whl
unset VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK
unset VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST
unset VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64
unset VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS
printf 'NEMO_RL_COMMIT=%s\n' "\$(git rev-parse HEAD)"
printf 'VLLM_COMMIT=%s\n' "\$(git -C ${CUSTOM_VLLM_ROOT} rev-parse HEAD)"
uv run --frozen python -c 'import flashinfer, vllm; print("vLLM=" + vllm.__version__); print("FlashInfer=" + flashinfer.__version__)'
uv run --frozen examples/run_grpo.py \
  --config ${CONFIG} \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  cluster.segment_size=${SEGMENT_SIZE} \
  policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
  policy.generation.vllm_cfg.enforce_eager=false \
  "policy.generation.vllm_cfg.quantization_ignored_layer_kws=[lm_head,mlp.gate]" \
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \
  ++policy.generation.vllm_kwargs.linear_backend=${BACKEND} \
  grpo.max_num_steps=${MAX_STEPS} \
  grpo.val_at_start=false \
  checkpointing.enabled=false \
  checkpointing.checkpoint_dir=${EXPERIMENT_ROOT}/checkpoints \
  logger.log_dir=${EXPERIMENT_ROOT}/logs \
  logger.wandb_enabled=false
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre
export COMMAND
export GPUS_PER_NODE
export BASE_LOG_DIR=${EXPERIMENT_ROOT}

SBATCH_ARGS=(
    --nodes="${NUM_NODES}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --exclusive
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --segment="${SEGMENT_SIZE}"
    --time="${WALLTIME}"
    --job-name="q30-mx-${BACKEND#flashinfer_}-${RUN_ID}"
    --output="${EXPERIMENT_ROOT}/slurm-%j.out"
)
if [[ -n "${QOS}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi

printf 'backend=%s\n' "${BACKEND}"
printf 'experiment_root=%s\n' "${EXPERIMENT_ROOT}"
printf 'sbatch_args='; printf ' %q' "${SBATCH_ARGS[@]}"; printf '\n'
printf '%s\n' "${COMMAND}"

case "${ACTION}" in
    dry-run)
        ;;
    test-only)
        sbatch --test-only "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
        ;;
    submit)
        sbatch "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
        ;;
esac
