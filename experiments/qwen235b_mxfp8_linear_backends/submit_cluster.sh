#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR_OVERRIDE:-$(realpath "${SCRIPT_DIR}/../..")}

BACKEND=${BACKEND:-flashinfer_cutedsl}
ACTION=${ACTION:-dry-run}
case "${BACKEND}" in
    flashinfer_cutedsl|flashinfer_cutlass) ;;
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

EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-a76062edee3a3ac23d47a93c7ce466f06a19111f}
MODEL=qwen3-235b
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml

ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
WALLTIME=${WALLTIME:-05:00:00}
NUM_NODES=${NUM_NODES:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SEGMENT_SIZE=${SEGMENT_SIZE:-16}
MAX_STEPS=${MAX_STEPS:-8}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
DEPENDENCY_JOB_ID=${DEPENDENCY_JOB_ID:-}

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
NEMO_RL_STATUS_PATHS=(.)
RUNTIME_NEMO_RL_STATUS_EXCLUSION=
if [[ "${CUSTOM_VLLM_ROOT}" == "${REPO_DIR}/"* ]]; then
    custom_vllm_relative=${CUSTOM_VLLM_ROOT#"${REPO_DIR}/"}
    NEMO_RL_STATUS_PATHS+=(":(exclude)${custom_vllm_relative}")
    RUNTIME_NEMO_RL_STATUS_EXCLUSION=" ':(exclude)${custom_vllm_relative}'"
fi
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${WORK_ROOT}/experiments/qwen235b-mxfp8-linear-backends/${RUN_ID}/${BACKEND}}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/.cache/qwen235b-mxfp8-linear-backends/${BACKEND}}
HF_HOME=${HF_HOME:-${WORK_ROOT}/.cache/huggingface}
DRIVER_VENV=${DRIVER_VENV:-${CACHE_ROOT}/driver-venv}
WORKER_VENV=${WORKER_VENV:-/tmp/nemo-rl-qwen235b-${BACKEND}-workers}
WANDB_MODE=${WANDB_MODE:-disabled}
SUBMIT_NEMO_RL_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD)
SUBMIT_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT}

if [[ "${ACTION}" != "dry-run" ]]; then
    [[ -f "${CONTAINER}" ]] || { echo "Missing container: ${CONTAINER}" >&2; exit 1; }
    [[ -d "${CUSTOM_VLLM_ROOT}/.git" ]] || {
        echo "Custom vLLM is not prepared at ${CUSTOM_VLLM_ROOT}" >&2
        exit 1
    }
    [[ -f "${CUSTOM_VLLM_ROOT}/nemo-rl.env" ]] || {
        echo "Custom vLLM environment is not prepared at ${CUSTOM_VLLM_ROOT}" >&2
        exit 1
    }
    if [[ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=all -- "${NEMO_RL_STATUS_PATHS[@]}")" ]]; then
        echo "NeMo-RL source is not clean at ${REPO_DIR}" >&2
        exit 1
    fi
    SUBMIT_VLLM_COMMIT=$(git -C "${CUSTOM_VLLM_ROOT}" rev-parse HEAD)
    [[ "${SUBMIT_VLLM_COMMIT}" == "${EXPECTED_VLLM_COMMIT}" ]] || {
        echo "Unexpected vLLM commit: ${SUBMIT_VLLM_COMMIT}" >&2
        exit 1
    }
fi

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}" "${HF_HOME}"

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO_DIR}
runtime_nemo_rl_commit=\$(git rev-parse HEAD)
[[ "\${runtime_nemo_rl_commit}" == "${SUBMIT_NEMO_RL_COMMIT}" ]] || {
  echo "NeMo-RL runtime commit mismatch: expected ${SUBMIT_NEMO_RL_COMMIT}, found \${runtime_nemo_rl_commit}" >&2
  exit 1
}
if [[ -n "\$(git status --porcelain --untracked-files=all -- .${RUNTIME_NEMO_RL_STATUS_EXCLUSION})" ]]; then
  echo "NeMo-RL source is not clean at job start: ${REPO_DIR}" >&2
  exit 1
fi
runtime_vllm_commit=\$(git -C ${CUSTOM_VLLM_ROOT} rev-parse HEAD)
[[ "\${runtime_vllm_commit}" == "${SUBMIT_VLLM_COMMIT}" ]] || {
  echo "vLLM runtime commit mismatch: expected ${SUBMIT_VLLM_COMMIT}, found \${runtime_vllm_commit}" >&2
  exit 1
}
rm -f ${EXPERIMENT_ROOT}/run_manifest.json
export HF_HOME=${HF_HOME}
export NCCL_NVLS_ENABLE=0
export RAY_CGRAPH_get_timeout=2400
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=${WORKER_VENV}
export NRL_VENV_BOOTSTRAP_PACKAGES='--torch-backend cu130 torch==2.11.0 numpy setuptools setuptools-rust setuptools-scm'
export NRL_VENV_NO_BUILD_ISOLATION_PACKAGES=vllm
export NVTE_CUDA_ARCHS=100
export SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1
export TORCH_CUDA_ARCH_LIST=10.0
export UV_PROJECT_ENVIRONMENT=${DRIVER_VENV}
export UV_LOCK_TIMEOUT=7200
export WANDB_MODE=${WANDB_MODE}
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_aarch64.whl
source ${CUSTOM_VLLM_ROOT}/nemo-rl.env
printf 'NEMO_RL_COMMIT=%s\n' "\${runtime_nemo_rl_commit}"
printf 'VLLM_COMMIT=%s\n' "\${runtime_vllm_commit}"
if [[ ! -x ${DRIVER_VENV}/bin/python ]]; then
  uv venv ${DRIVER_VENV}
fi
uv pip install --python ${DRIVER_VENV}/bin/python setuptools_rust
uv run --frozen --extra vllm python - <<'PY'
from pathlib import Path

import flashinfer
import vllm

vllm_path = Path(vllm.__file__).resolve()
custom_vllm_root = Path("${CUSTOM_VLLM_ROOT}").resolve()
if not vllm_path.is_relative_to(custom_vllm_root):
    raise RuntimeError(
        f"Expected vLLM from {custom_vllm_root}, but imported {vllm_path}"
    )

print(f"vLLM={vllm.__version__} path={vllm_path}")
print(f"FlashInfer={flashinfer.__version__}")
PY
export MXFP8_NEMO_RL_COMMIT="\${runtime_nemo_rl_commit}"
export MXFP8_VLLM_COMMIT="\${runtime_vllm_commit}"
${DRIVER_VENV}/bin/python - <<'PY'
import json
import os
from pathlib import Path

manifest = {
    "model": "${MODEL}",
    "nemo_rl_commit": os.environ["MXFP8_NEMO_RL_COMMIT"],
    "vllm_commit": os.environ["MXFP8_VLLM_COMMIT"],
    "container": "${CONTAINER}",
    "recipe": "${CONFIG}",
    "cuda_graph": True,
    "quantization_ignored_layer_kws": ["lm_head", "mlp.gate"],
    "moe_backend": "flashinfer_trtllm",
    "linear_backend": "${BACKEND}",
}
manifest_path = Path("${EXPERIMENT_ROOT}/run_manifest.json")
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
uv run --frozen --extra vllm examples/run_grpo.py \
  --config ${CONFIG} \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  cluster.segment_size=${SEGMENT_SIZE} \
  policy.generation.vllm_cfg.enforce_eager=false \
  "policy.generation.vllm_cfg.quantization_ignored_layer_kws=[lm_head,mlp.gate]" \
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \
  ++policy.generation.vllm_kwargs.linear_backend=${BACKEND} \
  +policy.generation.vllm_kwargs.distributed_timeout_seconds=2400 \
  grpo.max_num_steps=${MAX_STEPS} \
  grpo.val_at_start=false \
  checkpointing.enabled=false \
  checkpointing.checkpoint_dir=${EXPERIMENT_ROOT}/checkpoints \
  logger.log_dir=${EXPERIMENT_ROOT}/logs \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=true
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
    --job-name="q235-mx-${BACKEND#flashinfer_}-${RUN_ID}"
    --output="${EXPERIMENT_ROOT}/slurm-%j.out"
)
if [[ -n "${QOS}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi
if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDENCY_JOB_ID}")
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
