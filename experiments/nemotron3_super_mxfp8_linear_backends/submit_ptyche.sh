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
MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
CONFIG=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g.yaml
PROVENANCE_HELPER=experiments/mxfp8_linear_backend_model_matrix/provenance.sh
source "${SCRIPT_DIR}/../mxfp8_linear_backend_model_matrix/provenance.sh"

ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
WALLTIME=${WALLTIME:-05:00:00}
NUM_NODES=${NUM_NODES:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SEGMENT_SIZE=${SEGMENT_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-8}
NUM_PROMPTS_PER_STEP=${NUM_PROMPTS_PER_STEP:-32}
NUM_GENERATIONS_PER_PROMPT=${NUM_GENERATIONS_PER_PROMPT:-8}
TRAIN_GLOBAL_BATCH_SIZE=${TRAIN_GLOBAL_BATCH_SIZE:-256}
MAX_TOTAL_SEQUENCE_LENGTH=${MAX_TOTAL_SEQUENCE_LENGTH:-8192}
MAX_INPUT_SEQUENCE_LENGTH=${MAX_INPUT_SEQUENCE_LENGTH:-8192}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GENERATION_TENSOR_PARALLEL_SIZE=${GENERATION_TENSOR_PARALLEL_SIZE:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
PRECISION=${PRECISION:-fp8}
IS_MX=${IS_MX:-true}
LOGPROB_BATCH_SIZE=${LOGPROB_BATCH_SIZE:-1}
LOGPROB_CHUNK_SIZE=${LOGPROB_CHUNK_SIZE:-2048}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-true}
DEFER_FP32_LOGITS=${DEFER_FP32_LOGITS:-true}
SEQUENCE_PACKING=${SEQUENCE_PACKING:-true}
case "${LOGPROB_BATCH_SIZE}" in
    ''|*[!0-9]*|0|0*) echo "LOGPROB_BATCH_SIZE must be a positive integer" >&2; exit 2 ;;
esac
for boolean_name in IS_MX ACTIVATION_CHECKPOINTING SEQUENCE_PACKING DEFER_FP32_LOGITS; do
    boolean_value=${!boolean_name}
    case "${boolean_value}" in
        true) printf -v "${boolean_name}_PYTHON" '%s' True ;;
        false) printf -v "${boolean_name}_PYTHON" '%s' False ;;
        *) echo "${boolean_name} must be true or false" >&2; exit 2 ;;
    esac
done
case "${LOGPROB_CHUNK_SIZE}" in
    null) LOGPROB_CHUNK_SIZE_PYTHON=None ;;
    ''|*[!0-9]*|0|0*) echo "LOGPROB_CHUNK_SIZE must be null or a positive integer" >&2; exit 2 ;;
    *) LOGPROB_CHUNK_SIZE_PYTHON=${LOGPROB_CHUNK_SIZE} ;;
esac
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
NEMO_RL_STATUS_PATHS=(. ":(exclude)pyproject.toml" ":(exclude)uv.lock")
RUNTIME_NEMO_RL_STATUS_EXCLUSIONS=" ':(exclude)pyproject.toml' ':(exclude)uv.lock'"
if [[ "${CUSTOM_VLLM_ROOT}" == "${REPO_DIR}/"* ]]; then
    custom_vllm_relative=${CUSTOM_VLLM_ROOT#"${REPO_DIR}/"}
    NEMO_RL_STATUS_PATHS+=(":(exclude)${custom_vllm_relative}")
    RUNTIME_NEMO_RL_STATUS_EXCLUSIONS+=" ':(exclude)${custom_vllm_relative}'"
fi
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${WORK_ROOT}/experiments/nemotron3-super-mxfp8-linear-backends/${RUN_ID}/${BACKEND}}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/.cache/nemotron3-super-mxfp8-linear-backends/${BACKEND}}
HF_HOME=${HF_HOME:-${WORK_ROOT}/.cache/huggingface}
DRIVER_VENV=${DRIVER_VENV:-${CACHE_ROOT}/driver-venv}
WORKER_VENV=${WORKER_VENV:-/tmp/nemo-rl-nemotron3-super-${BACKEND}-${RUN_ID}-workers}
WANDB_MODE=${WANDB_MODE:-disabled}
SUBMIT_NEMO_RL_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD)
SUBMIT_DEPENDENCY_STATE_SHA256=
SUBMIT_RECIPE_SHA256=
SUBMIT_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT}
SUBMIT_VLLM_SOURCE_SHA256=dry-run-not-validated
SUBMIT_VLLM_DEPENDENCY_STATE_SHA256=dry-run-not-validated

if [[ "${ACTION}" == "dry-run" ]]; then
    SUBMIT_DEPENDENCY_STATE_SHA256=$(mxfp8_dependency_state_sha256 "${REPO_DIR}")
    SUBMIT_RECIPE_SHA256=$(mxfp8_file_sha256 "${REPO_DIR}/${CONFIG}")
else
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
    SUBMIT_DEPENDENCY_STATE_SHA256=$(mxfp8_dependency_state_sha256 "${REPO_DIR}")
    SUBMIT_RECIPE_SHA256=$(mxfp8_file_sha256 "${REPO_DIR}/${CONFIG}")
    mxfp8_assert_vllm_tracked_state "${CUSTOM_VLLM_ROOT}" || {
        echo "Custom vLLM tracked files are not clean at ${CUSTOM_VLLM_ROOT}" >&2
        exit 1
    }
    SUBMIT_VLLM_COMMIT=$(git -C "${CUSTOM_VLLM_ROOT}" rev-parse HEAD)
    [[ "${SUBMIT_VLLM_COMMIT}" == "${EXPECTED_VLLM_COMMIT}" ]] || {
        echo "Unexpected vLLM commit: ${SUBMIT_VLLM_COMMIT}" >&2
        exit 1
    }
    SUBMIT_VLLM_SOURCE_SHA256=$(mxfp8_vllm_source_sha256 "${CUSTOM_VLLM_ROOT}")
    SUBMIT_VLLM_DEPENDENCY_STATE_SHA256=$(mxfp8_vllm_dependency_state_sha256 "${CUSTOM_VLLM_ROOT}")
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
if [[ -n "\$(git status --porcelain --untracked-files=all -- .${RUNTIME_NEMO_RL_STATUS_EXCLUSIONS})" ]]; then
  echo "NeMo-RL source is not clean at job start: ${REPO_DIR}" >&2
  exit 1
fi
source ${REPO_DIR}/${PROVENANCE_HELPER}
runtime_dependency_state_sha256=\$(mxfp8_dependency_state_sha256 ${REPO_DIR})
[[ "\${runtime_dependency_state_sha256}" == "${SUBMIT_DEPENDENCY_STATE_SHA256}" ]] || {
  echo "Dependency state mismatch: expected ${SUBMIT_DEPENDENCY_STATE_SHA256}, found \${runtime_dependency_state_sha256}" >&2
  exit 1
}
runtime_recipe_sha256=\$(mxfp8_file_sha256 ${REPO_DIR}/${CONFIG})
[[ "\${runtime_recipe_sha256}" == "${SUBMIT_RECIPE_SHA256}" ]] || {
  echo "Recipe content mismatch: expected ${SUBMIT_RECIPE_SHA256}, found \${runtime_recipe_sha256}" >&2
  exit 1
}
mxfp8_assert_vllm_tracked_state ${CUSTOM_VLLM_ROOT} || {
  echo "Custom vLLM tracked files are not clean at job start: ${CUSTOM_VLLM_ROOT}" >&2
  exit 1
}
runtime_vllm_commit=\$(git -C ${CUSTOM_VLLM_ROOT} rev-parse HEAD)
[[ "\${runtime_vllm_commit}" == "${SUBMIT_VLLM_COMMIT}" ]] || {
  echo "vLLM runtime commit mismatch: expected ${SUBMIT_VLLM_COMMIT}, found \${runtime_vllm_commit}" >&2
  exit 1
}
runtime_vllm_source_sha256=\$(mxfp8_vllm_source_sha256 ${CUSTOM_VLLM_ROOT})
[[ "\${runtime_vllm_source_sha256}" == "${SUBMIT_VLLM_SOURCE_SHA256}" ]] || {
  echo "vLLM source fingerprint mismatch: expected ${SUBMIT_VLLM_SOURCE_SHA256}, found \${runtime_vllm_source_sha256}" >&2
  exit 1
}
runtime_vllm_dependency_state_sha256=\$(mxfp8_vllm_dependency_state_sha256 ${CUSTOM_VLLM_ROOT})
[[ "\${runtime_vllm_dependency_state_sha256}" == "${SUBMIT_VLLM_DEPENDENCY_STATE_SHA256}" ]] || {
  echo "vLLM dependency state mismatch: expected ${SUBMIT_VLLM_DEPENDENCY_STATE_SHA256}, found \${runtime_vllm_dependency_state_sha256}" >&2
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
export MXFP8_DEPENDENCY_STATE_SHA256="\${runtime_dependency_state_sha256}"
export MXFP8_VLLM_COMMIT="\${runtime_vllm_commit}"
export MXFP8_VLLM_SOURCE_SHA256="\${runtime_vllm_source_sha256}"
export MXFP8_VLLM_DEPENDENCY_STATE_SHA256="\${runtime_vllm_dependency_state_sha256}"
${DRIVER_VENV}/bin/python - <<'PY'
import json
import os
from pathlib import Path

manifest = {
    "model": "${MODEL}",
    "nemo_rl_commit": os.environ["MXFP8_NEMO_RL_COMMIT"],
    "dependency_state_sha256": os.environ["MXFP8_DEPENDENCY_STATE_SHA256"],
    "vllm_commit": os.environ["MXFP8_VLLM_COMMIT"],
    "vllm_source_sha256": os.environ["MXFP8_VLLM_SOURCE_SHA256"],
    "vllm_dependency_state_sha256": os.environ["MXFP8_VLLM_DEPENDENCY_STATE_SHA256"],
    "vllm_tracked_files_clean": True,
    "container": "${CONTAINER}",
    "recipe": "${CONFIG}",
    "recipe_sha256": "${SUBMIT_RECIPE_SHA256}",
    "cuda_graph": True,
    "precision": "${PRECISION}",
    "is_mx": ${IS_MX_PYTHON},
    "quantization_ignored_layer_kws": ["lm_head", "mlp.gate"],
    "moe_backend": "flashinfer_trtllm",
    "num_nodes": ${NUM_NODES},
    "gpus_per_node": ${GPUS_PER_NODE},
    "segment_size": ${SEGMENT_SIZE},
    "num_prompts_per_step": ${NUM_PROMPTS_PER_STEP},
    "num_generations_per_prompt": ${NUM_GENERATIONS_PER_PROMPT},
    "train_global_batch_size": ${TRAIN_GLOBAL_BATCH_SIZE},
    "max_total_sequence_length": ${MAX_TOTAL_SEQUENCE_LENGTH},
    "max_input_sequence_length": ${MAX_INPUT_SEQUENCE_LENGTH},
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "max_model_len": ${MAX_MODEL_LEN},
    "generation_tensor_parallel_size": ${GENERATION_TENSOR_PARALLEL_SIZE},
    "max_steps": ${MAX_STEPS},
    "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION},
    "logprob_batch_size": ${LOGPROB_BATCH_SIZE},
    "logprob_chunk_size": ${LOGPROB_CHUNK_SIZE_PYTHON},
    "activation_checkpointing": ${ACTIVATION_CHECKPOINTING_PYTHON},
    "defer_fp32_logits": ${DEFER_FP32_LOGITS_PYTHON},
    "sequence_packing": ${SEQUENCE_PACKING_PYTHON},
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
  grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP} \
  grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT} \
  policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
  policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH} \
  policy.generation.max_new_tokens=${MAX_NEW_TOKENS} \
  policy.generation.vllm_cfg.max_model_len=${MAX_MODEL_LEN} \
  data.max_input_seq_length=${MAX_INPUT_SEQUENCE_LENGTH} \
  policy.generation.vllm_cfg.tensor_parallel_size=${GENERATION_TENSOR_PARALLEL_SIZE} \
  policy.generation.vllm_cfg.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
  policy.generation.vllm_cfg.enforce_eager=false \
  policy.generation.vllm_cfg.precision=${PRECISION} \
  ++policy.generation.vllm_cfg.is_mx=${IS_MX} \
  policy.logprob_batch_size=${LOGPROB_BATCH_SIZE} \
  policy.logprob_chunk_size=${LOGPROB_CHUNK_SIZE} \
  policy.megatron_cfg.activation_checkpointing=${ACTIVATION_CHECKPOINTING} \
  policy.megatron_cfg.defer_fp32_logits=${DEFER_FP32_LOGITS} \
  policy.sequence_packing.enabled=${SEQUENCE_PACKING} \
  "++policy.generation.vllm_cfg.quantization_ignored_layer_kws=[lm_head,mlp.gate]" \
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
    --job-name="n3s-mx-${BACKEND#flashinfer_}-${RUN_ID}"
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
