#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR_OVERRIDE:-$(realpath "${SCRIPT_DIR}/../..")}

BACKEND=${BACKEND:-flashinfer_cutedsl}
ACTION=${ACTION:-dry-run}
case "${BACKEND}" in
    flashinfer_cutedsl|flashinfer_cutlass|flashinfer_trtllm|flashinfer_trtllm_adaptive) ;;
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
EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-a76062edee3a3ac23d47a93c7ce466f06a19111f}
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml

ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
WALLTIME=${WALLTIME:-05:00:00}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SEGMENT_SIZE=${SEGMENT_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-8}
MAX_TOTAL_SEQUENCE_LENGTH=${MAX_TOTAL_SEQUENCE_LENGTH:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-${MAX_TOTAL_SEQUENCE_LENGTH}}
MAX_INPUT_SEQUENCE_LENGTH=${MAX_INPUT_SEQUENCE_LENGTH:-${MAX_TOTAL_SEQUENCE_LENGTH}}
NUM_PROMPTS_PER_STEP=${NUM_PROMPTS_PER_STEP:-64}
NUM_GENERATIONS_PER_PROMPT=${NUM_GENERATIONS_PER_PROMPT:-32}
TRAIN_GLOBAL_BATCH_SIZE=${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-false}
SEQUENCE_PACKING=${SEQUENCE_PACKING:-true}
LOGPROB_BATCH_SIZE=${LOGPROB_BATCH_SIZE:-2}
LOGPROB_CHUNK_SIZE=${LOGPROB_CHUNK_SIZE:-null}
DEFER_FP32_LOGITS=${DEFER_FP32_LOGITS:-false}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

EFFECTIVE_BACKEND=${BACKEND}
ADAPTIVE_ENV_SETUP=$(cat <<'EOF'
unset VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK
unset VLLM_MXFP8_DENSE_TRTLLM_LAYOUT
unset VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M
unset VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE
unset VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256
unset VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST
unset VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64
EOF
)

if [[ "${BACKEND}" == "flashinfer_trtllm_adaptive" ]]; then
    EFFECTIVE_BACKEND=flashinfer_trtllm
    TACTIC_ARTIFACT_DIR=${TACTIC_ARTIFACT_DIR:-${SCRIPT_DIR}/artifacts/qwen3_30ba3b_cg_output_shmoo}
    TACTIC_FILE=${TACTIC_FILE:-${TACTIC_ARTIFACT_DIR}/exact_tactics.json}
    TACTIC_SHA256=${TACTIC_SHA256:-79bc8b3614cc93eeca6c88c8271aa6d5f53a16fe59dceffa05fd27879d7281b5}
    LAYER_ALLOWLIST_FILE=${LAYER_ALLOWLIST_FILE:-${TACTIC_ARTIFACT_DIR}/layer_allowlist.txt}
    [[ -f "${TACTIC_FILE}" ]] || { echo "Missing tactic table: ${TACTIC_FILE}" >&2; exit 1; }
    [[ -f "${LAYER_ALLOWLIST_FILE}" ]] || {
        echo "Missing layer allowlist: ${LAYER_ALLOWLIST_FILE}" >&2
        exit 1
    }
    if command -v sha256sum >/dev/null 2>&1; then
        actual_tactic_sha=$(sha256sum "${TACTIC_FILE}" | awk '{print $1}')
    else
        actual_tactic_sha=$(shasum -a 256 "${TACTIC_FILE}" | awk '{print $1}')
    fi
    [[ "${actual_tactic_sha}" == "${TACTIC_SHA256}" ]] || {
        echo "Tactic table SHA256 mismatch: ${actual_tactic_sha}" >&2
        exit 1
    }
    LAYER_ALLOWLIST_B64=$(base64 < "${LAYER_ALLOWLIST_FILE}" | tr -d '\n')
    ADAPTIVE_ENV_SETUP=$(cat <<EOF
export VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK=1
export VLLM_MXFP8_DENSE_TRTLLM_LAYOUT=adaptive
export VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M=256
export VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE=${TACTIC_FILE}
export VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256=${TACTIC_SHA256}
export VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64=${LAYER_ALLOWLIST_B64}
EOF
)
fi

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${WORK_ROOT}/experiments/qwen30b-mxfp8-linear-backends/${RUN_ID}/${BACKEND}}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/.cache/qwen30b-mxfp8-linear-backends/${BACKEND}}
HF_HOME=${HF_HOME:-${WORK_ROOT}/.cache/huggingface}
DRIVER_VENV=${DRIVER_VENV:-${CACHE_ROOT}/driver-venv}
WORKER_VENV=${WORKER_VENV:-/tmp/nemo-rl-qwen30b-${BACKEND}-${RUN_ID}-workers}
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
export NRL_VENV_BOOTSTRAP_PACKAGES='--torch-backend cu130 torch==2.11.0 numpy setuptools setuptools-rust setuptools-scm'
export NRL_VENV_NO_BUILD_ISOLATION_PACKAGES=vllm
export NVTE_CUDA_ARCHS=100
export SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1
export TORCH_CUDA_ARCH_LIST=10.0
export UV_PROJECT_ENVIRONMENT=${DRIVER_VENV}
export UV_LOCK_TIMEOUT=7200
export WANDB_MODE=${WANDB_MODE}
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_aarch64.whl
${ADAPTIVE_ENV_SETUP}
printf 'NEMO_RL_COMMIT=%s\n' "\$(git rev-parse HEAD)"
printf 'VLLM_COMMIT=%s\n' "\$(git -C ${CUSTOM_VLLM_ROOT} rev-parse HEAD)"
if [[ ! -x ${DRIVER_VENV}/bin/python ]]; then
  uv venv ${DRIVER_VENV}
fi
uv pip install --python ${DRIVER_VENV}/bin/python setuptools_rust
uv run --frozen --extra vllm python -c 'import flashinfer, vllm; print("vLLM=" + vllm.__version__); print("FlashInfer=" + flashinfer.__version__)'
uv run --frozen --extra vllm examples/run_grpo.py \
  --config ${CONFIG} \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  cluster.segment_size=${SEGMENT_SIZE} \
  grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP} \
  grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT} \
  policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
  policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH} \
  policy.logprob_batch_size=${LOGPROB_BATCH_SIZE} \
  policy.logprob_chunk_size=${LOGPROB_CHUNK_SIZE} \
  policy.megatron_cfg.activation_checkpointing=${ACTIVATION_CHECKPOINTING} \
  policy.megatron_cfg.defer_fp32_logits=${DEFER_FP32_LOGITS} \
  policy.sequence_packing.enabled=${SEQUENCE_PACKING} \
  policy.generation.max_new_tokens=${MAX_NEW_TOKENS} \
  policy.generation.vllm_cfg.max_model_len=${MAX_TOTAL_SEQUENCE_LENGTH} \
  policy.generation.vllm_cfg.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
  data.max_input_seq_length=${MAX_INPUT_SEQUENCE_LENGTH} \
  policy.generation.vllm_cfg.enforce_eager=false \
  "policy.generation.vllm_cfg.quantization_ignored_layer_kws=[lm_head,mlp.gate]" \
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \
  ++policy.generation.vllm_kwargs.linear_backend=${EFFECTIVE_BACKEND} \
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
