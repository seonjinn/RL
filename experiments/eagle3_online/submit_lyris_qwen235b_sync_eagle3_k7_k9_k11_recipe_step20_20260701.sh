#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-login-lyris}"
REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701}"
CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
TARGET_MODEL="${TARGET_MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65}"
DRAFT_MODEL="${DRAFT_MODEL:-${HF_HOME}/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba}"
PARD_DRAFT_MODEL="${PARD_DRAFT_MODEL:-${HF_HOME}/hub/models--amd--PARD-Qwen3-0.6B/snapshots/f9f650fbab180c26498817718f0db5cae8f25136}"
PARD_DRAFT_TP="${PARD_DRAFT_TP:-8}"
CONFIG="${CONFIG:-examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g.yaml}"
RUN_ID="${RUN_ID:-20260701_lyris_qwen235b_sync_eagle3_k7_k9_k11_recipe_step20_cudagraph}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${RUN_ID}}"
RUN_TAG="${RUN_TAG:-20260701}"
MODE_LABEL="${MODE_LABEL:-sync}"
STEP_LABEL="${STEP_LABEL:-step20}"
WANDB_HOME="${WANDB_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/wandb_netrc_home}"
WANDB_PROJECT="${WANDB_PROJECT:-sna-nemorl-specdec-lyris}"
MEGATRON_CHECKPOINT_DIR="${MEGATRON_CHECKPOINT_DIR:-}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
WALLTIME="${WALLTIME:-05:00:00}"
MAX_STEPS="${MAX_STEPS:-20}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
OUT="${OUT:-${ROOT_DIR}/docs/latest_lyris_nemorl_qwen235b_sync_eagle3_k7_k9_k11_recipe_step20_cudagraph_20260701_jobs.csv}"
VARIANTS="${VARIANTS:-baseline,eagle3_k7,eagle3_k9,eagle3_k11}"

IFS=',' read -r -a variants <<< "${VARIANTS}"
for variant in "${variants[@]}"; do
  case "${variant}" in
    baseline|baseline_noarrms|eagle3_k5|eagle3_k7|eagle3_k9|eagle3_k11|pard_k1|pard_k7|pard_k9|pard_k16) ;;
    *)
      echo "ERROR: unsupported variant: ${variant}" >&2
      exit 2
      ;;
  esac
done

render_command() {
  local variant="$1"
  local k="$2"
  local log_root="${RUN_ROOT}/logs/qwen235b_${MODE_LABEL}_${variant}"
  local checkpoint_root="${RUN_ROOT}/megatron_checkpoints/qwen235b_${MODE_LABEL}_${variant}"
  local training_checkpoint_root="${RUN_ROOT}/training_checkpoints/qwen235b_${MODE_LABEL}_${variant}"
  local node_cache="/tmp/sna/${RUN_ID}_${variant}"
  local wandb_name="qwen235b_perfcfg_${MODE_LABEL}_${variant}_recipeosl8192_cudagraph_${STEP_LABEL}_${RUN_TAG}"
  local specdec_overrides=""

  if [[ -n "${MEGATRON_CHECKPOINT_DIR}" ]]; then
    checkpoint_root="${MEGATRON_CHECKPOINT_DIR}"
  fi

  case "${variant}" in
    baseline)
      ;;
    baseline_noarrms)
      specdec_overrides="++policy.generation.vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms=false"
      ;;
    eagle3_k*)
      specdec_overrides="policy.draft.enabled=false \
++policy.generation.vllm_kwargs.speculative_config.method=eagle3 \
++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT_MODEL} \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${k} \
++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1"
      ;;
    pard_k*)
      specdec_overrides="policy.draft.enabled=false \
++policy.generation.vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms=false \
++policy.generation.vllm_kwargs.speculative_config.method=pard \
++policy.generation.vllm_kwargs.speculative_config.model=${PARD_DRAFT_MODEL} \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${k} \
++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=${PARD_DRAFT_TP}"
      ;;
  esac

  cat <<EOF
set -euo pipefail
cd '${REMOTE_REPO}'
export HF_HOME='${HF_HOME}'
export HF_DATASETS_CACHE='${HF_HOME}/datasets'
export HOME='${WANDB_HOME}'
export NRL_IGNORE_VERSION_MISMATCH=1
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export NEMO_RL_VENV_DIR='${REMOTE_REPO}/venvs'
export NRL_MEGATRON_CHECKPOINT_DIR='${checkpoint_root}'
export NRL_MEGATRON_TOKENIZER_MODEL='${TARGET_MODEL}'
export NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=1800
export RAY_CGRAPH_GET_TIMEOUT=7200
export RAY_CGRAPH_get_timeout=7200
export NODE_LOCAL_CACHE_ROOT='${node_cache}'
export PIP_CACHE_DIR='${RUN_ROOT}/cache/pip'
export XDG_CACHE_HOME='${node_cache}/xdg'
export VLLM_CACHE_ROOT='${node_cache}/vllm'
export FLASHINFER_WORKSPACE_BASE='${node_cache}/flashinfer_workspace'
export FLASHINFER_CACHE_DIR='${node_cache}/flashinfer_workspace/.cache/flashinfer'
export TORCHINDUCTOR_CACHE_DIR='${node_cache}/torchinductor'
export TRITON_CACHE_DIR='${node_cache}/triton'
export CUDA_CACHE_PATH='${node_cache}/cuda'
export TORCH_EXTENSIONS_DIR='${node_cache}/torch_extensions'
export PYTHONPYCACHEPREFIX='${node_cache}/pycache'
export PYTHONDONTWRITEBYTECODE=1
export MEGATRON_DATASET_HELPERS_BUILD_DIR='${node_cache}/megatron_dataset_helpers'
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY='FLASHINFER_WORKSPACE_BASE,FLASHINFER_CACHE_DIR,TORCHINDUCTOR_CACHE_DIR,TRITON_CACHE_DIR,CUDA_CACHE_PATH,XDG_CACHE_HOME,TORCH_EXTENSIONS_DIR,PYTHONPYCACHEPREFIX'
export PYTHONPATH='${REMOTE_REPO}:${REMOTE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REMOTE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM'
python '${REMOTE_REPO}/examples/run_grpo.py' \
  --config '${REMOTE_REPO}/${CONFIG}' \
  policy.model_name='${TARGET_MODEL}' \
  policy.tokenizer.name='${TARGET_MODEL}' \
  policy.generation.vllm_cfg.enforce_eager=false \
  grpo.max_num_steps=${MAX_STEPS} \
  checkpointing.checkpoint_dir='${training_checkpoint_root}' \
  ${specdec_overrides} \
  logger.log_dir='${log_root}/nemo_logs' \
  logger.wandb_enabled=true \
  logger.wandb.project='${WANDB_PROJECT}' \
  logger.wandb.name='${wandb_name}'
EOF
}

render_sbatch() {
  local variant="$1"
  local job_variant="${variant//_/-}"
  local log_root="${RUN_ROOT}/logs/qwen235b_${MODE_LABEL}_${variant}"
  printf '%s' "sbatch --nodes=32 --account=${ACCOUNT} --job-name=${ACCOUNT}-specdec.q235-${MODE_LABEL}-${job_variant} --partition=${PARTITION} --time=${WALLTIME} --segment=16 --output=${log_root}/slurm-%j.out ray.sub"
}

if [[ "${DRY_RUN}" == "true" ]]; then
  for variant in "${variants[@]}"; do
    k="${variant##*_k}"
    [[ "${variant}" == "baseline" || "${variant}" == "baseline_noarrms" ]] && k=0
    echo "[DRY-RUN] variant=${variant}"
    echo "[DRY-RUN] $(render_sbatch "${variant}")"
    render_command "${variant}" "${k}"
  done
  exit 0
fi

remote_payload=$(cat <<'REMOTE'
set -euo pipefail
cd "${REMOTE_REPO}"

for required in \
  "${CONTAINER}" \
  "${TARGET_MODEL}/config.json" \
  "${DRAFT_MODEL}/config.json" \
  "${PARD_DRAFT_MODEL}/config.json" \
  "${CONFIG}" \
  ray.sub; do
  if [[ ! -s "${required}" ]]; then
    echo "ERROR: missing required file: ${required}" >&2
    exit 2
  fi
done

if [[ -n "${MEGATRON_CHECKPOINT_DIR}" && ! -d "${MEGATRON_CHECKPOINT_DIR}" ]]; then
  echo "ERROR: missing Megatron checkpoint directory: ${MEGATRON_CHECKPOINT_DIR}" >&2
  exit 2
fi

repo_head="$(git rev-parse HEAD)"
printf 'job_id,variant,k,repo_head,config,max_steps,max_new_tokens,enforce_eager,async_engine,attention_backend,nodes,gpus_per_node,segment,container,log_dir,wandb_project,wandb_name\n'

submit_variant() {
  local variant="$1"
  local k="$2"
  local command="$3"
  local job_variant="${variant//_/-}"
  local log_root="${RUN_ROOT}/logs/qwen235b_${MODE_LABEL}_${variant}"
  local wandb_name="qwen235b_perfcfg_${MODE_LABEL}_${variant}_recipeosl8192_cudagraph_${STEP_LABEL}_${RUN_TAG}"
  local sbatch_args=(
    --nodes=32
    --account="${ACCOUNT}"
    --job-name="${ACCOUNT}-specdec.q235-${MODE_LABEL}-${job_variant}"
    --partition="${PARTITION}"
    --time="${WALLTIME}"
    --segment=16
    --output="${log_root}/slurm-%j.out"
  )

  mkdir -p "${log_root}" "${RUN_ROOT}/cache"
  if [[ "${TEST_ONLY}" == "true" ]]; then
    CONTAINER="${CONTAINER}" \
    MOUNTS="/lustre:/lustre,/project:/project" \
    BASE_LOG_DIR="${log_root}" \
    GPUS_PER_NODE=4 \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_HOME}/datasets" \
    COMMAND="${command}" \
      sbatch --test-only "${sbatch_args[@]}" ray.sub >&2
    job_id="TEST_ONLY"
  else
    output=$(
      CONTAINER="${CONTAINER}" \
      MOUNTS="/lustre:/lustre,/project:/project" \
      BASE_LOG_DIR="${log_root}" \
      GPUS_PER_NODE=4 \
      HF_HOME="${HF_HOME}" \
      HF_DATASETS_CACHE="${HF_HOME}/datasets" \
      COMMAND="${command}" \
        sbatch "${sbatch_args[@]}" ray.sub
    )
    job_id="$(printf '%s\n' "${output}" | sed -n 's/^Submitted batch job //p' | tail -n 1)"
    if [[ -z "${job_id}" ]]; then
      printf '%s\n' "${output}" >&2
      exit 1
    fi
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${job_id}" "${variant}" "${k}" "${repo_head}" "${CONFIG}" \
    "${MAX_STEPS}" 8192 false recipe_true recipe_default 32 4 16 "${CONTAINER}" "${log_root}" \
    "${WANDB_PROJECT}" "${wandb_name}"
}
REMOTE
)

for variant in "${variants[@]}"; do
  k="${variant##*_k}"
  [[ "${variant}" == "baseline" || "${variant}" == "baseline_noarrms" ]] && k=0
  command="$(render_command "${variant}" "${k}")"
  printf -v quoted_variant '%q' "${variant}"
  printf -v quoted_k '%q' "${k}"
  printf -v quoted_command '%q' "${command}"
  remote_payload+=$'\n'"submit_variant ${quoted_variant} ${quoted_k} ${quoted_command}"
done

mkdir -p "$(dirname "${OUT}")"
printf '%s\n' "${remote_payload}" | \
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE_HOST}" \
    env \
      REMOTE_REPO="${REMOTE_REPO}" \
      CONTAINER="${CONTAINER}" \
      HF_HOME="${HF_HOME}" \
      TARGET_MODEL="${TARGET_MODEL}" \
      DRAFT_MODEL="${DRAFT_MODEL}" \
      PARD_DRAFT_MODEL="${PARD_DRAFT_MODEL}" \
      PARD_DRAFT_TP="${PARD_DRAFT_TP}" \
      CONFIG="${CONFIG}" \
      RUN_ROOT="${RUN_ROOT}" \
      RUN_TAG="${RUN_TAG}" \
      MODE_LABEL="${MODE_LABEL}" \
      STEP_LABEL="${STEP_LABEL}" \
      WANDB_PROJECT="${WANDB_PROJECT}" \
      MEGATRON_CHECKPOINT_DIR="${MEGATRON_CHECKPOINT_DIR}" \
      ACCOUNT="${ACCOUNT}" \
      PARTITION="${PARTITION}" \
      WALLTIME="${WALLTIME}" \
      MAX_STEPS="${MAX_STEPS}" \
      TEST_ONLY="${TEST_ONLY}" \
      bash -s | tee "${OUT}"
