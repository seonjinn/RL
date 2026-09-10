#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT="${Q30_PTV3_SOURCE_ROOT:-/home/sna/nemorl-q30-flashinfer-specdec-gate-20260831}"
readonly SOURCE_SHA=15554749ae24361b5d511e72ddf41ecab2615cdc
readonly RECIPE="${SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
readonly CONTAINER=/lustre/fsw/portfolios/coreai/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly PTV3_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa
readonly TARGET_MODEL=/lustre/fsw/portfolios/coreai/users/sna/hf-local/Qwen/Qwen3-30B-A3B
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-ptv3-swa-online-30step-20260910/math
readonly ACCOUNT="${Q30_PTV3_ACCOUNT:-nemotron_n4_post}"

usage() {
  echo "usage: $0 --render|--test-only|--submit {dflash_k5,dspark_k7}_fixed{5,10}" >&2
  exit 2
}

mode="${1:-}"
arm="${2:-}"
case "${mode}" in --render|--test-only|--submit) ;; *) usage ;; esac

method=""
method_label=""
k=0
interval=0
case "${arm}" in
  dflash_k5_fixed5) method=dflash; method_label=DFlash; k=5; interval=5 ;;
  dflash_k5_fixed10) method=dflash; method_label=DFlash; k=5; interval=10 ;;
  dspark_k7_fixed5) method=dspark; method_label=DSpark; k=7; interval=5 ;;
  dspark_k7_fixed10) method=dspark; method_label=DSpark; k=7; interval=10 ;;
  *) usage ;;
esac
readonly method method_label k interval

checkpoint="${PTV3_ROOT}/sd2p3swa-q30-base-ptv3swe-${method}-b8-16n/exported-checkpoint-44000"
readonly checkpoint
case "${k}" in
  5) CAPTURE_SIZES='[1,2,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,128,160,192,256,320,384,640,768]' ;;
  7) CAPTURE_SIZES='[1,2,4,7,8,14,16,28,32,56,64,112,128,224,256,448,512,896,1024]' ;;
  *) usage ;;
esac
readonly CAPTURE_SIZES

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="Qwen3-30BA3B-${method_label}K${k}-Fixed${interval}-PTV3SWA-44K-Online30-${timestamp}"
artifact_dir="${DURABLE_ROOT}/${run_id}"
readonly run_id artifact_dir

draft_overrides=(
  "policy.draft.speculator_type=${method}"
  'policy.draft.enabled=true'
  "policy.draft.model_name=${checkpoint}"
  'policy.draft.loss_weight=0.1'
  '++policy.draft.anchors_per_sample=2'
  '++policy.draft.mask_token_id=151669'
  '++policy.draft.target_hidden_state_layer_ids=[1,12,23,34,45]'
  'policy.draft.num_layers=5'
  'policy.draft.optimizer={lr:5e-6,min_lr:5e-7,weight_decay:0.01}'
  '++policy.draft.update_schedule.mode=fixed'
  '++policy.draft.update_schedule.action=sparse_update'
  "++policy.draft.update_schedule.fixed_interval=${interval}"
)
if [[ "${method}" == dflash ]]; then
  draft_overrides+=("++policy.draft.gamma=${k}")
else
  draft_overrides+=(
    "++policy.draft.block_size=${k}"
    '++policy.draft.markov_rank=256'
    '++policy.draft.markov_head_type=vanilla'
    '++policy.draft.confidence_enabled=true'
    '++policy.draft.confidence_with_markov=true'
  )
fi

runtime_overrides=(
  'policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm'
  '++policy.generation.vllm_kwargs.max_num_seqs=128'
  '++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE'
  "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${CAPTURE_SIZES}"
  "++policy.generation.vllm_kwargs.speculative_config.method=${method}"
  "++policy.generation.vllm_kwargs.speculative_config.model=${checkpoint}"
  "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${k}"
  '++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1'
)
if [[ "${method}" == dspark ]]; then
  runtime_overrides+=(
    '++policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN'
    '++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false'
  )
fi

printf -v overrides ' %q' \
  'grpo.max_num_steps=30' \
  "policy.model_name=${TARGET_MODEL}" \
  "policy.tokenizer.name=${TARGET_MODEL}" \
  'policy.sequence_packing.enabled=true' \
  '++policy.offload_optimizer_for_refit=false' \
  'checkpointing.enabled=false' \
  'logger.wandb_enabled=true' \
  'logger.wandb.project=sna-specdec' \
  '++logger.wandb.group=q30-ptv3-swa-44k-math-online30' \
  "logger.wandb.name=${run_id}" \
  "logger.log_dir=${artifact_dir}/logs" \
  "${draft_overrides[@]}" \
  "${runtime_overrides[@]}"

render() {
  local post_sync_lines=""
  if [[ "${method}" == dspark ]]; then
    post_sync_lines="export NRL_VENV_POST_SYNC_SCRIPT=${SOURCE_ROOT}/experiments/qwen3_30ba3b_draft_cadence_200step_20260826/prepare_vllm_dspark_fap_overlay.py
export NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
  fi
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${ACCOUNT}.${run_id}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output=${artifact_dir}/slurm-%j.out
#SBATCH --error=${artifact_dir}/slurm-%j.err
set -euo pipefail
export PATH=/cm/local/apps/slurm/25.11/bin:\${PATH}
test -n "\${WANDB_API_KEY:-}"
test "\$(git -C ${SOURCE_ROOT} rev-parse HEAD)" = "${SOURCE_SHA}"
test -z "\$(git -C ${SOURCE_ROOT} status --porcelain=v1 --untracked-files=all)"
test -r "${CONTAINER}"
test -f "${RECIPE}"
test -d "${TARGET_MODEL}"
test -f "${checkpoint}/model.safetensors"
mkdir -p "${artifact_dir}"
export CONTAINER="${CONTAINER}"
export MOUNTS=/lustre:/lustre,/home:/home,/raid:/raid
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export BASE_LOG_DIR="${artifact_dir}"
export Q30_NODE_ROOT="/raid/scratch/sna/q30-ptv3-online30-\${SLURM_JOB_ID}"
export Q30_MCORE_SOURCE="${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export Q30_MCORE_OVERLAY="\${Q30_NODE_ROOT}/mcore-overlay"
export Q30_VLLM_OVERLAY="\${Q30_NODE_ROOT}/vllm-overlay"
export NEMO_RL_VENV_DIR="\${Q30_NODE_ROOT}/venvs"
export PYTHONPATH="\${Q30_VLLM_OVERLAY}:\${Q30_MCORE_OVERLAY}:${SOURCE_ROOT}:\${PYTHONPATH:-}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export SETUP_COMMAND='set -euo pipefail; mkdir -p "\${Q30_MCORE_OVERLAY}"; cp -a "\${Q30_MCORE_SOURCE}/megatron" "\${Q30_MCORE_OVERLAY}/"; test -f "\${Q30_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp"'
${post_sync_lines}
export NRL_FORCE_REBUILD_VENVS=true
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=10
export COMMAND="cd ${SOURCE_ROOT} && uv run examples/run_grpo.py --config ${RECIPE}${overrides}"
exec bash "${SOURCE_ROOT}/ray.sub"
EOF
}

load_wandb_api_key() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    return
  fi
  # shellcheck disable=SC1091
  source "${HOME}/.bashrc" >/dev/null 2>&1 || true
  test -n "${WANDB_API_KEY:-}"
  export WANDB_API_KEY
}

if [[ "${mode}" == --render ]]; then
  render
  exit 0
fi

test -e "${SOURCE_ROOT}/.git"
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${SOURCE_SHA}"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)"
test -r "${CONTAINER}"
test -f "${RECIPE}"
test -d "${TARGET_MODEL}"
test -f "${checkpoint}/model.safetensors"
load_wandb_api_key
mkdir -p "${artifact_dir}"
sbatch_path="${artifact_dir}/job.sbatch"
render >"${sbatch_path}"
chmod 700 "${sbatch_path}"
test_output="$(sbatch --test-only "${sbatch_path}" 2>&1)"
printf '%s\n' "${test_output}" | tee "${artifact_dir}/test-only.txt"
if [[ "${mode}" == --test-only ]]; then
  exit 0
fi
sbatch "${sbatch_path}" | tee "${artifact_dir}/submission.txt"
