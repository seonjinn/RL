#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=/home/sna/nemorl-bf16-flashinfer-specdec-latest-main-20260909
readonly RECIPE="${SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260909_7023221.sqsh
readonly TARGET_MODEL=/lustre/fsw/portfolios/coreai/users/sna/hf-local/Qwen/Qwen3-30B-A3B
readonly PTV3_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-latest-main-bf16-flashinfer-specdec-20260909
readonly ACCOUNT="${Q30_LATEST_MAIN_ACCOUNT:-nemotron_n4_post}"
readonly MAX_STEPS="${Q30_LATEST_MAIN_MAX_STEPS:-3}"

usage() {
  echo "usage: $0 --render|--test-only|--submit baseline|dflash_k3|dspark_k3" >&2
  exit 2
}

if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Q30_LATEST_MAIN_MAX_STEPS must be a positive integer: ${MAX_STEPS}" >&2
  exit 2
fi

mode="${1:-}"
arm="${2:-}"
case "${mode}" in --render|--test-only|--submit) ;; *) usage ;; esac

method=""
checkpoint=""
arm_label=""
case "${arm}" in
  baseline) arm_label="Baseline" ;;
  dflash_k3|dspark_k3)
    method="${arm%%_k*}"
    if [[ "${method}" == dflash ]]; then
      arm_label="DFlashK3"
    else
      arm_label="DSparkK3"
    fi
    checkpoint="${PTV3_ROOT}/sd2p3swa-q30-base-ptv3swe-${method}-b8-16n/exported-checkpoint-44000"
    ;;
  *) usage ;;
esac

readonly CAPTURE_SIZES='[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512]'
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="Qwen3-30BA3B-latest-main-BF16-flashinfer-${arm_label}-${MAX_STEPS}step-FAP-${timestamp}"
artifact_dir="${DURABLE_ROOT}/${run_id}"

post_sync_lines=""
if [[ "${method}" == dspark ]]; then
  post_sync_lines="export NRL_VENV_POST_SYNC_SCRIPT=${SOURCE_ROOT}/experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/prepare_vllm_dspark_fap_overlay.py
export NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
fi

spec_overrides=(
  'policy.precision=bfloat16'
  'policy.draft.enabled=false'
  'policy.sequence_packing.enabled=true'
  '++policy.offload_optimizer_for_refit=false'
  'policy.generation.refit_transport=null'
  'policy.generation.vllm_cfg.refit_with_reload_api=false'
  'policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm'
  '++policy.generation.vllm_kwargs.max_num_seqs=128'
  '++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE'
  "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${CAPTURE_SIZES}"
)
if [[ "${arm}" == baseline ]]; then
  spec_overrides+=('++policy.generation.vllm_kwargs.speculative_config=null')
else
  spec_overrides+=(
    "++policy.generation.vllm_kwargs.speculative_config.method=${method}"
    "++policy.generation.vllm_kwargs.speculative_config.model=${checkpoint}"
    '++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3'
    '++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1'
  )
  if [[ "${method}" == dspark ]]; then
    spec_overrides+=(
      '++policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN'
      '++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false'
    )
  fi
fi

printf -v overrides ' %q' \
  "grpo.max_num_steps=${MAX_STEPS}" \
  "policy.model_name=${TARGET_MODEL}" \
  "policy.tokenizer.name=${TARGET_MODEL}" \
  'logger.wandb_enabled=true' \
  'logger.wandb.project=sna-specdec' \
  '++logger.wandb.group=q30-latest-main-bf16-flashinfer-specdec' \
  "logger.wandb.name=${run_id}" \
  "logger.log_dir=${artifact_dir}/logs" \
  "${spec_overrides[@]}"

render() {
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${ACCOUNT}.${run_id}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --time=02:00:00
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
test -z "\$(git -C ${SOURCE_ROOT} status --porcelain=v1 --untracked-files=all)"
test -r "${CONTAINER}"
test -f "${RECIPE}"
test -d "${TARGET_MODEL}"
$(if [[ -n "${checkpoint}" ]]; then printf 'test -f "%s/model.safetensors"\n' "${checkpoint}"; fi)
mkdir -p "${artifact_dir}"
git -C "${SOURCE_ROOT}" rev-parse HEAD | tee "${artifact_dir}/source_sha.txt"
git -C "${SOURCE_ROOT}" submodule status --recursive >"${artifact_dir}/submodules.txt"
export CONTAINER="${CONTAINER}"
export MOUNTS=/lustre:/lustre,/home:/home,/raid:/raid
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export BASE_LOG_DIR="${artifact_dir}"
export Q30_NODE_ROOT="/raid/scratch/sna/q30-latest-main-bf16-specdec-\${SLURM_JOB_ID}"
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
  if [[ -r "${HOME}/.netrc" ]]; then
    WANDB_API_KEY="$(python3 - <<'PY'
from netrc import netrc

credentials = netrc().authenticators("api.wandb.ai")
print(credentials[2] if credentials else "")
PY
)"
    export WANDB_API_KEY
  fi
  test -n "${WANDB_API_KEY:-}"
}

if [[ "${mode}" == --render ]]; then
  render
  exit 0
fi

test -e "${SOURCE_ROOT}/.git"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)"
test -r "${CONTAINER}"
test -f "${RECIPE}"
test -d "${TARGET_MODEL}"
[[ -z "${checkpoint}" ]] || test -f "${checkpoint}/model.safetensors"
load_wandb_api_key
mkdir -p "${artifact_dir}"
sbatch_path="${artifact_dir}/job.sbatch"
render >"${sbatch_path}"
chmod 700 "${sbatch_path}"
sbatch --test-only "${sbatch_path}" 2>&1 | tee "${artifact_dir}/test-only.txt"
if [[ "${mode}" == --test-only ]]; then
  exit 0
fi
sbatch "${sbatch_path}" | tee "${artifact_dir}/submission.txt"
