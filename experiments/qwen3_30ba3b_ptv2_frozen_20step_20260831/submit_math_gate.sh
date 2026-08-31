#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=/home/sna/nemorl-q30-cadence-product-20260826
readonly SOURCE_SHA=d5c8bfa987025949699f7cfff188b349480bb8b5
readonly RECIPE="${SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
readonly CONTAINER=/lustre/fsw/portfolios/coreai/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly PTV2_ROOT=/lustre/fsw/portfolios/coreai/users/sna/specdec_ptv23/ptv2_final
readonly TARGET_MODEL=/lustre/fsw/portfolios/coreai/users/sna/hf-local/Qwen/Qwen3-30B-A3B
readonly DURABLE_ROOT=/lustre/fsw/portfolios/coreai/users/sna/experiments/q30-ptv2-frozen-20step-20260831/math
readonly ACCOUNT="${Q30_PTV2_ACCOUNT:-nemotron_n3_post}"
readonly CAPTURE_SIZES='[1,2,4,8,12,16,24,32,40,48,56,64,128,256]'

usage() {
  echo "usage: $0 --render|--test-only|--submit baseline|dflash_k7|dspark_k5" >&2
  exit 2
}

mode="${1:-}"
arm="${2:-}"
case "${mode}" in --render|--test-only|--submit) ;; *) usage ;; esac
case "${arm}" in baseline|dflash_k7|dspark_k5) ;; *) usage ;; esac

checkpoint=""
method=""
k=0
case "${arm}" in
  dflash_k7)
    checkpoint="${PTV2_ROOT}/sd2en-q30-base-ptv2en-dflash-b8-16n/exported-checkpoint-25391"
    method=dflash
    k=7
    ;;
  dspark_k5)
    checkpoint="${PTV2_ROOT}/sd2en-q30-base-ptv2en-dspark-b8-16n/exported-checkpoint-25391"
    method=dspark
    k=5
    ;;
esac

run_id="q30-ptv2-math-${arm}-frozen-$(date -u +%Y%m%dT%H%M%SZ)"
artifact_dir="${DURABLE_ROOT}/${run_id}"
post_sync_lines=""
if [[ "${method}" == dspark ]]; then
  post_sync_lines="export NRL_VENV_POST_SYNC_SCRIPT=${SOURCE_ROOT}/experiments/qwen3_30ba3b_draft_cadence_200step_20260826/prepare_vllm_dspark_fap_overlay.py
export NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
fi

spec_overrides=(
  'policy.draft.enabled=false'
  'policy.sequence_packing.enabled=true'
  '++policy.generation.vllm_kwargs.disable_custom_all_reduce=true'
  '++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE'
  "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${CAPTURE_SIZES}"
)
if [[ "${arm}" == baseline ]]; then
  spec_overrides+=('++policy.generation.vllm_kwargs.speculative_config=null')
else
  spec_overrides+=(
    "++policy.generation.vllm_kwargs.speculative_config.method=${method}"
    "++policy.generation.vllm_kwargs.speculative_config.model=${checkpoint}"
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${k}"
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
  "grpo.max_num_steps=20" \
  "policy.model_name=${TARGET_MODEL}" \
  "policy.tokenizer.name=${TARGET_MODEL}" \
  "logger.wandb_enabled=true" \
  "logger.wandb.project=sna-specdec" \
  "logger.wandb.name=${run_id}" \
  "logger.log_dir=${artifact_dir}/logs" \
  "${spec_overrides[@]}"

render() {
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
test -n "\${WANDB_API_KEY:-}"
test "\$(git -C ${SOURCE_ROOT} rev-parse HEAD)" = "${SOURCE_SHA}"
test -z "\$(git -C ${SOURCE_ROOT} status --porcelain=v1 --untracked-files=all)"
test -r "${CONTAINER}"
test -f "${RECIPE}"
test -d "${TARGET_MODEL}"
$(if [[ -n "${checkpoint}" ]]; then printf 'test -f "%s/model.safetensors"\n' "${checkpoint}"; fi)
mkdir -p "${artifact_dir}"
export CONTAINER="${CONTAINER}"
export MOUNTS=/lustre:/lustre,/home:/home,/raid:/raid
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export BASE_LOG_DIR="${artifact_dir}"
export Q30_NODE_ROOT="/raid/scratch/sna/q30-ptv2-\${SLURM_JOB_ID}"
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
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${SOURCE_SHA}"
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
test_output="$(sbatch --test-only "${sbatch_path}" 2>&1)"
printf '%s\n' "${test_output}" | tee "${artifact_dir}/test-only.txt"
if [[ "${mode}" == --test-only ]]; then
  exit 0
fi
sbatch "${sbatch_path}" | tee "${artifact_dir}/submission.txt"
