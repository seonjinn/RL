#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=/home/sna/nemorl-q30-flashinfer-specdec-gate-20260831
readonly SOURCE_SHA=15554749ae24361b5d511e72ddf41ecab2615cdc
readonly RECIPE="${SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
readonly CONTAINER=/lustre/fsw/portfolios/coreai/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly PTV2_ROOT=/lustre/fsw/portfolios/coreai/users/sna/specdec_ptv23/ptv2_final
readonly LEGACY_ROOT=/lustre/fsw/portfolios/coreai/users/sna/modelopt-specdec/training
readonly TARGET_MODEL=/lustre/fsw/portfolios/coreai/users/sna/hf-local/Qwen/Qwen3-30B-A3B
readonly DURABLE_ROOT=/lustre/fsw/portfolios/coreai/users/sna/experiments/q30-ptv2-frozen-20step-20260831/math
readonly ACCOUNT="${Q30_PTV2_ACCOUNT:-nemotron_n3_post}"
readonly MAX_STEPS="${Q30_PTV2_MAX_STEPS:-20}"

if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Q30_PTV2_MAX_STEPS must be a positive integer: ${MAX_STEPS}" >&2
  exit 2
fi

usage() {
  echo "usage: $0 --render|--test-only|--submit baseline|{ptv2,legacy}_{dflash,dspark}_k{1,2,3,5,7}" >&2
  exit 2
}

mode="${1:-}"
arm="${2:-}"
case "${mode}" in --render|--test-only|--submit) ;; *) usage ;; esac

checkpoint=""
method=""
cohort=""
k=0
case "${arm}" in
  baseline)
    cohort=matched
    ;;
  dflash_k7)
    cohort=ptv2
    method=dflash
    k=7
    ;;
  dspark_k5)
    cohort=ptv2
    method=dspark
    k=5
    ;;
  *)
    if [[ "${arm}" =~ ^(ptv2|legacy)_(dflash|dspark)_k(1|2|3|5|7)$ ]]; then
      cohort="${BASH_REMATCH[1]}"
      method="${BASH_REMATCH[2]}"
      k="${BASH_REMATCH[3]}"
    else
      usage
    fi
    ;;
esac

if [[ "${cohort}" == ptv2 ]]; then
  checkpoint="${PTV2_ROOT}/sd2en-q30-base-ptv2en-${method}-b8-16n/exported-checkpoint-25391"
elif [[ "${cohort}" == legacy ]]; then
  case "${method}" in
    dflash)
      checkpoint="${LEGACY_ROOT}/lyris-q30b-nemo-dflash-b8-16n-migrated-oci-s4400/exported-checkpoint-14500"
      ;;
    dspark)
      checkpoint="${LEGACY_ROOT}/lyris-q30b-nemo-dspark-b8-16n-migrated-oci-s5700/exported-checkpoint-14500"
      ;;
  esac
fi

case "${k}" in
  0|1) CAPTURE_SIZES='[1,2,4,8,16,32,64,128,256]' ;;
  2) CAPTURE_SIZES='[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384]' ;;
  3) CAPTURE_SIZES='[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512]' ;;
  5) CAPTURE_SIZES='[1,2,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,128,160,192,256,320,384,640,768]' ;;
  7) CAPTURE_SIZES='[1,2,4,7,8,14,16,28,32,56,64,112,128,224,256,448,512,896,1024]' ;;
  *) usage ;;
esac
readonly CAPTURE_SIZES

if [[ "${cohort}" == matched ]]; then
  run_label=baseline
else
  run_label="${method}_k${k}"
fi
run_id="q30-${cohort}-math-${run_label}-frozen-$(date -u +%Y%m%dT%H%M%SZ)"
artifact_dir="${DURABLE_ROOT}/${run_id}"
post_sync_lines=""
if [[ "${method}" == dspark ]]; then
  post_sync_lines="export NRL_VENV_POST_SYNC_SCRIPT=${SOURCE_ROOT}/experiments/qwen3_30ba3b_draft_cadence_200step_20260826/prepare_vllm_dspark_fap_overlay.py
export NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
fi

spec_overrides=(
  'policy.draft.enabled=false'
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
  "grpo.max_num_steps=${MAX_STEPS}" \
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
export PATH=/cm/local/apps/slurm/25.11/bin:\${PATH}
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
sbatch_args=()
if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  sbatch_args+=("--dependency=${SBATCH_DEPENDENCY}")
fi
test_output="$(sbatch --test-only "${sbatch_args[@]}" "${sbatch_path}" 2>&1)"
printf '%s\n' "${test_output}" | tee "${artifact_dir}/test-only.txt"
if [[ "${mode}" == --test-only ]]; then
  exit 0
fi
sbatch "${sbatch_args[@]}" "${sbatch_path}" | tee "${artifact_dir}/submission.txt"
