#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=/home/sna/nemorl-bf16-flashinfer-specdec-cgscope-v2-20260910
readonly RECIPE="${SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260909_7023221.sqsh
readonly TARGET_MODEL=/lustre/fsw/portfolios/coreai/users/sna/hf-local/Qwen/Qwen3-30B-A3B
readonly DRAFTER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa/sd2p3swa-q30-base-ptv3swe-dspark-b8-16n/exported-checkpoint-44000
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-gbs2048-4k-vllm0251-dspark-20260911
readonly ACCOUNT="${Q30_GBS2048_ACCOUNT:-nemotron_n4_post}"
readonly MAX_STEPS="${Q30_GBS2048_MAX_STEPS:-20}"
readonly MAX_NUM_SEQS=128

usage() {
  echo "usage: $0 --render|--test-only|--submit baseline|dspark_k3|dspark_k5|dspark_k7|dspark_dynamic" >&2
  exit 2
}

if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Q30_GBS2048_MAX_STEPS must be a positive integer: ${MAX_STEPS}" >&2
  exit 2
fi

mode="${1:-}"
arm="${2:-}"
case "${mode}" in --render|--test-only|--submit) ;; *) usage ;; esac

k=0
arm_label=Baseline
case "${arm}" in
  baseline) ;;
  dspark_k3|dspark_k5|dspark_k7)
    k="${arm##*_k}"
    arm_label="DSparkK${k}"
    ;;
  dspark_dynamic)
    echo "vLLM 0.25.1 DynamicSD is diagnostic-only: scheduler-selected K with reduced draft work is not proven" >&2
    exit 3
    ;;
  *) usage ;;
esac

# FAP sees K+1 target verification tokens per request. DSpark's anchor is its
# first draft token, so the draft query width is K. Cover every rounded target
# and draft batch shape for 1..MAX_NUM_SEQS without capturing every integer.
target_query_width=$((k + 1))
draft_query_width="${target_query_width}"
if ((k > 0)); then
  draft_query_width="${k}"
fi
max_capture_size=$((MAX_NUM_SEQS * target_query_width))
capture_sizes='['
separator=''
for ((tokens=1; tokens<=max_capture_size; tokens++)); do
  include_size=false
  if ((tokens % target_query_width == 0)); then
    include_size=true
  elif ((tokens % draft_query_width == 0 && tokens / draft_query_width <= MAX_NUM_SEQS)); then
    draft_size_already_covered=false
    for ((requests=1; requests<=MAX_NUM_SEQS; requests++)); do
      target_tokens=$((requests * target_query_width))
      rounded_for_draft=$(((target_tokens + draft_query_width - 1) / draft_query_width * draft_query_width))
      if ((rounded_for_draft == tokens)); then
        draft_size_already_covered=true
        break
      fi
    done
    if [[ "${draft_size_already_covered}" == false ]]; then
      include_size=true
    fi
  fi
  if [[ "${include_size}" == true ]]; then
    capture_sizes+="${separator}${tokens}"
    separator=','
  fi
done
capture_sizes+=']'
readonly capture_sizes

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="Qwen3-30BA3B-GBS2048-4K-vLLM0251-${arm_label}-${MAX_STEPS}step-FAP-${timestamp}"
artifact_dir="${DURABLE_ROOT}/${run_id}"

setup_dspark=''
if ((k > 0)); then
  setup_dspark='; /opt/nemo_rl_venv/bin/python ${SOURCE_ROOT}/experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/prepare_vllm_dspark_fap_overlay.py --overlay-root "${Q30_VLLM_OVERLAY}"'
fi

overrides=(
  "grpo.max_num_steps=${MAX_STEPS}"
  "policy.model_name=${TARGET_MODEL}"
  "policy.tokenizer.name=${TARGET_MODEL}"
  'policy.precision=bfloat16'
  'policy.draft.enabled=false'
  'policy.sequence_packing.enabled=true'
  '++policy.offload_optimizer_for_refit=false'
  'policy.generation.refit_transport=null'
  'policy.generation.vllm_cfg.refit_with_reload_api=false'
  'policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm'
  "++policy.generation.vllm_kwargs.max_num_seqs=${MAX_NUM_SEQS}"
  'policy.generation.vllm_cfg.enforce_eager=false'
  '++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE'
  "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${capture_sizes}"
  'logger.wandb_enabled=true'
  'logger.wandb.project=sna-specdec'
  '++logger.wandb.group=q30-gbs2048-4k-vllm0251-dspark-fixed'
  "logger.wandb.name=${run_id}"
  "logger.log_dir=${artifact_dir}/logs"
)
if [[ "${arm}" == baseline ]]; then
  overrides+=('++policy.generation.vllm_kwargs.speculative_config=null')
else
  overrides+=(
    '++policy.generation.vllm_kwargs.speculative_config.method=dspark'
    "++policy.generation.vllm_kwargs.speculative_config.model=${DRAFTER}"
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${k}"
    '++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1'
    '++policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN'
    '++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false'
  )
fi
printf -v override_args ' %q' "${overrides[@]}"

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
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
test -n "\${WANDB_API_KEY:-}"
test -z "\$(git -C ${SOURCE_ROOT} status --porcelain=v1 --untracked-files=all)"
test -r "${CONTAINER}"
test -f "${RECIPE}"
test -d "${TARGET_MODEL}"
$(if ((k > 0)); then printf 'test -f "%s/model.safetensors"\n' "${DRAFTER}"; fi)
mkdir -p "${artifact_dir}"
git -C "${SOURCE_ROOT}" rev-parse HEAD | tee "${artifact_dir}/source_sha.txt"
git -C "${SOURCE_ROOT}" submodule status --recursive >"${artifact_dir}/submodules.txt"
export CONTAINER="${CONTAINER}"
export MOUNTS=/lustre:/lustre,/home:/home,/raid:/raid
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export BASE_LOG_DIR="${artifact_dir}"
export Q30_NODE_ROOT="/raid/scratch/sna/q30-gbs2048-dspark-\${SLURM_JOB_ID}"
export Q30_MCORE_SOURCE="${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export Q30_MCORE_OVERLAY="\${Q30_NODE_ROOT}/mcore-overlay"
export Q30_VLLM_OVERLAY="\${Q30_NODE_ROOT}/vllm-overlay"
export PYTHONPATH="\${Q30_VLLM_OVERLAY}:\${Q30_MCORE_OVERLAY}:${SOURCE_ROOT}:\${PYTHONPATH:-}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export SETUP_COMMAND='set -euo pipefail; mkdir -p "\${Q30_MCORE_OVERLAY}"; cp -a "\${Q30_MCORE_SOURCE}/megatron" "\${Q30_MCORE_OVERLAY}/"; test -f "\${Q30_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp"${setup_dspark}'
export COMMAND="cd ${SOURCE_ROOT} && /opt/nemo_rl_venv/bin/python examples/run_grpo.py --config ${RECIPE}${override_args}"
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
[[ "${arm}" == baseline ]] || test -f "${DRAFTER}/model.safetensors"
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
