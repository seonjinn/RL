#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=/home/sna/nemorl-bf16-flashinfer-specdec-cgscope-v2-20260910
readonly RECIPE="${SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260909_7023221.sqsh
readonly TARGET_MODEL=/lustre/fsw/portfolios/coreai/users/sna/hf-local/Qwen/Qwen3-30B-A3B
readonly PTV3_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-latest-main-bf16-flashinfer-specdec-20260909
readonly ACCOUNT="${Q30_LATEST_MAIN_ACCOUNT:-nemotron_n4_post}"
readonly MAX_STEPS="${Q30_LATEST_MAIN_MAX_STEPS:-3}"
readonly CONTEXT_LENGTH="${Q30_LATEST_MAIN_CONTEXT_LENGTH:-4096}"
readonly DIAGNOSTIC="${Q30_LATEST_MAIN_DIAGNOSTIC:-false}"
readonly NSYS_ENABLED="${Q30_LATEST_MAIN_NSYS:-false}"
readonly GRAPH_MODE="${Q30_LATEST_MAIN_GRAPH_MODE:-FAP}"

usage() {
  echo "usage: $0 --render|--test-only|--submit baseline|dflash_k3|dflash_k5|dflash_k7|dspark_k3|dspark_k5|dspark_k7" >&2
  exit 2
}

if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Q30_LATEST_MAIN_MAX_STEPS must be a positive integer: ${MAX_STEPS}" >&2
  exit 2
fi

case "${CONTEXT_LENGTH}" in
  4096|32768) ;;
  *)
    echo "Q30_LATEST_MAIN_CONTEXT_LENGTH must be 4096 or 32768: ${CONTEXT_LENGTH}" >&2
    exit 2
    ;;
esac

case "${DIAGNOSTIC}:${NSYS_ENABLED}:${GRAPH_MODE}" in
  false:false:FAP|true:true:FAP|true:true:NONE) ;;
  *)
    echo "invalid diagnostic controls: diagnostic=${DIAGNOSTIC} nsys=${NSYS_ENABLED} graph_mode=${GRAPH_MODE}" >&2
    exit 2
    ;;
esac

mode="${1:-}"
arm="${2:-}"
case "${mode}" in --render|--test-only|--submit) ;; *) usage ;; esac

method=""
checkpoint=""
arm_label=""
num_speculative_tokens=""
case "${arm}" in
  baseline) arm_label="Baseline" ;;
  dflash_k3|dflash_k5|dflash_k7|dspark_k3|dspark_k5|dspark_k7)
    method="${arm%%_k*}"
    num_speculative_tokens="${arm##*_k}"
    if [[ "${method}" == dflash ]]; then
      arm_label="DFlashK${num_speculative_tokens}"
    else
      arm_label="DSparkK${num_speculative_tokens}"
    fi
    checkpoint="${PTV3_ROOT}/sd2p3swa-q30-base-ptv3swe-${method}-b8-16n/exported-checkpoint-44000"
    ;;
  *) usage ;;
esac

context_segment=""
wandb_group="q30-latest-main-bf16-flashinfer-specdec"
walltime="02:00:00"
max_num_seqs=128
capture_sizes='[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512]'
if [[ "${CONTEXT_LENGTH}" == 32768 ]]; then
  context_segment="32K-CGScopeV2-"
  wandb_group="q30-latest-main-bf16-flashinfer-specdec-32k-cgscope-v2"
  walltime="04:00:00"
  max_num_seqs=16
  # Target verification schedules K+1 tokens per request. DSpark's
  # anchor-as-first drafter schedules K query tokens per request.
  target_query_width=$((${num_speculative_tokens:-0} + 1))
  draft_query_width=${target_query_width}
  if [[ "${method}" == dspark ]]; then
    draft_query_width=${num_speculative_tokens}
  fi
  max_capture_size=$((max_num_seqs * target_query_width))
  capture_sizes='['
  separator=''
  for ((tokens=1; tokens<=max_capture_size; tokens++)); do
    include_size=false
    if ((tokens % target_query_width == 0)); then
      include_size=true
    elif ((tokens % draft_query_width == 0 && tokens / draft_query_width <= max_num_seqs)); then
      draft_size_already_covered=false
      for ((requests=1; requests<=max_num_seqs; requests++)); do
        target_tokens=$((requests * target_query_width))
        rounded_for_draft=$((
          ((target_tokens + draft_query_width - 1) / draft_query_width) * draft_query_width
        ))
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
fi
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
graph_label="FAP"
cudagraph_mode="FULL_AND_PIECEWISE"
if [[ "${GRAPH_MODE}" == NONE ]]; then
  graph_label="NoGraph"
  cudagraph_mode="NONE"
fi
if [[ "${DIAGNOSTIC}" == true ]]; then
  context_segment="32K-CGDiag-${graph_label}-"
  wandb_group="q30-latest-main-bf16-flashinfer-specdec-32k-cgdiag"
fi
run_id="Qwen3-30BA3B-latest-main-BF16-flashinfer-${context_segment}${arm_label}-${MAX_STEPS}step-${graph_label}-${timestamp}"
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
  "++policy.generation.vllm_kwargs.max_num_seqs=${max_num_seqs}"
  'policy.generation.vllm_cfg.enforce_eager=false'
  "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=${cudagraph_mode}"
  "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${capture_sizes}"
)
if [[ "${CONTEXT_LENGTH}" == 32768 ]]; then
  spec_overrides+=(
    'grpo.num_prompts_per_step=16'
    'grpo.num_generations_per_prompt=16'
    'policy.train_global_batch_size=256'
    'policy.train_micro_batch_size=1'
    'policy.logprob_batch_size=1'
    'policy.max_total_sequence_length=32768'
    'policy.generation.max_new_tokens=32768'
    'policy.generation.vllm_cfg.max_model_len=32768'
    'policy.sequence_packing.train_mb_tokens=32768'
    'policy.sequence_packing.logprob_mb_tokens=32768'
    'policy.megatron_cfg.context_parallel_size=2'
    'policy.megatron_cfg.activation_checkpointing=true'
    'policy.megatron_cfg.empty_unused_memory_level=2'
    'policy.make_sequence_length_divisible_by=8'
    '++policy.generation.vllm_kwargs.max_num_batched_tokens=32768'
  )
fi
if [[ "${arm}" == baseline ]]; then
  spec_overrides+=('++policy.generation.vllm_kwargs.speculative_config=null')
else
  spec_overrides+=(
    "++policy.generation.vllm_kwargs.speculative_config.method=${method}"
    "++policy.generation.vllm_kwargs.speculative_config.model=${checkpoint}"
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${num_speculative_tokens}"
    '++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1'
  )
  if [[ "${method}" == dspark ]]; then
    spec_overrides+=(
      '++policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN'
      '++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false'
    )
  fi
fi

if [[ "${DIAGNOSTIC}" == true ]]; then
  spec_overrides+=('grpo.seed=42')
fi

printf -v overrides ' %q' \
  "grpo.max_num_steps=${MAX_STEPS}" \
  "policy.model_name=${TARGET_MODEL}" \
  "policy.tokenizer.name=${TARGET_MODEL}" \
  'logger.wandb_enabled=true' \
  'logger.wandb.project=sna-specdec' \
  "++logger.wandb.group=${wandb_group}" \
  "logger.wandb.name=${run_id}" \
  "logger.log_dir=${artifact_dir}/logs" \
  "${spec_overrides[@]}"

render() {
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${ACCOUNT}.${run_id}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --time=${walltime}
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
export UV_CACHE_DIR="\${Q30_NODE_ROOT}/uv-cache"
export PYTHONPATH="\${Q30_VLLM_OVERLAY}:\${Q30_MCORE_OVERLAY}:${SOURCE_ROOT}:\${PYTHONPATH:-}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export SETUP_COMMAND='set -euo pipefail; mkdir -p "\${Q30_MCORE_OVERLAY}"; cp -a "\${Q30_MCORE_SOURCE}/megatron" "\${Q30_MCORE_OVERLAY}/"; test -f "\${Q30_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp"'
${post_sync_lines}
$(if [[ "${NSYS_ENABLED}" == true ]]; then cat <<'NSYS'
export NRL_NSYS_WORKER_PATTERNS=vllm_generation_worker
export NRL_NSYS_PROFILE_STEP_RANGE=2:3
export NRL_NSYS_EXTRA_OPTIONS='{"cuda-graph-trace":"node","cpuctxsw":"none"}'
export RAY_LOG_SYNC_FREQUENCY=30
NSYS
fi)
export NRL_FORCE_REBUILD_VENVS=true
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=10
export COMMAND="export NEMO_RL_VENV_DIR=\"\${Q30_NODE_ROOT}/venvs\"; export UV_CACHE_DIR=\"\${Q30_NODE_ROOT}/uv-cache\"; mkdir -p \"\${NEMO_RL_VENV_DIR}\" \"\${UV_CACHE_DIR}\"; cd ${SOURCE_ROOT} && uv run examples/run_grpo.py --config ${RECIPE}${overrides}"
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
