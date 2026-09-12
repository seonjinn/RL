#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=/home/sna/nemorl-q30-dapo-concurrency-20260912
readonly RECIPE="${SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260909_7023221.sqsh
readonly VLLM_PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
readonly TARGET_MODEL=/lustre/fsw/portfolios/coreai/users/sna/hf-local/Qwen/Qwen3-30B-A3B
readonly DRAFTER_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-dapo-concurrency-20260912
readonly ACCOUNT="${Q30_DAPO47K_ACCOUNT:-coreai_dlalgo_llm}"
readonly MAX_STEPS="${Q30_DAPO47K_MAX_STEPS:-1}"
readonly MAX_NUM_SEQS="${3:-}"
readonly DEPENDENCY="${4:-}"

usage() {
  echo "usage: $0 --render|--test-only|--submit baseline|dflash_k5|dspark_k5 16|32|64|128 [gate_job_id]" >&2
  exit 2
}

if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Q30_DAPO47K_MAX_STEPS must be a positive integer: ${MAX_STEPS}" >&2
  exit 2
fi

mode="${1:-}"
arm="${2:-}"
case "${mode}" in --render|--test-only|--submit) ;; *) usage ;; esac

case "${MAX_NUM_SEQS}" in 16|32|64|128) ;; *) usage ;; esac
if [[ -n "${DEPENDENCY}" && ! "${DEPENDENCY}" =~ ^[1-9][0-9]*$ ]]; then usage; fi
[[ $# -le 4 ]] || usage

k=0
method=none
arm_label=Baseline
case "${arm}" in
  baseline) ;;
  dflash_k5) k=5; method=dflash; arm_label=DFlashK5 ;;
  dspark_k5) k=5; method=dspark; arm_label=DSparkK5 ;;
  *) usage ;;
esac

readonly DRAFTER="${DRAFTER_ROOT}/sd2p3swa-q30-base-ptv3swe-${method}-b8-16n/exported-checkpoint-44000"

# Include geometric request buckets for each query width, including terminal shapes.
widths=(1)
if ((k > 0)); then widths+=("$((k + 1))"); fi
if [[ "${method}" == dspark ]]; then widths+=("${k}"); fi
capture_values="$(
  for width in "${widths[@]}"; do
    for ((requests=1; requests<=MAX_NUM_SEQS; requests*=2)); do
      printf '%s\n' "$((requests * width))"
    done
  done | sort -nu | paste -sd, -
)"
readonly capture_sizes="[${capture_values}]"
readonly walltime=04:00:00

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="Qwen3-30BA3B-DAPO-${arm_label}-S${MAX_NUM_SEQS}-${MAX_STEPS}step-${timestamp}"
artifact_dir="${DURABLE_ROOT}/${run_id}"

setup_dspark=''
if [[ "${method}" == dspark ]]; then
  setup_dspark="; ${VLLM_PYTHON} ${SOURCE_ROOT}/experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/prepare_vllm_dspark_fap_overlay.py --overlay-root \"\${Q30_VLLM_OVERLAY}\""
fi

# This starts from the proven Qwen3 vLLM performance recipe and explicitly
# transplants the DAPOMath17K 47,104-response/49,152-total workload contract.
# This cohort varies concurrency plus graph shapes; every arm is frozen.
overrides=(
  "grpo.max_num_steps=${MAX_STEPS}"
  'grpo.num_prompts_per_step=128'
  'grpo.num_generations_per_prompt=16'
  'grpo.use_leave_one_out_baseline=false'
  'grpo.reward_scaling.enabled=true'
  'grpo.reward_scaling.target_min=-1.0'
  'grpo.reward_shaping.enabled=true'
  'grpo.reward_shaping.overlong_buffer_length=2048'
  'grpo.reward_shaping.max_response_length=47104'
  'loss_fn.force_on_policy_ratio=false'
  'loss_fn.reference_policy_kl_penalty=0.0'
  'loss_fn.ratio_clip_max=0.28'
  'loss_fn.ratio_clip_c=10'
  'loss_fn.use_on_policy_kl_approximation=true'
  'loss_fn.use_importance_sampling_correction=true'
  "policy.model_name=${TARGET_MODEL}"
  "policy.tokenizer.name=${TARGET_MODEL}"
  'policy.precision=bfloat16'
  '++policy.hf_config_overrides.router_aux_loss_coef=0'
  'policy.train_global_batch_size=2048'
  'policy.train_micro_batch_size=1'
  'policy.logprob_batch_size=1'
  'policy.logprob_chunk_size=2048'
  'policy.max_total_sequence_length=49152'
  'policy.sequence_packing.enabled=true'
  'policy.sequence_packing.train_mb_tokens=49152'
  'policy.sequence_packing.logprob_mb_tokens=49152'
  'policy.draft.enabled=false'
  '++policy.offload_optimizer_for_refit=false'
  'policy.generation.refit_transport=null'
  'policy.generation.vllm_cfg.refit_with_reload_api=false'
  'policy.megatron_cfg.activation_checkpointing=true'
  'policy.megatron_cfg.tensor_model_parallel_size=2'
  'policy.megatron_cfg.pipeline_model_parallel_size=1'
  'policy.megatron_cfg.expert_model_parallel_size=8'
  'policy.megatron_cfg.context_parallel_size=4'
  'policy.megatron_cfg.sequence_parallel=true'
  'policy.megatron_cfg.moe_router_dtype=fp32'
  'policy.megatron_cfg.bias_activation_fusion=false'
  'policy.megatron_cfg.defer_fp32_logits=true'
  'policy.megatron_cfg.moe_per_layer_logging=true'
  'policy.megatron_cfg.checkpoint.async_save=false'
  'policy.megatron_cfg.optimizer.lr=1.0e-06'
  'policy.megatron_cfg.optimizer.min_lr=1.0e-06'
  'policy.megatron_cfg.optimizer.weight_decay=0.1'
  'policy.megatron_cfg.scheduler.lr_decay_iters=null'
  'policy.megatron_cfg.scheduler.lr_warmup_iters=10'
  'policy.megatron_cfg.scheduler.lr_warmup_init=1.0e-07'
  'policy.megatron_cfg.empty_unused_memory_level=2'
  'policy.make_sequence_length_divisible_by=16'
  'policy.generation.max_new_tokens=47104'
  'policy.generation.vllm_cfg.max_model_len=49152'
  'policy.generation.vllm_cfg.gpu_memory_utilization=0.7'
  'policy.generation.vllm_cfg.enforce_eager=false'
  'policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm'
  "++policy.generation.vllm_kwargs.max_num_seqs=${MAX_NUM_SEQS}"
  '++policy.generation.vllm_kwargs.max_num_batched_tokens=49152'
  '++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE'
  "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${capture_sizes}"
  'data.max_input_seq_length=2048'
  'data.train.dataset_name=DAPOMath17K'
  'data.train.split_validation_size=0.0'
  'data.validation=null'
  'data.default.prompt_file=null'
  'env.math.num_workers=16'
  'env.math.math_verify_impl=dapo_math_verify'
  'cluster.gpus_per_node=4'
  'cluster.num_nodes=8'
  'cluster.segment_size=4'
  'checkpointing.enabled=false'
  'logger.wandb_enabled=true'
  'logger.wandb.project=sna-specdec'
  '++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false'
  '++logger.wandb.group=q30-dapo-gbs2048-concurrency-20260912'
  "logger.wandb.name=${run_id}"
  "logger.log_dir=${artifact_dir}/logs"
)
if [[ "${arm}" == baseline ]]; then
  overrides+=('++policy.generation.vllm_kwargs.speculative_config=null')
else
  overrides+=(
    "++policy.generation.vllm_kwargs.speculative_config.method=${method}"
    "++policy.generation.vllm_kwargs.speculative_config.model=${DRAFTER}"
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${k}"
    '++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1'
    '++policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN'
  )
fi
printf -v override_args ' %q' "${overrides[@]}"

render() {
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${ACCOUNT}.${run_id}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --time=${walltime}
$(if [[ -n "${DEPENDENCY}" ]]; then printf '#SBATCH --dependency=afterok:%s\n#SBATCH --kill-on-invalid-dep=yes\n' "${DEPENDENCY}"; fi)
#SBATCH --nodes=8
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
$(if ((k > 0)); then printf 'test -f "%s/model.safetensors"\n' "${DRAFTER}"; fi)
mkdir -p "${artifact_dir}"
git -C "${SOURCE_ROOT}" rev-parse HEAD | tee "${artifact_dir}/source_sha.txt"
git -C "${SOURCE_ROOT}" submodule status --recursive >"${artifact_dir}/submodules.txt"
export CONTAINER="${CONTAINER}"
export MOUNTS=/lustre:/lustre,/home:/home,/raid:/raid
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export BASE_LOG_DIR="${artifact_dir}"
export RAY_TMPDIR=/raid/scratch/sna/r\${SLURM_JOB_ID}
export Q30_NODE_ROOT="/raid/scratch/sna/q30-dapo-concurrency-\${SLURM_JOB_ID}"
export Q30_MCORE_SOURCE="${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export Q30_MCORE_OVERLAY="\${Q30_NODE_ROOT}/mcore-overlay"
export Q30_VLLM_OVERLAY="\${Q30_NODE_ROOT}/vllm-overlay"
export PYTHONPATH="\${Q30_VLLM_OVERLAY}:\${Q30_MCORE_OVERLAY}:${SOURCE_ROOT}:\${PYTHONPATH:-}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export SETUP_COMMAND='set -euo pipefail; mkdir -p "\${RAY_TMPDIR}"; mkdir -p "\${Q30_MCORE_OVERLAY}"; cp -a "\${Q30_MCORE_SOURCE}/megatron" "\${Q30_MCORE_OVERLAY}/"; test -f "\${Q30_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp"; test -x ${VLLM_PYTHON}; ${VLLM_PYTHON} -c "import vllm"${setup_dspark}'
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
