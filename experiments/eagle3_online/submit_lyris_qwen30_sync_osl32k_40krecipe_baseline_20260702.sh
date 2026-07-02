#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-login-lyris}"
REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-2261912}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260702_lyris_qwen30_sync_osl32k_baseline_matched_step20_r1}"
RUN_ID="${RUN_ID:-20260702_lyris_qwen30_sync_osl32k_40krecipe_8n4g_baseline_smoke3_r1}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${RUN_ID}}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
WALLTIME="${WALLTIME:-01:00:00}"
MAX_STEPS="${MAX_STEPS:-3}"
WANDB_NAME="${WANDB_NAME:-qwen30ba3b_perfcfg_sync_osl32k_40krecipe_8n4g_baseline_cudagraph_smoke3_lyris_r1_20260702}"
SUBMIT="${SUBMIT:-false}"

ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" bash -s -- \
  "${REMOTE_REPO}" \
  "${SOURCE_JOB_ID}" \
  "${SOURCE_RUN_ROOT}" \
  "${RUN_ROOT}" \
  "${ACCOUNT}" \
  "${PARTITION}" \
  "${WALLTIME}" \
  "${MAX_STEPS}" \
  "${WANDB_NAME}" \
  "${SUBMIT}" <<'REMOTE'
set -euo pipefail

remote_repo="$1"
source_job_id="$2"
source_run_root="$3"
run_root="$4"
account="$5"
partition="$6"
walltime="$7"
max_steps="$8"
wandb_name="$9"
submit="${10}"

source_log="${source_run_root}/slurm-${source_job_id}.out"
if [[ ! -f "${source_log}" ]]; then
  echo "ERROR: source log does not exist: ${source_log}" >&2
  exit 1
fi

assignment="$(grep -m1 '^+ COMMAND=' "${source_log}")"
assignment="${assignment#+ }"
if [[ "${assignment}" != COMMAND=* ]]; then
  echo "ERROR: could not recover COMMAND from ${source_log}" >&2
  exit 1
fi

# Reuse the known-good launch environment while changing only the recipe and
# topology required by the repository's existing 40K performance config.
eval "${assignment}"

old_config="${remote_repo}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
new_config="${remote_repo}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g-40K.yaml"
old_cache_root="/tmp/sna/nemorl_qwen30ba3b-baseline-tritonattn-step20-r1"
new_cache_root="/tmp/sna/nemorl_qwen30ba3b-osl32k-40krecipe-8n4g-baseline-smoke3-r1"
old_checkpoint_root="${remote_repo}/nrl_megatron_ckpts_online_draft_qwen30ba3b-baseline-tritonattn-step20-r1"
new_checkpoint_root="${remote_repo}/nrl_megatron_ckpts_qwen30ba3b-40krecipe-8n4g-baseline-r1"

if [[ "${COMMAND}" != *"${old_config}"* ]]; then
  echo "ERROR: source command is not based on ${old_config}" >&2
  exit 1
fi

COMMAND="${COMMAND//${source_run_root}/${run_root}}"
COMMAND="${COMMAND//${old_config}/${new_config}}"
COMMAND="${COMMAND//${old_cache_root}/${new_cache_root}}"
COMMAND="${COMMAND//${old_checkpoint_root}/${new_checkpoint_root}}"
COMMAND+=" cluster.num_nodes=8 cluster.gpus_per_node=4"
COMMAND+=" grpo.max_num_steps=${max_steps}"
COMMAND+=" grpo.num_prompts_per_step=16 grpo.num_generations_per_prompt=16"
COMMAND+=" policy.train_global_batch_size=256"
COMMAND+=" policy.max_total_sequence_length=32768"
COMMAND+=" policy.megatron_cfg.tensor_model_parallel_size=4"
COMMAND+=" policy.megatron_cfg.pipeline_model_parallel_size=1"
COMMAND+=" policy.megatron_cfg.expert_model_parallel_size=8"
COMMAND+=" policy.megatron_cfg.context_parallel_size=8"
COMMAND+=" policy.megatron_cfg.sequence_parallel=true"
COMMAND+=" policy.megatron_cfg.activation_checkpointing=true"
COMMAND+=" policy.megatron_cfg.empty_unused_memory_level=1"
COMMAND+=" policy.make_sequence_length_divisible_by=64"
COMMAND+=" policy.generation.vllm_cfg.tensor_parallel_size=2"
COMMAND+=" policy.generation.vllm_cfg.enforce_eager=false"
COMMAND+=" ++policy.generation.vllm_kwargs.attention_backend=TRITON_ATTN"
COMMAND+=" ++policy.generation.vllm_kwargs.max_num_batched_tokens=32768"
COMMAND+=" ++policy.generation.vllm_kwargs.max_num_seqs=16"
COMMAND+=" ++policy.generation.vllm_kwargs.kernel_config.moe_backend=triton"
COMMAND+=" logger.log_dir=${run_root}/nemo_logs"
COMMAND+=" logger.wandb_enabled=true"
COMMAND+=" logger.wandb.project=sna-nemorl-specdec-lyris"
COMMAND+=" logger.wandb.name=${wandb_name}"

for required in \
  "--config ${new_config}" \
  'cluster.num_nodes=8' \
  'cluster.gpus_per_node=4' \
  'grpo.num_prompts_per_step=16' \
  'grpo.num_generations_per_prompt=16' \
  'policy.max_total_sequence_length=32768' \
  'policy.megatron_cfg.tensor_model_parallel_size=4' \
  'policy.megatron_cfg.expert_model_parallel_size=8' \
  'policy.megatron_cfg.context_parallel_size=8' \
  'policy.megatron_cfg.activation_checkpointing=true' \
  'policy.generation.vllm_cfg.tensor_parallel_size=2' \
  'policy.generation.vllm_cfg.enforce_eager=false' \
  '++policy.generation.vllm_kwargs.attention_backend=TRITON_ATTN' \
  '++policy.generation.vllm_kwargs.kernel_config.moe_backend=triton'; do
  if [[ " ${COMMAND} " != *" ${required} "* ]]; then
    echo "ERROR: reconstructed command is missing ${required}" >&2
    exit 1
  fi
done

if [[ "${COMMAND}" == *"${old_config}"* ]]; then
  echo "ERROR: old 4n4g recipe remains in reconstructed command" >&2
  exit 1
fi

mkdir -p "${run_root}"
export COMMAND
export CONTAINER="/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly.sqsh"
export MOUNTS="/lustre:/lustre,/project:/project"
export BASE_LOG_DIR="${run_root}"
export ACCOUNT="${account}"
export PARTITION="${partition}"
export GPUS_PER_NODE=4

sbatch_args=(
  --parsable
  --nodes=8
  --account="${account}"
  --job-name="${account}-specdec.q30-32k-40krecipe-baseline-step${max_steps}"
  --partition="${partition}"
  --time="${walltime}"
  --segment=8
  --output="${run_root}/slurm-%j.out"
)

echo "source_job_id=${source_job_id}"
echo "run_root=${run_root}"
echo "wandb_name=${wandb_name}"
echo "recipe=grpo-qwen3-30ba3b-4n8g-40K.yaml"
echo "topology=8_nodes_x_4_gpus_preserving_32_recipe_gpus"
echo "max_steps=${max_steps}"

test_only_output="$(sbatch --test-only "${sbatch_args[@]}" "${remote_repo}/ray.sub" 2>&1)"
echo "${test_only_output}"
test_only_id="$(sed -nE 's/.*Job ([0-9]+).*/\1/p' <<<"${test_only_output}" | head -1)"
echo "test_only_job_id=${test_only_id}"

if [[ "${submit}" == "true" ]]; then
  job_id="$(sbatch "${sbatch_args[@]}" "${remote_repo}/ray.sub")"
  echo "job_id=${job_id}"
else
  echo "submission=skipped (set SUBMIT=true after reviewing test-only output)"
fi
REMOTE
