#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-login-lyris}"
REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-2261912}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260702_lyris_qwen30_sync_osl32k_baseline_matched_step20_r1}"
RUN_ID="${RUN_ID:-20260702_lyris_qwen30_sync_osl32k_baseline_level2_step20_r1}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${RUN_ID}}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
WALLTIME="${WALLTIME:-03:00:00}"
WANDB_NAME="${WANDB_NAME:-qwen30ba3b_perfcfg_sync_osl32k_baseline_level2_cudagraph_cp2_pad8_step20_lyris_r1_20260702}"
SUBMIT="${SUBMIT:-false}"

ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" bash -s -- \
  "${REMOTE_REPO}" \
  "${SOURCE_JOB_ID}" \
  "${SOURCE_RUN_ROOT}" \
  "${RUN_ROOT}" \
  "${ACCOUNT}" \
  "${PARTITION}" \
  "${WALLTIME}" \
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
wandb_name="$8"
submit="$9"

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

# The source is a shell-xtrace assignment emitted by our own ray.sub launcher.
eval "${assignment}"

old_cache_root="/tmp/sna/nemorl_qwen30ba3b-baseline-tritonattn-step20-r1"
new_cache_root="/tmp/sna/nemorl_qwen30ba3b-osl32k-baseline-level2-step20-r1"
COMMAND="${COMMAND//${source_run_root}/${run_root}}"
COMMAND="${COMMAND//${old_cache_root}/${new_cache_root}}"
COMMAND+=" policy.megatron_cfg.empty_unused_memory_level=2"
COMMAND+=" logger.wandb.name=${wandb_name}"

for required in \
  'grpo.max_num_steps=20' \
  'policy.generation.vllm_cfg.enforce_eager=false' \
  'policy.max_total_sequence_length=32768' \
  'policy.megatron_cfg.context_parallel_size=2' \
  'policy.megatron_cfg.empty_unused_memory_level=2'; do
  if [[ " ${COMMAND} " != *" ${required} "* ]]; then
    echo "ERROR: reconstructed command is missing ${required}" >&2
    exit 1
  fi
done

source_driver_log="${source_run_root}/${source_job_id}-logs/ray-driver.log"
if [[ ! -f "${source_driver_log}" ]]; then
  echo "ERROR: source driver log does not exist: ${source_driver_log}" >&2
  exit 1
fi

master_config="$(grep -m1 'MasterConfig' "${source_driver_log}")"
if [[ "${master_config}" != *"'max_new_tokens': 32768"* ]]; then
  echo "ERROR: source job did not resolve policy.generation.max_new_tokens=32768" >&2
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
  --nodes=4
  --account="${account}"
  --job-name="${account}-specdec.q30-32k-baseline-level2"
  --partition="${partition}"
  --time="${walltime}"
  --segment=4
  --output="${run_root}/slurm-%j.out"
)

echo "source_job_id=${source_job_id}"
echo "run_root=${run_root}"
echo "wandb_name=${wandb_name}"
echo "config_delta=policy.megatron_cfg.empty_unused_memory_level:1->2"

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
