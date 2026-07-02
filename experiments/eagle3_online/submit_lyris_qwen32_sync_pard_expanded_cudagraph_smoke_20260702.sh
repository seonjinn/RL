#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-login-lyris}"
REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701}"
K="${K:-9}"
MAX_STEPS="${MAX_STEPS:-3}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
WALLTIME="${WALLTIME:-01:00:00}"
SUBMIT="${SUBMIT:-false}"

case "${K}" in
  9)
    SOURCE_JOB_ID="${SOURCE_JOB_ID:-2261211}"
    SOURCE_RUN_ID="20260702_qwen32_sync_pardk9_drafttp2_noarrms_cudagraph_step20_r4"
    CAPTURE_SIZE="${CAPTURE_SIZE:-640}"
    ;;
  16)
    SOURCE_JOB_ID="${SOURCE_JOB_ID:-2261212}"
    SOURCE_RUN_ID="20260702_qwen32_sync_pardk16_drafttp2_noarrms_cudagraph_step20_r4"
    CAPTURE_SIZE="${CAPTURE_SIZE:-1088}"
    ;;
  *)
    echo "ERROR: K must be 9 or 16" >&2
    exit 2
    ;;
esac

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${SOURCE_RUN_ID}}"
RUN_ID="${RUN_ID:-20260702_lyris_qwen32_sync_pardk${K}_drafttp2_noarrms_cgcap${CAPTURE_SIZE}_smoke${MAX_STEPS}_r1}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${RUN_ID}}"
WANDB_NAME="${WANDB_NAME:-qwen32_perfcfg_sync_pardk${K}_drafttp2_noarrms_cgcap${CAPTURE_SIZE}_cudagraph_smoke${MAX_STEPS}_r1_20260702}"

ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" bash -s -- \
  "${REMOTE_REPO}" \
  "${SOURCE_JOB_ID}" \
  "${SOURCE_RUN_ROOT}" \
  "${RUN_ROOT}" \
  "${RUN_ID}" \
  "${K}" \
  "${CAPTURE_SIZE}" \
  "${MAX_STEPS}" \
  "${WANDB_NAME}" \
  "${ACCOUNT}" \
  "${PARTITION}" \
  "${WALLTIME}" \
  "${SUBMIT}" <<'REMOTE'
set -euo pipefail

remote_repo="$1"
source_job_id="$2"
source_run_root="$3"
run_root="$4"
run_id="$5"
k="$6"
capture_size="$7"
max_steps="$8"
wandb_name="$9"
account="${10}"
partition="${11}"
walltime="${12}"
submit="${13}"

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
eval "${assignment}"

old_cache_root="/tmp/sna/${source_run_root##*/}"
new_cache_root="/tmp/sna/${run_id}"
COMMAND="${COMMAND//${source_run_root}/${run_root}}"
COMMAND="${COMMAND//${old_cache_root}/${new_cache_root}}"
COMMAND+=" grpo.max_num_steps=${max_steps}"
COMMAND+=" ++policy.generation.vllm_kwargs.compilation_config.max_cudagraph_capture_size=${capture_size}"
COMMAND+=" logger.log_dir=${run_root}/nemo_logs"
COMMAND+=" logger.wandb.name=${wandb_name}"

minimum_capture_size=$((64 * (k + 1)))
if (( capture_size < minimum_capture_size )); then
  echo "ERROR: capture_size=${capture_size} is below 64*(K+1)=${minimum_capture_size}" >&2
  exit 1
fi

for required in \
  "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${k}" \
  '++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true' \
  '++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=2' \
  '++policy.generation.vllm_kwargs.max_num_seqs=64' \
  'policy.generation.vllm_cfg.enforce_eager=false' \
  '++policy.generation.vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms=false' \
  "++policy.generation.vllm_kwargs.compilation_config.max_cudagraph_capture_size=${capture_size}"; do
  if [[ " ${COMMAND} " != *" ${required} "* ]]; then
    echo "ERROR: reconstructed command is missing ${required}" >&2
    exit 1
  fi
done

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
  --job-name="${account}-specdec.q32-pardk${k}-cgcap${capture_size}-smoke${max_steps}"
  --partition="${partition}"
  --time="${walltime}"
  --segment=4
  --network=sharp
  --output="${run_root}/slurm-%j.out"
)

echo "source_job_id=${source_job_id}"
echo "run_root=${run_root}"
echo "wandb_name=${wandb_name}"
echo "k=${k}"
echo "max_num_seqs=64"
echo "minimum_capture_size=${minimum_capture_size}"
echo "max_cudagraph_capture_size=${capture_size}"
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
