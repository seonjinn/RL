#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
JOB_NAME="${JOB_NAME:-q235b-speculators-eagle3}"
DRY_RUN="${DRY_RUN:-false}"
DEPENDENCY="${DEPENDENCY:-}"

SCRIPT="$ROOT_DIR/experiments/eagle3_qwen3_235b/slurm_speculators_offline_pipeline.sbatch"

echo "# Submit vLLM Speculators offline EAGLE3 pipeline"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION"
echo "NUM_NODES=$NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE TIME_LIMIT=$TIME_LIMIT"
echo "JOB_NAME=$JOB_NAME"
echo "DEPENDENCY=$DEPENDENCY"
echo "DRY_RUN=$DRY_RUN"

sbatch_args=(
  --nodes="$NUM_NODES"
  --account="$ACCOUNT"
  --partition="$PARTITION"
  --gres="gpu:$GPUS_PER_NODE"
  --time="$TIME_LIMIT"
  --job-name="$JOB_NAME"
)
if [[ -n "$DEPENDENCY" ]]; then
  sbatch_args+=(--dependency="$DEPENDENCY")
fi

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  printf '%q ' sbatch "${sbatch_args[@]}" "$SCRIPT"
  printf '\n'
  exit 0
fi

submit_out=$(
  sbatch "${sbatch_args[@]}" "$SCRIPT"
)
echo "$submit_out"
job_id="$(awk '/Submitted batch job/{print $4}' <<<"$submit_out" | tail -1)"
if [[ -n "$job_id" ]]; then
  echo "$job_id" > "$ROOT_DIR/latest_speculators_offline_pipeline_job.txt"
  echo "job_id=$job_id"
fi
