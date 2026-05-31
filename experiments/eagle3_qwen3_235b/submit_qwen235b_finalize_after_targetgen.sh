#!/usr/bin/env bash
set -euo pipefail

# Submit a lightweight finalization job for the Qwen3-235B 500K mixed-math
# corpus. By default it waits for all currently queued/running
# qwen3_235b-targetgen-resume jobs and then validates/merges the chunk files.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT_DIR/experiments/eagle3_qwen3_235b/finalize_qwen235b_mixed_500k_corpus.sh"

ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
JOB_NAME="${JOB_NAME:-qwen3_235b-finalize-mixed-500k}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
DRY_RUN="${DRY_RUN:-false}"
DEPENDENCY="${DEPENDENCY:-}"
DEPENDENCY_JOB_IDS="${DEPENDENCY_JOB_IDS:-}"
USER_NAME="${SLURM_USER:-${USER:-sna}}"

if [[ -z "$DEPENDENCY" ]]; then
  if [[ -z "$DEPENDENCY_JOB_IDS" ]]; then
    if command -v squeue >/dev/null 2>&1; then
      DEPENDENCY_JOB_IDS="$(
        squeue -u "$USER_NAME" -h -o '%i|%j' \
          | awk -F'|' '/qwen3_235b-targetgen-resume/ {print $1}' \
          | sort -n \
          | paste -sd, -
      )"
    fi
  fi
  if [[ -n "$DEPENDENCY_JOB_IDS" ]]; then
    DEPENDENCY="afterany:${DEPENDENCY_JOB_IDS//,/:}"
  fi
fi

mkdir -p "$ROOT_DIR/logs"

sbatch_args=(
  --nodes=1
  --ntasks=1
  --account="$ACCOUNT"
  --partition="$PARTITION"
  --gres="gpu:$GPUS_PER_NODE"
  --time="$TIME_LIMIT"
  --mem=0
  --job-name="$JOB_NAME"
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
)
if [[ -n "$DEPENDENCY" ]]; then
  sbatch_args+=(--dependency="$DEPENDENCY")
fi

wrap="cd '$ROOT_DIR' && bash '$SCRIPT'"

echo "# Submit Qwen3-235B mixed 500K finalizer"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT GPUS_PER_NODE=$GPUS_PER_NODE"
echo "JOB_NAME=$JOB_NAME"
echo "DEPENDENCY=$DEPENDENCY"
echo "DRY_RUN=$DRY_RUN"

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  printf '%q ' sbatch "${sbatch_args[@]}" --wrap "$wrap"
  printf '\n'
  exit 0
fi

submit_out="$(sbatch "${sbatch_args[@]}" --wrap "$wrap")"
echo "$submit_out"
job_id="$(awk '/Submitted batch job/{print $4}' <<<"$submit_out" | tail -1)"
if [[ -n "$job_id" ]]; then
  echo "$job_id" > "$ROOT_DIR/latest_qwen235b_finalize_job.txt"
  echo "job_id=$job_id"
fi
