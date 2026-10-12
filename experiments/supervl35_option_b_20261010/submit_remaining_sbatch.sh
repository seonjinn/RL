#!/bin/bash
set -euo pipefail
umask 077

[[ $# -eq 1 && "$1" =~ ^[1-9][0-9]*$ ]] || {
    echo "Usage: NRL_EXPERIMENT_SOURCE_COMMIT=<prepared-commit> submit_remaining_sbatch.sh <bf16-bf16-job-id>" >&2
    exit 2
}
baseline_job_id=$1
source_commit=${NRL_EXPERIMENT_SOURCE_COMMIT:?Set NRL_EXPERIMENT_SOURCE_COMMIT to the prepared runtime commit}
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || {
    echo "NRL_EXPERIMENT_SOURCE_COMMIT must be a full 40-character SHA" >&2
    exit 2
}
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
submitter=${NRL_ARM_SUBMITTER:-$launcher_dir/submit_arm_sbatch.sh}
[[ -f "$submitter" ]] || { echo "Arm launcher is missing: $submitter" >&2; exit 1; }
run_root=${NRL_RUN_ROOT:-$HOME/supervl35-optionb-control}
mkdir -p "$run_root"
if [[ -e "$run_root/matrix-afterok-$baseline_job_id" ]]; then
    echo "Existing dependent submissions for baseline $baseline_job_id; inspect them before retrying" >&2
    exit 1
fi
record_dir="$run_root/matrix-parallel-$baseline_job_id"
if ! mkdir "$record_dir" 2>/dev/null; then
    echo "Submission record already exists: $record_dir" >&2
    echo "Inspect its job IDs before retrying to avoid duplicate 32-node jobs" >&2
    exit 1
fi
record="$record_dir/submissions.tsv"
printf 'source_commit\tarm\tjob_id\n' >"$record"
printf 'Recording submissions in %s\n' "$record"

for arm in bf16-mxfp8 mxfp8-fp8params-on mxfp8-fp8params-off \
           mxfp8-bf16params-on mxfp8-bf16params-off; do
    if output=$(SLURM_AFTEROK_JOB_ID='' \
        NRL_SBATCH_PARSABLE=1 NRL_EXPERIMENT_SOURCE_COMMIT="$source_commit" NRL_RUN_ROOT="$run_root" \
        /bin/bash "$submitter" "$arm"); then
        printf '%s\n' "$output"
    else
        printf '%s\n' "$output" >&2
        echo "Stopped while submitting $arm; inspect already printed job IDs before retrying" >&2
        exit 1
    fi
    job_line=${output##*$'\n'}
    job_id=${job_line%%;*}
    [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || {
        echo "Could not parse the submitted job ID for $arm; inspect output before retrying" >&2
        exit 1
    }
    printf 'Submitted %s as %s without dependency\n' "$arm" "$job_id"
    printf '%s\t%s\t%s\n' "$source_commit" "$arm" "$job_id" >>"$record"
done
