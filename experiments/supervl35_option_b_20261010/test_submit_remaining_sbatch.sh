#!/bin/bash
set -euo pipefail

if [[ ${NRL_TEST_SUBMIT:-0} == 1 ]]; then
    arm=$1
    case "$arm" in
        bf16-mxfp8) id=1002 ;;
        mxfp8-fp8params-on) id=1003 ;;
        mxfp8-fp8params-off) id=1004 ;;
        mxfp8-bf16params-on) id=1005 ;;
        mxfp8-bf16params-off) id=1006 ;;
        *) exit 2 ;;
    esac
    printf '%s %s %s\n' "$arm" "$SLURM_AFTEROK_JOB_ID" "$NRL_EXPERIMENT_SOURCE_COMMIT" >>"$NRL_TEST_LOG"
    printf '%s\n' "$id"
    exit
fi

launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ -f "$launcher_dir/submit_remaining_sbatch.sh" ]] || {
    echo "Remaining-arm launcher is missing" >&2
    exit 1
}
temp_dir=$(mktemp -d)
trap 'rm -r "$temp_dir"' EXIT
log="$temp_dir/submissions.log"
source_commit=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
NRL_TEST_SUBMIT=1 NRL_TEST_LOG="$log" NRL_ARM_SUBMITTER="$launcher_dir/test_submit_remaining_sbatch.sh" \
    NRL_EXPERIMENT_SOURCE_COMMIT="$source_commit" NRL_RUN_ROOT="$temp_dir" \
    bash "$launcher_dir/submit_remaining_sbatch.sh" 1001 >/dev/null

expected=$(printf '%s\n' \
    "bf16-mxfp8 1001 $source_commit" \
    "mxfp8-fp8params-on 1002 $source_commit" \
    "mxfp8-fp8params-off 1003 $source_commit" \
    "mxfp8-bf16params-on 1004 $source_commit" \
    "mxfp8-bf16params-off 1005 $source_commit")
actual=$(<"$log")
[[ "$actual" == "$expected" ]] || {
    printf 'Unexpected submission chain:\n%s\n' "$actual" >&2
    exit 1
}
[[ $(wc -l <"$temp_dir/matrix-afterok-1001/submissions.tsv") -eq 6 ]]
if NRL_TEST_SUBMIT=1 NRL_TEST_LOG="$log" NRL_ARM_SUBMITTER="$launcher_dir/test_submit_remaining_sbatch.sh" \
    NRL_EXPERIMENT_SOURCE_COMMIT="$source_commit" NRL_RUN_ROOT="$temp_dir" \
    bash "$launcher_dir/submit_remaining_sbatch.sh" 1001 >/dev/null 2>&1; then
    echo "Duplicate matrix submission was not rejected" >&2
    exit 1
fi
[[ $(wc -l <"$log") -eq 5 ]]
