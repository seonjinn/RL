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
    printf '%s %s %s\n' "$arm" "${SLURM_AFTEROK_JOB_ID:-}" "$NRL_EXPERIMENT_SOURCE_COMMIT" >>"$NRL_TEST_LOG"
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
SLURM_AFTEROK_JOB_ID=777 NRL_TEST_SUBMIT=1 NRL_TEST_LOG="$log" NRL_ARM_SUBMITTER="$launcher_dir/test_submit_remaining_sbatch.sh" \
    NRL_EXPERIMENT_SOURCE_COMMIT="$source_commit" NRL_RUN_ROOT="$temp_dir" \
    bash "$launcher_dir/submit_remaining_sbatch.sh" 1001 >/dev/null

expected=$(printf '%s\n' \
    "bf16-mxfp8  $source_commit" \
    "mxfp8-fp8params-on  $source_commit" \
    "mxfp8-fp8params-off  $source_commit" \
    "mxfp8-bf16params-on  $source_commit" \
    "mxfp8-bf16params-off  $source_commit")
actual=$(<"$log")
[[ "$actual" == "$expected" ]] || {
    printf 'Unexpected submission dependencies:\n%s\n' "$actual" >&2
    exit 1
}
record="$temp_dir/matrix-parallel-1001/submissions.tsv"
[[ $(wc -l <"$record") -eq 6 ]] || { echo "Wrong record count" >&2; exit 1; }
[[ $(head -n 1 "$record") == $'source_commit\tarm\tjob_id' ]] || { echo "Wrong record header" >&2; exit 1; }
[[ $(awk -F '\t' 'NF != 3 {count++} END {print count+0}' "$record") -eq 0 ]] || {
    echo "Unexpected record fields" >&2
    exit 1
}
if NRL_TEST_SUBMIT=1 NRL_TEST_LOG="$log" NRL_ARM_SUBMITTER="$launcher_dir/test_submit_remaining_sbatch.sh" \
    NRL_EXPERIMENT_SOURCE_COMMIT="$source_commit" NRL_RUN_ROOT="$temp_dir" \
    bash "$launcher_dir/submit_remaining_sbatch.sh" 1001 >/dev/null 2>&1; then
    echo "Duplicate matrix submission was not rejected" >&2
    exit 1
fi
[[ $(wc -l <"$log") -eq 5 ]] || { echo "Duplicate submissions occurred" >&2; exit 1; }

old_root="$temp_dir/old-run"
mkdir -p "$old_root/matrix-afterok-1001"
if NRL_TEST_SUBMIT=1 NRL_TEST_LOG="$log" NRL_ARM_SUBMITTER="$launcher_dir/test_submit_remaining_sbatch.sh" \
    NRL_EXPERIMENT_SOURCE_COMMIT="$source_commit" NRL_RUN_ROOT="$old_root" \
    bash "$launcher_dir/submit_remaining_sbatch.sh" 1001 >/dev/null 2>&1; then
    echo "Existing chained submissions were not detected" >&2
    exit 1
fi
[[ $(wc -l <"$log") -eq 5 ]] || { echo "Duplicate submissions occurred" >&2; exit 1; }
