#!/usr/bin/env bash

set -euo pipefail

BASE=/lustre/fsw/coreai_dlalgo_llm/users/sna
EXP_ROOT=$BASE/pretyche_force_on_policy_ratio_q30_q32_20260707
SMOKE_JOBS=$EXP_ROOT/results/smoke_jobs.tsv
PERFORMANCE_JOBS=$EXP_ROOT/results/jobs.tsv
STATUS_FILE=$EXP_ROOT/results/watcher_status.tsv
SUBMITTER=$EXP_ROOT/scripts/submit_force_on_policy_matrix.sh
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-28800}

test -s "$SMOKE_JOBS"
test -x "$SUBMITTER"
test "$(awk 'END {print NR - 1}' "$SMOKE_JOBS")" -eq 4
test ! -e "$PERFORMANCE_JOBS"
[[ $POLL_SECONDS =~ ^[0-9]+$ ]]
[[ $MAX_WAIT_SECONDS =~ ^[0-9]+$ ]]

deadline=$(( $(date +%s) + MAX_WAIT_SECONDS ))

job_state() {
    local job_id=$1
    local state
    state=$(sacct -j "$job_id" -n -X -o State | awk 'NF {print $1; exit}')
    if [[ -z $state ]]; then
        state=$(squeue -j "$job_id" -h -o '%T' | awk 'NF {print $1; exit}')
    fi
    printf '%s' "${state:-UNKNOWN}"
}

write_status() {
    local tmp_file=$STATUS_FILE.tmp.$$
    printf 'timestamp\trun_key\tjob_id\tstate\n' > "$tmp_file"
    while IFS=$'\t' read -r run_key job_id _; do
        [[ $run_key == run_key ]] && continue
        printf '%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "$run_key" "$job_id" "$(job_state "$job_id")" \
            >> "$tmp_file"
    done < "$SMOKE_JOBS"
    mv "$tmp_file" "$STATUS_FILE"
    cat "$STATUS_FILE"
}

while true; do
    write_status
    active=0
    failed=0
    while IFS=$'\t' read -r run_key job_id _; do
        [[ $run_key == run_key ]] && continue
        state=$(job_state "$job_id")
        case "$state" in
            COMPLETED)
                ;;
            PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING|UNKNOWN)
                active=1
                ;;
            *)
                printf 'Smoke job %s (%s) reached terminal failure state %s.\n' \
                    "$job_id" "$run_key" "$state" >&2
                failed=1
                ;;
        esac
    done < "$SMOKE_JOBS"

    [[ $failed == 0 ]] || exit 1
    [[ $active == 1 ]] || break
    if (( $(date +%s) >= deadline )); then
        printf 'Timed out waiting for smoke jobs after %s seconds.\n' \
            "$MAX_WAIT_SECONDS" >&2
        exit 2
    fi
    sleep "$POLL_SECONDS"
done

verify_smoke_gate() {
    TEST_ONLY=1 SMOKE_ONLY=0 "$SUBMITTER"
}

verify_smoke_gate
TEST_ONLY=0 SMOKE_ONLY=0 "$SUBMITTER"

test -s "$PERFORMANCE_JOBS"
test "$(awk 'END {print NR - 1}' "$PERFORMANCE_JOBS")" -eq 4
performance_ids=$(awk 'NR > 1 {print $2}' "$PERFORMANCE_JOBS" | paste -sd, -)

for iteration in $(seq 1 10); do
    printf 'PERFORMANCE_MONITOR iteration=%s timestamp=%s\n' \
        "$iteration" "$(date --iso-8601=seconds)"
    squeue -j "$performance_ids" -h -o '%i|%T|%r|%S|%M|%N' || true
    if [[ $iteration -lt 10 ]]; then
        sleep 30
    fi
done
