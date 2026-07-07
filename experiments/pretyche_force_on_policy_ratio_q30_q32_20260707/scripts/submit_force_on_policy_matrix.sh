#!/usr/bin/env bash

set -euo pipefail

BASE=/lustre/fsw/coreai_dlalgo_llm/users/sna
REPO=$BASE/nemo-rl-main-pr3030-q235-20260701
EXP_ROOT=$BASE/pretyche_force_on_policy_ratio_q30_q32_20260707
CONTRACT=$EXP_ROOT/manifests/config_contract.tsv
RUNNER=$EXP_ROOT/scripts/run_force_on_policy_benchmark.sbatch
EXPECTED_REPO_SHA=d4cfecf90db41cdf142629963b54b67ab479ab02
EXPECTED_CONTAINER_SHA=bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510
TEST_ONLY=${TEST_ONLY:-1}
SMOKE_ONLY=${SMOKE_ONLY:-1}
RUN_FILTER=${RUN_FILTER:-.*}
AFTEROK_JOB_ID=${AFTEROK_JOB_ID:-}
DEPENDENCY_LABEL=none
dependency_args=()

if [[ -n $AFTEROK_JOB_ID ]]; then
    [[ $AFTEROK_JOB_ID =~ ^[0-9]+$ ]]
    DEPENDENCY_LABEL=afterok:$AFTEROK_JOB_ID
    dependency_args=(--dependency=afterok:$AFTEROK_JOB_ID)
fi

if [[ $SMOKE_ONLY == 1 ]]; then
    MAX_STEPS=2
    TIME_LIMIT=02:00:00
    JOBS_TSV=$EXP_ROOT/results/smoke_jobs.tsv
else
    MAX_STEPS=20
    TIME_LIMIT=05:00:00
    JOBS_TSV=$EXP_ROOT/results/jobs.tsv
fi

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_REPO_SHA"
test "$(git -C "$REPO" rev-parse '@{u}')" = "$EXPECTED_REPO_SHA"
test "$(git -C "$REPO" remote get-url origin)" = https://github.com/seonjinn/RL.git
test -z "$(git -C "$REPO" status --porcelain --ignore-submodules=untracked)"
! git -C "$REPO" submodule status --recursive | grep -Eq '^[-+U]'
test -x "$RUNNER"
test -s "$CONTRACT"
CONTAINER=$BASE/containers/nemo_rl_nightly_20260630_0215.sqsh
test -s "$CONTAINER"
test "$(sha256sum "$CONTAINER" | awk '{print $1}')" = "$EXPECTED_CONTAINER_SHA"
test -r "$HOME/.nemo_rl_tokens"

runner_sha=$(sha256sum "$RUNNER" | awk '{print $1}')

verify_smoke_gate() {
    local smoke_jobs=$EXP_ROOT/results/smoke_jobs.tsv
    test -s "$smoke_jobs"
    test "$(awk 'END {print NR - 1}' "$smoke_jobs")" -eq 4

    while IFS=$'\t' read -r run_key job_id model config_name force_value nodes segment steps repo_sha recorded_runner_sha dependency; do
        if [[ $run_key == run_key ]]; then
            continue
        fi
        test "$steps" = 2
        test "$repo_sha" = "$EXPECTED_REPO_SHA"
        test "$recorded_runner_sha" = "$runner_sha"
        local state
        state=$(sacct -j "$job_id" -n -X -o State | awk 'NF {print $1; exit}')
        test "$state" = COMPLETED

        local log=$EXP_ROOT/results/$run_key/2step/run.log
        test -s "$log"
        grep -q 'train_data_step2.jsonl' "$log"
        ! grep -Eiq 'Traceback|RuntimeError|AssertionError|illegal memory|Segmentation fault|NCCL watchdog|CUDA error|out of memory' "$log"
        if [[ $force_value == true ]]; then
            grep -q 'force_on_policy_ratio enabled' "$log"
            grep -q 'Skipping prev_logprobs (force_on_policy_ratio=True)' "$log"
        else
            ! grep -q 'Skipping prev_logprobs (force_on_policy_ratio=True)' "$log"
        fi
    done < "$smoke_jobs"
}

if [[ $SMOKE_ONLY == 0 ]]; then
    verify_smoke_gate
fi

if [[ $TEST_ONLY == 0 ]]; then
    mkdir -p "$(dirname "$JOBS_TSV")"
    if [[ -s $JOBS_TSV ]] && [[ $(wc -l < "$JOBS_TSV") -gt 1 ]]; then
        printf 'Refusing duplicate submission: %s already contains jobs.\n' "$JOBS_TSV" >&2
        exit 2
    fi
    printf 'run_key\tjob_id\tmodel\tconfig_name\tforce_on_policy_ratio\tnodes\tsegment\tsteps\trepo_sha\trunner_sha\tdependency\n' > "$JOBS_TSV"
fi

submit_case() {
    local run_key=$1
    local model=$2
    local config_name=$3
    local force_value=$4
    local nodes=$5
    local segment=$6
    local -a submit_mode=(--test-only)
    if [[ $TEST_ONLY == 0 ]]; then
        submit_mode=(--parsable)
    fi

    local output
    output=$(sbatch "${submit_mode[@]}" \
        --account=coreai_dlalgo_llm \
        --partition=36x2-a01r \
        --nodes="$nodes" \
        --exclusive \
        --segment="$segment" \
        --time="$TIME_LIMIT" \
        --comment=metrics \
        "${dependency_args[@]}" \
        --job-name="coreai_dlalgo_llm-nemorl.force-ratio-${run_key}-${MAX_STEPS}s" \
        --export="ALL,CONFIG_NAME=${config_name},RUN_KEY=${run_key},MODEL=${model},FORCE_ON_POLICY_RATIO=${force_value},MAX_STEPS=${MAX_STEPS},EXPECTED_RUNNER_SHA=${runner_sha}" \
        "$RUNNER" 2>&1)
    printf '%s\t%s\n' "$run_key" "$output"

    if [[ $TEST_ONLY == 0 ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$run_key" "$output" "$model" "$config_name" "$force_value" \
            "$nodes" "$segment" "$MAX_STEPS" "$EXPECTED_REPO_SHA" "$runner_sha" \
            "$DEPENDENCY_LABEL" >> "$JOBS_TSV"
    fi
}

while IFS=$'\t' read -r run_key model config_name force_value nodes segment global_batch_size steps; do
    if [[ $run_key == run_key ]] || [[ ! $run_key =~ $RUN_FILTER ]]; then
        continue
    fi
    test "$global_batch_size" = 2048
    test "$steps" = 20
    submit_case "$run_key" "$model" "$config_name" "$force_value" "$nodes" "$segment"
done < "$CONTRACT"
