#!/usr/bin/env bash

set -euo pipefail

BASE=/lustre/fsw/coreai_dlalgo_llm/users/sna
REPO=$BASE/nemo-rl-main-pr3030-q235-20260701
EXP_ROOT=$BASE/pretyche_force_on_policy_ratio_llama_qwen_async_20260707
CONTRACT=$EXP_ROOT/manifests/config_contract.tsv
RUNNER=$EXP_ROOT/scripts/run_force_on_policy_benchmark.sbatch
CONTAINER=$BASE/containers/nemo_rl_nightly_20260630_0215.sqsh
JOBS_TSV=$EXP_ROOT/results/jobs.tsv
EXPECTED_REPO_SHA=d4cfecf90db41cdf142629963b54b67ab479ab02
EXPECTED_CONTAINER_SHA=bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510
TEST_ONLY=${TEST_ONLY:-1}
RUN_FILTER=${RUN_FILTER:-.*}

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_REPO_SHA"
test "$(git -C "$REPO" rev-parse '@{u}')" = "$EXPECTED_REPO_SHA"
test "$(git -C "$REPO" remote get-url origin)" = https://github.com/seonjinn/RL.git
test -z "$(git -C "$REPO" status --porcelain --ignore-submodules=untracked)"
! git -C "$REPO" submodule status --recursive | grep -Eq '^[-+U]'
test -x "$RUNNER"
test -s "$CONTRACT"
test -s "$CONTAINER"
test "$(sha256sum "$CONTAINER" | awk '{print $1}')" = "$EXPECTED_CONTAINER_SHA"
test -r "$HOME/.nemo_rl_tokens"

runner_sha=$(sha256sum "$RUNNER" | awk '{print $1}')

if [[ $TEST_ONLY == 0 ]]; then
    mkdir -p "$(dirname "$JOBS_TSV")"
    if [[ -s $JOBS_TSV ]] && [[ $(wc -l < "$JOBS_TSV") -gt 1 ]]; then
        printf 'Refusing duplicate submission: %s already contains jobs.\n' "$JOBS_TSV" >&2
        exit 2
    fi
    printf 'run_key\tjob_id\tmodel\tconfig_name\tmode\tforce_on_policy_ratio\tnodes\tgpus_per_node\tsegment\tglobal_batch_size\tsteps\ttime_limit\trepo_sha\tcontainer_sha\trunner_sha\n' > "$JOBS_TSV"
fi

submit_case() {
    local run_key=$1
    local model=$2
    local config_name=$3
    local mode=$4
    local force_value=$5
    local nodes=$6
    local gpus_per_node=$7
    local segment=$8
    local global_batch_size=$9
    local steps=${10}
    local time_limit=${11}
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
        --time="$time_limit" \
        --comment=metrics \
        --job-name="coreai_dlalgo_llm-nemorl.force-ratio-${run_key}-20s" \
        --export="ALL,CONFIG_NAME=${config_name},RUN_KEY=${run_key},MODEL=${model},MODE=${mode},FORCE_ON_POLICY_RATIO=${force_value},EXPECTED_NODES=${nodes},SEGMENT_SIZE=${segment},EXPECTED_RUNNER_SHA=${runner_sha}" \
        "$RUNNER" 2>&1)
    printf '%s\t%s\n' "$run_key" "$output"

    if [[ $TEST_ONLY == 0 ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$run_key" "$output" "$model" "$config_name" "$mode" "$force_value" \
            "$nodes" "$gpus_per_node" "$segment" "$global_batch_size" "$steps" \
            "$time_limit" "$EXPECTED_REPO_SHA" "$EXPECTED_CONTAINER_SHA" "$runner_sha" \
            >> "$JOBS_TSV"
    fi
}

while IFS=$'\t' read -r run_key model config_name mode force_value nodes gpus_per_node segment global_batch_size steps time_limit; do
    if [[ $run_key == run_key ]] || [[ ! $run_key =~ $RUN_FILTER ]]; then
        continue
    fi
    test "$gpus_per_node" = 4
    test "$nodes" = "$segment"
    test "$global_batch_size" = 2048
    test "$steps" = 20
    [[ $time_limit == 02:00:00 || $time_limit == 03:00:00 || $time_limit == 04:00:00 ]]
    [[ $config_name != *8g* ]]
    submit_case "$run_key" "$model" "$config_name" "$mode" "$force_value" \
        "$nodes" "$gpus_per_node" "$segment" "$global_batch_size" "$steps" "$time_limit"
done < "$CONTRACT"
