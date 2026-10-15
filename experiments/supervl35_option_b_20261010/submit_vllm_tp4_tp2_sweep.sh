#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 {bf16|mxfp8|mxfp8_tp4|moe_all|all_decoder} {1024|10240} {1024|10240} {1|2|4|16|32|64|128|256|512|1024}" >&2
    exit 2
}

[[ $# == 4 ]] || usage
arm=$1
isl=$2
osl=$3
requested_concurrency=$4
case "$arm" in
    bf16|mxfp8|mxfp8_tp4|moe_all|all_decoder) ;;
    *) usage ;;
esac
case "$isl/$osl" in
    1024/1024|1024/10240|10240/1024) ;;
    *) usage ;;
esac
case "$requested_concurrency" in
    1|2|4|16|32|64|128|256|512|1024) ;;
    *) usage ;;
esac

concurrency=$requested_concurrency
if [[ "$concurrency" == 1 ]]; then
    concurrency=2
    echo "Requested C=1 is measured as C=2 total (one request per DP rank)." >&2
fi
multiplier=${BENCH_REQUEST_MULTIPLIER:-10}
[[ "$multiplier" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid request multiplier" >&2; exit 2; }
trials=${BENCH_TRIALS:-1}
high_concurrency_trials=${BENCH_HIGH_CONCURRENCY_TRIALS:-$trials}
[[ "$trials" =~ ^[1-9][0-9]*$ && "$high_concurrency_trials" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid trial count" >&2
    exit 2
}
if [[ "$isl/$osl/$concurrency" == 1024/10240/1024 && "$multiplier" -gt 5 ]]; then
    echo "1K/10K c1024 needs two five-times-request jobs within the four-hour limit" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$script_dir" rev-parse --show-toplevel)
if [[ -n $(git -C "$repo" status --porcelain) ]]; then
    echo "Commit and push source changes before submitting" >&2
    exit 1
fi
git -C "$repo" pull --ff-only
bench_code_commit=91350afefd
git -C "$repo" diff --quiet "$bench_code_commit" HEAD -- \
    experiments/supervl35_option_b_20261010/bench_vllm_generation.py \
    experiments/supervl35_option_b_20261010/bench_vllm_dp.py \
    experiments/supervl35_option_b_20261010/run_vllm_ab.sh \
    experiments/supervl35_option_b_20261010/configs/bf16.yaml \
    experiments/supervl35_option_b_20261010/configs/moe.yaml || {
        echo "Staged benchmark differs from the local source; restage before submitting" >&2
        exit 1
    }
git -C "$repo" merge-base --is-ancestor HEAD '@{upstream}' || {
    echo "Push HEAD before submitting" >&2
    exit 1
}

brr=${MARS_BRR:-$HOME/.local/share/mars-cluster-access/bin/brr}
test -x "$brr"
workspace=supervl35-optionb-20261010
stage=tp2-scope-v1-20261014
root=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010
image=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/images/vllm029-20261006/nemo_rl_main_aligned_20261007_7776525.sqsh
account=${SLURM_ACCOUNT:-nemotron_sw_post}
target_max_num_seqs=$concurrency
if [[ "$arm" == bf16 || "$arm" == mxfp8_tp4 ]]; then
    kind=length_sweep
    if [[ "$arm" == bf16 ]]; then
        bench_arm=bf16
    else
        bench_arm=moe
    fi
    cap_arg="BENCH_MAX_NUM_SEQS=$target_max_num_seqs"
else
    kind=dp2_length_sweep
    if [[ "$arm" == mxfp8 ]]; then
        bench_arm=moe
    else
        bench_arm=$arm
    fi
    cap_arg="BENCH_MAX_NUM_SEQS_PER_RANK=$((target_max_num_seqs / 2))"
fi

name="sna-svl35-${arm}-${isl}-${osl}-c${concurrency}-$(date -u +%m%d%H%M%S)-$$"
payload="set -euo pipefail; vllm_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python; bash $stage/apply_vllm_029_modular_activation.sh \"\$vllm_python\"; env BENCHMARK_KIND=$kind BENCH_ARMS=$bench_arm BENCH_ISL=$isl BENCH_OSL=$osl BENCH_CONCURRENCIES=$concurrency BENCH_REQUEST_MULTIPLIER=$multiplier BENCH_TRIALS=$trials BENCH_HIGH_CONCURRENCY_TRIALS=$high_concurrency_trials BENCH_ENABLE_PREFIX_CACHING=1 $cap_arg BENCH_CODE_COMMIT=$bench_code_commit NRL_BENCH_IMAGE=$image bash $stage/run_vllm_ab.sh"
command=("$brr" job submit oci-hsg
    --workspace "$workspace" --image "$image"
    --account "$account" --partition batch --qos normal
    --nodes 1 --gpus-per-node 4 --ntasks-per-node 1
    --time-limit 04:00:00 --job-name "$name" --container-name "$name"
    --mounts /lustre:/lustre:ro --mounts "$root:$root:rw"
    --mounts /raid/scratch:/raid/scratch:rw
    -- /bin/bash -lc "$payload")

printf 'ISL=%s OSL=%s offered_concurrency=%s target_max_num_seqs=%s requests_per_trial=%s trials=%s arm=%s\n' \
    "$isl" "$osl" "$concurrency" "$target_max_num_seqs" "$((concurrency * multiplier))" "$trials" "$arm"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
BRR_CLI_MINT_TIMEOUT=20 BRR_CLI_MINT_DEADLINE=45 BRR_JOB_TIMEOUT=120 "${command[@]}"
