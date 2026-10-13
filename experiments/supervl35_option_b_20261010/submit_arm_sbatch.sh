#!/bin/bash
set -euo pipefail

durable=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010

if [[ ${1:-} == --run ]]; then
    shift
    [[ $# -eq 5 ]] || { echo "Expected arm, source commit, launcher, run root, and image" >&2; exit 2; }
    arm=$1
    source_commit=$2
    launcher_dir=$3
    run_root=$4
    image=$5
    test "${SLURM_NNODES:?}" -eq 32
    export NRL_EXPERIMENT_CONFIG_DIR="$launcher_dir/configs"

    mounts="/lustre:/lustre:ro,$durable:$durable:rw,$launcher_dir:$launcher_dir:ro,$run_root:$run_root:rw,/raid/scratch:/raid/scratch:rw"
    if [[ -d /home/svc-dss/cache ]]; then
        mounts+=",/home/svc-dss/cache:/home/svc-dss/cache:ro"
    fi
    srun --kill-on-bad-exit=1 --nodes=32 --ntasks=32 --ntasks-per-node=1 \
        --gpus-per-node=4 --cpus-per-task=144 --mpi=pmix \
        --no-container-mount-home --container-image="$image" \
        --container-mounts="$mounts" --container-workdir="$run_root" \
        --output="$run_root/srun-%j-%t.log" \
        /bin/bash "$launcher_dir/run_arms.sh" "$source_commit" "$arm"
    exit
fi

arm=${1:?Usage: submit_arm_sbatch.sh ARM}
case "$arm" in
    bf16-bf16|bf16-mxfp8|mxfp8-bf16params-off|mxfp8-bf16params-on|mxfp8-fp8params-off|mxfp8-fp8params-on) ;;
    *) echo "Unknown precision arm: $arm" >&2; exit 2 ;;
esac
[[ -n ${WANDB_API_KEY:-} ]] || {
    echo "WANDB_API_KEY must be exported for W&B-enabled performance runs" >&2
    exit 1
}
export WANDB_API_KEY
export MM_TRAINER_WANDB_PROJECT=${MM_TRAINER_WANDB_PROJECT:-nemo-rl-supervl35-mxfp8}
afterok_job_id=${SLURM_AFTEROK_JOB_ID:-}
if [[ -n "$afterok_job_id" && ! "$afterok_job_id" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLURM_AFTEROK_JOB_ID must be a positive numeric job ID" >&2
    exit 2
fi

launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$launcher_dir" rev-parse --show-toplevel)
[[ -z $(git -C "$repo" status --porcelain) ]] || {
    echo "Commit all source and launcher changes before submitting" >&2
    exit 1
}
previous_commit=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" -c fetch.recurseSubmodules=false -c submodule.recurse=false \
    pull --ff-only --no-recurse-submodules
launcher_commit=$(git -C "$repo" rev-parse HEAD)
if [[ "$launcher_commit" != "$previous_commit" ]]; then
    exec /bin/bash "$launcher_dir/submit_arm_sbatch.sh" "$arm"
fi
patched_source=af93c00e9f8daff2f446d2b00882b0a9b4619da5
source_commit=${NRL_EXPERIMENT_SOURCE_COMMIT:-$patched_source}
git -C "$repo" cat-file -e "$source_commit^{commit}"
if ! git -C "$repo" merge-base --is-ancestor "$source_commit" "$launcher_commit" || \
   ! git -C "$repo" diff --quiet "$source_commit" "$launcher_commit" -- . ':(exclude)experiments/supervl35_option_b_20261010'; then
    echo "The prepared runtime does not match the experiment source" >&2
    exit 1
fi

run_root=${NRL_RUN_ROOT:-$HOME/supervl35-optionb-control}
image=${NRL_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/images/vllm029-20261006/nemo_rl_main_aligned_20261007_7776525.sqsh}
account=${SLURM_ACCOUNT:-coreai_dlalgo_nemorl}
test -r "$image"
test -d "$durable/runtime"
if ! compgen -G "$durable/runtime/supervl35-runtime-${source_commit:0:12}-ray*.tar.gz.sha256" >/dev/null; then
    echo "Missing prepared runtime for $source_commit under $durable/runtime" >&2
    exit 1
fi
mkdir -p "$run_root"
run_root=$(cd "$run_root" && pwd)
name="${account}.supervl35-${arm}-$(date -u +%Y%m%d%H%M%S)"

args=(
    --nodes=32 --ntasks-per-node=1 --cpus-per-task=144
    --gpus-per-node=4 --segment=8 --exclusive --mem=0
    --account="$account" --partition=batch --qos=normal --time=04:00:00
    --export=ALL
    --job-name="$name" --output="$run_root/slurm-%j.out"
)
if [[ -n "$afterok_job_id" ]]; then
    args+=("--dependency=afterok:$afterok_job_id")
fi
command=("$launcher_dir/submit_arm_sbatch.sh" --run "$arm" "$source_commit" "$launcher_dir" "$run_root" "$image")
sbatch --test-only "${args[@]}" "${command[@]}"
if [[ ${NRL_SBATCH_PARSABLE:-0} == 1 ]]; then
    sbatch --parsable "${args[@]}" "${command[@]}"
else
    sbatch "${args[@]}" "${command[@]}"
fi
