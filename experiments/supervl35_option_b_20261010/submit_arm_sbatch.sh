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

launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$launcher_dir" rev-parse --show-toplevel)
[[ -z $(git -C "$repo" status --porcelain) ]] || {
    echo "Commit all source and launcher changes before submitting" >&2
    exit 1
}
previous_commit=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" pull --ff-only
launcher_commit=$(git -C "$repo" rev-parse HEAD)
if [[ "$launcher_commit" != "$previous_commit" ]]; then
    exec /bin/bash "$launcher_dir/submit_arm_sbatch.sh" "$arm"
fi
source_commit=${NRL_EXPERIMENT_SOURCE_COMMIT:-c8734274ac5883cab9c942b88a37820805a6aeae}
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
    --job-name="$name" --output="$run_root/slurm-%j.out"
)
command=("$launcher_dir/submit_arm_sbatch.sh" --run "$arm" "$source_commit" "$launcher_dir" "$run_root" "$image")
sbatch --test-only "${args[@]}" "${command[@]}"
sbatch "${args[@]}" "${command[@]}"
