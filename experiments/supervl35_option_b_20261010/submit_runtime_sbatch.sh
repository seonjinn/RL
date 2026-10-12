#!/bin/bash
set -euo pipefail

durable=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ${1:-} == --run ]]; then
    shift
    [[ $# -eq 4 ]] || { echo "Expected source commit, checkout, run root, and image" >&2; exit 2; }
    source_commit=$1
    source_checkout=$2
    run_root=$3
    image=$4
    test "${SLURM_NNODES:?}" -eq 1
    test "$(git -C "$source_checkout" rev-parse HEAD)" = "$source_commit"

    mounts="$HOME:$HOME:ro,$run_root:$run_root:rw,/lustre:/lustre:ro,$durable:$durable:rw,/raid/scratch:/raid/scratch:rw"
    srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 \
        --gpus-per-node=4 --cpus-per-task=144 \
        --no-container-mount-home --container-image="$image" \
        --container-mounts="$mounts" --container-workdir="$run_root" \
        /usr/bin/env SOURCE_CHECKOUT="$source_checkout" \
        /bin/bash "$launcher_dir/prepare_runtime.sh" "$source_commit"
    exit
fi

[[ $# -eq 0 ]] || { echo "Usage: submit_runtime_sbatch.sh" >&2; exit 2; }
repo=$(git -C "$launcher_dir" rev-parse --show-toplevel)
[[ -z $(git -C "$repo" status --porcelain) ]] || {
    echo "Commit all source and launcher changes before submitting" >&2
    exit 1
}
previous_commit=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" pull --ff-only
source_commit=$(git -C "$repo" rev-parse HEAD)
if [[ "$source_commit" != "$previous_commit" ]]; then
    exec /bin/bash "$launcher_dir/submit_runtime_sbatch.sh"
fi
git -C "$repo" merge-base --is-ancestor af93c00e9f8daff2f446d2b00882b0a9b4619da5 "$source_commit" || {
    echo "Source commit $source_commit does not include the bounded broadcast fix" >&2
    exit 1
}
git -C "$repo" submodule update --init --recursive -- \
    3rdparty/Gym-workspace/Gym \
    3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
[[ -f "$repo/3rdparty/Gym-workspace/Gym/pyproject.toml" ]] || {
    echo "Gym submodule is missing its pyproject.toml" >&2
    exit 1
}
[[ -f "$repo/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/__init__.py" ]] || {
    echo "Megatron-LM submodule is not initialized" >&2
    exit 1
}
[[ -z $(git -C "$repo" status --porcelain) ]] || {
    echo "Source checkout changed while preparing submodules" >&2
    exit 1
}

run_root=${NRL_RUN_ROOT:-$HOME/supervl35-optionb-control}
image=${NRL_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/images/vllm029-20261006/nemo_rl_main_aligned_20261007_7776525.sqsh}
account=${SLURM_ACCOUNT:-coreai_dlalgo_nemorl}
[[ -r "$image" ]] || { echo "Container image is not readable: $image" >&2; exit 1; }
[[ -d "$durable" ]] || { echo "Experiment directory is missing: $durable" >&2; exit 1; }
if compgen -G "$durable/runtime/supervl35-runtime-${source_commit:0:12}-ray*.tar.gz" >/dev/null || \
   compgen -G "$durable/runtime/supervl35-runtime-${source_commit:0:12}-ray*.tar.gz.sha256" >/dev/null; then
    echo "Runtime artifact already exists for $source_commit under $durable/runtime" >&2
    exit 1
fi
mkdir -p "$run_root"
run_root=$(cd "$run_root" && pwd)

args=(
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=144
    --gpus-per-node=4 --exclusive --mem=0
    --account="$account" --partition=batch --qos=normal --time=01:00:00
    --job-name="${account}.supervl35-runtime-${source_commit:0:12}"
    --output="$run_root/runtime-build-%j.out"
)
command=("$launcher_dir/submit_runtime_sbatch.sh" --run "$source_commit" "$repo" "$run_root" "$image")
sbatch --test-only "${args[@]}" "${command[@]}"
sbatch "${args[@]}" "${command[@]}"

echo "After completion, verify $durable/runtime/supervl35-runtime-${source_commit:0:12}-ray*.tar.gz.sha256"
echo "Then submit with NRL_EXPERIMENT_SOURCE_COMMIT=$source_commit"
