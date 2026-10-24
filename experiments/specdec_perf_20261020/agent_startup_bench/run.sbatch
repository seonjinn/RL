#!/bin/bash
#SBATCH --job-name=agent-startup-bench
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out

# sbatch --export=ALL,SETUP=<openhands setup dir>,SIF=<agent sif> run.sbatch
# SETUP must be the real (symlink-resolved) path: the venv records it as its interpreter home.
set -uo pipefail
: "${SETUP:?OpenHands setup dir with OpenHands/ and miniforge3/}"
: "${SIF:?agent container image}"
HERE=$SLURM_SUBMIT_DIR
OUT=$HERE/results/run_$(date +%Y%m%d_%H%M%S)_$SLURM_JOB_ID
LOCAL=/dev/shm/agent_startup_bench_$SLURM_JOB_ID
mkdir -p "$OUT" "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
export APPTAINER_CACHEDIR=$LOCAL/apptainer_cache
echo "setup=$SETUP sif=$SIF node=$(hostname) cpus=$(nproc)" | tee "$OUT/meta.txt"

run_batch() {
    local variant=$1 src=$2 n=$3 rep=$4 i
    local mounts="--mount type=bind,src=$src/OpenHands,dst=/openhands_setup/OpenHands,ro
        --mount type=bind,src=$src/OpenHands,dst=$SETUP/OpenHands,ro
        --mount type=bind,src=$src/miniforge3,dst=/openhands_setup/miniforge3,ro
        --mount type=bind,src=$src/miniforge3,dst=$SETUP/miniforge3,ro
        --mount type=bind,src=$HERE,dst=/bench,ro"
    for i in $(seq "$n"); do
        (
            f=$OUT/$variant.n$n.r$rep.$i.csv
            t0=$(date +%s.%N)
            # shellcheck disable=SC2086
            apptainer exec --writable-tmpfs --cleanenv --pid --no-mount home,tmp,bind-paths $mounts "$SIF" \
                bash /bench/bench_one.sh >"$f" 2>"$f.err"
            echo "total,$t0,$(date +%s.%N),$?" >>"$f"
        ) &
    done
    wait
    echo "$(date +%T) $variant n=$n rep=$rep done"
}

# Repeated n=1 batches separate a cold Lustre client cache from a warm one.
for rep in 1 2 3; do run_batch lustre "$SETUP" 1 "$rep"; done
for n in 8 32 64; do run_batch lustre "$SETUP" "$n" 1; done

t0=$(date +%s.%N)
rsync -a --exclude=/OpenHands/logs --exclude=/OpenHands/.eval_sessions --exclude=/OpenHands/evaluation/oh \
    "$SETUP/OpenHands" "$SETUP/miniforge3" "$LOCAL/"
rc=$?
mkdir -p "$LOCAL/OpenHands/logs" "$LOCAL/OpenHands/.eval_sessions" "$LOCAL/OpenHands/evaluation/oh"
echo "copy,$t0,$(date +%s.%N),$rc" >"$OUT/copy.csv"
du -sh "$LOCAL" | tee -a "$OUT/meta.txt"

for rep in 1 2 3; do run_batch shm "$LOCAL" 1 "$rep"; done
for n in 8 32 64; do run_batch shm "$LOCAL" "$n" 1; done