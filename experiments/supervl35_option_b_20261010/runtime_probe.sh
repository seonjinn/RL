#!/bin/bash
set -euo pipefail
umask 077
archive=$1
output=$2
sample=$3
mkdir -p "$output"
output=$(realpath "$output")
exec >"$output/probe.log" 2>&1
local_root="/raid/scratch/sna/supervl35-probe-${SLURM_JOB_ID}"
mkdir -p "$local_root/source" "$local_root/tmp" "$local_root/pycache"
export TMPDIR="$local_root/tmp" PYTHONPYCACHEPREFIX="$local_root/pycache"
export UV_CACHE_DIR="$local_root/uv" TRITON_CACHE_DIR="$local_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor"
tar -xf "$archive" -C "$local_root/source"
export NRL_EXPERIMENT_SOURCE="$local_root/source"
export PYTHONPATH="$NRL_EXPERIMENT_SOURCE:$NRL_EXPERIMENT_SOURCE/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$NRL_EXPERIMENT_SOURCE/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:$NRL_EXPERIMENT_SOURCE/3rdparty/Gym-workspace/Gym:$NRL_EXPERIMENT_SOURCE/examples/nemo_gym/supervl3p5"
date -u
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
for python in /opt/nemo_rl_venv/bin/python /opt/ray_venvs/*/bin/python; do
    if test -x "$python"; then
        "$python" "$NRL_EXPERIMENT_SOURCE/experiments/supervl35_option_b_20261010/runtime_probe.py" "$output" "$sample"
    fi
done
printf '%s\n' PROBE_COMPLETE
