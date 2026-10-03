#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:05:00

set -euo pipefail

srun --nodes=1 --ntasks=1 --cpus-per-task=144 --exact \
  --mem=0 bash -c '
set -euo pipefail
path=$(awk -F: '\''$1 == "0" { print $3 }'\'' /proc/self/cgroup)
limit=$(cat "/sys/fs/cgroup${path}/memory.max")
printf "node=%s\nSLURM_MEM_PER_NODE=%s\ncgroup_memory_max=%s\n" \
  "$(hostname)" "${SLURM_MEM_PER_NODE:-unset-for-mem-zero}" "$limit"
if [[ "$limit" != max ]] && (( limit <= 128 * 1024 * 1024 * 1024 )); then
  echo "The Ray-sized step is still restricted by a small memory default" >&2
  exit 1
fi
nvidia-smi --query-gpu=name --format=csv,noheader
'
