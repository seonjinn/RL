#!/bin/bash
set -euo pipefail

launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
durable_root=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010

if [[ ${1:-} == --node ]]; then
    [[ $# -eq 4 ]] || exit 2
    job_id=$2
    arm=$3
    head_ip=$4
    read -r node_ip _ <<< "$(hostname -I)"
    [[ "$node_ip" == "$head_ip" ]] || exit 0
    source_log="/raid/scratch/sna/supervl35-$job_id-$arm/node.log"
    driver_log="$durable_root/runs/$job_id/$arm/driver.log"
    bash "$launcher_dir/sync_driver_log.sh" "$source_log" "$driver_log" --once
    printf 'Captured %s bytes from rank 0 on %s to %s\n' \
        "$(wc -c <"$driver_log")" "$(hostname)" "$driver_log"
    exit 0
fi

[[ $# -eq 2 && "$1" =~ ^[1-9][0-9]*$ ]] || {
    echo "Usage: capture_running_driver_log.sh JOB_ID ARM" >&2
    exit 2
}
job_id=$1
arm=$2
case "$arm" in
    bf16-bf16|bf16-mxfp8|mxfp8-bf16params-off|mxfp8-bf16params-on|mxfp8-fp8params-off|mxfp8-fp8params-on) ;;
    *) echo "Unknown precision arm: $arm" >&2; exit 2 ;;
esac
run_root=${NRL_RUN_ROOT:-$HOME/supervl35-optionb-control}
head_file="$run_root/runs/$job_id/$arm/state/head-ip"
[[ -s "$head_file" ]] || { echo "Missing Ray head IP: $head_file" >&2; exit 1; }
head_ip=$(<"$head_file")

srun --jobid="$job_id" --overlap --nodes=32 --ntasks=32 \
    --ntasks-per-node=1 --cpus-per-task=1 \
    /bin/bash "$launcher_dir/capture_running_driver_log.sh" --node "$job_id" "$arm" "$head_ip"
