#!/usr/bin/env bash

set -euo pipefail

runtime_root=${1:?node-local runtime output directory}
durable_root=${2:?durable attempt directory}
ray_tmpdir=${3:?node-local Ray log directory}
job_id=${4:?SLURM job id}
process_exit=${5:?process exit code}

mkdir -p "${durable_root}"
for artifact in overrides.txt recipe.txt identity.txt process-completed.txt; do
    if [[ -f "${runtime_root}/${artifact}" ]]; then
        cp "${runtime_root}/${artifact}" "${durable_root}/${artifact}"
    fi
done

if [[ -s "${runtime_root}/train.log" ]]; then
    tail -c 16777216 "${runtime_root}/train.log" \
        | gzip -c > "${durable_root}/train-tail.log.gz.tmp"
    mv "${durable_root}/train-tail.log.gz.tmp" \
        "${durable_root}/train-tail.log.gz"
fi

if [[ "${process_exit}" != 0 && -d "${ray_tmpdir}" ]]; then
    tar -czf "${durable_root}/ray-logs-failure-${job_id}.tar.gz.tmp" \
        -C "${ray_tmpdir}" .
    mv "${durable_root}/ray-logs-failure-${job_id}.tar.gz.tmp" \
        "${durable_root}/ray-logs-failure-${job_id}.tar.gz"
fi

printf '%s\n' "${process_exit}" > "${durable_root}/runtime-exit.txt.tmp"
mv "${durable_root}/runtime-exit.txt.tmp" "${durable_root}/runtime-exit.txt"
