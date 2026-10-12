#!/bin/bash
set -euo pipefail
umask 022

test -f ./run_vllm_ab.sh
root=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010/vllm_profile
destination=$PWD/results

for spec in \
    7902274:isl10240_osl1024:c16:bf16 \
    7902295:isl10240_osl1024:c16:moe \
    7902375:isl1024_osl10240:c16:bf16 \
    7902376:isl1024_osl10240:c16:moe \
    7902432:isl10240_osl1024:c128:bf16 \
    7902433:isl10240_osl1024:c128:moe \
    7902434:isl1024_osl10240:c128:bf16 \
    7902437:isl1024_osl10240:c128:moe
do
    IFS=: read -r job_id lengths concurrency arm <<< "$spec"
    source_dir=$root/$lengths/$concurrency/$job_id
    test "$(<"$source_dir/exit-code")" = 0
    test -s "$source_dir/$arm.json"
    test -s "$source_dir/trace.tar"
    sha256sum -c "$source_dir/trace.tar.sha256"
    tar -tf "$source_dir/trace.tar" |
        awk '/\.pt\.trace\.json\.gz$/{count++} END{exit(count == 4 ? 0 : 1)}'

    install -d -m 0755 "$destination/$job_id"
    install -m 0644 \
        "$source_dir/$arm.json" \
        "$source_dir/trace.tar" \
        "$source_dir/trace.tar.sha256" \
        "$source_dir/provenance.txt" \
        "$source_dir/exit-code" \
        "$source_dir/driver.log" \
        "$destination/$job_id/"
    printf 'Collected %s %s %s\n' "$job_id" "$arm" "$lengths/$concurrency"
done
