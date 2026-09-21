#!/usr/bin/env bash

set -euo pipefail

expected_head=${1:?expected commit}
bundle=${2:?immutable source bundle}
bundle_sha=${3:?bundle sha256}
result_parent=${4:?fresh Lustre result parent}
stage=${5:?one-based segment stage}
: "${SLURM_ARRAY_TASK_ID:?submit this script as a SLURM array}"

repo_root=$(git rev-parse --show-toplevel)
cd "${repo_root}"
condition_label=
method=
arm=
seqs=
previous_step=
stop_step=
eval "$(python3 -m research.qwen3_8b_rp25_swa.segment_matrix \
    --task-index "${SLURM_ARRAY_TASK_ID}" --stage "${stage}" --shell)"
[[ -n "${condition_label}" && -n "${method}" && -n "${arm}" && -n "${seqs}" \
    && -n "${previous_step}" && -n "${stop_step}" ]] || exit 64

result_root=${result_parent}/${condition_label}
phase=online-packed-300-${seqs}
exec bash research/qwen3_8b_rp25_swa/run_online_canary.sbatch \
    "${expected_head}" "${bundle}" "${bundle_sha}" "${result_root}" \
    "${method}" "${phase}" "${arm}" "${stop_step}" "${previous_step}"
