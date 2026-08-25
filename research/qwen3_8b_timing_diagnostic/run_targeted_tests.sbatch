#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ARCHIVE:?SOURCE_ARCHIVE must name the immutable source archive}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

scratch_root="/raid/scratch/q8-timing-test-${SLURM_JOB_ID}"
source_root="${scratch_root}/source"
mkdir -p "${source_root}" "${scratch_root}/uv-cache" "${scratch_root}/venv"
tar -xf "${SOURCE_ARCHIVE}" -C "${source_root}"

cd "${source_root}"
export UV_CACHE_DIR="${scratch_root}/uv-cache"
export UV_PROJECT_ENVIRONMENT="${scratch_root}/venv"

uv run --frozen pytest \
  tests/unit/algorithms/test_grpo.py::test_refit_policy_generation_records_full_weight_sync_time \
  tests/unit/algorithms/test_grpo_sync_draft_schedule.py::test_scheduled_refit_records_total_sync_and_transaction_timings \
  -q
