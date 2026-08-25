#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ARCHIVE:?SOURCE_ARCHIVE must name the immutable source archive}"
: "${CONTAINER:?CONTAINER must name the immutable container image}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

scratch_root="/raid/scratch/q8-timing-test-${SLURM_JOB_ID}"
source_root="${scratch_root}/source"
mkdir -p "${source_root}" "${scratch_root}/uv-cache" "${scratch_root}/venv"
tar -xf "${SOURCE_ARCHIVE}" -C "${source_root}"

quoted_source=$(printf '%q' "${source_root}")
quoted_cache=$(printf '%q' "${scratch_root}/uv-cache")
quoted_venv=$(printf '%q' "${scratch_root}/venv")
command="cd ${quoted_source} && UV_CACHE_DIR=${quoted_cache} UV_PROJECT_ENVIRONMENT=${quoted_venv} uv run --frozen pytest tests/unit/algorithms/test_grpo.py::test_refit_policy_generation_records_full_weight_sync_time tests/unit/algorithms/test_grpo_sync_draft_schedule.py::test_scheduled_refit_records_total_sync_and_transaction_timings -q"

exec srun \
  --container-image="${CONTAINER}" \
  --container-mounts="/lustre:/lustre,${source_root}:${source_root},${scratch_root}:${scratch_root}" \
  --no-container-mount-home \
  bash -lc "${command}"
