#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

: "${REPO:?Set REPO to the pinned integration checkout}"
: "${EXPECTED_SOURCE_SHA:?Pin the full integration SHA}"
: "${CONTAINER:?Set CONTAINER to the GPU-validated immutable derived image}"
: "${RESULT_ROOT:?Set RESULT_ROOT to a fresh comparison result directory}"
export SRUN_BIN=${SRUN_BIN:-/cm/local/apps/slurm/25.11/bin/srun}
export SLURM_COMMAND_PATH=${SLURM_COMMAND_PATH:-/cm/local/apps/slurm/25.11/bin}
export PATH="${SLURM_COMMAND_PATH}:${PATH}"
export CLUSTER=oci PARTITION=batch PERFORMANCE_RECIPE=1
export PERFORMANCE_PROFILE=runtime-aligned
export QUANT_SCOPE=moe MOE_BACKEND=flashinfer_trtllm MOE_ROUTER_DTYPE=fp32
export NRL_FORCE_REBUILD_VENVS=false ACTOR_VENV_ROOT=/opt/ray_venvs
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export RUN_GROUP=${RUN_GROUP:-ray258-20261003}
export WANDB_HOME=${WANDB_HOME:-/home/${USER}}

cd "${REPO}"
git pull --ff-only
git submodule update --init --recursive
[[ "$(git rev-parse HEAD)" == "${EXPECTED_SOURCE_SHA}" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ -s "${CONTAINER}" && -s "${CONTAINER}.runtime.json" ]]
sha256sum -c "${CONTAINER}.sha256"

# Never carry small-workload or memory diagnostics into performance cells.
unset KV_CACHE_MEMORY_BYTES SUPER_GPU_MEMORY_UTILIZATION GPU_MEMORY_UTILIZATION
unset DIAGNOSTIC_NUM_PROMPTS_PER_STEP DIAGNOSTIC_NUM_GENERATIONS_PER_PROMPT
unset DIAGNOSTIC_TRAIN_GLOBAL_BATCH_SIZE DIAGNOSTIC_MAX_NEW_TOKENS
unset NRL_LOG_LEVEL REFIT_PREQUANTIZE_OVERRIDE REFIT_TRANSPORT_OVERRIDE
unset VLLM_ENFORCE_EAGER ASYNC_RECOMPUTE_KV_CACHE ASYNC_IN_FLIGHT_WEIGHT_UPDATES
unset NRL_REFIT_BUFFER_MEMORY_RATIO NRL_REFIT_NUM_STREAMS NRL_REFIT_VALIDATE_RECEIVE
unset NRL_REFIT_MXFP8_USE_WEIGHT_LOADER NRL_REFIT_RUNTIME_FINGERPRINT NRL_XFERDTENSOR_GOLDEN
unset VLLM_PADDING_SOURCE VLLM_PADDING_SHA MODEL_SNAPSHOT_OVERRIDE SOURCE_ARCHIVE_OVERRIDE

mkdir -p "${RESULT_ROOT}/submission"
preflight_log="${RESULT_ROOT}/submission/preflight-${SLURM_JOB_ID}.log"
bash experiments/mxfp8_vllm029_fullperf_20260930/run_preflight.sbatch \
  2>&1 | tee "${preflight_log}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_SOURCE_SHA}" ]]

sshare -n -P -u "${USER}" \
  -A nemotron_sw_post,nemotron_n4_post,coreai_dlalgo_llm,coreai_dlalgo_nemorl \
  -o Account,User,FairShare >"${RESULT_ROOT}/submission/fairshare-${SLURM_JOB_ID}.txt"
SLURM_ACCOUNT=$(python3 - "${RESULT_ROOT}/submission/fairshare-${SLURM_JOB_ID}.txt" <<'PY'
import csv
import os
from pathlib import Path
import sys

rows = csv.reader(Path(sys.argv[1]).open(), delimiter="|")
accounts = [(float(row[2]), row[0].strip()) for row in rows
            if len(row) == 3 and row[1] == os.environ["USER"] and row[2]]
if not accounts:
    raise SystemExit("No usable account FairShare data")
print(max(accounts)[1])
PY
)
export SLURM_ACCOUNT
printf 'source=%s\nimage=%s\naccount=%s\n' \
  "${EXPECTED_SOURCE_SHA}" "${CONTAINER}" "${SLURM_ACCOUNT}"

python3 experiments/mxfp8_vllm029_fullperf_20260930/submit_matrix.py \
  --expected-sha "${EXPECTED_SOURCE_SHA}" \
  --preflight-log "${preflight_log}" \
  --ledger "${RESULT_ROOT}/submission/matrix-${SLURM_JOB_ID}.json"
