#!/usr/bin/env bash
# Ported from Nemo-RL-nano-v3-omni/scripts/nanov3_mpo.sh so super-vllm20 can be
# submitted with the same launcher contract as nano:
#   - reads ${NEMORL}/.env if present (SBATCH_ACCOUNT, CONTAINER, MOUNTS,
#     MPO_MODEL_NAME, MPO_DATA_PATH, etc.)
#   - builds $COMMAND that invokes examples/run_vlm_mpo.py against
#     examples/omni/nanov3_mpo.yaml
#   - sbatches ray.sub
#
# RUNNER (uv idiom for prebuilt vs fresh-checkout containers)
#   The default RUNNER is `uv run --no-sync`, matching the idiom used elsewhere
#   in this repo (run_interactive_step_3_nanov3_vision_rl.sh) and in
#   Megatron-LM / Megatron-Bridge functional tests. Rationale:
#     - The super-omni-rl-vllm-v20 prebuilt sqsh container activates a baked
#       /opt/nemo_rl_venv (set as $VIRTUAL_ENV) that already matches the repo
#       lock plus a hand-maintained custom vLLM 0.20.1 fork overlay.
#     - Plain `uv run` would re-trigger `uv sync` and silently undo those
#       overlays, surfacing as either an ImportError on vllm or a wandb auth
#       failure -- the typical "didn't manage to make it work on my end"
#       symptom from reviewers on this MR.
#     - `--no-sync` tells uv "use $VIRTUAL_ENV as-is, do not touch deps", which
#       reuses the baked venv intact while keeping the launcher uniform with
#       the rest of the repo.
#
#   If you are NOT using the prebuilt container (fresh checkout where uv
#   should build the venv from scratch), override RUNNER:
#       RUNNER='uv run --extra mcore --extra vllm' bash scripts/vlm_mpo.sh
#
# The custom vLLM build (tools/build-custom-vllm.sh) is only needed if you are
# starting from a blank repo; the .env container already ships vLLM 0.20.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -f "${NEMORL}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${NEMORL}/.env"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-examples/omni/nanov3_mpo.yaml}"
NUM_NODES="${NUM_NODES:-4}"
# Use a fresh default run name each launch so checkpoints/logs do not resume prior runs.
JOB_NAME_BASE="${JOB_NAME_BASE:-mpo}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%3N)}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_BASE}-${RUN_ID}}"
# wandb run name / project decoupled from JOB_NAME so you can rename the
# wandb run without changing the SLURM job name or the checkpoint dir.
# Default name is fixed (nanov3-mpo-test) so every launch shows up under
# the same display name in the wandb UI; wandb's own run-id keeps each
# launch unique. Override with `WANDB_NAME=...` if you want a one-off name.
WANDB_NAME="${WANDB_NAME:-nanov3-mpo-test}"
WANDB_PROJECT="${WANDB_PROJECT:-vlm-mpo-dev-mmpr}"
MODEL_NAME="${MPO_MODEL_NAME:-${MODEL_NAME:-}}"
DATA_PATH="${MPO_DATA_PATH:-${DATA_PATH:-}}"
: "${MODEL_NAME:?Set MPO_MODEL_NAME or MODEL_NAME, or define it in ${NEMORL}/.env}"
: "${DATA_PATH:?Set MPO_DATA_PATH or DATA_PATH, or define it in ${NEMORL}/.env}"
RESULTS_ROOT="${RESULTS_ROOT:-${NEMORL}/results}"
RESULTS_DIR="${RESULTS_ROOT}/${JOB_NAME}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT or define it in ${NEMORL}/.env}"
SBATCH_PARTITION="${SBATCH_PARTITION:-${PARTITION:-batch}}"
SBATCH_TIME="${SBATCH_TIME:-4:00:00}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

export CONTAINER="${CONTAINER:?Set CONTAINER or define it in ${NEMORL}/.env}"
export MOUNTS="${MOUNTS:-/lustre:/lustre}"
# wandb auth: bind-mount $HOME/.netrc into the container if it exists on the
# host. Without this, wandb.init() crashes inside the container with
# "No API key configured. Use `wandb login` to log in." even though the user
# has already run `wandb login` on the submission host. Mirrors the same
# bind-mount used in MPO/sbatch.sh. If you prefer to inject the key directly,
# export WANDB_API_KEY before invoking this script and ray.sub will forward it.
if [[ -f "${HOME}/.netrc" ]] && [[ "${MOUNTS}" != *"/root/.netrc"* ]]; then
  export MOUNTS="${MOUNTS},${HOME}/.netrc:/root/.netrc:ro"
  echo "[scripts/vlm_mpo.sh] auto-mounting ${HOME}/.netrc -> /root/.netrc (for wandb)"
fi
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export CACHE_ROOT="${CACHE_ROOT:-${NEMORL}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${RUN_ID}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
export NEMO_RL_TRAIN_STEP_MEM_DIAG="${NEMO_RL_TRAIN_STEP_MEM_DIAG:-1}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-0}"

if [[ ! -f "${NEMORL}/ray.sub" ]]; then
  echo "ray.sub not found under NEMORL=${NEMORL}" >&2
  exit 1
fi

if [[ "${CONFIG_PATH}" = /* ]]; then
  CONFIG_ABS_PATH="${CONFIG_PATH}"
else
  CONFIG_ABS_PATH="${NEMORL}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_ABS_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# Default to `uv run --no-sync` so the prebuilt container's baked $VIRTUAL_ENV
# is reused as-is without re-triggering uv sync; see "RUNNER" note at top of
# this file for why.
RUNNER="${RUNNER:-uv run --no-sync}"

# PYTHONPATH override: force the on-lustre working tree to take precedence over
# the baked /opt/nemo_rl_venv/lib/python*/site-packages/nemo_rl. Without this,
# `import nemo_rl.algorithms.mpo` resolves to the upstream copy baked into the
# container image (which was built before this PR existed and therefore lacks
# the mpo subpackage). uv pip install -e is not stable here -- every fresh
# container attach loses any prior editable install, so PYTHONPATH is the only
# reliable mechanism.
#
# Optional wandb version override: opt-in only. Set WANDB_PIN_VERSION=<ver> to
# pin wandb inside the container before the driver runs (e.g. to roll back a
# breaking wandb release). Empty (default) skips this entirely.
if [[ -n "${WANDB_PIN_VERSION:-}" ]]; then
  WANDB_PIN_SNIPPET="(python -c \"import wandb, sys; sys.exit(0 if wandb.__version__ == '${WANDB_PIN_VERSION}' else 1)\" 2>/dev/null || pip install --quiet --no-input --no-deps 'wandb==${WANDB_PIN_VERSION}') && "
else
  WANDB_PIN_SNIPPET=""
fi
export COMMAND="\
mkdir -p '${HF_HOME}' '${HF_MODULES_CACHE}' '${NRL_MEGATRON_CHECKPOINT_DIR}' '${TRITON_CACHE_DIR}' '${TMPDIR}' '${RESULTS_DIR}' && \
export PYTHONPATH=\"${NEMORL}:\${PYTHONPATH:-}\" && \
${WANDB_PIN_SNIPPET}${RUNNER} examples/run_vlm_mpo.py --config '${CONFIG_PATH}' \
cluster.num_nodes=$NUM_NODES \
policy.model_name='${MODEL_NAME}' \
checkpointing.checkpoint_dir='${RESULTS_DIR}' \
data.data_path='${DATA_PATH}' \
logger.wandb.name='${WANDB_NAME}' \
logger.wandb.project='${WANDB_PROJECT}'"

cd "${NEMORL}"

sbatch \
    --nodes=${NUM_NODES} \
    --account=${SBATCH_ACCOUNT} \
    --job-name=${JOB_NAME} \
    --partition=${SBATCH_PARTITION} \
    --time=${SBATCH_TIME} \
    --gres=gpu:${GPUS_PER_NODE} \
    ray.sub
