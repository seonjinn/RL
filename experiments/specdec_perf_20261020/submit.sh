#!/bin/bash
# Submit one GRPO arm on OCI-HSG with ray.sub (run from anywhere inside the repo).
#
#   submit.sh <config.yaml> <run_name> [max_steps] [extra overrides...]
#
# Env knobs:
#   NUM_ACTOR_NODES      nodes to allocate (default 4, the 4n4g recipe size)
#   FORCE_REBUILD_VENVS  default true. The nightly image sets NEMO_RL_VENV_DIR=/opt/ray_venvs,
#                        so venvs are per-job and container-local. Its prebuilt venvs pin the
#                        image's own Megatron-Bridge and vLLM, and that Bridge imports
#                        te_checkpoint, which this checkout's Megatron-LM lacks.
#   CONTAINER            sqsh image (default: latest staged nightly symlink)
#   TIME                 walltime (default 04:00:00)
#   TEST_ONLY=1          sbatch --test-only
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

CONFIG=$1
RUN_NAME=$2
STEPS=${3:-20}
shift $(( $# < 3 ? $# : 3 ))

U=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna
EXP_DIR=experiments/specdec_perf_20261020
account=coreai_dlalgo_nemorl
NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-4}
# ray.sub defaults to 8 and rejects GB200's gpu:4 GRES unless this is exported
export GPUS_PER_NODE=4

REBUILD=""
if [[ "${FORCE_REBUILD_VENVS:-true}" == true ]]; then
  REBUILD="NRL_FORCE_REBUILD_VENVS=true "
fi

# grpo_math_1B.yaml's draft block keeps Eagle3-only keys that the Megatron DFlash/DSpark
# draft configs (extra="forbid") reject at MasterConfig validation, so online arms drop them
DRAFT_DELETES=""
if grep -q draft_online_ "${CONFIG}"; then
  for k in token_chunk_size learning_rate ttt_steps ttt_step_loss_decay train_embed_and_head; do
    DRAFT_DELETES+="'~policy.draft.${k}' "
  done
fi

# checkpointing stays off: workspace runs must not grow lustre inode/space usage
COMMAND="${REBUILD}uv run examples/run_grpo.py --config ${CONFIG} grpo.max_num_steps=${STEPS} checkpointing.enabled=false logger.log_dir=${EXP_DIR}/exp_logs/${RUN_NAME} logger.wandb_enabled=True logger.wandb.project='sna-async-grpo-gb200' logger.wandb.name='${RUN_NAME}' ${DRAFT_DELETES}$*"

mkdir -p "${EXP_DIR}/logs"
export CONTAINER=${CONTAINER:-$U/containers/nemo_rl_nightly.sqsh}
export HF_HOME=$U/hf_home
export HF_DATASETS_CACHE=$U/hf_home/cache
# The login-shell HF_TOKEN is revoked and 401s even on public repos; model and data are cached
unset HF_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_API_KEY
export MOUNTS="/lustre:/lustre"
export COMMAND
export BASE_LOG_DIR=${EXP_DIR}/logs/${RUN_NAME}
mkdir -p "${BASE_LOG_DIR}"

echo "commit=$(git rev-parse HEAD) container=$(readlink -f "$CONTAINER")"
echo "COMMAND=${COMMAND}"
sbatch ${TEST_ONLY:+--test-only} \
  --nodes="${NUM_ACTOR_NODES}" \
  --account="${account}" \
  --job-name="${account}.${RUN_NAME}" \
  --partition=batch \
  --time="${TIME:-04:00:00}" \
  --gres=gpu:${GPUS_PER_NODE} \
  --segment "${NUM_ACTOR_NODES}" \
  ray.sub