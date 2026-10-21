#!/bin/bash
# Submit one async SWE GRPO arm on OCI-HSG with ray.sub (run from anywhere inside the repo).
# Port of ruit's run_grpo_qwen3_30b_swe_scale_gen.sh to latest main.
#
#   submit_swe.sh <config.yaml> <run_name> [max_steps] [extra overrides...]
#
# Env knobs:
#   NUM_VLLM_REPLICAS  vLLM TP2 replicas, a multiple of 4 (default 4). Derives the rest as in
#                      ruit's launcher: R/2 generation nodes + 1 training node, R/4 prompts per
#                      step x 8 generations, GBS = 2R, agent concurrency = max(768, GBS)
#   CONTAINER, TIME, TEST_ONLY  as in submit.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# The Gym mount and the worker venvs come from the submodule checkouts, whose pins move with
# the branch
git submodule update --init --recursive

CONFIG=$1
RUN_NAME=$2
STEPS=${3:-20}
shift $(( $# < 3 ? $# : 3 ))

U=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna
EXP_DIR=experiments/specdec_perf_20261020
account=coreai_dlalgo_nemorl
export GPUS_PER_NODE=4
# ray.sub sizes each worker step's memory by its CPU share; 140 of 144 CPUs asks for more
# than a GB200 node has free and intermittently fails step creation
export CPUS_PER_WORKER=96

R=${NUM_VLLM_REPLICAS:-4}
if (( R < 4 || R % 4 != 0 )); then
  echo "NUM_VLLM_REPLICAS must be a positive multiple of 4 (got ${R})" >&2
  exit 1
fi
GEN_NODES=$(( R * 2 / GPUS_PER_NODE ))
NUM_NODES=$(( GEN_NODES + 1 ))
GPP=8
PPS=$(( R / 4 ))
GBS=$(( PPS * GPP ))
CONCURRENCY=$(( GBS > 768 ? GBS : 768 ))

SWE_OVERRIDES="cluster.num_nodes=${NUM_NODES} policy.generation.colocated.resources.num_nodes=${GEN_NODES} policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} grpo.num_prompts_per_step=${PPS} grpo.num_generations_per_prompt=${GPP} policy.train_global_batch_size=${GBS}"
for split in train val; do
  SWE_OVERRIDES+=" env.nemo_gym.swe_agents_${split}.responses_api_agents.swe_agents.concurrency=${CONCURRENCY}"
done

# Gym server venvs go on node-local disk; building them on lustre hangs. Parallel editable
# builds can exceed uv's default 300 s lock timeout. Worker venvs are rebuilt for the same
# reason as in submit.sh: the image's prebuilt ones do not match this checkout.
GYM_ENV="NRL_FORCE_REBUILD_VENVS=true NEMO_GYM_VENV_DIR=/opt/gym_venvs UV_LOCK_TIMEOUT=3600 UV_HTTP_TIMEOUT=3600"

# The first OpenHands setup on lustre (miniforge + poetry install) outlasts Gym's 600 s server
# spinup default; later runs reuse the cached tree.
SWE_OVERRIDES+=" ++env.nemo_gym.server_spinup_timeout_seconds=3600"

# checkpointing stays off: workspace runs must not grow lustre inode/space usage. The env flag
# skips the per-step train_data_step*.jsonl dump of full agent trajectories (see submit.sh).
COMMAND="${GYM_ENV} uv run examples/nemo_gym/run_grpo_nemo_gym.py --config ${CONFIG} grpo.max_num_steps=${STEPS} checkpointing.enabled=false ++env.should_log_nemo_gym_responses=true ${SWE_OVERRIDES} logger.log_dir=${EXP_DIR}/exp_logs/${RUN_NAME} logger.wandb_enabled=True logger.wandb.project='sna-async-grpo-gb200' logger.wandb.name='${RUN_NAME}' $*"

mkdir -p "${EXP_DIR}/logs"
export CONTAINER=${CONTAINER:-$U/containers/nemo_rl_nightly.sqsh}
export HF_HOME=$U/hf_home
export HF_DATASETS_CACHE=$U/hf_home/cache
# The login-shell HF_TOKEN is revoked and 401s even on public repos; model and data are local
unset HF_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_API_KEY
# The container's prebuilt Gym venv installs Gym editable from /opt/nemo-rl, so mount this
# checkout's Gym over it
export MOUNTS="/lustre:/lustre,$PWD/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
export COMMAND
export BASE_LOG_DIR=${EXP_DIR}/logs/${RUN_NAME}
mkdir -p "${BASE_LOG_DIR}"

echo "commit=$(git rev-parse HEAD) container=$(readlink -f "$CONTAINER")"
echo "replicas=${R} gen_nodes=${GEN_NODES} nodes=${NUM_NODES} pps=${PPS} gpp=${GPP} gbs=${GBS} concurrency=${CONCURRENCY}"
echo "COMMAND=${COMMAND}"
sbatch ${TEST_ONLY:+--test-only} \
  --nodes="${NUM_NODES}" \
  --account="${account}" \
  --job-name="${account}.${RUN_NAME}" \
  --partition=batch \
  --time="${TIME:-04:00:00}" \
  --gres=gpu:${GPUS_PER_NODE} \
  --segment "${NUM_NODES}" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"data_loading","description":"Async SWE GRPO: training GPUs idle while agents run"}}' \
  ray.sub