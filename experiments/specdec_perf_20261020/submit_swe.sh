#!/bin/bash
# Submit one async SWE GRPO arm on OCI-HSG with ray.sub (run from anywhere inside the repo).
# Port of ruit's run_grpo_qwen3_30b_swe_scale_gen.sh to latest main.
#
#   submit_swe.sh <config.yaml> <run_name> [max_steps] [extra overrides...]
#
# Env knobs (defaults reproduce ruit's 3-node Q30 layout):
#   NUM_VLLM_REPLICAS     vLLM engines (default 4)
#   GEN_TP                GPUs per engine, passed as vllm_cfg.tensor_parallel_size (default 2).
#                         Engines must fill whole nodes.
#   TRAIN_NODES           training nodes, added after the generation nodes (default 1)
#   NUM_PROMPTS_PER_STEP  prompts per step, 8 generations each (default: generation GPUs / 8,
#                         so GBS = one rollout per generation GPU, as in ruit's launcher).
#                         Agent concurrency = max(768, GBS).
#   SEGMENT               nodes per NVL72 block for --segment (default: all nodes). A block holds
#                         18 nodes, so jobs above that need it; match the config's
#                         cluster.segment_size.
#   ACCOUNT               Slurm account (default coreai_dlalgo_nemorl)
#   ENTRY                 training script (default examples/nemo_gym/run_grpo_nemo_gym.py, the legacy
#                         async path; examples/run_grpo_single_controller.py for the sc_* configs)
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
account=${ACCOUNT:-coreai_dlalgo_nemorl}
export GPUS_PER_NODE=4
# ray.sub sizes each worker step's memory by its CPU share; 140 of 144 CPUs asks for more
# than a GB200 node has free and intermittently fails step creation
export CPUS_PER_WORKER=96

R=${NUM_VLLM_REPLICAS:-4}
GEN_TP=${GEN_TP:-2}
TRAIN_NODES=${TRAIN_NODES:-1}
GEN_GPUS=$(( R * GEN_TP ))
if (( R < 1 || GEN_GPUS % GPUS_PER_NODE != 0 )); then
  echo "NUM_VLLM_REPLICAS x GEN_TP must fill whole ${GPUS_PER_NODE}-GPU nodes (got ${R} x ${GEN_TP})" >&2
  exit 1
fi
GEN_NODES=$(( GEN_GPUS / GPUS_PER_NODE ))
NUM_NODES=$(( GEN_NODES + TRAIN_NODES ))
GPP=8
PPS=${NUM_PROMPTS_PER_STEP:-$(( GEN_GPUS / GPP ))}
if (( PPS < 1 )); then
  echo "Fewer than ${GPP} generation GPUs: set NUM_PROMPTS_PER_STEP" >&2
  exit 1
fi
GBS=$(( PPS * GPP ))
CONCURRENCY=$(( GBS > 768 ? GBS : 768 ))

SWE_OVERRIDES="cluster.num_nodes=${NUM_NODES} policy.generation.colocated.resources.num_nodes=${GEN_NODES} policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} policy.generation.vllm_cfg.tensor_parallel_size=${GEN_TP} grpo.num_prompts_per_step=${PPS} grpo.num_generations_per_prompt=${GPP} policy.train_global_batch_size=${GBS}"
for split in train val; do
  SWE_OVERRIDES+=" env.nemo_gym.swe_agents_${split}.responses_api_agents.swe_agents.concurrency=${CONCURRENCY}"
done

export CONTAINER=${CONTAINER:-$U/containers/nemo_rl_nightly.sqsh}

# swe_agents runs its rollouts as Ray tasks SPREAD over all nodes, with its own venv python as
# py_executable, so Gym server venvs must exist on every node. The image ships /opt/gym_venvs
# empty, so build them once on lustre per Gym commit and image (the venv python links into the
# image's /opt/uv). skip_venv_if_present then reuses them without reinstalling into a venv
# other jobs import from. It trusts any venv whose python exists, so only one job may build:
# set GYM_VENV_BUILDER=1 on the first arm and submit the rest once its venvs are marked built.
# Parallel editable builds can exceed uv's default 300 s lock timeout. Worker venvs are rebuilt
# for the same reason as in submit.sh: the image's prebuilt ones do not match this checkout.
# The checkout goes first on PYTHONPATH, also as in submit.sh, so base-venv Ray workers import
# this nemo_rl (and init_ray's worker_process_setup_hook) rather than the image's older copy.
GYM_VENV_DIR=$U/gym_venvs/$(git -C 3rdparty/Gym-workspace/Gym rev-parse --short=12 HEAD)-$(basename "$(readlink -f "${CONTAINER}")" .sqsh)
GYM_VENVS=$(find "${GYM_VENV_DIR}" -mindepth 3 -maxdepth 3 -name .venv 2>/dev/null || true)
GYM_UNBUILT=$(find "${GYM_VENV_DIR}" -mindepth 3 -maxdepth 3 -name .venv ! -exec test -f {}/.nemo-gym-setup-complete \; -print 2>/dev/null || true)
if [[ -z ${GYM_VENV_BUILDER:-} ]] && [[ -z ${GYM_VENVS} || -n ${GYM_UNBUILT} ]]; then
  echo "Gym venvs in ${GYM_VENV_DIR} are missing or still building. Submit one arm with GYM_VENV_BUILDER=1 and wait for its .nemo-gym-setup-complete markers." >&2
  exit 1
fi
GYM_ENV="NRL_FORCE_REBUILD_VENVS=true NEMO_GYM_VENV_DIR=${GYM_VENV_DIR} UV_LOCK_TIMEOUT=3600 UV_HTTP_TIMEOUT=3600 PYTHONPATH=$(pwd)\${PYTHONPATH:+:\$PYTHONPATH}"
SWE_OVERRIDES+=" ++env.nemo_gym.skip_venv_if_present=true"

# The first OpenHands setup on lustre (miniforge + poetry install) outlasts Gym's 600 s server
# spinup default; later runs reuse the cached tree.
SWE_OVERRIDES+=" ++env.nemo_gym.server_spinup_timeout_seconds=3600"
# Gym's wait for the vLLM endpoints has the same 600 s default. Q235 engines with a drafter take
# longer than that to load and capture CUDA graphs.
SWE_OVERRIDES+=" ++env.nemo_gym.model_endpoint_readiness_timeout_seconds=3600"

# checkpointing stays off: workspace runs must not grow lustre inode/space usage. The env flag
# skips the per-step train_data_step*.jsonl dump of full agent trajectories (see submit.sh).
ENTRY=${ENTRY:-examples/nemo_gym/run_grpo_nemo_gym.py}
COMMAND="${GYM_ENV} uv run ${ENTRY} --config ${CONFIG} grpo.max_num_steps=${STEPS} checkpointing.enabled=false ++env.should_log_nemo_gym_responses=true ${SWE_OVERRIDES} logger.log_dir=${EXP_DIR}/exp_logs/${RUN_NAME} logger.wandb_enabled=True logger.wandb.project='sna-async-grpo-gb200' logger.wandb.name='${RUN_NAME}' $*"

mkdir -p "${EXP_DIR}/logs"
export HF_HOME=$U/hf_home
export HF_DATASETS_CACHE=$U/hf_home/cache
# Model and data are local; offline mode stops Hub 429s from hundreds of workers
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
echo "replicas=${R} gen_tp=${GEN_TP} gen_nodes=${GEN_NODES} train_nodes=${TRAIN_NODES} nodes=${NUM_NODES} pps=${PPS} gpp=${GPP} gbs=${GBS} concurrency=${CONCURRENCY}"
echo "COMMAND=${COMMAND}"
sbatch ${TEST_ONLY:+--test-only} \
  --nodes="${NUM_NODES}" \
  --account="${account}" \
  --job-name="${account}.${RUN_NAME}" \
  --partition=batch \
  --time="${TIME:-04:00:00}" \
  --gres=gpu:${GPUS_PER_NODE} \
  --segment "${SEGMENT:-${NUM_NODES}}" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"data_loading","description":"Async SWE GRPO: training GPUs idle while agents run"}}' \
  ray.sub