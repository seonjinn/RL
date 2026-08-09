#!/bin/bash

set -euo pipefail

dispatcher=${1:-}
mode=${2:-submit}
round=${3:-}
case "${dispatcher}" in
  baseline|hybridep) ;;
  *) printf 'Usage: %s {baseline|hybridep} {submit|test-only} ROUND\n' "$0" >&2; exit 2 ;;
esac
case "${mode}" in
  submit|test-only) ;;
  *) printf 'Usage: %s {baseline|hybridep} {submit|test-only} ROUND\n' "$0" >&2; exit 2 ;;
esac
case "${round}" in
  1|2|3) ;;
  *) printf 'Usage: %s {baseline|hybridep} {submit|test-only} ROUND\n' "$0" >&2; exit 2 ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
submit_script=${script_dir}/../pr2964-20step-20260807/submit_performance_20step.sh
work_root=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna
source_experiment_root=${work_root}/experiments/pr2964-20step-20260807

export MAX_NUM_STEPS_OVERRIDE=200
export TIME_LIMIT_OVERRIDE=04:00:00
export EXPERIMENT_ROOT_OVERRIDE=${work_root}/experiments/pr2964-q30-4hour-20260809
export RUN_NAME_OVERRIDE=qwen3-30ba3b-sync-${dispatcher}-pr2964-200step-round${round}
export CHECKPOINTING_ENABLED_OVERRIDE=true
export CHECKPOINT_DIR_OVERRIDE=${EXPERIMENT_ROOT_OVERRIDE}/checkpoints/${dispatcher}
export CHECKPOINT_SAVE_PERIOD_OVERRIDE=200
export CHECKPOINT_MUST_SAVE_BY_OVERRIDE=00:03:15:00
export CHECKPOINT_METRIC_NAME_OVERRIDE=null
export CHECKPOINT_KEEP_TOP_K_OVERRIDE=1
export CHECKPOINT_FT_KEEP_LATEST_K_OVERRIDE=1
export CHECKPOINT_SAVE_OPTIMIZER_OVERRIDE=true
export VALIDATION_REPO_OVERRIDE=${source_experiment_root}/RL-latest-bridge-validation-20260808
export VALIDATION_HEAD_OVERRIDE=541413bd2912561950413b39809db40590a652bb
export MCORE_SOURCE_OVERRIDE=${source_experiment_root}/Megatron-LM-14499-routing-34b55f
export MCORE_EXPECTED_COMMIT_OVERRIDE=34b55f24f0826c9aebd6693ecb60648cd934737d
export HYBRIDEP_DEPENDENCY_ANCESTOR_OVERRIDE=a9aaa395c37963a9fd8a7320d61a516c7b714e57
export CONTAINER_OVERRIDE=${work_root}/containers/nemo-rl-nightly-cw-fallback-20260808/nemo_rl_nightly_20260805_15171871.sqsh
export HYBRID_EP_WHEEL_OVERRIDE=${source_experiment_root}/deepep-wheels/17cfb817bccec3a9c247013360cc550c2bac441e-dmabuf-506072/deep_ep-1.2.1+17cfb81-cp313-cp313-linux_x86_64.whl
export HYBRID_EP_WHEEL_SHA256_OVERRIDE=f181085dcbfdcb88bc2a33f9df52d4acfd99d1f5e3a73a03ce3dfa38947f559d
export SLURM_EXCLUDE=pool0-0167,pool0-0272,pool0-0337
export NRL_FORCE_REBUILD_VENVS_OVERRIDE=false

exec bash "${submit_script}" qwen3-30ba3b "${dispatcher}" "${mode}"
