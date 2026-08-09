#!/bin/bash

set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${experiment_dir}/performance_case.sh"

assert_contains() {
  local haystack=$1
  local needle=$2
  [[ "${haystack}" == *"${needle}"* ]] || {
    printf 'expected %q in %q\n' "${needle}" "${haystack}" >&2
    exit 1
  }
}

assert_not_contains() {
  local haystack=$1
  local needle=$2
  [[ "${haystack}" != *"${needle}"* ]] || {
    printf 'did not expect %q in %q\n' "${needle}" "${haystack}" >&2
    exit 1
  }
}

render_case qwen3-30ba3b baseline /tmp/q30-baseline
q30_baseline=$(printf '%s\n' "${driver_args[@]}")
[[ "${num_nodes}" == 4 ]]
[[ "${segment_size}" == 4 ]]
assert_contains "${q30_baseline}" 'grpo-qwen3-30ba3b-4n8g.yaml'
assert_contains "${q30_baseline}" 'grpo.max_num_steps=20'
assert_contains "${q30_baseline}" 'policy.megatron_cfg.moe_token_dispatcher_type=alltoall'
assert_contains "${q30_baseline}" 'checkpointing.enabled=false'
assert_contains "${q30_baseline}" 'logger.tensorboard_enabled=true'
assert_not_contains "${q30_baseline}" 'moe_flex_dispatcher_backend=hybridep'
assert_not_contains "${q30_baseline}" 'moe_hybridep_prepad_packed_inputs=true'

MAX_NUM_STEPS_OVERRIDE=1000
render_case qwen3-30ba3b baseline /tmp/q30-baseline-maxsteps
q30_baseline_maxsteps=$(printf '%s\n' "${driver_args[@]}")
unset MAX_NUM_STEPS_OVERRIDE
assert_contains "${q30_baseline_maxsteps}" 'grpo.max_num_steps=1000'

CHECKPOINTING_ENABLED_OVERRIDE=true
CHECKPOINT_DIR_OVERRIDE=/tmp/q30-checkpoints
CHECKPOINT_SAVE_PERIOD_OVERRIDE=200
CHECKPOINT_MUST_SAVE_BY_OVERRIDE=00:03:15:00
CHECKPOINT_METRIC_NAME_OVERRIDE=null
CHECKPOINT_KEEP_TOP_K_OVERRIDE=1
CHECKPOINT_FT_KEEP_LATEST_K_OVERRIDE=1
CHECKPOINT_SAVE_OPTIMIZER_OVERRIDE=true
render_case qwen3-30ba3b baseline /tmp/q30-baseline-checkpoint
q30_baseline_checkpoint=$(printf '%s\n' "${driver_args[@]}")
unset CHECKPOINTING_ENABLED_OVERRIDE
unset CHECKPOINT_DIR_OVERRIDE
unset CHECKPOINT_SAVE_PERIOD_OVERRIDE
unset CHECKPOINT_MUST_SAVE_BY_OVERRIDE
unset CHECKPOINT_METRIC_NAME_OVERRIDE
unset CHECKPOINT_KEEP_TOP_K_OVERRIDE
unset CHECKPOINT_FT_KEEP_LATEST_K_OVERRIDE
unset CHECKPOINT_SAVE_OPTIMIZER_OVERRIDE
assert_contains "${q30_baseline_checkpoint}" 'checkpointing.enabled=true'
assert_contains "${q30_baseline_checkpoint}" 'checkpointing.checkpoint_dir=/tmp/q30-checkpoints'
assert_contains "${q30_baseline_checkpoint}" 'checkpointing.save_period=200'
assert_contains "${q30_baseline_checkpoint}" 'checkpointing.checkpoint_must_save_by=00:03:15:00'
assert_contains "${q30_baseline_checkpoint}" 'checkpointing.metric_name=null'
assert_contains "${q30_baseline_checkpoint}" 'checkpointing.keep_top_k=1'
assert_contains "${q30_baseline_checkpoint}" '++checkpointing.ft_keep_latest_k=1'
assert_contains "${q30_baseline_checkpoint}" 'checkpointing.save_optimizer=true'
assert_not_contains "${q30_baseline_checkpoint}" 'checkpointing.enabled=false'

render_case qwen3-30ba3b hybridep /tmp/q30-hybridep
q30_hybridep=$(printf '%s\n' "${driver_args[@]}")
assert_contains "${q30_hybridep}" '++policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true'

render_case qwen3-235b hybridep /tmp/q235-hybridep
q235_hybridep=$(printf '%s\n' "${driver_args[@]}")
[[ "${num_nodes}" == 16 ]]
[[ -z "${segment_size}" ]]
assert_contains "${q235_hybridep}" 'grpo-qwen3-235b-16n8g.yaml'
assert_contains "${q235_hybridep}" 'policy.megatron_cfg.moe_token_dispatcher_type=flex'
assert_contains "${q235_hybridep}" '++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep'
assert_contains "${q235_hybridep}" "++policy.megatron_cfg.env_vars.NVLINK_DOMAIN_SIZE='8'"
assert_contains "${q235_hybridep}" '++policy.generation.vllm_kwargs.disable_custom_all_reduce=true'
assert_contains "${q235_hybridep}" '++policy.generation.vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms=false'
assert_not_contains "${q235_hybridep}" 'logger.tensorboard_enabled=true'
assert_not_contains "${q235_hybridep}" 'moe_hybridep_prepad_packed_inputs=true'

render_case qwen3-235b baseline /tmp/q235-baseline
q235_baseline=$(printf '%s\n' "${driver_args[@]}")
assert_contains "${q235_baseline}" 'policy.megatron_cfg.moe_token_dispatcher_type=alltoall'
assert_contains "${q235_baseline}" '++policy.generation.vllm_kwargs.disable_custom_all_reduce=true'
assert_contains "${q235_baseline}" '++policy.generation.vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms=false'

render_case nemotron3-super baseline /tmp/super-baseline
super_baseline=$(printf '%s\n' "${driver_args[@]}")
[[ "${num_nodes}" == 32 ]]
assert_contains "${super_baseline}" 'grpo-nemotron3-super-120BA12B-32n8g.yaml'
assert_contains "${super_baseline}" '++policy.generation.vllm_kwargs.disable_custom_all_reduce=true'
assert_contains "${super_baseline}" '++policy.generation.vllm_kwargs.moe_backend=triton'
assert_contains "${super_baseline}" 'policy.generation.vllm_cfg.enforce_eager=true'
assert_contains "${super_baseline}" 'logger.tensorboard_enabled=true'
assert_not_contains "${super_baseline}" 'moe_flex_dispatcher_backend=hybridep'

render_case nemotron3-super hybridep /tmp/super-hybridep
super_hybridep=$(printf '%s\n' "${driver_args[@]}")
assert_contains "${super_hybridep}" '++policy.generation.vllm_kwargs.disable_custom_all_reduce=true'
assert_contains "${super_hybridep}" '++policy.generation.vllm_kwargs.moe_backend=triton'
assert_contains "${super_hybridep}" 'policy.generation.vllm_cfg.enforce_eager=true'
assert_contains "${super_hybridep}" '++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep'
assert_contains "${super_hybridep}" '++policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true'

submit_script=$(<"${experiment_dir}/submit_performance_20step.sh")
assert_contains "${submit_script}" '--comment=${job_reaper_comment}'
assert_contains "${submit_script}" '"exemptIdleTimeMins":"90"'
assert_contains "${submit_script}" 'model initialization and colocated vLLM startup'
assert_contains "${submit_script}" 'nemotron3-super) default_time_limit=08:00:00'
assert_contains "${submit_script}" 'experiment_root=${EXPERIMENT_ROOT_OVERRIDE:-${work_root}/experiments/pr2964-20step-20260807}'
assert_contains "${submit_script}" 'max_num_steps=${MAX_NUM_STEPS_OVERRIDE:-20}'
assert_contains "${submit_script}" 'time_limit=${TIME_LIMIT_OVERRIDE:-${default_time_limit}}'
assert_contains "${submit_script}" 'repo=${VALIDATION_REPO_OVERRIDE:-${work_root}/experiments/pr2964-20step-20260807/RL}'
assert_contains "${submit_script}" 'validation_head=${VALIDATION_HEAD_OVERRIDE:-a028b33bcde0ef8aeb9fcc626a2e0c57fb568d2f}'
assert_contains "${submit_script}" 'test "$(git -C "${repo}" rev-parse HEAD)" = "${validation_head}"'
assert_contains "${submit_script}" 'mcore_source=${MCORE_SOURCE_OVERRIDE:-${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM}'
assert_contains "${submit_script}" 'mcore_commit=${MCORE_EXPECTED_COMMIT_OVERRIDE:-$(git -C "${mcore_source}" rev-parse HEAD)}'
assert_contains "${submit_script}" 'hybridep_dependency_ancestor=${HYBRIDEP_DEPENDENCY_ANCESTOR_OVERRIDE:-a9aaa395c37963a9fd8a7320d61a516c7b714e57}'
assert_contains "${submit_script}" 'test "$(git -C "${mcore_source}" rev-parse HEAD)" = "${mcore_commit}"'
assert_contains "${submit_script}" 'git -C "${repo}" merge-base --is-ancestor "${hybridep_dependency_ancestor}" HEAD'
assert_contains "${submit_script}" 'PYTHONPATH="${overlay}:${repo}:${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${mcore_source}"'
assert_contains "${submit_script}" "import megatron.core; print(megatron.core.__file__)"
assert_contains "${submit_script}" 'mcore_source=%s\nmcore_commit=%s\n'
assert_contains "${submit_script}" 'force_rebuild_venvs=${NRL_FORCE_REBUILD_VENVS_OVERRIDE:-false}'
assert_contains "${submit_script}" 'export NRL_FORCE_REBUILD_VENVS="${force_rebuild_venvs}"'
assert_contains "${submit_script}" 'export UV_FROZEN=1'
assert_contains "${submit_script}" 'nrl_force_rebuild_venvs=%s\n'
assert_contains "${submit_script}" 'max_num_steps=%s\ntime_limit=%s\n'

long_experiment_dir=$(cd "${experiment_dir}/.." && pwd)/pr2964-q30-4hour-20260809
long_submit_path=${long_experiment_dir}/submit_q30_4hour.sh
[[ -f "${long_submit_path}" ]] || {
  printf 'missing four-hour launcher: %s\n' "${long_submit_path}" >&2
  exit 1
}
long_submit_script=$(<"${long_submit_path}")
assert_contains "${long_submit_script}" 'round=${3:-}'
assert_contains "${long_submit_script}" 'attempt=${ATTEMPT_SUFFIX_OVERRIDE:-}'
assert_contains "${long_submit_script}" 'retry[1-9]|retry[1-9][0-9]) ;;'
assert_contains "${long_submit_script}" '1|2|3) ;;'
assert_contains "${long_submit_script}" 'MAX_NUM_STEPS_OVERRIDE=200'
assert_contains "${long_submit_script}" 'TIME_LIMIT_OVERRIDE=04:00:00'
assert_contains "${long_submit_script}" 'CHECKPOINTING_ENABLED_OVERRIDE=true'
assert_contains "${long_submit_script}" 'CHECKPOINT_DIR_OVERRIDE=${EXPERIMENT_ROOT_OVERRIDE}/checkpoints/${dispatcher}'
assert_contains "${long_submit_script}" 'CHECKPOINT_SAVE_PERIOD_OVERRIDE=200'
assert_contains "${long_submit_script}" 'CHECKPOINT_MUST_SAVE_BY_OVERRIDE=00:03:15:00'
assert_contains "${long_submit_script}" 'CHECKPOINT_METRIC_NAME_OVERRIDE=null'
assert_contains "${long_submit_script}" 'CHECKPOINT_KEEP_TOP_K_OVERRIDE=1'
assert_contains "${long_submit_script}" 'CHECKPOINT_FT_KEEP_LATEST_K_OVERRIDE=1'
assert_contains "${long_submit_script}" 'CHECKPOINT_SAVE_OPTIMIZER_OVERRIDE=true'
assert_contains "${long_submit_script}" 'VALIDATION_HEAD_OVERRIDE=541413bd2912561950413b39809db40590a652bb'
assert_contains "${long_submit_script}" 'HYBRIDEP_DEPENDENCY_ANCESTOR_OVERRIDE=4846673cf66cb47fc1eecf0ea22d17c1bead8f75'
assert_contains "${long_submit_script}" 'MCORE_EXPECTED_COMMIT_OVERRIDE=34b55f24f0826c9aebd6693ecb60648cd934737d'
assert_contains "${long_submit_script}" 'SLURM_EXCLUDE=${SLURM_EXCLUDE_OVERRIDE:-pool0-0167,pool0-0272,pool0-0337}'
assert_contains "${long_submit_script}" 'RUN_NAME_OVERRIDE=qwen3-30ba3b-sync-${dispatcher}-pr2964-200step-round${round}${attempt:+-${attempt}}'
assert_contains "${long_submit_script}" 'exec bash "${submit_script}" qwen3-30ba3b "${dispatcher}" "${mode}"'

focused_test_script=$(<"${experiment_dir}/submit_hybridep_prepadding_tests.sh")
assert_contains "${focused_test_script}" 'tests/unit/models/megatron/test_hybridep_data.py::test_hybridep_prepads_packed_inputs_before_model_forward'
assert_not_contains "${focused_test_script}" 'tests/unit/models/megatron/test_megatron_data.py::test_hybridep_'
assert_contains "${focused_test_script}" 'HYBRIDEP_ENABLE_COVERAGE'
assert_contains "${focused_test_script}" 'coverage.json'

report_html=$(<"${experiment_dir}/report/index.html")
assert_contains "${report_html}" 'assets/step-time-improvement.png'
assert_contains "${report_html}" 'assets/throughput-improvement.png'
assert_contains "${report_html}" 'Positive bars mean HybridEP improved the metric'
assert_contains "${report_html}" 'train/mean_total_tokens_per_sample'
assert_contains "${report_html}" '3.16% fewer tokens per sample'
assert_contains "${report_html}" 'job 507302 · external routing MCore'
assert_not_contains "${report_html}" "507302 completed 20/20 with Bridge's stock MCore"
assert_contains "${report_html}" 'Steady-state window sensitivity · Steps 5–20'
assert_contains "${report_html}" 'n=16/16'
assert_contains "${report_html}" 'ratio-of-sums policy TPS is 52.09% higher'
assert_contains "${report_html}" 'Job 507275 completed 20/20 in 5:16:18'
assert_contains "${report_html}" '896.590 ± 264.189 s'
assert_contains "${report_html}" '3.765 ± 0.926'
assert_contains "${report_html}" 'byte-for-byte identical to PR #2964 head'
assert_not_contains "${report_html}" 'Super 507275 completed 15 steps'
assert_contains "${report_html}" 'Active 200-step resumed Qwen3-30B-A3B A/B'
assert_contains "${report_html}" '508637 → 508638 → 508639'
assert_contains "${report_html}" '508612 → 508613 → 508614'
assert_contains "${report_html}" 'pool0-0343'

printf 'performance-case-tests-pass\n'
