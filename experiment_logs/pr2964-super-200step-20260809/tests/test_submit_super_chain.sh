#!/bin/bash

set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
test_root=$(mktemp -d /tmp/pr2964-super-chain.XXXXXX)
trap 'rm -rf -- "${test_root}"' EXIT

fake_launcher=${test_root}/fake-round-launcher.sh
cat > "${fake_launcher}" <<'EOF'
#!/bin/bash
set -euo pipefail
mode=$1
round=$2
case "${mode}" in
  test-only)
    printf 'test-only|%s|\n' "${round}" >> "${FAKE_CALL_LOG}"
    printf 'test-only-ok\n'
    ;;
  submit)
    printf '%s|%s\n' "${round}" "${JOB_DEPENDENCY:-}" >> "${FAKE_CALL_LOG}"
    printf '%s\n' "$((600000 + round))"
    ;;
  *)
    exit 2
    ;;
esac
EOF
chmod +x "${fake_launcher}"

plan_output=$(bash "${experiment_dir}/submit_super_chain.sh" plan 3)
[[ "${plan_output}" == $'round=1 dependency=none\nround=2 dependency=afterok:<round-1-job>\nround=3 dependency=afterok:<round-2-job>' ]]

call_log=${test_root}/calls.tsv
manifest=${test_root}/job-chain.env
FAKE_CALL_LOG=${call_log} \
ROUND_LAUNCHER_OVERRIDE=${fake_launcher} \
CHAIN_MANIFEST_OVERRIDE=${manifest} \
  bash "${experiment_dir}/submit_super_chain.sh" submit 3 > "${test_root}/submit.out"

expected_calls=$'test-only|1|\n1|\n2|afterok:600001\n3|afterok:600002'
[[ "$(<"${call_log}")" == "${expected_calls}" ]]

expected_manifest=$'hybridep_round1=600001\nhybridep_round2=600002\nhybridep_round3=600003'
[[ "$(<"${manifest}")" == "${expected_manifest}" ]]

grep -Fxq 'submitted_round=1 job_id=600001 dependency=none' "${test_root}/submit.out"
grep -Fxq 'submitted_round=2 job_id=600002 dependency=afterok:600001' "${test_root}/submit.out"
grep -Fxq 'submitted_round=3 job_id=600003 dependency=afterok:600002' "${test_root}/submit.out"

if bash "${experiment_dir}/submit_super_chain.sh" plan 0 > /dev/null 2>&1; then
  printf 'round count 0 must be rejected\n' >&2
  exit 1
fi

if bash "${experiment_dir}/submit_super_chain.sh" plan 25 > /dev/null 2>&1; then
  printf 'round count 25 must be rejected\n' >&2
  exit 1
fi

fake_base_submit=${test_root}/fake-base-submit.sh
cat > "${fake_base_submit}" <<'EOF'
#!/bin/bash
set -euo pipefail
printf 'args=%s|%s|%s\n' "$1" "$2" "$3"
printf 'max_steps=%s\n' "${MAX_NUM_STEPS_OVERRIDE}"
printf 'time_limit=%s\n' "${TIME_LIMIT_OVERRIDE}"
printf 'experiment_root=%s\n' "${EXPERIMENT_ROOT_OVERRIDE}"
printf 'run_name=%s\n' "${RUN_NAME_OVERRIDE}"
printf 'checkpoint_enabled=%s\n' "${CHECKPOINTING_ENABLED_OVERRIDE}"
printf 'checkpoint_dir=%s\n' "${CHECKPOINT_DIR_OVERRIDE}"
printf 'save_period=%s\n' "${CHECKPOINT_SAVE_PERIOD_OVERRIDE}"
printf 'save_deadline=%s\n' "${CHECKPOINT_MUST_SAVE_BY_OVERRIDE}"
printf 'keep_top_k=%s\n' "${CHECKPOINT_KEEP_TOP_K_OVERRIDE}"
printf 'ft_keep_latest_k=%s\n' "${CHECKPOINT_FT_KEEP_LATEST_K_OVERRIDE}"
printf 'save_optimizer=%s\n' "${CHECKPOINT_SAVE_OPTIMIZER_OVERRIDE}"
EOF
chmod +x "${fake_base_submit}"

round_output=$(BASE_SUBMIT_SCRIPT_OVERRIDE=${fake_base_submit} \
  bash "${experiment_dir}/submit_super_4hour.sh" test-only 7)
[[ "${round_output}" == *'args=nemotron3-super|hybridep|test-only'* ]]
[[ "${round_output}" == *'max_steps=200'* ]]
[[ "${round_output}" == *'time_limit=04:00:00'* ]]
[[ "${round_output}" == *'experiment_root=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-super-200step-20260809'* ]]
[[ "${round_output}" == *'run_name=nemotron3-super-sync-hybridep-pr2964-200step-round7'* ]]
[[ "${round_output}" == *'checkpoint_enabled=true'* ]]
[[ "${round_output}" == *'checkpoint_dir=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-super-200step-20260809/checkpoints/hybridep'* ]]
[[ "${round_output}" == *'save_period=200'* ]]
[[ "${round_output}" == *'save_deadline=00:03:15:00'* ]]
[[ "${round_output}" == *'keep_top_k=1'* ]]
[[ "${round_output}" == *'ft_keep_latest_k=1'* ]]
[[ "${round_output}" == *'save_optimizer=true'* ]]

if BASE_SUBMIT_SCRIPT_OVERRIDE=${fake_base_submit} \
  bash "${experiment_dir}/submit_super_4hour.sh" submit 0 > /dev/null 2>&1; then
  printf 'round 0 must be rejected\n' >&2
  exit 1
fi

printf 'super-chain-tests-pass\n'
