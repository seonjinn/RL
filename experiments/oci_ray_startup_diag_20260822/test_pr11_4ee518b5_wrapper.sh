#!/usr/bin/env bash
# shellcheck disable=SC2016

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="${SCRIPT_DIR}/pr11_4ee518b5.sbatch"
DISTRIBUTED_HARNESS="${SCRIPT_DIR}/run_tp2_pp2_cp2_refit.py"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-/home/sna/pr3757-final-4ee518b5}"
EXPECTED_HEAD=4ee518b5dc2ed16f75e31876b477ea5ecf7d8c9b

test -f "${WRAPPER}"
test -f "${DISTRIBUTED_HARNESS}"
test "$(git -C "${CANDIDATE_ROOT}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
candidate_status="$(git -C "${CANDIDATE_ROOT}" status --porcelain --untracked-files=all --ignore-submodules=none)"
test -z "${candidate_status}"
bash -n "${WRAPPER}"

required_contracts=(
  '#SBATCH --account=nemotron_n3_post'
  '#SBATCH --partition=batch'
  '#SBATCH --qos=normal'
  '#SBATCH --nodes=2'
  '#SBATCH --gres=gpu:4'
  '#SBATCH --time=04:00:00'
  'expected_head=4ee518b5dc2ed16f75e31876b477ea5ecf7d8c9b'
  'expected_per_node_bootstrap_sha=670920393de28eb76548015c01c2bdb20c5a745ed7979ad5d579758991e8a5d2'
  'expected_per_node_launcher_sha=51e9e0d44a499a457b365d3dbe1bbc5d16dae2edb2049f11db462c001c7a4313'
  'expected_distributed_harness_sha=2fc298a9ba248648db362ded7ebb052e490606010568878955bbb5cc7b5f684c'
  'source_dir=/home/sna/pr3757-final-4ee518b5'
  'harness_root=/home/sna/pr3757-pr11-harness-pernode/experiments/oci_ray_startup_diag_20260822'
  'result_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr3757-final-4ee518b5-gate'
  'scratch_root=/raid/scratch/sna/pr3757-final-${SLURM_JOB_ID}'
  'venv_dir=${scratch_root}/venv'
  'ray_harness=${harness_root}/ray_then_pytest.sh'
  'ray_contract=${harness_root}/test_ray_then_pytest_harness.sh'
  'per_node_bootstrap=${harness_root}/pr11_per_node_bootstrap.sh'
  'per_node_launcher=${harness_root}/pr11_per_node_launcher.sh'
  'distributed_harness=${harness_root}/run_tp2_pp2_cp2_refit.py'
  'q30_runner=${source_dir}/scripts/run_focused_qwen_moe_draft_tests.sh'
  'source_status="$(git -C'
  'submodule_status="$(git -C'
  'stage=per-node-mcore-build'
  '/bin/bash "${per_node_launcher}"'
  'pr11-node-bootstrap-${SLURM_JOB_ID}-${node_name}.txt'
  'installed_distributions_sha256'
  'per_node_distribution_sha256='
  'bash "${ray_harness}"'
  'tests/unit/models/policy/test_dflash_worker_validation.py::test_disabled_typed_dflash_config_has_no_refit_metadata'
  'tests/unit/weight_sync/test_nccl_reshard_utils.py::test_build_refit_info_tp2_pp2_cp2_stage_mesh_keeps_both_cp_replicas'
  'tests/unit/models/policy/test_dflash_worker_validation.py'
  'tests/unit/models/megatron/test_draft_refit.py'
  'tests/unit/weight_sync/test_nccl_reshard_utils.py'
  'tests/unit/utils/test_packed_tensor.py::test_packed_broadcast_midstream_failure_reaches_consumer'
  'tests/unit/utils/test_packed_tensor.py::test_packed_broadcast_consumer_drains_after_load_failure'
  'tests/unit/models/generation/test_vllm_quant_lifecycle.py::test_real_quant_lifecycle_finalizes_cotrained_draft_after_target'
  'tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_real_quant_reload_finalizes_cotrained_draft_after_target'
  'tests/unit/models/policy/test_draft_config.py::test_eagle3_draft_config_rejects_near_miss_typo_keys'
  'tests/unit/algorithms/test_draft_soft_ce.py::test_streaming_soft_ce_rejects_out_of_range_bin_ids'
  'tests/unit/algorithms/test_projected_draft_soft_ce.py::test_projected_soft_ce_rejects_out_of_range_bin_ids'
  'tests/unit/algorithms/test_projected_draft_soft_ce.py::test_projected_soft_ce_validates_explicit_bin_ids_once_per_call'
  'tests/unit/models/megatron/test_draft_step_state.py::test_split_accumulation_matches_monolithic_normalization'
  'tests/unit/algorithms/test_draft_loss_wrapper.py::test_draft_loss_wrapper_defers_raw_stats_for_split_step'
  'tests/unit/algorithms/test_dspark_objective.py::test_dspark_objective_matches_dense_component_oracles'
  'tests/unit/models/megatron/test_dspark_provider.py::test_factory_rejects_markov_confidence_without_confidence'
  'tests/unit/models/megatron/test_dspark_provider.py::test_objective_stats_rejects_malformed_hidden_and_head'
  'test "${#pytest_targets[@]}" -eq 18'
  'tests/unit/models/megatron/test_draft_optimizer.py::test_draft_weight_decay_override_keeps_norm_and_bias_at_zero_decay'
  'tests/unit/models/megatron/test_dspark_heads.py::test_tp2_markov_head_sums_replicated_w1_gradients'
  'tests/unit/models/megatron/test_dspark_heads.py::test_tp2_confidence_path_sanitizes_ids_without_tp_gradient_summing'
  'tests/unit/models/megatron/test_dspark_heads.py::test_tp2_markov_head_megatron_checkpoint_round_trip'
  'test "${#q30_extra_targets[@]}" -eq 4'
  'stage=q30-focused-runner'
  '/bin/bash "${q30_runner}" "${q30_extra_targets[@]}"'
  '--nnodes=2'
  '--nproc-per-node=4'
  'run_tp2_pp2_cp2_refit.py'
  'wrapper_sha256='
  'ray_harness_sha256='
  'distributed_harness_sha256='
  'per_node_bootstrap_sha256='
  'per_node_launcher_sha256='
  'q30_runner_sha256='
  'exclusions=none'
  'final_stage=${stage}'
)

for contract in "${required_contracts[@]}"; do
  grep -Fq -- "${contract}" "${WRAPPER}"
done

sha256_file() {
  local path=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

read_wrapper_pin() {
  local name=$1
  awk -F= -v name="${name}" '$1 == name { print $2; exit }' "${WRAPPER}"
}

test "$(read_wrapper_pin expected_harness_sha)" = \
  "$(sha256_file "${SCRIPT_DIR}/ray_then_pytest.sh")"
test "$(read_wrapper_pin expected_contract_sha)" = \
  "$(sha256_file "${SCRIPT_DIR}/test_ray_then_pytest_harness.sh")"

test "$(grep -Fc '/bin/bash "${per_node_launcher}"' "${WRAPPER}")" -eq 1
test "$(grep -Fc 'bash "${ray_harness}"' "${WRAPPER}")" -eq 1
test "$(grep -Fc '/bin/bash "${q30_runner}" "${q30_extra_targets[@]}"' "${WRAPPER}")" -eq 1
test "$(grep -Fc 'tests/unit/algorithms/test_projected_draft_soft_ce.py::test_projected_soft_ce_rejects_out_of_range_bin_ids' "${WRAPPER}")" -eq 1
test "$(grep -Fc 'tests/unit/algorithms/test_projected_draft_soft_ce.py::test_projected_soft_ce_validates_explicit_bin_ids_once_per_call' "${WRAPPER}")" -eq 1

if grep -Eq '^[[:space:]]+env[[:space:]]*\\$' "${WRAPPER}"; then
  echo "bare env is forbidden in the per-node srun launch" >&2
  exit 1
fi

if grep -Fq 'bash "${per_node_bootstrap}"' "${WRAPPER}"; then
  echo "bootstrap must be entered through the absolute launcher" >&2
  exit 1
fi

trap_line="$(grep -n '^trap finish EXIT$' "${WRAPPER}" | cut -d: -f1)"
first_source_check_line="$(grep -n '^actual_head=' "${WRAPPER}" | cut -d: -f1)"
test "${trap_line}" -lt "${first_source_check_line}"

if grep -Eq -- '(^|[[:space:]])(-k|--deselect|--ignore|--ignore-glob|--exclude)(=|[[:space:]]|$)' "${WRAPPER}"; then
  echo "test deselection is forbidden for the bounded PR11 MCore suite" >&2
  exit 1
fi

if grep -Eq -- '(^|[[:space:]])(scancel)([[:space:]]|$)' "${WRAPPER}"; then
  echo "scheduler cancellation command found in wrapper" >&2
  exit 1
fi

required_distributed_contracts=(
  'SOURCE = Path(os.environ["SOURCE_ROOT"])'
  'EXPECTED_SHA = os.environ["EXPECTED_HEAD"]'
  'assert world_size == 8'
  '"_run_tp2_pp2_cp2_worker_draft_refit"'
  '"_run_tp2_pp2_cp2_worker_draft_failure_consensus"'
  '"_run_cp_lane_manifest_mismatch"'
  'tests=3 topology=TP2xPP2xCP2'
)
for contract in "${required_distributed_contracts[@]}"; do
  grep -Fq -- "${contract}" "${DISTRIBUTED_HARNESS}"
done
if grep -Fq '/home/' "${DISTRIBUTED_HARNESS}"; then
  echo "distributed harness must use the exact wrapper-provided source" >&2
  exit 1
fi

focused_nodes=(
  tests/unit/utils/test_packed_tensor.py::test_packed_broadcast_midstream_failure_reaches_consumer
  tests/unit/utils/test_packed_tensor.py::test_packed_broadcast_consumer_drains_after_load_failure
  tests/unit/models/generation/test_vllm_quant_lifecycle.py::test_real_quant_lifecycle_finalizes_cotrained_draft_after_target
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_real_quant_reload_finalizes_cotrained_draft_after_target
  tests/unit/models/policy/test_draft_config.py::test_eagle3_draft_config_rejects_near_miss_typo_keys
  tests/unit/algorithms/test_draft_soft_ce.py::test_streaming_soft_ce_rejects_out_of_range_bin_ids
  tests/unit/algorithms/test_projected_draft_soft_ce.py::test_projected_soft_ce_rejects_out_of_range_bin_ids
  tests/unit/algorithms/test_projected_draft_soft_ce.py::test_projected_soft_ce_validates_explicit_bin_ids_once_per_call
  tests/unit/models/megatron/test_draft_step_state.py::test_split_accumulation_matches_monolithic_normalization
  tests/unit/algorithms/test_draft_loss_wrapper.py::test_draft_loss_wrapper_defers_raw_stats_for_split_step
  tests/unit/algorithms/test_dspark_objective.py::test_dspark_objective_matches_dense_component_oracles
  tests/unit/models/megatron/test_dspark_provider.py::test_factory_rejects_markov_confidence_without_confidence
  tests/unit/models/megatron/test_dspark_provider.py::test_objective_stats_rejects_malformed_hidden_and_head
  tests/unit/models/megatron/test_draft_optimizer.py::test_draft_weight_decay_override_keeps_norm_and_bias_at_zero_decay
  tests/unit/models/megatron/test_dspark_heads.py::test_tp2_markov_head_sums_replicated_w1_gradients
  tests/unit/models/megatron/test_dspark_heads.py::test_tp2_confidence_path_sanitizes_ids_without_tp_gradient_summing
  tests/unit/models/megatron/test_dspark_heads.py::test_tp2_markov_head_megatron_checkpoint_round_trip
)
for node in "${focused_nodes[@]}"; do
  relative_path="${node%%::*}"
  test_name="${node##*::}"
  test -f "${CANDIDATE_ROOT}/${relative_path}"
  grep -Eq "^(async )?def ${test_name}\\(" "${CANDIDATE_ROOT}/${relative_path}"
done

projected_bounds_test="${CANDIDATE_ROOT}/tests/unit/algorithms/test_projected_draft_soft_ce.py"
grep -Fq '[pytest.param(4, id="one_tile"), pytest.param(5, id="multiple_tiles")]' \
  "${projected_bounds_test}"
grep -Fq '[pytest.param(-1, id="negative"), pytest.param(2, id="too_large")]' \
  "${projected_bounds_test}"
grep -Fq 'def test_projected_soft_ce_validates_explicit_bin_ids_once_per_call(' \
  "${projected_bounds_test}"
grep -Fq 'assert validation_calls == ["min", "max"]' "${projected_bounds_test}"

q30_runner="${CANDIDATE_ROOT}/scripts/run_focused_qwen_moe_draft_tests.sh"
test -f "${q30_runner}"
for q30_path in \
  tests/unit/models/megatron/test_dflash_model.py \
  tests/unit/models/megatron/test_dflash_asymmetric_tp_export.py \
  tests/unit/models/megatron/test_dspark_training_provider.py \
  tests/unit/test_qwen_moe_draft_linux_runner_contract.py; do
  grep -Fq "${q30_path}" "${q30_runner}"
done
grep -Fq 'python -m pytest -q --mcore-only "${MCORE_TESTS[@]}" "$@"' "${q30_runner}"

candidate_distributed_tests="${CANDIDATE_ROOT}/tests/unit/models/megatron/test_draft_refit.py"
for distributed_name in \
  _run_tp2_pp2_cp2_worker_draft_refit \
  _run_tp2_pp2_cp2_worker_draft_failure_consensus \
  _run_cp_lane_manifest_mismatch; do
  grep -Eq "^def ${distributed_name}\\(" "${candidate_distributed_tests}"
done

echo "PR11_4EE518B5_WRAPPER_CONTRACT_PASS"
