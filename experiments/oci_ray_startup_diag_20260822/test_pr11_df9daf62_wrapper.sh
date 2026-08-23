#!/usr/bin/env bash
# shellcheck disable=SC2016

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="${SCRIPT_DIR}/pr11_df9daf62.sbatch"

test -f "${WRAPPER}"
bash -n "${WRAPPER}"

required_contracts=(
  '#SBATCH --account=nemotron_n3_post'
  '#SBATCH --partition=batch'
  '#SBATCH --qos=normal'
  '#SBATCH --nodes=2'
  '#SBATCH --gres=gpu:4'
  '#SBATCH --time=04:00:00'
  'expected_head=df9daf62fe4625609b3a71abd7179007cd6970f9'
  'expected_harness_sha=e2a640fd54b9f6cc4c122ab5844507c78de6e26413a74d81d941c88b0eba0422'
  'expected_contract_sha=b7d849fabc0e00e261eab4fa48e45a01061390535e159b533dbd34f18699b822'
  'expected_per_node_bootstrap_sha=192c96fa7f4adb1305f419f1ccb8909fbca1c286ad0f6a229431c231d8e0f60e'
  'source_dir=/home/sna/pr3757-raydiag-df9daf62'
  'harness_root=/home/sna/pr3757-pr11-harness-pernode/experiments/oci_ray_startup_diag_20260822'
  'result_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr3757-reconcile-df9daf62-gate'
  'scratch_root=/raid/scratch/sna/pr3757-raydiag-${SLURM_JOB_ID}'
  'venv_dir=${scratch_root}/venv'
  'ray_harness=${harness_root}/ray_then_pytest.sh'
  'ray_contract=${harness_root}/test_ray_then_pytest_harness.sh'
  'per_node_bootstrap=${harness_root}/pr11_per_node_bootstrap.sh'
  'distributed_harness=${harness_root}/run_tp2_pp2_cp2_refit.py'
  'source_status="$(git -C'
  'submodule_status="$(git -C'
  'stage=per-node-mcore-build'
  'NODE_SCRATCH_ROOT='
  'bash "${per_node_bootstrap}"'
  'pr11-node-bootstrap-${SLURM_JOB_ID}-${node_name}.txt'
  'installed_distributions_sha256'
  'per_node_distribution_sha256='
  'bash "${ray_harness}"'
  'tests/unit/models/policy/test_dflash_worker_validation.py::test_disabled_typed_dflash_config_has_no_refit_metadata'
  'tests/unit/weight_sync/test_nccl_reshard_utils.py::test_build_refit_info_tp2_pp2_cp2_stage_mesh_keeps_both_cp_replicas'
  'tests/unit/models/policy/test_dflash_worker_validation.py'
  'tests/unit/models/megatron/test_draft_refit.py'
  'tests/unit/weight_sync/test_nccl_reshard_utils.py'
  'test "${#pytest_targets[@]}" -eq 5'
  '--nnodes=2'
  '--nproc-per-node=4'
  'run_tp2_pp2_cp2_refit.py'
  'wrapper_sha256='
  'ray_harness_sha256='
  'distributed_harness_sha256='
  'per_node_bootstrap_sha256='
  'exclusions=none'
  'final_stage=${stage}'
)

for contract in "${required_contracts[@]}"; do
  grep -Fq -- "${contract}" "${WRAPPER}"
done

test "$(grep -Fc 'bash "${per_node_bootstrap}"' "${WRAPPER}")" -eq 1
test "$(grep -Fc 'bash "${ray_harness}"' "${WRAPPER}")" -eq 1

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

echo "PR11_DF9DAF62_WRAPPER_CONTRACT_PASS"
