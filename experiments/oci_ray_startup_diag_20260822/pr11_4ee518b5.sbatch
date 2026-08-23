#!/usr/bin/env bash
#SBATCH --job-name=pr3757-final-4ee518b5
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr3757-final-4ee518b5-gate/slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr3757-final-4ee518b5-gate/slurm-%j.err

set -euo pipefail

expected_head=4ee518b5dc2ed16f75e31876b477ea5ecf7d8c9b
expected_harness_sha=ec5efaeced419d426d4b4454714c3714da37eca4098cadcf20897ac82596340d
expected_contract_sha=f28b0a3b99c28b2b76cbd2a88a22a644fd576ce2beb555fa2823e3905e3c4639
expected_per_node_bootstrap_sha=670920393de28eb76548015c01c2bdb20c5a745ed7979ad5d579758991e8a5d2
expected_per_node_launcher_sha=51e9e0d44a499a457b365d3dbe1bbc5d16dae2edb2049f11db462c001c7a4313
expected_distributed_harness_sha=2fc298a9ba248648db362ded7ebb052e490606010568878955bbb5cc7b5f684c
expected_container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
source_dir=/home/sna/pr3757-final-4ee518b5
harness_root=/home/sna/pr3757-pr11-harness-pernode/experiments/oci_ray_startup_diag_20260822
result_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr3757-final-4ee518b5-gate
container_path=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
scratch_root=/raid/scratch/sna/pr3757-final-${SLURM_JOB_ID}
venv_dir=${scratch_root}/venv
ray_harness=${harness_root}/ray_then_pytest.sh
ray_contract=${harness_root}/test_ray_then_pytest_harness.sh
per_node_bootstrap=${harness_root}/pr11_per_node_bootstrap.sh
per_node_launcher=${harness_root}/pr11_per_node_launcher.sh
distributed_harness=${harness_root}/run_tp2_pp2_cp2_refit.py
q30_runner=${source_dir}/scripts/run_focused_qwen_moe_draft_tests.sh
result_path=""
live_log=""
stage=init

finish() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ -n "${result_path}" ]]; then
    {
      echo "final_stage=${stage}"
      echo "result=$([[ ${rc} -eq 0 ]] && echo PASS || echo FAIL)"
      echo "exit_code=${rc}"
      echo "job_id=${SLURM_JOB_ID}"
      echo "expected_head=${expected_head}"
      echo "actual_head=${actual_head:-unset}"
      echo "expected_uv_lock_sha=${expected_uv_lock_sha:-unset}"
      echo "source_root=${source_dir}"
      echo "venv=${venv_dir}"
      echo "wrapper=${BASH_SOURCE[0]}"
      echo "wrapper_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
      echo "ray_harness_sha256=${expected_harness_sha}"
      echo "ray_contract_sha256=${expected_contract_sha}"
      echo "per_node_bootstrap_sha256=${expected_per_node_bootstrap_sha}"
      echo "per_node_launcher_sha256=${expected_per_node_launcher_sha}"
      echo "distributed_harness_sha256=${expected_distributed_harness_sha}"
      echo "q30_runner_sha256=${q30_runner_sha256:-unset}"
      echo "container_sha256=${expected_container_sha}"
      echo "ray_receipt=${result_root}/ray-diagnostic-receipt-${SLURM_JOB_ID}.txt"
      echo "focused=legacy-pr11,packed-failure,real-quant-lifecycle,lower-fix-core,projected-bin-bounds,sync-count-regression"
      echo "q30_runner=focused-qwen-moe-draft-tests"
      echo "distributed_topology=TP2xPP2xCP2"
      echo "distributed_cases=3"
      echo "per_node_distribution_sha256=${per_node_distribution_sha256:-unset}"
      echo "exclusions=none"
    } > "${result_path}"
  fi
  exit "${rc}"
}
trap finish EXIT

test -d "${result_root}"
result_root="$(realpath "${result_root}")"
case "${result_root}" in
  /lustre/*) ;;
  *) echo "result root must resolve below /lustre" >&2; exit 2 ;;
esac
result_path=${result_root}/pr11-result-${SLURM_JOB_ID}.txt
live_log=${result_root}/pr11-live-${SLURM_JOB_ID}.log
exec > >(tee "${live_log}") 2>&1

stage=source-provenance
actual_head="$(git -C "${source_dir}" rev-parse HEAD)"
test "${actual_head}" = "${expected_head}"
source_status="$(git -C "${source_dir}" status --porcelain --untracked-files=all --ignore-submodules=none)"
test -z "${source_status}"
submodule_status="$(git -C "${source_dir}" submodule status --recursive)"
if printf '%s\n' "${submodule_status}" | grep -Eq '^[-+U]'; then
  printf '%s\n' "${submodule_status}" >&2
  exit 11
fi

expected_uv_lock_sha="$(sha256sum "${source_dir}/uv.lock" | awk '{print $1}')"
q30_runner_sha256="$(sha256sum "${q30_runner}" | awk '{print $1}')"
test "$(sha256sum "${ray_harness}" | awk '{print $1}')" = "${expected_harness_sha}"
test "$(sha256sum "${ray_contract}" | awk '{print $1}')" = "${expected_contract_sha}"
test "$(sha256sum "${per_node_bootstrap}" | awk '{print $1}')" = \
  "${expected_per_node_bootstrap_sha}"
test "$(sha256sum "${per_node_launcher}" | awk '{print $1}')" = \
  "${expected_per_node_launcher_sha}"
test "$(sha256sum "${distributed_harness}" | awk '{print $1}')" = \
  "${expected_distributed_harness_sha}"
test "$(sha256sum "${container_path}" | awk '{print $1}')" = \
  "${expected_container_sha}"

container_args=(
  "--container-image=${container_path}"
  "--container-mounts=/home:/home,/raid/scratch:/raid/scratch,/lustre:/lustre"
  "--mpi=pmix"
)

stage=per-node-mcore-build
srun \
  --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  "${container_args[@]}" \
  /bin/bash "${per_node_launcher}" \
    "${source_dir}" \
    "${scratch_root}" \
    "${result_root}" \
    "${expected_head}" \
    "${expected_uv_lock_sha}" \
    "${per_node_bootstrap}"

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
test "${#allocated_nodes[@]}" -eq 2
per_node_distribution_sha256=""
for node_name in "${allocated_nodes[@]}"; do
  receipt="${result_root}/pr11-node-bootstrap-${SLURM_JOB_ID}-${node_name}.txt"
  test -f "${receipt}"
  grep -Fxq "node=${node_name}" "${receipt}"
  grep -Fxq "expected_head=${expected_head}" "${receipt}"
  grep -Fxq "actual_head=${expected_head}" "${receipt}"
  grep -Fxq "expected_uv_lock_sha=${expected_uv_lock_sha}" "${receipt}"
  grep -Fxq "actual_uv_lock_sha=${expected_uv_lock_sha}" "${receipt}"
  grep -Fxq 'uv_sync_locked=complete' "${receipt}"
  grep -Fxq 'transformer_engine_import=PASS' "${receipt}"
  grep -Fxq 'result=PASS' "${receipt}"
  node_distribution_sha="$(awk -F= '$1 == "installed_distributions_sha256" {print $2}' "${receipt}")"
  test "${#node_distribution_sha}" -eq 64
  if [[ -z "${per_node_distribution_sha256}" ]]; then
    per_node_distribution_sha256="${node_distribution_sha}"
  else
    test "${node_distribution_sha}" = "${per_node_distribution_sha256}"
  fi
done
printf '%s\n' \
  "nodes=${allocated_nodes[*]}" \
  "expected_head=${expected_head}" \
  "expected_uv_lock_sha=${expected_uv_lock_sha}" \
  "per_node_distribution_sha256=${per_node_distribution_sha256}" \
  'result=PASS' \
  > "${result_root}/pr11-per-node-bootstrap-summary-${SLURM_JOB_ID}.txt"

stage=mcore-pytest
srun \
  --nodes=1 \
  --ntasks=1 \
  "${container_args[@]}" \
  /bin/bash -s -- \
  "${source_dir}" \
  "${result_root}" \
  "${scratch_root}" \
  "${venv_dir}" \
  "${expected_head}" \
  "${expected_uv_lock_sha}" \
  "${ray_harness}" <<'INNER'
set -euo pipefail

source_dir=$1
result_root=$2
scratch_root=$3
venv_dir=$4
expected_head=$5
expected_uv_lock_sha=$6
ray_harness=$7

cd "${source_dir}"
export UV_PROJECT_ENVIRONMENT="${venv_dir}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export PYTHONPATH="${source_dir}:${source_dir}/3rdparty/Megatron-Bridge/src:${source_dir}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
cudnn_lib="${UV_PROJECT_ENVIRONMENT}/lib/python3.13/site-packages/nvidia/cudnn/lib"
test -e "${cudnn_lib}/libcudnn.so.9"
export CUDNN_HOME="${cudnn_lib}"
export CUDNN_PATH="${cudnn_lib}"
export LD_LIBRARY_PATH="${cudnn_lib}:${LD_LIBRARY_PATH:-}"
"${UV_PROJECT_ENVIRONMENT}/bin/python" -c \
  'import torch; import transformer_engine; import megatron.core'

pytest_targets=(
  tests/unit/models/policy/test_dflash_worker_validation.py::test_disabled_typed_dflash_config_has_no_refit_metadata
  tests/unit/weight_sync/test_nccl_reshard_utils.py::test_build_refit_info_tp2_pp2_cp2_stage_mesh_keeps_both_cp_replicas
  tests/unit/models/policy/test_dflash_worker_validation.py
  tests/unit/models/megatron/test_draft_refit.py
  tests/unit/weight_sync/test_nccl_reshard_utils.py
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
)
test "${#pytest_targets[@]}" -eq 18

export SOURCE_ROOT="${source_dir}"
export EXPECTED_HEAD="${expected_head}"
export EXPECTED_UV_LOCK_SHA="${expected_uv_lock_sha}"
export DURABLE_RESULT_ROOT="${result_root}"
bash "${ray_harness}" "${pytest_targets[@]}"
INNER

stage=q30-focused-runner
q30_extra_targets=(
  tests/unit/models/megatron/test_draft_optimizer.py::test_draft_weight_decay_override_keeps_norm_and_bias_at_zero_decay
  tests/unit/models/megatron/test_dspark_heads.py::test_tp2_markov_head_sums_replicated_w1_gradients
  tests/unit/models/megatron/test_dspark_heads.py::test_tp2_confidence_path_sanitizes_ids_without_tp_gradient_summing
  tests/unit/models/megatron/test_dspark_heads.py::test_tp2_markov_head_megatron_checkpoint_round_trip
)
test "${#q30_extra_targets[@]}" -eq 4
srun \
  --nodes=1 \
  --ntasks=1 \
  "${container_args[@]}" \
  /bin/bash -s -- \
  "${source_dir}" \
  "${venv_dir}" \
  "${q30_runner}" \
  "${q30_extra_targets[@]}" <<'Q30_INNER'
set -euo pipefail

source_dir=$1
venv_dir=$2
q30_runner=$3
shift 3
q30_extra_targets=("$@")

cd "${source_dir}"
export UV_PROJECT_ENVIRONMENT="${venv_dir}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export PYTHONPATH="${source_dir}:${source_dir}/3rdparty/Megatron-Bridge/src:${source_dir}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
cudnn_lib="${venv_dir}/lib/python3.13/site-packages/nvidia/cudnn/lib"
test -e "${cudnn_lib}/libcudnn.so.9"
export CUDNN_HOME="${cudnn_lib}"
export CUDNN_PATH="${cudnn_lib}"
export LD_LIBRARY_PATH="${cudnn_lib}:${LD_LIBRARY_PATH:-}"
/bin/bash "${q30_runner}" "${q30_extra_targets[@]}"
Q30_INNER

stage=tp2-pp2-cp2
master_addr="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
master_port=$((20000 + SLURM_JOB_ID % 20000))
srun \
  --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  "${container_args[@]}" \
  /bin/bash -lc "
set -euo pipefail
source_dir='${source_dir}'
venv_dir='${venv_dir}'
distributed_harness='${distributed_harness}'
cd \"\${source_dir}\"
export SOURCE_ROOT=\"\${source_dir}\"
export EXPECTED_HEAD='${expected_head}'
export PATH=\"\${venv_dir}/bin:\${PATH}\"
export PYTHONPATH=\"\${source_dir}:\${source_dir}/3rdparty/Megatron-Bridge/src:\${source_dir}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:\${PYTHONPATH:-}\"
cudnn_lib=\"\${venv_dir}/lib/python3.13/site-packages/nvidia/cudnn/lib\"
test -e \"\${cudnn_lib}/libcudnn.so.9\"
export CUDNN_HOME=\"\${cudnn_lib}\"
export CUDNN_PATH=\"\${cudnn_lib}\"
export LD_LIBRARY_PATH=\"\${cudnn_lib}:\${LD_LIBRARY_PATH:-}\"
\"\${venv_dir}/bin/python\" -m torch.distributed.run \\
  --nnodes=2 \\
  --nproc-per-node=4 \\
  --node-rank=\"\${SLURM_NODEID}\" \\
  --master-addr='${master_addr}' \\
  --master-port='${master_port}' \\
  \"\${distributed_harness}\"
"

stage=complete
