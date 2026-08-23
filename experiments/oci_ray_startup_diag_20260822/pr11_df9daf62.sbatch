#!/usr/bin/env bash
#SBATCH --job-name=pr3757-raydiag-df9daf62
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr3757-reconcile-df9daf62-gate/slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr3757-reconcile-df9daf62-gate/slurm-%j.err

set -euo pipefail

expected_head=df9daf62fe4625609b3a71abd7179007cd6970f9
expected_harness_sha=e2a640fd54b9f6cc4c122ab5844507c78de6e26413a74d81d941c88b0eba0422
expected_contract_sha=b7d849fabc0e00e261eab4fa48e45a01061390535e159b533dbd34f18699b822
expected_per_node_bootstrap_sha=670920393de28eb76548015c01c2bdb20c5a745ed7979ad5d579758991e8a5d2
expected_per_node_launcher_sha=51e9e0d44a499a457b365d3dbe1bbc5d16dae2edb2049f11db462c001c7a4313
expected_distributed_harness_sha=60ca1766817447ad47747ad972a38cef0bd9db8f8a0e87eb6f03cd312dd0b95c
expected_container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
source_dir=/home/sna/pr3757-raydiag-df9daf62
legacy_source_dir=/home/sna/pr3757-reconcile-df9daf62
harness_root=/home/sna/pr3757-pr11-harness-pernode/experiments/oci_ray_startup_diag_20260822
result_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/pr3757-reconcile-df9daf62-gate
container_path=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
scratch_root=/raid/scratch/sna/pr3757-raydiag-${SLURM_JOB_ID}
venv_dir=${scratch_root}/venv
ray_harness=${harness_root}/ray_then_pytest.sh
ray_contract=${harness_root}/test_ray_then_pytest_harness.sh
per_node_bootstrap=${harness_root}/pr11_per_node_bootstrap.sh
per_node_launcher=${harness_root}/pr11_per_node_launcher.sh
distributed_harness=${harness_root}/run_tp2_pp2_cp2_refit.py
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
      echo "container_sha256=${expected_container_sha}"
      echo "ray_receipt=${result_root}/ray-diagnostic-receipt-${SLURM_JOB_ID}.txt"
      echo "focused=disabled-typed-config,tp2-pp2-cp2-stage-mesh"
      echo "full=policy-worker-validation,draft-refit,nccl-reshard-utils"
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

legacy_head="$(git -C "${legacy_source_dir}" rev-parse HEAD)"
test "${legacy_head}" = "${expected_head}"
legacy_status="$(git -C "${legacy_source_dir}" status --porcelain --untracked-files=all --ignore-submodules=none)"
test -z "${legacy_status}"
legacy_submodule_status="$(git -C "${legacy_source_dir}" submodule status --recursive)"
if printf '%s\n' "${legacy_submodule_status}" | grep -Eq '^[-+U]'; then
  printf '%s\n' "${legacy_submodule_status}" >&2
  exit 12
fi

expected_uv_lock_sha="$(sha256sum "${source_dir}/uv.lock" | awk '{print $1}')"
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
)
test "${#pytest_targets[@]}" -eq 5

export SOURCE_ROOT="${source_dir}"
export EXPECTED_HEAD="${expected_head}"
export EXPECTED_UV_LOCK_SHA="${expected_uv_lock_sha}"
export DURABLE_RESULT_ROOT="${result_root}"
bash "${ray_harness}" "${pytest_targets[@]}"
INNER

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
