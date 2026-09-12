#!/usr/bin/env bash
set -euo pipefail

readonly source_root=/home/sna/nemorl-q30-openhands-full-rl-20260912
readonly experiment="${source_root}/experiments/q30_openhands_full_rl_20260912"
readonly container=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260909_7023221.sqsh
readonly durable_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-openhands-full-rl-20260912
readonly account=nemotron_n4_post
readonly mode="${1:-}"
case "${mode}" in
  --render|--test-only|--submit) ;;
  *) echo "usage: $0 --render|--test-only|--submit" >&2; exit 2 ;;
esac
if [[ -n "${SWE_GATE_DEPENDENCY:-}" && ! "${SWE_GATE_DEPENDENCY}" =~ ^[0-9]+$ ]]; then
  echo 'SWE_GATE_DEPENDENCY must be a numeric job ID' >&2
  exit 2
fi
readonly stamp="$(date -u +%Y%m%dT%H%M%SZ)"
readonly artifact_dir="${durable_root}/baseline-3step-${stamp}"

render() {
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=sna.swe2-full-rl-baseline-3step
#SBATCH --account=${account}
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output=${artifact_dir}/slurm-%j.out
#SBATCH --error=${artifact_dir}/slurm-%j.err
set -euo pipefail
export PATH=/cm/local/apps/slurm/25.11/bin:\${PATH}
test -n "\${WANDB_API_KEY:-}"
test -z "\$(git -C ${source_root} status --porcelain=v1 --untracked-files=all)"
git -C ${source_root} rev-parse HEAD >${artifact_dir}/source_sha.txt
git -C ${source_root} submodule status --recursive >${artifact_dir}/submodules.txt
export SWE_SOURCE_ROOT=${source_root}
export SWE_ARTIFACT_DIR=${artifact_dir}
export SWE_NODE_ROOT=/raid/scratch/sna/swe2-full-rl-\${SLURM_JOB_ID}
export UV_CACHE_DIR=\${SWE_NODE_ROOT}/uv-cache
export NEMO_GYM_VENV_DIR=\${SWE_NODE_ROOT}/gym-venvs
export UV_LINK_MODE=hardlink
export TMPDIR=\${SWE_NODE_ROOT}/tmp
export XDG_CACHE_HOME=\${SWE_NODE_ROOT}/cache
export TRITON_CACHE_DIR=\${SWE_NODE_ROOT}/triton
export TORCH_EXTENSIONS_DIR=\${SWE_NODE_ROOT}/torch-extensions
export VLLM_CACHE_ROOT=\${SWE_NODE_ROOT}/vllm-cache
export WANDB_CACHE_DIR=\${SWE_NODE_ROOT}/wandb-cache
export PYTHONPATH=\${SWE_NODE_ROOT}/mcore-overlay:${source_root}/3rdparty/Gym-workspace/Gym:${source_root}:\${PYTHONPATH:-}
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export CONTAINER=${container}
export MOUNTS=/lustre:/lustre,/home:/home,/raid:/raid,\${SWE_NODE_ROOT}/Gym:${source_root}/3rdparty/Gym-workspace/Gym
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export BASE_LOG_DIR=${artifact_dir}
export SETUP_COMMAND='bash ${experiment}/setup_node.sh'
export COMMAND='cd ${source_root} && export UV_CACHE_DIR="\${SWE_NODE_ROOT}/uv-cache" && uv run --no-project --python /opt/nemo_rl_venv/bin/python examples/nemo_gym/prefetch_venvs.py ${experiment}/gate.yaml && uv run --no-project --python /opt/nemo_rl_venv/bin/python examples/nemo_gym/run_grpo_nemo_gym.py --config ${experiment}/gate.yaml'
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash ${experiment}/stage_node.sh
cd ${source_root}
exec bash ray.sub
EOF
}

if [[ "${mode}" == --render ]]; then
  render
  exit 0
fi
test -z "$(git -C "${source_root}" status --porcelain=v1 --untracked-files=all)"
test -r "${container}"
test -f "${experiment}/gate.yaml"
test -n "${WANDB_API_KEY:-}"
export WANDB_API_KEY
mkdir -p "${artifact_dir}"
render >"${artifact_dir}/job.sbatch"
dependency_args=()
if [[ -n "${SWE_GATE_DEPENDENCY:-}" ]]; then
  dependency_args=(--dependency="afterok:${SWE_GATE_DEPENDENCY}" --kill-on-invalid-dep=yes)
fi
sbatch "${dependency_args[@]}" --test-only "${artifact_dir}/job.sbatch" 2>&1 | tee "${artifact_dir}/test-only.txt"
if [[ "${mode}" == --submit ]]; then
  sbatch "${dependency_args[@]}" "${artifact_dir}/job.sbatch" | tee "${artifact_dir}/submission.txt"
fi
