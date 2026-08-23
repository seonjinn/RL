#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=qwen3_30ba3b_dflash_dspark_20step_20260822
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_dflash_dspark_20step_20260822
readonly SOURCE_ROOT=/home/sna/nemorl-pr11-final-df9

usage() {
  echo "usage: $0 --render|--test-only|--submit" >&2
  exit 2
}

render() {
  local root="$1" run artifact script
  run="q30-container-python-diag-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
  artifact="${root}/diagnostics/${run}"
  script="${artifact}/job.sbatch"
  mkdir -p "${artifact}"
  cat >"${script}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=sna-q30-pydiag
#SBATCH --account=nemotron_n4_post
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --output=${artifact}/slurm-%j.out
#SBATCH --error=${artifact}/slurm-%j.err
set -euo pipefail
export CONTAINER="${CONTAINER}"
export MOUNTS="/lustre:/lustre,/home:/home,/raid:/raid"
export SCRATCH="/raid/scratch/\${SLURM_JOB_ID}/${EXPERIMENT}-python-diag"
export TMPDIR="\${SCRATCH}/tmp"
export RAY_TMPDIR="\${SCRATCH}/ray"
export TRITON_CACHE_DIR="\${SCRATCH}/triton"
export UV_CACHE_DIR_OVERRIDE="\${SCRATCH}/uv"
export UV_PROJECT_ENVIRONMENT="\${SCRATCH}/venv"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 bash -c 'mkdir -p "\${SCRATCH}/tmp" "\${SCRATCH}/ray" "\${SCRATCH}/triton" "\${SCRATCH}/uv" "\${SCRATCH}/venv"'
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 --no-container-mount-home --mpi=pmix --container-mounts="\${MOUNTS},\${UV_CACHE_DIR_OVERRIDE}:/root/.cache/uv" --container-image="\${CONTAINER}" --container-workdir="${SOURCE_ROOT}" bash -lc 'python3 - <<PY
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import sys

def package(name):
    spec = importlib.util.find_spec(name)
    result = {
        "spec": None if spec is None else {"origin": spec.origin, "locations": list(spec.submodule_search_locations or [])},
        "distribution": None,
    }
    try:
        dist = importlib.metadata.distribution(name)
        result["distribution"] = {
            "version": dist.version,
            "metadata_name": dist.metadata.get("Name"),
            "root": str(dist.locate_file("")),
            "exceptions_py_exists": (dist.locate_file("urllib3/exceptions.py")).is_file() if name == "urllib3" else None,
        }
    except importlib.metadata.PackageNotFoundError:
        result["distribution"] = "missing"
    return result

result = {
    "executable": sys.executable,
    "version": sys.version,
    "sys_path": sys.path,
    "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "UV_CACHE_DIR", "UV_CACHE_DIR_OVERRIDE", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")},
    "requests": package("requests"),
    "urllib3": package("urllib3"),
}
for name in ("requests", "urllib3"):
    try:
        module = __import__(name)
        result[name]["import_file"] = getattr(module, "__file__", None)
        result[name]["import_version"] = getattr(module, "__version__", None)
    except Exception as error:
        result[name]["import_error"] = f"{type(error).__name__}: {error}"
print(json.dumps(result, sort_keys=True, indent=2))
PY'
SBATCH
  chmod 700 "${script}"
  printf '%s\n' "${script}"
}

case "${1:-}" in
  --render) render "${Q30_20STEP_DIAGNOSTIC_RENDER_ROOT:?Q30_20STEP_DIAGNOSTIC_RENDER_ROOT is required}" ;;
  --test-only) sbatch --test-only "$(render "${DURABLE_ROOT}")" 2>&1 ;;
  --submit) sbatch "$(render "${DURABLE_ROOT}")" ;;
  *) usage ;;
esac
