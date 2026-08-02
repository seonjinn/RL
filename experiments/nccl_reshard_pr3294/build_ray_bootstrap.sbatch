#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=build_ray_bootstrap

set -euo pipefail

: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

RAY_VERSION=${RAY_VERSION:-2.56.1}
REQUESTS_VERSION=${REQUESTS_VERSION:-2.34.2}
URLLIB3_VERSION=${URLLIB3_VERSION:-2.7.0}
UV_VERSION=${UV_VERSION:-0.11.28}
PYTHON_VERSION=${PYTHON_VERSION:-3.13}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-ray-${RAY_VERSION}-py${PYTHON_VERSION/./}}

mkdir -p "${OUTPUT_DIR}"
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/nrl-ray-bootstrap.XXXXXX")
venv_dir="${work_dir}/${OUTPUT_PREFIX}"
archive="${OUTPUT_DIR}/${OUTPUT_PREFIX}.tar.gz"
metadata="${archive}.metadata.txt"
partial_archive="${archive}.partial.${SLURM_JOB_ID}"

cleanup() {
  rm -rf "${work_dir}"
  rm -f "${partial_archive}"
}
trap cleanup EXIT

command -v uv >/dev/null
uv venv --python "${PYTHON_VERSION}" "${venv_dir}"
uv pip install --python "${venv_dir}/bin/python" \
  "ray[default]==${RAY_VERSION}" \
  "requests==${REQUESTS_VERSION}" \
  "urllib3==${URLLIB3_VERSION}" \
  "uv==${UV_VERSION}"

"${venv_dir}/bin/python" -c \
  "import ray, requests, urllib3; print(ray.__version__, requests.__version__, urllib3.__version__)"
test -x "${venv_dir}/bin/ray"
test -x "${venv_dir}/bin/uv"

tar -czf "${partial_archive}" -C "${work_dir}" "${OUTPUT_PREFIX}"
mv "${partial_archive}" "${archive}"
{
  echo "ray=${RAY_VERSION}"
  echo "requests=${REQUESTS_VERSION}"
  echo "urllib3=${URLLIB3_VERSION}"
  echo "uv=${UV_VERSION}"
  echo "python=${PYTHON_VERSION}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "sha256=$(sha256sum "${archive}" | awk '{print $1}')"
} >"${metadata}"

cleanup
trap - EXIT
ls -lh "${archive}" "${metadata}"
