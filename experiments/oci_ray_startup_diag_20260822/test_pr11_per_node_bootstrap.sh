#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP="${SCRIPT_DIR}/pr11_per_node_bootstrap.sh"
LAUNCHER="${SCRIPT_DIR}/pr11_per_node_launcher.sh"
fixture_root="$(mktemp -d)"

cleanup() {
  rm -rf -- "${fixture_root}"
}
trap cleanup EXIT

source_root="${fixture_root}/home/source"
durable_root="${fixture_root}/lustre/results"
fake_bin="${fixture_root}/bin"
shadow_bin="${fixture_root}/shadow-bin"
legacy_node0_scratch="${fixture_root}/legacy/node0/raid/scratch/job"
legacy_node1_scratch="${fixture_root}/legacy/node1/raid/scratch/job"
mkdir -p "${source_root}" "${durable_root}" "${fake_bin}" "${shadow_bin}"

git -C "${source_root}" init -q
git -C "${source_root}" config user.email test@example.com
git -C "${source_root}" config user.name Test
printf '%s\n' 'version = 1' > "${source_root}/uv.lock"
git -C "${source_root}" add uv.lock
git -C "${source_root}" commit -q -m fixture
expected_head="$(git -C "${source_root}" rev-parse HEAD)"
expected_lock_sha="$(shasum -a 256 "${source_root}/uv.lock" | awk '{print $1}')"

cat > "${fake_bin}/uv" <<'FAKE_UV'
#!/usr/bin/env bash
set -euo pipefail
test "$*" = 'sync --locked --extra mcore --group test'
test -n "${UV_PROJECT_ENVIRONMENT:-}"
mkdir -p "${UV_PROJECT_ENVIRONMENT}/bin"
cat > "${UV_PROJECT_ENVIRONMENT}/bin/python" <<'FAKE_PYTHON'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == '-c' ]]; then
  exit 0
fi
printf '%s\n' 'alpha==1.0' 'transformer-engine==2.15.0'
FAKE_PYTHON
chmod +x "${UV_PROJECT_ENVIRONMENT}/bin/python"
FAKE_UV
chmod +x "${fake_bin}/uv"

printf '%s\n' '#!/usr/bin/env bash' 'exit 99' > "${shadow_bin}/env"
chmod 0644 "${shadow_bin}/env"
set +e
"${shadow_bin}/env" >/dev/null 2>&1
shadow_rc=$?
set -e
test "${shadow_rc}" -ne 0

PATH="${fake_bin}:${PATH}" \
UV_PROJECT_ENVIRONMENT="${legacy_node0_scratch}/venv" \
  uv sync --locked --extra mcore --group test

test -x "${legacy_node0_scratch}/venv/bin/python"
if [[ -e "${legacy_node1_scratch}/venv/bin/python" ]]; then
  echo 'fixture must reproduce node0-only venv visibility' >&2
  exit 1
fi

for node_name in node0 node1; do
  node_scratch="${fixture_root}/nodes/${node_name}/raid/scratch/job"
  PATH="${shadow_bin}:${fake_bin}:/usr/bin:/bin" \
  PR11_BOOTSTRAP_ALLOW_TEST_PATHS=1 \
  SLURM_JOB_ID=9001 \
  SLURMD_NODENAME="${node_name}" \
    /bin/bash "${LAUNCHER}" \
      "${source_root}" \
      "${node_scratch}" \
      "${durable_root}" \
      "${expected_head}" \
      "${expected_lock_sha}" \
      "${BOOTSTRAP}"
done

for node_name in node0 node1; do
  receipt="${durable_root}/pr11-node-bootstrap-9001-${node_name}.txt"
  test -f "${receipt}"
  grep -Fxq "node=${node_name}" "${receipt}"
  grep -Fxq "expected_head=${expected_head}" "${receipt}"
  grep -Fxq "actual_head=${expected_head}" "${receipt}"
  grep -Fxq "expected_uv_lock_sha=${expected_lock_sha}" "${receipt}"
  grep -Fxq "actual_uv_lock_sha=${expected_lock_sha}" "${receipt}"
  grep -Fxq 'uv_sync_locked=complete' "${receipt}"
  grep -Fxq 'transformer_engine_import=PASS' "${receipt}"
  grep -Fxq 'installed_distributions_sha256=7940af6c5ab9d2ce8013680c47b90ab2f7919b72b372be3ebe7a084a28c68daf' "${receipt}"
done

cmp \
  "${durable_root}/installed-distributions-9001-node0.txt" \
  "${durable_root}/installed-distributions-9001-node1.txt"

echo PR11_PER_NODE_BOOTSTRAP_CONTRACT_PASS
