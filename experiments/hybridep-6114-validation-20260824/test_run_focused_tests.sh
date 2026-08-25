#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_root=$(mktemp -d)
trap 'rm -rf -- "${fixture_root}"' EXIT

mkdir -p "${fixture_root}/bin" "${fixture_root}/venv/bin" "${fixture_root}/project" "${fixture_root}/megatron-lm"

cat >"${fixture_root}/bin/uv" <<'EOF'
#!/bin/bash
echo "Focused test runner must use the selected Python interpreter directly" >&2
exit 20
EOF
chmod +x "${fixture_root}/bin/uv"

cat >"${fixture_root}/venv/bin/python" <<'EOF'
#!/bin/bash
set -euo pipefail
[[ -z ${PYTEST_ADDOPTS:-} ]]
[[ " $* " == *" -p no:pytest-shard "* ]]
if [[ " $* " == *"test_routers.py"* ]]; then
  [[ ${PYTHONPATH%%:*} == "${MEGATRON_LM_ROOT}" ]]
else
  [[ ${PYTHONPATH%%:*} == "${PROJECT_ROOT}" ]]
fi
printf 'python|%s\n' "$*" >>"${FOCUSED_TEST_CALLS}"
EOF
chmod +x "${fixture_root}/venv/bin/python"

export PATH="${fixture_root}/bin:${PATH}"
export PROJECT_ROOT="${fixture_root}/project"
export MEGATRON_LM_ROOT="${fixture_root}/megatron-lm"
export UV_PROJECT_ENVIRONMENT="${fixture_root}/venv"
export FOCUSED_TEST_CALLS="${fixture_root}/calls"
export PYTEST_ADDOPTS="--num-shards=128 --shard-id=127"

bash "${script_dir}/run_focused_tests.sh"

[[ $(grep -c '^python|' "${FOCUSED_TEST_CALLS}") -eq 4 ]]
if grep -q ' -k ' "${FOCUSED_TEST_CALLS}"; then
  echo "Focused test runner must use exact pytest node IDs instead of -k selectors" >&2
  exit 1
fi

echo "focused test runner contract passed"
