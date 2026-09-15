#!/usr/bin/env bash
set -euo pipefail
driver_python=${1:?validated driver interpreter}
source_root=${2:?checked-out project root}
test -x "${driver_python}"
test -f "${source_root}/pyproject.toml"
# The backend dependencies were already installed from the frozen lock.
uv pip install --python "${driver_python}" --no-deps --editable "${source_root}"
"${driver_python}" - "${source_root}" <<'PY'
from importlib.metadata import requires
from pathlib import Path
import sys
import tomllib

project = tomllib.loads((Path(sys.argv[1]) / "pyproject.toml").read_text())
declared = project["project"]["dependencies"]
installed = requires("nemo-rl") or []

def tq_requirements(requirements: list[str]) -> list[str]:
    return ["".join(req.split()) for req in requirements
            if req.lower().startswith("transferqueue")]

expected = tq_requirements(declared)
actual = tq_requirements(installed)
if len(expected) != 1 or actual != expected:
    raise SystemExit("DRIVER_METADATA_PREFLIGHT=FAIL: TransferQueue pin mismatch")
print("DRIVER_METADATA_PREFLIGHT=PASS", sys.executable)
PY
