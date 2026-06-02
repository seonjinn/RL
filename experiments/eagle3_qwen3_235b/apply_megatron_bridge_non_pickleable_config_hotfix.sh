#!/usr/bin/env bash
set -euo pipefail

# Apply the Megatron-Bridge config sanitizer hotfix needed after 235B job 3116849
# failed in qkv_config broadcast with:
#   TypeError: cannot pickle 'torch._C._distributed_c10d.ProcessGroup' object
#
# Run from the NeMo-RL checkout root, for example:
#   cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL
#   bash /path/to/apply_megatron_bridge_non_pickleable_config_hotfix.sh

ROOT="${1:-$(pwd)}"
TARGET="${ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/models/conversion/utils.py"

if [[ ! -f "${TARGET}" ]]; then
  echo "ERROR: target file not found: ${TARGET}" >&2
  exit 2
fi

if grep -q "_is_torch_process_group" "${TARGET}"; then
  echo "Hotfix already present: ${TARGET}"
  exit 0
fi

python3 - "${TARGET}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

replacements = [
    (
        "import functools\nimport re\n",
        "import functools\nimport pickle\nimport re\n",
    ),
    (
        "    raise ValueError(f\"Parameter '{param_name}' not found in model at VP stage {vp_stage}\")\n\n\n"
        "def remove_non_pickleables(obj, max_depth: int = 3, current_depth: int = 0):\n",
        "    raise ValueError(f\"Parameter '{param_name}' not found in model at VP stage {vp_stage}\")\n\n\n"
        "def _is_torch_process_group(obj):\n"
        "    cls = type(obj)\n"
        "    return cls.__name__ == \"ProcessGroup\" and \"distributed\" in cls.__module__\n\n\n"
        "def _is_pickleable_leaf(obj):\n"
        "    try:\n"
        "        pickle.dumps(obj)\n"
        "    except Exception:\n"
        "        return False\n"
        "    return True\n\n\n"
        "def remove_non_pickleables(obj, max_depth: int = 3, current_depth: int = 0):\n",
    ),
    (
        "    # Stop recursion if max depth reached\n"
        "    if current_depth >= max_depth:\n"
        "        return obj\n",
        "    if _is_torch_process_group(obj):\n"
        "        return None\n\n"
        "    # Stop recursion if max depth reached\n"
        "    if current_depth >= max_depth:\n"
        "        return obj if _is_pickleable_leaf(obj) else None\n",
    ),
    (
        "    # For primitive types and other safe objects, return as-is\n"
        "    return obj\n",
        "    # For primitive types and other safe objects, return as-is. Drop any\n"
        "    # remaining non-pickleable leaves before broadcast_object_list.\n"
        "    return obj if _is_pickleable_leaf(obj) else None\n",
    ),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(f"Expected patch context not found in {path}: {old[:80]!r}")
    text = text.replace(old, new, 1)

path.write_text(text)
print(f"Applied hotfix to {path}")
PY

grep -n "_is_torch_process_group\|_is_pickleable_leaf\|pickle" "${TARGET}"
