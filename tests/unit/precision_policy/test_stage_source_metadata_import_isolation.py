from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_metadata_stager_imports_without_runtime_or_capture_dependencies() -> None:
    code = """
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])

BLOCKED = (
    'nemo_rl',
    'pydantic',
    'typing_extensions',
    'tools.capture_precision_policy_source_evidence',
)

class BlockRuntimeDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(f'{name}.') for name in BLOCKED):
            raise ImportError(f'{fullname} imports are blocked')
        return None

sys.meta_path.insert(0, BlockRuntimeDependencies())
import tools.stage_precision_policy_source_metadata
"""
    repository_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        (sys.executable, "-S", "-P", "-c", code, str(repository_root)),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
