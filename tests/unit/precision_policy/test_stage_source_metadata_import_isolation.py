from __future__ import annotations

import subprocess
import sys


def test_metadata_stager_imports_without_runtime_or_capture_dependencies() -> None:
    code = """
import importlib.abc
import sys

BLOCKED = (
    'nemo_rl',
    'pydantic',
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
    result = subprocess.run(
        (sys.executable, "-c", code),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
