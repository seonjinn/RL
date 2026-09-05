from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.precision_policy_source_artifacts import (
    checkpoint_metadata_artifact_identities,
)


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


def test_checkpoint_identity_returns_fresh_nested_wire_values() -> None:
    identity = checkpoint_metadata_artifact_identities()[-1]
    artifact = dict(identity.artifact)
    header_lengths = artifact["mtp_header_byte_lengths"]
    weight_block_size = artifact["weight_block_size"]
    assert isinstance(header_lengths, dict)
    assert isinstance(weight_block_size, list)

    header_lengths["model-00185-of-00213.safetensors"] = 1
    weight_block_size[0] = 1

    fresh_artifact = identity.artifact
    assert fresh_artifact["mtp_header_byte_lengths"] == {
        "model-00185-of-00213.safetensors": 254184,
        "model-00186-of-00213.safetensors": 127080,
    }
    assert fresh_artifact["weight_block_size"] == [128, 128]
