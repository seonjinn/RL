import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).parents[1]


def _manifest() -> ModuleType:
    path = ROOT / "dataset_manifest.py"
    spec = importlib.util.spec_from_file_location("dataset_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_hashes_every_cache_file_deterministically(tmp_path: Path) -> None:
    dataset = tmp_path / "cache"
    (dataset / "nested").mkdir(parents=True)
    (dataset / "a.json").write_text("alpha")
    (dataset / "nested" / "b.parquet").write_text("beta")
    manifest = _manifest().build_manifest(dataset)

    assert [entry["path"] for entry in manifest["files"]] == [
        "a.json",
        "nested/b.parquet",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert len(manifest["tree_sha256"]) == 64


def test_pair_verification_rejects_any_cache_drift(tmp_path: Path) -> None:
    module = _manifest()
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    payload = {
        "schema_version": 1,
        "dataset_revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
        "files": [{"path": "data.parquet", "sha256": "a" * 64, "size": 1}],
        "tree_sha256": "b" * 64,
    }
    left.write_text(json.dumps(payload))
    right.write_text(json.dumps({**payload, "tree_sha256": "c" * 64}))

    with pytest.raises(ValueError, match="cache manifest mismatch"):
        module.verify_pair(left, right)


def test_source_parquet_sha_is_required(tmp_path: Path) -> None:
    module = _manifest()
    dataset = tmp_path / "cache"
    dataset.mkdir()
    (dataset / "data.parquet").write_text("wrong")

    with pytest.raises(ValueError, match="source parquet SHA256"):
        module.build_manifest(dataset, expected_source_parquet_sha256="f" * 64)
