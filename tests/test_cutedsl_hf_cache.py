# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
PREPARER = EXPERIMENT_DIR / "prepare_hf_cache.py"
MATRIX_PAYLOAD = EXPERIMENT_DIR / "run_cutedsl_matrix.sbatch"


def _load_preparer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_hf_cache", PREPARER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_snapshot_download(cache_dir: Path, calls: list[dict[str, Any]]):
    revisions = {
        ("Qwen/Qwen3-30B-A3B", None): "a" * 40,
        ("nvidia/OpenMathInstruct-2", "dataset"): "b" * 40,
    }

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        repo_id = kwargs["repo_id"]
        repo_type = kwargs.get("repo_type")
        revision = revisions[(repo_id, repo_type)]
        if kwargs.get("local_files_only"):
            assert kwargs["revision"] == revision
        repo_dir = Path(kwargs["cache_dir"]) / (
            ("datasets--" if repo_type == "dataset" else "models--")
            + repo_id.replace("/", "--")
        )
        snapshot = repo_dir / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text("{}\n")
        return str(snapshot)

    return download


def _fake_load_dataset(calls: list[dict[str, Any]], *, num_rows: int = 17):
    class Dataset:
        def __len__(self) -> int:
            return num_rows

    def load_dataset(path: str, **kwargs: Any) -> Dataset:
        calls.append({"path": path, **kwargs})
        return Dataset()

    return load_dataset


def test_prepare_cache_pins_revisions_and_reuses_completed_manifest(
    tmp_path: Path,
) -> None:
    module = _load_preparer()
    cache_dir = tmp_path / "hf_home"
    shared_manifest = cache_dir / "nemo2606_qwen3_30ba3b_manifest.json"
    calls: list[dict[str, Any]] = []
    download = _fake_snapshot_download(cache_dir, calls)
    dataset_calls: list[dict[str, Any]] = []
    load_dataset = _fake_load_dataset(dataset_calls)

    first = module.prepare_cache(
        cache_dir,
        shared_manifest,
        download,
        load_dataset,
    )
    assert [call.get("local_files_only", False) for call in calls] == [False, False]
    assert first["schema_version"] == 1
    assert first["repositories"] == {
        "model": {
            "repo_id": "Qwen/Qwen3-30B-A3B",
            "repo_type": None,
            "revision": "a" * 40,
            "file_count": 1,
        },
        "dataset": {
            "repo_id": "nvidia/OpenMathInstruct-2",
            "repo_type": "dataset",
            "revision": "b" * 40,
            "file_count": 1,
            "split": "train_1M",
            "num_rows": 17,
        },
    }
    assert dataset_calls == [
        {
            "path": "nvidia/OpenMathInstruct-2",
            "split": "train_1M",
            "revision": "b" * 40,
            "cache_dir": str(cache_dir / "datasets"),
        }
    ]
    assert json.loads(shared_manifest.read_text()) == first

    calls.clear()
    dataset_calls.clear()
    second = module.prepare_cache(
        cache_dir,
        shared_manifest,
        download,
        load_dataset,
    )
    assert second == first
    assert [call.get("local_files_only") for call in calls] == [True, True]
    assert dataset_calls == []


def test_prepare_cache_offline_verifies_processed_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_preparer()
    cache_dir = tmp_path / "hf_home"
    shared_manifest = cache_dir / "nemo2606_qwen3_30ba3b_manifest.json"
    snapshot_calls: list[dict[str, Any]] = []
    download = _fake_snapshot_download(cache_dir, snapshot_calls)
    dataset_calls: list[dict[str, Any]] = []
    load_dataset = _fake_load_dataset(dataset_calls)
    module.prepare_cache(
        cache_dir,
        shared_manifest,
        download,
        load_dataset,
    )

    snapshot_calls.clear()
    dataset_calls.clear()
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    manifest = module.prepare_cache(
        cache_dir,
        shared_manifest,
        download,
        load_dataset,
    )

    assert manifest["repositories"]["dataset"]["num_rows"] == 17
    assert dataset_calls == [
        {
            "path": "nvidia/OpenMathInstruct-2",
            "split": "train_1M",
            "cache_dir": str(cache_dir / "datasets"),
        }
    ]


def test_prepare_cache_rejects_unpinned_snapshot_directory(tmp_path: Path) -> None:
    module = _load_preparer()

    def invalid_download(**_: Any) -> str:
        snapshot = tmp_path / "snapshots" / "main"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}\n")
        return str(snapshot)

    with pytest.raises(ValueError, match="40-character hexadecimal revision"):
        module.prepare_cache(
            tmp_path / "hf_home",
            tmp_path / "manifest.json",
            invalid_download,
            _fake_load_dataset([]),
        )


def test_snapshot_download_honors_rate_limit_retry_after() -> None:
    module = _load_preparer()
    sleeps: list[float] = []

    class Response:
        status_code = 429
        headers = {"retry-after": "7"}

    class RateLimitError(Exception):
        response = Response()

    attempts = 0

    def download(**_: Any) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError
        return "/cache/snapshots/" + "a" * 40

    result = module._snapshot_download_with_retry(
        download,
        {"repo_id": "Qwen/Qwen3-30B-A3B"},
        sleep=sleeps.append,
    )
    assert result.endswith("a" * 40)
    assert attempts == 2
    assert sleeps == [7.0]


def test_matrix_warms_shared_hf_cache_before_forcing_offline_mode() -> None:
    source = MATRIX_PAYLOAD.read_text()
    required = (
        'SHARED_HF_HOME="${CONTAINER_REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g/results/hf_home"',
        'export HF_HOME="${SHARED_HF_HOME}"',
        "export SHARED_HF_MANIFEST",
        "prepare_hf_cache.py",
        '--job-manifest "${CONTAINER_RESULT_DIR}/hf_cache_manifest.json"',
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "export HF_DATASETS_OFFLINE=1",
    )
    for fragment in required:
        assert fragment in source, fragment

    warm = source.index("prepare_hf_cache.py")
    offline = source.index("export HF_HUB_OFFLINE=1")
    grpo = source.index("examples/run_grpo.py", offline)
    assert warm < offline < grpo
