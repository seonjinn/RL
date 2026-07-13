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
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
PREPARER = EXPERIMENT_DIR / "prepare_hf_cache.py"
MATRIX_PAYLOAD = EXPERIMENT_DIR / "run_cutedsl_matrix.sbatch"
PROFILE_LOADER = EXPERIMENT_DIR / "lib/cluster_profile.sh"


def _shared_hf_preflight_source() -> str:
    source = MATRIX_PAYLOAD.read_text()
    return source.split("# CUTEDSL_SHARED_HF_PREFLIGHT_START\n", 1)[1].split(
        "# CUTEDSL_SHARED_HF_PREFLIGHT_END\n", 1
    )[0]


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
            assert kwargs["revision"] in ("main", revision)
        repo_dir = Path(kwargs["cache_dir"]) / (
            ("datasets--" if repo_type == "dataset" else "models--")
            + repo_id.replace("/", "--")
        )
        snapshot = repo_dir / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text("{}\n")
        return str(snapshot)

    return download


def _fake_load_dataset(
    calls: list[dict[str, Any]], *, num_rows: int = 1_000_000
):
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
            "num_rows": 1_000_000,
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

    assert manifest["repositories"]["dataset"]["num_rows"] == 1_000_000
    assert dataset_calls == [
        {
            "path": "nvidia/OpenMathInstruct-2",
            "split": "train_1M",
            "cache_dir": str(cache_dir / "datasets"),
        }
    ]


def test_offline_main_does_not_require_writing_shared_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_preparer()
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    shared_manifest = hf_home / "nemo2606_qwen3_30ba3b_manifest.json"
    shared_manifest.write_text('{"schema_version": 1, "repositories": {}}\n')
    job_manifest = tmp_path / "result/hf_cache_manifest.json"
    manifest = {"schema_version": 1, "repositories": {}}
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            hf_home=hf_home,
            shared_manifest=shared_manifest,
            job_manifest=job_manifest,
        ),
    )
    monkeypatch.setattr(module, "prepare_cache", lambda *_: manifest)
    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = object()  # type: ignore[attr-defined]
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    real_open = Path.open

    def reject_shared_lock(path: Path, *args: Any, **kwargs: Any):
        if path.name == ".nemo2606-cache.lock":
            raise AssertionError("offline verification must not write a shared lock")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_shared_lock)

    assert module.main() == 0
    assert json.loads(job_manifest.read_text()) == manifest


def test_offline_missing_manifest_bootstraps_from_local_cache_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_preparer()
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    shared_manifest = hf_home / "nemo2606_qwen3_30ba3b_manifest.json"
    snapshot_calls: list[dict[str, Any]] = []
    dataset_calls: list[dict[str, Any]] = []
    replacements: list[tuple[Path, Path]] = []
    real_replace = module.os.replace

    def observe_atomic_replace(source: Path, destination: Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert destination_path == shared_manifest
        assert not destination_path.exists()
        value = json.loads(source_path.read_text())
        assert value["repositories"]["dataset"]["num_rows"] == 1_000_000
        replacements.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr(module.os, "replace", observe_atomic_replace)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    manifest = module.prepare_cache(
        hf_home,
        shared_manifest,
        _fake_snapshot_download(hf_home, snapshot_calls),
        _fake_load_dataset(dataset_calls),
    )

    assert [call["revision"] for call in snapshot_calls] == ["main", "main"]
    assert [call["local_files_only"] for call in snapshot_calls] == [True, True]
    assert dataset_calls == [
        {
            "path": "nvidia/OpenMathInstruct-2",
            "split": "train_1M",
            "cache_dir": str(hf_home / "datasets"),
        }
    ]
    assert json.loads(shared_manifest.read_text()) == manifest
    assert len(replacements) == 1
    assert not replacements[0][0].exists()


@pytest.mark.parametrize("failure", ("missing_snapshot", "wrong_row_count"))
def test_offline_bootstrap_rejects_incomplete_local_cache_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    module = _load_preparer()
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    shared_manifest = hf_home / "nemo2606_qwen3_30ba3b_manifest.json"
    snapshot_calls: list[dict[str, Any]] = []
    dataset_calls: list[dict[str, Any]] = []
    download = _fake_snapshot_download(hf_home, snapshot_calls)
    if failure == "missing_snapshot":

        def missing_download(**kwargs: Any) -> str:
            snapshot_calls.append(kwargs)
            raise FileNotFoundError("cached snapshot is incomplete")

        download = missing_download
    load_dataset = _fake_load_dataset(
        dataset_calls,
        num_rows=17 if failure == "wrong_row_count" else 1_000_000,
    )
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    with pytest.raises((FileNotFoundError, ValueError)):
        module.prepare_cache(
            hf_home,
            shared_manifest,
            download,
            load_dataset,
        )

    assert snapshot_calls
    assert all(call["local_files_only"] is True for call in snapshot_calls)
    assert not shared_manifest.exists()
    assert not list(hf_home.glob(".*.tmp"))


def test_offline_missing_manifest_rechecks_after_lock_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_preparer()
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    shared_manifest = hf_home / "nemo2606_qwen3_30ba3b_manifest.json"
    job_manifest = tmp_path / "result/hf_cache_manifest.json"
    snapshot_calls: list[dict[str, Any]] = []
    dataset_calls: list[dict[str, Any]] = []
    download = _fake_snapshot_download(hf_home, snapshot_calls)
    load_dataset = _fake_load_dataset(dataset_calls)
    completed = {
        "schema_version": 1,
        "repositories": {
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
                "num_rows": 1_000_000,
            },
        },
    }
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            hf_home=hf_home,
            shared_manifest=shared_manifest,
            job_manifest=job_manifest,
        ),
    )
    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = load_dataset  # type: ignore[attr-defined]
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    lock_calls = 0

    def racing_flock(_: int, __: int) -> None:
        nonlocal lock_calls
        lock_calls += 1
        shared_manifest.write_text(json.dumps(completed) + "\n")

    monkeypatch.setattr(module.fcntl, "flock", racing_flock)

    assert module.main() == 0
    assert lock_calls == 1
    assert json.loads(shared_manifest.read_text()) == completed
    assert json.loads(job_manifest.read_text()) == completed
    assert [call["revision"] for call in snapshot_calls] == ["a" * 40, "b" * 40]
    assert all(call["local_files_only"] is True for call in snapshot_calls)


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


@pytest.mark.parametrize(
    ("profile_name", "expected_hf_home"),
    (
        (
            "pre_tyche",
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home",
        ),
        (
            "lyris",
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home",
        ),
        (
            "aws_dfw",
            "/lustre/fsw/portfolios/nemotron/projects/"
            "nemotron_sw_post/users/sna/hf_home",
        ),
    ),
)
def test_cluster_profile_exports_absolute_shared_hf_home(
    profile_name: str,
    expected_hf_home: str,
) -> None:
    command = f"""
set -euo pipefail
source {PROFILE_LOADER!s}
export CUTEDSL_CLUSTER_PROFILE={profile_name}
load_cutedsl_cluster_profile
env | grep '^CUTEDSL_SHARED_HF_HOME='
"""
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"CUTEDSL_SHARED_HF_HOME={expected_hf_home}"
    loader = PROFILE_LOADER.read_text()
    assert '[[ "${CUTEDSL_SHARED_HF_HOME-}" != /* ]]' in loader


@pytest.mark.parametrize("case", ("relative", "missing_root"))
def test_matrix_shared_hf_preflight_fails_closed(
    tmp_path: Path,
    case: str,
) -> None:
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    manifest = hf_home / "nemo2606_qwen3_30ba3b_manifest.json"
    manifest.write_text('{"schema_version": 1, "repositories": {}}\n')
    configured_home = str(hf_home)
    if case == "relative":
        configured_home = "relative/hf_home"
    elif case == "missing_root":
        configured_home = str(tmp_path / "absent")

    env = os.environ.copy()
    env["CUTEDSL_SHARED_HF_HOME"] = configured_home
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + _shared_hf_preflight_source()],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "shared Hugging Face cache" in result.stderr
    assert configured_home not in result.stderr


@pytest.mark.parametrize("manifest_exists", (False, True))
def test_matrix_shared_hf_preflight_accepts_existing_cache_root(
    tmp_path: Path,
    manifest_exists: bool,
) -> None:
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    manifest = hf_home / "nemo2606_qwen3_30ba3b_manifest.json"
    if manifest_exists:
        manifest.write_text('{"schema_version": 1, "repositories": {}}\n')
    env = os.environ.copy()
    env["CUTEDSL_SHARED_HF_HOME"] = str(hf_home)

    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            + _shared_hf_preflight_source()
            + '\nprintf "%s\\n" "$HF_HOME" "$SHARED_HF_MANIFEST"\n',
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(hf_home), str(manifest)]


def test_matrix_verifies_profile_cache_offline_before_grpo() -> None:
    source = MATRIX_PAYLOAD.read_text()
    required = (
        'SHARED_HF_HOME="${CUTEDSL_SHARED_HF_HOME}"',
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
    assert 'assert repository["num_rows"] == 1_000_000' in source

    assert "results/hf_home" not in source
    verify = source.index("prepare_hf_cache.py")
    offline = source.index("export HF_HUB_OFFLINE=1")
    image_python = source.index('command -v python3 >/dev/null')
    image_bootstrap = source.index(
        'python3 \\\n    "${CONTAINER_REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g/prepare_hf_cache.py"'
    )
    uv_install = source.index('curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh"')
    uv_sync = source.index('"${UV_BIN}" sync --locked')
    grpo = source.index("examples/run_grpo.py", verify)
    assert offline < image_python < image_bootstrap < uv_install < uv_sync < grpo

    manifest_block = source.split("manifest = {", 1)[1].split("}\n(", 1)[0]
    assert "SHARED_HF_HOME" not in manifest_block
    assert "HF_HOME" not in manifest_block
