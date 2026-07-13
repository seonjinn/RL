#!/usr/bin/env python3
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

"""Prepare immutable Hugging Face inputs for the CuTeDSL performance matrix."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MODEL_REPO_ID = "Qwen/Qwen3-30B-A3B"
DATASET_REPO_ID = "nvidia/OpenMathInstruct-2"
DATASET_SPLIT = "train_1M"
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
REPOSITORIES = (
    ("model", MODEL_REPO_ID, None),
    ("dataset", DATASET_REPO_ID, "dataset"),
)

SnapshotDownload = Callable[..., str]
DatasetLoad = Callable[..., Any]
Sleep = Callable[[float], None]


def _rate_limit_response(error: BaseException) -> Any | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 429:
            return response
        current = current.__cause__ or current.__context__
    return None


def _call_with_rate_limit_retry(
    operation: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    label: str,
    sleep: Sleep = time.sleep,
    max_attempts: int = 6,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return operation(**kwargs)
        except Exception as error:
            response = _rate_limit_response(error)
            if response is None or attempt == max_attempts:
                raise
            headers = getattr(response, "headers", {})
            retry_after = headers.get("retry-after", headers.get("Retry-After", "60"))
            try:
                delay = max(1.0, min(float(retry_after), 300.0))
            except (TypeError, ValueError):
                delay = 60.0
            print(
                f"Hugging Face rate limit while caching {label}; "
                f"retrying in {delay:.0f}s ({attempt}/{max_attempts})"
            )
            sleep(delay)
    raise AssertionError("unreachable")


def _snapshot_download_with_retry(
    snapshot_download: SnapshotDownload,
    kwargs: dict[str, Any],
    *,
    sleep: Sleep = time.sleep,
    max_attempts: int = 6,
) -> str:
    return _call_with_rate_limit_retry(
        snapshot_download,
        kwargs,
        label=kwargs["repo_id"],
        sleep=sleep,
        max_attempts=max_attempts,
    )


def _snapshot_file_count(snapshot: Path) -> int:
    count = sum(1 for path in snapshot.rglob("*") if path.is_file())
    if count == 0:
        raise ValueError(f"Hugging Face snapshot contains no files: {snapshot}")
    return count


def _validate_revision(snapshot: Path) -> str:
    revision = snapshot.name
    if REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(
            "Hugging Face snapshot path must end in a 40-character hexadecimal "
            f"revision: {snapshot}"
        )
    return revision


def _download_repository(
    hf_home: Path,
    repo_id: str,
    repo_type: str | None,
    snapshot_download: SnapshotDownload,
    *,
    revision: str,
    local_files_only: bool,
) -> tuple[Path, str, int]:
    snapshot = Path(
        _snapshot_download_with_retry(
            snapshot_download,
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "cache_dir": str(hf_home / "hub"),
                "local_files_only": local_files_only,
            },
        )
    ).resolve()
    resolved_revision = _validate_revision(snapshot)
    if local_files_only and resolved_revision != revision:
        raise ValueError(
            f"Cached revision for {repo_id} changed: {resolved_revision} != {revision}"
        )
    return snapshot, resolved_revision, _snapshot_file_count(snapshot)


def _read_completed_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"Invalid shared Hugging Face cache manifest: {path}")
    repositories = value.get("repositories")
    if not isinstance(repositories, dict):
        raise ValueError(f"Invalid repository map in Hugging Face manifest: {path}")
    return value


def _offline_mode_enabled() -> bool:
    return any(
        os.environ.get(name) == "1"
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    )


def _load_required_dataset(
    hf_home: Path,
    load_dataset: DatasetLoad,
    *,
    revision: str | None,
) -> int:
    kwargs: dict[str, Any] = {
        "path": DATASET_REPO_ID,
        "split": DATASET_SPLIT,
        "cache_dir": str(hf_home / "datasets"),
    }
    if revision is not None:
        kwargs["revision"] = revision
    dataset = _call_with_rate_limit_retry(
        load_dataset,
        kwargs,
        label=f"{DATASET_REPO_ID}:{DATASET_SPLIT}",
    )
    num_rows = len(dataset)
    if not isinstance(num_rows, int) or isinstance(num_rows, bool) or num_rows <= 0:
        raise ValueError(
            f"Materialized dataset {DATASET_REPO_ID}:{DATASET_SPLIT} is empty"
        )
    return num_rows


def prepare_cache(
    hf_home: Path,
    shared_manifest: Path,
    snapshot_download: SnapshotDownload,
    load_dataset: DatasetLoad,
) -> dict[str, Any]:
    """Populate or verify the shared model and dataset snapshots."""
    hf_home.mkdir(parents=True, exist_ok=True)
    completed = _read_completed_manifest(shared_manifest)
    offline = _offline_mode_enabled()
    if offline and completed is None:
        raise ValueError(
            "Offline Hugging Face verification requires a completed shared manifest"
        )
    repositories: dict[str, dict[str, Any]] = {}
    for label, repo_id, repo_type in REPOSITORIES:
        existing = completed.get("repositories", {}).get(label) if completed else None
        if completed:
            if not isinstance(existing, dict):
                raise ValueError(f"Missing {label} repository in {shared_manifest}")
            if (
                existing.get("repo_id") != repo_id
                or existing.get("repo_type") != repo_type
            ):
                raise ValueError(f"Repository identity changed for {label}")
            revision = existing.get("revision")
            if (
                not isinstance(revision, str)
                or REVISION_PATTERN.fullmatch(revision) is None
            ):
                raise ValueError(f"Invalid cached revision for {label}")
            _, resolved_revision, file_count = _download_repository(
                hf_home,
                repo_id,
                repo_type,
                snapshot_download,
                revision=revision,
                local_files_only=True,
            )
            if file_count != existing.get("file_count"):
                raise ValueError(f"Cached file count changed for {label}")
        else:
            _, resolved_revision, file_count = _download_repository(
                hf_home,
                repo_id,
                repo_type,
                snapshot_download,
                revision="main",
                local_files_only=False,
            )
        repository = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "revision": resolved_revision,
            "file_count": file_count,
        }
        if label == "dataset":
            if completed:
                if existing.get("split") != DATASET_SPLIT:
                    raise ValueError("Cached dataset split changed")
                expected_num_rows = existing.get("num_rows")
                if (
                    not isinstance(expected_num_rows, int)
                    or isinstance(expected_num_rows, bool)
                    or expected_num_rows <= 0
                ):
                    raise ValueError("Invalid cached dataset row count")
                if offline:
                    num_rows = _load_required_dataset(
                        hf_home,
                        load_dataset,
                        revision=None,
                    )
                    if num_rows != expected_num_rows:
                        raise ValueError("Cached dataset row count changed")
                else:
                    num_rows = expected_num_rows
            else:
                num_rows = _load_required_dataset(
                    hf_home,
                    load_dataset,
                    revision=resolved_revision,
                )
            repository.update({"split": DATASET_SPLIT, "num_rows": num_rows})
        repositories[label] = repository

    manifest = {"schema_version": 1, "repositories": repositories}
    if completed is not None and manifest != completed:
        raise ValueError(
            "Shared Hugging Face cache manifest changed during verification"
        )
    if completed is None:
        temporary = shared_manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, shared_manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--shared-manifest", type=Path, required=True)
    parser.add_argument("--job-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.hf_home.mkdir(parents=True, exist_ok=True)
    lock_path = args.hf_home / ".nemo2606-cache.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        from datasets import load_dataset
        from huggingface_hub import snapshot_download

        manifest = prepare_cache(
            args.hf_home,
            args.shared_manifest,
            snapshot_download,
            load_dataset,
        )
    args.job_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.job_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
