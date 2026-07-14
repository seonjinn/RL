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
"""Shared provenance construction for precomputed SFT validation events."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from nemo_rl.algorithms.sft_validation_artifact import ValidationArtifactFingerprint

if TYPE_CHECKING:
    from nemo_rl.algorithms.sft import MasterConfig


_PACKED_DATASET_NAME = "megatron_sft_packed"


def content_sha256(path: Path) -> str:
    """Hash one file or a directory tree using stable relative-path ordering."""
    if path.is_symlink():
        raise ValueError(f"Validation provenance rejects symbolic links: {path}")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise ValueError(f"Validation provenance path does not exist: {path}")

    digest = hashlib.sha256(b"validation-directory-sha256-v1\x00")
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Validation provenance directory is empty: {path}")
    for candidate in files:
        if candidate.is_symlink():
            raise ValueError(
                f"Validation provenance rejects symbolic links: {candidate}"
            )
        relative_path = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, byteorder="big"))
        digest.update(relative_path)
        digest.update(bytes.fromhex(content_sha256(candidate)))
    return digest.hexdigest()


def verify_content_sha256(path: Path, expected_sha256: str, *, label: str) -> str:
    """Hash an actual provenance input and reject a mismatched claimed digest."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"Expected {label} SHA-256 must have 64 characters")
    actual_sha256 = content_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Actual {label} SHA-256 does not match the expected value: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def derive_preprocessing_sha256(
    config: MasterConfig,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Hash the canonical artifact-relevant subset of a resolved SFT config."""
    megatron_config = config.policy.get("megatron_cfg")
    if not isinstance(megatron_config, Mapping):
        raise ValueError(
            "Validation artifact preprocessing provenance requires policy.megatron_cfg"
        )
    provenance = {
        "data": {
            "train": config.data.get("train"),
            "validation": config.data.get("validation"),
            "default": config.data.get("default"),
            "max_input_seq_length": config.data.get("max_input_seq_length"),
            "add_bos": config.data.get("add_bos"),
            "add_eos": config.data.get("add_eos"),
            "add_generation_prompt": config.data.get("add_generation_prompt"),
            "shuffle": config.data.get("shuffle"),
        },
        "policy": {
            "tokenizer": config.policy.get("tokenizer"),
            "max_total_sequence_length": config.policy.get("max_total_sequence_length"),
            "sequence_packing": config.policy.get("sequence_packing"),
            "dynamic_batching": config.policy.get("dynamic_batching"),
            "make_sequence_length_divisible_by": config.policy.get(
                "make_sequence_length_divisible_by"
            ),
            "megatron_cfg": {
                "enabled": megatron_config.get("enabled"),
                "tensor_model_parallel_size": megatron_config.get(
                    "tensor_model_parallel_size"
                ),
                "context_parallel_size": megatron_config.get("context_parallel_size"),
                "prepacked_sft_loss_mode": megatron_config.get(
                    "prepacked_sft_loss_mode"
                ),
            },
        },
        "sft": {
            "val_batches": config.sft.val_batches,
            "val_global_batch_size": config.sft.val_global_batch_size,
            "val_micro_batch_size": config.sft.val_micro_batch_size,
        },
    }
    canonical = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    preprocessing_sha256 = hashlib.sha256(canonical).hexdigest()
    if expected_sha256 is not None and expected_sha256 != preprocessing_sha256:
        raise ValueError(
            "Validation artifact expected preprocessing SHA-256 does not match "
            f"resolved config: expected {expected_sha256}, got {preprocessing_sha256}"
        )
    return preprocessing_sha256


def validate_validation_source_config(config: MasterConfig) -> None:
    """Reject validation rows that a single external dataset digest cannot prove."""
    _validation_dataset_configs(config.data)
    default_config = config.data.get("default")
    if default_config is not None and not isinstance(default_config, Mapping):
        raise ValueError("Validation artifact data.default must be a mapping")

    for train_index, train_config in enumerate(_train_dataset_configs(config.data)):
        if "split_validation_size" in train_config:
            split_validation_size = train_config["split_validation_size"]
        elif (
            isinstance(default_config, Mapping)
            and "split_validation_size" in default_config
        ):
            split_validation_size = default_config["split_validation_size"]
        else:
            if train_config.get("dataset_name") == _PACKED_DATASET_NAME:
                continue
            raise ValueError(
                "Validation artifact production cannot prove train split is "
                f"disabled for data.train[{train_index}]; set "
                "split_validation_size=0 explicitly"
            )
        if (
            not isinstance(split_validation_size, (int, float))
            or isinstance(split_validation_size, bool)
            or not math.isfinite(float(split_validation_size))
            or split_validation_size != 0
        ):
            raise ValueError(
                "Validation artifact production does not support train-derived "
                "validation; "
                f"data.train[{train_index}].split_validation_size must be absent or 0"
            )


def build_validation_artifact_fingerprint(
    *,
    dataset_sha256: str,
    tokenizer_sha256: str,
    preprocessing_sha256: str,
    container_sha256: str,
    repository_root: Path,
) -> ValidationArtifactFingerprint:
    """Build an artifact fingerprint from explicit inputs and checked-out source."""
    submodule_commits = _submodule_commits(repository_root)
    _require_clean_repository_tree(repository_root, submodule_commits)
    return ValidationArtifactFingerprint(
        dataset_sha256=dataset_sha256,
        tokenizer_sha256=tokenizer_sha256,
        preprocessing_sha256=preprocessing_sha256,
        nemo_rl_commit=_git_output(repository_root, "rev-parse", "HEAD"),
        submodule_commits=submodule_commits,
        container_sha256=container_sha256,
    )


def _validation_dataset_configs(
    data_config: Mapping[str, object],
) -> list[Mapping[str, object]]:
    validation = data_config.get("validation")
    if isinstance(validation, Mapping):
        return [validation]
    if (
        isinstance(validation, list)
        and validation
        and all(isinstance(item, Mapping) for item in validation)
    ):
        return validation
    raise ValueError(
        "Validation artifact production requires an explicit configured validation "
        "dataset"
    )


def _train_dataset_configs(
    data_config: Mapping[str, object],
) -> list[Mapping[str, object]]:
    train = data_config.get("train")
    if isinstance(train, Mapping):
        return [train]
    if (
        isinstance(train, list)
        and train
        and all(isinstance(item, Mapping) for item in train)
    ):
        return train
    raise ValueError("Validation artifact production requires configured train data")


def _git_output(repository_root: Path, *args: str) -> str:
    return _git_stdout(repository_root, *args).strip()


def _git_stdout(repository_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Could not read Git metadata: {' '.join(args)}") from error


def _submodule_commits(repository_root: Path) -> tuple[tuple[str, str], ...]:
    commits: list[tuple[str, str]] = []
    for line in _git_stdout(
        repository_root, "submodule", "status", "--recursive"
    ).splitlines():
        if not line or line[0] != " ":
            raise RuntimeError(
                "Validation artifact production requires clean submodules"
            )
        fields = line[1:].split(maxsplit=2)
        if len(fields) < 2:
            raise RuntimeError("Could not parse Git submodule status")
        commits.append((fields[1], fields[0]))
    if not commits:
        raise RuntimeError(
            "Validation artifact production requires initialized submodules"
        )
    return tuple(sorted(commits))


def _require_clean_repository_tree(
    repository_root: Path,
    submodule_commits: tuple[tuple[str, str], ...],
) -> None:
    repositories = ((".", repository_root),) + tuple(
        (path, repository_root / path) for path, _ in submodule_commits
    )
    for display_path, repository in repositories:
        status = _git_output(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=all",
        )
        if status:
            raise RuntimeError(
                "Validation artifact production requires a clean repository and "
                f"submodules; {display_path!r} has tracked or untracked changes"
            )
