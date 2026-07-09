# pyright: reportMissingImports=false

import dataclasses
import hashlib
import json
import random
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from examples import run_sft
import examples.prepare_sft_validation_event as producer_module
from examples.prepare_sft_validation_event import (
    build_validation_artifact_fingerprint,
    build_precomputed_validation_event,
    derive_validation_artifact_eligibility,
    derive_preprocessing_sha256,
    digest_validation_event_data,
    load_master_config,
    main as producer_main,
    validate_validation_source_config,
)
import nemo_rl.algorithms.sft as run_sft_sft
import nemo_rl.algorithms.sft_validation_artifact as artifact_module
from nemo_rl.algorithms.sft_validation_artifact import (
    MemoryBudget,
    PrecomputedValidationEvent,
    ValidationArtifactEligibility,
    ValidationArtifactFingerprint,
    clone_validation_event_data,
    load_validation_event,
    save_validation_event,
    tensor_content_sha256,
)
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.data.megatron_sft_packed import megatron_sft_packed_preprocessor


_SUPER_V3_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "configs"
    / "sft_superv3_prepacked.yaml"
)


def test_preprocessing_digest_changes_for_relevant_hydra_override() -> None:
    baseline = load_master_config(_SUPER_V3_CONFIG, [])
    changed = load_master_config(
        _SUPER_V3_CONFIG,
        ["data.max_input_seq_length=131072"],
    )

    assert derive_preprocessing_sha256(baseline) != derive_preprocessing_sha256(changed)


def test_preprocessing_digest_changes_for_train_split_config() -> None:
    baseline = load_master_config(_SUPER_V3_CONFIG, [])
    changed = load_master_config(
        _SUPER_V3_CONFIG,
        ["++data.train.split_validation_size=0"],
    )

    assert derive_preprocessing_sha256(baseline) != derive_preprocessing_sha256(changed)


def test_preprocessing_digest_ignores_logger_run_name_override() -> None:
    baseline = load_master_config(_SUPER_V3_CONFIG, [])
    changed = load_master_config(
        _SUPER_V3_CONFIG,
        ["logger.wandb.name=artifact-provenance-test"],
    )

    assert derive_preprocessing_sha256(baseline) == derive_preprocessing_sha256(changed)


def test_preprocessing_digest_rejects_mismatched_expected_value() -> None:
    config = load_master_config(_SUPER_V3_CONFIG, [])

    with pytest.raises(ValueError, match="expected preprocessing SHA-256"):
        derive_preprocessing_sha256(config, expected_sha256="0" * 64)


def test_cli_rejects_preprocessing_mismatch_before_data_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_sft_validation_event.py",
            "--config",
            str(_SUPER_V3_CONFIG),
            "--artifact-dir",
            str(tmp_path / "artifact"),
            "--dataset-sha256",
            "a" * 64,
            "--tokenizer-sha256",
            "b" * 64,
            "--container-sha256",
            "f" * 64,
            "--preprocessing-sha256",
            "0" * 64,
        ],
    )

    with (
        patch.object(producer_module, "get_tokenizer") as tokenizer_loader,
        patch.object(producer_module, "save_validation_event") as publisher,
        pytest.raises(ValueError, match="expected preprocessing SHA-256"),
    ):
        producer_main()

    tokenizer_loader.assert_not_called()
    publisher.assert_not_called()


def test_cli_rejects_train_derived_validation_before_data_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_sft_validation_event.py",
            "--config",
            str(_SUPER_V3_CONFIG),
            "--artifact-dir",
            str(tmp_path / "artifact"),
            "--dataset-sha256",
            "a" * 64,
            "--tokenizer-sha256",
            "b" * 64,
            "--container-sha256",
            "f" * 64,
            "++data.train.split_validation_size=0.1",
        ],
    )

    with (
        patch.object(producer_module, "get_tokenizer") as tokenizer_loader,
        patch.object(producer_module, "setup_data") as data_loader,
        patch.object(producer_module, "save_validation_event") as publisher,
        pytest.raises(ValueError, match="train-derived validation"),
    ):
        producer_main()

    tokenizer_loader.assert_not_called()
    data_loader.assert_not_called()
    publisher.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        ["++data.train.split_validation_size=0"],
    ],
)
def test_super_explicit_validation_accepts_absent_or_zero_train_split(
    overrides: list[str],
) -> None:
    config = load_master_config(_SUPER_V3_CONFIG, overrides)

    validate_validation_source_config(config)


def test_validation_source_requires_explicit_validation_config() -> None:
    config = load_master_config(_SUPER_V3_CONFIG, ["data.validation=null"])

    with pytest.raises(ValueError, match="explicit configured validation dataset"):
        validate_validation_source_config(config)


def test_validation_source_rejects_train_split_in_default_config() -> None:
    config = load_master_config(
        _SUPER_V3_CONFIG,
        ["++data.default.split_validation_size=0.1"],
    )

    with pytest.raises(ValueError, match="train-derived validation"):
        validate_validation_source_config(config)


def test_validation_source_rejects_unproven_train_dataset_default_split() -> None:
    config = load_master_config(
        _SUPER_V3_CONFIG,
        ["data.train.dataset_name=ResponseDataset"],
    )

    with pytest.raises(ValueError, match="cannot prove train split is disabled"):
        validate_validation_source_config(config)


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")


def _initialize_git_repository(repository: Path, tracked_file: str) -> None:
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "artifact-test@example.com")
    _git(repository, "config", "user.name", "Artifact Test")
    (repository / tracked_file).write_text(f"clean {tracked_file}\n")
    _git(repository, "add", tracked_file)
    _git(repository, "commit", "-m", f"add {tracked_file}")


def _recursive_git_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    leaf = tmp_path / "leaf"
    _initialize_git_repository(leaf, "leaf.txt")

    child = tmp_path / "child"
    _initialize_git_repository(child, "child.txt")
    _git(
        child,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(leaf),
        "modules/leaf",
    )
    _git(child, "commit", "-am", "add leaf submodule")

    root = tmp_path / "root"
    _initialize_git_repository(root, "root.txt")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "modules/child",
    )
    _git(root, "commit", "-am", "add child submodule")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
    )
    return root, root / "modules" / "child", root / "modules" / "child/modules/leaf"


def _source_fingerprint(repository: Path) -> ValidationArtifactFingerprint:
    return build_validation_artifact_fingerprint(
        dataset_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        preprocessing_sha256="c" * 64,
        container_sha256="d" * 64,
        repository_root=repository,
    )


def test_source_fingerprint_accepts_clean_root_and_recursive_submodules(
    tmp_path: Path,
) -> None:
    root, child, leaf = _recursive_git_fixture(tmp_path)
    for repository in (root, child, leaf):
        assert (
            _git(
                repository,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=all",
            )
            == ""
        )
    recursive_status = _git(root, "submodule", "status", "--recursive")
    status_lines = recursive_status.splitlines()
    assert [line[:1] for line in status_lines] == [" ", " "], repr(recursive_status)

    fingerprint = _source_fingerprint(root)

    assert fingerprint.nemo_rl_commit == _git(root, "rev-parse", "HEAD")
    assert [path for path, _ in fingerprint.submodule_commits] == [
        "modules/child",
        "modules/child/modules/leaf",
    ]


@pytest.mark.parametrize(
    "dirty_case",
    [
        "tracked-root",
        "untracked-root",
        "tracked-submodule",
        "untracked-submodule",
    ],
)
def test_source_fingerprint_rejects_dirty_repository_tree(
    tmp_path: Path,
    dirty_case: str,
) -> None:
    root, child, leaf = _recursive_git_fixture(tmp_path)
    if dirty_case == "tracked-root":
        (root / "root.txt").write_text("dirty root\n")
    elif dirty_case == "untracked-root":
        (root / "untracked.txt").write_text("untracked root\n")
    elif dirty_case == "tracked-submodule":
        (child / "child.txt").write_text("dirty child\n")
    else:
        (leaf / "untracked.txt").write_text("untracked leaf\n")

    with pytest.raises(RuntimeError, match="clean repository and submodules"):
        _source_fingerprint(root)


class _ResponseDatasetFixture:
    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        self.task_spec = None
        self.dataset = [{"task_name": task_name}]
        self.preprocessor = None

    def processor(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {}


def _data_config_fixture() -> DataConfig:
    return cast(
        DataConfig,
        {
            "train": {"dataset_name": "train"},
            "validation": {"dataset_name": "validation"},
            "add_bos": False,
            "add_eos": True,
            "add_generation_prompt": False,
            "max_input_seq_length": 16,
        },
    )


def test_setup_data_skips_configured_validation_when_requested() -> None:
    loaded_config_names: list[str] = []

    def load_dataset(config: dict[str, str]) -> _ResponseDatasetFixture:
        dataset_name = config["dataset_name"]
        loaded_config_names.append(dataset_name)
        return _ResponseDatasetFixture(dataset_name)

    with patch.object(run_sft, "load_response_dataset", side_effect=load_dataset):
        dataset, val_dataset = run_sft.setup_data(
            tokenizer=object(),
            data_config=_data_config_fixture(),
            load_validation=False,
        )

    assert dataset is not None
    assert val_dataset is None
    assert loaded_config_names == ["train"]


def test_setup_data_loads_configured_validation_by_default() -> None:
    loaded_config_names: list[str] = []

    def load_dataset(config: dict[str, str]) -> _ResponseDatasetFixture:
        dataset_name = config["dataset_name"]
        loaded_config_names.append(dataset_name)
        return _ResponseDatasetFixture(dataset_name)

    with patch.object(run_sft, "load_response_dataset", side_effect=load_dataset):
        dataset, val_dataset = run_sft.setup_data(
            tokenizer=object(),
            data_config=_data_config_fixture(),
        )

    assert dataset is not None
    assert val_dataset is not None
    assert loaded_config_names == ["train", "validation"]


def _runtime_main_config(*, precomputed: bool) -> run_sft_sft.MasterConfig:
    config = load_master_config(_SUPER_V3_CONFIG, [])
    if precomputed:
        config.sft.validation_input_mode = "precomputed_event"
        config.sft.validation_execution_mode = "event_batch"
        config.sft.validation_event_cache_mode = "off"
        config.sft.validation_precomputed_manifest = "/tmp/validation.manifest.json"
        config.sft.validation_precomputed_dataset_sha256 = "a" * 64
        config.sft.validation_precomputed_tokenizer_sha256 = "b" * 64
        config.sft.validation_precomputed_container_sha256 = "c" * 64
    return config


def _setup_result(master_config: run_sft_sft.MasterConfig) -> tuple[object, ...]:
    return (
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        master_config,
    )


def test_run_sft_main_artifact_failure_precedes_all_runtime_side_effects() -> None:
    config = _runtime_main_config(precomputed=True)
    expected_fingerprint = _fingerprint()

    with (
        patch.object(
            run_sft,
            "parse_args",
            return_value=(SimpleNamespace(config="runtime.yaml"), []),
        ),
        patch.object(run_sft, "register_omegaconf_resolvers"),
        patch.object(run_sft, "load_config", return_value=object()),
        patch.object(
            run_sft.OmegaConf,
            "to_container",
            return_value=config.model_dump(),
        ),
        patch.object(run_sft, "validate_validation_source_config"),
        patch.object(
            run_sft,
            "derive_preprocessing_sha256",
            return_value="d" * 64,
        ),
        patch.object(
            run_sft,
            "build_validation_artifact_fingerprint",
            return_value=expected_fingerprint,
        ),
        patch.object(
            run_sft,
            "load_validation_event",
            side_effect=ValueError("fingerprint mismatch"),
        ) as artifact_loader,
        patch.object(run_sft, "init_ray") as ray_initializer,
        patch.object(run_sft, "get_tokenizer") as tokenizer_loader,
        patch.object(run_sft, "setup_data") as data_loader,
        patch.object(run_sft, "setup") as setup_runtime,
        patch.object(run_sft, "sft_train") as train_runtime,
        pytest.raises(ValueError, match="fingerprint mismatch"),
    ):
        run_sft.main()

    artifact_loader.assert_called_once_with(
        Path(config.sft.validation_precomputed_manifest),
        expected_fingerprint,
    )
    ray_initializer.assert_not_called()
    tokenizer_loader.assert_not_called()
    data_loader.assert_not_called()
    setup_runtime.assert_not_called()
    train_runtime.assert_not_called()


def test_run_sft_main_precomputed_event_loads_once_and_is_forwarded() -> None:
    config = _runtime_main_config(precomputed=True)
    expected_fingerprint = _fingerprint()
    event = _event_fixture()
    tokenizer = object()
    train_dataset = object()
    setup_result = _setup_result(config)
    call_order: list[str] = []

    def load_event(*_args: object, **_kwargs: object) -> PrecomputedValidationEvent:
        call_order.append("load_validation_event")
        return event

    def initialize_ray() -> None:
        call_order.append("init_ray")

    def load_tokenizer(*_args: object, **_kwargs: object) -> object:
        call_order.append("get_tokenizer")
        return tokenizer

    def load_data(*_args: object, **_kwargs: object) -> tuple[object, None]:
        call_order.append("setup_data")
        return train_dataset, None

    def setup_training(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        call_order.append("setup")
        return setup_result

    with (
        patch.object(
            run_sft,
            "parse_args",
            return_value=(SimpleNamespace(config="runtime.yaml"), []),
        ),
        patch.object(run_sft, "register_omegaconf_resolvers"),
        patch.object(run_sft, "load_config", return_value=object()),
        patch.object(
            run_sft.OmegaConf,
            "to_container",
            return_value=config.model_dump(),
        ),
        patch.object(run_sft, "validate_validation_source_config"),
        patch.object(
            run_sft,
            "derive_preprocessing_sha256",
            return_value="d" * 64,
        ),
        patch.object(
            run_sft,
            "build_validation_artifact_fingerprint",
            return_value=expected_fingerprint,
        ),
        patch.object(
            run_sft,
            "load_validation_event",
            side_effect=load_event,
        ) as artifact_loader,
        patch.object(run_sft, "get_next_experiment_dir", return_value="/tmp/logs"),
        patch.object(run_sft, "init_ray", side_effect=initialize_ray),
        patch.object(
            run_sft,
            "get_tokenizer",
            side_effect=load_tokenizer,
        ),
        patch.object(run_sft, "setup_data", side_effect=load_data) as data_loader,
        patch.object(run_sft, "setup", side_effect=setup_training) as setup_runtime,
        patch.object(run_sft, "sft_train") as train_runtime,
    ):
        run_sft.main()

    assert call_order == [
        "load_validation_event",
        "init_ray",
        "get_tokenizer",
        "setup_data",
        "setup",
    ]
    artifact_loader.assert_called_once_with(
        Path(config.sft.validation_precomputed_manifest),
        expected_fingerprint,
    )
    called_config = setup_runtime.call_args.args[0]
    data_loader.assert_called_once_with(
        tokenizer,
        called_config.data,
        load_validation=False,
    )
    setup_runtime.assert_called_once_with(
        called_config,
        tokenizer,
        train_dataset,
        None,
    )
    assert set(train_runtime.call_args.kwargs) == {"precomputed_validation_event"}
    assert train_runtime.call_args.kwargs["precomputed_validation_event"] is event


def test_run_sft_main_default_mode_keeps_validation_data_loading() -> None:
    config = _runtime_main_config(precomputed=False)
    tokenizer = object()
    train_dataset = object()
    val_dataset = object()
    setup_result = _setup_result(config)

    with (
        patch.object(
            run_sft,
            "parse_args",
            return_value=(SimpleNamespace(config="runtime.yaml"), []),
        ),
        patch.object(run_sft, "register_omegaconf_resolvers"),
        patch.object(run_sft, "load_config", return_value=object()),
        patch.object(
            run_sft.OmegaConf,
            "to_container",
            return_value=config.model_dump(),
        ),
        patch.object(run_sft, "load_validation_event") as artifact_loader,
        patch.object(run_sft, "get_next_experiment_dir", return_value="/tmp/logs"),
        patch.object(run_sft, "init_ray"),
        patch.object(run_sft, "get_tokenizer", return_value=tokenizer),
        patch.object(
            run_sft,
            "setup_data",
            return_value=(train_dataset, val_dataset),
        ) as data_loader,
        patch.object(run_sft, "setup", return_value=setup_result) as setup_runtime,
        patch.object(run_sft, "sft_train") as train_runtime,
    ):
        run_sft.main()

    artifact_loader.assert_not_called()
    called_config = setup_runtime.call_args.args[0]
    data_loader.assert_called_once_with(tokenizer, called_config.data)
    setup_runtime.assert_called_once_with(
        called_config,
        tokenizer,
        train_dataset,
        val_dataset,
    )
    assert train_runtime.call_args.kwargs == {"precomputed_validation_event": None}


def _packed_validation_batch(batch_index: int) -> BatchedDataDict:
    row_ids = torch.arange(batch_index * 64, (batch_index + 1) * 64)
    input_ids = torch.stack((row_ids, row_ids + 1000), dim=1)
    return BatchedDataDict(
        input_ids=input_ids,
        target_ids=input_ids + 1,
        token_mask=torch.ones((64, 2), dtype=torch.float32),
        position_ids=torch.tensor([0, 1], dtype=torch.int64).repeat(64, 1),
        input_lengths=torch.full((64,), 2, dtype=torch.int64),
        processed_token_counts=torch.full((64,), 2, dtype=torch.int64),
        sample_mask=torch.ones(64, dtype=torch.float32),
        packed_cu_seqlens=torch.tensor([[0, 2]], dtype=torch.int32).repeat(64, 1),
        packed_cu_seqlens_lengths=torch.full((64,), 2, dtype=torch.int64),
        packed_max_seqlens=torch.full((64,), 2, dtype=torch.int64),
        idx=row_ids.tolist(),
        task_name=["megatron_sft_packed"] * 64,
    )


class _PackedTokenizerFixture:
    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = None

    def apply_chat_template(
        self, messages: list[dict[str, str]], **_kwargs: object
    ) -> list[int]:
        token_ids: list[int] = []
        for message in messages:
            content = message["content"].strip()
            if content:
                token_ids.extend(int(piece) for piece in content.split())
        return token_ids

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<unk>"
        return 0


def _write_packed_validation_rows(path: Path, row_count: int) -> None:
    with path.open("w") as file_handle:
        for row_index in range(row_count):
            batch_index = row_index // 64
            assistant_tokens = " ".join(
                str(1000 + batch_index * 10 + offset)
                for offset in range(batch_index + 1)
            )
            record = {
                "messages": [
                    {"role": "system", "content": str(row_index + 10)},
                    {"role": "assistant", "content": assistant_tokens},
                ]
            }
            file_handle.write(json.dumps(record) + "\n")


def _real_producer_config(data_path: Path) -> run_sft_sft.MasterConfig:
    data_config = {
        "max_input_seq_length": 8,
        "add_bos": False,
        "add_eos": True,
        "add_generation_prompt": False,
        "shuffle": False,
        "num_workers": 0,
        "train": {
            "dataset_name": "megatron_sft_packed",
            "data_path": str(data_path),
            "chat_key": "messages",
        },
        "validation": {
            "dataset_name": "megatron_sft_packed",
            "data_path": str(data_path),
            "chat_key": "messages",
        },
        "default": {
            "prompt_file": None,
            "system_prompt_file": None,
            "megatron_sft_prompt_format": "identity",
            "megatron_sft_pad_token": "<unk>",
            "megatron_sft_assistant_prefix_len": 0,
            "megatron_sft_context_parallel_size": 1,
        },
    }
    return run_sft_sft.MasterConfig.model_construct(
        data=data_config,
        policy={
            "tokenizer": {
                "name": "fixture",
                "chat_template": None,
                "chat_template_kwargs": None,
            },
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {
                "enabled": True,
                "train_mb_tokens": 8,
                "algorithm": "modified_first_fit_decreasing",
                "sequence_length_round": 1,
            },
            "max_total_sequence_length": 8,
            "make_sequence_length_divisible_by": 1,
            "megatron_cfg": {
                "enabled": True,
                "tensor_model_parallel_size": 1,
                "context_parallel_size": 1,
                "prepacked_sft_loss_mode": "labels",
            },
        },
        sft=run_sft_sft.SFTConfig(
            val_batches=4,
            val_global_batch_size=64,
            val_micro_batch_size=1,
        ),
        logger={},
        cluster={},
        checkpointing={},
    )


def _real_validation_dataset(
    config: run_sft_sft.MasterConfig,
    tokenizer: _PackedTokenizerFixture,
) -> AllTaskProcessedDataset:
    _, val_dataset = run_sft.setup_data(tokenizer, config.data)
    assert val_dataset is not None
    return val_dataset


def _producer_config_fixture() -> run_sft_sft.MasterConfig:
    return run_sft_sft.MasterConfig.model_construct(
        data={
            "train": {"dataset_name": "megatron_sft_packed"},
            "validation": {"dataset_name": "megatron_sft_packed"},
            "shuffle": False,
            "num_workers": 0,
        },
        policy={
            "dynamic_batching": {"enabled": False},
            "megatron_cfg": {"enabled": True, "prepacked_sft_loss_mode": "labels"},
            "sequence_packing": {"enabled": True},
        },
        sft=SimpleNamespace(
            val_batches=4,
            val_global_batch_size=64,
            val_micro_batch_size=1,
        ),
    )


def _producer_dataset_fixture(
    batches: list[BatchedDataDict],
) -> AllTaskProcessedDataset:
    return cast(
        AllTaskProcessedDataset,
        SimpleNamespace(
            batches=batches,
            task_data_processors={
                "megatron_sft_packed": (
                    None,
                    partial(
                        megatron_sft_packed_preprocessor,
                        prompt_format="identity",
                    ),
                )
            },
            task_data_preprocessors={},
        ),
    )


class _RngConsumingFixtureDataLoader:
    def __init__(self, dataset: Any, **kwargs: object) -> None:
        assert kwargs["batch_size"] == 64
        assert kwargs["shuffle"] is False
        assert kwargs["drop_last"] is True
        self._batches: list[BatchedDataDict] = dataset.batches

    def __iter__(self) -> Iterator[BatchedDataDict]:
        random.random()
        np.random.random()
        torch.rand(1)
        return iter(self._batches)


def _producer_event(
    batches: list[BatchedDataDict],
) -> PrecomputedValidationEvent:
    with patch(
        "examples.prepare_sft_validation_event.StatefulDataLoader",
        _RngConsumingFixtureDataLoader,
    ):
        return build_precomputed_validation_event(
            _producer_config_fixture(),
            SimpleNamespace(pad_token_id=0),
            _producer_dataset_fixture(batches),
        )


def test_producer_unit_combination_matches_runtime_helper() -> None:
    batches = [_packed_validation_batch(batch_index) for batch_index in range(4)]
    expected_token_counts = (128, 128, 128, 128)
    live_batches = list(batches)

    produced = _producer_event(batches)
    live = run_sft_sft._combine_validation_event_batches(
        live_batches,
        global_batch_size=64,
        pad_token_id=0,
    )

    assert produced.num_valid_tokens == expected_token_counts
    assert produced.payload_digest == digest_validation_event_data(live)
    assert set(produced.data) == set(live)
    for key, value in live.items():
        if torch.is_tensor(value):
            assert torch.equal(produced.data[key], value)
        else:
            assert produced.data[key] == value
    assert produced.data["input_ids"][:, 0].tolist() == list(range(256))


def test_producer_real_loader_preserves_rows_and_distinct_batch_token_counts(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "validation.jsonl.packed"
    _write_packed_validation_rows(data_path, 256)
    tokenizer = _PackedTokenizerFixture()
    config = _real_producer_config(data_path)
    val_dataset = _real_validation_dataset(config, tokenizer)

    produced = build_precomputed_validation_event(config, tokenizer, val_dataset)

    assert produced.num_valid_tokens == (64, 128, 192, 256)
    assert produced.data["input_ids"][:, 0].tolist() == list(range(10, 266))


@pytest.mark.parametrize("row_count", [192, 255])
def test_producer_real_loader_rejects_fewer_than_four_complete_batches(
    tmp_path: Path,
    row_count: int,
) -> None:
    data_path = tmp_path / "validation.jsonl.packed"
    _write_packed_validation_rows(data_path, row_count)
    tokenizer = _PackedTokenizerFixture()
    config = _real_producer_config(data_path)
    val_dataset = _real_validation_dataset(config, tokenizer)

    with pytest.raises(ValueError, match="four complete validation batches"):
        build_precomputed_validation_event(config, tokenizer, val_dataset)


def _super_v3_integration_overrides(data_path: Path) -> list[str]:
    return [
        f"data.train.data_path={data_path}",
        f"data.validation.data_path={data_path}",
        "data.max_input_seq_length=8",
        "data.num_workers=0",
        "policy.max_total_sequence_length=8",
        "policy.megatron_cfg.tensor_model_parallel_size=1",
        "policy.megatron_cfg.context_parallel_size=1",
        "sft.val_batches=4",
        "sft.val_global_batch_size=64",
        "sft.val_micro_batch_size=1",
    ]


def test_producer_cli_wires_resolved_config_into_real_data_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "validation.jsonl.packed"
    artifact_directory = tmp_path / "artifact"
    _write_packed_validation_rows(data_path, 256)
    overrides = _super_v3_integration_overrides(data_path)
    resolved_config = load_master_config(_SUPER_V3_CONFIG, overrides)
    expected_preprocessing_sha256 = derive_preprocessing_sha256(resolved_config)
    captured_fingerprint: ValidationArtifactFingerprint | None = None

    def capture_fingerprint(**kwargs: Any) -> ValidationArtifactFingerprint:
        nonlocal captured_fingerprint
        captured_fingerprint = dataclasses.replace(
            _fingerprint(),
            dataset_sha256=kwargs["dataset_sha256"],
            tokenizer_sha256=kwargs["tokenizer_sha256"],
            preprocessing_sha256=kwargs["preprocessing_sha256"],
            container_sha256=kwargs["container_sha256"],
        )
        return captured_fingerprint

    monkeypatch.setattr(
        producer_module,
        "get_tokenizer",
        lambda _config: _PackedTokenizerFixture(),
    )
    monkeypatch.setattr(
        producer_module,
        "build_validation_artifact_fingerprint",
        capture_fingerprint,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_sft_validation_event.py",
            "--config",
            str(_SUPER_V3_CONFIG),
            "--artifact-dir",
            str(artifact_directory),
            "--dataset-sha256",
            "a" * 64,
            "--tokenizer-sha256",
            "b" * 64,
            "--container-sha256",
            "f" * 64,
            "--preprocessing-sha256",
            expected_preprocessing_sha256,
            *overrides,
        ],
    )

    producer_main()

    assert captured_fingerprint is not None
    assert captured_fingerprint.preprocessing_sha256 == expected_preprocessing_sha256
    manifest = next(artifact_directory.glob("*.json"))
    loaded = load_validation_event(
        manifest,
        captured_fingerprint,
        MemoryBudget(available_bytes=10_000_000),
    )
    assert loaded.num_valid_tokens == (64, 128, 192, 256)
    assert loaded.data["input_ids"][:, 0].tolist() == list(range(10, 266))


def test_producer_rejects_unknown_validation_dataset_contract() -> None:
    config = _producer_config_fixture()
    config.data["validation"] = {"dataset_name": "unknown"}

    with pytest.raises(ValueError, match="megatron_sft_packed"):
        derive_validation_artifact_eligibility(
            config,
            _producer_dataset_fixture([]),
        )


def test_repeated_production_preserves_rng_and_serialized_artifact(tmp_path) -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    python_state = random.getstate()
    numpy_state = cast(tuple[Any, ...], np.random.get_state())
    torch_state = torch.get_rng_state()

    first = _producer_event([_packed_validation_batch(index) for index in range(4)])
    second = _producer_event([_packed_validation_batch(index) for index in range(4)])

    assert random.getstate() == python_state
    current_numpy_state = cast(tuple[Any, ...], np.random.get_state())
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)

    first_manifest = save_validation_event(
        tmp_path / "first", first, _fingerprint(), _supported_eligibility()
    )
    second_manifest = save_validation_event(
        tmp_path / "second", second, _fingerprint(), _supported_eligibility()
    )

    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    first_tensor = (
        first_manifest.parent / _manifest_content(first_manifest)["tensor_file"]
    )
    second_tensor = (
        second_manifest.parent / _manifest_content(second_manifest)["tensor_file"]
    )
    assert first_tensor.read_bytes() == second_tensor.read_bytes()


def _fingerprint() -> ValidationArtifactFingerprint:
    return ValidationArtifactFingerprint(
        dataset_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        preprocessing_sha256="c" * 64,
        nemo_rl_commit="d" * 40,
        submodule_commits=(("Megatron-LM", "e" * 40),),
        container_sha256="f" * 64,
    )


def _event_fixture(*, offset: int = 0) -> PrecomputedValidationEvent:
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(offset, offset + 6, dtype=torch.int64).reshape(
                2, 3
            ),
            "input_lengths": torch.tensor([3, 2], dtype=torch.int64),
            "token_mask": torch.tensor([[True, True, False], [True, False, False]]),
            "sample_mask": torch.ones(2, dtype=torch.float32),
            "processed_token_counts": torch.tensor([2, 1], dtype=torch.int64),
            "idx": [17, 23],
            "task_name": ["megatron_sft_packed", "validation_aux"],
        }
    )
    return PrecomputedValidationEvent(
        data=data,
        num_valid_tokens=(2, 1, 0, 3),
        payload_digest=digest_validation_event_data(data),
        retained_bytes=sum(
            value.nbytes for value in data.values() if torch.is_tensor(value)
        ),
    )


def _memory_budget() -> MemoryBudget:
    return MemoryBudget(available_bytes=1_000_000)


def _supported_eligibility() -> ValidationArtifactEligibility:
    return ValidationArtifactEligibility.from_producer_facts(
        prepacked_input=True,
        raw_online_packing=False,
        stochastic_preprocessing=False,
        dynamic_batching=False,
        multimodal_data=False,
    )


def _manifest_content(manifest) -> dict[str, Any]:
    return json.loads(manifest.read_text())


def _write_manifest(manifest, content: dict[str, Any]) -> None:
    manifest.write_text(json.dumps(content))


def test_validation_artifact_round_trip_preserves_tensor_contract(tmp_path) -> None:
    event = _event_fixture()

    manifest = save_validation_event(
        tmp_path, event, _fingerprint(), _supported_eligibility()
    )
    loaded = load_validation_event(manifest, _fingerprint(), _memory_budget())

    assert loaded.num_valid_tokens == event.num_valid_tokens
    assert loaded.payload_digest == event.payload_digest
    for key, value in event.data.items():
        if torch.is_tensor(value):
            assert torch.equal(loaded.data[key], value)
        else:
            assert loaded.data[key] == value
    content = _manifest_content(manifest)
    assert content["tensor_file"].startswith("validation-")
    assert content["tensor_file"].endswith(".safetensors")


def test_validation_artifact_round_trip_preserves_runtime_metadata(tmp_path) -> None:
    event = _event_fixture()
    event.data["processed_token_counts"] = torch.tensor([2, 1], dtype=torch.int64)
    event.data["idx"] = [17, 23]
    event.data["task_name"] = ["megatron_sft_packed", "megatron_sft_packed"]
    event = dataclasses.replace(
        event,
        payload_digest=digest_validation_event_data(event.data),
        retained_bytes=sum(
            value.nbytes for value in event.data.values() if torch.is_tensor(value)
        ),
    )

    manifest = save_validation_event(
        tmp_path, event, _fingerprint(), _supported_eligibility()
    )
    loaded = load_validation_event(manifest, _fingerprint(), _memory_budget())

    assert set(loaded.data) == set(event.data)
    assert torch.equal(
        loaded.data["processed_token_counts"], event.data["processed_token_counts"]
    )
    assert loaded.data["idx"] == event.data["idx"]
    assert loaded.data["task_name"] == event.data["task_name"]
    assert _manifest_content(manifest)["metadata"] == {
        "idx": [17, 23],
        "task_name": ["megatron_sft_packed", "megatron_sft_packed"],
    }


@pytest.mark.parametrize(
    "field",
    ["processed_token_counts", "idx", "task_name"],
)
def test_validation_artifact_requires_runtime_metadata(
    tmp_path: Path, field: str
) -> None:
    event = _event_fixture()
    del event.data[field]
    event = dataclasses.replace(
        event,
        retained_bytes=sum(
            value.nbytes for value in event.data.values() if torch.is_tensor(value)
        ),
    )

    with pytest.raises(ValueError, match=f"missing required.*{field}"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


def test_validation_artifact_load_rejects_metadata_digest_mismatch(
    tmp_path: Path,
) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content["metadata"]["idx"][0] += 1
    _write_manifest(manifest, content)

    with pytest.raises(ValueError, match="payload digest"):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


def test_validation_event_digest_includes_ordered_list_metadata() -> None:
    event = _event_fixture()
    baseline = digest_validation_event_data(event.data)
    changed = clone_validation_event_data(event.data)
    changed["task_name"] = list(reversed(changed["task_name"]))

    assert digest_validation_event_data(changed) != baseline


def test_validation_event_digest_ignores_mapping_insertion_order() -> None:
    event = _event_fixture()
    reversed_data = BatchedDataDict(reversed(list(event.data.items())))

    assert digest_validation_event_data(reversed_data) == digest_validation_event_data(
        event.data
    )


def test_validation_artifact_load_preserves_driver_rng_and_generator(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    generator = torch.Generator().manual_seed(404)
    python_state = random.getstate()
    numpy_state = cast(tuple[Any, ...], np.random.get_state())
    torch_state = torch.get_rng_state().clone()
    generator_state = generator.get_state().clone()

    load_validation_event(manifest, _fingerprint(), _memory_budget())

    assert random.getstate() == python_state
    current_numpy_state = cast(tuple[Any, ...], np.random.get_state())
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.equal(generator.get_state(), generator_state)


def test_validation_artifact_rejects_unknown_non_tensor_value(tmp_path) -> None:
    event = _event_fixture()
    event.data["messages"] = ["unsupported"]

    with pytest.raises(TypeError, match="tensor-only"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


@pytest.mark.parametrize(
    "field", ["dataset_sha256", "tokenizer_sha256", "preprocessing_sha256"]
)
def test_load_fails_closed_on_fingerprint_mismatch(tmp_path, field: str) -> None:
    fingerprint = _fingerprint()
    manifest = save_validation_event(
        tmp_path, _event_fixture(), fingerprint, _supported_eligibility()
    )
    changed = dataclasses.replace(fingerprint, **{field: "f" * 64})

    with pytest.raises(ValueError, match=field):
        load_validation_event(manifest, changed, _memory_budget())


def test_load_rejects_corrupted_tensor_bytes(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    tensor_path = manifest.parent / _manifest_content(manifest)["tensor_file"]
    content = bytearray(tensor_path.read_bytes())
    content[-1] ^= 1
    tensor_path.write_bytes(content)

    with pytest.raises(ValueError, match="SHA-256"):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


def test_save_rejects_non_cpu_tensor(tmp_path) -> None:
    event = _event_fixture()
    event.data["input_ids"] = torch.empty(1, device="meta")

    with pytest.raises(ValueError, match="CPU tensors only"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_save_rejects_cuda_tensor(tmp_path) -> None:
    event = _event_fixture()
    event.data["input_ids"] = torch.zeros(1, device="cuda")

    with pytest.raises(ValueError, match="CPU tensors only"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


def test_load_enforces_three_copy_memory_headroom(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )

    with pytest.raises(MemoryError, match="three-copy headroom"):
        load_validation_event(manifest, _fingerprint(), MemoryBudget(available_bytes=1))


def test_submission_clone_cannot_mutate_canonical_event() -> None:
    canonical = _event_fixture()
    canonical.data["idx"] = [17, 23]
    canonical.data["task_name"] = ["megatron_sft_packed", "megatron_sft_packed"]
    submitted = clone_validation_event_data(canonical.data)
    submitted["input_ids"][0, 0] = -1
    submitted["idx"][0] = -1
    submitted["task_name"][0] = "mutated"

    assert canonical.data["input_ids"][0, 0].item() == 0
    assert canonical.data["idx"] == [17, 23]
    assert canonical.data["task_name"] == [
        "megatron_sft_packed",
        "megatron_sft_packed",
    ]


def test_submission_clone_rejects_unknown_sft_tensor_key() -> None:
    canonical = _event_fixture()
    canonical.data["pixel_values"] = torch.zeros((2, 3, 4, 4))

    with pytest.raises(ValueError, match="unknown SFT tensor keys.*pixel_values"):
        clone_validation_event_data(canonical.data)


def test_tensor_content_hash_is_independent_of_tensor_layout() -> None:
    contiguous = torch.arange(12, dtype=torch.int64).reshape(3, 4)

    assert tensor_content_sha256(contiguous) == tensor_content_sha256(
        contiguous.transpose(0, 1).contiguous().transpose(0, 1)
    )


def test_load_rejects_unknown_manifest_key(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content["unexpected"] = True
    _write_manifest(manifest, content)

    with pytest.raises(ValueError, match="unknown keys"):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


def test_load_rejects_v2_before_applying_v3_schema(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content["artifact_version"] = 2
    del content["metadata"]
    _write_manifest(manifest, content)

    with pytest.raises(ValueError, match="Unsupported validation artifact version"):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


def test_interrupted_publish_preserves_previous_artifact_pair(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )

    with patch.object(
        artifact_module,
        "_atomic_write",
        side_effect=OSError("interrupted manifest publish"),
    ):
        with pytest.raises(OSError, match="interrupted"):
            save_validation_event(
                tmp_path,
                _event_fixture(offset=100),
                _fingerprint(),
                _supported_eligibility(),
            )

    loaded = load_validation_event(manifest, _fingerprint(), _memory_budget())
    assert loaded.data["input_ids"][0, 0].item() == 0


def test_concurrent_writers_are_serialized(tmp_path) -> None:
    original_save = artifact_module.save_safetensors_file
    state_lock = threading.Lock()
    active_writers = 0
    maximum_active_writers = 0

    def observed_save(*args, **kwargs) -> None:
        nonlocal active_writers, maximum_active_writers
        with state_lock:
            active_writers += 1
            maximum_active_writers = max(maximum_active_writers, active_writers)
        try:
            time.sleep(0.05)
            original_save(*args, **kwargs)
        finally:
            with state_lock:
                active_writers -= 1

    events = [_event_fixture(), _event_fixture(offset=100)]
    with patch.object(artifact_module, "save_safetensors_file", observed_save):
        with ThreadPoolExecutor(max_workers=2) as executor:
            manifests = list(
                executor.map(
                    lambda event: save_validation_event(
                        tmp_path, event, _fingerprint(), _supported_eligibility()
                    ),
                    events,
                )
            )

    assert maximum_active_writers == 1
    assert manifests[0] == manifests[1]
    loaded = load_validation_event(manifests[0], _fingerprint(), _memory_budget())
    assert loaded.data["input_ids"][0, 0].item() in {0, 100}


def test_forged_small_manifest_fails_memory_check_before_tensor_load(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content["retained_bytes"] = 0
    for record in content["tensors"].values():
        record["nbytes"] = 0
    _write_manifest(manifest, content)

    with patch.object(artifact_module, "load_safetensors_file") as tensor_loader:
        with pytest.raises(MemoryError, match="three-copy headroom"):
            load_validation_event(
                manifest,
                _fingerprint(),
                MemoryBudget(available_bytes=1),
            )

    tensor_loader.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("dataset_sha256", "A" * 64, id="uppercase-dataset"),
        pytest.param("tokenizer_sha256", "b" * 63, id="short-tokenizer"),
        pytest.param("preprocessing_sha256", "z" * 64, id="nonhex-preprocessing"),
        pytest.param("container_sha256", "", id="empty-container"),
    ],
)
def test_save_rejects_malformed_sha256_fingerprint(
    tmp_path, field: str, value: str
) -> None:
    fingerprint = dataclasses.replace(_fingerprint(), **{field: value})

    with pytest.raises(ValueError, match=field):
        save_validation_event(
            tmp_path, _event_fixture(), fingerprint, _supported_eligibility()
        )


@pytest.mark.parametrize("commit", ["", "g" * 40, "a" * 39])
def test_save_rejects_malformed_nemo_rl_commit(tmp_path, commit: str) -> None:
    fingerprint = dataclasses.replace(_fingerprint(), nemo_rl_commit=commit)

    with pytest.raises(ValueError, match="nemo_rl_commit"):
        save_validation_event(
            tmp_path, _event_fixture(), fingerprint, _supported_eligibility()
        )


@pytest.mark.parametrize(
    "submodules",
    [
        pytest.param((), id="empty"),
        pytest.param((("a", "e" * 40), ("a", "f" * 40)), id="duplicate-path"),
        pytest.param((("z", "e" * 40), ("a", "f" * 40)), id="unsorted"),
        pytest.param((("../escape", "e" * 40),), id="parent-path"),
        pytest.param((("/absolute", "e" * 40),), id="absolute-path"),
        pytest.param(((".", "e" * 40),), id="current-directory-path"),
        pytest.param((("module", "g" * 40),), id="invalid-commit"),
    ],
)
def test_save_rejects_invalid_recursive_submodule_entries(
    tmp_path, submodules: tuple[tuple[str, str], ...]
) -> None:
    fingerprint = dataclasses.replace(_fingerprint(), submodule_commits=submodules)

    with pytest.raises(ValueError, match="submodule_commits"):
        save_validation_event(
            tmp_path, _event_fixture(), fingerprint, _supported_eligibility()
        )


@pytest.mark.parametrize(
    "num_valid_tokens",
    [
        pytest.param((True, 1, 2, 3), id="bool"),
        pytest.param((-1, 1, 2, 3), id="negative"),
    ],
)
def test_save_rejects_nonexact_token_counts(tmp_path, num_valid_tokens) -> None:
    event = dataclasses.replace(_event_fixture(), num_valid_tokens=num_valid_tokens)

    with pytest.raises(ValueError, match="non-negative integers"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


@pytest.mark.parametrize("retained_bytes", [True, -1])
def test_save_rejects_nonexact_retained_bytes(tmp_path, retained_bytes) -> None:
    event = dataclasses.replace(_event_fixture(), retained_bytes=retained_bytes)

    with pytest.raises(ValueError, match="retained_bytes"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


@pytest.mark.parametrize(
    "budget",
    [
        pytest.param(MemoryBudget(available_bytes=True), id="bool-available"),
        pytest.param(MemoryBudget(available_bytes=-1), id="negative-available"),
        pytest.param(
            MemoryBudget(available_bytes=1_000_000, required_copy_count=True),
            id="bool-copies",
        ),
        pytest.param(
            MemoryBudget(available_bytes=1_000_000, required_copy_count=0),
            id="zero-copies",
        ),
    ],
)
def test_load_rejects_nonexact_memory_budget(tmp_path, budget: MemoryBudget) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )

    with pytest.raises(ValueError, match="MemoryBudget"):
        load_validation_event(manifest, _fingerprint(), budget)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("artifact_version", True, id="bool-version"),
        pytest.param("retained_bytes", True, id="bool-retained-bytes"),
        pytest.param("retained_bytes", -1, id="negative-retained-bytes"),
    ],
)
def test_load_rejects_nonexact_manifest_integers(
    tmp_path, field: str, value: object
) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content[field] = value
    _write_manifest(manifest, content)

    with pytest.raises(ValueError, match=field):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("nbytes", True, id="bool-bytes"),
        pytest.param("nbytes", -1, id="negative-bytes"),
        pytest.param("shape", [2, -1], id="negative-dimension"),
        pytest.param("shape", [2, True], id="bool-dimension"),
    ],
)
def test_load_rejects_nonexact_tensor_record_integers(
    tmp_path, field: str, value: object
) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content["tensors"]["input_ids"][field] = value
    _write_manifest(manifest, content)

    with patch.object(artifact_module, "load_safetensors_file") as tensor_loader:
        with pytest.raises(ValueError, match=field):
            load_validation_event(manifest, _fingerprint(), _memory_budget())

    tensor_loader.assert_not_called()


def test_save_rejects_unknown_sft_tensor_key(tmp_path) -> None:
    event = _event_fixture()
    event.data["pixel_values"] = torch.zeros((2, 3, 4, 4))

    with pytest.raises(ValueError, match="unknown SFT tensor keys.*pixel_values"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


def test_save_requires_complete_sft_tensor_schema(tmp_path) -> None:
    event = _event_fixture()
    del event.data["sample_mask"]

    with pytest.raises(ValueError, match="missing required SFT tensor keys"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


def test_save_requires_complete_packed_metadata_group(tmp_path) -> None:
    event = _event_fixture()
    event.data["packed_cu_seqlens"] = torch.tensor([[0, 3], [0, 2]], dtype=torch.int32)

    with pytest.raises(ValueError, match="packed metadata must include"):
        save_validation_event(tmp_path, event, _fingerprint(), _supported_eligibility())


def test_supported_producer_eligibility_round_trips(tmp_path) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )

    assert _manifest_content(manifest)["eligibility"] == {
        "prepacked_input": True,
        "dynamic_batching": False,
        "multimodal_data": False,
        "raw_online_packing": False,
        "stochastic_preprocessing": False,
    }
    load_validation_event(manifest, _fingerprint(), _memory_budget())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("prepacked_input", False, id="not-prepacked"),
        pytest.param("raw_online_packing", True, id="online-packing"),
        pytest.param("stochastic_preprocessing", True, id="stochastic"),
        pytest.param("dynamic_batching", True, id="dynamic-batching"),
        pytest.param("multimodal_data", True, id="multimodal"),
    ],
)
def test_save_rejects_unsupported_producer_fact_before_publication(
    tmp_path, field: str, value: bool
) -> None:
    artifact_directory = tmp_path / "artifact"
    eligibility = dataclasses.replace(_supported_eligibility(), **{field: value})

    with pytest.raises(ValueError, match="producer eligibility"):
        save_validation_event(
            artifact_directory, _event_fixture(), _fingerprint(), eligibility
        )

    assert not artifact_directory.exists()


def test_save_requires_explicit_producer_eligibility_before_publication(
    tmp_path,
) -> None:
    artifact_directory = tmp_path / "artifact"
    untyped_save: Any = save_validation_event

    with pytest.raises(TypeError, match="eligibility"):
        untyped_save(artifact_directory, _event_fixture(), _fingerprint())

    assert not artifact_directory.exists()


@pytest.mark.parametrize(
    "field",
    [
        "prepacked_input",
        "raw_online_packing",
        "stochastic_preprocessing",
        "dynamic_batching",
        "multimodal_data",
    ],
)
def test_producer_eligibility_requires_exact_booleans(field: str) -> None:
    facts: dict[str, object] = {
        "prepacked_input": True,
        "raw_online_packing": False,
        "stochastic_preprocessing": False,
        "dynamic_batching": False,
        "multimodal_data": False,
    }
    facts[field] = 1
    untyped_factory: Any = ValidationArtifactEligibility.from_producer_facts

    with pytest.raises(TypeError, match=field):
        untyped_factory(**facts)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("prepacked_input", False, id="not-prepacked"),
        pytest.param("raw_online_packing", True, id="online-packing"),
        pytest.param("stochastic_preprocessing", True, id="stochastic"),
        pytest.param("dynamic_batching", True, id="dynamic-batching"),
        pytest.param("multimodal_data", True, id="multimodal"),
    ],
)
def test_load_rejects_ineligible_artifact(tmp_path, field: str, value: object) -> None:
    manifest = save_validation_event(
        tmp_path, _event_fixture(), _fingerprint(), _supported_eligibility()
    )
    content = _manifest_content(manifest)
    content["eligibility"][field] = value
    _write_manifest(manifest, content)

    with pytest.raises(ValueError, match="eligibility"):
        load_validation_event(manifest, _fingerprint(), _memory_budget())


def test_tensor_content_sha256_handles_scalar_tensor() -> None:
    scalar = torch.tensor(7, dtype=torch.int64)

    assert (
        tensor_content_sha256(scalar)
        == hashlib.sha256(scalar.numpy().tobytes()).hexdigest()
    )
