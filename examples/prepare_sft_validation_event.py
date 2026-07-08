# pyright: reportMissingImports=false

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

"""Produce a deterministic, tensor-only SFT validation event artifact."""

import argparse
import hashlib
import json
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from examples.run_sft import setup_data
from nemo_rl.algorithms.sft import (
    MasterConfig,
    _build_sft_collate_fn,
    _combine_validation_event_batches,
    _validate_packed_validation_metadata,
)
from nemo_rl.algorithms.sft_validation_artifact import (
    PrecomputedValidationEvent,
    ValidationArtifactEligibility,
    save_validation_event,
    tensor_content_sha256,
)
from nemo_rl.algorithms.sft_validation_provenance import (
    _validation_dataset_configs,
    build_validation_artifact_fingerprint,
    derive_preprocessing_sha256,
    validate_validation_source_config,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.megatron_sft_packed import megatron_sft_packed_preprocessor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


_PACKED_DATASET_NAME = "megatron_sft_packed"
_PERSISTED_TENSOR_KEYS = frozenset(
    {
        "input_ids",
        "input_lengths",
        "sample_mask",
        "token_mask",
        "position_ids",
        "target_ids",
        "packed_cu_seqlens",
        "packed_cu_seqlens_lengths",
        "packed_max_seqlens",
    }
)
_RUNTIME_ONLY_KEYS = frozenset({"idx", "processed_token_counts", "task_name"})


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse producer arguments and preserve Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Produce a deterministic SFT validation event artifact"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory where the validation artifact is published",
    )
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument(
        "--preprocessing-sha256",
        help="Optional expected digest of the resolved preprocessing config",
    )
    parser.add_argument("--container-sha256", required=True)
    return parser.parse_known_args()


def load_master_config(config_path: str | Path, overrides: list[str]) -> MasterConfig:
    """Load one fully resolved SFT config after applying Hydra overrides."""
    register_omegaconf_resolvers()
    config = load_config(config_path)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved SFT config must be a mapping")
    return MasterConfig(**cast(Any, resolved))


def derive_validation_artifact_eligibility(
    config: MasterConfig,
    val_dataset: AllTaskProcessedDataset,
) -> ValidationArtifactEligibility:
    """Prove that the configured validation path is deterministic packed text SFT."""
    validate_validation_source_config(config)
    validation_configs = _validation_dataset_configs(config.data)
    if any(
        validation_config.get("dataset_name") != _PACKED_DATASET_NAME
        for validation_config in validation_configs
    ):
        raise ValueError(
            "Validation artifact production requires every validation dataset to use "
            f"{_PACKED_DATASET_NAME!r}"
        )
    if config.data["shuffle"] is not False:
        raise ValueError("Validation artifact production requires data.shuffle=false")

    dynamic_batching = config.policy["dynamic_batching"]
    if (
        not isinstance(dynamic_batching, Mapping)
        or dynamic_batching.get("enabled") is not False
    ):
        raise ValueError(
            "Validation artifact production requires policy.dynamic_batching.enabled=false"
        )

    megatron_config = config.policy.get("megatron_cfg")
    if (
        not isinstance(megatron_config, Mapping)
        or megatron_config.get("enabled") is not True
        or megatron_config.get("prepacked_sft_loss_mode") != "labels"
    ):
        raise ValueError(
            "Validation artifact production requires the text-only Megatron "
            "prepacked SFT labels path"
        )
    _validate_packed_processor_contract(val_dataset)

    return ValidationArtifactEligibility.from_producer_facts(
        prepacked_input=True,
        raw_online_packing=False,
        stochastic_preprocessing=False,
        dynamic_batching=False,
        multimodal_data=False,
    )


def build_precomputed_validation_event(
    config: MasterConfig,
    tokenizer: AutoTokenizer,
    val_dataset: AllTaskProcessedDataset,
) -> PrecomputedValidationEvent:
    """Build one deterministic four-batch validation event without Ray."""
    derive_validation_artifact_eligibility(config, val_dataset)
    _validate_event_batch_config(config)
    pad_token_id = tokenizer.pad_token_id
    if not isinstance(pad_token_id, int):
        raise ValueError(
            "Validation artifact production requires an integer pad_token_id"
        )
    num_workers = config.data.get("num_workers")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool):
        raise ValueError(
            "Validation artifact production requires integer data.num_workers"
        )

    with _preserved_rng_state():
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=config.sft.val_global_batch_size,
            shuffle=False,
            collate_fn=_build_sft_collate_fn(config.policy),
            drop_last=True,
            num_workers=num_workers,
        )
        batches = list(islice(val_dataloader, 4))
        if len(batches) != 4:
            raise ValueError(
                "Validation artifact production requires four complete validation batches; "
                f"collected {len(batches)}"
            )
        for batch in batches:
            if "packed_cu_seqlens" not in batch:
                raise ValueError(
                    "Validation artifact production requires packed validation data"
                )
            _validate_packed_validation_metadata(batch)
        token_counts = [_valid_token_count(batch) for batch in batches]
        num_valid_tokens: tuple[int, int, int, int] = (
            token_counts[0],
            token_counts[1],
            token_counts[2],
            token_counts[3],
        )
        combined = _combine_validation_event_batches(
            batches,
            global_batch_size=config.sft.val_global_batch_size,
            pad_token_id=pad_token_id,
        )
        event_data = _event_tensor_data(combined)

    return PrecomputedValidationEvent(
        data=event_data,
        num_valid_tokens=num_valid_tokens,
        payload_digest=digest_validation_event_data(event_data),
        retained_bytes=sum(tensor.nbytes for tensor in event_data.values()),
    )


def digest_validation_event_data(data: Mapping[str, object]) -> str:
    """Return the canonical digest of the tensor-only artifact payload."""
    tensor_data = _event_tensor_data(data, clone_tensors=False)
    records = {
        key: {
            "dtype": str(tensor.dtype),
            "sha256": tensor_content_sha256(tensor),
            "shape": list(tensor.shape),
        }
        for key, tensor in sorted(tensor_data.items())
    }
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_packed_processor_contract(val_dataset: AllTaskProcessedDataset) -> None:
    processors = val_dataset.task_data_processors
    if not isinstance(processors, dict) or set(processors) != {_PACKED_DATASET_NAME}:
        raise ValueError(
            "Validation artifact production requires only the known "
            "megatron_sft_packed processor contract"
        )
    _, processor = processors[_PACKED_DATASET_NAME]
    processor_function = processor
    while isinstance(processor_function, partial):
        processor_function = processor_function.func
    if processor_function is not megatron_sft_packed_preprocessor:
        raise ValueError(
            "Validation artifact production requires the known "
            "megatron_sft_packed processor contract"
        )
    if val_dataset.task_data_preprocessors:
        raise ValueError(
            "Validation artifact production rejects validation preprocessors because "
            "their determinism cannot be proven"
        )


def _validate_event_batch_config(config: MasterConfig) -> None:
    if config.sft.val_batches != 4:
        raise ValueError(
            "Validation artifact production requires sft.val_batches=4; "
            f"got {config.sft.val_batches}"
        )
    if config.sft.val_global_batch_size != 64:
        raise ValueError(
            "Validation artifact production requires sft.val_global_batch_size=64; "
            f"got {config.sft.val_global_batch_size}"
        )
    if config.sft.val_micro_batch_size != 1:
        raise ValueError(
            "Validation artifact production requires sft.val_micro_batch_size=1; "
            f"got {config.sft.val_micro_batch_size}"
        )


def _valid_token_count(batch: BatchedDataDict[Any]) -> int:
    return int((batch["sample_mask"].unsqueeze(-1) * batch["token_mask"]).sum().item())


def _event_tensor_data(
    data: Mapping[str, object],
    *,
    clone_tensors: bool = True,
) -> BatchedDataDict[Any]:
    tensors = BatchedDataDict[Any]()
    for key, value in data.items():
        if key in _PERSISTED_TENSOR_KEYS:
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Validation artifact tensor {key!r} must be a torch.Tensor"
                )
            tensor: Any = value
            if tensor.device.type != "cpu":
                raise ValueError(
                    "Validation artifact production supports CPU tensors only"
                )
            tensors[key] = (
                tensor.detach().contiguous().clone() if clone_tensors else tensor
            )
        elif key not in _RUNTIME_ONLY_KEYS:
            raise ValueError(
                f"Validation artifact production cannot persist unknown batch key {key!r}"
            )
    return tensors


@contextmanager
def _preserved_rng_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def main() -> None:
    """Load config, produce a validation event, and atomically publish it."""
    args, overrides = parse_args()
    config_path = (
        args.config
        if args.config
        else str(Path(__file__).parent / "configs" / "sft.yaml")
    )
    config = load_master_config(config_path, overrides)
    validate_validation_source_config(config)
    preprocessing_sha256 = derive_preprocessing_sha256(
        config,
        expected_sha256=args.preprocessing_sha256,
    )

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    _, val_dataset = setup_data(tokenizer, config.data)
    if val_dataset is None:
        raise ValueError("Validation artifact production requires validation data")
    eligibility = derive_validation_artifact_eligibility(config, val_dataset)
    event = build_precomputed_validation_event(config, tokenizer, val_dataset)
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = build_validation_artifact_fingerprint(
        dataset_sha256=args.dataset_sha256,
        tokenizer_sha256=args.tokenizer_sha256,
        preprocessing_sha256=preprocessing_sha256,
        container_sha256=args.container_sha256,
        repository_root=repository_root,
    )
    manifest = save_validation_event(
        Path(args.artifact_dir),
        event,
        fingerprint,
        eligibility=eligibility,
    )
    print(f"Published validation event artifact: {manifest}")


if __name__ == "__main__":
    main()
