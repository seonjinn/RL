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

"""Immutable checkpoint identities shared by metadata staging and capture."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


STAGED_CHECKPOINT_DIRECTORY = "checkpoints"
STAGED_CONFIG_FILENAME = "config.json"
STAGED_INDEX_FILENAME = "model.safetensors.index.json"
STAGED_HEADER_MANIFEST_FILENAME = "safetensors_header_manifest.json"
STAGED_HEADER_LENGTHS_FILENAME = "safetensors_header_byte_lengths.json"

_ARTIFACT_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_HEX_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class CheckpointMetadataArtifactIdentity:
    """Typed immutable identity for metadata fetched from one checkpoint."""

    artifact_id: str
    repository: str
    revision: str
    config_sha256: str
    index_sha256: str
    header_manifest_sha256: str
    shards: int
    tensors: int
    mtp_header_byte_lengths: tuple[tuple[str, int], ...] = ()
    catalog_admission: str | None = None
    quant_method: str | None = None
    remainder_evidence: str | None = None
    weight_block_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact_id, str)
            or _ARTIFACT_ID_PATTERN.fullmatch(self.artifact_id) is None
        ):
            raise ValueError("checkpoint artifact id is invalid")
        if not isinstance(self.repository, str):
            raise ValueError("checkpoint repository must contain owner and name")
        repository_parts = self.repository.split("/")
        if len(repository_parts) != 2 or any(not part for part in repository_parts):
            raise ValueError("checkpoint repository must contain owner and name")
        if (
            not isinstance(self.revision, str)
            or _HEX_REVISION_PATTERN.fullmatch(self.revision) is None
        ):
            raise ValueError("checkpoint revision must be a lowercase Git SHA")
        for digest in (
            self.config_sha256,
            self.index_sha256,
            self.header_manifest_sha256,
        ):
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError("checkpoint digest must be lowercase SHA256")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in (self.shards, self.tensors)
        ):
            raise ValueError("checkpoint shard and tensor counts must be positive")
        if not isinstance(self.mtp_header_byte_lengths, tuple) or any(
            not isinstance(entry, tuple) or len(entry) != 2
            for entry in self.mtp_header_byte_lengths
        ):
            raise ValueError("checkpoint MTP header lengths must use frozen tuples")
        if any(
            not isinstance(name, str) or not name
            for name, _ in self.mtp_header_byte_lengths
        ) or any(
            isinstance(length, bool) or not isinstance(length, int) or length <= 0
            for _, length in self.mtp_header_byte_lengths
        ):
            raise ValueError("checkpoint MTP header lengths are invalid")
        header_names = tuple(name for name, _ in self.mtp_header_byte_lengths)
        if len(set(header_names)) != len(header_names):
            raise ValueError("checkpoint MTP header lengths are invalid")
        optional_quantization_fields = (
            self.catalog_admission,
            self.quant_method,
            self.remainder_evidence,
            self.weight_block_size,
        )
        if any(field is not None for field in optional_quantization_fields) and any(
            field is None for field in optional_quantization_fields
        ):
            raise ValueError(
                "checkpoint quantization metadata must be entirely present or absent"
            )
        if self.catalog_admission is not None and any(
            not isinstance(field, str) or not field
            for field in (
                self.catalog_admission,
                self.quant_method,
                self.remainder_evidence,
            )
        ):
            raise ValueError("checkpoint quantization metadata must contain strings")
        if self.weight_block_size is not None:
            if (
                not isinstance(self.weight_block_size, tuple)
                or len(self.weight_block_size) != 2
            ):
                raise ValueError("checkpoint weight block size must be a frozen pair")
            if any(
                isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0
                for extent in self.weight_block_size
            ):
                raise ValueError("checkpoint weight block extents must be positive")

    @property
    def artifact(self) -> Mapping[str, object]:
        """Return a fresh wire-compatible artifact mapping."""
        artifact: dict[str, object] = {
            "kind": "immutable_hf_metadata",
            "repository": self.repository,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "index_sha256": self.index_sha256,
            "header_manifest_sha256": self.header_manifest_sha256,
        }
        if self.mtp_header_byte_lengths:
            artifact["mtp_header_byte_lengths"] = dict(self.mtp_header_byte_lengths)
        if self.catalog_admission is not None:
            artifact["catalog_admission"] = self.catalog_admission
            artifact["quant_method"] = self.quant_method
            artifact["remainder_evidence"] = self.remainder_evidence
        artifact["shards"] = self.shards
        artifact["tensors"] = self.tensors
        if self.weight_block_size is not None:
            artifact["weight_block_size"] = list(self.weight_block_size)
        return artifact


class CheckpointMetadataArtifactSpec(Protocol):
    """Structural metadata-only view accepted by the network stager."""

    @property
    def artifact_id(self) -> str: ...

    @property
    def artifact(self) -> Mapping[str, object]: ...


_CHECKPOINT_METADATA_ARTIFACT_IDENTITIES = (
    CheckpointMetadataArtifactIdentity(
        artifact_id="qwen3_bf16",
        repository="Qwen/Qwen3-30B-A3B",
        revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        config_sha256=(
            "2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297"
        ),
        index_sha256=(
            "df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24"
        ),
        header_manifest_sha256=(
            "72d48dbc90e484781cffc7962ae19ceb477bd252981b4c9554d7f5792107d970"
        ),
        shards=16,
        tensors=18867,
    ),
    CheckpointMetadataArtifactIdentity(
        artifact_id="kimi_k2",
        repository="moonshotai/Kimi-K2-Base",
        revision="ce72df012259dcc55d945e890f815fe7ef69159c",
        config_sha256=(
            "8c13ae1049df55f29b3bdcae69a562433f243ff70dac251d819ecad8dbdf7439"
        ),
        index_sha256=(
            "c1f1d16c853f20467ae81361d2a92223650d39efa005f9c872a7cc14425ddcbc"
        ),
        header_manifest_sha256=(
            "ff7de9c047659d7cbc0cbee8734e60dade5384d48bda8a3600e33eb84a69fe41"
        ),
        shards=61,
        tensors=139644,
    ),
    CheckpointMetadataArtifactIdentity(
        artifact_id="kimi_k25",
        repository="moonshotai/Kimi-K2.5",
        revision="4d01dfe0332d63057c186e0b262165819efb6611",
        config_sha256=(
            "acd5bb01a16f64b309599cd6ed196be056f613c99d6bc9300692b82cd10882f6"
        ),
        index_sha256=(
            "bdba19b127c4d1dc57dc3b6f3366c10739c7e7f13baf3f5424b556469a4dbc1b"
        ),
        header_manifest_sha256=(
            "1f869fba2e6a9c4de7376fb6b277f545a78f6e0276075748589c438e35374012"
        ),
        shards=64,
        tensors=208550,
    ),
    CheckpointMetadataArtifactIdentity(
        artifact_id="kimi_k3",
        repository="moonshotai/Kimi-K3",
        revision="f831ab66814297da540d832a5235f8e904f29d06",
        config_sha256=(
            "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
        ),
        index_sha256=(
            "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
        ),
        header_manifest_sha256=(
            "35fc99eb32a3bce794e86f9ac7c1f4cdf55df197e60444b0c8c47dc25b95594b"
        ),
        shards=96,
        tensors=497220,
    ),
    CheckpointMetadataArtifactIdentity(
        artifact_id="nemotron_lightning_nvfp4",
        repository="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        revision="cc84af2fe71647d87f4486c064f320e1e7535243",
        config_sha256=(
            "f1d98b530846087dc08b574a219713a94f945bf6583dc7230a19ebf1e8c50933"
        ),
        index_sha256=(
            "3c3bc7efa8d658c2e909a0b9020eb0f72064e6647de348856af4dee9895bead9"
        ),
        header_manifest_sha256=(
            "b70b7d010a9aea3783f6bca9081a59afa41a80a97ff51d8e0ced2f41fb5f6714"
        ),
        shards=52,
        tensors=18487,
    ),
    CheckpointMetadataArtifactIdentity(
        artifact_id="qwen_a95b_fp8",
        repository="Qwen/Qwen3.8-2.4T-A95B-FP8",
        revision="d2dc35658bcf77e66643428cb52e774cc3b5bd29",
        config_sha256=(
            "b7396b749964c6afb5387c58e6425db8628e85f8ae66739d284eb1c8f42c4d4e"
        ),
        index_sha256=(
            "67f75ab10833869c951b5c8e02ddcf4fa11974a8dcb950c51193680c90a4f77c"
        ),
        header_manifest_sha256=(
            "cc5b309051da3d5fc508b8609247ce0f49aa0592839786cad9d7ddddfd8344c3"
        ),
        shards=213,
        tensors=287119,
        mtp_header_byte_lengths=(
            ("model-00185-of-00213.safetensors", 254184),
            ("model-00186-of-00213.safetensors", 127080),
        ),
        catalog_admission="both_logical_axes_divisible_by_128",
        quant_method="fp8",
        remainder_evidence="unsupported_not_observed",
        weight_block_size=(128, 128),
    ),
)

CHECKPOINT_METADATA_ARTIFACT_IDS = tuple(
    identity.artifact_id for identity in _CHECKPOINT_METADATA_ARTIFACT_IDENTITIES
)
if len(set(CHECKPOINT_METADATA_ARTIFACT_IDS)) != len(CHECKPOINT_METADATA_ARTIFACT_IDS):
    raise RuntimeError("checkpoint metadata artifact ids must be unique")


def checkpoint_metadata_artifact_identities() -> tuple[
    CheckpointMetadataArtifactIdentity, ...
]:
    """Return ordered immutable identities for metadata staging and capture."""
    return _CHECKPOINT_METADATA_ARTIFACT_IDENTITIES
