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

"""Capture reviewed precision-policy evidence without reading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    MXFP8_FORMAT,
    AxisExtentRounding,
    ComponentDescriptor,
    FormatDescriptor,
    LiteralComponentAxisSpec,
    LogicalComponentAxisSpec,
    resolve_component_axes,
)
from nemo_rl.precision_policy.source_formats import SOURCE_FORMAT_CATALOG


SOURCE_EVIDENCE_SCHEMA = "precision-policy-source-format-evidence.v1"
PRODUCER_EVIDENCE_SCHEMA = "precision-policy-producer-implementations.v1"
CAPTURE_RECEIPT_SCHEMA = "precision-policy-source-capture-receipt.v1"
STAGED_CHECKPOINT_DIRECTORY = "checkpoints"
STAGED_CONFIG_FILENAME = "config.json"
STAGED_INDEX_FILENAME = "model.safetensors.index.json"
STAGED_HEADER_MANIFEST_FILENAME = "safetensors_header_manifest.json"
STAGED_HEADER_LENGTHS_FILENAME = "safetensors_header_byte_lengths.json"
_ARTIFACT_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")

ROOT_TE_REVISION = "42b840051647eef89761a16dfdff87e82bb253ab"
ROOT_TE_PACKAGE = "2.15.0+42b8400"
BRIDGE_TE_REVISION = "4329ff84bfbdaa778a33cba02a15fb0807c64689"
BRIDGE_TE_PACKAGE = "2.17.1+4329ff84"
COMPRESSED_TENSORS_REVISION = "f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0"
COMPRESSED_TENSORS_PACKAGE = "0.17.0"
COMPRESSED_TENSORS_WHEEL_SHA256 = (
    "4a1b89b508f7efb8ffb4eee8a6e69e0452d9b080cae130146025c64fbe9fa9aa"
)
MODELOPT_LIGHTNING_REVISION = "c897fbeaaff66d53d61033f107885b7c5432f235"
MODELOPT_LIGHTNING_PACKAGE = "0.44.0rc5"

_AUTOMODEL_ADAPTER_PATH = (
    "3rdparty/Automodel-workspace/Automodel/"
    "nemo_automodel/components/models/kimi_k25_vl/state_dict_adapter.py"
)
_AUTOMODEL_ADAPTER_SHA256 = (
    "f09f9a2833d9597fbffbe6514adbfabaaab017471c0d7959f43b8011cae70a36"
)
_TE_MXFP8_PATH = "transformer_engine/pytorch/tensor/mxfp8_tensor.py"
_TE_MXFP8_SHA256 = "3f0cbdc95195e6e5719c007bf9b541252ea3cd81733a290c650beb9c63c0c6ed"
_TE_SOURCE_SHA256: dict[str, str] = {
    "transformer_engine/common/cast/mxfp8/dequantize_mxfp8.cuh": (
        "d9b5de5e73413f3b8856406d9eac8b8a8f3d769f136ab79bbbdd15e3a1889831"
    ),
    "transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh": (
        "84c1868d34795d67b2f48cbee6663c132b5ba7033ed6206f9f1cf89d8b4e6c96"
    ),
    "transformer_engine/common/cast/mxfp8/swizzle.cuh": (
        "fb7a56e4252c502517d3ce94c270254b01ff9f24bff22b79e539ba11ddbc4547"
    ),
    "transformer_engine/common/include/transformer_engine/transformer_engine.h": (
        "1a77cbf9fdfaf92f6961ca3cbd407c33bcede538189b9c5a857aaca9dcf73921"
    ),
    "transformer_engine/pytorch/csrc/quantizer.cpp": (
        "a46fff857056a90c33a48c8a8cc557a3081103df6ff68fc9f53f4ec90f34f2cf"
    ),
    "transformer_engine/pytorch/onnx_extensions.py": (
        "468aac0f4bdae6cd8a3ad9a845afb436f5f422bd05f90a86f63ca5d849f8b3b9"
    ),
    _TE_MXFP8_PATH: _TE_MXFP8_SHA256,
    "transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py": (
        "2ba10d82f36f93c1aae17654967aba01646bfac830036c29f451dee0e6560363"
    ),
}
_COMPRESSED_TENSORS_SOURCE_SHA256: dict[str, str] = {
    "src/compressed_tensors/compressors/mx_utils.py": (
        "9b116a7e3406199316a58c5e306ac3840c123d014e8d650c168ad284c169ab75"
    ),
    "src/compressed_tensors/compressors/mxfp4/base.py": (
        "992cfef1b99de4cf8589b714c6ef01106079fc1c065e2e2693b52dd3a6dfac50"
    ),
    "src/compressed_tensors/compressors/nvfp4/base.py": (
        "7850e3b679ccd87f0b0238d178e2765d9cc64337d865840d417e0a4b7dbb155c"
    ),
    "src/compressed_tensors/compressors/nvfp4/helpers.py": (
        "148bb7fddfd33130dcc7ee523d623963fdbddd3143a0767d7b10278ca446901c"
    ),
    "src/compressed_tensors/compressors/pack_quantized/base.py": (
        "a6a532a0b2ae19b7ebfb425d73776653ee8d117b3f69778f05e44e67175cfc9d"
    ),
    "src/compressed_tensors/compressors/pack_quantized/helpers.py": (
        "8619308666eba5e8a442d1647c34b6b5f9716b13ed2051ed17fb1ea683b0db72"
    ),
    "src/compressed_tensors/quantization/lifecycle/forward_helpers.py": (
        "8b3399fda143cc249c231e291d938c987a7da4793eeae35b3136b3390bbde6c8"
    ),
}
_MODELOPT_LIGHTNING_SOURCE_SHA256: dict[str, str] = {
    "modelopt/torch/quantization/qtensor/nvfp4_tensor.py": (
        "0aa20d06cefbf97031294681e2112f37650b4e08cefa38936632eac161aeeb6d"
    ),
}

_EXPECTED_IMPLEMENTATIONS: dict[str, dict[str, str]] = {
    "compressed_tensors_format_spec": {
        "kind": "locked_registry_source",
        "lock_path": "uv.lock",
        "source_revision": COMPRESSED_TENSORS_REVISION,
        "head_revision": COMPRESSED_TENSORS_REVISION,
        "package_identity": COMPRESSED_TENSORS_PACKAGE,
        "wheel_sha256": COMPRESSED_TENSORS_WHEEL_SHA256,
    },
    "megatron_bridge": {
        "kind": "git_submodule",
        "path": "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge",
        "gitlink_revision": "b11414c71b15e54d333eb49346ed199f20fa9021",
        "head_revision": "b11414c71b15e54d333eb49346ed199f20fa9021",
    },
    "nemo_automodel": {
        "kind": "git_submodule",
        "path": "3rdparty/Automodel-workspace/Automodel",
        "gitlink_revision": "1814c6c93a66b9d59d254960ef6a99a64249b671",
        "head_revision": "1814c6c93a66b9d59d254960ef6a99a64249b671",
    },
    "megatron_core": {
        "kind": "nested_git_submodule",
        "path": (
            "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
        ),
        "gitlink_revision": "7c9c3a027c503ae9ae1e8ad7b14397abb8269378",
        "head_revision": "7c9c3a027c503ae9ae1e8ad7b14397abb8269378",
    },
    "modelopt_lightning_producer": {
        "kind": "versioned_git_source",
        "source_revision": MODELOPT_LIGHTNING_REVISION,
        "head_revision": MODELOPT_LIGHTNING_REVISION,
        "package_identity": MODELOPT_LIGHTNING_PACKAGE,
    },
    "transformer_engine_root_runtime": {
        "kind": "locked_git_dependency",
        "lock_path": "uv.lock",
        "source_revision": ROOT_TE_REVISION,
        "package_identity": ROOT_TE_PACKAGE,
    },
    "transformer_engine_bridge_source": {
        "kind": "declared_git_dependency",
        "lock_path": ("3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/uv.lock"),
        "source_revision": BRIDGE_TE_REVISION,
        "package_identity": BRIDGE_TE_PACKAGE,
    },
}

_EXPECTED_SOURCE_CONTRACTS: dict[str, dict[str, object]] = {
    "compressed_tensors_int4_pack": {
        "implementation": "compressed_tensors_format_spec",
        "sources": {
            path: digest
            for path, digest in _COMPRESSED_TENSORS_SOURCE_SHA256.items()
            if path
            in {
                "src/compressed_tensors/compressors/pack_quantized/base.py",
                "src/compressed_tensors/compressors/pack_quantized/helpers.py",
                "src/compressed_tensors/quantization/lifecycle/forward_helpers.py",
            }
        },
        "contract": {
            "catalog_admission": (
                "input_features_divisible_by_32_and_packed_axis_divisible_by_8"
            ),
            "format_identifier": "pack-quantized",
            "group_axis": "input_features",
            "group_remainder_behavior": "reject_nondivisible_by_32",
            "group_size": 32,
            "nibble_order": "value_i_at_bits_4_times_i_lsb_first",
            "offset_binary_bias": 8,
            "packed_axis": "input_features",
            "packed_carrier_dtype": "int32",
            "packed_values_per_word": 8,
            "source_pack_remainder_behavior": "zero_pad_high_unused_values",
            "unpacked_value_dtype": "int8",
        },
    },
    "compressed_tensors_mxfp4_pack": {
        "implementation": "compressed_tensors_format_spec",
        "sources": {
            path: digest
            for path, digest in _COMPRESSED_TENSORS_SOURCE_SHA256.items()
            if path
            in {
                "src/compressed_tensors/compressors/mx_utils.py",
                "src/compressed_tensors/compressors/mxfp4/base.py",
                "src/compressed_tensors/compressors/nvfp4/base.py",
                "src/compressed_tensors/compressors/nvfp4/helpers.py",
                "src/compressed_tensors/quantization/lifecycle/forward_helpers.py",
            }
        },
        "contract": {
            "block_axis": "input_features",
            "block_remainder_behavior": "reject_nondivisible_by_32",
            "block_size": 32,
            "element_encoding": "e2m1_sign_bit_3_magnitude_bits_0_to_2",
            "element_values": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            "format_identifier": "mxfp4-pack-quantized",
            "nibble_order": "first_value_low_second_value_high",
            "packed_axis": "input_features",
            "packed_carrier_dtype": "uint8",
            "packed_values_per_byte": 2,
            "scale_carrier_dtype": "uint8",
            "scale_encoding": "e8m0_bias_127",
        },
    },
    "kimi_k25_automodel_int4": {
        "implementation": "nemo_automodel",
        "path": _AUTOMODEL_ADAPTER_PATH,
        "sha256": _AUTOMODEL_ADAPTER_SHA256,
        "contract": {
            "declared_group_size": 32,
            "group_count_formula": "ceil(input_features/32)",
            "group_remainder_behavior": "equal_reshape_or_runtime_error",
            "integration_admission": "reject_input_features_nondivisible_by_32",
            "logical_shape_dtype": "int64",
            "logical_shape_extent": 2,
            "nibble_order": "value_i_at_bits_4_times_i_lsb_first",
            "offset_binary_bias": 8,
            "packed_dtype": "int32",
            "packed_values_per_word": 8,
            "packed_remainder_behavior": "reject_nondivisible_by_8",
            "scale_dtype": "float16",
            "scale_shape_rounding": "ceil_group_count_with_equal_chunks",
        },
    },
    "modelopt_nvfp4_pack": {
        "implementation": "modelopt_lightning_producer",
        "sources": _MODELOPT_LIGHTNING_SOURCE_SHA256,
        "contract": {
            "block_axis": "input_features",
            "block_size": 16,
            "catalog_admission": "input_features_divisible_by_16",
            "element_encoding": "e2m1_sign_bit_3_magnitude_bits_0_to_2",
            "element_values": [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
            ],
            "global_scale_dtype": "float32",
            "nibble_order": "first_value_low_second_value_high",
            "packed_axis": "input_features",
            "packed_carrier_dtype": "uint8",
            "packed_values_per_byte": 2,
            "per_block_scale_dtype": "float8_e4m3fn",
            "source_remainder_behavior": "pad_input_features_to_block_size",
            "target_algorithm_binding": (
                "immutable_checkpoint_quantized_layers_and_config_group"
            ),
        },
    },
    "transformer_engine_native_mxfp8": {
        "implementation": "transformer_engine_root_runtime",
        "sources": _TE_SOURCE_SHA256,
        "contract": {
            "catalog_format_evidence": "none_native_storage_requires_normalization",
            "column_scale_shape": "[round_up(M/32,4),round_up(K,128)]",
            "compact_padding_fill": "unspecified_ignored",
            "compact_python_representation": "not_proven",
            "logical_admission": "M_and_K_exactly_divisible_by_32",
            "row_scale_shape": "[round_up(M,128),round_up(K/32,4)]",
            "scale_dtype": "e8m0",
            "storage_dtype": "uint8",
            "swizzle": "optional_128x4_tiles",
            "swizzled_padding_fill": "zero",
            "values_dtype": "e4m3",
        },
    },
}

_EXPECTED_ARTIFACTS: dict[str, dict[str, object]] = {
    "compressed_tensors_0_17_0": {
        "kind": "pinned_local_source",
        "source_contracts": [
            "compressed_tensors_int4_pack",
            "compressed_tensors_mxfp4_pack",
        ],
    },
    "qwen3_bf16": {
        "kind": "immutable_hf_metadata",
        "repository": "Qwen/Qwen3-30B-A3B",
        "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "config_sha256": (
            "2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297"
        ),
        "index_sha256": (
            "df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24"
        ),
        "header_manifest_sha256": (
            "72d48dbc90e484781cffc7962ae19ceb477bd252981b4c9554d7f5792107d970"
        ),
        "shards": 16,
        "tensors": 18867,
    },
    "kimi_k2": {
        "kind": "immutable_hf_metadata",
        "repository": "moonshotai/Kimi-K2-Base",
        "revision": "ce72df012259dcc55d945e890f815fe7ef69159c",
        "config_sha256": (
            "8c13ae1049df55f29b3bdcae69a562433f243ff70dac251d819ecad8dbdf7439"
        ),
        "index_sha256": (
            "c1f1d16c853f20467ae81361d2a92223650d39efa005f9c872a7cc14425ddcbc"
        ),
        "header_manifest_sha256": (
            "ff7de9c047659d7cbc0cbee8734e60dade5384d48bda8a3600e33eb84a69fe41"
        ),
        "shards": 61,
        "tensors": 139644,
    },
    "kimi_k25": {
        "kind": "immutable_hf_metadata",
        "repository": "moonshotai/Kimi-K2.5",
        "revision": "4d01dfe0332d63057c186e0b262165819efb6611",
        "config_sha256": (
            "acd5bb01a16f64b309599cd6ed196be056f613c99d6bc9300692b82cd10882f6"
        ),
        "index_sha256": (
            "bdba19b127c4d1dc57dc3b6f3366c10739c7e7f13baf3f5424b556469a4dbc1b"
        ),
        "header_manifest_sha256": (
            "1f869fba2e6a9c4de7376fb6b277f545a78f6e0276075748589c438e35374012"
        ),
        "shards": 64,
        "tensors": 208550,
    },
    "kimi_k3": {
        "kind": "immutable_hf_metadata",
        "repository": "moonshotai/Kimi-K3",
        "revision": "f831ab66814297da540d832a5235f8e904f29d06",
        "config_sha256": (
            "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
        ),
        "index_sha256": (
            "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
        ),
        "header_manifest_sha256": (
            "35fc99eb32a3bce794e86f9ac7c1f4cdf55df197e60444b0c8c47dc25b95594b"
        ),
        "shards": 96,
        "tensors": 497220,
    },
    "nemotron_lightning_nvfp4": {
        "kind": "immutable_hf_metadata",
        "repository": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "revision": "cc84af2fe71647d87f4486c064f320e1e7535243",
        "config_sha256": (
            "f1d98b530846087dc08b574a219713a94f945bf6583dc7230a19ebf1e8c50933"
        ),
        "index_sha256": (
            "3c3bc7efa8d658c2e909a0b9020eb0f72064e6647de348856af4dee9895bead9"
        ),
        "header_manifest_sha256": (
            "b70b7d010a9aea3783f6bca9081a59afa41a80a97ff51d8e0ced2f41fb5f6714"
        ),
        "shards": 52,
        "tensors": 18487,
    },
    "modelopt_0_44_0rc5": {
        "kind": "pinned_local_source",
        "source_contract": "modelopt_nvfp4_pack",
    },
    "qwen_a95b_fp8": {
        "kind": "immutable_hf_metadata",
        "repository": "Qwen/Qwen3.8-2.4T-A95B-FP8",
        "revision": "d2dc35658bcf77e66643428cb52e774cc3b5bd29",
        "config_sha256": (
            "b7396b749964c6afb5387c58e6425db8628e85f8ae66739d284eb1c8f42c4d4e"
        ),
        "index_sha256": (
            "67f75ab10833869c951b5c8e02ddcf4fa11974a8dcb950c51193680c90a4f77c"
        ),
        "header_manifest_sha256": (
            "cc5b309051da3d5fc508b8609247ce0f49aa0592839786cad9d7ddddfd8344c3"
        ),
        "mtp_header_byte_lengths": {
            "model-00185-of-00213.safetensors": 254184,
            "model-00186-of-00213.safetensors": 127080,
        },
        "catalog_admission": "both_logical_axes_divisible_by_128",
        "quant_method": "fp8",
        "remainder_evidence": "unsupported_not_observed",
        "shards": 213,
        "tensors": 287119,
        "weight_block_size": [128, 128],
    },
    "automodel_kimi_k25": {
        "kind": "pinned_local_source",
        "source_contract": "kimi_k25_automodel_int4",
    },
    "transformer_engine_mxfp8": {
        "kind": "pinned_local_source",
        "source_contract": "transformer_engine_native_mxfp8",
    },
}

_EXPECTED_CLAIM_ARTIFACTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("block-fp8.e4m3-f32-scale-inv-block128x128.v1", "values"): ("kimi_k2",),
    ("block-fp8.e4m3-f32-scale-inv-block128x128.v1", "inverse_scales"): ("kimi_k2",),
    ("block-fp8.e4m3-bf16-scale-inv-block128x128.v1", "values"): ("qwen_a95b_fp8",),
    ("block-fp8.e4m3-bf16-scale-inv-block128x128.v1", "inverse_scales"): (
        "qwen_a95b_fp8",
    ),
    ("packed-int4.i32-bf16-group32-shape-i32.v1", "packed_values"): (
        "kimi_k25",
        "compressed_tensors_0_17_0",
    ),
    ("packed-int4.i32-bf16-group32-shape-i32.v1", "group_scales"): (
        "kimi_k25",
        "compressed_tensors_0_17_0",
    ),
    ("packed-int4.i32-bf16-group32-shape-i32.v1", "logical_shape"): (
        "kimi_k25",
        "compressed_tensors_0_17_0",
    ),
    ("packed-int4.i32-f16-group32-shape-i64.v1", "packed_values"): (
        "automodel_kimi_k25",
    ),
    ("packed-int4.i32-f16-group32-shape-i64.v1", "group_scales"): (
        "automodel_kimi_k25",
    ),
    ("packed-int4.i32-f16-group32-shape-i64.v1", "logical_shape"): (
        "automodel_kimi_k25",
    ),
    ("mxfp4.u8-u8-block32-input-features.v1", "packed_values"): (
        "kimi_k3",
        "compressed_tensors_0_17_0",
    ),
    ("mxfp4.u8-u8-block32-input-features.v1", "block_scales"): (
        "kimi_k3",
        "compressed_tensors_0_17_0",
    ),
    ("nvfp4.u8-e4m3-f32-block16-input-features.v1", "packed_values"): (
        "nemotron_lightning_nvfp4",
        "modelopt_0_44_0rc5",
    ),
    ("nvfp4.u8-e4m3-f32-block16-input-features.v1", "block_scales"): (
        "nemotron_lightning_nvfp4",
        "modelopt_0_44_0rc5",
    ),
    ("nvfp4.u8-e4m3-f32-block16-input-features.v1", "global_scale"): (
        "nemotron_lightning_nvfp4",
        "modelopt_0_44_0rc5",
    ),
}

_EXPECTED_OBSERVATIONS: dict[str, tuple[str, str, str, str, str]] = {
    **{
        f"qwen3.main.{projection}": (
            "checkpoint.qwen3_bf16",
            "qwen3_bf16",
            "main",
            projection,
            "bf16.logical.v1",
        )
        for projection in ("gate", "up", "down")
    },
    **{
        f"k2.main.{projection}": (
            "checkpoint.kimi_k2",
            "kimi_k2",
            "main",
            projection,
            "block-fp8.e4m3-f32-scale-inv-block128x128.v1",
        )
        for projection in ("gate", "up", "down")
    },
    **{
        f"k25.checkpoint.main.{projection}": (
            "checkpoint.kimi_k25",
            "kimi_k25",
            "main",
            projection,
            "packed-int4.i32-bf16-group32-shape-i32.v1",
        )
        for projection in ("gate", "up", "down")
    },
    **{
        f"k25.automodel.main.{projection}": (
            "automodel.kimi_k25",
            "automodel_kimi_k25",
            "main",
            projection,
            "packed-int4.i32-f16-group32-shape-i64.v1",
        )
        for projection in ("gate", "up", "down")
    },
    **{
        f"k3.main.{projection}": (
            "checkpoint.kimi_k3",
            "kimi_k3",
            "main",
            projection,
            "mxfp4.u8-u8-block32-input-features.v1",
        )
        for projection in ("gate", "up", "down")
    },
    **{
        f"lightning.main.{projection}": (
            "checkpoint.nemotron_lightning_nvfp4",
            "nemotron_lightning_nvfp4",
            "main",
            projection,
            "nvfp4.u8-e4m3-f32-block16-input-features.v1",
        )
        for projection in ("up", "down")
    },
    **{
        f"a95b.{graph.replace('.', '-')}.{projection}": (
            "checkpoint.qwen_a95b_fp8",
            "qwen_a95b_fp8",
            graph,
            projection,
            "block-fp8.e4m3-bf16-scale-inv-block128x128.v1",
        )
        for graph in ("main", "mtp.0")
        for projection in ("gate", "up", "down")
    },
}

_EXPECTED_METADATA_LOCATIONS: dict[str, str] = {
    **{
        f"qwen3.main.{projection}": "model-00001-of-00016.safetensors"
        for projection in ("gate", "up", "down")
    },
    **{
        f"k2.main.{projection}": "model-2-of-61.safetensors"
        for projection in ("gate", "up", "down")
    },
    **{
        f"k25.checkpoint.main.{projection}": "model-00002-of-000064.safetensors"
        for projection in ("gate", "up", "down")
    },
    **{
        f"k25.automodel.main.{projection}": _AUTOMODEL_ADAPTER_PATH
        for projection in ("gate", "up", "down")
    },
    **{
        f"k3.main.{projection}": "model-00002-of-000096.safetensors"
        for projection in ("gate", "up", "down")
    },
    **{
        f"lightning.main.{projection}": "model-00002-of-00052.safetensors"
        for projection in ("up", "down")
    },
    **{
        f"a95b.main.{projection}": (
            "model-00002-of-00213.safetensors"
            if projection == "down"
            else "model-00001-of-00213.safetensors"
        )
        for projection in ("gate", "up", "down")
    },
    **{
        f"a95b.mtp-0.{projection}": (
            "model-00186-of-00213.safetensors"
            if projection == "down"
            else "model-00185-of-00213.safetensors"
        )
        for projection in ("gate", "up", "down")
    },
}

_EXPECTED_RAW_PREFIXES: dict[str, str] = {
    **{
        f"qwen3.main.{projection}": (f"model.layers.0.mlp.experts.0.{projection}_proj")
        for projection in ("gate", "up", "down")
    },
    **{
        f"k2.main.{projection}": (f"model.layers.1.mlp.experts.0.{projection}_proj")
        for projection in ("gate", "up", "down")
    },
    **{
        f"k25.checkpoint.main.{projection}": (
            f"language_model.model.layers.1.mlp.experts.0.{projection}_proj"
        )
        for projection in ("gate", "up", "down")
    },
    **{
        f"k25.automodel.main.{projection}": (
            f"language_model.model.layers.5.mlp.experts.0.{projection}_proj"
        )
        for projection in ("gate", "up", "down")
    },
    "k3.main.gate": ("language_model.model.layers.1.block_sparse_moe.experts.0.w1"),
    "k3.main.up": "language_model.model.layers.1.block_sparse_moe.experts.0.w3",
    "k3.main.down": ("language_model.model.layers.1.block_sparse_moe.experts.0.w2"),
    **{
        f"lightning.main.{projection}": (
            f"backbone.layers.1.mixer.experts.0.{projection}_proj"
        )
        for projection in ("up", "down")
    },
    **{
        f"a95b.main.{projection}": (f"model.layers.0.mlp.experts.0.{projection}_proj")
        for projection in ("gate", "up", "down")
    },
    **{
        f"a95b.mtp-0.{projection}": (f"mtp.layers.0.mlp.experts.0.{projection}_proj")
        for projection in ("gate", "up", "down")
    },
}

_EXPECTED_RAW_SUFFIXES: dict[str, dict[str, str]] = {
    "bf16.logical.v1": {"logical_values": ".weight"},
    "block-fp8.e4m3-f32-scale-inv-block128x128.v1": {
        "values": ".weight",
        "inverse_scales": ".weight_scale_inv",
    },
    "block-fp8.e4m3-bf16-scale-inv-block128x128.v1": {
        "values": ".weight",
        "inverse_scales": ".weight_scale_inv",
    },
    "packed-int4.i32-bf16-group32-shape-i32.v1": {
        "packed_values": ".weight_packed",
        "group_scales": ".weight_scale",
        "logical_shape": ".weight_shape",
    },
    "packed-int4.i32-f16-group32-shape-i64.v1": {
        "packed_values": ".weight_packed",
        "group_scales": ".weight_scale",
        "logical_shape": ".weight_shape",
    },
    "mxfp4.u8-u8-block32-input-features.v1": {
        "packed_values": ".weight_packed",
        "block_scales": ".weight_scale",
    },
    "nvfp4.u8-e4m3-f32-block16-input-features.v1": {
        "packed_values": ".weight",
        "block_scales": ".weight_scale",
        "global_scale": ".weight_scale_2",
    },
}

_EXPECTED_SOURCE_DTYPES: dict[str, tuple[str, ...]] = {
    **{f"qwen3.main.{projection}": ("BF16",) for projection in ("gate", "up", "down")},
    **{
        f"k2.main.{projection}": ("F8_E4M3", "F32")
        for projection in ("gate", "up", "down")
    },
    **{
        f"k25.checkpoint.main.{projection}": ("I32", "BF16", "I32")
        for projection in ("gate", "up", "down")
    },
    **{
        f"k25.automodel.main.{projection}": ("int32", "float16", "int64")
        for projection in ("gate", "up", "down")
    },
    **{f"k3.main.{projection}": ("U8", "U8") for projection in ("gate", "up", "down")},
    **{
        f"lightning.main.{projection}": ("U8", "F8_E4M3", "F32")
        for projection in ("up", "down")
    },
    **{
        f"a95b.{graph.replace('.', '-')}.{projection}": ("F8_E4M3", "BF16")
        for graph in ("main", "mtp.0")
        for projection in ("gate", "up", "down")
    },
}


class EvidenceError(ValueError):
    """Evidence is incomplete or differs from its reviewed immutable pin."""


@dataclass(frozen=True)
class CheckpointObservationSpec:
    """Reviewed selection rule for one observation in raw checkpoint metadata."""

    format_id: str
    graph: str
    logical_shape_config_paths: tuple[tuple[str, ...], ...]
    observation_id: str
    producer: str
    projection: str
    raw_prefix: str
    source_dtypes: tuple[str, ...]
    suffixes: tuple[str, ...]
    logical_axes: tuple[str, ...] = ("output_features", "input_features")


@dataclass(frozen=True)
class ConfigMembershipRequirement:
    """Reviewed representative members and cardinality for one config array."""

    path: tuple[str, ...]
    members: tuple[str, ...]
    cardinality: int


@dataclass(frozen=True)
class CheckpointArtifactSpec:
    """Immutable identity and observation rules for one staged checkpoint."""

    artifact_id: str
    artifact: Mapping[str, object]
    observations: tuple[CheckpointObservationSpec, ...]
    config_requirements: tuple[tuple[tuple[str, ...], object], ...] = ()
    config_membership_requirements: tuple[ConfigMembershipRequirement, ...] = ()


def _object_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise EvidenceError(f"{field} must be a JSON object with string keys")
    return cast(Mapping[str, object], value)


def _object_sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvidenceError(f"{field} must be a JSON array")
    return cast(Sequence[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{field} must be a non-empty string")
    return value


def _positive_int_sequence(value: object, field: str) -> tuple[int, ...]:
    values = _object_sequence(value, field)
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in values
    ):
        raise EvidenceError(f"{field} must contain positive integers")
    return tuple(cast(int, item) for item in values)


def _read_metadata_json(path: Path, field: str) -> tuple[Mapping[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"missing staged metadata file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{field} is not valid UTF-8 JSON") from error
    return _object_mapping(value, field), raw


def _canonical_compact_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _resolve_config_path(config: Mapping[str, object], path: tuple[str, ...]) -> object:
    value: object = config
    for name in path:
        value = _object_mapping(value, ".".join(path)).get(name)
    return value


def _validated_shard_name(value: object, field: str) -> str:
    shard_name = _string(value, field)
    path = PurePosixPath(shard_name)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.name != shard_name
        or path.suffix != ".safetensors"
    ):
        raise EvidenceError(f"{field} must be a traversal-safe safetensors basename")
    return shard_name


def capture_staged_checkpoint_evidence(
    staged_metadata_root: Path,
    spec: CheckpointArtifactSpec,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    tuple[dict[str, str], ...],
]:
    """Derive reviewed observations from raw config, index, and header metadata."""
    staged_metadata_root = staged_metadata_root.resolve(strict=True)
    if not staged_metadata_root.is_dir():
        raise EvidenceError("staged metadata root must be a directory")
    if _ARTIFACT_ID_PATTERN.fullmatch(spec.artifact_id) is None:
        raise EvidenceError(f"invalid staged artifact id: {spec.artifact_id}")
    checkpoints_root = staged_metadata_root / STAGED_CHECKPOINT_DIRECTORY
    if checkpoints_root.is_symlink():
        raise EvidenceError("staged checkpoints directory must not be a symlink")
    try:
        checkpoints_root = checkpoints_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise EvidenceError("missing staged checkpoints directory") from error
    if not checkpoints_root.is_dir():
        raise EvidenceError("staged checkpoints path must be a directory")
    artifact_root = checkpoints_root / spec.artifact_id
    if artifact_root.is_symlink():
        raise EvidenceError(
            f"symlinked staged artifact is forbidden: {spec.artifact_id}"
        )
    try:
        artifact_root = artifact_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise EvidenceError(
            f"missing staged artifact directory: {spec.artifact_id}"
        ) from error
    if not artifact_root.is_dir() or artifact_root.parent != checkpoints_root:
        raise EvidenceError(f"invalid staged artifact directory: {spec.artifact_id}")
    config_path = artifact_root / STAGED_CONFIG_FILENAME
    index_path = artifact_root / STAGED_INDEX_FILENAME
    header_path = artifact_root / STAGED_HEADER_MANIFEST_FILENAME
    config, raw_config = _read_metadata_json(config_path, f"{spec.artifact_id}.config")
    index, raw_index = _read_metadata_json(index_path, f"{spec.artifact_id}.index")
    header_manifest, raw_header_manifest = _read_metadata_json(
        header_path, f"{spec.artifact_id}.header_manifest"
    )
    opened_metadata: list[tuple[Path, bytes]] = [
        (config_path, raw_config),
        (index_path, raw_index),
        (header_path, raw_header_manifest),
    ]

    expected = dict(spec.artifact)
    observed_hashes = {
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "index_sha256": hashlib.sha256(raw_index).hexdigest(),
        "header_manifest_sha256": hashlib.sha256(
            _canonical_compact_json(header_manifest)
        ).hexdigest(),
    }
    for field, observed in observed_hashes.items():
        if expected.get(field) != observed:
            raise EvidenceError(
                f"{spec.artifact_id} {field} differs from immutable pin"
            )

    weight_map = _object_mapping(index.get("weight_map"), "index.weight_map")
    if not weight_map or any(
        not isinstance(tensor_name, str) or not tensor_name
        for tensor_name in weight_map
    ):
        raise EvidenceError("index.weight_map must contain non-empty tensor names")
    validated_weight_map = {
        tensor_name: _validated_shard_name(
            shard_name, f"index.weight_map[{tensor_name}]"
        )
        for tensor_name, shard_name in weight_map.items()
    }
    if set(header_manifest) != set(weight_map):
        raise EvidenceError("index/header tensor keys differ")
    shard_names = set(validated_weight_map.values())
    if expected.get("tensors") != len(header_manifest):
        raise EvidenceError(f"{spec.artifact_id} tensor count differs from pin")
    if expected.get("shards") != len(shard_names):
        raise EvidenceError(f"{spec.artifact_id} shard count differs from pin")

    parsed_headers: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    for tensor_name, raw_header in header_manifest.items():
        header = _object_mapping(raw_header, f"header[{tensor_name}]")
        if set(header) != {"dtype", "shape", "shard"}:
            raise EvidenceError(f"header fields differ for {tensor_name}")
        dtype = _string(header.get("dtype"), f"header[{tensor_name}].dtype")
        shape = _positive_int_sequence_allow_scalar(
            header.get("shape"), f"header[{tensor_name}].shape"
        )
        shard = _string(header.get("shard"), f"header[{tensor_name}].shard")
        indexed_shard = validated_weight_map[tensor_name]
        if shard != indexed_shard:
            raise EvidenceError(
                f"index/header shard mismatch for {tensor_name}: "
                f"{indexed_shard!r} != {shard!r}"
            )
        parsed_headers[tensor_name] = (dtype, shard, shape)

    raw_config_fields: list[dict[str, object]] = []
    for config_path_parts, required_value in spec.config_requirements:
        observed_value = _resolve_config_path(config, config_path_parts)
        if observed_value != required_value:
            dotted_path = ".".join(config_path_parts)
            raise EvidenceError(
                f"{spec.artifact_id} config {dotted_path} differs from pin"
            )
        raw_config_fields.append(
            {"path": ".".join(config_path_parts), "value": observed_value}
        )

    raw_config_memberships: list[dict[str, object]] = []
    for requirement in spec.config_membership_requirements:
        dotted_path = ".".join(requirement.path)
        values = _object_sequence(
            _resolve_config_path(config, requirement.path),
            f"{spec.artifact_id} config {dotted_path}",
        )
        if any(not isinstance(value, str) or not value for value in values):
            raise EvidenceError(
                f"{spec.artifact_id} config {dotted_path} must contain strings"
            )
        if len(values) != requirement.cardinality or any(
            member not in values for member in requirement.members
        ):
            raise EvidenceError(
                f"{spec.artifact_id} config {dotted_path} membership differs from pin"
            )
        raw_config_memberships.append(
            {
                "cardinality": len(values),
                "path": dotted_path,
                "representative_members": list(requirement.members),
            }
        )

    expected_header_lengths = expected.get("mtp_header_byte_lengths")
    if expected_header_lengths is not None:
        lengths_path = artifact_root / STAGED_HEADER_LENGTHS_FILENAME
        header_lengths, raw_header_lengths = _read_metadata_json(
            lengths_path, f"{spec.artifact_id}.header_byte_lengths"
        )
        opened_metadata.append((lengths_path, raw_header_lengths))
        if header_lengths != expected_header_lengths:
            raise EvidenceError(
                f"{spec.artifact_id} header byte lengths differ from pin"
            )

    catalog_by_id = {
        descriptor.format_id: descriptor for descriptor in SOURCE_FORMAT_CATALOG
    }
    observations: list[dict[str, object]] = []
    for observation_spec in spec.observations:
        descriptor = catalog_by_id.get(observation_spec.format_id)
        if descriptor is None:
            raise EvidenceError(f"unknown source format: {observation_spec.format_id}")
        roles = tuple(component.role for component in descriptor.components)
        if len(roles) != len(observation_spec.suffixes) or len(roles) != len(
            observation_spec.source_dtypes
        ):
            raise EvidenceError(
                f"component selection is incomplete for {observation_spec.observation_id}"
            )
        components: list[dict[str, object]] = []
        raw_shapes: dict[str, tuple[int, ...]] = {}
        raw_names: list[str] = []
        for role, suffix, expected_dtype in zip(
            roles,
            observation_spec.suffixes,
            observation_spec.source_dtypes,
            strict=True,
        ):
            raw_name = observation_spec.raw_prefix + suffix
            raw_names.append(raw_name)
            try:
                source_dtype, shard, shape = parsed_headers[raw_name]
            except KeyError as error:
                raise EvidenceError(
                    f"missing raw sibling for {observation_spec.observation_id}: "
                    f"{raw_name}"
                ) from error
            if source_dtype != expected_dtype:
                raise EvidenceError(
                    f"exact raw dtype differs for {observation_spec.observation_id}: "
                    f"{role}"
                )
            raw_shapes[role] = shape
            components.append(
                {
                    "metadata_location": shard,
                    "raw_name": raw_name,
                    "role": role,
                    "shape": list(shape),
                    "source_dtype": source_dtype,
                }
            )
        if len(observation_spec.logical_shape_config_paths) != len(
            observation_spec.logical_axes
        ):
            raise EvidenceError(
                "logical config paths do not match axis count for "
                f"{observation_spec.observation_id}"
            )
        logical_shape_values: list[int] = []
        for config_path_parts in observation_spec.logical_shape_config_paths:
            value = _resolve_config_path(config, config_path_parts)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise EvidenceError(
                    f"{spec.artifact_id} config {'.'.join(config_path_parts)} "
                    "must be a positive integer"
                )
            logical_shape_values.append(value)
        logical_shape = tuple(logical_shape_values)
        for component_descriptor in descriptor.components:
            resolved_shape = tuple(
                extent
                for _, extent in resolve_component_axes(
                    component_descriptor,
                    logical_axes=observation_spec.logical_axes,
                    logical_shape=logical_shape,
                )
            )
            if raw_shapes[component_descriptor.role] != resolved_shape:
                raise EvidenceError(
                    f"component shape differs for {observation_spec.observation_id}: "
                    f"{component_descriptor.role}"
                )
        observations.append(
            {
                "artifact": spec.artifact_id,
                "components": components,
                "format_id": observation_spec.format_id,
                "graph": observation_spec.graph,
                "logical_axes": list(observation_spec.logical_axes),
                "logical_shape": list(logical_shape),
                "observation_id": observation_spec.observation_id,
                "producer": observation_spec.producer,
                "projection": observation_spec.projection,
                "raw_siblings": raw_names,
            }
        )
    receipt = tuple(
        {
            "path": path.relative_to(staged_metadata_root).as_posix(),
            "sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        }
        for path, raw in opened_metadata
    )
    captured_artifact = dict(expected)
    if raw_config_fields:
        captured_artifact["raw_config_fields"] = raw_config_fields
    if raw_config_memberships:
        captured_artifact["raw_config_memberships"] = raw_config_memberships
    return captured_artifact, observations, receipt


def _axis_payload(
    axis: LogicalComponentAxisSpec | LiteralComponentAxisSpec,
) -> dict[str, object]:
    if isinstance(axis, LiteralComponentAxisSpec):
        return {
            "axis": axis.axis_name,
            "divisor": axis.extent,
            "kind": "literal",
            "rounding": AxisExtentRounding.EXACT.value,
        }
    return {
        "axis": axis.logical_axis,
        "divisor": axis.divisor,
        "kind": "logical",
        "rounding": axis.rounding.value,
    }


def _component_contract(component: ComponentDescriptor) -> dict[str, object]:
    axes = (
        None
        if component.component_axes is None
        else [_axis_payload(axis) for axis in component.component_axes]
    )
    return {
        "axes": axes,
        "dtype": component.dtype,
        "encoding": component.encoding,
    }


def _catalog_payload(
    catalog: Sequence[FormatDescriptor] = SOURCE_FORMAT_CATALOG,
) -> list[dict[str, object]]:
    return [
        {
            "components": [
                {"role": component.role, **_component_contract(component)}
                for component in descriptor.components
            ],
            "family": descriptor.family,
            "format_id": descriptor.format_id,
        }
        for descriptor in catalog
    ]


def _expected_capture_receipt() -> dict[str, object]:
    opened_metadata: list[dict[str, str]] = []
    for artifact_id in (
        "qwen3_bf16",
        "kimi_k2",
        "kimi_k25",
        "kimi_k3",
        "nemotron_lightning_nvfp4",
        "qwen_a95b_fp8",
    ):
        artifact = _EXPECTED_ARTIFACTS[artifact_id]
        prefix = f"{STAGED_CHECKPOINT_DIRECTORY}/{artifact_id}"
        opened_metadata.extend(
            (
                {
                    "path": f"{prefix}/{STAGED_CONFIG_FILENAME}",
                    "sha256": f"sha256:{artifact['config_sha256']}",
                },
                {
                    "path": f"{prefix}/{STAGED_INDEX_FILENAME}",
                    "sha256": f"sha256:{artifact['index_sha256']}",
                },
                {
                    "path": f"{prefix}/{STAGED_HEADER_MANIFEST_FILENAME}",
                    "sha256": f"sha256:{artifact['header_manifest_sha256']}",
                },
            )
        )
        header_lengths = artifact.get("mtp_header_byte_lengths")
        if header_lengths is not None:
            opened_metadata.append(
                {
                    "path": f"{prefix}/{STAGED_HEADER_LENGTHS_FILENAME}",
                    "sha256": "sha256:"
                    + hashlib.sha256(
                        _canonical_compact_json(header_lengths)
                    ).hexdigest(),
                }
            )
    return {
        "opened_metadata": opened_metadata,
        "schema_version": CAPTURE_RECEIPT_SCHEMA,
    }


def _expected_source_artifacts() -> dict[str, dict[str, object]]:
    artifacts = {
        artifact_id: dict(artifact)
        for artifact_id, artifact in _EXPECTED_ARTIFACTS.items()
    }
    for spec in checkpoint_artifact_specs():
        if spec.config_requirements:
            artifacts[spec.artifact_id]["raw_config_fields"] = [
                {"path": ".".join(path), "value": value}
                for path, value in spec.config_requirements
            ]
        if spec.config_membership_requirements:
            artifacts[spec.artifact_id]["raw_config_memberships"] = [
                {
                    "cardinality": requirement.cardinality,
                    "path": ".".join(requirement.path),
                    "representative_members": list(requirement.members),
                }
                for requirement in spec.config_membership_requirements
            ]
    return artifacts


def validate_source_format_evidence(document: Mapping[str, object]) -> None:
    """Fail closed unless staged metadata proves every literal catalog field."""
    if set(document) != {
        "artifacts",
        "capture_mode",
        "capture_receipt",
        "catalog",
        "claims",
        "observations",
        "schema_version",
    }:
        raise EvidenceError("source evidence fields are incomplete or unexpected")
    if document.get("schema_version") != SOURCE_EVIDENCE_SCHEMA:
        raise EvidenceError("source evidence has an unsupported schema_version")
    if document.get("capture_mode") != "metadata_only_no_tensor_payloads":
        raise EvidenceError("source evidence must declare metadata-only capture")
    if document.get("catalog") != _catalog_payload():
        raise EvidenceError("source evidence catalog differs from literal catalog")
    if document.get("capture_receipt") != _expected_capture_receipt():
        raise EvidenceError("source evidence capture receipt differs from raw metadata")

    artifacts = _object_mapping(document.get("artifacts"), "artifacts")
    expected_artifacts = _expected_source_artifacts()
    if artifacts != expected_artifacts:
        if artifacts.get("qwen_a95b_fp8") != expected_artifacts["qwen_a95b_fp8"]:
            raise EvidenceError(
                "A95B block geometry or exact remainder evidence is missing"
            )
        raise EvidenceError("source artifact evidence differs from immutable pins")

    claims = _object_sequence(document.get("claims"), "claims")
    expected_components = {
        (descriptor.format_id, component.role): _component_contract(component)
        for descriptor in SOURCE_FORMAT_CATALOG
        if descriptor is not BF16_FORMAT and descriptor is not MXFP8_FORMAT
        for component in descriptor.components
    }
    claims_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for index, value in enumerate(claims):
        claim = _object_mapping(value, f"claims[{index}]")
        if set(claim) != {"contract", "evidence", "format_id", "role"}:
            raise EvidenceError(f"claims[{index}] fields are incomplete or unexpected")
        key = (
            _string(claim.get("format_id"), f"claims[{index}].format_id"),
            _string(claim.get("role"), f"claims[{index}].role"),
        )
        if key in claims_by_key:
            raise EvidenceError(f"duplicate source-format claim: {key}")
        claims_by_key[key] = claim
    if set(claims_by_key) != set(expected_components):
        raise EvidenceError("catalog component claims are incomplete or unexpected")
    for key, expected_contract in expected_components.items():
        claim = claims_by_key[key]
        if claim.get("contract") != expected_contract:
            raise EvidenceError(f"claim contract differs from catalog: {key}")
        evidence = tuple(
            _string(item, f"claim {key} evidence")
            for item in _object_sequence(claim.get("evidence"), f"claim {key} evidence")
        )
        if evidence != _EXPECTED_CLAIM_ARTIFACTS[key]:
            raise EvidenceError(f"claim evidence differs from reviewed sources: {key}")

    observations = _object_sequence(document.get("observations"), "observations")
    observations_by_id: dict[str, Mapping[str, object]] = {}
    catalog_by_id = {
        descriptor.format_id: descriptor for descriptor in SOURCE_FORMAT_CATALOG
    }
    for index, value in enumerate(observations):
        observation = _object_mapping(value, f"observations[{index}]")
        if set(observation) != {
            "artifact",
            "components",
            "format_id",
            "graph",
            "logical_axes",
            "logical_shape",
            "observation_id",
            "producer",
            "projection",
            "raw_siblings",
        }:
            raise EvidenceError(
                f"observations[{index}] fields are incomplete or unexpected"
            )
        observation_id = _string(
            observation.get("observation_id"),
            f"observations[{index}].observation_id",
        )
        if observation_id in observations_by_id:
            raise EvidenceError(f"duplicate observation_id: {observation_id}")
        observations_by_id[observation_id] = observation
        expected_identity = _EXPECTED_OBSERVATIONS.get(observation_id)
        observed_identity = (
            observation.get("producer"),
            observation.get("artifact"),
            observation.get("graph"),
            observation.get("projection"),
            observation.get("format_id"),
        )
        if expected_identity != observed_identity:
            raise EvidenceError(f"unexpected observation identity: {observation_id}")
        format_id = cast(str, observation["format_id"])
        descriptor = catalog_by_id[format_id]
        logical_axes = tuple(
            _string(item, f"{observation_id}.logical_axes")
            for item in _object_sequence(
                observation.get("logical_axes"), f"{observation_id}.logical_axes"
            )
        )
        if logical_axes != ("output_features", "input_features"):
            raise EvidenceError(f"logical axes differ for {observation_id}")
        logical_shape = _positive_int_sequence(
            observation.get("logical_shape"), f"{observation_id}.logical_shape"
        )
        components = _object_sequence(
            observation.get("components"), f"{observation_id}.components"
        )
        component_by_role: dict[str, Mapping[str, object]] = {}
        raw_names: list[str] = []
        for component_index, component_value in enumerate(components):
            component = _object_mapping(
                component_value,
                f"{observation_id}.components[{component_index}]",
            )
            if set(component) != {
                "metadata_location",
                "raw_name",
                "role",
                "shape",
                "source_dtype",
            }:
                raise EvidenceError(
                    f"{observation_id}.components[{component_index}] fields are "
                    "incomplete or unexpected"
                )
            role = _string(
                component.get("role"),
                f"{observation_id}.components[{component_index}].role",
            )
            if role in component_by_role:
                raise EvidenceError(f"duplicate observed component role: {role}")
            component_by_role[role] = component
            raw_names.append(
                _string(
                    component.get("raw_name"),
                    f"{observation_id}.components[{component_index}].raw_name",
                )
            )
            metadata_location = _string(
                component.get("metadata_location"),
                f"{observation_id}.components[{component_index}].metadata_location",
            )
            if metadata_location != _EXPECTED_METADATA_LOCATIONS[observation_id]:
                raise EvidenceError(f"metadata location differs for {observation_id}")
        expected_roles = tuple(component.role for component in descriptor.components)
        if tuple(component_by_role) != expected_roles:
            raise EvidenceError(f"component roles are incomplete for {observation_id}")
        siblings = tuple(
            _string(item, f"{observation_id}.raw_siblings")
            for item in _object_sequence(
                observation.get("raw_siblings"), f"{observation_id}.raw_siblings"
            )
        )
        if siblings != tuple(raw_names) or len(set(siblings)) != len(siblings):
            raise EvidenceError(f"raw sibling set differs for {observation_id}")
        expected_names = tuple(
            _EXPECTED_RAW_PREFIXES[observation_id]
            + _EXPECTED_RAW_SUFFIXES[format_id][role]
            for role in expected_roles
        )
        if tuple(raw_names) != expected_names:
            raise EvidenceError(f"raw component names differ for {observation_id}")
        for component_index, component_descriptor in enumerate(descriptor.components):
            component = component_by_role[component_descriptor.role]
            source_dtype = _string(
                component.get("source_dtype"),
                f"{observation_id}.{component_descriptor.role}.source_dtype",
            )
            if source_dtype != _EXPECTED_SOURCE_DTYPES[observation_id][component_index]:
                raise EvidenceError(
                    f"exact raw dtype differs for {observation_id}: "
                    f"{component_descriptor.role}"
                )
            observed_shape = _positive_int_sequence_allow_scalar(
                component.get("shape"),
                f"{observation_id}.{component_descriptor.role}.shape",
            )
            try:
                expected_shape = tuple(
                    extent
                    for _, extent in resolve_component_axes(
                        component_descriptor,
                        logical_axes=logical_axes,
                        logical_shape=logical_shape,
                    )
                )
            except ValueError as error:
                raise EvidenceError(
                    f"component axes cannot resolve for {observation_id}: {error}"
                ) from error
            if observed_shape != expected_shape:
                raise EvidenceError(
                    f"component shape differs for {observation_id}: "
                    f"{component_descriptor.role}"
                )
    if set(observations_by_id) != set(_EXPECTED_OBSERVATIONS):
        raise EvidenceError("representative source observations are incomplete")


def _positive_int_sequence_allow_scalar(value: object, field: str) -> tuple[int, ...]:
    values = _object_sequence(value, field)
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in values
    ):
        raise EvidenceError(f"{field} must contain positive integers")
    return tuple(cast(int, item) for item in values)


def validate_producer_implementation_evidence(
    document: Mapping[str, object],
) -> None:
    """Fail closed on any producer or TE provenance identity mismatch."""
    if set(document) != {
        "implementations",
        "runtime_inspection",
        "schema_version",
        "source_contracts",
    }:
        raise EvidenceError("producer evidence fields are incomplete or unexpected")
    if document.get("schema_version") != PRODUCER_EVIDENCE_SCHEMA:
        raise EvidenceError("producer evidence has an unsupported schema_version")
    implementations = _object_mapping(
        document.get("implementations"), "implementations"
    )
    if implementations != _EXPECTED_IMPLEMENTATIONS:
        raise EvidenceError("producer implementation evidence differs from exact pins")
    source_contracts = _object_mapping(
        document.get("source_contracts"), "source_contracts"
    )
    if source_contracts != _EXPECTED_SOURCE_CONTRACTS:
        raise EvidenceError("producer source contracts differ from exact pins")
    runtime = _object_mapping(document.get("runtime_inspection"), "runtime_inspection")
    if runtime != {
        "package_identity": ROOT_TE_PACKAGE,
        "source_revision": ROOT_TE_REVISION,
        "status": "matched_root_lock",
    }:
        raise EvidenceError(
            "effective Transformer Engine runtime inspection is required and must "
            "match the NeMo-RL root lock"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _gitlink(repository: Path, relative_path: str) -> str:
    entry = _git(repository, "ls-tree", "HEAD", "--", relative_path)
    fields = entry.split()
    if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
        raise EvidenceError(f"missing gitlink for {relative_path}")
    return fields[2]


def _locked_te_identity(lock_path: Path) -> tuple[str, str]:
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise EvidenceError(f"{lock_path} has no package array")
    identities: set[tuple[str, str]] = set()
    for raw_package in packages:
        if not isinstance(raw_package, dict) or raw_package.get("name") != (
            "transformer-engine"
        ):
            continue
        version = raw_package.get("version")
        source = raw_package.get("source")
        git_source = source.get("git") if isinstance(source, dict) else None
        if (
            isinstance(version, str)
            and isinstance(git_source, str)
            and "#" in git_source
        ):
            identities.add((version, git_source.rsplit("#", maxsplit=1)[1]))
    if len(identities) != 1:
        raise EvidenceError(
            f"{lock_path} must contain exactly one git-pinned Transformer Engine"
        )
    return identities.pop()


def _locked_registry_wheel_sha256(
    lock_path: Path,
    *,
    package_name: str,
    package_version: str,
) -> str:
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise EvidenceError(f"{lock_path} has no package array")
    matching_packages = [
        raw_package
        for raw_package in packages
        if isinstance(raw_package, dict)
        and raw_package.get("name") == package_name
        and raw_package.get("version") == package_version
    ]
    if len(matching_packages) != 1:
        raise EvidenceError(
            f"{lock_path} must contain exactly one {package_name} {package_version}"
        )
    wheels = matching_packages[0].get("wheels")
    if not isinstance(wheels, list):
        raise EvidenceError(f"{package_name} {package_version} has no locked wheels")
    expected_filename = (
        f"{package_name.replace('-', '_')}-{package_version}-py3-none-any.whl"
    )
    hashes = {
        raw_wheel.get("hash")
        for raw_wheel in wheels
        if isinstance(raw_wheel, dict)
        and isinstance(raw_wheel.get("url"), str)
        and cast(str, raw_wheel["url"]).endswith(expected_filename)
        and isinstance(raw_wheel.get("hash"), str)
    }
    if len(hashes) != 1:
        raise EvidenceError(
            f"{package_name} {package_version} must have one universal wheel pin"
        )
    locked_hash = cast(str, hashes.pop())
    if not locked_hash.startswith("sha256:"):
        raise EvidenceError(
            f"{package_name} {package_version} wheel must use a SHA256 pin"
        )
    return locked_hash.removeprefix("sha256:")


def _installed_te_identity() -> tuple[str, str]:
    try:
        distribution = importlib.metadata.distribution("transformer-engine")
    except importlib.metadata.PackageNotFoundError as error:
        raise EvidenceError("Transformer Engine runtime is not installed") from error
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise EvidenceError("Transformer Engine runtime has no direct_url.json")
    direct_url = json.loads(direct_url_text)
    if not isinstance(direct_url, dict):
        raise EvidenceError("Transformer Engine direct_url.json is not an object")
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict):
        raise EvidenceError("Transformer Engine runtime has no VCS identity")
    commit_id = vcs_info.get("commit_id")
    if not isinstance(commit_id, str):
        raise EvidenceError("Transformer Engine runtime has no VCS commit")
    return distribution.version, commit_id


def capture_producer_implementation_evidence(
    repository_root: Path,
    compressed_tensors_source_root: Path,
    modelopt_lightning_source_root: Path,
    transformer_engine_source_root: Path,
    *,
    inspect_runtime: bool,
) -> dict[str, object]:
    """Read pinned local repositories, locks, and source text identities."""
    repository_root = repository_root.resolve(strict=True)
    compressed_tensors_source_root = compressed_tensors_source_root.resolve(strict=True)
    modelopt_lightning_source_root = modelopt_lightning_source_root.resolve(strict=True)
    transformer_engine_source_root = transformer_engine_source_root.resolve(strict=True)
    bridge_relative = _EXPECTED_IMPLEMENTATIONS["megatron_bridge"]["path"]
    automodel_relative = _EXPECTED_IMPLEMENTATIONS["nemo_automodel"]["path"]
    bridge_root = repository_root / bridge_relative
    automodel_root = repository_root / automodel_relative
    mcore_relative_to_bridge = "3rdparty/Megatron-LM"
    mcore_root = bridge_root / mcore_relative_to_bridge

    observed_implementations = {
        "compressed_tensors_format_spec": {
            **_EXPECTED_IMPLEMENTATIONS["compressed_tensors_format_spec"],
            "head_revision": _git(compressed_tensors_source_root, "rev-parse", "HEAD"),
        },
        "megatron_bridge": {
            **_EXPECTED_IMPLEMENTATIONS["megatron_bridge"],
            "gitlink_revision": _gitlink(repository_root, bridge_relative),
            "head_revision": _git(bridge_root, "rev-parse", "HEAD"),
        },
        "nemo_automodel": {
            **_EXPECTED_IMPLEMENTATIONS["nemo_automodel"],
            "gitlink_revision": _gitlink(repository_root, automodel_relative),
            "head_revision": _git(automodel_root, "rev-parse", "HEAD"),
        },
        "megatron_core": {
            **_EXPECTED_IMPLEMENTATIONS["megatron_core"],
            "gitlink_revision": _gitlink(bridge_root, mcore_relative_to_bridge),
            "head_revision": _git(mcore_root, "rev-parse", "HEAD"),
        },
        "modelopt_lightning_producer": {
            **_EXPECTED_IMPLEMENTATIONS["modelopt_lightning_producer"],
            "head_revision": _git(modelopt_lightning_source_root, "rev-parse", "HEAD"),
        },
        "transformer_engine_root_runtime": {
            **_EXPECTED_IMPLEMENTATIONS["transformer_engine_root_runtime"],
        },
        "transformer_engine_bridge_source": {
            **_EXPECTED_IMPLEMENTATIONS["transformer_engine_bridge_source"],
        },
    }

    root_lock_identity = _locked_te_identity(repository_root / "uv.lock")
    if root_lock_identity != (ROOT_TE_PACKAGE, ROOT_TE_REVISION):
        raise EvidenceError("NeMo-RL root Transformer Engine lock differs from pin")
    bridge_lock_identity = _locked_te_identity(bridge_root / "uv.lock")
    if bridge_lock_identity != (BRIDGE_TE_PACKAGE, BRIDGE_TE_REVISION):
        raise EvidenceError("Bridge Transformer Engine declaration differs from pin")

    compressed_tensors_wheel_sha256 = _locked_registry_wheel_sha256(
        repository_root / "uv.lock",
        package_name="compressed-tensors",
        package_version=COMPRESSED_TENSORS_PACKAGE,
    )
    if compressed_tensors_wheel_sha256 != COMPRESSED_TENSORS_WHEEL_SHA256:
        raise EvidenceError("compressed-tensors wheel lock differs from pin")
    if (
        _git(compressed_tensors_source_root, "rev-parse", "HEAD")
        != COMPRESSED_TENSORS_REVISION
    ):
        raise EvidenceError("local compressed-tensors source differs from pin")
    for relative_path, expected_sha256 in _COMPRESSED_TENSORS_SOURCE_SHA256.items():
        if _sha256(compressed_tensors_source_root / relative_path) != expected_sha256:
            raise EvidenceError(
                f"pinned compressed-tensors source differs: {relative_path}"
            )

    if (
        _git(modelopt_lightning_source_root, "rev-parse", "HEAD")
        != MODELOPT_LIGHTNING_REVISION
    ):
        raise EvidenceError("local ModelOpt Lightning source differs from pin")
    for relative_path, expected_sha256 in _MODELOPT_LIGHTNING_SOURCE_SHA256.items():
        if _sha256(modelopt_lightning_source_root / relative_path) != expected_sha256:
            raise EvidenceError(
                f"pinned ModelOpt NVFP4 source differs: {relative_path}"
            )

    automodel_adapter = repository_root / _AUTOMODEL_ADAPTER_PATH
    if _sha256(automodel_adapter) != _AUTOMODEL_ADAPTER_SHA256:
        raise EvidenceError(
            "pinned K2.5 Automodel adapter source differs from evidence"
        )
    if _git(transformer_engine_source_root, "rev-parse", "HEAD") != ROOT_TE_REVISION:
        raise EvidenceError("local Transformer Engine source differs from root lock")
    for relative_path, expected_sha256 in _TE_SOURCE_SHA256.items():
        te_source = transformer_engine_source_root / relative_path
        if _sha256(te_source) != expected_sha256:
            raise EvidenceError(
                f"pinned Transformer Engine MXFP8 source differs: {relative_path}"
            )

    if not inspect_runtime:
        raise EvidenceError(
            "--inspect-runtime is required for admissible producer evidence"
        )
    runtime_package, runtime_revision = _installed_te_identity()
    if (runtime_package, runtime_revision) != (ROOT_TE_PACKAGE, ROOT_TE_REVISION):
        raise EvidenceError(
            "effective Transformer Engine runtime differs from the root lock"
        )
    runtime_inspection = {
        "package_identity": runtime_package,
        "source_revision": runtime_revision,
        "status": "matched_root_lock",
    }

    evidence: dict[str, object] = {
        "implementations": observed_implementations,
        "runtime_inspection": runtime_inspection,
        "schema_version": PRODUCER_EVIDENCE_SCHEMA,
        "source_contracts": _EXPECTED_SOURCE_CONTRACTS,
    }
    validate_producer_implementation_evidence(evidence)
    return evidence


def checkpoint_artifact_specs() -> tuple[CheckpointArtifactSpec, ...]:
    """Return the immutable HF checkpoint pins consumed by staging and capture."""
    checkpoint_artifact_ids = (
        "qwen3_bf16",
        "kimi_k2",
        "kimi_k25",
        "kimi_k3",
        "nemotron_lightning_nvfp4",
        "qwen_a95b_fp8",
    )
    dimensions_by_artifact: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "qwen3_bf16": (("moe_intermediate_size",), ("hidden_size",)),
        "kimi_k2": (("moe_intermediate_size",), ("hidden_size",)),
        "kimi_k25": (
            ("text_config", "moe_intermediate_size"),
            ("text_config", "hidden_size"),
        ),
        "kimi_k3": (
            ("text_config", "moe_intermediate_size"),
            ("text_config", "routed_expert_hidden_size"),
        ),
        "nemotron_lightning_nvfp4": (
            ("moe_intermediate_size",),
            ("hidden_size",),
        ),
        "qwen_a95b_fp8": (("moe_intermediate_size",), ("hidden_size",)),
    }
    requirements_by_artifact: dict[str, tuple[tuple[tuple[str, ...], object], ...]] = {
        "qwen3_bf16": (),
        "kimi_k2": (
            (("quantization_config", "quant_method"), "fp8"),
            (("quantization_config", "fmt"), "e4m3"),
            (("quantization_config", "weight_block_size"), [128, 128]),
        ),
        "kimi_k25": (
            (
                ("text_config", "quantization_config", "format"),
                "pack-quantized",
            ),
            (
                ("text_config", "quantization_config", "quant_method"),
                "compressed-tensors",
            ),
            (
                (
                    "text_config",
                    "quantization_config",
                    "config_groups",
                    "group_0",
                    "weights",
                    "group_size",
                ),
                32,
            ),
            (
                (
                    "text_config",
                    "quantization_config",
                    "config_groups",
                    "group_0",
                    "weights",
                    "num_bits",
                ),
                4,
            ),
            (
                (
                    "text_config",
                    "quantization_config",
                    "config_groups",
                    "group_0",
                    "weights",
                    "type",
                ),
                "int",
            ),
            (
                (
                    "text_config",
                    "quantization_config",
                    "config_groups",
                    "group_0",
                    "weights",
                    "symmetric",
                ),
                True,
            ),
        ),
        "kimi_k3": (
            (
                ("text_config", "quantization_config", "format"),
                "mxfp4-pack-quantized",
            ),
            (
                ("text_config", "quantization_config", "quant_method"),
                "compressed-tensors",
            ),
            (
                (
                    "text_config",
                    "quantization_config",
                    "config_groups",
                    "group_0",
                    "weights",
                    "group_size",
                ),
                32,
            ),
            (
                (
                    "text_config",
                    "quantization_config",
                    "config_groups",
                    "group_0",
                    "weights",
                    "num_bits",
                ),
                4,
            ),
            (
                (
                    "text_config",
                    "quantization_config",
                    "config_groups",
                    "group_0",
                    "weights",
                    "scale_dtype",
                ),
                "torch.uint8",
            ),
            (
                (
                    "text_config",
                    "quantization_config",
                    "config_groups",
                    "group_0",
                    "weights",
                    "type",
                ),
                "float",
            ),
            (
                (
                    "text_config",
                    "quantization_config",
                    "config_groups",
                    "group_0",
                    "weights",
                    "symmetric",
                ),
                True,
            ),
        ),
        "nemotron_lightning_nvfp4": (
            (("quantization_config", "quant_method"), "modelopt"),
            (("quantization_config", "quant_algo"), "MIXED_PRECISION"),
            (
                ("quantization_config", "producer"),
                {"name": "modelopt", "version": "0.44.0rc5"},
            ),
            (
                (
                    "quantization_config",
                    "config_groups",
                    "group_1",
                    "weights",
                    "group_size",
                ),
                16,
            ),
            (
                (
                    "quantization_config",
                    "config_groups",
                    "group_1",
                    "weights",
                    "num_bits",
                ),
                4,
            ),
            (
                (
                    "quantization_config",
                    "config_groups",
                    "group_1",
                    "weights",
                    "type",
                ),
                "float",
            ),
            *tuple(
                requirement
                for target in (
                    "backbone.layers.1.mixer.experts.0.up_proj",
                    "backbone.layers.1.mixer.experts.0.down_proj",
                )
                for requirement in (
                    (
                        (
                            "quantization_config",
                            "quantized_layers",
                            target,
                            "quant_algo",
                        ),
                        "W4A16_NVFP4",
                    ),
                    (
                        (
                            "quantization_config",
                            "quantized_layers",
                            target,
                            "group_size",
                        ),
                        16,
                    ),
                )
            ),
        ),
        "qwen_a95b_fp8": (
            (("quantization_config", "quant_method"), "fp8"),
            (("quantization_config", "weight_block_size"), [128, 128]),
        ),
    }
    membership_requirements_by_artifact: dict[
        str, tuple[ConfigMembershipRequirement, ...]
    ] = {artifact_id: () for artifact_id in checkpoint_artifact_ids}
    membership_requirements_by_artifact["nemotron_lightning_nvfp4"] = (
        ConfigMembershipRequirement(
            path=(
                "quantization_config",
                "config_groups",
                "group_1",
                "targets",
            ),
            members=(
                "backbone.layers.1.mixer.experts.0.up_proj",
                "backbone.layers.1.mixer.experts.0.down_proj",
            ),
            cardinality=5935,
        ),
    )
    specs: list[CheckpointArtifactSpec] = []
    for artifact_id in checkpoint_artifact_ids:
        observation_specs: list[CheckpointObservationSpec] = []
        for observation_id, identity in _EXPECTED_OBSERVATIONS.items():
            producer, observation_artifact, graph, projection, format_id = identity
            if observation_artifact != artifact_id or not producer.startswith(
                "checkpoint."
            ):
                continue
            descriptor = next(
                item for item in SOURCE_FORMAT_CATALOG if item.format_id == format_id
            )
            roles = tuple(component.role for component in descriptor.components)
            logical_shape_config_paths = dimensions_by_artifact[artifact_id]
            if projection == "down":
                logical_shape_config_paths = tuple(reversed(logical_shape_config_paths))
            observation_specs.append(
                CheckpointObservationSpec(
                    format_id=format_id,
                    graph=graph,
                    logical_shape_config_paths=logical_shape_config_paths,
                    observation_id=observation_id,
                    producer=producer,
                    projection=projection,
                    raw_prefix=_EXPECTED_RAW_PREFIXES[observation_id],
                    source_dtypes=_EXPECTED_SOURCE_DTYPES[observation_id],
                    suffixes=tuple(
                        _EXPECTED_RAW_SUFFIXES[format_id][role] for role in roles
                    ),
                )
            )
        specs.append(
            CheckpointArtifactSpec(
                artifact_id=artifact_id,
                artifact=_EXPECTED_ARTIFACTS[artifact_id],
                observations=tuple(observation_specs),
                config_requirements=requirements_by_artifact[artifact_id],
                config_membership_requirements=(
                    membership_requirements_by_artifact[artifact_id]
                ),
            )
        )
    return tuple(specs)


def _derive_automodel_observations(
    checkpoint_observations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    checkpoint_by_projection = {
        cast(str, observation["projection"]): observation
        for observation in checkpoint_observations
        if observation.get("artifact") == "kimi_k25"
    }
    format_id = "packed-int4.i32-f16-group32-shape-i64.v1"
    descriptor = next(
        item for item in SOURCE_FORMAT_CATALOG if item.format_id == format_id
    )
    roles = tuple(component.role for component in descriptor.components)
    observations: list[dict[str, object]] = []
    for projection in ("gate", "up", "down"):
        observation_id = f"k25.automodel.main.{projection}"
        try:
            checkpoint_observation = checkpoint_by_projection[projection]
        except KeyError as error:
            raise EvidenceError(
                f"missing K2.5 checkpoint orientation for Automodel {projection}"
            ) from error
        logical_axes = tuple(
            _string(item, f"{observation_id}.logical_axes")
            for item in _object_sequence(
                checkpoint_observation.get("logical_axes"),
                f"{observation_id}.logical_axes",
            )
        )
        logical_shape = _positive_int_sequence(
            checkpoint_observation.get("logical_shape"),
            f"{observation_id}.logical_shape",
        )
        raw_names = [
            _EXPECTED_RAW_PREFIXES[observation_id]
            + _EXPECTED_RAW_SUFFIXES[format_id][role]
            for role in roles
        ]
        components: list[dict[str, object]] = []
        for index, component in enumerate(descriptor.components):
            try:
                shape = [
                    extent
                    for _, extent in resolve_component_axes(
                        component,
                        logical_axes=logical_axes,
                        logical_shape=logical_shape,
                    )
                ]
            except ValueError as error:
                raise EvidenceError(
                    "Automodel group-32 catalog admission rejects logical shape "
                    f"for {observation_id}: {error}"
                ) from error
            components.append(
                {
                    "metadata_location": _EXPECTED_METADATA_LOCATIONS[observation_id],
                    "raw_name": raw_names[index],
                    "role": component.role,
                    "shape": shape,
                    "source_dtype": _EXPECTED_SOURCE_DTYPES[observation_id][index],
                }
            )
        observations.append(
            {
                "artifact": "automodel_kimi_k25",
                "components": components,
                "format_id": format_id,
                "graph": "main",
                "logical_axes": list(logical_axes),
                "logical_shape": list(logical_shape),
                "observation_id": observation_id,
                "producer": "automodel.kimi_k25",
                "projection": projection,
                "raw_siblings": raw_names,
            }
        )
    return observations


def _claims_payload() -> list[dict[str, object]]:
    return [
        {
            "contract": _component_contract(component),
            "evidence": list(
                _EXPECTED_CLAIM_ARTIFACTS[(descriptor.format_id, component.role)]
            ),
            "format_id": descriptor.format_id,
            "role": component.role,
        }
        for descriptor in SOURCE_FORMAT_CATALOG
        if descriptor is not BF16_FORMAT and descriptor is not MXFP8_FORMAT
        for component in descriptor.components
    ]


def load_staged_source_format_evidence(staged_metadata_root: Path) -> dict[str, object]:
    """Derive source-format evidence from staged raw checkpoint metadata."""
    staged_metadata_root = staged_metadata_root.resolve(strict=True)
    if not staged_metadata_root.is_dir():
        raise EvidenceError("staged metadata root must be a directory")
    artifacts: dict[str, object] = {}
    checkpoint_observations: list[dict[str, object]] = []
    opened_metadata: list[dict[str, str]] = []
    for spec in checkpoint_artifact_specs():
        artifact, observations, receipt = capture_staged_checkpoint_evidence(
            staged_metadata_root, spec
        )
        artifacts[spec.artifact_id] = artifact
        checkpoint_observations.extend(observations)
        opened_metadata.extend(receipt)
    artifacts["automodel_kimi_k25"] = _EXPECTED_ARTIFACTS["automodel_kimi_k25"]
    artifacts["compressed_tensors_0_17_0"] = _EXPECTED_ARTIFACTS[
        "compressed_tensors_0_17_0"
    ]
    artifacts["modelopt_0_44_0rc5"] = _EXPECTED_ARTIFACTS["modelopt_0_44_0rc5"]
    artifacts["transformer_engine_mxfp8"] = _EXPECTED_ARTIFACTS[
        "transformer_engine_mxfp8"
    ]
    automodel_observations = _derive_automodel_observations(checkpoint_observations)
    observations_by_id = {
        cast(str, observation["observation_id"]): observation
        for observation in (*checkpoint_observations, *automodel_observations)
    }
    observation_order = (
        "qwen3.main.gate",
        "qwen3.main.up",
        "qwen3.main.down",
        "k2.main.gate",
        "k2.main.up",
        "k2.main.down",
        "k25.checkpoint.main.gate",
        "k25.automodel.main.gate",
        "k25.checkpoint.main.up",
        "k25.automodel.main.up",
        "k25.checkpoint.main.down",
        "k25.automodel.main.down",
        "k3.main.gate",
        "k3.main.up",
        "k3.main.down",
        "lightning.main.up",
        "lightning.main.down",
        "a95b.main.gate",
        "a95b.main.up",
        "a95b.main.down",
        "a95b.mtp-0.gate",
        "a95b.mtp-0.up",
        "a95b.mtp-0.down",
    )
    document: dict[str, object] = {
        "artifacts": artifacts,
        "capture_mode": "metadata_only_no_tensor_payloads",
        "capture_receipt": {
            "opened_metadata": opened_metadata,
            "schema_version": CAPTURE_RECEIPT_SCHEMA,
        },
        "catalog": _catalog_payload(),
        "claims": _claims_payload(),
        "observations": [observations_by_id[item] for item in observation_order],
        "schema_version": SOURCE_EVIDENCE_SCHEMA,
    }
    validate_source_format_evidence(document)
    return document


def _canonical_json(document: Mapping[str, object]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _materialize(path: Path, document: Mapping[str, object], *, check: bool) -> None:
    expected = _canonical_json(document)
    if check:
        if not path.is_file() or path.read_text() != expected:
            raise EvidenceError(f"captured evidence is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected)


def capture(
    *,
    repository_root: Path,
    compressed_tensors_source_root: Path,
    modelopt_lightning_source_root: Path,
    staged_metadata_root: Path,
    transformer_engine_source_root: Path,
    output_directory: Path,
    check: bool,
    inspect_runtime: bool,
) -> None:
    """Validate staged metadata and materialize deterministic evidence fixtures."""
    source_evidence = load_staged_source_format_evidence(staged_metadata_root)
    producer_evidence = capture_producer_implementation_evidence(
        repository_root,
        compressed_tensors_source_root,
        modelopt_lightning_source_root,
        transformer_engine_source_root,
        inspect_runtime=inspect_runtime,
    )
    _materialize(
        output_directory / "source_format_evidence.json",
        source_evidence,
        check=check,
    )
    _materialize(
        output_directory / "producer_implementations.json",
        producer_evidence,
        check=check,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--compressed-tensors-source-root", required=True, type=Path)
    parser.add_argument("--modelopt-lightning-source-root", required=True, type=Path)
    parser.add_argument("--staged-metadata-root", required=True, type=Path)
    parser.add_argument("--transformer-engine-source-root", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--inspect-runtime", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        capture(
            repository_root=arguments.repository_root,
            compressed_tensors_source_root=arguments.compressed_tensors_source_root,
            modelopt_lightning_source_root=arguments.modelopt_lightning_source_root,
            staged_metadata_root=arguments.staged_metadata_root,
            transformer_engine_source_root=arguments.transformer_engine_source_root,
            output_directory=arguments.output_directory,
            check=arguments.check,
            inspect_runtime=arguments.inspect_runtime,
        )
    except (EvidenceError, OSError, subprocess.CalledProcessError) as error:
        print(f"evidence capture failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
