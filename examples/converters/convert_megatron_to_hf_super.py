# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from megatron.bridge import AutoBridge

""" NOTE: this script requires mcore. Make sure to launch with the mcore extra:
uv run --extra mcore torchrun --nproc_per_node=8 --nnodes=1 examples/converters/convert_megatron_to_hf_super.py \
  --config <path_to_ckpt>/config.yaml \
  --megatron-ckpt-path <path_to_ckpt>/policy/weights/iter_xxxxx \
  --hf-ckpt-path <path_to_save_hf_ckpt> \
  --tp <tensor_model_parallel_size> \
  --pp <pipeline_model_parallel_size> \
  --ep <expert_model_parallel_size> \
  --etp <expert_tensor_parallel_size>

The converter reads policy.model_name from --config and uses that parent HF
checkpoint only to recover explicitly allowlisted tensors below if the DCP
export does not yield them.
"""

# These are static preprocessing buffers required by the HF Super/Omni checkpoint
# layout. They were not yielded by the DCP export for the RL checkpoints we tested,
# but they are not trainable RL weights.
STATIC_PREPROCESSING_TENSOR_NAMES = (
    "sound_encoder.encoder.feature_extractor.featurizer.fb",
    "sound_encoder.encoder.feature_extractor.featurizer.window",
    "vision_model.radio_model.input_conditioner.norm_mean",
    "vision_model.radio_model.input_conditioner.norm_std",
)

# Some Super parent HF checkpoints include an MTP head even when the RL DCP has
# MTP disabled (for example mtp_num_layers: null). In that case the trainable MTP
# tensors cannot come from the DCP, but they are still needed for the HF weight
# map to be complete. Keep this fallback scoped to the explicit HF MTP namespace.
MTP_FALLBACK_TENSOR_PREFIXES = ("mtp.",)

# This RL checkpoint does not yield the tied lexical embedding/output tensors
# from DCP, but the HF weight map requires them. Keep the parent fallback limited
# to these exact tensor names so missing trainable tensors still fail closed.
TIED_EMBEDDING_FALLBACK_TENSOR_NAMES = (
    "backbone.embeddings.weight",
    "lm_head.weight",
)


def is_allowed_parent_fallback_tensor(name: str) -> bool:
    return (
        name in STATIC_PREPROCESSING_TENSOR_NAMES
        or name in TIED_EMBEDDING_FALLBACK_TENSOR_NAMES
        or any(name.startswith(prefix) for prefix in MTP_FALLBACK_TENSOR_PREFIXES)
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Torch DCP checkpoint to HF checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file in the checkpoint directory",
    )
    parser.add_argument(
        "--megatron-ckpt-path",
        type=str,
        default=None,
        help="Path to Megatron checkpoint",
    )
    parser.add_argument(
        "--hf-ckpt-path", type=str, default=None, help="Path to save HF checkpoint"
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="Tensor model parallel size"
    )
    parser.add_argument(
        "--pp", type=int, default=1, help="Pipeline model parallel size"
    )
    parser.add_argument(
        "--ep", type=int, default=1, help="Expert model parallel size"
    )
    parser.add_argument(
        "--etp", type=int, default=1, help="Expert tensor parallel size"
    )
    # Parse known args for the script
    args = parser.parse_args()

    return args


def install_parent_allowlisted_safetensors_save() -> None:
    """Fill only explicitly allowlisted tensors from the parent HF checkpoint."""
    from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

    def save_generator_with_parent_static_fallback(self, generator, output_path, strict=True):
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0

        if rank != 0:
            for _ in generator:
                pass
            return

        from safetensors.torch import save_file

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        key_to_filename_map = self.key_to_filename_map
        all_expected_keys = set(key_to_filename_map.keys())

        if not key_to_filename_map:
            buffered_tensors = dict(generator)
            if buffered_tensors:
                save_file(buffered_tensors, output_path / "model.safetensors")
            return

        filename_to_keys_map = defaultdict(set)
        for key, filename in key_to_filename_map.items():
            filename_to_keys_map[filename].add(key)

        files_to_save = dict(filename_to_keys_map)
        buffered_tensors = {}
        all_yielded_keys = set()
        all_saved_keys = set()

        def save_complete_file(filename, keys_for_file, tensor_source):
            tensors_to_save = {key: tensor_source[key] for key in keys_for_file}
            save_file(tensors_to_save, output_path / filename)
            all_saved_keys.update(keys_for_file)

        for name, tensor in generator:
            all_yielded_keys.add(name)
            if name not in all_expected_keys:
                if strict:
                    raise KeyError(
                        f"Tensor '{name}' from generator not found in the original model structure. "
                        "To ignore, set strict=False."
                    )
                print(f"Warning: tensor '{name}' from generator not found in original model structure. Skipping.")
                continue

            buffered_tensors[name] = tensor

            for filename in list(files_to_save.keys()):
                keys_for_file = files_to_save[filename]
                if keys_for_file.issubset(buffered_tensors.keys()):
                    save_complete_file(filename, keys_for_file, buffered_tensors)
                    for key in keys_for_file:
                        del buffered_tensors[key]
                    del files_to_save[filename]

        if files_to_save:
            missing_by_file = {
                filename: sorted(keys_for_file - all_yielded_keys)
                for filename, keys_for_file in files_to_save.items()
            }
            all_missing_keys = sorted(
                {key for missing_keys in missing_by_file.values() for key in missing_keys}
            )
            unexpected_missing_keys = [
                key for key in all_missing_keys if not is_allowed_parent_fallback_tensor(key)
            ]
            if unexpected_missing_keys:
                raise RuntimeError(
                    "DCP export did not yield tensors outside the explicit parent fallback allowlist "
                    "(static preprocessing tensors, tied embedding/output tensors, and HF mtp.* tensors): "
                    f"{unexpected_missing_keys[:20]}. Refusing to fetch arbitrary weights from the "
                    "parent HF checkpoint."
                )

            fallback_keys = sorted(
                key for key in all_missing_keys if is_allowed_parent_fallback_tensor(key)
            )
            if fallback_keys:
                static_fallback_keys = [
                    key for key in fallback_keys if key in STATIC_PREPROCESSING_TENSOR_NAMES
                ]
                mtp_fallback_keys = [
                    key
                    for key in fallback_keys
                    if any(key.startswith(prefix) for prefix in MTP_FALLBACK_TENSOR_PREFIXES)
                ]
                tied_embedding_fallback_keys = [
                    key
                    for key in fallback_keys
                    if key in TIED_EMBEDDING_FALLBACK_TENSOR_NAMES
                ]
                print(
                    "Warning: fetching these explicitly allowlisted tensors from "
                    "policy.model_name because the DCP export did not yield them:"
                )
                if static_fallback_keys:
                    print(f"  - static preprocessing tensors: {len(static_fallback_keys)}")
                    for key in static_fallback_keys:
                        print(f"    - {key}")
                if mtp_fallback_keys:
                    print(f"  - HF mtp.* tensors: {len(mtp_fallback_keys)}")
                    for key in mtp_fallback_keys:
                        print(f"    - {key}")
                if tied_embedding_fallback_keys:
                    print(f"  - tied embedding/output tensors: {len(tied_embedding_fallback_keys)}")
                    for key in tied_embedding_fallback_keys:
                        print(f"    - {key}")

            for filename in list(files_to_save.keys()):
                keys_for_file = files_to_save[filename]
                missing_for_file = missing_by_file[filename]
                if missing_for_file:
                    print(f"  - {filename}: parent HF fallback for {len(missing_for_file)} explicit tensors:")
                    for key in missing_for_file:
                        print(f"    - {key}")
                fallback_tensors = self.load_tensors(missing_for_file) if missing_for_file else {}
                tensor_source = {}
                for key in keys_for_file:
                    tensor_source[key] = buffered_tensors.get(key, fallback_tensors.get(key))
                    if tensor_source[key] is None:
                        raise KeyError(f"Tensor '{key}' was not yielded and is missing from source fallback.")
                save_complete_file(filename, keys_for_file, tensor_source)
                for key in list(keys_for_file):
                    buffered_tensors.pop(key, None)
                del files_to_save[filename]

        if buffered_tensors:
            print(f"Warning: {len(buffered_tensors)} yielded tensors were not part of the parent HF weight map.")

        unsaved_keys = all_expected_keys - all_saved_keys
        if unsaved_keys:
            raise RuntimeError(f"{len(unsaved_keys)} tensors were not written: {sorted(unsaved_keys)[:20]}")

        extra_keys = all_yielded_keys - all_expected_keys
        if extra_keys:
            print(f"Success: wrote all HF tensors; ignored {len(extra_keys)} extra generated tensors.")
        else:
            print("Success: wrote all HF tensors.")

        original_index_file = self.path / "model.safetensors.index.json"
        if original_index_file.exists():
            with open(original_index_file, "r", encoding="utf-8") as f:
                original_index_data = json.load(f)
            new_index_data = {
                "metadata": original_index_data.get("metadata", {}),
                "weight_map": {key: key_to_filename_map[key] for key in sorted(all_saved_keys)},
            }
            with open(output_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
                json.dump(new_index_data, f, indent=4)

    SafeTensorsStateSource.save_generator = save_generator_with_parent_static_fallback


def export_model_from_megatron_gpu(
    hf_model_name: str,
    input_path: str,
    output_path: str,
    hf_tokenizer_path: str,
    overwrite: bool = False,
    hf_overrides: Optional[dict[str, Any]] = {},
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
):
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"HF checkpoint already exists at {output_path}. Delete it to run or set overwrite=True."
        )

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **hf_overrides
    )

    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.sequence_parallel = True

    # FIXME: This is a hack to enable cuda graph for the model.
    model_provider.enable_cuda_graph=True
    model_provider.use_te_rng_tracker=True

    # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0, seed_kwargs={"te_rng_tracker": model_provider.use_te_rng_tracker})

    # Load the Megatron model directly
    megatron_model = bridge.load_megatron_model(input_path, wrap_with_ddp=False)

    install_parent_allowlisted_safetensors_save()

    bridge.save_hf_pretrained(
        megatron_model,
        output_path,
        source_path=hf_model_name,
        show_progress=True,
    )


def main():
    """Main entry point."""
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["policy"]["model_name"]
    tokenizer_name = config["policy"]["tokenizer"]["name"]
    hf_overrides = config["policy"].get("hf_overrides", {}) or {}

    export_model_from_megatron_gpu(
        hf_model_name=model_name,
        input_path=args.megatron_ckpt_path,
        output_path=args.hf_ckpt_path,
        hf_tokenizer_path=tokenizer_name,
        hf_overrides=hf_overrides,
        overwrite=True,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
    )


if __name__ == "__main__":
    main()
