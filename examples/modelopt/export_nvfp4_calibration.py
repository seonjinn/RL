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

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from nemo_rl.modelopt.calibration_artifact import (
    load_nvfp4_calibration,
    save_nvfp4_calibration,
)


def collect_nvfp4_input_amax(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect enabled input quantizer amax values by exact HF weight name."""
    input_amax: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        quantizer = getattr(module, "input_quantizer", None)
        if quantizer is None or not bool(getattr(quantizer, "is_enabled", False)):
            continue
        if not module_name or getattr(module, "weight", None) is None:
            raise RuntimeError(
                f"Enabled input quantizer {module_name!r} has no HF projection weight"
            )

        amax = getattr(quantizer, "_amax", None)
        if amax is None:
            amax = getattr(quantizer, "amax", None)
        if not isinstance(amax, torch.Tensor):
            raise RuntimeError(
                f"Enabled input quantizer {module_name!r} has no tensor amax"
            )

        projection_name = f"{module_name}.weight"
        if projection_name in input_amax:
            raise RuntimeError(
                f"Duplicate HF projection input amax {projection_name!r}"
            )
        input_amax[projection_name] = amax.detach().cpu().clone()
    return input_amax


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixed NVFP4 input calibration for BF16 rollout refits."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--quant-cfg", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-count", required=True, type=_positive_int)
    parser.add_argument("--sequence-length", required=True, type=_positive_int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.dataset == "random":
        raise ValueError("NVFP4 rollout calibration requires a real dataset")

    # These dependencies are optional and heavy outside the standalone exporter.
    from transformers import AutoModelForCausalLM

    from nemo_rl.algorithms.utils import set_seed
    from nemo_rl.modelopt.models.policy.workers.utils import (
        get_tokenizer,
        quantize_model,
    )
    from nemo_rl.modelopt.utils import resolve_nvfp4_real_quant_mode

    if resolve_nvfp4_real_quant_mode(args.quant_cfg) != "w4a4":
        raise ValueError("NVFP4 calibration export requires a W4A4 quant_cfg")

    set_seed(args.seed)
    tokenizer = get_tokenizer(args.model, max_seq_len=args.sequence_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    quantize_model(
        model=model,
        quant_cfg=args.quant_cfg,
        tokenizer=tokenizer,
        calib_size=args.sample_count,
        is_megatron=False,
        data=args.dataset,
        max_sample_length=args.sequence_length,
    )
    input_amax = collect_nvfp4_input_amax(model)
    save_nvfp4_calibration(
        args.output,
        input_amax,
        model_id=args.model,
        model_revision=args.model_revision,
        quant_cfg=args.quant_cfg,
        dataset=args.dataset,
        sample_count=args.sample_count,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    load_nvfp4_calibration(
        args.output,
        model_id=args.model,
        model_revision=args.model_revision,
        quant_cfg=args.quant_cfg,
        expected_projection_names=set(input_amax),
    )


if __name__ == "__main__":
    main()
