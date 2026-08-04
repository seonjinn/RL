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
    normalize_quant_cfg_identity,
    save_nvfp4_calibration,
)


def _quantizer_amax(quantizer: object, quantizer_name: str) -> torch.Tensor:
    if not bool(getattr(quantizer, "is_enabled", False)):
        raise RuntimeError(f"Required input quantizer {quantizer_name!r} is disabled")

    amax = getattr(quantizer, "_amax", None)
    if amax is None:
        amax = getattr(quantizer, "amax", None)
    if not isinstance(amax, torch.Tensor) or amax.is_meta:
        raise RuntimeError(
            f"Required input quantizer {quantizer_name!r} has no tensor amax"
        )

    value = amax.detach().cpu().clone()
    if value.ndim != 0:
        raise RuntimeError(
            f"Required input quantizer {quantizer_name!r} must have a scalar amax"
        )
    if not bool(torch.isfinite(value).item()) or not bool((value > 0).item()):
        raise RuntimeError(
            f"Required input quantizer {quantizer_name!r} must have a finite, "
            "positive amax"
        )
    return value


def _is_fused_expert_candidate(module: nn.Module) -> bool:
    if hasattr(module, "gate_up_proj_input_quantizer") or hasattr(
        module, "down_proj_input_quantizer"
    ):
        return True
    return any(
        isinstance(parameter, torch.Tensor) and parameter.ndim == 3
        for parameter in (
            getattr(module, "gate_up_proj", None),
            getattr(module, "down_proj", None),
        )
    )


def _fused_expert_count(module_name: str, module: nn.Module) -> int:
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    if not isinstance(gate_up, torch.Tensor) or gate_up.ndim != 3:
        raise RuntimeError(
            f"Fused experts {module_name!r} require a 3-D gate_up_proj parameter"
        )
    if not isinstance(down, torch.Tensor) or down.ndim != 3:
        raise RuntimeError(
            f"Fused experts {module_name!r} require a 3-D down_proj parameter"
        )

    num_experts = getattr(module, "num_experts", gate_up.shape[0])
    if (
        not isinstance(num_experts, int)
        or isinstance(num_experts, bool)
        or num_experts <= 0
    ):
        raise RuntimeError(
            f"Fused experts {module_name!r} have invalid num_experts {num_experts!r}"
        )
    if gate_up.shape[0] != num_experts or down.shape[0] != num_experts:
        raise RuntimeError(
            f"Fused experts {module_name!r} num_experts {num_experts} does not "
            f"match parameter shapes {tuple(gate_up.shape)} and {tuple(down.shape)}"
        )

    intermediate_dim, remainder = divmod(gate_up.shape[1], 2)
    if (
        remainder
        or intermediate_dim <= 0
        or down.shape[1] != gate_up.shape[2]
        or down.shape[2] != intermediate_dim
    ):
        raise RuntimeError(
            f"Fused experts {module_name!r} have inconsistent gate_up_proj shape "
            f"{tuple(gate_up.shape)} and down_proj shape {tuple(down.shape)}"
        )
    return num_experts


def _store_input_amax(
    input_amax: dict[str, torch.Tensor],
    projection_name: str,
    amax: torch.Tensor,
) -> None:
    if projection_name in input_amax:
        raise RuntimeError(f"Duplicate HF projection input amax {projection_name!r}")
    input_amax[projection_name] = amax.clone()


def _collect_fused_expert_input_amax(
    input_amax: dict[str, torch.Tensor],
    module_name: str,
    module: nn.Module,
) -> None:
    if not module_name:
        raise RuntimeError("Fused experts must have an HF module name")
    num_experts = _fused_expert_count(module_name, module)

    quantizer_amax: dict[str, torch.Tensor] = {}
    for projection_group in ("gate_up_proj", "down_proj"):
        quantizer_name = f"{projection_group}_input_quantizer"
        quantizer = getattr(module, quantizer_name, None)
        if quantizer is None:
            raise RuntimeError(
                f"Fused experts {module_name!r} are missing {quantizer_name}"
            )
        quantizer_amax[projection_group] = _quantizer_amax(
            quantizer,
            f"{module_name}.{quantizer_name}",
        )

    for expert_idx in range(num_experts):
        expert_prefix = f"{module_name}.{expert_idx}"
        for projection_name in ("gate_proj", "up_proj"):
            _store_input_amax(
                input_amax,
                f"{expert_prefix}.{projection_name}.weight",
                quantizer_amax["gate_up_proj"],
            )
        _store_input_amax(
            input_amax,
            f"{expert_prefix}.down_proj.weight",
            quantizer_amax["down_proj"],
        )


def collect_nvfp4_input_amax(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect enabled input quantizer amax values by exact HF weight name."""
    input_amax: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if _is_fused_expert_candidate(module):
            _collect_fused_expert_input_amax(input_amax, module_name, module)
            continue

        quantizer = getattr(module, "input_quantizer", None)
        if quantizer is None or not bool(getattr(quantizer, "is_enabled", False)):
            continue
        if not module_name or getattr(module, "weight", None) is None:
            raise RuntimeError(
                f"Enabled input quantizer {module_name!r} has no HF projection weight"
            )

        projection_name = f"{module_name}.weight"
        _store_input_amax(
            input_amax,
            projection_name,
            _quantizer_amax(quantizer, f"{module_name}.input_quantizer"),
        )
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
    quant_cfg_identity = normalize_quant_cfg_identity(args.quant_cfg)

    set_seed(args.seed)
    tokenizer = get_tokenizer(
        args.model,
        max_seq_len=args.sequence_length,
        revision=args.model_revision,
    )
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
        quant_cfg=quant_cfg_identity,
        dataset=args.dataset,
        sample_count=args.sample_count,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    load_nvfp4_calibration(
        args.output,
        model_id=args.model,
        model_revision=args.model_revision,
        quant_cfg=quant_cfg_identity,
        expected_projection_names=set(input_amax),
    )


if __name__ == "__main__":
    main()
