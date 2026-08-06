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

import os
import types
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import vllm  # noqa: F401
import zmq
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (  # pyrefly: ignore[import-error]
    TensorQuantizer,
)

from nemo_rl.modelopt.calibration_artifact import load_nvfp4_calibration
from nemo_rl.modelopt.models.generation.nvfp4_refit import (
    NVFP4Calibration,
    NVFP4RefitMode,
    nvfp4_refit_group,
    serialize_bf16_nvfp4_group,
)
from nemo_rl.modelopt.utils import (
    MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS,
    matches_quant_ignore_pattern,
)
from nemo_rl.models.generation.vllm.checkpoint_engine import VllmCheckpointEngineMixin
from nemo_rl.models.generation.vllm.vllm_backend import (
    IPCWeightManifestError,
    VllmInternalWorkerExtension,
    WeightUpdateFinalizer,
    WeightUpdateTransport,
)
from nemo_rl.weight_sync.nccl_reshard_utils import (
    HFToLocalParamMap,
    LocalParamSpec,
    RefitCtx,
)
from nemo_rl.weight_sync.refit_transforms import (
    RefitTransformRequest,
    RefitTransformResponse,
)

_FUSED_MODELOPT_MOE_SUFFIXES = {
    ".experts.w13_weight": "w13_weight",
    ".experts.w13_weight_scale": "w13_weight_scale",
    ".experts.w13_weight_scale_2": "w13_weight_scale_2",
    ".experts.w2_weight": "down_proj.weight",
    ".experts.w2_weight_scale": "down_proj.weight_scale",
    ".experts.w2_weight_scale_2": "down_proj.weight_scale_2",
    ".experts.w13_input_scale": "w13_input_scale",
    ".experts.w2_input_scale": "w2_input_scale",
}
_MODELOPT_COMPONENT_SUFFIXES = (
    ".weight_scale_2",
    ".weight_scale",
    ".input_scale",
    ".weight",
)
_RealQuantSource = Literal["bf16", "modelopt"]


@dataclass(frozen=True)
class _RealQuantSourceInfo:
    source: _RealQuantSource
    bf16_names: frozenset[str]
    w13_num_shards_by_prefix: dict[str, int]


def _input_scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected an HF projection weight name, got {weight_name!r}")
    return weight_name.removesuffix(".weight") + ".input_scale"


def _vllm_calibration_provenance(model_config: Any) -> tuple[str, str]:
    """Return the explicit HF model identity held by vLLM 0.25 ModelConfig."""
    model_id = model_config.model
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("vLLM model config requires a non-empty model id")

    configured_revision = model_config.revision
    if not isinstance(configured_revision, str) or not configured_revision:
        raise ValueError(
            "BF16 W4A4 calibration requires an explicit model revision in "
            "the vLLM model config"
        )

    resolved_revision = getattr(model_config.hf_config, "_commit_hash", None)
    if isinstance(resolved_revision, str) and resolved_revision:
        return model_id, resolved_revision
    return model_id, configured_revision


def _match_fused_modelopt_moe_weight(name: str) -> tuple[str, str] | None:
    return next(
        (
            (suffix, target)
            for suffix, target in _FUSED_MODELOPT_MOE_SUFFIXES.items()
            if name.endswith(suffix)
        ),
        None,
    )


def _is_ignored_real_quant_tensor(name: str, ignore_patterns: list[str]) -> bool:
    """Return whether a real-quant manifest tensor matches an ignore rule."""
    return matches_quant_ignore_pattern(name, ignore_patterns)


def _model_prefix_variants(name: str) -> set[str]:
    """Return vLLM/HF name variants with or without the leading model scope."""
    variants = {name}
    if name.startswith("model."):
        variants.add(name.removeprefix("model."))
    else:
        variants.add(f"model.{name}")
    return variants


def _modelopt_target_kind(module: torch.nn.Module) -> Literal["linear", "moe"] | None:
    """Classify a receiver module that owns an NVFP4 quantized destination."""
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None:
        return None

    try:
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )
    except ImportError:
        return None
    try:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )
    except ImportError:
        ParallelLMHead = LinearBase

    quant_config = getattr(quant_method, "quant_config", None)
    if not isinstance(quant_config, ModelOptNvFp4Config):
        return None
    if isinstance(module, RoutedExperts):
        return "moe"
    if isinstance(module, (LinearBase, ParallelLMHead)):
        return "linear"
    return None


def _mapped_weight_name_variants(model: torch.nn.Module, name: str) -> set[str]:
    """Map complete HF weight names through the receiver's vLLM mapper."""
    source_variants = _model_prefix_variants(name)
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is not None:
        apply_list = getattr(mapper, "apply_list", None)
        if not callable(apply_list):
            raise TypeError("vLLM hf_to_vllm_mapper must provide apply_list()")
        apply_list = cast(Callable[[list[str]], list[str]], apply_list)
        mapped_names = set(apply_list([name]))
        if not mapped_names:
            return set()
        for source_name in sorted(source_variants - {name}):
            mapped_names.update(apply_list([source_name]))
        return mapped_names

    # Older model implementations expose only packed_modules_mapping. Keep
    # this compatibility path separate from WeightsMapper's complete-name API.
    packed_mapping = getattr(model, "packed_modules_mapping", {})
    packed_reverse = {
        original: fused
        for fused, originals in packed_mapping.items()
        for original in originals
    }
    mapped_variants: set[str] = set()
    for source_name in source_variants:
        parts = source_name.split(".")
        if len(parts) >= 2 and parts[-2] in packed_reverse:
            parts[-2] = packed_reverse[parts[-2]]
        mapped_variants.add(".".join(parts))
    return mapped_variants


def _traverse_module_path(
    model: torch.nn.Module,
    module_path: list[str],
) -> torch.nn.Module | None:
    current_module = model
    try:
        for part in module_path:
            if isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError):
        return None

    routed_experts = getattr(current_module, "routed_experts", None)
    if isinstance(routed_experts, torch.nn.Module):
        return routed_experts
    return current_module


def _resolve_hf_quant_target_module(
    model: torch.nn.Module,
    name: str,
) -> torch.nn.Module | None:
    """Resolve an HF parameter to its vLLM receiver module.

    WeightsMapper owns complete-name renaming, including stacked projections.
    Module traversal then resolves the mapped destination and normalizes a
    MoERunner-like owner to its RoutedExperts child when present.
    """
    for mapped_name in sorted(_mapped_weight_name_variants(model, name)):
        module_path = mapped_name.split(".")[:-1]
        target_module = _traverse_module_path(model, module_path)
        if target_module is not None:
            return target_module

        if len(module_path) >= 3 and module_path[-2].isdigit():
            target_module = _traverse_module_path(model, module_path[:-2])
            if target_module is not None:
                return target_module
    return None


def _is_bf16_quantization_candidate(
    name: str,
    shape: tuple[int, ...] | list[int] | torch.Size,
    *,
    model: torch.nn.Module,
) -> bool:
    """Return whether a BF16 tensor belongs to a ModelOpt-owned destination."""
    if not name.endswith(".weight") or len(shape) != 2:
        return False
    target_module = _resolve_hf_quant_target_module(model, name)
    return _modelopt_target_kind(target_module) is not None if target_module else False


def _is_modelopt_manifest_name(name: str) -> bool:
    """Return whether a manifest name can belong to a ModelOpt target."""
    return _match_fused_modelopt_moe_weight(name) is not None or any(
        name.endswith(suffix) for suffix in _MODELOPT_COMPONENT_SUFFIXES
    )


def _is_receiver_modelopt_component(model: torch.nn.Module, name: str) -> bool:
    """Return whether a packed component belongs to a quantized receiver."""
    target_module = _resolve_hf_quant_target_module(model, name)
    return _modelopt_target_kind(target_module) is not None if target_module else False


def _validate_modelopt_manifest(
    state_dict_info: dict[str, Any],
    names: set[str],
    *,
    require_input_scales: bool,
) -> dict[str, int]:
    """Validate complete prepacked ModelOpt component families."""
    fused_names = {
        name for name in names if _match_fused_modelopt_moe_weight(name) is not None
    }
    w13_num_shards_by_prefix = (
        _w13_num_shards_from_state_dict_info(
            {name: state_dict_info[name] for name in fused_names},
            require_input_scales=require_input_scales,
        )
        if fused_names
        else {}
    )

    ordinary_names = names - fused_names
    if not ordinary_names:
        return w13_num_shards_by_prefix

    families: dict[str, set[str]] = {}
    for name in ordinary_names:
        suffix = next(
            (
                component_suffix
                for component_suffix in _MODELOPT_COMPONENT_SUFFIXES
                if name.endswith(component_suffix)
            ),
            None,
        )
        if suffix is None:
            raise ValueError(f"Unsupported ModelOpt real-quant tensor name: {name}")
        prefix = name[: -len(suffix)]
        families.setdefault(prefix, set()).add(suffix)

    required_suffixes = {".weight", ".weight_scale", ".weight_scale_2"}
    if require_input_scales:
        required_suffixes.add(".input_scale")
    for prefix, suffixes in families.items():
        missing = required_suffixes - suffixes
        if missing:
            raise RuntimeError(
                f"Incomplete ModelOpt weight family for {prefix}: "
                f"missing {sorted(missing)}"
            )
    return w13_num_shards_by_prefix


def _classify_real_quant_source(
    state_dict_info: dict[str, Any],
    *,
    model: torch.nn.Module,
    ignore_patterns: list[str],
    require_input_scales: bool,
) -> _RealQuantSourceInfo:
    """Classify a real-quant refit manifest before receiving any payload."""
    eligible = {
        name: info
        for name, info in state_dict_info.items()
        if not _is_ignored_real_quant_tensor(name, ignore_patterns)
    }
    receiver_names = {
        name
        for name in eligible
        if _is_modelopt_manifest_name(name)
        and _is_receiver_modelopt_component(model, name)
    }
    packed_names = {
        name
        for name in receiver_names
        if _match_fused_modelopt_moe_weight(name) is not None
        or not name.endswith(".weight")
        or eligible[name][1] != torch.bfloat16
    }
    bf16_names = {
        name
        for name, (shape, dtype) in eligible.items()
        if dtype == torch.bfloat16
        and _is_bf16_quantization_candidate(
            name,
            shape,
            model=model,
        )
    }
    if bf16_names and packed_names:
        raise ValueError(
            "mixed BF16 and ModelOpt real-quant manifest entries in "
            "quantizable destinations: "
            f"BF16={sorted(bf16_names)}, ModelOpt={sorted(packed_names)}"
        )
    if bf16_names:
        return _RealQuantSourceInfo(
            source="bf16",
            bf16_names=frozenset(bf16_names),
            w13_num_shards_by_prefix={},
        )

    if packed_names:
        w13_num_shards_by_prefix = _validate_modelopt_manifest(
            eligible,
            packed_names,
            require_input_scales=require_input_scales,
        )
        return _RealQuantSourceInfo(
            source="modelopt",
            bf16_names=frozenset(),
            w13_num_shards_by_prefix=w13_num_shards_by_prefix,
        )

    raise ValueError(
        "no receiver ModelOpt quantization targets found in real-quant manifest"
    )


def _w13_num_shards_from_state_dict_info(
    state_dict_info: dict[str, Any],
    *,
    require_input_scales: bool = False,
) -> dict[str, int]:
    """Validate complete fused-MoE families and resolve their W13 layout."""
    num_shards_by_prefix: dict[str, int] = {}
    input_shards_by_prefix: dict[str, int] = {}
    targets_by_prefix: dict[str, set[str]] = {}
    for name, (shape, _dtype) in state_dict_info.items():
        matched = _match_fused_modelopt_moe_weight(name)
        if matched is None:
            continue
        suffix, target = matched
        prefix = name[: -len(suffix)]
        if target.startswith("down_proj."):
            target = "w2_" + target.removeprefix("down_proj.")
        targets_by_prefix.setdefault(prefix, set()).add(target)
        if target == "w13_input_scale":
            if len(shape) == 1:
                input_shards = 1
            elif len(shape) == 2 and shape[1] in {1, 2}:
                input_shards = shape[1]
            else:
                raise ValueError(
                    f"Expected one or two W13 input scales per expert for {name}, "
                    f"got {tuple(shape)}"
                )
            input_shards_by_prefix[prefix] = input_shards
        if target != "w13_weight_scale_2":
            continue
        if len(shape) == 1:
            num_shards = 1
        elif len(shape) == 2 and shape[1] in {1, 2}:
            num_shards = shape[1]
        else:
            raise ValueError(
                f"Expected one or two W13 global scales per expert for {name}, "
                f"got {tuple(shape)}"
            )
        num_shards_by_prefix[prefix] = num_shards

    required_targets = {
        "w13_weight",
        "w13_weight_scale",
        "w13_weight_scale_2",
        "w2_weight",
        "w2_weight_scale",
        "w2_weight_scale_2",
    }
    if require_input_scales:
        required_targets.update({"w13_input_scale", "w2_input_scale"})
    for prefix, targets in targets_by_prefix.items():
        missing = required_targets - targets
        if missing:
            raise RuntimeError(
                f"Incomplete ModelOpt MoE export family for {prefix}: "
                f"missing {sorted(missing)}"
            )
    if set(num_shards_by_prefix) != set(targets_by_prefix):
        missing = set(targets_by_prefix) - set(num_shards_by_prefix)
        raise RuntimeError(
            "ModelOpt MoE export families are missing W13 global scales: "
            f"{sorted(missing)}"
        )
    if require_input_scales:
        mismatched = {
            prefix
            for prefix, num_shards in num_shards_by_prefix.items()
            if input_shards_by_prefix.get(prefix) != num_shards
        }
        if mismatched:
            raise RuntimeError(
                "ModelOpt MoE W13 input/global scale layouts disagree for: "
                f"{sorted(mismatched)}"
            )
    return num_shards_by_prefix


def _batch_fused_modelopt_moe_weights(
    weights: list[tuple[str, torch.Tensor]],
    *,
    w13_num_shards_by_prefix: dict[str, int],
) -> list[tuple[str, torch.Tensor]]:
    """Map fused ModelOpt payloads to vLLM per-projection checkpoint names.

    ``w2`` weights and block scales stay batched so vLLM can
    tensor-parallel-shard the full ``[E, ...]`` tensor at once.  Its scalar
    loader still requires an expert id, so only the tiny per-expert global
    scales are exposed as scalar views.

    Gated ``w13`` payloads are the exception on vLLM >= 0.25: they are emitted
    as per-expert 2-D shards instead, because ``RoutedExperts.load_weights``'
    fused-3D branch mis-transposes packed NVFP4. See the comment at the
    emission site below.
    """
    batched: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        matched = _match_fused_modelopt_moe_weight(name)
        if matched is None:
            batched.append((name, tensor))
            continue

        suffix, target = matched
        prefix = name[: -len(suffix)]
        if tensor.ndim == 0:
            raise ValueError(
                f"Fused ModelOpt MoE tensor must have an expert dimension: {name}"
            )

        if target in {"w13_weight", "w13_weight_scale"}:
            target_suffix = "weight" if target == "w13_weight" else "weight_scale"
            if w13_num_shards_by_prefix.get(prefix) == 1:
                batched.append(
                    (
                        f"{prefix}.experts.0.up_proj.{target_suffix}",
                        tensor,
                    )
                )
                continue
            if tensor.ndim < 2 or tensor.shape[1] % 2 != 0:
                raise ValueError(
                    f"Expected fused gate/up tensor with an even projection "
                    f"dimension for {name}, got {tuple(tensor.shape)}"
                )
            # Emit per-expert 2-D shards rather than batched 3-D tensors:
            # gated models (e.g. Qwen3-MoE) route batched tensors through
            # vLLM 0.25's RoutedExperts.load_weights fused branch, whose
            # orientation heuristic compares the last dim against the
            # unpacked hidden size and mis-transposes packed NVFP4 weights
            # (K/2 uint8) and block scales (K/16). Per-expert 2-D loads take
            # the same weight_loader path as the initial disk load.
            gate, up = tensor.chunk(2, dim=1)
            batched.extend(
                (
                    f"{prefix}.experts.{expert_id}.{projection}.{target_suffix}",
                    expert_weight,
                )
                for projection, shard in (
                    ("gate_proj", gate),
                    ("up_proj", up),
                )
                for expert_id, expert_weight in enumerate(shard.unbind(0))
            )
            continue

        if target == "w13_input_scale":
            if tensor.ndim == 1:
                tensor = tensor[:, None]
            if tensor.ndim != 2 or tensor.shape[1] not in {1, 2}:
                raise ValueError(
                    f"Expected one or two W13 input scales per expert for {name}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.shape[1] == 1:
                batched.extend(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.input_scale",
                        expert_scale[0],
                    )
                    for expert_id, expert_scale in enumerate(tensor.unbind(0))
                )
                continue
            for expert_id, expert_scale in enumerate(tensor.unbind(0)):
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.gate_proj.input_scale",
                        expert_scale[0],
                    )
                )
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.input_scale",
                        expert_scale[1],
                    )
                )
            continue

        if target == "w2_input_scale":
            if tensor.ndim == 2 and tensor.shape[1] == 1:
                tensor = tensor[:, 0]
            if tensor.ndim != 1:
                raise ValueError(
                    f"Expected one down-projection input scale per expert for "
                    f"{name}, got {tuple(tensor.shape)}"
                )
            batched.extend(
                (f"{prefix}.experts.{expert_id}.down_proj.input_scale", scale)
                for expert_id, scale in enumerate(tensor.unbind(0))
            )
            continue

        if target == "w13_weight_scale_2":
            if tensor.ndim == 1:
                tensor = tensor[:, None]
            if tensor.ndim != 2 or tensor.shape[1] not in {1, 2}:
                raise ValueError(
                    f"Expected one or two W13 global scales per expert for {name}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.shape[1] == 1:
                batched.extend(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.weight_scale_2",
                        expert_scale[0],
                    )
                    for expert_id, expert_scale in enumerate(tensor.unbind(0))
                )
                continue
            for expert_id, expert_scale in enumerate(tensor.unbind(0)):
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.gate_proj.weight_scale_2",
                        expert_scale[0],
                    )
                )
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.weight_scale_2",
                        expert_scale[1],
                    )
                )
            continue

        if not target.endswith("weight_scale_2"):
            batched.append((f"{prefix}.experts.0.{target}", tensor))
            continue

        if tensor.ndim == 1:
            expert_scales = tensor
        elif tensor.ndim == 2 and tensor.shape[1] == 1:
            expert_scales = tensor[:, 0]
        else:
            raise ValueError(
                f"Expected one global scale per expert for {name}, got "
                f"shape {tuple(tensor.shape)}"
            )

        batched.extend(
            (f"{prefix}.experts.{expert_id}.{target}", expert_scale)
            for expert_id, expert_scale in enumerate(expert_scales.unbind(0))
        )

    return batched


def _detach_pending_layerwise_weights(
    reload_roots: tuple[torch.nn.Module, ...],
    source_storage_ptrs: set[int],
) -> None:
    """Own deferred weights before a transport buffer may be reused.

    Completed layers have already released their buffered arguments, so this
    clones only tensors from a layer split across transport batches. Only the
    cached layerwise-reload subgraphs are inspected.
    """
    if not source_storage_ptrs:
        return
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    for reload_root in reload_roots:
        for module in reload_root.modules():
            info = get_layerwise_info(module)
            for _, arguments in info.loaded_weights:
                loaded_weight = arguments.arguments.get("loaded_weight")
                if not isinstance(loaded_weight, torch.Tensor):
                    continue
                if loaded_weight.untyped_storage().data_ptr() in source_storage_ptrs:
                    arguments.arguments["loaded_weight"] = loaded_weight.clone()


def _iter_modelopt_quant_modules(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """Return modules whose runtime layout is owned by vLLM ModelOpt methods."""
    return [
        (module_name, module)
        for module_name, module in model.named_modules()
        if _modelopt_target_kind(module) is not None
    ]


def _modelopt_layerwise_reload_roots(
    model: torch.nn.Module,
    *,
    include_fp8_kv_cache: bool,
) -> list[torch.nn.Module]:
    """Select disjoint roots that require vLLM's native reload lifecycle.

    Ordinary parameters are already updated in place by vLLM's checkpoint
    loaders.  Restricting layerwise reconstruction to ModelOpt runtime layouts
    and attention scale owners avoids materializing unrelated non-persistent
    buffers.  In vLLM 0.20, whole-model reconstruction can otherwise break a
    derived buffer that aliases a child parameter (for example Nemotron-H's
    ``conv_weights`` view of ``conv1d.weight``).
    """
    from vllm.model_executor.layers.attention import Attention, MLAAttention
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    modelopt_modules = {module for _, module in _iter_modelopt_quant_modules(model)}
    attention_types = (Attention, MLAAttention)
    quant_roots: list[torch.nn.Module] = []
    attention_roots: list[torch.nn.Module] = []
    visited: set[torch.nn.Module] = set()

    def collect(module: torch.nn.Module) -> None:
        if module in visited:
            return
        visited.add(module)
        if (
            include_fp8_kv_cache
            and isinstance(module, attention_types)
            and isinstance(getattr(module, "quant_method", None), BaseKVCacheMethod)
            and "fp8" in str(getattr(module, "kv_cache_dtype", "auto")).lower()
        ):
            attention_roots.append(module)
            return
        if module in modelopt_modules:
            quant_roots.append(module)
            return
        for child in module.children():
            collect(child)

    collect(model)
    # Match vLLM's ordering contract: process quantized modules before the
    # attention owners that finalize KV-cache scales.
    return quant_roots + attention_roots


def _require_complete_modelopt_layerwise_reload(model: torch.nn.Module) -> None:
    """Reject ModelOpt layers that vLLM would otherwise finalize partially."""
    candidates = _iter_modelopt_quant_modules(model)

    if not candidates:
        return

    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    incomplete = []
    for module_name, module in candidates:
        info = get_layerwise_info(module)
        if info.load_numel_total is None:
            # A completed layer is processed and reset immediately by vLLM.
            continue
        if info.load_numel == info.load_numel_total:
            continue
        buffered = sorted({name for name, _ in info.loaded_weights})
        incomplete.append(
            f"{module_name or '<root>'}: {info.load_numel}/"
            f"{info.load_numel_total} elements, buffered={buffered}"
        )

    if incomplete:
        details = "; ".join(incomplete[:8])
        suffix = "; ..." if len(incomplete) > 8 else ""
        raise RuntimeError(
            "ModelOpt layerwise reload is incomplete for "
            f"{len(incomplete)} layer(s): {details}{suffix}"
        )


if os.environ.get("VLLM_MODELOPT_REAL_QUANT", "0") == "1":
    from nemo_rl.modelopt.models.generation.vllm_modelopt import (
        register_nemo_modelopt_nvfp4,
    )

    register_nemo_modelopt_nvfp4()


class VllmQuantInternalWorkerExtension(VllmInternalWorkerExtension):
    _nrl_w13_num_shards_by_prefix: dict[str, int]
    _nrl_real_quant_source: _RealQuantSource = "modelopt"
    _nrl_bf16_staging: dict[str, dict[str, torch.Tensor]] = {}
    _nrl_bf16_quantizable_names: set[str] = set()
    _nrl_bf16_mode: NVFP4RefitMode = "w4a16"
    _nrl_bf16_calibration: NVFP4Calibration | None = None
    _nrl_bf16_expected_input_scale_names: set[str] = set()
    _nrl_bf16_input_scale_cache: dict[str, torch.Tensor] = {}
    _nrl_collective_group_members: dict[str, tuple[str, ...]] = {}
    _nrl_collective_grouped_projections: dict[str, str] = {}
    _nrl_collective_bf16_staging: dict[str, dict[str, torch.Tensor]] = {}
    _nrl_modelopt_reload_roots: tuple[torch.nn.Module, ...] | None = None

    def maybe_init_zmq(self) -> None:
        """Use a longer timeout only for ModelOpt real-quant refits."""
        super().maybe_init_zmq()
        if self._is_real_quant_model():
            self.zmq_socket.setsockopt(zmq.SNDTIMEO, MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS)
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS)

    def _is_real_quant_model(self) -> bool:
        return os.environ.get("VLLM_MODELOPT_REAL_QUANT", "0") == "1"

    def _get_modelopt_reload_roots(self) -> tuple[torch.nn.Module, ...]:
        """Return the invariant ModelOpt layerwise-reload subgraphs."""
        if self._nrl_modelopt_reload_roots is None:
            self._nrl_modelopt_reload_roots = tuple(
                _modelopt_layerwise_reload_roots(
                    self.model_runner.model,
                    include_fp8_kv_cache=self._uses_fp8_kv_cache(),
                )
            )
        return self._nrl_modelopt_reload_roots

    @contextmanager
    def _weight_update_lifecycle(
        self, transport: WeightUpdateTransport
    ) -> Iterator[WeightUpdateFinalizer]:
        """Use vLLM's native layerwise reload lifecycle for real quantization."""
        if not self._is_real_quant_model():
            with super()._weight_update_lifecycle(transport) as finalize:
                yield finalize
            return

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        model = self.model_runner.model
        reload_roots = self._get_modelopt_reload_roots()

        def finalize() -> None:
            try:
                self._require_complete_bf16_refit_groups()
                with torch.device(self.device):
                    _require_complete_modelopt_layerwise_reload(model)
                    for reload_root in reload_roots:
                        finalize_layerwise_reload(reload_root, self.model_config)
                # NCCL-Reshard owns its completion fence after this finalizer.
                # Legacy collective return and IPC COMPLETE acknowledgment do not.
                if transport != "nccl_reshard":
                    torch.accelerator.synchronize()
            except Exception as error:
                if transport == "ipc":
                    raise RuntimeError(
                        f"ModelOpt real-quant refit post-processing failed: {error}"
                    ) from error
                raise

        try:
            # Layerwise loading may reconstruct backend CustomOps as soon as a
            # layer becomes complete. Keep vLLM's worker config available for
            # that online processing as well as deferred finalization.
            with set_current_vllm_config(self.model_runner.vllm_config):
                with torch.device(self.device):
                    for reload_root in reload_roots:
                        initialize_layerwise_reload(reload_root)
                yield finalize
        except IPCWeightManifestError as error:
            raise RuntimeError(
                f"ModelOpt real-quant refit rejected: {error}"
            ) from error
        except Exception as error:
            if transport == "collective":
                raise RuntimeError(
                    "ModelOpt real-quant collective refit failed"
                ) from error
            raise

    def _weight_update_errors_are_fatal(self) -> bool:
        return self._is_real_quant_model()

    def _synchronize_before_ipc_data_ack(self) -> None:
        """Fence all accelerator streams used by ModelOpt post-load methods."""
        if self._is_real_quant_model():
            torch.accelerator.synchronize()
            return
        super()._synchronize_before_ipc_data_ack()

    def _require_complete_bf16_refit_groups(self) -> None:
        """Reject logical BF16 groups that never received all members."""
        if self._nrl_real_quant_source != "bf16":
            return

        incomplete = []
        for group_name, tensors in sorted(self._nrl_collective_bf16_staging.items()):
            expected = set(self._nrl_collective_group_members[group_name])
            missing = sorted(expected - set(tensors))
            if missing:
                incomplete.append(f"collective group {group_name}: missing {missing}")
        for group_name, tensors in sorted(self._nrl_bf16_staging.items()):
            expected = nvfp4_refit_group(next(iter(tensors)))[1]
            missing = sorted(set(expected) - set(tensors))
            if missing:
                incomplete.append(f"{group_name}: missing {missing}")
        missing_input_scales = sorted(
            self._nrl_bf16_expected_input_scale_names
            - self._nrl_bf16_input_scale_cache.keys()
        )
        if missing_input_scales:
            incomplete.append(f"static input scales: missing {missing_input_scales}")
        if incomplete:
            raise RuntimeError(
                "BF16 NVFP4 refit is incomplete: " + "; ".join(incomplete)
            )

    def prepare_refit_info(
        self,
        state_dict_info: dict[str, Any],
        serialized_fp8_config: dict[str, Any] | None = None,
    ) -> RefitTransformResponse:
        if not self._is_real_quant_model():
            return super().prepare_refit_info(state_dict_info, serialized_fp8_config)
        self.state_dict_info = state_dict_info
        quant_config = (
            self.model_runner.vllm_config.model_config.hf_config.quantization_config
        )
        ignore_patterns = quant_config.get("ignore", []) or []
        require_input_scales = (
            str(quant_config.get("quant_algo", "")).upper() == "NVFP4"
        )
        source_info = _classify_real_quant_source(
            state_dict_info,
            model=self.model_runner.model,
            ignore_patterns=ignore_patterns,
            require_input_scales=require_input_scales,
        )
        self._nrl_real_quant_source = source_info.source
        self._nrl_bf16_quantizable_names = set(source_info.bf16_names)
        self._nrl_bf16_staging = {}
        self._nrl_bf16_mode = "w4a4" if require_input_scales else "w4a16"
        self._nrl_bf16_calibration = None
        self._nrl_bf16_expected_input_scale_names = set()
        self._nrl_bf16_input_scale_cache = {}
        self._nrl_collective_group_members = {}
        self._nrl_collective_grouped_projections = {}
        self._nrl_collective_bf16_staging = {}
        if self._nrl_real_quant_source == "bf16" and self._nrl_bf16_mode == "w4a4":
            calibration_path = os.environ.get("VLLM_MODELOPT_CALIBRATION_PATH")
            if not calibration_path:
                raise ValueError(
                    "BF16 W4A4 refit requires VLLM_MODELOPT_CALIBRATION_PATH"
                )
            quant_cfg = os.environ.get("VLLM_MODELOPT_CALIBRATION_QUANT_CFG")
            if not quant_cfg:
                raise ValueError(
                    "BF16 W4A4 refit requires VLLM_MODELOPT_CALIBRATION_QUANT_CFG"
                )
            model_config = self.model_runner.vllm_config.model_config
            model_id, model_revision = _vllm_calibration_provenance(model_config)
            self._nrl_bf16_calibration = load_nvfp4_calibration(
                calibration_path,
                model_id=model_id,
                model_revision=model_revision,
                quant_cfg=quant_cfg,
                expected_projection_names=self._nrl_bf16_quantizable_names,
            )
            self._nrl_bf16_expected_input_scale_names = {
                _input_scale_name(name) for name in self._nrl_bf16_quantizable_names
            }

        self._get_modelopt_reload_roots()
        self._nrl_w13_num_shards_by_prefix = source_info.w13_num_shards_by_prefix
        if (
            self._nrl_w13_num_shards_by_prefix
            and self.model_runner.vllm_config.parallel_config.enable_expert_parallel
        ):
            raise RuntimeError(
                "Fused ModelOpt MoE refits require all experts local; "
                "vLLM expert parallelism is unsupported"
            )
        if self._nrl_real_quant_source == "bf16":
            return [
                RefitTransformRequest(
                    parameter_names=tuple(sorted(self._nrl_bf16_quantizable_names)),
                    source_format="bf16",
                    target_format=f"nvfp4_{self._nrl_bf16_mode}",
                    transform_location="destination",
                )
            ]
        return None

    def build_hf_to_local_param_map(self, refit_info: dict) -> HFToLocalParamMap:
        """Prepare NVFP4 completion groups before building receive specs."""
        group_members: dict[str, list[str]] = {}
        grouped_projections: dict[str, str] = {}
        for layer_name in refit_info["layer_names"]:
            for param_info in refit_info["per_layer_params"][layer_name]:
                if not str(param_info.get("transform_id", "")).startswith(
                    "bf16_to_nvfp4_"
                ):
                    continue
                name = str(param_info["name"])
                completion_key = str(param_info.get("completion_key", name))
                group_members.setdefault(completion_key, []).append(name)
                grouped_projection = param_info.get("grouped_expert_proj")
                if grouped_projection:
                    grouped_projections[name] = str(grouped_projection)

        self._nrl_collective_group_members = {
            key: tuple(names) for key, names in group_members.items()
        }
        self._nrl_collective_grouped_projections = grouped_projections
        self._nrl_collective_bf16_staging = {}
        return super().build_hf_to_local_param_map(refit_info)

    def _build_receiver_transform_param_spec(
        self,
        *,
        hf_name: str,
        param_info: dict[str, Any],
        value_param: torch.Tensor,
        merged_slice: tuple[slice, ...] | None,
        wire_local_shape: tuple[int, ...] | None,
        wire_dtype: torch.dtype,
    ) -> LocalParamSpec | None:
        """Build destination-local BF16 scratch for NVFP4 NCCL conversion."""
        del merged_slice
        transform_id = str(param_info.get("transform_id", ""))
        if not transform_id.startswith("bf16_to_nvfp4_"):
            return None
        expected_transform = f"bf16_to_nvfp4_{self._nrl_bf16_mode}"
        if transform_id != expected_transform:
            raise ValueError(
                f"NVFP4 receiver for {hf_name!r} expected transform "
                f"{expected_transform!r}, got {transform_id!r}."
            )
        if param_info.get("finalize_scope") != "model":
            raise ValueError(
                f"NVFP4 receiver for {hf_name!r} requires finalize_scope='model'."
            )
        if wire_dtype != torch.bfloat16:
            raise ValueError(
                f"NVFP4 receiver for {hf_name!r} requires a BF16 wire tensor, "
                f"got {wire_dtype}."
            )
        if wire_local_shape is None:
            raise ValueError(
                f"NVFP4 receiver for {hf_name!r} cannot derive its destination-local "
                "BF16 scratch shape from placement metadata."
            )
        if len(wire_local_shape) not in {2, 3} or wire_local_shape[-1] % 16:
            raise ValueError(
                f"NVFP4 receiver for {hf_name!r} requires a 2-D projection or "
                f"grouped 3-D expert tensor with local K divisible by 16, got "
                f"{wire_local_shape}."
            )

        wire_components = param_info.get(
            "wire_components", param_info.get("components", [])
        )
        wire_global_shape = tuple(param_info.get("global_shape", ()))
        actual_wire = [
            (
                component.get("role"),
                tuple(component.get("global_shape", ())),
                str(component.get("dtype")),
            )
            for component in wire_components
        ]
        expected_wire = [("weight", wire_global_shape, "torch.bfloat16")]
        if actual_wire != expected_wire:
            raise ValueError(
                f"NVFP4 receiver for {hf_name!r} has wire family "
                f"{actual_wire!r}, expected {expected_wire!r}."
            )

        destination_components = param_info.get("destination_components", [])
        expected_destination = [
            (
                "weight",
                (*wire_global_shape[:-1], wire_global_shape[-1] // 2),
                "torch.uint8",
                "codec",
            ),
            (
                "weight_scale",
                (*wire_global_shape[:-1], wire_global_shape[-1] // 16),
                "torch.float8_e4m3fn",
                "codec",
            ),
            (
                "weight_scale_2",
                wire_global_shape[:-2],
                "torch.float32",
                "codec",
            ),
        ]
        if self._nrl_bf16_mode == "w4a4":
            expected_destination.append(
                (
                    "input_scale",
                    wire_global_shape[:-2],
                    "torch.float32",
                    "calibration",
                )
            )
        actual_destination = [
            (
                component.get("role"),
                tuple(component.get("global_shape", ())),
                str(component.get("dtype")),
                component.get("source"),
            )
            for component in destination_components
        ]
        if actual_destination != expected_destination:
            raise ValueError(
                f"NVFP4 receiver for {hf_name!r} has destination family "
                f"{actual_destination!r}, expected {expected_destination!r}."
            )

        completion_key = str(param_info.get("completion_key", hf_name))

        def pre(_base: torch.Tensor) -> RefitCtx:
            return RefitCtx(
                buf=torch.empty(
                    wire_local_shape,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            )

        def post(ctx: RefitCtx) -> None:
            self._stage_collective_bf16_weight(
                completion_key=completion_key,
                hf_name=hf_name,
                weight=ctx.buf,
            )

        return LocalParamSpec(base=value_param, pre=pre, post=post)

    def _stage_collective_bf16_weight(
        self,
        *,
        completion_key: str,
        hf_name: str,
        weight: torch.Tensor,
    ) -> None:
        """Load one completed collective group through canonical NVFP4 serialization."""
        expected_names = self._nrl_collective_group_members.get(completion_key)
        if expected_names is None or hf_name not in expected_names:
            raise RuntimeError(
                f"Unknown NVFP4 collective completion group {completion_key!r} "
                f"for {hf_name!r}."
            )
        staged = self._nrl_collective_bf16_staging.setdefault(completion_key, {})
        if hf_name in staged:
            raise RuntimeError(
                f"Duplicate NVFP4 collective tensor {hf_name!r} in "
                f"completion group {completion_key!r}."
            )
        staged[hf_name] = weight
        if set(staged) != set(expected_names):
            return

        expanded_weights: list[tuple[str, torch.Tensor]] = []
        for name in expected_names:
            tensor = staged[name]
            grouped_projection = self._nrl_collective_grouped_projections.get(name)
            if grouped_projection is None:
                expanded_weights.append((name, tensor))
                continue
            if tensor.ndim != 3:
                raise ValueError(
                    f"Grouped NVFP4 collective tensor {name!r} must be [E, M, K], "
                    f"got {tuple(tensor.shape)}."
                )
            suffix = f".experts.{grouped_projection}.weight"
            if not name.endswith(suffix):
                raise ValueError(
                    f"Grouped NVFP4 collective tensor {name!r} does not match "
                    f"projection {grouped_projection!r}."
                )
            prefix = name.removesuffix(suffix)
            expanded_weights.extend(
                (
                    f"{prefix}.experts.{expert_id}.{grouped_projection}.weight",
                    expert_weight,
                )
                for expert_id, expert_weight in enumerate(tensor.unbind(0))
            )

        self._nrl_collective_bf16_staging.pop(completion_key)
        self._nrl_bf16_quantizable_names.update(name for name, _ in expanded_weights)
        if self._nrl_bf16_mode == "w4a4":
            self._nrl_bf16_expected_input_scale_names.update(
                _input_scale_name(name) for name, _ in expanded_weights
            )
        self._load_weights(expanded_weights)

    @contextmanager
    def _patch_named_parameters_to_include_buffers(self, model):
        """Temporarily patches model.named_parameters() to also yield input_quantizer buffers.

        Weights arrive pre-folded from the Megatron side, so only input_quantizer
        amax buffers need to be loaded. Weight quantizer buffers are skipped.
        """
        original_named_parameters = model.named_parameters
        # input_quantizer buffers we attached a weight_loader to and must
        # clean up on exit; pre-existing loaders (if any) are left untouched.
        patched_quantizer_buffers = []

        def input_amax_loader(param, loaded_weight, *args, **kwargs):
            param.copy_(torch.max(param, loaded_weight))

        def new_named_parameters(self, *args, **kwargs):
            yield from original_named_parameters(*args, **kwargs)
            for name, buf in self.named_buffers(*args, **kwargs):
                if "input_quantizer" not in name:
                    continue
                if not hasattr(buf, "weight_loader"):
                    buf.weight_loader = input_amax_loader
                    patched_quantizer_buffers.append(buf)
                yield name, buf

        model.named_parameters = types.MethodType(new_named_parameters, model)
        try:
            yield
        finally:
            model.named_parameters = original_named_parameters
            for buf in patched_quantizer_buffers:
                del buf.weight_loader

    @contextmanager
    def _attach_input_quantizer_amax_loaders(self, model):
        """Eagerly attach weight_loaders to input_quantizer amax buffers.

        vLLM >= 0.25 loads refit weights through per-module
        ``load_weights`` (e.g. ``LinearBase.load_weights``), which resolves
        targets via ``getattr`` and calls ``param.weight_loader(param,
        loaded_weight, shard_id)`` directly — it never iterates
        ``model.named_parameters()``, so the lazy attach in
        ``_patch_named_parameters_to_include_buffers`` no longer fires and
        quantizer amax buffers arrive without a loader (AttributeError:
        'Tensor' object has no attribute 'weight_loader').
        """

        def input_amax_loader(param, loaded_weight, *args, **kwargs):
            param.copy_(torch.max(param, loaded_weight))

        attached = []
        for name, buf in model.named_buffers():
            if "input_quantizer" not in name:
                continue
            if not hasattr(buf, "weight_loader"):
                buf.weight_loader = input_amax_loader
                attached.append(buf)
        try:
            yield
        finally:
            for buf in attached:
                del buf.weight_loader

    def _load_bf16_weights(
        self,
        weights: list[tuple[str, torch.Tensor]],
        *,
        ignore_patterns: list[str],
    ) -> tuple[
        list[tuple[str, torch.Tensor]],
        dict[str, torch.Tensor],
    ]:
        """Serialize complete BF16 groups and return static-scale replay state."""
        direct_weights: list[tuple[str, torch.Tensor]] = []
        incoming_groups: dict[str, dict[str, torch.Tensor]] = {}
        incoming_names: set[str] = set()
        for name, weight in weights:
            ignored = _is_ignored_real_quant_tensor(name, ignore_patterns)
            suffix = name.rsplit(".", 1)[-1]
            if ignored and suffix in {
                "weight_scale",
                "weight_scale_2",
                "input_scale",
            }:
                continue
            if ignored or name not in self._nrl_bf16_quantizable_names:
                direct_weights.append((name, weight))
                continue

            group_name, _expected_names = nvfp4_refit_group(name)
            incoming_groups.setdefault(group_name, {})[name] = weight
            incoming_names.add(name)

        serialized_weights: list[tuple[str, torch.Tensor]] = []
        replayed_input_scales: dict[str, torch.Tensor] = {}
        for group_name, incoming in incoming_groups.items():
            group_tensors = dict(self._nrl_bf16_staging.get(group_name, {}))
            group_tensors.update(incoming)
            expected_names = nvfp4_refit_group(next(iter(group_tensors)))[1]
            if set(group_tensors) != set(expected_names):
                self._nrl_bf16_staging[group_name] = {
                    name: tensor.detach().clone() if name in incoming_names else tensor
                    for name, tensor in group_tensors.items()
                }
                continue

            group_weights = serialize_bf16_nvfp4_group(
                group_tensors,
                mode=self._nrl_bf16_mode,
                calibration=self._nrl_bf16_calibration,
            )
            if self._nrl_bf16_mode == "w4a4":
                expected_input_scales = {
                    _input_scale_name(name) for name in expected_names
                }
                actual_input_scales = {
                    name for name, _ in group_weights if name.endswith(".input_scale")
                }
                if actual_input_scales != expected_input_scales:
                    missing = sorted(expected_input_scales - actual_input_scales)
                    unexpected = sorted(actual_input_scales - expected_input_scales)
                    raise RuntimeError(
                        "BF16 W4A4 serialization produced an invalid input-scale "
                        f"family for {group_name}: missing {missing}; "
                        f"unexpected {unexpected}"
                    )

                fixed_group_weights = []
                for name, serialized_tensor in group_weights:
                    fixed_tensor = serialized_tensor
                    if name in expected_input_scales:
                        if name in self._nrl_bf16_input_scale_cache:
                            fixed_tensor = self._nrl_bf16_input_scale_cache[name]
                        if name in replayed_input_scales:
                            raise RuntimeError(
                                f"Duplicate BF16 W4A4 input scale {name!r}"
                            )
                        replayed_input_scales[name] = fixed_tensor
                    fixed_group_weights.append((name, fixed_tensor))
                group_weights = fixed_group_weights

            serialized_weights.extend(group_weights)
            self._nrl_bf16_staging.pop(group_name, None)

        return direct_weights + serialized_weights, replayed_input_scales

    def _load_weights(self, weights):
        """Load pre-folded weights and input_quantizer amax buffers.

        Weights arrive already folded from the Megatron side (weight_quantizer
        applied during export), so no fold_weight step is needed here.
        """
        if self._is_real_quant_model():
            weights = list(weights)
            source_storage_ptrs = {
                tensor.untyped_storage().data_ptr() for _, tensor in weights
            }
            quant_config = (
                self.model_runner.vllm_config.model_config.hf_config.quantization_config
            )
            ignore_patterns = quant_config.get("ignore", []) or []
            if self._nrl_real_quant_source == "bf16":
                source_weights = list(weights)
                weights, replayed_input_scales = self._load_bf16_weights(
                    source_weights,
                    ignore_patterns=ignore_patterns,
                )
                if not weights:
                    return None
                try:
                    with torch.device(self.device):
                        result = super()._load_weights(weights)
                    for name, tensor in replayed_input_scales.items():
                        self._nrl_bf16_input_scale_cache.setdefault(
                            name,
                            tensor.detach().clone(),
                        )
                    return result
                finally:
                    with torch.device(self.device):
                        _detach_pending_layerwise_weights(
                            self._get_modelopt_reload_roots(),
                            source_storage_ptrs,
                        )

            filtered = []
            for name, weight in weights:
                suffix = name.rsplit(".", 1)[-1]
                ignored = matches_quant_ignore_pattern(name, ignore_patterns)
                if ignored and suffix in {
                    "weight_scale",
                    "weight_scale_2",
                    "input_scale",
                }:
                    continue

                filtered.append((name, weight))
            if any(
                _match_fused_modelopt_moe_weight(name) is not None
                for name, _ in filtered
            ):
                weights = _batch_fused_modelopt_moe_weights(
                    filtered,
                    w13_num_shards_by_prefix=self._nrl_w13_num_shards_by_prefix,
                )
            else:
                weights = filtered
            if not weights:
                return None
            try:
                with torch.device(self.device):
                    return super()._load_weights(weights)
            finally:
                with torch.device(self.device):
                    _detach_pending_layerwise_weights(
                        self._get_modelopt_reload_roots(),
                        source_storage_ptrs,
                    )

        with ExitStack() as contexts:
            for _, child in self.model_runner.model.named_children():
                contexts.enter_context(
                    self._patch_named_parameters_to_include_buffers(child)
                )
            contexts.enter_context(
                self._attach_input_quantizer_amax_loaders(self.model_runner.model)
            )
            return super()._load_weights(weights)

    def get_weight_snapshot(self, name: str) -> torch.Tensor:
        """Return a CPU copy of a named parameter for before/after comparison."""
        model = self.model_runner.model
        for n, p in model.named_parameters():
            if n == name:
                return p.detach().cpu().clone()
        raise KeyError(f"Parameter '{name}' not found in model")

    def get_quantizer_stats(self) -> dict:
        """Return summary statistics for all TensorQuantizer modules.

        Matches the interface of MegatronQuantPolicyWorker.get_quantizer_stats().
        """
        total = 0
        enabled = 0
        with_amax = 0
        positive_amax = 0
        model = self.model_runner.model
        for _, module in model.named_modules():
            if isinstance(module, TensorQuantizer):
                total += 1
                if module.is_enabled:
                    enabled += 1
                    if hasattr(module, "amax") and module.amax is not None:
                        with_amax += 1
                        if (module.amax > 0).all():
                            positive_amax += 1
        return {
            "total": total,
            "enabled": enabled,
            "with_amax": with_amax,
            "positive_amax": positive_amax,
        }


class VllmQuantInternalWorkerExtensionWithCheckpointEngine(
    VllmCheckpointEngineMixin, VllmQuantInternalWorkerExtension
):
    """ModelOpt worker extension with checkpoint-engine refit support."""
