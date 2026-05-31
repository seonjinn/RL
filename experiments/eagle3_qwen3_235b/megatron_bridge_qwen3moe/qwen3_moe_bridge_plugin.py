"""Qwen3 MoE bridge registration for the NeMo 25.07 Megatron-Bridge stack.

This is intentionally a narrow compatibility shim.  It reuses the container's
``/opt/Megatron-Bridge`` and ``/opt/megatron-lm`` packages, then registers only
the Qwen3 MoE HF architecture with the old bridge dispatch registry.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch
import torch.nn.functional as F

try:
    from megatron.core.models.gpt import GPTModel
except ImportError:  # pragma: no cover - depends on MCore package layout
    from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.model_bridge import MegatronModelBridge
try:
    from megatron.bridge.models.param_mapping import AutoMapping as _BridgeAutoMapping
except ImportError:  # pragma: no cover - depends on Megatron-Bridge version
    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping as _BridgeAutoMapping
    except ImportError:  # pragma: no cover - old stacks only expose TPAwareMapping
        _BridgeAutoMapping = None

from megatron.bridge.models.param_mapping import GatedMLPMapping, QKVMapping, TPAwareMapping

try:
    from transformers import Qwen3MoeForCausalLM
except ImportError:  # pragma: no cover - depends on transformers package layout
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


def _register_grouped_linear_module_types() -> None:
    register_module_type = getattr(TPAwareMapping, "register_module_type", None)
    if not callable(register_module_type):
        return
    # With the current Qwen3-235B rollout layout ETP is 1, so grouped expert
    # tensors are full-sized inside each TP rank. Treating these TE grouped
    # modules as column/row TP-sharded makes Megatron-Bridge scatter 1/TP-sized
    # chunks into full-sized expert tensors during HF import.
    for module_type, parallelism_type in (
        ("TEColumnParallelGroupedLinear", "replicated"),
        ("TERowParallelGroupedLinear", "replicated"),
    ):
        register_module_type(module_type, parallelism_type)


_register_grouped_linear_module_types()


def _target_tensor_from_param(
    megatron_param: str, module: torch.nn.Module
) -> Optional[torch.Tensor]:
    param_name = megatron_param.rsplit(".", 1)[-1]
    tensor = getattr(module, param_name, None)
    return tensor if isinstance(tensor, torch.Tensor) else None


@contextmanager
def _temporary_weight_attr(
    megatron_param: str, module: torch.nn.Module
) -> Iterator[torch.nn.Module]:
    if hasattr(module, "weight"):
        yield module
        return
    target = _target_tensor_from_param(megatron_param, module)
    if target is None:
        yield module
        return

    sentinel = object()
    old_weight = module.__dict__.get("weight", sentinel)
    module.__dict__["weight"] = target
    try:
        yield module
    finally:
        if old_weight is sentinel:
            module.__dict__.pop("weight", None)
        else:
            module.__dict__["weight"] = old_weight


if _BridgeAutoMapping is None:

    class AutoMapping(TPAwareMapping):
        """Compatibility subset of newer Megatron-Bridge AutoMapping."""

        _REPLICATED_PARAM_SUFFIXES = (
            "decoder.final_layernorm.weight",
            "self_attention.linear_qkv.layer_norm_weight",
            "self_attention.linear_qkv.layer_norm_bias",
            "mlp.router.weight",
            "pre_mlp_layernorm.weight",
            "self_attention.q_layernorm.weight",
            "self_attention.k_layernorm.weight",
        )

        def _is_replicated_param(self) -> bool:
            return self.megatron_param.endswith(self._REPLICATED_PARAM_SUFFIXES)

        def _detect_parallelism_type(self, module: torch.nn.Module) -> str:
            if self._is_replicated_param():
                return "replicated"
            return super()._detect_parallelism_type(module)

        def _target_tensor(self, module: torch.nn.Module) -> Optional[torch.Tensor]:
            tensor = _target_tensor_from_param(self.megatron_param, module)
            if tensor is not None:
                return tensor
            for tensor in module.parameters(recurse=False):
                return tensor
            for tensor in module.buffers(recurse=False):
                return tensor
            return None

        @staticmethod
        def _broadcast_device(target: torch.Tensor) -> torch.device:
            if target.device.type == "cuda":
                return target.device
            if not torch.cuda.is_available():
                return target.device
            return torch.device("cuda", torch.cuda.current_device())

        def hf_to_megatron(
            self,
            hf_weights: torch.Tensor,
            megatron_module: torch.nn.Module,
        ) -> torch.Tensor:
            if not self._is_replicated_param():
                with _temporary_weight_attr(self.megatron_param, megatron_module):
                    return super().hf_to_megatron(hf_weights, megatron_module)

            target = self._target_tensor(megatron_module)
            if target is None:
                return self._replicated_mapping.hf_to_megatron(
                    hf_weights, megatron_module
                )

            if self.tp_size == 1:
                return hf_weights.to(device=target.device, dtype=target.dtype)

            broadcast_device = self._broadcast_device(target)
            if self.tp_rank == 0:
                tensor = hf_weights.to(
                    device=broadcast_device, dtype=target.dtype
                ).contiguous()
            else:
                tensor = torch.empty(
                    target.shape,
                    device=broadcast_device,
                    dtype=target.dtype,
                )
            global_src = torch.distributed.get_global_rank(
                group=self.tp_group, group_rank=0
            )
            torch.distributed.broadcast(tensor, src=global_src, group=self.tp_group)
            return tensor.to(device=target.device) if target.device != broadcast_device else tensor

else:
    AutoMapping = _BridgeAutoMapping


class GroupedGatedMLPMapping(GatedMLPMapping):
    """Gated MLP mapping for grouped expert modules exposing weight0/weight1."""

    def hf_to_megatron(
        self,
        hf_weights: dict[str, torch.Tensor],
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor:
        with _temporary_weight_attr(self.megatron_param, megatron_module):
            return super().hf_to_megatron(hf_weights, megatron_module)


@dataclass
class Qwen3MoEModelProvider(GPTModelProvider):
    """Megatron-Core provider values for Qwen3 MoE models."""

    normalization: str = "RMSNorm"
    activation_func: Callable[..., Any] = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    kv_channels: Optional[int] = 128
    num_query_groups: int = 8
    seq_length: int = 40960
    max_position_embeddings: int = 40960
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    vocab_size: int = 151936
    share_embeddings_and_output_weights: Optional[bool] = False
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 1000000.0
    position_embedding_type: str = "rope"
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True

    num_moe_experts: int = 128
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_router_topk: int = 8
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_ffn_hidden_size: int = 1536
    expert_tensor_parallel_size: int = 1

    def finalize(self) -> "Qwen3MoEModelProvider":
        """Match newer Megatron-Bridge provider API when running on older stacks."""
        expert_tensor_parallel_size = getattr(self, "expert_tensor_parallel_size", None)
        finalize = getattr(super(), "finalize", None)
        if callable(finalize):
            result = finalize()
            provider = self if result is None else result
        else:
            provider = self
        # Older Bridge finalize can recouple MoE ETP to TP; preserve explicit ETP.
        if expert_tensor_parallel_size is not None:
            provider.expert_tensor_parallel_size = expert_tensor_parallel_size
        return provider


def _hf_config(hf_pretrained: Any) -> Any:
    return getattr(hf_pretrained, "config", hf_pretrained)


def _hf_generation_config(hf_pretrained: Any) -> Any:
    return getattr(hf_pretrained, "generation_config", None)


def _dtype_from_config(hf_config: Any) -> torch.dtype:
    dtype = getattr(hf_config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        lowered = dtype.lower().replace("torch.", "")
        if lowered in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if lowered in {"float16", "fp16", "half"}:
            return torch.float16
        if lowered in {"float32", "fp32"}:
            return torch.float32
    return torch.float32


@MegatronModelBridge.register_bridge(source=Qwen3MoeForCausalLM, target=GPTModel)
class Qwen3MoEBridge(MegatronModelBridge):
    """Bridge Qwen3MoeForCausalLM HF configs/weights to Megatron GPTModel."""

    def _dtype_from_hf(self, hf_config: Any) -> torch.dtype:
        dtype_from_hf = getattr(super(), "dtype_from_hf", None)
        if callable(dtype_from_hf):
            return dtype_from_hf(hf_config, default=torch.float32)
        dtype_from_hf = getattr(self, "dtype_from_hf", None)
        if callable(dtype_from_hf):
            return dtype_from_hf(hf_config, default=torch.float32)
        return _dtype_from_config(hf_config)

    def _make_vocab_size_divisible_by(self, vocab_size: int) -> int:
        make_divisible = getattr(super(), "make_vocab_size_divisible_by", None)
        if callable(make_divisible):
            return make_divisible(vocab_size)
        make_divisible = getattr(self, "make_vocab_size_divisible_by", None)
        if callable(make_divisible):
            return make_divisible(vocab_size)
        return 128

    def provider_bridge(self, hf_pretrained: Any) -> Qwen3MoEModelProvider:
        hf_config = _hf_config(hf_pretrained)
        params_dtype = self._dtype_from_hf(hf_config)

        return Qwen3MoEModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=getattr(hf_config, "head_dim", None),
            num_moe_experts=hf_config.num_experts,
            moe_router_topk=hf_config.num_experts_per_tok,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self._make_vocab_size_divisible_by(hf_config.vocab_size),
            rotary_base=hf_config.rope_theta,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            max_position_embeddings=hf_config.max_position_embeddings,
            fp16=(params_dtype == torch.float16),
            bf16=(params_dtype == torch.bfloat16),
            params_dtype=params_dtype,
            generation_config=_hf_generation_config(hf_pretrained),
            qk_layernorm=True,
            moe_grouped_gemm=True,
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        }
        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in param_mappings.items()
        ]
        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                GroupedGatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                GroupedGatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)


def registration_status() -> dict[str, Any]:
    from megatron.bridge.models.causal_bridge import CausalLMBridge

    supported_models = CausalLMBridge.list_supported_models()
    return {
        "registered": "Qwen3MoeForCausalLM" in supported_models,
        "supported_models": supported_models,
        "bridge_class": Qwen3MoEBridge.__name__,
        "provider_class": Qwen3MoEModelProvider.__name__,
    }


__all__ = ["Qwen3MoEBridge", "Qwen3MoEModelProvider", "registration_status"]
