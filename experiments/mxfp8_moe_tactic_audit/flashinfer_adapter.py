"""Pinned FlashInfer 0.6.13 boundary for the MXFP8 MoE tactic audit."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import importlib.metadata
import operator
from typing import Any

import torch  # pyright: ignore[reportMissingImports]

try:
    from .schema import ReplayProfile, TacticPair
except ImportError:  # pragma: no cover - direct script execution
    from schema import ReplayProfile, TacticPair


SUPPORTED_FLASHINFER_VERSION = "0.6.13"
FLASHINFER_DISTRIBUTION = "flashinfer-python"
MOE_CUSTOM_OP = "flashinfer::trtllm_fp8_block_scale_moe"
MOE_RUNNER = "MoERunner"


@dataclass(frozen=True)
class MoeKernelCase:
    """One in-memory MXFP8 MoE workload replay case."""

    profile: ReplayProfile
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    output: torch.Tensor
    activation_type: int
    routing_method_type: int
    local_expert_offset: int


def assert_supported_flashinfer() -> None:
    """Fail unless the installed FlashInfer distribution is exactly 0.6.13."""
    try:
        installed = importlib.metadata.version(FLASHINFER_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"MXFP8 MoE tactic audit requires FlashInfer {SUPPORTED_FLASHINFER_VERSION}; "
            f"{FLASHINFER_DISTRIBUTION} is not installed"
        ) from error
    if installed != SUPPORTED_FLASHINFER_VERSION:
        raise RuntimeError(
            f"MXFP8 MoE tactic audit requires FlashInfer {SUPPORTED_FLASHINFER_VERSION}; "
            f"found {installed}"
        )


def normalize_tactic_pair(raw: object) -> TacticPair:
    """Normalize one FlashInfer tactic iterable to a typed FC1/FC2 pair."""
    if isinstance(raw, (str, bytes, bytearray)):
        raise TypeError("FlashInfer tactic must be a two-integer iterable")
    try:
        values = tuple(raw)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("FlashInfer tactic must be a two-integer iterable") from error
    if len(values) != 2:
        raise ValueError("FlashInfer tactic must contain exactly FC1 and FC2 IDs")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("FlashInfer tactic IDs must be integers")
        try:
            normalized.append(int(operator.index(value)))
        except TypeError as error:
            raise TypeError("FlashInfer tactic IDs must be integers") from error
    return TacticPair(gemm1=normalized[0], gemm2=normalized[1])


def _load_flashinfer_bindings() -> tuple[Any, Any, Any, Any]:
    """Load optional FlashInfer symbols only inside the adapter boundary."""
    jit_moe = importlib.import_module("flashinfer.jit.fused_moe")
    enums = importlib.import_module("flashinfer.tllm_enums")
    return (
        jit_moe.gen_trtllm_gen_fused_moe_sm100_module,
        enums.DtypeTrtllmGen,
        enums.Fp8QuantizationType,
        enums.WeightLayout,
    )


def enumerate_valid_tactics(case: MoeKernelCase) -> tuple[TacticPair, ...]:
    """Enumerate legal paired FC1/FC2 tactics through TRTLLM Gen MoE."""
    assert_supported_flashinfer()
    module_factory, dtype_enum, fp8_enum, weight_layout = _load_flashinfer_bindings()
    signature = case.profile.signature
    moe_op = module_factory().build_and_load()
    raw_tactics = moe_op.trtllm_get_valid_moe_configs(
        dtype_enum.MxE4m3,
        dtype_enum.MxE4m3,
        fp8_enum.MxFp8,
        signature.top_k,
        signature.hidden_size,
        signature.intermediate_size,
        signature.local_num_experts,
        case.activation_type,
        True,
        weight_layout.MajorK.value,
        False,
        signature.num_tokens,
        False,
    )
    return tuple(
        sorted(
            {normalize_tactic_pair(raw) for raw in raw_tactics},
            key=lambda tactic: (tactic.gemm1, tactic.gemm2),
        )
    )


def _last_positive_power_of_two(value: int) -> int:
    return 1 << (max(value, 1).bit_length() - 1)


def tune_max_num_tokens(case: MoeKernelCase) -> int:
    """Return the upstream-tested token cap used to pin the cache bucket."""
    return max(_last_positive_power_of_two(case.profile.signature.num_tokens), 16)


def _profile_shape(shape: torch.Size | tuple[int, ...], bucket: int) -> tuple[int, ...]:
    values = tuple(shape)
    return (bucket, *values[1:])


def cache_key_for_case(case: MoeKernelCase, *, has_gemm1_lora_delta: bool) -> str:
    """Build the exact hash-free AutoTuner file key for one replay case."""
    signature = case.profile.signature
    bucket = min(
        _last_positive_power_of_two(signature.num_tokens),
        tune_max_num_tokens(case),
    )
    profile_shapes = (
        _profile_shape(case.output.shape, bucket),
        (0,),
        (bucket, signature.top_k),
        (0,),
        _profile_shape(case.hidden_states.shape, bucket),
        _profile_shape(case.hidden_states_scale.shape, bucket),
        (bucket, signature.top_k, 2 * signature.intermediate_size)
        if has_gemm1_lora_delta
        else (0,),
        (0,),
    )
    return str((MOE_CUSTOM_OP, MOE_RUNNER, profile_shapes, ()))


def _get_autotuner() -> Any:
    """Return FlashInfer's process-local AutoTuner singleton."""
    autotuner = importlib.import_module("flashinfer.autotuner")
    return autotuner.AutoTuner.get()


def _validate_moe_file_key(cache_key: str) -> None:
    try:
        parsed = ast.literal_eval(cache_key)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            "cache_key must be the exact FlashInfer MoE file key"
        ) from error
    if (
        not isinstance(parsed, tuple)
        or len(parsed) != 4
        or parsed[0] != MOE_CUSTOM_OP
        or parsed[1] != MOE_RUNNER
        or not isinstance(parsed[2], tuple)
        or parsed[3] != ()
    ):
        raise ValueError("cache_key must be the exact FlashInfer MoE file key")


@contextmanager
def force_tactic(cache_key: str, tactic: TacticPair) -> Iterator[None]:
    """Temporarily force one paired tactic without leaking AutoTuner state."""
    _validate_moe_file_key(cache_key)
    tuner = _get_autotuner()
    file_configs_snapshot = tuner._file_configs.copy()
    profiling_cache_snapshot = tuner.profiling_cache.copy()
    try:
        tuner._file_configs.clear()
        tuner.profiling_cache.clear()
        tuner._file_configs[cache_key] = (MOE_RUNNER, [tactic.gemm1, tactic.gemm2])
        yield
    finally:
        tuner._file_configs.clear()
        tuner._file_configs.update(file_configs_snapshot)
        tuner.profiling_cache.clear()
        tuner.profiling_cache.update(profiling_cache_snapshot)


@contextmanager
def force_stock_tactic(cache_key: str) -> Iterator[None]:
    """Temporarily force FlashInfer's built-in ``[-1, -1]`` fallback pair."""
    _validate_moe_file_key(cache_key)
    tuner = _get_autotuner()
    file_configs_snapshot = tuner._file_configs.copy()
    profiling_cache_snapshot = tuner.profiling_cache.copy()
    try:
        tuner._file_configs.clear()
        tuner.profiling_cache.clear()
        tuner._file_configs[cache_key] = (MOE_RUNNER, [-1, -1])
        yield
    finally:
        tuner._file_configs.clear()
        tuner._file_configs.update(file_configs_snapshot)
        tuner.profiling_cache.clear()
        tuner.profiling_cache.update(profiling_cache_snapshot)


def _load_moe_runtime() -> tuple[Any, Any, Any, Any]:
    flashinfer = importlib.import_module("flashinfer")
    fused_moe = importlib.import_module("flashinfer.fused_moe")
    enums = importlib.import_module("flashinfer.tllm_enums")
    return (
        flashinfer,
        fused_moe.trtllm_fp8_block_scale_routed_moe,
        enums.Fp8QuantizationType,
        enums.WeightLayout,
    )


def build_kernel_case(profile: ReplayProfile, device: torch.device) -> MoeKernelCase:
    """Build deterministic kernel-ready MXFP8 tensors for one replay profile."""
    assert_supported_flashinfer()
    if device.type != "cuda":
        raise ValueError("MXFP8 MoE kernel cases require a CUDA device")
    flashinfer, _, _, _ = _load_moe_runtime()
    signature = profile.signature
    generator = torch.Generator(device=device)
    generator.manual_seed(int(profile.signature_key[:16], 16))

    hidden_bf16 = torch.randn(
        signature.num_tokens,
        signature.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gemm1_bf16 = torch.randn(
        signature.local_num_experts,
        2 * signature.intermediate_size,
        signature.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gemm2_bf16 = torch.randn(
        signature.local_num_experts,
        signature.hidden_size,
        signature.intermediate_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    hidden_states, hidden_states_scale = flashinfer.mxfp8_quantize(hidden_bf16, False)
    gemm1_weights, gemm1_weights_scale = flashinfer.mxfp8_quantize(gemm1_bf16, True)
    gemm2_weights, gemm2_weights_scale = flashinfer.mxfp8_quantize(gemm2_bf16, True)
    hidden_states_scale = hidden_states_scale.view(torch.uint8).reshape(
        signature.num_tokens, -1
    )
    gemm1_weights_scale = gemm1_weights_scale.view(torch.uint8).reshape(
        signature.local_num_experts, 2 * signature.intermediate_size, -1
    )
    gemm2_weights_scale = gemm2_weights_scale.view(torch.uint8).reshape(
        signature.local_num_experts, signature.hidden_size, -1
    )
    return MoeKernelCase(
        profile=profile,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        output=torch.empty(
            signature.num_tokens,
            signature.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        activation_type=flashinfer.ActivationType.Swiglu.value,
        routing_method_type=flashinfer.RoutingMethodType.RenormalizeNaive.value,
        local_expert_offset=0,
    )


def run_moe_pair(
    case: MoeKernelCase,
    packed_topk: torch.Tensor,
    *,
    do_finalize: bool,
    gemm1_lora_delta: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Run the routed full MoE pair through the pinned FlashInfer wrapper."""
    assert_supported_flashinfer()
    _, routed_moe, fp8_enum, weight_layout = _load_moe_runtime()
    signature = case.profile.signature
    result = routed_moe(
        topk_ids=packed_topk,
        routing_bias=None,
        hidden_states=case.hidden_states,
        hidden_states_scale=case.hidden_states_scale,
        gemm1_weights=case.gemm1_weights,
        gemm1_weights_scale=case.gemm1_weights_scale,
        gemm2_weights=case.gemm2_weights,
        gemm2_weights_scale=case.gemm2_weights_scale,
        num_experts=signature.global_num_experts,
        top_k=signature.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=signature.intermediate_size,
        local_expert_offset=case.local_expert_offset,
        local_num_experts=signature.local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=case.routing_method_type,
        use_shuffled_weight=True,
        weight_layout=weight_layout.MajorK.value,
        do_finalize=do_finalize,
        output=case.output,
        tune_max_num_tokens=tune_max_num_tokens(case),
        fp8_quantization_type=fp8_enum.MxFp8,
        activation_type=case.activation_type,
        gemm1_lora_delta=gemm1_lora_delta,
    )
    if isinstance(result, torch.Tensor):
        return (result,)
    return tuple(result)
