"""Pinned FlashInfer 0.6.13 boundary for the MXFP8 MoE tactic audit."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import importlib.metadata
import operator
from pathlib import Path
from typing import Any, cast

import torch  # pyright: ignore[reportMissingImports]

try:
    from .schema import ReplayProfile, TacticPair
except ImportError:  # pragma: no cover - direct script execution
    from schema import ReplayProfile, TacticPair


SUPPORTED_FLASHINFER_VERSION = "0.6.13"
FLASHINFER_DISTRIBUTION = "flashinfer-python"
MOE_CUSTOM_OP = "flashinfer::trtllm_fp8_block_scale_moe"
MOE_RUNNER = "MoERunner"
MOE_LOG_KEY = (MOE_CUSTOM_OP, MOE_RUNNER)
PREPACKED_ARTIFACT_FORMAT = "flashinfer_mxfp8_moe_prepacked_v1"
PRODUCTION_TUNE_MAX_NUM_TOKENS = 8192


class TacticDispatchError(RuntimeError):
    """Raised when FlashInfer does not dispatch the forced file-config tactic."""


class IntermediateApiUnavailable(RuntimeError):
    """Raised when FlashInfer cannot expose the activated FC1 intermediate."""


class MonolithicApiUnavailable(RuntimeError):
    """Raised when the pinned FlashInfer monolithic MoE API is unavailable."""


@dataclass(frozen=True)
class PrepackedMoeWeights:
    """Validated shuffled MXFP8 expert weights ready for TRTLLM Gen MoE."""

    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    local_expert_offset: int


@dataclass(frozen=True)
class MoePairResult:
    """Typed result from one complete paired FC1/FC2 MoE invocation."""

    final_output: torch.Tensor | None
    activated_intermediate: torch.Tensor | None


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
    weight_layout: str
    use_shuffled_weight: bool
    prepacked_weight_format: str


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


def _next_positive_power_of_two(value: int) -> int:
    normalized = max(value, 1)
    return 1 << (normalized - 1).bit_length()


def _map_to_hybrid_bucket(value: int, maximum: int) -> int:
    """Mirror the token-bucket mapper pinned by FlashInfer 0.6.13."""
    if value <= 0:
        return 1
    if value >= maximum:
        return maximum
    if value <= 256:
        return _next_positive_power_of_two(value)
    if value <= 2048:
        return min(((value + 255) // 256) * 256, maximum)
    if value <= 4096:
        return min(((value + 511) // 512) * 512, maximum)
    return min(_next_positive_power_of_two(value), maximum)


def tune_max_num_tokens(case: MoeKernelCase) -> int:
    """Return the upstream-tested token cap used to pin the cache bucket."""
    return max(_next_positive_power_of_two(case.profile.signature.num_tokens), 16)


def _profile_shape(shape: torch.Size | tuple[int, ...], bucket: int) -> tuple[int, ...]:
    values = tuple(shape)
    return (bucket, *values[1:])


def cache_key_for_case(
    case: MoeKernelCase,
    *,
    has_gemm1_lora_delta: bool,
    router_logits: torch.Tensor | None = None,
) -> str:
    """Build the exact hash-free AutoTuner file key for one replay case."""
    signature = case.profile.signature
    if router_logits is not None:
        expected_shape = (signature.num_tokens, signature.global_num_experts)
        if tuple(router_logits.shape) != expected_shape:
            raise ValueError(
                f"router_logits shape mismatch: {tuple(router_logits.shape)} "
                f"!= {expected_shape}"
            )
        if has_gemm1_lora_delta:
            raise ValueError("monolithic replay does not support gemm1_lora_delta")
    bucket = _map_to_hybrid_bucket(
        signature.num_tokens,
        PRODUCTION_TUNE_MAX_NUM_TOKENS
        if router_logits is not None
        else tune_max_num_tokens(case),
    )
    if router_logits is None:
        profile_shapes = (
            _profile_shape(case.output.shape, bucket),
            (0,),
            (bucket, signature.top_k),
            (bucket,),
            _profile_shape(case.hidden_states.shape, bucket),
            _profile_shape(case.hidden_states_scale.shape, bucket),
            (bucket, signature.top_k, 2 * signature.intermediate_size)
            if has_gemm1_lora_delta
            else (0,),
            (0,),
        )
    else:
        profile_shapes = (
            _profile_shape(case.output.shape, bucket),
            _profile_shape(router_logits.shape, bucket),
            (0,),
            (0,),
            _profile_shape(case.hidden_states.shape, bucket),
            _profile_shape(case.hidden_states_scale.shape, bucket),
            (0,),
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
def _force_tactic_ids(cache_key: str, tactic_ids: list[int]) -> Iterator[None]:
    _validate_moe_file_key(cache_key)
    tuner = _get_autotuner()
    file_configs_snapshot = tuner._file_configs.copy()
    profiling_cache_snapshot = tuner.profiling_cache.copy()
    log_key_was_present = MOE_LOG_KEY in tuner._logged_file_hits
    try:
        tuner._file_configs.clear()
        tuner.profiling_cache.clear()
        tuner._logged_file_hits.discard(MOE_LOG_KEY)
        tuner._file_configs[cache_key] = (MOE_RUNNER, list(tactic_ids))
        yield
        if MOE_LOG_KEY not in tuner._logged_file_hits:
            raise TacticDispatchError(
                "forced tactic did not log an exact file hit for "
                f"{MOE_CUSTOM_OP}/{MOE_RUNNER}"
            )
        expected = (MOE_RUNNER, list(tactic_ids))
        if tuner._file_configs.get(cache_key) != expected:
            raise TacticDispatchError(
                f"forced file hit used the wrong tactic; expected {tactic_ids!r}"
            )
    finally:
        tuner._file_configs.clear()
        tuner._file_configs.update(file_configs_snapshot)
        tuner.profiling_cache.clear()
        tuner.profiling_cache.update(profiling_cache_snapshot)
        if log_key_was_present:
            tuner._logged_file_hits.add(MOE_LOG_KEY)
        else:
            tuner._logged_file_hits.discard(MOE_LOG_KEY)


@contextmanager
def force_tactic(cache_key: str, tactic: TacticPair) -> Iterator[None]:
    """Force and verify one paired tactic without leaking AutoTuner state."""
    with _force_tactic_ids(cache_key, [tactic.gemm1, tactic.gemm2]):
        yield


@contextmanager
def force_stock_tactic(cache_key: str) -> Iterator[None]:
    """Force and verify FlashInfer's literal ``[-1, -1]`` fallback pair."""
    with _force_tactic_ids(cache_key, [-1, -1]):
        yield


def observed_forced_cache_event(cache_key: str) -> str:
    """Return the cache event evidenced by the active forced dispatch."""
    tuner = _get_autotuner()
    configured = tuner._file_configs.get(cache_key)
    if MOE_LOG_KEY not in tuner._logged_file_hits or not (
        isinstance(configured, tuple)
        and len(configured) == 2
        and configured[0] == MOE_RUNNER
        and isinstance(configured[1], list)
    ):
        raise TacticDispatchError("forced dispatch has no observed file-cache event")
    return "fallback" if configured[1] == [-1, -1] else "cache hit"


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


def _load_monolithic_moe_runtime() -> tuple[Any, Any, Any]:
    try:
        fused_moe = importlib.import_module("flashinfer.fused_moe")
        monolithic_moe = fused_moe.trtllm_fp8_block_scale_moe
        fp8_quantization_type = fused_moe.Fp8QuantizationType
        weight_layout = fused_moe.WeightLayout
    except (AttributeError, ImportError) as error:
        raise MonolithicApiUnavailable(
            "FlashInfer monolithic trtllm_fp8_block_scale_moe is unavailable"
        ) from error
    if not callable(monolithic_moe):
        raise MonolithicApiUnavailable(
            "FlashInfer monolithic trtllm_fp8_block_scale_moe is not callable"
        )
    return monolithic_moe, fp8_quantization_type, weight_layout


def assert_monolithic_replay_supported() -> None:
    """Fail before shmoo setup unless the pinned monolithic API is callable."""
    assert_supported_flashinfer()
    _load_monolithic_moe_runtime()


def _validate_prepacked_tensor(
    payload: Mapping[str, object],
    name: str,
    *,
    local_num_experts: int,
    dtypes: tuple[torch.dtype, ...],
    device: torch.device,
) -> torch.Tensor:
    tensor = payload.get(name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"prepacked artifact is missing tensor {name}")
    tensor = cast(Any, tensor)
    if tensor.ndim == 0 or tensor.shape[0] != local_num_experts:
        raise ValueError(
            f"prepacked artifact {name} expert axis mismatch: "
            f"{tuple(tensor.shape)} does not start with {local_num_experts}"
        )
    if tensor.dtype not in dtypes:
        raise ValueError(
            f"prepacked artifact {name} dtype mismatch: {tensor.dtype} not in {dtypes}"
        )
    if tensor.device != device:
        raise ValueError(
            f"prepacked artifact {name} device mismatch: {tensor.device} != {device}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"prepacked artifact {name} must be contiguous")
    return tensor


def load_prepacked_weights(
    path: Path, profile: ReplayProfile, device: torch.device
) -> PrepackedMoeWeights:
    """Load and validate a kernel-ready shuffled Qwen MXFP8 artifact."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("prepacked artifact must be a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("prepacked artifact is missing metadata")

    signature = profile.signature
    required_metadata: dict[str, object] = {
        "format": PREPACKED_ARTIFACT_FORMAT,
        "flashinfer_version": SUPPORTED_FLASHINFER_VERSION,
        "model_revision": signature.model_revision,
        "quantization": "MXFP8",
        "weight_layout": "MajorK",
        "use_shuffled_weight": True,
        "activation": "SwiGLU",
        "gated_rows_reordered": True,
        "matrix_a_shuffled": True,
        "matrix_sf_a_shuffled": True,
        "global_num_experts": signature.global_num_experts,
        "local_num_experts": signature.local_num_experts,
        "hidden_size": signature.hidden_size,
        "intermediate_size": signature.intermediate_size,
    }
    for name, expected in required_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"prepacked artifact metadata {name} mismatch: "
                f"{metadata.get(name)!r} != {expected!r}"
            )
    local_expert_offset = metadata.get("local_expert_offset")
    if (
        isinstance(local_expert_offset, bool)
        or not isinstance(local_expert_offset, int)
        or local_expert_offset < 0
        or local_expert_offset + signature.local_num_experts
        > signature.global_num_experts
    ):
        raise ValueError("prepacked artifact metadata local_expert_offset is invalid")

    experts = signature.local_num_experts
    if signature.hidden_size % 32 or signature.intermediate_size % 32:
        raise ValueError("MXFP8 prepacked dimensions must be divisible by 32")
    scale_dtypes = (torch.uint8, torch.float8_e8m0fnu)
    return PrepackedMoeWeights(
        gemm1_weights=_validate_prepacked_tensor(
            payload,
            "gemm1_weights",
            local_num_experts=experts,
            dtypes=(torch.float8_e4m3fn,),
            device=device,
        ),
        gemm1_weights_scale=_validate_prepacked_tensor(
            payload,
            "gemm1_weights_scale",
            local_num_experts=experts,
            dtypes=scale_dtypes,
            device=device,
        ),
        gemm2_weights=_validate_prepacked_tensor(
            payload,
            "gemm2_weights",
            local_num_experts=experts,
            dtypes=(torch.float8_e4m3fn,),
            device=device,
        ),
        gemm2_weights_scale=_validate_prepacked_tensor(
            payload,
            "gemm2_weights_scale",
            local_num_experts=experts,
            dtypes=scale_dtypes,
            device=device,
        ),
        local_expert_offset=local_expert_offset,
    )


def _prepare_synthetic_weights(
    profile: ReplayProfile, device: torch.device, flashinfer: Any
) -> PrepackedMoeWeights:
    """Prepare explicit smoke weights with the upstream MXFP8 shuffle pipeline."""
    signature = profile.signature
    generator = torch.Generator(device=device)
    generator.manual_seed(int(profile.signature_key[:16], 16))
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

    quant_gemm1: list[torch.Tensor] = []
    scale_gemm1: list[torch.Tensor] = []
    quant_gemm2: list[torch.Tensor] = []
    scale_gemm2: list[torch.Tensor] = []
    for expert in range(signature.local_num_experts):
        weight1, scale1 = flashinfer.mxfp8_quantize(gemm1_bf16[expert], False)
        weight2, scale2 = flashinfer.mxfp8_quantize(gemm2_bf16[expert], False)
        quant_gemm1.append(weight1)
        scale_gemm1.append(scale1.view(torch.uint8))
        quant_gemm2.append(weight2)
        scale_gemm2.append(scale2.view(torch.uint8))

    epilogue_tile_m = 128
    gemm1_rows = 2 * signature.intermediate_size
    shuffled_gemm1: list[torch.Tensor] = []
    shuffled_scale1: list[torch.Tensor] = []
    shuffled_gemm2: list[torch.Tensor] = []
    shuffled_scale2: list[torch.Tensor] = []
    for expert in range(signature.local_num_experts):
        weight1 = quant_gemm1[expert].clone().reshape(gemm1_rows, -1)
        scale1 = scale_gemm1[expert].clone().reshape(gemm1_rows, -1)
        weight1 = flashinfer.reorder_rows_for_gated_act_gemm(weight1)
        scale1 = flashinfer.reorder_rows_for_gated_act_gemm(scale1)
        shuffled_gemm1.append(
            flashinfer.shuffle_matrix_a(weight1.view(torch.uint8), epilogue_tile_m)
            .contiguous()
            .view(quant_gemm1[expert].dtype)
        )
        shuffled_scale1.append(
            flashinfer.shuffle_matrix_sf_a(
                scale1.view(torch.uint8).reshape(gemm1_rows, -1), epilogue_tile_m
            )
            .contiguous()
            .view(scale_gemm1[expert].dtype)
        )
        shuffled_gemm2.append(
            flashinfer.shuffle_matrix_a(
                quant_gemm2[expert].view(torch.uint8), epilogue_tile_m
            )
            .contiguous()
            .view(quant_gemm2[expert].dtype)
        )
        shuffled_scale2.append(
            flashinfer.shuffle_matrix_sf_a(
                scale_gemm2[expert]
                .view(torch.uint8)
                .reshape(signature.hidden_size, -1),
                epilogue_tile_m,
            )
            .contiguous()
            .view(scale_gemm2[expert].dtype)
        )
    return PrepackedMoeWeights(
        gemm1_weights=torch.stack(shuffled_gemm1),
        gemm1_weights_scale=torch.stack(shuffled_scale1),
        gemm2_weights=torch.stack(shuffled_gemm2),
        gemm2_weights_scale=torch.stack(shuffled_scale2),
        local_expert_offset=0,
    )


def build_kernel_case(
    profile: ReplayProfile,
    device: torch.device,
    *,
    weights_path: Path | None,
    synthetic_smoke: bool = False,
) -> MoeKernelCase:
    """Build a replay case from prepacked Qwen weights or explicit smoke weights."""
    assert_supported_flashinfer()
    if device.type != "cuda":
        raise ValueError("MXFP8 MoE kernel cases require a CUDA device")
    if synthetic_smoke == (weights_path is not None):
        raise ValueError("provide exactly one of weights_path or synthetic_smoke")
    flashinfer, _, _, _ = _load_moe_runtime()
    if synthetic_smoke:
        prepared = _prepare_synthetic_weights(profile, device, flashinfer)
    else:
        assert weights_path is not None
        prepared = load_prepacked_weights(weights_path, profile, device)
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
    hidden_states, hidden_states_scale = flashinfer.mxfp8_quantize(hidden_bf16, False)
    hidden_states_scale = hidden_states_scale.view(torch.uint8).reshape(
        signature.num_tokens, -1
    )
    return MoeKernelCase(
        profile=profile,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=prepared.gemm1_weights,
        gemm1_weights_scale=prepared.gemm1_weights_scale,
        gemm2_weights=prepared.gemm2_weights,
        gemm2_weights_scale=prepared.gemm2_weights_scale,
        output=torch.empty(
            signature.num_tokens,
            signature.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        activation_type=flashinfer.ActivationType.Swiglu.value,
        routing_method_type=flashinfer.RoutingMethodType.RenormalizeNaive.value,
        local_expert_offset=prepared.local_expert_offset,
        weight_layout="MajorK",
        use_shuffled_weight=True,
        prepacked_weight_format=PREPACKED_ARTIFACT_FORMAT,
    )


def run_moe_pair(
    case: MoeKernelCase,
    packed_topk: torch.Tensor,
    *,
    do_finalize: bool,
    gemm1_lora_delta: torch.Tensor | None,
) -> MoePairResult:
    """Run the routed full MoE pair through the pinned FlashInfer wrapper."""
    assert_supported_flashinfer()
    _, routed_moe, fp8_enum, weight_layout = _load_moe_runtime()
    if (
        case.prepacked_weight_format != PREPACKED_ARTIFACT_FORMAT
        or case.weight_layout != "MajorK"
        or not case.use_shuffled_weight
    ):
        raise ValueError("MoE case does not contain validated prepacked MajorK weights")
    signature = case.profile.signature
    try:
        raw_result = routed_moe(
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
            use_shuffled_weight=case.use_shuffled_weight,
            weight_layout=weight_layout.MajorK.value,
            do_finalize=do_finalize,
            output=case.output,
            tune_max_num_tokens=tune_max_num_tokens(case),
            fp8_quantization_type=fp8_enum.MxFp8,
            activation_type=case.activation_type,
            gemm1_lora_delta=gemm1_lora_delta,
        )
    except (TypeError, NotImplementedError) as error:
        if not do_finalize and gemm1_lora_delta is not None:
            raise IntermediateApiUnavailable(
                "FlashInfer intermediate invocation is unavailable"
            ) from error
        raise

    if do_finalize:
        if isinstance(raw_result, torch.Tensor):
            final_output = raw_result
        elif (
            isinstance(raw_result, (tuple, list))
            and raw_result
            and isinstance(raw_result[0], torch.Tensor)
        ):
            final_output = raw_result[0]
        else:
            raise RuntimeError("unexpected FlashInfer final-output return contract")
        return MoePairResult(
            final_output=final_output,
            activated_intermediate=None,
        )

    expected_elements = (
        signature.num_tokens * signature.top_k * signature.intermediate_size
    )
    if (
        gemm1_lora_delta is None
        or not isinstance(raw_result, (tuple, list))
        or len(raw_result) != 4
        or not isinstance(raw_result[3], torch.Tensor)
        or raw_result[3].dtype != torch.bfloat16
        or raw_result[3].numel() != expected_elements
    ):
        raise IntermediateApiUnavailable(
            "unexpected FlashInfer activated-intermediate return contract"
        )
    return MoePairResult(
        final_output=None,
        activated_intermediate=raw_result[3].reshape(
            signature.num_tokens, signature.top_k, signature.intermediate_size
        ),
    )


def run_monolithic_moe_pair(
    case: MoeKernelCase, router_logits: torch.Tensor
) -> MoePairResult:
    """Run the production-shaped monolithic FlashInfer FC1+FC2 invocation."""
    assert_supported_flashinfer()
    monolithic_moe, fp8_enum, weight_layout = _load_monolithic_moe_runtime()
    if (
        case.prepacked_weight_format != PREPACKED_ARTIFACT_FORMAT
        or case.weight_layout != "MajorK"
        or not case.use_shuffled_weight
    ):
        raise ValueError("MoE case does not contain validated prepacked MajorK weights")
    signature = case.profile.signature
    expected_shape = (signature.num_tokens, signature.global_num_experts)
    if tuple(router_logits.shape) != expected_shape:
        raise ValueError(
            f"router_logits shape mismatch: {tuple(router_logits.shape)} "
            f"!= {expected_shape}"
        )
    if router_logits.device != case.hidden_states.device:
        raise ValueError("router_logits must be on the hidden-states device")
    if not router_logits.is_floating_point() or not router_logits.is_contiguous():
        raise ValueError("router_logits must be a contiguous floating-point tensor")

    raw_result = monolithic_moe(
        routing_logits=router_logits,
        routing_bias=None,
        hidden_states=case.hidden_states,
        hidden_states_scale=case.hidden_states_scale,
        gemm1_weights=case.gemm1_weights,
        gemm1_weights_scale=case.gemm1_weights_scale,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
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
        use_shuffled_weight=case.use_shuffled_weight,
        weight_layout=weight_layout.MajorK,
        fp8_quantization_type=fp8_enum.MxFp8,
        activation_type=case.activation_type,
    )
    if isinstance(raw_result, torch.Tensor):
        final_output = raw_result
    elif (
        isinstance(raw_result, (tuple, list))
        and raw_result
        and isinstance(raw_result[0], torch.Tensor)
    ):
        final_output = raw_result[0]
    else:
        raise RuntimeError("unexpected FlashInfer final-output return contract")
    return MoePairResult(final_output=final_output, activated_intermediate=None)
