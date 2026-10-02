"""Exercise FlashInfer's MXFP8 TRTLLM MoE at Qwen3 decode shapes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import flashinfer
import torch


def _load_flashinfer_test_helpers():
    source_root = Path(os.environ["FLASHINFER_SOURCE_ROOT"])
    if not (source_root / "tests/moe/trtllm_gen_fused_moe_utils.py").is_file():
        raise FileNotFoundError(
            "FLASHINFER_SOURCE_ROOT must point to the FlashInfer v0.6.18 source tree"
        )

    # Import the installed package first, then expose only the matching upstream
    # test helpers. This keeps the kernel under test tied to the nightly image.
    sys.path.insert(0, str(source_root))
    from tests.moe import trtllm_gen_fused_moe_utils as helpers

    return helpers


def test_qwen3_mxfp8_trtllm_decode_shapes() -> None:
    helpers = _load_flashinfer_test_helpers()
    original_check_accuracy = helpers.check_accuracy
    observed_shapes: list[tuple[int, ...]] = []

    def check_finite_accuracy(reference, actual, *args, **kwargs) -> None:
        assert torch.isfinite(reference).all(), "BF16 reference contains non-finite values"
        assert torch.isfinite(actual).all(), "TRTLLM MXFP8 output contains non-finite values"
        observed_shapes.append(tuple(actual.shape))
        original_check_accuracy(reference, actual, *args, **kwargs)

    # The upstream matrix skips hidden sizes above 1024 only to control CI time.
    # Qwen3-30B-A3B uses H=2048, so this focused GB200 test deliberately runs it.
    helpers.skip_checks = lambda *args, **kwargs: None
    helpers.check_accuracy = check_finite_accuracy

    routing_config = {
        "num_experts": 128,
        "top_k": 8,
        "padding": 8,
        "n_groups": None,
        "top_k_groups": None,
        "routed_scaling": None,
        "has_routing_bias": False,
        "routing_method_type": helpers.RoutingMethodType.Renormalize,
        "compatible_moe_impls": [helpers.FP8BlockScaleMoe],
        "compatible_intermediate_size": [768],
        "enable_autotune": False,
    }
    weight_processing = {
        "use_shuffled_weight": True,
        "layout": helpers.WeightLayout.MajorK,
        "compatible_moe_impls": [helpers.FP8BlockScaleMoe],
    }
    token_counts = (1, 8, 16, 31, 32, 33, 64, 96, 128)

    print(
        f"flashinfer={flashinfer.__version__} "
        "model=Qwen3-30B-A3B experts=128 hidden=2048 intermediate=768 top_k=8"
    )
    for num_tokens in token_counts:
        helpers.run_moe_test(
            num_tokens=num_tokens,
            hidden_size=2048,
            intermediate_size=768,
            moe_impl=helpers.FP8BlockScaleMoe(
                fp8_quantization_type=helpers.QuantMode.FP8_BLOCK_SCALE_MXFP8
            ),
            routing_config=routing_config,
            weight_processing=weight_processing,
            activation_type=helpers.ActivationType.Swiglu,
            cache_permute_indices={},
        )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"tokens={num_tokens} finite_and_accurate=true")

    assert observed_shapes == [(num_tokens, 2048) for num_tokens in token_counts]


if __name__ == "__main__":
    test_qwen3_mxfp8_trtllm_decode_shapes()
    print("FlashInfer TRTLLM MXFP8 Qwen3 shape checks passed")
