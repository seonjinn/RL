"""Check FlashInfer MXFP8 activation quantization at failing prompt lengths."""

from __future__ import annotations

import torch
from flashinfer import mxfp8_dequantize_host, mxfp8_quantize


def _check_mxfp8_activation_quantization_prompt_shape(
    num_tokens: int,
    backend: str,
) -> None:
    """Odd token counts must not produce invalid values or scales."""
    torch.manual_seed(42)
    hidden_states = (
        torch.randn(
            (num_tokens, 2048),
            dtype=torch.float32,
            device="cuda",
        )
        * 16
    ).to(torch.bfloat16)

    quantized, scales = mxfp8_quantize(
        hidden_states,
        is_sf_swizzled_layout=False,
        alignment=32,
        backend=backend,
    )
    torch.cuda.synchronize()

    assert quantized.shape == hidden_states.shape
    assert scales.numel() == hidden_states.numel() // 32
    assert not torch.isnan(quantized.float()).any()
    assert not torch.isinf(quantized.float()).any()
    assert not (scales == 255).any(), "E8M0 byte 255 represents a non-finite scale"

    dequantized = mxfp8_dequantize_host(
        quantized.cpu().view(torch.uint8),
        scales.cpu().view(torch.uint8).reshape(-1),
        is_sf_swizzled_layout=False,
    )
    error = (dequantized.float() - hidden_states.cpu().float()).abs()
    assert not torch.isnan(error).any()
    assert not torch.isinf(error).any()
    assert (error > 8).float().mean().item() <= 0.001
    print(
        f"backend={backend} tokens={num_tokens} "
        f"quantized_shape={tuple(quantized.shape)} scale_shape={tuple(scales.shape)}"
    )


def test_mxfp8_activation_quantization_prompt_shapes() -> None:
    for backend in ("cuda", "cute-dsl"):
        for num_tokens in (49, 50, 51, 81, 82, 83):
            _check_mxfp8_activation_quantization_prompt_shape(num_tokens, backend)


if __name__ == "__main__":
    test_mxfp8_activation_quantization_prompt_shapes()
    print("FlashInfer MXFP8 prompt-shape checks passed")
