"""Check FlashInfer MXFP8 activation quantization at failing prompt lengths."""

from __future__ import annotations

import pytest
import torch
from flashinfer import mxfp8_dequantize_host, mxfp8_quantize


@pytest.mark.parametrize("num_tokens", [49, 50, 51, 81, 82, 83])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_mxfp8_activation_quantization_prompt_shapes(
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
    assert scales.shape == (num_tokens, hidden_states.shape[1] // 32)
    assert not torch.isnan(quantized.float()).any()
    assert not torch.isinf(quantized.float()).any()
    assert not (scales == 255).any(), "E8M0 byte 255 represents a non-finite scale"

    dequantized = mxfp8_dequantize_host(
        quantized.cpu().view(torch.uint8),
        scales.cpu().view(torch.uint8),
        is_sf_swizzled_layout=False,
    )
    error = (dequantized.float() - hidden_states.cpu().float()).abs()
    assert not torch.isnan(error).any()
    assert not torch.isinf(error).any()
    assert (error > 8).float().mean().item() <= 0.001
