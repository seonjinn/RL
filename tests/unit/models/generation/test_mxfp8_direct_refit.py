"""Exact-value and storage contracts for direct CuTeDSL refit finalization."""

import pytest
import torch

from nemo_rl.utils import mxfp8_direct_refit as direct


def reference_swizzle(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    padded = torch.zeros(
        ((n + 127) // 128 * 128, (k + 127) // 128 * 4),
        dtype=scale.dtype,
        device=scale.device,
    )
    padded[:n, : k // 32] = scale[:n, : k // 32]
    return (
        padded.view((n + 127) // 128, 4, 32, (k + 127) // 128, 4)
        .transpose(1, 3)
        .contiguous()
        .view(-1)
    )


@pytest.mark.parametrize("n,k", [(1, 32), (130, 160), (256, 512), (384, 768)])
def test_direct_refit_preserves_exact_values_and_storage(
    monkeypatch, n: int, k: int
) -> None:
    def cpu_swizzle(src, dst, *, n, k):
        dst.copy_(reference_swizzle(src, n, k))

    monkeypatch.setattr(direct, "swizzle_into", cpu_swizzle)
    value = torch.randn(n, k).to(torch.float8_e4m3fn)
    scale = torch.randint(1, 255, (n, k // 32), dtype=torch.uint8)
    runtime_weight = torch.empty_like(value).t()
    runtime_scale = torch.full_like(reference_swizzle(scale, n, k), 255)
    pointers = (runtime_weight.data_ptr(), runtime_scale.data_ptr())
    for _ in range(2):
        direct.copy_into_runtime(value, scale, runtime_weight, runtime_scale)
        assert torch.equal(runtime_weight.float(), value.t().float())
        assert torch.equal(runtime_scale, reference_swizzle(scale, n, k))
        assert pointers == (runtime_weight.data_ptr(), runtime_scale.data_ptr())
        scale.add_(1)


@pytest.mark.parametrize(
    "bad", ["shape", "stride", "dtype", "scale_size", "scale_dtype"]
)
def test_invalid_runtime_rejected_before_mutation(monkeypatch, bad: str) -> None:
    value = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
    scale = torch.ones(128, 4, dtype=torch.uint8)
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn).t()
    target = torch.full((512,), 37, dtype=torch.uint8)
    if bad == "shape":
        weight = weight[:64]
    if bad == "stride":
        weight = weight.contiguous()
    if bad == "dtype":
        weight = weight.float()
    if bad == "scale_size":
        target = target[:128]
    if bad == "scale_dtype":
        target = target.float()
    old_weight, old_target = weight.float().clone(), target.clone()
    with pytest.raises(ValueError):
        direct.copy_into_runtime(value, scale, weight, target)
    assert torch.equal(weight.float(), old_weight)
    assert torch.equal(target, old_target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "n,k",
    [
        (1, 32),
        (127, 96),
        (128, 128),
        (130, 160),
        (256, 512),
        (384, 768),
        (2048, 2048),
        (6144, 2048),
    ],
)
def test_cuda_swizzle_into_matches_reference(n: int, k: int) -> None:
    # Noncontiguous checkpoint scales exercise loader-provided strides.
    source = torch.randint(0, 256, (n, k // 32 * 2), device="cuda", dtype=torch.uint8)[
        :, ::2
    ]
    target = torch.empty_like(reference_swizzle(source, n, k))
    direct.swizzle_into(source, target, n=n, k=k)
    assert torch.equal(target, reference_swizzle(source, n, k))
