from dataclasses import dataclass
from typing import Any

import pytest
import torch

from nemo_rl.models.policy.workers.mxfp8_refit_source import (
    extract_native_mxfp8_components,
)


@dataclass
class FakeMXFP8Tensor:
    shape: tuple[int, ...]
    rowwise_data: torch.Tensor
    rowwise_scale_inv: torch.Tensor
    with_gemm_swizzled_scales: bool
    metadata: dict[str, Any] | None = None

    def get_metadata(self) -> dict[str, Any]:
        return self.metadata or {
            "rowwise_data": self.rowwise_data,
            "rowwise_scale_inv": self.rowwise_scale_inv,
            "with_gemm_swizzled_scales": self.with_gemm_swizzled_scales,
        }


def test_extract_native_mxfp8_components_crops_padding() -> None:
    source = FakeMXFP8Tensor(
        shape=(64, 256),
        rowwise_data=torch.arange(64 * 256, dtype=torch.uint8).reshape(64, 256),
        rowwise_scale_inv=torch.arange(128 * 8, dtype=torch.uint8).reshape(128, 8),
        with_gemm_swizzled_scales=False,
    )

    components = extract_native_mxfp8_components(source)

    assert components.weight.shape == (64, 256)
    assert components.weight.dtype == torch.float8_e4m3fn
    assert components.weight_scale.shape == (64, 8)
    assert components.weight_scale.dtype == torch.uint8


def test_extract_native_mxfp8_components_rejects_missing_data() -> None:
    source = FakeMXFP8Tensor(
        shape=(64, 256),
        rowwise_data=torch.empty(64, 256, dtype=torch.uint8),
        rowwise_scale_inv=torch.empty(128, 8, dtype=torch.uint8),
        with_gemm_swizzled_scales=False,
        metadata={
            "rowwise_scale_inv": torch.empty(128, 8, dtype=torch.uint8),
            "with_gemm_swizzled_scales": False,
        },
    )

    with pytest.raises(KeyError, match="rowwise_data"):
        extract_native_mxfp8_components(source)


@pytest.mark.parametrize(
    ("data_dtype", "scale_dtype", "invalid_key"),
    [
        (torch.float16, torch.uint8, "rowwise_data"),
        (torch.uint8, torch.float16, "rowwise_scale_inv"),
    ],
)
def test_extract_native_mxfp8_components_rejects_invalid_dtype(
    data_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    invalid_key: str,
) -> None:
    source = FakeMXFP8Tensor(
        shape=(64, 256),
        rowwise_data=torch.empty(64, 256, dtype=data_dtype),
        rowwise_scale_inv=torch.empty(128, 8, dtype=scale_dtype),
        with_gemm_swizzled_scales=False,
    )

    with pytest.raises(ValueError, match=invalid_key):
        extract_native_mxfp8_components(source)


def test_extract_native_mxfp8_components_rejects_unaligned_k() -> None:
    source = FakeMXFP8Tensor(
        shape=(64, 255),
        rowwise_data=torch.empty(64, 255, dtype=torch.uint8),
        rowwise_scale_inv=torch.empty(128, 8, dtype=torch.uint8),
        with_gemm_swizzled_scales=False,
    )

    with pytest.raises(
        ValueError,
        match=r"Native MXFP8 refit requires K divisible by 32; got \(64, 255\)",
    ):
        extract_native_mxfp8_components(source)


def test_extract_native_mxfp8_components_rejects_swizzled_scales() -> None:
    source = FakeMXFP8Tensor(
        shape=(64, 256),
        rowwise_data=torch.empty(64, 256, dtype=torch.uint8),
        rowwise_scale_inv=torch.empty(128, 8, dtype=torch.uint8),
        with_gemm_swizzled_scales=True,
    )

    with pytest.raises(
        ValueError,
        match="Native MXFP8 refit requires compact rowwise scales",
    ):
        extract_native_mxfp8_components(source)
