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
        shape=(3, 64),
        rowwise_data=torch.arange(3 * 64, dtype=torch.uint8).reshape(3, 64),
        rowwise_scale_inv=torch.arange(5 * 4, dtype=torch.uint8).reshape(5, 4),
        with_gemm_swizzled_scales=False,
    )

    components = extract_native_mxfp8_components(source)

    assert components.weight.shape == (3, 64)
    assert components.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(components.weight.view(torch.uint8), source.rowwise_data)
    assert components.weight_scale.shape == (3, 2)
    assert components.weight_scale.dtype == torch.uint8
    assert torch.equal(components.weight_scale, source.rowwise_scale_inv[:3, :2])


def test_extract_native_mxfp8_components_preserves_grouped_expert_values() -> None:
    shape = (2, 3, 64)
    source = FakeMXFP8Tensor(
        shape=shape,
        rowwise_data=torch.arange(2 * 3 * 64, dtype=torch.uint8).reshape(shape),
        rowwise_scale_inv=torch.arange(8 * 4, dtype=torch.uint8).reshape(8, 4),
        with_gemm_swizzled_scales=False,
    )

    components = extract_native_mxfp8_components(source)

    assert components.weight.shape == shape
    assert torch.equal(components.weight.view(torch.uint8), source.rowwise_data)
    assert components.weight_scale.shape == (2, 3, 2)
    assert torch.equal(
        components.weight_scale,
        source.rowwise_scale_inv[:6, :2].reshape(2, 3, 2),
    )


@pytest.mark.parametrize("missing_key", ["rowwise_data", "rowwise_scale_inv"])
def test_extract_native_mxfp8_components_rejects_missing_component(
    missing_key: str,
) -> None:
    metadata: dict[str, Any] = {
        "rowwise_data": torch.empty(3, 64, dtype=torch.uint8),
        "rowwise_scale_inv": torch.empty(5, 4, dtype=torch.uint8),
        "with_gemm_swizzled_scales": False,
    }
    del metadata[missing_key]
    source = FakeMXFP8Tensor(
        shape=(3, 64),
        rowwise_data=torch.empty(3, 64, dtype=torch.uint8),
        rowwise_scale_inv=torch.empty(5, 4, dtype=torch.uint8),
        with_gemm_swizzled_scales=False,
        metadata=metadata,
    )

    with pytest.raises(ValueError, match=missing_key):
        extract_native_mxfp8_components(source)


@pytest.mark.parametrize(
    ("invalid_key", "invalid_value"),
    [
        ("rowwise_data", None),
        ("rowwise_scale_inv", "not a tensor"),
    ],
)
def test_extract_native_mxfp8_components_rejects_non_tensor_component(
    invalid_key: str,
    invalid_value: Any,
) -> None:
    source = FakeMXFP8Tensor(
        shape=(3, 64),
        rowwise_data=torch.empty(3, 64, dtype=torch.uint8),
        rowwise_scale_inv=torch.empty(5, 4, dtype=torch.uint8),
        with_gemm_swizzled_scales=False,
    )
    source.metadata = {
        "rowwise_data": source.rowwise_data,
        "rowwise_scale_inv": source.rowwise_scale_inv,
        "with_gemm_swizzled_scales": False,
        invalid_key: invalid_value,
    }

    with pytest.raises(ValueError, match=rf"{invalid_key}.*torch.Tensor"):
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
        shape=(3, 64),
        rowwise_data=torch.empty(3, 64, dtype=data_dtype),
        rowwise_scale_inv=torch.empty(5, 4, dtype=scale_dtype),
        with_gemm_swizzled_scales=False,
    )

    with pytest.raises(ValueError, match=invalid_key):
        extract_native_mxfp8_components(source)


def test_extract_native_mxfp8_components_rejects_undersized_data() -> None:
    source = FakeMXFP8Tensor(
        shape=(3, 64),
        rowwise_data=torch.empty(3 * 64 - 1, dtype=torch.uint8),
        rowwise_scale_inv=torch.empty(5, 4, dtype=torch.uint8),
        with_gemm_swizzled_scales=False,
    )

    with pytest.raises(ValueError, match=r"rowwise_data.*192 bytes"):
        extract_native_mxfp8_components(source)


@pytest.mark.parametrize(
    "scale_shape",
    [(2, 2), (3, 1), (3,)],
)
def test_extract_native_mxfp8_components_rejects_undersized_scale(
    scale_shape: tuple[int, ...],
) -> None:
    source = FakeMXFP8Tensor(
        shape=(3, 64),
        rowwise_data=torch.empty(3, 64, dtype=torch.uint8),
        rowwise_scale_inv=torch.empty(scale_shape, dtype=torch.uint8),
        with_gemm_swizzled_scales=False,
    )

    with pytest.raises(ValueError, match=r"rowwise_scale_inv.*logical shape \(3, 2\)"):
        extract_native_mxfp8_components(source)


def test_extract_native_mxfp8_components_rejects_unaligned_k() -> None:
    source = FakeMXFP8Tensor(
        shape=(3, 63),
        rowwise_data=torch.empty(3, 63, dtype=torch.uint8),
        rowwise_scale_inv=torch.empty(5, 4, dtype=torch.uint8),
        with_gemm_swizzled_scales=False,
    )

    with pytest.raises(
        ValueError,
        match=r"Native MXFP8 refit requires K divisible by 32; got \(3, 63\)",
    ):
        extract_native_mxfp8_components(source)


def test_extract_native_mxfp8_components_rejects_swizzled_scales() -> None:
    source = FakeMXFP8Tensor(
        shape=(3, 64),
        rowwise_data=torch.empty(3, 64, dtype=torch.uint8),
        rowwise_scale_inv=torch.empty(5, 4, dtype=torch.uint8),
        with_gemm_swizzled_scales=True,
    )

    with pytest.raises(
        ValueError,
        match="Native MXFP8 refit requires compact rowwise scales",
    ):
        extract_native_mxfp8_components(source)
