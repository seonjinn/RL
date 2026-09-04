# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import FrozenInstanceError, fields, replace
from pickle import dumps, loads

import pytest

import nemo_rl.precision_policy as precision_policy
import nemo_rl.precision_policy.source_storage as source_storage_module
from nemo_rl.precision_policy.source_dtype import CanonicalSourceDType
from nemo_rl.precision_policy.source_storage import (
    IDENTITY_PERMUTATION_ID,
    IDENTITY_SWIZZLE_ID,
    SourceDerivedRealization,
    SourceExtentRounding,
    SourceLiteralAxisExtent,
    SourceNormalizationContract,
    SourceNormalizationKind,
    SourceNormalizerManifest,
    SourceNormalizedAxisExtent,
    SourcePaddingSemantics,
    SourcePhysicalAxisSpec,
    SourceStorageComponent,
    SourceStorageRealization,
    SourceStorageRealizationInventory,
    source_normalizer_manifest_digest,
    source_realization_is_wire_eligible,
    source_realizations_have_exact_physical_representation,
    source_storage_inventory_digest,
    validate_source_storage_realization_inventory,
)


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _contract(
    capability_id: str = "test.identity.v1",
    kind: SourceNormalizationKind = SourceNormalizationKind.IDENTITY,
    character: str = "1",
) -> SourceNormalizationContract:
    return SourceNormalizationContract(
        capability_id=capability_id,
        kind=kind,
        contract_digest=_digest(character),
    )


def _identity_axes(shape: tuple[int, ...]) -> tuple[SourcePhysicalAxisSpec, ...]:
    return tuple(
        SourcePhysicalAxisSpec(
            axis_name=f"axis_{axis_index}",
            extent=SourceNormalizedAxisExtent(
                normalized_axis_indices=(axis_index,),
                divisor=1,
                rounding=SourceExtentRounding.EXACT,
                alignment=1,
            ),
        )
        for axis_index in range(len(shape))
    )


def _component(
    *,
    component_id: str = "main.weight.values",
    native_name: str = "model.weight",
    component_role: str = "logical_values",
    dtype: CanonicalSourceDType = CanonicalSourceDType.BFLOAT16,
    normalized_shape: tuple[int, ...] = (128, 928, 2688),
    physical_shape: tuple[int, ...] | None = None,
    physical_axes: tuple[SourcePhysicalAxisSpec, ...] | None = None,
    storage_encoding: str = "plain_bfloat16",
    padding_semantics: SourcePaddingSemantics = SourcePaddingSemantics.NO_PADDING,
    padding_fill_encoding: str | None = None,
    permutation_id: str = IDENTITY_PERMUTATION_ID,
    swizzle_id: str = IDENTITY_SWIZZLE_ID,
) -> SourceStorageComponent:
    return SourceStorageComponent(
        graph_instance_id="main",
        native_component_id=component_id,
        source_native_name=native_name,
        component_role=component_role,
        carrier_dtype=dtype,
        physical_shape=(normalized_shape if physical_shape is None else physical_shape),
        physical_axes=(
            _identity_axes(normalized_shape) if physical_axes is None else physical_axes
        ),
        storage_encoding=storage_encoding,
        padding_semantics=padding_semantics,
        padding_fill_encoding=padding_fill_encoding,
        permutation_id=permutation_id,
        swizzle_id=swizzle_id,
    )


def _realization(
    *,
    realization_id: str = "main.weight.identity",
    output_record_id: str = "main.weight",
    output_dtype: CanonicalSourceDType = CanonicalSourceDType.BFLOAT16,
    output_shape: tuple[int, ...] = (128, 928, 2688),
    output_numeric_encoding: str = "plain_bfloat16",
    components: tuple[SourceStorageComponent, ...] | None = None,
    normalization: SourceNormalizationContract | None = None,
) -> SourceStorageRealization:
    return SourceStorageRealization(
        realization_id=realization_id,
        graph_instance_id="main",
        output_record_id=output_record_id,
        components=components or (_component(normalized_shape=output_shape),),
        output_dtype=output_dtype,
        output_shape=output_shape,
        output_numeric_encoding=output_numeric_encoding,
        normalization=normalization or _contract(),
    )


def test_identity_bf16_realization_preserves_logical_eih_view() -> None:
    realization = _realization()
    manifest = SourceNormalizerManifest(schema_version=1, contracts=(_contract(),))
    inventory = SourceStorageRealizationInventory(
        graph_instance_id="main",
        normalizer_manifest=manifest,
        realizations=(realization,),
    )

    assert realization.output_shape == (128, 928, 2688)
    assert realization.components[0].physical_shape == (128, 928, 2688)
    assert source_realization_is_wire_eligible(realization)
    assert source_normalizer_manifest_digest(manifest).startswith("sha256:")
    assert source_storage_inventory_digest(inventory).startswith("sha256:")
    assert loads(dumps(inventory)) == inventory
    with pytest.raises(FrozenInstanceError):
        realization.output_shape = (1,)  # type: ignore[misc]


def test_identity_normalization_does_not_depend_on_physical_axis_labels() -> None:
    shape = (128, 928, 2688)
    axes = tuple(
        replace(axis, axis_name=axis_name)
        for axis, axis_name in zip(
            _identity_axes(shape),
            ("experts", "intermediate", "hidden"),
            strict=True,
        )
    )

    realization = _realization(
        components=(
            _component(
                normalized_shape=shape,
                physical_axes=axes,
            ),
        ),
    )

    assert tuple(
        axis.axis_name for axis in realization.components[0].physical_axes
    ) == ("experts", "intermediate", "hidden")


def test_inventory_digest_streams_realizations_without_an_outer_payload_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _realization()
    realizations = tuple(
        replace(
            base,
            realization_id=f"main.weight.{index}",
            output_record_id=f"main.weight.{index}",
            components=(
                replace(
                    base.components[0],
                    native_component_id=f"main.weight.{index}.values",
                    source_native_name=f"model.weight.{index}",
                ),
            ),
        )
        for index in range(1_000)
    )
    inventory = SourceStorageRealizationInventory(
        graph_instance_id="main",
        normalizer_manifest=SourceNormalizerManifest(
            schema_version=1,
            contracts=(_contract(),),
        ),
        realizations=realizations,
    )
    original = source_storage_module._canonical_json_bytes
    realization_payload_count = 0

    def reject_outer_inventory_payload(payload: object) -> bytes:
        nonlocal realization_payload_count
        if isinstance(payload, dict):
            assert payload.get("type") != "source_storage_realization_inventory"
            if payload.get("type") == "source_storage_realization":
                realization_payload_count += 1
        return original(payload)

    monkeypatch.setattr(
        source_storage_module,
        "_canonical_json_bytes",
        reject_outer_inventory_payload,
    )

    assert source_storage_inventory_digest(inventory).startswith("sha256:")
    assert realization_payload_count == len(realizations)


def test_bf16_logical_eih_and_trtllm_packed_storage_are_not_direct_copy_equal() -> None:
    logical = _realization()
    repack = _contract(
        "trtllm.block-major-repack.v1",
        SourceNormalizationKind.REPACK,
        "2",
    )
    packed_axes = (
        SourcePhysicalAxisSpec(
            "experts",
            SourceNormalizedAxisExtent(
                normalized_axis_indices=(0,),
                divisor=1,
                rounding=SourceExtentRounding.EXACT,
                alignment=1,
            ),
        ),
        SourcePhysicalAxisSpec(
            "hidden_blocks",
            SourceNormalizedAxisExtent(
                normalized_axis_indices=(2,),
                divisor=64,
                rounding=SourceExtentRounding.EXACT,
                alignment=1,
            ),
        ),
        SourcePhysicalAxisSpec(
            "intermediate_padded",
            SourceNormalizedAxisExtent(
                normalized_axis_indices=(1,),
                divisor=1,
                rounding=SourceExtentRounding.EXACT,
                alignment=128,
            ),
        ),
        SourcePhysicalAxisSpec("block", SourceLiteralAxisExtent(64)),
    )
    packed = _realization(
        realization_id="main.weight.trtllm",
        components=(
            _component(
                component_id="main.weight.trtllm",
                physical_shape=(128, 42, 1024, 64),
                physical_axes=packed_axes,
                storage_encoding="trtllm.block-major-bf16.v1",
                padding_semantics=SourcePaddingSemantics.ZERO_FILLED,
                padding_fill_encoding="raw-zero",
                permutation_id="trtllm.w13-to-w31.v1",
            ),
        ),
        normalization=repack,
    )

    assert not source_realizations_have_exact_physical_representation(logical, packed)


def test_exact_physical_comparison_ignores_only_endpoint_native_identity() -> None:
    first = _realization()
    second = replace(
        first,
        realization_id="main.weight.other-endpoint",
        graph_instance_id="destination",
        output_record_id="destination.weight",
        components=(
            replace(
                first.components[0],
                graph_instance_id="destination",
                native_component_id="destination.weight.values",
                source_native_name="runtime.weight",
            ),
        ),
    )

    assert source_realizations_have_exact_physical_representation(first, second)


def test_exact_physical_comparison_binds_every_layout_and_normalization_fact() -> None:
    output_shape = (3, 3)
    physical_axes = tuple(
        SourcePhysicalAxisSpec(
            axis_name=axis_name,
            extent=SourceNormalizedAxisExtent(
                normalized_axis_indices=(axis_index,),
                divisor=1,
                rounding=SourceExtentRounding.EXACT,
                alignment=4,
            ),
        )
        for axis_index, axis_name in enumerate(("rows", "columns"))
    )
    normalization = _contract(
        "test.composite-normalize.v1",
        SourceNormalizationKind.COMPOSITE,
        "2",
    )
    values = _component(
        component_id="main.weight.values",
        native_name="model.weight.values",
        normalized_shape=output_shape,
        physical_shape=(4, 4),
        physical_axes=physical_axes,
        padding_semantics=SourcePaddingSemantics.ZERO_FILLED,
        padding_fill_encoding="raw-zero",
    )
    scales = _component(
        component_id="main.weight.scales",
        native_name="model.weight.scales",
        component_role="block_scales",
        dtype=CanonicalSourceDType.UINT8,
        normalized_shape=output_shape,
        physical_shape=(4, 4),
        physical_axes=physical_axes,
        storage_encoding="uint8-carried-e8m0",
        padding_semantics=SourcePaddingSemantics.ZERO_FILLED,
        padding_fill_encoding="raw-zero",
    )
    baseline = _realization(
        output_shape=output_shape,
        components=(values, scales),
        normalization=normalization,
    )
    ceil_axes = (
        replace(
            physical_axes[0],
            extent=replace(
                physical_axes[0].extent,
                rounding=SourceExtentRounding.CEIL,
            ),
        ),
        physical_axes[1],
    )
    wider_axes = (
        replace(
            physical_axes[0],
            extent=replace(physical_axes[0].extent, alignment=8),
        ),
        physical_axes[1],
    )
    mutations = {
        "output dtype": replace(
            baseline,
            output_dtype=CanonicalSourceDType.FLOAT32,
        ),
        "output shape": replace(baseline, output_shape=(3, 4)),
        "output encoding": replace(
            baseline,
            output_numeric_encoding="other_bfloat16",
        ),
        "normalizer contract": replace(
            baseline,
            normalization=_contract(
                "test.other-normalize.v1",
                SourceNormalizationKind.COMPOSITE,
                "3",
            ),
        ),
        "component order": replace(
            baseline,
            components=tuple(reversed(baseline.components)),
        ),
        "component role": replace(
            baseline,
            components=(replace(values, component_role="other_values"), scales),
        ),
        "carrier dtype": replace(
            baseline,
            components=(
                replace(values, carrier_dtype=CanonicalSourceDType.FLOAT32),
                scales,
            ),
        ),
        "physical shape": replace(
            baseline,
            components=(
                replace(values, physical_shape=(8, 4), physical_axes=wider_axes),
                scales,
            ),
        ),
        "physical formula": replace(
            baseline,
            components=(replace(values, physical_axes=ceil_axes), scales),
        ),
        "physical axis order": replace(
            baseline,
            components=(
                replace(values, physical_axes=tuple(reversed(physical_axes))),
                scales,
            ),
        ),
        "storage encoding": replace(
            baseline,
            components=(replace(values, storage_encoding="other_bfloat16"), scales),
        ),
        "padding semantics": replace(
            baseline,
            components=(
                replace(
                    values,
                    padding_semantics=SourcePaddingSemantics.UNSPECIFIED_IGNORED,
                    padding_fill_encoding=None,
                ),
                scales,
            ),
        ),
        "padding fill": replace(
            baseline,
            components=(
                replace(values, padding_fill_encoding="semantic-zero"),
                scales,
            ),
        ),
        "permutation": replace(
            baseline,
            components=(
                replace(values, permutation_id="test.transpose.v1"),
                scales,
            ),
        ),
        "swizzle": replace(
            baseline,
            components=(replace(values, swizzle_id="test.tile-swizzle.v1"), scales),
        ),
    }

    for fact_name, candidate in mutations.items():
        assert not source_realizations_have_exact_physical_representation(
            baseline,
            candidate,
        ), fact_name


def test_te_rowwise_compact_and_gemm_swizzled_scale_realizations_are_explicit() -> None:
    normalized_shape = (3, 32, 29)
    physical_axes = (
        SourcePhysicalAxisSpec(
            "flattened_m",
            SourceNormalizedAxisExtent(
                normalized_axis_indices=(0, 1),
                divisor=1,
                rounding=SourceExtentRounding.EXACT,
                alignment=128,
            ),
        ),
        SourcePhysicalAxisSpec(
            "scale_k",
            SourceNormalizedAxisExtent(
                normalized_axis_indices=(2,),
                divisor=1,
                rounding=SourceExtentRounding.EXACT,
                alignment=4,
            ),
        ),
    )
    crop = _contract(
        "te.mxfp8.rowwise-compact-normalize.v1",
        SourceNormalizationKind.CROP,
        "2",
    )
    unswizzle = _contract(
        "te.mxfp8.rowwise-gemm-unswizzle.v1",
        SourceNormalizationKind.COMPOSITE,
        "3",
    )
    compact = _realization(
        realization_id="main.scale.rowwise-compact",
        output_record_id="main.scale",
        output_dtype=CanonicalSourceDType.E8M0,
        output_shape=normalized_shape,
        output_numeric_encoding="mxfp8_e8m0_scale",
        components=(
            _component(
                component_id="main.scale.compact",
                native_name="model.weight_rowwise_scale_inv",
                component_role="block_scales",
                dtype=CanonicalSourceDType.UINT8,
                normalized_shape=normalized_shape,
                physical_shape=(128, 32),
                physical_axes=physical_axes,
                storage_encoding="uint8-carried-e8m0",
                padding_semantics=SourcePaddingSemantics.UNSPECIFIED_IGNORED,
            ),
        ),
        normalization=crop,
    )
    swizzled = _realization(
        realization_id="main.scale.rowwise-gemm-swizzled",
        output_record_id="main.scale",
        output_dtype=CanonicalSourceDType.E8M0,
        output_shape=normalized_shape,
        output_numeric_encoding="mxfp8_e8m0_scale",
        components=(
            _component(
                component_id="main.scale.swizzled",
                native_name="model.weight_rowwise_scale_inv",
                component_role="block_scales",
                dtype=CanonicalSourceDType.UINT8,
                normalized_shape=normalized_shape,
                physical_shape=(128, 32),
                physical_axes=physical_axes,
                storage_encoding="uint8-carried-e8m0",
                padding_semantics=SourcePaddingSemantics.ZERO_FILLED,
                padding_fill_encoding="raw-zero",
                swizzle_id="te.mxfp8.gemm-128x4.v1",
            ),
        ),
        normalization=unswizzle,
    )
    inventory = SourceStorageRealizationInventory(
        graph_instance_id="main",
        normalizer_manifest=SourceNormalizerManifest(
            schema_version=1,
            contracts=(crop, unswizzle),
        ),
        realizations=(swizzled, compact),
    )

    assert tuple(item.realization_id for item in inventory.realizations) == (
        "main.scale.rowwise-compact",
        "main.scale.rowwise-gemm-swizzled",
    )
    assert compact.components[0].physical_shape == (128, 32)
    assert swizzled.components[0].swizzle_id == "te.mxfp8.gemm-128x4.v1"
    assert not source_realizations_have_exact_physical_representation(compact, swizzled)


def test_te_columnwise_scale_alignment_resolves_from_normalized_grid() -> None:
    normalized_shape = (3, 928)
    columnwise = _realization(
        realization_id="main.scale.columnwise",
        output_record_id="main.columnwise-scale",
        output_dtype=CanonicalSourceDType.E8M0,
        output_shape=normalized_shape,
        output_numeric_encoding="mxfp8_e8m0_scale",
        components=(
            _component(
                component_id="main.scale.columnwise",
                component_role="block_scales",
                dtype=CanonicalSourceDType.UINT8,
                normalized_shape=normalized_shape,
                physical_shape=(4, 1024),
                physical_axes=(
                    SourcePhysicalAxisSpec(
                        "scale_m",
                        SourceNormalizedAxisExtent(
                            normalized_axis_indices=(0,),
                            divisor=1,
                            rounding=SourceExtentRounding.EXACT,
                            alignment=4,
                        ),
                    ),
                    SourcePhysicalAxisSpec(
                        "k",
                        SourceNormalizedAxisExtent(
                            normalized_axis_indices=(1,),
                            divisor=1,
                            rounding=SourceExtentRounding.EXACT,
                            alignment=128,
                        ),
                    ),
                ),
                storage_encoding="uint8-carried-e8m0",
                padding_semantics=SourcePaddingSemantics.UNSPECIFIED_IGNORED,
            ),
        ),
        normalization=_contract(
            "te.mxfp8.columnwise-compact-normalize.v1",
            SourceNormalizationKind.CROP,
            "4",
        ),
    )

    assert columnwise.components[0].physical_shape == (4, 1024)


def test_realization_rejects_physical_shape_that_disagrees_with_formula() -> None:
    with pytest.raises(ValueError, match="physical shape.*normalized-axis formulas"):
        _realization(
            normalization=_contract(
                "test.repack.v1",
                SourceNormalizationKind.REPACK,
            ),
            components=(
                _component(
                    physical_shape=(128, 928, 2048),
                    physical_axes=_identity_axes((128, 928, 2688)),
                ),
            ),
        )


@pytest.mark.parametrize(
    "padding_semantics",
    (SourcePaddingSemantics.ZERO_FILLED, SourcePaddingSemantics.UNSPECIFIED_IGNORED),
)
def test_unpadded_formula_rejects_a_padded_storage_contract(
    padding_semantics: SourcePaddingSemantics,
) -> None:
    with pytest.raises(ValueError, match="padding semantics.*physical extent"):
        _realization(
            normalization=_contract(
                "test.repack.v1",
                SourceNormalizationKind.REPACK,
            ),
            components=(
                _component(
                    padding_semantics=padding_semantics,
                    padding_fill_encoding=(
                        "raw-zero"
                        if padding_semantics is SourcePaddingSemantics.ZERO_FILLED
                        else None
                    ),
                ),
            ),
        )


def test_aligned_padding_formula_rejects_no_padding_semantics() -> None:
    with pytest.raises(ValueError, match="padding semantics.*physical extent"):
        _realization(
            output_shape=(3,),
            normalization=_contract(
                "test.crop.v1",
                SourceNormalizationKind.CROP,
            ),
            components=(
                _component(
                    normalized_shape=(3,),
                    physical_shape=(4,),
                    physical_axes=(
                        SourcePhysicalAxisSpec(
                            "aligned",
                            SourceNormalizedAxisExtent(
                                normalized_axis_indices=(0,),
                                divisor=1,
                                rounding=SourceExtentRounding.EXACT,
                                alignment=4,
                            ),
                        ),
                    ),
                ),
            ),
        )


def test_ceil_division_remainder_requires_padding_semantics() -> None:
    physical_axes = (
        SourcePhysicalAxisSpec(
            "packed_blocks",
            SourceNormalizedAxisExtent(
                normalized_axis_indices=(0,),
                divisor=32,
                rounding=SourceExtentRounding.CEIL,
                alignment=1,
            ),
        ),
    )
    component = _component(
        normalized_shape=(33,),
        physical_shape=(2,),
        physical_axes=physical_axes,
        storage_encoding="packed-block-32",
        padding_semantics=SourcePaddingSemantics.UNSPECIFIED_IGNORED,
    )

    realization = _realization(
        output_shape=(33,),
        components=(component,),
        normalization=_contract(
            "test.unpack-block-32.v1",
            SourceNormalizationKind.UNFLATTEN,
        ),
    )

    assert realization.components[0].physical_shape == (2,)
    with pytest.raises(ValueError, match="padding semantics.*physical extent"):
        replace(
            realization,
            components=(
                replace(
                    component,
                    padding_semantics=SourcePaddingSemantics.NO_PADDING,
                ),
            ),
        )


def test_alternative_realizations_must_agree_on_normalized_output() -> None:
    reinterpret = _contract(
        "test.reinterpret.v1",
        SourceNormalizationKind.REINTERPRET,
        "6",
    )
    first = _realization(normalization=reinterpret)
    second = _realization(
        realization_id="main.weight.alternative",
        components=(
            _component(
                component_id="main.weight.alternative",
                native_name="model.weight.alternative",
            ),
        ),
        normalization=reinterpret,
    )
    manifest = SourceNormalizerManifest(schema_version=1, contracts=(reinterpret,))

    with pytest.raises(ValueError, match="alternative realizations.*output"):
        SourceStorageRealizationInventory(
            graph_instance_id="main",
            normalizer_manifest=manifest,
            realizations=(
                first,
                replace(second, output_numeric_encoding="different_bfloat16"),
            ),
        )


def test_realization_requires_exact_normalizer_manifest_membership() -> None:
    allowed = _contract()
    unknown = _contract("test.other.v1", SourceNormalizationKind.IDENTITY, "2")

    with pytest.raises(ValueError, match="normalization contract.*manifest"):
        SourceStorageRealizationInventory(
            graph_instance_id="main",
            normalizer_manifest=SourceNormalizerManifest(
                schema_version=1,
                contracts=(allowed,),
            ),
            realizations=(_realization(normalization=unknown),),
        )


def test_native_component_identity_has_one_canonical_metadata_record() -> None:
    first = _realization()
    second = replace(
        first,
        realization_id="main.weight.alternative",
        components=(
            replace(
                first.components[0],
                source_native_name="model.other_weight",
            ),
        ),
    )

    with pytest.raises(ValueError, match="native component ID.*conflicting"):
        SourceStorageRealizationInventory(
            graph_instance_id="main",
            normalizer_manifest=SourceNormalizerManifest(
                schema_version=1,
                contracts=(_contract(),),
            ),
            realizations=(first, second),
        )


@pytest.mark.parametrize("digest", ["", "1" * 64, "sha256:ABC", None])
def test_normalizer_contract_rejects_missing_or_noncanonical_digest(
    digest: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="digest"):
        SourceNormalizationContract(
            capability_id="test.identity.v1",
            kind=SourceNormalizationKind.IDENTITY,
            contract_digest=digest,  # type: ignore[arg-type]
        )


def test_identity_normalization_rejects_uint8_carrier_for_e4m3_view() -> None:
    with pytest.raises(ValueError, match="identity normalization"):
        _realization(
            output_dtype=CanonicalSourceDType.E4M3,
            output_numeric_encoding="mxfp8_e4m3_values",
            components=(
                _component(
                    dtype=CanonicalSourceDType.UINT8,
                    storage_encoding="uint8-carried-e4m3",
                ),
            ),
        )


def test_padding_semantic_mismatch_prevents_direct_copy_compatibility() -> None:
    shape = (3, 32, 29)
    axes = (
        SourcePhysicalAxisSpec(
            "flattened_m",
            SourceNormalizedAxisExtent(
                normalized_axis_indices=(0, 1),
                divisor=1,
                rounding=SourceExtentRounding.EXACT,
                alignment=128,
            ),
        ),
        SourcePhysicalAxisSpec(
            "scale_k",
            SourceNormalizedAxisExtent(
                normalized_axis_indices=(2,),
                divisor=1,
                rounding=SourceExtentRounding.EXACT,
                alignment=4,
            ),
        ),
    )
    normalization = _contract("test.crop.v1", SourceNormalizationKind.CROP)
    unspecified = _realization(
        output_shape=shape,
        components=(
            _component(
                dtype=CanonicalSourceDType.UINT8,
                normalized_shape=shape,
                physical_shape=(128, 32),
                physical_axes=axes,
                storage_encoding="uint8-carried-e8m0",
                padding_semantics=SourcePaddingSemantics.UNSPECIFIED_IGNORED,
            ),
        ),
        normalization=normalization,
    )
    zeroed = replace(
        unspecified,
        realization_id="main.weight.zeroed",
        components=(
            replace(
                unspecified.components[0],
                native_component_id="main.weight.zeroed",
                padding_semantics=SourcePaddingSemantics.ZERO_FILLED,
                padding_fill_encoding="raw-zero",
            ),
        ),
    )

    assert not source_realizations_have_exact_physical_representation(
        unspecified, zeroed
    )


@pytest.mark.parametrize(
    ("semantics", "fill"),
    [
        (SourcePaddingSemantics.ZERO_FILLED, None),
        (SourcePaddingSemantics.UNSPECIFIED_IGNORED, "raw-zero"),
        (SourcePaddingSemantics.NO_PADDING, "raw-zero"),
    ],
)
def test_padding_semantics_conditionally_require_or_forbid_fill_encoding(
    semantics: SourcePaddingSemantics,
    fill: str | None,
) -> None:
    with pytest.raises(ValueError, match="padding.*fill"):
        _component(
            normalized_shape=(3,),
            physical_shape=(4,),
            physical_axes=(
                SourcePhysicalAxisSpec(
                    "aligned",
                    SourceNormalizedAxisExtent(
                        normalized_axis_indices=(0,),
                        divisor=1,
                        rounding=SourceExtentRounding.EXACT,
                        alignment=4,
                    ),
                ),
            ),
            padding_semantics=semantics,
            padding_fill_encoding=fill,
        )


def test_backend_derived_realization_has_no_raw_component_or_wire_path() -> None:
    derivation = _contract(
        "backend.derived-from-owner.v1",
        SourceNormalizationKind.BACKEND_DERIVATION,
        "5",
    )
    witness = SourceDerivedRealization(
        realization_id="main.derived",
        graph_instance_id="main",
        output_record_id="main.derived",
        output_dtype=CanonicalSourceDType.BFLOAT16,
        output_shape=(128,),
        output_numeric_encoding="plain_bfloat16",
        derivation=derivation,
    )
    inventory = SourceStorageRealizationInventory(
        graph_instance_id="main",
        normalizer_manifest=SourceNormalizerManifest(
            schema_version=1,
            contracts=(derivation,),
        ),
        realizations=(witness,),
    )

    assert not hasattr(witness, "components")
    assert not source_realization_is_wire_eligible(witness)
    assert inventory.realizations == (witness,)


def test_source_storage_contracts_are_exported_from_public_boundary() -> None:
    expected_exports = {
        "SourceDerivedRealization": SourceDerivedRealization,
        "SourceExtentRounding": SourceExtentRounding,
        "SourceLiteralAxisExtent": SourceLiteralAxisExtent,
        "SourceNormalizationContract": SourceNormalizationContract,
        "SourceNormalizationKind": SourceNormalizationKind,
        "SourceNormalizedAxisExtent": SourceNormalizedAxisExtent,
        "SourceNormalizerManifest": SourceNormalizerManifest,
        "SourcePaddingSemantics": SourcePaddingSemantics,
        "SourcePhysicalAxisSpec": SourcePhysicalAxisSpec,
        "SourceStorageComponent": SourceStorageComponent,
        "SourceStorageRealization": SourceStorageRealization,
        "SourceStorageRealizationInventory": SourceStorageRealizationInventory,
        "source_normalizer_manifest_digest": source_normalizer_manifest_digest,
        "source_realization_is_wire_eligible": source_realization_is_wire_eligible,
        "source_realizations_have_exact_physical_representation": (
            source_realizations_have_exact_physical_representation
        ),
        "source_storage_inventory_digest": source_storage_inventory_digest,
        "validate_source_storage_realization_inventory": (
            validate_source_storage_realization_inventory
        ),
    }

    assert {
        name: getattr(precision_policy, name, None) for name in expected_exports
    } == expected_exports


def test_realization_schema_is_metadata_only_and_has_no_payload_accessor() -> None:
    assert tuple(field.name for field in fields(SourceStorageRealization)) == (
        "realization_id",
        "graph_instance_id",
        "output_record_id",
        "components",
        "output_dtype",
        "output_shape",
        "output_numeric_encoding",
        "normalization",
    )
    assert not {
        "payload",
        "tensor",
        "buffer",
        "accessor",
        "normalizer_callable",
        "transformed_payload",
    } & {field.name for field in fields(SourceStorageComponent)}
