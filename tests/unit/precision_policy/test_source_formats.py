from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from nemo_rl.precision_policy.semantic import (
    BF16_FORMAT,
    MXFP8_FORMAT,
    AxisExtentRounding,
    ComponentDescriptor,
    ComponentRole,
    FormatDescriptor,
    LiteralComponentAxisSpec,
    LogicalComponentAxisSpec,
    resolve_component_axes,
)
from nemo_rl.precision_policy.source_formats import (
    SOURCE_FORMAT_CATALOG,
    build_source_format_catalog,
)
from tools.capture_precision_policy_source_evidence import (
    CheckpointArtifactSpec,
    CheckpointObservationSpec,
    EvidenceError,
    capture_staged_checkpoint_evidence,
    load_staged_source_format_evidence,
    validate_producer_implementation_evidence,
    validate_source_format_evidence,
)


FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "precision_policy"

EXPECTED_SOURCE_FORMATS: tuple[
    tuple[str, tuple[str, tuple[tuple[str, str, str, object], ...]]], ...
] = (
    (
        "bf16.logical.v1",
        ("bf16", (("logical_values", "bfloat16", "plain_bfloat16", None),)),
    ),
    (
        "mxfp8.e4m3-e8m0-block32-input-features.v1",
        (
            "mxfp8",
            (
                ("values", "e4m3", "mxfp8_e4m3_values", None),
                (
                    "block_scales",
                    "e8m0",
                    "mxfp8_e8m0_scale",
                    (
                        ("output_features", 1, "exact"),
                        ("input_features", 32, "ceil"),
                    ),
                ),
            ),
        ),
    ),
    (
        "block-fp8.e4m3-f32-scale-inv-block128x128.v1",
        (
            "block_fp8",
            (
                ("values", "e4m3", "float8_e4m3_values", None),
                (
                    "inverse_scales",
                    "float32",
                    "inverse_scale_float32",
                    (
                        ("output_features", 128, "exact"),
                        ("input_features", 128, "exact"),
                    ),
                ),
            ),
        ),
    ),
    (
        "block-fp8.e4m3-bf16-scale-inv-block128x128.v1",
        (
            "block_fp8",
            (
                ("values", "e4m3", "float8_e4m3_values", None),
                (
                    "inverse_scales",
                    "bfloat16",
                    "inverse_scale_bfloat16",
                    (
                        ("output_features", 128, "exact"),
                        ("input_features", 128, "exact"),
                    ),
                ),
            ),
        ),
    ),
    (
        "packed-int4.i32-bf16-group32-shape-i32.v1",
        (
            "packed_int4",
            (
                (
                    "packed_values",
                    "int32",
                    "int4_offset_binary_pack8",
                    (
                        ("output_features", 1, "exact"),
                        ("input_features", 8, "exact"),
                    ),
                ),
                (
                    "group_scales",
                    "bfloat16",
                    "symmetric_group_scale",
                    (
                        ("output_features", 1, "exact"),
                        ("input_features", 32, "exact"),
                    ),
                ),
                (
                    "logical_shape",
                    "int32",
                    "logical_shape_vector",
                    (("literal", 2, "exact"),),
                ),
            ),
        ),
    ),
    (
        "packed-int4.i32-f16-group32-shape-i64.v1",
        (
            "packed_int4",
            (
                (
                    "packed_values",
                    "int32",
                    "int4_offset_binary_pack8",
                    (
                        ("output_features", 1, "exact"),
                        ("input_features", 8, "exact"),
                    ),
                ),
                (
                    "group_scales",
                    "float16",
                    "symmetric_group_scale",
                    (
                        ("output_features", 1, "exact"),
                        ("input_features", 32, "exact"),
                    ),
                ),
                (
                    "logical_shape",
                    "int64",
                    "logical_shape_vector",
                    (("literal", 2, "exact"),),
                ),
            ),
        ),
    ),
    (
        "mxfp4.u8-u8-block32-input-features.v1",
        (
            "mxfp4",
            (
                (
                    "packed_values",
                    "uint8",
                    "mxfp4_pack2",
                    (
                        ("output_features", 1, "exact"),
                        ("input_features", 2, "exact"),
                    ),
                ),
                (
                    "block_scales",
                    "uint8",
                    "mxfp4_block_scale",
                    (
                        ("output_features", 1, "exact"),
                        ("input_features", 32, "exact"),
                    ),
                ),
            ),
        ),
    ),
    (
        "nvfp4.u8-e4m3-f32-block16-input-features.v1",
        (
            "nvfp4",
            (
                (
                    "packed_values",
                    "uint8",
                    "nvfp4_pack2",
                    (
                        ("output_features", 1, "exact"),
                        ("input_features", 2, "exact"),
                    ),
                ),
                (
                    "block_scales",
                    "e4m3",
                    "nvfp4_block_scale",
                    (
                        ("output_features", 1, "exact"),
                        ("input_features", 16, "exact"),
                    ),
                ),
                ("global_scale", "float32", "nvfp4_global_scale", ()),
            ),
        ),
    ),
)


def _serialize_axes(
    axes: tuple[LogicalComponentAxisSpec | LiteralComponentAxisSpec, ...] | None,
) -> object:
    if axes is None:
        return None
    return tuple(
        (
            axis.logical_axis,
            axis.divisor,
            axis.rounding.value,
        )
        if isinstance(axis, LogicalComponentAxisSpec)
        else (axis.axis_name, axis.extent, AxisExtentRounding.EXACT.value)
        for axis in axes
    )


def _serialize_catalog(
    catalog: Sequence[FormatDescriptor],
) -> tuple[
    tuple[str, tuple[str, tuple[tuple[str, str, str | None, object], ...]]], ...
]:
    return tuple(
        (
            descriptor.format_id,
            (
                descriptor.family,
                tuple(
                    (
                        component.role,
                        component.dtype,
                        component.encoding,
                        _serialize_axes(component.component_axes),
                    )
                    for component in descriptor.components
                ),
            ),
        )
        for descriptor in catalog
    )


def _load_fixture(name: str) -> dict[str, Any]:
    value = json.loads((FIXTURE_ROOT / name).read_text())
    assert isinstance(value, dict)
    return value


def _claim(document: Mapping[str, Any], format_id: str, role: str) -> dict[str, Any]:
    claims = document["claims"]
    assert isinstance(claims, list)
    return next(
        item
        for item in claims
        if item["format_id"] == format_id and item["role"] == role
    )


def test_source_format_catalog_is_literal_ordered_and_reuses_builtins() -> None:
    assert _serialize_catalog(SOURCE_FORMAT_CATALOG) == EXPECTED_SOURCE_FORMATS
    catalog_by_id = {item.format_id: item for item in SOURCE_FORMAT_CATALOG}
    assert catalog_by_id[BF16_FORMAT.format_id] is BF16_FORMAT
    assert catalog_by_id[MXFP8_FORMAT.format_id] is MXFP8_FORMAT
    assert len(catalog_by_id) == len(SOURCE_FORMAT_CATALOG)


def test_catalog_ids_are_canonical_and_not_model_or_repository_aliases() -> None:
    forbidden_fragments = ("kimi", "qwen", "nemotron", "automodel", "bridge")
    assert all(
        fragment not in descriptor.format_id.lower()
        for descriptor in SOURCE_FORMAT_CATALOG
        for fragment in forbidden_fragments
    )


def test_catalog_construction_rejects_every_duplicate_stable_id() -> None:
    with pytest.raises(ValueError, match="duplicate source format_id"):
        build_source_format_catalog((BF16_FORMAT, BF16_FORMAT))

    conflicting = FormatDescriptor(
        BF16_FORMAT.format_id,
        "conflicting_family",
        (ComponentDescriptor(ComponentRole("other_values"), "float32"),),
    )
    with pytest.raises(ValueError, match="duplicate source format_id"):
        build_source_format_catalog((BF16_FORMAT, conflicting))


def test_catalog_construction_rejects_recreated_builtin_object() -> None:
    recreated = FormatDescriptor(
        BF16_FORMAT.format_id,
        BF16_FORMAT.family,
        BF16_FORMAT.components,
    )
    assert recreated == BF16_FORMAT
    assert recreated is not BF16_FORMAT
    with pytest.raises(ValueError, match="must reuse its canonical object"):
        build_source_format_catalog((recreated,))


def test_block_fp8_scale_dtype_is_part_of_canonical_identity() -> None:
    by_id = {item.format_id: item for item in SOURCE_FORMAT_CATALOG}
    k2 = by_id["block-fp8.e4m3-f32-scale-inv-block128x128.v1"]
    a95b = by_id["block-fp8.e4m3-bf16-scale-inv-block128x128.v1"]
    assert k2 is not a95b
    assert k2.components[1].dtype == "float32"
    assert a95b.components[1].dtype == "bfloat16"
    assert k2.components[1].encoding == "inverse_scale_float32"
    assert a95b.components[1].encoding == "inverse_scale_bfloat16"


def test_a95b_exact_block_geometry_resolves_observed_orientations() -> None:
    descriptor = next(
        item
        for item in SOURCE_FORMAT_CATALOG
        if item.format_id == "block-fp8.e4m3-bf16-scale-inv-block128x128.v1"
    )
    scales = descriptor.components[1]
    logical_axes = ("output_features", "input_features")
    assert resolve_component_axes(
        scales,
        logical_axes=logical_axes,
        logical_shape=(2048, 8192),
    ) == (("output_features", 16), ("input_features", 64))
    assert resolve_component_axes(
        scales,
        logical_axes=logical_axes,
        logical_shape=(8192, 2048),
    ) == (("output_features", 64), ("input_features", 16))

    for nondivisible_shape in ((2049, 8192), (2048, 8193)):
        with pytest.raises(ValueError, match="exactly divisible by 128"):
            resolve_component_axes(
                scales,
                logical_axes=logical_axes,
                logical_shape=nondivisible_shape,
            )


def test_u8_and_packed_int4_carriers_do_not_collapse_distinct_formats() -> None:
    by_id = {item.format_id: item for item in SOURCE_FORMAT_CATALOG}
    checkpoint_int4 = by_id["packed-int4.i32-bf16-group32-shape-i32.v1"]
    automodel_int4 = by_id["packed-int4.i32-f16-group32-shape-i64.v1"]
    mxfp4 = by_id["mxfp4.u8-u8-block32-input-features.v1"]
    nvfp4 = by_id["nvfp4.u8-e4m3-f32-block16-input-features.v1"]

    assert checkpoint_int4.components[1:] != automodel_int4.components[1:]
    assert mxfp4.components[0].dtype == nvfp4.components[0].dtype == "uint8"
    assert mxfp4.family != nvfp4.family
    assert mxfp4.components != nvfp4.components


def test_automodel_group32_format_rejects_nondivisible_width() -> None:
    descriptor = next(
        item
        for item in SOURCE_FORMAT_CATALOG
        if item.format_id == "packed-int4.i32-f16-group32-shape-i64.v1"
    )
    logical_axes = ("output_features", "input_features")
    logical_shape = (7, 40)
    assert resolve_component_axes(
        descriptor.components[0],
        logical_axes=logical_axes,
        logical_shape=logical_shape,
    ) == (("output_features", 7), ("input_features", 5))
    with pytest.raises(ValueError, match="exactly divisible by 32"):
        resolve_component_axes(
            descriptor.components[1],
            logical_axes=logical_axes,
            logical_shape=logical_shape,
        )


def test_raw_checkpoint_capture_derives_observation_from_independent_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "checkpoints" / "example"
    artifact_root.mkdir(parents=True)
    config_path = artifact_root / "config.json"
    index_path = artifact_root / "model.safetensors.index.json"
    header_path = artifact_root / "safetensors_header_manifest.json"
    config_path.write_text(
        '{"input_features":8,"model_type":"example","output_features":4}\n'
    )
    tensor_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 64},
                "weight_map": {tensor_name: "model-00001-of-00001.safetensors"},
            },
            separators=(",", ":"),
        )
    )
    header_manifest = {
        tensor_name: {
            "dtype": "BF16",
            "shape": [4, 8],
            "shard": "model-00001-of-00001.safetensors",
        }
    }
    header_path.write_text(json.dumps(header_manifest))
    # A pre-shaped output beside the raw inputs must neither be read nor trusted.
    (artifact_root / "source_format_observations.json").write_text(
        '{"observations":"poison"}'
    )
    tensor_payload = artifact_root / "model-00001-of-00001.safetensors"
    tensor_payload.write_bytes(b"tensor payload must remain unread")

    artifact = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "header_manifest_sha256": hashlib.sha256(
            json.dumps(
                header_manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "kind": "immutable_hf_metadata",
        "repository": "example/repository",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "shards": 1,
        "tensors": 1,
    }
    spec = CheckpointArtifactSpec(
        artifact_id="example",
        artifact=artifact,
        observations=(
            CheckpointObservationSpec(
                format_id=BF16_FORMAT.format_id,
                graph="main",
                logical_shape_config_paths=(
                    ("output_features",),
                    ("input_features",),
                ),
                observation_id="example.main.gate",
                producer="checkpoint.example",
                projection="gate",
                raw_prefix="model.layers.0.mlp.experts.0.gate_proj",
                source_dtypes=("BF16",),
                suffixes=(".weight",),
            ),
        ),
    )
    config_digest = cast(str, artifact["config_sha256"])
    index_digest = cast(str, artifact["index_sha256"])
    header_raw_digest = hashlib.sha256(header_path.read_bytes()).hexdigest()
    original_open = Path.open
    actual_opened_paths: list[Path] = []

    def tracked_open(path: Path, *args: object, **kwargs: object):
        actual_opened_paths.append(path)
        if path.suffix == ".safetensors":
            raise AssertionError("capture tool attempted to open a tensor payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    captured_artifact, observations, capture_receipt = (
        capture_staged_checkpoint_evidence(tmp_path, spec)
    )

    assert captured_artifact == artifact
    assert observations == [
        {
            "artifact": "example",
            "components": [
                {
                    "metadata_location": "model-00001-of-00001.safetensors",
                    "raw_name": tensor_name,
                    "role": "logical_values",
                    "shape": [4, 8],
                    "source_dtype": "BF16",
                }
            ],
            "format_id": BF16_FORMAT.format_id,
            "graph": "main",
            "logical_axes": ["output_features", "input_features"],
            "logical_shape": [4, 8],
            "observation_id": "example.main.gate",
            "producer": "checkpoint.example",
            "projection": "gate",
            "raw_siblings": [tensor_name],
        }
    ]
    assert capture_receipt == (
        {
            "path": "checkpoints/example/config.json",
            "sha256": f"sha256:{config_digest}",
        },
        {
            "path": "checkpoints/example/model.safetensors.index.json",
            "sha256": f"sha256:{index_digest}",
        },
        {
            "path": "checkpoints/example/safetensors_header_manifest.json",
            "sha256": f"sha256:{header_raw_digest}",
        },
    )
    assert actual_opened_paths == [config_path, index_path, header_path]

    monkeypatch.undo()

    with pytest.raises(EvidenceError, match="invalid staged artifact id"):
        capture_staged_checkpoint_evidence(
            tmp_path,
            replace(spec, artifact_id=".."),
        )

    symlinked_artifact = tmp_path / "checkpoints" / "escape"
    symlinked_artifact.symlink_to(artifact_root, target_is_directory=True)
    with pytest.raises(EvidenceError, match="symlinked staged artifact"):
        capture_staged_checkpoint_evidence(
            tmp_path,
            replace(spec, artifact_id="escape"),
        )

    config_path.write_text(
        '{"input_features":8,"model_type":"example","output_features":5}\n'
    )
    changed_artifact = {
        **artifact,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    with pytest.raises(EvidenceError, match="component shape"):
        capture_staged_checkpoint_evidence(
            tmp_path,
            replace(spec, artifact=changed_artifact),
        )


def test_raw_checkpoint_capture_rejects_index_header_disagreement(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "checkpoints" / "example"
    artifact_root.mkdir(parents=True)
    config_path = artifact_root / "config.json"
    index_path = artifact_root / "model.safetensors.index.json"
    header_path = artifact_root / "safetensors_header_manifest.json"
    tensor_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    config_path.write_text('{"input_features":8,"output_features":4}')
    index_path.write_text(
        json.dumps({"weight_map": {tensor_name: "model-00001-of-00001.safetensors"}})
    )
    header_manifest = {
        tensor_name: {
            "dtype": "BF16",
            "shape": [4, 8],
            "shard": "wrong.safetensors",
        }
    }
    header_path.write_text(json.dumps(header_manifest))
    artifact = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "header_manifest_sha256": hashlib.sha256(
            json.dumps(
                header_manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "kind": "immutable_hf_metadata",
        "repository": "example/repository",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "shards": 1,
        "tensors": 1,
    }
    spec = CheckpointArtifactSpec(
        artifact_id="example",
        artifact=artifact,
        observations=(
            CheckpointObservationSpec(
                format_id=BF16_FORMAT.format_id,
                graph="main",
                logical_shape_config_paths=(
                    ("output_features",),
                    ("input_features",),
                ),
                observation_id="example.main.gate",
                producer="checkpoint.example",
                projection="gate",
                raw_prefix="model.layers.0.mlp.experts.0.gate_proj",
                source_dtypes=("BF16",),
                suffixes=(".weight",),
            ),
        ),
    )

    with pytest.raises(EvidenceError, match="index/header shard mismatch"):
        capture_staged_checkpoint_evidence(tmp_path, spec)


def test_source_evidence_requires_exact_raw_dtype_and_logical_axis_order() -> None:
    original = _load_fixture("source_format_evidence.json")

    canonicalized_dtype = copy.deepcopy(original)
    observation = next(
        item
        for item in canonicalized_dtype["observations"]
        if item["observation_id"] == "a95b.main.gate"
    )
    observation["components"][0]["source_dtype"] = "e4m3"
    with pytest.raises(EvidenceError, match="exact raw dtype"):
        validate_source_format_evidence(canonicalized_dtype)

    reordered_axes = copy.deepcopy(original)
    observation = next(
        item
        for item in reordered_axes["observations"]
        if item["observation_id"] == "a95b.main.gate"
    )
    observation["logical_axes"] = ["input_features", "output_features"]
    observation["logical_shape"] = [8192, 2048]
    with pytest.raises(EvidenceError, match="logical axes"):
        validate_source_format_evidence(reordered_axes)


def test_source_evidence_fixture_is_complete_and_matches_catalog() -> None:
    validate_source_format_evidence(_load_fixture("source_format_evidence.json"))


def test_source_evidence_gate_rejects_capture_receipt_mutation() -> None:
    changed = _load_fixture("source_format_evidence.json")
    changed["capture_receipt"]["opened_metadata"][0]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(EvidenceError, match="capture receipt"):
        validate_source_format_evidence(changed)


def test_checkpoint_evidence_records_exact_quantization_vocabulary() -> None:
    evidence = _load_fixture("source_format_evidence.json")
    artifacts = evidence["artifacts"]

    k25_fields = {
        item["path"]: item["value"]
        for item in artifacts["kimi_k25"]["raw_config_fields"]
    }
    assert (
        k25_fields["text_config.quantization_config.config_groups.group_0.weights.type"]
        == "int"
    )
    assert (
        k25_fields[
            "text_config.quantization_config.config_groups.group_0.weights.symmetric"
        ]
        is True
    )

    k3_fields = {
        item["path"]: item["value"]
        for item in artifacts["kimi_k3"]["raw_config_fields"]
    }
    assert (
        k3_fields["text_config.quantization_config.config_groups.group_0.weights.type"]
        == "float"
    )
    assert (
        k3_fields[
            "text_config.quantization_config.config_groups.group_0.weights.symmetric"
        ]
        is True
    )

    lightning_fields = {
        item["path"]: item["value"]
        for item in artifacts["nemotron_lightning_nvfp4"]["raw_config_fields"]
    }
    assert lightning_fields["quantization_config.quant_algo"] == "MIXED_PRECISION"
    assert lightning_fields["quantization_config.producer"] == {
        "name": "modelopt",
        "version": "0.44.0rc5",
    }
    for target in (
        "backbone.layers.1.mixer.experts.0.up_proj",
        "backbone.layers.1.mixer.experts.0.down_proj",
    ):
        assert (
            lightning_fields[
                f"quantization_config.quantized_layers.{target}.quant_algo"
            ]
            == "W4A16_NVFP4"
        )
        assert (
            lightning_fields[
                f"quantization_config.quantized_layers.{target}.group_size"
            ]
            == 16
        )

    changed = copy.deepcopy(evidence)
    raw_fields = changed["artifacts"]["nemotron_lightning_nvfp4"]["raw_config_fields"]
    next(
        item
        for item in raw_fields
        if item["path"].endswith("experts.0.up_proj.quant_algo")
    )["value"] = "NVFP4"
    with pytest.raises(EvidenceError, match="source artifact evidence"):
        validate_source_format_evidence(changed)


def test_constructed_format_claims_require_defining_source_contracts() -> None:
    evidence = _load_fixture("source_format_evidence.json")

    assert _claim(
        evidence,
        "packed-int4.i32-bf16-group32-shape-i32.v1",
        "packed_values",
    )["evidence"] == ["kimi_k25", "compressed_tensors_0_17_0"]
    assert _claim(
        evidence,
        "mxfp4.u8-u8-block32-input-features.v1",
        "packed_values",
    )["evidence"] == ["kimi_k3", "compressed_tensors_0_17_0"]
    assert _claim(
        evidence,
        "nvfp4.u8-e4m3-f32-block16-input-features.v1",
        "packed_values",
    )["evidence"] == ["nemotron_lightning_nvfp4", "modelopt_0_44_0rc5"]


def test_representative_observations_cover_projections_graphs_and_producers() -> None:
    evidence = _load_fixture("source_format_evidence.json")
    observations = evidence["observations"]
    observed = {
        (item["producer"], item["graph"], item["projection"]) for item in observations
    }
    assert {
        ("checkpoint.kimi_k2", "main", projection)
        for projection in ("gate", "up", "down")
    } <= observed
    assert {
        ("checkpoint.kimi_k25", "main", projection)
        for projection in ("gate", "up", "down")
    } <= observed
    assert {
        ("automodel.kimi_k25", "main", projection)
        for projection in ("gate", "up", "down")
    } <= observed
    assert {
        ("checkpoint.kimi_k3", "main", projection)
        for projection in ("gate", "up", "down")
    } <= observed
    assert {
        ("checkpoint.nemotron_lightning_nvfp4", "main", projection)
        for projection in ("up", "down")
    } <= observed
    assert {
        ("checkpoint.qwen_a95b_fp8", graph, projection)
        for graph in ("main", "mtp.0")
        for projection in ("gate", "up", "down")
    } <= observed


def test_source_evidence_gate_rejects_missing_or_changed_component_contract() -> None:
    original = _load_fixture("source_format_evidence.json")
    format_id = "block-fp8.e4m3-bf16-scale-inv-block128x128.v1"

    missing_role = copy.deepcopy(original)
    missing_role["claims"].remove(_claim(missing_role, format_id, "inverse_scales"))
    with pytest.raises(ValueError, match="catalog component claims"):
        validate_source_format_evidence(missing_role)

    for field, replacement in (
        ("dtype", "float32"),
        ("encoding", "unproven_encoding"),
        ("axes", None),
    ):
        changed = copy.deepcopy(original)
        _claim(changed, format_id, "inverse_scales")["contract"][field] = replacement
        with pytest.raises(ValueError, match="claim contract"):
            validate_source_format_evidence(changed)

    for field, replacement in (("divisor", 64), ("rounding", "ceil")):
        changed = copy.deepcopy(original)
        axes = _claim(changed, format_id, "inverse_scales")["contract"]["axes"]
        axes[0][field] = replacement
        with pytest.raises(ValueError, match="claim contract"):
            validate_source_format_evidence(changed)


def test_source_evidence_gate_rejects_unproven_a95b_remainder_or_geometry() -> None:
    original = _load_fixture("source_format_evidence.json")
    assert original["artifacts"]["qwen_a95b_fp8"]["mtp_header_byte_lengths"] == {
        "model-00185-of-00213.safetensors": 254184,
        "model-00186-of-00213.safetensors": 127080,
    }

    for field, replacement in (
        ("weight_block_size", [128, 64]),
        ("catalog_admission", "allow_nondivisible"),
        ("remainder_evidence", "ceil_observed"),
    ):
        changed = copy.deepcopy(original)
        a95b = changed["artifacts"]["qwen_a95b_fp8"]
        a95b[field] = replacement
        with pytest.raises(ValueError, match="A95B"):
            validate_source_format_evidence(changed)


def test_source_evidence_gate_rejects_observation_shape_or_sibling_loss() -> None:
    original = _load_fixture("source_format_evidence.json")
    changed_shape = copy.deepcopy(original)
    observation = next(
        item
        for item in changed_shape["observations"]
        if item["observation_id"] == "a95b.main.gate"
    )
    observation["components"][1]["shape"] = [16, 63]
    with pytest.raises(ValueError, match="component shape"):
        validate_source_format_evidence(changed_shape)

    missing_sibling = copy.deepcopy(original)
    observation = next(
        item
        for item in missing_sibling["observations"]
        if item["observation_id"] == "a95b.main.gate"
    )
    observation["components"].pop()
    with pytest.raises(ValueError, match="component roles"):
        validate_source_format_evidence(missing_sibling)


def test_producer_evidence_keeps_root_and_bridge_te_identities_distinct() -> None:
    evidence = _load_fixture("producer_implementations.json")
    implementations = evidence["implementations"]
    root_te = implementations["transformer_engine_root_runtime"]
    bridge_te = implementations["transformer_engine_bridge_source"]
    assert root_te["source_revision"] == ("42b840051647eef89761a16dfdff87e82bb253ab")
    assert root_te["package_identity"] == "2.15.0+42b8400"
    assert bridge_te["source_revision"] == ("4329ff84bfbdaa778a33cba02a15fb0807c64689")
    assert bridge_te["package_identity"] == "2.17.1+4329ff84"
    assert root_te != bridge_te
    with pytest.raises(EvidenceError, match="effective Transformer Engine runtime"):
        validate_producer_implementation_evidence(evidence)

    evidence["runtime_inspection"] = {
        "package_identity": "2.15.0+42b8400",
        "source_revision": "42b840051647eef89761a16dfdff87e82bb253ab",
        "status": "matched_root_lock",
    }
    validate_producer_implementation_evidence(evidence)


def test_producer_evidence_pins_every_te_native_realization_source() -> None:
    evidence = _load_fixture("producer_implementations.json")
    evidence["runtime_inspection"] = {
        "package_identity": "2.15.0+42b8400",
        "source_revision": "42b840051647eef89761a16dfdff87e82bb253ab",
        "status": "matched_root_lock",
    }
    te_contract = evidence["source_contracts"]["transformer_engine_native_mxfp8"]
    sources = te_contract["sources"]
    assert {
        "transformer_engine/common/cast/mxfp8/dequantize_mxfp8.cuh",
        "transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh",
        "transformer_engine/common/cast/mxfp8/swizzle.cuh",
        "transformer_engine/common/include/transformer_engine/transformer_engine.h",
        "transformer_engine/pytorch/csrc/quantizer.cpp",
        "transformer_engine/pytorch/onnx_extensions.py",
        "transformer_engine/pytorch/tensor/mxfp8_tensor.py",
        "transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py",
    } == set(sources)
    assert te_contract["contract"]["compact_padding_fill"] == "unspecified_ignored"
    assert te_contract["contract"]["swizzled_padding_fill"] == "zero"
    validate_producer_implementation_evidence(evidence)

    missing_source = copy.deepcopy(evidence)
    del missing_source["source_contracts"]["transformer_engine_native_mxfp8"][
        "sources"
    ]["transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh"]
    with pytest.raises(EvidenceError, match="producer source contracts"):
        validate_producer_implementation_evidence(missing_source)

    false_uniform_padding = copy.deepcopy(evidence)
    false_uniform_padding["source_contracts"]["transformer_engine_native_mxfp8"][
        "contract"
    ]["compact_padding_fill"] = "zero"
    with pytest.raises(EvidenceError, match="producer source contracts"):
        validate_producer_implementation_evidence(false_uniform_padding)


def test_producer_evidence_pins_numeric_encoding_sources() -> None:
    evidence = _load_fixture("producer_implementations.json")
    evidence["runtime_inspection"] = {
        "package_identity": "2.15.0+42b8400",
        "source_revision": "42b840051647eef89761a16dfdff87e82bb253ab",
        "status": "matched_root_lock",
    }

    implementations = evidence["implementations"]
    assert implementations["compressed_tensors_format_spec"] == {
        "head_revision": "f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0",
        "kind": "locked_registry_source",
        "lock_path": "uv.lock",
        "package_identity": "0.17.0",
        "source_revision": "f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0",
        "wheel_sha256": (
            "4a1b89b508f7efb8ffb4eee8a6e69e0452d9b080cae130146025c64fbe9fa9aa"
        ),
    }
    assert implementations["modelopt_lightning_producer"] == {
        "head_revision": "c897fbeaaff66d53d61033f107885b7c5432f235",
        "kind": "versioned_git_source",
        "package_identity": "0.44.0rc5",
        "source_revision": "c897fbeaaff66d53d61033f107885b7c5432f235",
    }

    contracts = evidence["source_contracts"]
    assert (
        contracts["compressed_tensors_int4_pack"]["contract"]["nibble_order"]
        == "value_i_at_bits_4_times_i_lsb_first"
    )
    assert (
        contracts["kimi_k25_automodel_int4"]["contract"]["nibble_order"]
        == "value_i_at_bits_4_times_i_lsb_first"
    )
    assert (
        contracts["compressed_tensors_mxfp4_pack"]["contract"]["scale_encoding"]
        == "e8m0_bias_127"
    )
    assert (
        contracts["modelopt_nvfp4_pack"]["contract"]["nibble_order"]
        == "first_value_low_second_value_high"
    )
    validate_producer_implementation_evidence(evidence)

    changed = copy.deepcopy(evidence)
    changed["source_contracts"]["modelopt_nvfp4_pack"]["sources"][
        "modelopt/torch/quantization/qtensor/nvfp4_tensor.py"
    ] = "0" * 64
    with pytest.raises(EvidenceError, match="producer source contracts"):
        validate_producer_implementation_evidence(changed)


def test_producer_evidence_gate_rejects_every_missing_or_swapped_identity() -> None:
    original = _load_fixture("producer_implementations.json")
    expected_fields = {
        "megatron_bridge": ("gitlink_revision", "head_revision"),
        "nemo_automodel": ("gitlink_revision", "head_revision"),
        "megatron_core": ("gitlink_revision", "head_revision"),
        "transformer_engine_root_runtime": (
            "source_revision",
            "package_identity",
        ),
        "transformer_engine_bridge_source": (
            "source_revision",
            "package_identity",
        ),
    }
    for implementation, fields in expected_fields.items():
        for field in fields:
            changed = copy.deepcopy(original)
            del changed["implementations"][implementation][field]
            with pytest.raises(ValueError, match="producer implementation evidence"):
                validate_producer_implementation_evidence(changed)

    swapped = copy.deepcopy(original)
    implementations = swapped["implementations"]
    (
        implementations["transformer_engine_root_runtime"],
        implementations["transformer_engine_bridge_source"],
    ) = (
        implementations["transformer_engine_bridge_source"],
        implementations["transformer_engine_root_runtime"],
    )
    with pytest.raises(ValueError, match="producer implementation evidence"):
        validate_producer_implementation_evidence(swapped)


def test_staged_loader_fails_closed_without_reviewed_metadata(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError, match="missing staged checkpoints directory"):
        load_staged_source_format_evidence(tmp_path)
