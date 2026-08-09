from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import stat

import pytest

from experiments.mxfp8_moe_tactic_audit.build_experiment_override_cache import (
    build_experiment_override_cache,
)


MOE_OP = "flashinfer::trtllm_fp8_block_scale_moe"
MOE_RUNNER = "MoERunner"
QWEN_BUCKET_128_SHAPES = (
    (128, 2048),
    (128, 128),
    (128,),
    (128,),
    (128, 2048),
    (128, 64),
    (0,),
    (0,),
)


def _cache_key(shapes: tuple[tuple[int, ...], ...]) -> str:
    return str((MOE_OP, MOE_RUNNER, shapes, ()))


def _write_stock_cache(path: Path, *, duplicate_target: bool = False) -> dict[str, object]:
    target = _cache_key(QWEN_BUCKET_128_SHAPES)
    retained = _cache_key(((64, 2048), *QWEN_BUCKET_128_SHAPES[1:]))
    payload: dict[str, object] = {
        "_metadata": {
            "flashinfer_version": "0.6.13",
            "runtime_marker": "vllm-0.25.1",
        },
        target: [MOE_RUNNER, [16, 530]],
        retained: [MOE_RUNNER, [8, 229]],
        "cutedsl::dense-key": ["DenseRunner", 17],
    }
    if duplicate_target:
        alternate_key = f" {target}"
        payload[alternate_key] = [MOE_RUNNER, [16, 530]]
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")
    return payload


def _write_audit_summary(path: Path) -> None:
    payload = {
        "decision": "NO_PROMOTION",
        "gates": {
            "minimum_weighted_gain": 0.02,
            "maximum_cv": 0.03,
            "maximum_high_weight_regression": 0.01,
        },
        "measurement_context": {
            "flashinfer_version": "0.6.13",
            "model": "Qwen3-30B-A3B",
            "nemo_rl_commit": "d678bfbf2d3b05df8c628d5004a4d08756be4250",
            "prepacked_weight_sha256": "b" * 64,
            "stock_tactic": [16, 530],
            "vllm_commit": "b9eea5bbbec24a2af6acd0d92c02a3640a748e9c",
        },
        "same_run_comparison": {
            "candidates": [
                {
                    "maximum_cv": 0.037384,
                    "maximum_profile_regression": 0.0,
                    "pair": [32, 574],
                    "speedup_vs_stock": 1.015216,
                    "weighted_median_gain": 0.012167304,
                }
            ]
        },
        "source_artifacts": {
            "focused_measurements": {
                "sha256": "1" * 64,
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def _native_builder(request: Mapping[str, object]) -> Mapping[str, object]:
    stock_path = Path(str(request["stock_path"]))
    candidate_path = Path(str(request["candidate_path"]))
    payload = json.loads(stock_path.read_text(encoding="ascii"))
    promoted = request["promoted"]
    assert isinstance(promoted, Mapping)
    assert len(promoted) == 1
    target, tactic = next(iter(promoted.items()))
    assert tactic == {"gemm1": 32, "gemm2": 574}
    payload[target] = [MOE_RUNNER, [32, 574]]
    candidate_path.write_text(json.dumps(payload, indent=2), encoding="ascii")
    return {
        "exact_hit_validated": True,
        "fallback_miss_validated": True,
        "method": "injected-native-builder",
    }


def test_builds_one_key_unqualified_override_with_reproducible_evidence(
    tmp_path: Path,
) -> None:
    stock_path = tmp_path / "stock.json"
    stock_payload = _write_stock_cache(stock_path)
    audit_path = tmp_path / "audit.json"
    _write_audit_summary(audit_path)
    output_dir = tmp_path / "override"

    manifest = build_experiment_override_cache(
        stock_cache=stock_path,
        audit_summary=audit_path,
        output_dir=output_dir,
        native_builder=_native_builder,
    )

    candidate_path = output_dir / "cache" / "autotune_configs.json"
    candidate_payload = json.loads(candidate_path.read_text(encoding="ascii"))
    target = _cache_key(QWEN_BUCKET_128_SHAPES)
    assert set(candidate_payload) == set(stock_payload)
    assert candidate_payload["_metadata"] == stock_payload["_metadata"]
    assert candidate_payload[target] == [MOE_RUNNER, [32, 574]]
    assert sum(
        candidate_payload[key] != value for key, value in stock_payload.items()
    ) == 1

    semantic_diff = json.loads(
        (output_dir / "semantic_diff.json").read_text(encoding="ascii")
    )
    assert semantic_diff == {
        "changed_entries": 1,
        "key": target,
        "new_value": [MOE_RUNNER, [32, 574]],
        "old_value": [MOE_RUNNER, [16, 530]],
        "unchanged_entries": 3,
    }

    native_validation = json.loads(
        (output_dir / "native_validation.json").read_text(encoding="ascii")
    )
    assert native_validation["exact_hit_validated"] is True
    assert native_validation["fallback_miss_validated"] is True

    manifest_path = output_dir / "experiment_override_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="ascii"))
    assert manifest == manifest_payload
    assert manifest_payload["artifact_class"] == "UNQUALIFIED_EXPERIMENT_OVERRIDE"
    assert manifest_payload["production_eligible"] is False
    assert manifest_payload["bucket"] == 128
    assert manifest_payload["old_tactic"] == [16, 530]
    assert manifest_payload["new_tactic"] == [32, 574]
    assert manifest_payload["failed_gates"] == {
        "maximum_cv": {
            "limit": 0.03,
            "observed": 0.037384,
            "passed": False,
        },
        "minimum_weighted_gain": {
            "limit": 0.02,
            "observed": 0.012167304,
            "passed": False,
        },
    }
    assert manifest_payload["sha256"]["stock_cache"] == hashlib.sha256(
        stock_path.read_bytes()
    ).hexdigest()
    assert manifest_payload["sha256"]["candidate_cache"] == hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    assert manifest_payload["sha256"]["audit_summary"] == hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()

    for artifact in (
        candidate_path,
        output_dir / "semantic_diff.json",
        output_dir / "native_validation.json",
        manifest_path,
    ):
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o444


def test_rejects_existing_output_directory_without_invoking_native_builder(
    tmp_path: Path,
) -> None:
    stock_path = tmp_path / "stock.json"
    _write_stock_cache(stock_path)
    audit_path = tmp_path / "audit.json"
    _write_audit_summary(audit_path)
    output_dir = tmp_path / "override"
    output_dir.mkdir()
    invoked = False

    def forbidden_builder(request: Mapping[str, object]) -> Mapping[str, object]:
        nonlocal invoked
        invoked = True
        return {}

    with pytest.raises(FileExistsError, match="output directory already exists"):
        build_experiment_override_cache(
            stock_cache=stock_path,
            audit_summary=audit_path,
            output_dir=output_dir,
            native_builder=forbidden_builder,
        )

    assert not invoked


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("wrong_stock_tactic", "expected stock tactic"),
        ("missing_target", "exactly one bucket-128 Qwen"),
        ("duplicate_target", "exactly one bucket-128 Qwen"),
    ],
)
def test_rejects_ambiguous_or_unexpected_bucket_128_stock_entry(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    stock_path = tmp_path / "stock.json"
    payload = _write_stock_cache(
        stock_path,
        duplicate_target=mutation == "duplicate_target",
    )
    target = _cache_key(QWEN_BUCKET_128_SHAPES)
    if mutation == "wrong_stock_tactic":
        payload[target] = [MOE_RUNNER, [9, 9]]
    elif mutation == "missing_target":
        del payload[target]
    stock_path.write_text(json.dumps(payload, indent=2), encoding="ascii")
    audit_path = tmp_path / "audit.json"
    _write_audit_summary(audit_path)

    with pytest.raises(ValueError, match=error):
        build_experiment_override_cache(
            stock_cache=stock_path,
            audit_summary=audit_path,
            output_dir=tmp_path / "override",
            native_builder=_native_builder,
        )


@pytest.mark.parametrize("mutation", ["metadata", "keyset", "second_value"])
def test_rejects_native_builder_changes_outside_the_single_override(
    tmp_path: Path,
    mutation: str,
) -> None:
    stock_path = tmp_path / "stock.json"
    _write_stock_cache(stock_path)
    audit_path = tmp_path / "audit.json"
    _write_audit_summary(audit_path)

    def corrupting_builder(request: Mapping[str, object]) -> Mapping[str, object]:
        _native_builder(request)
        candidate_path = Path(str(request["candidate_path"]))
        payload = json.loads(candidate_path.read_text(encoding="ascii"))
        if mutation == "metadata":
            payload["_metadata"]["runtime_marker"] = "changed"
        elif mutation == "keyset":
            payload["unexpected"] = ["OtherRunner", 1]
        else:
            payload["cutedsl::dense-key"] = ["DenseRunner", 23]
        candidate_path.write_text(json.dumps(payload, indent=2), encoding="ascii")
        return {"exact_hit_validated": True, "fallback_miss_validated": True}

    with pytest.raises(RuntimeError, match="candidate cache must preserve"):
        build_experiment_override_cache(
            stock_cache=stock_path,
            audit_summary=audit_path,
            output_dir=tmp_path / "override",
            native_builder=corrupting_builder,
        )
