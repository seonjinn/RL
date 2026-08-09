"""Build a reproducible, explicitly unqualified FC1/FC2 cache override."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import cast


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
DEFAULT_STOCK_TACTIC = (16, 530)
DEFAULT_OVERRIDE_TACTIC = (32, 574)

NativeBuilder = Callable[[Mapping[str, object]], Mapping[str, object]]


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise ValueError(f"{path} must contain a JSON object with string keys")
    return cast(dict[str, object], payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _parse_exact_moe_key(cache_key: str) -> tuple[tuple[int, ...], ...] | None:
    try:
        parsed = ast.literal_eval(cache_key)
    except (SyntaxError, ValueError):
        return None
    if (
        not isinstance(parsed, tuple)
        or len(parsed) != 4
        or parsed[0] != MOE_OP
        or parsed[1] != MOE_RUNNER
        or parsed[3] != ()
        or not isinstance(parsed[2], tuple)
    ):
        return None
    shapes: list[tuple[int, ...]] = []
    for shape in parsed[2]:
        if (
            not isinstance(shape, tuple)
            or not all(isinstance(dimension, int) for dimension in shape)
        ):
            return None
        shapes.append(shape)
    return tuple(shapes)


def _parse_tactic(value: object, cache_key: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or value[0] != MOE_RUNNER
        or not isinstance(value[1], list)
        or len(value[1]) != 2
        or not all(isinstance(item, int) for item in value[1])
    ):
        raise ValueError(f"invalid MoERunner cache value for {cache_key}")
    return cast(tuple[int, int], tuple(value[1]))


def _target_key(
    stock: Mapping[str, object], expected_stock_tactic: tuple[int, int]
) -> str:
    matches = [
        key
        for key in stock
        if key != "_metadata" and _parse_exact_moe_key(key) == QWEN_BUCKET_128_SHAPES
    ]
    if len(matches) != 1:
        raise ValueError(
            "stock cache must contain exactly one bucket-128 Qwen MoERunner key"
        )
    target = matches[0]
    actual = _parse_tactic(stock[target], target)
    if actual != expected_stock_tactic:
        raise ValueError(
            f"expected stock tactic {expected_stock_tactic}, found {actual}"
        )
    return target


def _retained_tactics(
    stock: Mapping[str, object], target: str
) -> dict[str, dict[str, int]]:
    retained: dict[str, dict[str, int]] = {}
    for cache_key, value in stock.items():
        if cache_key == target or _parse_exact_moe_key(cache_key) is None:
            continue
        gemm1, gemm2 = _parse_tactic(value, cache_key)
        retained[cache_key] = {"gemm1": gemm1, "gemm2": gemm2}
    return retained


def _absent_key(stock: Mapping[str, object]) -> str:
    existing = set(stock)
    for bucket in range(129, 1_000_129):
        shapes = ((bucket, 2048), *QWEN_BUCKET_128_SHAPES[1:])
        cache_key = str((MOE_OP, MOE_RUNNER, shapes, ()))
        if cache_key not in existing:
            return cache_key
    raise ValueError("could not derive an absent exact MoERunner cache key")


def _default_native_builder(request: Mapping[str, object]) -> Mapping[str, object]:
    repository_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "experiments.mxfp8_moe_tactic_audit.qualify_cache",
        "--build-and-validate-cache",
    ]
    result = subprocess.run(
        command,
        cwd=repository_root,
        input=json.dumps(request, ensure_ascii=True),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"FlashInfer native cache validation failed: {details}")
    return {
        "exact_hit_validated": True,
        "fallback_miss_validated": True,
        "method": "qualify_cache --build-and-validate-cache",
    }


def _candidate_metrics(
    audit: Mapping[str, object], override_tactic: tuple[int, int]
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    gates = audit.get("gates")
    comparison = audit.get("same_run_comparison")
    if not isinstance(gates, Mapping) or not isinstance(comparison, Mapping):
        raise ValueError("audit summary is missing gates or same_run_comparison")
    candidates = comparison.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("audit summary candidates must be a list")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("pair") == list(override_tactic)
    ]
    if len(matches) != 1:
        raise ValueError("audit summary must contain exactly one override tactic row")
    return gates, cast(Mapping[str, object], matches[0])


def _failed_gates(
    gates: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    minimum_gain = gates.get("minimum_weighted_gain")
    maximum_cv = gates.get("maximum_cv")
    observed_gain = candidate.get("weighted_median_gain")
    observed_cv = candidate.get("maximum_cv")
    values = (minimum_gain, maximum_cv, observed_gain, observed_cv)
    if not all(isinstance(value, (int, float)) for value in values):
        raise ValueError("audit summary contains invalid gate metrics")
    failed = {
        "maximum_cv": {
            "limit": maximum_cv,
            "observed": observed_cv,
            "passed": cast(float, observed_cv) <= cast(float, maximum_cv),
        },
        "minimum_weighted_gain": {
            "limit": minimum_gain,
            "observed": observed_gain,
            "passed": cast(float, observed_gain) >= cast(float, minimum_gain),
        },
    }
    failed = {
        name: result
        for name, result in failed.items()
        if not cast(Mapping[str, object], result)["passed"]
    }
    if not failed:
        raise ValueError("override tactic does not fail any recorded qualification gate")
    return failed


def _validate_candidate(
    stock: Mapping[str, object],
    candidate: Mapping[str, object],
    target: str,
    override_tactic: tuple[int, int],
) -> dict[str, object]:
    if set(candidate) != set(stock) or candidate.get("_metadata") != stock.get(
        "_metadata"
    ):
        raise RuntimeError("candidate cache must preserve metadata and the stock keyset")
    changed = [key for key, value in stock.items() if candidate[key] != value]
    expected_value = [MOE_RUNNER, list(override_tactic)]
    if changed != [target] or candidate[target] != expected_value:
        raise RuntimeError(
            "candidate cache must preserve every entry except the single override"
        )
    return {
        "changed_entries": 1,
        "key": target,
        "new_value": expected_value,
        "old_value": stock[target],
        "unchanged_entries": len(stock) - 1,
    }


def build_experiment_override_cache(
    *,
    stock_cache: Path,
    audit_summary: Path,
    output_dir: Path,
    expected_stock_tactic: tuple[int, int] = DEFAULT_STOCK_TACTIC,
    override_tactic: tuple[int, int] = DEFAULT_OVERRIDE_TACTIC,
    native_builder: NativeBuilder | None = None,
) -> dict[str, object]:
    """Build one read-only experiment cache without weakening promotion gates."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    stock = _load_json_object(stock_cache)
    audit = _load_json_object(audit_summary)
    target = _target_key(stock, expected_stock_tactic)
    gates, candidate_metrics = _candidate_metrics(audit, override_tactic)
    failed_gates = _failed_gates(gates, candidate_metrics)
    request: dict[str, object] = {
        "stock_path": str(stock_cache.resolve()),
        "candidate_path": str(
            (output_dir / "cache" / "autotune_configs.json").resolve()
        ),
        "promoted": {
            target: {"gemm1": override_tactic[0], "gemm2": override_tactic[1]}
        },
        "retained": _retained_tactics(stock, target),
        "absent_key": _absent_key(stock),
    }

    try:
        candidate_path = output_dir / "cache" / "autotune_configs.json"
        candidate_path.parent.mkdir(parents=True)
        validation = dict((native_builder or _default_native_builder)(request))
        if validation.get("exact_hit_validated") is not True or validation.get(
            "fallback_miss_validated"
        ) is not True:
            raise RuntimeError("native builder did not validate exact hit and fallback")
        candidate = _load_json_object(candidate_path)
        semantic_diff = _validate_candidate(
            stock, candidate, target, override_tactic
        )

        semantic_path = output_dir / "semantic_diff.json"
        native_path = output_dir / "native_validation.json"
        manifest_path = output_dir / "experiment_override_manifest.json"
        _write_json(semantic_path, semantic_diff)
        _write_json(native_path, validation)
        manifest: dict[str, object] = {
            "artifact_class": "UNQUALIFIED_EXPERIMENT_OVERRIDE",
            "bucket": 128,
            "cache_key": target,
            "failed_gates": failed_gates,
            "new_tactic": list(override_tactic),
            "old_tactic": list(expected_stock_tactic),
            "production_eligible": False,
            "runtime_context": audit.get("measurement_context", {}),
            "schema_version": 1,
            "sha256": {
                "audit_summary": _sha256_file(audit_summary),
                "candidate_cache": _sha256_file(candidate_path),
                "metadata": _sha256_json(stock.get("_metadata")),
                "native_validation": _sha256_file(native_path),
                "semantic_diff": _sha256_file(semantic_path),
                "stock_cache": _sha256_file(stock_cache),
            },
            "source_artifacts": audit.get("source_artifacts", {}),
            "unchanged_entries": len(stock) - 1,
        }
        _write_json(manifest_path, manifest)
        for artifact in (candidate_path, semantic_path, native_path, manifest_path):
            artifact.chmod(0o444)
        return manifest
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def _tactic(value: str) -> tuple[int, int]:
    try:
        gemm1, gemm2 = value.split(",", maxsplit=1)
        return int(gemm1), int(gemm2)
    except ValueError as error:
        raise argparse.ArgumentTypeError("tactic must be GEMM1,GEMM2") from error


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-cache", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-stock-tactic", type=_tactic, default=DEFAULT_STOCK_TACTIC
    )
    parser.add_argument(
        "--override-tactic", type=_tactic, default=DEFAULT_OVERRIDE_TACTIC
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the experiment-only cache override from CLI arguments."""
    args = _parse_args(argv)
    build_experiment_override_cache(
        stock_cache=args.stock_cache,
        audit_summary=args.audit_summary,
        output_dir=args.output_dir,
        expected_stock_tactic=args.expected_stock_tactic,
        override_tactic=args.override_tactic,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
