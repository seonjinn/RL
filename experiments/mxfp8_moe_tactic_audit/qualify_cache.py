"""Qualify MXFP8 MoE tactics and emit a standard FlashInfer cache artifact."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import MappingProxyType
from typing import Any, cast

if __package__:
    from .flashinfer_adapter import MOE_CUSTOM_OP, MOE_RUNNER
    from .schema import TacticMeasurement, TacticPair
else:  # pragma: no cover - direct script validation entry point
    from flashinfer_adapter import MOE_CUSTOM_OP, MOE_RUNNER
    from schema import TacticMeasurement, TacticPair


MANIFEST_SCHEMA_VERSION = 1
MIN_WEIGHTED_GAIN = 0.02
MAX_CV = 0.03
MAX_HIGH_WEIGHT_REGRESSION = 0.01
HIGH_WEIGHT_FRACTION = 0.05
MIN_MICRO_COSINE_SIMILARITY = 0.999
MAX_MXFP8_ABS_ERROR = 0.1
DEFAULT_CACHE_SUBPROCESS_TIMEOUT_SECONDS = 120.0

ARTIFACT_FINGERPRINT_FIELDS = frozenset(
    {
        "trace_set_sha256",
        "selected_profiles_sha256",
        "shmoo_results_sha256",
    }
)
RUNTIME_FINGERPRINT_FIELDS = frozenset(
    {
        "model_revision",
        "container_sha256",
        "vllm_commit",
        "flashinfer_version",
        "cuda_version",
        "gpu_name",
        "tp_size",
        "ep_size",
        "dp_size",
        "cuda_graph_mode",
    }
)
SOURCE_FINGERPRINT_FIELDS = ARTIFACT_FINGERPRINT_FIELDS | RUNTIME_FINGERPRINT_FIELDS


@dataclass(frozen=True)
class BucketAudit:
    """Aggregated stock-normalized qualification metrics for one cache bucket."""

    cache_key: str
    stock: TacticPair
    candidate: TacticPair
    weighted_gain: float
    max_cv: float
    worst_high_weight_regression: float
    all_correct: bool
    signature_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualificationDecision:
    """Selected tactic and promotion verdict for one exact FlashInfer key."""

    cache_key: str
    selected: TacticPair
    promoted: bool
    reason: str
    signature_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualificationInputs:
    """Typed public inputs for fail-closed pair-only cache qualification."""

    stock_cache: Path
    selected_profiles: Path
    shmoo_results: Path
    nsys_pairs: Path
    trace_summary: Path
    runtime_provenance: Path
    output_dir: Path


@dataclass(frozen=True)
class CacheProvenance:
    """Artifact inputs and runtime identity recorded in the cache manifest."""

    trace_paths: tuple[Path, ...]
    selected_profiles: Path
    shmoo_results: Path
    model_revision: str
    container_sha256: str
    vllm_commit: str
    flashinfer_version: str
    cuda_version: str
    gpu_name: str
    tp_size: int
    ep_size: int
    dp_size: int
    cuda_graph_mode: str

    def __post_init__(self) -> None:
        """Normalize paths and reject incomplete provenance."""
        object.__setattr__(self, "trace_paths", tuple(self.trace_paths))
        if not self.trace_paths:
            raise ValueError("trace_paths must not be empty")
        text_fields = (
            "model_revision",
            "vllm_commit",
            "flashinfer_version",
            "cuda_version",
            "gpu_name",
            "cuda_graph_mode",
        )
        for field_name in text_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a nonempty string")
        if len(self.container_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.container_sha256
        ):
            raise ValueError("container_sha256 must be a lowercase SHA256")
        for field_name in ("tp_size", "ep_size", "dp_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be positive")

    def runtime_fingerprints(self) -> Mapping[str, str]:
        """Return runtime fields used to accept or reject the candidate path."""
        return {
            "model_revision": self.model_revision,
            "container_sha256": self.container_sha256,
            "vllm_commit": self.vllm_commit,
            "flashinfer_version": self.flashinfer_version,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "tp_size": str(self.tp_size),
            "ep_size": str(self.ep_size),
            "dp_size": str(self.dp_size),
            "cuda_graph_mode": self.cuda_graph_mode,
        }

    def source_fingerprints(self) -> Mapping[str, str]:
        """Hash source artifacts and combine them with runtime provenance."""
        fingerprints = dict(self.runtime_fingerprints())
        fingerprints.update(
            {
                "trace_set_sha256": _sha256_file_set(self.trace_paths),
                "selected_profiles_sha256": _sha256_file(self.selected_profiles),
                "shmoo_results_sha256": _sha256_file(self.shmoo_results),
            }
        )
        return fingerprints


@dataclass(frozen=True)
class CacheManifest:
    """Versioned identity and entry counts for a candidate FlashInfer cache."""

    stock_sha256: str
    candidate_sha256: str
    source_fingerprints: Mapping[str, str]
    promoted_entries: int
    retained_entries: int

    def __post_init__(self) -> None:
        """Freeze and validate the complete manifest contract."""
        fingerprints = dict(self.source_fingerprints)
        if frozenset(fingerprints) != SOURCE_FINGERPRINT_FIELDS:
            missing = sorted(SOURCE_FINGERPRINT_FIELDS - frozenset(fingerprints))
            unexpected = sorted(frozenset(fingerprints) - SOURCE_FINGERPRINT_FIELDS)
            raise ValueError(
                f"invalid source fingerprint fields: missing={missing}, "
                f"unexpected={unexpected}"
            )
        for field_name, value in (
            ("stock_sha256", self.stock_sha256),
            ("candidate_sha256", self.candidate_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA256")
        for field_name in ARTIFACT_FINGERPRINT_FIELDS:
            value = fingerprints[field_name]
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA256")
        for field_name in RUNTIME_FINGERPRINT_FIELDS:
            if not fingerprints[field_name]:
                raise ValueError(f"{field_name} must be nonempty")
        for field_name, value in (
            ("promoted_entries", self.promoted_entries),
            ("retained_entries", self.retained_entries),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        object.__setattr__(
            self,
            "source_fingerprints",
            MappingProxyType(dict(sorted(fingerprints.items()))),
        )

    def to_json(self) -> dict[str, object]:
        """Serialize the stable versioned manifest payload."""
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "stock_sha256": self.stock_sha256,
            "candidate_sha256": self.candidate_sha256,
            "source_fingerprints": dict(self.source_fingerprints),
            "promoted_entries": self.promoted_entries,
            "retained_entries": self.retained_entries,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> CacheManifest:
        """Parse a manifest while rejecting missing or unexpected fields."""
        expected = {
            "schema_version",
            "stock_sha256",
            "candidate_sha256",
            "source_fingerprints",
            "promoted_entries",
            "retained_entries",
        }
        if set(payload) != expected:
            raise ValueError("invalid cache manifest fields")
        if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported cache manifest schema version")
        raw_fingerprints = payload["source_fingerprints"]
        if not isinstance(raw_fingerprints, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_fingerprints.items()
        ):
            raise ValueError("source_fingerprints must be a string mapping")
        stock_sha256 = payload["stock_sha256"]
        candidate_sha256 = payload["candidate_sha256"]
        promoted_entries = payload["promoted_entries"]
        retained_entries = payload["retained_entries"]
        if not isinstance(stock_sha256, str) or not isinstance(candidate_sha256, str):
            raise ValueError("cache SHA256 fields must be strings")
        if not isinstance(promoted_entries, int) or not isinstance(
            retained_entries, int
        ):
            raise ValueError("cache entry counts must be integers")
        return cls(
            stock_sha256=stock_sha256,
            candidate_sha256=candidate_sha256,
            source_fingerprints=cast(Mapping[str, str], raw_fingerprints),
            promoted_entries=promoted_entries,
            retained_entries=retained_entries,
        )


def _sha256_file(path: Path) -> str:
    """Return the SHA256 of one artifact without loading it all into memory."""
    digest = sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_set(paths: Sequence[Path]) -> str:
    """Return a path-independent SHA256 for a multiset of trace artifacts."""
    member_digests = sorted(_sha256_file(path) for path in paths)
    payload = json.dumps(member_digests, ensure_ascii=True, separators=(",", ":"))
    return sha256(payload.encode("ascii")).hexdigest()


def _measurement_is_correct(measurement: TacticMeasurement) -> bool:
    """Apply the Task 7 row-level finite, deterministic, and micro gate."""
    numeric_values = (
        measurement.median_us,
        measurement.p95_us,
        measurement.cv,
        measurement.max_abs_error,
        measurement.cosine_similarity,
    )
    return (
        measurement.failure is None
        and measurement.finite
        and measurement.deterministic
        and all(math.isfinite(value) for value in numeric_values)
        and measurement.warmups == 3
        and measurement.repetitions >= 10
        and measurement.median_us > 0
        and measurement.p95_us > 0
        and measurement.cv >= 0
        and 0 <= measurement.max_abs_error <= MAX_MXFP8_ABS_ERROR
        and measurement.cosine_similarity >= MIN_MICRO_COSINE_SIMILARITY
    )


def _weighted_median(values: Sequence[tuple[float, float]]) -> float:
    """Return the lower weighted median under deterministic gain ordering."""
    total_weight = math.fsum(weight for _, weight in values)
    midpoint = total_weight / 2
    cumulative = 0.0
    for value, weight in sorted(values, key=lambda item: item[0]):
        cumulative += weight
        if cumulative >= midpoint:
            return value
    raise ValueError("weighted median requires positive finite weights")


def audit_bucket(
    *,
    cache_key: str,
    stock: TacticPair,
    candidate: TacticPair,
    profile_weights: Mapping[str, float],
    measurements: Sequence[TacticMeasurement],
) -> BucketAudit:
    """Aggregate candidate performance against the stock row for each profile."""
    if not profile_weights:
        raise ValueError("profile_weights must not be empty")
    for signature_key, weight in profile_weights.items():
        if not signature_key:
            raise ValueError("profile signature keys must be nonempty")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("profile weights must be finite and positive")

    indexed: dict[tuple[str, TacticPair], TacticMeasurement] = {}
    for measurement in measurements:
        key = (measurement.signature_key, measurement.tactic)
        if key in indexed:
            raise ValueError(
                "duplicate shmoo row for profile and tactic: "
                f"{measurement.signature_key}/{measurement.tactic}"
            )
        indexed[key] = measurement

    rows: list[tuple[float, TacticMeasurement, TacticMeasurement]] = []
    all_correct = True
    for signature_key, weight in profile_weights.items():
        try:
            stock_row = indexed[(signature_key, stock)]
            candidate_row = indexed[(signature_key, candidate)]
        except KeyError as error:
            raise ValueError(
                f"missing stock or candidate shmoo row for profile {signature_key}"
            ) from error
        all_correct = all_correct and _measurement_is_correct(stock_row)
        all_correct = all_correct and _measurement_is_correct(candidate_row)
        rows.append((weight, stock_row, candidate_row))

    max_cv = max(candidate_row.cv for _, _, candidate_row in rows)
    if not all_correct:
        return BucketAudit(
            cache_key=cache_key,
            stock=stock,
            candidate=candidate,
            weighted_gain=0.0,
            max_cv=max_cv,
            worst_high_weight_regression=0.0,
            all_correct=False,
            signature_keys=tuple(sorted(profile_weights)),
        )

    total_weight = math.fsum(weight for weight, _, _ in rows)
    gains = [
        (
            (stock_row.median_us - candidate_row.median_us) / stock_row.median_us,
            weight,
        )
        for weight, stock_row, candidate_row in rows
    ]
    high_weight_regressions = [
        (candidate_row.median_us - stock_row.median_us) / stock_row.median_us
        for weight, stock_row, candidate_row in rows
        if weight / total_weight >= HIGH_WEIGHT_FRACTION
    ]
    return BucketAudit(
        cache_key=cache_key,
        stock=stock,
        candidate=candidate,
        weighted_gain=_weighted_median(gains),
        max_cv=max_cv,
        worst_high_weight_regression=(
            max(high_weight_regressions) if high_weight_regressions else 0.0
        ),
        all_correct=True,
        signature_keys=tuple(sorted(profile_weights)),
    )


def qualify_bucket(bucket: BucketAudit) -> QualificationDecision:
    """Promote only candidates that satisfy every binding Task 7 gate."""
    if not bucket.all_correct:
        reason = "candidate failed correctness checks"
    elif not all(
        math.isfinite(value)
        for value in (
            bucket.weighted_gain,
            bucket.max_cv,
            bucket.worst_high_weight_regression,
        )
    ):
        reason = "qualification metrics are not finite"
    elif bucket.weighted_gain < MIN_WEIGHTED_GAIN:
        reason = "weighted gain below 2%"
    elif bucket.max_cv > MAX_CV:
        reason = "coefficient of variation above 3%"
    elif bucket.worst_high_weight_regression > MAX_HIGH_WEIGHT_REGRESSION:
        reason = "high-weight regression above 1%"
    else:
        return QualificationDecision(
            cache_key=bucket.cache_key,
            selected=bucket.candidate,
            promoted=True,
            reason="candidate passed qualification gates",
            signature_keys=bucket.signature_keys,
        )
    return QualificationDecision(
        cache_key=bucket.cache_key,
        selected=bucket.stock,
        promoted=False,
        reason=reason,
        signature_keys=bucket.signature_keys,
    )


def _parse_moe_file_key(cache_key: str) -> tuple[tuple[int, ...], ...]:
    """Parse an exact pinned FlashInfer MoE file key and return its shapes."""
    try:
        parsed = ast.literal_eval(cache_key)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            "cache key must be an exact FlashInfer MoE file key"
        ) from error
    if (
        not isinstance(parsed, tuple)
        or len(parsed) != 4
        or parsed[0] != MOE_CUSTOM_OP
        or parsed[1] != MOE_RUNNER
        or not isinstance(parsed[2], tuple)
        or not all(
            isinstance(shape, tuple)
            and all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension >= 0
                for dimension in shape
            )
            for shape in parsed[2]
        )
        or parsed[3] != ()
    ):
        raise ValueError("cache key must be an exact FlashInfer MoE file key")
    return cast(tuple[tuple[int, ...], ...], parsed[2])


def _validate_moe_file_key(cache_key: str) -> None:
    """Reject keys outside the exact pinned FlashInfer MoE file-key shape."""
    _parse_moe_file_key(cache_key)


def _load_json_object(path: Path) -> dict[str, object]:
    """Read a JSON object with string keys."""
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, object], payload)


def _positive_number(value: object, field_name: str) -> float:
    """Parse one finite positive JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return number


def _selected_profile_contract(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    """Load selected profile weights and call counts without weakening the producer."""
    payload = _load_json_object(path)
    covered_weight = _positive_number(payload.get("covered_weight"), "covered_weight")
    if covered_weight < 0.95 or covered_weight > 1.0:
        raise ValueError("selected profile coverage must be in [0.95, 1]")
    raw_profiles = payload.get("selected_profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("selected_profiles must be a nonempty array")
    weights: dict[str, float] = {}
    call_counts: dict[str, int] = {}
    for index, row in enumerate(raw_profiles):
        if not isinstance(row, Mapping):
            raise ValueError(f"selected_profiles[{index}] must be an object")
        signature_key = row.get("signature_key")
        call_count = row.get("call_count")
        if not isinstance(signature_key, str) or not signature_key:
            raise ValueError(f"selected_profiles[{index}] has no signature_key")
        if signature_key in weights:
            raise ValueError(f"duplicate selected signature {signature_key}")
        if (
            isinstance(call_count, bool)
            or not isinstance(call_count, int)
            or call_count <= 0
        ):
            raise ValueError(f"selected_profiles[{index}] has invalid call_count")
        weights[signature_key] = _positive_number(
            row.get("normalized_weight"),
            f"selected_profiles[{index}].normalized_weight",
        )
        call_counts[signature_key] = call_count
    if not math.isclose(
        math.fsum(weights.values()), covered_weight, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("selected profile weights do not match covered_weight")
    return weights, call_counts


def _trace_contract(
    path: Path, profile_weights: Mapping[str, float]
) -> tuple[dict[str, str], tuple[Path, ...]]:
    """Load exact signature/cache-key bindings and raw trace provenance."""
    payload = _load_json_object(path)
    raw_paths = payload.get("trace_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("trace_summary.trace_paths must be a nonempty array")
    trace_paths: list[Path] = []
    for index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"trace_paths[{index}] must be a nonempty string")
        trace_path = (path.parent / raw_path).resolve()
        if not trace_path.is_file():
            raise ValueError(f"trace path does not exist: {trace_path}")
        trace_paths.append(trace_path)
    if len(set(trace_paths)) != len(trace_paths):
        raise ValueError("trace_summary contains duplicate trace paths")

    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("trace_summary.profiles must be a nonempty array")
    bindings: dict[str, str] = {}
    for index, row in enumerate(raw_profiles):
        if not isinstance(row, Mapping):
            raise ValueError(f"trace profile {index} must be an object")
        signature_key = row.get("signature_key")
        cache_key = row.get("cache_key")
        if not isinstance(signature_key, str) or not signature_key:
            raise ValueError(f"trace profile {index} has no signature_key")
        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError(f"trace profile {index} has no cache_key")
        if signature_key in bindings:
            raise ValueError(f"duplicate trace signature {signature_key}")
        _validate_moe_file_key(cache_key)
        expected_weight = profile_weights.get(signature_key)
        if expected_weight is None or not math.isclose(
            _positive_number(row.get("call_weight"), "trace call_weight"),
            expected_weight,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("trace mapping does not bind selected profile weights")
        bindings[signature_key] = cache_key
    if set(bindings) != set(profile_weights):
        raise ValueError("trace and selected profile signatures differ")
    return bindings, tuple(trace_paths)


def _csv_tactic(value: object, field_name: str) -> TacticPair:
    """Parse one comma-delimited pair from the NSys consumer schema."""
    if not isinstance(value, str):
        raise ValueError(f"NSys {field_name} must be a tactic pair")
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"NSys {field_name} must be a tactic pair")
    try:
        return TacticPair(int(parts[0]), int(parts[1]))
    except ValueError as error:
        raise ValueError(f"NSys {field_name} must be a tactic pair") from error


def _nsys_pair_contract(
    path: Path,
    *,
    profile_bindings: Mapping[str, str],
    call_counts: Mapping[str, int],
) -> dict[str, set[TacticPair]]:
    """Validate pair-only NSys rows and return measured candidate pairs."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("NSys pair CSV has no header")
            required = {
                "signature_key",
                "cache_key",
                "arm",
                "component",
                "tactic",
                "comparison_tactic",
                "cache_event",
                "call_weight",
                "call_count",
                "mean_us",
            }
            if set(reader.fieldnames) != required:
                raise ValueError("NSys pair CSV fields do not match the typed contract")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"cannot read NSys pair CSV {path}: {error}") from error
    if not rows:
        raise ValueError("NSys pair CSV must not be empty")

    observed_bindings: dict[str, str] = {}
    candidate_tactics = {signature: set() for signature in profile_bindings}
    paired_arms: dict[
        tuple[str, TacticPair], dict[str, tuple[TacticPair, int, int]]
    ] = {}
    for index, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"NSys pair CSV row {index} is malformed")
        signature_key = row["signature_key"]
        cache_key = row["cache_key"]
        if signature_key not in profile_bindings:
            raise ValueError(f"NSys row has unselected signature {signature_key}")
        _validate_moe_file_key(cache_key)
        previous = observed_bindings.setdefault(signature_key, cache_key)
        if previous != cache_key or profile_bindings[signature_key] != cache_key:
            raise ValueError("NSys and trace signature/cache-key mappings disagree")
        if row["component"] != "FC1+FC2/GEMM1+GEMM2":
            raise ValueError("NSys qualification input must contain pair-only timings")
        if row["arm"] not in {"stock", "candidate"}:
            raise ValueError("NSys pair CSV has invalid arm")
        if row["cache_event"].strip().lower() not in {"cache hit", "fallback"}:
            raise ValueError("NSys pair CSV has invalid cache_event")
        tactic = _csv_tactic(row["tactic"], "tactic")
        comparison_tactic = _csv_tactic(row["comparison_tactic"], "comparison_tactic")
        if row["arm"] == "candidate":
            if tactic != comparison_tactic:
                raise ValueError(
                    "candidate NSys tactic does not match comparison_tactic"
                )
            candidate_tactics[signature_key].add(tactic)
        try:
            call_weight = int(row["call_weight"])
            call_count = int(row["call_count"])
            mean_us = float(row["mean_us"])
        except ValueError as error:
            raise ValueError(
                f"NSys pair CSV row {index} has invalid numerics"
            ) from error
        if (
            call_weight != call_counts[signature_key]
            or call_count <= 0
            or not math.isfinite(mean_us)
            or mean_us <= 0
        ):
            raise ValueError(f"NSys pair CSV row {index} has unbound timing evidence")
        arm_rows = paired_arms.setdefault((signature_key, comparison_tactic), {})
        if row["arm"] in arm_rows:
            raise ValueError("NSys pair CSV contains a duplicate comparison arm")
        arm_rows[row["arm"]] = (tactic, call_weight, call_count)
    for (signature_key, comparison_tactic), arm_rows in paired_arms.items():
        if set(arm_rows) != {"stock", "candidate"}:
            raise ValueError(
                "NSys pair CSV requires matching stock and candidate arms for "
                f"{signature_key}/{comparison_tactic}"
            )
        _, stock_weight, stock_count = arm_rows["stock"]
        candidate_tactic, candidate_weight, candidate_count = arm_rows["candidate"]
        if candidate_tactic != comparison_tactic or (
            stock_weight,
            stock_count,
        ) != (candidate_weight, candidate_count):
            raise ValueError("NSys stock and candidate comparison arms do not match")
    if set(observed_bindings) != set(profile_bindings) or any(
        not tactics for tactics in candidate_tactics.values()
    ):
        raise ValueError("NSys pair CSV does not cover every selected signature")
    return candidate_tactics


def _load_measurements(path: Path) -> tuple[TacticMeasurement, ...]:
    """Parse exact shmoo JSONL rows and reject duplicates."""
    measurements: list[TacticMeasurement] = []
    seen: set[tuple[str, TacticPair]] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read shmoo results {path}: {error}") from error
    if not lines:
        raise ValueError("shmoo results must not be empty")
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"blank shmoo row at line {line_number}")
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"shmoo row {line_number} must be an object")
        measurement = TacticMeasurement.from_json(cast(Mapping[str, object], raw))
        key = (measurement.signature_key, measurement.tactic)
        if key in seen:
            raise ValueError(f"duplicate shmoo row at line {line_number}")
        seen.add(key)
        measurements.append(measurement)
    return tuple(measurements)


def _runtime_provenance(path: Path) -> dict[str, object]:
    """Parse the exact typed runtime provenance object used by the public CLI."""
    payload = _load_json_object(path)
    if set(payload) != RUNTIME_FINGERPRINT_FIELDS:
        raise ValueError("runtime provenance fields do not match the typed contract")
    return payload


def _get_autotuner() -> Any:
    """Load the optional pinned FlashInfer AutoTuner only on cache operations."""
    autotuner = importlib.import_module("flashinfer.autotuner")
    return autotuner.AutoTuner.get()


def _run_cache_subprocess(
    stock_path: Path,
    candidate_path: Path,
    promoted: Mapping[str, TacticPair],
    retained: Mapping[str, TacticPair],
    absent_key: str,
    timeout_seconds: float,
) -> None:
    """Build and validate the standard cache without touching the parent tuner."""
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((str(repository_root), existing_pythonpath))
        if existing_pythonpath
        else str(repository_root)
    )
    request = {
        "stock_path": str(stock_path),
        "candidate_path": str(candidate_path),
        "promoted": {key: tactic.to_json() for key, tactic in promoted.items()},
        "retained": {key: tactic.to_json() for key, tactic in retained.items()},
        "absent_key": absent_key,
    }
    manifest_path = candidate_path.with_name("cache_manifest.json")
    candidate_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "experiments.mxfp8_moe_tactic_audit.qualify_cache",
        "--build-and-validate-cache",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=repository_root,
            env=environment,
            input=json.dumps(request, ensure_ascii=True),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        candidate_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"cache subprocess timed out after {timeout_seconds:g} seconds "
            "during FlashInfer load/modify/save/lookup"
        ) from error
    if result.returncode != 0:
        candidate_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"cache subprocess failed: {details}")


def _cache_tactic(value: object, cache_key: str) -> TacticPair:
    """Parse one serialized AutoTuner runner/tactic value."""
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
        or value[0] != MOE_RUNNER
        or not isinstance(value[1], Sequence)
        or isinstance(value[1], (str, bytes, bytearray))
    ):
        raise ValueError(f"invalid MoERunner cache value for {cache_key}")
    tactic_values = value[1]
    if len(tactic_values) != 2:
        raise ValueError(f"invalid MoERunner tactic pair for {cache_key}")
    return TacticPair(gemm1=tactic_values[0], gemm2=tactic_values[1])  # type: ignore[arg-type]


def _absent_key_from(source_key: str, existing_keys: set[str]) -> str:
    """Derive a valid exact MoE shape key that is absent from the stock cache."""
    profile_shapes = _parse_moe_file_key(source_key)
    for shape_index, shape in enumerate(profile_shapes):
        if not shape:
            continue
        for offset in range(1, 1_000_001):
            changed_shape = (shape[0] + offset, *shape[1:])
            changed_shapes = (
                *profile_shapes[:shape_index],
                changed_shape,
                *profile_shapes[shape_index + 1 :],
            )
            cache_key = str((MOE_CUSTOM_OP, MOE_RUNNER, changed_shapes, ()))
            if cache_key not in existing_keys:
                return cache_key
    raise ValueError("could not derive an absent FlashInfer MoE cache key")


def _write_manifest(path: Path, manifest: CacheManifest) -> None:
    """Write the deterministic ASCII cache manifest."""
    path.write_text(
        json.dumps(manifest.to_json(), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="ascii",
    )


def _write_qualification_decisions(
    path: Path,
    *,
    manifest: CacheManifest,
    decisions: Sequence[QualificationDecision],
    stock: Mapping[str, object],
    nsys_pairs: Path | None,
) -> None:
    """Emit the authoritative cache-build decisions beside the manifest."""
    rows: list[dict[str, object]] = []
    for decision in sorted(decisions, key=lambda item: item.cache_key):
        rows.append(
            {
                "cache_key": decision.cache_key,
                "promoted": decision.promoted,
                "reason": decision.reason,
                "selected": decision.selected.to_json(),
                "signature_keys": list(decision.signature_keys),
                "stock": _cache_tactic(
                    stock[decision.cache_key], decision.cache_key
                ).to_json(),
            }
        )
    payload = {
        "cache_manifest_sha256": _sha256_file(path.with_name("cache_manifest.json")),
        "decisions": rows,
        "selected_profiles_sha256": manifest.source_fingerprints[
            "selected_profiles_sha256"
        ],
        "shmoo_results_sha256": manifest.source_fingerprints["shmoo_results_sha256"],
        "trace_set_sha256": manifest.source_fingerprints["trace_set_sha256"],
    }
    if nsys_pairs is not None:
        payload["nsys_pairs_sha256"] = _sha256_file(nsys_pairs)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def select_qualification_decisions(
    *,
    stock: Mapping[str, TacticPair],
    profile_bindings: Mapping[str, str],
    profile_weights: Mapping[str, float],
    measurements: Sequence[TacticMeasurement],
    nsys_candidate_tactics: Mapping[str, set[TacticPair]],
) -> tuple[QualificationDecision, ...]:
    """Choose the strongest fully measured tactic that passes every gate."""
    signatures_by_key: dict[str, list[str]] = {}
    for signature_key, cache_key in profile_bindings.items():
        signatures_by_key.setdefault(cache_key, []).append(signature_key)

    measured_by_signature: dict[str, set[TacticPair]] = {}
    for measurement in measurements:
        if measurement.signature_key in profile_bindings:
            measured_by_signature.setdefault(measurement.signature_key, set()).add(
                measurement.tactic
            )

    decisions: list[QualificationDecision] = []
    for cache_key, raw_signatures in sorted(signatures_by_key.items()):
        signatures = tuple(sorted(raw_signatures))
        stock_tactic = stock.get(cache_key)
        if stock_tactic is None:
            raise ValueError(
                f"selected cache key is absent from stock cache: {cache_key}"
            )
        tactic_sets: list[set[TacticPair]] = []
        for signature in signatures:
            measured = measured_by_signature.get(signature, set())
            nsys_measured = nsys_candidate_tactics.get(signature, set())
            tactic_sets.append(measured & nsys_measured)
        complete_tactics = set.intersection(*tactic_sets) if tactic_sets else set()
        if stock_tactic not in complete_tactics:
            raise ValueError(
                f"stock tactic lacks complete shmoo/NSys evidence for {cache_key}"
            )

        weights = {signature: profile_weights[signature] for signature in signatures}
        eligible: list[tuple[BucketAudit, QualificationDecision]] = []
        for tactic in sorted(
            complete_tactics - {stock_tactic},
            key=lambda item: (item.gemm1, item.gemm2),
        ):
            audit = audit_bucket(
                cache_key=cache_key,
                stock=stock_tactic,
                candidate=tactic,
                profile_weights=weights,
                measurements=measurements,
            )
            decision = qualify_bucket(audit)
            if decision.promoted:
                eligible.append((audit, decision))
        if eligible:
            _, selected = max(
                eligible,
                key=lambda item: (
                    item[0].weighted_gain,
                    -item[0].max_cv,
                    -item[0].worst_high_weight_regression,
                    -item[0].candidate.gemm1,
                    -item[0].candidate.gemm2,
                ),
            )
            decisions.append(selected)
        else:
            decisions.append(
                QualificationDecision(
                    cache_key=cache_key,
                    selected=stock_tactic,
                    promoted=False,
                    reason="no fully measured candidate passed qualification gates",
                    signature_keys=signatures,
                )
            )
    if not decisions:
        raise ValueError("no selected cache keys were available for qualification")
    return tuple(decisions)


def qualify_and_build_cache(inputs: QualificationInputs) -> CacheManifest:
    """Validate public artifacts, qualify exact keys, and build the candidate."""
    profile_weights, call_counts = _selected_profile_contract(inputs.selected_profiles)
    profile_bindings, trace_paths = _trace_contract(
        inputs.trace_summary, profile_weights
    )
    nsys_candidate_tactics = _nsys_pair_contract(
        inputs.nsys_pairs,
        profile_bindings=profile_bindings,
        call_counts=call_counts,
    )
    measurements = _load_measurements(inputs.shmoo_results)
    stock_payload = _load_json_object(inputs.stock_cache)
    stock = {
        cache_key: _cache_tactic(value, cache_key)
        for cache_key, value in stock_payload.items()
        if cache_key != "_metadata" and cache_key in set(profile_bindings.values())
    }
    decisions = select_qualification_decisions(
        stock=stock,
        profile_bindings=profile_bindings,
        profile_weights=profile_weights,
        measurements=measurements,
        nsys_candidate_tactics=nsys_candidate_tactics,
    )
    runtime = _runtime_provenance(inputs.runtime_provenance)
    provenance = CacheProvenance(
        trace_paths=trace_paths,
        selected_profiles=inputs.selected_profiles,
        shmoo_results=inputs.shmoo_results,
        model_revision=cast(str, runtime["model_revision"]),
        container_sha256=cast(str, runtime["container_sha256"]),
        vllm_commit=cast(str, runtime["vllm_commit"]),
        flashinfer_version=cast(str, runtime["flashinfer_version"]),
        cuda_version=cast(str, runtime["cuda_version"]),
        gpu_name=cast(str, runtime["gpu_name"]),
        tp_size=cast(int, runtime["tp_size"]),
        ep_size=cast(int, runtime["ep_size"]),
        dp_size=cast(int, runtime["dp_size"]),
        cuda_graph_mode=cast(str, runtime["cuda_graph_mode"]),
    )
    return build_candidate_cache(
        inputs.stock_cache,
        decisions,
        inputs.output_dir,
        provenance=provenance,
        nsys_pairs=inputs.nsys_pairs,
    )


def build_candidate_cache(
    stock_cache: Path,
    decisions: Sequence[QualificationDecision],
    output: Path,
    *,
    provenance: CacheProvenance,
    nsys_pairs: Path | None = None,
    subprocess_timeout_seconds: float = DEFAULT_CACHE_SUBPROCESS_TIMEOUT_SECONDS,
) -> CacheManifest:
    """Replace promoted exact MoE keys through FlashInfer's native cache APIs."""
    if (
        isinstance(subprocess_timeout_seconds, bool)
        or not isinstance(subprocess_timeout_seconds, (int, float))
        or not math.isfinite(subprocess_timeout_seconds)
        or subprocess_timeout_seconds <= 0
    ):
        raise ValueError("subprocess_timeout_seconds must be finite and positive")
    stock_cache = stock_cache.resolve()
    output = output.resolve()
    candidate_path = output / "autotune_configs.json"
    if candidate_path == stock_cache:
        raise ValueError("candidate cache must not overwrite the stock cache")

    decision_by_key: dict[str, QualificationDecision] = {}
    for decision in decisions:
        if decision.cache_key in decision_by_key:
            raise ValueError(
                f"duplicate qualification decision for {decision.cache_key}"
            )
        decision_by_key[decision.cache_key] = decision
    promoted = {
        decision.cache_key: decision.selected
        for decision in decisions
        if decision.promoted
    }
    for cache_key in promoted:
        _validate_moe_file_key(cache_key)

    stock_payload = _load_json_object(stock_cache)
    stock_entries = {key for key in stock_payload if key != "_metadata"}
    missing_promoted = sorted(set(promoted) - stock_entries)
    if missing_promoted:
        raise ValueError(
            f"promoted key is not present in stock cache: {missing_promoted[0]}"
        )

    exact_moe_keys: list[str] = []
    retained: dict[str, TacticPair] = {}
    for cache_key, value in stock_payload.items():
        if cache_key == "_metadata":
            continue
        try:
            _validate_moe_file_key(cache_key)
        except ValueError:
            continue
        exact_moe_keys.append(cache_key)
        if cache_key in promoted or retained:
            continue
        retained[cache_key] = _cache_tactic(value, cache_key)
    if not exact_moe_keys:
        raise ValueError(
            "stock cache must contain an exact FlashInfer MoE key for validation"
        )
    absent_key = _absent_key_from(exact_moe_keys[0], stock_entries)

    output.mkdir(parents=True, exist_ok=True)
    _run_cache_subprocess(
        stock_path=stock_cache,
        candidate_path=candidate_path,
        promoted=promoted,
        retained=retained,
        absent_key=absent_key,
        timeout_seconds=float(subprocess_timeout_seconds),
    )

    candidate_payload = _load_json_object(candidate_path)
    if set(candidate_payload) != set(stock_payload):
        raise RuntimeError("candidate cache changed the stock JSON object keys")
    for key, stock_value in stock_payload.items():
        if key not in promoted and candidate_payload[key] != stock_value:
            raise RuntimeError(f"candidate cache changed retained entry {key}")
    for key, tactic in promoted.items():
        if candidate_payload[key] != [MOE_RUNNER, [tactic.gemm1, tactic.gemm2]]:
            raise RuntimeError(f"candidate cache serialized the wrong tactic for {key}")

    manifest = CacheManifest(
        stock_sha256=_sha256_file(stock_cache),
        candidate_sha256=_sha256_file(candidate_path),
        source_fingerprints=provenance.source_fingerprints(),
        promoted_entries=len(promoted),
        retained_entries=len(stock_entries) - len(promoted),
    )
    _write_manifest(output / "cache_manifest.json", manifest)
    _write_qualification_decisions(
        output / "qualification_decisions.json",
        manifest=manifest,
        decisions=decisions,
        stock=stock_payload,
        nsys_pairs=nsys_pairs,
    )
    return manifest


def select_cache_path(
    *,
    stock_path: Path,
    candidate_path: Path,
    manifest_path: Path,
    runtime_fingerprints: Mapping[str, str],
) -> Path:
    """Select the candidate only when its hash and every runtime field match."""
    try:
        manifest = CacheManifest.from_json(_load_json_object(manifest_path))
        runtime_matches = all(
            runtime_fingerprints.get(field) == manifest.source_fingerprints[field]
            for field in RUNTIME_FINGERPRINT_FIELDS
        )
        candidate_matches = _sha256_file(candidate_path) == manifest.candidate_sha256
        stock_matches = _sha256_file(stock_path) == manifest.stock_sha256
    except (OSError, ValueError, json.JSONDecodeError):
        return stock_path
    if runtime_matches and candidate_matches and stock_matches:
        return candidate_path
    return stock_path


def _expected_tactics(
    request: Mapping[str, object], field_name: str
) -> dict[str, TacticPair]:
    """Parse one exact-key tactic mapping from the child request."""
    raw_expected = request.get(field_name)
    if not isinstance(raw_expected, Mapping):
        raise ValueError(f"{field_name} must be an object")
    expected: dict[str, TacticPair] = {}
    for cache_key, value in raw_expected.items():
        if not isinstance(cache_key, str) or not isinstance(value, Mapping):
            raise ValueError(f"{field_name} contains an invalid entry")
        _validate_moe_file_key(cache_key)
        expected[cache_key] = TacticPair.from_json(cast(Mapping[str, object], value))
    return expected


def _lookup_tactic(tuner: Any, autotuner: Any, cache_key: str) -> tuple[bool, object]:
    """Exercise pinned search_cache and choose_one for one exact file key."""
    # FlashInfer and torch are optional outside native cache construction.
    torch = importlib.import_module("torch")
    profile_shapes = _parse_moe_file_key(cache_key)

    def forward(
        self: object,
        inputs: list[object],
        tactic: object = -1,
        do_preparation: bool = False,
        **kwargs: object,
    ) -> None:
        raise RuntimeError("lookup validation unexpectedly invoked MoERunner.forward")

    def get_valid_tactics(
        self: object, inputs: list[object], profile: object
    ) -> list[int]:
        return [-1]

    runner_type = type(
        MOE_RUNNER,
        (autotuner.TunableRunner,),
        {
            "forward": forward,
            "get_valid_tactics": get_valid_tactics,
            "__module__": __name__,
        },
    )
    runner = runner_type()
    inputs = [torch.empty(shape, device="meta") for shape in profile_shapes]
    input_shapes = tuple(value.size() for value in inputs)
    tuning_config = autotuner.TuningConfig()
    hit, runner_id, searched_tactic, _ = tuner.search_cache(
        MOE_CUSTOM_OP,
        [runner],
        input_shapes,
        tuning_config,
        inputs=inputs,
    )
    if runner_id != 0:
        raise RuntimeError(f"lookup returned invalid runner index {runner_id}")
    selected_runner, selected_tactic = tuner.choose_one(
        MOE_CUSTOM_OP,
        [runner],
        tuning_config,
        inputs,
    )
    if selected_runner is not runner or selected_tactic != searched_tactic:
        raise RuntimeError(
            "search_cache and choose_one returned inconsistent lookup results"
        )
    return bool(hit), selected_tactic


def _build_and_validate_cache_from_stdin() -> int:
    """Build and validate a candidate entirely in this short-lived process."""
    request = json.loads(sys.stdin.read())
    if not isinstance(request, Mapping):
        raise ValueError("cache subprocess request must be an object")
    stock_path_value = request.get("stock_path")
    candidate_path_value = request.get("candidate_path")
    absent_key = request.get("absent_key")
    if not all(
        isinstance(value, str) and value
        for value in (stock_path_value, candidate_path_value, absent_key)
    ):
        raise ValueError("cache subprocess paths and absent_key must be strings")
    stock_path = Path(cast(str, stock_path_value))
    candidate_path = Path(cast(str, candidate_path_value))
    absent_key = cast(str, absent_key)
    _validate_moe_file_key(absent_key)
    promoted = _expected_tactics(request, "promoted")
    retained = _expected_tactics(request, "retained")

    autotuner = importlib.import_module("flashinfer.autotuner")
    tuner = _get_autotuner()
    if tuner.load_configs(str(stock_path)) is False:
        raise RuntimeError("stock cache metadata does not match the child runtime")
    for cache_key, tactic in promoted.items():
        loaded = tuner._file_configs.get(cache_key)
        if loaded is None or loaded[0] != MOE_RUNNER:
            raise RuntimeError(
                f"promoted key is unavailable after stock load: {cache_key}"
            )
        tuner._file_configs[cache_key] = (
            MOE_RUNNER,
            (tactic.gemm1, tactic.gemm2),
        )
    shutil.copyfile(stock_path, candidate_path)
    tuner.save_configs(str(candidate_path))

    tuner._file_configs.clear()
    tuner.profiling_cache.clear()
    if tuner.load_configs(str(candidate_path)) is False:
        raise RuntimeError("candidate cache metadata does not match the child runtime")
    for expected in (promoted, retained):
        for cache_key, tactic in expected.items():
            hit, selected_tactic = _lookup_tactic(tuner, autotuner, cache_key)
            expected_tactic = (tactic.gemm1, tactic.gemm2)
            if not hit or selected_tactic != expected_tactic:
                raise RuntimeError(
                    f"cache lookup for {cache_key} returned hit={hit}, "
                    f"tactic={selected_tactic!r}; expected {expected_tactic!r}"
                )
    miss, heuristic_tactic = _lookup_tactic(tuner, autotuner, absent_key)
    if miss or heuristic_tactic != -1:
        raise RuntimeError(
            f"absent key returned hit={miss}, tactic={heuristic_tactic!r}; "
            "expected a heuristic miss"
        )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse public qualification inputs or the private child-process mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-and-validate-cache", action="store_true")
    parser.add_argument("--stock-cache", type=Path)
    parser.add_argument("--selected-profiles", type=Path)
    parser.add_argument("--shmoo-results", type=Path)
    parser.add_argument("--nsys-pairs", type=Path)
    parser.add_argument("--trace-summary", type=Path)
    parser.add_argument("--runtime-provenance", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run public qualification or private fresh-process cache validation."""
    args = _parse_args(argv)
    if args.build_and_validate_cache:
        return _build_and_validate_cache_from_stdin()
    public_fields = (
        "stock_cache",
        "selected_profiles",
        "shmoo_results",
        "nsys_pairs",
        "trace_summary",
        "runtime_provenance",
        "output_dir",
    )
    missing = [name for name in public_fields if getattr(args, name) is None]
    if missing:
        print(
            "qualification error: missing required arguments: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing),
            file=sys.stderr,
        )
        return 2
    try:
        qualify_and_build_cache(
            QualificationInputs(
                stock_cache=args.stock_cache,
                selected_profiles=args.selected_profiles,
                shmoo_results=args.shmoo_results,
                nsys_pairs=args.nsys_pairs,
                trace_summary=args.trace_summary,
                runtime_provenance=args.runtime_provenance,
                output_dir=args.output_dir,
            )
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"qualification error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
