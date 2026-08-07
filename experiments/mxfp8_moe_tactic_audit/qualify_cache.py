"""Qualify MXFP8 MoE tactics and emit a standard FlashInfer cache artifact."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
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
        "container",
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
class CacheProvenance:
    """Artifact inputs and runtime identity recorded in the cache manifest."""

    trace_paths: tuple[Path, ...]
    selected_profiles: Path
    shmoo_results: Path
    model_revision: str
    container: str
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
            "container",
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
        for field_name in ("tp_size", "ep_size", "dp_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be positive")

    def runtime_fingerprints(self) -> Mapping[str, str]:
        """Return runtime fields used to accept or reject the candidate path."""
        return {
            "model_revision": self.model_revision,
            "container": self.container,
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
    return (
        measurement.failure is None
        and measurement.finite
        and measurement.deterministic
        and measurement.median_us > 0
        and measurement.cv >= 0
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
                "stock": _cache_tactic(stock[decision.cache_key], decision.cache_key).to_json(),
            }
        )
    payload = {
        "cache_manifest_sha256": _sha256_file(path.with_name("cache_manifest.json")),
        "decisions": rows,
        "selected_profiles_sha256": manifest.source_fingerprints["selected_profiles_sha256"],
        "shmoo_results_sha256": manifest.source_fingerprints["shmoo_results_sha256"],
        "trace_set_sha256": manifest.source_fingerprints["trace_set_sha256"],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def build_candidate_cache(
    stock_cache: Path,
    decisions: Sequence[QualificationDecision],
    output: Path,
    *,
    provenance: CacheProvenance,
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
    """Parse the private fresh-process validation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-and-validate-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the private fresh-process validation mode."""
    args = _parse_args(argv)
    if not args.build_and_validate_cache:
        raise SystemExit("--build-and-validate-cache is required")
    return _build_and_validate_cache_from_stdin()


if __name__ == "__main__":
    raise SystemExit(main())
