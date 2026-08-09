"""Build reviewed, provenance-bound MXFP8 MoE tactic-audit reports."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Sequence

from .collect_results import (
    COMPARABILITY_METADATA_FIELDS,
    EvidenceError,
    RunSummary,
    compare_manifests,
    find_driver_log,
    load_json_object,
    load_jsonl,
    sha256_file,
    summarize_run,
)
from .compare_gsm8k import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    exact_mcnemar_p_value,
    paired_outcome_bootstrap_ci,
    paired_outcomes_sha256,
)
from .plot_results import PLOT_NAMES, write_complete_plots, write_unavailable_plots
from .qualify_cache import (
    RUNTIME_FINGERPRINT_FIELDS,
    CacheManifest,
)


REPORT_BASENAME = "mxfp8_moe_tactic_audit_latest"
CORRECTNESS_GATES = (
    "micro_correctness",
    "cuda_graph_replay",
    "deterministic_generation",
)
PAIR_COMPONENT = "FC1+FC2/GEMM1+GEMM2"
STAGE_COMPONENTS = ("FC1/GEMM1", "FC2/GEMM2")


@dataclass(frozen=True)
class AuditInputs:
    """Every execution, tactic, component, and provenance source for an audit."""

    stock_runs: tuple[Path, ...]
    candidate_runs: tuple[Path, ...]
    cache_manifest: Path
    stock_cache: Path
    candidate_cache: Path
    trace_summary: Path
    qualification_decisions: Path
    selected_profiles: Path
    shmoo: Path
    nsys: Path
    correctness: Path
    gsm8k: Path


@dataclass(frozen=True)
class AuditReport:
    """Rendered state and the decisive evidence messages."""

    verdict: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _Collected:
    stock_runs: tuple[RunSummary, ...]
    candidate_runs: tuple[RunSummary, ...]
    component_speedups: tuple[tuple[str, float], ...]
    component_distribution: tuple[tuple[str, float], ...]
    tactic_change_share: float
    cache_hit_share: float
    source_hashes: tuple[tuple[str, str], ...]
    cache_manifest_bindings: tuple[tuple[str, str], ...]
    metadata_caption: str
    failed_gates: tuple[str, ...]
    covered_weight: float
    gsm8k: dict[str, object]


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or positive and number <= 0:
        raise EvidenceError(
            f"{label} must be finite" + (" and positive" if positive else "")
        )
    return number


def _tactic(value: object, label: str) -> tuple[int, int]:
    if isinstance(value, dict):
        first, second = value.get("gemm1"), value.get("gemm2")
    elif isinstance(value, list) and len(value) == 2:
        first, second = value
    else:
        raise EvidenceError(f"{label} must be a GEMM1/GEMM2 pair")
    if (
        isinstance(first, bool)
        or not isinstance(first, int)
        or first < 0
        or isinstance(second, bool)
        or not isinstance(second, int)
        or second < 0
    ):
        raise EvidenceError(f"{label} must contain nonnegative integer tactic IDs")
    return (first, second)


def _cache(path: Path) -> dict[str, tuple[int, int]]:
    payload = load_json_object(path)
    entries: dict[str, tuple[int, int]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or key == "_metadata":
            continue
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[1], list)
        ):
            raise EvidenceError(f"invalid cache entry {key}")
        entries[key] = _tactic(value[1], f"cache entry {key}")
    if not entries:
        raise EvidenceError(f"{path} has no tactic entries")
    return entries


def _trace_set_sha256(trace_summary: Path) -> str:
    """Match qualify_cache's path-independent raw trace-set digest exactly."""
    raw_paths = load_json_object(trace_summary).get("trace_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise EvidenceError("trace summary must list raw trace_paths")
    paths: list[Path] = []
    for index, value in enumerate(raw_paths, start=1):
        if not isinstance(value, str) or not value:
            raise EvidenceError("trace summary has invalid trace path")
        path = (trace_summary.parent / value).resolve()
        if not path.is_file():
            raise EvidenceError(f"trace summary raw trace member {index} is missing")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise EvidenceError("trace summary duplicates a raw trace path")
    member_digests = sorted(sha256_file(path) for path in paths)
    payload = json.dumps(member_digests, ensure_ascii=True, separators=(",", ":"))
    return sha256(payload.encode("ascii")).hexdigest()


def _profiles(trace_path: Path, selected_path: Path) -> dict[str, tuple[str, float]]:
    trace = load_json_object(trace_path).get("profiles")
    selected = load_json_object(selected_path)
    if not isinstance(trace, list) or not trace:
        raise EvidenceError("trace summary must contain profiles")
    if _number(selected.get("covered_weight"), "covered_weight") < 0.95:
        raise EvidenceError("trace/profile coverage is below 95%")
    selected_rows = selected.get("selected_profiles")
    if not isinstance(selected_rows, list):
        raise EvidenceError("selected profiles are missing")
    selected_weights: dict[str, float] = {}
    for row in selected_rows:
        if not isinstance(row, dict) or not isinstance(row.get("signature_key"), str):
            raise EvidenceError("selected profile is malformed")
        selected_weights[row["signature_key"]] = _number(
            row.get("normalized_weight"), "normalized_weight", positive=True
        )
    result: dict[str, tuple[str, float]] = {}
    for row in trace:
        if not isinstance(row, dict):
            raise EvidenceError("trace profile is malformed")
        signature, cache_key = row.get("signature_key"), row.get("cache_key")
        if (
            not isinstance(signature, str)
            or not isinstance(cache_key, str)
            or signature not in selected_weights
        ):
            raise EvidenceError(
                "trace summary does not bind selected signature to cache key"
            )
        if signature in result:
            raise EvidenceError(f"duplicate trace signature {signature}")
        result[signature] = (
            cache_key,
            _number(row.get("call_weight"), "call_weight", positive=True),
        )
    if set(result) != set(selected_weights):
        raise EvidenceError("trace and selected profile signatures differ")
    return result


def _validate_provenance(inputs: AuditInputs) -> tuple[tuple[str, str], ...]:
    manifest = load_json_object(inputs.cache_manifest)
    try:
        parsed_manifest = CacheManifest.from_json(manifest)
    except ValueError as error:
        raise EvidenceError(f"invalid cache manifest: {error}") from error
    if manifest.get("stock_sha256") != sha256_file(inputs.stock_cache) or manifest.get(
        "candidate_sha256"
    ) != sha256_file(inputs.candidate_cache):
        raise EvidenceError(
            "cache manifest does not bind supplied stock/candidate cache files"
        )
    fingerprints = parsed_manifest.source_fingerprints
    expected = {
        "trace_set_sha256": _trace_set_sha256(inputs.trace_summary),
        "selected_profiles_sha256": sha256_file(inputs.selected_profiles),
        "shmoo_results_sha256": sha256_file(inputs.shmoo),
    }
    if any(fingerprints.get(key) != value for key, value in expected.items()):
        raise EvidenceError(
            "cache manifest source fingerprints do not bind supplied artifacts"
        )
    decisions = load_json_object(inputs.qualification_decisions)
    decision_expected = {
        "cache_manifest_sha256": sha256_file(inputs.cache_manifest),
        "nsys_pairs_sha256": sha256_file(inputs.nsys),
        "trace_set_sha256": _trace_set_sha256(inputs.trace_summary),
        "selected_profiles_sha256": sha256_file(inputs.selected_profiles),
        "shmoo_results_sha256": sha256_file(inputs.shmoo),
    }
    if any(decisions.get(key) != value for key, value in decision_expected.items()):
        raise EvidenceError(
            "qualification decisions do not bind supplied source artifacts"
        )
    return tuple(
        (name, value)
        for name, value in (
            ("stock_sha256", manifest["stock_sha256"]),
            ("candidate_sha256", manifest["candidate_sha256"]),
            *sorted(fingerprints.items()),
            *(
                (f"qualification.{key}", value)
                for key, value in sorted(decision_expected.items())
            ),
        )
        if isinstance(name, str) and isinstance(value, str)
    )


def _decision_tactics(
    inputs: AuditInputs, profiles: dict[str, tuple[str, float]]
) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    stock, candidate = _cache(inputs.stock_cache), _cache(inputs.candidate_cache)
    payload = load_json_object(inputs.qualification_decisions)
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise EvidenceError("qualification decisions are missing")
    decisions: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for row in raw_decisions:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("cache_key"), str)
            or not isinstance(row.get("promoted"), bool)
        ):
            raise EvidenceError("qualification decision is malformed")
        key = row["cache_key"]
        if key in decisions or key not in stock or key not in candidate:
            raise EvidenceError("qualification/cache keys are incomplete or duplicate")
        selected = _tactic(row.get("selected"), f"decision {key}")
        signatures = row.get("signature_keys")
        if (
            not isinstance(signatures, list)
            or not signatures
            or not all(isinstance(value, str) and value for value in signatures)
        ):
            raise EvidenceError(f"decision {key} has no producer signature bindings")
        if any(
            profiles.get(signature, (None, 0.0))[0] != key for signature in signatures
        ):
            raise EvidenceError(
                f"decision {key} signatures do not bind selected profiles"
            )
        recorded_stock = _tactic(row.get("stock"), f"decision stock {key}")
        if recorded_stock != stock[key]:
            raise EvidenceError(
                f"decision {key} does not bind to the stock cache tactic"
            )
        expected_candidate = candidate[key] if row["promoted"] else stock[key]
        if selected != expected_candidate:
            raise EvidenceError(
                f"decision {key} does not bind to the cache-selected tactic"
            )
        decisions[key] = (stock[key], expected_candidate)
    needed = {cache_key for cache_key, _ in profiles.values()}
    if not needed <= set(decisions):
        raise EvidenceError(
            "qualification decisions do not cover every trace cache key"
        )
    return decisions


def _successful_shmoo(
    inputs: AuditInputs,
    profiles: dict[str, tuple[str, float]],
    tactics: dict[str, tuple[tuple[int, int], tuple[int, int]]],
) -> float:
    rows: dict[tuple[str, tuple[int, int]], float] = {}
    for row in load_jsonl(inputs.shmoo):
        signature = row.get("signature_key")
        if not isinstance(signature, str) or signature not in profiles:
            continue
        tactic = _tactic(row.get("tactic"), "shmoo tactic")
        if (
            row.get("finite") is not True
            or row.get("deterministic") is not True
            or row.get("failure") is not None
        ):
            continue
        median = _number(row.get("median_us"), "shmoo median_us", positive=True)
        key = (signature, tactic)
        if key in rows:
            raise EvidenceError("duplicate successful shmoo row")
        rows[key] = median
    changed_weight = 0.0
    total_weight = 0.0
    for signature, (cache_key, weight) in profiles.items():
        stock, candidate = tactics[cache_key]
        if (signature, stock) not in rows or (signature, candidate) not in rows:
            raise EvidenceError(
                "selected stock/promoted tactic has no successful shmoo row"
            )
        if stock != candidate:
            changed_weight += weight
        total_weight += weight
    return changed_weight / total_weight


def _component_speedups(
    inputs: AuditInputs,
    profiles: dict[str, tuple[str, float]],
    tactics: dict[str, tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...], float]:
    try:
        rows = list(
            csv.DictReader(inputs.nsys.read_text(encoding="utf-8").splitlines())
        )
    except OSError as error:
        raise EvidenceError(f"cannot read NSys component CSV: {error}") from error
    selected_rows = load_json_object(inputs.selected_profiles).get("selected_profiles")
    if not isinstance(selected_rows, list):
        raise EvidenceError("selected profiles are missing")
    expected_call_weights: dict[str, int] = {}
    for selected in selected_rows:
        if not isinstance(selected, dict) or not isinstance(
            selected.get("signature_key"), str
        ):
            raise EvidenceError("selected profile is malformed")
        call_weight = selected.get("call_count")
        if (
            isinstance(call_weight, bool)
            or not isinstance(call_weight, int)
            or call_weight <= 0
        ):
            raise EvidenceError(
                "selected profile call_count must be a positive integer"
            )
        expected_call_weights[selected["signature_key"]] = call_weight
    indexed: dict[
        tuple[str, str, str, str, tuple[int, int], tuple[int, int]],
        tuple[float, int, str],
    ] = {}
    for row in rows:
        event = row.get("cache_event", "").strip().lower()
        if event not in {"cache hit", "fallback"}:
            raise EvidenceError("NSys component row has invalid cache event")
        signature = row.get("signature_key")
        cache_key = row.get("cache_key")
        arm = row.get("arm")
        component = row.get("component")
        if not (
            isinstance(signature, str)
            and signature
            and isinstance(cache_key, str)
            and cache_key
            and isinstance(arm, str)
            and arm
            and isinstance(component, str)
            and component
        ):
            raise EvidenceError("NSys component row is malformed")
        if component not in {*STAGE_COMPONENTS, PAIR_COMPONENT} or arm not in {
            "stock",
            "candidate",
        }:
            raise EvidenceError("NSys component row has invalid arm or component")
        try:
            tactic = tuple(int(item) for item in row.get("tactic", "").split(","))
            comparison_tactic = tuple(
                int(item) for item in row.get("comparison_tactic", "").split(",")
            )
        except ValueError as error:
            raise EvidenceError("NSys component row has invalid tactic") from error
        if len(tactic) != 2 or len(comparison_tactic) != 2:
            raise EvidenceError("NSys component row has invalid tactic")
        try:
            count_value = float(row.get("call_count", "nan"))
            timing_value = float(row.get("mean_us", "nan"))
        except (TypeError, ValueError) as error:
            raise EvidenceError(
                "NSys component row has malformed numeric field"
            ) from error
        count_number = _number(count_value, "NSys call_count", positive=True)
        if not count_number.is_integer():
            raise EvidenceError("NSys call_count must be an integer")
        count = int(count_number)
        try:
            call_weight_value = float(row.get("call_weight", "nan"))
        except (TypeError, ValueError) as error:
            raise EvidenceError(
                "NSys component row has malformed numeric field"
            ) from error
        call_weight_number = _number(
            call_weight_value, "NSys call_weight", positive=True
        )
        if not call_weight_number.is_integer():
            raise EvidenceError("NSys call_weight must be an integer")
        if int(call_weight_number) != expected_call_weights.get(signature):
            raise EvidenceError("NSys call_weight does not bind selected profile")
        timing = _number(timing_value, "NSys mean_us", positive=True)
        key = (
            signature,
            cache_key,
            arm,
            component,
            (tactic[0], tactic[1]),
            (comparison_tactic[0], comparison_tactic[1]),
        )
        if key in indexed:
            raise EvidenceError("duplicate NSys component row")
        indexed[key] = (timing, count, event)
    observed_components = {key[3] for key in indexed}
    if PAIR_COMPONENT in observed_components and len(observed_components) != 1:
        raise EvidenceError(
            "NSys component evidence contains mixed pair and stage timings"
        )
    if observed_components == {PAIR_COMPONENT}:
        report_components = (PAIR_COMPONENT,)
    elif observed_components == set(STAGE_COMPONENTS):
        report_components = STAGE_COMPONENTS
    else:
        raise EvidenceError("NSys component evidence is incomplete")

    results: list[tuple[str, float]] = []
    distribution: list[tuple[str, float]] = []
    hit_weight = fallback_weight = 0.0
    for component in report_components:
        weighted = total = 0.0
        for signature, (cache_key, trace_weight) in profiles.items():
            stock, candidate = tactics[cache_key]
            stock_row = indexed.get(
                (signature, cache_key, "stock", component, stock, candidate)
            )
            candidate_row = indexed.get(
                (signature, cache_key, "candidate", component, candidate, candidate)
            )
            if stock_row is None or candidate_row is None:
                raise EvidenceError(
                    f"NSys component evidence missing {component} for selected tactic"
                )
            stock_time, stock_calls, stock_event = stock_row
            candidate_time, candidate_calls, candidate_event = candidate_row
            if stock_calls != candidate_calls:
                raise EvidenceError("NSys stock/candidate component call counts differ")
            weight = trace_weight
            speedup = stock_time / candidate_time
            weighted += weight * speedup
            total += weight
            distribution.append((f"{component}\n{signature[:12]}", speedup))
            for event in (stock_event, candidate_event):
                if event == "cache hit":
                    hit_weight += weight
                else:
                    fallback_weight += weight
        results.append((component, weighted / total))
    if hit_weight + fallback_weight == 0:
        raise EvidenceError("NSys CSV has no explicit cache hit/fallback rows")
    return (
        tuple(results),
        tuple(distribution),
        hit_weight / (hit_weight + fallback_weight),
    )


def _run_summaries(
    inputs: AuditInputs,
) -> tuple[tuple[RunSummary, ...], tuple[RunSummary, ...]]:
    if not inputs.stock_runs or not inputs.candidate_runs:
        raise EvidenceError("stock and candidate run sets are required")
    paths = (*inputs.stock_runs, *inputs.candidate_runs)
    if len({path.resolve() for path in paths}) != len(paths):
        raise EvidenceError("duplicate run paths cannot satisfy repetition evidence")
    stock = tuple(summarize_run(path) for path in inputs.stock_runs)
    candidate = tuple(summarize_run(path) for path in inputs.candidate_runs)
    if len(stock) != len(candidate):
        raise EvidenceError(
            "stock and candidate run counts differ; runs are not comparable"
        )
    if any(not run.all_metrics_finite for run in (*stock, *candidate)):
        raise EvidenceError("run metrics are not finite")
    if any(run.arm != "stock" for run in stock) or any(
        run.arm != "candidate" for run in candidate
    ):
        raise EvidenceError("run evidence arm does not match supplied arm")
    if len({run.run_id for run in stock}) != len(stock) or len(
        {run.run_id for run in candidate}
    ) != len(candidate):
        raise EvidenceError("run IDs must be unique within each arm")
    all_paths = (("stock", inputs.stock_runs), ("candidate", inputs.candidate_runs))
    for arm, arm_paths in all_paths:
        anchor = arm_paths[0] / "run_manifest.json"
        for path in arm_paths[1:]:
            differences = compare_manifests(anchor, path / "run_manifest.json")
            if differences:
                raise EvidenceError(
                    f"{arm} repetition manifests differ: " + ", ".join(differences)
                )
    for index, (left, right) in enumerate(zip(stock, candidate, strict=True)):
        if left.run_id != right.run_id:
            raise EvidenceError("stock/candidate logical comparison IDs differ")
        differences = compare_manifests(
            inputs.stock_runs[index] / "run_manifest.json",
            inputs.candidate_runs[index] / "run_manifest.json",
        )
        if differences:
            raise EvidenceError(
                "stock/candidate manifests differ: " + ", ".join(differences)
            )
    metadata_anchor = stock[0].metadata
    for run in (*stock, *candidate):
        for field in COMPARABILITY_METADATA_FIELDS:
            if run.metadata.get(field) != metadata_anchor.get(field):
                raise EvidenceError(f"run repetition {field} metadata differ")
    for run, path in zip((*stock, *candidate), paths, strict=True):
        manifest_run_kind = load_json_object(path / "run_manifest.json").get("run_kind")
        if run.metadata.get("run_kind") != manifest_run_kind:
            raise EvidenceError("run metadata does not bind manifest run kind")
    for arm, runs, paths_for_arm in (
        ("stock", stock, inputs.stock_runs),
        ("candidate", candidate, inputs.candidate_runs),
    ):
        cache_hashes = {
            load_json_object(path / "run_manifest.json").get("cache_sha256")
            for path in paths_for_arm
        }
        if len(cache_hashes) != 1 or not all(
            isinstance(value, str) and value for value in cache_hashes
        ):
            raise EvidenceError(f"{arm} repetitions do not share one cache identity")
    stock_cache = load_json_object(inputs.stock_runs[0] / "run_manifest.json").get(
        "cache_sha256"
    )
    candidate_cache = load_json_object(
        inputs.candidate_runs[0] / "run_manifest.json"
    ).get("cache_sha256")
    if stock_cache == candidate_cache:
        raise EvidenceError(
            "stock and candidate arms must use distinct cache identities"
        )
    return stock, candidate


def _source_hashes(inputs: AuditInputs) -> tuple[tuple[str, str], ...]:
    paths = (
        ("cache manifest", inputs.cache_manifest),
        ("stock cache", inputs.stock_cache),
        ("candidate cache", inputs.candidate_cache),
        ("trace summary", inputs.trace_summary),
        ("qualification decisions", inputs.qualification_decisions),
        ("selected profiles", inputs.selected_profiles),
        ("shmoo JSONL", inputs.shmoo),
        ("NSys components", inputs.nsys),
        ("correctness", inputs.correctness),
        ("GSM8K", inputs.gsm8k),
    )
    run_paths = tuple(
        (f"{arm} {path.name} {artifact}", file)
        for arm, runs in (
            ("stock", inputs.stock_runs),
            ("candidate", inputs.candidate_runs),
        )
        for path in runs
        for artifact, file in (
            ("driver", find_driver_log(path)),
            ("evidence", path / "run_evidence.json"),
            ("manifest", path / "run_manifest.json"),
        )
    )
    raw_trace_paths = load_json_object(inputs.trace_summary).get("trace_paths")
    if not isinstance(raw_trace_paths, list):
        raise EvidenceError("trace summary must list raw trace_paths")
    trace_paths = tuple(
        (
            f"raw trace member {index + 1}",
            (inputs.trace_summary.parent / value).resolve(),
        )
        for index, value in enumerate(raw_trace_paths)
        if isinstance(value, str)
    )
    if len(trace_paths) != len(raw_trace_paths):
        raise EvidenceError("trace summary has invalid trace path")
    return tuple(
        (label, sha256_file(path)) for label, path in (*paths, *trace_paths, *run_paths)
    )


def _collect(inputs: AuditInputs) -> _Collected:
    cache_manifest_bindings = _validate_provenance(inputs)
    profiles = _profiles(inputs.trace_summary, inputs.selected_profiles)
    tactics = _decision_tactics(inputs, profiles)
    tactic_share = _successful_shmoo(inputs, profiles, tactics)
    components, component_distribution, cache_hit_share = _component_speedups(
        inputs, profiles, tactics
    )
    stock, candidate = _run_summaries(inputs)
    correctness, gsm8k = (
        load_json_object(inputs.correctness),
        load_json_object(inputs.gsm8k),
    )
    manifest = CacheManifest.from_json(load_json_object(inputs.cache_manifest))
    expected_runtime = {
        field: manifest.source_fingerprints[field]
        for field in RUNTIME_FINGERPRINT_FIELDS
    }
    for run, path in (
        *zip(stock, inputs.stock_runs, strict=True),
        *zip(candidate, inputs.candidate_runs, strict=True),
    ):
        required_runtime = set(RUNTIME_FINGERPRINT_FIELDS) | {
            "cache_file_sha256",
            "cache_sha256",
            "container_sha256",
            "nemo_rl_commit",
        }
        missing_runtime = required_runtime - set(run.runtime_fingerprints)
        if missing_runtime:
            raise EvidenceError(
                "runtime evidence is missing independently observed fields: "
                + ", ".join(sorted(missing_runtime))
            )
        if any(
            run.runtime_fingerprints.get(name) != value
            for name, value in expected_runtime.items()
        ):
            raise EvidenceError(
                "independently observed runtime fingerprints do not match cache manifest"
            )
        run_manifest = load_json_object(path / "run_manifest.json")
        manifest_cache = run_manifest.get("cache_sha256")
        expected_cache = (
            manifest.stock_sha256 if run.arm == "stock" else manifest.candidate_sha256
        )
        if manifest_cache != expected_cache:
            raise EvidenceError("run manifest cache hash does not bind its audit arm")
        if run.runtime_fingerprints.get("cache_sha256") != manifest_cache:
            raise EvidenceError(
                "observed runtime cache hash does not match the run manifest"
            )
        if run.runtime_fingerprints.get("cache_file_sha256") != manifest_cache:
            raise EvidenceError(
                "observed runtime cache file hash does not match the run manifest"
            )
        if run.runtime_fingerprints.get("container_sha256") != run_manifest.get(
            "container_sha256"
        ):
            raise EvidenceError(
                "observed runtime container hash does not match the run manifest"
            )
        if run.runtime_fingerprints.get("nemo_rl_commit") != run_manifest.get(
            "nemo_rl_commit"
        ):
            raise EvidenceError(
                "observed runtime NeMo-RL checkout does not match run manifest"
            )
        if run.runtime_fingerprints.get("vllm_commit") != run_manifest.get(
            "vllm_commit"
        ):
            raise EvidenceError(
                "observed runtime vLLM checkout does not match run manifest"
            )
    required_gsm8k = (
        "provenance_matched",
        "matched_examples",
        "stock_accuracy",
        "candidate_accuracy",
        "candidate_only_wins",
        "stock_only_wins",
        "both_correct",
        "both_wrong",
        "accuracy_delta",
        "mcnemar_p_value",
        "delta_ci95",
        "bootstrap_seed",
        "bootstrap_samples",
        "paired_outcomes",
        "paired_outcomes_sha256",
        "passed",
    )
    if any(name not in gsm8k for name in required_gsm8k):
        raise EvidenceError("GSM8K comparison is missing paired evidence fields")
    if (
        gsm8k.get("provenance_matched") is not True
        or gsm8k.get("matched_examples") != 1319
    ):
        raise EvidenceError(
            "GSM8K comparison provenance or matched example count is invalid"
        )
    count_fields = (
        "candidate_only_wins",
        "stock_only_wins",
        "both_correct",
        "both_wrong",
    )
    counts: list[int] = []
    for name in count_fields:
        value = gsm8k[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EvidenceError("GSM8K paired counts are invalid")
        counts.append(value)
    if sum(counts) != 1319:
        raise EvidenceError("GSM8K paired counts are invalid")
    delta_ci95 = gsm8k.get("delta_ci95")
    if not isinstance(delta_ci95, list) or len(delta_ci95) != 2:
        raise EvidenceError("GSM8K delta_ci95 is invalid")
    for name in (
        "stock_accuracy",
        "candidate_accuracy",
        "accuracy_delta",
        "mcnemar_p_value",
    ):
        _number(gsm8k[name], f"GSM8K {name}")
    for value in delta_ci95:
        _number(value, "GSM8K delta_ci95")
    stock_accuracy = (counts[2] + counts[1]) / 1319
    candidate_accuracy = (counts[2] + counts[0]) / 1319
    accuracy_delta = candidate_accuracy - stock_accuracy
    if (
        not math.isclose(
            _number(gsm8k["stock_accuracy"], "GSM8K stock_accuracy"), stock_accuracy
        )
        or not math.isclose(
            _number(gsm8k["candidate_accuracy"], "GSM8K candidate_accuracy"),
            candidate_accuracy,
        )
        or not math.isclose(
            _number(gsm8k["accuracy_delta"], "GSM8K accuracy_delta"), accuracy_delta
        )
    ):
        raise EvidenceError("GSM8K accuracies or delta disagree with paired cells")
    p_value = _number(gsm8k["mcnemar_p_value"], "GSM8K mcnemar_p_value")
    if not 0 <= p_value <= 1:
        raise EvidenceError("GSM8K mcnemar_p_value must be in [0, 1]")
    expected_p_value = exact_mcnemar_p_value(counts[0], counts[1])
    if not math.isclose(p_value, expected_p_value, rel_tol=0.0, abs_tol=1e-12):
        raise EvidenceError("GSM8K McNemar p-value disagrees with paired cells")
    lower = _number(delta_ci95[0], "GSM8K delta_ci95")
    upper = _number(delta_ci95[1], "GSM8K delta_ci95")
    if not -1 <= lower <= upper <= 1 or not lower <= accuracy_delta <= upper:
        raise EvidenceError("GSM8K delta_ci95 bounds are inconsistent")
    outcomes = gsm8k.get("paired_outcomes")
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != 1319
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in {-1, 0, 1}
            for value in outcomes
        )
    ):
        raise EvidenceError("GSM8K paired outcomes are invalid")
    paired_outcomes = tuple(outcomes)
    if (
        paired_outcomes.count(1) != counts[0]
        or paired_outcomes.count(-1) != counts[1]
        or paired_outcomes_sha256(paired_outcomes)
        != gsm8k.get("paired_outcomes_sha256")
    ):
        raise EvidenceError("GSM8K paired outcomes disagree with paired cells")
    if (
        gsm8k.get("bootstrap_seed") != BOOTSTRAP_SEED
        or gsm8k.get("bootstrap_samples") != BOOTSTRAP_SAMPLES
    ):
        raise EvidenceError("GSM8K bootstrap contract is invalid")
    expected_ci = paired_outcome_bootstrap_ci(
        paired_outcomes, BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES
    )
    if not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip((lower, upper), expected_ci, strict=True)
    ):
        raise EvidenceError("GSM8K delta_ci95 was not reproduced from paired outcomes")
    expected_passed = p_value >= 0.05 and lower <= 0 <= upper
    if gsm8k.get("passed") is not expected_passed:
        raise EvidenceError("GSM8K passed field disagrees with paired statistics")
    expected_hashes = {
        arm: {
            run.run_id: sha256_file(path / "run_manifest.json")
            for run, path in zip(runs, paths, strict=True)
        }
        for arm, runs, paths in (
            ("stock", stock, inputs.stock_runs),
            ("candidate", candidate, inputs.candidate_runs),
        )
    }
    for label, payload in (("correctness", correctness), ("GSM8K", gsm8k)):
        comparison_ids = payload.get("comparison_run_ids")
        if (
            not isinstance(comparison_ids, list)
            or len(comparison_ids) != 1
            or not isinstance(comparison_ids[0], str)
        ):
            raise EvidenceError(f"{label} must bind exactly one evaluated run pair")
        run_id = comparison_ids[0]
        if (
            payload.get("stock_arm_id") != "stock"
            or payload.get("candidate_arm_id") != "candidate"
            or payload.get("stock_run_manifests")
            != {run_id: expected_hashes["stock"].get(run_id)}
            or payload.get("candidate_run_manifests")
            != {run_id: expected_hashes["candidate"].get(run_id)}
            or run_id not in expected_hashes["stock"]
            or run_id not in expected_hashes["candidate"]
        ):
            raise EvidenceError(
                f"{label} does not bind exact stock/candidate run artifacts"
            )
    failed = tuple(
        key for key in CORRECTNESS_GATES if correctness.get(key) is not True
    ) + (("GSM8K",) if gsm8k.get("passed") is not True else ())
    caption = "; ".join(
        f"{run.arm}/{run.run_id}: batch={run.metadata['batch']}, topology={run.metadata['topology']}"
        for run in (*stock, *candidate)
    )
    return _Collected(
        stock,
        candidate,
        components,
        component_distribution,
        tactic_share,
        cache_hit_share,
        _source_hashes(inputs),
        cache_manifest_bindings,
        caption,
        failed,
        _number(
            load_json_object(inputs.selected_profiles).get("covered_weight"),
            "covered_weight",
        ),
        gsm8k,
    )


def _means(runs: Sequence[RunSummary]) -> tuple[float, float]:
    return (
        fmean(run.generated_tokens_per_second_per_gpu for run in runs),
        fmean(run.total_step_seconds for run in runs),
    )


def _variation(runs: Sequence[RunSummary]) -> float | None:
    if len(runs) < 2:
        return None
    values = [run.generated_tokens_per_second_per_gpu for run in runs]
    return pstdev(values) / fmean(values)


def _first_run(runs: Sequence[RunSummary]) -> RunSummary:
    if not runs:
        raise EvidenceError("run summaries are unexpectedly empty")
    return runs[0]


def _markdown(collected: _Collected, verdict: str, reasons: Sequence[str]) -> str:
    stock_tokens, stock_time = _means(collected.stock_runs)
    candidate_tokens, candidate_time = _means(collected.candidate_runs)
    lines = [
        "# MXFP8 MoE Tactic Audit",
        "",
        f"## {verdict}",
        "",
        *[f"- {reason}" for reason in reasons],
        "",
        "## Metadata",
        "",
        f"{collected.metadata_caption}",
        "",
        "## Raw Run Summary",
        "",
        "| Arm | Runs | tok/s/GPU | Total step s | Realized tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Stock | {len(collected.stock_runs)} | {stock_tokens:.2f} | {stock_time:.2f} | {sum(run.realized_generated_tokens for run in collected.stock_runs)} |",
        f"| Candidate | {len(collected.candidate_runs)} | {candidate_tokens:.2f} | {candidate_time:.2f} | {sum(run.realized_generated_tokens for run in collected.candidate_runs)} |",
        f"| Candidate / Stock | - | {candidate_tokens / stock_tokens:.4f} | {candidate_time / stock_time:.4f} (Total step time / stock; lower is better) | - |",
        "",
        "## Raw Steps 3-8",
        "",
        "| Arm/run | Step | Reward | Loss | Generation KL Error | Mean generation length | Realized tokens | tok/s/GPU | Total step s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, runs in (
        ("Stock", collected.stock_runs),
        ("Candidate", collected.candidate_runs),
    ):
        for run in runs:
            for step in run.steps:
                lines.append(
                    f"| {arm}/{run.run_id} | {step.step} | {step.reward:.4f} | {step.loss:.4f} | {step.kl:.4f} | {step.mean_generation_length:.2f} | {step.realized_generated_tokens} | {step.generated_tokens_per_second_per_gpu:.2f} | {step.total_step_seconds:.2f} |"
                )
    lines.extend(
        (
            "",
            "## Tactic and Cache Evidence",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Achieved replay coverage | {collected.covered_weight:.1%} |",
            *[
                f"| {name} call-weighted micro speedup | {value:.4f} |"
                for name, value in collected.component_speedups
            ],
            f"| Tactic-change share | {collected.tactic_change_share:.1%} |",
            f"| Cache hit share | {collected.cache_hit_share:.1%} |",
            f"| Fallback share | {1.0 - collected.cache_hit_share:.1%} |",
            "",
            "## GSM8K Paired Comparison",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Matched examples | {collected.gsm8k['matched_examples']} |",
            f"| Stock accuracy | {_number(collected.gsm8k['stock_accuracy'], 'GSM8K stock_accuracy'):.4f} |",
            f"| Candidate accuracy | {_number(collected.gsm8k['candidate_accuracy'], 'GSM8K candidate_accuracy'):.4f} |",
            f"| Candidate-only wins | {collected.gsm8k['candidate_only_wins']} |",
            f"| Stock-only wins | {collected.gsm8k['stock_only_wins']} |",
            f"| McNemar p-value | {_number(collected.gsm8k['mcnemar_p_value'], 'GSM8K mcnemar_p_value'):.4g} |",
            "",
            "## Figures",
            "",
            *[f"![{name}]({name}.png)" for name in PLOT_NAMES],
            "",
            "## Cache Manifest Bindings",
            "",
            "| Binding | SHA256 |",
            "| --- | --- |",
            *[
                f"| {name} | `{digest}` |"
                for name, digest in collected.cache_manifest_bindings
            ],
            "",
            "## Source Hashes",
            "",
            "| Source | SHA256 |",
            "| --- | --- |",
            *[f"| {name} | `{digest}` |" for name, digest in collected.source_hashes],
            "",
        )
    )
    return "\n".join(lines)


def _html(markdown: str, verdict: str) -> str:
    lines = markdown.splitlines()
    body = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8"><title>MXFP8 MoE Tactic Audit</title><style>body{font-family:Arial,sans-serif;margin:2rem;max-width:1200px}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #aaa;padding:.35rem;text-align:left}figure{margin:1.5rem 0}img{max-width:100%;height:auto}.verdict{font-weight:bold}</style></head><body>',
        f'<h1>MXFP8 MoE Tactic Audit</h1><p class="verdict">{escape(verdict)}</p>',
    ]
    in_table = False
    for line in lines[3:]:
        if line.startswith("## "):
            if in_table:
                body.append("</table>")
                in_table = False
            body.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<p>{escape(line[2:])}</p>")
        elif line.startswith("| "):
            cells = [escape(cell.strip()) for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":", " "} for cell in cells):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
            tag = (
                "th"
                if "| Arm" in line or "| Metric" in line or "| Source" in line
                else "td"
            )
            body.append(
                "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"
            )
        elif line.startswith("!["):
            name = line[2 : line.index("]")]
            source = line[line.index("(") + 1 : line.index(")")]
            body.append(
                f'<figure><img src="{escape(source)}" alt="{escape(name)}"><figcaption>{escape(name)}</figcaption></figure>'
            )
        elif line:
            body.append(f"<p>{escape(line)}</p>")
    if in_table:
        body.append("</table>")
    body.append("</body></html>\n")
    return "\n".join(body)


def _write(output_dir: Path, markdown: str, verdict: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{REPORT_BASENAME}.md").write_text(markdown, encoding="utf-8")
    (output_dir / f"{REPORT_BASENAME}.html").write_text(
        _html(markdown, verdict), encoding="utf-8"
    )


def _write_unexecuted(output_dir: Path, state: str, reason: str) -> AuditReport:
    write_unavailable_plots(output_dir, state)
    markdown = "\n".join(
        (
            "# MXFP8 MoE Tactic Audit",
            "",
            f"## {state}",
            "",
            f"- {reason}",
            "- Performance values are not reported.",
            "",
            "## Required Evidence",
            "",
            "- steps 3-8; refit/rollout/logprob/train; explicit realized token counts; finite reward/loss/KL; 95% trace coverage; measured FC1+FC2 pair timing or validated stage timings; cache hit/fallback; GSM8K; trace and qualification provenance.",
            "",
            "## Figures",
            "",
            *[f"![{name}]({name}.png)" for name in PLOT_NAMES],
            "",
        )
    )
    _write(output_dir, markdown, state)
    return AuditReport(state, (reason,))


def write_template(output_dir: Path) -> AuditReport:
    """Render the explicitly not-yet-executed report when no artifacts exist."""
    return _write_unexecuted(
        output_dir,
        "NOT YET EXECUTED",
        "Template generated without execution artifacts; Task 11 supplies measured evidence.",
    )


def build_report(inputs: AuditInputs, output_dir: Path) -> AuditReport:
    """Render complete evidence, rejected executed evidence, or incomplete state."""
    try:
        collected = _collect(inputs)
    except EvidenceError as error:
        has_execution = any(
            path.exists()
            for run in (*inputs.stock_runs, *inputs.candidate_runs)
            for path in (run / "run_evidence.json", run / "run_manifest.json")
        )
        return _write_unexecuted(
            output_dir,
            "INCOMPLETE" if has_execution else "NOT YET EXECUTED",
            str(error),
        )
    stock_tokens, stock_time = _means(collected.stock_runs)
    candidate_tokens, candidate_time = _means(collected.candidate_runs)
    variations = tuple(
        value
        for value in (
            _variation(collected.stock_runs),
            _variation(collected.candidate_runs),
        )
        if value is not None
    )
    if collected.failed_gates:
        verdict, reasons = (
            "REJECT",
            ("failed correctness gates: " + ", ".join(collected.failed_gates),),
        )
    elif len(collected.stock_runs) < 2 or len(collected.candidate_runs) < 2:
        verdict, reasons = (
            "INCOMPLETE",
            (
                "Promotion requires at least two comparable runs per arm; executed values are retained below.",
            ),
        )
    else:
        variation = max(variations)
        speedup = candidate_tokens / stock_tokens - 1.0
        no_regression = (
            candidate_tokens >= stock_tokens and candidate_time <= stock_time
        )
        if speedup > variation and no_regression:
            verdict, reasons = (
                "KEEP",
                (
                    f"End-to-end speedup {speedup:.2%} exceeds measured run-to-run variation {variation:.2%}.",
                    "All correctness gates passed and no primary metric regressed.",
                ),
            )
        else:
            verdict, reasons = (
                "REJECT",
                (
                    f"End-to-end speedup {speedup:.2%}; measured run-to-run variation {variation:.2%}.",
                    "Stock FlashInfer autotuning is sufficient for this workload.",
                ),
            )
    reasons = (
        *reasons,
        f"Run-to-run variation: {'not available' if not variations else f'{max(variations):.2%}'}. Within-run step variation is displayed separately and is not a promotion gate.",
    )
    per_step = tuple(
        (
            run.run_id,
            run.arm,
            step.step,
            step.generated_tokens_per_second_per_gpu,
            step.total_step_seconds,
        )
        for run in (*collected.stock_runs, *collected.candidate_runs)
        for step in run.steps
    )
    write_complete_plots(
        output_dir,
        component_speedups=collected.component_distribution,
        tactic_change_share=collected.tactic_change_share,
        cache_hit_share=collected.cache_hit_share,
        normalized_throughput=candidate_tokens / stock_tokens,
        normalized_total_step_time=candidate_time / stock_time,
        per_step=per_step,
        metadata_caption=collected.metadata_caption,
    )
    _write(output_dir, _markdown(collected, verdict, reasons), verdict)
    return AuditReport(verdict, tuple(reasons))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stock-run", type=Path, action="append", default=[])
    parser.add_argument("--candidate-run", type=Path, action="append", default=[])
    for name in (
        "cache-manifest",
        "stock-cache",
        "candidate-cache",
        "trace-summary",
        "qualification-decisions",
        "selected-profiles",
        "shmoo",
        "nsys",
        "correctness",
        "gsm8k",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        AuditInputs(
            stock_runs=tuple(args.stock_run),
            candidate_runs=tuple(args.candidate_run),
            cache_manifest=args.cache_manifest,
            stock_cache=args.stock_cache,
            candidate_cache=args.candidate_cache,
            trace_summary=args.trace_summary,
            qualification_decisions=args.qualification_decisions,
            selected_profiles=args.selected_profiles,
            shmoo=args.shmoo,
            nsys=args.nsys,
            correctness=args.correctness,
            gsm8k=args.gsm8k,
        ),
        args.output_dir,
    )
    return 0 if report.verdict == "KEEP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
