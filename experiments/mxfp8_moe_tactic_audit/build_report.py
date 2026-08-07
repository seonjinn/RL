"""Build a fail-closed HTML and Markdown MXFP8 MoE tactic-audit report."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
import math
from pathlib import Path
from typing import Sequence

from .collect_results import (
    EvidenceError,
    RunSummary,
    compare_manifests,
    find_driver_log,
    load_json_object,
    load_jsonl,
    require_boolean_gates,
    sha256_file,
    summarize_run,
)
from .plot_results import write_complete_plots, write_incomplete_plots


REPORT_BASENAME = "mxfp8_moe_tactic_audit_latest"
REQUIRED_CORRECTNESS_GATES = (
    "micro_correctness",
    "cuda_graph_replay",
    "deterministic_generation",
    "gsm8k",
)


@dataclass(frozen=True)
class AuditInputs:
    """Artifact paths required to make a performance decision."""

    stock_run: Path
    candidate_run: Path
    cache_manifest: Path
    correctness: Path
    gsm8k: Path
    nsys: Path
    selected_profiles: Path
    shmoo: Path


@dataclass(frozen=True)
class AuditReport:
    """The rendered report conclusion and reasons supporting it."""

    verdict: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _CompleteEvidence:
    stock: RunSummary
    candidate: RunSummary
    micro_speedups: tuple[tuple[str, float], ...]
    tactic_change_share: float
    cache_hit_share: float
    coverage: float
    source_hashes: tuple[tuple[str, str], ...]
    manifest_differences: tuple[str, ...]
    gsm8k: dict[str, object]


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


def _tactic_id(value: object) -> int:
    """Return one explicitly typed nonnegative tactic identifier."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError("shmoo tactic IDs must be nonnegative integers")
    return value


def _profile_weights(path: Path) -> dict[str, float]:
    payload = load_json_object(path)
    coverage = _finite_number(payload.get("covered_weight"), "covered_weight")
    if coverage < 0.95:
        raise EvidenceError("selected profile coverage is below 95%")
    rows = payload.get("selected_profiles")
    if not isinstance(rows, list) or not rows:
        raise EvidenceError("selected_profiles must be a nonempty list")
    weights: dict[str, float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise EvidenceError(f"selected_profiles[{index}] must be an object")
        key = row.get("signature_key")
        weight = row.get("normalized_weight")
        if not isinstance(key, str) or not key:
            raise EvidenceError(f"selected_profiles[{index}] has no signature_key")
        parsed_weight = _finite_number(weight, f"selected_profiles[{index}].normalized_weight")
        if parsed_weight <= 0 or key in weights:
            raise EvidenceError("selected profiles must have unique positive weights")
        weights[key] = parsed_weight
    return weights


def _micro_speedups(shmoo_path: Path, weights: dict[str, float]) -> tuple[tuple[tuple[str, float], ...], float]:
    rows_by_signature: dict[str, list[tuple[tuple[int, int], float]]] = {
        key: [] for key in weights
    }
    for row in load_jsonl(shmoo_path):
        key = row.get("signature_key")
        tactic = row.get("tactic")
        if not isinstance(key, str) or key not in rows_by_signature:
            continue
        if not isinstance(tactic, dict):
            continue
        gemm1 = _tactic_id(tactic.get("gemm1"))
        gemm2 = _tactic_id(tactic.get("gemm2"))
        median_us = _finite_number(row.get("median_us"), "shmoo median_us")
        if median_us <= 0:
            raise EvidenceError("shmoo median_us must be positive")
        rows_by_signature[key].append(((gemm1, gemm2), median_us))

    fc1_weighted = 0.0
    fc2_weighted = 0.0
    total_weight = 0.0
    changed_weight = 0.0
    for key, weight in weights.items():
        rows = rows_by_signature[key]
        if len(rows) < 2:
            raise EvidenceError(f"shmoo has no stock/candidate pair for {key}")
        stock_tactic, stock_time = rows[0]
        candidate_tactic, candidate_time = min(rows, key=lambda item: item[1])
        speedup = stock_time / candidate_time
        if candidate_tactic[0] != stock_tactic[0]:
            fc1_weighted += weight * speedup
        else:
            fc1_weighted += weight
        if candidate_tactic[1] != stock_tactic[1]:
            fc2_weighted += weight * speedup
        else:
            fc2_weighted += weight
        if candidate_tactic != stock_tactic:
            changed_weight += weight
        total_weight += weight
    return (
        (("FC1/GEMM1", fc1_weighted / total_weight), ("FC2/GEMM2", fc2_weighted / total_weight)),
        changed_weight / total_weight,
    )


def _cache_hit_share(path: Path) -> float:
    try:
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    except OSError as error:
        raise EvidenceError(f"cannot read NSys CSV {path}: {error}") from error
    events = [row.get("cache_event", "").strip().lower() for row in rows]
    hit_count = sum(event == "cache hit" for event in events)
    fallback_count = sum(event == "fallback" for event in events)
    if hit_count + fallback_count == 0:
        raise EvidenceError("NSys CSV has no cache hit/fallback observations")
    return hit_count / (hit_count + fallback_count)


def _source_hashes(inputs: AuditInputs) -> tuple[tuple[str, str], ...]:
    sources = (
        ("stock ray-driver.log", find_driver_log(inputs.stock_run)),
        ("candidate ray-driver.log", find_driver_log(inputs.candidate_run)),
        ("stock manifest", inputs.stock_run / "run_manifest.json"),
        ("candidate manifest", inputs.candidate_run / "run_manifest.json"),
        ("cache manifest", inputs.cache_manifest),
        ("correctness", inputs.correctness),
        ("GSM8K", inputs.gsm8k),
        ("NSys CSV", inputs.nsys),
        ("selected profiles", inputs.selected_profiles),
        ("shmoo JSONL", inputs.shmoo),
    )
    return tuple((label, sha256_file(path)) for label, path in sources)


def _collect(inputs: AuditInputs) -> _CompleteEvidence:
    stock = summarize_run(inputs.stock_run)
    candidate = summarize_run(inputs.candidate_run)
    if not stock.all_metrics_finite or not candidate.all_metrics_finite:
        raise EvidenceError("NeMo-RL reward/loss/KL or timing metrics are not finite")
    manifest_differences = compare_manifests(
        inputs.stock_run / "run_manifest.json", inputs.candidate_run / "run_manifest.json"
    )
    if manifest_differences:
        raise EvidenceError("non-cache manifests differ: " + ", ".join(manifest_differences))
    cache_manifest = load_json_object(inputs.cache_manifest)
    for field in ("stock_sha256", "candidate_sha256", "source_fingerprints"):
        if field not in cache_manifest:
            raise EvidenceError(f"cache manifest missing {field}")
    correctness = load_json_object(inputs.correctness)
    require_boolean_gates(correctness, REQUIRED_CORRECTNESS_GATES)
    gsm8k = load_json_object(inputs.gsm8k)
    if gsm8k.get("passed") is not True:
        raise EvidenceError("matched GSM8K correctness gate failed or is absent")
    weights = _profile_weights(inputs.selected_profiles)
    micro_speedups, tactic_change_share = _micro_speedups(inputs.shmoo, weights)
    return _CompleteEvidence(
        stock=stock,
        candidate=candidate,
        micro_speedups=micro_speedups,
        tactic_change_share=tactic_change_share,
        cache_hit_share=_cache_hit_share(inputs.nsys),
        coverage=_finite_number(load_json_object(inputs.selected_profiles).get("covered_weight"), "covered_weight"),
        source_hashes=_source_hashes(inputs),
        manifest_differences=manifest_differences,
        gsm8k=gsm8k,
    )


def _markdown_complete(evidence: _CompleteEvidence, verdict: str, reasons: Sequence[str]) -> str:
    normalized_throughput = evidence.candidate.generated_tokens_per_second_per_gpu / evidence.stock.generated_tokens_per_second_per_gpu
    normalized_step_speed = evidence.stock.total_step_seconds / evidence.candidate.total_step_seconds
    lines = [
        "# MXFP8 MoE Tactic Audit",
        "",
        f"## {verdict}",
        "",
        "Measured Qwen3-30B-A3B evidence: steps 3-8, six steady-state steps per arm.",
        "",
        "## Decision",
        "",
        *[f"- {reason}" for reason in reasons],
        "",
        "## Correctness and Coverage",
        "",
        f"- Replay coverage: {evidence.coverage:.1%} (required: 95%).",
        "- Correctness: micro, CUDA Graph replay, deterministic generation, GSM8K, and NeMo-RL phases refit/rollout/logprob/train passed.",
        f"- GSM8K: matched gate passed; stock accuracy {evidence.gsm8k.get('stock_accuracy')}, candidate accuracy {evidence.gsm8k.get('candidate_accuracy')}.",
        "- Cache behavior: cache hit and fallback are reported from the NSys CSV; fallback remains stock FlashInfer behavior.",
        "",
        "## Raw End-to-End Table",
        "",
        "| Arm | Generation tok/s/GPU | Total step s | Realized tokens | Variation |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Stock | {evidence.stock.generated_tokens_per_second_per_gpu:.2f} | {evidence.stock.total_step_seconds:.2f} | {evidence.stock.realized_generated_tokens} | {evidence.stock.variation:.2%} |",
        f"| Candidate | {evidence.candidate.generated_tokens_per_second_per_gpu:.2f} | {evidence.candidate.total_step_seconds:.2f} | {evidence.candidate.realized_generated_tokens} | {evidence.candidate.variation:.2%} |",
        f"| Stock-normalized | {normalized_throughput:.4f} | {normalized_step_speed:.4f} speed | - | - |",
        "",
        "## Raw Tactic and Cache Table",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        *[f"| {label} call-weighted micro speedup | {value:.4f} |" for label, value in evidence.micro_speedups],
        f"| Tactic-change share | {evidence.tactic_change_share:.1%} |",
        f"| Cache hit share | {evidence.cache_hit_share:.1%} |",
        f"| Fallback share | {1.0 - evidence.cache_hit_share:.1%} |",
        "",
        "## Source hashes",
        "",
        "| Source | SHA256 |",
        "| --- | --- |",
        *[f"| {label} | `{digest}` |" for label, digest in evidence.source_hashes],
        "",
        "Figures: call-weighted FC1/GEMM1 and FC2/GEMM2 micro speedup; tactic-change plus cache hit/fallback shares; stock-normalized tok/s/GPU plus step speed; and per-step variation. The baseline 1.0 line is behind normalized bars. No confidence bands are rendered because no repeated end-to-end runs exist.",
        "",
    ]
    return "\n".join(lines)


def _markdown_incomplete(reason: str) -> str:
    return "\n".join(
        [
            "# MXFP8 MoE Tactic Audit",
            "",
            "## REJECT",
            "",
            "## INCOMPLETE EVIDENCE",
            "",
            f"- {reason}",
            "- This is a not-yet-executed report template. Performance values are not reported.",
            "- Required evidence: six measured steps 3-8; successful refit/rollout/logprob/train phases; realized token counts; finite reward/loss/KL; matched manifests; 95% replay coverage; FC1/GEMM1 and FC2/GEMM2 shmoo data; cache hit/fallback NSys data; and matched GSM8K evidence.",
            "- Stock FlashInfer autotuning is sufficient for this workload until complete evidence proves otherwise.",
            "",
            "## Raw tables",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            "| Steps 3-8 / realized tokens / finite reward-loss-KL | not reported |",
            "| FC1/GEMM1 and FC2/GEMM2 weighted micro speedup | not reported |",
            "| Tactic-change / cache hit / fallback | not reported |",
            "| Stock-normalized tok/s/GPU / step time / per-step variation | not reported |",
            "| 95% coverage / GSM8K | not reported |",
            "",
            "## Source hashes",
            "",
            "No source artifacts were supplied, so hashes are not reported.",
            "",
        ]
    )


def _write_report(output_dir: Path, markdown: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{REPORT_BASENAME}.md").write_text(markdown, encoding="utf-8")
    html = "<!doctype html><html><head><meta charset=\"utf-8\"><title>MXFP8 MoE Tactic Audit</title></head><body><pre>" + escape(markdown) + "</pre></body></html>\n"
    (output_dir / f"{REPORT_BASENAME}.html").write_text(html, encoding="utf-8")


def build_report(inputs: AuditInputs, output_dir: Path) -> AuditReport:
    """Render complete evidence or a non-numeric REJECT template on any gap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        evidence = _collect(inputs)
    except EvidenceError as error:
        reason = str(error)
        write_incomplete_plots(output_dir)
        _write_report(output_dir, _markdown_incomplete(reason))
        return AuditReport(verdict="REJECT", reasons=(reason,))

    normalized_throughput = evidence.candidate.generated_tokens_per_second_per_gpu / evidence.stock.generated_tokens_per_second_per_gpu
    normalized_step_speed = evidence.stock.total_step_seconds / evidence.candidate.total_step_seconds
    end_to_end_speedup = normalized_throughput - 1.0
    variation = max(evidence.stock.variation, evidence.candidate.variation)
    no_primary_metric_regression = normalized_throughput >= 1.0 and normalized_step_speed >= 1.0
    keep = end_to_end_speedup > variation and no_primary_metric_regression
    if keep:
        verdict = "KEEP"
        reasons = (
            "All correctness gates passed.",
            f"End-to-end generation speedup {end_to_end_speedup:.2%} exceeds run-to-run variation {variation:.2%}.",
            "No primary metric regressed.",
        )
    else:
        verdict = "REJECT"
        reasons = (
            "Stock FlashInfer autotuning is sufficient for this workload.",
            f"End-to-end generation speedup {end_to_end_speedup:.2%}; run-to-run variation {variation:.2%}.",
            f"Primary metrics did not regress: {no_primary_metric_regression}.",
            "Microbenchmark opportunity is not by itself an end-to-end promotion result.",
        )
    write_complete_plots(
        output_dir,
        micro_speedups=evidence.micro_speedups,
        tactic_change_share=evidence.tactic_change_share,
        cache_hit_share=evidence.cache_hit_share,
        normalized_generation_throughput=normalized_throughput,
        normalized_step_speed=normalized_step_speed,
        step_values=tuple(
            (stock.step, stock.generated_tokens_per_second_per_gpu, candidate.generated_tokens_per_second_per_gpu)
            for stock, candidate in zip(evidence.stock.steps, evidence.candidate.steps, strict=True)
        ),
    )
    _write_report(output_dir, _markdown_complete(evidence, verdict, reasons))
    return AuditReport(verdict=verdict, reasons=reasons)


def write_template(output_dir: Path) -> AuditReport:
    """Create a clearly non-numeric report before hardware artifacts exist."""
    write_incomplete_plots(output_dir)
    _write_report(output_dir, _markdown_incomplete("hardware artifacts have not been collected"))
    return AuditReport(verdict="REJECT", reasons=("hardware artifacts have not been collected",))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse report inputs or the explicit no-artifact template mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", action="store_true")
    for name in ("stock-run", "candidate-run", "cache-manifest", "correctness", "gsm8k", "nsys", "selected-profiles", "shmoo"):
        parser.add_argument(f"--{name}", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the report from collected artifacts."""
    args = parse_args(argv)
    if args.template:
        write_template(args.output_dir)
        return 0
    values = {
        "stock_run": args.stock_run,
        "candidate_run": args.candidate_run,
        "cache_manifest": args.cache_manifest,
        "correctness": args.correctness,
        "gsm8k": args.gsm8k,
        "nsys": args.nsys,
        "selected_profiles": args.selected_profiles,
        "shmoo": args.shmoo,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise SystemExit("missing report inputs: " + ", ".join(missing))
    report = build_report(AuditInputs(**values), args.output_dir)
    return 0 if report.verdict == "KEEP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
