"""Producer-shaped contracts for the MXFP8 MoE tactic-audit report."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

from experiments.mxfp8_moe_tactic_audit.build_report import (
    AuditInputs,
    build_report,
    write_template,
)
from experiments.mxfp8_moe_tactic_audit.collect_results import summarize_run, write_run_evidence


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _trace_set_sha(paths: tuple[Path, ...]) -> str:
    return sha256(json.dumps(sorted(_sha(path) for path in paths), separators=(",", ":")).encode("ascii")).hexdigest()


def write_grpo_run(
    root: Path,
    *,
    arm: str,
    run_id: str,
    throughput: float,
    total_step_seconds: float,
    complete: bool = True,
) -> Path:
    """Write actual GRPO labels and launcher-shaped execution evidence."""
    run = root / f"{arm}-{run_id}"
    log_dir = run / "logs"
    log_dir.mkdir(parents=True)
    blocks = []
    for _step in range(1, 9):
        blocks.append(
            "\n".join(
                (
                    "Training Results:",
                    "  • Loss: 0.2000",
                    "  • Generation KL Error: 0.0100",
                    "  • Avg Reward: 0.5000",
                    "  • Mean Generation Length: 4000.0000",
                    f"  • Total step time: {total_step_seconds:.2f}s",
                    "  • generation: 55.00s (26.2%)",
                    f"    - E2E (Tokens/sec/gpu): {throughput:.2f}",
                    "    - Generation Worker Group (Tokens/sec/gpu): "
                    f"{throughput:.2f}",
                )
            )
        )
    (log_dir / "ray-driver.log").write_text("\n".join(blocks), encoding="utf-8")
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "cache_sha256": "a" * 64 if arm == "stock" else "b" * 64,
                "container_sha256": "c" * 64,
                "execution_inputs_sha256": "d" * 64,
                "model_snapshot_sha256": "e" * 64,
                "nemo_rl_commit": "f" * 40,
                "recipe_sha256": "1" * 64,
                "run_kind": f"validation-{arm}",
                "scripts_sha256": "2" * 64,
                "vllm_commit": "3" * 40,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    if complete:
        (run / "run_evidence.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "exit_code": 0,
                    "metadata": {
                        "batch": "16 prompts x 8 generations",
                        "run_id": run_id,
                        "topology": "4 nodes x 4 GPUs",
                    },
                    "runtime_fingerprints": {"producer": "fixture"},
                    "phases": {
                        "logprob": "success",
                        "refit": "success",
                        "rollout": "success",
                        "train": "success",
                    },
                    "steps": [
                        {"realized_generated_tokens": 64000, "step": step}
                        for step in range(3, 9)
                    ],
                },
                sort_keys=True,
            ),
            encoding="ascii",
        )
    return run


def write_audit_artifacts(root: Path) -> dict[str, Path]:
    """Write cache-bound shmoo, trace, component, and provenance evidence."""
    trace_summary = root / "trace_summary.json"
    raw_trace = root / "trace-rank0.jsonl"
    raw_trace.write_text('{"producer":"routing-trace"}\n', encoding="ascii")
    trace_summary.write_text(
        json.dumps(
            {
                "trace_paths": [raw_trace.name],
                "profiles": [
                    {"cache_key": "cache-1", "call_weight": 1.0, "signature_key": "sig-1"}
                ]
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    selected_profiles = root / "selected_profiles.json"
    selected_profiles.write_text(
        json.dumps(
            {
                "covered_weight": 0.96,
                "selected_profiles": [
                    {"call_count": 10, "normalized_weight": 1.0, "signature_key": "sig-1"}
                ],
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    shmoo = root / "measurements.jsonl"
    rows = [
        {"deterministic": True, "failure": "kernel failed", "finite": False, "median_us": 0.0, "signature_key": "sig-1", "tactic": {"gemm1": 9, "gemm2": 9}},
        {"deterministic": True, "failure": None, "finite": True, "median_us": 90.0, "signature_key": "sig-1", "tactic": {"gemm1": 3, "gemm2": 4}},
        {"deterministic": True, "failure": None, "finite": True, "median_us": 100.0, "signature_key": "sig-1", "tactic": {"gemm1": 1, "gemm2": 2}},
    ]
    shmoo.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="ascii")
    stock_cache = root / "stock_cache.json"
    candidate_cache = root / "candidate_cache.json"
    stock_cache.write_text(json.dumps({"cache-1": ["MoERunner", [1, 2]]}, sort_keys=True), encoding="ascii")
    candidate_cache.write_text(json.dumps({"cache-1": ["MoERunner", [3, 4]]}, sort_keys=True), encoding="ascii")
    cache_manifest = root / "cache_manifest.json"
    cache_manifest.write_text(
        json.dumps(
            {
                "candidate_sha256": _sha(candidate_cache),
                "source_fingerprints": {
                    "selected_profiles_sha256": _sha(selected_profiles),
                    "shmoo_results_sha256": _sha(shmoo),
                    "trace_set_sha256": _trace_set_sha((raw_trace,)),
                },
                "stock_sha256": _sha(stock_cache),
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    qualification_decisions = root / "qualification_decisions.json"
    qualification_decisions.write_text(
        json.dumps(
            {
                "cache_manifest_sha256": _sha(cache_manifest),
                "decisions": [
                    {"cache_key": "cache-1", "promoted": True, "selected": {"gemm1": 3, "gemm2": 4}, "stock": {"gemm1": 1, "gemm2": 2}, "signature_keys": ["sig-1"]}
                ],
                "trace_set_sha256": _trace_set_sha((raw_trace,)),
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    correctness = root / "correctness.json"
    correctness.write_text(json.dumps({"cuda_graph_replay": True, "deterministic_generation": True, "micro_correctness": True}, sort_keys=True), encoding="ascii")
    gsm8k = root / "gsm8k_comparison.json"
    gsm8k.write_text(json.dumps({"accuracy_delta": 0.0, "both_correct": 700, "both_wrong": 600, "candidate_accuracy": 0.6, "candidate_only_wins": 10, "delta_ci95": [-0.01, 0.01], "matched_examples": 1319, "mcnemar_p_value": 1.0, "passed": True, "provenance_matched": True, "stock_accuracy": 0.6, "stock_only_wins": 9}, sort_keys=True), encoding="ascii")
    nsys = root / "nsys_components.csv"
    with nsys.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "cache_event", "cache_key", "call_count", "component", "median_us", "signature_key", "tactic"])
        writer.writeheader()
        writer.writerows(
            [
                {"arm": "stock", "cache_event": "cache hit", "cache_key": "cache-1", "call_count": 10, "component": "FC1/GEMM1", "median_us": 60.0, "signature_key": "sig-1", "tactic": "1,2"},
                {"arm": "candidate", "cache_event": "cache hit", "cache_key": "cache-1", "call_count": 10, "component": "FC1/GEMM1", "median_us": 50.0, "signature_key": "sig-1", "tactic": "3,4"},
                {"arm": "stock", "cache_event": "fallback", "cache_key": "cache-1", "call_count": 10, "component": "FC2/GEMM2", "median_us": 40.0, "signature_key": "sig-1", "tactic": "1,2"},
                {"arm": "candidate", "cache_event": "cache hit", "cache_key": "cache-1", "call_count": 10, "component": "FC2/GEMM2", "median_us": 30.0, "signature_key": "sig-1", "tactic": "3,4"},
            ]
        )
    return {
        "cache_manifest": cache_manifest,
        "candidate_cache": candidate_cache,
        "correctness": correctness,
        "gsm8k": gsm8k,
        "nsys": nsys,
        "qualification_decisions": qualification_decisions,
        "selected_profiles": selected_profiles,
        "shmoo": shmoo,
        "stock_cache": stock_cache,
        "trace_summary": trace_summary,
    }


def complete_inputs(root: Path, *, repeated: bool = True) -> AuditInputs:
    """Build a complete evidence set with one or two runs per arm."""
    stock_runs = (write_grpo_run(root, arm="stock", run_id="one", throughput=9500.0, total_step_seconds=210.0),)
    candidate_runs = (write_grpo_run(root, arm="candidate", run_id="candidate-one", throughput=9700.0, total_step_seconds=205.0),)
    if repeated:
        stock_runs += (write_grpo_run(root, arm="stock", run_id="two", throughput=9550.0, total_step_seconds=209.0),)
        candidate_runs += (write_grpo_run(root, arm="candidate", run_id="candidate-two", throughput=9750.0, total_step_seconds=204.0),)
    return AuditInputs(stock_runs=stock_runs, candidate_runs=candidate_runs, **write_audit_artifacts(root))


def test_summarize_run_parses_actual_grpo_labels_and_evidence_tokens(tmp_path: Path) -> None:
    run = write_grpo_run(tmp_path, arm="stock", run_id="one", throughput=9500.0, total_step_seconds=210.0)

    summary = summarize_run(run)

    assert summary.measured_steps == 6
    assert summary.steps[0].reward == 0.5
    assert summary.steps[0].kl == 0.01
    assert summary.steps[0].loss == 0.2
    assert summary.realized_generated_tokens == 6 * 64000


def test_launcher_evidence_is_derived_from_grpo_producer_masks_and_phase_markers(tmp_path: Path) -> None:
    run = write_grpo_run(tmp_path, arm="stock", run_id="producer", throughput=9500.0, total_step_seconds=210.0)
    (run / "run_evidence.json").unlink()
    log = run / "logs" / "ray-driver.log"
    with log.open("a", encoding="utf-8") as handle:
        for step in range(3, 9):
            for phase in ("refit", "rollout", "logprob", "train"):
                handle.write(f"\n[MXFP8_MOE_AUDIT] step={step} phase={phase} status=success")
            dump = run / "logs" / "exp_001" / f"train_data_step{step}.jsonl"
            dump.parent.mkdir(parents=True, exist_ok=True)
            dump.write_text(json.dumps({"token_loss_mask": [1, 1, 0, 1]}) + "\n", encoding="ascii")

    write_run_evidence(run, arm="stock", run_id="producer", metadata={"batch": "fixture", "run_id": "producer", "topology": "fixture"}, runtime_fingerprints={"producer": "fixture"})

    assert summarize_run(run).realized_generated_tokens == 18


def test_report_binds_shuffled_shmoo_rows_and_ignores_failed_fastest_row(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path)
    output_dir = tmp_path / "report"

    report = build_report(inputs, output_dir)

    markdown = (output_dir / "mxfp8_moe_tactic_audit_latest.md").read_text()
    assert report.verdict == "KEEP"
    assert "FC1/GEMM1 call-weighted micro speedup | 1.2000" in markdown
    assert "FC2/GEMM2 call-weighted micro speedup | 1.3333" in markdown
    assert "Run-to-run variation" in markdown
    assert "Total step time / stock" in markdown
    assert "Cache Manifest Bindings" in markdown
    assert "stock stock-one manifest" in markdown
    html = (output_dir / "mxfp8_moe_tactic_audit_latest.html").read_text()
    assert "<table>" in html
    assert "<img src=" in html
    assert "Raw Steps 3-8" in html
    assert "Generation KL Error" in html
    assert "run=one; batch=16 prompts x 8 generations; topology=4 nodes x 4 GPUs" in html


def test_one_run_per_arm_is_incomplete_for_promotion_but_preserves_raw_values(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path, repeated=False)
    output_dir = tmp_path / "report"

    report = build_report(inputs, output_dir)

    markdown = (output_dir / "mxfp8_moe_tactic_audit_latest.md").read_text()
    assert report.verdict == "INCOMPLETE"
    assert "9500.00" in markdown
    assert "9700.00" in markdown
    assert "at least two comparable runs per arm" in markdown


def test_failed_gate_rejects_but_preserves_raw_evidence(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path)
    inputs.correctness.write_text(json.dumps({"cuda_graph_replay": False, "deterministic_generation": True, "micro_correctness": True}), encoding="ascii")
    output_dir = tmp_path / "report"

    report = build_report(inputs, output_dir)

    markdown = (output_dir / "mxfp8_moe_tactic_audit_latest.md").read_text()
    assert report.verdict == "REJECT"
    assert "9500.00" in markdown
    assert "failed correctness gates: cuda_graph_replay" in markdown


def test_nsys_rows_must_bind_the_trace_cache_key(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path)
    with inputs.nsys.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["cache_key"] = "wrong-cache"
    with inputs.nsys.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = build_report(inputs, tmp_path / "report")

    assert report.verdict == "INCOMPLETE"
    assert "NSys component evidence missing" in report.reasons[0]


def test_missing_artifacts_render_not_yet_executed_without_numeric_claims(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path)
    inputs.nsys.unlink()
    output_dir = tmp_path / "report"

    report = build_report(inputs, output_dir)

    markdown = (output_dir / "mxfp8_moe_tactic_audit_latest.md").read_text()
    assert report.verdict == "INCOMPLETE"
    assert "not reported" in markdown
    assert "9500.00" not in markdown


def test_template_is_explicitly_not_yet_executed_without_source_artifacts(
    tmp_path: Path,
) -> None:
    report = write_template(tmp_path / "report")

    markdown = (tmp_path / "report" / "mxfp8_moe_tactic_audit_latest.md").read_text()
    assert report.verdict == "NOT YET EXECUTED"
    assert "Template generated without execution artifacts" in markdown
    assert "Performance values are not reported" in markdown
    html = (tmp_path / "report" / "mxfp8_moe_tactic_audit_latest.html").read_text()
    assert html.count("<img src=") == 4
