"""Contracts for the fail-closed MXFP8 MoE tactic-audit report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from experiments.mxfp8_moe_tactic_audit.build_report import (
    AuditInputs,
    build_report,
    write_template,
)
from experiments.mxfp8_moe_tactic_audit.collect_results import (
    compare_manifests,
    summarize_run,
)


def _sha() -> str:
    return "a" * 64


def write_run_fixture(
    root: Path,
    *,
    steps: Iterable[int],
    tokens_per_second_per_gpu: float,
    generation_seconds: float,
    total_step_seconds: float,
    arm: str,
    complete: bool = True,
    nested_log: bool = False,
) -> Path:
    """Write a minimal validation run with real metric and phase evidence."""
    run = root / arm
    run.mkdir()
    blocks = []
    for _step in steps:
        blocks.append(
            "\n".join(
                [
                    "Training Results:",
                    f"  • Total step time: {total_step_seconds:.2f}s",
                    f"  • generation: {generation_seconds:.2f}s (26.2%)",
                    f"    - E2E (Tokens/sec/gpu): {tokens_per_second_per_gpu:.2f}",
                    "    - Generation Worker Group (Tokens/sec/gpu): "
                    f"{tokens_per_second_per_gpu:.2f}",
                    "  • Realized generated tokens: 64000",
                    "  • Reward: 0.5",
                    "  • KL: 0.01",
                    "  • Loss: 0.2",
                ]
            )
        )
    log_dir = run / "validation-logs" if nested_log else run
    log_dir.mkdir(exist_ok=True)
    (log_dir / "ray-driver.log").write_text("\n".join(blocks), encoding="utf-8")
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "cache_sha256": _sha() if arm == "stock" else "b" * 64,
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
        (run / "phase_status.json").write_text(
            json.dumps(
                {
                    "refit": "success",
                    "rollout": "success",
                    "logprob": "success",
                    "train": "success",
                },
                sort_keys=True,
            ),
            encoding="ascii",
        )
    return run


def write_audit_artifacts(root: Path) -> dict[str, Path]:
    """Write compact, schema-shaped evidence for an executable report fixture."""
    selected_profiles = root / "selected_profiles.json"
    selected_profiles.write_text(
        json.dumps(
            {
                "covered_weight": 0.96,
                "total_gpu_time_us": 1000.0,
                "selected_profiles": [
                    {
                        "signature_key": "sig-1",
                        "call_count": 10,
                        "normalized_weight": 1.0,
                    }
                ],
            }
        ),
        encoding="ascii",
    )
    shmoo = root / "measurements.jsonl"
    rows = [
        {"signature_key": "sig-1", "tactic": {"gemm1": 1, "gemm2": 2}, "median_us": 100.0},
        {"signature_key": "sig-1", "tactic": {"gemm1": 3, "gemm2": 4}, "median_us": 90.0},
    ]
    shmoo.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="ascii",
    )
    cache_manifest = root / "cache_manifest.json"
    cache_manifest.write_text(
        json.dumps(
            {
                "stock_sha256": _sha(),
                "candidate_sha256": "b" * 64,
                "promoted_entries": 1,
                "retained_entries": 4,
                "source_fingerprints": {"trace_set_sha256": "4" * 64},
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    correctness = root / "correctness.json"
    correctness.write_text(
        json.dumps(
            {
                "micro_correctness": True,
                "cuda_graph_replay": True,
                "deterministic_generation": True,
                "gsm8k": True,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    gsm8k = root / "gsm8k_comparison.json"
    gsm8k.write_text(
        json.dumps(
            {
                "passed": True,
                "stock_accuracy": 0.6,
                "candidate_accuracy": 0.6,
                "accuracy_delta": 0.0,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    nsys = root / "nsys_cache.csv"
    nsys.write_text("cache_event\ncache hit\nfallback\ncache hit\n", encoding="ascii")
    return {
        "cache_manifest": cache_manifest,
        "correctness": correctness,
        "gsm8k": gsm8k,
        "nsys": nsys,
        "selected_profiles": selected_profiles,
        "shmoo": shmoo,
    }


def test_summarize_run_requires_all_six_measured_steps(tmp_path: Path) -> None:
    run = write_run_fixture(
        tmp_path,
        steps=range(1, 9),
        tokens_per_second_per_gpu=9500.0,
        generation_seconds=55.0,
        total_step_seconds=210.0,
        arm="stock",
        nested_log=True,
    )

    summary = summarize_run(run, first_step=3, last_step=8)

    assert summary.measured_steps == 6
    assert summary.generated_tokens_per_second_per_gpu > 0
    assert summary.all_metrics_finite
    assert summary.realized_generated_tokens == 6 * 64000


def test_manifest_comparison_allows_only_explicit_cache_identity(tmp_path: Path) -> None:
    stock = write_run_fixture(
        tmp_path,
        steps=range(1, 9),
        tokens_per_second_per_gpu=9500.0,
        generation_seconds=55.0,
        total_step_seconds=210.0,
        arm="stock",
    )
    candidate = write_run_fixture(
        tmp_path,
        steps=range(1, 9),
        tokens_per_second_per_gpu=9600.0,
        generation_seconds=54.0,
        total_step_seconds=208.0,
        arm="candidate",
    )

    assert compare_manifests(stock / "run_manifest.json", candidate / "run_manifest.json") == ()
    candidate_manifest = json.loads((candidate / "run_manifest.json").read_text())
    candidate_manifest["recipe_sha256"] = "9" * 64
    (candidate / "run_manifest.json").write_text(json.dumps(candidate_manifest))
    assert compare_manifests(stock / "run_manifest.json", candidate / "run_manifest.json")


def test_report_renders_complete_evidence_and_paper_assets(tmp_path: Path) -> None:
    stock = write_run_fixture(
        tmp_path,
        steps=range(1, 9),
        tokens_per_second_per_gpu=9500.0,
        generation_seconds=55.0,
        total_step_seconds=210.0,
        arm="stock",
    )
    candidate = write_run_fixture(
        tmp_path,
        steps=range(1, 9),
        tokens_per_second_per_gpu=9700.0,
        generation_seconds=53.0,
        total_step_seconds=205.0,
        arm="candidate",
        nested_log=True,
    )
    artifacts = write_audit_artifacts(tmp_path)
    output_dir = tmp_path / "report"

    report = build_report(
        AuditInputs(stock_run=stock, candidate_run=candidate, **artifacts), output_dir
    )

    markdown = (output_dir / "mxfp8_moe_tactic_audit_latest.md").read_text()
    for required in (
        "FC1/GEMM1",
        "FC2/GEMM2",
        "95%",
        "cache hit",
        "fallback",
        "GSM8K",
        "steps 3-8",
        "KEEP",
        "Source hashes",
    ):
        assert required in markdown
    assert report.verdict == "KEEP"
    for plot_name in (
        "mxfp8_moe_tactic_audit_micro_speedup",
        "mxfp8_moe_tactic_audit_tactic_cache_shares",
        "mxfp8_moe_tactic_audit_end_to_end",
        "mxfp8_moe_tactic_audit_step_variation",
    ):
        assert (output_dir / f"{plot_name}.png").is_file()
        assert (output_dir / f"{plot_name}.pdf").is_file()
    assert (output_dir / "mxfp8_moe_tactic_audit_latest.html").is_file()


def test_report_rejects_incomplete_evidence_without_performance_numbers(
    tmp_path: Path,
) -> None:
    stock = write_run_fixture(
        tmp_path,
        steps=range(1, 9),
        tokens_per_second_per_gpu=9500.0,
        generation_seconds=55.0,
        total_step_seconds=210.0,
        arm="stock",
        complete=False,
    )
    candidate = write_run_fixture(
        tmp_path,
        steps=range(1, 9),
        tokens_per_second_per_gpu=9700.0,
        generation_seconds=53.0,
        total_step_seconds=205.0,
        arm="candidate",
    )
    artifacts = write_audit_artifacts(tmp_path)
    output_dir = tmp_path / "report"

    report = build_report(
        AuditInputs(stock_run=stock, candidate_run=candidate, **artifacts), output_dir
    )

    markdown = (output_dir / "mxfp8_moe_tactic_audit_latest.md").read_text()
    assert report.verdict == "REJECT"
    assert "INCOMPLETE EVIDENCE" in markdown
    assert "not reported" in markdown
    assert "## Raw tables" in markdown
    assert "| Metric | Value |" in markdown


def test_template_renders_placeholder_plots_in_a_new_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "report"

    report = write_template(output_dir)

    assert report.verdict == "REJECT"
    assert (output_dir / "mxfp8_moe_tactic_audit_micro_speedup.png").is_file()
