"""Producer-shaped contracts for the MXFP8 MoE tactic-audit report."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from experiments.mxfp8_moe_tactic_audit import plot_results
from experiments.mxfp8_moe_tactic_audit.build_report import (
    AuditInputs,
    _component_speedups,
    build_report,
    write_template,
)
from experiments.mxfp8_moe_tactic_audit.collect_results import (
    comparison_artifact_bindings,
    summarize_run,
    write_run_evidence,
)
from experiments.mxfp8_moe_tactic_audit.compare_gsm8k import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    paired_outcome_bootstrap_ci,
    paired_outcomes_sha256,
)
from experiments.mxfp8_moe_tactic_audit.nsys_to_component_csv import convert


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _trace_set_sha(paths: tuple[Path, ...]) -> str:
    return sha256(
        json.dumps(sorted(_sha(path) for path in paths), separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


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
                    f"    - Generation Worker Group (Tokens/sec/gpu): {throughput:.2f}",
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
                "run_kind": "validation",
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
                        "generation_settings": "greedy generation fixture",
                        "run_id": run_id,
                        "run_kind": "validation",
                        "topology": "4 nodes x 4 GPUs",
                    },
                    "runtime_fingerprints": {
                        "cache_sha256": "a" * 64 if arm == "stock" else "b" * 64,
                        "producer": "fixture",
                    },
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
                    {
                        "cache_key": "cache-1",
                        "call_weight": 1.0,
                        "signature_key": "sig-1",
                    }
                ],
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
                    {
                        "call_count": 10,
                        "normalized_weight": 1.0,
                        "signature_key": "sig-1",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    shmoo = root / "measurements.jsonl"
    rows = [
        {
            "deterministic": True,
            "failure": "kernel failed",
            "finite": False,
            "median_us": 0.0,
            "signature_key": "sig-1",
            "tactic": {"gemm1": 9, "gemm2": 9},
        },
        {
            "deterministic": True,
            "failure": None,
            "finite": True,
            "median_us": 90.0,
            "signature_key": "sig-1",
            "tactic": {"gemm1": 3, "gemm2": 4},
        },
        {
            "deterministic": True,
            "failure": None,
            "finite": True,
            "median_us": 100.0,
            "signature_key": "sig-1",
            "tactic": {"gemm1": 1, "gemm2": 2},
        },
    ]
    shmoo.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="ascii",
    )
    stock_cache = root / "stock_cache.json"
    candidate_cache = root / "candidate_cache.json"
    stock_cache.write_text(
        json.dumps({"cache-1": ["MoERunner", [1, 2]]}, sort_keys=True), encoding="ascii"
    )
    candidate_cache.write_text(
        json.dumps({"cache-1": ["MoERunner", [3, 4]]}, sort_keys=True), encoding="ascii"
    )
    cache_manifest = root / "cache_manifest.json"
    runtime_fingerprints = {
        "container_sha256": "c" * 64,
        "cuda_graph_mode": "required",
        "cuda_version": "13.0",
        "dp_size": "16",
        "ep_size": "1",
        "flashinfer_version": "0.6.13",
        "gpu_name": "NVIDIA GB200",
        "model_revision": "fixture-model-revision",
        "tp_size": "4",
        "vllm_commit": "3" * 40,
    }
    cache_manifest.write_text(
        json.dumps(
            {
                "candidate_sha256": _sha(candidate_cache),
                "source_fingerprints": {
                    **runtime_fingerprints,
                    "selected_profiles_sha256": _sha(selected_profiles),
                    "shmoo_results_sha256": _sha(shmoo),
                    "trace_set_sha256": _trace_set_sha((raw_trace,)),
                },
                "schema_version": 1,
                "stock_sha256": _sha(stock_cache),
                "promoted_entries": 1,
                "retained_entries": 0,
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
                    {
                        "cache_key": "cache-1",
                        "promoted": True,
                        "selected": {"gemm1": 3, "gemm2": 4},
                        "stock": {"gemm1": 1, "gemm2": 2},
                        "signature_keys": ["sig-1"],
                    }
                ],
                "selected_profiles_sha256": _sha(selected_profiles),
                "shmoo_results_sha256": _sha(shmoo),
                "trace_set_sha256": _trace_set_sha((raw_trace,)),
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    correctness = root / "correctness.json"
    correctness.write_text(
        json.dumps(
            {
                "cuda_graph_replay": True,
                "deterministic_generation": True,
                "micro_correctness": True,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    gsm8k = root / "gsm8k_comparison.json"
    paired_outcomes = (0,) * 1300 + (1,) * 10 + (-1,) * 9
    delta_ci95 = paired_outcome_bootstrap_ci(
        paired_outcomes, BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES
    )
    gsm8k.write_text(
        json.dumps(
            {
                "accuracy_delta": 1 / 1319,
                "both_correct": 700,
                "both_wrong": 600,
                "candidate_accuracy": 710 / 1319,
                "candidate_only_wins": 10,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "delta_ci95": delta_ci95,
                "matched_examples": 1319,
                "mcnemar_p_value": 1.0,
                "passed": True,
                "paired_outcomes": paired_outcomes,
                "paired_outcomes_sha256": paired_outcomes_sha256(paired_outcomes),
                "provenance_matched": True,
                "stock_accuracy": 709 / 1319,
                "stock_only_wins": 9,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    nsys = root / "nsys_components.csv"
    with nsys.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm",
                "cache_event",
                "cache_key",
                "call_count",
                "call_weight",
                "component",
                "mean_us",
                "signature_key",
                "tactic",
                "comparison_tactic",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "arm": "stock",
                    "cache_event": "cache hit",
                    "cache_key": "cache-1",
                    "call_count": 10,
                    "call_weight": 10,
                    "component": "FC1/GEMM1",
                    "mean_us": 60.0,
                    "signature_key": "sig-1",
                    "tactic": "1,2",
                    "comparison_tactic": "3,4",
                },
                {
                    "arm": "candidate",
                    "cache_event": "cache hit",
                    "cache_key": "cache-1",
                    "call_count": 10,
                    "call_weight": 10,
                    "component": "FC1/GEMM1",
                    "mean_us": 50.0,
                    "signature_key": "sig-1",
                    "tactic": "3,4",
                    "comparison_tactic": "3,4",
                },
                {
                    "arm": "stock",
                    "cache_event": "fallback",
                    "cache_key": "cache-1",
                    "call_count": 10,
                    "call_weight": 10,
                    "component": "FC2/GEMM2",
                    "mean_us": 40.0,
                    "signature_key": "sig-1",
                    "tactic": "1,2",
                    "comparison_tactic": "3,4",
                },
                {
                    "arm": "candidate",
                    "cache_event": "cache hit",
                    "cache_key": "cache-1",
                    "call_count": 10,
                    "call_weight": 10,
                    "component": "FC2/GEMM2",
                    "mean_us": 30.0,
                    "signature_key": "sig-1",
                    "tactic": "3,4",
                    "comparison_tactic": "3,4",
                },
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
    stock_runs = (
        write_grpo_run(
            root, arm="stock", run_id="one", throughput=9500.0, total_step_seconds=210.0
        ),
    )
    candidate_runs = (
        write_grpo_run(
            root,
            arm="candidate",
            run_id="one",
            throughput=9700.0,
            total_step_seconds=205.0,
        ),
    )
    if repeated:
        stock_runs += (
            write_grpo_run(
                root,
                arm="stock",
                run_id="two",
                throughput=9550.0,
                total_step_seconds=209.0,
            ),
        )
        candidate_runs += (
            write_grpo_run(
                root,
                arm="candidate",
                run_id="two",
                throughput=9750.0,
                total_step_seconds=204.0,
            ),
        )
    artifacts = write_audit_artifacts(root)
    cache_hashes = {
        "stock": _sha(artifacts["stock_cache"]),
        "candidate": _sha(artifacts["candidate_cache"]),
    }
    manifest_runtime = json.loads(
        artifacts["cache_manifest"].read_text(encoding="ascii")
    )["source_fingerprints"]
    assert isinstance(manifest_runtime, dict)
    for arm, runs in (("stock", stock_runs), ("candidate", candidate_runs)):
        for run in runs:
            manifest_path = run / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            manifest["cache_sha256"] = cache_hashes[arm]
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True), encoding="ascii"
            )
            evidence_path = run / "run_evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="ascii"))
            evidence["runtime_fingerprints"] = {
                **manifest_runtime,
                "cache_file_sha256": cache_hashes[arm],
                "cache_sha256": cache_hashes[arm],
                "container_sha256": "c" * 64,
                "nemo_rl_commit": "f" * 40,
            }
            evidence_path.write_text(
                json.dumps(evidence, sort_keys=True), encoding="ascii"
            )
    produced_bindings = [
        comparison_artifact_bindings(stock_run, candidate_run)
        for stock_run, candidate_run in zip(stock_runs, candidate_runs, strict=True)
    ]
    bindings = {
        arm: {
            run_id: digest
            for produced in produced_bindings
            for run_id, digest in cast(
                dict[str, str], produced[f"{arm}_run_manifests"]
            ).items()
        }
        for arm in ("stock", "candidate")
    }
    for artifact in (artifacts["correctness"], artifacts["gsm8k"]):
        payload = json.loads(artifact.read_text(encoding="ascii"))
        payload.update(
            {
                "stock_run_manifests": bindings["stock"],
                "candidate_run_manifests": bindings["candidate"],
                "comparison_run_ids": sorted(bindings["stock"]),
                "stock_arm_id": "stock",
                "candidate_arm_id": "candidate",
            }
        )
        artifact.write_text(json.dumps(payload, sort_keys=True), encoding="ascii")
    return AuditInputs(
        stock_runs=stock_runs, candidate_runs=candidate_runs, **artifacts
    )


def test_summarize_run_parses_actual_grpo_labels_and_evidence_tokens(
    tmp_path: Path,
) -> None:
    run = write_grpo_run(
        tmp_path, arm="stock", run_id="one", throughput=9500.0, total_step_seconds=210.0
    )

    summary = summarize_run(run)

    assert summary.measured_steps == 6
    assert summary.steps[0].reward == 0.5
    assert summary.steps[0].kl == 0.01
    assert summary.steps[0].loss == 0.2
    assert summary.realized_generated_tokens == 6 * 64000


def test_launcher_evidence_is_derived_from_grpo_producer_masks_and_phase_markers(
    tmp_path: Path,
) -> None:
    run = write_grpo_run(
        tmp_path,
        arm="stock",
        run_id="producer",
        throughput=9500.0,
        total_step_seconds=210.0,
    )
    (run / "run_evidence.json").unlink()
    log = run / "logs" / "ray-driver.log"
    with log.open("a", encoding="utf-8") as handle:
        for step in range(3, 9):
            for phase in ("refit", "rollout", "logprob", "train"):
                handle.write(
                    f"\n[MXFP8_MOE_AUDIT] step={step} phase={phase} status=success"
                )
            dump = run / "logs" / "exp_001" / f"train_data_step{step}.jsonl"
            dump.parent.mkdir(parents=True, exist_ok=True)
            dump.write_text(
                json.dumps({"token_loss_mask": [1, 1, 0, 1]}) + "\n", encoding="ascii"
            )

    write_run_evidence(
        run,
        arm="stock",
        run_id="producer",
        metadata={
            "batch": "fixture",
            "generation_settings": "fixture",
            "run_id": "producer",
            "run_kind": "validation",
            "topology": "fixture",
        },
        runtime_fingerprints={"producer": "fixture"},
    )

    assert summarize_run(run).realized_generated_tokens == 18


def test_nsys_producer_converts_tagged_summary_to_report_schema(tmp_path: Path) -> None:
    raw = tmp_path / "nvtx.csv"
    raw.write_text(
        "Range,Instances,Total Time (ns)\n"
        '"MXFP8_MOE_AUDIT|signature_key=sig-1|cache_key=cache-1|arm=stock|component=FC1/GEMM1|tactic=1,2|comparison_tactic=3,4|cache_event=cache hit|call_weight=10",2,120000\n',
        encoding="ascii",
    )
    output = tmp_path / "components.csv"

    convert(raw, output)

    rows = list(csv.DictReader(output.open(encoding="ascii")))
    assert rows == [
        {
            "signature_key": "sig-1",
            "cache_key": "cache-1",
            "arm": "stock",
            "component": "FC1/GEMM1",
            "tactic": "1,2",
            "comparison_tactic": "3,4",
            "cache_event": "cache hit",
            "call_weight": "10",
            "call_count": "2",
            "mean_us": "60",
        }
    ]


def test_nsys_producer_output_is_accepted_by_the_report_consumer(
    tmp_path: Path,
) -> None:
    inputs = complete_inputs(tmp_path)
    source_rows = list(csv.DictReader(inputs.nsys.open(encoding="ascii")))
    raw = tmp_path / "nvtx.csv"
    with raw.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Range", "Instances", "Total Time (ns)"]
        )
        writer.writeheader()
        for row in source_rows:
            tag = "|".join(
                (
                    "MXFP8_MOE_AUDIT",
                    f"signature_key={row['signature_key']}",
                    f"cache_key={row['cache_key']}",
                    f"arm={row['arm']}",
                    f"component={row['component']}",
                    f"tactic={row['tactic']}",
                    f"comparison_tactic={row['comparison_tactic']}",
                    f"cache_event={row['cache_event']}",
                    f"call_weight={row['call_weight']}",
                )
            )
            writer.writerow(
                {
                    "Range": tag,
                    "Instances": row["call_count"],
                    "Total Time (ns)": float(row["mean_us"])
                    * float(row["call_count"])
                    * 1000,
                }
            )

    convert(raw, inputs.nsys)

    report = build_report(inputs, tmp_path / "report")
    assert report.verdict == "KEEP", report.reasons


def test_component_plot_preserves_each_profile_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[list[str]] = []

    def capture_bars(_ax: object, data: pd.DataFrame, **_kwargs: object) -> None:
        if "Component" in data:
            observed.append(data["Component"].tolist())

    monkeypatch.setattr(plot_results, "_bars", capture_bars)
    plot_results.write_complete_plots(
        tmp_path,
        component_speedups=(
            ("FC1/GEMM1\nsignature-a", 1.1),
            ("FC1/GEMM1\nsignature-b", 1.2),
            ("FC2/GEMM2\nsignature-a", 1.3),
        ),
        tactic_change_share=0.5,
        cache_hit_share=1.0,
        normalized_throughput=1.1,
        normalized_total_step_time=0.9,
        per_step=(("one", "stock", 3, 100.0, 2.0),),
        metadata_caption="fixture",
    )

    assert observed[0] == [
        "FC1/GEMM1\nsignature-a",
        "FC1/GEMM1\nsignature-b",
        "FC2/GEMM2\nsignature-a",
    ]


def test_component_weighting_uses_profile_weight_not_nsys_instances(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            {
                "selected_profiles": [
                    {"signature_key": "sig-a", "call_count": 100},
                    {"signature_key": "sig-b", "call_count": 1},
                ]
            }
        ),
        encoding="ascii",
    )
    nsys = tmp_path / "components.csv"
    with nsys.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for component in ("FC1/GEMM1", "FC2/GEMM2"):
            for signature, key, weight, calls, stock_time, candidate_time in (
                ("sig-a", "key-a", 100, 1, 20, 10),
                ("sig-b", "key-b", 1, 100, 10, 20),
            ):
                for arm, tactic, timing in (
                    ("stock", "1,2", stock_time),
                    ("candidate", "3,4", candidate_time),
                ):
                    writer.writerow(
                        {
                            "signature_key": signature,
                            "cache_key": key,
                            "arm": arm,
                            "component": component,
                            "tactic": tactic,
                            "comparison_tactic": "3,4",
                            "cache_event": "cache hit",
                            "call_weight": weight,
                            "call_count": calls,
                            "mean_us": timing,
                        }
                    )
    inputs = AuditInputs(
        stock_runs=(),
        candidate_runs=(),
        cache_manifest=tmp_path,
        stock_cache=tmp_path,
        candidate_cache=tmp_path,
        trace_summary=tmp_path,
        qualification_decisions=tmp_path,
        selected_profiles=selected,
        shmoo=tmp_path,
        nsys=nsys,
        correctness=tmp_path,
        gsm8k=tmp_path,
    )

    components, _, _ = _component_speedups(
        inputs,
        {"sig-a": ("key-a", 0.75), "sig-b": ("key-b", 0.25)},
        {"key-a": ((1, 2), (3, 4)), "key-b": ((1, 2), (3, 4))},
    )

    assert components == (("FC1/GEMM1", 1.625), ("FC2/GEMM2", 1.625))


def test_report_binds_shuffled_shmoo_rows_and_ignores_failed_fastest_row(
    tmp_path: Path,
) -> None:
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
    assert "stock/one: batch=16 prompts x 8 generations" in html
    assert "candidate/two: batch=16 prompts x 8 generations" in html
    assert "trace-rank0.jsonl" not in markdown


def test_one_run_per_arm_is_incomplete_for_promotion_but_preserves_raw_values(
    tmp_path: Path,
) -> None:
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
    correctness = json.loads(inputs.correctness.read_text(encoding="ascii"))
    correctness["cuda_graph_replay"] = False
    inputs.correctness.write_text(json.dumps(correctness), encoding="ascii")
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


def test_fractional_nsys_call_count_is_incomplete(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path)
    rows = list(csv.DictReader(inputs.nsys.open(encoding="ascii")))
    rows[0]["call_count"] = "1.5"
    with inputs.nsys.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = build_report(inputs, tmp_path / "report")

    assert report.verdict == "INCOMPLETE"
    assert "call_count must be an integer" in report.reasons[0]


def test_gsm8k_inconsistent_accuracy_is_incomplete(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path)
    gsm8k = json.loads(inputs.gsm8k.read_text(encoding="ascii"))
    gsm8k["stock_accuracy"] = 0.99
    inputs.gsm8k.write_text(json.dumps(gsm8k), encoding="ascii")

    report = build_report(inputs, tmp_path / "report")

    assert report.verdict == "INCOMPLETE"
    assert "accuracies or delta disagree" in report.reasons[0]


def test_gsm8k_inconsistent_passed_field_is_incomplete(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path)
    gsm8k = json.loads(inputs.gsm8k.read_text(encoding="ascii"))
    gsm8k["passed"] = False
    inputs.gsm8k.write_text(json.dumps(gsm8k), encoding="ascii")

    report = build_report(inputs, tmp_path / "report")

    assert report.verdict == "INCOMPLETE"
    assert "passed field disagrees" in report.reasons[0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mcnemar_p_value", 0.25, "McNemar p-value disagrees"),
        ("mcnemar_p_value", 1.1, "must be in [0, 1]"),
        ("delta_ci95", [0.1, -0.1], "bounds are inconsistent"),
        ("delta_ci95", [-1.1, 0.1], "bounds are inconsistent"),
        ("paired_outcomes_sha256", "0" * 64, "paired outcomes disagree"),
    ],
)
def test_gsm8k_statistical_evidence_is_recomputed(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    inputs = complete_inputs(tmp_path)
    gsm8k = json.loads(inputs.gsm8k.read_text(encoding="ascii"))
    gsm8k[field] = value
    inputs.gsm8k.write_text(json.dumps(gsm8k), encoding="ascii")

    report = build_report(inputs, tmp_path / "report")

    assert report.verdict == "INCOMPLETE"
    assert message in report.reasons[0]


def test_repetitions_require_identical_generation_settings(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path)
    evidence_path = inputs.candidate_runs[1] / "run_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="ascii"))
    evidence["metadata"]["generation_settings"] = "different"
    evidence_path.write_text(json.dumps(evidence), encoding="ascii")

    report = build_report(inputs, tmp_path / "report")

    assert report.verdict == "INCOMPLETE"
    assert "generation_settings metadata differ" in report.reasons[0]


def test_observed_container_and_cache_file_hashes_bind_run_manifest(
    tmp_path: Path,
) -> None:
    inputs = complete_inputs(tmp_path)
    evidence_path = inputs.stock_runs[0] / "run_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="ascii"))
    evidence["runtime_fingerprints"]["container_sha256"] = "0" * 64
    evidence_path.write_text(json.dumps(evidence), encoding="ascii")

    report = build_report(inputs, tmp_path / "report")

    assert report.verdict == "INCOMPLETE"
    assert "independently observed runtime fingerprints" in report.reasons[0]


def test_runtime_mismatch_is_rejected_as_incomplete(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path)
    evidence_path = inputs.candidate_runs[0] / "run_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="ascii"))
    evidence["runtime_fingerprints"]["gpu_name"] = "NVIDIA H100"
    evidence_path.write_text(json.dumps(evidence), encoding="ascii")

    report = build_report(inputs, tmp_path / "report")

    assert report.verdict == "INCOMPLETE"
    assert "independently observed runtime fingerprints" in report.reasons[0]


def test_unrelated_correctness_or_gsm8k_binding_is_incomplete(tmp_path: Path) -> None:
    inputs = complete_inputs(tmp_path)
    for artifact in (inputs.correctness, inputs.gsm8k):
        payload = json.loads(artifact.read_text(encoding="ascii"))
        payload["comparison_run_ids"] = ["unrelated"]
        artifact.write_text(json.dumps(payload), encoding="ascii")

    report = build_report(inputs, tmp_path / "report")

    assert report.verdict == "INCOMPLETE"
    assert "does not bind exact stock/candidate run artifacts" in report.reasons[0]


def test_shared_comparison_ids_are_valid_but_per_arm_cache_mismatch_is_not(
    tmp_path: Path,
) -> None:
    inputs = complete_inputs(tmp_path)
    assert build_report(inputs, tmp_path / "valid-report").verdict == "KEEP"
    manifest_path = inputs.stock_runs[1] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["cache_sha256"] = "9" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    report = build_report(inputs, tmp_path / "report")

    assert report.verdict == "INCOMPLETE"
    assert "stock repetitions do not share one cache identity" in report.reasons[0]


def test_missing_artifacts_render_not_yet_executed_without_numeric_claims(
    tmp_path: Path,
) -> None:
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
