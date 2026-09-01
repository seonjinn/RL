from __future__ import annotations

import importlib.util
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "experiments" / "vllm_028_nemotron_bf16_matrix"
REPORT_PATH = PACKAGE_ROOT / "build_mrv2_patch_matrix_report.py"
FIXED_K_CONTAINER_DIGEST = (
    "sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4"
)


def load_report() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_build_mrv2_patch_matrix_report", REPORT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixed_k_runtime_fields(model: str = "super") -> dict[str, str]:
    return {
        "tensor_parallel_size": "2" if model == "super" else "8",
        "expert_parallel": "False" if model == "super" else "True",
        "vllm_commit": "2cf0a69",
        "container_digest": FIXED_K_CONTAINER_DIGEST,
        "harness_commit": "a" * 40,
        "cuda_graph_capture_completed": "83",
        "cuda_graph_capture_total": "83",
    }


def payload(*, method: str, tok_s_gpu: float) -> dict[str, Any]:
    dynamic = method == "mtp_dynamic_max_k5"
    metrics = (
        {
            "num_drafts": 10.0,
            "num_draft_tokens": 30.0,
            "num_accepted_tokens": 18.0,
            "acceptance_rate": 0.6,
            "mean_acceptance_length": 2.8,
        }
        if dynamic
        else {}
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "runtime": {
            "vllm_version": "0.28.0",
            "environment": {"SLURM_JOB_ID": "123" if dynamic else "122"},
        },
        "config": {
            "model_key": "super",
            "method_key": method,
            "runner_key": "mrv2",
            "model": (
                "/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
                "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e"
            ),
            "isl": 1000,
            "osl": 10000,
            "batch_size": 8,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "engine_gpus": 2,
            "total_gpus": 2,
            "enable_expert_parallel": False,
            "dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "enforce_eager": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": 42,
            "warmup_repeats": 1,
            "measure_repeats": 1,
            "max_num_seqs": 512,
            "max_num_batched_tokens": 32768,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": 0.9,
            "disable_custom_all_reduce": False,
            "attention_backend": "auto",
            "moe_backend": "flashinfer_trtllm",
            "mamba_ssm_cache_dtype": "float32",
            "mamba_backend": "flashinfer",
            "model_loader_extra_config": {
                "enable_multithread_load": True,
                "num_threads": 48,
            },
            "k_selection_basis": ("active_scheduled_batch" if dynamic else "baseline"),
            "effective_k": None if dynamic else 0,
            "requested_batch_schedule_k": 3 if dynamic else None,
            "speculative_config": (
                {
                    "method": "mtp",
                    "num_speculative_tokens": 5,
                    "num_speculative_tokens_per_batch_size": [
                        [1, 4, 5],
                        [5, 16, 3],
                        [17, 64, 2],
                        [65, 128, 1],
                        [129, 512, 0],
                    ],
                }
                if dynamic
                else None
            ),
        },
        "results": [
            {
                "bs": 8,
                "latency_s": 100.0,
                "output_tok_s": tok_s_gpu * 2,
                "output_tok_s_per_gpu": tok_s_gpu,
                "expected_output_tokens": 80000,
                "actual_output_tokens": 80000,
                "tokens_ok": True,
                "spec_decode_metrics": metrics,
            }
        ],
        "summary": {"tokens_ok": True, "spec_decode_metrics": metrics},
        "runtime_provenance": {
            "vllm_base_commit": "2cf0a6915ce544dc493a0990f2ea38d81601128a",
            "patched_vllm_head": "bf0719d13bbc74da8af86d2b326f7b16ba4f7462",
            "patchset_manifest_sha256": (
                "238e2ffcc14d2fb2f0fc07c419004820efa9a3cfd284c3a41515e84e39aecc25"
            ),
            "container_artifact_sha256": (
                "5ae5c3e3d630d95e1129b71384fe9c5c437a77288492ada30da94f93b8582066"
            ),
            "ray_version": None,
            "harness_commit": "f3a842c0ad3c91f97e37b4f58ef84bd9148e281a",
            "harness_manifest_sha256": (
                "d182f2e26920c6c68e1961d13cbfec60340cf4e3394f65ac3d560589146fc873"
            ),
        },
    }


def evidence(*, dynamic: bool) -> dict[str, Any]:
    return {
        "method_key": "mtp_dynamic_max_k5" if dynamic else "baseline",
        "piecewise_completed": [
            {
                "line": "Capturing CUDA graphs (PIECEWISE): 83/83",
                "count": 83,
            },
            *(
                [
                    {
                        "line": "Capturing prefill CUDA graphs (PIECEWISE): 83/83",
                        "count": 83,
                    }
                ]
                if dynamic
                else []
            ),
        ],
        "full_completed": [
            {"line": "Capturing CUDA graphs (FULL): 51/51", "count": 51},
            *(
                [
                    {
                        "line": "Capturing prefill CUDA graphs (FULL): 375/375",
                        "count": 375,
                    }
                ]
                if dynamic
                else []
            ),
        ],
        "drafter_decode_completed": (
            [
                {
                    "line": "Capturing decode CUDA graphs (FULL): 51/51",
                    "count": 51,
                }
            ]
            if dynamic
            else []
        ),
    }


def test_normalize_rows_matches_baseline_and_reports_active_batch_width() -> None:
    report = load_report()

    rows = report.normalize_rows(
        [
            (payload(method="baseline", tok_s_gpu=100.0), evidence(dynamic=False)),
            (
                payload(method="mtp_dynamic_max_k5", tok_s_gpu=175.0),
                evidence(dynamic=True),
            ),
        ]
    )

    baseline, dynamic = rows
    assert baseline["throughput_speedup"] == pytest.approx(1.0)
    assert dynamic["throughput_speedup"] == pytest.approx(1.75)
    assert dynamic["requested_batch_schedule_k"] == 3
    assert dynamic["mean_draft_width"] == pytest.approx(3.0)
    assert dynamic["acceptance_rate"] == pytest.approx(0.6)
    assert dynamic["mean_accepted_length"] == pytest.approx(2.8)
    assert dynamic["cuda_graph_verified"] is True
    assert dynamic["drafter_decode_full_verified"] is True


def test_normalize_rows_rejects_dynamic_without_drafter_decode_graph() -> None:
    report = load_report()
    incomplete_evidence = evidence(dynamic=True)
    incomplete_evidence["drafter_decode_completed"] = []

    with pytest.raises(ValueError, match="drafter decode"):
        report.normalize_rows(
            [
                (payload(method="baseline", tok_s_gpu=100.0), evidence(dynamic=False)),
                (
                    payload(method="mtp_dynamic_max_k5", tok_s_gpu=175.0),
                    incomplete_evidence,
                ),
            ]
        )


def test_normalize_rows_rejects_unmatched_performance_configuration() -> None:
    report = load_report()
    baseline = payload(method="baseline", tok_s_gpu=100.0)
    baseline["config"]["model_loader_extra_config"]["num_threads"] = 47

    with pytest.raises(ValueError, match="missing matched baseline"):
        report.normalize_rows(
            [
                (baseline, evidence(dynamic=False)),
                (
                    payload(method="mtp_dynamic_max_k5", tok_s_gpu=175.0),
                    evidence(dynamic=True),
                ),
            ]
        )


def test_normalize_rows_rejects_incomplete_capture_identity() -> None:
    report = load_report()
    incomplete_evidence = evidence(dynamic=True)
    incomplete_evidence["piecewise_completed"] = [
        {"line": "Capturing CUDA graphs (PIECEWISE): 83/83", "count": 83}
    ]

    with pytest.raises(ValueError, match="drafter prefill PIECEWISE"):
        report.normalize_rows(
            [
                (payload(method="baseline", tok_s_gpu=100.0), evidence(dynamic=False)),
                (
                    payload(method="mtp_dynamic_max_k5", tok_s_gpu=175.0),
                    incomplete_evidence,
                ),
            ]
        )


def test_normalize_rows_rejects_unapproved_harness_mismatch() -> None:
    report = load_report()
    baseline = payload(method="baseline", tok_s_gpu=100.0)
    baseline["runtime_provenance"]["harness_commit"] = (
        "85cfe1a8a2f7c97684e65e5428a679cab7058142"
    )
    baseline["runtime_provenance"]["harness_manifest_sha256"] = (
        "86012387865c84b68150f0aa2e758dbb7b5ccc75be07ec15dc1c9663d6b87147"
    )

    with pytest.raises(ValueError, match="harness provenance mismatch"):
        report.normalize_rows(
            [
                (baseline, evidence(dynamic=False)),
                (
                    payload(method="mtp_dynamic_max_k5", tok_s_gpu=175.0),
                    evidence(dynamic=True),
                ),
            ]
        )


def test_load_cells_rejects_duplicate_canonical_results(tmp_path: Path) -> None:
    report = load_report()
    leaf = tmp_path / "super" / "isl1k_osl10k" / "baseline" / "bs8"
    for job_id in ("1", "2"):
        job_dir = leaf / f"job-{job_id}"
        job_dir.mkdir(parents=True)
        (job_dir / "result.json").write_text(json.dumps({}), encoding="utf-8")
        (job_dir / "cuda_graph_evidence.json").write_text(
            json.dumps({}), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="exactly one canonical result"):
        report.load_cells(
            tmp_path, [report.ResultKey("super", 1000, 10000, "baseline", 8)]
        )


def test_normalize_fixed_k_rows_recomputes_matched_speedups() -> None:
    report = load_report()
    raw_rows = [
        {
            "model": "super",
            "isl": "1000",
            "osl": "10000",
            "method": "baseline",
            "batch_size": "32",
            "output_tok_s_per_gpu": "100.0",
            "speedup_vs_baseline": "1.0",
            "tokens_ok": "True",
            "actual_output_tokens": "320000",
            "expected_output_tokens": "320000",
            "weight_dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "runner": "mrv1",
            "vllm_version": "0.28.0",
            "cudagraph_mode": "PIECEWISE",
            "enforce_eager": "False",
            "cuda_graph_verified": "True",
            **fixed_k_runtime_fields(),
        },
        *[
            {
                "model": "super",
                "isl": "1000",
                "osl": "10000",
                "method": f"mtp_static_k{k}",
                "batch_size": "32",
                "output_tok_s_per_gpu": str(value),
                "speedup_vs_baseline": str(value / 100.0),
                "tokens_ok": "True",
                "actual_output_tokens": "320000",
                "expected_output_tokens": "320000",
                "weight_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "runner": "mrv1",
                "vllm_version": "0.28.0",
                "cudagraph_mode": "PIECEWISE",
                "enforce_eager": "False",
                "cuda_graph_verified": "True",
                **fixed_k_runtime_fields(),
            }
            for k, value in enumerate((130.0, 180.0, 258.0, 225.0, 226.0), start=1)
        ],
    ]

    rows = report.normalize_fixed_k_rows(raw_rows, require_complete_matrix=False)

    assert [row["k"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["throughput_speedup"] for row in rows] == pytest.approx(
        [1.30, 1.80, 2.58, 2.25, 2.26]
    )
    assert report.best_fixed_k(rows)[0] == 3


def test_normalize_fixed_k_rows_rejects_inexact_tokens() -> None:
    report = load_report()
    raw_rows = [
        {
            "model": "super",
            "isl": "1000",
            "osl": "10000",
            "method": "baseline",
            "batch_size": "1",
            "output_tok_s_per_gpu": "100.0",
            "speedup_vs_baseline": "1.0",
            "tokens_ok": "True",
            "actual_output_tokens": "9999",
            "expected_output_tokens": "10000",
            "weight_dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "runner": "mrv1",
            "vllm_version": "0.28.0",
            "cudagraph_mode": "PIECEWISE",
            "enforce_eager": "False",
            "cuda_graph_verified": "True",
            **fixed_k_runtime_fields(),
        }
    ]

    with pytest.raises(ValueError, match="exact output tokens"):
        report.normalize_fixed_k_rows(raw_rows, require_complete_matrix=False)


@pytest.mark.parametrize(
    ("field", "value"),
    (("tensor_parallel_size", "8"), ("cuda_graph_capture_completed", "82")),
)
def test_normalize_fixed_k_rows_rejects_topology_or_incomplete_graph_capture(
    field: str, value: str
) -> None:
    report = load_report()
    row = {
        "model": "super",
        "isl": "1000",
        "osl": "10000",
        "method": "baseline",
        "batch_size": "1",
        "output_tok_s_per_gpu": "100.0",
        "speedup_vs_baseline": "1.0",
        "tokens_ok": "True",
        "actual_output_tokens": "10000",
        "expected_output_tokens": "10000",
        "weight_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "runner": "mrv1",
        "vllm_version": "0.28.0",
        "cudagraph_mode": "PIECEWISE",
        "enforce_eager": "False",
        "cuda_graph_verified": "True",
        **fixed_k_runtime_fields(),
    }
    row[field] = value

    with pytest.raises(ValueError, match="topology, provenance, or graph capture"):
        report.normalize_fixed_k_rows([row], require_complete_matrix=False)


def test_normalize_mrv1_dynamic_rows_recomputes_matched_speedup() -> None:
    report = load_report()
    common = {
        "model": "super",
        "isl": "1000",
        "osl": "10000",
        "batch_size": "8",
        "tokens_ok": "True",
        "actual_output_tokens": "80000",
        "expected_output_tokens": "80000",
        "weight_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "runner": "mrv1",
        "vllm_version": "0.28.0",
        "cudagraph_mode": "PIECEWISE",
        "enforce_eager": "False",
        "cuda_graph_verified": "True",
        **fixed_k_runtime_fields(),
    }
    raw_rows = [
        {
            **common,
            "method": "baseline",
            "output_tok_s_per_gpu": "100.0",
            "speedup_vs_baseline": "1.0",
        },
        {
            **common,
            "method": "mtp_dynamic_max_k5",
            "output_tok_s_per_gpu": "180.0",
            "speedup_vs_baseline": "1.8",
            "acceptance_rate": "0.75",
            "mean_acceptance_length": "3.5",
            "requested_batch_schedule_k": "3",
            "k_selection_basis": "active_scheduled_batch",
        },
    ]

    rows = report.normalize_mrv1_dynamic_rows(raw_rows, require_complete_matrix=False)

    assert rows == [
        {
            "model": "super",
            "isl": 1000,
            "osl": 10000,
            "concurrency": 8,
            "tok_s_gpu": 180.0,
            "baseline_tok_s_gpu": 100.0,
            "throughput_speedup": 1.8,
            "acceptance_rate": 0.75,
            "mean_accepted_length": 3.5,
            "job_id": "",
        }
    ]


def test_normalize_mrv1_dynamic_rows_requires_all_32_settings() -> None:
    report = load_report()

    with pytest.raises(ValueError, match="32 baselines and 32 DynamicSD rows"):
        report.normalize_mrv1_dynamic_rows([], require_complete_matrix=True)


def test_normalize_mrv1_dynamic_rows_rejects_unapproved_harness_pair() -> None:
    report = load_report()
    common = {
        "model": "super",
        "isl": "1000",
        "osl": "10000",
        "batch_size": "8",
        "tokens_ok": "True",
        "actual_output_tokens": "80000",
        "expected_output_tokens": "80000",
        "weight_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "runner": "mrv1",
        "vllm_version": "0.28.0",
        "cudagraph_mode": "PIECEWISE",
        "enforce_eager": "False",
        "cuda_graph_verified": "True",
        **fixed_k_runtime_fields(),
    }
    raw_rows = [
        {
            **common,
            "method": "baseline",
            "output_tok_s_per_gpu": "100.0",
            "speedup_vs_baseline": "1.0",
            "harness_commit": "a" * 40,
        },
        {
            **common,
            "method": "mtp_dynamic_max_k5",
            "output_tok_s_per_gpu": "180.0",
            "speedup_vs_baseline": "1.8",
            "acceptance_rate": "0.75",
            "mean_acceptance_length": "3.5",
            "requested_batch_schedule_k": "3",
            "k_selection_basis": "active_scheduled_batch",
            "harness_commit": "b" * 40,
        },
    ]

    with pytest.raises(ValueError, match="harness provenance mismatch"):
        report.normalize_mrv1_dynamic_rows(raw_rows, require_complete_matrix=False)


def test_relative_href_tracks_actual_output_and_input_paths(tmp_path: Path) -> None:
    report = load_report()
    output_html = tmp_path / "public" / "reports" / "report.html"
    input_csv = tmp_path / "artifacts" / "fixed-k" / "results.csv"

    assert (
        report.relative_href(output_html, input_csv)
        == "../../artifacts/fixed-k/results.csv"
    )


class _TechniquePageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.h1 = 0
        self.h2 = 0
        self.svg = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "h1":
            self.h1 += 1
        elif tag == "h2":
            self.h2 += 1
        elif tag == "svg":
            self.svg += 1


def test_render_html_explains_schedule_fixed_k_and_cohort_boundaries() -> None:
    report = load_report()
    mrv2_rows = report.normalize_rows(
        [
            (payload(method="baseline", tok_s_gpu=100.0), evidence(dynamic=False)),
            (
                payload(method="mtp_dynamic_max_k5", tok_s_gpu=175.0),
                evidence(dynamic=True),
            ),
        ]
    )
    fixed_k_rows = report.normalize_fixed_k_rows(
        [
            {
                "model": "super",
                "isl": "1000",
                "osl": "10000",
                "method": "baseline",
                "batch_size": "8",
                "output_tok_s_per_gpu": "100.0",
                "speedup_vs_baseline": "1.0",
                "tokens_ok": "True",
                "actual_output_tokens": "80000",
                "expected_output_tokens": "80000",
                "weight_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "runner": "mrv1",
                "vllm_version": "0.28.0",
                "cudagraph_mode": "PIECEWISE",
                "enforce_eager": "False",
                "cuda_graph_verified": "True",
                **fixed_k_runtime_fields(),
            },
            *[
                {
                    "model": "super",
                    "isl": "1000",
                    "osl": "10000",
                    "method": f"mtp_static_k{k}",
                    "batch_size": "8",
                    "output_tok_s_per_gpu": str(value),
                    "speedup_vs_baseline": str(value / 100.0),
                    "tokens_ok": "True",
                    "actual_output_tokens": "80000",
                    "expected_output_tokens": "80000",
                    "weight_dtype": "bfloat16",
                    "kv_cache_dtype": "fp8",
                    "runner": "mrv1",
                    "vllm_version": "0.28.0",
                    "cudagraph_mode": "PIECEWISE",
                    "enforce_eager": "False",
                    "cuda_graph_verified": "True",
                    **fixed_k_runtime_fields(),
                }
                for k, value in enumerate((150.0, 190.0, 240.0, 246.0, 325.0), start=1)
            ],
        ],
        require_complete_matrix=False,
    )
    mrv1_dynamic_rows = report.normalize_mrv1_dynamic_rows(
        [
            {
                "model": "super",
                "isl": "1000",
                "osl": "10000",
                "method": "baseline",
                "batch_size": "8",
                "output_tok_s_per_gpu": "100.0",
                "speedup_vs_baseline": "1.0",
                "tokens_ok": "True",
                "actual_output_tokens": "80000",
                "expected_output_tokens": "80000",
                "weight_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "runner": "mrv1",
                "vllm_version": "0.28.0",
                "cudagraph_mode": "PIECEWISE",
                "enforce_eager": "False",
                "cuda_graph_verified": "True",
                **fixed_k_runtime_fields(),
            },
            {
                "model": "super",
                "isl": "1000",
                "osl": "10000",
                "method": "mtp_dynamic_max_k5",
                "batch_size": "8",
                "output_tok_s_per_gpu": "180.0",
                "speedup_vs_baseline": "1.8",
                "acceptance_rate": "0.75",
                "mean_acceptance_length": "3.5",
                "requested_batch_schedule_k": "3",
                "k_selection_basis": "active_scheduled_batch",
                "tokens_ok": "True",
                "actual_output_tokens": "80000",
                "expected_output_tokens": "80000",
                "weight_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "runner": "mrv1",
                "vllm_version": "0.28.0",
                "cudagraph_mode": "PIECEWISE",
                "enforce_eager": "False",
                "cuda_graph_verified": "True",
                **fixed_k_runtime_fields(),
            },
        ],
        require_complete_matrix=False,
    )

    rendered = report.render_html(
        mrv2_rows,
        fixed_k_rows=fixed_k_rows,
        mrv1_dynamic_rows=mrv1_dynamic_rows,
        csv_href="../data/mrv2/results.csv",
        fixed_k_csv_href="../data/mrv1/results.csv",
        source_manifest_href="../data/mrv2/source_manifest.json",
    )
    parser = _TechniquePageParser()
    parser.feed(rendered)

    assert (parser.h1, parser.h2, parser.svg) == (1, 3, 1)
    assert "Does it work?" in rendered
    assert "active scheduled batch" in rendered
    assert "num_speculative_tokens_per_batch_size" in rendered
    assert "K1" in rendered and "K5" in rendered
    assert "K5 (3.25×)" in rendered
    assert "DynamicSD (MRV1)" in rendered
    assert "DynamicSD (MRV2)" in rendered
    assert "1.80×" in rendered
    assert "1.75×" in rendered
    assert "MRV1 fixed-K" in rendered and "MRV2 DynamicMTP" in rendered
    assert "C=512 proves positive-K drafting occurred" in rendered
    assert "most admitted decode work occurred" not in rendered
    assert "../data/mrv1/results.csv" in rendered
    assert "../data/mrv2/source_manifest.json" in rendered
