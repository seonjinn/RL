from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "experiments/vllm_028_nemotron_bf16_matrix"


def load_module(module_name: str) -> ModuleType:
    path = PACKAGE_ROOT / f"{module_name}.py"
    assert path.is_file(), f"missing reporting module: {path}"
    spec = importlib.util.spec_from_file_location(
        f"vllm028_nemotron_bf16_{module_name}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(
    *,
    model: str,
    method: str,
    batch_size: int,
    tok_s_per_gpu: float,
    job_id: str,
) -> dict[str, Any]:
    effective_k = 0 if method == "baseline" else 4
    metrics: dict[str, Any] = {}
    if method != "baseline":
        metrics = {
            "acceptance_rate": 0.75,
            "mean_acceptance_length": 3.5,
            "num_drafts": 100.0,
            "num_draft_tokens": 400.0,
            "num_accepted_tokens": 300.0,
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "config": {
            "model_key": model,
            "method_key": method,
            "runner_key": "mrv1",
            "isl": 1000,
            "osl": 10000,
            "batch_size": batch_size,
            "effective_k": effective_k,
            "k_selection_basis": "baseline" if method == "baseline" else "static",
            "dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "tensor_parallel_size": 2,
            "enable_expert_parallel": False,
            "cudagraph_mode": "PIECEWISE",
            "enforce_eager": False,
        },
        "results": [
            {
                "bs": batch_size,
                "actual_output_tokens": batch_size * 10000,
                "expected_output_tokens": batch_size * 10000,
                "output_tok_s": tok_s_per_gpu * 2,
                "output_tok_s_per_gpu": tok_s_per_gpu,
                "latency_s": 10.0,
                "tokens_ok": True,
                "spec_decode_metrics": metrics,
            }
        ],
        "runtime": {
            "vllm_version": "0.28.0",
            "environment": {"SLURM_JOB_ID": job_id},
        },
        "runtime_provenance": {
            "vllm_commit": "2cf0a69",
            "container_digest": "sha256:test",
            "harness_commit": "deadbeef",
        },
    }


def test_expected_mrv1_keys_cover_contiguous_k0_through_k5_and_dynamic() -> None:
    reporting = load_module("build_public_report")
    contract = load_module("contract").build_contract_matrix()

    keys = reporting.expected_mrv1_keys(contract)

    assert len(keys) == 224
    assert {key.method_key for key in keys} == {
        "baseline",
        "mtp_static_k1",
        "mtp_static_k2",
        "mtp_static_k3",
        "mtp_static_k4",
        "mtp_static_k5",
        "mtp_dynamic_max_k5",
    }


def test_parse_cuda_graph_evidence_requires_observed_piecewise_capture() -> None:
    reporting = load_module("build_public_report")
    log = """
enforce_eager=False
cudagraph_mode: <CUDAGraphMode.PIECEWISE: 1>
Profiling CUDA graph memory: PIECEWISE=83 (largest=1024)
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|x| 83/83
CUDA graph pool memory: 1.49 GiB (actual), 1.36 GiB (estimated)
"""

    evidence = reporting.parse_cuda_graph_evidence(log)

    assert evidence["verified"] is True
    assert evidence["capture_completed"] == 83
    assert evidence["capture_total"] == 83
    assert evidence["pool_memory_gib"] == 1.49


def test_normalize_rows_matches_baseline_and_computes_speedup() -> None:
    reporting = load_module("build_public_report")
    baseline = payload(
        model="super",
        method="baseline",
        batch_size=1,
        tok_s_per_gpu=100.0,
        job_id="1",
    )
    k4 = payload(
        model="super",
        method="mtp_static_k4",
        batch_size=1,
        tok_s_per_gpu=175.0,
        job_id="2",
    )
    logs = {
        "1": (
            "enforce_eager=False PIECEWISE=83 "
            "Capturing CUDA graphs (PIECEWISE): 83/83"
        ),
        "2": (
            "enforce_eager=False PIECEWISE=83 "
            "Capturing CUDA graphs (PIECEWISE): 83/83"
        ),
    }

    rows = reporting.normalize_rows([baseline, k4], logs_by_job_id=logs)

    assert rows[0]["speedup_vs_baseline"] == 1.0
    assert rows[1]["speedup_vs_baseline"] == 1.75
    assert rows[1]["acceptance_rate"] == 0.75
    assert rows[1]["mean_acceptance_length"] == 3.5
    assert all(row["cuda_graph_verified"] for row in rows)


def test_normalize_rows_rejects_non_exact_token_output() -> None:
    reporting = load_module("build_public_report")
    broken = payload(
        model="super",
        method="baseline",
        batch_size=1,
        tok_s_per_gpu=100.0,
        job_id="1",
    )
    broken["results"][0]["actual_output_tokens"] = 9999

    try:
        reporting.normalize_rows([broken], logs_by_job_id={"1": ""})
    except ValueError as exc:
        assert "exact output-token" in str(exc)
    else:
        raise AssertionError("non-exact output must be rejected")


def test_load_payloads_uses_canonical_hierarchy(tmp_path: Path) -> None:
    reporting = load_module("build_public_report")
    key = reporting.ResultKey(
        "super", 1000, 10000, "mrv1", "baseline", 1
    )
    result_path = (
        tmp_path
        / "super"
        / "isl1000_osl10000"
        / "mrv1"
        / "baseline"
        / "bs1"
        / "result.json"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            payload(
                model="super",
                method="baseline",
                batch_size=1,
                tok_s_per_gpu=100.0,
                job_id="1",
            )
        ),
        encoding="utf-8",
    )

    loaded = reporting.load_payloads(tmp_path, [key])

    assert len(loaded) == 1
    assert loaded[0]["config"]["method_key"] == "baseline"


def test_render_html_documents_dynamic_config_native_depth_and_rows() -> None:
    reporting = load_module("build_public_report")
    rows = reporting.normalize_rows(
        [
            payload(
                model="super",
                method="baseline",
                batch_size=1,
                tok_s_per_gpu=100.0,
                job_id="1",
            ),
            payload(
                model="super",
                method="mtp_static_k4",
                batch_size=1,
                tok_s_per_gpu=175.0,
                job_id="2",
            ),
        ],
        logs_by_job_id={
            "1": "enforce_eager=False PIECEWISE=83 Capturing CUDA graphs (PIECEWISE): 83/83",
            "2": "enforce_eager=False PIECEWISE=83 Capturing CUDA graphs (PIECEWISE): 83/83",
        },
    )
    schedule = [
        {"start": 1, "end": 4, "k": 5},
        {"start": 5, "end": 512, "k": 0},
    ]

    html = reporting.render_html(rows, dynamic_schedule=schedule)

    assert "num_nextn_predict_layers=1" in html
    assert "active scheduled batch" in html
    assert "mtp_static_k4" in html
    assert "1–4" in html
    assert "CUDA Graph" in html


def test_build_artifacts_writes_normalized_json_and_html(tmp_path: Path) -> None:
    reporting = load_module("build_public_report")
    key = reporting.ResultKey(
        "super", 1000, 10000, "mrv1", "baseline", 1
    )
    source = reporting.result_path(tmp_path / "results", key)
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            payload(
                model="super",
                method="baseline",
                batch_size=1,
                tok_s_per_gpu=100.0,
                job_id="1",
            )
        ),
        encoding="utf-8",
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "slurm-1.out").write_text(
        "enforce_eager=False PIECEWISE=83 "
        "Capturing CUDA graphs (PIECEWISE): 83/83",
        encoding="utf-8",
    )
    output_json = tmp_path / "normalized.json"
    output_html = tmp_path / "report.html"

    reporting.build_artifacts(
        result_root=tmp_path / "results",
        log_dir=log_dir,
        output_json=output_json,
        output_html=output_html,
        keys=[key],
        dynamic_schedule=[{"start": 1, "end": 512, "k": 0}],
    )

    assert json.loads(output_json.read_text(encoding="utf-8"))[0]["job_id"] == "1"
    assert "Canonical rows" in output_html.read_text(encoding="utf-8")
