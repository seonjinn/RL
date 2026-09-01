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
    revisions = {
        "super": "d51eab0d1f979ebc26b546e634a04f450d99158e",
        "ultra": "624ba927cfbef0427354998700de3d51173c8c04",
    }
    model_names = {
        "super": "NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "ultra": "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    }
    tp = 2 if model == "super" else 8
    effective_k: int | None = 0 if method == "baseline" else 4
    speculative_config = None
    if method.startswith("mtp_static_k"):
        effective_k = int(method.removeprefix("mtp_static_k"))
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": effective_k,
        }
    elif method == "mtp_dynamic_max_k5":
        effective_k = None
        speculative_config = {
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
            "model": f"/checkpoints/{model_names[model]}/snapshots/{revisions[model]}",
            "method_key": method,
            "speculative_config": speculative_config,
            "runner_key": "mrv1",
            "isl": 1000,
            "osl": 10000,
            "batch_size": batch_size,
            "effective_k": effective_k,
            "requested_batch_schedule_k": (
                5 if method == "mtp_dynamic_max_k5" else None
            ),
            "k_selection_basis": (
                "baseline"
                if method == "baseline"
                else "active_scheduled_batch"
                if method == "mtp_dynamic_max_k5"
                else "static"
            ),
            "dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": 1,
            "engine_gpus": tp,
            "total_gpus": tp,
            "enable_expert_parallel": model == "ultra",
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
            "container_digest": (
                "sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4"
            ),
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


def test_parse_cuda_graph_evidence_rejects_full_only_capture() -> None:
    reporting = load_module("build_public_report")
    log = """
enforce_eager=False
Profiling CUDA graph memory: PIECEWISE=83 (largest=1024)
Capturing CUDA graphs (FULL): 100%|x| 1/1
"""

    evidence = reporting.parse_cuda_graph_evidence(log)

    assert evidence["verified"] is False


def test_parse_cuda_graph_evidence_requires_profiled_capture_count() -> None:
    reporting = load_module("build_public_report")
    log = """
enforce_eager=False
Profiling CUDA graph memory: PIECEWISE=83 (largest=1024)
Capturing CUDA graphs (PIECEWISE): 100%|x| 1/1
"""

    evidence = reporting.parse_cuda_graph_evidence(log)

    assert evidence["verified"] is False


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


def test_normalize_rows_accepts_legacy_baseline_without_selection_basis() -> None:
    reporting = load_module("build_public_report")
    baseline = payload(
        model="super",
        method="baseline",
        batch_size=1,
        tok_s_per_gpu=100.0,
        job_id="1",
    )
    baseline["config"].pop("k_selection_basis")

    rows = reporting.normalize_rows(
        [baseline],
        logs_by_job_id={
            "1": "enforce_eager=False PIECEWISE=83 "
            "Capturing CUDA graphs (PIECEWISE): 83/83"
        },
    )

    assert rows[0]["effective_k"] == 0


def test_normalize_rows_canonicalizes_legacy_static_and_dynamic_k_provenance() -> None:
    reporting = load_module("build_public_report")
    baseline = payload(
        model="super",
        method="baseline",
        batch_size=1,
        tok_s_per_gpu=100.0,
        job_id="1",
    )
    static = payload(
        model="super",
        method="mtp_static_k4",
        batch_size=1,
        tok_s_per_gpu=170.0,
        job_id="2",
    )
    static["config"].pop("k_selection_basis")
    dynamic = payload(
        model="super",
        method="mtp_dynamic_max_k5",
        batch_size=1,
        tok_s_per_gpu=180.0,
        job_id="3",
    )
    dynamic["config"]["effective_k"] = 5
    dynamic["config"].pop("requested_batch_schedule_k")
    dynamic["config"].pop("k_selection_basis")
    log = "enforce_eager=False PIECEWISE=83 Capturing CUDA graphs (PIECEWISE): 83/83"

    rows = reporting.normalize_rows(
        [baseline, static, dynamic],
        logs_by_job_id={"1": log, "2": log, "3": log},
    )

    assert rows[1]["k_selection_basis"] == "static"
    assert rows[2]["effective_k"] is None
    assert rows[2]["requested_batch_schedule_k"] == 5
    assert rows[2]["k_selection_basis"] == "active_scheduled_batch"


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


def test_normalize_rows_rejects_method_k_and_batch_drift() -> None:
    reporting = load_module("build_public_report")
    wrong_k = payload(
        model="super",
        method="mtp_static_k4",
        batch_size=1,
        tok_s_per_gpu=175.0,
        job_id="2",
    )
    wrong_k["config"]["speculative_config"]["num_speculative_tokens"] = 5
    wrong_k["results"][0]["bs"] = 2

    try:
        reporting.normalize_rows([wrong_k], logs_by_job_id={"2": ""})
    except ValueError as exc:
        assert "method" in str(exc) or "batch size" in str(exc)
    else:
        raise AssertionError("method K and batch drift must be rejected")


def test_normalize_rows_rejects_runtime_and_topology_drift() -> None:
    reporting = load_module("build_public_report")
    baseline = payload(
        model="super",
        method="baseline",
        batch_size=1,
        tok_s_per_gpu=100.0,
        job_id="1",
    )
    baseline["runtime"]["vllm_version"] = "0.27.0"
    baseline["config"]["tensor_parallel_size"] = 8

    try:
        reporting.normalize_rows([baseline], logs_by_job_id={"1": ""})
    except ValueError as exc:
        assert "runtime" in str(exc) or "topology" in str(exc)
    else:
        raise AssertionError("runtime and topology drift must be rejected")


def test_normalize_rows_rejects_checkpoint_and_container_drift() -> None:
    reporting = load_module("build_public_report")
    baseline = payload(
        model="super",
        method="baseline",
        batch_size=1,
        tok_s_per_gpu=100.0,
        job_id="1",
    )
    baseline["config"]["model"] = "/checkpoints/wrong-model"
    baseline["runtime_provenance"]["container_digest"] = "sha256:wrong"

    try:
        reporting.normalize_rows([baseline], logs_by_job_id={"1": ""})
    except ValueError as exc:
        assert "checkpoint" in str(exc) or "container" in str(exc)
    else:
        raise AssertionError("checkpoint and container drift must be rejected")


def test_normalize_rows_rejects_empty_or_duplicate_job_ids() -> None:
    reporting = load_module("build_public_report")
    first = payload(
        model="super",
        method="baseline",
        batch_size=1,
        tok_s_per_gpu=100.0,
        job_id="9",
    )
    second = payload(
        model="super",
        method="mtp_static_k4",
        batch_size=1,
        tok_s_per_gpu=175.0,
        job_id="9",
    )

    try:
        reporting.normalize_rows([first, second], logs_by_job_id={"9": ""})
    except ValueError as exc:
        assert "job ID" in str(exc)
    else:
        raise AssertionError("duplicate job IDs must be rejected")


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
    assert html.count("<td>Static K4</td>") == 2
    assert "source_data.xlsx" not in html


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
    output_csv = tmp_path / "results.csv"
    output_html = tmp_path / "report.html"

    reporting.build_artifacts(
        result_root=tmp_path / "results",
        log_dir=log_dir,
        output_json=output_json,
        output_csv=output_csv,
        output_html=output_html,
        keys=[key],
        dynamic_schedule=[{"start": 1, "end": 512, "k": 0}],
    )

    assert json.loads(output_json.read_text(encoding="utf-8"))[0]["job_id"] == "1"
    csv_text = output_csv.read_text(encoding="utf-8")
    assert "speedup_vs_baseline" in csv_text
    assert b"\r" not in output_csv.read_bytes()
    assert "Canonical rows" in output_html.read_text(encoding="utf-8")
