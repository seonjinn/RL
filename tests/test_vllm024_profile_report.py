from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/vllm024_profile_report.py"
REAL_INPUT_ROOT = (
    ROOT
    / "experiments/vllm_024_dynamicsd/report/20260704_vllm_native_completed"
)


def load_module() -> ModuleType:
    assert MODULE_PATH.exists(), "profile report module is not implemented"
    spec = importlib.util.spec_from_file_location("vllm024_profile_report", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def real_input_paths() -> list[Path]:
    paths = sorted(REAL_INPUT_ROOT.rglob("*.json"))
    assert len(paths) == 60, "expected 60 checked-in vLLM-native result JSON files"
    return paths


def test_load_profile_results_normalizes_real_inputs() -> None:
    module = load_module()

    rows = module.load_profile_results(real_input_paths())

    assert len(rows) == 154
    assert rows["source_status"].value_counts().to_dict() == {
        "complete": 136,
        "partial": 18,
    }
    assert rows["method"].value_counts().to_dict() == {
        "baseline": 32,
        "dflash": 32,
        "pard": 32,
        "pard2": 26,
        "suffix": 32,
    }
    assert rows["context_profile"].value_counts().to_dict() == {
        "Native 32K": 114,
        "YaRN 64K": 20,
        "YaRN total-128K": 20,
    }

    baseline = rows.loc[
        rows["source"].astype(str).str.endswith(
            "native32k/math/baseline/matrix/baseline_t0p0/result.json"
        )
        & rows["batch_size"].eq(32)
    ].iloc[0]
    assert baseline["runtime"] == "vLLM 0.24.0"
    assert baseline["domain"] == "Math"
    assert baseline["model"] == "Qwen3-8B"
    assert baseline["temperature"] == 0.0
    assert baseline["top_p"] == 1.0
    assert baseline["isl"] == 4096
    assert baseline["osl"] == 32768
    assert baseline["context_profile"] == "Native 32K"
    assert baseline["position_encoding"] == "native"
    assert baseline["cuda_graph"] == "NONE"
    assert baseline["source_status"] == "complete"
    assert baseline["method"] == "baseline"
    assert pd.isna(baseline["k"])
    assert math.isclose(baseline["tok_s_gpu"], 164.68264711913437)
    assert math.isnan(float(baseline["acceptance_rate"]))
    assert math.isnan(float(baseline["mean_accept_len"]))

    partial = rows.loc[
        rows["source"].astype(str).str.endswith(
            "native32k/math/pard2/matrix/pard2_t0p0/result.json"
        )
        & rows["batch_size"].eq(8)
    ].iloc[0]
    assert partial["runtime"] == "vLLM 0.24.0"
    assert partial["domain"] == "Math"
    assert partial["context_profile"] == "Native 32K"
    assert partial["source_status"] == "partial"
    assert partial["method"] == "pard2"
    assert partial["k"] == 15
    assert math.isclose(partial["tok_s_gpu"], 36.822565954623336)
    assert math.isclose(partial["acceptance_rate"], 0.0010346207983057633)
    assert math.isclose(partial["mean_accept_len"], 1.0155193119745864)


def test_match_profile_baselines_matches_real_native_rows() -> None:
    module = load_module()

    rows = module.match_profile_baselines(module.load_profile_results(real_input_paths()))

    baseline = rows.loc[
        rows["source"].astype(str).str.endswith(
            "native32k/math/baseline/matrix/baseline_t0p0/result.json"
        )
        & rows["batch_size"].eq(4)
    ].iloc[0]
    assert math.isclose(baseline["throughput_speedup"], 1.0)
    assert math.isclose(baseline["latency_speedup"], 1.0)
    assert baseline["throughput_speedup_label"] == "1.00x"
    assert baseline["latency_speedup_label"] == "1.00x"

    dflash = rows.loc[
        rows["source"].astype(str).str.endswith(
            "native32k/math/dflash/matrix/dflash_t0p0/result.json"
        )
        & rows["batch_size"].eq(4)
    ].iloc[0]
    assert dflash["source_status"] == "complete"
    assert dflash["method"] == "dflash"
    assert math.isclose(dflash["baseline_tok_s_gpu"], 37.279732126789526)
    assert math.isclose(dflash["baseline_latency_s"], 878.976272912987)
    assert math.isclose(dflash["throughput_speedup"], 1.486798745603985)
    assert math.isclose(dflash["latency_speedup"], 1.486798745603985)
    assert dflash["throughput_speedup_label"] == "1.49x"
    assert dflash["latency_speedup_label"] == "1.49x"

    partial = rows.loc[
        rows["source"].astype(str).str.endswith(
            "native32k/math/pard2/matrix/pard2_t0p0/result.json"
        )
        & rows["batch_size"].eq(8)
    ].iloc[0]
    assert partial["source_status"] == "partial"
    assert math.isclose(partial["baseline_tok_s_gpu"], 74.10213105299202)
    assert math.isclose(partial["baseline_latency_s"], 884.4010161210317)
    assert math.isclose(partial["throughput_speedup"], 0.49691642374347816)
    assert math.isclose(partial["latency_speedup"], 0.49691642374347816)


def test_match_profile_baselines_requires_full_exact_key() -> None:
    module = load_module()

    rows = pd.DataFrame(
        [
            {
                "source_status": "complete",
                "runtime": "vLLM 0.24.0",
                "domain": "Math",
                "model": "Qwen3-8B",
                "temperature": 0.0,
                "top_p": 1.0,
                "batch_size": 4,
                "isl": 4096,
                "osl": 32768,
                "context_profile": "Native 32K",
                "position_encoding": "native",
                "cuda_graph": "NONE",
                "setup": "triton-tp1-pp1",
                "attention_backend": "TRITON_ATTN",
                "method": "baseline",
                "k": math.nan,
                "tok_s_gpu": 40.0,
                "latency_s": 100.0,
                "acceptance_rate": math.nan,
                "mean_accept_len": math.nan,
                "job_id": "2270001",
                "source": "baseline.json",
            },
            {
                "source_status": "complete",
                "runtime": "vLLM 0.24.0",
                "domain": "Math",
                "model": "Qwen3-8B",
                "temperature": 0.0,
                "top_p": 1.0,
                "batch_size": 4,
                "isl": 4096,
                "osl": 32768,
                "context_profile": "Native 32K",
                "position_encoding": "native",
                "cuda_graph": "NONE",
                "setup": "flashinfer-tp1-pp1",
                "attention_backend": "TRITON_ATTN",
                "method": "pard2",
                "k": 15,
                "tok_s_gpu": 20.0,
                "latency_s": 200.0,
                "acceptance_rate": 0.01,
                "mean_accept_len": 1.1,
                "job_id": "2270002",
                "source": "pard2.json",
            },
        ]
    )

    matched = module.match_profile_baselines(rows)
    spec = matched.loc[matched["method"].eq("pard2")].iloc[0]

    assert math.isnan(float(spec["baseline_tok_s_gpu"]))
    assert math.isnan(float(spec["baseline_latency_s"]))
    assert math.isnan(float(spec["throughput_speedup"]))
    assert math.isnan(float(spec["latency_speedup"]))
    assert spec["throughput_speedup_label"] == "waiting matched baseline"
    assert spec["latency_speedup_label"] == "waiting matched baseline"


def test_render_profile_section_marks_complete_and_partial_sources() -> None:
    module = load_module()

    rows = module.match_profile_baselines(module.load_profile_results(real_input_paths()))
    rendered = module.render_profile_section(rows)

    assert '<section class="section" id="vllm024-profile">' in rendered
    assert "vLLM 0.24 / Native Profile Results" in rendered
    assert "Native 32K" in rendered
    assert "YaRN 64K" in rendered
    assert "YaRN total-128K" in rendered
    for heading in [
        "Domain",
        "Temperature",
        "ISL",
        "OSL",
        "Batch",
        "Method / K",
        "tok/s/GPU",
        "Throughput speedup",
        "Latency speedup",
        "Acceptance",
        "Mean accept length",
        "Source",
    ]:
        assert heading in rendered
    assert "Persisted batch results from interrupted sources are shown as partial." in rendered
    assert 'class="source-status complete"' in rendered
    assert 'class="source-status partial"' in rendered
    assert "pard2_t0p0/result.json" in rendered
    assert "baseline_t0p0/result.json" in rendered
