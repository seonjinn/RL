from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest


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
    return sorted(REAL_INPUT_ROOT.rglob("*.json"))


def require_real_input_paths() -> list[Path]:
    paths = real_input_paths()
    if len(paths) != 60:
        pytest.skip("60-file vLLM-native corpus is absent until Task 4 data commit")
    return paths


def make_runtime(
    *,
    job_id: str | None = "1234567",
    vllm_version: str = "0.24.0",
    torch_version: str = "2.11.0+cu130",
    cuda_version: str = "13.0",
    platform_name: str = "Linux-6.17.0-aarch64",
) -> dict[str, object]:
    environment: dict[str, object] = {
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_ATTENTION_BACKEND": None,
        "CUDA_VISIBLE_DEVICES": None,
    }
    if job_id is not None:
        environment["SLURM_JOB_ID"] = job_id
    return {
        "python": "3.12.3",
        "platform": platform_name,
        "vllm_version": vllm_version,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_count": 4,
        "gpu_names": ["NVIDIA GB200"] * 4,
        "environment": environment,
    }


def make_config(
    *,
    profile: str = "native32k",
    domain: str = "math",
    method: str = "baseline",
    temperature: float = 0.0,
    top_p: float = 1.0,
    disable_custom_all_reduce: bool = True,
    prompt_offset: int = 0,
    seed: int = 0,
    measure_repeats: int = 1,
    warmup_repeats: int = 1,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    if profile == "native32k":
        model = (
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
            "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
        )
        osl = 32_768
        batch_sizes: list[int] = [1, 2, 4]
        max_model_len = 40_960
        max_num_seqs = 32
        max_num_batched_tokens = 131_072
        prompt_count_loaded = 32
    elif profile == "yarn64k":
        model = "/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm024-dynamicsd/long-context-models/yarn4/qwen3-8b"
        osl = 65_536
        batch_sizes = [1]
        max_model_len = 69_632
        max_num_seqs = 1
        max_num_batched_tokens = 131_072
        prompt_count_loaded = 1
        warmup_repeats = 0
        temperature = 1.0
    elif profile == "yarn128k":
        model = "/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm024-dynamicsd/long-context-models/yarn4/qwen3-8b"
        osl = 126_976
        batch_sizes = [1]
        max_model_len = 131_072
        max_num_seqs = 1
        max_num_batched_tokens = 65_536
        prompt_count_loaded = 1
        warmup_repeats = 0
        temperature = 1.0
    else:
        raise AssertionError(f"unsupported profile {profile}")

    prompt_jsonl = (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm024-dynamicsd/datasets/"
        "math_500_data_prompts_qmath_20260617.jsonl"
        if domain == "math"
        else "/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark/data/"
        "swebench_verified_prompts_all.jsonl"
    )

    config: dict[str, object] = {
        "attention_backend": "TRITON_ATTN",
        "batch_sizes": batch_sizes,
        "cuda_profiler_range": False,
        "cudagraph_mode": "NONE",
        "disable_custom_all_reduce": disable_custom_all_reduce,
        "draft_model": "/drafts/default",
        "dtype": "bfloat16",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "engine_gpus": 1,
        "gpu_memory_utilization": 0.9,
        "isl": 4096,
        "kv_cache_dtype": "auto",
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "measure_repeats": measure_repeats,
        "mode": method,
        "model": model,
        "moe_backend": "auto",
        "osl": osl,
        "pipeline_parallel_size": 1,
        "pp": 1,
        "prompt_count_loaded": prompt_count_loaded,
        "prompt_jsonl": prompt_jsonl,
        "prompt_offset": prompt_offset,
        "seed": seed,
        "speculative_config": None,
        "tag": f"matrix_{method}_t{int(temperature)}p{int(top_p)}",
        "temperature": temperature,
        "tensor_parallel_size": 1,
        "top_p": top_p,
        "total_gpus": 4,
        "tp": 1,
        "warmup_repeats": warmup_repeats,
    }
    if method != "baseline":
        k = {"dflash": 15, "pard": 12, "pard2": 15, "suffix": 32}.get(method)
        speculative: dict[str, object] = {
            "method": "draft_model" if method == "pard" else method,
            "num_speculative_tokens": k,
        }
        if method in {"dflash", "pard", "pard2"}:
            speculative["model"] = f"/drafts/{method}"
            speculative["draft_tensor_parallel_size"] = 1
        if method in {"pard", "pard2"}:
            speculative["parallel_drafting"] = True
        if method == "suffix":
            speculative["suffix_decoding_max_tree_depth"] = 32
        config["draft_model"] = f"/drafts/{method}"
        config["speculative_config"] = speculative
    if extra:
        config.update(extra)
    return config


def make_result(
    batch_size: int,
    *,
    tok_s_gpu: float,
    latency_s: float,
    acceptance_rate: float | None = None,
    mean_accept_len: float | None = None,
) -> dict[str, object]:
    spec_decode_metrics: dict[str, object] = {}
    if acceptance_rate is not None:
        spec_decode_metrics["acceptance_rate"] = acceptance_rate
    if mean_accept_len is not None:
        spec_decode_metrics["mean_acceptance_length"] = mean_accept_len
    return {
        "bs": batch_size,
        "latency_s_mean": latency_s,
        "latency_s_median": latency_s,
        "mean_latency_s": latency_s,
        "latency_s": latency_s,
        "output_tokens": 32_768,
        "output_tok_s": tok_s_gpu * 4,
        "output_tok_s_per_gpu": tok_s_gpu,
        "prompt_count_used": batch_size,
        "num_batches": 1,
        "repeats": [
            {
                "repeat": 0,
                "latency_s": latency_s,
                "output_tokens": 32_768,
                "output_tok_s": tok_s_gpu * 4,
                "spec_decode_metrics": spec_decode_metrics,
            }
        ],
        "spec_decode_metrics": spec_decode_metrics,
    }


def write_payload(
    path: Path,
    *,
    status: str = "complete",
    runtime: object | None = None,
    config: dict[str, object] | None = None,
    results: list[dict[str, object]] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "status": status,
        "runtime": runtime if runtime is not None else make_runtime(),
        "config": config if config is not None else make_config(),
        "results": results if results is not None else [make_result(1, tok_s_gpu=10.0, latency_s=100.0)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def baseline_frame_row(
    *,
    runtime_family: str = "vllm_native",
    domain: str = "Math",
    source: str = "baseline.json",
    method: str = "baseline",
    k: float = math.nan,
) -> dict[str, object]:
    return {
        "runtime_family": runtime_family,
        "runtime": "vLLM 0.24.0",
        "runtime_provenance": "runtime-sig",
        "domain": domain,
        "model": "Qwen3-8B",
        "model_checkpoint": "/models/qwen3-8b@snapshot",
        "temperature": 0.0,
        "top_p": 1.0,
        "batch_size": 4,
        "isl": 4096,
        "osl": 32768,
        "context_profile": "Native 32K",
        "position_encoding": "native",
        "cuda_graph": "NONE",
        "setup_signature": "setup-sig",
        "attention_backend": "TRITON_ATTN",
        "method": method,
        "k": k,
        "tok_s_gpu": 40.0,
        "latency_s": 100.0,
        "acceptance_rate": math.nan,
        "mean_accept_len": math.nan,
        "job_id": "1234567",
        "source_status": "complete",
        "source": source,
    }


def test_load_profile_results_normalizes_real_inputs() -> None:
    module = load_module()

    rows = module.load_profile_results(require_real_input_paths())

    assert len(rows) == 154
    assert rows["runtime_family"].value_counts().to_dict() == {"vllm_native": 154}
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
    assert "cuda_version" in str(baseline["runtime_provenance"])
    assert "torch_version" in str(baseline["runtime_provenance"])
    assert baseline["domain"] == "Math"
    assert baseline["model"] == "Qwen3-8B"
    assert str(baseline["model_checkpoint"]).endswith(
        "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert baseline["temperature"] == 0.0
    assert baseline["top_p"] == 1.0
    assert baseline["isl"] == 4096
    assert baseline["osl"] == 32768
    assert baseline["context_profile"] == "Native 32K"
    assert baseline["position_encoding"] == "native"
    assert baseline["cuda_graph"] == "NONE"
    assert baseline["source_status"] == "complete"
    assert baseline["method"] == "baseline"
    assert "disable_custom_all_reduce" in str(baseline["setup_signature"])
    assert "prompt_offset" in str(baseline["setup_signature"])
    assert "measure_repeats" in str(baseline["setup_signature"])
    assert "warmup_repeats" in str(baseline["setup_signature"])
    assert pd.isna(baseline["k"])
    assert math.isclose(float(baseline["tok_s_gpu"]), 164.68264711913437)
    assert math.isnan(float(baseline["acceptance_rate"]))
    assert math.isnan(float(baseline["mean_accept_len"]))


def test_load_profile_results_filters_non_vllm_native_runtime_and_derives_profiles_and_job_id(
    tmp_path: Path,
) -> None:
    module = load_module()
    native32k = write_payload(
        tmp_path / "native/job-1234567/result.json",
        config=make_config(profile="native32k", domain="math", method="baseline"),
        results=[make_result(1, tok_s_gpu=10.0, latency_s=100.0)],
    )
    yarn64k = write_payload(
        tmp_path / "yarn64k/job-1234568/result.json",
        runtime=make_runtime(job_id="1234568"),
        config=make_config(profile="yarn64k", domain="math", method="dflash"),
        results=[make_result(1, tok_s_gpu=12.0, latency_s=80.0, acceptance_rate=0.2, mean_accept_len=4.0)],
    )
    yarn128k = write_payload(
        tmp_path / "yarn128k/job-12345678901/result.json",
        runtime=make_runtime(job_id=None),
        config=make_config(profile="yarn128k", domain="swe", method="suffix"),
        results=[make_result(1, tok_s_gpu=8.0, latency_s=120.0, acceptance_rate=0.8, mean_accept_len=12.0)],
    )
    angelslim = write_payload(
        tmp_path / "angelslim/job-1234569/result.json",
        runtime="AngelSlim",
        config=make_config(profile="native32k", domain="math", method="baseline"),
        results=[make_result(1, tok_s_gpu=99.0, latency_s=1.0)],
    )

    rows = module.load_profile_results([native32k, yarn64k, yarn128k, angelslim])

    assert len(rows) == 3
    assert rows["runtime_family"].value_counts().to_dict() == {"vllm_native": 3}
    assert rows["context_profile"].tolist() == [
        "Native 32K",
        "YaRN total-128K",
        "YaRN 64K",
    ]
    assert rows["job_id"].tolist() == ["1234567", "12345678901", "1234568"]


def test_match_profile_baselines_matches_real_native_rows_without_row_multiplication() -> None:
    module = load_module()

    loaded = module.load_profile_results(require_real_input_paths())
    rows = module.match_profile_baselines(loaded)

    assert len(rows) == len(loaded)
    baseline = rows.loc[
        rows["source"].astype(str).str.endswith(
            "native32k/math/baseline/matrix/baseline_t0p0/result.json"
        )
        & rows["batch_size"].eq(4)
    ].iloc[0]
    assert math.isclose(float(baseline["throughput_speedup"]), 1.0)
    assert math.isclose(float(baseline["latency_speedup"]), 1.0)
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
    assert math.isclose(float(dflash["baseline_tok_s_gpu"]), 37.279732126789526)
    assert math.isclose(float(dflash["baseline_latency_s"]), 878.976272912987)
    assert math.isclose(float(dflash["throughput_speedup"]), 1.486798745603985)
    assert math.isclose(float(dflash["latency_speedup"]), 1.486798745603985)


def test_match_profile_baselines_requires_exact_runtime_and_setup_signature(
    tmp_path: Path,
) -> None:
    module = load_module()
    baseline = write_payload(
        tmp_path / "baseline/job-2222222/result.json",
        config=make_config(
            profile="native32k",
            method="baseline",
            disable_custom_all_reduce=True,
            prompt_offset=0,
        ),
        results=[make_result(4, tok_s_gpu=40.0, latency_s=100.0)],
    )
    setup_mismatch = write_payload(
        tmp_path / "setup/job-2222223/result.json",
        config=make_config(
            profile="native32k",
            method="pard2",
            disable_custom_all_reduce=False,
            prompt_offset=0,
        ),
        results=[make_result(4, tok_s_gpu=20.0, latency_s=200.0, acceptance_rate=0.01, mean_accept_len=1.1)],
    )
    runtime_mismatch = write_payload(
        tmp_path / "runtime/job-2222224/result.json",
        runtime=make_runtime(torch_version="9.9.9"),
        config=make_config(
            profile="native32k",
            method="dflash",
            disable_custom_all_reduce=True,
            prompt_offset=0,
        ),
        results=[make_result(4, tok_s_gpu=30.0, latency_s=120.0, acceptance_rate=0.2, mean_accept_len=4.2)],
    )
    prompt_mismatch = write_payload(
        tmp_path / "prompt/job-2222225/result.json",
        config=make_config(
            profile="native32k",
            method="suffix",
            disable_custom_all_reduce=True,
            prompt_offset=8,
        ),
        results=[make_result(4, tok_s_gpu=80.0, latency_s=50.0, acceptance_rate=0.9, mean_accept_len=16.0)],
    )

    rows = module.match_profile_baselines(
        module.load_profile_results([baseline, setup_mismatch, runtime_mismatch, prompt_mismatch])
    )

    nonbaseline = rows.loc[rows["method"].ne("baseline")].copy()
    assert nonbaseline["throughput_speedup_label"].tolist() == [
        "waiting matched baseline",
        "waiting matched baseline",
        "waiting matched baseline",
    ]
    assert nonbaseline["latency_speedup_label"].tolist() == [
        "waiting matched baseline",
        "waiting matched baseline",
        "waiting matched baseline",
    ]


def test_match_profile_baselines_targeted_retry_matches_full_sweep_but_nested_setup_difference_does_not(
    tmp_path: Path,
) -> None:
    module = load_module()
    baseline = write_payload(
        tmp_path / "baseline/job-3333331/result.json",
        config=make_config(
            profile="native32k",
            method="baseline",
            extra={
                "batch_sizes": [1, 2, 4, 8, 16, 32],
                "prompt_count_loaded": 32,
            },
        ),
        results=[
            make_result(16, tok_s_gpu=160.0, latency_s=100.0),
            make_result(32, tok_s_gpu=320.0, latency_s=100.0),
        ],
    )
    targeted_retry = write_payload(
        tmp_path / "retry/job-3333332/result.json",
        config=make_config(
            profile="native32k",
            method="pard2",
            extra={
                "batch_sizes": [16, 32],
                "prompt_count_loaded": 2,
            },
        ),
        results=[
            make_result(16, tok_s_gpu=80.0, latency_s=200.0, acceptance_rate=0.1, mean_accept_len=1.2),
            make_result(32, tok_s_gpu=160.0, latency_s=200.0, acceptance_rate=0.1, mean_accept_len=1.2),
        ],
    )
    nested_difference = write_payload(
        tmp_path / "nested/job-3333333/result.json",
        config=make_config(
            profile="native32k",
            method="dflash",
            extra={
                "batch_sizes": [16],
                "prompt_count_loaded": 1,
                "engine_metadata": {
                    "batch_sizes": [99],
                    "prompt_count_loaded": 777,
                    "max_num_seqs": 999,
                },
            },
        ),
        results=[
            make_result(16, tok_s_gpu=120.0, latency_s=120.0, acceptance_rate=0.2, mean_accept_len=4.0),
        ],
    )

    rows = module.match_profile_baselines(
        module.load_profile_results([baseline, targeted_retry, nested_difference])
    )

    retry_rows = rows.loc[rows["source"].astype(str).str.endswith("retry/job-3333332/result.json")]
    assert retry_rows["throughput_speedup_label"].tolist() == ["0.50x", "0.50x"]
    assert retry_rows["latency_speedup_label"].tolist() == ["0.50x", "0.50x"]

    nested_row = rows.loc[rows["source"].astype(str).str.endswith("nested/job-3333333/result.json")].iloc[0]
    assert nested_row["throughput_speedup_label"] == "waiting matched baseline"
    assert nested_row["latency_speedup_label"] == "waiting matched baseline"


def test_match_profile_baselines_prefers_complete_duplicate_baseline_without_row_multiplication() -> None:
    module = load_module()
    rows = pd.DataFrame(
        [
            baseline_frame_row(source="partial-baseline.json") | {"source_status": "partial", "tok_s_gpu": 20.0, "latency_s": 200.0},
            baseline_frame_row(source="complete-baseline.json"),
            baseline_frame_row(method="pard2", k=15.0, source="spec.json")
            | {
                "tok_s_gpu": 10.0,
                "latency_s": 400.0,
                "acceptance_rate": 0.1,
                "mean_accept_len": 1.2,
            },
        ]
    )

    matched = module.match_profile_baselines(rows)
    spec = matched.loc[matched["source"].eq("spec.json")].iloc[0]

    assert len(matched) == len(rows)
    assert math.isclose(float(spec["baseline_tok_s_gpu"]), 40.0)
    assert math.isclose(float(spec["baseline_latency_s"]), 100.0)
    assert math.isclose(float(spec["throughput_speedup"]), 0.25)
    assert math.isclose(float(spec["latency_speedup"]), 0.25)


def test_match_profile_baselines_rejects_ambiguous_duplicate_exact_keys() -> None:
    module = load_module()
    rows = pd.DataFrame(
        [
            baseline_frame_row(source="baseline-a.json"),
            baseline_frame_row(source="baseline-b.json"),
            baseline_frame_row(method="dflash", k=15.0, source="spec.json")
            | {
                "tok_s_gpu": 20.0,
                "latency_s": 200.0,
                "acceptance_rate": 0.2,
                "mean_accept_len": 4.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="ambiguous duplicate baseline"):
        module.match_profile_baselines(rows)


def test_speedup_cell_emits_bounded_magnitude_sensitive_color_properties() -> None:
    module = load_module()

    def alpha(cell: str, property_name: str) -> float:
        match = re.search(rf"{property_name}:rgba\([^)]*,([0-9.]+)\)", cell)
        assert match is not None
        return float(match.group(1))

    mild_speedup = module._speedup_cell(
        baseline_frame_row(source="mild-speedup.json") | {"throughput_speedup": 1.01}
    )
    strong_speedup = module._speedup_cell(
        baseline_frame_row(source="strong-speedup.json") | {"throughput_speedup": 100.0}
    )
    mild_slowdown = module._speedup_cell(
        baseline_frame_row(source="mild-slowdown.json") | {"throughput_speedup": 0.99}
    )
    strong_slowdown = module._speedup_cell(
        baseline_frame_row(source="strong-slowdown.json") | {"throughput_speedup": 0.05}
    )

    blue_alphas = [
        alpha(mild_speedup, "--matrix-blue"),
        alpha(strong_speedup, "--matrix-blue"),
    ]
    red_alphas = [
        alpha(mild_slowdown, "--matrix-red"),
        alpha(strong_slowdown, "--matrix-red"),
    ]
    assert blue_alphas[0] < blue_alphas[1]
    assert red_alphas[0] < red_alphas[1]
    assert all(0.12 <= value <= 0.36 for value in blue_alphas + red_alphas)
    assert "--matrix-text:#17406d" in mild_speedup
    assert "--matrix-text:#17406d" in strong_speedup
    assert "--matrix-text:#8f1d16" in mild_slowdown
    assert "--matrix-text:#8f1d16" in strong_slowdown
    assert "#ffffff" not in mild_speedup + strong_speedup + mild_slowdown + strong_slowdown


def test_speedup_cell_preserves_partial_unmatched_state_and_escapes_title() -> None:
    module = load_module()
    cell = module._speedup_cell(
        baseline_frame_row(source='spec<&>"\'.json', method="suffix", k=32.0)
        | {
            "source_status": "partial",
            "throughput_speedup": math.nan,
            "acceptance_rate": 0.7,
            "mean_accept_len": 8.0,
        }
    )

    assert 'class="speed-cell empty waiting partial"' in cell
    assert "waiting baseline†" in cell
    assert (
        'title="status: partial; source: spec&lt;&amp;&gt;&quot;&#x27;.json; '
        'tok/s/GPU: 40.00; acceptance: 70.00%; mean accepted length: 8.00"'
    ) in cell


def test_render_profile_section_renders_matrix_states_and_separate_k_rows() -> None:
    module = load_module()
    rows = pd.DataFrame(
        [
            baseline_frame_row(source="baseline<&>.json")
            | {
                "batch_size": 1,
                "throughput_speedup": 1.0,
                "latency_speedup": 1.0,
                "throughput_speedup_label": "1.00x",
                "latency_speedup_label": "1.00x",
            },
            baseline_frame_row(
                source='spec<&>"\'.json',
                method="dflash",
                k=15.0,
            )
            | {
                "batch_size": 1,
                "tok_s_gpu": 90.0,
                "latency_s": 44.0,
                "acceptance_rate": 0.625,
                "mean_accept_len": 4.5,
                "throughput_speedup": 2.25,
                "latency_speedup": 2.25,
                "throughput_speedup_label": "2.25x",
                "latency_speedup_label": "2.25x",
            },
            baseline_frame_row(source="spec-k7.json", method="dflash", k=7.0)
            | {
                "batch_size": 1,
                "tok_s_gpu": 50.0,
                "latency_s": 80.0,
                "acceptance_rate": 0.4,
                "mean_accept_len": 3.5,
                "throughput_speedup": 1.25,
                "latency_speedup": 1.25,
                "throughput_speedup_label": "1.25x",
                "latency_speedup_label": "1.25x",
            },
            baseline_frame_row(source="slowdown.json", method="dflash", k=15.0)
            | {
                "batch_size": 2,
                "tok_s_gpu": 30.0,
                "latency_s": 133.0,
                "acceptance_rate": 0.5,
                "mean_accept_len": 3.0,
                "throughput_speedup": 0.75,
                "latency_speedup": 0.75,
                "throughput_speedup_label": "0.75x",
                "latency_speedup_label": "0.75x",
            },
            baseline_frame_row(source="partial.json", method="dflash", k=15.0)
            | {
                "batch_size": 4,
                "source_status": "partial",
                "tok_s_gpu": 20.0,
                "latency_s": 200.0,
                "acceptance_rate": 0.25,
                "mean_accept_len": 2.0,
                "throughput_speedup": 0.5,
                "latency_speedup": 0.5,
                "throughput_speedup_label": "0.50x",
                "latency_speedup_label": "0.50x",
            },
            baseline_frame_row(source="unmatched.json", method="suffix", k=32.0)
            | {
                "batch_size": 8,
                "source_status": "partial",
                "tok_s_gpu": 25.0,
                "latency_s": 160.0,
                "acceptance_rate": 0.7,
                "mean_accept_len": 8.0,
                "throughput_speedup": math.nan,
                "latency_speedup": math.nan,
                "throughput_speedup_label": "waiting matched baseline",
                "latency_speedup_label": "waiting matched baseline",
            },
            baseline_frame_row(
                runtime_family="angelslim",
                domain="AngelSlim <drop>",
                source="angelslim.json",
            )
            | {
                "throughput_speedup": 99.0,
                "latency_speedup": 99.0,
                "throughput_speedup_label": "99.00x",
                "latency_speedup_label": "99.00x",
            },
        ]
    )

    rendered = module.render_profile_section(rows)
    matrix = module._profile_matrix(rows, "Native 32K")

    assert '<section class="section" id="vllm024-profile">' in rendered
    assert '<table class="native-speedup-matrix">' in rendered
    assert all(f">B{batch}<" in rendered for batch in (1, 2, 4, 8, 16, 32))
    assert 'class="speed-cell speedup"' in rendered
    assert 'class="speed-cell neutral"' in rendered
    assert 'class="speed-cell slowdown"' in rendered
    assert 'class="speed-cell slowdown partial"' in rendered
    assert "0.50x†" in rendered
    assert 'class="speed-cell empty">n/a</td>' in rendered
    assert 'class="speed-cell empty waiting partial"' in rendered
    assert "waiting baseline†" in rendered
    assert "spec&lt;&amp;&gt;&quot;&#x27;.json" in rendered
    assert "baseline&lt;&amp;&gt;.json" in rendered
    assert matrix.count("<td>DFlash K=7</td>") == 1
    assert matrix.count("<td>DFlash K=15</td>") == 1
    assert matrix.index("DFlash K=7") < matrix.index("DFlash K=15")
    assert matrix.index("Math Temp 0.0") < matrix.index("Math Temp 1.0")
    assert matrix.index("Math Temp 1.0") < matrix.index("SWE Temp 0.0")
    assert matrix.index("SWE Temp 0.0") < matrix.index("SWE Temp 1.0")
    assert '<details class="native-profile-details">' in rendered
    assert "Detailed native metrics and sources" in rendered
    assert "AngelSlim &lt;drop&gt;" not in rendered


def test_render_profile_section_rejects_duplicate_native_matrix_cells() -> None:
    module = load_module()
    duplicate = baseline_frame_row(method="dflash", k=15.0, source="spec-a.json") | {
        "throughput_speedup": 2.0,
        "latency_speedup": 2.0,
    }
    rows = pd.DataFrame(
        [
            duplicate,
            duplicate | {"source": "spec-b.json", "job_id": "7654321"},
        ]
    )

    with pytest.raises(ValueError, match="duplicate native matrix cell"):
        module.render_profile_section(rows)


def test_render_profile_section_follows_profile_order() -> None:
    module = load_module()
    rows = pd.DataFrame(
        [
            baseline_frame_row(source="native.json")
            | {
                "throughput_speedup": 1.0,
                "latency_speedup": 1.0,
            },
            baseline_frame_row(source="yarn64.json")
            | {
                "osl": 65_536,
                "context_profile": "YaRN 64K",
                "position_encoding": "yarn4",
                "throughput_speedup": 1.0,
                "latency_speedup": 1.0,
            },
            baseline_frame_row(source="yarn128.json")
            | {
                "osl": 126_976,
                "context_profile": "YaRN total-128K",
                "position_encoding": "yarn4",
                "throughput_speedup": 1.0,
                "latency_speedup": 1.0,
            },
        ]
    )

    rendered = module.render_profile_section(rows)
    positions = [
        rendered.index(f'data-profile="{profile}"') for profile in module.PROFILE_ORDER
    ]

    assert positions == sorted(positions)
