from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/vllm024_dflare_report.py"
BUILDER_PATH = ROOT / "scripts/build_latest_specdec_html_pages.py"


def load_module() -> ModuleType:
    assert MODULE_PATH.exists(), "DFlare report module is not implemented"
    spec = importlib.util.spec_from_file_location("vllm024_dflare_report", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_result(
    path: Path,
    *,
    status: str = "complete",
    run_mode: str = "spec",
    osl: int = 65_536,
    temperature: float = 0.0,
    include_baseline: bool = False,
    dataset: str = "math500",
) -> Path:
    results: dict[str, object] = {
        "samples": 4,
        "spec_decode_time_per_token_s": 0.125,
        "spec_decode_tok_s": 8.0,
        "mean_acceptance_length": 5.5,
        "acceptance_rate": 0.375,
    }
    if include_baseline:
        results.update(
            baseline_decode_time_per_token_s=0.25,
            baseline_decode_tok_s=4.0,
            decode_throughput_speedup=2.0,
        )
    payload = {
        "schema_version": 1,
        "backend": "angelslim_transformers_native",
        "status": status,
        "config": {
            "target_model": (
                "/models/qwen3-8b"
                if osl == 32_768
                else "/models/yarn4/qwen3-8b"
            ),
            "draft_model": (
                "/models/qwen3-8b-dflare"
                if osl == 32_768
                else "/models/yarn4/qwen3-8b-dflare"
            ),
            "draft_arch": "dflare",
            "dataset": dataset,
            "max_samples": 4,
            "input_length": 4096,
            "max_new_tokens": osl,
            "ignore_eos": True,
            "temperature": temperature,
            "top_p": 1.0,
            "block_size": 16,
            "world_size": 4,
            "run_mode": run_mode,
        },
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_completed_dflare_results_excludes_noncomplete(tmp_path: Path) -> None:
    module = load_module()
    complete = write_result(
        tmp_path / "64k/math/job-2271721/result.json",
        status="complete",
    )
    running = write_result(
        tmp_path / "64k/math/job-2271722/result.json",
        status="running",
    )

    rows = module.load_completed_dflare_results([running, complete])

    assert rows["status"].tolist() == ["complete"]
    row = rows.iloc[0]
    assert row["method"] == "dflare_k16"
    assert row["domain"] == "Math"
    assert row["model"] == "Qwen3-8B"
    assert row["context_profile"] == "YaRN 64K"
    assert row["position_encoding"] == "yarn4"
    assert row["batch_size"] == 1
    assert row["job_id"] == "2271721"
    assert row["tok_s_gpu"] == 8.0
    assert row["acceptance_rate"] == 0.375
    assert row["mean_accept_len"] == 5.5


def test_spec_only_result_has_no_invented_speedup(tmp_path: Path) -> None:
    module = load_module()
    result = write_result(
        tmp_path / "128k/swe/job-2271728/result.json",
        run_mode="spec",
        osl=126_976,
        temperature=1.0,
    )

    rows = module.match_dflare_baselines(
        module.load_completed_dflare_results([result])
    )

    row = rows.iloc[0]
    assert math.isnan(float(row["speedup"]))
    assert row["speedup_label"] == "waiting matched baseline"
    assert row["context_profile"] == "YaRN total-128K"


def test_paired_result_preserves_exact_angelslim_speedup(tmp_path: Path) -> None:
    module = load_module()
    result = write_result(
        tmp_path / "32k/math/job-2271128/result.json",
        run_mode="both",
        osl=32_768,
        include_baseline=True,
    )

    rows = module.match_dflare_baselines(
        module.load_completed_dflare_results([result])
    )

    row = rows.iloc[0]
    assert row["context_profile"] == "Native 32K"
    assert row["speedup"] == 2.0
    assert row["speedup_label"] == "2.00x"


def test_target_profile_filter_excludes_short_canaries(tmp_path: Path) -> None:
    module = load_module()
    paths = [
        write_result(tmp_path / "canary/job-2270811/result.json", osl=256),
        write_result(tmp_path / "64k/job-2271723/result.json", osl=65_536),
    ]
    rows = module.load_completed_dflare_results(paths)

    target_rows = module.target_profile_rows(rows)

    assert target_rows["context_profile"].tolist() == ["YaRN 64K"]
    assert target_rows["job_id"].tolist() == ["2271723"]


def test_source_paths_can_be_made_repository_relative(tmp_path: Path) -> None:
    module = load_module()
    result = write_result(tmp_path / "results/job-2271723/result.json")
    rows = module.load_completed_dflare_results([result])

    relative = module.relativize_sources(rows, tmp_path)

    assert relative.iloc[0]["source"] == "results/job-2271723/result.json"


def test_render_groups_context_profiles_and_required_columns(tmp_path: Path) -> None:
    module = load_module()
    paths = [
        write_result(
            tmp_path / "32k/math/job-2271128/result.json",
            run_mode="both",
            osl=32_768,
            include_baseline=True,
        ),
        write_result(tmp_path / "64k/math/job-2271721/result.json", osl=65_536),
        write_result(
            tmp_path / "128k/swe/job-2271728/result.json",
            osl=126_976,
            temperature=1.0,
            dataset="swe_verified",
        ),
    ]
    rows = module.match_dflare_baselines(
        module.load_completed_dflare_results(paths)
    )

    rendered = module.render_dflare_section(rows)

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
        "Speedup",
        "Acceptance",
        "Mean accept length",
        "Job ID",
    ]:
        assert heading in rendered
    assert "2271128" in rendered
    assert "2271721" in rendered
    assert "2271728" in rendered
    assert "waiting matched baseline" in rendered


def test_render_does_not_show_incomplete_rows(tmp_path: Path) -> None:
    module = load_module()
    complete = write_result(tmp_path / "64k/math/job-2271721/result.json")
    rows = module.load_completed_dflare_results([complete])
    running = rows.iloc[0].copy()
    running["status"] = "running"
    running["job_id"] = "2271999"

    rendered = module.render_dflare_section(
        pd.concat([rows, running.to_frame().T], ignore_index=True)
    )

    assert "2271721" in rendered
    assert "2271999" not in rendered


def test_existing_standalone_builder_includes_dflare_section() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "load_completed_dflare_results" in source
    assert "match_dflare_baselines" in source
    assert "render_dflare_section" in source
    assert "dflare_completed_latest.csv" in source


def test_standalone_builder_uses_tracked_public_data_fallback() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert 'PUBLIC_DATA = ROOT / "public/data"' in source
    assert "resolve_data_source" in source
    assert 'VLLM_ADDED_INPUT = PUBLIC_DATA / "vllm_standalone_added_results_latest.csv"' in source
    assert "else matrix(main)" in source


def test_pages_index_reports_completed_dflare_rows() -> None:
    source = (ROOT / "scripts/build_pages_index.py").read_text(encoding="utf-8")

    assert "dflare_completed_latest.csv" in source
    assert "dflare_summary" in source
    assert "DFLARE_PROFILES" in source
    assert ".isin(DFLARE_PROFILES)" in source
    assert "do not yet include a direct DFlare row" not in source
