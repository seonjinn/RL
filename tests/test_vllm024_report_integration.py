from __future__ import annotations

import hashlib
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import build_latest_specdec_html_pages as latest
from scripts import build_pages_index as index


class RecordingHTMLParser(HTMLParser):
    pass


def parse_html(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    parser = RecordingHTMLParser()
    parser.feed(text)
    parser.close()
    return text


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_task5_latest_builder_can_write_to_temp_outputs_without_checkout_side_effects(
    tmp_path: Path,
) -> None:
    production_latest = ROOT / "docs/vllm_standalone_results_latest.html"
    production_historical = ROOT / "docs/vllm_standalone_results_20260621.html"
    production_added = ROOT / "docs/vllm_standalone_added_results_latest.csv"
    production_public_index = ROOT / "public/index.html"
    latest_before = sha256(production_latest)
    historical_before = sha256(production_historical)
    added_before = sha256(production_added)
    public_index_before = sha256(production_public_index)

    temp_html = tmp_path / "docs/vllm_standalone_results_latest.html"
    temp_added = tmp_path / "docs/vllm_standalone_added_results_latest.csv"
    temp_completed = tmp_path / "report/dflare_completed_latest.csv"
    temp_public_data = tmp_path / "public/data"

    latest.build_latest_vllm_outputs(
        output_html=temp_html,
        added_csv_out=temp_added,
        completed_csv_out=temp_completed,
        public_data_dir=temp_public_data,
    )

    assert temp_html.exists()
    assert temp_added.exists()
    assert temp_completed.exists()
    assert (temp_public_data / "vllm024_profiles_latest.csv").exists()
    assert (temp_public_data / "dflare_completed_latest.csv").exists()
    assert (temp_public_data / "dflare_job_status_latest.csv").exists()
    assert sha256(production_latest) == latest_before
    assert sha256(production_historical) == historical_before
    assert sha256(production_added) == added_before
    assert sha256(production_public_index) == public_index_before


def test_task5_latest_vllm_html_contains_native_and_status_sections(tmp_path: Path) -> None:
    temp_html = tmp_path / "docs/vllm_standalone_results_latest.html"
    temp_added = tmp_path / "docs/vllm_standalone_added_results_latest.csv"
    temp_completed = tmp_path / "report/dflare_completed_latest.csv"
    temp_public_data = tmp_path / "public/data"
    latest.build_latest_vllm_outputs(
        output_html=temp_html,
        added_csv_out=temp_added,
        completed_csv_out=temp_completed,
        public_data_dir=temp_public_data,
    )

    html_text = parse_html(temp_html)

    assert "vLLM 0.24 / Native Profile Results" in html_text
    assert "vLLM 0.24 / DFlare Completed Results" in html_text
    assert html_text.count('class="native-profile-grid"') >= 4
    assert html_text.count('class="native-speedup-matrix"') >= 16
    assert (
        'class="native-profile" data-profile="Native 32K" '
        'data-cuda-graph="PIECEWISE"'
    ) in html_text
    assert "CUDA Graph ON (PIECEWISE)" in html_text
    assert "waiting baseline" in html_text
    assert "speed-cell slowdown" in html_text
    assert "speed-cell speedup" in html_text
    assert "speed-cell neutral" in html_text
    assert re.search(
        r'class="speed-cell (?:slowdown|neutral|speedup|empty waiting) partial"[^>]*>'
        r"[^<]*†</td>",
        html_text,
    )
    assert '<details class="native-profile-details">' in html_text
    assert ".native-profile-grid{display:grid" in html_text
    assert ".native-profile-matrix{min-width:0}" in html_text
    assert ".native-speedup-matrix{font-size:14px}" in html_text
    assert "@media(max-width:1000px){.native-profile-grid{grid-template-columns:1fr}" in html_text
    style_text = html_text.split("<style>", 1)[1].split("</style>", 1)[0]
    speed_cell_selectors = [
        selector.strip()
        for selector in re.findall(r"([^{}]+)\{", style_text)
        if ".speed-cell" in selector
    ]
    assert speed_cell_selectors
    assert all(
        selector.startswith(".native-speedup-matrix .speed-cell")
        for selector in speed_cell_selectors
    )
    assert "DFlare Failure and Status" in html_text
    assert "2272937" in html_text
    assert "2272938" in html_text
    assert "2272941" in html_text
    assert "2272942" in html_text
    assert "2274775" in html_text
    assert "2274776" in html_text
    assert "2274777" in html_text
    assert "2274778" in html_text
    assert "slurm_wall_time_5h" in html_text
    assert "retry_compact_transport_8h_backfill" in html_text
    assert "gather_object_cuda_oom_after_generation" in html_text
    assert "retry_of" in html_text
    assert "vllm024_profiles_latest.csv" in html_text
    assert "dflare_job_status_latest.csv" in html_text
    assert (temp_public_data / "vllm024_profiles_latest.csv").exists()
    assert (temp_public_data / "dflare_completed_latest.csv").exists()
    assert (temp_public_data / "dflare_job_status_latest.csv").exists()


def test_task6_latest_vllm_html_contains_sync_rl_and_speedbench_status(
    tmp_path: Path,
) -> None:
    temp_html = tmp_path / "docs/vllm_standalone_results_latest.html"
    temp_added = tmp_path / "docs/vllm_standalone_added_results_latest.csv"
    temp_completed = tmp_path / "report/dflare_completed_latest.csv"
    temp_public_data = tmp_path / "public/data"
    latest.build_latest_vllm_outputs(
        output_html=temp_html,
        added_csv_out=temp_added,
        completed_csv_out=temp_completed,
        public_data_dir=temp_public_data,
    )

    html_text = parse_html(temp_html)

    assert "Sync-RL SWE and SPEED-Bench Status" in html_text
    assert "Official SPEED-Bench" in html_text
    assert "Sync-RL overlay" in html_text
    assert "official-modelopt" in html_text
    assert "sync-rl-overlay-user" in html_text
    assert "temperature 1.0 / top_p 1.0" in html_text
    assert "Qwen3-32B Math DynamicSD" in html_text
    assert "DAPO-Math-17k" in html_text
    assert "OpenMathInstruct-2" in html_text
    assert "45.65%" in html_text
    assert "46.59%" in html_text
    assert "33.77%" in html_text
    assert "35.55%" in html_text
    assert "26.6%" in html_text
    assert "13.7%" in html_text
    assert "26.59%" not in html_text
    assert "13.75%" not in html_text
    assert "3376.99%" not in html_text
    assert "3555.14%" not in html_text
    assert "1.51x" in html_text
    assert "1.55x" in html_text
    assert "No completed SPEED-Bench official or overlay result.json artifacts are present in this checkout." in html_text
    assert "Qwen3-32B" in html_text
    assert "Qwen3-235B-A22B" in html_text
    assert "NVIDIA-Nemotron-3-Super-120B-A12B-BF16" in html_text
    assert "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" in html_text
    assert ">32K<" in html_text or " 32K " in html_text
    assert ">64K<" in html_text or " 64K " in html_text
    assert "Official launcher support" in html_text
    assert "Sync-RL overlay support" in html_text
    assert "Official limitations" in html_text
    assert "Overlay gates" in html_text

    qwen_table = html_text.split("<h3>Qwen SWE Sync-RL support</h3>", 1)[1].split(
        "</table>", 1
    )[0]
    nemotron_table = html_text.split(
        "<h3>Nemotron SPEED-Bench support</h3>", 1
    )[1].split("</table>", 1)[0]
    assert "<th>Supported</th>" in qwen_table
    assert "<th>Integration only</th>" in qwen_table
    assert "<th>Official launcher support</th>" not in qwen_table
    assert "<th>Official launcher support</th>" in nemotron_table
    assert "<th>Sync-RL overlay support</th>" in nemotron_table
    assert "<th>Official limitations</th>" in nemotron_table
    assert "<th>Overlay gates</th>" in nemotron_table


def test_task6_nemotron_support_uses_runner_capabilities_and_dynamic_gate() -> None:
    _, nemotron = latest.load_sync_speedbench_support()

    assert not nemotron.empty
    for row in nemotron.to_dict(orient="records"):
        assert "native MTP static" not in row["official_support"]
        assert "native MTP static" in row["overlay_support"]
        assert "native MTP static: low-level runner capability only" in row[
            "official_limitations"
        ]
        assert "no official Nemotron MTP launcher" in row["official_limitations"]
        assert "native MTP dynamic" not in row["official_support"]
        assert "native MTP dynamic" in row["overlay_support"]
        assert "native MTP dynamic unsupported" in row["official_limitations"]
        assert "signed model/profile calibration artifact" in row["overlay_gates"]
        assert "excluded from smoke" in row["overlay_gates"]


def test_task6_latest_vllm_html_contains_perfcfg_dynamic_replay_results(
    tmp_path: Path,
) -> None:
    temp_html = tmp_path / "docs/vllm_standalone_results_latest.html"
    temp_added = tmp_path / "docs/vllm_standalone_added_results_latest.csv"
    temp_completed = tmp_path / "report/dflare_completed_latest.csv"
    temp_public_data = tmp_path / "public/data"
    latest.build_latest_vllm_outputs(
        output_html=temp_html,
        added_csv_out=temp_added,
        completed_csv_out=temp_completed,
        public_data_dir=temp_public_data,
    )

    html_text = parse_html(temp_html)

    assert "Performance-Recipe DynamicSD Replay" in html_text
    assert "historical schedule replay" in html_text
    assert "excluded from calibrated claims" in html_text
    for job_id in ("2294695", "2294696", "2294697", "2294699", "2294694", "2294734"):
        assert job_id in html_text
    assert "Qwen3-30B-A3B" in html_text
    assert "Qwen3-32B" in html_text
    assert "Qwen3-235B-A22B" in html_text
    assert (temp_public_data / "vllm024_perfcfg_dynamic_replay_20260706.csv").exists()


def test_final_review_design_and_plan_have_exactly_one_trailing_newline() -> None:
    paths = (
        ROOT
        / "docs/superpowers/specs/2026-07-06-vllm024-sync-rl-swe-speedbench-design.md",
        ROOT
        / "docs/superpowers/plans/2026-07-06-vllm024-sync-rl-swe-speedbench.md",
    )

    for path in paths:
        payload = path.read_bytes()
        assert payload.endswith(b"\n")
        assert not payload.endswith(b"\n\n")


def test_task6_task5_probe_report_uses_repo_root_safe_module_loading() -> None:
    report_text = (ROOT / ".superpowers/sdd/task-5-report.md").read_text(
        encoding="utf-8"
    )

    assert 'import sys' in report_text
    assert 'sys.path.insert(0, str(path.parent))' in report_text
    assert "sys.modules[spec.name] = bench" in report_text
    assert 'save_dir = root / "save"' in report_text


def test_task5_index_publishes_new_artifacts_and_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_docs = tmp_path / "docs"
    temp_public = tmp_path / "public"
    temp_reports = temp_public / "reports"
    temp_data = temp_public / "data"
    temp_archive = temp_public / "archive"
    temp_figures = temp_public / "figures"
    temp_docs.mkdir(parents=True)
    temp_reports.mkdir(parents=True)
    temp_data.mkdir(parents=True)
    temp_archive.mkdir(parents=True)
    temp_figures.mkdir(parents=True)
    latest_html = temp_docs / "vllm_standalone_results_latest.html"
    added_csv = temp_docs / "vllm_standalone_added_results_latest.csv"
    latest.build_latest_vllm_outputs(
        output_html=latest_html,
        added_csv_out=added_csv,
        completed_csv_out=temp_data / "dflare_completed_latest.csv",
        public_data_dir=temp_data,
    )
    (temp_docs / "vllm_standalone_results_20260619.html").write_text(
        "<!doctype html><title>6/19</title>",
        encoding="utf-8",
    )

    monkeypatch.setattr(index, "DOCS", temp_docs)
    monkeypatch.setattr(index, "PUBLIC", temp_public)
    monkeypatch.setattr(index, "REPORTS", temp_reports)
    monkeypatch.setattr(index, "DATA", temp_data)
    monkeypatch.setattr(index, "ARCHIVE", temp_archive)
    monkeypatch.setattr(index, "FIGURES", temp_figures)
    index.build()

    html_text = parse_html(temp_public / "index.html")

    assert "8 completed target-profile DFlare job(s)" in html_text
    assert "12 performance row(s)" in html_text
    assert "16 failure/status row(s)" in html_text
    assert "job 2272941." in html_text
    assert 'href="data/vllm024_profiles_latest.csv"' in html_text
    assert 'href="data/dflare_completed_latest.csv"' in html_text
    assert 'href="data/dflare_job_status_latest.csv"' in html_text


def test_task5_index_chooses_max_numeric_job_id_for_latest_dflare_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_docs = tmp_path / "docs"
    temp_public = tmp_path / "public"
    temp_reports = temp_public / "reports"
    temp_data = temp_public / "data"
    temp_archive = temp_public / "archive"
    temp_figures = temp_public / "figures"
    temp_docs.mkdir(parents=True)
    temp_reports.mkdir(parents=True)
    temp_data.mkdir(parents=True)
    temp_archive.mkdir(parents=True)
    temp_figures.mkdir(parents=True)
    (temp_docs / "vllm_standalone_results_latest.html").write_text(
        "<!doctype html><title>latest</title>",
        encoding="utf-8",
    )
    (temp_docs / "vllm_standalone_results_20260619.html").write_text(
        "<!doctype html><title>6/19</title>",
        encoding="utf-8",
    )
    completed = pd.DataFrame(
        [
            {
                "status": "complete",
                "context_profile": "Native 32K",
                "domain": "SWE",
                "temperature": 1.0,
                "tok_s_gpu": 1.0,
                "acceptance_rate": 0.1,
                "mean_accept_len": 1.0,
                "job_id": "2272941",
            },
            {
                "status": "complete",
                "context_profile": "Native 32K",
                "domain": "Math",
                "temperature": 0.0,
                "tok_s_gpu": 2.0,
                "acceptance_rate": 0.2,
                "mean_accept_len": 2.0,
                "job_id": "2272937",
            },
        ]
    )
    completed_path = tmp_path / "report/dflare_completed_latest.csv"
    completed_path.parent.mkdir(parents=True)
    completed.to_csv(completed_path, index=False)
    status = pd.DataFrame(
        [
            {"job_id": "2274775", "state": "RUNNING"},
        ]
    )
    status_path = tmp_path / "report/dflare_job_status_latest.csv"
    status.to_csv(status_path, index=False)

    monkeypatch.setattr(index, "DOCS", temp_docs)
    monkeypatch.setattr(index, "PUBLIC", temp_public)
    monkeypatch.setattr(index, "REPORTS", temp_reports)
    monkeypatch.setattr(index, "DATA", temp_data)
    monkeypatch.setattr(index, "ARCHIVE", temp_archive)
    monkeypatch.setattr(index, "FIGURES", temp_figures)
    monkeypatch.setattr(index, "DFLARE_COMPLETED", completed_path)
    monkeypatch.setattr(index, "DFLARE_STATUS", status_path)
    index.build()

    html_text = parse_html(temp_public / "index.html")
    assert "job 2272941." in html_text
    assert "job 2272937." not in html_text
