from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import build_latest_specdec_html_pages as latest  # noqa: E402
from scripts import build_pages_index as index  # noqa: E402


NEMOTRON_SMOKE_ROOT = (
    ROOT
    / "experiments/vllm_024_dynamicsd/report/results/nemotron_mtp_smoke_20260704"
)
NEMOTRON_K_SWEEP_ROOT = (
    ROOT
    / "experiments/vllm_024_dynamicsd/report/results/"
    "nemotron_mtp_k_sweep_osl4k_20260706"
)
NEMOTRON_OSL16K_FULL_ROOT = (
    ROOT
    / "experiments/vllm_024_dynamicsd/report/results/"
    "nemotron_mtp_osl16k_20260706"
)
EXPECTED_NEMOTRON_SMOKE_RESULTS = (
    (
        "super",
        "baseline",
        "Nemotron-3-Super-120B-A12B-BF16",
        "2326451",
    ),
    (
        "super",
        "mtp_static",
        "Nemotron-3-Super-120B-A12B-BF16",
        "2326452",
    ),
    (
        "super",
        "mtp_dynamic",
        "Nemotron-3-Super-120B-A12B-BF16",
        "2326453",
    ),
    (
        "ultra",
        "baseline",
        "Nemotron-3-Ultra-550B-A55B-BF16",
        "2326448",
    ),
    (
        "ultra",
        "mtp_static",
        "Nemotron-3-Ultra-550B-A55B-BF16",
        "2326449",
    ),
    (
        "ultra",
        "mtp_dynamic",
        "Nemotron-3-Ultra-550B-A55B-BF16",
        "2326450",
    ),
)
EXPECTED_NEMOTRON_SMOKE_MODEL_PATHS = {
    "super": (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e"
    ),
    "ultra": (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04"
    ),
}
EXPECTED_NEMOTRON_K_SWEEP_RESULTS = (
    ("super", "baseline", 0, "2335027"),
    ("super", "k1", 1, "2335049"),
    ("super", "k3", 3, "2335028"),
    ("super", "k5", 5, "2335033"),
    ("ultra", "baseline", 0, "2335029"),
    ("ultra", "k1", 1, "2335295"),
    ("ultra", "k3", 3, "2335297"),
    ("ultra", "k5", 5, "2335030"),
)
EXPECTED_NEMOTRON_OSL16K_FULL_RESULTS = (
    ("super", "baseline", 0, "2335018"),
    ("super", "k3", 3, "2335019"),
    ("super", "k5", 5, "2335035"),
    ("ultra", "baseline", 0, "2335020"),
    ("ultra", "k5", 5, "2335021"),
)


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


@pytest.fixture
def nemotron_smoke_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(latest, "ROOT", tmp_path)
    result_root = tmp_path / NEMOTRON_SMOKE_ROOT.name
    shutil.copytree(NEMOTRON_SMOKE_ROOT, result_root)
    return result_root


@pytest.fixture
def nemotron_k_sweep_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(latest, "ROOT", tmp_path)
    result_root = tmp_path / NEMOTRON_K_SWEEP_ROOT.name
    shutil.copytree(NEMOTRON_K_SWEEP_ROOT, result_root)
    return result_root


@pytest.fixture
def nemotron_osl16k_full_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(latest, "ROOT", tmp_path)
    result_root = tmp_path / NEMOTRON_OSL16K_FULL_ROOT.name
    shutil.copytree(NEMOTRON_OSL16K_FULL_ROOT, result_root)
    return result_root


def replace_json_value(
    path: Path,
    field_path: tuple[str, ...],
    value: object,
) -> None:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    parent = payload
    for key in field_path[:-1]:
        child = parent[key]
        assert isinstance(child, dict)
        parent = child
    parent[field_path[-1]] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


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


@pytest.mark.parametrize(
    ("model_key", "mode", "_model", "_job_id"),
    EXPECTED_NEMOTRON_SMOKE_RESULTS,
    ids=lambda value: str(value),
)
def test_nemotron_legacy_smoke_rejects_each_missing_expected_payload(
    nemotron_smoke_root: Path,
    model_key: str,
    mode: str,
    _model: str,
    _job_id: str,
) -> None:
    relative_path = Path(model_key) / mode / "result.json"
    (nemotron_smoke_root / relative_path).unlink()

    with pytest.raises(ValueError, match=re.escape(relative_path.as_posix())):
        latest.load_nemotron_mtp_legacy_smoke_rows(nemotron_smoke_root)


def test_nemotron_legacy_smoke_rejects_unexpected_result_payload(
    nemotron_smoke_root: Path,
) -> None:
    relative_path = Path("unexpected/cohort/result.json")
    result_path = nemotron_smoke_root / relative_path
    result_path.parent.mkdir(parents=True)
    shutil.copy2(
        nemotron_smoke_root / "super/baseline/result.json",
        result_path,
    )

    with pytest.raises(ValueError, match="unexpected payloads") as error:
        latest.load_nemotron_mtp_legacy_smoke_rows(nemotron_smoke_root)

    assert relative_path.as_posix() in str(error.value)


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    (
        (("status",), "running"),
        (("runtime", "vllm_version"), "0.23.0"),
        (("config", "cudagraph_mode"), "FULL"),
        (("config", "compilation_config", "cudagraph_mode"), "FULL"),
        (("config", "max_new_tokens"), 256),
        (("config", "temperature"), 0.0),
        (("config", "top_p"), 1.0),
    ),
    ids=(
        "status",
        "vllm-version",
        "cudagraph-mode",
        "compiled-cudagraph-mode",
        "max-new-tokens",
        "temperature",
        "top-p",
    ),
)
def test_nemotron_legacy_smoke_rejects_mismatched_shared_metadata(
    nemotron_smoke_root: Path,
    field_path: tuple[str, ...],
    bad_value: object,
) -> None:
    result_path = nemotron_smoke_root / "super/baseline/result.json"
    replace_json_value(result_path, field_path, bad_value)
    field_name = ".".join(field_path)

    with pytest.raises(ValueError, match=re.escape(field_name)):
        latest.load_nemotron_mtp_legacy_smoke_rows(nemotron_smoke_root)


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    (
        (("config", "mode"), "wrong_method"),
        (("config", "model"), "/models/wrong-model"),
        (("runtime", "environment", "SLURM_JOB_ID"), "9999999"),
    ),
    ids=("method", "model", "job-id"),
)
@pytest.mark.parametrize(
    ("model_key", "mode", "expected_model", "expected_job_id"),
    EXPECTED_NEMOTRON_SMOKE_RESULTS,
    ids=lambda value: str(value),
)
def test_nemotron_legacy_smoke_rejects_mismatched_payload_identity(
    nemotron_smoke_root: Path,
    model_key: str,
    mode: str,
    expected_model: str,
    expected_job_id: str,
    field_path: tuple[str, ...],
    bad_value: object,
) -> None:
    result_path = nemotron_smoke_root / model_key / mode / "result.json"
    replace_json_value(result_path, field_path, bad_value)
    field_name = ".".join(field_path)
    expected_value = {
        "config.mode": mode,
        "config.model": expected_model,
        "runtime.environment.SLURM_JOB_ID": expected_job_id,
    }[field_name]

    with pytest.raises(ValueError, match=re.escape(field_name)) as error:
        latest.load_nemotron_mtp_legacy_smoke_rows(nemotron_smoke_root)

    assert expected_value in str(error.value)


@pytest.mark.parametrize(
    "model_key",
    ("super", "ultra"),
)
@pytest.mark.parametrize(
    "mutation",
    ("checkpoint-root", "snapshot-revision"),
)
def test_nemotron_legacy_smoke_rejects_wrong_checkpoint_path_or_revision(
    nemotron_smoke_root: Path,
    model_key: str,
    mutation: str,
) -> None:
    result_path = nemotron_smoke_root / model_key / "baseline/result.json"
    expected_model_path = EXPECTED_NEMOTRON_SMOKE_MODEL_PATHS[model_key]
    if mutation == "checkpoint-root":
        bad_model_path = expected_model_path.replace(
            "/users/sna/hf_home/",
            "/users/other/hf_home/",
        )
    else:
        bad_model_path = str(Path(expected_model_path).parent / ("0" * 40))
    replace_json_value(result_path, ("config", "model"), bad_model_path)

    with pytest.raises(ValueError, match=r"config\.model") as error:
        latest.load_nemotron_mtp_legacy_smoke_rows(nemotron_smoke_root)

    assert expected_model_path in str(error.value)


def test_nemotron_legacy_smoke_rejects_nonempty_runtime_image_sha256(
    nemotron_smoke_root: Path,
) -> None:
    result_path = nemotron_smoke_root / "super/baseline/result.json"
    replace_json_value(
        result_path,
        ("config", "runtime_image_sha256"),
        "sha256:unexpected",
    )

    with pytest.raises(ValueError, match="config.runtime_image_sha256"):
        latest.load_nemotron_mtp_legacy_smoke_rows(nemotron_smoke_root)


def test_nemotron_legacy_smoke_accepts_empty_runtime_image_sha256(
    nemotron_smoke_root: Path,
) -> None:
    result_path = nemotron_smoke_root / "super/baseline/result.json"
    replace_json_value(result_path, ("config", "runtime_image_sha256"), "")

    rows = latest.load_nemotron_mtp_legacy_smoke_rows(nemotron_smoke_root)

    assert len(rows) == 6


@pytest.mark.parametrize(
    "metric",
    ("total_output_tokens", "total_rollout_time_s", "output_tok_s_per_gpu"),
)
def test_nemotron_legacy_smoke_rejects_summary_aggregate_not_in_raw_batches(
    nemotron_smoke_root: Path,
    metric: str,
) -> None:
    result_path = nemotron_smoke_root / "super/mtp_static/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"][metric] += 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"summary\.{metric}"):
        latest.load_nemotron_mtp_legacy_smoke_rows(nemotron_smoke_root)


@pytest.mark.parametrize(
    "counter",
    (
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
        "num_accepted_tokens_per_pos",
    ),
)
def test_nemotron_legacy_smoke_rejects_spec_counters_not_in_raw_batches(
    nemotron_smoke_root: Path,
    counter: str,
) -> None:
    result_path = nemotron_smoke_root / "super/mtp_dynamic/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    value = payload["summary"]["spec_decode_metrics"][counter]
    if isinstance(value, list):
        value[0] += 1
    else:
        payload["summary"]["spec_decode_metrics"][counter] += 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"summary\.spec_decode_metrics\.{counter}",
    ):
        latest.load_nemotron_mtp_legacy_smoke_rows(nemotron_smoke_root)


def test_pages_report_publishes_and_links_exact_nemotron_smoke_evidence(
    tmp_path: Path,
) -> None:
    public_root = tmp_path / "public"
    report_path = public_root / "reports/vllm_standalone_results_latest.html"
    public_data = public_root / "data"
    latest.build_latest_vllm_outputs(
        output_html=report_path,
        added_csv_out=tmp_path / "docs/vllm_standalone_added_results_latest.csv",
        completed_csv_out=tmp_path / "report/dflare_completed_latest.csv",
        public_data_dir=public_data,
        nemotron_evidence_href_root="../data/nemotron_mtp_smoke_20260704",
    )

    html_text = parse_html(report_path)
    smoke_section = html_text.split(
        "<h2>Nemotron Native MTP Legacy Smoke</h2>", 1
    )[1].split("</section>", 1)[0]
    evidence_hrefs = re.findall(
        r'href="([^"]*nemotron_mtp_smoke_20260704/[^"]*/result\.json)"',
        smoke_section,
    )
    expected_relative_paths = {
        Path(model_key) / mode / "result.json"
        for model_key, mode, _model, _job_id in EXPECTED_NEMOTRON_SMOKE_RESULTS
    }

    assert len(evidence_hrefs) == 6
    assert {
        Path(href).relative_to("../data/nemotron_mtp_smoke_20260704")
        for href in evidence_hrefs
    } == expected_relative_paths
    for href in evidence_hrefs:
        assert (report_path.parent / href).resolve().is_file()

    evidence_root = public_data / "nemotron_mtp_smoke_20260704"
    published_paths = {
        path.relative_to(evidence_root)
        for path in evidence_root.glob("**/result.json")
    }
    assert published_paths == expected_relative_paths
    for relative_path in expected_relative_paths:
        assert sha256(evidence_root / relative_path) == sha256(
            NEMOTRON_SMOKE_ROOT / relative_path
        )


def test_latest_vllm_html_contains_separate_nemotron_native_mtp_legacy_smoke(
    tmp_path: Path,
) -> None:
    temp_html = tmp_path / "docs/vllm_standalone_results_latest.html"
    latest.build_latest_vllm_outputs(
        output_html=temp_html,
        added_csv_out=tmp_path / "docs/vllm_standalone_added_results_latest.csv",
        completed_csv_out=tmp_path / "report/dflare_completed_latest.csv",
        public_data_dir=tmp_path / "public/data",
    )

    html_text = parse_html(temp_html)
    heading = "Nemotron Native MTP Legacy Smoke"
    assert heading in html_text
    smoke_section = html_text.split(f"<h2>{heading}</h2>", 1)[1].split(
        "</section>", 1
    )[0]

    for label in (
        "Every row is legacy capability smoke",
        "vLLM 0.24.0",
        "CUDA Graph PIECEWISE",
        "OSL/max_new_tokens 128",
        "temperature 1.0",
        "top_p 0.95",
        "one measured realization",
        "runtime_image_sha256 missing",
        "uncalibrated dynamic schedules",
        "excluded from calibrated DynamicSD/DynamicMTP claims",
        "natural EOS",
    ):
        assert label in smoke_section

    for header in (
        "Model",
        "Method",
        "Job ID",
        "Output tok/s/GPU",
        "Baseline-relative throughput speedup",
        "Rollout-time speedup",
        "Output-token ratio",
        "Acceptance rate",
        "Mean acceptance length",
        "Static K / dynamic schedule",
        "Validity",
    ):
        assert f"<th>{header}</th>" in smoke_section

    table_body = smoke_section.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    rows = re.findall(r"<tr>.*?</tr>", table_body, flags=re.DOTALL)
    assert len(rows) == 6
    assert "Nemotron-3-Super-120B-A12B-BF16" in table_body
    assert "Nemotron-3-Ultra-550B-A55B-BF16" in table_body

    job_ids = ("2326451", "2326452", "2326453", "2326448", "2326449", "2326450")
    for job_id in job_ids:
        assert html_text.count(job_id) == 1
        row = next(row for row in rows if job_id in row)
        assert "legacy capability smoke" in row
        assert "one measured realization" in row
        assert "runtime_image_sha256 missing" in row

    result_root = (
        "../experiments/vllm_024_dynamicsd/report/results/"
        "nemotron_mtp_smoke_20260704/"
    )
    assert smoke_section.count(f'href="{result_root}') == 6

    super_static = next(row for row in rows if "2326452" in row)
    assert "4011/3968 = 1.0108x" in super_static
    assert "n/a (invalid: output-token ratio outside 1%)" in super_static
    assert "1.58x" not in super_static

    super_dynamic = next(row for row in rows if "2326453" in row)
    assert "3959/3968 = 0.9977x" in super_dynamic
    assert "0.99x (directional only)" in super_dynamic
    assert "uncalibrated" in super_dynamic
    assert "natural EOS" in super_dynamic

    ultra_static = next(row for row in rows if "2326449" in row)
    ultra_dynamic = next(row for row in rows if "2326450" in row)
    assert "4096/4096 = 1.0000x" in ultra_static
    assert "1.66x (directional only)" in ultra_static
    assert "4096/4096 = 1.0000x" in ultra_dynamic
    assert "1.53x (directional only)" in ultra_dynamic


@pytest.mark.parametrize(
    ("model_key", "method_key", "_k", "_job_id"),
    EXPECTED_NEMOTRON_K_SWEEP_RESULTS,
    ids=lambda value: str(value),
)
def test_nemotron_k_sweep_rejects_each_missing_expected_payload(
    nemotron_k_sweep_root: Path,
    model_key: str,
    method_key: str,
    _k: int,
    _job_id: str,
) -> None:
    relative_path = Path(model_key) / method_key / "result.json"
    (nemotron_k_sweep_root / relative_path).unlink()

    with pytest.raises(ValueError, match=re.escape(relative_path.as_posix())):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_unexpected_result_payload(
    nemotron_k_sweep_root: Path,
) -> None:
    relative_path = Path("super/k7/result.json")
    result_path = nemotron_k_sweep_root / relative_path
    result_path.parent.mkdir(parents=True)
    shutil.copy2(
        nemotron_k_sweep_root / "super/k5/result.json",
        result_path,
    )

    with pytest.raises(ValueError, match="unexpected payloads") as error:
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)

    assert relative_path.as_posix() in str(error.value)


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    (
        (("status",), "running"),
        (("runtime", "vllm_version"), "0.23.0"),
        (("config", "runtime_image_sha256"), "wrong-image-sha"),
        (("config", "cudagraph_mode"), "FULL"),
        (("config", "compilation_config", "cudagraph_mode"), "FULL"),
        (("config", "temperature"), 0.0),
        (("config", "top_p"), 0.95),
        (("config", "max_new_tokens"), 2048),
        (("config", "num_prompts"), 4),
        (("config", "samples_per_prompt"), 2),
        (("config", "rollout_batches"), 2),
        (("config", "scenario"), "offline_generation"),
        (("config", "sync_barrier"), "none"),
        (("config", "source_recipe"), "other"),
    ),
    ids=(
        "status",
        "vllm-version",
        "runtime-image",
        "cudagraph-mode",
        "compiled-cudagraph-mode",
        "temperature",
        "top-p",
        "max-new-tokens",
        "num-prompts",
        "samples-per-prompt",
        "rollout-barriers",
        "scenario",
        "sync-barrier",
        "source-recipe",
    ),
)
def test_nemotron_k_sweep_rejects_mismatched_shared_metadata(
    nemotron_k_sweep_root: Path,
    field_path: tuple[str, ...],
    bad_value: object,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    replace_json_value(result_path, field_path, bad_value)

    with pytest.raises(ValueError, match=re.escape(".".join(field_path))):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    ("relative_path", "field_path", "bad_value"),
    (
        (
            "super/k3/result.json",
            ("config", "enable_mamba_cache_stochastic_rounding"),
            True,
        ),
        (
            "ultra/k3/result.json",
            ("config", "enable_mamba_cache_stochastic_rounding"),
            False,
        ),
        (
            "ultra/k3/result.json",
            ("config", "mamba_cache_philox_rounds"),
            4,
        ),
    ),
    ids=("super-rounding", "ultra-rounding", "ultra-philox-rounds"),
)
def test_nemotron_k_sweep_rejects_model_specific_mamba_cache_mismatch(
    nemotron_k_sweep_root: Path,
    relative_path: str,
    field_path: tuple[str, ...],
    bad_value: object,
) -> None:
    result_path = nemotron_k_sweep_root / relative_path
    replace_json_value(result_path, field_path, bad_value)

    with pytest.raises(ValueError, match=re.escape(".".join(field_path))):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    (
        (("config", "temperature"), 1),
        (("config", "top_p"), True),
        (("config", "seed"), 1234.0),
        (("config", "tensor_parallel_size"), 2.0),
        (("config", "topology", "tensor_parallel_size"), 2.0),
        (("config", "node_count"), 1.0),
        (("config", "topology", "nodes"), True),
        (("config", "pipeline_parallel_size"), 1.0),
        (("config", "total_gpus"), 2.0),
        (("runtime", "gpu_count"), 4.0),
    ),
    ids=(
        "temperature-int",
        "top-p-bool",
        "seed-float",
        "config-tp-float",
        "topology-tp-float",
        "config-nodes-float",
        "topology-nodes-bool",
        "config-pp-float",
        "total-gpus-float",
        "runtime-gpu-count-float",
    ),
)
def test_nemotron_k_sweep_rejects_non_strict_json_scalar_types(
    nemotron_k_sweep_root: Path,
    field_path: tuple[str, ...],
    bad_value: object,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    replace_json_value(result_path, field_path, bad_value)

    with pytest.raises(ValueError, match=re.escape(".".join(field_path))):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    (
        (("config", "model_config_hash"), "0" * 64),
        (("config", "model_checkpoint_hash"), "0" * 64),
        (("config", "model_view_marker_hash"), "0" * 64),
        (("config", "drafter_config_hash"), "0" * 64),
        (("config", "drafter_checkpoint_hash"), "0" * 64),
        (("config", "drafter_view_marker_hash"), "0" * 64),
        (("config", "prompt_set_hash"), "0" * 64),
        (("config", "prompt_batch_hashes"), ["0" * 64] * 3),
        (("config", "pipeline_parallel_size"), 2),
        (("config", "total_gpus"), 4),
        (("config", "distributed_executor_backend"), "ray"),
        (("config", "topology", "pipeline_parallel_size"), 2),
        (("config", "topology", "distributed_executor_backend"), "ray"),
        (("config", "context_profile"), "unverified"),
        (("config", "rope_config_hash"), "0" * 64),
    ),
    ids=(
        "model-config-hash",
        "model-checkpoint-hash",
        "model-view-marker-hash",
        "drafter-config-hash",
        "drafter-checkpoint-hash",
        "drafter-view-marker-hash",
        "prompt-set-hash",
        "prompt-batch-hashes",
        "pipeline-parallel-size",
        "total-gpus",
        "executor-backend",
        "topology-pipeline-parallel-size",
        "topology-executor-backend",
        "context-profile",
        "rope-config-hash",
    ),
)
def test_nemotron_k_sweep_rejects_wrong_cohort_identity(
    nemotron_k_sweep_root: Path,
    field_path: tuple[str, ...],
    bad_value: object,
) -> None:
    result_path = nemotron_k_sweep_root / "super/k1/result.json"
    replace_json_value(result_path, field_path, bad_value)

    with pytest.raises(ValueError, match=re.escape(".".join(field_path))):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    "hash_field",
    ("prompt_sha256", "source_prompt_sha256"),
)
def test_nemotron_k_sweep_rejects_request_prompt_hash_mismatch(
    nemotron_k_sweep_root: Path,
    hash_field: str,
) -> None:
    result_path = nemotron_k_sweep_root / "super/k1/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["rollout_batches"][0]["requests"][0][hash_field] = "0" * 64
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=hash_field):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_coherent_request_provenance_mutation(
    nemotron_k_sweep_root: Path,
) -> None:
    for model_key, method_key, _k, _job_id in EXPECTED_NEMOTRON_K_SWEEP_RESULTS:
        result_path = nemotron_k_sweep_root / model_key / method_key / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        request = payload["rollout_batches"][0]["requests"][0]
        request["prompt_id"] = "coherently-mutated-prompt"
        request["prompt_sha256"] = "0" * 64
        request["source_prompt_sha256"] = "1" * 64
        result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="request_provenance_hash"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_wrong_config_seed(
    nemotron_k_sweep_root: Path,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    replace_json_value(result_path, ("config", "seed"), 4321)

    with pytest.raises(ValueError, match=r"config\.seed"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    ("batch_index", "request_index", "field", "bad_value"),
    (
        (0, 0, "seed", 4321),
        (1, 17, "seed", 4321),
        (2, 31, "seed", 4321),
        (0, 1, "sample_index", 3),
        (1, 2, "min_tokens", 1),
        (1, 3, "max_tokens", 2048),
        (2, 4, "ignore_eos", True),
        (2, 5, "prompt_tokens", 1),
    ),
    ids=(
        "first-batch-seed",
        "middle-batch-seed",
        "last-batch-seed",
        "sample-index",
        "min-tokens",
        "max-tokens",
        "ignore-eos",
        "prompt-tokens",
    ),
)
def test_nemotron_k_sweep_rejects_wrong_request_seed_or_protocol(
    nemotron_k_sweep_root: Path,
    batch_index: int,
    request_index: int,
    field: str,
    bad_value: object,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["rollout_batches"][batch_index]["requests"][request_index][field] = (
        bad_value
    )
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    ("model_key", "method_key", "_k", "expected_job_id"),
    EXPECTED_NEMOTRON_K_SWEEP_RESULTS,
    ids=lambda value: str(value),
)
def test_nemotron_k_sweep_rejects_wrong_job_id_for_every_payload(
    nemotron_k_sweep_root: Path,
    model_key: str,
    method_key: str,
    _k: int,
    expected_job_id: str,
) -> None:
    result_path = nemotron_k_sweep_root / model_key / method_key / "result.json"
    replace_json_value(
        result_path,
        ("runtime", "environment", "SLURM_JOB_ID"),
        "9999999",
    )

    with pytest.raises(ValueError, match="runtime.environment.SLURM_JOB_ID") as error:
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)

    assert expected_job_id in str(error.value)


@pytest.mark.parametrize("model_key", ("super", "ultra"))
def test_nemotron_k_sweep_rejects_wrong_checkpoint_revision(
    nemotron_k_sweep_root: Path,
    model_key: str,
) -> None:
    result_path = nemotron_k_sweep_root / model_key / "baseline/result.json"
    expected_model_path = EXPECTED_NEMOTRON_SMOKE_MODEL_PATHS[model_key]
    replace_json_value(
        result_path,
        ("config", "model"),
        str(Path(expected_model_path).parent / ("0" * 40)),
    )

    with pytest.raises(ValueError, match=r"config\.model") as error:
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)

    assert expected_model_path in str(error.value)


@pytest.mark.parametrize(
    ("model_key", "field_path", "bad_value"),
    (
        ("super", ("config", "topology", "tensor_parallel_size"), 4),
        ("super", ("config", "topology", "nodes"), 2),
        ("ultra", ("config", "topology", "tensor_parallel_size"), 4),
        ("ultra", ("config", "topology", "nodes"), 1),
    ),
    ids=("super-tp", "super-nodes", "ultra-tp", "ultra-nodes"),
)
def test_nemotron_k_sweep_rejects_wrong_topology(
    nemotron_k_sweep_root: Path,
    model_key: str,
    field_path: tuple[str, ...],
    bad_value: object,
) -> None:
    result_path = nemotron_k_sweep_root / model_key / "baseline/result.json"
    replace_json_value(result_path, field_path, bad_value)

    with pytest.raises(ValueError, match=re.escape(".".join(field_path))):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    ("model_key", "expected_active_gpus", "bad_runtime_gpu_count"),
    (("super", 2, 2), ("ultra", 8, 8)),
)
def test_nemotron_k_sweep_rejects_wrong_runtime_gpu_inventory(
    nemotron_k_sweep_root: Path,
    model_key: str,
    expected_active_gpus: int,
    bad_runtime_gpu_count: int,
) -> None:
    result_path = nemotron_k_sweep_root / model_key / "baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["config"]["total_gpus"] == expected_active_gpus
    payload["runtime"]["gpu_count"] = bad_runtime_gpu_count
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"runtime\.gpu_count"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    ("relative_path", "field_path", "bad_value"),
    (
        ("super/baseline/result.json", ("config", "speculative_config"), {}),
        ("super/k1/result.json", ("config", "mode"), "mtp_dynamic"),
        (
            "super/k3/result.json",
            ("config", "speculative_config", "num_speculative_tokens"),
            1,
        ),
    ),
    ids=("baseline-spec-config", "fixed-k-mode", "fixed-k-value"),
)
def test_nemotron_k_sweep_rejects_wrong_fixed_k_method_configuration(
    nemotron_k_sweep_root: Path,
    relative_path: str,
    field_path: tuple[str, ...],
    bad_value: object,
) -> None:
    result_path = nemotron_k_sweep_root / relative_path
    replace_json_value(result_path, field_path, bad_value)

    with pytest.raises(ValueError, match=re.escape(".".join(field_path))):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_wrong_rollout_barrier_count(
    nemotron_k_sweep_root: Path,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["rollout_batches"].pop()
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rollout_batches"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_non_natural_eos_request(
    nemotron_k_sweep_root: Path,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["rollout_batches"][0]["requests"][0]["ignore_eos"] = True
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ignore_eos"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_summary_csv_row_or_metric_mismatch(
    nemotron_k_sweep_root: Path,
) -> None:
    summary_path = nemotron_k_sweep_root / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[summary["job_id"] == 2335033, "output_tok_s_per_gpu"] = 1.0
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match="output_tok_s_per_gpu"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_fractional_csv_integer_counter(
    nemotron_k_sweep_root: Path,
) -> None:
    summary_path = nemotron_k_sweep_root / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary["total_output_tokens"] = summary["total_output_tokens"].astype(float)
    row = summary["result_path"] == "super/baseline/result.json"
    summary.loc[row, "total_output_tokens"] += 1e-8
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match="total_output_tokens"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_fractional_spec_summary_counter(
    nemotron_k_sweep_root: Path,
) -> None:
    result_path = nemotron_k_sweep_root / "super/k3/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"]["spec_decode_metrics"]["num_drafts"] += 1e-8
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"spec_decode_metrics\.num_drafts"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_validates_required_csv_columns_before_result_path(
    nemotron_k_sweep_root: Path,
) -> None:
    summary_path = nemotron_k_sweep_root / "summary.csv"
    summary = pd.read_csv(summary_path).drop(columns="result_path")
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match="missing columns: result_path"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    "metric",
    ("total_output_tokens", "total_rollout_time_s", "output_tok_s_per_gpu"),
)
def test_nemotron_k_sweep_rejects_summary_and_csv_aggregate_not_in_raw_batches(
    nemotron_k_sweep_root: Path,
    metric: str,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"][metric] += 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    summary_path = nemotron_k_sweep_root / "summary.csv"
    summary = pd.read_csv(summary_path)
    row = summary["result_path"] == "super/baseline/result.json"
    summary.loc[row, metric] = payload["summary"][metric]
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match=rf"summary\.{metric}"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_batch_output_tokens_not_backed_by_requests(
    nemotron_k_sweep_root: Path,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["rollout_batches"][0]["output_tokens"] += 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="actual_output_tokens"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    "vector_field",
    (
        "actual_output_tokens",
        "planned_output_tokens",
        "forced_output_mask",
        "output_token_hashes",
    ),
)
def test_nemotron_k_sweep_rejects_truncated_per_request_vector(
    nemotron_k_sweep_root: Path,
    vector_field: str,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    batch = payload["rollout_batches"][0]
    removed = batch[vector_field].pop()
    if vector_field == "actual_output_tokens":
        batch[vector_field][0] += removed
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=vector_field):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize("bad_value", (True, 0, None))
def test_nemotron_k_sweep_requires_every_forced_output_mask_value_false(
    nemotron_k_sweep_root: Path,
    bad_value: object,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["rollout_batches"][0]["forced_output_mask"][0] = bad_value
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"forced_output_mask\[0\]"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    ("vector_field", "bad_value"),
    (
        ("actual_output_tokens", 4097),
        ("actual_output_tokens", 1.0),
        ("planned_output_tokens", -1),
        ("planned_output_tokens", 4095),
        ("planned_output_tokens", 4097),
        ("planned_output_tokens", True),
    ),
    ids=(
        "actual-above-planned",
        "actual-float",
        "planned-negative",
        "planned-below-request-cap",
        "planned-above-request-cap",
        "planned-bool",
    ),
)
def test_nemotron_k_sweep_rejects_invalid_natural_eos_token_vectors(
    nemotron_k_sweep_root: Path,
    vector_field: str,
    bad_value: object,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    batch = payload["rollout_batches"][0]
    original = batch[vector_field][0]
    batch[vector_field][0] = bad_value
    if vector_field == "actual_output_tokens" and isinstance(bad_value, int):
        batch["output_tokens"] += bad_value - original
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{vector_field}\[0\]"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    "bad_hash",
    ("", "0" * 63, "g" * 64, "A" * 64, 0, None),
    ids=("empty", "short", "non-hex", "uppercase", "integer", "null"),
)
def test_nemotron_k_sweep_rejects_invalid_output_token_hash(
    nemotron_k_sweep_root: Path,
    bad_hash: object,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["rollout_batches"][0]["output_token_hashes"][0] = bad_hash
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"output_token_hashes\[0\]"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_raw_rollout_time_summary_mismatch(
    nemotron_k_sweep_root: Path,
) -> None:
    result_path = nemotron_k_sweep_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    batch = payload["rollout_batches"][0]
    batch["rollout_time_s"] += 1.0
    batch["output_tok_s"] = batch["output_tokens"] / batch["rollout_time_s"]
    batch["output_tok_s_per_gpu"] = batch["output_tok_s"] / 2
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"summary\.total_rollout_time_s"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    ("metric", "csv_metric"),
    (
        ("num_drafts", None),
        ("num_draft_tokens", None),
        ("num_accepted_tokens", None),
        ("num_accepted_tokens_per_pos", None),
        ("acceptance_rate", "acceptance_rate"),
        ("mean_acceptance_length", "mean_accept_len"),
        ("accepted_tokens_per_draft", None),
        ("acceptance_rate_per_pos", None),
    ),
)
def test_nemotron_k_sweep_rejects_spec_summary_not_derived_from_batches(
    nemotron_k_sweep_root: Path,
    metric: str,
    csv_metric: str | None,
) -> None:
    result_path = nemotron_k_sweep_root / "super/k3/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    current = payload["summary"]["spec_decode_metrics"][metric]
    if isinstance(current, list):
        payload["summary"]["spec_decode_metrics"][metric][0] += 1
    else:
        payload["summary"]["spec_decode_metrics"][metric] += 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    if csv_metric is not None:
        summary_path = nemotron_k_sweep_root / "summary.csv"
        summary = pd.read_csv(summary_path)
        row = summary["result_path"] == "super/k3/result.json"
        summary.loc[row, csv_metric] = payload["summary"]["spec_decode_metrics"][
            metric
        ]
        summary.to_csv(summary_path, index=False)

    with pytest.raises(
        ValueError,
        match=rf"summary\.spec_decode_metrics\.{metric}",
    ):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


def test_nemotron_k_sweep_rejects_unexpected_summary_csv_row(
    nemotron_k_sweep_root: Path,
) -> None:
    summary_path = nemotron_k_sweep_root / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary = pd.concat([summary, summary.iloc[[0]]], ignore_index=True)
    summary.loc[len(summary) - 1, "result_path"] = "super/k7/result.json"
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match="summary.csv"):
        latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)


@pytest.mark.parametrize(
    ("output_tokens", "baseline_tokens", "expected"),
    (
        (99, 100, True),
        (101, 100, True),
        (98, 100, False),
        (102, 100, False),
        (99 * 10**30, 100 * 10**30, True),
        (101 * 10**30 + 1, 100 * 10**30, False),
    ),
)
def test_output_work_validity_uses_inclusive_exact_integer_bounds(
    output_tokens: int,
    baseline_tokens: int,
    expected: bool,
) -> None:
    assert (
        latest._output_work_within_one_percent(output_tokens, baseline_tokens)
        is expected
    )


def test_nemotron_k_sweep_rows_derive_metrics_and_gate_rollout_speedup(
    nemotron_k_sweep_root: Path,
) -> None:
    rows = latest.load_nemotron_mtp_k_sweep_rows(nemotron_k_sweep_root)

    assert len(rows) == 8
    assert rows["job_id"].tolist() == [
        job_id for _model, _method, _k, job_id in EXPECTED_NEMOTRON_K_SWEEP_RESULTS
    ]
    for model_key in ("super", "ultra"):
        model_rows = rows[rows["model_key"] == model_key].set_index("method_key")
        baseline_payload = json.loads(
            (nemotron_k_sweep_root / model_key / "baseline/result.json").read_text(
                encoding="utf-8"
            )
        )
        baseline_tokens = sum(
            batch["output_tokens"] for batch in baseline_payload["rollout_batches"]
        )
        baseline_time = sum(
            batch["rollout_time_s"] for batch in baseline_payload["rollout_batches"]
        )
        baseline_tok_s_gpu = (
            baseline_tokens / baseline_time / baseline_payload["config"]["total_gpus"]
        )
        for method_key in ("baseline", "k1", "k3", "k5"):
            payload = json.loads(
                (nemotron_k_sweep_root / model_key / method_key / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            rollout_batches = payload["rollout_batches"]
            output_tokens = sum(batch["output_tokens"] for batch in rollout_batches)
            rollout_time = sum(batch["rollout_time_s"] for batch in rollout_batches)
            output_tok_s_gpu = (
                output_tokens / rollout_time / payload["config"]["total_gpus"]
            )
            row = model_rows.loc[method_key]
            output_ratio = output_tokens / baseline_tokens
            assert row["output_tok_s_gpu"] == pytest.approx(output_tok_s_gpu)
            assert row["throughput_speedup"] == pytest.approx(
                output_tok_s_gpu / baseline_tok_s_gpu
            )
            assert row["output_token_ratio"] == pytest.approx(output_ratio)
            if latest._output_work_within_one_percent(
                output_tokens,
                baseline_tokens,
            ):
                assert row["rollout_time_speedup"] == pytest.approx(
                    baseline_time / rollout_time
                )
            else:
                assert pd.isna(row["rollout_time_speedup"])
            if method_key != "baseline":
                num_drafts = sum(
                    batch["spec_decode_metrics"]["num_drafts"]
                    for batch in rollout_batches
                )
                num_draft_tokens = sum(
                    batch["spec_decode_metrics"]["num_draft_tokens"]
                    for batch in rollout_batches
                )
                num_accepted_tokens = sum(
                    batch["spec_decode_metrics"]["num_accepted_tokens"]
                    for batch in rollout_batches
                )
                assert row["acceptance_rate"] == pytest.approx(
                    num_accepted_tokens / num_draft_tokens
                )
                assert row["mean_acceptance_length"] == pytest.approx(
                    1.0 + num_accepted_tokens / num_drafts
                )

    super_rows = rows[rows["model_key"] == "super"].set_index("method_key")
    ultra_rows = rows[rows["model_key"] == "ultra"].set_index("method_key")
    assert super_rows.loc["k5", "throughput_speedup"] == pytest.approx(
        1.6863337864738492
    )
    assert ultra_rows.loc["k5", "throughput_speedup"] == pytest.approx(
        1.9729748161942076
    )
    assert ultra_rows.loc["k1", "throughput_speedup"] == pytest.approx(
        1.3471281687409997
    )
    assert ultra_rows.loc["k3", "throughput_speedup"] == pytest.approx(
        1.8754073247286647
    )
    assert bool(ultra_rows.loc["k3", "time_speedup_valid"])
    assert not bool(ultra_rows.loc["k5", "time_speedup_valid"])
    assert bool(ultra_rows.loc["k3", "selected_by_policy"])
    assert not bool(ultra_rows.loc["k5", "selected_by_policy"])


def test_latest_vllm_html_contains_validated_nemotron_native_mtp_k_sweep(
    tmp_path: Path,
) -> None:
    temp_html = tmp_path / "docs/vllm_standalone_results_latest.html"
    latest.build_latest_vllm_outputs(
        output_html=temp_html,
        added_csv_out=tmp_path / "docs/vllm_standalone_added_results_latest.csv",
        completed_csv_out=tmp_path / "report/dflare_completed_latest.csv",
        public_data_dir=tmp_path / "public/data",
    )

    html_text = parse_html(temp_html)
    heading = "Nemotron Native MTP OSL 4K K Sweep"
    assert html_text.count(heading) == 1
    section = html_text.split(f"<h2>{heading}</h2>", 1)[1].split("</section>", 1)[0]
    legacy_section = html_text.split(
        "<h2>Nemotron Native MTP Legacy Smoke</h2>", 1
    )[1].split("</section>", 1)[0]

    assert heading not in legacy_section
    assert "validated fixed-K evidence, not DynamicMTP" in section
    assert "OpenMath natural-EOS Sync-RL-style rollout, so output work can differ" in section
    assert "Super TP2 / 1 node" in section
    assert "Ultra TP8 / 2 nodes" in section
    assert "Super best K5 at 1.686x" in section
    assert "Ultra absolute best K5 at 1.973x" in section
    assert "matched-work K3 reaches 1.875x" in section
    assert "Mamba cache stochastic rounding with 5 Philox rounds" in section
    assert "smallest-K-within-2% policy" in section
    assert 'aria-label="Nemotron Native MTP OSL 4K throughput speedup by fixed K' in section
    assert 'text-anchor="middle"' in section
    assert "1.0x baseline" in section
    assert ">1.69x</text>" in section
    chart = section.split("<svg", 1)[1].split("</svg>", 1)[0]
    assert ">0</text>" not in chart
    assert ">Super<" in section
    assert ">Ultra<" in section

    for header in (
        "Model",
        "Method",
        "Job ID",
        "tok/s/GPU",
        "Throughput speedup",
        "Rollout-time speedup",
        "Output ratio",
        "Acceptance rate",
        "Mean acceptance length",
        "Validity",
    ):
        assert f"<th>{header}</th>" in section

    table_body = section.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    table_rows = re.findall(r"<tr>.*?</tr>", table_body, flags=re.DOTALL)
    assert len(table_rows) == 8
    for _model, _method, _k, job_id in EXPECTED_NEMOTRON_K_SWEEP_RESULTS:
        assert section.count(job_id) == 1

    super_k5 = next(row for row in table_rows if "2335033" in row)
    assert "1.686x" in super_k5
    assert "1.688x" in super_k5
    ultra_k3 = next(row for row in table_rows if "2335297" in row)
    assert "1.875x" in ultra_k3
    assert "1.884x" in ultra_k3
    assert "selected by matched-work smallest-K-within-2% policy" in ultra_k3


def test_nemotron_k_sweep_chart_uses_shared_explicit_model_series_colors() -> None:
    section = latest.render_nemotron_mtp_k_sweep_section()
    chart = section.split("<svg", 1)[1].split("</svg>", 1)[0]

    assert latest.NEMOTRON_MODEL_SERIES_COLORS == {
        "Super": "#2563eb",
        "Ultra": "#dc2626",
    }
    for model, color in latest.NEMOTRON_MODEL_SERIES_COLORS.items():
        assert re.search(
            rf'<rect[^>]+fill="{re.escape(color)}"[^>]*>'
            rf'<text[^>]*>{model}</text>',
            chart,
        )
        assert re.search(rf'<polyline[^>]+stroke="{re.escape(color)}"', chart)


def test_pages_report_publishes_and_links_exact_nemotron_k_sweep_evidence(
    tmp_path: Path,
) -> None:
    public_root = tmp_path / "public"
    report_path = public_root / "reports/vllm_standalone_results_latest.html"
    public_data = public_root / "data"
    latest.build_latest_vllm_outputs(
        output_html=report_path,
        added_csv_out=tmp_path / "docs/vllm_standalone_added_results_latest.csv",
        completed_csv_out=tmp_path / "report/dflare_completed_latest.csv",
        public_data_dir=public_data,
        nemotron_evidence_href_root="../data/nemotron_mtp_smoke_20260704",
        nemotron_k_sweep_evidence_href_root=(
            "../data/nemotron_mtp_k_sweep_osl4k_20260706"
        ),
    )

    html_text = parse_html(report_path)
    section = html_text.split(
        "<h2>Nemotron Native MTP OSL 4K K Sweep</h2>", 1
    )[1].split("</section>", 1)[0]
    href_root = "../data/nemotron_mtp_k_sweep_osl4k_20260706"
    evidence_hrefs = re.findall(
        rf'href="({re.escape(href_root)}/[^"]*/result\.json)"',
        section,
    )
    expected_relative_paths = {
        Path(model_key) / method_key / "result.json"
        for model_key, method_key, _k, _job_id in EXPECTED_NEMOTRON_K_SWEEP_RESULTS
    }

    assert len(evidence_hrefs) == 8
    assert {
        Path(href).relative_to(href_root) for href in evidence_hrefs
    } == expected_relative_paths
    for href in evidence_hrefs:
        assert (report_path.parent / href).resolve().is_file()

    evidence_root = public_data / "nemotron_mtp_k_sweep_osl4k_20260706"
    published_paths = {
        path.relative_to(evidence_root)
        for path in evidence_root.glob("**/result.json")
    }
    assert published_paths == expected_relative_paths
    for relative_path in expected_relative_paths:
        assert sha256(evidence_root / relative_path) == sha256(
            NEMOTRON_K_SWEEP_ROOT / relative_path
        )


@pytest.mark.parametrize(
    ("model_key", "method_key", "_k", "_job_id"),
    EXPECTED_NEMOTRON_OSL16K_FULL_RESULTS,
    ids=lambda value: str(value),
)
def test_nemotron_osl16k_full_rejects_each_missing_expected_payload(
    nemotron_osl16k_full_root: Path,
    model_key: str,
    method_key: str,
    _k: int,
    _job_id: str,
) -> None:
    relative_path = Path(model_key) / method_key / "result.json"
    (nemotron_osl16k_full_root / relative_path).unlink()

    with pytest.raises(ValueError, match=re.escape(relative_path.as_posix())):
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)


def test_nemotron_osl16k_full_rejects_unexpected_result_payload(
    nemotron_osl16k_full_root: Path,
) -> None:
    relative_path = Path("ultra/k3/result.json")
    result_path = nemotron_osl16k_full_root / relative_path
    result_path.parent.mkdir(parents=True)
    shutil.copy2(
        nemotron_osl16k_full_root / "ultra/k5/result.json",
        result_path,
    )

    with pytest.raises(ValueError, match="unexpected payloads") as error:
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)

    assert relative_path.as_posix() in str(error.value)


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    (
        (("status",), "running"),
        (("runtime", "vllm_version"), "0.23.0"),
        (("runtime", "gpu_count"), 8),
        (("config", "runtime_image_sha256"), "wrong-image-sha"),
        (("config", "cudagraph_mode"), "FULL"),
        (("config", "compilation_config", "cudagraph_mode"), "FULL"),
        (("config", "temperature"), 0.0),
        (("config", "top_p"), 0.95),
        (("config", "max_new_tokens"), 4096),
        (("config", "num_prompts"), 8),
        (("config", "samples_per_prompt"), 2),
        (("config", "rollout_batches"), 2),
        (("config", "requests_per_rollout_batch"), 32),
        (("config", "seed"), 4321),
        (("config", "scenario"), "offline_generation"),
        (("config", "sync_barrier"), "none"),
        (("config", "source_recipe"), "other"),
        (("config", "prompt_jsonl"), "/tmp/other_openmath_prompts.jsonl"),
        (("config", "prompt_set_hash"), "0" * 64),
        (("config", "prompt_batch_hashes"), ["0" * 64] * 3),
        (("config", "model_config_hash"), "0" * 64),
        (("config", "model_checkpoint_hash"), "0" * 64),
        (("config", "model_view_marker_hash"), "0" * 64),
        (("config", "drafter_checkpoint_hash"), "0" * 64),
        (("config", "context_profile"), "unverified"),
        (("config", "rope_config_hash"), "0" * 64),
    ),
    ids=lambda value: str(value),
)
def test_nemotron_osl16k_full_rejects_wrong_cohort_identity(
    nemotron_osl16k_full_root: Path,
    field_path: tuple[str, ...],
    bad_value: object,
) -> None:
    result_path = nemotron_osl16k_full_root / "super/baseline/result.json"
    replace_json_value(result_path, field_path, bad_value)

    with pytest.raises(ValueError, match=re.escape(".".join(field_path))):
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)


@pytest.mark.parametrize(
    ("model_key", "method_key", "_k", "expected_job_id"),
    EXPECTED_NEMOTRON_OSL16K_FULL_RESULTS,
    ids=lambda value: str(value),
)
def test_nemotron_osl16k_full_rejects_wrong_job_id_for_every_payload(
    nemotron_osl16k_full_root: Path,
    model_key: str,
    method_key: str,
    _k: int,
    expected_job_id: str,
) -> None:
    result_path = nemotron_osl16k_full_root / model_key / method_key / "result.json"
    replace_json_value(
        result_path,
        ("runtime", "environment", "SLURM_JOB_ID"),
        "9999999",
    )

    with pytest.raises(ValueError, match="runtime.environment.SLURM_JOB_ID") as error:
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)

    assert expected_job_id in str(error.value)


@pytest.mark.parametrize("model_key", ("super", "ultra"))
def test_nemotron_osl16k_full_rejects_wrong_checkpoint_revision(
    nemotron_osl16k_full_root: Path,
    model_key: str,
) -> None:
    result_path = nemotron_osl16k_full_root / model_key / "baseline/result.json"
    expected_model_path = EXPECTED_NEMOTRON_SMOKE_MODEL_PATHS[model_key]
    replace_json_value(
        result_path,
        ("config", "model"),
        str(Path(expected_model_path).parent / ("0" * 40)),
    )

    with pytest.raises(ValueError, match=r"config\.model") as error:
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)

    assert expected_model_path in str(error.value)


@pytest.mark.parametrize(
    ("model_key", "field_path", "bad_value"),
    (
        ("super", ("config", "topology", "tensor_parallel_size"), 4),
        ("super", ("config", "node_count"), 2),
        ("ultra", ("config", "tensor_parallel_size"), 4),
        ("ultra", ("config", "topology", "nodes"), 1),
    ),
    ids=("super-tp", "super-nodes", "ultra-tp", "ultra-nodes"),
)
def test_nemotron_osl16k_full_rejects_wrong_topology(
    nemotron_osl16k_full_root: Path,
    model_key: str,
    field_path: tuple[str, ...],
    bad_value: object,
) -> None:
    result_path = nemotron_osl16k_full_root / model_key / "baseline/result.json"
    replace_json_value(result_path, field_path, bad_value)

    with pytest.raises(ValueError, match=re.escape(".".join(field_path))):
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("seed", 4321),
        ("sample_index", 3),
        ("max_tokens", 4096),
        ("min_tokens", 1),
        ("ignore_eos", True),
    ),
)
def test_nemotron_osl16k_full_rejects_wrong_request_protocol(
    nemotron_osl16k_full_root: Path,
    field: str,
    bad_value: object,
) -> None:
    result_path = nemotron_osl16k_full_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["rollout_batches"][1]["requests"][2][field] = bad_value
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)


def test_nemotron_osl16k_full_rejects_coherent_request_provenance_mutation(
    nemotron_osl16k_full_root: Path,
) -> None:
    for model_key, method_key, _k, _job_id in (
        EXPECTED_NEMOTRON_OSL16K_FULL_RESULTS
    ):
        result_path = nemotron_osl16k_full_root / model_key / method_key / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        request = payload["rollout_batches"][0]["requests"][0]
        request["prompt_id"] = "coherently-mutated-prompt"
        request["prompt_sha256"] = "0" * 64
        request["source_prompt_sha256"] = "1" * 64
        result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="request_provenance_hash"):
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)


def test_nemotron_osl16k_full_reconciles_summary_csv_with_raw_batches(
    nemotron_osl16k_full_root: Path,
) -> None:
    rows = latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)

    assert len(rows) == 5
    assert rows["job_id"].tolist() == [
        job_id
        for _model, _method, _k, job_id in EXPECTED_NEMOTRON_OSL16K_FULL_RESULTS
    ]
    indexed = rows.set_index(["model_key", "method_key"])
    for model_key, method_key, _k, _job_id in EXPECTED_NEMOTRON_OSL16K_FULL_RESULTS:
        payload = json.loads(
            (
                nemotron_osl16k_full_root
                / model_key
                / method_key
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        batches = payload["rollout_batches"]
        output_tokens = sum(batch["output_tokens"] for batch in batches)
        rollout_time = sum(batch["rollout_time_s"] for batch in batches)
        output_tok_s_gpu = output_tokens / rollout_time / payload["config"]["total_gpus"]
        baseline = json.loads(
            (
                nemotron_osl16k_full_root
                / model_key
                / "baseline/result.json"
            ).read_text(encoding="utf-8")
        )
        baseline_tokens = sum(
            batch["output_tokens"] for batch in baseline["rollout_batches"]
        )
        baseline_time = sum(
            batch["rollout_time_s"] for batch in baseline["rollout_batches"]
        )
        baseline_tok_s_gpu = (
            baseline_tokens
            / baseline_time
            / baseline["config"]["total_gpus"]
        )
        row = indexed.loc[(model_key, method_key)]
        assert row["output_tok_s_gpu"] == pytest.approx(output_tok_s_gpu)
        assert row["throughput_speedup"] == pytest.approx(
            output_tok_s_gpu / baseline_tok_s_gpu
        )
        assert row["output_token_ratio"] == pytest.approx(
            output_tokens / baseline_tokens
        )
        if method_key != "baseline":
            num_drafts = sum(
                batch["spec_decode_metrics"]["num_drafts"] for batch in batches
            )
            num_draft_tokens = sum(
                batch["spec_decode_metrics"]["num_draft_tokens"]
                for batch in batches
            )
            num_accepted_tokens = sum(
                batch["spec_decode_metrics"]["num_accepted_tokens"]
                for batch in batches
            )
            assert row["acceptance_rate"] == pytest.approx(
                num_accepted_tokens / num_draft_tokens
            )
            assert row["mean_acceptance_length"] == pytest.approx(
                1.0 + num_accepted_tokens / num_drafts
            )

    assert indexed.loc[("super", "k3"), "throughput_speedup"] == pytest.approx(
        1.709845465759743
    )
    assert indexed.loc[("super", "k5"), "throughput_speedup"] == pytest.approx(
        1.7916630541624878
    )
    assert pd.isna(indexed.loc[("super", "k3"), "rollout_time_speedup"])
    assert pd.isna(indexed.loc[("super", "k5"), "rollout_time_speedup"])
    assert not bool(indexed.loc[("super", "k3"), "selected_by_policy"])
    assert not bool(indexed.loc[("super", "k5"), "selected_by_policy"])
    assert indexed.loc[("ultra", "k5"), "throughput_speedup"] == pytest.approx(
        2.114279153339312
    )
    assert bool(indexed.loc[("ultra", "k5"), "selected_by_policy"])
    assert indexed.loc[("ultra", "k5"), "rollout_time_speedup"] == pytest.approx(
        2.098458292569296
    )
    assert indexed.loc[("ultra", "k5"), "output_token_ratio"] == pytest.approx(
        1.007539278157702
    )


def test_nemotron_osl16k_full_rejects_summary_csv_metric_mismatch(
    nemotron_osl16k_full_root: Path,
) -> None:
    summary_path = nemotron_osl16k_full_root / "summary.csv"
    summary = pd.read_csv(summary_path, float_precision="round_trip")
    row = summary["result_path"] == "ultra/k5/result.json"
    summary.loc[row, "throughput_speedup"] += 0.001
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match="throughput_speedup"):
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)


def test_nemotron_osl16k_full_rejects_one_ulp_summary_csv_drift(
    nemotron_osl16k_full_root: Path,
) -> None:
    summary_path = nemotron_osl16k_full_root / "summary.csv"
    summary = pd.read_csv(summary_path, float_precision="round_trip")
    row = summary["result_path"] == "super/k3/result.json"
    current = float(summary.loc[row, "time_speedup"].iloc[0])
    summary.loc[row, "time_speedup"] = math.nextafter(current, math.inf)
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match="time_speedup"):
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)


def test_nemotron_osl16k_full_rejects_aggregate_not_in_raw_batches(
    nemotron_osl16k_full_root: Path,
) -> None:
    result_path = nemotron_osl16k_full_root / "ultra/k5/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"]["total_output_tokens"] += 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    summary_path = nemotron_osl16k_full_root / "summary.csv"
    summary = pd.read_csv(summary_path, float_precision="round_trip")
    row = summary["result_path"] == "ultra/k5/result.json"
    summary.loc[row, "total_output_tokens"] += 1
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match=r"summary\.total_output_tokens"):
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)


def test_nemotron_osl16k_raw_validation_uses_osl16k_cohort_label(
    nemotron_osl16k_full_root: Path,
) -> None:
    result_path = nemotron_osl16k_full_root / "super/baseline/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["rollout_batches"][0]["actual_output_tokens"][0] += 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="actual_output_tokens") as error:
        latest.load_nemotron_mtp_osl16k_full_rows(nemotron_osl16k_full_root)

    assert "Nemotron MTP OSL 16K full payload" in str(error.value)
    assert "Nemotron MTP OSL 4K K-sweep payload" not in str(error.value)


def test_latest_vllm_html_contains_separate_nemotron_native_mtp_osl16k_full(
    tmp_path: Path,
) -> None:
    temp_html = tmp_path / "docs/vllm_standalone_results_latest.html"
    latest.build_latest_vllm_outputs(
        output_html=temp_html,
        added_csv_out=tmp_path / "docs/vllm_standalone_added_results_latest.csv",
        completed_csv_out=tmp_path / "report/dflare_completed_latest.csv",
        public_data_dir=tmp_path / "public/data",
    )

    html_text = parse_html(temp_html)
    heading = "Nemotron Native MTP OSL 16K Full"
    assert html_text.count(heading) == 1
    section = html_text.split(f"<h2>{heading}</h2>", 1)[1].split(
        "</section>", 1
    )[0]
    osl4k_section = html_text.split(
        "<h2>Nemotron Native MTP OSL 4K K Sweep</h2>", 1
    )[1].split("</section>", 1)[0]

    assert heading not in osl4k_section
    assert "validated fixed-K evidence, not DynamicMTP" in section
    assert "OpenMath natural-EOS Sync-RL-style rollout" in section
    assert "OSL cap 16K" in section
    assert "output work can differ" in section
    assert "Super K3 1.710x and K5 1.792x throughput" in section
    assert "90.05% and 92.16%" in section
    assert "Ultra K5 2.114x throughput" in section
    assert "2.098x rollout-time speedup" in section
    assert "100.75% work ratio" in section
    assert "54.94% acceptance" in section
    assert "mean accepted length 3.75" in section
    assert 'aria-label="Nemotron Native MTP OSL 16K fixed-K throughput speedup' in section
    assert "1.0x baseline" in section

    for header in (
        "Model",
        "Method / K",
        "Job ID",
        "tok/s/GPU",
        "Throughput speedup",
        "Rollout-time speedup",
        "Output-token ratio",
        "Acceptance",
        "Mean accept length",
        "Validity / evidence",
    ):
        assert f"<th>{header}</th>" in section

    table_body = section.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    table_rows = re.findall(r"<tr>.*?</tr>", table_body, flags=re.DOTALL)
    assert len(table_rows) == 5
    super_k3 = next(row for row in table_rows if "2335019" in row)
    super_k5 = next(row for row in table_rows if "2335035" in row)
    ultra_k5 = next(row for row in table_rows if "2335021" in row)
    assert "1.710x" in super_k3
    assert "n/a (output-token ratio outside 1%)" in super_k3
    assert "1.792x" in super_k5
    assert "n/a (output-token ratio outside 1%)" in super_k5
    assert "2.114x" in ultra_k5
    assert "2.098x" in ultra_k5


def test_latest_vllm_html_describes_each_report_cohort_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        latest,
        "publish_nemotron_mtp_k_sweep_evidence",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        latest,
        "render_nemotron_mtp_k_sweep_section",
        lambda **_kwargs: "",
    )
    temp_html = tmp_path / "docs/vllm_standalone_results_latest.html"
    latest.build_latest_vllm_outputs(
        output_html=temp_html,
        added_csv_out=tmp_path / "docs/vllm_standalone_added_results_latest.csv",
        completed_csv_out=tmp_path / "report/dflare_completed_latest.csv",
        public_data_dir=tmp_path / "public/data",
    )

    html_text = parse_html(temp_html)

    for scope in (
        "Nemotron legacy OSL 128 smoke",
        "Nemotron natural-EOS OSL 4K",
        "Nemotron natural-EOS OSL 16K",
        "Older matrices: ISL 4096 / OSL 32768",
        "multiple cohorts, not one global ISL/OSL",
    ):
        assert scope in html_text
    assert "intentionally focused on matched ISL4096/OSL32768 comparisons" not in html_text


def test_nemotron_osl16k_chart_uses_shared_explicit_model_series_colors() -> None:
    section = latest.render_nemotron_mtp_osl16k_full_section()
    chart = section.split("<svg", 1)[1].split("</svg>", 1)[0]

    for model, color in latest.NEMOTRON_MODEL_SERIES_COLORS.items():
        assert re.search(
            rf'<rect[^>]+fill="{re.escape(color)}"[^>]*>'
            rf'<text[^>]*>{model}</text>',
            chart,
        )
        assert re.search(rf'<polyline[^>]+stroke="{re.escape(color)}"', chart)


def test_pages_report_publishes_and_links_exact_nemotron_osl16k_evidence(
    tmp_path: Path,
) -> None:
    public_root = tmp_path / "public"
    report_path = public_root / "reports/vllm_standalone_results_latest.html"
    public_data = public_root / "data"
    href_root = "../data/nemotron_mtp_osl16k_20260706"
    latest.build_latest_vllm_outputs(
        output_html=report_path,
        added_csv_out=tmp_path / "docs/vllm_standalone_added_results_latest.csv",
        completed_csv_out=tmp_path / "report/dflare_completed_latest.csv",
        public_data_dir=public_data,
        nemotron_osl16k_evidence_href_root=href_root,
    )

    html_text = parse_html(report_path)
    section = html_text.split(
        "<h2>Nemotron Native MTP OSL 16K Full</h2>", 1
    )[1].split("</section>", 1)[0]
    evidence_hrefs = re.findall(
        rf'href="({re.escape(href_root)}/[^"]*/result\.json)"',
        section,
    )
    expected_relative_paths = {
        Path(model_key) / method_key / "result.json"
        for model_key, method_key, _k, _job_id in (
            EXPECTED_NEMOTRON_OSL16K_FULL_RESULTS
        )
    }

    assert len(evidence_hrefs) == 5
    assert {
        Path(href).relative_to(href_root) for href in evidence_hrefs
    } == expected_relative_paths
    for href in evidence_hrefs:
        assert (report_path.parent / href).resolve().is_file()

    evidence_root = public_data / "nemotron_mtp_osl16k_20260706"
    published_paths = {
        path.relative_to(evidence_root)
        for path in evidence_root.glob("**/result.json")
    }
    assert published_paths == expected_relative_paths
    for relative_path in expected_relative_paths:
        assert sha256(evidence_root / relative_path) == sha256(
            NEMOTRON_OSL16K_FULL_ROOT / relative_path
        )


def test_public_hub_describes_latest_vllm_page_as_multi_cohort() -> None:
    html_text = parse_html(ROOT / "public/index.html")

    for scope in (
        "legacy Nemotron OSL128 smoke",
        "natural-EOS Nemotron OSL4K/OSL16K cohorts",
        "older ISL4096/OSL32768 Math/SWE matrices",
        "not one global ISL/OSL",
    ):
        assert scope in html_text
    assert "intentionally scoped to matched ISL4096/OSL32768 comparisons" not in html_text


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
