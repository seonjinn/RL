from __future__ import annotations

import hashlib
import json
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
    ("ultra", "k1", 1, "2335051"),
    ("ultra", "k3", 3, "2335053"),
    ("ultra", "k5", 5, "2335030"),
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
        baseline = json.loads(
            (nemotron_k_sweep_root / model_key / "baseline/result.json").read_text(
                encoding="utf-8"
            )
        )["summary"]
        for method_key in ("baseline", "k1", "k3", "k5"):
            summary = json.loads(
                (nemotron_k_sweep_root / model_key / method_key / "result.json").read_text(
                    encoding="utf-8"
                )
            )["summary"]
            row = model_rows.loc[method_key]
            output_ratio = summary["total_output_tokens"] / baseline["total_output_tokens"]
            assert row["output_tok_s_gpu"] == pytest.approx(
                summary["output_tok_s_per_gpu"]
            )
            assert row["throughput_speedup"] == pytest.approx(
                summary["output_tok_s_per_gpu"] / baseline["output_tok_s_per_gpu"]
            )
            assert row["output_token_ratio"] == pytest.approx(output_ratio)
            if abs(output_ratio - 1.0) <= 0.01:
                assert row["rollout_time_speedup"] == pytest.approx(
                    baseline["total_rollout_time_s"] / summary["total_rollout_time_s"]
                )
            else:
                assert pd.isna(row["rollout_time_speedup"])

    super_rows = rows[rows["model_key"] == "super"].set_index("method_key")
    ultra_rows = rows[rows["model_key"] == "ultra"].set_index("method_key")
    assert super_rows.loc["k5", "throughput_speedup"] == pytest.approx(
        1.6863337864738492
    )
    assert ultra_rows.loc["k5", "throughput_speedup"] == pytest.approx(
        1.9729748161942076
    )
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
    assert "K3 is within 2% of best" in section
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
    ultra_k3 = next(row for row in table_rows if "2335053" in row)
    assert "1.952x" in ultra_k3
    assert "n/a (output-token ratio outside 1%)" in ultra_k3
    assert "selected by smallest-K-within-2% policy" in ultra_k3


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
