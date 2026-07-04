from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

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


def test_task5_latest_vllm_html_contains_native_and_status_sections() -> None:
    latest.main()

    html_text = parse_html(ROOT / "docs/vllm_standalone_results_latest.html")

    assert "vLLM 0.24 / Native Profile Results" in html_text
    assert "vLLM 0.24 / DFlare Completed Results" in html_text
    assert "DFlare Failure and Status" in html_text
    assert "2272937" in html_text
    assert "2272938" in html_text
    assert "2272941" in html_text
    assert "2272942" in html_text
    assert "slurm_wall_time_5h" in html_text
    assert "gather_object_cuda_oom_after_generation" in html_text
    assert "retry_of" in html_text
    assert "vllm024_profiles_latest.csv" in html_text
    assert "dflare_job_status_latest.csv" in html_text
    assert (ROOT / "public/data/vllm024_profiles_latest.csv").exists()
    assert (ROOT / "public/data/dflare_completed_latest.csv").exists()
    assert (ROOT / "public/data/dflare_job_status_latest.csv").exists()


def test_task5_index_publishes_new_artifacts_and_counts() -> None:
    latest.main()
    index.build()

    html_text = parse_html(ROOT / "public/index.html")

    assert "8 completed target-profile DFlare job(s)" in html_text
    assert "12 performance row(s)" in html_text
    assert 'href="data/vllm024_profiles_latest.csv"' in html_text
    assert 'href="data/dflare_completed_latest.csv"' in html_text
    assert 'href="data/dflare_job_status_latest.csv"' in html_text
