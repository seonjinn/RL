# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contracts for the self-contained latest-main NanoV3 CUDA Graph report."""

import csv
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
RENDERER = (
    REPO_ROOT / "experiments/cuda_graph/render_latestmain_nanov3_cg_matrix_report.py"
)


class _SectionParser(HTMLParser):
    """Collect section ids without depending on a third-party HTML parser."""

    def __init__(self) -> None:
        super().__init__()
        self.section_ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "section":
            self.section_ids.add(dict(attrs).get("id", ""))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_renderer_writes_escaped_self_contained_matrix_report(tmp_path: Path) -> None:
    """CSV evidence becomes a linkable report without exposing raw operational data."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_csv(
        results_dir / "latestmain_nanov3_cg_matrix_smoke.csv",
        [
            {
                "cluster": "ptyche",
                "scope": "attn",
                "job_id": "1234",
                "state": "COMPLETED",
                "source_sha": "deadbeef",
                "container": "nemo_rl_latestmain_nanov3.sqsh",
                "script": "scopes/01_attn.sh",
                "log_link": "logs/1234.out",
                "wandb_link": "https://wandb.ai/example/run",
                "reason": "<script>alert('hostile')</script>",
            }
        ],
    )
    _write_csv(
        results_dir / "latestmain_nanov3_cg_matrix_runs.csv",
        [
            {
                "scope": "attn",
                "accuracy_metric": "reward_mean",
                "baseline_value": "0.76",
                "candidate_value": "0.77",
                "delta": "0.01",
            }
        ],
    )
    _write_csv(
        results_dir / "latestmain_nanov3_cg_matrix_performance.csv",
        [
            {
                "scope": "attn",
                "timing/train/total_step_time": "2.5",
                "performance/tokens_per_sec_per_gpu": "1234.5",
                "timing/train/generation": "1.1",
                "timing/train/policy_training": "0.8",
                "timing/train/policy_and_reference_logprobs": "0.6",
            }
        ],
    )
    output_path = results_dir / "latestmain_nanov3_cg_matrix_report.html"

    result = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--results-dir",
            str(results_dir),
            "--output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    rendered_html = output_path.read_text()
    parser = _SectionParser()
    parser.feed(rendered_html)
    assert parser.section_ids == {
        "provenance",
        "job-status",
        "performance",
        "convergence",
    }
    assert "Updated (UTC)" in rendered_html
    assert "deadbeef" in rendered_html
    assert "nemo_rl_latestmain_nanov3.sqsh" in rendered_html
    assert "scopes/01_attn.sh" in rendered_html
    assert "logs/1234.out" in rendered_html
    assert "https://wandb.ai/example/run" in rendered_html
    assert "E2E step time (s)" in rendered_html
    assert "tokens/s/GPU" in rendered_html
    assert "&lt;script&gt;alert(&#x27;hostile&#x27;)&lt;/script&gt;" in rendered_html
    assert "<script>alert" not in rendered_html
    assert "checkpoint" not in rendered_html.lower()
    assert "api_key" not in rendered_html.lower()


def test_renderer_handles_missing_result_csvs(tmp_path: Path) -> None:
    """The page remains publishable before the first smoke job is submitted."""
    output_path = tmp_path / "report.html"
    result = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--results-dir",
            str(tmp_path / "absent"),
            "--output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    rendered_html = output_path.read_text()
    assert "No result CSV is available yet." in rendered_html
    assert "provenance" in rendered_html
