from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "cuda_graph"
    / "nemotron_thd_te_graph_20260731"
    / "analyze_cuda_graph_calls.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_cuda_graph_calls", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_cuda_api_summary_namespaces_driver_and_runtime_launches() -> None:
    module = _load_module()
    report = """\
Processing [worker.nsys-rep]...
"Time (%)","Total Time (ns)","Num Calls","Average (ns)","Name"
"40.0","400","8","50.0","cudaGraphLaunch_ptsz"
"30.0","300","3","100.0","cuGraphLaunch"
"30.0","300","30","10.0","cudaMemcpyAsync"
"""

    assert module.parse_cuda_api_summary(report) == {
        "nsys_cuda_api_calls": 41,
        "nsys_cuda_graph_launch_calls": 11,
    }


def test_summarize_profiles_keeps_nsys_share_distinct_from_runtime_coverage() -> None:
    module = _load_module()

    assert module.summarize_profiles(
        [
            {"nsys_cuda_api_calls": 100, "nsys_cuda_graph_launch_calls": 20},
            {"nsys_cuda_api_calls": 80, "nsys_cuda_graph_launch_calls": 0},
            {"nsys_cuda_api_calls": 120, "nsys_cuda_graph_launch_calls": 10},
        ]
    ) == {
        "nsys_profile_count": 3,
        "nsys_profiles_with_cuda_graph_launches": 2,
        "nsys_process_coverage_pct": 66.666667,
        "nsys_total_cuda_api_calls": 300,
        "nsys_total_cuda_graph_launch_calls": 30,
        "nsys_cuda_graph_launch_share_of_cuda_api_calls_pct": 10.0,
        "nsys_cuda_graph_launch_calls_min": 0,
        "nsys_cuda_graph_launch_calls_median": 10,
        "nsys_cuda_graph_launch_calls_max": 20,
    }


def test_cli_writes_labeled_json_for_all_profiles(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "worker-1.nsys-rep").touch()
    (profiles / "worker-2.nsys-rep").touch()
    fake_nsys = tmp_path / "nsys"
    fake_nsys.write_text(
        """#!/bin/sh
case "$*" in
  *worker-1.nsys-rep*) graph_calls=4 ;;
  *) graph_calls=0 ;;
esac
cat <<EOF
"Time (%)","Total Time (ns)","Num Calls","Average (ns)","Name"
"50","100","$graph_calls","25","cudaGraphLaunch"
"50","100","20","5","cudaMemcpyAsync"
EOF
"""
    )
    fake_nsys.chmod(0o755)
    output = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--label",
            f"positive={profiles}",
            "--nsys",
            str(fake_nsys),
            "--output-json",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["positive"]["nsys_profile_count"] == 2
    assert payload["positive"]["nsys_profiles_with_cuda_graph_launches"] == 1
    assert payload["positive"]["nsys_total_cuda_graph_launch_calls"] == 4
