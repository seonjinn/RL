from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCRIPT = (
    REPO_ROOT / "experiments" / "qwen30b_mxfp8_linear_backends" / "summarize_results.py"
)


def _load_summary_module():
    spec = importlib.util.spec_from_file_location(
        "qwen30b_backend_summary", SUMMARY_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_steps_and_summarize_steady_state() -> None:
    summary = _load_summary_module()
    log = """
Training Results:
  • Mean Generation Length: 4000.00
  • Total step time: 258.07s
  • generation: 56.20s (21.8%)
    - E2E (Tokens/sec/gpu): 1992.83
    - Generation Worker Group (Tokens/sec/gpu): 9150.66
Training Results:
  • Mean Generation Length: 32000.00
  • Total step time: 216.19s
  • generation: 54.66s (25.3%)
    - E2E (Tokens/sec/gpu): 2365.21
    - Generation Worker Group (Tokens/sec/gpu): 9354.79
Training Results:
  • Mean Generation Length: 30000.00
  • Total step time: 210.00s
  • generation: 50.00s (23.8%)
    - E2E (Tokens/sec/gpu): 2400.00
    - Generation Worker Group (Tokens/sec/gpu): 9600.00
"""

    steps = summary.parse_training_results(log)

    assert [step.step for step in steps] == [1, 2, 3]
    assert steps[1].generation_tokens_per_sec_per_gpu == 9354.79
    assert steps[2].generation_seconds == 50.0
    assert steps[1].mean_generation_length == 32000.0
    steady = summary.summarize_steps(steps, first_step=2)
    assert steady.num_steps == 2
    assert steady.generation_tokens_per_sec_per_gpu_mean == 9477.395
    assert steady.generation_seconds_mean == 52.33
    assert steady.total_step_seconds_mean == 213.095
    assert steady.mean_generation_length_mean == 31000.0


def test_parse_rejects_incomplete_metric_block() -> None:
    summary = _load_summary_module()
    log = """
Training Results:
  • Mean Generation Length: 4000.00
  • Total step time: 258.07s
  • generation: 56.20s (21.8%)
"""

    assert summary.parse_training_results(log) == []


def test_write_results_accepts_backend_subset(tmp_path: Path) -> None:
    summary = _load_summary_module()
    log = """
Training Results:
  • Mean Generation Length: 32000.00
  • Total step time: 220.00s
  • generation: 60.00s (27.3%)
    - E2E (Tokens/sec/gpu): 2000.00
    - Generation Worker Group (Tokens/sec/gpu): 9000.00
"""
    backends = ("flashinfer_cutedsl", "flashinfer_trtllm_adaptive")
    for backend in backends:
        log_dir = tmp_path / "run" / backend / "123-logs"
        log_dir.mkdir(parents=True)
        (log_dir / "ray-driver.log").write_text(log)

    summary.write_results(
        tmp_path / "run", tmp_path / "summary", first_step=1, backends=backends
    )

    output = (tmp_path / "summary" / "summary.json").read_text()
    assert "flashinfer_cutedsl" in output
    assert "flashinfer_trtllm_adaptive" in output
    assert "flashinfer_cutlass" not in output
