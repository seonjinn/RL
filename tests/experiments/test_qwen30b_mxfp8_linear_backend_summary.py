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
  • Total step time: 258.07s
  • generation: 56.20s (21.8%)
    - E2E (Tokens/sec/gpu): 1992.83
    - Generation Worker Group (Tokens/sec/gpu): 9150.66
Training Results:
  • Total step time: 216.19s
  • generation: 54.66s (25.3%)
    - E2E (Tokens/sec/gpu): 2365.21
    - Generation Worker Group (Tokens/sec/gpu): 9354.79
Training Results:
  • Total step time: 210.00s
  • generation: 50.00s (23.8%)
    - E2E (Tokens/sec/gpu): 2400.00
    - Generation Worker Group (Tokens/sec/gpu): 9600.00
"""

    steps = summary.parse_training_results(log)

    assert [step.step for step in steps] == [1, 2, 3]
    assert steps[1].generation_tokens_per_sec_per_gpu == 9354.79
    assert steps[2].generation_seconds == 50.0
    steady = summary.summarize_steps(steps, first_step=2)
    assert steady.num_steps == 2
    assert steady.generation_tokens_per_sec_per_gpu_mean == 9477.395
    assert steady.generation_seconds_mean == 52.33
    assert steady.total_step_seconds_mean == 213.095


def test_parse_rejects_incomplete_metric_block() -> None:
    summary = _load_summary_module()
    log = """
Training Results:
  • Total step time: 258.07s
  • generation: 56.20s (21.8%)
"""

    assert summary.parse_training_results(log) == []
