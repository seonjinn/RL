import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from analyze_results import (  # noqa: E402
    aggregate_throughput,
    latency_speedup,
    select_steps,
    throughput_speedup,
    write_comparison_csv,
)


def test_select_steps_drops_warmup() -> None:
    values = {"1": 100.0, "2": 20.0, "3": 30.0}

    assert select_steps(values, first_step=2) == [20.0, 30.0]


def test_throughput_speedup_uses_mxfp8_over_bf16() -> None:
    assert throughput_speedup(bf16=100.0, mxfp8=125.0) == 1.25


def test_latency_speedup_uses_bf16_over_mxfp8() -> None:
    assert latency_speedup(bf16=100.0, mxfp8=80.0) == 1.25


def test_aggregate_throughput_weights_by_work() -> None:
    tokens = {"1": 100.0, "2": 200.0, "3": 400.0}
    seconds = {"1": 100.0, "2": 10.0, "3": 30.0}

    assert aggregate_throughput(tokens, seconds, gpu_count=2, first_step=2) == 7.5


def test_csv_uses_unix_newlines(tmp_path: Path) -> None:
    output_path = tmp_path / "comparison.csv"
    write_comparison_csv([], output_path)

    assert b"\r\n" not in output_path.read_bytes()
