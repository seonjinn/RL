from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO_ROOT / "experiments" / "vllm_024_dynamicsd"
sys.path.insert(0, str(EXPERIMENT))

import speedbench_dataset as adapter  # pyright: ignore[reportMissingImports]  # noqa: E402


def test_subset_discovery_accepts_only_requested_config(tmp_path: Path) -> None:
    parquet = tmp_path / "throughput_1k" / "test.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"payload")

    assert adapter.discover_prepared_parquet_paths(
        tmp_path,
        expected_configs=("throughput_1k",),
    ) == (Path("throughput_1k/test.parquet"),)


def test_subset_discovery_rejects_unrequested_config(tmp_path: Path) -> None:
    for config in ("throughput_1k", "qualitative"):
        parquet = tmp_path / config / "test.parquet"
        parquet.parent.mkdir(parents=True)
        parquet.write_bytes(config.encode())

    with pytest.raises(ValueError, match="unexpected parquet paths"):
        adapter.discover_prepared_parquet_paths(
            tmp_path,
            expected_configs=("throughput_1k",),
        )


def test_stage_defaults_to_throughput_only_config() -> None:
    completed = subprocess.run(
        [str(EXPERIMENT / "stage_speedbench.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CLUSTER": "lyris",
            "DRY_RUN": "true",
            "REQUIRE_GIT_PULL": "false",
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SPEED_CONFIGS=throughput_1k" in completed.stdout
    assert 'for speed_config in $SPEED_CONFIGS' in completed.stdout
    assert '--expected-config "$speed_config"' in completed.stdout
    assert "--config all" not in completed.stdout
