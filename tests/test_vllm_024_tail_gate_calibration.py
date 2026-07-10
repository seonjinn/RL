from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATOR = REPO_ROOT / "experiments" / "vllm_024_upgrade" / "calibrate_tail_gate.py"
CSV_FIELDS = (
    "model",
    "target_tp",
    "draft_tp",
    "cluster",
    "container",
    "container_sha256",
    "vllm_commit",
    "gpu",
    "B",
    "S",
    "K",
    "T_T",
    "T_D",
    "T_V",
    "W_t",
    "W_d",
    "C_dense",
    "C_attn",
    "kappa_theoretical",
    "F_eff",
    "BW_peak",
    "F_peak",
    "c_comm",
)


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for batch in (1, 8, 32):
        for sequence in (2048, 4096):
            for k in (1, 5):
                rows.append(
                    {
                        "model": "Qwen/Qwen3-32B",
                        "target_tp": "2",
                        "draft_tp": "1",
                        "cluster": "lyris-gb200",
                        "container": "/lustre/test/nemo-rl.sqsh",
                        "container_sha256": "a" * 64,
                        "vllm_commit": "ee0da84a",
                        "gpu": "GB200",
                        "B": str(batch),
                        "S": str(sequence),
                        "K": str(k),
                        "T_T": str(0.8 + batch * 0.03 + sequence / 1_000_000),
                        "T_D": str(0.11 + batch * 0.01 + k * 0.01),
                        "T_V": str(0.6 + batch * 0.02 + k * 0.03),
                        "W_t": "64000000000",
                        "W_d": "2000000000",
                        "C_dense": "100000000000",
                        "C_attn": "10000000",
                        "kappa_theoretical": "262144",
                        "F_eff": "1000000000000000",
                        "BW_peak": "8000000000000",
                        "F_peak": "4500000000000000",
                        "c_comm": "0.001",
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _run(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(CALIBRATOR),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_calibration_writes_deterministic_efficientrollout_schema_and_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "measurements.csv"
    _write_csv(source, _rows())
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = _run(source, first_output)
    second = _run(source, second_output)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_json = next(first_output.glob("*.json"))
    second_json = next(second_output.glob("*.json"))
    assert first_json.read_bytes() == second_json.read_bytes()
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["hardware"]["gpu"] == "GB200"
    assert payload["hardware"]["tp"] == 2
    assert payload["model"]["name"] == "Qwen/Qwen3-32B"
    assert payload["calibration"]["per_gamma"] == {}
    for section, key in (
        ("hardware", "BW_eff"),
        ("model", "W_t"),
        ("model", "W_d"),
        ("model", "C_dense"),
        ("model", "C_attn"),
        ("calibration", "eta_d"),
        ("calibration", "kappa_eff"),
        ("calibration", "F_eff"),
        ("calibration", "c_T"),
        ("calibration", "c_D"),
        ("calibration", "c_V"),
    ):
        assert payload[section][key] > 0
    assert payload["metadata"] == {
        "calibration_schema": "efficientrollout-sd-toggle-v1",
        "cluster": "lyris-gb200",
        "container": "/lustre/test/nemo-rl.sqsh",
        "container_sha256": "a" * 64,
        "draft_tp": 1,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "k_values": [1, 5],
        "measurement_rows": 12,
        "model": "Qwen/Qwen3-32B",
        "target_tp": 2,
        "vllm_commit": "ee0da84a",
    }
    sidecar = Path(f"{first_json}.sha256")
    assert sidecar.read_text(encoding="utf-8") == (
        f"{hashlib.sha256(first_json.read_bytes()).hexdigest()}  {first_json.name}\n"
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("model", "", "missing required value: model"),
        ("target_tp", "", "missing required value: target_tp"),
        ("cluster", "", "missing required value: cluster"),
    ),
)
def test_calibration_rejects_missing_provenance(
    tmp_path: Path, field: str, value: str, error: str
) -> None:
    source = tmp_path / "measurements.csv"
    rows = _rows()
    rows[0][field] = value
    _write_csv(source, rows)

    result = _run(source, tmp_path / "output")

    assert result.returncode == 2
    assert error in result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model", "Qwen/Qwen3-30B-A3B"),
        ("target_tp", "1"),
        ("draft_tp", "2"),
        ("cluster", "oci-hsg-gb200"),
    ),
)
def test_calibration_rejects_mixed_measurement_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    source = tmp_path / "measurements.csv"
    rows = _rows()
    rows[-1][field] = value
    _write_csv(source, rows)

    result = _run(source, tmp_path / "output")

    assert result.returncode == 2
    assert f"mixed {field}" in result.stderr
