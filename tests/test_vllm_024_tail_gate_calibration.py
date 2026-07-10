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
    "target_checkpoint_revision",
    "draft_checkpoint_revision",
    "calibration_timestamp",
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
    bandwidth = 1.0e9
    kappa = 100.0
    target_weights = 1.0e6
    draft_weights = 2.0e5
    eta_d = 1.5
    c_comm = 1.0e-5
    target_overhead_us = 20.0
    draft_overhead_us = {1: 10.0, 3: 30.0, 5: 50.0}
    verify_overhead_us = {1: 15.0, 3: 45.0, 5: 75.0}
    for batch in (1, 8, 32):
        for sequence in (2048, 4096):
            for k in (1, 3, 5):
                target_base = (target_weights + kappa * batch * sequence) / bandwidth
                draft_base = (
                    eta_d * draft_weights + kappa * batch * sequence
                ) / bandwidth
                verify_base = target_base
                rows.append(
                    {
                        "model": "Qwen/Qwen3-32B",
                        "target_tp": "2",
                        "draft_tp": "1",
                        "cluster": "lyris-gb200",
                        "container": "/lustre/test/nemo-rl.sqsh",
                        "container_sha256": "a" * 64,
                        "vllm_commit": "ee0da84a",
                        "target_checkpoint_revision": "1" * 40,
                        "draft_checkpoint_revision": "2" * 40,
                        "calibration_timestamp": "2026-07-10T12:34:56Z",
                        "gpu": "GB200",
                        "B": str(batch),
                        "S": str(sequence),
                        "K": str(k),
                        "T_T": str(
                            (target_base + c_comm + target_overhead_us * batch * 1e-6)
                            * 1e3
                        ),
                        "T_D": str(
                            (draft_base + c_comm + draft_overhead_us[k] * batch * 1e-6)
                            * 1e3
                        ),
                        "T_V": str(
                            (
                                verify_base
                                + c_comm
                                + verify_overhead_us[k] * batch * 1e-6
                            )
                            * 1e3
                        ),
                        "W_t": str(target_weights),
                        "W_d": str(draft_weights),
                        "C_dense": "1000",
                        "C_attn": "1",
                        "kappa_theoretical": str(int(kappa)),
                        "F_eff": "1000000000000",
                        "BW_peak": "2000000000",
                        "F_peak": "4000000000000",
                        "c_comm": str(c_comm),
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
    assert payload["calibration"]["c_T"] == 0.0
    assert payload["calibration"]["c_D"] == 0.0
    assert payload["calibration"]["c_V"] == 0.0
    per_gamma = payload["calibration"]["per_gamma"]
    assert set(per_gamma) == {"1", "3", "5"}
    assert len({per_gamma[key]["c_D"] for key in per_gamma}) == 3
    assert len({per_gamma[key]["c_V"] for key in per_gamma}) == 3
    for fit in per_gamma.values():
        assert fit["c_T"] >= 0.0
        assert fit["c_D"] >= 0.0
        assert fit["c_V"] >= 0.0
        assert fit["c_D"] < min(float(row["T_D"]) for row in _rows()) * 1000.0
        assert fit["c_V"] < min(float(row["T_V"]) for row in _rows()) * 1000.0
    for section, key in (
        ("hardware", "BW_eff"),
        ("model", "W_t"),
        ("model", "W_d"),
        ("model", "C_dense"),
        ("model", "C_attn"),
        ("calibration", "eta_d"),
        ("calibration", "kappa_eff"),
        ("calibration", "F_eff"),
    ):
        assert payload[section][key] > 0
    assert payload["metadata"] == {
        "calibration_schema": "efficientrollout-sd-toggle-v1",
        "cluster": "lyris-gb200",
        "container": "/lustre/test/nemo-rl.sqsh",
        "container_sha256": "a" * 64,
        "draft_tp": 1,
        "draft_checkpoint_revision": "2" * 40,
        "calibration_timestamp": "2026-07-10T12:34:56Z",
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "k_values": [1, 3, 5],
        "measurement_rows": 18,
        "model": "Qwen/Qwen3-32B",
        "target_tp": 2,
        "target_checkpoint_revision": "1" * 40,
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
        (
            "target_checkpoint_revision",
            "",
            "missing required value: target_checkpoint_revision",
        ),
        ("calibration_timestamp", "", "missing required value: calibration_timestamp"),
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
        ("target_checkpoint_revision", "3" * 40),
        ("draft_checkpoint_revision", "4" * 40),
        ("calibration_timestamp", "2026-07-11T12:34:56Z"),
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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_tp", "0"),
        ("target_tp", "1.5"),
        ("draft_tp", "-1"),
        ("draft_tp", "main"),
    ),
)
def test_calibration_rejects_non_positive_integer_tp(
    tmp_path: Path, field: str, value: str
) -> None:
    source = tmp_path / "measurements.csv"
    rows = _rows()
    for row in rows:
        row[field] = value
    _write_csv(source, rows)

    result = _run(source, tmp_path / "output")

    assert result.returncode == 2
    assert f"{field} must be a positive integer" in result.stderr


@pytest.mark.parametrize(
    "field", ("target_checkpoint_revision", "draft_checkpoint_revision")
)
def test_calibration_rejects_mutable_checkpoint_revisions(
    tmp_path: Path, field: str
) -> None:
    source = tmp_path / "measurements.csv"
    rows = _rows()
    for row in rows:
        row[field] = "main"
    _write_csv(source, rows)

    result = _run(source, tmp_path / "output")

    assert result.returncode == 2
    assert (
        f"{field} must be an exact 40-character hexadecimal revision" in result.stderr
    )
