import hashlib
import json
from pathlib import Path

import pytest

from experiments.mxfp8_adaptive_rollout_v0251.contract import (
    AdaptiveInputs,
    build_arm_environment,
)
from experiments.mxfp8_adaptive_rollout_v0251.summarize import summarize_log


def _write_table(path: Path) -> str:
    payload = {"schema_version": 1, "metadata": {}, "entries": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_baseline_uses_same_custom_source_without_adaptive_table(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vllm"
    source.mkdir()

    env = build_arm_environment("baseline", source=source)

    assert env["PYTHONPATH"] == str(source)
    assert env["NEMORL_MXFP8_LINEAR_BACKEND"] == "flashinfer_cutedsl"
    assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE" not in env


def test_adaptive_requires_matching_table_hash(tmp_path: Path) -> None:
    source = tmp_path / "vllm"
    source.mkdir()
    table = tmp_path / "tactics.json"
    digest = _write_table(table)
    inputs = AdaptiveInputs(
        tactic_file=table,
        tactic_sha256=digest,
        layer_allowlist_b64="MTI4MCw4MTkyCg==",
    )

    env = build_arm_environment("adaptive", source=source, adaptive=inputs)

    assert env["NEMORL_MXFP8_LINEAR_BACKEND"] == "flashinfer_trtllm"
    assert env["VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256"] == digest
    assert env["VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M"] == "256"

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_arm_environment(
            "adaptive",
            source=source,
            adaptive=AdaptiveInputs(
                tactic_file=table,
                tactic_sha256="0" * 64,
                layer_allowlist_b64="MTI4MCw4MTkyCg==",
            ),
        )


def test_summary_rejects_partial_log(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("NEMORL_CANARY arm=baseline event=start\n", encoding="utf-8")

    summary = summarize_log(log)

    assert summary["complete"] is False
    assert summary["elapsed_seconds"] is None


def test_summary_reads_completed_run(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "NEMORL_CANARY arm=adaptive event=start epoch=10.0",
                "NEMORL_CANARY event=model_ready epoch=14.5",
                "NEMORL_CANARY event=outputs tokens=8192",
                "NEMORL_CANARY arm=adaptive event=complete epoch=20.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_log(log)

    assert summary == {
        "arm": "adaptive",
        "complete": True,
        "elapsed_seconds": 10.0,
        "model_load_seconds": 4.5,
        "output_tokens": 8192,
    }


def test_arm_reuses_locked_driver_interpreter_for_ray_actors() -> None:
    root = Path(__file__).parents[3]
    launcher = (
        root / "experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh"
    ).read_text(encoding="utf-8")

    assert "export NEMO_RL_PY_EXECUTABLES_SYSTEM=1" in launcher
