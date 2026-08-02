import hashlib
import json
from pathlib import Path

import pytest

from experiments.mxfp8_adaptive_rollout_v0251.contract import (
    AdaptiveInputs,
    build_arm_environment,
)
from experiments.mxfp8_adaptive_rollout_v0251.runtime_overlay import (
    prepare_runtime_overlay,
)
from experiments.mxfp8_adaptive_rollout_v0251.summarize import summarize_log


def _write_table(path: Path) -> str:
    payload = {"schema_version": 1, "metadata": {}, "entries": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_baseline_uses_same_custom_source_without_adaptive_table(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    (runtime_root / "vllm").mkdir(parents=True)

    env = build_arm_environment("baseline", runtime_root=runtime_root)

    assert env["PYTHONPATH"] == str(runtime_root)
    assert env["VLLM_SUBPROCESS_PYTHONPATH"] == str(runtime_root)
    assert env["NEMORL_MXFP8_LINEAR_BACKEND"] == "flashinfer_cutedsl"
    assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE" not in env


def test_adaptive_requires_matching_table_hash(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    (runtime_root / "vllm").mkdir(parents=True)
    table = tmp_path / "tactics.json"
    digest = _write_table(table)
    inputs = AdaptiveInputs(
        tactic_file=table,
        tactic_sha256=digest,
        layer_allowlist_b64="MTI4MCw4MTkyCg==",
    )

    env = build_arm_environment(
        "adaptive", runtime_root=runtime_root, adaptive=inputs
    )

    assert env["NEMORL_MXFP8_LINEAR_BACKEND"] == "flashinfer_trtllm"
    assert env["VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256"] == digest
    assert env["VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M"] == "256"

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_arm_environment(
            "adaptive",
            runtime_root=runtime_root,
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


def test_overlay_builder_only_clears_pythonpath_during_discovery() -> None:
    root = Path(__file__).parents[3]
    launcher = (
        root / "experiments/mxfp8_adaptive_rollout_v0251/run_ab.sh"
    ).read_text(encoding="utf-8")

    builder = launcher.split("builder_python=(", maxsplit=1)[1].split(")", maxsplit=1)[0]
    runtime = launcher.split("runtime_python=(", maxsplit=1)[1].split(")", maxsplit=1)[0]
    assert "-u PYTHONPATH" in builder
    assert "-u PYTHONPATH" not in runtime


def test_runtime_overlay_preserves_wheel_extensions_and_overlays_source(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "installed" / "vllm"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("ORIGIN = 'wheel'\n", encoding="utf-8")
    extension = installed / "_C_stable_libtorch.abi3.so"
    extension.write_bytes(b"compiled-extension")
    (installed / "wheel_only.py").write_text("VALUE = 1\n", encoding="utf-8")

    source = tmp_path / "source" / "vllm"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("ORIGIN = 'source'\n", encoding="utf-8")
    (source / "patched.py").write_text("VALUE = 2\n", encoding="utf-8")

    runtime_root = prepare_runtime_overlay(
        installed_package=installed,
        source_package=source,
        destination_base=tmp_path / "runtime",
        source_revision="deadbeef",
    )

    runtime_package = runtime_root / "vllm"
    assert (runtime_package / "__init__.py").read_text(encoding="utf-8") == (
        "ORIGIN = 'source'\n"
    )
    assert (runtime_package / "patched.py").is_file()
    assert (runtime_package / "wheel_only.py").is_file()
    assert (runtime_package / extension.name).read_bytes() == b"compiled-extension"


def test_runtime_overlay_rejects_wheel_without_stable_extension(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "installed" / "vllm"
    source = tmp_path / "source" / "vllm"
    installed.mkdir(parents=True)
    source.mkdir(parents=True)

    with pytest.raises(ValueError, match="_C_stable_libtorch"):
        prepare_runtime_overlay(
            installed_package=installed,
            source_package=source,
            destination_base=tmp_path / "runtime",
            source_revision="deadbeef",
        )


def test_runtime_overlay_is_immutable_and_content_addressed(tmp_path: Path) -> None:
    installed = tmp_path / "installed" / "vllm"
    source = tmp_path / "source" / "vllm"
    installed.mkdir(parents=True)
    source.mkdir(parents=True)
    extension = installed / "_C_stable_libtorch.abi3.so"
    extension.write_bytes(b"extension-v1")

    first = prepare_runtime_overlay(
        installed_package=installed,
        source_package=source,
        destination_base=tmp_path / "runtime",
        source_revision="deadbeef",
    )
    reused = prepare_runtime_overlay(
        installed_package=installed,
        source_package=source,
        destination_base=tmp_path / "runtime",
        source_revision="deadbeef",
    )
    extension.write_bytes(b"extension-v2")
    second = prepare_runtime_overlay(
        installed_package=installed,
        source_package=source,
        destination_base=tmp_path / "runtime",
        source_revision="deadbeef",
    )

    assert reused == first
    assert second != first
    assert first.is_dir()
    assert second.is_dir()
