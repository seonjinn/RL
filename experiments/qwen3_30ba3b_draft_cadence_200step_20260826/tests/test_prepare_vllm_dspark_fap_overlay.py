"""Tests for the fail-closed DSpark FAP vLLM overlay."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from prepare_vllm_dspark_fap_overlay import parse_args, prepare_overlay  # noqa: E402


STOCK_GUARD = """        if has_trtllm_support:
            return AttentionCGSupport.UNIFORM_BATCH
"""
PATCHED_GUARD = """        # trtllm-gen only supports causal attention.
        if has_trtllm_support and not vllm_config.attention_config.use_non_causal:
            return AttentionCGSupport.UNIFORM_BATCH
"""


def write_package(root: Path, guard: str) -> Path:
    package = root / "site-packages" / "vllm"
    backend = package / "v1" / "attention" / "backends" / "flashinfer.py"
    backend.parent.mkdir(parents=True)
    backend.write_text(f"before\n{guard}after\n")
    (package / "__init__.py").write_text('__version__ = "0.25.1"\n')
    return package


def test_prepare_overlay_patches_exact_guard_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = write_package(tmp_path / "source", STOCK_GUARD)
    source_backend = source / "v1" / "attention" / "backends" / "flashinfer.py"
    original = source_backend.read_bytes()

    overlay_package = prepare_overlay(source, tmp_path / "overlay")

    patched_backend = (
        overlay_package / "v1" / "attention" / "backends" / "flashinfer.py"
    )
    assert patched_backend.read_text() == f"before\n{PATCHED_GUARD}after\n"
    assert source_backend.read_bytes() == original
    receipt = json.loads(
        (tmp_path / "overlay" / "dspark-fap-vllm-48167-attention-guard.json").read_text()
    )
    assert receipt["schema_version"] == 1
    assert receipt["upstream_pr"] == "https://github.com/vllm-project/vllm/pull/48167"
    assert receipt["status"] == "applied"
    assert receipt["patched_sha256"] == hashlib.sha256(
        patched_backend.read_bytes()
    ).hexdigest()


def test_prepare_overlay_accepts_exact_already_patched_source(tmp_path: Path) -> None:
    source = write_package(tmp_path / "source", PATCHED_GUARD)

    overlay_package = prepare_overlay(source, tmp_path / "overlay")

    receipt = json.loads(
        (tmp_path / "overlay" / "dspark-fap-vllm-48167-attention-guard.json").read_text()
    )
    assert receipt["status"] == "already-patched"
    assert (
        overlay_package / "v1" / "attention" / "backends" / "flashinfer.py"
    ).read_text() == f"before\n{PATCHED_GUARD}after\n"


def test_prepare_overlay_rejects_source_drift(tmp_path: Path) -> None:
    source = write_package(
        tmp_path / "source",
        "        if has_trtllm_support and unknown_condition:\n"
        "            return AttentionCGSupport.UNIFORM_BATCH\n",
    )

    with pytest.raises(ValueError, match="expected vLLM 0.25.1 attention guard"):
        prepare_overlay(source, tmp_path / "overlay")

    assert not (tmp_path / "overlay").exists()


def test_parse_args_uses_worker_overlay_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay_root = tmp_path / "overlay"
    monkeypatch.setenv("Q30_VLLM_OVERLAY", str(overlay_root))
    monkeypatch.setattr(sys, "argv", ["prepare_vllm_dspark_fap_overlay.py"])

    args = parse_args()

    assert args.overlay_root == overlay_root
